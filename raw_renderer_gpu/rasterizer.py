"""
GPU render() — pure PyTorch, no C++ compilation needed.

Rasterisation is face-batched so GPU memory stays bounded.
EnvMap shading is sample-batched for the same reason.
Everything else (projection, interpolation, BRDF) is fully vectorised
over all hit pixels at once.
"""

import numpy as np
import torch
from dataclasses import dataclass
from PIL import Image
from typing import Union


# ─────────────────────────────────────────── tensor-based types ──────────────

@dataclass
class SHLight:
    """SH lighting: 9 coefficients per RGB channel."""
    coeffs: torch.Tensor          # (9, 3)


@dataclass
class PBRMat:
    """Cook-Torrance metallic-roughness material."""
    albedo:    torch.Tensor                       # (3,) or (..., 3)
    metallic:  Union[float, torch.Tensor] = 0.0
    roughness: float = 0.5


@dataclass
class PhongMat:
    """Phong material."""
    base_color: torch.Tensor      # (3,)
    ka:         float = 0.05
    kd:         float = 0.80
    ks:         float = 0.30
    shininess:  float = 32.0


@dataclass
class PointLightGPU:
    """Point light source."""
    position: torch.Tensor        # (3,)
    color:    torch.Tensor        # (3,)


@dataclass
class EnvMapLightGPU:
    """Pre-processed environment map (flat samples)."""
    dirs:         torch.Tensor    # (P, 3)
    image_flat:   torch.Tensor    # (P, 3)
    solid_angles: torch.Tensor    # (P,)


# ─────────────────────────────────────────── camera math ─────────────────────

def _look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    """4×4 view matrix: world → camera space."""
    f = target - eye
    f /= np.linalg.norm(f)
    r = np.cross(f, up)
    r /= np.linalg.norm(r)
    u = np.cross(r, f)
    M = np.eye(4, dtype=np.float32)
    M[0, :3] = r
    M[0, 3] = -r.dot(eye)
    M[1, :3] = u
    M[1, 3] = -u.dot(eye)
    M[2, :3] = -f
    M[2, 3] = f.dot(eye)
    return M


def _perspective(fov_deg: float, aspect: float, near: float = 0.1, far: float = 100.0) -> np.ndarray:
    """4×4 perspective projection matrix (OpenGL/NDC convention)."""
    t = np.tan(np.radians(fov_deg) / 2)
    M = np.zeros((4, 4), dtype=np.float32)
    M[0, 0] = 1 / (aspect * t)
    M[1, 1] = 1 / t
    M[2, 2] = -(far + near) / (far - near)
    M[2, 3] = -2 * far * near / (far - near)
    M[3, 2] = -1
    return M


# ─────────────────────────────────────────── small helpers ───────────────────

def _cuda(x, dev, dtype=torch.float32):
    return torch.from_numpy(np.ascontiguousarray(x)).to(dev, dtype=dtype)


def _norm(x, dim=-1):
    return x / (x.norm(dim=dim, keepdim=True) + 1e-8)


# ─────────────────────────────────────────── stage 1: projection ─────────────

def _project(verts, MVP, W, H):
    homo = torch.cat([verts, verts.new_ones(len(verts), 1)], 1)  # (V, 4)
    clip = homo @ MVP.T                                           # (V, 4)
    ndc = clip[:, :3] / clip[:, 3:4]                            # (V, 3)
    s = verts.new_empty(ndc.shape)
    s[:, 0] = (ndc[:, 0] + 1.0) * 0.5 * W
    s[:, 1] = (1.0 - ndc[:, 1]) * 0.5 * H
    s[:, 2] = ndc[:, 2]
    return s                                                       # (V, 3)


# ─────────────────────────────────────────── stage 2: rasterise ──────────────

def _rasterize(sverts, faces, W, H, face_batch=32):
    """
    Per-pixel closest-triangle test, processed in face batches to limit VRAM.
    Returns face_ids (H*W,) int64  [−1 = background]
            bary     (H*W, 3) float32
    """
    dev = sverts.device
    P = W * H
    F_n = faces.shape[0]

    px = torch.arange(W, device=dev, dtype=torch.float32) + 0.5
    py = torch.arange(H, device=dev, dtype=torch.float32) + 0.5
    py_g, px_g = torch.meshgrid(py, px, indexing='ij')
    pcx = px_g.reshape(P)        # (P,)
    pcy = py_g.reshape(P)

    best_z = sverts.new_full((P,),   float('inf'))
    face_ids = torch.full((P,), -1, device=dev, dtype=torch.long)
    bary = sverts.new_zeros(P, 3)
    aP = torch.arange(P, device=dev)

    for fi in range(0, F_n, face_batch):
        fb = faces[fi: fi + face_batch]          # (B, 3)
        ax = sverts[fb[:, 0], 0]
        ay = sverts[fb[:, 0], 1]
        az = sverts[fb[:, 0], 2]
        bx = sverts[fb[:, 1], 0]
        by = sverts[fb[:, 1], 1]
        bz = sverts[fb[:, 1], 2]
        cx = sverts[fb[:, 2], 0]
        cy = sverts[fb[:, 2], 1]
        cz = sverts[fb[:, 2], 2]

        denom = (by - cy)*(ax - cx) + (cx - bx)*(ay - cy)   # (B,)
        ok = denom.abs() > 1e-6

        # barycentric coords for all pixels × batch faces  →  (P, B)
        w0 = ((by-cy)*(pcx[:, None]-cx) + (cx-bx)
              * (pcy[:, None]-cy)) / (denom + 1e-10)
        w1 = ((cy-ay)*(pcx[:, None]-cx) + (ax-cx)
              * (pcy[:, None]-cy)) / (denom + 1e-10)
        w2 = 1.0 - w0 - w1

        inside = (w0 >= 0) & (w1 >= 0) & (w2 >= 0) & ok    # (P, B)
        z = torch.where(inside, w0*az + w1*bz + w2*cz,
                        sverts.new_full((), float('inf')))

        bz_best, bl = z.min(1)                               # (P,)
        imp = bz_best < best_z
        best_z = torch.where(imp, bz_best, best_z)
        face_ids = torch.where(imp, fi + bl, face_ids)
        bary = torch.where(imp[:, None],
                           torch.stack(
                               [w0[aP, bl], w1[aP, bl], w2[aP, bl]], 1),
                           bary)

    return face_ids, bary


# ─────────────────────────────────────────── stage 3a: interpolate ───────────

def _interp(verts, fn, vn, faces, face_ids, bary, hit, smooth):
    fi = face_ids[hit]
    bw = bary[hit]                               # (M, 3)
    i0, i1, i2 = faces[fi, 0], faces[fi, 1], faces[fi, 2]

    frag_pos = bw[:, 0:1]*verts[i0] + bw[:, 1:2] * \
        verts[i1] + bw[:, 2:3]*verts[i2]

    if smooth:
        raw_n = bw[:, 0:1]*vn[i0] + bw[:, 1:2]*vn[i1] + bw[:, 2:3]*vn[i2]
        N = _norm(raw_n)
    else:
        N = fn[fi]

    return frag_pos, N                             # (M, 3), (M, 3)


# ─────────────────────────────────────────── SH helpers ──────────────────────

def _sh_basis(d):
    """d: (..., 3) → (..., 9)"""
    x, y, z = d[..., 0], d[..., 1], d[..., 2]
    return torch.stack([
        torch.ones_like(x) * 0.282095,
        0.488603*y, 0.488603*z, 0.488603*x,
        1.092548*x*y, 1.092548*y*z,
        0.315392*(3*z**2 - 1),
        1.092548*x*z, 0.546274*(x**2 - y**2),
    ], dim=-1)                                     # (..., 9)


def _sh_irradiance(coeffs_t, N):
    """ZH band-limiting weights for diffuse — coeffs_t: (9,3); N: (...,3) → (...,3)"""
    Y = _sh_basis(N)                               # (..., 9)
    A = N.new_tensor([
        torch.pi,
        2*torch.pi/3, 2*torch.pi/3, 2*torch.pi/3,
        torch.pi/4,   torch.pi/4,   torch.pi/4, torch.pi/4, torch.pi/4,
    ])
    return ((A * Y) @ coeffs_t)       # (..., 3)
    # return ((A * Y) @ coeffs_t).clamp(min=0)       # (..., 3)


def _sh_phong_filtered_radiance(coeffs_t, dirs, shininess):
    """Phong-lobe SH filter — coeffs_t: (9,3); dirs: (...,3) → (...,3)"""
    Y = _sh_basis(dirs)
    B_0 = 2 * torch.pi / (shininess + 1)
    B_1 = 2 * torch.pi / (shininess + 2)
    B_2 = torch.pi * (3.0 / (shininess + 3) - 1.0 / (shininess + 1))
    norm = (shininess + 2) / (2 * torch.pi)
    B = Y.new_tensor([B_0,
                      B_1, B_1, B_1,
                      B_2, B_2, B_2, B_2, B_2]) * norm
    return ((B * Y) @ coeffs_t).clamp(min=0)       # (..., 3)


# ─────────────────────────────────────────── CT micro-terms ──────────────────

def _ggx_D(NdH, alpha2):
    d = NdH**2 * (alpha2 - 1.0) + 1.0
    return alpha2 / (torch.pi * d**2 + 1e-7)


def _schlick_F(VdH, F0):
    """VdH: (...); F0: (3,) → (..., 3)"""
    return F0 + (1.0 - F0) * (1.0 - VdH.unsqueeze(-1)) ** 5


def _smith_G(NdV, NdL, k):
    return (NdV/(NdV*(1-k)+k+1e-7)) * (NdL/(NdL*(1-k)+k+1e-7))


def _f0_mat(albedo, metallic):
    return 0.04*(1-metallic) + albedo*metallic


# ─────────────────────────────────────────── Phong shaders ───────────────────

def _phong_point(frag_pos, N, cam_pos, mat: PhongMat, light: PointLightGPU):
    V = _norm(cam_pos - frag_pos)
    L = _norm(light.position - frag_pos)
    NdL = (N*L).sum(1, keepdim=True).clamp(min=0)
    diff = mat.kd * light.color * mat.base_color / torch.pi * NdL
    R = _norm(2*NdL*N - L)
    RdV = (R*V).sum(1, keepdim=True).clamp(min=0)
    spec = mat.ks * light.color * RdV**mat.shininess
    amb = mat.ka * light.color * mat.base_color
    return (amb + diff + spec).clamp(0, 1)


def _phong_envmap(frag_pos, N, cam_pos, mat: PhongMat, light: EnvMapLightGPU, sbatch=128):
    S = light.dirs.shape[0]
    norm_f = (mat.shininess + 2.0) / (2.0 * torch.pi)
    V = _norm(cam_pos - frag_pos)
    NdV = (N*V).sum(1, keepdim=True)             # (M, 1)
    diff = frag_pos.new_zeros(frag_pos.shape)
    spec = frag_pos.new_zeros(frag_pos.shape)
    for si in range(0, S, sbatch):
        L_b = light.dirs[si:si+sbatch]           # (B, 3)
        r_b = light.image_flat[si:si+sbatch]
        dw_b = light.solid_angles[si:si+sbatch]  # (B,)
        NdL = (N @ L_b.T).clamp(min=0)           # (M, B)
        mask = (NdL > 1e-4).float()
        diff += (NdL * dw_b * mask) @ r_b         # (M, 3)
        LdV = V @ L_b.T                           # (M, B)
        RdV = (2.0*NdL*NdV - LdV).clamp(min=0) * mask
        spec += (RdV**mat.shininess * dw_b) @ r_b
    mean_rad = light.image_flat.mean(0)
    return (mat.ka*mean_rad*mat.base_color + mat.kd*diff*mat.base_color/torch.pi + mat.ks*spec*norm_f).clamp(0, 1)


def _phong_sh(frag_pos, N, cam_pos, mat: PhongMat, light: SHLight):
    irr = _sh_irradiance(light.coeffs, N)
    diff = mat.kd * irr * mat.base_color / torch.pi
    V = _norm(cam_pos - frag_pos)
    NdV = (N*V).sum(1, keepdim=True).clamp(min=0)
    R = _norm(2*NdV*N - V)
    L_R = _sh_phong_filtered_radiance(light.coeffs, R, mat.shininess)
    spec = mat.ks * L_R
    return (diff + spec).clamp(0, 1)


# ─────────────────────────────────────────── Cook-Torrance shaders ───────────

def _ct_point(frag_pos, N, cam_pos, mat: PBRMat, light: PointLightGPU):
    F0 = _f0_mat(mat.albedo, mat.metallic)
    alpha2 = mat.roughness**4
    k = (mat.roughness + 1)**2 / 8.0
    V = _norm(cam_pos - frag_pos)
    NdV = (N*V).sum(1).clamp(min=1e-4)            # (M,)
    L = _norm(light.position - frag_pos)
    NdL = (N*L).sum(1)                            # (M,)
    hit = NdL > 1e-4
    out = frag_pos.new_zeros(frag_pos.shape)
    if not hit.any():
        return out
    N_h, V_h, L_h = N[hit], V[hit], L[hit]
    NdV_h = NdV[hit]
    NdL_h = NdL[hit]
    H_v = _norm(L_h + V_h)
    NdH = (N_h*H_v).sum(1).clamp(0, 1)
    VdH = (V_h*H_v).sum(1).clamp(0, 1)
    D = _ggx_D(NdH, alpha2)
    F = _schlick_F(VdH, F0)                       # (M_h, 3)
    G = _smith_G(NdV_h, NdL_h, k)
    spec = F * (D*G / (4*NdV_h + 1e-7))[:, None] * light.color
    k_d = (1 - F) * (1 - mat.metallic)
    diff = k_d * mat.albedo / torch.pi * light.color * NdL_h[:, None]
    out[hit] = (diff + spec).clamp(0, 1)
    return out


def _ct_envmap(frag_pos, N, cam_pos, mat: PBRMat, light: EnvMapLightGPU, sbatch=64):
    F0 = _f0_mat(mat.albedo, mat.metallic)
    alpha2 = mat.roughness**4
    k = alpha2 / 2.0                              # IBL Smith k
    S = light.dirs.shape[0]
    V = _norm(cam_pos - frag_pos)
    NdV = (N*V).sum(1).clamp(min=1e-4)           # (M,)
    spec = frag_pos.new_zeros(frag_pos.shape)
    diff_irr = frag_pos.new_zeros(frag_pos.shape)
    F_sum = frag_pos.new_zeros(frag_pos.shape)
    n_valid = frag_pos.new_zeros(frag_pos.shape[0])
    for si in range(0, S, sbatch):
        L_b = light.dirs[si:si+sbatch]            # (B, 3)
        r_b = light.image_flat[si:si+sbatch]
        dw_b = light.solid_angles[si:si+sbatch]   # (B,)
        NdL = N @ L_b.T                           # (M, B)
        mask = (NdL > 1e-4)
        mf = mask.float()
        NdV_e = NdV[:, None]                      # (M, 1)
        LdV = V @ L_b.T                           # (M, B)
        H_len = (2.0 + 2.0*LdV).clamp(min=1e-8).sqrt()
        NdH = ((NdL + NdV_e) / H_len).clamp(0, 1)
        VdH = ((LdV + 1.0) / H_len).clamp(0, 1)
        D = _ggx_D(NdH, alpha2)                   # (M, B)
        F = _schlick_F(VdH, F0)                   # (M, B, 3)
        G = _smith_G(NdV_e, NdL, k)               # (M, B)
        w = (D*G*dw_b / (4*NdV_e + 1e-7)) * mf   # (M, B)
        spec += (F * w[:, :, None] * r_b).sum(1)
        diff_irr += ((NdL*dw_b*mf) @ r_b)
        F_sum += (F * mf[:, :, None]).sum(1)
        n_valid += mf.sum(1)
    F_mean = F_sum / n_valid[:, None].clamp(min=1)
    k_d = (1 - F_mean) * (1 - mat.metallic)
    diff = k_d * mat.albedo / torch.pi * diff_irr
    return (diff + spec).clamp(0, 1)


# ─────────────────────────────────────────── public shading API ──────────────

def shade_ct_sh(
    normals:   torch.Tensor,                    # [..., 3] unit normals
    # [..., 3] per-pixel albedo in [0, 1]
    albedo:    torch.Tensor,
    # [9, 3]  SH lighting coefficients
    sh_coeffs: torch.Tensor,
    metallic:  Union[float, torch.Tensor] = 0.0,
) -> torch.Tensor:
    """
    Differentiable Cook-Torrance + SH irradiance shading.

    All tensors must reside on the same device. Works on any leading batch
    dimensions (e.g. flat (M,3) or spatial (H,W,3)).
    Returns RGB in [0, 1] with the same leading shape as albedo.
    """
    irr = _sh_irradiance(sh_coeffs, normals)
    k_d = 1.0 - metallic
    # if ((k_d * albedo / torch.pi * irr) > 1.0).any():
    #     print("clamped to < 1 ")
    # elif ((k_d * albedo / torch.pi * irr) < 0).any():
    #     print("clamped to > 0 ")
    # return (k_d * albedo / torch.pi * irr).clamp(0.0, 1.0)
    return (k_d * albedo / torch.pi * irr)


# ─────────────────────────────────────────── geometry extraction ─────────────

def rasterize_geometry(
    mesh,
    camera,
    width:  int = 512,
    height: int = 512,
    smooth: bool = False,
    device: str = "cuda",
) -> tuple:
    """
    Rasterise mesh and return per-pixel geometry tensors without shading.

    Returns
    -------
    normals  : Tensor [H, W, 3]  float32 – unit normals (zero on background)
    frag_pos : Tensor [H, W, 3]  float32 – world-space fragment positions
    mask     : Tensor [H, W]     bool    – True on foreground pixels
    cam_pos  : Tensor [3]        float32 – camera position
    """
    W, H = width, height

    verts = _cuda(mesh.vertices,       device)
    faces = _cuda(mesh.faces,          device, dtype=torch.long)
    fn = _cuda(mesh.normals,        device)
    vn = _cuda(mesh.vertex_normals, device)
    cam_t = _cuda(camera.position.astype(np.float32), device)
    MVP = _cuda((_perspective(camera.fov_deg, W/H) @
                 _look_at(camera.position, camera.target, camera.up)
                 ).astype(np.float32), device)

    sverts = _project(verts, MVP, W, H)
    face_ids, bary = _rasterize(sverts, faces, W, H)

    hit = face_ids >= 0                          # (H*W,)
    normals_flat = verts.new_zeros(H * W, 3)
    frag_flat = verts.new_zeros(H * W, 3)

    if hit.any():
        fp, N = _interp(verts, fn, vn, faces, face_ids, bary, hit, smooth)
        normals_flat[hit] = N
        frag_flat[hit] = fp

    return (
        normals_flat.reshape(H, W, 3),
        frag_flat.reshape(H, W, 3),
        hit.reshape(H, W),
        cam_t,
    )


# ─────────────────────────────────────────── main render ─────────────────────

def render(
    mesh,
    camera,
    material,              # PhongMat | PBRMat
    light,                 # PointLightGPU | EnvMapLightGPU | SHLight
    width:       int = 512,
    height:      int = 512,
    smooth:      bool = False,
    output_path: str = "render_gpu.png",
) -> np.ndarray:
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    W, H = width, height

    verts = _cuda(mesh.vertices,       dev)
    faces = _cuda(mesh.faces,          dev, dtype=torch.long)
    fn = _cuda(mesh.normals,        dev)
    vn = _cuda(mesh.vertex_normals, dev)
    cam_t = _cuda(camera.position.astype(np.float32), dev)
    MVP = _cuda((_perspective(camera.fov_deg, W/H) @
                 _look_at(camera.position, camera.target, camera.up)
                 ).astype(np.float32), dev)

    sverts = _project(verts, MVP, W, H)
    face_ids, bary = _rasterize(sverts, faces, W, H)

    hit = face_ids >= 0                          # (H*W,)
    fb = verts.new_zeros(H*W, 3)

    if hit.any():
        frag_pos, N = _interp(verts, fn, vn, faces,
                              face_ids, bary, hit, smooth)

        def _dev(t):
            return t.to(dev) if isinstance(t, torch.Tensor) else t

        if isinstance(material, PhongMat):
            mat = PhongMat(base_color=_dev(material.base_color),
                           ka=material.ka, kd=material.kd,
                           ks=material.ks, shininess=material.shininess)
            if isinstance(light, PointLightGPU):
                col = _phong_point(frag_pos, N, cam_t, mat,
                                   PointLightGPU(_dev(light.position), _dev(light.color)))
            elif isinstance(light, EnvMapLightGPU):
                col = _phong_envmap(frag_pos, N, cam_t, mat,
                                    EnvMapLightGPU(_dev(light.dirs), _dev(light.image_flat), _dev(light.solid_angles)))
            elif isinstance(light, SHLight):
                col = _phong_sh(frag_pos, N, cam_t, mat,
                                SHLight(_dev(light.coeffs)))
            else:
                raise TypeError(type(light))

        elif isinstance(material, PBRMat):
            mat = PBRMat(albedo=_dev(material.albedo),
                         metallic=material.metallic, roughness=material.roughness)
            if isinstance(light, PointLightGPU):
                col = _ct_point(frag_pos, N, cam_t, mat,
                                PointLightGPU(_dev(light.position), _dev(light.color)))
            elif isinstance(light, EnvMapLightGPU):
                col = _ct_envmap(frag_pos, N, cam_t, mat,
                                 EnvMapLightGPU(_dev(light.dirs), _dev(light.image_flat), _dev(light.solid_angles)))
            elif isinstance(light, SHLight):
                col = shade_ct_sh(N, _dev(material.albedo), _dev(
                    light.coeffs), material.metallic)
            else:
                raise TypeError(type(light))

        else:
            raise TypeError(type(material))

        fb[hit] = col

    if (fb > 0).any():
        print("clamped image")
    img_u8 = (fb.reshape(H, W, 3)*255).clamp(0, 255).byte().cpu().numpy()
    Image.fromarray(img_u8).save(output_path)
    print(f"Saved {output_path}  ({W}×{H})")
    return img_u8
