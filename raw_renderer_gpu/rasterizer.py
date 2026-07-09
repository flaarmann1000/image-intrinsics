"""
GPU render() — pure PyTorch, no C++ compilation needed.

Rasterisation is face-batched so GPU memory stays bounded.
EnvMap shading is sample-batched for the same reason.
Everything else (projection, interpolation, BRDF) is fully vectorised
over all hit pixels at once.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from dataclasses import dataclass
from PIL import Image
from typing import Optional, Union
from pathlib import Path


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

def _sh_basis(d, order: int = 2):
    """d: (..., 3) → (..., 9) for order 2, (..., 16) for order 3.

    Band ordering follows the Ramamoorthi convention used throughout:
    [DC | y,z,x | xy,yz,3z²−1,xz,x²−y² | band-3 m=−3..3].
    """
    x, y, z = d[..., 0], d[..., 1], d[..., 2]
    terms = [
        torch.ones_like(x) * 0.282095,
        0.488603*y, 0.488603*z, 0.488603*x,
        1.092548*x*y, 1.092548*y*z,
        0.315392*(3*z**2 - 1),
        1.092548*x*z, 0.546274*(x**2 - y**2),
    ]
    if order >= 3:
        terms += [
            0.590044 * y * (3*x**2 - y**2),
            2.890611 * x * y * z,
            0.457046 * y * (5*z**2 - 1),
            0.373176 * z * (5*z**2 - 3),
            0.457046 * x * (5*z**2 - 1),
            1.445306 * z * (x**2 - y**2),
            0.590044 * x * (x**2 - 3*y**2),
        ]
    return torch.stack(terms, dim=-1)              # (..., 9|16)


def _sh_order_of(coeffs_t) -> int:
    """Infer the SH order from the coefficient count (9 → 2, 16 → 3)."""
    n = coeffs_t.shape[0]
    if n == 9:
        return 2
    if n == 16:
        return 3
    raise ValueError(f"Expected 9 (order 2) or 16 (order 3) SH coefficients, got {n}")


def _sh_irradiance(coeffs_t, N):
    """ZH band-limiting weights for diffuse — coeffs_t: (9|16,3); N: (...,3) → (...,3).

    Band 3 has A₃ = 0 (odd Lambertian ZH bands above 1 vanish), so order-3
    lighting affects the diffuse term only through its lower bands.
    """
    order = _sh_order_of(coeffs_t)
    Y = _sh_basis(N, order=order)                  # (..., 9|16)
    A_vals = [
        torch.pi,
        2*torch.pi/3, 2*torch.pi/3, 2*torch.pi/3,
        torch.pi/4,   torch.pi/4,   torch.pi/4, torch.pi/4, torch.pi/4,
    ]
    if order >= 3:
        A_vals += [0.0] * 7
    A = N.new_tensor(A_vals)
    return ((A * Y) @ coeffs_t).clamp(min=0)       # (..., 3)


def _sh_phong_filtered_radiance(coeffs_t, dirs, shininess):
    """Phong-lobe SH filter — coeffs_t: (9,3); dirs: (...,3) → (...,3)"""
    Y = _sh_basis(dirs)
    # accept plain float or 0-d tensor as well as per-pixel (..., 1) tensor
    if not torch.is_tensor(shininess):
        shininess = torch.tensor(shininess, dtype=Y.dtype, device=Y.device)
    if shininess.dim() == 0:
        shininess = shininess.unsqueeze(-1)   # → (1,), broadcasts over pixels
    B_0 = 2 * torch.pi / (shininess + 1)
    B_1 = 2 * torch.pi / (shininess + 2)
    B_2 = torch.pi * (3.0 / (shininess + 3) - 1.0 / (shininess + 1))
    norm = (shininess + 2) / (2 * torch.pi)
    B = torch.cat([B_0, B_1, B_1, B_1, B_2, B_2, B_2, B_2, B_2], dim=-1) * norm
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


# def shade_phong_sh(frag_pos, N, cam_pos, mat: PhongMat, light: SHLight):
#     irr = _sh_irradiance(light.coeffs, N)
#     diff = mat.kd * irr * mat.base_color / torch.pi
#     V = _norm(cam_pos - frag_pos)
#     NdV = (N*V).sum(1, keepdim=True).clamp(min=0)
#     R = _norm(2*NdV*N - V)
#     L_R = _sh_phong_filtered_radiance(light.coeffs, R, mat.shininess)
#     spec = mat.ks * L_R
#     return (diff + spec).clamp(0, 1)


def shade_phong_sh(V, N, ka, kd, ks, shininess, base_color, coeffs,
                   return_components: bool = False):
    irr  = _sh_irradiance(coeffs, N)
    diff = kd * irr * base_color / torch.pi
    NdV  = (N * V).sum(1, keepdim=True).clamp(min=0)
    R    = _norm(2 * NdV * N - V)
    ks_pos = (ks > 0) if not torch.is_tensor(ks) else torch.any(ks > 0)
    if ks_pos:
        L_R  = _sh_phong_filtered_radiance(coeffs, R, shininess)
        spec = ks * L_R
    else:
        L_R  = diff.new_zeros(diff.shape)
        spec = diff.new_zeros(diff.shape)
    composite = diff + spec
    if not return_components:
        return composite
    return composite, {"irr": irr, "diff": diff, "R": R, "L_spec": L_R, "spec": spec}


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

    # roughness = max(float(mat.roughness), 0.12)    
    roughness = float(mat.roughness)
    alpha2 = roughness ** 4
    k = alpha2 / 2.0

    S = light.dirs.shape[0]
    V = _norm(cam_pos - frag_pos)
    NdV = (N * V).sum(1).clamp(min=1e-4)

    spec = frag_pos.new_zeros(frag_pos.shape)
    diff_irr = frag_pos.new_zeros(frag_pos.shape)
    F_sum = frag_pos.new_zeros(frag_pos.shape)
    n_valid = frag_pos.new_zeros(frag_pos.shape[0])

    for si in range(0, S, sbatch):
        L_b = light.dirs[si:si + sbatch]
        r_b = light.image_flat[si:si + sbatch]
        dw_b = light.solid_angles[si:si + sbatch]

        NdL_raw = N @ L_b.T
        mask = NdL_raw > 1e-4
        mf = mask.float()
        NdL = NdL_raw.clamp(min=1e-4)

        NdV_e = NdV[:, None]
        LdV = V @ L_b.T

        H_len = (2.0 + 2.0 * LdV).clamp(min=1e-8).sqrt()
        NdH = ((NdL_raw + NdV_e) / H_len).clamp(0, 1)
        VdH = ((LdV + 1.0) / H_len).clamp(0, 1)

        D = _ggx_D(NdH, alpha2)
        F = _schlick_F(VdH, F0)
        G = _smith_G(NdV_e, NdL, k)

        w = (D * G * dw_b / (4 * NdV_e + 1e-7)) * mf

        spec += (F * w[:, :, None] * r_b).sum(1)
        diff_irr += ((NdL_raw.clamp(min=0) * dw_b * mf) @ r_b)

        F_sum += (F * mf[:, :, None]).sum(1)
        n_valid += mf.sum(1)

    F_mean = F_sum / n_valid[:, None].clamp(min=1)
    k_d = (1 - F_mean) * (1 - mat.metallic)
    diff = k_d * mat.albedo / torch.pi * diff_irr

    return (diff + spec).clamp(0, 1)


# ─────────────────────────────────────────── public shading API ──────────────

# def shade_ct_sh(
#     view: torch.Tensor,
#     normals:   torch.Tensor,                    # [..., 3] unit normals
#     # [..., 3] per-pixel albedo in [0, 1]
#     albedo:    torch.Tensor,
#     # [9, 3]  SH lighting coefficients
#     sh_coeffs: torch.Tensor,
#     metallic:  Union[float, torch.Tensor] = 0.0,
#     roughness: Union[float, torch.Tensor] = 0.0,
# ) -> torch.Tensor:
#     """
#     Differentiable Cook-Torrance + SH irradiance shading.

#     All tensors must reside on the same device. Works on any leading batch
#     dimensions (e.g. flat (M,3) or spatial (H,W,3)).
#     Returns RGB in [0, 1] with the same leading shape as albedo.
#     """
#     irr = _sh_irradiance(sh_coeffs, normals)
#     k_d = 1.0 - metallic

#     return (k_d * albedo / torch.pi * irr)


# ── LUT configuration ─────────────────────────────────────────────────────────

_LUT_PATH = Path(__file__).parent / "ggx_sh_lut.npz"
_LUT_N_ROUGH = 512    # roughness resolution (uniform grid over [0, 1])
_LUT_N_THETA = 8192   # integration resolution

_LUT_PATH_O3 = Path(__file__).parent / "ggx_sh_lut_o3.npz"
_LUT_CACHE: dict = {}                       # n_bands -> (N, n_bands) tensor


# ── LUT computation ───────────────────────────────────────────────────────────

def _compute_ggx_sh_lut(
    n_roughness: int = _LUT_N_ROUGH,
    n_theta:     int = _LUT_N_THETA,
    n_bands:     int = 3,
) -> np.ndarray:
    """
    Numerically integrate GGX zonal SH band weights for bands 0..n_bands-1.

    Kernel:   w(θ) = D_GGX(n·h = cos(θ/2), α)        ← no cos θ factor
    Bands:    h_l = 2π ∫₀^{π/2} w(θ) P_l(cosθ) sinθ dθ
    Stored:   raw h_l (no h_1 normalisation)

    Limit:  α → 0  ⇒  D → δ  ⇒  ∫D dω_l = 4  ⇒  h_l = 4 for all l.
    """

    theta = np.linspace(0.0, np.pi / 2.0, n_theta, dtype=np.float64)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    cos_th2 = np.cos(theta / 2.0)**2

    Pl = [
        np.ones_like(cos_t),
        cos_t,
        0.5 * (3.0 * cos_t**2 - 1.0),
        0.5 * (5.0 * cos_t**3 - 3.0 * cos_t),
    ][:n_bands]

    lut = np.zeros((n_roughness, n_bands), dtype=np.float32)

    for i in range(n_roughness):
        r = float(i) / max(n_roughness - 1, 1)
        a = r ** 2
        a2 = a ** 2

        if a2 < 1e-12:                       # delta limit
            lut[i] = [4.0] * n_bands
            continue

        D = a2 / (np.pi * (cos_th2 * (a2 - 1.0) + 1.0) ** 2)
        # NO cos_t factor in the kernel
        h = np.array([
            2.0 * np.pi * np.trapezoid(D * p * sin_t, theta)
            for p in Pl
        ])
        lut[i] = h                            # raw, unnormalised

    return lut


# ── LUT loading / caching ─────────────────────────────────────────────────────

def _get_ggx_sh_lut(
    device:     torch.device,
    cache_path: Optional[Path] = None,
    n_bands:    int = 3,
) -> torch.Tensor:
    """
    Return the GGX SH LUT on `device`, with `n_bands` zonal bands (3 for
    order-2 SH, 4 for order-3).

    First call:  computes the LUT, saves it to its cache file, caches in memory.
    Later calls: returns the in-memory tensor (moved to `device` if needed).
    """
    if n_bands not in _LUT_CACHE:
        path = cache_path if cache_path is not None else (
            _LUT_PATH if n_bands == 3 else _LUT_PATH_O3)
        if path.exists():
            data = np.load(path)
            _LUT_CACHE[n_bands] = torch.from_numpy(data["lut"])
        else:
            print(f"[ggx_sh] {n_bands}-band LUT not found — computing … ", end="", flush=True)
            lut_np = _compute_ggx_sh_lut(n_bands=n_bands)
            np.savez_compressed(path, lut=lut_np)
            _LUT_CACHE[n_bands] = torch.from_numpy(lut_np)
            print(f"done  →  saved to {path}")

    return _LUT_CACHE[n_bands].to(device)


# ── LUT interpolation ─────────────────────────────────────────────────────────

def _lut_lookup(lut: torch.Tensor, roughness: torch.Tensor) -> torch.Tensor:
    """
    Differentiable linear interpolation into a uniform 1-D LUT.

    Parameters
    ----------
    lut       : (N, C)   table indexed uniformly over roughness ∈ [0, 1]
    roughness : (...)    values in [0, 1], any shape

    Returns
    -------
    (..., C)  — gradients flow through `roughness` via the lerp weight `t`.
    """
    N = lut.shape[0]
    idx_f = roughness.clamp(0.0, 1.0) * (N - 1)
    idx_lo = idx_f.long().clamp(0, N - 2)              # integer, no grad needed
    t = (idx_f - idx_lo.float()).unsqueeze(-1)     # (..., 1)  differentiable
    return lut[idx_lo] + t * (lut[idx_lo + 1] - lut[idx_lo])  # (..., C)


# ── GGX SH specular filter ────────────────────────────────────────────────────

def _sh_ggx_filtered_radiance(
    coeffs_t:  torch.Tensor,   # (9|16, 3)
    dirs:      torch.Tensor,   # (..., 3)  unit reflection directions
    roughness: torch.Tensor,   # (..., 1)  in [0, 1]
    lut:       torch.Tensor,   # (N, 3|4)
) -> torch.Tensor:
    """
    GGX-lobe SH filter — analogue of _sh_phong_filtered_radiance.

    Convolves the SH-encoded environment with the GGX lobe centred on `dirs`,
    with per-element width controlled by `roughness`. The SH order is inferred
    from the coefficient count; order-3 needs a 4-band LUT.

    Returns (..., 3), clamped to ≥ 0.
    """
    order = _sh_order_of(coeffs_t)
    if lut.shape[1] < order + 1:
        raise ValueError(f"order-{order} SH needs a {order + 1}-band GGX LUT, "
                         f"got {lut.shape[1]} bands")
    Y = _sh_basis(dirs, order=order)             # (..., 9|16)
    Bvals = _lut_lookup(lut, roughness.squeeze(-1))  # (..., n_bands)

    parts = [Bvals[..., 0:1],                    # band 0 (1 coeff)
             Bvals[..., 1:2].expand(*Bvals.shape[:-1], 3),
             Bvals[..., 2:3].expand(*Bvals.shape[:-1], 5)]
    if order >= 3:
        parts.append(Bvals[..., 3:4].expand(*Bvals.shape[:-1], 7))
    B = torch.cat(parts, dim=-1)                 # (..., 9|16)

    return ((B * Y) @ coeffs_t).clamp(min=0.0)   # (..., 3)


# ── Main shader ───────────────────────────────────────────────────────────────

def shade_ct_sh(
    view:             torch.Tensor,
    normals:          torch.Tensor,
    albedo:           torch.Tensor,
    sh_coeffs:        torch.Tensor,
    metallic:         Union[float, torch.Tensor] = 0.0,
    roughness:        Union[float, torch.Tensor] = 0.5,
    lut:              Optional[torch.Tensor] = None,
    diffuse_fresnel:  bool = True,
    return_components: bool = False,
) -> torch.Tensor:
    """
    Differentiable Cook-Torrance shading with SH-environment specular.

    Parameters
    ----------
    view      : (..., 3)  unit view directions (camera → fragment, or reversed)
    normals   : (..., 3)  unit surface normals
    albedo    : (..., 3)  base colour in [0, 1]
    sh_coeffs : (9, 3)    SH lighting coefficients
    metallic  : scalar or (..., 1)  metallic factor in [0, 1]
    roughness : scalar or (..., 1)  perceptual roughness in [0, 1]
    lut       : optional pre-loaded GGX SH LUT; fetched/computed if None
    diffuse_fresnel : if True, multiply the diffuse by (1-F) on top of
                (1-metallic). Default False to match Blender's Principled BSDF.

    Returns
    -------
    (..., 3)  shaded RGB
    """
    device = albedo.device

    def _as_tensor(x: Union[float, torch.Tensor]) -> torch.Tensor:
        if torch.is_tensor(x):
            return x.to(device)
        return torch.full(
            albedo.shape[:-1] + (1,), float(x),
            dtype=albedo.dtype, device=device,
        )

    roughness = _as_tensor(roughness)
    metallic = _as_tensor(metallic)

    if lut is None:
        lut = _get_ggx_sh_lut(device, n_bands=_sh_order_of(sh_coeffs) + 1)

    # ── Fresnel-Schlick ───────────────────────────────────────────────────────
    # F0: dielectrics reflect ~4%; metals reflect with their albedo tint
    f0 = 0.04 * (1.0 - metallic) + albedo * metallic          # (..., 3)
    NdotV_raw = (normals * view).sum(-1, keepdim=True)
    NdotV = NdotV_raw.clamp(min=0.0)
    F = f0 + (1.0 - f0) * (1.0 - NdotV).pow(5)              # (..., 3)

    # ── Smith G1 — view term  (IBL variant: k = α⁴/2) ────────────────────────
    alpha = roughness ** 2                          # α = roughness²
    k = alpha ** 2 / 2.0                        # k = α²/2
    G1 = NdotV / (NdotV * (1.0 - k) + k + 1e-6)  # (..., 1)

    # ── Diffuse (Lambertian, energy-conserving) ───────────────────────────────
    irr = _sh_irradiance(sh_coeffs, normals)       # (..., 3)
    # Diffuse weight. Default matches Blender's Principled BSDF, which weights the
    # diffuse layer by (1-metallic) only. The (1-F) "energy taken by specular"
    # cross-term darkens the diffuse and was the main cause of the shader being
    # too dark vs BlenderProc; enable diffuse_fresnel=True to restore it.
    k_d = (1.0 - metallic)
    if diffuse_fresnel:
        k_d = (1.0 - F) * k_d
    diff = k_d * albedo / torch.pi * irr

    # ── Specular (GGX SH) ─────────────────────────────────────────────────────
    R = _norm(2.0 * NdotV_raw * normals - view)
    L_spec = _sh_ggx_filtered_radiance(sh_coeffs, R, roughness, lut)
    spec = F * G1 * L_spec / 4.0
    # spec = G1 * L_spec / 4.0

    # import matplotlib.pyplot as plt
    # n = normals[NdotV.squeeze() <= 0][:1].numpy()   # shape: (N, 3)
    # v = view[NdotV.squeeze() <= 0][:1].numpy()   # shape: (N, 3)

    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')

    # ax.quiver(
    #     np.zeros(len(n)),  # x origins
    #     np.zeros(len(n)),  # y origins
    #     np.zeros(len(n)),  # z origins
    #     n[:, 0],            # dx
    #     n[:, 1],            # dy
    #     n[:, 2],            # dz
    #     color="red",
    # )

    # ax.quiver(
    #     np.zeros(len(v)),  # x origins
    #     np.zeros(len(v)),  # y origins
    #     np.zeros(len(v)),  # z origins
    #     v[:, 0],            # dx
    #     v[:, 1],            # dy
    #     v[:, 2],            # dz
    # )

    # plt.show()

    front = (NdotV_raw > 0).to(albedo.dtype)
    composite = (diff + spec) * front
    if not return_components:
        return composite
    return composite, {
        "NdotV":  NdotV,
        "F":      F,
        "G1":     G1,
        "irr":    irr,
        "k_d":    k_d,
        "R":      R,
        "L_spec": L_spec,
        "diff":   diff,
        "spec":   spec,
    }


# ─────────────────────────────────────────── env-map CT shader ───────────────

def _env_bilinear(env_pixels: torch.Tensor, dirs: torch.Tensor) -> torch.Tensor:
    """Differentiable bilinear lookup of a flat (H*W, 3) equirect env map
    (EnvMap layout: W = 2H, texel centers at half-integers, pole = +Y,
    u = atan2(z, x)) at arbitrary unit directions (..., 3).

    Gradients flow into env_pixels (gather) AND into dirs (through the
    bilinear weights), so lobe-shape parameters like roughness receive
    gradients from the env content."""
    P = env_pixels.shape[0]
    H = int(math.sqrt(P / 2))
    W = 2 * H
    assert H * W == P, f"env_pixels ({P}) is not an equirect W=2H grid"
    x, y, z = dirs[..., 0], dirs[..., 1], dirs[..., 2]
    u = (torch.atan2(z, x) / (2 * math.pi) + 0.5) * W - 0.5      # texel coords
    v = (torch.acos(y.clamp(-1.0, 1.0)) / math.pi) * H - 0.5
    u0 = torch.floor(u);  wu = (u - u0).unsqueeze(-1)
    v0 = torch.floor(v);  wv = (v - v0).unsqueeze(-1)
    iu0 = (u0.long() % W); iu1 = (iu0 + 1) % W                    # wrap in phi
    iv0 = v0.long().clamp(0, H - 1); iv1 = (iv0 + 1).clamp(0, H - 1)  # clamp at poles
    p00 = env_pixels[iv0 * W + iu0]
    p01 = env_pixels[iv0 * W + iu1]
    p10 = env_pixels[iv1 * W + iu0]
    p11 = env_pixels[iv1 * W + iu1]
    return ((p00 * (1 - wu) + p01 * wu) * (1 - wv)
            + (p10 * (1 - wu) + p11 * wu) * wv)


def _spec_ggx_importance(
    view:        torch.Tensor,   # (M, 3)
    normals:     torch.Tensor,   # (M, 3)
    env_pixels:  torch.Tensor,   # (H*W, 3)
    F0:          torch.Tensor,   # (M, 3)
    NdotV_raw:   torch.Tensor,   # (M, 1)
    alpha2:      torch.Tensor,   # (M, 1) or scalar tensor — α² with α = roughness²
    k_smith:     torch.Tensor,   # (M, 1) — Smith k, α²/2 (matches quadrature path)
    n_samples:   int = 64,
) -> torch.Tensor:
    """GGX half-vector importance sampling of the specular env integral.

    Deterministic stratified samples (identical across pixels and calls, so
    L-BFGS closures see a noise-free objective): xi1 stratified, xi2 golden-
    ratio sequence. The D term cancels against the pdf, leaving the classic
    estimator  spec = mean_s F(VdH) G(NdV, NdL) VdH / (NdH NdV) * L_env(L).
    Valid at all roughness values (no texel-grid aliasing)."""
    M      = normals.shape[0]
    device = normals.device
    dtype  = normals.dtype
    NdotV  = NdotV_raw.clamp(min=1e-4)                       # (M, 1)
    alpha  = alpha2.clamp(min=1e-12).sqrt()                  # (M, 1) — GGX α

    # tangent frame around N (branchless up-vector selection)
    up   = torch.where(normals[:, 2:3].abs() < 0.999,
                       torch.tensor([0.0, 0.0, 1.0], device=device, dtype=dtype).expand_as(normals),
                       torch.tensor([1.0, 0.0, 0.0], device=device, dtype=dtype).expand_as(normals))
    t1 = _norm(torch.cross(up, normals, dim=-1))             # (M, 3)
    t2 = torch.cross(normals, t1, dim=-1)                    # (M, 3)

    s   = torch.arange(n_samples, device=device, dtype=dtype)
    xi1 = (s + 0.5) / n_samples                              # (S,) stratified
    xi2 = torch.frac(s * 0.6180339887498949)                 # (S,) golden ratio

    # GGX D sampling: cosθ_h = sqrt((1-ξ)/(1+(α²-1)ξ))
    cos_th = ((1.0 - xi1) / (1.0 + (alpha2 - 1.0) * xi1)).clamp(min=0.0).sqrt()  # (M, S)
    sin_th = (1.0 - cos_th ** 2).clamp(min=0.0).sqrt()
    phi    = 2 * math.pi * xi2                                # (S,)
    Hvec = (sin_th * torch.cos(phi)).unsqueeze(-1) * t1.unsqueeze(1) \
         + (sin_th * torch.sin(phi)).unsqueeze(-1) * t2.unsqueeze(1) \
         + cos_th.unsqueeze(-1) * normals.unsqueeze(1)        # (M, S, 3)

    VdH = (view.unsqueeze(1) * Hvec).sum(-1)                  # (M, S)
    L   = 2.0 * VdH.unsqueeze(-1) * Hvec - view.unsqueeze(1)  # (M, S, 3)
    NdL_raw = (normals.unsqueeze(1) * L).sum(-1)              # (M, S)
    mf  = ((NdL_raw > 1e-4) & (VdH > 1e-4)).to(dtype)
    NdL = NdL_raw.clamp(min=1e-4)
    NdH = (normals.unsqueeze(1) * Hvec).sum(-1).clamp(min=1e-4)

    F = _schlick_F(VdH.clamp(0, 1), F0.unsqueeze(1))          # (M, S, 3)
    G = _smith_G(NdotV, NdL, k_smith)                         # (M, S)

    radiance = _env_bilinear(env_pixels, L)                   # (M, S, 3)
    w = (G * VdH / (NdH * NdotV + 1e-7)) * mf                 # (M, S)
    return (F * w.unsqueeze(-1) * radiance).mean(1)           # (M, 3)


def shade_ct_env(
    view:              torch.Tensor,
    normals:           torch.Tensor,
    albedo:            torch.Tensor,
    env_pixels:        torch.Tensor,       # (P, 3) — may be learnable
    env_dirs:          torch.Tensor,       # (P, 3)
    env_dw:            torch.Tensor,       # (P,)
    metallic:          Union[float, torch.Tensor] = 0.0,
    roughness:         Union[float, torch.Tensor] = 0.5,
    sbatch:            int = 64,
    diffuse_fresnel:   bool = True,
    return_components: bool = False,
    spec_importance:   bool = False,
    spec_samples:      int = 64,
) -> torch.Tensor:
    """
    Differentiable Cook-Torrance shading with explicit env-map integration.

    Drop-in companion to shade_ct_sh: same flat (M, 3) interface, same front-face
    masking, but integrates over explicit (P, 3) env-map samples instead of SH.
    Gradients flow through env_pixels, albedo, metallic, and roughness.

    Parameters
    ----------
    view, normals, albedo : (M, 3)  flat foreground-pixel arrays
    env_pixels            : (P, 3)  env-map radiance (learnable)
    env_dirs              : (P, 3)  sample directions (unit sphere)
    env_dw                : (P,)    solid angles
    metallic, roughness   : scalar or (M, 1) Tensor
    sbatch                : env-map samples per memory batch
    diffuse_fresnel       : if True, multiply diffuse by (1-F) on top of
                            (1-metallic). Default False (matches Blender).
    return_components     : if True, return (composite, dict)
    spec_importance       : if True, compute the SPECULAR term by GGX
                            half-vector importance sampling (spec_samples
                            deterministic stratified samples, bilinear env
                            lookup) instead of the texel-grid Riemann sum.
                            The Riemann sum aliases for lobes narrower than a
                            texel (roughness ≲ 0.3 on the 32x64 grid, with a
                            float32 instability peaking near roughness 0.1);
                            importance sampling stays valid at ALL roughness.
                            Requires env_pixels on an equirect grid with
                            W = 2H (the EnvMap layout). Diffuse is unchanged.

    Returns
    -------
    (M, 3) shaded RGB, optionally paired with a components dict containing
    NdotV (M,1), F_avg (M,3), irr (M,3), k_d (M,3), diff (M,3), spec (M,3).
    """
    device = albedo.device

    def _as_tensor(x):
        if torch.is_tensor(x):
            return x.to(device)
        return torch.full(
            albedo.shape[:-1] + (1,), float(x),
            dtype=albedo.dtype, device=device,
        )

    metallic_t  = _as_tensor(metallic)
    # roughness_t = _as_tensor(roughness).clamp(min=0.05)
    roughness_t = _as_tensor(roughness)
    alpha2      = roughness_t ** 4
    k_smith     = alpha2 / 2.0

    F0        = _f0_mat(albedo, metallic_t)                   # (M, 3)
    NdotV_raw = (normals * view).sum(-1, keepdim=True)        # (M, 1)
    NdotV     = NdotV_raw.clamp(min=1e-4)                     # (M, 1)

    P         = env_pixels.shape[0]
    spec      = albedo.new_zeros(albedo.shape)
    diff_irr  = albedo.new_zeros(albedo.shape)
    F_sum     = albedo.new_zeros(albedo.shape)
    n_valid   = albedo.new_zeros(albedo.shape[0])

    for si in range(0, P, sbatch):
        L_b  = env_dirs[si:si + sbatch]             # (B, 3)
        r_b  = env_pixels[si:si + sbatch]           # (B, 3)
        dw_b = env_dw[si:si + sbatch]               # (B,)

        NdL_raw = normals @ L_b.T                   # (M, B)
        mf      = (NdL_raw > 1e-4).float()
        NdL     = NdL_raw.clamp(min=1e-4)

        LdV   = view @ L_b.T                        # (M, B)
        H_len = (2.0 + 2.0 * LdV).clamp(min=1e-8).sqrt()
        NdH   = ((NdL_raw + NdotV) / H_len).clamp(0, 1)
        VdH   = ((LdV + 1.0)       / H_len).clamp(0, 1)

        F = _schlick_F(VdH, F0.unsqueeze(1))                  # (M, B, 3)

        if not spec_importance:
            D = _ggx_D(NdH, alpha2)                           # (M, B)
            G = _smith_G(NdotV, NdL, k_smith)                 # (M, B)
            w = (D * G * dw_b / (4 * NdotV + 1e-7)) * mf      # (M, B)
            spec += (F * w[:, :, None] * r_b).sum(1)
        diff_irr += (NdL_raw.clamp(min=0) * dw_b * mf) @ r_b
        F_sum    += (F * mf[:, :, None]).sum(1)
        n_valid  += mf.sum(1)

    if spec_importance:
        spec = _spec_ggx_importance(view, normals, env_pixels, F0,
                                    NdotV_raw, alpha2, k_smith, spec_samples)

    F_mean    = F_sum / n_valid[:, None].clamp(min=1)         # (M, 3)
    # Diffuse weight: (1-metallic) by default (matches Blender's Principled BSDF);
    # pass diffuse_fresnel=True to also apply the (1-F) cross-term.
    k_d       = (1.0 - metallic_t)                            # (M, 3)
    if diffuse_fresnel:
        k_d = (1.0 - F_mean) * k_d
    diff      = k_d * albedo / torch.pi * diff_irr            # (M, 3)
    front     = (NdotV_raw > 0).to(albedo.dtype)
    composite = (diff + spec) * front
    if not return_components:
        return composite
    return composite, {
        "NdotV": NdotV,
        "F_avg": F_mean,
        "irr":   diff_irr,
        "k_d":   k_d,
        "diff":  diff,
        "spec":  spec,
    }


# ─────────────────────────────────────────── env-map Phong shader ────────────

def shade_phong_env(
    view:              torch.Tensor,
    normals:           torch.Tensor,
    albedo:            torch.Tensor,
    env_pixels:        torch.Tensor,       # (P, 3) — may be learnable
    env_dirs:          torch.Tensor,       # (P, 3)
    env_dw:            torch.Tensor,       # (P,)
    ka:                float = 0.0,
    kd:                float = 1.0,
    ks:                Union[float, torch.Tensor] = 0.5,
    shininess:         Union[float, torch.Tensor] = 32.0,
    sbatch:            int = 128,
    return_components: bool = False,
) -> torch.Tensor:
    """
    Differentiable Phong shading with explicit env-map integration.

    Companion to shade_phong_sh: same flat (M, 3) interface, integrates over
    explicit (P, 3) env-map samples. Gradients flow through env_pixels, albedo,
    ks, and shininess.

    Parameters
    ----------
    view, normals, albedo : (M, 3)
    env_pixels            : (P, 3)  env-map radiance (learnable)
    env_dirs              : (P, 3)  sample directions
    env_dw                : (P,)    solid angles
    ka, kd                : scene-level ambient / diffuse scalars
    ks                    : scalar or (M, 1) specular coefficient
    shininess             : scalar or (M, 1) Phong exponent
    return_components     : if True, return (composite, dict)
    """
    device = albedo.device

    def _as_tensor(x):
        if torch.is_tensor(x):
            return x.to(device)
        return torch.full(
            albedo.shape[:-1] + (1,), float(x),
            dtype=albedo.dtype, device=device,
        )

    ks_t        = _as_tensor(ks)
    shininess_t = _as_tensor(shininess)
    norm_f      = (shininess_t + 2.0) / (2.0 * torch.pi)  # (M,1) or scalar

    P           = env_pixels.shape[0]
    NdotV_raw   = (normals * view).sum(-1, keepdim=True)   # (M, 1)
    front       = (NdotV_raw > 0).to(albedo.dtype)

    diff_irr    = albedo.new_zeros(albedo.shape)
    spec        = albedo.new_zeros(albedo.shape)

    for si in range(0, P, sbatch):
        L_b  = env_dirs[si:si + sbatch]    # (B, 3)
        r_b  = env_pixels[si:si + sbatch]  # (B, 3)
        dw_b = env_dw[si:si + sbatch]      # (B,)

        NdL_raw = normals @ L_b.T          # (M, B)
        mf      = (NdL_raw > 1e-4).float()
        LdV     = view @ L_b.T             # (M, B)

        # R·V = 2*(N·L)*(N·V) - L·V, clamped and masked to front-facing lights
        RdV     = (2.0 * NdL_raw.clamp(min=0) * NdotV_raw - LdV).clamp(min=0) * mf

        # spec_contrib: (M, B)  —  shininess_t (M,1) broadcasts over B
        spec_contrib = RdV ** shininess_t * norm_f * dw_b * mf
        spec     += spec_contrib @ r_b                            # (M, 3)
        diff_irr += (NdL_raw.clamp(min=0) * dw_b * mf) @ r_b    # (M, 3)

    mean_rad  = env_pixels.mean(0)
    diff      = kd * albedo / torch.pi * diff_irr
    spec_out  = ks_t * spec
    amb       = ka * mean_rad * albedo
    composite = (amb + diff + spec_out) * front

    if not return_components:
        return composite
    return composite, {
        "NdotV": NdotV_raw,
        "irr":   diff_irr,
        "diff":  diff,
        "spec":  spec_out,
    }


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
    # amp: float = 1,
    amp: float = 5,
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
    # fb = verts.new_zeros(H*W, 3)
    fb = verts.new_ones(H*W, 3)

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
                # def shade_phong_sh(frag_pos, N, cam_pos, ka, kd, ks, shininess, base_color, coeffs):
                V = _norm(cam_t - frag_pos)
                col = shade_phong_sh(V, N, mat.ka, mat.kd,
                                     mat.ks, mat.shininess, mat.base_color, _dev(light.coeffs))
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
                V = _norm(cam_t - frag_pos)
                col = shade_ct_sh(V, N, _dev(material.albedo), _dev(
                    light.coeffs), material.metallic, material.roughness)
            else:
                raise TypeError(type(light))

        else:
            raise TypeError(type(material))

        fb[hit] = col * amp

    img_u8 = (fb.reshape(H, W, 3)*255).clamp(0, 255).byte().cpu().numpy()
    Image.fromarray(img_u8).save(output_path)
    print(f"Saved {output_path}  ({W}×{H})")
    return img_u8
