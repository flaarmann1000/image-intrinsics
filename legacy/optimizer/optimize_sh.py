"""
Physics-based intrinsic decomposition with Spherical Harmonics lighting.

Inspired by nvdiffrecmc (NeurIPS 2022) — "Shape, Light, and Material
Decomposition from Images using Monte Carlo Rendering and Denoising".

In nvdiffrecmc the per-pixel normal is obtained by differentiably rasterising
a reconstructed 3D mesh.  Here we have a 2D image sequence and a monocular
normal map from Marigold, so we skip rasterisation and feed the normals
directly into the same physically-based renderer.

Model
-----
    I_k(p) = albedo(p) ⊙ relu( Y(n(p)) @ c_k )

    n(p)   : unit normal at pixel p  [H, W, 3]  — from Marigold, camera space
    Y(n)   : order-2 SH basis  [H, W, 9]
    c_k    : per-image SH lighting coefficients  [9, 3]  (learnable)
    albedo : shared diffuse reflectance  [H, W, 3]  (learnable)

The order-2 SH basis has 9 coefficients and captures all low-frequency
(distant) illumination.  For Lambertian surfaces the clamped-cosine filter
is a smooth SH kernel; absorbing it into the learned c_k keeps the model
minimal and avoids assuming a known coordinate frame.

Losses
------
    data    : ‖albedo ⊙ relu(Y @ c_k) − I_k‖²  (reconstruction)
    sparse  : isotropic TV on albedo  (piece-wise constant reflectance)
    white   : ‖mean(albedo) − 0.5‖²  (scale ambiguity: anchor mean brightness)
"""

import numpy as np
import torch
import torch.nn.functional as F
import cv2


# ─────────────────────────────────────────────────────────────────────────────
# SH utilities
# ─────────────────────────────────────────────────────────────────────────────

def sh_basis(n: torch.Tensor) -> torch.Tensor:
    """
    Evaluate the order-2 real SH basis at unit normals.

    Parameters
    ----------
    n : [H, W, 3]  (x, y, z)  unit normals

    Returns
    -------
    Y : [H, W, 9]
    """
    nx, ny, nz = n[..., 0], n[..., 1], n[..., 2]
    # from niessner paper
    # return torch.stack([
    #     torch.ones_like(nx),           # l=0  m= 0
    #     ny,                            # l=1  m=-1
    #     nz,                            # l=1  m= 0
    #     nx,                            # l=1  m= 1
    #     nx * ny,                       # l=2  m=-2
    #     ny * nz,                       # l=2  m=-1
    #     (3.0 * nz ** 2 - 1.0) / 2.0,  # l=2  m= 0
    #     nx * nz,                       # l=2  m= 1
    #     (nx ** 2 - ny ** 2) / 2.0,    # l=2  m= 2
    # ], dim=-1)  # [H, W, 9]

    # from dataset gen
    return torch.stack([
        0.282095 * torch.ones_like(nx),
        0.488603 * ny,
        0.488603 * nz,
        0.488603 * nx,
        1.092548 * nx * ny,
        1.092548 * ny * nz,
        0.315392 * (3.0 * nz ** 2 - 1.0),
        1.092548 * nx * nz,
        0.546274 * (nx ** 2 - ny ** 2),
    ], dim=-1)


def render_shading(Y: torch.Tensor, coeffs: torch.Tensor) -> torch.Tensor:
    """
    Compute irradiance  E(p) = relu( Y(p) @ coeffs ).

    Parameters
    ----------
    Y      : [H, W, 9]
    coeffs : [9, 3]   SH coefficients (one set per RGB channel)

    Returns
    -------
    shading : [H, W, 3]
    """
    return F.relu(Y @ coeffs)  # [H, W, 3]


def _tv(x: torch.Tensor) -> torch.Tensor:
    """Isotropic total variation on a [1, C, H, W] tensor."""
    dh = x[..., 1:, :] - x[..., :-1, :]
    dw = x[..., :, 1:] - x[..., :, :-1]
    return (dh ** 2 + 1e-8).sqrt().mean() + (dw ** 2 + 1e-8).sqrt().mean()


# ─────────────────────────────────────────────────────────────────────────────
# Normal map loading
# ─────────────────────────────────────────────────────────────────────────────

def load_normals(path: str, target_hw: tuple) -> torch.Tensor:
    """
    Load normal map, decode to unit normals, resize to target_hw.

    Marigold encodes normals as  n_color = (n_world + 1) / 2 ∈ [0, 1]³,
    so we invert:  n = n_color * 2 − 1  and re-normalise.

    Parameters
    ----------
    path      : path to the PNG file
    target_hw : (H, W) to resize to

    Returns
    -------
    normals : [H, W, 3]  float32 tensor, unit vectors
    """
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    n = img.astype(np.float32) / \
        255.0 if img.dtype == np.uint8 else img.astype(np.float32)

    H, W = target_hw
    if n.shape[:2] != (H, W):
        n = cv2.resize(n, (W, H), interpolation=cv2.INTER_LINEAR)

    n = n * 2.0 - 1.0                           # [0,1] → [-1,1]
    norm = np.linalg.norm(n, axis=-1, keepdims=True).clip(1e-6)
    n /= norm                                    # unit vectors
    return torch.from_numpy(n)                   # [H, W, 3]


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def decompose(images_np, normals_path="marigold/normals.png",
              n_iter=2000, lr=5e-3,
              lambda_sparse=0.5,
              lambda_white=0.1):
    """
    Physics-based intrinsic decomposition using SH-rendered irradiance.

    Parameters
    ----------
    images_np      : list of ndarray [H, W, 3] uint8
    normals_path   : path to Marigold normal map PNG
    n_iter         : Adam iterations
    lr             : learning rate
    lambda_sparse  : TV weight on albedo
    lambda_white   : scale-anchor weight  (‖mean(albedo)−0.5‖²)

    Returns
    -------
    albedo   : ndarray [H, W, 3] float in [0, 1]
    shadings : list of ndarray [H, W, 3] float  (RGB irradiance per image)
    history  : list of scalar loss values (every 200 iters)
    """
    eps = 1e-6
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    N = len(images_np)
    H, W = images_np[0].shape[:2]

    # ── Geometry: normal map ─────────────────────────────────────────────────
    normals = load_normals(normals_path, (H, W)).to(device)  # [H, W, 3]
    Y = sh_basis(normals)                                     # [H, W, 9]

    # ── Images ───────────────────────────────────────────────────────────────
    imgs = [
        # torch.from_numpy(img.astype("float32") / 255.0).to(device)
        torch.from_numpy(img.astype("float32")).to(device)
        for img in images_np
    ]  # each [H, W, 3]

    # ── Learnable parameters ─────────────────────────────────────────────────
    # Albedo: initialise as mean of input images (reasonable starting point)
    # albedo_init = sum(imgs) / N
    albedo_init = (sum(imgs) / N).clamp(0.05, 0.95)
    log_albedo = torch.log(albedo_init).requires_grad_(True)

    # SH coefficients per image: initialise to a gentle ambient (c[0] ≈ 1, rest 0)
    sh_init = torch.zeros(N, 9, 3, device=device)
    # sh_init[:, 0, :] = 0.5                         # ambient term
    sh_init[:, 0, :] = 1.5
    sh_coeffs = sh_init.clone().requires_grad_(True)

    optimizer = torch.optim.Adam([log_albedo, sh_coeffs], lr=lr)

    def to_chw(x):
        return x.permute(2, 0, 1).unsqueeze(0)    # [1, C, H, W]

    history = []
    for i in range(n_iter):
        optimizer.zero_grad()

        albedo = torch.exp(log_albedo)             # [H, W, 3]

        loss_data = torch.tensor(0.0, device=device)
        for k in range(N):
            shading_k = render_shading(Y, sh_coeffs[k])   # [H, W, 3]
            recon_k = albedo * shading_k
            loss_data = loss_data + ((recon_k - imgs[k]) ** 2).mean()

        loss_sparse = lambda_sparse * _tv(to_chw(log_albedo))
        loss_white = lambda_white * ((albedo.mean() - 0.5) ** 2)

        loss = loss_data + loss_sparse + loss_white
        # loss = loss_data
        loss.backward()
        optimizer.step()

        if i % 200 == 0:
            history.append(loss.item())
            print(f"[{i:4d}] total={loss.item():.5f}  "
                  f"data={loss_data.item():.5f}  "
                  f"sparse={loss_sparse.item():.5f}  "
                  f"white={loss_white.item():.5f}")

    def to_np(t):
        return t.detach().cpu().numpy()

    albedo_out = to_np(torch.exp(log_albedo).clamp(0, 1))
    shadings = [
        to_np(render_shading(Y, sh_coeffs[k]).clamp(0))
        for k in range(N)
    ]

    return albedo_out, shadings, history


# ─────────────────────────────────────────────────────────────────────────────
# PyTorch3D renderer variant — same pipeline as create_sh_dataset.py
# ─────────────────────────────────────────────────────────────────────────────

class Pt3dSHShader(torch.nn.Module):
    """
    Flat-face-normal SH shader, identical to HardSHShader in create_sh_dataset.py.
    Pass sh_coeffs=[9, 3] as a keyword argument when calling the renderer.
    """

    def __init__(self, device="cuda", blend_params=None):
        super().__init__()
        from pytorch3d.renderer import BlendParams
        self.blend_params = blend_params or BlendParams(
            background_color=(0, 0, 0))

    def forward(self, fragments, meshes, **kwargs):
        from pytorch3d.renderer import hard_rgb_blend
        coeffs = kwargs["sh_coeffs"]  # [9, 3]

        faces = meshes.faces_packed()
        verts = meshes.verts_packed()
        v0, v1, v2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
        face_normals = F.normalize(
            torch.cross(v1 - v0, v2 - v0, dim=1), dim=1
        )

        pix_to_face = fragments.pix_to_face   # [N, H, W, K]
        mask = pix_to_face >= 0
        pixel_normals = torch.zeros(
            *pix_to_face.shape, 3, device=verts.device, dtype=verts.dtype
        )
        pixel_normals[mask] = face_normals[pix_to_face[mask]]

        basis = sh_basis(pixel_normals)                    # [..., 9]
        lighting = (basis @ coeffs).clamp(min=0.0)        # [..., 3]

        albedo = meshes.sample_textures(fragments)
        colors = albedo * lighting
        colors[~mask] = 0.0

        return hard_rgb_blend(colors, fragments, self.blend_params)


class Pt3dAlbedoShader(torch.nn.Module):
    """Renders raw vertex-color albedo without any lighting."""

    def __init__(self, blend_params=None):
        super().__init__()
        from pytorch3d.renderer import BlendParams
        self.blend_params = blend_params or BlendParams(
            background_color=(0, 0, 0))

    def forward(self, fragments, meshes, **kwargs):
        from pytorch3d.renderer import hard_rgb_blend
        albedo = meshes.sample_textures(fragments)
        return hard_rgb_blend(albedo, fragments, self.blend_params)


def decompose_pytorch3d(images_np, mesh, cameras, raster_settings=None,
                        n_iter=2000, lr=5e-3, lambda_white=0.0,
                        alternating=False):
    """
    Physics-based SH decomposition using the pytorch3d renderer pipeline from
    create_sh_dataset.py.

    Instead of loading a pre-baked normal map PNG, this function rasterizes the
    provided mesh once to obtain per-pixel face normals, then optimises vertex
    albedo and per-image SH coefficients so that

        albedo(p) * relu( Y(n(p)) @ c_k )  ≈  I_k(p)

    where n(p) comes from the mesh geometry (flat face normals, same as the
    dataset renderer) rather than a Marigold image.

    Parameters
    ----------
    images_np       : list of ndarray [H, W, 3] float32 in [0, 1]
    mesh            : pytorch3d Meshes (geometry only; texture is ignored)
    cameras         : pytorch3d camera object
    raster_settings : RasterizationSettings; defaults to image_size=H
    n_iter          : Adam iterations
    lr              : learning rate
    lambda_white    : ‖mean(albedo)−0.5‖² weight
    alternating     : if True, alternate between optimising albedo (odd steps)
                      and SH coefficients (even steps) instead of joint updates

    Returns
    -------
    albedo_image : ndarray [H, W, 3]  float in [0, 1]
    shadings     : list of ndarray [H, W, 3]
    history      : list of scalar losses (every 200 iters)
    """
    from pytorch3d.renderer import RasterizationSettings, MeshRasterizer, TexturesVertex
    from pytorch3d.ops import interpolate_face_attributes
    from pytorch3d.structures import Meshes

    device = mesh.verts_packed().device
    N = len(images_np)
    H, W = images_np[0].shape[:2]

    if raster_settings is None:
        raster_settings = RasterizationSettings(
            image_size=H, blur_radius=0.0, faces_per_pixel=1
        )

    imgs = [torch.from_numpy(img.astype("float32")).to(device)
            for img in images_np]

    # ── Precompute geometry — rasterize once, normals & SH basis are constant ─
    V = mesh.verts_packed().shape[0]
    geom_mesh = Meshes(
        verts=mesh.verts_list(),
        faces=mesh.faces_list(),
        textures=TexturesVertex(
            verts_features=torch.zeros(1, V, 3, device=device)),
    )
    rasterizer = MeshRasterizer(
        cameras=cameras, raster_settings=raster_settings)
    fragments = rasterizer(geom_mesh)

    pix_to_face = fragments.pix_to_face        # [1, H, W, 1]
    bary_coords = fragments.bary_coords       # [1, H, W, 1, 3]
    mask_hw = (pix_to_face[0, :, :, 0] >= 0).unsqueeze(-1)  # [H, W, 1]

    faces_idx = mesh.faces_packed()            # [F, 3]
    verts_pos = mesh.verts_packed()            # [V, 3]
    v0, v1, v2 = verts_pos[faces_idx[:, 0]
                           ], verts_pos[faces_idx[:, 1]], verts_pos[faces_idx[:, 2]]
    face_normals = F.normalize(torch.cross(v1 - v0, v2 - v0, dim=1), dim=1)

    safe_idx = pix_to_face[0, :, :, 0].clamp(min=0)      # [H, W]
    pixel_normals = face_normals[safe_idx] * mask_hw      # [H, W, 3]
    Y = sh_basis(pixel_normals)                           # [H, W, 9]

    # ── Learnable parameters ─────────────────────────────────────────────────
    log_verts_rgb = torch.log(
        torch.full((V, 3), 0.5, device=device)
    ).requires_grad_(True)
    # verts_rgb = torch.full((V, 3), 0.5, device=device).requires_grad_(True)

    # sh_init = torch.zeros(N, 9, 3, device=device)
    # sh_init[:, 0, :] = 1.5
    sh_higher = torch.zeros(N, 8, 3, device=device).requires_grad_(True)  # bands 1-8 only
    # sh_coeffs = sh_init.clone().requires_grad_(True)

    if alternating:
        opt_albedo = torch.optim.Adam([log_verts_rgb], lr=lr)
        # opt_albedo = torch.optim.Adam([verts_rgb], lr=lr)
        # opt_sh = torch.optim.Adam([sh_coeffs],     lr=lr)
        opt_sh = torch.optim.Adam([sh_higher],     lr=lr)
    else:
        # optimizer = torch.optim.Adam([verts_rgb, sh_higher], lr=lr)
        optimizer = torch.optim.Adam([log_verts_rgb, sh_higher], lr=lr)
        # optimizer = torch.optim.Adam([log_verts_rgb, sh_coeffs], lr=lr)

    def _forward():
        verts_rgb = torch.exp(log_verts_rgb)
        # verts_rgb = torch.exp(log_verts_rgb).clamp(0, 1)

        face_verts_rgb = verts_rgb[faces_idx]
        pixel_albedo = interpolate_face_attributes(
            pix_to_face, bary_coords, face_verts_rgb
        )[0, :, :, 0, :] * mask_hw
        loss_data = torch.tensor(0.0, device=device)
        
        band0 = torch.ones(N, 1, 3, device=device)*3          # fixed, no grad
        sh_full = torch.cat([band0, sh_higher], dim=1)       # [N, 9, 3]
        for k in range(N):            
            recon_k = pixel_albedo * (Y @ sh_full[k]).clamp(min=0.0)
            # recon_k = pixel_albedo * (Y @ sh_full[k])
            loss_data = loss_data + ((recon_k - imgs[k]) ** 2).mean()
        # Per-channel anchor: fixes each of the 3 independent scale d.o.f.
        # A scalar mean would only pin one linear combination of λ_R, λ_G, λ_B.
        masked_albedo = pixel_albedo[mask_hw.expand_as(pixel_albedo)]  # only mesh pixels
        loss_white = lambda_white * ((masked_albedo.mean() - 0.5) ** 2)
        # print(f"loss_white: {loss_white} for lambda_white: {lambda_white} as mean is {verts_rgb.mean(dim=0)}")
        return loss_data + loss_white, loss_data, loss_white

    history = []
    for i in range(n_iter):
        if alternating:
            # even steps: fix albedo, update SH
            opt_sh.zero_grad()
            loss, loss_data, loss_white = _forward()
            loss.backward()
            opt_sh.step()
            # odd steps: fix SH, update albedo
            opt_albedo.zero_grad()
            loss, loss_data, loss_white = _forward()
            loss.backward()
            opt_albedo.step()
        else:
            optimizer.zero_grad()
            loss, loss_data, loss_white = _forward()
            loss.backward()
            optimizer.step()

        if i % 200 == 0:
            history.append(loss.item())
            print(f"[{i:4d}] total={loss.item():.5f}  "
                  f"data={loss_data.item():.5f}  "
                  f"white={loss_white.item():.5f}")

    # ── Outputs ───────────────────────────────────────────────────────────────
    verts_rgb_final = torch.exp(log_verts_rgb).clamp(0, 1).detach()
    # verts_rgb_final = verts_rgb.clamp(0, 1).detach()
    pixel_albedo_out = interpolate_face_attributes(
        pix_to_face, bary_coords, verts_rgb_final[faces_idx]
    )[0, :, :, 0, :] * mask_hw
    albedo_out = pixel_albedo_out.cpu().numpy()
    
    band0 = torch.ones(N, 1, 3, device=device) *3          # fixed, no grad
    sh_full = torch.cat([band0, sh_higher], dim=1)       # [N, 9, 3]

    shadings = [
        (F.relu(Y @ sh_full[k].detach()) * mask_hw).cpu().numpy()
        # (F.relu(Y @ sh_coeffs[k].detach()) * mask_hw).cpu().numpy()
        for k in range(N)
    ]

    return albedo_out, shadings, history
