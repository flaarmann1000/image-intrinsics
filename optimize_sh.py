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
    return torch.stack([
        torch.ones_like(nx),           # l=0  m= 0
        ny,                            # l=1  m=-1
        nz,                            # l=1  m= 0
        nx,                            # l=1  m= 1
        nx * ny,                       # l=2  m=-2
        ny * nz,                       # l=2  m=-1
        (3.0 * nz ** 2 - 1.0) / 2.0,  # l=2  m= 0
        nx * nz,                       # l=2  m= 1
        (nx ** 2 - ny ** 2) / 2.0,    # l=2  m= 2
    ], dim=-1)  # [H, W, 9]


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
    Load a Marigold normal map, decode to unit normals, resize to target_hw.

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
    n = img.astype(np.float32) / 255.0 if img.dtype == np.uint8 else img.astype(np.float32)

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
        torch.from_numpy(img.astype("float32") / 255.0).to(device)
        for img in images_np
    ]  # each [H, W, 3]

    # ── Learnable parameters ─────────────────────────────────────────────────
    # Albedo: initialise as mean of input images (reasonable starting point)
    albedo_init = sum(imgs) / N
    log_albedo  = torch.log(albedo_init.clamp(eps)).requires_grad_(True)

    # SH coefficients per image: initialise to a gentle ambient (c[0] ≈ 1, rest 0)
    sh_init  = torch.zeros(N, 9, 3, device=device)
    sh_init[:, 0, :] = 0.5                         # ambient term
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
            recon_k   = albedo * shading_k
            loss_data = loss_data + ((recon_k - imgs[k]) ** 2).mean()

        loss_sparse = lambda_sparse  * _tv(to_chw(log_albedo))
        loss_white  = lambda_white   * ((albedo.mean() - 0.5) ** 2)

        loss = loss_data + loss_sparse + loss_white
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
    shadings   = [
        to_np(render_shading(Y, sh_coeffs[k]).clamp(0))
        for k in range(N)
    ]

    return albedo_out, shadings, history
