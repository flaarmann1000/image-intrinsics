"""
Differentiable CT+SH intrinsic decomposition.

Forward model (Cook-Torrance with SH irradiance):

    I_k(p) = k_d(p) * albedo(p) / π * irr_SH(n(p), c_k)

    irr_SH  — Ramamoorthi & Hanrahan (2001) SH irradiance
    F0      = 0.04*(1−metallic) + albedo*metallic
    F_ap    = F0 + (1−F0)*(1−NdV)^5          (Schlick approximation)
    k_d     = (1−F_ap) * (1−metallic)

Optimises:
    log_albedo   [H, W, 3]     shared diffuse reflectance
    sh_coeffs    [N, 9, 3]     per-image SH lighting coefficients
"""

import glob
import os

import numpy as np
import torch
from PIL import Image

from raw_renderer_gpu import shade_ct_sh


def _tv(x: torch.Tensor) -> torch.Tensor:
    """Isotropic total variation. x: [..., H, W] (last two dims are spatial)."""
    dh = x[..., 1:, :] - x[..., :-1, :]
    dw = x[..., :, 1:] - x[..., :, :-1]
    return (dh**2 + 1e-8).sqrt().mean() + (dw**2 + 1e-8).sqrt().mean()


def optimize(
    images,                   # list of [H, W, 3] float32 in [0, 1]
    normals,                  # Tensor or ndarray [H, W, 3] unit normals
    # Tensor or ndarray [H, W, 3] world positions (optional)
    frag_pos=None,
    # Tensor or ndarray [3] camera position (optional)
    cam_pos=None,
    mask=None,                # Tensor or ndarray [H, W] bool (optional)
    metallic=0.0,             # float | ndarray [H,W,1] | Tensor [H,W,1]
    n_iter: int = 2000,
    lr: float = 5e-3,
    lambda_sparse: float = 0.5,
    lambda_white: float = 0.1,
    log_every: int = 200,
):
    """
    Estimate per-pixel albedo and per-image SH coefficients via Adam.

    Normals and optional frag_pos / cam_pos come from rasterize_geometry()
    (called once as a precompute step before the loop) or from a loaded PNG.

    Parameters
    ----------
    images        : list of [H, W, 3] float32 in [0, 1]
    normals       : [H, W, 3] unit surface normals
    frag_pos      : [H, W, 3] world-space fragment positions (for accurate NdV)
    cam_pos       : [3] camera world position (for accurate NdV)
    mask          : [H, W] bool — only foreground pixels contribute to loss
    metallic      : fixed CT metallic — scalar float or per-pixel [H,W,1] array/Tensor
    n_iter        : Adam iterations
    lr            : learning rate
    lambda_sparse : TV regularisation weight on albedo
    lambda_white  : scale-anchor weight  (||mean(albedo) − 0.5||²)
    log_every     : print / record loss interval

    Returns
    -------
    albedo    : ndarray [H, W, 3] float32 in [0, 1]
    sh_coeffs : ndarray [N, 9, 3] float32
    shadings  : list of ndarray [H, W, 3] float32
    history   : list of float, total loss recorded every log_every steps
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    N_imgs = len(images)

    def _t(x, dtype=torch.float32):
        if isinstance(x, torch.Tensor):
            return x.to(device, dtype)
        return torch.from_numpy(np.asarray(x, dtype=np.float32)).to(device)

    imgs_t = torch.stack([_t(img) for img in images])   # [N, H, W, 3]
    N_t = _t(normals)                                  # [H, W, 3]

    # Per-pixel view direction — use exact geometry when available, else (0,0,1)
    if frag_pos is not None and cam_pos is not None:
        V_raw = _t(cam_pos) - _t(frag_pos)               # [H, W, 3]
        V_t = V_raw / (V_raw.norm(dim=-1, keepdim=True) + 1e-8)
    else:
        V_t = N_t.new_tensor([0., 0., 1.]).expand_as(N_t)

    NdV = (N_t * V_t).sum(-1, keepdim=True).clamp(min=1e-4)  # [H, W, 1]
    if mask is not None:
        if isinstance(mask, torch.Tensor):
            mask_t = mask.to(device=device, dtype=torch.bool).unsqueeze(-1)
        else:
            mask_t = torch.from_numpy(np.asarray(
                mask, dtype=bool)).to(device).unsqueeze(-1)
    else:
        mask_t = None

    # metallic: scalar or per-pixel [H,W,1]
    if isinstance(metallic, (int, float)):
        metallic_t = torch.tensor(float(metallic), device=device)
    else:
        metallic_t = _t(metallic)   # [H,W,1] stays broadcastable

    # ── Learnable parameters ─────────────────────────────────────────────────
    albedo_init = imgs_t.mean(0).clamp(0.05, 0.95)              # [H, W, 3]
    log_albedo = torch.log(albedo_init).requires_grad_(True)

    sh_init = torch.zeros(N_imgs, 9, 3, device=device)
    # warm-start ambient
    sh_init[:, 0, :] = 1.5
    sh_coeffs = sh_init.clone().requires_grad_(True)

    opt = torch.optim.Adam([log_albedo, sh_coeffs], lr=lr)

    # ── Forward pass ─────────────────────────────────────────────────────────
    def _forward():
        albedo = torch.exp(log_albedo)                          # [H, W, 3]

        loss_data = albedo.new_zeros(())
        for k in range(N_imgs):
            recon = shade_ct_sh(albedo, N_t, sh_coeffs[k], metallic_t)
            diff = (recon - imgs_t[k]) ** 2
            if mask_t is not None:
                diff = diff[mask_t.expand_as(diff)]
            loss_data = loss_data + diff.mean()

        loss_sparse = lambda_sparse * _tv(log_albedo.permute(2, 0, 1))
        loss_white = lambda_white * (torch.exp(log_albedo).mean() - 0.5) ** 2

        return loss_data + loss_sparse + loss_white, loss_data, loss_sparse, loss_white

    # ── Optimisation loop ────────────────────────────────────────────────────
    history = []
    for i in range(n_iter):
        opt.zero_grad()
        loss, l_d, l_s, l_w = _forward()
        loss.backward()
        opt.step()

        if i % log_every == 0:
            history.append(loss.item())
            print(f"[{i:4d}] total={loss.item():.5f}  "
                  f"data={l_d.item():.5f}  "
                  f"sparse={l_s.item():.5f}  "
                  f"white={l_w.item():.5f}")

    # ── Outputs ───────────────────────────────────────────────────────────────
    albedo_out = torch.exp(log_albedo).clamp(0, 1).detach().cpu().numpy()
    sh_out = sh_coeffs.detach().cpu().numpy()

    with torch.no_grad():
        albedo = torch.exp(log_albedo)
        shadings = []
        for k in range(N_imgs):
            s = shade_ct_sh(albedo, N_t, sh_coeffs[k], metallic_t)
            if mask_t is not None:
                s = s * mask_t
            shadings.append(s.cpu().numpy())

    return albedo_out, sh_out, shadings, history


# ── tiny dataset convenience ──────────────────────────────────────────────────

_TINY_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "raw_dataset",
)


def optimize_tiny(
    normal_set: str = "normals_a",   # "normals_a" | "normals_b"
    n_variants=None,                 # int | None → use all available variants
    **kwargs,
) -> tuple:
    """
    Run intrinsic decomposition on the rendered_tiny_gpu dataset.

    Loads normals from raw_dataset/raw_tiny/<normal_set>.png and rendered
    images from raw_dataset/rendered_tiny_gpu/<normal_set>_variant_*.png,
    then calls optimize() with sensible defaults for the tiny (3×6 px) case.

    Parameters
    ----------
    normal_set  : which normal map to use — "normals_a" (8 unique) or "normals_b" (9 unique)
    n_variants  : how many rendered variants to load (None → all 10)
    **kwargs    : forwarded to optimize() — override n_iter, lr, lambda_sparse, etc.

    Returns
    -------
    Same as optimize(): (albedo, sh_coeffs, shadings, history)
    """
    raw_dir = os.path.join(_TINY_ROOT, "raw_tiny")
    rend_dir = os.path.join(_TINY_ROOT, "rendered_tiny_gpu")

    # Decode normals: (N*0.5+0.5)*255 → unit normals, shape (3, 6, 3)
    normals_u8 = np.array(
        Image.open(os.path.join(raw_dir, f"{normal_set}.png")), dtype=np.float32
    )
    normals = normals_u8 / 255.0 * 2.0 - 1.0                   # (3, 6, 3)
    normals = normals / \
        (np.linalg.norm(normals, axis=-1, keepdims=True) + 1e-8)

    # Load rendered variants in sorted order
    pattern = os.path.join(rend_dir, f"{normal_set}_variant_*.png")
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(
            f"No rendered tiny images found at {pattern!r}. "
            "Run create_raw_gpu_dataset.render_tiny_gpu() first."
        )
    if n_variants is not None:
        paths = paths[:n_variants]

    images = [
        np.array(Image.open(p), dtype=np.float32) / 255.0      # (3, 6, 3)
        for p in paths
    ]

    print(f"optimize_tiny: {normal_set}  {len(images)} variants  "
          f"shape={normals.shape}  device={'cuda' if torch.cuda.is_available() else 'cpu'}")

    # Tiny patches have no background — no mask needed, no frag_pos / cam_pos
    kwargs.setdefault("metallic",      0.0)
    kwargs.setdefault("n_iter",        2000)
    # TV is ill-defined on 3×6; disable
    kwargs.setdefault("lambda_sparse", 0.0)
    kwargs.setdefault("lambda_white",  0.0)

    return optimize(images, normals, **kwargs)
