"""
Fixed-view inverse rendering with known normals and unknown lighting.

Inspired by nvdiffrecmc, but adapted to a simpler setup:
- no mesh optimization
- no rasterization
- no Monte Carlo sampling / denoising
- normals are given directly from a monocular normal map

We optimize:
    - shared base color       [H, W, 3]
    - shared roughness        [H, W, 1]
    - shared metallic         [H, W, 1]
    - per-image environment   [E_H, E_W, 3]

Renderer:
    Cook-Torrance / GGX microfacet BRDF
    integrated over a learnable lat-long environment map.

Interface:
    Same function signature as the original code.
    Same 3-return structure for compatibility:
        albedo_out, shadings, history

Additional outputs are attached to:
    decompose.last_result = {
        "base_color": ...,
        "roughness": ...,
        "metallic": ...,
        "envmaps": ...,
        "diffuse": ...,
        "specular": ...,
        "history": ...
    }
"""

import math
import numpy as np
import torch
import torch.nn.functional as F
import cv2


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def _safe_normalize(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True).clamp_min(eps)


def _tv(x: torch.Tensor) -> torch.Tensor:
    """Isotropic total variation on a [1, C, H, W] tensor."""
    dh = x[..., 1:, :] - x[..., :-1, :]
    dw = x[..., :, 1:] - x[..., :, :-1]
    return (dh.square() + 1e-8).sqrt().mean() + (dw.square() + 1e-8).sqrt().mean()


def _to_chw(x: torch.Tensor) -> torch.Tensor:
    """[H, W, C] -> [1, C, H, W]"""
    return x.permute(2, 0, 1).unsqueeze(0)


def _resize_hw3(t: torch.Tensor, out_hw):
    return F.interpolate(
        t.permute(2, 0, 1).unsqueeze(0),
        size=out_hw,
        mode="bilinear",
        align_corners=False
    ).squeeze(0).permute(1, 2, 0)

# ─────────────────────────────────────────────────────────────────────────────
# Normal map loading
# ─────────────────────────────────────────────────────────────────────────────


def load_normals(path: str, target_hw: tuple) -> torch.Tensor:
    """
    Load a Marigold normal map, decode to unit normals, resize to target_hw.

    Assumes PNG-like encoding:
        n_color = (n + 1) / 2
        n = n_color * 2 - 1
    """
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(path)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if img.dtype == np.uint8:
        n = img.astype(np.float32) / 255.0
    else:
        n = img.astype(np.float32)
        if n.max() > 1.0:
            n /= 65535.0 if img.dtype == np.uint16 else n.max()

    H, W = target_hw
    if n.shape[:2] != (H, W):
        n = cv2.resize(n, (W, H), interpolation=cv2.INTER_LINEAR)

    n = n * 2.0 - 1.0
    n /= np.linalg.norm(n, axis=-1, keepdims=True).clip(1e-6)
    return torch.from_numpy(n.astype(np.float32))


# ─────────────────────────────────────────────────────────────────────────────
# Environment parameterization
# ─────────────────────────────────────────────────────────────────────────────

def _latlong_directions(env_h: int, env_w: int, device) -> tuple:
    """
    Lat-long environment map sampling directions and solid-angle weights.

    Returns
    -------
    dirs   : [S, 3] unit directions
    omega  : [S, 1] solid angle per texel
    """
    theta = (torch.arange(env_h, device=device,
             dtype=torch.float32) + 0.5) * (math.pi / env_h)
    phi = (torch.arange(env_w, device=device, dtype=torch.float32) +
           0.5) * (2.0 * math.pi / env_w)

    th, ph = torch.meshgrid(theta, phi, indexing="ij")

    sin_th = torch.sin(th)
    cos_th = torch.cos(th)
    sin_ph = torch.sin(ph)
    cos_ph = torch.cos(ph)

    # Camera/world convention:
    # x = right, y = up, z = forward
    dirs = torch.stack([
        sin_th * cos_ph,   # x
        cos_th,            # y
        sin_th * sin_ph,   # z
    ], dim=-1).reshape(-1, 3)

    dtheta = math.pi / env_h
    dphi = 2.0 * math.pi / env_w
    omega = (sin_th * dtheta * dphi).reshape(-1, 1)

    return dirs, omega


# ─────────────────────────────────────────────────────────────────────────────
# PBR shading
# ─────────────────────────────────────────────────────────────────────────────

def _schlick_fresnel(u: torch.Tensor, f0: torch.Tensor) -> torch.Tensor:
    """
    u  : [..., 1]
    f0 : [..., 3]
    """
    return f0 + (1.0 - f0) * (1.0 - u).clamp(0.0, 1.0).pow(5)


def _ggx_distribution(n_dot_h: torch.Tensor, alpha: torch.Tensor, eps: float) -> torch.Tensor:
    """
    n_dot_h : [..., 1]
    alpha   : [..., 1]
    """
    a2 = alpha.square()
    denom = n_dot_h.square() * (a2 - 1.0) + 1.0
    return a2 / (math.pi * denom.square().clamp_min(eps))


def _smith_g1_schlick_ggx(n_dot_x: torch.Tensor, roughness: torch.Tensor) -> torch.Tensor:
    """
    Schlick-GGX masking term.
    roughness : [..., 1]
    """
    k = ((roughness + 1.0).square()) / 8.0
    return n_dot_x / (n_dot_x * (1.0 - k) + k).clamp_min(1e-8)


def render_environment_pbr(
    normals: torch.Tensor,        # [H, W, 3]
    base_color: torch.Tensor,     # [H, W, 3]
    roughness: torch.Tensor,      # [H, W, 1]
    metallic: torch.Tensor,       # [H, W, 1]
    envmap: torch.Tensor,         # [E_H, E_W, 3]
    view_dir: torch.Tensor,       # [3]
    light_chunk: int = 1,
    pixel_chunk: int = 64,
    eps: float = 1e-6,
):
    """
    Memory-efficient PBR rendering under a lat-long environment map.

    Returns
    -------
    total    : [H, W, 3]
    diffuse  : [H, W, 3]
    specular : [H, W, 3]
    """
    device = normals.device
    H, W, _ = normals.shape
    P = H * W

    wi_all, omega_all = _latlong_directions(
        envmap.shape[0], envmap.shape[1], device=device)
    Li_all = envmap.reshape(-1, 3)
    S = wi_all.shape[0]

    # Flatten per-pixel inputs
    n_all = normals.reshape(P, 3)
    bc_all = base_color.reshape(P, 3)
    r_all = roughness.reshape(P, 1).clamp(0.04, 1.0)
    m_all = metallic.reshape(P, 1).clamp(0.0, 1.0)

    v = view_dir.view(1, 3).expand(P, 3)
    n_dot_v_all = (n_all * v).sum(dim=-1, keepdim=True).clamp_min(eps)

    dielectric_f0 = torch.full_like(bc_all, 0.04)
    f0_all = dielectric_f0 * (1.0 - m_all) + bc_all * m_all
    diffuse_brdf_all = bc_all * (1.0 - m_all) / math.pi
    alpha_all = r_all.square()

    total_all = torch.zeros(P, 3, device=device, dtype=normals.dtype)
    diff_all = torch.zeros(P, 3, device=device, dtype=normals.dtype)
    spec_all = torch.zeros(P, 3, device=device, dtype=normals.dtype)

    for p0 in range(0, P, pixel_chunk):
        p1 = min(p0 + pixel_chunk, P)

        n = n_all[p0:p1]                    # [Bp, 3]
        bc = bc_all[p0:p1]                  # [Bp, 3]
        r = r_all[p0:p1]                    # [Bp, 1]
        m = m_all[p0:p1]                    # [Bp, 1]
        v_local = v[p0:p1]                  # [Bp, 3]
        n_dot_v = n_dot_v_all[p0:p1]        # [Bp, 1]
        f0 = f0_all[p0:p1]                  # [Bp, 3]
        diffuse_brdf = diffuse_brdf_all[p0:p1]  # [Bp, 3]
        alpha = alpha_all[p0:p1]            # [Bp, 1]

        total_chunk = torch.zeros_like(bc)
        diff_chunk_total = torch.zeros_like(bc)
        spec_chunk_total = torch.zeros_like(bc)

        for s0 in range(0, S, light_chunk):
            s1 = min(s0 + light_chunk, S)

            wi = wi_all[s0:s1]              # [Bs, 3]
            Li = Li_all[s0:s1]              # [Bs, 3]
            omega = omega_all[s0:s1]        # [Bs, 1]

            # [Bp, Bs, 3]
            wi_exp = wi.unsqueeze(0)
            v_exp = v_local.unsqueeze(1)
            n_exp = n.unsqueeze(1)

            n_dot_l = (n_exp * wi_exp).sum(dim=-1,
                                           # [Bp, Bs, 1]
                                           keepdim=True).clamp_min(0.0)
            active = (n_dot_l > 0.0).to(n_dot_l.dtype)

            h = _safe_normalize(wi_exp + v_exp)
            n_dot_h = (n_exp * h).sum(dim=-1, keepdim=True).clamp_min(0.0)
            v_dot_h = (v_exp * h).sum(dim=-1, keepdim=True).clamp_min(0.0)

            D = _ggx_distribution(n_dot_h, alpha.unsqueeze(1), eps)
            G = _smith_g1_schlick_ggx(n_dot_v.unsqueeze(1), r.unsqueeze(1)) * \
                _smith_g1_schlick_ggx(n_dot_l, r.unsqueeze(1))
            Ff = _schlick_fresnel(v_dot_h, f0.unsqueeze(1))

            spec_brdf = (D * G * Ff) / \
                (4.0 * n_dot_v.unsqueeze(1) * n_dot_l + eps)
            spec_brdf = spec_brdf * active

            Li_exp = Li.unsqueeze(0)                # [1, Bs, 3]
            omega_exp = omega.view(1, -1, 1)        # [1, Bs, 1]
            geom_w = n_dot_l * omega_exp            # [Bp, Bs, 1]

            diff_term = (diffuse_brdf.unsqueeze(
                1) * Li_exp * geom_w).sum(dim=1)
            spec_term = (spec_brdf * Li_exp * geom_w).sum(dim=1)

            diff_chunk_total += diff_term
            spec_chunk_total += spec_term
            total_chunk += diff_term + spec_term

        total_all[p0:p1] = total_chunk
        diff_all[p0:p1] = diff_chunk_total
        spec_all[p0:p1] = spec_chunk_total

    total = total_all.reshape(H, W, 3)
    diffuse = diff_all.reshape(H, W, 3)
    specular = spec_all.reshape(H, W, 3)
    return total, diffuse, specular


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def decompose(images_np,
              normals_path="marigold/normals.png",
              n_iter=2000,
              lr=5e-3,
              lambda_sparse=0.5,
              lambda_white=0.1):
    """
    Fixed-view inverse rendering with shared PBR material maps and
    per-image unknown environment lighting.

    Parameters
    ----------
    images_np      : list of ndarray [H, W, 3] uint8
    normals_path   : path to normal map PNG
    n_iter         : Adam iterations
    lr             : learning rate
    lambda_sparse  : regularization strength (re-used as a main smoothness knob)
    lambda_white   : anchor on mean base color to reduce scale ambiguity

    Returns
    -------
    albedo_out : ndarray [H, W, 3] float in [0, 1]
        Interpreted here as base color.

    shadings : list of ndarray [H, W, 3]
        Full rendered appearance images under the estimated lighting
        (kept for compatibility with the old interface).

    history : list of scalar loss values (every 100 iters)

    Additional results are stored in:
        decompose.last_result["roughness"]
        decompose.last_result["metallic"]
        decompose.last_result["envmaps"]
        decompose.last_result["diffuse"]
        decompose.last_result["specular"]
    """
    if len(images_np) == 0:
        raise ValueError("images_np must contain at least one image.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    eps = 1e-6

    N = len(images_np)
    H, W = images_np[0].shape[:2]

    max_side = 128
    scale = min(1.0, max_side / max(H, W))
    H_opt = max(32, int(round(H * scale)))
    W_opt = max(32, int(round(W * scale)))

    # Basic consistency checks
    for idx, img in enumerate(images_np):
        if img.shape[:2] != (H, W):
            raise ValueError(f"All images must have the same size. "
                             f"Image 0 is {(H, W)}, image {idx} is {img.shape[:2]}.")

    # ── Inputs ───────────────────────────────────────────────────────────────
    normals = load_normals(normals_path, (H_opt, W_opt)).to(device)
    normals = _safe_normalize(normals)

    imgs = []
    for img in images_np:
        if img.shape[:2] != (H_opt, W_opt):
            img_small = cv2.resize(img, (W_opt, H_opt),
                                   interpolation=cv2.INTER_AREA)
        else:
            img_small = img
        imgs.append(torch.from_numpy(
            img_small.astype(np.float32) / 255.0).to(device))

    img_mean = sum(imgs) / N

    # Fixed camera/view direction.
    # Assumes normals are in camera space and +z points forward.
    view_dir = torch.tensor(
        [0.0, 0.0, 1.0], device=device, dtype=torch.float32)
    view_dir = view_dir / view_dir.norm()

    # ── Learnable shared materials ───────────────────────────────────────────
    # Base color (shared)
    log_base = torch.log(img_mean.clamp_min(0.05)).requires_grad_(True)

    # Roughness / metallic logits (shared)
    rough_logits = torch.full((H_opt, W_opt, 1), 0.0,
                              device=device, requires_grad=True)
    metal_logits = torch.full((H_opt, W_opt, 1), -2.2,
                              device=device, requires_grad=True)

    # ── Learnable per-image environment maps ────────────────────────────────
    # Small env for tractability; increase if your GPU can handle it.
    # env_h, env_w = 16, 32
    env_h, env_w = 2, 4

    # Initialize as weak gray ambient light
    env_log = torch.full((N, env_h, env_w, 3), -1.5,
                         device=device, requires_grad=True)

    optimizer = torch.optim.Adam(
        [log_base, rough_logits, metal_logits, env_log],
        lr=lr
    )

    history = []

    # Regularization weights
    lambda_rough = 0.25 * lambda_sparse
    lambda_metal = 0.25 * lambda_sparse
    lambda_env_tv = 0.02 * lambda_sparse
    lambda_env_l2 = 0.001
    lambda_binary_metal = 0.01

    for i in range(n_iter):
        print(f"starting iter {i}")
        optimizer.zero_grad()

        base_color = torch.sigmoid(log_base)                  # [H, W, 3]
        roughness = torch.sigmoid(rough_logits)             # [H, W, 1]
        metallic = torch.sigmoid(metal_logits)             # [H, W, 1]
        envmaps = F.softplus(env_log)                     # positive radiance

        loss_data = torch.tensor(0.0, device=device)
        loss_env_tv = torch.tensor(0.0, device=device)
        renders_cache = []

        for k in range(N):
            print(f"  rendering image {k}")
            render_k, _, _ = render_environment_pbr(
                normals=normals,
                base_color=base_color,
                roughness=roughness,
                metallic=metallic,
                envmap=envmaps[k],
                view_dir=view_dir,
                light_chunk=1,
                pixel_chunk=64,
                eps=eps,
            )
            # renders_cache.append(render_k) # not needed
            loss_data = loss_data + (render_k - imgs[k]).square().mean()

            loss_env_tv = loss_env_tv + \
                _tv(envmaps[k].permute(2, 0, 1).unsqueeze(0))

        # Shared map smoothness
        loss_base_tv = lambda_sparse * _tv(_to_chw(base_color))
        loss_rough_tv = lambda_rough * \
            _tv(_to_chw(roughness.expand(-1, -1, 3)))
        loss_metal_tv = lambda_metal * _tv(_to_chw(metallic.expand(-1, -1, 3)))

        # Base color scale anchor
        loss_white = lambda_white * (base_color.mean() - 0.5).square()

        # Mild metallic bimodality prior: encourages near-dielectric or near-metal
        # without forcing it too hard.
        loss_metal_binary = lambda_binary_metal * \
            (metallic * (1.0 - metallic)).mean()

        # Lighting regularization
        loss_env_tv = lambda_env_tv * loss_env_tv / N
        loss_env_l2 = lambda_env_l2 * envmaps.square().mean()

        loss = (
            loss_data
            + loss_base_tv
            + loss_rough_tv
            + loss_metal_tv
            + loss_white
            + loss_metal_binary
            + loss_env_tv
            + loss_env_l2
        )

        loss.backward()
        optimizer.step()

        if i % 1 == 0:
            history.append(float(loss.item()))
            print(
                f"[{i:4d}] total={loss.item():.6f}  "
                f"data={loss_data.item():.6f}  "
                f"base_tv={loss_base_tv.item():.6f}  "
                f"rough_tv={loss_rough_tv.item():.6f}  "
                f"metal_tv={loss_metal_tv.item():.6f}  "
                f"white={loss_white.item():.6f}  "
                f"env_tv={loss_env_tv.item():.6f}  "
                f"env_l2={loss_env_l2.item():.6f}"
            )

    # ── Final outputs ────────────────────────────────────────────────────────
    with torch.no_grad():
        base_color = torch.sigmoid(log_base)
        roughness = torch.sigmoid(rough_logits)
        metallic = torch.sigmoid(metal_logits)
        envmaps = F.softplus(env_log)

        final_renders = []
        final_diffuse = []
        final_specular = []

        for k in range(N):
            total_k, diff_k, spec_k = render_environment_pbr(
                normals=normals,
                base_color=base_color,
                roughness=roughness,
                metallic=metallic,
                envmap=envmaps[k],
                view_dir=view_dir,
                light_chunk=1,
                pixel_chunk=64,
                eps=eps,
            )
            final_renders.append(total_k.clamp(0.0, 1.0))
            final_diffuse.append(diff_k.clamp(0.0, 1.0))
            final_specular.append(spec_k.clamp(0.0, 1.0))

    def _to_np(t):
        return t.detach().cpu().numpy()

    base_color_up = _resize_hw3(base_color.clamp(0, 1), (H, W))
    albedo_out = _to_np(base_color_up)

    roughness_up = _resize_hw3(roughness.expand(-1, -1, 3), (H, W))[..., :1]
    metallic_up = _resize_hw3(metallic.expand(-1, -1, 3), (H, W))[..., :1]

    shadings = [
        _to_np(_resize_hw3(x, (H, W)).clamp(0, 1))
        for x in final_renders
    ]

    # Attach extra results without changing the legacy 3-output API
    decompose.last_result = {
        "base_color": albedo_out,
        "roughness": _to_np(roughness_up[..., 0].clamp(0.0, 1.0)),
        "metallic": _to_np(metallic_up[..., 0].clamp(0.0, 1.0)),
        "envmaps": _to_np(envmaps),
        "diffuse": [_to_np(x) for x in final_diffuse],
        "specular": [_to_np(x) for x in final_specular],
        "history": history,
    }

    return albedo_out, shadings, history
