"""
Synthetic CT + Phong dataset: generation (Phase 1) + decomposition (Phase 2).

Phase 1 renders a sphere (or Suzanne / Stanford bunny) under material configs
× 6 light setups × shader types.

  Shaders:
    ct_sh    — Cook-Torrance + SH lighting
    ct_env   — Cook-Torrance + env-map lighting
    phong_sh — Phong + SH lighting
    phong_env— Phong + env-map lighting

Phase 2 runs intrinsic decomposition recovering per-pixel material + per-image lighting:
    ct_sh    : albedo, metallic, roughness, SH coefficients
    ct_env   : albedo, metallic, roughness, env-map pixels
    phong_sh : albedo, shininess, ks, SH coefficients
    phong_env: albedo, shininess, ks, env-map pixels

Usage
-----
    python raw_optimizer/synthetic_ct_dataset.py --phase 1 --shader ct_sh --mesh sphere
    python raw_optimizer/synthetic_ct_dataset.py --phase 2 --shader phong_sh --mesh sphere
    python raw_optimizer/synthetic_ct_dataset.py --phase 2 --shader all --mat sphere_default
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from functools import partial
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import wandb
from PIL import Image

_REPO_ROOT = Path(__file__).parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from raw_renderer_gpu import (
    rasterize_geometry, shade_ct_sh, shade_ct_env, shade_phong_sh, shade_phong_env,
    SHLight, EnvMapLightGPU,
    Camera, EnvMap, SHLighting, build_sh_basis, generate_mesh, load_obj,
)
from raw_renderer_gpu.rasterizer import _norm, _get_ggx_sh_lut, _sh_irradiance, _sh_basis, _lut_lookup
from raw_optimizer.optimizer import _tv
from raw_optimizer.helper import _albedo_rmse


# ─────────────────────────────────────── constants ───────────────────────────

SYNTHETIC_ROOT = _REPO_ROOT / "synthetic_ct"
DATASET_ROOT   = SYNTHETIC_ROOT / "dataset"
RESULTS_ROOT   = SYNTHETIC_ROOT / "results"

MATERIAL_CONFIGS: dict[str, dict] = {
    "default":          dict(albedo=[0.5, 0.5, 0.5], metallic=0.1, roughness=0.4),
    "albedo_0":         dict(albedo=[0.8, 0.3, 0.2], metallic=0.1, roughness=0.4),
    "albedo_1":         dict(albedo=[0.2, 0.5, 0.8], metallic=0.1, roughness=0.4),
    "metallic_0":       dict(albedo=[0.5, 0.5, 0.5], metallic=0.0, roughness=0.4),
    "metallic_1":       dict(albedo=[0.5, 0.5, 0.5], metallic=0.8, roughness=0.4),
    "roughness_0":      dict(albedo=[0.5, 0.5, 0.5], metallic=0.1, roughness=0.1),
    "roughness_1":      dict(albedo=[0.5, 0.5, 0.5], metallic=0.1, roughness=0.8),
    # checkerboard: two colors sampled from [0.1, 0.9] per channel
    "albedo_checker":   dict(albedo_checker=([0.8, 0.3, 0.2], [0.2, 0.5, 0.8]),
                             metallic=0.1, roughness=0.4, n_tiles=4),
    # random patch textures for all parameters
    "all_texture":      dict(albedo_range=([0.1, 0.1, 0.1], [0.9, 0.9, 0.9]),
                             metallic_range=(0.0, 1.0),
                             roughness_range=(0.1, 0.9),
                             n_tiles=16, seed=42),
}

PHONG_MATERIAL_CONFIGS: dict[str, dict] = {
    "default":           dict(albedo=[0.5, 0.5, 0.5], shininess=32.0,  ks=0.5, ka=0.0, kd=1.0),
    "albedo_0":          dict(albedo=[0.8, 0.3, 0.2], shininess=32.0,  ks=0.5, ka=0.0, kd=1.0),
    "albedo_1":          dict(albedo=[0.2, 0.5, 0.8], shininess=32.0,  ks=0.5, ka=0.0, kd=1.0),
    "shininess_0":       dict(albedo=[0.5, 0.5, 0.5], shininess=4.0,   ks=0.5, ka=0.0, kd=1.0),
    "shininess_1":       dict(albedo=[0.5, 0.5, 0.5], shininess=128.0, ks=0.5, ka=0.0, kd=1.0),
    "ks_0":              dict(albedo=[0.5, 0.5, 0.5], shininess=32.0,  ks=0.1, ka=0.0, kd=1.0),
    "ks_1":              dict(albedo=[0.5, 0.5, 0.5], shininess=32.0,  ks=0.9, ka=0.0, kd=1.0),
    # checkerboard: two values sampled from per-parameter allowed range
    "albedo_checker":    dict(albedo_checker=([0.8, 0.3, 0.2], [0.2, 0.5, 0.8]),
                              shininess=32.0, ks=0.5, ka=0.0, kd=1.0, n_tiles=4),
    "shininess_checker": dict(albedo=[0.5, 0.5, 0.5],
                              shininess_checker=(4.0, 63.0),
                              ks=0.5, ka=0.0, kd=1.0, n_tiles=4),
    "ks_checker":        dict(albedo=[0.5, 0.5, 0.5], shininess=32.0,
                              ks_checker=(0.1, 0.9), ka=0.0, kd=1.0, n_tiles=4),
    # random patch textures for all parameters
    "all_texture":       dict(albedo_range=([0.1, 0.1, 0.1], [0.9, 0.9, 0.9]),
                              shininess_range=(4.0, 63.0),
                              ks_range=(0.1, 0.9),
                              ka=0.0, kd=1.0, n_tiles=16, seed=42),
}

SHININESS_RANGE = (1.0, 63.0)

LIGHT_ANGLES_DEG = [0, 18, 36, 54, 72, 90]
LIGHT_COLOR      = np.array([1.0, 0.9, 0.8], dtype=np.float32)
LIGHT_INTENSITY  = 2.0

DEFAULT_CAMERA = Camera(  # type: ignore[call-arg]
    position=np.array([0.0, 0.0, 3.0], dtype=np.float32),
    target  =np.array([0.0, 0.0, 0.0], dtype=np.float32),
)

DEFAULT_CFG = dict(
    optimizer      = "LBFGS",
    n_iter         = 50,
    lbfgs_max_iter = 20,
    lr             = 1.0,
    lambda_sparse  = 0.0,
    lambda_white   = 0.0,
    lambda_tv      = 0.0,
    sbatch         = 64,
    log_every      = 20,
    loss           = "L2",
    shininess_min  = SHININESS_RANGE[0],
    shininess_max  = SHININESS_RANGE[1],
    spec_warmup_steps    = 0,
    min_metallic_steps   = 0,
    init_spec_zero       = False,
    init_roughness_zero  = False,
    lambda_metallic_l1        = 0.0,
    lambda_metallic_binarize  = 0.0,
    lr_end            = 0.0,
    lr_schedule       = "none",
    lr_schedule_step  = 50,
    lr_schedule_gamma = 0.5,
    # per-parameter transform ("none" | "sigmoid" | "log" | "softplus")
    tr_albedo    = "none",
    tr_metallic  = "none",
    tr_roughness = "none",
    tr_env       = "none",
    # rescale albedo+lighting toward GT after every N steps (0 = disabled)
    rescale_every = 0,
    # accumulate gradients over chunks of this many images per step (0 = all
    # images in one autograd graph). Bounds peak memory to ~img_batch images;
    # numerically identical to full-batch up to float summation order.
    img_batch = 0,
    # SH lighting order for the ct_sh shader: 2 (9 coeffs, default) or 3
    # (16 coeffs). Band 3 has zero Lambertian irradiance weight, so order 3
    # only sharpens the SPECULAR term. GT SH given as (9,3) is zero-padded.
    sh_order = 2,
    # integer stride for downsampling images + GT maps before optimization
    # (nearest/strided, keeps GT crisp). 1 = full resolution.
    downsample = 1,
    # cap on the number of PER-IMAGE wandb previews (recons, env maps, err maps)
    # logged each step. Scalar metrics still use ALL images; this only limits the
    # image uploads, which otherwise dominate runtime for large N. None = all.
    wandb_max_images = None,
    # diffuse Fresnel: multiply the diffuse by (1-F) on top of (1-metallic).
    # MUST match the data generator + final shadings + relight (all default True,
    # i.e. shade_ct_sh/shade_ct_env default), or recon_rmse decouples from the
    # data loss. True = energy-conserving (specular takes energy from diffuse).
    diffuse_fresnel = True,
    # Huber transition point (linear radiance), used when loss == "huber"
    huber_delta = 0.05,
    # ct_env only: compute the specular term by GGX importance sampling
    # (deterministic, valid at all roughness) instead of the texel-grid
    # Riemann sum, which aliases below roughness ~0.3 on the 32x64 grid.
    spec_importance = False,
    spec_samples    = 64,
    # hold out the last N images as a validation set: they are excluded from
    # optimization and, at every log step, re-rendered with the CURRENT
    # intrinsics + their GT lighting -> "relight_rmse"/"relight_mae".
    # Requires GT SH (sh_XXX.npy) in the scene dir. 0 = off.
    val_images = 0,
)


# ─────────────────────────────────────── helpers ─────────────────────────────

def _load_mesh(name: str):
    if name == "sphere":
        return generate_mesh("sphere")
    if name == "suzanne":
        return load_obj(str(_REPO_ROOT / "assets" / "obj" / "suzanne.obj"))
    if name == "bunny":
        return load_obj(str(_REPO_ROOT / "assets" / "obj" / "stanford-bunny.obj"))
    raise ValueError(f"Unknown mesh {name!r}. Choose sphere / suzanne / bunny.")


def _make_lights(angle_deg: float):
    """Build SH and env-map lights for a single XZ-plane rotation angle.

    0° = frontal (+Z toward camera); 90° = side (+X).
    Returns (sh_light, env_light, direction_np, sh_coeffs_np, env_raw).
    """
    theta     = np.radians(angle_deg)
    direction = np.array([np.sin(theta), 0.0, np.cos(theta)], dtype=np.float32)
    sh_raw    = SHLighting.directional(direction, LIGHT_COLOR, intensity=LIGHT_INTENSITY)
    sh_light  = SHLight(coeffs=torch.from_numpy(sh_raw.coeffs))
    env_raw   = EnvMap.from_sh(sh_raw)
    env_light = EnvMapLightGPU(
        dirs        =torch.from_numpy(env_raw._dirs),
        image_flat  =torch.from_numpy(env_raw._image_flat),
        solid_angles=torch.from_numpy(env_raw._solid_angles),
    )
    return sh_light, env_light, direction, sh_raw.coeffs, env_raw


def _sh_coeffs_to_env_img(coeffs: np.ndarray, resolution: int = 64) -> np.ndarray:
    """(9|16,3) SH coefficients → (H,W,3) float32 image normalized to [0,1]."""
    coeffs = np.asarray(coeffs, np.float32)
    order = 3 if coeffs.shape[0] == 16 else 2
    dirs = EnvMap._sh_grid_dirs(resolution)
    img = np.maximum(build_sh_basis(dirs, order=order) @ coeffs, 0.0)
    mx = float(img.max())
    return img / max(mx, 1e-8)


def _env_flat_to_img(env_flat: np.ndarray, env_H: int, env_W: int) -> np.ndarray:
    """(P,3) flat env-map → (H,W,3) float32 image normalized to [0,1]."""
    img = env_flat.reshape(env_H, env_W, 3)
    mx = float(img.max())
    return img / max(mx, 1e-8)


def _softplus_inv(x: torch.Tensor) -> torch.Tensor:
    """Inverse of softplus: softplus(result) ≈ x for x > 0."""
    return torch.log(torch.expm1(x))


# ── Lighting mode helpers ──────────────────────────────────────────────────────

def _get_light_angles(n_lights: int, full_circle: bool) -> list:
    if full_circle:
        return np.linspace(0, 360, n_lights, endpoint=False).tolist()
    return np.linspace(0, 90, n_lights).tolist()


def _scene_suffix(light_mode: str, n_lights: int, full_circle: bool) -> str:
    """Return empty string for the default config (directional, 6, quarter-circle)."""
    # if light_mode == "directional" and n_lights == 6 and not full_circle:
    #     return ""
    mode_short = {"directional": "dir", "random_sh": "rsh", "circular": "circ"}[light_mode]
    suffix = f"_{mode_short}{n_lights}"
    if full_circle:
        suffix += "full"
    return suffix


def _make_lights_random_sh(seed: int, n_dirs: int = 2, min_front_irr: float = 0.3) -> tuple:
    """Random but plausible SH lighting as a sum of n_dirs directional lights.

    Resamples up to 10 times until the mean RGB irradiance at the front-facing
    normal (+Z, toward camera) reaches min_front_irr. Falls back to the best
    attempt found if the threshold is never met.
    """
    rng = np.random.default_rng(seed)

    # Band-limiting weights for diffuse SH irradiance (L=0,1,2)
    _A = np.array([np.pi, 2*np.pi/3, 2*np.pi/3, 2*np.pi/3,
                   np.pi/4, np.pi/4, np.pi/4, np.pi/4, np.pi/4], dtype=np.float32)
    # Front-facing normal: +Z (camera is along +Z in this scene convention)
    _front_Y = build_sh_basis(np.array([0.0, 0.0, 1.0], dtype=np.float32))  # (9,)
    _AY = _A * _front_Y  # (9,) — reusable dot-product weights

    def _sample() -> tuple:
        c = np.zeros((9, 3), dtype=np.float32)
        for _ in range(n_dirs):
            phi   = rng.uniform(0, 2 * np.pi)
            cos_t = rng.uniform(-0.3, 1.0)
            sin_t = float(np.sqrt(max(0.0, 1.0 - cos_t ** 2)))
            d = np.array([sin_t * np.cos(phi), cos_t, sin_t * np.sin(phi)], dtype=np.float32)
            d /= np.linalg.norm(d)
            intensity = rng.uniform(0.5, 2.0)
            color     = rng.uniform([0.6, 0.5, 0.4], [1.4, 1.3, 1.2]).astype(np.float32)
            c += build_sh_basis(d)[:, None] * (color * intensity)[None, :]
        irr = float(_AY @ c.mean(axis=-1))  # mean RGB irradiance at front normal
        return c, irr

    best_coeffs, best_irr = _sample()
    for _ in range(9):
        if best_irr >= min_front_irr:
            break
        coeffs, irr = _sample()
        if irr > best_irr:
            best_coeffs, best_irr = coeffs, irr

    coeffs    = best_coeffs
    sh_raw    = SHLighting(coeffs)
    sh_light  = SHLight(coeffs=torch.from_numpy(coeffs))
    env_raw   = EnvMap.from_sh(sh_raw)
    env_light = EnvMapLightGPU(
        dirs        =torch.from_numpy(env_raw._dirs),
        image_flat  =torch.from_numpy(env_raw._image_flat),
        solid_angles=torch.from_numpy(env_raw._solid_angles),
    )
    return sh_light, env_light, None, coeffs, env_raw


def _make_lights_circular(seed: int, n_sources: int = 2,
                          radius_range=(0.1, 0.5),
                          intensity_range=(2.0, 10.0)) -> tuple:
    """Circular area light sources rendered into an env map, approximated with SH."""
    rng   = np.random.default_rng(seed)
    env_H, env_W = 64, 128
    i_idx = np.arange(env_H)
    j_idx = np.arange(env_W)
    theta = np.pi * (i_idx + 0.5) / env_H           # (env_H,)
    phi   = 2 * np.pi * (j_idx + 0.5) / env_W       # (env_W,)
    dirs_grid = np.stack([                           # (env_H, env_W, 3)
        np.outer(np.sin(theta), np.cos(phi)),
        np.tile(np.cos(theta)[:, None], (1, env_W)),
        np.outer(np.sin(theta), np.sin(phi)),
    ], axis=-1).astype(np.float32)

    env_img = np.zeros((env_H, env_W, 3), dtype=np.float32)
    for _ in range(n_sources):
        phi_s   = rng.uniform(0, 2 * np.pi)
        cos_t_s = rng.uniform(-0.5, 1.0)
        sin_t_s = float(np.sqrt(max(0.0, 1.0 - cos_t_s ** 2)))
        center  = np.array([sin_t_s * np.cos(phi_s), cos_t_s, sin_t_s * np.sin(phi_s)],
                            dtype=np.float32)
        center /= np.linalg.norm(center)
        radius    = rng.uniform(*radius_range)
        color     = rng.uniform(0.3, 1.0, size=3).astype(np.float32)
        intensity = rng.uniform(*intensity_range)
        cos_angle = np.clip((dirs_grid * center[None, None, :]).sum(-1), -1.0, 1.0)
        mask      = cos_angle >= float(np.cos(radius))
        env_img[mask] += color * intensity

    env_raw   = EnvMap(env_img)
    sh_raw    = SHLighting.from_env_map(env_raw)
    sh_light  = SHLight(coeffs=torch.from_numpy(sh_raw.coeffs))
    env_light = EnvMapLightGPU(
        dirs        =torch.from_numpy(env_raw._dirs),
        image_flat  =torch.from_numpy(env_raw._image_flat),
        solid_angles=torch.from_numpy(env_raw._solid_angles),
    )
    return sh_light, env_light, None, sh_raw.coeffs, env_raw


def _get_light_entries(light_mode: str, n_lights: int, full_circle: bool) -> list:
    """Return [(key, make_fn), ...] for each light in the dataset."""
    if light_mode == "directional":
        angles = _get_light_angles(n_lights, full_circle)
        return [(f"light_{int(a):02d}deg", partial(_make_lights, angle_deg=a)) for a in angles]
    if light_mode == "random_sh":
        return [(f"light_{i:03d}", partial(_make_lights_random_sh, seed=i)) for i in range(n_lights)]
    if light_mode == "circular":
        return [(f"light_{i:03d}", partial(_make_lights_circular, seed=i)) for i in range(n_lights)]
    raise ValueError(f"Unknown light_mode {light_mode!r}")


# ── Checkerboard texture helpers ──────────────────────────────────────────────

def _checker_uv(normals_m: np.ndarray) -> tuple:
    """Spherical UV from unit normals (M, 3). Returns (u, v), each (M,) in [0, 1]."""
    nx, ny, nz = normals_m[:, 0], normals_m[:, 1], normals_m[:, 2]
    u = 0.5 + np.arctan2(nz, nx) / (2 * np.pi)
    v = 0.5 - np.arcsin(ny.clip(-1.0, 1.0)) / np.pi
    return u, v


def _make_checker_map(normals_m: np.ndarray, val_a, val_b, n_tiles: int = 4) -> np.ndarray:
    """Per-pixel map (M, C) alternating val_a / val_b on a spherical checkerboard.

    val_a / val_b can be scalars or RGB lists; C is inferred from them.
    """
    u, v = _checker_uv(normals_m)
    cell_a = ((np.floor(u * n_tiles).astype(int) + np.floor(v * n_tiles).astype(int)) % 2) == 0
    a = np.asarray(val_a, dtype=np.float32).reshape(-1)
    b = np.asarray(val_b, dtype=np.float32).reshape(-1)
    C, M = len(a), normals_m.shape[0]
    result = np.empty((M, C), dtype=np.float32)
    result[ cell_a] = a
    result[~cell_a] = b
    return result


def _make_random_patch_map(normals_m: np.ndarray, val_low, val_high,
                           n_tiles: int = 16, seed: int = 42) -> np.ndarray:
    """Per-pixel map (M, C) with independent random values per spherical UV patch.

    Each of the n_tiles x n_tiles grid cells gets a uniformly random value in
    [val_low, val_high].  val_low / val_high are scalars or RGB lists.
    """
    u, v = _checker_uv(normals_m)
    cell_u = np.floor(u * n_tiles).astype(int).clip(0, n_tiles - 1)
    cell_v = np.floor(v * n_tiles).astype(int).clip(0, n_tiles - 1)
    low  = np.asarray(val_low,  dtype=np.float32).reshape(-1)
    high = np.asarray(val_high, dtype=np.float32).reshape(-1)
    C = len(low)
    rng = np.random.default_rng(seed)
    grid = rng.random((n_tiles, n_tiles, C)).astype(np.float32) * (high - low) + low
    return grid[cell_v, cell_u]  # (M, C)


def _scatter_np(flat_m: np.ndarray, mask_flat: np.ndarray, H: int, W: int) -> np.ndarray:
    """Scatter foreground values (M, C) to (H, W, C); background stays zero."""
    C   = flat_m.shape[1] if flat_m.ndim > 1 else 1
    out = np.zeros((H * W, C), dtype=np.float32)
    out[mask_flat] = flat_m.reshape(-1, C)
    return out.reshape(H, W, C)


# ── Domain transforms ─────────────────────────────────────────────────────────

NAMED_TRANSFORMS: dict[str, dict] = {
    "none": dict(albedo="none", metallic="none", roughness="none",
                 shininess="none", ks="none", env="none"),
    "all":  dict(albedo="log",  metallic="sigmoid", roughness="sigmoid",
                 shininess="sigmoid", ks="sigmoid", env="softplus"),
    "only_softplus":  dict(albedo="none",  metallic="none", roughness="none",
                 shininess="none", ks="none", env="softplus"),
    "only_shininess":  dict(albedo="none",  metallic="none", roughness="none",
                 shininess="log", ks="none", env="none"),
}


def _transforms_folder(tr: dict) -> str:
    if tr == NAMED_TRANSFORMS["none"]: return "no_transforms"
    if tr == NAMED_TRANSFORMS["all"]:  return "all_transforms"
    if tr == NAMED_TRANSFORMS["only_softplus"]:  return "only_softplus_transforms"
    if tr == NAMED_TRANSFORMS["only_shininess"]:  return "only_shininess_transforms"
    parts = [f"{k}={v}" for k, v in sorted(tr.items()) if v != "none"]
    return "tr_" + ",".join(parts)


def _parse_transforms(spec: str) -> dict:
    if spec in NAMED_TRANSFORMS:
        return dict(NAMED_TRANSFORMS[spec])
    base = dict(NAMED_TRANSFORMS["none"])
    for part in spec.split(","):
        k, v = part.strip().split("=")
        base[k] = v
    return base


def _fwd_albedo(p: torch.Tensor, t: str) -> torch.Tensor:
    if t == "sigmoid": return torch.sigmoid(p) 
    if t == "log": return torch.exp(p) 
    return  p

def _fwd_metallic(p: torch.Tensor, t: str) -> torch.Tensor:
    if t == "sigmoid":    return torch.sigmoid(p)
    if t == "sigmoid_sq": return torch.sigmoid(p) ** 2
    return p

def _fwd_roughness(p: torch.Tensor, t: str) -> torch.Tensor:
    if t == "sigmoid":    return torch.sigmoid(p)
    if t == "sigmoid_sq": return torch.sigmoid(p) ** 2
    return p

def _fwd_shininess(p: torch.Tensor, t: str, s_min: float, s_max: float) -> torch.Tensor:
    if t == "sigmoid":
        return s_min + (s_max - s_min) * torch.sigmoid(p)
    elif t == "log":
        return torch.exp(p)
    else:
        return p

def _fwd_ks(p: torch.Tensor, t: str) -> torch.Tensor:
    if t == "sigmoid":    return torch.sigmoid(p)
    if t == "sigmoid_sq": return torch.sigmoid(p) ** 2
    return p

def _fwd_env(p: torch.Tensor, t: str) -> torch.Tensor:
    import torch.nn.functional as F
    return F.softplus(p) if t == "softplus" else p


def _init_albedo(base: torch.Tensor, t: str) -> torch.Tensor:
    """base: (H,W,3). Returns raw param in transform space."""
    if t == "log":     return torch.log(base)
    if t == "sigmoid": return torch.logit(base.clamp(1e-6, 1 - 1e-6))
    return base.clone()

def _init_scalar(val: float, H: int, W: int, t: str,
                 squeeze_fn=None, dev=None) -> torch.Tensor:
    """Initialize a (H,W,1) scalar param for a fixed value."""
    dtype = torch.float32
    if t in ("sigmoid", "sigmoid_r"):
        v = np.clip(val, 1e-6, 1-1e-6)
        raw = float(np.log(v / (1 - v)))
    elif t == "sigmoid_sq":
        v = np.clip(np.sqrt(np.clip(val, 0, 1)), 1e-6, 1-1e-6)
        raw = float(np.log(v / (1 - v)))
    else:
        raw = float(val)
    return torch.full((H, W, 1), raw, dtype=dtype, device=dev)


def _init_map(arr: np.ndarray, t: str, dev) -> torch.Tensor:
    """Initialize a (H, W, 1) raw param from a spatial GT map."""
    x = torch.from_numpy(arr.astype(np.float32)).to(dev)
    if t in ("sigmoid", "sigmoid_r"):
        return torch.logit(x)
    elif t == "sigmoid_sq":
        return torch.logit(x.clamp(1e-6, 1 - 1e-6).sqrt())
    else:
        return x.clone()

def _init_env(gt_flat: np.ndarray, t: str, dev) -> torch.Tensor:
    gt_t = torch.from_numpy(gt_flat.astype(np.float32)).to(dev)
    return _softplus_inv(gt_t) if t == "softplus" else gt_t.clone()


def _albedo_lighting_scale(
    albedo_param: torch.Tensor,
    tr_ab: str,
    flat_mask: torch.Tensor,
    gt_ab_m: torch.Tensor,
) -> torch.Tensor:
    """Per-channel LS scale aligning estimated albedo to GT. No side effects."""
    ab_m = _fwd_albedo(albedo_param, tr_ab).detach().reshape(-1, 3)[flat_mask]
    return (gt_ab_m * ab_m).sum(0) / (ab_m * ab_m).sum(0).clamp(1e-8)  # (3,)


def _rescale_albedo_lighting(
    albedo_param: torch.Tensor,
    lighting_params: list,
    tr_ab: str,
    flat_mask: torch.Tensor,
    gt_ab_m: torch.Tensor,
) -> torch.Tensor:
    """Rescale albedo and lighting in-place to align estimated albedo with GT.

    Returns the applied scale (3,) for logging.
    lighting_params: list of tensors with shape (..., 3) — sh_coeffs or env_maps.
    """
    scale = _albedo_lighting_scale(albedo_param, tr_ab, flat_mask, gt_ab_m)  # (3,)
    cur_out = _fwd_albedo(albedo_param, tr_ab)                           # (H, W, 3)
    new_out = (cur_out * scale[None, None, :])
    if tr_ab == "log":
        albedo_param.data.copy_(torch.log(new_out))
    elif tr_ab == "sigmoid":
        albedo_param.data.copy_(torch.logit(new_out.clamp(1e-6, 1 - 1e-6)))
    else:
        albedo_param.data.copy_(new_out)
    for lp in lighting_params:
        lp.data /= scale  # (..., 3) / (3,) — broadcasts over all leading dims
    return scale


# Canonical learnable-parameter sets per shader
_CT_SH_PARAMS    = frozenset({"albedo", "sh",  "metallic",  "roughness"})
_CT_ENV_PARAMS   = frozenset({"albedo", "env", "metallic",  "roughness"})
_PHONG_SH_PARAMS = frozenset({"albedo", "sh",  "shininess", "ks"})
_PHONG_ENV_PARAMS= frozenset({"albedo", "env", "shininess", "ks"})


def _write_dataset_meta(scene_name: str, light_mode: str, n_lights: int,
                        full_circle: bool, light_keys: list) -> None:
    meta_path = DATASET_ROOT / scene_name / "dataset_meta.json"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with open(meta_path, "w") as fh:
        json.dump({"light_mode": light_mode, "n_lights": n_lights,
                   "full_circle": full_circle, "light_keys": light_keys}, fh, indent=2)


def _read_dataset_meta(scene_name: str) -> Optional[dict]:
    meta_path = DATASET_ROOT / scene_name / "dataset_meta.json"
    if meta_path.exists():
        with open(meta_path) as fh:
            return json.load(fh)
    return None


def _all_renders_exist(scene_name: str, shader_type: str,
                       light_keys: Optional[list] = None) -> bool:
    """True when every light's render.png is present for this scene × shader."""
    keys = light_keys if light_keys is not None else [f"light_{int(a):02d}deg" for a in LIGHT_ANGLES_DEG]
    return all(
        (DATASET_ROOT / scene_name / shader_type / k / "render.png").exists()
        for k in keys
    )


def _scatter(flat: torch.Tensor, flat_mask: torch.Tensor, H: int, W: int) -> torch.Tensor:
    """Scatter foreground pixels (M, C) back to (H, W, C)."""
    C   = flat.shape[-1] if flat.dim() > 1 else 1
    buf = torch.zeros(H * W, C, device=flat.device, dtype=torch.float32)
    buf[flat_mask] = flat.reshape(-1, C).float()
    return buf.reshape(H, W, C)


def _save_component_images(
    components: dict[str, torch.Tensor],
    flat_mask:  torch.Tensor,
    H: int, W: int,
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    scalar_names    = {"NdotV", "G1"}
    direction_names = {"R"}

    def _to_u8(arr):
        return (arr.clip(0, 1) * 255).astype(np.uint8)

    for name, comp in components.items():
        comp_cpu = comp.detach().float().cpu()
        if name in scalar_names:
            full = _scatter(comp_cpu, flat_mask, H, W).numpy()
            fg   = full[flat_mask.reshape(H, W).cpu().numpy()]
            vmin, vmax = float(fg.min()), float(fg.max())
            normed = (full - vmin) / max(vmax - vmin, 1e-8)
            gray = (normed.squeeze(-1) * 255).clip(0, 255).astype(np.uint8)
            Image.fromarray(gray, mode="L").save(out_dir / f"{name}.png")
        elif name in direction_names:
            full = _scatter(comp_cpu, flat_mask, H, W).numpy()
            Image.fromarray(_to_u8(full * 0.5 + 0.5)).save(out_dir / f"{name}.png")
        else:
            C    = comp_cpu.shape[-1] if comp_cpu.dim() > 1 else 1
            full = _scatter(comp_cpu, flat_mask, H, W).numpy()
            if C == 1:
                full = np.repeat(full, 3, axis=-1)
            Image.fromarray(_to_u8(full)).save(out_dir / f"{name}.png")


def _save_render(composite: torch.Tensor, flat_mask: torch.Tensor,
                 H: int, W: int, path: Path) -> None:
    img = _scatter(composite.detach(), flat_mask, H, W).cpu().numpy()
    Image.fromarray((img.clip(0, 1) * 255).astype(np.uint8)).save(path)
    np.save(path.with_suffix(".npy"), img.astype(np.float32))


def _save_config_json(path: Path, *, mesh_name, mat_cfg, angle_deg, direction,
                      sh_coeffs_np, width, height, light_type, light_mode="directional") -> None:
    with open(path, "w") as fh:
        json.dump({
            "mesh_name": mesh_name,
            "material":  mat_cfg,
            "light": {
                "light_mode": light_mode,
                "angle_deg":  angle_deg,
                "direction":  direction.tolist() if direction is not None else None,
                "color":      LIGHT_COLOR.tolist(),
                "intensity":  LIGHT_INTENSITY,
                "sh_coeffs":  sh_coeffs_np.tolist(),
            },
            "render_resolution": [width, height],
            "light_type": light_type,
        }, fh, indent=2)


# ─────────────────────────────────────── Phase 1 ─────────────────────────────

def generate_dataset(
    mesh_name:          str            = "sphere",
    width:              int            = 128,
    height:             int            = 128,
    shader:             str            = "all",
    device:             str            = "cuda",
    skip_existing:      bool           = False,
    mat_configs_filter: Optional[set]  = None,
    light_mode:         str            = "directional",
    n_lights:           int            = 6,
    full_circle:        bool           = False,
) -> None:
    """Render material × light × shader combinations.

    shader      : "ct_sh" | "ct_env" | "phong_sh" | "phong_env" | "all"
    light_mode  : "directional" | "random_sh" | "circular"
    n_lights    : number of light configurations per scene
    full_circle : for directional mode — span [0°, 360°) instead of [0°, 90°]
    """
    dev          = device
    mesh         = _load_mesh(mesh_name)
    light_entries = _get_light_entries(light_mode, n_lights, full_circle)
    light_keys    = [k for k, _ in light_entries]
    suffix        = _scene_suffix(light_mode, n_lights, full_circle)

    ct_mat_cfgs    = {k: v for k, v in MATERIAL_CONFIGS.items()
                      if mat_configs_filter is None or k in mat_configs_filter}
    phong_mat_cfgs = {k: v for k, v in PHONG_MATERIAL_CONFIGS.items()
                      if mat_configs_filter is None or k in mat_configs_filter}

    normals_hw, frag_pos_hw, mask_hw, cam_pos = rasterize_geometry(
        mesh, DEFAULT_CAMERA, width=width, height=height, smooth=True, device=dev,
    )
    flat_mask  = mask_hw.reshape(-1)
    normals_m  = normals_hw.reshape(-1, 3)[flat_mask]
    frag_pos_m = frag_pos_hw.reshape(-1, 3)[flat_mask]
    view_m     = _norm(cam_pos.unsqueeze(0) - frag_pos_m)
    M          = int(flat_mask.sum())

    normals_vis = ((normals_hw.cpu().numpy() * 0.5 + 0.5) * 255).clip(0, 255).astype(np.uint8)
    mask_np_gen = mask_hw.cpu().numpy()

    do_ct_sh    = shader in ("ct_sh",    "all")
    do_ct_env   = shader in ("ct_env",   "all")
    do_phong_sh = shader in ("phong_sh", "all")
    do_phong_env= shader in ("phong_env","all")

    # ── CT scenes ────────────────────────────────────────────────────────────
    if do_ct_sh or do_ct_env:
        lut = _get_ggx_sh_lut(torch.device(dev))
        for mat_id, mat_cfg in ct_mat_cfgs.items():
            scene_name   = f"{mesh_name}_{mat_id}{suffix}"
            need_ct_sh   = do_ct_sh  and not (skip_existing and _all_renders_exist(scene_name, "ct_sh",  light_keys))
            need_ct_env  = do_ct_env and not (skip_existing and _all_renders_exist(scene_name, "ct_env", light_keys))
            if not need_ct_sh and not need_ct_env:
                print(f"[Phase 1] {scene_name}  skipped (all renders exist)")
                continue
            n_tiles      = mat_cfg.get("n_tiles", 16)
            seed         = mat_cfg.get("seed", 42)
            normals_m_np = normals_m.cpu().numpy()
            flat_mask_np = flat_mask.cpu().numpy()

            # ── albedo ───────────────────────────────────────────────────────
            if "albedo_range" in mat_cfg:
                low, high   = mat_cfg["albedo_range"]
                albedo_flat = _make_random_patch_map(normals_m_np, low, high, n_tiles, seed)
                albedo_t    = torch.from_numpy(albedo_flat).to(dev)
                albedo_hw   = _scatter_np(albedo_flat, flat_mask_np, height, width)
            elif "albedo_checker" in mat_cfg:
                col_a, col_b = mat_cfg["albedo_checker"]
                albedo_flat  = _make_checker_map(normals_m_np, col_a, col_b, n_tiles)
                albedo_t     = torch.from_numpy(albedo_flat).to(dev)
                albedo_hw    = _scatter_np(albedo_flat, flat_mask_np, height, width)
            else:
                albedo_t  = torch.tensor(mat_cfg["albedo"], device=dev,
                                         dtype=torch.float32).expand(M, 3).contiguous()
                albedo_hw = (np.tile(np.array(mat_cfg["albedo"], dtype=np.float32),
                                     (height, width, 1)) * mask_np_gen[:, :, None])

            # ── metallic ─────────────────────────────────────────────────────
            if "metallic_range" in mat_cfg:
                m_low, m_high  = mat_cfg["metallic_range"]
                met_flat       = _make_random_patch_map(normals_m_np, m_low, m_high, n_tiles, seed + 1)
                metallic_t     = torch.from_numpy(met_flat).to(dev)   # (M, 1)
                metallic_hw    = _scatter_np(met_flat, flat_mask_np, height, width)
                metallic       = float(met_flat.mean())
            else:
                metallic       = mat_cfg["metallic"]
                metallic_t     = None
                metallic_hw    = np.full((height, width, 1), metallic, dtype=np.float32) * mask_np_gen[:, :, None]

            # ── roughness ────────────────────────────────────────────────────
            if "roughness_range" in mat_cfg:
                r_low, r_high  = mat_cfg["roughness_range"]
                rough_flat     = _make_random_patch_map(normals_m_np, r_low, r_high, n_tiles, seed + 2)
                roughness_t    = torch.from_numpy(rough_flat).to(dev)  # (M, 1)
                roughness_hw   = _scatter_np(rough_flat, flat_mask_np, height, width)
                roughness      = float(rough_flat.mean())
            else:
                roughness      = mat_cfg["roughness"]
                roughness_t    = None
                roughness_hw   = np.full((height, width, 1), roughness, dtype=np.float32) * mask_np_gen[:, :, None]

            gt_dir = DATASET_ROOT / scene_name / "gt"
            gt_dir.mkdir(parents=True, exist_ok=True)
            gt_albedo_img = (albedo_hw * 255).clip(0, 255).astype(np.uint8)
            Image.fromarray(gt_albedo_img).save(gt_dir / "albedo.png")
            np.save(gt_dir / "albedo.npy", albedo_hw.astype(np.float32))
            for name, hw in [("metallic", metallic_hw), ("roughness", roughness_hw)]:
                gray = (hw[:, :, 0] * 255).clip(0, 255).astype(np.uint8)
                Image.fromarray(gray, mode="L").save(gt_dir / f"{name}.png")
                np.save(gt_dir / f"{name}.npy", hw.astype(np.float32))
            Image.fromarray(normals_vis).save(gt_dir / "normals.png")

            for light_id, make_fn in light_entries:
                sh_light, env_light, direction, sh_coeffs_np, env_raw = make_fn()
                angle_deg = float(np.degrees(np.arctan2(float(direction[0]), float(direction[2])))) if direction is not None else None

                with torch.no_grad():
                    if need_ct_sh:
                        sh_dir = DATASET_ROOT / scene_name / "ct_sh" / light_id
                        (sh_dir / "components").mkdir(parents=True, exist_ok=True)
                        composite, comps = shade_ct_sh(  # type: ignore[misc]
                            view_m, normals_m, albedo_t,
                            sh_light.coeffs.to(dev),
                            metallic_t if metallic_t is not None else metallic,
                            roughness_t if roughness_t is not None else roughness,
                            lut=lut, return_components=True,
                        )
                        _save_render(composite, flat_mask, height, width, sh_dir / "render.png")
                        _save_component_images(comps, flat_mask, height, width, sh_dir / "components")
                        _save_config_json(sh_dir / "config.json",
                                          mesh_name=mesh_name, mat_cfg=mat_cfg,
                                          angle_deg=angle_deg, direction=direction,
                                          sh_coeffs_np=sh_coeffs_np,
                                          width=width, height=height, light_type="ct_sh",
                                          light_mode=light_mode)
                        sh_env_img = _sh_coeffs_to_env_img(sh_coeffs_np)
                        Image.fromarray((sh_env_img * 255).astype(np.uint8)).save(
                            sh_dir / "sh_env_map.png")
                        np.save(sh_dir / "sh_env_map.npy", sh_env_img.astype(np.float32))

                    if need_ct_env:
                        env_dir = DATASET_ROOT / scene_name / "ct_env" / light_id
                        (env_dir / "components").mkdir(parents=True, exist_ok=True)
                        env_pix_t  = env_light.image_flat.to(dev)
                        env_dirs_t = env_light.dirs.to(dev)
                        env_dw_t   = env_light.solid_angles.to(dev)
                        composite_env, comps_env = shade_ct_env(  # type: ignore[misc]
                            view_m, normals_m, albedo_t,
                            env_pix_t, env_dirs_t, env_dw_t,
                            metallic_t if metallic_t is not None else metallic,
                            roughness_t if roughness_t is not None else roughness,
                            return_components=True,
                        )
                        _save_render(composite_env, flat_mask, height, width, env_dir / "render.png")
                        _save_component_images(comps_env, flat_mask, height, width, env_dir / "components")
                        _save_config_json(env_dir / "config.json",
                                          mesh_name=mesh_name, mat_cfg=mat_cfg,
                                          angle_deg=angle_deg, direction=direction,
                                          sh_coeffs_np=sh_coeffs_np,
                                          width=width, height=height, light_type="ct_env",
                                          light_mode=light_mode)
                        env_img = _env_flat_to_img(env_raw._image_flat,
                                                   *env_raw.image.shape[:2])
                        Image.fromarray((env_img * 255).astype(np.uint8)).save(
                            env_dir / "env_map.png")
                        np.save(env_dir / "env_map.npy", env_img.astype(np.float32))

            _write_dataset_meta(scene_name, light_mode, n_lights, full_circle, light_keys)
            print(f"[Phase 1] {scene_name}  done")

    # ── Phong scenes ─────────────────────────────────────────────────────────
    if do_phong_sh or do_phong_env:
        for mat_id, mat_cfg in phong_mat_cfgs.items():
            scene_name     = f"{mesh_name}_phong_{mat_id}{suffix}"
            need_phong_sh  = do_phong_sh  and not (skip_existing and _all_renders_exist(scene_name, "phong_sh",  light_keys))
            need_phong_env = do_phong_env and not (skip_existing and _all_renders_exist(scene_name, "phong_env", light_keys))
            if not need_phong_sh and not need_phong_env:
                print(f"[Phase 1] {scene_name}  skipped (all renders exist)")
                continue
            n_tiles      = mat_cfg.get("n_tiles", 16)
            seed         = mat_cfg.get("seed", 42)
            normals_m_np = normals_m.cpu().numpy()
            flat_mask_np = flat_mask.cpu().numpy()
            ka, kd = mat_cfg["ka"], mat_cfg["kd"]

            # ── albedo ───────────────────────────────────────────────────────
            if "albedo_range" in mat_cfg:
                low, high   = mat_cfg["albedo_range"]
                albedo_flat = _make_random_patch_map(normals_m_np, low, high, n_tiles, seed)
                albedo_t    = torch.from_numpy(albedo_flat).to(dev)
                albedo_hw   = _scatter_np(albedo_flat, flat_mask_np, height, width)
            elif "albedo_checker" in mat_cfg:
                col_a, col_b = mat_cfg["albedo_checker"]
                albedo_flat  = _make_checker_map(normals_m_np, col_a, col_b, n_tiles)
                albedo_t     = torch.from_numpy(albedo_flat).to(dev)
                albedo_hw    = _scatter_np(albedo_flat, flat_mask_np, height, width)
            else:
                albedo_t  = torch.tensor(mat_cfg["albedo"], device=dev,
                                         dtype=torch.float32).expand(M, 3).contiguous()
                albedo_hw = (np.tile(np.array(mat_cfg["albedo"], dtype=np.float32),
                                     (height, width, 1)) * mask_np_gen[:, :, None])

            # ── shininess ────────────────────────────────────────────────────
            if "shininess_range" in mat_cfg:
                s_low, s_high = mat_cfg["shininess_range"]
                shin_flat = _make_random_patch_map(normals_m_np, s_low, s_high, n_tiles, seed + 1)
                shin_t    = torch.from_numpy(shin_flat).to(dev)            # (M, 1)
                shin_hw   = _scatter_np(shin_flat, flat_mask_np, height, width)
                shin = float(shin_flat.mean())
            elif "shininess_checker" in mat_cfg:
                shin_a, shin_b = mat_cfg["shininess_checker"]
                shin_flat = _make_checker_map(normals_m_np, shin_a, shin_b, n_tiles)
                shin_t    = torch.from_numpy(shin_flat).to(dev)
                shin_hw   = _scatter_np(shin_flat, flat_mask_np, height, width)
                shin = float(np.mean([shin_a, shin_b]))
            else:
                shin   = mat_cfg["shininess"]
                shin_t = None
                shin_hw = np.full((height, width, 1), shin, dtype=np.float32) * mask_np_gen[:, :, None]

            # ── ks ───────────────────────────────────────────────────────────
            if "ks_range" in mat_cfg:
                k_low, k_high = mat_cfg["ks_range"]
                ks_flat = _make_random_patch_map(normals_m_np, k_low, k_high, n_tiles, seed + 2)
                ks_t    = torch.from_numpy(ks_flat).to(dev)                # (M, 1)
                ks_hw   = _scatter_np(ks_flat, flat_mask_np, height, width)
                ks = float(ks_flat.mean())
            elif "ks_checker" in mat_cfg:
                ks_a, ks_b = mat_cfg["ks_checker"]
                ks_flat = _make_checker_map(normals_m_np, ks_a, ks_b, n_tiles)
                ks_t    = torch.from_numpy(ks_flat).to(dev)
                ks_hw   = _scatter_np(ks_flat, flat_mask_np, height, width)
                ks = float(np.mean([ks_a, ks_b]))
            else:
                ks   = mat_cfg["ks"]
                ks_t = None
                ks_hw = np.full((height, width, 1), ks, dtype=np.float32) * mask_np_gen[:, :, None]

            gt_dir = DATASET_ROOT / scene_name / "gt"
            gt_dir.mkdir(parents=True, exist_ok=True)
            Image.fromarray((albedo_hw * 255).clip(0, 255).astype(np.uint8)).save(
                gt_dir / "albedo.png")
            np.save(gt_dir / "albedo.npy", albedo_hw.astype(np.float32))

            Image.fromarray(
                (shin_hw[:, :, 0] / SHININESS_RANGE[1] * 255).clip(0, 255).astype(np.uint8),
                mode="L").save(gt_dir / "shininess.png")
            np.save(gt_dir / "shininess.npy", shin_hw.astype(np.float32))

            Image.fromarray(
                (ks_hw[:, :, 0] * 255).clip(0, 255).astype(np.uint8),
                mode="L").save(gt_dir / "ks.png")
            np.save(gt_dir / "ks.npy", ks_hw.astype(np.float32))
            Image.fromarray(normals_vis).save(gt_dir / "normals.png")

            for light_id, make_fn in light_entries:
                sh_light, env_light, direction, sh_coeffs_np, env_raw = make_fn()
                angle_deg = float(np.degrees(np.arctan2(float(direction[0]), float(direction[2])))) if direction is not None else None

                with torch.no_grad():
                    if need_phong_sh:
                        ps_dir = DATASET_ROOT / scene_name / "phong_sh" / light_id
                        (ps_dir / "components").mkdir(parents=True, exist_ok=True)
                        composite_ps, comps_ps = shade_phong_sh(  # type: ignore[misc]
                            view_m, normals_m,
                            ka, kd,
                            ks_t   if ks_t   is not None else ks,
                            shin_t if shin_t is not None else shin,
                            albedo_t, sh_light.coeffs.to(dev),
                            return_components=True,
                        )
                        _save_render(composite_ps, flat_mask, height, width, ps_dir / "render.png")
                        _save_component_images(comps_ps, flat_mask, height, width, ps_dir / "components")
                        _save_config_json(ps_dir / "config.json",
                                          mesh_name=mesh_name, mat_cfg=mat_cfg,
                                          angle_deg=angle_deg, direction=direction,
                                          sh_coeffs_np=sh_coeffs_np,
                                          width=width, height=height, light_type="phong_sh",
                                          light_mode=light_mode)
                        sh_env_img = _sh_coeffs_to_env_img(sh_coeffs_np)
                        Image.fromarray((sh_env_img * 255).astype(np.uint8)).save(
                            ps_dir / "sh_env_map.png")
                        np.save(ps_dir / "sh_env_map.npy", sh_env_img.astype(np.float32))

                    if need_phong_env:
                        pe_dir = DATASET_ROOT / scene_name / "phong_env" / light_id
                        (pe_dir / "components").mkdir(parents=True, exist_ok=True)
                        env_pix_t  = env_light.image_flat.to(dev)
                        env_dirs_t = env_light.dirs.to(dev)
                        env_dw_t   = env_light.solid_angles.to(dev)
                        composite_pe, comps_pe = shade_phong_env(  # type: ignore[misc]
                            view_m, normals_m, albedo_t,
                            env_pix_t, env_dirs_t, env_dw_t,
                            ka, kd,
                            ks_t   if ks_t   is not None else ks,
                            shin_t if shin_t is not None else shin,
                            return_components=True,
                        )
                        _save_render(composite_pe, flat_mask, height, width, pe_dir / "render.png")
                        _save_component_images(comps_pe, flat_mask, height, width, pe_dir / "components")
                        _save_config_json(pe_dir / "config.json",
                                          mesh_name=mesh_name, mat_cfg=mat_cfg,
                                          angle_deg=angle_deg, direction=direction,
                                          sh_coeffs_np=sh_coeffs_np,
                                          width=width, height=height, light_type="phong_env",
                                          light_mode=light_mode)
                        env_img = _env_flat_to_img(env_raw._image_flat,
                                                   *env_raw.image.shape[:2])
                        Image.fromarray((env_img * 255).astype(np.uint8)).save(
                            pe_dir / "env_map.png")
                        np.save(pe_dir / "env_map.npy", env_img.astype(np.float32))

            _write_dataset_meta(scene_name, light_mode, n_lights, full_circle, light_keys)
            print(f"[Phase 1] {scene_name}  done")

    print("[Phase 1] Complete.")


# ─────────────────────────────────────── Phase 2 helpers ─────────────────────

def _save_grad_step(
    step:       int,
    named_params: dict,        # {name: tensor} — learnable raw params
    pre_raw:    dict,          # {name: tensor} — param.data clone before opt step
    gt_map:     dict,          # {name: np.ndarray} — GT values in physical space, broadcast to param shape
    fwd_map:    dict,          # {name: callable(raw) -> physical}
    fwd_comps_fn,              # callable() -> list[dict] of shade components per image
    losses:     tuple,         # (total, data, sparse, white, tv)
    grad_log_dir: Path,
    flat_mask:  torch.Tensor,
    H: int, W: int,
) -> None:
    """Save one gradient-flow snapshot as a compressed npz file."""
    data = {}
    data["loss_total"]  = np.float32(losses[0])
    data["loss_data"]   = np.float32(losses[1])
    data["loss_sparse"] = np.float32(losses[2])
    data["loss_white"]  = np.float32(losses[3])
    data["loss_tv"]     = np.float32(losses[4])
    with torch.no_grad():
        for name, p in named_params.items():
            raw_np  = p.data.detach().cpu().numpy()
            data[f"{name}_raw"]    = raw_np
            data[f"{name}_update"] = raw_np - pre_raw[name].cpu().numpy()
            if p.grad is not None:
                data[f"{name}_grad"] = p.grad.detach().cpu().numpy()
            val = fwd_map[name](p.data)
            val_np = val.detach().cpu().numpy()
            data[f"{name}_value"] = val_np
            if name in gt_map and gt_map[name] is not None:
                gt_np = np.broadcast_to(gt_map[name], val_np.shape)
                data[f"{name}_gt_error"] = val_np - gt_np

        comps_per_img = fwd_comps_fn()
        for k, comps in enumerate(comps_per_img):
            for cname, ctensor in comps.items():
                c_np = ctensor.detach().float().cpu()
                C = c_np.shape[-1] if c_np.dim() > 1 else 1
                full = torch.zeros(H * W, C, dtype=torch.float32)
                full[flat_mask.cpu()] = c_np.reshape(-1, C)
                data[f"shade_k{k:02d}_{cname}"] = full.reshape(H, W, C).numpy()

    np.savez_compressed(grad_log_dir / f"step_{step:05d}.npz", **data)


def _optimizer_name(cfg) -> str:
    return str(cfg.get("optimizer", "LBFGS")).upper()


def _make_optimizer(params, cfg):
    name = _optimizer_name(cfg)
    if name == "LM":
        return None          # LM is not a torch.optim.Optimizer; see _build_lm
    if name == "LBFGS":
        return torch.optim.LBFGS(
            params, lr=cfg["lr"],
            max_iter=cfg["lbfgs_max_iter"],
            line_search_fn="strong_wolfe",
            tolerance_grad=0,
            tolerance_change=0,

        )
    return torch.optim.Adam(params, lr=cfg["lr"])


def _make_scheduler(opt, cfg, n_steps):
    """Return an LR scheduler for Adam, or None (LBFGS / 'none').

    cfg keys used:
        lr_schedule       : "none" | "cosine" | "step" | "linear" | "exponential"
        lr_end            : target LR at the end of training (default 0).
                            Used by cosine (eta_min), linear (end_factor), exponential (gamma).
                            Ignored by step (use lr_schedule_gamma instead).
        lr_schedule_step  : step-size in iters for "step" mode (default 50)
        lr_schedule_gamma : per-step decay factor for "step" mode (default 0.5)
    """
    if _optimizer_name(cfg) in ("LBFGS", "LM"):
        return None
    mode    = cfg.get("lr_schedule", "none")
    lr_0    = cfg["lr"]
    lr_end  = cfg.get("lr_end", 0.0)
    if mode == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=n_steps, eta_min=lr_end,
        )
    if mode == "step":
        return torch.optim.lr_scheduler.StepLR(
            opt, step_size=cfg.get("lr_schedule_step", 50),
            gamma=cfg.get("lr_schedule_gamma", 0.5),
        )
    if mode == "linear":
        end_factor = lr_end / lr_0 if lr_0 > 0 else 0.0
        return torch.optim.lr_scheduler.LinearLR(
            opt, start_factor=1.0, end_factor=end_factor, total_iters=n_steps,
        )
    if mode == "exponential":
        # compute per-step gamma so that lr_0 * gamma^n_steps == lr_end
        floor     = max(lr_end, 1e-12)
        gamma     = (floor / lr_0) ** (1.0 / max(n_steps, 1)) if lr_0 > 0 else 1.0
        return torch.optim.lr_scheduler.ExponentialLR(opt, gamma=gamma)
    return None


def _loss_fn(recon, target, mask_t, mode, huber_delta=0.05):
    resid = recon - target
    if mode == "L1":
        diff = resid.abs()
    elif mode == "huber":
        a = resid.abs()
        diff = torch.where(a <= huber_delta, 0.5 * resid ** 2,
                           huber_delta * (a - 0.5 * huber_delta))
    else:
        diff = resid ** 2
    return diff[mask_t.expand_as(diff)].mean()


def _opt_step(opt, forward_fn, cfg):
    """Single optimizer step; returns (total_loss, loss_data, loss_sparse, loss_white, loss_tv)."""
    if cfg["optimizer"] == "LBFGS":
        def closure():
            opt.zero_grad()
            loss, *_ = forward_fn()
            loss.backward()
            return loss
        try:
            opt.step(closure)
        except (IndexError, TypeError):
            opt.state.clear()  # line search failed; reset LBFGS state so next iter starts fresh
        with torch.no_grad():
            return forward_fn()
    else:
        opt.zero_grad()
        result = forward_fn()
        result[0].backward()
        opt.step()
        return result


def _opt_step_img_batched(opt, forward_fn, cfg, n_imgs, img_batch):
    """Single optimizer step with gradient accumulation over image chunks.

    Behaves like _opt_step, but each closure evaluation runs forward+backward
    on `img_batch` images at a time, so the autograd graph only ever holds one
    chunk. The summed loss/gradients equal the full-batch ones up to float
    summation order — NOT stochastic mini-batching, so it is safe under LBFGS.

    forward_fn must accept an iterable of image indices (or None for all) and
    scale its image-independent loss terms by len(indices)/n_imgs.
    """
    def closure():
        opt.zero_grad()
        total = 0.0
        for b in range(0, n_imgs, img_batch):
            loss_b, *_ = forward_fn(range(b, min(b + img_batch, n_imgs)))
            loss_b.backward()
            total += float(loss_b.detach())
        return torch.tensor(total)

    if cfg["optimizer"] == "LBFGS":
        try:
            opt.step(closure)
        except (IndexError, TypeError):
            opt.state.clear()  # line search failed; reset LBFGS state so next iter starts fresh
    else:
        closure()
        opt.step()
    # re-evaluate at the accepted parameters for accurate logging (graph-free,
    # so memory stays bounded regardless of n_imgs)
    with torch.no_grad():
        return forward_fn(None)


# ─────────────────────────────────────── wandb log helper ────────────────────

def _structured_scalar_log(*, loss, l_d, l_s, l_w, l_tv, loss_ml, loss_mb,
                           scale3, gt_metrics, relight, recon_rmse, recon_mae, lr):
    """Per-step wandb scalars grouped into three sections by name prefix:
      loss/*  — loss terms + albedo scales
      rmse/*  — intrinsics (albedo/roughness/metallic) + recon + relight RMSE
      mae/*   — the same, MAE
    """
    d = {
        "loss/total": float(loss), "loss/data": float(l_d),
        "loss/tv": float(l_tv), "loss/sparse": float(l_s), "loss/white": float(l_w),
        "loss/metallic_l1": float(loss_ml), "loss/metallic_binarize": float(loss_mb),
        "lr": float(lr),
    }
    if recon_rmse is not None:
        d["rmse/recon"] = float(recon_rmse)
    if recon_mae is not None:
        d["mae/recon"] = float(recon_mae)
    if scale3 is not None:
        d["loss/albedo_scale_r"] = float(scale3[0])
        d["loss/albedo_scale_g"] = float(scale3[1])
        d["loss/albedo_scale_b"] = float(scale3[2])
        d["loss/albedo_scale_mean"] = float(sum(float(x) for x in scale3) / 3.0)
    for k, v in (gt_metrics or {}).items():          # e.g. albedo_rmse -> rmse/albedo
        base, kind = k.rsplit("_", 1)
        d[f"{kind}/{base}"] = v
    for k, v in (relight or {}).items():             # relight_rmse -> rmse/relight
        base, kind = k.rsplit("_", 1)
        d[f"{kind}/{base}"] = v
    return d


# ─────────────────────────────────────── LM residual helpers ─────────────────

def _sqrt_res(x, eps=1e-12):
    """Square-root trick: a non-negative loss term c_i is represented by the
    residual sqrt(c_i), so sum_i r_i^2 == sum_i c_i exactly. eps keeps the
    derivative of sqrt finite at 0."""
    return (x + eps).sqrt()


def _tv_residuals(x, scale):
    """Residuals whose squared sum equals `scale * _tv(x)` (isotropic TV)."""
    dh = x[..., 1:, :] - x[..., :-1, :]
    dw = x[..., :, 1:] - x[..., :, :-1]
    rh = _sqrt_res(scale * (dh**2 + 1e-8).sqrt() / dh.numel())
    rw = _sqrt_res(scale * (dw**2 + 1e-8).sqrt() / dw.numel())
    return torch.cat([rh.reshape(-1), rw.reshape(-1)])


from raw_optimizer.levenberg_marquardt import pcg  # noqa: E402


class _CGBackend:
    """Matrix-free LM step: solve (J^T J + lam D) d = J^T r by preconditioned CG.

    Never forms the P x P matrix (P = 1.3e6 at 512^2), so memory is O(P). Every
    matvec is  J^T (J v)  evaluated with torch.func jvp/vjp, chunked over images
    so the autograd graph only ever holds `img_chunk` images. Handles *all*
    regularizers exactly, including the pixel-coupling ones (TV, sparse, white).

    Preconditioner: block-Jacobi from the exact Gauss-Newton diagonal blocks —
    a 5x5 block per pixel and a (3*n_sh)^2 block per image — which are cheap to
    build from the per-pixel jacobians and dramatically cut the CG count.
    """

    def __init__(self, *, unflatten, data_res_fn, reg_res_fn, blocks_fn,
                 pix_idx, sh_off, n_sh3, n_imgs, img_chunk, tol, maxiter, dev, dtype):
        self.unflatten, self.data_res_fn, self.reg_res_fn = unflatten, data_res_fn, reg_res_fn
        self.blocks_fn = blocks_fn
        self.pix_idx, self.sh_off, self.n_sh3 = pix_idx, sh_off, n_sh3
        self.n_imgs, self.img_chunk = n_imgs, max(1, int(img_chunk))
        self.tol, self.maxiter = float(tol), int(maxiter)
        self.dev, self.dtype = dev, dtype
        self.last_cg_iters = 0

    def _chunks(self, idx):
        for s in range(0, idx.numel(), self.img_chunk):
            yield idx[s:s + self.img_chunk]

    def build(self, params, idx):
        self._theta = torch.cat([p.detach().reshape(-1) for p in params])
        self._idx = idx
        rhs = torch.zeros_like(self._theta)
        loss = 0.0
        for sub in self._chunks(idx):
            f = lambda th, _s=sub: self.data_res_fn(self.unflatten(th), _s)
            r, vjp_fn = torch.func.vjp(f, self._theta)
            rhs += vjp_fn(r)[0]
            loss += float(r.pow(2).sum())
            del vjp_fn
        if self.reg_res_fn is not None:
            fr = lambda th: self.reg_res_fn(self.unflatten(th))
            r, vjp_fn = torch.func.vjp(fr, self._theta)
            rhs += vjp_fn(r)[0]
            loss += float(r.pow(2).sum())
            del vjp_fn
        self.rhs = rhs
        self._A, self._C = self.blocks_fn(params, idx)     # (M,npix,npix), (K,n_sh3,n_sh3)
        return loss

    def _JtJv(self, v):
        out = torch.zeros_like(v)
        for sub in self._chunks(self._idx):
            f = lambda th, _s=sub: self.data_res_fn(self.unflatten(th), _s)
            _, Jv = torch.func.jvp(f, (self._theta,), (v,))
            _, vjp_fn = torch.func.vjp(f, self._theta)
            out += vjp_fn(Jv)[0]
            del vjp_fn, Jv
        if self.reg_res_fn is not None:
            fr = lambda th: self.reg_res_fn(self.unflatten(th))
            _, Jv = torch.func.jvp(fr, (self._theta,), (v,))
            _, vjp_fn = torch.func.vjp(fr, self._theta)
            out += vjp_fn(Jv)[0]
            del vjp_fn, Jv
        return out

    def solve(self, damping, kind):
        if kind == "fletcher":
            d = torch.zeros_like(self.rhs)
            if self.pix_idx is not None:
                d[self.pix_idx.reshape(-1)] = torch.diagonal(
                    self._A, dim1=-2, dim2=-1).reshape(-1)
            if self._C is not None:
                d[self.sh_off:self.sh_off + self._C.shape[0] * self.n_sh3] = torch.diagonal(
                    self._C, dim1=-2, dim2=-1).reshape(-1)
            d = d.clamp_min(1e-12)
        else:
            d = None

        def matvec(v):
            damp = damping * (d * v if d is not None else v)
            return self._JtJv(v) + damp

        # block-Jacobi preconditioner, refreshed for this damping value
        eye_p = torch.eye(self._A.shape[-1], device=self.dev, dtype=self.dtype) \
            if self.pix_idx is not None else None
        Ainv = Cinv = None
        if self.pix_idx is not None:
            Ad = self._A + damping * (torch.diag_embed(torch.diagonal(self._A, dim1=-2, dim2=-1))
                                      if kind == "fletcher" else eye_p)
            Ainv = torch.linalg.inv(Ad + 1e-12 * eye_p)
        if self._C is not None:
            eye_s = torch.eye(self.n_sh3, device=self.dev, dtype=self.dtype)
            Cd = self._C + damping * (torch.diag_embed(torch.diagonal(self._C, dim1=-2, dim2=-1))
                                      if kind == "fletcher" else eye_s)
            Cinv = torch.linalg.inv(Cd + 1e-12 * eye_s)

        def precond(v):
            out = v.clone()
            if Ainv is not None:
                vp = v[self.pix_idx]                                  # (M, npix)
                out[self.pix_idx] = torch.einsum('mij,mj->mi', Ainv, vp)
            if Cinv is not None:
                n = Cinv.shape[0] * self.n_sh3
                vs = v[self.sh_off:self.sh_off + n].view(-1, self.n_sh3)
                out[self.sh_off:self.sh_off + n] = torch.einsum(
                    'kij,kj->ki', Cinv, vs).reshape(-1)
            return out

        x, iters = pcg(matvec, self.rhs, precond, tol=self.tol, maxiter=self.maxiter)
        self.last_cg_iters = iters
        if not torch.isfinite(x).all():
            return None
        return x


class _SchurBackend:
    """Exact LM step via Schur complement over the per-pixel blocks.

    Each pixel's 5 raw params touch only its own residuals, so the pixel Hessian A
    is block-diagonal (5x5 per pixel) *provided no regularizer couples pixels*
    (no TV / sparse / white). Eliminate it and solve the tiny reduced SH system:

        S      = C - B^T (A+lam)^-1 B                  (K*3*n_sh squared, 29 MB at K=100)
        d_sh   = S^-1 (g_s - B^T (A+lam)^-1 g_p)
        d_pix  = (A+lam)^-1 (g_p - B d_sh)

    Exact, and fast: the reduced system is only (K*3*n_sh)^2 (2700^2 at K=100).
    The cost is memory for the cross block B, which is (M*npix) x (K*3*n_sh):
    ~0.8 GB at 124^2/K=100, ~3.5 GB at 256^2, ~14 GB at 512^2. Above
    `lm_schur_max_gb` this backend refuses to run and tells you to use 'cg'
    (which is O(P) and is the auto choice when a pixel-coupling regularizer is on).
    """

    def __init__(self, *, blocks_full_fn, pix_idx, sh_off, n_sh3, dev, dtype):
        self.blocks_full_fn = blocks_full_fn
        self.pix_idx, self.sh_off, self.n_sh3 = pix_idx, sh_off, n_sh3
        self.dev, self.dtype = dev, dtype
        self.last_cg_iters = 0

    def build(self, params, idx):
        self._params, self._idx = params, idx
        self._P = sum(p.numel() for p in params)
        A, C, B, g_p, g_s, loss = self.blocks_full_fn(params, idx)
        self._A, self._C, self._B, self._gp, self._gs = A, C, B, g_p, g_s
        return loss

    def solve(self, damping, kind):
        A, C, B, gp, gs = self._A, self._C, self._B, self._gp, self._gs
        M, npix = A.shape[0], A.shape[-1]
        K = C.shape[0]
        eye_p = torch.eye(npix, device=self.dev, dtype=self.dtype)
        eye_s = torch.eye(self.n_sh3, device=self.dev, dtype=self.dtype)
        if kind == "fletcher":
            Ad = A + damping * torch.diag_embed(torch.diagonal(A, dim1=-2, dim2=-1))
            Cd = C + damping * torch.diag_embed(torch.diagonal(C, dim1=-2, dim2=-1))
        else:
            Ad, Cd = A + damping * eye_p, C + damping * eye_s
        try:
            Ainv = torch.linalg.inv(Ad + 1e-12 * eye_p)                  # (M, npix, npix)
        except RuntimeError:
            return None
        # T[p] = Ainv[p] @ B[p]  -> (M, npix, K*n_sh3)
        T = torch.einsum('mij,mja->mia', Ainv, B)
        S = torch.block_diag(*Cd) - torch.einsum('mia,mib->ab', B, T)    # (K*n_sh3, K*n_sh3)
        rhs_s = gs.reshape(-1) - torch.einsum('mia,mi->a', T, gp)
        try:
            L = torch.linalg.cholesky(S)
            d_sh = torch.cholesky_solve(rhs_s.unsqueeze(-1), L).squeeze(-1)
        except RuntimeError:
            return None
        d_pix = torch.einsum('mij,mj->mi', Ainv,
                             gp - torch.einsum('mia,a->mi', B, d_sh))    # (M, npix)
        delta = torch.zeros(self._P, device=self.dev, dtype=self.dtype)
        delta[self.pix_idx] = d_pix
        delta[self.sh_off:self.sh_off + K * self.n_sh3] = d_sh
        if not torch.isfinite(delta).all():
            return None
        return delta


def _lm_cfg_kwargs(cfg) -> dict:
    return dict(
        solver            = cfg.get("lm_solver", "cholesky"),
        damping           = cfg.get("lm_damping", "standard"),
        damping_init      = float(cfg.get("lm_damping_init", 1e-3)),
        damping_factor    = float(cfg.get("lm_damping_factor", 10.0)),
        damping_min       = float(cfg.get("lm_damping_min", 1e-12)),
        damping_max       = float(cfg.get("lm_damping_max", 1e10)),
        adaptive_damping  = bool(cfg.get("lm_adaptive_damping", True)),
        learning_rate     = float(cfg.get("lm_learning_rate", 1.0)),
        attempts_per_step = int(cfg.get("lm_attempts_per_step", 5)),
        jacobian_max_num_rows = int(cfg.get("lm_jacobian_max_num_rows", 0) or 0),        
        jacobian_mode     = cfg.get("lm_jacobian_mode", "auto"),
    )
# cfg keys consumed directly by the linear backends (not by LevenbergMarquardt):
#   lm_linear_solver auto|dense|cg|schur   lm_dense_max_params
#   lm_image_chunk   lm_cg_tol   lm_cg_maxiter   lm_schur_max_gb


# ─────────────────────────────────────── Phase 2: SH optimizer ───────────────

def _optimize_ct_sh(
    images:       list,
    normals_hw:   torch.Tensor,
    frag_pos_hw:  torch.Tensor,
    mask_hw:      torch.Tensor,
    cam_pos:      torch.Tensor,
    gt_metallic:  Union[float, np.ndarray],
    gt_roughness: Union[float, np.ndarray],
    cfg:          dict,
    wandb_run=None,
    gt_sh_coeffs: Optional[list] = None,
    gt_albedo:    Optional[np.ndarray] = None,
    opt_params:   Optional[frozenset] = None,
    transforms:   Optional[dict] = None,
    init_from_gt: bool = False,
    log_gradients: bool = False,
    grad_log_dir: Optional[Path] = None,
    val_images:   Optional[list] = None,
    val_sh_coeffs: Optional[list] = None,
    init_maps:    Optional[dict] = None,
) -> tuple:
    dev    = normals_hw.device
    ftype  = normals_hw.dtype
    H, W   = normals_hw.shape[:2]
    N_imgs = len(images)
    op = opt_params if opt_params is not None else _CT_SH_PARAMS
    if transforms is not None:
        tr_ab, tr_met, tr_rou = transforms["albedo"], transforms["metallic"], transforms["roughness"]
    else:
        tr_ab  = cfg.get("tr_albedo",   "none")
        tr_met = cfg.get("tr_metallic",  "none")
        tr_rou = cfg.get("tr_roughness", "none")

    def _t(x):
        return torch.from_numpy(np.asarray(x, np.float32)).to(dev, ftype) \
            if not isinstance(x, torch.Tensor) else x.to(dev, ftype)

    _sh_ord = int(cfg.get("sh_order", 2))
    if _sh_ord not in (2, 3):
        raise ValueError(f"sh_order must be 2 or 3, got {_sh_ord}")
    n_sh = (_sh_ord + 1) ** 2                     # 9 or 16
    _diffuse_fresnel = bool(cfg.get("diffuse_fresnel", True))
    _n_log = N_imgs if cfg.get("wandb_max_images") is None else min(N_imgs, int(cfg["wandb_max_images"]))

    def _pad_sh(arr):
        """(9,3) GT coefficients → (n_sh,3), zero-padding band 3 if needed."""
        arr = np.asarray(arr, np.float32)
        if arr.shape[0] < n_sh:
            arr = np.concatenate([arr, np.zeros((n_sh - arr.shape[0], 3), np.float32)])
        return arr

    imgs_t    = torch.stack([_t(img) for img in images])
    flat_mask = mask_hw.reshape(-1)
    N_m       = normals_hw.reshape(-1, 3)[flat_mask]
    fp_m      = frag_pos_hw.reshape(-1, 3)[flat_mask]
    view_m    = _norm(cam_pos.unsqueeze(0) - fp_m)
    lut       = _get_ggx_sh_lut(dev, n_bands=_sh_ord + 1).to(ftype)
    mask_t    = mask_hw.unsqueeze(-1).to(dev)

    learnable    = []
    named_params = {}   # for gradient logging

    if "albedo" in op:
        if init_from_gt and gt_albedo is not None:
            base = torch.from_numpy(
                np.broadcast_to(np.asarray(gt_albedo, np.float32), (H, W, 3)).copy()            
            ).to(dev, ftype)
        else:
            base = imgs_t.mean(0)
        albedo_param = _init_albedo(base, tr_ab).requires_grad_(True)
        learnable.append(albedo_param)
        named_params["albedo"] = albedo_param
    else:
        # frozen albedo: GT only when init_from_gt, else the SAME init a normal run
        # would use (mean image). A frozen param must never silently become GT —
        # that turns a warm-up (e.g. sh_only) into an inverse crime.
        if init_from_gt and gt_albedo is not None:
            base = torch.from_numpy(
                np.broadcast_to(np.asarray(gt_albedo, np.float32), (H, W, 3)).copy()
            ).to(dev, ftype)
        else:
            base = imgs_t.mean(0)
        albedo_param = _init_albedo(base, tr_ab)

    if "sh" in op:
        if init_from_gt and gt_sh_coeffs is not None:
            sh_coeffs = torch.stack([
                torch.from_numpy(_pad_sh(gt_sh_coeffs[k])).to(dev, ftype) for k in range(N_imgs)
            ]).requires_grad_(True)
        else:
            sh_init = torch.zeros(N_imgs, n_sh, 3, device=dev, dtype=ftype)
            sh_init[:, 0, :] = 1.5
            sh_coeffs = sh_init.clone().requires_grad_(True)
        learnable.append(sh_coeffs)
        named_params["sh"] = sh_coeffs
    else:
        sh_coeffs = torch.stack([
            torch.from_numpy(_pad_sh(gt_sh_coeffs[k])).to(dev, ftype) for k in range(N_imgs)
        ])

    _gt_met_np = np.asarray(gt_metallic, np.float32)
    _gt_rou_np = np.asarray(gt_roughness, np.float32)
    _flat_mask_s = flat_mask.cpu().numpy()
    _gt_met_scalar = float(_gt_met_np.reshape(-1, 1)[_flat_mask_s].mean() if _gt_met_np.ndim > 0
                           else float(_gt_met_np))
    _gt_rou_scalar = float(_gt_rou_np.reshape(-1, 1)[_flat_mask_s].mean() if _gt_rou_np.ndim > 0
                           else float(_gt_rou_np))

    if "metallic" in op:
        if init_from_gt:
            metallic_raw = _init_map(_gt_met_np.reshape(H, W, 1), tr_met, dev).to(ftype).requires_grad_(True)
        else:
            _mv = 0.0 if cfg.get("init_spec_zero", False) else 0.5
            metallic_raw = _init_scalar(_mv, H, W, tr_met, dev=dev).to(ftype).requires_grad_(True)
        learnable.append(metallic_raw)
        named_params["metallic"] = metallic_raw
    else:
        # frozen metallic: GT only when init_from_gt, else the normal init.
        if init_from_gt:
            metallic_raw = (_init_map(_gt_met_np.reshape(H, W, 1), tr_met, dev).to(ftype)
                            if _gt_met_np.ndim > 0
                            else _init_scalar(_gt_met_scalar, H, W, tr_met, dev=dev).to(ftype))
        else:
            _mv = 0.0 if cfg.get("init_spec_zero", False) else 0.5
            metallic_raw = _init_scalar(_mv, H, W, tr_met, dev=dev).to(ftype)

    if "roughness" in op:
        if init_from_gt:
            roughness_raw = _init_map(_gt_rou_np.reshape(H, W, 1), tr_rou, dev).to(ftype).requires_grad_(True)
        else:
            _rv = 1.0 if cfg.get("init_spec_zero", False) else (0.1 if cfg.get("init_roughness_zero", False) else 0.5)
            roughness_raw = _init_scalar(_rv, H, W, tr_rou, dev=dev).to(ftype).requires_grad_(True)
        learnable.append(roughness_raw)
        named_params["roughness"] = roughness_raw
    else:
        # frozen roughness: GT only when init_from_gt, else the normal init.
        if init_from_gt:
            roughness_raw = (_init_map(_gt_rou_np.reshape(H, W, 1), tr_rou, dev).to(ftype)
                             if _gt_rou_np.ndim > 0
                             else _init_scalar(_gt_rou_scalar, H, W, tr_rou, dev=dev).to(ftype))
        else:
            _rv = 1.0 if cfg.get("init_spec_zero", False) else (0.1 if cfg.get("init_roughness_zero", False) else 0.5)
            roughness_raw = _init_scalar(_rv, H, W, tr_rou, dev=dev).to(ftype)

    # ── warm-start overrides (natural-space maps from a previous phase) ────────
    # init_maps["albedo"|"sh"|"metallic"|"roughness"] override the init in-place
    # (values are natural, i.e. post-transform); used for curriculum chaining.
    if init_maps is not None:
        with torch.no_grad():
            if init_maps.get("albedo") is not None:
                _am = np.broadcast_to(np.asarray(init_maps["albedo"], np.float32), (H, W, 3)).copy()
                albedo_param.data.copy_(_init_albedo(torch.from_numpy(_am).to(dev, ftype), tr_ab))
            if init_maps.get("sh") is not None:
                sh_coeffs.data.copy_(torch.stack([
                    torch.from_numpy(_pad_sh(init_maps["sh"][k])).to(dev, ftype)
                    for k in range(N_imgs)]))
            if init_maps.get("metallic") is not None:
                _mm = np.broadcast_to(np.asarray(init_maps["metallic"], np.float32), (H, W, 1)).copy()
                metallic_raw.data.copy_(_init_map(_mm, tr_met, dev).to(ftype))
            if init_maps.get("roughness") is not None:
                _rm = np.broadcast_to(np.asarray(init_maps["roughness"], np.float32), (H, W, 1)).copy()
                roughness_raw.data.copy_(_init_map(_rm, tr_rou, dev).to(ftype))

    # ── coarse-to-fine lighting: freeze SH bands above `sh_active_order` ───────
    # Keeps the full order-`sh_order` machinery but only lets the first (a+1)^2
    # coefficients move (a = sh_active_order), by zeroing the higher bands at init
    # and masking their gradient. Used for the "SH1 first, then SH2" curriculum.
    _sh_active = cfg.get("sh_active_order")
    if _sh_active is not None and "sh" in op:
        _na = (int(_sh_active) + 1) ** 2
        if _na < n_sh:
            with torch.no_grad():
                sh_coeffs[:, _na:, :] = 0.0
            _sh_band_mask = torch.zeros(1, n_sh, 1, device=dev, dtype=ftype)
            _sh_band_mask[:, :_na, :] = 1.0
            sh_coeffs.register_hook(lambda g, _m=_sh_band_mask: g * _m)

    # ── optional Gaussian noise on the metallic/roughness init (natural space) ─
    _spec_noise = float(cfg.get("init_spec_noise_std", 0.0) or 0.0)
    if _spec_noise > 0:
        if cfg.get("init_seed") is not None:
            torch.manual_seed(int(cfg["init_seed"]))
        with torch.no_grad():
            if "metallic" in op:
                _mn = (_fwd_metallic(metallic_raw, tr_met)
                       + torch.randn_like(metallic_raw) * _spec_noise).clamp(1e-4, 1 - 1e-4)
                metallic_raw.data.copy_(_init_map(_mn.cpu().numpy(), tr_met, dev).to(ftype))
            if "roughness" in op:
                _rn = (_fwd_roughness(roughness_raw, tr_rou)
                       + torch.randn_like(roughness_raw) * _spec_noise).clamp(1e-4, 1 - 1e-4)
                roughness_raw.data.copy_(_init_map(_rn.cpu().numpy(), tr_rou, dev).to(ftype))

    def _get_met(): return _fwd_metallic(metallic_raw, tr_met)
    def _get_rou(): return _fwd_roughness(roughness_raw, tr_rou)

    # ── Precompute geometry-only terms (never change during optimisation) ─────
    with torch.no_grad():
        _A_vals = [
            torch.pi,
            2*torch.pi/3, 2*torch.pi/3, 2*torch.pi/3,
            torch.pi/4,   torch.pi/4,   torch.pi/4, torch.pi/4, torch.pi/4,
        ]
        if _sh_ord >= 3:
            _A_vals += [0.0] * 7        # Lambertian ZH weight of band 3 is zero
        _A = N_m.new_tensor(_A_vals)
        _AY        = _A * _sh_basis(N_m, order=_sh_ord)               # (M, n_sh)
        _NdotV_raw = (N_m * view_m).sum(-1, keepdim=True)             # (M, 1)
        _NdotV     = _NdotV_raw.clamp(min=0.0)                        # (M, 1)
        _R         = _norm(2.0 * _NdotV_raw * N_m - view_m)           # (M, 3)
        _Y_R       = _sh_basis(_R, order=_sh_ord)                     # (M, n_sh)
        _front     = (_NdotV_raw > 0).to(ftype)                       # (M, 1)
        _imgs_m    = imgs_t.reshape(N_imgs, -1, 3)[:, flat_mask, :]   # (N, M, 3)

    opt   = _make_optimizer(learnable, cfg) if learnable else None
    sched = _make_scheduler(opt, cfg, cfg["n_iter"]) if opt is not None else None

    # ── Levenberg-Marquardt (alternative to LBFGS / Adam) ─────────────────────
    # LM minimises sum(r^2) directly, so _forward's scalar loss is re-expressed as
    # residuals: the data term exactly for L2 (r = recon-target), and every
    # non-negative penalty (L1/huber data, TV, metallic L1/binarize) via the
    # square-root trick. Scaling matches _forward, so sum(r^2) == its loss.
    _lm = None
    _lm_frac = [1.0]                       # chunk fraction, mirrors _forward's `frac`
    if _optimizer_name(cfg) == "LM" and learnable:
        from .levenberg_marquardt import LevenbergMarquardt
        _lm_names = [n for n in ("albedo", "sh", "metallic", "roughness") if n in named_params]
        _M = int(flat_mask.sum())
        _denom = float(N_imgs * _M * 3)    # so sum(r^2) over ALL images == loss_data

        def _lm_unpack(pt):
            d = dict(zip(_lm_names, pt))
            return (d.get("albedo",    albedo_param.detach()),
                    d.get("sh",        sh_coeffs.detach()),
                    d.get("metallic",  metallic_raw.detach()),
                    d.get("roughness", roughness_raw.detach()))

        def _lm_maps(pt):
            ab_p, sh_c, met_r, rou_r = _lm_unpack(pt)
            ab = _fwd_albedo(ab_p, tr_ab)
            return (ab, ab.reshape(-1, 3)[flat_mask], sh_c,
                    _fwd_metallic(met_r, tr_met), _fwd_roughness(rou_r, tr_rou))

        def _lm_data_residuals(pt, idx):
            ab_hw, ab_m, sh_c, met_hw, rou_hw = _lm_maps(pt)
            met_m = met_hw.reshape(-1, 1)[flat_mask]
            rou_m = rou_hw.reshape(-1, 1)[flat_mask]
            rec = torch.stack([
                shade_ct_sh(view_m, N_m, ab_m, sh_c[k], met_m, rou_m, lut=lut,
                            diffuse_fresnel=_diffuse_fresnel)
                for k in idx.tolist()])                       # (n, M, 3)
            resid = rec - _imgs_m[idx]
            if cfg["loss"] == "L2":
                return (resid / math.sqrt(_denom)).reshape(-1)
            if cfg["loss"] == "L1":
                return _sqrt_res(resid.abs() / _denom).reshape(-1)
            _d = cfg.get("huber_delta", 0.05)
            _a = resid.abs()
            hub = torch.where(_a <= _d, 0.5 * resid**2, _d * (_a - 0.5 * _d))
            return _sqrt_res(hub / _denom).reshape(-1)

        def _lm_reg_residuals(pt):
            ab_hw, _, _, met_hw, rou_hw = _lm_maps(pt)
            fr = _lm_frac[0]
            out = []
            if cfg["lambda_sparse"]:
                out.append(_tv_residuals(ab_hw.permute(2, 0, 1), fr * cfg["lambda_sparse"]))
            if cfg["lambda_tv"]:
                s = fr * cfg["lambda_tv"]
                out.append(_tv_residuals(ab_hw.permute(2, 0, 1), s))
                out.append(_tv_residuals(met_hw.permute(2, 0, 1), s))
                out.append(_tv_residuals(rou_hw.permute(2, 0, 1), s))
            if cfg["lambda_white"]:                            # already quadratic
                out.append((math.sqrt(fr * cfg["lambda_white"])
                            * (ab_hw.mean() - 0.5)).reshape(1))
            _m = met_hw.reshape(-1, 1)[flat_mask]
            if cfg.get("lambda_metallic_l1", 0.0):
                out.append(_sqrt_res(fr * cfg["lambda_metallic_l1"] * _m.abs() / _m.numel()).reshape(-1))
            if cfg.get("lambda_metallic_binarize", 0.0):
                out.append(_sqrt_res(fr * cfg["lambda_metallic_binarize"]
                                     * (_m * (1.0 - _m)).clamp(min=0) / _m.numel()).reshape(-1))
            if not out:
                return ab_hw.new_zeros(0)
            return torch.cat(out)

        # ── block-sparse (structured) normal equations ────────────────────────
        # The data residual r[k,p,:] depends ONLY on pixel p's own 5 raw params
        # (albedo rgb, metallic, roughness) and on image k's SH coeffs. So J is
        # block-sparse and a dense jacrev/jacfwd wastes ~99.9% of its work. Build
        # J^T J directly from per-pixel blocks:
        #     H_pp[p]   = sum_k  Jp[k,p]^T Jp[k,p]        (npix x npix)
        #     H_ss[k]   = sum_p  Js[k,p]^T Js[k,p]        (3*n_sh x 3*n_sh)
        #     H_ps[p,k] =        Jp[k,p]^T Js[k,p]        (npix x 3*n_sh)
        # Each per-pixel jacobian costs only 3 VJPs (output dim 3), vmapped over
        # pixels — independent of the residual count.
        # Shared layout + per-pixel jacobian (used by the structured-dense, CG and
        # Schur paths alike).
        _gidx = torch.nonzero(flat_mask, as_tuple=False).squeeze(-1)   # grid idx of masked px
        _nsh  = sh_coeffs.shape[1]
        _nsh3 = _nsh * 3
        _inv_sd = 1.0 / math.sqrt(_denom)
        _offs, _o = {}, 0
        for _nm, _par in zip(_lm_names, learnable):   # NB: don't shadow the _t() helper
            _offs[_nm] = _o
            _o += _par.numel()
        _P = _o
        _cols, _local = [], []
        if "albedo" in _offs:
            _cols.append(_offs["albedo"] + _gidx[:, None] * 3
                         + torch.arange(3, device=dev, dtype=torch.long))
            _local += [0, 1, 2]
        if "metallic" in _offs:
            _cols.append((_offs["metallic"] + _gidx)[:, None]); _local += [3]
        if "roughness" in _offs:
            _cols.append((_offs["roughness"] + _gidx)[:, None]); _local += [4]
        _pix_idx = torch.cat(_cols, 1) if _cols else None          # (M, npix)
        _local_t = torch.tensor(_local, device=dev, dtype=torch.long)
        _npix = len(_local)
        _has_sh = "sh" in _offs
        _ar_nsh3 = torch.arange(_nsh3, device=dev, dtype=torch.long)
        _lm_shapes = [tuple(p.shape) for p in learnable]
        _lm_numels = [p.numel() for p in learnable]

        def _lm_unflat(theta):
            out, i = [], 0
            for shp, n in zip(_lm_shapes, _lm_numels):
                out.append(theta[i:i + n].view(shp)); i += n
            return tuple(out)

        def _px_fn(ab_r, met_r, rou_r, sh_k, v, n):
            ab = _fwd_albedo(ab_r, tr_ab)
            me = _fwd_metallic(met_r, tr_met)
            ro = _fwd_roughness(rou_r, tr_rou)
            return shade_ct_sh(v[None], n[None], ab[None], sh_k, me[None], ro[None],
                               lut=lut, diffuse_fresnel=_diffuse_fresnel)[0] * _inv_sd

        _jac_px = torch.func.vmap(
            torch.func.jacrev(_px_fn, argnums=(0, 1, 2, 3)),
            in_dims=(0, 0, 0, None, 0, 0))

        def _lm_pixsh_maps(pt):
            d = dict(zip(_lm_names, pt))
            return (d.get("albedo",    albedo_param.detach()).reshape(-1, 3)[flat_mask],
                    d.get("sh",        sh_coeffs.detach()),
                    d.get("metallic",  metallic_raw.detach()).reshape(-1, 1)[flat_mask],
                    d.get("roughness", roughness_raw.detach()).reshape(-1, 1)[flat_mask])

        def _lm_jac_and_resid(pt, k):
            """Per-pixel jacobian blocks + residual for one image."""
            ab_m, sh_c, me_m, ro_m = _lm_pixsh_maps(pt)
            with torch.no_grad():
                rec = shade_ct_sh(view_m, N_m, _fwd_albedo(ab_m, tr_ab), sh_c[k],
                                  _fwd_metallic(me_m, tr_met), _fwd_roughness(ro_m, tr_rou),
                                  lut=lut, diffuse_fresnel=_diffuse_fresnel)
                r_k = (rec - _imgs_m[k]) * _inv_sd                       # (M, 3)
            Ja, Jm, Jr, Js = _jac_px(ab_m, me_m, ro_m, sh_c[k], view_m, N_m)
            Jpix = torch.cat([Ja, Jm, Jr], dim=-1)[..., _local_t]        # (M, 3, npix)
            return Jpix, Js.reshape(_gidx.numel(), 3, _nsh3), r_k

        def _lm_blocks(pt, idx):
            """Gauss-Newton diagonal blocks only: A (M,npix,npix), C (K,n_sh3,n_sh3).
            Used as the CG block-Jacobi preconditioner (data term only — a
            preconditioner never needs to be exact)."""
            A = torch.zeros(_gidx.numel(), _npix, _npix, device=dev, dtype=ftype) if _npix else None
            C = torch.zeros(len(idx), _nsh3, _nsh3, device=dev, dtype=ftype) if _has_sh else None
            for j, k in enumerate(idx.tolist()):
                Jpix, Jsf, _ = _lm_jac_and_resid(pt, k)
                if _npix:
                    A += torch.einsum('mci,mcj->mij', Jpix, Jpix)
                if _has_sh:
                    C[j] = torch.einsum('mca,mcb->ab', Jsf, Jsf)
            return A, C

        def _reg_px_fn(ab_r, met_r, rou_r):
            """Per-pixel regularizer residuals — only the pixel-SEPARABLE ones, which
            is all Schur permits. Squared-summed they reproduce the scalar terms."""
            me = _fwd_metallic(met_r, tr_met)
            fr = _lm_frac[0]
            out = []
            if cfg.get("lambda_metallic_l1", 0.0):
                out.append(_sqrt_res(fr * cfg["lambda_metallic_l1"] * me.abs() / _M))
            if cfg.get("lambda_metallic_binarize", 0.0):
                out.append(_sqrt_res(fr * cfg["lambda_metallic_binarize"]
                                     * (me * (1.0 - me)).clamp(min=0) / _M))
            if not out:
                return ab_r.new_zeros(0)
            return torch.cat(out)

        _jac_reg_px = torch.func.vmap(torch.func.jacrev(_reg_px_fn, argnums=(0, 1, 2)))
        _has_px_reg = bool(cfg.get("lambda_metallic_l1", 0.0)
                           or cfg.get("lambda_metallic_binarize", 0.0))

        def _lm_blocks_full(pt, idx):
            """A, C, B, g_p, g_s, loss  — everything the Schur complement needs.
            B is (M, npix, K*n_sh3): the memory bound of this backend.
            Includes the pixel-separable regularizers (they add to A and g_p only)."""
            K = len(idx)
            nb = _gidx.numel() * _npix * K * _nsh3 * (8 if ftype == torch.float64 else 4)
            cap = float(cfg.get("lm_schur_max_gb", 4.0)) * 1e9
            if nb > cap:
                raise MemoryError(
                    f"Schur cross-block B needs {nb/1e9:.1f} GB (> lm_schur_max_gb="
                    f"{cap/1e9:.1f}). Use lm_linear_solver='cg' (O(P) memory), or "
                    f"lower the light count / resolution.")
            A = torch.zeros(_gidx.numel(), _npix, _npix, device=dev, dtype=ftype)
            C = torch.zeros(K, _nsh3, _nsh3, device=dev, dtype=ftype)
            B = torch.zeros(_gidx.numel(), _npix, K * _nsh3, device=dev, dtype=ftype)
            gp = torch.zeros(_gidx.numel(), _npix, device=dev, dtype=ftype)
            gs = torch.zeros(K, _nsh3, device=dev, dtype=ftype)
            loss = 0.0
            for j, k in enumerate(idx.tolist()):
                Jpix, Jsf, r_k = _lm_jac_and_resid(pt, k)
                loss += float(r_k.pow(2).sum())
                A  += torch.einsum('mci,mcj->mij', Jpix, Jpix)
                gp += torch.einsum('mci,mc->mi',  Jpix, r_k)
                C[j] = torch.einsum('mca,mcb->ab', Jsf, Jsf)
                gs[j] = torch.einsum('mca,mc->a',  Jsf, r_k)
                B[:, :, j * _nsh3:(j + 1) * _nsh3] = torch.einsum('mci,mca->mia', Jpix, Jsf)

            # Pixel-separable regularizers: they touch only each pixel's own params,
            # so they add to A and g_p (never to B or C). Omitting them would make
            # the Schur step solve a DIFFERENT system than the loss LM accepts on.
            if _has_px_reg:
                ab_m, _, me_m, ro_m = _lm_pixsh_maps(pt)
                with torch.no_grad():
                    r_reg = torch.func.vmap(_reg_px_fn)(ab_m, me_m, ro_m)       # (M, nr)
                    loss += float(r_reg.pow(2).sum())
                Ra, Rm, Rr = _jac_reg_px(ab_m, me_m, ro_m)
                Jreg = torch.cat([Ra, Rm, Rr], dim=-1)[..., _local_t]           # (M, nr, npix)
                A  += torch.einsum('mri,mrj->mij', Jreg, Jreg)
                gp += torch.einsum('mri,mr->mi',  Jreg, r_reg)
            return A, C, B, gp, gs, loss

        # ── block-sparse (structured) DENSE normal equations ──────────────────
        _lm_structured = bool(cfg.get("lm_structured", False))
        _structured_gn = None
        if _lm_structured:

            def _structured_gn(pt, idx):
                d = dict(zip(_lm_names, pt))
                ab_p  = d.get("albedo",    albedo_param.detach())
                sh_c  = d.get("sh",        sh_coeffs.detach())
                met_r = d.get("metallic",  metallic_raw.detach())
                rou_r = d.get("roughness", roughness_raw.detach())
                ab_m = ab_p.reshape(-1, 3)[flat_mask]
                me_m = met_r.reshape(-1, 1)[flat_mask]
                ro_m = rou_r.reshape(-1, 1)[flat_mask]
                ab_f = _fwd_albedo(ab_m, tr_ab)
                me_f = _fwd_metallic(me_m, tr_met)
                ro_f = _fwd_roughness(ro_m, tr_rou)

                JJ  = torch.zeros(_P, _P, device=dev, dtype=ftype)
                rhs = torch.zeros(_P, device=dev, dtype=ftype)
                flat = JJ.view(-1)
                loss = 0.0
                Hpp = gp = None
                for k in idx.tolist():
                    with torch.no_grad():
                        rec = shade_ct_sh(view_m, N_m, ab_f, sh_c[k], me_f, ro_f,
                                          lut=lut, diffuse_fresnel=_diffuse_fresnel)
                        r_k = (rec - _imgs_m[k]) * _inv_sd                  # (M, 3)
                        loss += float(r_k.pow(2).sum())
                    Ja, Jm, Jr, Js = _jac_px(ab_m, me_m, ro_m, sh_c[k], view_m, N_m)
                    Jpix = torch.cat([Ja, Jm, Jr], dim=-1)[..., _local_t]   # (M, 3, npix)
                    Jsf  = Js.reshape(_gidx.numel(), 3, _nsh3)              # (M, 3, 3*n_sh)
                    if _npix:
                        _hpp = torch.einsum('mci,mcj->mij', Jpix, Jpix)
                        _gp  = torch.einsum('mci,mc->mi',  Jpix, r_k)
                        Hpp = _hpp if Hpp is None else Hpp + _hpp
                        gp  = _gp  if gp  is None else gp  + _gp
                    if _has_sh:
                        sh_i = _offs["sh"] + k * _nsh3 + _ar_nsh3           # (3*n_sh,)
                        _hss = torch.einsum('mca,mcb->ab', Jsf, Jsf)
                        _gs  = torch.einsum('mca,mc->a',  Jsf, r_k)
                        flat.index_add_(0, (sh_i[:, None] * _P + sh_i[None, :]).reshape(-1),
                                        _hss.reshape(-1))
                        rhs.index_add_(0, sh_i, _gs)
                        if _npix:
                            _hps = torch.einsum('mci,mca->mia', Jpix, Jsf)  # (M, npix, 3*n_sh)
                            rws = _pix_idx[:, :, None].expand(-1, -1, _nsh3)
                            cls = sh_i.view(1, 1, -1).expand(rws.shape[0], _npix, -1)
                            flat.index_add_(0, (rws * _P + cls).reshape(-1), _hps.reshape(-1))
                            flat.index_add_(0, (cls * _P + rws).reshape(-1), _hps.reshape(-1))
                if _npix:
                    rws = _pix_idx[:, :, None].expand(-1, -1, _npix)
                    cls = _pix_idx[:, None, :].expand(-1, _npix, -1)
                    flat.index_add_(0, (rws * _P + cls).reshape(-1), Hpp.reshape(-1))
                    rhs.index_add_(0, _pix_idx.reshape(-1), gp.reshape(-1))
                return JJ, rhs, loss

        _has_reg = any(cfg.get(k, 0.0) for k in
                       ("lambda_sparse", "lambda_tv", "lambda_white",
                        "lambda_metallic_l1", "lambda_metallic_binarize"))
        # Regularizers that couple DIFFERENT pixels: they destroy the block-diagonal
        # pixel Hessian that the exact Schur complement relies on.
        _pix_coupled = any(cfg.get(k, 0.0) for k in
                           ("lambda_tv", "lambda_sparse", "lambda_white"))

        # ── choose the linear solver ──────────────────────────────────────────
        # P = M*5 + K*3*n_sh. A dense P x P matrix is 25 GB at 124^2 and 6.9 TB at
        # 512^2, so above ~20k params we must never form it:
        #   schur : exact, eliminates the block-diagonal per-pixel 5x5 blocks and
        #           solves the tiny reduced SH system. Needs pixel-separable regs.
        #   cg    : matrix-free preconditioned CG. O(P) memory, keeps every
        #           regularizer, inner solve is inexact (truncated Newton).
        _lin = str(cfg.get("lm_linear_solver", "auto")).lower()
        if _lin == "auto":
            if _P <= int(cfg.get("lm_dense_max_params", 20000)):
                _lin = "dense"
            else:
                _lin = "cg" if _pix_coupled else "schur"
        if _lin == "schur" and _pix_coupled:
            raise ValueError(
                "lm_linear_solver='schur' needs pixel-separable regularizers: set "
                "lambda_tv=lambda_sparse=lambda_white=0 (metallic_l1/binarize are fine), "
                "or use lm_linear_solver='cg'.")
        if _lin == "dense" and _P > 40000:
            print(f"  [LM] WARNING: dense solver with P={_P} -> JtJ is "
                  f"{_P**2*(8 if ftype==torch.float64 else 4)/1e9:.1f} GB")

        _lm_backend = None
        if _lin == "cg":
            _lm_backend = _CGBackend(
                unflatten=_lm_unflat, data_res_fn=_lm_data_residuals,
                reg_res_fn=_lm_reg_residuals if _has_reg else None,
                blocks_fn=_lm_blocks, pix_idx=_pix_idx,
                sh_off=_offs.get("sh", 0), n_sh3=_nsh3, n_imgs=N_imgs,
                img_chunk=int(cfg.get("lm_image_chunk", 8)),
                tol=float(cfg.get("lm_cg_tol", 1e-4)),
                maxiter=int(cfg.get("lm_cg_maxiter", 50)),
                dev=dev, dtype=ftype)
        elif _lin == "schur":
            _lm_backend = _SchurBackend(
                blocks_full_fn=_lm_blocks_full, pix_idx=_pix_idx,
                sh_off=_offs.get("sh", 0), n_sh3=_nsh3, dev=dev, dtype=ftype)

        _lm = LevenbergMarquardt(
            learnable, _lm_data_residuals, n_samples=N_imgs,
            reg_residuals_fn=_lm_reg_residuals if _has_reg else None,
            structured_gn_fn=_structured_gn if _lin == "dense" else None,
            linear_backend=_lm_backend,
            **_lm_cfg_kwargs(cfg))
        _lm_bs = int(cfg.get("lm_batch_size", 0) or 0)
        _lm_full = not (0 < _lm_bs < N_imgs)
        _jac_desc = ('block-sparse' if (_lin == 'dense' and _lm_structured)
                     else ('blocks+autograd' if _lin in ('cg', 'schur')
                           else cfg.get('lm_jacobian_mode', 'auto')))
        print(f"  [LM] P={_lm.n_params}  samples={N_imgs}  "
              f"{'full batch' if _lm_full else f'batch={_lm_bs}'}  "
              f"linear={_lin}  damping={cfg.get('lm_damping','standard')}  jacobian={_jac_desc}")

    _step = [0]
    _loss_ml = [torch.zeros((), device=dev, dtype=ftype)]
    _loss_mb = [torch.zeros((), device=dev, dtype=ftype)]
    _recon_rmse = [0.0]           # per-image recon RMSE/MAE, stashed by _forward
    _recon_mae  = [0.0]

    def _forward(img_indices=None):
        # img_indices: iterable of image indices for gradient-accumulation
        # chunks (None = all images). Image-independent loss terms are scaled
        # by the chunk fraction so the chunk losses sum to the full-batch loss.
        idx   = None if img_indices is None else list(img_indices)
        n_sel = N_imgs if idx is None else len(idx)
        frac  = n_sel / N_imgs
        _spec_warmup = _step[0] < cfg.get("spec_warmup_steps", 0)
        albedo      = _fwd_albedo(albedo_param, tr_ab)                # (H, W, 3)
        albedo_m    = albedo.reshape(-1, 3)[flat_mask]                 # (M, 3)
        metallic    = _get_met()                                       # (H, W, 1)
        roughness   = _get_rou()                                       # (H, W, 1)
        metallic_m  = metallic.reshape(-1, 1)[flat_mask]               # (M, 1)
        roughness_m = roughness.reshape(-1, 1)[flat_mask]               # (M, 1)
        met_m_true  = metallic_m                                       # for regularisation
        if _spec_warmup:
            metallic_m  = albedo_m.new_zeros(albedo_m.shape[0], 1)
            roughness_m = albedo_m.new_ones(albedo_m.shape[0], 1)
        if _step[0] < cfg.get("min_metallic_steps", 0):
            metallic_m = metallic_m.clamp(min=0.1)

        # ── material terms (identical for all N images) ───────────────────
        f0   = 0.04 * (1.0 - metallic_m) + albedo_m * metallic_m     # (M, 3)
        F    = f0 + (1.0 - f0) * (1.0 - _NdotV).pow(5)               # (M, 3)
        alpha = roughness_m ** 2
        G1    = _NdotV / (_NdotV * (1.0 - alpha**2/2.0) + alpha**2/2.0 + 1e-6)  # (M, 1)
        # Diffuse weight. Must match the data generator (shade_ct_sh) and the
        # final shadings/relight, or recon_rmse decouples from the data loss and
        # the inverse crime can't reach 0. diffuse_fresnel=True multiplies by
        # (1-F) (energy taken by specular); default True = shade_ct_sh default.
        k_d   = 1.0 - metallic_m                                       # (M, 1)
        if _diffuse_fresnel:
            k_d = (1.0 - F) * k_d                                      # (M, 3)

        # ── specular SH filter B (roughness-dependent, recomputed each step) ─
        Bvals = _lut_lookup(lut, roughness_m.squeeze(-1))              # (M, n_bands)
        _bp = [Bvals[..., 0:1],
               Bvals[..., 1:2].expand(-1, 3),
               Bvals[..., 2:3].expand(-1, 5)]
        if _sh_ord >= 3:
            _bp.append(Bvals[..., 3:4].expand(-1, 7))
        BY = torch.cat(_bp, dim=-1) * _Y_R                             # (M, n_sh)

        # ── batch SH projection over the selected images ───────────────────
        # (1, M, 9) @ (n, 9, 3) → (n, M, 3) via broadcast matmul
        sh_sel   = sh_coeffs if idx is None else sh_coeffs[idx]       # (n, 9, 3)
        imgs_sel = _imgs_m   if idx is None else _imgs_m[idx]         # (n, M, 3)
        irr_all    = (_AY.unsqueeze(0) @ sh_sel).clamp(min=0)         # (n, M, 3)
        L_spec_all = (BY.unsqueeze(0)  @ sh_sel).clamp(min=0)         # (n, M, 3)

        diff_m  = k_d * albedo_m / torch.pi * irr_all                 # (n, M, 3)
        spec_m  = F * G1 * L_spec_all / 4.0                           # (n, M, 3)
        recon_m = (diff_m + spec_m) * _front                          # (n, M, 3)

        # ── loss in masked pixel space (scatter back not needed) ──────────
        resid = recon_m - imgs_sel
        with torch.no_grad():                        # per-image recon RMSE/MAE (for logging)
            _rd = resid.detach()
            _recon_rmse[0] = float(_rd.pow(2).mean(dim=(-2, -1)).sqrt().mean())
            _recon_mae[0]  = float(_rd.abs().mean(dim=(-2, -1)).mean())
        if cfg["loss"] == "L1":
            err = resid.abs()
        elif cfg["loss"] == "huber":
            _d = cfg.get("huber_delta", 0.05)
            _a = resid.abs()
            err = torch.where(_a <= _d, 0.5 * resid ** 2, _d * (_a - 0.5 * _d))
        else:
            err = resid ** 2
        # mean over images (n_sel/N_imgs lets gradient-accumulation chunks sum
        # to the full-batch value): loss_data is comparable across dataset
        # sizes. Regularizer lambdas are absolute (not divided by N).
        loss_data = err.mean() * (n_sel / N_imgs)

        loss_sparse = frac * cfg["lambda_sparse"] * _tv(albedo.permute(2, 0, 1))
        loss_white  = frac * cfg["lambda_white"]  * (albedo.mean() - 0.5) ** 2
        loss_tv     = frac * cfg["lambda_tv"] * (
            _tv(albedo.permute(2, 0, 1)) +
            _tv(metallic.permute(2, 0, 1)) +
            _tv(roughness.permute(2, 0, 1))
        )
        loss_metallic_l1       = frac * cfg.get("lambda_metallic_l1",       0.0) * met_m_true.abs().mean()
        loss_metallic_binarize = frac * cfg.get("lambda_metallic_binarize",  0.0) * (met_m_true * (1.0 - met_m_true)).mean()
        _loss_ml[0] = loss_metallic_l1.detach()
        _loss_mb[0] = loss_metallic_binarize.detach()
        return loss_data + loss_sparse + loss_white + loss_tv + loss_metallic_l1 + loss_metallic_binarize, loss_data, loss_sparse, loss_white, loss_tv

    def _forward_components():
        with torch.no_grad():
            albedo_m    = _fwd_albedo(albedo_param, tr_ab).reshape(-1, 3)[flat_mask]
            metallic_m  = _fwd_metallic(metallic_raw, tr_met).reshape(-1, 1)[flat_mask]
            roughness_m = _fwd_roughness(roughness_raw, tr_rou).reshape(-1, 1)[flat_mask]
            result = []
            for k in range(N_imgs):
                _, comps = shade_ct_sh(view_m, N_m, albedo_m, sh_coeffs[k],
                                       metallic_m, roughness_m, lut=lut,
                                       diffuse_fresnel=_diffuse_fresnel, return_components=True)
                result.append(comps)
        return result

    gt_map_grad = {
        "albedo":    np.broadcast_to(np.asarray(gt_albedo,   np.float32), (H, W, 3)).copy() if gt_albedo is not None else None,
        "sh":        np.stack([_pad_sh(s) for s in gt_sh_coeffs]).astype(np.float32) if gt_sh_coeffs is not None else None,
        "metallic":  np.broadcast_to(np.asarray(gt_metallic,  np.float32), (H, W, 1)).copy(),
        "roughness": np.broadcast_to(np.asarray(gt_roughness, np.float32), (H, W, 1)).copy(),
    }
    fwd_map_grad = {
        "albedo":    lambda p: _fwd_albedo(p, tr_ab),
        "sh":        lambda p: p,
        "metallic":  lambda p: _fwd_metallic(p, tr_met),
        "roughness": lambda p: _fwd_roughness(p, tr_rou),
    }

    if log_gradients and grad_log_dir is not None:
        grad_log_dir.mkdir(parents=True, exist_ok=True)

    history = []

    # Precompute flat GT tensors for per-step RMSE logging
    _flat_mask_np = flat_mask.cpu().numpy()
    _gt_ab_m = (torch.from_numpy(
                    np.asarray(gt_albedo, np.float32).reshape(-1, 3)[_flat_mask_np]
                ).to(dev, ftype) if gt_albedo is not None else None)
    _gt_met_arr = np.asarray(gt_metallic, np.float32)
    _gt_met_m = torch.from_numpy(
        _gt_met_arr.reshape(-1, 1)[_flat_mask_np] if _gt_met_arr.ndim > 0
        else np.full((int(flat_mask.sum()), 1), float(_gt_met_arr), np.float32)
    ).to(dev, ftype)
    _gt_rou_arr = np.asarray(gt_roughness, np.float32)
    _gt_rou_m = torch.from_numpy(
        _gt_rou_arr.reshape(-1, 1)[_flat_mask_np] if _gt_rou_arr.ndim > 0
        else np.full((int(flat_mask.sum()), 1), float(_gt_rou_arr), np.float32)
    ).to(dev, ftype)

    def _gt_rmse_metrics(ab_m, met_m, rou_m):
        """Pixel-level RMSE/MAE against GT intrinsics (only when GT is available)."""
        out = {}
        if _gt_ab_m is not None:
            scale = (_gt_ab_m * ab_m).sum(0) / (ab_m * ab_m).sum(0).clamp(1e-8)
            out["albedo_rmse"] = float((ab_m * scale - _gt_ab_m).pow(2).mean().sqrt())
            out["albedo_mae"]  = float((ab_m * scale - _gt_ab_m).abs().mean())
        out["metallic_rmse"]  = float((met_m - _gt_met_m).pow(2).mean().sqrt())
        out["metallic_mae"]   = float((met_m - _gt_met_m).abs().mean())
        out["roughness_rmse"] = float((rou_m - _gt_rou_m).pow(2).mean().sqrt())
        out["roughness_mae"]  = float((rou_m - _gt_rou_m).abs().mean())
        return out

    # ── held-out relighting metric ────────────────────────────────────────────
    _val_imgs_m = _val_sh = None
    if val_images and val_sh_coeffs is not None:
        _val_imgs_m = torch.stack([_t(v) for v in val_images]) \
            .reshape(len(val_images), -1, 3)[:, flat_mask, :]              # (V, M, 3)
        _val_sh = torch.stack([
            torch.from_numpy(_pad_sh(s)).to(dev, ftype)
            for s in val_sh_coeffs])                                       # (V, n_sh, 3)

    def _relight_metrics(ab_m, met_m, rou_m):
        """Render the held-out images with CURRENT intrinsics + their GT
        lighting; report the per-image error vs the observed val images."""
        if _val_imgs_m is None:
            return {}
        rs, ms = [], []
        with torch.no_grad():
            for k in range(_val_imgs_m.shape[0]):
                recon = shade_ct_sh(view_m, N_m, ab_m, _val_sh[k],
                                    met_m, rou_m, lut=lut, diffuse_fresnel=_diffuse_fresnel)
                d = recon - _val_imgs_m[k]
                rs.append(float(d.pow(2).mean().sqrt()))
                ms.append(float(d.abs().mean()))
        return {"relight_rmse": float(np.mean(rs)), "relight_mae": float(np.mean(ms))}

    with torch.no_grad():
        _il, _ild, _ils, _ilw, _iltv = _forward()
        _im = float(_get_met()[mask_hw].mean())
        _ir = float(_get_rou()[mask_hw].mean())
    print(f"  [init]          loss={float(_il):.3e}  data={float(_ild):.3e}"
          f"  metallic={_im:.3f}  roughness={_ir:.3f}")
    if wandb_run is not None:
        with torch.no_grad():
            _ab_t  = _fwd_albedo(albedo_param, tr_ab).detach()
            _ab_m  = _ab_t.reshape(-1, 3)[flat_mask]
            _met_m = _get_met().detach().reshape(-1, 1)[flat_mask]
            _rou_m = _get_rou().detach().reshape(-1, 1)[flat_mask]
            _init_scale = (_albedo_lighting_scale(albedo_param, tr_ab, flat_mask, _gt_ab_m)
                           if _gt_ab_m is not None else None)
        wandb_run.log(_structured_scalar_log(
            loss=_il, l_d=_ild, l_s=_ils, l_w=_ilw, l_tv=_iltv,
            loss_ml=_loss_ml[0], loss_mb=_loss_mb[0], scale3=_init_scale,
            gt_metrics=_gt_rmse_metrics(_ab_m, _met_m, _rou_m),
            relight=_relight_metrics(_ab_m, _met_m, _rou_m),
            recon_rmse=_recon_rmse[0], recon_mae=_recon_mae[0],
            lr=opt.param_groups[0]["lr"] if opt is not None else 0.0,
        ), step=-1)
    _rescale_every = cfg.get("rescale_every", 0)
    _img_batch = int(cfg.get("img_batch", 0) or 0)
    _use_img_batch = 0 < _img_batch < N_imgs
    if _use_img_batch and log_gradients:
        print("  [img_batch] disabled: incompatible with log_gradients")
        _use_img_batch = False
    _lm_info = {}
    t0 = time.perf_counter()
    for i in range(cfg["n_iter"]):
        _step[0] = i
        if _lm is not None:
            # full batch (idx=None) or a random image mini-batch of `lm_batch_size`
            if _lm_full:
                _idx, _lm_frac[0] = None, 1.0
            else:
                _idx = torch.randperm(N_imgs, device=dev)[:_lm_bs]
                _lm_frac[0] = _lm_bs / N_imgs
            _lm_info = _lm.step(_idx)
            with torch.no_grad():                       # report the full-batch loss
                loss, l_d, l_s, l_w, l_tv = _forward()
        elif opt is not None:
            if log_gradients:
                pre_raw = {n: p.data.clone() for n, p in named_params.items()}
            if _use_img_batch:
                loss, l_d, l_s, l_w, l_tv = _opt_step_img_batched(
                    opt, _forward, cfg, N_imgs, _img_batch)
            else:
                loss, l_d, l_s, l_w, l_tv = _opt_step(opt, _forward, cfg)
            if sched is not None:
                sched.step()
            if (_rescale_every > 0 and (i + 1) % _rescale_every == 0
                    and "albedo" in op and "sh" in op and _gt_ab_m is not None):
                with torch.no_grad():
                    _rescale_albedo_lighting(
                        albedo_param, [sh_coeffs], tr_ab, flat_mask, _gt_ab_m)
            if log_gradients and grad_log_dir is not None:
                _save_grad_step(i, named_params, pre_raw, gt_map_grad, fwd_map_grad,
                                _forward_components,
                                (float(loss), float(l_d), float(l_s), float(l_w), float(l_tv)),
                                grad_log_dir, flat_mask, H, W)
        else:
            with torch.no_grad():
                loss, l_d, l_s, l_w, l_tv = _forward()
        if i % cfg["log_every"] == 0:
            elapsed = time.perf_counter() - t0
            with torch.no_grad():
                met_map = _get_met().detach()
                rou_map = _get_rou().detach()
                met = float(met_map[mask_hw].mean())
                rou = float(rou_map[mask_hw].mean())
            history.append(float(loss))
            print(f"  [{i:4d}] {elapsed:6.1f}s  loss={float(loss):.3e}  data={float(l_d):.3e}"
                  f"  metallic={met:.3f}  roughness={rou:.3f}")
            if wandb_run is not None:
                with torch.no_grad():
                    _ab_t  = _fwd_albedo(albedo_param, tr_ab).detach()
                    _ab_m  = _ab_t.reshape(-1, 3)[flat_mask]
                    _met_m = met_map.reshape(-1, 1)[flat_mask]
                    _rou_m = rou_map.reshape(-1, 1)[flat_mask]
                    _step_scale = (_albedo_lighting_scale(albedo_param, tr_ab, flat_mask, _gt_ab_m)
                                   if _gt_ab_m is not None else None)
                _payload = _structured_scalar_log(
                    loss=loss, l_d=l_d, l_s=l_s, l_w=l_w, l_tv=l_tv,
                    loss_ml=_loss_ml[0], loss_mb=_loss_mb[0], scale3=_step_scale,
                    gt_metrics=_gt_rmse_metrics(_ab_m, _met_m, _rou_m),
                    relight=_relight_metrics(_ab_m, _met_m, _rou_m),
                    recon_rmse=_recon_rmse[0], recon_mae=_recon_mae[0],
                    lr=opt.param_groups[0]["lr"] if opt is not None else 0.0)
                _payload["elapsed_s"] = elapsed
                if _lm_info:
                    _payload["lm/damping"]  = _lm_info["damping"]
                    _payload["lm/accepted"] = float(_lm_info["accepted"])
                    _payload["lm/attempts"] = _lm_info["attempts"]
                wandb_run.log(_payload, step=i)
    total_time = time.perf_counter() - t0

    with torch.no_grad():
        albedo_out = _fwd_albedo(albedo_param, tr_ab).cpu().numpy()
        sh_out     = sh_coeffs.cpu().numpy()
        met_out    = _get_met().cpu().numpy()
        rou_out    = _get_rou().cpu().numpy()
        met_m      = torch.from_numpy(met_out).to(dev).reshape(-1, 1)[flat_mask]
        rou_m      = torch.from_numpy(rou_out).to(dev).reshape(-1, 1)[flat_mask]
        albedo_t2  = _fwd_albedo(albedo_param, tr_ab)
        shadings   = []
        for k in range(N_imgs):
            albedo_m = albedo_t2.reshape(-1, 3)[flat_mask]
            recon_m  = shade_ct_sh(view_m, N_m, albedo_m, sh_coeffs[k],
                                   met_m, rou_m, lut=lut, diffuse_fresnel=_diffuse_fresnel)
            s = albedo_t2.new_zeros(H, W, 3)
            s.reshape(-1, 3)[flat_mask] = recon_m
            s *= mask_t
            shadings.append(s.cpu().numpy())

    return albedo_out, sh_out, met_out, rou_out, shadings, history, total_time


# ─────────────────────────────────────── Phase 2: env-map optimizer ──────────

def _optimize_ct_env(
    images:       list,
    normals_hw:   torch.Tensor,
    frag_pos_hw:  torch.Tensor,
    mask_hw:      torch.Tensor,
    cam_pos:      torch.Tensor,
    gt_metallic:  Union[float, np.ndarray],
    gt_roughness: Union[float, np.ndarray],
    env_dirs:     np.ndarray,
    env_dw:       np.ndarray,
    cfg:          dict,
    wandb_run=None,
    env_H:        int = 64,
    env_W:        int = 128,
    gt_sh_coeffs: Optional[list] = None,
    gt_albedo:    Optional[np.ndarray] = None,
    opt_params:   Optional[frozenset] = None,
    transforms:   Optional[dict] = None,
    init_from_gt: bool = False,
    log_gradients: bool = False,
    grad_log_dir: Optional[Path] = None,
    val_images:   Optional[list] = None,
    val_sh_coeffs: Optional[list] = None,
) -> tuple:
    dev    = normals_hw.device
    ftype  = normals_hw.dtype
    H, W   = normals_hw.shape[:2]
    N_imgs = len(images)
    P      = env_dirs.shape[0]
    _n_log = N_imgs if cfg.get("wandb_max_images") is None else min(N_imgs, int(cfg["wandb_max_images"]))
    op = opt_params if opt_params is not None else _CT_ENV_PARAMS
    _diffuse_fresnel = bool(cfg.get("diffuse_fresnel", True))
    if transforms is not None:
        tr_ab, tr_met, tr_rou, tr_env = transforms["albedo"], transforms["metallic"], transforms["roughness"], transforms["env"]
    else:
        tr_ab  = cfg.get("tr_albedo",   "none")
        tr_met = cfg.get("tr_metallic",  "none")
        tr_rou = cfg.get("tr_roughness", "none")
        tr_env = cfg.get("tr_env",       "none")

    def _t(x):
        return torch.from_numpy(np.asarray(x, np.float32)).to(dev, ftype) \
            if not isinstance(x, torch.Tensor) else x.to(dev, ftype)

    imgs_t     = torch.stack([_t(img) for img in images])
    flat_mask  = mask_hw.reshape(-1)
    N_m        = normals_hw.reshape(-1, 3)[flat_mask]
    fp_m       = frag_pos_hw.reshape(-1, 3)[flat_mask]
    view_m     = _norm(cam_pos.unsqueeze(0) - fp_m)
    env_dirs_t = _t(env_dirs)
    env_dw_t   = _t(env_dw)
    mask_t     = mask_hw.unsqueeze(-1).to(dev)

    learnable    = []
    named_params = {}

    if "albedo" in op:
        if init_from_gt and gt_albedo is not None:
            base = torch.from_numpy(
                np.broadcast_to(np.asarray(gt_albedo, np.float32), (H, W, 3)).copy()
            ).to(dev, ftype)
        else:
            base = imgs_t.mean(0)
        albedo_param = _init_albedo(base, tr_ab).requires_grad_(True)
        learnable.append(albedo_param)
        named_params["albedo"] = albedo_param
    else:
        base = torch.from_numpy(
            np.broadcast_to(np.asarray(gt_albedo, np.float32), (H, W, 3)).copy()
        ).to(dev, ftype)
        albedo_param = _init_albedo(base, tr_ab)

    if "env" in op:
        if init_from_gt and gt_sh_coeffs is not None:
            gt_ef = np.stack([
                EnvMap.from_sh(SHLighting(gt_sh_coeffs[k]))._image_flat
                for k in range(N_imgs)
            ]).astype(np.float32)
            gt_ef_t = torch.from_numpy(gt_ef).to(dev, ftype)
            env_raw_params = (_softplus_inv(gt_ef_t) if tr_env == "softplus" else gt_ef_t.clone()).requires_grad_(True)
        else:
            env_raw_params = torch.zeros(N_imgs, P, 3, device=dev, dtype=ftype).requires_grad_(True)
        learnable.append(env_raw_params)
        named_params["env"] = env_raw_params
    else:
        gt_ef = np.stack([
            EnvMap.from_sh(SHLighting(gt_sh_coeffs[k]))._image_flat
            for k in range(N_imgs)
        ]).astype(np.float32)
        gt_ef_t = torch.from_numpy(gt_ef).to(dev, ftype)
        env_raw_params = _softplus_inv(gt_ef_t) if tr_env == "softplus" else gt_ef_t.clone()

    _gt_met_np = np.asarray(gt_metallic, np.float32)
    _gt_rou_np = np.asarray(gt_roughness, np.float32)
    _flat_mask_s = flat_mask.cpu().numpy()
    _gt_met_scalar = float(_gt_met_np.reshape(-1, 1)[_flat_mask_s].mean() if _gt_met_np.ndim > 0
                           else float(_gt_met_np))
    _gt_rou_scalar = float(_gt_rou_np.reshape(-1, 1)[_flat_mask_s].mean() if _gt_rou_np.ndim > 0
                           else float(_gt_rou_np))

    if "metallic" in op:
        if init_from_gt:
            metallic_raw = _init_map(_gt_met_np.reshape(H, W, 1), tr_met, dev).to(ftype).requires_grad_(True)
        else:
            _mv = 0.0 if cfg.get("init_spec_zero", False) else 0.5
            metallic_raw = _init_scalar(_mv, H, W, tr_met, dev=dev).to(ftype).requires_grad_(True)
        learnable.append(metallic_raw)
        named_params["metallic"] = metallic_raw
    else:
        if _gt_met_np.ndim > 0:
            metallic_raw = _init_map(_gt_met_np.reshape(H, W, 1), tr_met, dev).to(ftype)
        else:
            metallic_raw = _init_scalar(_gt_met_scalar, H, W, tr_met, dev=dev).to(ftype)

    if "roughness" in op:
        if init_from_gt:
            roughness_raw = _init_map(_gt_rou_np.reshape(H, W, 1), tr_rou, dev).to(ftype).requires_grad_(True)
        else:
            _rv = 1.0 if cfg.get("init_spec_zero", False) else (0.1 if cfg.get("init_roughness_zero", False) else 0.5)
            roughness_raw = _init_scalar(_rv, H, W, tr_rou, dev=dev).to(ftype).requires_grad_(True)
        learnable.append(roughness_raw)
        named_params["roughness"] = roughness_raw
    else:
        if _gt_rou_np.ndim > 0:
            roughness_raw = _init_map(_gt_rou_np.reshape(H, W, 1), tr_rou, dev).to(ftype)
        else:
            roughness_raw = _init_scalar(_gt_rou_scalar, H, W, tr_rou, dev=dev).to(ftype)

    def _get_met(): return _fwd_metallic(metallic_raw, tr_met)
    def _get_rou(): return _fwd_roughness(roughness_raw, tr_rou)

    opt   = _make_optimizer(learnable, cfg) if learnable else None
    sched = _make_scheduler(opt, cfg, cfg["n_iter"]) if opt is not None else None
    _step = [0]
    _loss_ml = [torch.zeros((), device=dev, dtype=ftype)]
    if _optimizer_name(cfg) == "LM":
        # Each env map is 2048x3 free params PER IMAGE, so P explodes (>600k at 100
        # images) and J^T J is not formable. Use LBFGS/Adam for ct_env, or ct_sh.
        raise NotImplementedError(
            "optimizer='LM' is only supported for shader='ct_sh' (ct_env's per-image "
            "env-map parameters make the P x P normal equations intractable)")

    _loss_mb = [torch.zeros((), device=dev, dtype=ftype)]
    _recon_rmse = [0.0]           # per-image recon RMSE/MAE, stashed by _forward
    _recon_mae  = [0.0]

    def _forward(img_indices=None):
        if img_indices is None:
            img_indices = range(N_imgs)
        img_indices = list(img_indices)
        frac = len(img_indices) / N_imgs
        _spec_warmup = _step[0] < cfg.get("spec_warmup_steps", 0)
        albedo      = _fwd_albedo(albedo_param, tr_ab)
        albedo_m    = albedo.reshape(-1, 3)[flat_mask]
        metallic    = _fwd_metallic(metallic_raw,  tr_met)
        roughness   = _fwd_roughness(roughness_raw, tr_rou)
        metallic_m  = metallic.reshape(-1, 1)[flat_mask]
        roughness_m = roughness.reshape(-1, 1)[flat_mask]
        if _spec_warmup:
            metallic_m  = albedo_m.new_zeros(albedo_m.shape[0], 1)
            roughness_m = albedo_m.new_ones(albedo_m.shape[0], 1)
        if _step[0] < cfg.get("min_metallic_steps", 0):
            metallic_m = metallic_m.clamp(min=0.1)
        loss_data = albedo.new_zeros(())
        _rr_sum = 0.0
        _ra_sum = 0.0
        for k in img_indices:
            env_pix_k = _fwd_env(env_raw_params[k], tr_env)
            recon_m   = shade_ct_env(view_m, N_m, albedo_m,
                                     env_pix_k, env_dirs_t, env_dw_t,
                                     metallic_m, roughness_m,
                                     sbatch=cfg.get("sbatch", 64),
                                     spec_importance=cfg.get("spec_importance", False),
                                     spec_samples=cfg.get("spec_samples", 64), diffuse_fresnel=_diffuse_fresnel)
            recon = albedo.new_zeros(H, W, 3)
            recon.reshape(-1, 3)[flat_mask] = recon_m
            loss_data = loss_data + _loss_fn(recon, imgs_t[k], mask_t, cfg["loss"],
                                             cfg.get("huber_delta", 0.05))
            with torch.no_grad():
                _rd = (recon_m - imgs_t[k].reshape(-1, 3)[flat_mask]).detach()
                _rr_sum += float(_rd.pow(2).mean().sqrt())
                _ra_sum += float(_rd.abs().mean())
        # mean over images (chunks sum to the full-batch value): loss_data is
        # comparable across dataset sizes. Regularizer lambdas stay absolute.
        loss_data = loss_data / N_imgs
        _n_sel = max(len(img_indices), 1)
        _recon_rmse[0] = _rr_sum / _n_sel
        _recon_mae[0]  = _ra_sum / _n_sel
        loss_sparse = frac * cfg["lambda_sparse"] * _tv(albedo_param.permute(2, 0, 1))
        loss_white  = frac * cfg["lambda_white"]  * (_fwd_albedo(albedo_param, tr_ab).mean() - 0.5) ** 2
        loss_tv     = frac * cfg["lambda_tv"] * (
            _tv(albedo_param.permute(2, 0, 1)) +
            _tv(metallic_raw.permute(2, 0, 1)) +
            _tv(roughness_raw.permute(2, 0, 1))
        )
        met_m = metallic.reshape(-1, 1)[flat_mask]
        loss_metallic_l1       = frac * cfg.get("lambda_metallic_l1",       0.0) * met_m.abs().mean()
        loss_metallic_binarize = frac * cfg.get("lambda_metallic_binarize",  0.0) * (met_m * (1.0 - met_m)).mean()
        _loss_ml[0] = loss_metallic_l1.detach()
        _loss_mb[0] = loss_metallic_binarize.detach()
        return loss_data + loss_sparse + loss_white + loss_tv + loss_metallic_l1 + loss_metallic_binarize, loss_data, loss_sparse, loss_white, loss_tv

    def _forward_components():
        with torch.no_grad():
            albedo_m    = _fwd_albedo(albedo_param, tr_ab).reshape(-1, 3)[flat_mask]
            metallic_m  = _fwd_metallic(metallic_raw, tr_met).reshape(-1, 1)[flat_mask]
            roughness_m = _fwd_roughness(roughness_raw, tr_rou).reshape(-1, 1)[flat_mask]
            result = []
            for k in range(N_imgs):
                env_pix_k = _fwd_env(env_raw_params[k], tr_env)
                _, comps = shade_ct_env(view_m, N_m, albedo_m,
                                        env_pix_k, env_dirs_t, env_dw_t,
                                        metallic_m, roughness_m,
                                        sbatch=cfg.get("sbatch", 64),
                                     spec_importance=cfg.get("spec_importance", False),
                                     spec_samples=cfg.get("spec_samples", 64), return_components=True)
                result.append(comps)
        return result

    gt_ef_np = np.stack([
        EnvMap.from_sh(SHLighting(gt_sh_coeffs[k]))._image_flat
        for k in range(N_imgs)
    ]).astype(np.float32) if gt_sh_coeffs is not None else None
    gt_map_grad = {
        "albedo":    np.broadcast_to(np.asarray(gt_albedo, np.float32), (H, W, 3)).copy() if gt_albedo is not None else None,
        "env":       gt_ef_np,
        "metallic":  np.broadcast_to(np.asarray(gt_metallic,  np.float32), (H, W, 1)).copy(),
        "roughness": np.broadcast_to(np.asarray(gt_roughness, np.float32), (H, W, 1)).copy(),
    }
    fwd_map_grad = {
        "albedo":    lambda p: _fwd_albedo(p, tr_ab),
        "env":       lambda p: _fwd_env(p, tr_env),
        "metallic":  lambda p: _fwd_metallic(p, tr_met),
        "roughness": lambda p: _fwd_roughness(p, tr_rou),
    }

    if log_gradients and grad_log_dir is not None:
        grad_log_dir.mkdir(parents=True, exist_ok=True)

    # ── GT RMSE helpers (mirrors _optimize_ct_sh) ─────────────────────────────
    _flat_mask_np = flat_mask.cpu().numpy()
    _gt_ab_m = (torch.from_numpy(
                    np.asarray(gt_albedo, np.float32).reshape(-1, 3)[_flat_mask_np]
                ).to(dev, ftype) if gt_albedo is not None else None)
    _gt_met_arr = np.asarray(gt_metallic, np.float32)
    _gt_met_m = torch.from_numpy(
        _gt_met_arr.reshape(-1, 1)[_flat_mask_np] if _gt_met_arr.ndim > 0
        else np.full((int(flat_mask.sum()), 1), float(_gt_met_arr), np.float32)
    ).to(dev, ftype)
    _gt_rou_arr = np.asarray(gt_roughness, np.float32)
    _gt_rou_m = torch.from_numpy(
        _gt_rou_arr.reshape(-1, 1)[_flat_mask_np] if _gt_rou_arr.ndim > 0
        else np.full((int(flat_mask.sum()), 1), float(_gt_rou_arr), np.float32)
    ).to(dev, ftype)

    def _gt_rmse_metrics(ab_m, met_m, rou_m):
        out = {}
        if _gt_ab_m is not None:
            scale = (_gt_ab_m * ab_m).sum(0) / (ab_m * ab_m).sum(0).clamp(1e-8)
            out["albedo_rmse"] = float((ab_m * scale - _gt_ab_m).pow(2).mean().sqrt())
            out["albedo_mae"]  = float((ab_m * scale - _gt_ab_m).abs().mean())
        out["metallic_rmse"]  = float((met_m - _gt_met_m).pow(2).mean().sqrt())
        out["roughness_rmse"] = float((rou_m - _gt_rou_m).pow(2).mean().sqrt())
        out["metallic_mae"]   = float((met_m - _gt_met_m).abs().mean())
        out["roughness_mae"]  = float((rou_m - _gt_rou_m).abs().mean())
        return out

    # ── held-out relighting metric ────────────────────────────────────────────
    # GT lighting expressed on the optimizer's env grid (rectified SH radiance,
    # the representation this shader lights with).
    _val_imgs_m = _val_env = None
    if val_images and val_sh_coeffs is not None:
        _val_imgs_m = torch.stack([_t(v) for v in val_images]) \
            .reshape(len(val_images), -1, 3)[:, flat_mask, :]              # (V, M, 3)
        _val_env = torch.stack([
            torch.from_numpy(np.maximum(
                build_sh_basis(env_dirs) @ np.asarray(s, np.float32), 0.0)
            ).to(dev, ftype)
            for s in val_sh_coeffs])                                       # (V, P, 3)

    def _relight_metrics(ab_m, met_m, rou_m):
        """Render the held-out images with CURRENT intrinsics + their GT
        lighting; report the per-image error vs the observed val images."""
        if _val_imgs_m is None:
            return {}
        rs, ms = [], []
        with torch.no_grad():
            for k in range(_val_imgs_m.shape[0]):
                recon = shade_ct_env(view_m, N_m, ab_m, _val_env[k],
                                     env_dirs_t, env_dw_t, met_m, rou_m,
                                     sbatch=cfg.get("sbatch", 64),
                                     spec_importance=cfg.get("spec_importance", False),
                                     spec_samples=cfg.get("spec_samples", 64), diffuse_fresnel=_diffuse_fresnel)
                d = recon - _val_imgs_m[k]
                rs.append(float(d.pow(2).mean().sqrt()))
                ms.append(float(d.abs().mean()))
        return {"relight_rmse": float(np.mean(rs)), "relight_mae": float(np.mean(ms))}

    history = []
    with torch.no_grad():
        _il, _ild, _ils, _ilw, _iltv = _forward()
        _im = float(_get_met()[mask_hw].mean())
        _ir = float(_get_rou()[mask_hw].mean())
    print(f"  [init]          loss={float(_il):.3e}  data={float(_ild):.3e}"
          f"  metallic={_im:.3f}  roughness={_ir:.3f}")
    if wandb_run is not None:
        with torch.no_grad():
            _ab_t  = _fwd_albedo(albedo_param, tr_ab).detach()
            _ab_m  = _ab_t.reshape(-1, 3)[flat_mask]
            _met_m = _get_met().detach().reshape(-1, 1)[flat_mask]
            _rou_m = _get_rou().detach().reshape(-1, 1)[flat_mask]
            _init_scale = (_albedo_lighting_scale(albedo_param, tr_ab, flat_mask, _gt_ab_m)
                           if _gt_ab_m is not None else None)
        wandb_run.log(_structured_scalar_log(
            loss=_il, l_d=_ild, l_s=_ils, l_w=_ilw, l_tv=_iltv,
            loss_ml=_loss_ml[0], loss_mb=_loss_mb[0], scale3=_init_scale,
            gt_metrics=_gt_rmse_metrics(_ab_m, _met_m, _rou_m),
            relight=_relight_metrics(_ab_m, _met_m, _rou_m),
            recon_rmse=_recon_rmse[0], recon_mae=_recon_mae[0],
            lr=opt.param_groups[0]["lr"] if opt is not None else 0.0,
        ), step=-1)
    _rescale_every = cfg.get("rescale_every", 0)
    _img_batch = int(cfg.get("img_batch", 0) or 0)
    _use_img_batch = 0 < _img_batch < N_imgs
    if _use_img_batch and log_gradients:
        print("  [img_batch] disabled: incompatible with log_gradients")
        _use_img_batch = False
    t0 = time.perf_counter()
    for i in range(cfg["n_iter"]):
        _step[0] = i
        if opt is not None:
            if log_gradients:
                pre_raw = {n: p.data.clone() for n, p in named_params.items()}
            if _use_img_batch:
                loss, l_d, l_s, l_w, l_tv = _opt_step_img_batched(
                    opt, _forward, cfg, N_imgs, _img_batch)
            else:
                loss, l_d, l_s, l_w, l_tv = _opt_step(opt, _forward, cfg)
            if sched is not None:
                sched.step()
            if (_rescale_every > 0 and (i + 1) % _rescale_every == 0
                    and "albedo" in op and "env" in op and _gt_ab_m is not None):
                with torch.no_grad():
                    _rescale_albedo_lighting(
                        albedo_param, [env_raw_params], tr_ab, flat_mask, _gt_ab_m)
            if log_gradients and grad_log_dir is not None:
                _save_grad_step(i, named_params, pre_raw, gt_map_grad, fwd_map_grad,
                                _forward_components,
                                (float(loss), float(l_d), float(l_s), float(l_w), float(l_tv)),
                                grad_log_dir, flat_mask, H, W)
        else:
            with torch.no_grad():
                loss, l_d, l_s, l_w, l_tv = _forward()
        if i % cfg["log_every"] == 0:
            elapsed = time.perf_counter() - t0
            with torch.no_grad():
                met_map = _get_met().detach()
                rou_map = _get_rou().detach()
                met = float(met_map[mask_hw].mean())
                rou = float(rou_map[mask_hw].mean())
            history.append(float(loss))
            print(f"  [{i:4d}] {elapsed:6.1f}s  loss={float(loss):.3e}  data={float(l_d):.3e}"
                  f"  metallic={met:.3f}  roughness={rou:.3f}")
            if wandb_run is not None:
                with torch.no_grad():
                    _ab_t  = _fwd_albedo(albedo_param, tr_ab).detach()
                    _ab_m  = _ab_t.reshape(-1, 3)[flat_mask]
                    _met_m = met_map.reshape(-1, 1)[flat_mask]
                    _rou_m = rou_map.reshape(-1, 1)[flat_mask]
                    _step_scale = (_albedo_lighting_scale(albedo_param, tr_ab, flat_mask, _gt_ab_m)
                                   if _gt_ab_m is not None else None)
                _payload = _structured_scalar_log(
                    loss=loss, l_d=l_d, l_s=l_s, l_w=l_w, l_tv=l_tv,
                    loss_ml=_loss_ml[0], loss_mb=_loss_mb[0], scale3=_step_scale,
                    gt_metrics=_gt_rmse_metrics(_ab_m, _met_m, _rou_m),
                    relight=_relight_metrics(_ab_m, _met_m, _rou_m),
                    recon_rmse=_recon_rmse[0], recon_mae=_recon_mae[0],
                    lr=opt.param_groups[0]["lr"] if opt is not None else 0.0)
                _payload["elapsed_s"] = elapsed
                wandb_run.log(_payload, step=i)

    total_time = time.perf_counter() - t0

    with torch.no_grad():
        albedo_out   = _fwd_albedo(albedo_param, tr_ab).cpu().numpy()
        env_maps_out = _fwd_env(env_raw_params, tr_env).cpu().numpy()
        met_out      = _fwd_metallic(metallic_raw, tr_met).cpu().numpy()
        rou_out      = _fwd_roughness(roughness_raw, tr_rou).cpu().numpy()
        met_m        = torch.from_numpy(met_out).to(dev).reshape(-1, 1)[flat_mask]
        rou_m        = torch.from_numpy(rou_out).to(dev).reshape(-1, 1)[flat_mask]
        albedo_t2    = _fwd_albedo(albedo_param, tr_ab)
        shadings = []
        for k in range(N_imgs):
            albedo_m  = albedo_t2.reshape(-1, 3)[flat_mask]
            env_pix_k = torch.from_numpy(env_maps_out[k]).to(dev)
            recon_m   = shade_ct_env(view_m, N_m, albedo_m,
                                     env_pix_k, env_dirs_t, env_dw_t,
                                     met_m, rou_m,
                                     sbatch=cfg.get("sbatch", 64),
                                     spec_importance=cfg.get("spec_importance", False),
                                     spec_samples=cfg.get("spec_samples", 64),
                                     diffuse_fresnel=_diffuse_fresnel)
            s = albedo_t2.new_zeros(H, W, 3)
            s.reshape(-1, 3)[flat_mask] = recon_m
            s *= mask_t
            shadings.append(s.cpu().numpy())

    return albedo_out, env_maps_out, met_out, rou_out, shadings, history, total_time


# ─────────────────────────────────────── Phase 2: Phong SH optimizer ─────────

def _optimize_phong_sh(
    images:       list,
    normals_hw:   torch.Tensor,
    frag_pos_hw:  torch.Tensor,
    mask_hw:      torch.Tensor,
    cam_pos:      torch.Tensor,
    gt_shininess: float,
    gt_ks:        float,
    ka:           float,
    kd:           float,
    cfg:          dict,
    wandb_run=None,
    gt_sh_coeffs: Optional[list] = None,
    gt_albedo:    Optional[np.ndarray] = None,
    opt_params:   Optional[frozenset] = None,
    transforms:   Optional[dict] = None,
    init_from_gt: bool = False,
    log_gradients: bool = False,
    grad_log_dir: Optional[Path] = None,
) -> tuple:
    dev    = normals_hw.device
    H, W   = normals_hw.shape[:2]
    N_imgs = len(images)
    _n_log = N_imgs if cfg.get("wandb_max_images") is None else min(N_imgs, int(cfg["wandb_max_images"]))
    s_min  = cfg.get("shininess_min", DEFAULT_CFG["shininess_min"])
    s_max  = cfg.get("shininess_max", DEFAULT_CFG["shininess_max"])
    op     = opt_params if opt_params is not None else _PHONG_SH_PARAMS
    tr     = transforms if transforms is not None else NAMED_TRANSFORMS["none"]
    tr_ab, tr_shin, tr_ks = tr["albedo"], tr["shininess"], tr["ks"]

    def _t(x):
        return torch.from_numpy(np.asarray(x, np.float32)).to(dev) \
            if not isinstance(x, torch.Tensor) else x.to(dev, torch.float32)

    imgs_t    = torch.stack([_t(img) for img in images])
    flat_mask = mask_hw.reshape(-1)
    N_m       = normals_hw.reshape(-1, 3)[flat_mask]
    fp_m      = frag_pos_hw.reshape(-1, 3)[flat_mask]
    view_m    = _norm(cam_pos.unsqueeze(0) - fp_m)
    mask_t    = mask_hw.unsqueeze(-1).to(dev)

    learnable    = []
    named_params = {}

    if "albedo" in op:
        if init_from_gt and gt_albedo is not None:
            base = torch.from_numpy(
                np.broadcast_to(np.asarray(gt_albedo, np.float32), (H, W, 3)).copy()
            ).to(dev)
        else:
            base = imgs_t.mean(0)
        albedo_param = _init_albedo(base, tr_ab).requires_grad_(True)
        learnable.append(albedo_param)
        named_params["albedo"] = albedo_param
    else:
        base = torch.from_numpy(
            np.broadcast_to(np.asarray(gt_albedo, np.float32), (H, W, 3)).copy()
        ).to(dev)
        albedo_param = _init_albedo(base, tr_ab)

    if "sh" in op:
        if init_from_gt and gt_sh_coeffs is not None:
            sh_coeffs = torch.stack([
                torch.from_numpy(gt_sh_coeffs[k]).to(dev) for k in range(N_imgs)
            ]).requires_grad_(True)
        else:
            sh_init = torch.zeros(N_imgs, 9, 3, device=dev)
            sh_init[:, 0, :] = 1.5
            sh_coeffs = sh_init.clone().requires_grad_(True)
        learnable.append(sh_coeffs)
        named_params["sh"] = sh_coeffs
    else:
        sh_coeffs = torch.stack([
            torch.from_numpy(gt_sh_coeffs[k]).to(dev) for k in range(N_imgs)
        ])

    if "shininess" in op:
        if init_from_gt:
            _gt_shin = np.broadcast_to(np.asarray(gt_shininess, np.float32), (H, W, 1)).copy()
            if tr_shin == "sigmoid":
                sv = np.clip((_gt_shin - s_min) / (s_max - s_min), 1e-6, 1 - 1e-6)
                raw_arr = np.log(sv / (1 - sv))
            elif tr_shin == "log":
                raw_arr = np.log(_gt_shin.clip(1e-7))
            else:
                raw_arr = _gt_shin
            shininess_raw = torch.from_numpy(raw_arr.astype(np.float32)).to(dev).requires_grad_(True)
        else:
            if tr_shin == "sigmoid":
                s0 = 0.0
            elif tr_shin == "log":
                s0 = float(np.log(0.5 * (s_min + s_max)))
            else:
                s0 = 0.5 * (s_min + s_max)
            shininess_raw = torch.full((H, W, 1), s0, dtype=torch.float32, device=dev).requires_grad_(True)
        learnable.append(shininess_raw)
        named_params["shininess"] = shininess_raw
    else:
        _gt_shin_s = float(np.asarray(gt_shininess).mean())
        if tr_shin == "sigmoid":
            sv = float(np.clip((_gt_shin_s - s_min) / (s_max - s_min), 1e-6, 1 - 1e-6))
            shin_raw_val = float(np.log(sv / (1 - sv)))
        elif tr_shin == "log":
            shin_raw_val = float(np.log(_gt_shin_s))
        else:
            shin_raw_val = float(_gt_shin_s)
        shininess_raw = torch.full((H, W, 1), shin_raw_val, dtype=torch.float32, device=dev)

    if "ks" in op:
        if init_from_gt:
            _gt_ks = np.broadcast_to(np.asarray(gt_ks, np.float32), (H, W, 1)).copy()
            if tr_ks == "sigmoid":
                kv = np.clip(_gt_ks, 1e-6, 1 - 1e-6)
                raw_arr = np.log(kv / (1 - kv))
            else:
                raw_arr = _gt_ks
            ks_raw = torch.from_numpy(raw_arr.astype(np.float32)).to(dev).requires_grad_(True)
        else:
            ks0 = (-10.0 if tr_ks == "sigmoid" else 0.0) if cfg.get("init_spec_zero", False) \
                  else (0.0 if tr_ks == "sigmoid" else 0.5)
            ks_raw = torch.full((H, W, 1), ks0, dtype=torch.float32, device=dev).requires_grad_(True)
        learnable.append(ks_raw)
        named_params["ks"] = ks_raw
    else:
        _gt_ks_s = float(np.asarray(gt_ks).mean())
        if tr_ks == "sigmoid":
            kv = float(np.clip(_gt_ks_s, 1e-6, 1 - 1e-6))
            ks_raw_val = float(np.log(kv / (1 - kv)))
        else:
            ks_raw_val = float(_gt_ks_s)
        ks_raw = torch.full((H, W, 1), ks_raw_val, dtype=torch.float32, device=dev)

    opt   = _make_optimizer(learnable, cfg)
    sched = _make_scheduler(opt, cfg, cfg["n_iter"])
    _step = [0]

    def _forward():
        _spec_warmup = _step[0] < cfg.get("spec_warmup_steps", 0)
        albedo      = _fwd_albedo(albedo_param, tr_ab)
        albedo_m    = albedo.reshape(-1, 3)[flat_mask]
        shininess_m = _fwd_shininess(shininess_raw, tr_shin, s_min, s_max).reshape(-1, 1)[flat_mask]
        ks_m        = _fwd_ks(ks_raw, tr_ks).reshape(-1, 1)[flat_mask]
        if _spec_warmup:
            ks_m = albedo_m.new_zeros(albedo_m.shape[0], 1)
        loss_data   = albedo.new_zeros(())
        for k in range(N_imgs):
            recon_m = shade_phong_sh(view_m, N_m, ka, kd, ks_m, shininess_m,
                                     albedo_m, sh_coeffs[k])
            recon   = albedo.new_zeros(H, W, 3)
            recon.reshape(-1, 3)[flat_mask] = recon_m
            loss_data = loss_data + _loss_fn(recon, imgs_t[k], mask_t, cfg["loss"])
        loss_sparse = cfg["lambda_sparse"] * _tv(albedo_param.permute(2, 0, 1))
        loss_white  = cfg["lambda_white"]  * (_fwd_albedo(albedo_param, tr_ab).mean() - 0.5) ** 2
        loss_tv     = cfg["lambda_tv"] * (
            _tv(albedo_param.permute(2, 0, 1)) +
            _tv(shininess_raw.permute(2, 0, 1)) +
            _tv(ks_raw.permute(2, 0, 1))
        )
        return loss_data + loss_sparse + loss_white + loss_tv, loss_data, loss_sparse, loss_white, loss_tv

    def _forward_components():
        with torch.no_grad():
            albedo_m    = _fwd_albedo(albedo_param, tr_ab).reshape(-1, 3)[flat_mask]
            shininess_m = _fwd_shininess(shininess_raw, tr_shin, s_min, s_max).reshape(-1, 1)[flat_mask]
            ks_m        = _fwd_ks(ks_raw, tr_ks).reshape(-1, 1)[flat_mask]
            result = []
            for k in range(N_imgs):
                _, comps = shade_phong_sh(view_m, N_m, ka, kd, ks_m, shininess_m,
                                          albedo_m, sh_coeffs[k], return_components=True)
                result.append(comps)
        return result

    gt_map_grad = {
        "albedo":    np.broadcast_to(np.asarray(gt_albedo, np.float32), (H, W, 3)).copy() if gt_albedo is not None else None,
        "sh":        np.stack([_pad_sh(s) for s in gt_sh_coeffs]).astype(np.float32) if gt_sh_coeffs is not None else None,
        "shininess": np.broadcast_to(np.asarray(gt_shininess, np.float32), (H, W, 1)).copy(),
        "ks":        np.broadcast_to(np.asarray(gt_ks, np.float32), (H, W, 1)).copy(),
    }
    fwd_map_grad = {
        "albedo":    lambda p: _fwd_albedo(p, tr_ab),
        "sh":        lambda p: p,
        "shininess": lambda p: _fwd_shininess(p, tr_shin, s_min, s_max),
        "ks":        lambda p: _fwd_ks(p, tr_ks),
    }

    if log_gradients and grad_log_dir is not None:
        grad_log_dir.mkdir(parents=True, exist_ok=True)

    _gt_shin_mean = float(np.asarray(gt_shininess).mean())
    _gt_ks_mean   = float(np.asarray(gt_ks).mean())
    history = []
    with torch.no_grad():
        _il, _ild, _ils, _ilw, _iltv = _forward()
        _is = float(_fwd_shininess(shininess_raw, tr_shin, s_min, s_max)[mask_hw].mean())
        _ik = float(_fwd_ks(ks_raw, tr_ks)[mask_hw].mean())
    print(f"  [init]          loss={float(_il):.3e}  data={float(_ild):.3e}"
          f"  shininess={_is:.1f}  ks={_ik:.3f}")
    if wandb_run is not None:
        with torch.no_grad():
            _shin_map  = _fwd_shininess(shininess_raw, tr_shin, s_min, s_max).detach()
            _ks_map    = _fwd_ks(ks_raw, tr_ks).detach()
            _est_sh_np = sh_coeffs.detach().cpu().numpy()
            _ab_t   = _fwd_albedo(albedo_param, tr_ab).detach()
            _ab_m   = _ab_t.reshape(-1, 3)[flat_mask]
            _shin_m = _shin_map.reshape(-1, 1)[flat_mask]
            _ks_m   = _ks_map.reshape(-1, 1)[flat_mask]
            _recons, _errs = [], []
            for _k in range(_n_log):
                _r = _ab_t.new_zeros(H, W, 3)
                _r.reshape(-1, 3)[flat_mask] = shade_phong_sh(
                    view_m, N_m, ka, kd, _ks_m, _shin_m, _ab_m, sh_coeffs[_k])
                _r *= mask_t
                _recons.append(wandb.Image(_r.float().cpu().numpy()))
                _errs.append(wandb.Image((_r - imgs_t[_k]).abs().mul(mask_t).float().cpu().numpy()))
        wandb_run.log({
            "loss": float(_il), "loss_data": float(_ild), "data_rmse": float(_ild) ** 0.5,
            "loss_sparse": float(_ils), "loss_white": float(_ilw), "loss_tv": float(_iltv),
            "pred_albedo":     wandb.Image(_ab_t.float().cpu().numpy()),
            "pred_shininess":  wandb.Image((_shin_map / s_max).squeeze(-1).cpu().numpy()),
            "pred_ks":         wandb.Image(_ks_map.squeeze(-1).cpu().numpy()),
            "est_sh_env_maps": [wandb.Image(_sh_coeffs_to_env_img(_est_sh_np[k])) for k in range(_n_log)],
            "recons":          _recons,
            "recon_err_maps":  _errs,
            "shininess_mean":     _is,
            "ks_mean":            _ik,
            "shininess_err_mean": abs(_is - _gt_shin_mean),
            "ks_err_mean":        abs(_ik - _gt_ks_mean),
            "lr": opt.param_groups[0]["lr"] if opt is not None else 0.0,
        }, step=-1)
    t0 = time.perf_counter()
    for i in range(cfg["n_iter"]):
        _step[0] = i
        if log_gradients:
            pre_raw = {n: p.data.clone() for n, p in named_params.items()}
        loss, l_d, l_s, l_w, l_tv = _opt_step(opt, _forward, cfg)
        if sched is not None:
            sched.step()
        if log_gradients and grad_log_dir is not None:
            _save_grad_step(i, named_params, pre_raw, gt_map_grad, fwd_map_grad,
                            _forward_components,
                            (float(loss), float(l_d), float(l_s), float(l_w), float(l_tv)),
                            grad_log_dir, flat_mask, H, W)
        if i % cfg["log_every"] == 0:
            elapsed = time.perf_counter() - t0
            with torch.no_grad():
                shin_map = _fwd_shininess(shininess_raw, tr_shin, s_min, s_max).detach()
                ks_map   = _fwd_ks(ks_raw, tr_ks).detach()
                shin_val = float(shin_map[mask_hw].mean())
                ks_val   = float(ks_map[mask_hw].mean())
            history.append(float(loss))
            print(f"  [{i:4d}] {elapsed:6.1f}s  loss={float(loss):.3e}  data={float(l_d):.3e}"
                  f"  shininess={shin_val:.1f}  ks={ks_val:.3f}")
            if wandb_run is not None:
                est_sh_np = sh_coeffs.detach().cpu().numpy()
                with torch.no_grad():
                    _ab_t   = _fwd_albedo(albedo_param, tr_ab).detach()
                    _ab_m   = _ab_t.reshape(-1, 3)[flat_mask]
                    _shin_m = shin_map.reshape(-1, 1)[flat_mask]
                    _ks_m   = ks_map.reshape(-1, 1)[flat_mask]
                    _recons, _errs = [], []
                    for _k in range(_n_log):
                        _r = _ab_t.new_zeros(H, W, 3)
                        _r.reshape(-1, 3)[flat_mask] = shade_phong_sh(
                            view_m, N_m, ka, kd, _ks_m, _shin_m, _ab_m, sh_coeffs[_k])
                        _r *= mask_t
                        _recons.append(wandb.Image(_r.float().cpu().numpy()))
                        _errs.append(wandb.Image((_r - imgs_t[_k]).abs().mul(mask_t).float().cpu().numpy()))
                wandb_run.log({
                    "loss": float(loss), "loss_data": float(l_d), "data_rmse": float(l_d) ** 0.5,
                    "loss_sparse": float(l_s), "loss_white": float(l_w), "loss_tv": float(l_tv),
                    "elapsed_s": elapsed,
                    "pred_albedo":     wandb.Image(
                        _fwd_albedo(albedo_param, tr_ab).detach().float().cpu().numpy()),
                    "pred_shininess":  wandb.Image(
                        (shin_map / s_max).squeeze(-1).cpu().numpy()),
                    "pred_ks":         wandb.Image(ks_map.squeeze(-1).cpu().numpy()),
                    "est_sh_env_maps": [wandb.Image(_sh_coeffs_to_env_img(est_sh_np[k]))
                                        for k in range(_n_log)],
                    "recons":          _recons,
                    "recon_err_maps":  _errs,
                    "shininess_mean":     shin_val,
                    "ks_mean":            ks_val,
                    "shininess_err_mean": abs(shin_val - _gt_shin_mean),
                    "ks_err_mean":        abs(ks_val   - _gt_ks_mean),
                    "lr": opt.param_groups[0]["lr"] if opt is not None else 0.0,
                }, step=i)
    total_time = time.perf_counter() - t0

    with torch.no_grad():
        albedo_out = _fwd_albedo(albedo_param, tr_ab).cpu().numpy()
        sh_out     = sh_coeffs.cpu().numpy()
        shin_out   = _fwd_shininess(shininess_raw, tr_shin, s_min, s_max).cpu().numpy()
        ks_out     = _fwd_ks(ks_raw, tr_ks).cpu().numpy()
        shin_m_t   = torch.from_numpy(shin_out).to(dev).reshape(-1, 1)[flat_mask]
        ks_m_t     = torch.from_numpy(ks_out).to(dev).reshape(-1, 1)[flat_mask]
        albedo_t2  = _fwd_albedo(albedo_param, tr_ab)
        shadings   = []
        for k in range(N_imgs):
            albedo_m = albedo_t2.reshape(-1, 3)[flat_mask]
            recon_m  = shade_phong_sh(view_m, N_m, ka, kd, ks_m_t, shin_m_t,
                                      albedo_m, sh_coeffs[k])
            s = albedo_t2.new_zeros(H, W, 3)
            s.reshape(-1, 3)[flat_mask] = recon_m
            s *= mask_t
            shadings.append(s.cpu().numpy())

    return albedo_out, sh_out, shin_out, ks_out, shadings, history, total_time


# ─────────────────────────────────────── Phase 2: Phong env-map optimizer ────

def _optimize_phong_env(
    images:       list,
    normals_hw:   torch.Tensor,
    frag_pos_hw:  torch.Tensor,
    mask_hw:      torch.Tensor,
    cam_pos:      torch.Tensor,
    gt_shininess: float,
    gt_ks:        float,
    ka:           float,
    kd:           float,
    env_dirs:     np.ndarray,
    env_dw:       np.ndarray,
    cfg:          dict,
    wandb_run=None,
    env_H:        int = 64,
    env_W:        int = 128,
    gt_sh_coeffs: Optional[list] = None,
    gt_albedo:    Optional[np.ndarray] = None,
    opt_params:   Optional[frozenset] = None,
    transforms:   Optional[dict] = None,
    init_from_gt: bool = False,
    log_gradients: bool = False,
    grad_log_dir: Optional[Path] = None,
) -> tuple:
    dev    = normals_hw.device
    H, W   = normals_hw.shape[:2]
    N_imgs = len(images)
    _n_log = N_imgs if cfg.get("wandb_max_images") is None else min(N_imgs, int(cfg["wandb_max_images"]))
    P      = env_dirs.shape[0]
    s_min  = cfg.get("shininess_min", DEFAULT_CFG["shininess_min"])
    s_max  = cfg.get("shininess_max", DEFAULT_CFG["shininess_max"])
    op     = opt_params if opt_params is not None else _PHONG_ENV_PARAMS
    tr     = transforms if transforms is not None else NAMED_TRANSFORMS["none"]
    tr_ab, tr_shin, tr_ks, tr_env = tr["albedo"], tr["shininess"], tr["ks"], tr["env"]

    def _t(x):
        return torch.from_numpy(np.asarray(x, np.float32)).to(dev) \
            if not isinstance(x, torch.Tensor) else x.to(dev, torch.float32)

    imgs_t     = torch.stack([_t(img) for img in images])
    flat_mask  = mask_hw.reshape(-1)
    N_m        = normals_hw.reshape(-1, 3)[flat_mask]
    fp_m       = frag_pos_hw.reshape(-1, 3)[flat_mask]
    view_m     = _norm(cam_pos.unsqueeze(0) - fp_m)
    env_dirs_t = _t(env_dirs)
    env_dw_t   = _t(env_dw)
    mask_t     = mask_hw.unsqueeze(-1).to(dev)

    learnable    = []
    named_params = {}

    if "albedo" in op:
        if init_from_gt and gt_albedo is not None:
            base = torch.from_numpy(
                np.broadcast_to(np.asarray(gt_albedo, np.float32), (H, W, 3)).copy()
            ).to(dev)
        else:
            base = imgs_t.mean(0)
        albedo_param = _init_albedo(base, tr_ab).requires_grad_(True)
        learnable.append(albedo_param)
        named_params["albedo"] = albedo_param
    else:
        base = torch.from_numpy(
            np.broadcast_to(np.asarray(gt_albedo, np.float32), (H, W, 3)).copy()
        ).to(dev)
        albedo_param = _init_albedo(base, tr_ab)

    if "env" in op:
        if init_from_gt and gt_sh_coeffs is not None:
            gt_ef = np.stack([
                EnvMap.from_sh(SHLighting(gt_sh_coeffs[k]))._image_flat
                for k in range(N_imgs)
            ]).astype(np.float32)
            gt_ef_t = torch.from_numpy(gt_ef).to(dev)
            env_raw_params = (_softplus_inv(gt_ef_t) if tr_env == "softplus" else gt_ef_t.clone()).requires_grad_(True)
        else:
            env_raw_params = torch.zeros(N_imgs, P, 3, device=dev).requires_grad_(True)
        learnable.append(env_raw_params)
        named_params["env"] = env_raw_params
    else:
        gt_ef = np.stack([
            EnvMap.from_sh(SHLighting(gt_sh_coeffs[k]))._image_flat
            for k in range(N_imgs)
        ]).astype(np.float32)
        gt_ef_t = torch.from_numpy(gt_ef).to(dev)
        env_raw_params = _softplus_inv(gt_ef_t) if tr_env == "softplus" else gt_ef_t.clone()

    if "shininess" in op:
        if init_from_gt:
            _gt_shin = np.broadcast_to(np.asarray(gt_shininess, np.float32), (H, W, 1)).copy()
            if tr_shin == "sigmoid":
                sv = np.clip((_gt_shin - s_min) / (s_max - s_min), 1e-6, 1 - 1e-6)
                raw_arr = np.log(sv / (1 - sv))
            elif tr_shin == "log":
                raw_arr = np.log(_gt_shin.clip(1e-7))
            else:
                raw_arr = _gt_shin
            shininess_raw = torch.from_numpy(raw_arr.astype(np.float32)).to(dev).requires_grad_(True)
        else:
            if tr_shin == "sigmoid":
                s0 = 0.0
            elif tr_shin == "log":
                s0 = float(np.log(0.5 * (s_min + s_max)))
            else:
                s0 = 0.5 * (s_min + s_max)
            shininess_raw = torch.full((H, W, 1), s0, dtype=torch.float32, device=dev).requires_grad_(True)
        learnable.append(shininess_raw)
        named_params["shininess"] = shininess_raw
    else:
        _gt_shin_s = float(np.asarray(gt_shininess).mean())
        if tr_shin == "sigmoid":
            sv = float(np.clip((_gt_shin_s - s_min) / (s_max - s_min), 1e-6, 1 - 1e-6))
            shin_raw_val = float(np.log(sv / (1 - sv)))
        elif tr_shin == "log":
            shin_raw_val = float(np.log(_gt_shin_s))
        else:
            shin_raw_val = float(_gt_shin_s)
        shininess_raw = torch.full((H, W, 1), shin_raw_val, dtype=torch.float32, device=dev)

    if "ks" in op:
        if init_from_gt:
            _gt_ks = np.broadcast_to(np.asarray(gt_ks, np.float32), (H, W, 1)).copy()
            if tr_ks == "sigmoid":
                kv = np.clip(_gt_ks, 1e-6, 1 - 1e-6)
                raw_arr = np.log(kv / (1 - kv))
            else:
                raw_arr = _gt_ks
            ks_raw = torch.from_numpy(raw_arr.astype(np.float32)).to(dev).requires_grad_(True)
        else:
            ks0 = (-10.0 if tr_ks == "sigmoid" else 0.0) if cfg.get("init_spec_zero", False) \
                  else (0.0 if tr_ks == "sigmoid" else 0.5)
            ks_raw = torch.full((H, W, 1), ks0, dtype=torch.float32, device=dev).requires_grad_(True)
        learnable.append(ks_raw)
        named_params["ks"] = ks_raw
    else:
        _gt_ks_s = float(np.asarray(gt_ks).mean())
        if tr_ks == "sigmoid":
            kv = float(np.clip(_gt_ks_s, 1e-6, 1 - 1e-6))
            ks_raw_val = float(np.log(kv / (1 - kv)))
        else:
            ks_raw_val = float(_gt_ks_s)
        ks_raw = torch.full((H, W, 1), ks_raw_val, dtype=torch.float32, device=dev)

    opt   = _make_optimizer(learnable, cfg)
    sched = _make_scheduler(opt, cfg, cfg["n_iter"])
    _step = [0]

    def _forward(img_indices=None):
        if img_indices is None:
            img_indices = range(N_imgs)
        img_indices = list(img_indices)
        frac = len(img_indices) / N_imgs
        _spec_warmup = _step[0] < cfg.get("spec_warmup_steps", 0)
        albedo      = _fwd_albedo(albedo_param, tr_ab)
        albedo_m    = albedo.reshape(-1, 3)[flat_mask]
        shininess_m = _fwd_shininess(shininess_raw, tr_shin, s_min, s_max).reshape(-1, 1)[flat_mask]
        ks_m        = _fwd_ks(ks_raw, tr_ks).reshape(-1, 1)[flat_mask]
        if _spec_warmup:
            ks_m = albedo_m.new_zeros(albedo_m.shape[0], 1)
        loss_data   = albedo.new_zeros(())
        for k in img_indices:
            env_pix_k = _fwd_env(env_raw_params[k], tr_env)
            recon_m   = shade_phong_env(view_m, N_m, albedo_m,
                                        env_pix_k, env_dirs_t, env_dw_t,
                                        ka, kd, ks_m, shininess_m,
                                        sbatch=cfg.get("sbatch", 64),
                                     spec_importance=cfg.get("spec_importance", False),
                                     spec_samples=cfg.get("spec_samples", 64))
            recon = albedo.new_zeros(H, W, 3)
            recon.reshape(-1, 3)[flat_mask] = recon_m
            loss_data = loss_data + _loss_fn(recon, imgs_t[k], mask_t, cfg["loss"])
        loss_sparse = frac * cfg["lambda_sparse"] * _tv(albedo_param.permute(2, 0, 1))
        loss_white  = frac * cfg["lambda_white"]  * (_fwd_albedo(albedo_param, tr_ab).mean() - 0.5) ** 2
        loss_tv     = frac * cfg["lambda_tv"] * (
            _tv(albedo_param.permute(2, 0, 1)) +
            _tv(shininess_raw.permute(2, 0, 1)) +
            _tv(ks_raw.permute(2, 0, 1))
        )
        return loss_data + loss_sparse + loss_white + loss_tv, loss_data, loss_sparse, loss_white, loss_tv

    def _forward_components():
        with torch.no_grad():
            albedo_m    = _fwd_albedo(albedo_param, tr_ab).reshape(-1, 3)[flat_mask]
            shininess_m = _fwd_shininess(shininess_raw, tr_shin, s_min, s_max).reshape(-1, 1)[flat_mask]
            ks_m        = _fwd_ks(ks_raw, tr_ks).reshape(-1, 1)[flat_mask]
            result = []
            for k in range(N_imgs):
                env_pix_k = _fwd_env(env_raw_params[k], tr_env)
                _, comps = shade_phong_env(view_m, N_m, albedo_m,
                                           env_pix_k, env_dirs_t, env_dw_t,
                                           ka, kd, ks_m, shininess_m,
                                           sbatch=cfg.get("sbatch", 64),
                                     spec_importance=cfg.get("spec_importance", False),
                                     spec_samples=cfg.get("spec_samples", 64), return_components=True)
                result.append(comps)
        return result

    gt_ef_np = np.stack([
        EnvMap.from_sh(SHLighting(gt_sh_coeffs[k]))._image_flat
        for k in range(N_imgs)
    ]).astype(np.float32) if gt_sh_coeffs is not None else None
    gt_map_grad = {
        "albedo":    np.broadcast_to(np.asarray(gt_albedo, np.float32), (H, W, 3)).copy() if gt_albedo is not None else None,
        "env":       gt_ef_np,
        "shininess": np.broadcast_to(np.asarray(gt_shininess, np.float32), (H, W, 1)).copy(),
        "ks":        np.broadcast_to(np.asarray(gt_ks, np.float32), (H, W, 1)).copy(),
    }
    fwd_map_grad = {
        "albedo":    lambda p: _fwd_albedo(p, tr_ab),
        "env":       lambda p: _fwd_env(p, tr_env),
        "shininess": lambda p: _fwd_shininess(p, tr_shin, s_min, s_max),
        "ks":        lambda p: _fwd_ks(p, tr_ks),
    }

    if log_gradients and grad_log_dir is not None:
        grad_log_dir.mkdir(parents=True, exist_ok=True)

    _gt_shin_mean = float(np.asarray(gt_shininess).mean())
    _gt_ks_mean   = float(np.asarray(gt_ks).mean())
    history = []
    with torch.no_grad():
        _il, _ild, _ils, _ilw, _iltv = _forward()
        _is = float(_fwd_shininess(shininess_raw, tr_shin, s_min, s_max)[mask_hw].mean())
        _ik = float(_fwd_ks(ks_raw, tr_ks)[mask_hw].mean())
    print(f"  [init]          loss={float(_il):.3e}  data={float(_ild):.3e}"
          f"  shininess={_is:.1f}  ks={_ik:.3f}")
    if wandb_run is not None:
        with torch.no_grad():
            _env_pix_all = _fwd_env(env_raw_params, tr_env).detach()
            _env_imgs_k  = [_env_flat_to_img(_env_pix_all[k].cpu().numpy(), env_H, env_W) for k in range(_n_log)]
            _env_avg_img = _env_flat_to_img(_env_pix_all.mean(0).cpu().numpy(), env_H, env_W)
            _shin_map = _fwd_shininess(shininess_raw, tr_shin, s_min, s_max).detach()
            _ks_map   = _fwd_ks(ks_raw, tr_ks).detach()
            _ab_t   = _fwd_albedo(albedo_param, tr_ab).detach()
            _ab_m   = _ab_t.reshape(-1, 3)[flat_mask]
            _shin_m = _shin_map.reshape(-1, 1)[flat_mask]
            _ks_m   = _ks_map.reshape(-1, 1)[flat_mask]
            _recons, _errs = [], []
            for _k in range(_n_log):
                _r = _ab_t.new_zeros(H, W, 3)
                _r.reshape(-1, 3)[flat_mask] = shade_phong_env(
                    view_m, N_m, _ab_m, _env_pix_all[_k], env_dirs_t, env_dw_t,
                    ka, kd, _ks_m, _shin_m, sbatch=cfg.get("sbatch", 64),
                                     spec_importance=cfg.get("spec_importance", False),
                                     spec_samples=cfg.get("spec_samples", 64))
                _r *= mask_t
                _recons.append(wandb.Image(_r.float().cpu().numpy()))
                _errs.append(wandb.Image((_r - imgs_t[_k]).abs().mul(mask_t).float().cpu().numpy()))
        wandb_run.log({
            "loss": float(_il), "loss_data": float(_ild), "data_rmse": float(_ild) ** 0.5,
            "loss_sparse": float(_ils), "loss_white": float(_ilw), "loss_tv": float(_iltv),
            "pred_albedo":    wandb.Image(_ab_t.float().cpu().numpy()),
            "pred_shininess": wandb.Image((_shin_map / s_max).squeeze(-1).cpu().numpy()),
            "pred_ks":        wandb.Image(_ks_map.squeeze(-1).cpu().numpy()),
            "est_env_maps":   [wandb.Image(img) for img in _env_imgs_k],
            "env_map_avg":    wandb.Image(_env_avg_img),
            "recons":         _recons,
            "recon_err_maps": _errs,
            "shininess_mean":     _is,
            "ks_mean":            _ik,
            "shininess_err_mean": abs(_is - _gt_shin_mean),
            "ks_err_mean":        abs(_ik - _gt_ks_mean),
            "lr": opt.param_groups[0]["lr"] if opt is not None else 0.0,
        }, step=-1)
    t0 = time.perf_counter()
    img_batch = cfg.get("img_batch", N_imgs) or N_imgs
    for i in range(cfg["n_iter"]):
        _step[0] = i
        if log_gradients:
            pre_raw = {n: p.data.clone() for n, p in named_params.items()}
        if img_batch >= N_imgs or log_gradients:
            loss, l_d, l_s, l_w, l_tv = _opt_step(opt, _forward, cfg)
        else:
            totals = [0.0] * 5

            def _accum():
                opt.zero_grad()
                totals[:] = [0.0] * 5
                for _b in range(0, N_imgs, img_batch):
                    _vals = _forward(range(_b, min(_b + img_batch, N_imgs)))
                    _vals[0].backward()
                    for _j, _v in enumerate(_vals):
                        totals[_j] += float(_v.detach())
                return torch.tensor(totals[0])

            if cfg.get("optimizer") == "LBFGS":
                opt.step(_accum)
            else:
                _accum()
                opt.step()
            loss, l_d, l_s, l_w, l_tv = totals
        if sched is not None:
            sched.step()
        if log_gradients and grad_log_dir is not None:
            _save_grad_step(i, named_params, pre_raw, gt_map_grad, fwd_map_grad,
                            _forward_components,
                            (float(loss), float(l_d), float(l_s), float(l_w), float(l_tv)),
                            grad_log_dir, flat_mask, H, W)
        if i % cfg["log_every"] == 0:
            elapsed = time.perf_counter() - t0
            with torch.no_grad():
                shin_map = _fwd_shininess(shininess_raw, tr_shin, s_min, s_max).detach()
                ks_map   = _fwd_ks(ks_raw, tr_ks).detach()
                shin_val = float(shin_map[mask_hw].mean())
                ks_val   = float(ks_map[mask_hw].mean())
            history.append(float(loss))
            print(f"  [{i:4d}] {elapsed:6.1f}s  loss={float(loss):.3e}  data={float(l_d):.3e}"
                  f"  shininess={shin_val:.1f}  ks={ks_val:.3f}")
            if wandb_run is not None:
                with torch.no_grad():
                    env_pix_all = _fwd_env(env_raw_params, tr_env).detach()
                    env_imgs_k  = [_env_flat_to_img(env_pix_all[k].cpu().numpy(), env_H, env_W)
                                   for k in range(_n_log)]
                    env_avg_img = _env_flat_to_img(env_pix_all.mean(0).cpu().numpy(), env_H, env_W)
                    _ab_t   = _fwd_albedo(albedo_param, tr_ab).detach()
                    _ab_m   = _ab_t.reshape(-1, 3)[flat_mask]
                    _shin_m = shin_map.reshape(-1, 1)[flat_mask]
                    _ks_m   = ks_map.reshape(-1, 1)[flat_mask]
                    _recons, _errs = [], []
                    for _k in range(_n_log):
                        _r = _ab_t.new_zeros(H, W, 3)
                        _r.reshape(-1, 3)[flat_mask] = shade_phong_env(
                            view_m, N_m, _ab_m, env_pix_all[_k], env_dirs_t, env_dw_t,
                            ka, kd, _ks_m, _shin_m, sbatch=cfg.get("sbatch", 64),
                                     spec_importance=cfg.get("spec_importance", False),
                                     spec_samples=cfg.get("spec_samples", 64))
                        _r *= mask_t
                        _recons.append(wandb.Image(_r.float().cpu().numpy()))
                        _errs.append(wandb.Image((_r - imgs_t[_k]).abs().mul(mask_t).float().cpu().numpy()))
                wandb_run.log({
                    "loss": float(loss), "loss_data": float(l_d), "data_rmse": float(l_d) ** 0.5,
                    "loss_sparse": float(l_s), "loss_white": float(l_w), "loss_tv": float(l_tv),
                    "elapsed_s": elapsed,
                    "pred_albedo":    wandb.Image(
                        _fwd_albedo(albedo_param, tr_ab).detach().float().cpu().numpy()),
                    "pred_shininess": wandb.Image(
                        (shin_map / s_max).squeeze(-1).cpu().numpy()),
                    "pred_ks":        wandb.Image(ks_map.squeeze(-1).cpu().numpy()),
                    "est_env_maps":   [wandb.Image(img) for img in env_imgs_k],
                    "env_map_avg":    wandb.Image(env_avg_img),
                    "recons":         _recons,
                    "recon_err_maps": _errs,
                    "shininess_mean":     shin_val,
                    "ks_mean":            ks_val,
                    "shininess_err_mean": abs(shin_val - _gt_shin_mean),
                    "ks_err_mean":        abs(ks_val   - _gt_ks_mean),
                    "lr": opt.param_groups[0]["lr"] if opt is not None else 0.0,
                }, step=i)
    total_time = time.perf_counter() - t0

    with torch.no_grad():
        albedo_out   = _fwd_albedo(albedo_param, tr_ab).cpu().numpy()
        env_maps_out = _fwd_env(env_raw_params, tr_env).cpu().numpy()
        shin_out     = _fwd_shininess(shininess_raw, tr_shin, s_min, s_max).cpu().numpy()
        ks_out       = _fwd_ks(ks_raw, tr_ks).cpu().numpy()
        shin_m_t     = torch.from_numpy(shin_out).to(dev).reshape(-1, 1)[flat_mask]
        ks_m_t       = torch.from_numpy(ks_out).to(dev).reshape(-1, 1)[flat_mask]
        albedo_t2    = _fwd_albedo(albedo_param, tr_ab)
        shadings     = []
        for k in range(N_imgs):
            albedo_m  = albedo_t2.reshape(-1, 3)[flat_mask]
            env_pix_k = torch.from_numpy(env_maps_out[k]).to(dev)
            recon_m   = shade_phong_env(view_m, N_m, albedo_m,
                                        env_pix_k, env_dirs_t, env_dw_t,
                                        ka, kd, ks_m_t, shin_m_t)
            s = albedo_t2.new_zeros(H, W, 3)
            s.reshape(-1, 3)[flat_mask] = recon_m
            s *= mask_t
            shadings.append(s.cpu().numpy())

    return albedo_out, env_maps_out, shin_out, ks_out, shadings, history, total_time


# ─────────────────────────────────────── Phase 2: main loop ──────────────────

def run_decomposition(
    mesh_name:          str            = "sphere",
    width:              int            = 128,
    height:             int            = 128,
    shader:             str            = "ct_sh",
    mat_filter:         Optional[str]  = None,
    cfg_overrides:      Optional[dict] = None,
    device:             str            = "cuda",
    opt_params:         Optional[frozenset] = None,
    skip_existing:      bool           = False,
    mat_configs_filter: Optional[set]  = None,
    transforms:         Optional[dict] = None,
    light_mode:         str            = "directional",
    n_lights:           int            = 6,
    full_circle:        bool           = False,
    init_from_gt:       bool           = False,
    log_gradients:      bool           = False,
) -> None:
    cfg = {**DEFAULT_CFG, **(cfg_overrides or {})}
    dev = device
    tr  = transforms if transforms is not None else NAMED_TRANSFORMS["none"]
    transform_folder = _transforms_folder(tr)
    tr_dir = RESULTS_ROOT / transform_folder
    tr_dir.mkdir(parents=True, exist_ok=True)
    tr_json = tr_dir / "transforms.json"
    if not tr_json.exists():
        with open(tr_json, "w") as fh:
            json.dump(tr, fh, indent=2)

    is_phong = shader.startswith("phong")
    _base_configs = PHONG_MATERIAL_CONFIGS if is_phong else MATERIAL_CONFIGS
    mat_configs = {k: v for k, v in _base_configs.items()
                   if mat_configs_filter is None or k in mat_configs_filter}

    # Folder/run suffix for selective optimization and regularization
    if opt_params is not None:
        result_shader = shader + "_op=" + ",".join(sorted(opt_params))
    else:
        result_shader = shader
    ls = cfg.get("lambda_sparse", 0.0)
    lw = cfg.get("lambda_white",  0.0)
    lt = cfg.get("lambda_tv",     0.0)
    if ls:
        result_shader += f"_ls={ls}"
    if lw:
        result_shader += f"_lw={lw}"
    if lt:
        result_shader += f"_lt={lt}"
    if init_from_gt:
        result_shader += "_gtinit"
    if log_gradients:
        result_shader += "_gradlog"

    suffix = _scene_suffix(light_mode, n_lights, full_circle)

    mesh = _load_mesh(mesh_name)
    normals_hw, frag_pos_hw, mask_hw, cam_pos = rasterize_geometry(
        mesh, DEFAULT_CAMERA, width=width, height=height, smooth=True, device=dev,
    )

    _sh_ref  = SHLighting.directional(np.array([0, 0, 1], dtype=np.float32),
                                      LIGHT_COLOR, intensity=LIGHT_INTENSITY)
    _env_ref = EnvMap.from_sh(_sh_ref)
    env_dirs, env_dw = _env_ref._dirs, _env_ref._solid_angles
    env_H, env_W     = _env_ref.image.shape[:2]

    def _save_gray(arr_hw1: np.ndarray, path: Path) -> None:
        Image.fromarray((arr_hw1.squeeze(-1) * 255).clip(0, 255).astype(np.uint8)).save(path)

    for mat_id, mat_cfg in mat_configs.items():
        base_prefix  = f"{mesh_name}_phong_{mat_id}" if is_phong else f"{mesh_name}_{mat_id}"
        scene_prefix = f"{base_prefix}{suffix}"
        if mat_filter and scene_prefix != mat_filter and base_prefix != mat_filter:
            continue

        # Read light keys from dataset_meta.json; fall back to LIGHT_ANGLES_DEG for old data
        meta = _read_dataset_meta(scene_prefix)
        if meta is not None:
            light_keys = meta["light_keys"]
        else:
            light_keys = [f"light_{int(a):02d}deg" for a in LIGHT_ANGLES_DEG]

        out_dir_check = tr_dir / scene_prefix / result_shader
        if skip_existing and (out_dir_check / "metrics.json").exists():
            print(f"\n[Phase 2] {scene_prefix}  ({result_shader})  skipped (metrics.json exists)")
            continue

        print(f"\n[Phase 2] {scene_prefix}  ({shader})")

        images, gt_sh_list = [], []
        for light_id in light_keys:
            img_path = DATASET_ROOT / scene_prefix / shader / light_id / "render.png"
            cfg_path = DATASET_ROOT / scene_prefix / shader / light_id / "config.json"
            if not img_path.exists():
                raise FileNotFoundError(f"{img_path} — run --phase 1 --shader {shader} first.")
            npy_path = img_path.with_suffix(".npy")
            if npy_path.exists():
                images.append(np.load(npy_path).astype(np.float32))
            else:
                images.append(np.array(Image.open(img_path), dtype=np.float32) / 255.0)
            with open(cfg_path) as fh:
                gt_sh_list.append(np.array(
                    json.load(fh)["light"]["sh_coeffs"], dtype=np.float32))

        mask_np = mask_hw.cpu().numpy()
        _gt_alb_npy = DATASET_ROOT / scene_prefix / "gt" / "albedo.npy"
        if _gt_alb_npy.exists():
            gt_color = np.load(_gt_alb_npy)               # (H, W, 3), already masked
        else:
            gt_color = (np.tile(np.array(mat_cfg["albedo"], dtype=np.float32), (height, width, 1))
                        * mask_np[:, :, None])
        gt_albedo_img = gt_color

        run = wandb.init(
            entity  ="DLVC-intrinsics",
            project ="synthetic_ct_decomp",
            config  =dict(**cfg, mesh_name=mesh_name, mat_id=mat_id,
                          material=mat_cfg, shader=shader,
                          opt_params=sorted(opt_params) if opt_params is not None else "all",
                          transforms=tr, transform_folder=transform_folder,
                          width=width, height=height, n_images=len(images),
                          light_mode=light_mode, init_from_gt=init_from_gt),
            name    =f"{scene_prefix}_{result_shader}",
            reinit  =True,
        )

        # GT SH env maps — shared across both CT and Phong SH/env shaders
        gt_sh_env_imgs = [_sh_coeffs_to_env_img(gt_sh) for gt_sh in gt_sh_list]

        if is_phong:
            _gt_shin_npy = DATASET_ROOT / scene_prefix / "gt" / "shininess.npy"
            _gt_ks_npy   = DATASET_ROOT / scene_prefix / "gt" / "ks.npy"
            gt_shin_map  = (np.load(_gt_shin_npy) if _gt_shin_npy.exists()
                            else np.full((height, width, 1), mat_cfg["shininess"], dtype=np.float32)
                                 * mask_np[:, :, None])
            gt_ks_map    = (np.load(_gt_ks_npy) if _gt_ks_npy.exists()
                            else np.full((height, width, 1), mat_cfg["ks"], dtype=np.float32)
                                 * mask_np[:, :, None])
            gt_shin = float(gt_shin_map[mask_np].mean())
            gt_ks   = float(gt_ks_map[mask_np].mean())
            gt_shin_img = gt_shin_map[:, :, 0] / SHININESS_RANGE[1]
            gt_ks_img   = gt_ks_map[:, :, 0]
            run.log({"gt_images":       [wandb.Image(img) for img in images],
                     "gt_albedo":       wandb.Image(gt_albedo_img),
                     "gt_shininess":    wandb.Image(gt_shin_img),
                     "gt_ks":           wandb.Image(gt_ks_img),
                     "gt_sh_env_maps":  [wandb.Image(e) for e in gt_sh_env_imgs]}, step=0)
        else:
            _gt_met_npy   = DATASET_ROOT / scene_prefix / "gt" / "metallic.npy"
            _gt_rough_npy = DATASET_ROOT / scene_prefix / "gt" / "roughness.npy"
            gt_met_map    = (np.load(_gt_met_npy) if _gt_met_npy.exists()
                             else np.full((height, width, 1),
                                          mat_cfg["metallic"], dtype=np.float32) * mask_np[:, :, None])
            gt_rough_map  = (np.load(_gt_rough_npy) if _gt_rough_npy.exists()
                             else np.full((height, width, 1),
                                          mat_cfg["roughness"], dtype=np.float32) * mask_np[:, :, None])
            gt_metallic      = float(gt_met_map[mask_np].mean())
            gt_roughness     = float(gt_rough_map[mask_np].mean())
            gt_metallic_img  = gt_met_map[:, :, 0]
            gt_roughness_img = gt_rough_map[:, :, 0]
            run.log({"gt_images":       [wandb.Image(img) for img in images],
                     "gt_albedo":       wandb.Image(gt_albedo_img),
                     "gt_metallic":     wandb.Image(gt_metallic_img),
                     "gt_roughness":    wandb.Image(gt_roughness_img),
                     "gt_sh_env_maps":  [wandb.Image(e) for e in gt_sh_env_imgs]}, step=0)

        # gradient log dir (created lazily by _optimize_* when log_gradients=True)
        grad_log_dir = tr_dir / scene_prefix / result_shader / "gradient_flow"

        # ── dispatch ──────────────────────────────────────────────────────────
        sh_out:   np.ndarray = np.empty(0)
        env_maps: np.ndarray = np.empty(0)

        if shader == "ct_sh":
            albedo, sh_out, mat_a, mat_b, shadings, history, elapsed = _optimize_ct_sh(
                images, normals_hw, frag_pos_hw, mask_hw, cam_pos,
                gt_met_map, gt_rough_map, cfg,  # type: ignore[possibly-undefined]
                wandb_run=run, gt_sh_coeffs=gt_sh_list,
                gt_albedo=gt_color, opt_params=opt_params, transforms=tr,
                init_from_gt=init_from_gt, log_gradients=log_gradients,
                grad_log_dir=grad_log_dir,
            )
        elif shader == "ct_env":
            albedo, env_maps, mat_a, mat_b, shadings, history, elapsed = _optimize_ct_env(
                images, normals_hw, frag_pos_hw, mask_hw, cam_pos,
                gt_met_map, gt_rough_map, env_dirs, env_dw, cfg,  # type: ignore[possibly-undefined]
                wandb_run=run, env_H=env_H, env_W=env_W,
                gt_sh_coeffs=gt_sh_list, gt_albedo=gt_color, opt_params=opt_params, transforms=tr,
                init_from_gt=init_from_gt, log_gradients=log_gradients,
                grad_log_dir=grad_log_dir,
            )
        elif shader == "phong_sh":
            albedo, sh_out, mat_a, mat_b, shadings, history, elapsed = _optimize_phong_sh(
                images, normals_hw, frag_pos_hw, mask_hw, cam_pos,
                gt_shin_map, gt_ks_map,  # type: ignore[possibly-undefined]  # spatial GT (broadcast-compatible)
                mat_cfg["ka"], mat_cfg["kd"], cfg,
                wandb_run=run, gt_sh_coeffs=gt_sh_list,
                gt_albedo=gt_color, opt_params=opt_params, transforms=tr,
                init_from_gt=init_from_gt, log_gradients=log_gradients,
                grad_log_dir=grad_log_dir,
            )
        else:  # phong_env
            albedo, env_maps, mat_a, mat_b, shadings, history, elapsed = _optimize_phong_env(
                images, normals_hw, frag_pos_hw, mask_hw, cam_pos,
                gt_shin_map, gt_ks_map,  # type: ignore[possibly-undefined]  # spatial GT
                mat_cfg["ka"], mat_cfg["kd"],
                env_dirs, env_dw, cfg, wandb_run=run, env_H=env_H, env_W=env_W,
                gt_sh_coeffs=gt_sh_list, gt_albedo=gt_color, opt_params=opt_params, transforms=tr,
                init_from_gt=init_from_gt, log_gradients=log_gradients,
                grad_log_dir=grad_log_dir,
            )

        # mat_a = metallic or shininess (H,W,1), mat_b = roughness or ks (H,W,1)
        gt_a = gt_shin   if is_phong else gt_metallic   # type: ignore[possibly-undefined]
        gt_b = gt_ks     if is_phong else gt_roughness  # type: ignore[possibly-undefined]
        a_label, b_label = ("shininess", "ks") if is_phong else ("metallic", "roughness")

        # ── albedo RMSE ───────────────────────────────────────────────────────
        est_px = torch.from_numpy(albedo[mask_np])
        gt_px  = torch.from_numpy(gt_color[mask_np])          # (M, 3) — handles flat and spatial GT
        rmse_t, scale_t = _albedo_rmse(est_px, gt_px)
        rmse   = float(rmse_t)
        scale  = scale_t.numpy()                                  # (3,) per-channel

        # Rescale lighting by 1/scale to match the albedo correction
        inv_scale = 1.0 / np.maximum(scale[None, None, :], 1e-8)  # (1,1,3)
        if shader in ("ct_sh", "phong_sh") and sh_out.size:
            sh_out_rescaled = sh_out * inv_scale
        else:
            sh_out_rescaled = sh_out
        if shader in ("ct_env", "phong_env") and env_maps.size:
            env_maps_rescaled = env_maps * inv_scale
        else:
            env_maps_rescaled = env_maps

        if is_phong:
            mat_a_err = np.abs(mat_a - gt_shin_map)    # type: ignore[possibly-undefined]
            mat_b_err = np.abs(mat_b - gt_ks_map)      # type: ignore[possibly-undefined]
        else:
            mat_a_err = np.abs(mat_a - gt_met_map)     # type: ignore[possibly-undefined]
            mat_b_err = np.abs(mat_b - gt_rough_map)   # type: ignore[possibly-undefined]
        mat_a_mean = float(mat_a[mask_np].mean())
        mat_b_mean = float(mat_b[mask_np].mean())

        albedo_scaled = (albedo * scale).clip(0, 1)
        albedo_err    = np.abs(albedo_scaled - gt_color) * mask_np[:, :, None]

        recon_err  = [np.abs(s - img) * mask_np[:, :, None]
                      for s, img in zip(shadings, images)]
        recon_rmse = float(np.mean([e[mask_np].mean() for e in recon_err]))

        metrics = dict(
            albedo_rmse=rmse, final_loss=float(history[-1]),
            recon_rmse=recon_rmse,
            albedo_scale=scale.tolist(), loss_history=history,
            **{f"{a_label}_est_mean": mat_a_mean,
               f"{a_label}_gt":       gt_a,
               f"{a_label}_err_mean": float(mat_a_err[mask_np].mean()),
               f"{b_label}_est_mean": mat_b_mean,
               f"{b_label}_gt":       gt_b,
               f"{b_label}_err_mean": float(mat_b_err[mask_np].mean())},
        )

        # Build final SH/env map images for wandb summary (scale-corrected)
        if shader in ("ct_sh", "phong_sh"):
            final_light_imgs = [wandb.Image(_sh_coeffs_to_env_img(sh_out_rescaled[k]))
                                for k in range(len(images))]
            light_img_key = "est_sh_env_maps"
        else:
            final_light_imgs = [wandb.Image(_env_flat_to_img(env_maps_rescaled[k], env_H, env_W))
                                for k in range(len(images))]
            light_img_key = "est_env_maps"

        run.log({
            "albedo_est":      wandb.Image(albedo.clip(0, 1)),
            "albedo_scaled":   wandb.Image(albedo_scaled),
            "albedo_err":      wandb.Image(albedo_err.mean(-1)),
            f"{a_label}_est":  wandb.Image(
                mat_a.squeeze(-1) / (SHININESS_RANGE[1] if is_phong else 1.0)),
            f"{b_label}_est":  wandb.Image(mat_b.squeeze(-1)),
            f"{a_label}_err":  wandb.Image(mat_a_err.squeeze(-1) * mask_np),
            f"{b_label}_err":  wandb.Image(mat_b_err.squeeze(-1) * mask_np),
            "reconstructions": [wandb.Image(s.clip(0, 1)) for s in shadings],
            "recon_errors":    [wandb.Image(e.mean(-1)) for e in recon_err],
            light_img_key:     final_light_imgs,
            "albedo_rmse":     rmse,
            "recon_rmse":      recon_rmse,
            "final_loss":      history[-1],
            "elapsed_s":       elapsed,
        }, step=cfg["n_iter"])
        run.finish()

        # ── save to disk ──────────────────────────────────────────────────────
        out_dir = tr_dir / scene_prefix / result_shader
        out_dir.mkdir(parents=True, exist_ok=True)
        recon_dir = out_dir / "reconstructions"
        recon_dir.mkdir(exist_ok=True)

        Image.fromarray((albedo.clip(0, 1) * 255).astype(np.uint8)).save(
            out_dir / "albedo_est.png")
        np.save(out_dir / "albedo_est.npy", albedo.astype(np.float32))   # raw, unscaled

        # shininess is in [s_min, s_max] — normalize to [0, 1] for 8-bit images
        a_norm = SHININESS_RANGE[1] if (is_phong and a_label == "shininess") else 1.0
        _save_gray(mat_a / a_norm, out_dir / f"{a_label}_est.png")
        np.save(out_dir / f"{a_label}_est.npy", mat_a.astype(np.float32))   # actual units
        _save_gray(mat_b,          out_dir / f"{b_label}_est.png")
        np.save(out_dir / f"{b_label}_est.npy", mat_b.astype(np.float32))
        _save_gray(mat_a_err / a_norm * mask_np[:, :, None], out_dir / f"{a_label}_err.png")
        np.save(out_dir / f"{a_label}_err.npy",
                (mat_a_err * mask_np[:, :, None]).astype(np.float32))       # actual units
        _save_gray(mat_b_err * mask_np[:, :, None],          out_dir / f"{b_label}_err.png")
        np.save(out_dir / f"{b_label}_err.npy",
                (mat_b_err * mask_np[:, :, None]).astype(np.float32))

        for k, (s, e) in enumerate(zip(shadings, recon_err)):
            lk = light_keys[k]
            Image.fromarray((s.clip(0, 1) * 255).astype(np.uint8)).save(
                recon_dir / f"recon_{lk}.png")
            np.save(recon_dir / f"recon_{lk}.npy", s.astype(np.float32))
            Image.fromarray((e.mean(-1) * 255).clip(0, 255).astype(np.uint8)).save(
                recon_dir / f"recon_err_{lk}.png")
            np.save(recon_dir / f"recon_err_{lk}.npy", e.astype(np.float32))

        if shader in ("ct_sh", "phong_sh"):
            np.save(out_dir / "sh_coeffs_est.npy", sh_out_rescaled)
            for k, sh_k in enumerate(sh_out_rescaled):
                sh_env_img = _sh_coeffs_to_env_img(sh_k)
                Image.fromarray((sh_env_img * 255).astype(np.uint8)).save(
                    out_dir / f"sh_env_map_{light_keys[k]}.png")
                np.save(out_dir / f"sh_env_map_{light_keys[k]}.npy", sh_env_img.astype(np.float32))
        else:
            np.save(out_dir / "env_maps_est.npy", env_maps_rescaled)
            for k, env_k in enumerate(env_maps_rescaled):
                env_img = _env_flat_to_img(env_k, env_H, env_W)
                Image.fromarray((env_img * 255).astype(np.uint8)).save(
                    out_dir / f"env_map_{light_keys[k]}.png")
                np.save(out_dir / f"env_map_{light_keys[k]}.npy", env_img.astype(np.float32))
            env_avg_img = _env_flat_to_img(env_maps_rescaled.mean(0), env_H, env_W)
            Image.fromarray((env_avg_img * 255).astype(np.uint8)).save(
                out_dir / "env_map_avg.png")

        with open(out_dir / "material_est.json", "w") as fh:
            json.dump({
                f"{a_label}_est_mean": mat_a_mean,
                f"{a_label}_gt":       gt_a,
                f"{a_label}_err_mean": float(mat_a_err[mask_np].mean()),
                f"{b_label}_est_mean": mat_b_mean,
                f"{b_label}_gt":       gt_b,
                f"{b_label}_err_mean": float(mat_b_err[mask_np].mean()),
            }, fh, indent=2)
        with open(out_dir / "metrics.json", "w") as fh:
            json.dump(metrics, fh, indent=2)

        print(f"  {elapsed:.1f}s  albedo RMSE={rmse:.4f}"
              f"  {a_label}={mat_a_mean:.3f}(GT={gt_a:.3f})"
              f"  {b_label}={mat_b_mean:.3f}(GT={gt_b:.3f})  -> {out_dir}")

    print("[Phase 2] Complete.")


# ─────────────────────────────────────── CLI ─────────────────────────────────

_ALL_SHADERS = ["ct_sh", "ct_env", "phong_sh", "phong_env"]


def _build_parser():
    p = argparse.ArgumentParser(description="Synthetic CT + Phong dataset + decomposer")
    p.add_argument("--mesh",     default="sphere",
                   choices=["sphere", "suzanne", "bunny", "all"])
    p.add_argument("--width",    type=int, default=128)
    p.add_argument("--height",   type=int, default=128)
    p.add_argument("--phase",    type=int, default=1, choices=[1, 2])
    p.add_argument("--shader",   default="ct_sh",
                   choices=_ALL_SHADERS + ["all"],
                   help="Shader type (default: ct_sh)")
    p.add_argument("--optimizer", default=None, choices=["LBFGS", "Adam"])
    p.add_argument("--n-iter",        type=int,   default=None)
    p.add_argument("--lr",            type=float, default=None)
    p.add_argument("--lambda-sparse", type=float, default=None)
    p.add_argument("--lambda-white",  type=float, default=None)
    p.add_argument("--lambda-tv",     type=float, default=None)
    p.add_argument("--mat",         default=None,
                   help="Single full scene name, e.g. sphere_default")
    p.add_argument("--mat-configs", default=None,
                   help="Comma-separated config keys to include, e.g. 'albedo_0,metallic_1'. "
                        "Default: all configs.")
    p.add_argument("--device",    default=None,
                   help="torch device, e.g. cuda, cuda:1, cpu (default: cuda if available)")
    p.add_argument("--opt-params", default=None,
                   help="Comma-separated learnable params, e.g. 'albedo,sh'. Default: all. "
                        "Results written to <shader>_op=<params> subfolder.")
    p.add_argument("--skip-existing", action="store_true",
                   help="Skip dataset renders / optimization runs whose output already exists.")
    p.add_argument("--transforms", default="none",
                   help="Parameter domain transforms: 'none', 'all', or custom 'k=v,...' pairs. "
                        "Default: none (no transforms).")
    p.add_argument("--light-mode", default="directional",
                   choices=["directional", "random_sh", "circular"],
                   help="Lighting mode for dataset generation (default: directional).")
    p.add_argument("--n-lights",   type=int, default=6,
                   help="Number of light configurations per scene (default: 6).")
    p.add_argument("--full-circle", action="store_true",
                   help="Spread directional lights over full 360° instead of 0–90°.")
    p.add_argument("--init-from-gt", action="store_true",
                   help="Initialize optimizable parameters from GT values (phase 2).")
    p.add_argument("--log-gradients", action="store_true",
                   help="Log per-step gradient flow snapshots to gradient_flow/ (phase 2).")
    return p


def main():
    args = _build_parser().parse_args()
    overrides = {k: v for k, v in [
        ("optimizer",      args.optimizer),
        ("n_iter",         args.n_iter),
        ("lr",             args.lr),
        ("lambda_sparse",  args.lambda_sparse),
        ("lambda_white",   args.lambda_white),
        ("lambda_tv",      args.lambda_tv),
    ] if v is not None}

    device        = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    meshes        = ["sphere", "suzanne", "bunny"] if args.mesh == "all" else [args.mesh]
    shaders       = _ALL_SHADERS if args.shader == "all" else [args.shader]
    opt_params         = frozenset(args.opt_params.split(",")) if args.opt_params else None
    skip_existing      = args.skip_existing
    mat_configs_filter = set(args.mat_configs.split(",")) if args.mat_configs else None
    transforms         = _parse_transforms(args.transforms)

    if args.phase == 1:
        for mesh in meshes:
            generate_dataset(mesh_name=mesh, width=args.width, height=args.height,
                             shader=args.shader, device=device,
                             skip_existing=skip_existing,
                             mat_configs_filter=mat_configs_filter,
                             light_mode=args.light_mode,
                             n_lights=args.n_lights,
                             full_circle=args.full_circle)
    else:
        for mesh in meshes:
            for sh in shaders:
                run_decomposition(
                    mesh_name          =mesh,
                    width              =args.width,
                    height             =args.height,
                    shader             =sh,
                    mat_filter         =args.mat,
                    cfg_overrides      =overrides or None,
                    device             =device,
                    opt_params         =opt_params,
                    skip_existing      =skip_existing,
                    mat_configs_filter =mat_configs_filter,
                    transforms         =transforms,
                    light_mode         =args.light_mode,
                    n_lights           =args.n_lights,
                    full_circle        =args.full_circle,
                    init_from_gt       =args.init_from_gt,
                    log_gradients      =args.log_gradients,
                )


if __name__ == "__main__":
    main()
