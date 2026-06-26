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
from raw_renderer_gpu.rasterizer import _norm, _get_ggx_sh_lut, _sh_irradiance
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
    init_spec_zero       = False,
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
    """(9,3) SH coefficients → (H,W,3) float32 image normalized to [0,1]."""
    img = EnvMap.from_sh(SHLighting(coeffs), resolution=resolution).image  # (H,W,3)
    mx = float(img.max())
    return img / max(mx, 1e-8)


def _env_flat_to_img(env_flat: np.ndarray, env_H: int, env_W: int) -> np.ndarray:
    """(P,3) flat env-map → (H,W,3) float32 image normalized to [0,1]."""
    img = env_flat.reshape(env_H, env_W, 3)
    mx = float(img.max())
    return img / max(mx, 1e-8)


def _softplus_inv(x: torch.Tensor) -> torch.Tensor:
    """Inverse of softplus: softplus(result) ≈ x for x > 0."""
    return torch.log(torch.expm1(x.clamp(min=1e-7)))


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
    return torch.exp(p) if t == "log" else p.clamp(0.05, 0.95)

def _fwd_metallic(p: torch.Tensor, t: str) -> torch.Tensor:
    return torch.sigmoid(p) if t == "sigmoid" else p.clamp(0.0, 1.0)

def _fwd_roughness(p: torch.Tensor, t: str) -> torch.Tensor:
    return torch.sigmoid(p) if t == "sigmoid" else p.clamp(0.0, 1.0)

def _fwd_shininess(p: torch.Tensor, t: str, s_min: float, s_max: float) -> torch.Tensor:
    if t == "sigmoid":
        return s_min + (s_max - s_min) * torch.sigmoid(p)
    elif t == "log":
        return torch.exp(p).clamp(s_min, s_max)
    else:
        return p.clamp(s_min, s_max)

def _fwd_ks(p: torch.Tensor, t: str) -> torch.Tensor:
    return torch.sigmoid(p) if t == "sigmoid" else p.clamp(0.0, 1.0)

def _fwd_env(p: torch.Tensor, t: str) -> torch.Tensor:
    import torch.nn.functional as F
    return F.softplus(p) if t == "softplus" else p.clamp(min=0.0)


def _init_albedo(base: torch.Tensor, t: str) -> torch.Tensor:
    """base: (H,W,3) clamped to (0.05,0.95). Returns raw param in transform space."""
    return torch.log(base) if t == "log" else base.clone()

def _init_scalar(val: float, H: int, W: int, t: str,
                 squeeze_fn=None, dev=None) -> torch.Tensor:
    """Initialize a (H,W,1) scalar param for a fixed value."""
    dtype = torch.float32
    if t == "sigmoid":
        raw = float(np.log(np.clip(val, 1e-6, 1-1e-6) / (1 - np.clip(val, 1e-6, 1-1e-6))))
    elif t == "sigmoid_r":  # kept for compatibility, identical to sigmoid
        raw = float(np.log(np.clip(val, 1e-6, 1-1e-6) / (1 - np.clip(val, 1e-6, 1-1e-6))))
    else:
        raw = float(val)
    return torch.full((H, W, 1), raw, dtype=dtype, device=dev)


def _init_map(arr: np.ndarray, t: str, dev) -> torch.Tensor:
    """Initialize a (H, W, 1) raw param from a spatial GT map."""
    x = torch.from_numpy(arr.astype(np.float32)).to(dev)
    if t == "sigmoid":
        return torch.logit(x.clamp(1e-6, 1 - 1e-6))
    elif t == "sigmoid_r":  # kept for compatibility, identical to sigmoid
        return torch.logit(x.clamp(1e-6, 1 - 1e-6))
    else:
        return x.clone()

def _init_env(gt_flat: np.ndarray, t: str, dev) -> torch.Tensor:
    gt_t = torch.from_numpy(gt_flat.astype(np.float32)).to(dev)
    return _softplus_inv(gt_t) if t == "softplus" else gt_t.clone()


def _rescale_albedo_lighting(
    albedo_param: torch.Tensor,
    lighting_params: list,
    tr_ab: str,
    flat_mask: torch.Tensor,
    gt_ab_m: torch.Tensor,
) -> None:
    """Rescale albedo and lighting in-place to align estimated albedo with GT.

    Computes per-channel scale = argmin_s ||est_albedo * s - gt_albedo|| and
    applies it: albedo_param *= scale, each lighting tensor /= scale.
    lighting_params: list of tensors with shape (..., 3) — sh_coeffs or env_maps.
    """
    ab_m = _fwd_albedo(albedo_param, tr_ab).reshape(-1, 3)[flat_mask]  # (M, 3)
    scale = (gt_ab_m * ab_m).sum(0) / (ab_m * ab_m).sum(0).clamp(1e-8)  # (3,)
    cur_out = _fwd_albedo(albedo_param, tr_ab)                           # (H, W, 3)
    new_out = (cur_out * scale[None, None, :]).clamp(min=1e-6)
    if tr_ab == "log":
        albedo_param.data.copy_(torch.log(new_out))
    else:
        albedo_param.data.copy_(new_out)
    for lp in lighting_params:
        lp.data /= scale  # (..., 3) / (3,) — broadcasts over all leading dims


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


def _make_optimizer(params, cfg):
    if cfg["optimizer"] == "LBFGS":
        return torch.optim.LBFGS(
            params, lr=cfg["lr"],
            max_iter=cfg["lbfgs_max_iter"],
            line_search_fn="strong_wolfe",
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
    if cfg.get("optimizer") == "LBFGS":
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


def _loss_fn(recon, target, mask_t, mode):
    diff = (recon - target).abs() if mode == "L1" else (recon - target) ** 2
    return diff[mask_t.expand_as(diff)].mean()


def _opt_step(opt, forward_fn, cfg):
    """Single optimizer step; returns (total_loss, loss_data, loss_sparse, loss_white, loss_tv)."""
    if cfg["optimizer"] == "LBFGS":
        def closure():
            opt.zero_grad()
            loss, *_ = forward_fn()
            loss.backward()
            return loss
        opt.step(closure)
        with torch.no_grad():
            return forward_fn()
    else:
        opt.zero_grad()
        result = forward_fn()
        result[0].backward()
        opt.step()
        return result


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
) -> tuple:
    dev    = normals_hw.device
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
        return torch.from_numpy(np.asarray(x, np.float32)).to(dev) \
            if not isinstance(x, torch.Tensor) else x.to(dev, torch.float32)

    imgs_t    = torch.stack([_t(img) for img in images])
    flat_mask = mask_hw.reshape(-1)
    N_m       = normals_hw.reshape(-1, 3)[flat_mask]
    fp_m      = frag_pos_hw.reshape(-1, 3)[flat_mask]
    view_m    = _norm(cam_pos.unsqueeze(0) - fp_m)
    lut       = _get_ggx_sh_lut(dev)
    mask_t    = mask_hw.unsqueeze(-1).to(dev)

    learnable    = []
    named_params = {}   # for gradient logging

    if "albedo" in op:
        if init_from_gt and gt_albedo is not None:
            base = torch.from_numpy(
                np.broadcast_to(np.asarray(gt_albedo, np.float32), (H, W, 3)).copy()
            ).to(dev).clamp(0.05, 0.95)
        else:
            base = imgs_t.mean(0).clamp(0.05, 0.95)
        albedo_param = _init_albedo(base, tr_ab).requires_grad_(True)
        learnable.append(albedo_param)
        named_params["albedo"] = albedo_param
    else:
        base = torch.from_numpy(
            np.broadcast_to(np.asarray(gt_albedo, np.float32), (H, W, 3)).copy()
        ).to(dev).clamp(0.05, 0.95)
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

    _gt_met_np = np.asarray(gt_metallic, np.float32)
    _gt_rou_np = np.asarray(gt_roughness, np.float32)
    _flat_mask_s = flat_mask.cpu().numpy()
    _gt_met_scalar = float(_gt_met_np.reshape(-1, 1)[_flat_mask_s].mean() if _gt_met_np.ndim > 0
                           else float(_gt_met_np))
    _gt_rou_scalar = float(_gt_rou_np.reshape(-1, 1)[_flat_mask_s].mean() if _gt_rou_np.ndim > 0
                           else float(_gt_rou_np))

    if "metallic" in op:
        if init_from_gt:
            metallic_raw = _init_map(_gt_met_np.reshape(H, W, 1), tr_met, dev).requires_grad_(True)
        else:
            m0 = (-10.0 if tr_met == "sigmoid" else 0.0) if cfg.get("init_spec_zero", False) \
                 else (0.5 if tr_met != "sigmoid" else 0.0)
            metallic_raw = torch.full((H, W, 1), m0, dtype=torch.float32, device=dev).requires_grad_(True)
        learnable.append(metallic_raw)
        named_params["metallic"] = metallic_raw
    else:
        if _gt_met_np.ndim > 0:
            metallic_raw = _init_map(_gt_met_np.reshape(H, W, 1), tr_met, dev)
        else:
            metallic_raw = _init_scalar(_gt_met_scalar, H, W, tr_met, dev=dev)

    if "roughness" in op:
        if init_from_gt:
            roughness_raw = _init_map(_gt_rou_np.reshape(H, W, 1), tr_rou, dev).requires_grad_(True)
        else:
            r0 = (10.0 if tr_rou == "sigmoid" else 1.0) if cfg.get("init_spec_zero", False) \
                 else (0.5 if tr_rou != "sigmoid" else 0.0)
            roughness_raw = torch.full((H, W, 1), r0, dtype=torch.float32, device=dev).requires_grad_(True)
        learnable.append(roughness_raw)
        named_params["roughness"] = roughness_raw
    else:
        if _gt_rou_np.ndim > 0:
            roughness_raw = _init_map(_gt_rou_np.reshape(H, W, 1), tr_rou, dev)
        else:
            roughness_raw = _init_scalar(_gt_rou_scalar, H, W, tr_rou, dev=dev)

    def _get_met(): return _fwd_metallic(metallic_raw, tr_met)
    def _get_rou(): return _fwd_roughness(roughness_raw, tr_rou)

    opt   = _make_optimizer(learnable, cfg) if learnable else None
    sched = _make_scheduler(opt, cfg, cfg["n_iter"]) if opt is not None else None
    _step = [0]
    _loss_ml = [torch.zeros((), device=dev)]  # metallic L1 loss, updated each _forward call
    _loss_mb = [torch.zeros((), device=dev)]  # metallic binarize loss, updated each _forward call

    def _forward():
        _spec_warmup = _step[0] < cfg.get("spec_warmup_steps", 0)
        albedo      = _fwd_albedo(albedo_param, tr_ab)
        albedo_m    = albedo.reshape(-1, 3)[flat_mask]
        metallic    = _get_met()
        roughness   = _get_rou()
        metallic_m  = metallic.reshape(-1, 1)[flat_mask]
        roughness_m = roughness.reshape(-1, 1)[flat_mask]
        if _spec_warmup:
            metallic_m  = albedo_m.new_zeros(albedo_m.shape[0], 1)
            roughness_m = albedo_m.new_ones(albedo_m.shape[0], 1)
        loss_data = albedo.new_zeros(())
        for k in range(N_imgs):
            recon_m = shade_ct_sh(view_m, N_m, albedo_m, sh_coeffs[k],
                                  metallic_m, roughness_m, lut=lut)
            recon = albedo.new_zeros(H, W, 3)
            recon.reshape(-1, 3)[flat_mask] = recon_m
            loss_data = loss_data + _loss_fn(recon, imgs_t[k], mask_t, cfg["loss"])
        loss_sparse = cfg["lambda_sparse"] * _tv(albedo_param.permute(2, 0, 1))
        loss_white  = cfg["lambda_white"]  * (_fwd_albedo(albedo_param, tr_ab).mean() - 0.5) ** 2
        loss_tv     = cfg["lambda_tv"] * (
            _tv(albedo_param.permute(2, 0, 1)) +
            _tv(metallic_raw.permute(2, 0, 1)) +
            _tv(roughness_raw.permute(2, 0, 1))
        )
        met_m = metallic.reshape(-1, 1)[flat_mask]
        loss_metallic_l1       = cfg.get("lambda_metallic_l1",       0.0) * met_m.abs().mean()
        loss_metallic_binarize = cfg.get("lambda_metallic_binarize",  0.0) * (met_m * (1.0 - met_m)).mean()
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
                                       metallic_m, roughness_m, lut=lut, return_components=True)
                result.append(comps)
        return result

    gt_map_grad = {
        "albedo":    np.broadcast_to(np.asarray(gt_albedo,   np.float32), (H, W, 3)).copy() if gt_albedo is not None else None,
        "sh":        np.stack(gt_sh_coeffs).astype(np.float32) if gt_sh_coeffs is not None else None,
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
                ).to(dev) if gt_albedo is not None else None)
    _gt_met_arr = np.asarray(gt_metallic, np.float32)
    _gt_met_m = torch.from_numpy(
        _gt_met_arr.reshape(-1, 1)[_flat_mask_np] if _gt_met_arr.ndim > 0
        else np.full((int(flat_mask.sum()), 1), float(_gt_met_arr), np.float32)
    ).to(dev)
    _gt_rou_arr = np.asarray(gt_roughness, np.float32)
    _gt_rou_m = torch.from_numpy(
        _gt_rou_arr.reshape(-1, 1)[_flat_mask_np] if _gt_rou_arr.ndim > 0
        else np.full((int(flat_mask.sum()), 1), float(_gt_rou_arr), np.float32)
    ).to(dev)

    def _gt_rmse_metrics(ab_m, met_m, rou_m):
        """Pixel-level RMSE against GT intrinsics (only when GT is available)."""
        out = {}
        if _gt_ab_m is not None:
            scale = (_gt_ab_m * ab_m).sum(0) / (ab_m * ab_m).sum(0).clamp(1e-8)
            out["albedo_rmse"] = float((ab_m * scale - _gt_ab_m).pow(2).mean().sqrt())
        out["metallic_rmse"]  = float((met_m - _gt_met_m).pow(2).mean().sqrt())
        out["roughness_rmse"] = float((rou_m - _gt_rou_m).pow(2).mean().sqrt())
        return out

    with torch.no_grad():
        _il, _ild, _ils, _ilw, _iltv = _forward()
        _im = float(_get_met()[mask_hw].mean())
        _ir = float(_get_rou()[mask_hw].mean())
    print(f"  [init]          loss={float(_il):.3e}  data={float(_ild):.3e}"
          f"  metallic={_im:.3f}  roughness={_ir:.3f}")
    if wandb_run is not None:
        with torch.no_grad():
            _met_map = _get_met().detach()
            _rou_map = _get_rou().detach()
            _est_sh_np = sh_coeffs.detach().cpu().numpy()
            _ab_t  = _fwd_albedo(albedo_param, tr_ab).clamp(0, 1)
            _ab_m  = _ab_t.reshape(-1, 3)[flat_mask]
            _met_m = _met_map.reshape(-1, 1)[flat_mask]
            _rou_m = _rou_map.reshape(-1, 1)[flat_mask]
            _recons, _errs = [], []
            for _k in range(N_imgs):
                _r = _ab_t.new_zeros(H, W, 3)
                _r.reshape(-1, 3)[flat_mask] = shade_ct_sh(
                    view_m, N_m, _ab_m, sh_coeffs[_k], _met_m, _rou_m, lut=lut)
                _r *= mask_t
                _recons.append(wandb.Image(_r.cpu().numpy()))
                _errs.append(wandb.Image((_r - imgs_t[_k]).abs().mul(mask_t).cpu().numpy()))
        wandb_run.log({
            "loss": float(_il), "loss_data": float(_ild),
            "loss_sparse": float(_ils), "loss_white": float(_ilw), "loss_tv": float(_iltv),
            "pred_albedo":     wandb.Image(_ab_t.cpu().numpy()),
            "pred_metallic":   wandb.Image(_met_map.squeeze(-1).cpu().numpy()),
            "pred_roughness":  wandb.Image(_rou_map.squeeze(-1).cpu().numpy()),
            "est_sh_env_maps": [wandb.Image(_sh_coeffs_to_env_img(_est_sh_np[k])) for k in range(N_imgs)],
            "recons":          _recons,
            "recon_err_maps":  _errs,
            "metallic_mean":       _im,
            "roughness_mean":      _ir,
            "metallic_err_mean":   abs(_im - _gt_met_scalar),
            "roughness_err_mean":  abs(_ir - _gt_rou_scalar),
            "loss_metallic_l1":       float(_loss_ml[0]),
            "loss_metallic_binarize": float(_loss_mb[0]),
            **_gt_rmse_metrics(_ab_m, _met_m, _rou_m),
            "lr": opt.param_groups[0]["lr"] if opt is not None else 0.0,
        }, step=-1)
    _rescale_every = cfg.get("rescale_every", 0)
    t0 = time.perf_counter()
    for i in range(cfg["n_iter"]):
        _step[0] = i
        if opt is not None:
            if log_gradients:
                pre_raw = {n: p.data.clone() for n, p in named_params.items()}
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
                est_sh_np = sh_coeffs.detach().cpu().numpy()
                with torch.no_grad():
                    _ab_t  = _fwd_albedo(albedo_param, tr_ab).clamp(0, 1)
                    _ab_m  = _ab_t.reshape(-1, 3)[flat_mask]
                    _met_m = met_map.reshape(-1, 1)[flat_mask]
                    _rou_m = rou_map.reshape(-1, 1)[flat_mask]
                    _recons, _errs = [], []
                    for _k in range(N_imgs):
                        _r = _ab_t.new_zeros(H, W, 3)
                        _r.reshape(-1, 3)[flat_mask] = shade_ct_sh(
                            view_m, N_m, _ab_m, sh_coeffs[_k], _met_m, _rou_m, lut=lut)
                        _r *= mask_t
                        _recons.append(wandb.Image(_r.cpu().numpy()))
                        _errs.append(wandb.Image((_r - imgs_t[_k]).abs().mul(mask_t).cpu().numpy()))
                wandb_run.log({
                    "loss": float(loss), "loss_data": float(l_d),
                    "loss_sparse": float(l_s), "loss_white": float(l_w), "loss_tv": float(l_tv),
                    "elapsed_s": elapsed,
                    "pred_albedo":     wandb.Image(
                        _fwd_albedo(albedo_param, tr_ab).clamp(0, 1).detach().cpu().numpy()),
                    "pred_metallic":   wandb.Image(met_map.squeeze(-1).cpu().numpy()),
                    "pred_roughness":  wandb.Image(rou_map.squeeze(-1).cpu().numpy()),
                    "est_sh_env_maps": [wandb.Image(_sh_coeffs_to_env_img(est_sh_np[k]))
                                        for k in range(N_imgs)],
                    "recons":          _recons,
                    "recon_err_maps":  _errs,
                    "metallic_mean":      met,
                    "roughness_mean":     rou,
                    "metallic_err_mean":  abs(met - _gt_met_scalar),
                    "roughness_err_mean": abs(rou - _gt_rou_scalar),
                    "loss_metallic_l1":       float(_loss_ml[0]),
                    "loss_metallic_binarize": float(_loss_mb[0]),
                    **_gt_rmse_metrics(_ab_m, _met_m, _rou_m),
                    "lr": opt.param_groups[0]["lr"] if opt is not None else 0.0,
                }, step=i)
    total_time = time.perf_counter() - t0

    with torch.no_grad():
        albedo_out = _fwd_albedo(albedo_param, tr_ab).clamp(0, 1).cpu().numpy()
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
                                   met_m, rou_m, lut=lut)
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
) -> tuple:
    dev    = normals_hw.device
    H, W   = normals_hw.shape[:2]
    N_imgs = len(images)
    P      = env_dirs.shape[0]
    op = opt_params if opt_params is not None else _CT_ENV_PARAMS
    if transforms is not None:
        tr_ab, tr_met, tr_rou, tr_env = transforms["albedo"], transforms["metallic"], transforms["roughness"], transforms["env"]
    else:
        tr_ab  = cfg.get("tr_albedo",   "none")
        tr_met = cfg.get("tr_metallic",  "none")
        tr_rou = cfg.get("tr_roughness", "none")
        tr_env = cfg.get("tr_env",       "none")

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
            ).to(dev).clamp(0.05, 0.95)
        else:
            base = imgs_t.mean(0).clamp(0.05, 0.95)
        albedo_param = _init_albedo(base, tr_ab).requires_grad_(True)
        learnable.append(albedo_param)
        named_params["albedo"] = albedo_param
    else:
        base = torch.from_numpy(
            np.broadcast_to(np.asarray(gt_albedo, np.float32), (H, W, 3)).copy()
        ).to(dev).clamp(0.05, 0.95)
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

    _gt_met_np = np.asarray(gt_metallic, np.float32)
    _gt_rou_np = np.asarray(gt_roughness, np.float32)
    _flat_mask_s = flat_mask.cpu().numpy()
    _gt_met_scalar = float(_gt_met_np.reshape(-1, 1)[_flat_mask_s].mean() if _gt_met_np.ndim > 0
                           else float(_gt_met_np))
    _gt_rou_scalar = float(_gt_rou_np.reshape(-1, 1)[_flat_mask_s].mean() if _gt_rou_np.ndim > 0
                           else float(_gt_rou_np))

    if "metallic" in op:
        if init_from_gt:
            metallic_raw = _init_map(_gt_met_np.reshape(H, W, 1), tr_met, dev).requires_grad_(True)
        else:
            m0 = (-10.0 if tr_met == "sigmoid" else 0.0) if cfg.get("init_spec_zero", False) \
                 else (0.5 if tr_met != "sigmoid" else 0.0)
            metallic_raw = torch.full((H, W, 1), m0, dtype=torch.float32, device=dev).requires_grad_(True)
        learnable.append(metallic_raw)
        named_params["metallic"] = metallic_raw
    else:
        if _gt_met_np.ndim > 0:
            metallic_raw = _init_map(_gt_met_np.reshape(H, W, 1), tr_met, dev)
        else:
            metallic_raw = _init_scalar(_gt_met_scalar, H, W, tr_met, dev=dev)

    if "roughness" in op:
        if init_from_gt:
            roughness_raw = _init_map(_gt_rou_np.reshape(H, W, 1), tr_rou, dev).requires_grad_(True)
        else:
            r0 = (10.0 if tr_rou == "sigmoid" else 1.0) if cfg.get("init_spec_zero", False) \
                 else (0.5 if tr_rou != "sigmoid" else 0.0)
            roughness_raw = torch.full((H, W, 1), r0, dtype=torch.float32, device=dev).requires_grad_(True)
        learnable.append(roughness_raw)
        named_params["roughness"] = roughness_raw
    else:
        if _gt_rou_np.ndim > 0:
            roughness_raw = _init_map(_gt_rou_np.reshape(H, W, 1), tr_rou, dev)
        else:
            roughness_raw = _init_scalar(_gt_rou_scalar, H, W, tr_rou, dev=dev)

    def _get_met(): return _fwd_metallic(metallic_raw, tr_met)
    def _get_rou(): return _fwd_roughness(roughness_raw, tr_rou)

    opt   = _make_optimizer(learnable, cfg) if learnable else None
    sched = _make_scheduler(opt, cfg, cfg["n_iter"]) if opt is not None else None
    _step = [0]
    _loss_ml = [torch.zeros((), device=dev)]  # metallic L1 loss, updated each _forward call
    _loss_mb = [torch.zeros((), device=dev)]  # metallic binarize loss, updated each _forward call

    def _forward(img_indices=None):
        if img_indices is None:
            img_indices = range(N_imgs)
        img_indices = list(img_indices)
        frac = len(img_indices) / N_imgs
        _spec_warmup = _step[0] < cfg.get("spec_warmup_steps", 0)
        albedo      = _fwd_albedo(albedo_param, tr_ab)
        albedo_m    = albedo.reshape(-1, 3)[flat_mask]
        metallic    = metallic_raw  if (_frozen_gt and not metallic_raw.requires_grad)  else _fwd_metallic(metallic_raw,  tr_met)
        roughness   = roughness_raw if (_frozen_gt and not roughness_raw.requires_grad) else _fwd_roughness(roughness_raw, tr_rou)
        metallic_m  = metallic.reshape(-1, 1)[flat_mask]
        roughness_m = roughness.reshape(-1, 1)[flat_mask]
        if _spec_warmup:
            metallic_m  = albedo_m.new_zeros(albedo_m.shape[0], 1)
            roughness_m = albedo_m.new_ones(albedo_m.shape[0], 1)
        loss_data = albedo.new_zeros(())
        for k in img_indices:
            env_pix_k = _fwd_env(env_raw_params[k], tr_env)
            recon_m   = shade_ct_env(view_m, N_m, albedo_m,
                                     env_pix_k, env_dirs_t, env_dw_t,
                                     metallic_m, roughness_m,
                                     sbatch=cfg.get("sbatch", 64))
            recon = albedo.new_zeros(H, W, 3)
            recon.reshape(-1, 3)[flat_mask] = recon_m
            loss_data = loss_data + _loss_fn(recon, imgs_t[k], mask_t, cfg["loss"])
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
                                        sbatch=cfg.get("sbatch", 64), return_components=True)
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
                ).to(dev) if gt_albedo is not None else None)
    _gt_met_arr = np.asarray(gt_metallic, np.float32)
    _gt_met_m = torch.from_numpy(
        _gt_met_arr.reshape(-1, 1)[_flat_mask_np] if _gt_met_arr.ndim > 0
        else np.full((int(flat_mask.sum()), 1), float(_gt_met_arr), np.float32)
    ).to(dev)
    _gt_rou_arr = np.asarray(gt_roughness, np.float32)
    _gt_rou_m = torch.from_numpy(
        _gt_rou_arr.reshape(-1, 1)[_flat_mask_np] if _gt_rou_arr.ndim > 0
        else np.full((int(flat_mask.sum()), 1), float(_gt_rou_arr), np.float32)
    ).to(dev)

    def _gt_rmse_metrics(ab_m, met_m, rou_m):
        out = {}
        if _gt_ab_m is not None:
            scale = (_gt_ab_m * ab_m).sum(0) / (ab_m * ab_m).sum(0).clamp(1e-8)
            out["albedo_rmse"] = float((ab_m * scale - _gt_ab_m).pow(2).mean().sqrt())
        out["metallic_rmse"]  = float((met_m - _gt_met_m).pow(2).mean().sqrt())
        out["roughness_rmse"] = float((rou_m - _gt_rou_m).pow(2).mean().sqrt())
        return out

    history = []
    with torch.no_grad():
        _il, _ild, _ils, _ilw, _iltv = _forward()
        _im = float(_get_met()[mask_hw].mean())
        _ir = float(_get_rou()[mask_hw].mean())
    print(f"  [init]          loss={float(_il):.3e}  data={float(_ild):.3e}"
          f"  metallic={_im:.3f}  roughness={_ir:.3f}")
    if wandb_run is not None:
        with torch.no_grad():
            _env_pix_all = _fwd_env(env_raw_params, tr_env).detach()
            _env_imgs_k  = [_env_flat_to_img(_env_pix_all[k].cpu().numpy(), env_H, env_W) for k in range(N_imgs)]
            _env_avg_img = _env_flat_to_img(_env_pix_all.mean(0).cpu().numpy(), env_H, env_W)
            _met_map = _get_met().detach()
            _rou_map = _get_rou().detach()
            _ab_t  = _fwd_albedo(albedo_param, tr_ab).clamp(0, 1)
            _ab_m  = _ab_t.reshape(-1, 3)[flat_mask]
            _met_m = _met_map.reshape(-1, 1)[flat_mask]
            _rou_m = _rou_map.reshape(-1, 1)[flat_mask]
            _recons, _errs = [], []
            for _k in range(N_imgs):
                _r = _ab_t.new_zeros(H, W, 3)
                _r.reshape(-1, 3)[flat_mask] = shade_ct_env(
                    view_m, N_m, _ab_m, _env_pix_all[_k], env_dirs_t, env_dw_t,
                    _met_m, _rou_m, sbatch=cfg.get("sbatch", 64))
                _r *= mask_t
                _recons.append(wandb.Image(_r.cpu().numpy()))
                _errs.append(wandb.Image((_r - imgs_t[_k]).abs().mul(mask_t).cpu().numpy()))
        wandb_run.log({
            "loss": float(_il), "loss_data": float(_ild),
            "loss_sparse": float(_ils), "loss_white": float(_ilw), "loss_tv": float(_iltv),
            "pred_albedo":    wandb.Image(_ab_t.cpu().numpy()),
            "pred_metallic":  wandb.Image(_met_map.squeeze(-1).cpu().numpy()),
            "pred_roughness": wandb.Image(_rou_map.squeeze(-1).cpu().numpy()),
            "est_env_maps":   [wandb.Image(img) for img in _env_imgs_k],
            "env_map_avg":    wandb.Image(_env_avg_img),
            "recons":         _recons,
            "recon_err_maps": _errs,
            "metallic_mean":      _im,
            "roughness_mean":     _ir,
            "metallic_err_mean":  abs(_im - _gt_met_scalar),
            "roughness_err_mean": abs(_ir - _gt_rou_scalar),
            "loss_metallic_l1":   float(_loss_ml[0]),
            **_gt_rmse_metrics(_ab_m, _met_m, _rou_m),
            "lr": opt.param_groups[0]["lr"] if opt is not None else 0.0,
        }, step=-1)
    _rescale_every = cfg.get("rescale_every", 0)
    t0 = time.perf_counter()
    img_batch = cfg.get("img_batch", N_imgs) or N_imgs
    for i in range(cfg["n_iter"]):
        _step[0] = i
        if opt is not None:
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
            if (_rescale_every > 0 and (i + 1) % _rescale_every == 0
                    and "albedo" in op and "env" in op and _gt_ab_m is not None):
                with torch.no_grad():
                    _rescale_albedo_lighting(
                        albedo_param, [env_maps], tr_ab, flat_mask, _gt_ab_m)
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
                    env_pix_all = _fwd_env(env_raw_params, tr_env).detach()
                    env_imgs_k  = [_env_flat_to_img(env_pix_all[k].cpu().numpy(), env_H, env_W)
                                   for k in range(N_imgs)]
                    env_avg_img = _env_flat_to_img(env_pix_all.mean(0).cpu().numpy(), env_H, env_W)
                    _ab_t  = _fwd_albedo(albedo_param, tr_ab).clamp(0, 1)
                    _ab_m  = _ab_t.reshape(-1, 3)[flat_mask]
                    _met_m = met_map.reshape(-1, 1)[flat_mask]
                    _rou_m = rou_map.reshape(-1, 1)[flat_mask]
                    _recons, _errs = [], []
                    for _k in range(N_imgs):
                        _r = _ab_t.new_zeros(H, W, 3)
                        _r.reshape(-1, 3)[flat_mask] = shade_ct_env(
                            view_m, N_m, _ab_m, env_pix_all[_k], env_dirs_t, env_dw_t,
                            _met_m, _rou_m, sbatch=cfg.get("sbatch", 64))
                        _r *= mask_t
                        _recons.append(wandb.Image(_r.cpu().numpy()))
                        _errs.append(wandb.Image((_r - imgs_t[_k]).abs().mul(mask_t).cpu().numpy()))
                wandb_run.log({
                    "loss": float(loss), "loss_data": float(l_d),
                    "loss_sparse": float(l_s), "loss_white": float(l_w), "loss_tv": float(l_tv),
                    "elapsed_s": elapsed,
                    "pred_albedo":    wandb.Image(
                        _fwd_albedo(albedo_param, tr_ab).clamp(0, 1).detach().cpu().numpy()),
                    "pred_metallic":  wandb.Image(met_map.squeeze(-1).cpu().numpy()),
                    "pred_roughness": wandb.Image(rou_map.squeeze(-1).cpu().numpy()),
                    "est_env_maps":   [wandb.Image(img) for img in env_imgs_k],
                    "env_map_avg":    wandb.Image(env_avg_img),
                    "recons":         _recons,
                    "recon_err_maps": _errs,
                    "metallic_mean":      met,
                    "roughness_mean":     rou,
                    "metallic_err_mean":  abs(met - _gt_met_scalar),
                    "roughness_err_mean": abs(rou - _gt_rou_scalar),
                    "loss_metallic_l1":       float(_loss_ml[0]),
                    "loss_metallic_binarize": float(_loss_mb[0]),
                    **_gt_rmse_metrics(_ab_m, _met_m, _rou_m),
                    "lr": opt.param_groups[0]["lr"] if opt is not None else 0.0,
                }, step=i)

    total_time = time.perf_counter() - t0

    with torch.no_grad():
        albedo_out   = _fwd_albedo(albedo_param, tr_ab).clamp(0, 1).cpu().numpy()
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
                                     met_m, rou_m)
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
            ).to(dev).clamp(0.05, 0.95)
        else:
            base = imgs_t.mean(0).clamp(0.05, 0.95)
        albedo_param = _init_albedo(base, tr_ab).requires_grad_(True)
        learnable.append(albedo_param)
        named_params["albedo"] = albedo_param
    else:
        base = torch.from_numpy(
            np.broadcast_to(np.asarray(gt_albedo, np.float32), (H, W, 3)).copy()
        ).to(dev).clamp(0.05, 0.95)
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
        "sh":        np.stack(gt_sh_coeffs).astype(np.float32) if gt_sh_coeffs is not None else None,
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
            _ab_t   = _fwd_albedo(albedo_param, tr_ab).clamp(0, 1)
            _ab_m   = _ab_t.reshape(-1, 3)[flat_mask]
            _shin_m = _shin_map.reshape(-1, 1)[flat_mask]
            _ks_m   = _ks_map.reshape(-1, 1)[flat_mask]
            _recons, _errs = [], []
            for _k in range(N_imgs):
                _r = _ab_t.new_zeros(H, W, 3)
                _r.reshape(-1, 3)[flat_mask] = shade_phong_sh(
                    view_m, N_m, ka, kd, _ks_m, _shin_m, _ab_m, sh_coeffs[_k])
                _r *= mask_t
                _recons.append(wandb.Image(_r.cpu().numpy()))
                _errs.append(wandb.Image((_r - imgs_t[_k]).abs().mul(mask_t).cpu().numpy()))
        wandb_run.log({
            "loss": float(_il), "loss_data": float(_ild),
            "loss_sparse": float(_ils), "loss_white": float(_ilw), "loss_tv": float(_iltv),
            "pred_albedo":     wandb.Image(_ab_t.cpu().numpy()),
            "pred_shininess":  wandb.Image((_shin_map / s_max).squeeze(-1).cpu().numpy()),
            "pred_ks":         wandb.Image(_ks_map.squeeze(-1).cpu().numpy()),
            "est_sh_env_maps": [wandb.Image(_sh_coeffs_to_env_img(_est_sh_np[k])) for k in range(N_imgs)],
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
                    _ab_t   = _fwd_albedo(albedo_param, tr_ab).clamp(0, 1)
                    _ab_m   = _ab_t.reshape(-1, 3)[flat_mask]
                    _shin_m = shin_map.reshape(-1, 1)[flat_mask]
                    _ks_m   = ks_map.reshape(-1, 1)[flat_mask]
                    _recons, _errs = [], []
                    for _k in range(N_imgs):
                        _r = _ab_t.new_zeros(H, W, 3)
                        _r.reshape(-1, 3)[flat_mask] = shade_phong_sh(
                            view_m, N_m, ka, kd, _ks_m, _shin_m, _ab_m, sh_coeffs[_k])
                        _r *= mask_t
                        _recons.append(wandb.Image(_r.cpu().numpy()))
                        _errs.append(wandb.Image((_r - imgs_t[_k]).abs().mul(mask_t).cpu().numpy()))
                wandb_run.log({
                    "loss": float(loss), "loss_data": float(l_d),
                    "loss_sparse": float(l_s), "loss_white": float(l_w), "loss_tv": float(l_tv),
                    "elapsed_s": elapsed,
                    "pred_albedo":     wandb.Image(
                        _fwd_albedo(albedo_param, tr_ab).clamp(0, 1).detach().cpu().numpy()),
                    "pred_shininess":  wandb.Image(
                        (shin_map / s_max).squeeze(-1).cpu().numpy()),
                    "pred_ks":         wandb.Image(ks_map.squeeze(-1).cpu().numpy()),
                    "est_sh_env_maps": [wandb.Image(_sh_coeffs_to_env_img(est_sh_np[k]))
                                        for k in range(N_imgs)],
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
        albedo_out = _fwd_albedo(albedo_param, tr_ab).clamp(0, 1).cpu().numpy()
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
            ).to(dev).clamp(0.05, 0.95)
        else:
            base = imgs_t.mean(0).clamp(0.05, 0.95)
        albedo_param = _init_albedo(base, tr_ab).requires_grad_(True)
        learnable.append(albedo_param)
        named_params["albedo"] = albedo_param
    else:
        base = torch.from_numpy(
            np.broadcast_to(np.asarray(gt_albedo, np.float32), (H, W, 3)).copy()
        ).to(dev).clamp(0.05, 0.95)
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
                                        sbatch=cfg.get("sbatch", 64))
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
                                           sbatch=cfg.get("sbatch", 64), return_components=True)
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
            _env_imgs_k  = [_env_flat_to_img(_env_pix_all[k].cpu().numpy(), env_H, env_W) for k in range(N_imgs)]
            _env_avg_img = _env_flat_to_img(_env_pix_all.mean(0).cpu().numpy(), env_H, env_W)
            _shin_map = _fwd_shininess(shininess_raw, tr_shin, s_min, s_max).detach()
            _ks_map   = _fwd_ks(ks_raw, tr_ks).detach()
            _ab_t   = _fwd_albedo(albedo_param, tr_ab).clamp(0, 1)
            _ab_m   = _ab_t.reshape(-1, 3)[flat_mask]
            _shin_m = _shin_map.reshape(-1, 1)[flat_mask]
            _ks_m   = _ks_map.reshape(-1, 1)[flat_mask]
            _recons, _errs = [], []
            for _k in range(N_imgs):
                _r = _ab_t.new_zeros(H, W, 3)
                _r.reshape(-1, 3)[flat_mask] = shade_phong_env(
                    view_m, N_m, _ab_m, _env_pix_all[_k], env_dirs_t, env_dw_t,
                    ka, kd, _ks_m, _shin_m, sbatch=cfg.get("sbatch", 64))
                _r *= mask_t
                _recons.append(wandb.Image(_r.cpu().numpy()))
                _errs.append(wandb.Image((_r - imgs_t[_k]).abs().mul(mask_t).cpu().numpy()))
        wandb_run.log({
            "loss": float(_il), "loss_data": float(_ild),
            "loss_sparse": float(_ils), "loss_white": float(_ilw), "loss_tv": float(_iltv),
            "pred_albedo":    wandb.Image(_ab_t.cpu().numpy()),
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
                                   for k in range(N_imgs)]
                    env_avg_img = _env_flat_to_img(env_pix_all.mean(0).cpu().numpy(), env_H, env_W)
                    _ab_t   = _fwd_albedo(albedo_param, tr_ab).clamp(0, 1)
                    _ab_m   = _ab_t.reshape(-1, 3)[flat_mask]
                    _shin_m = shin_map.reshape(-1, 1)[flat_mask]
                    _ks_m   = ks_map.reshape(-1, 1)[flat_mask]
                    _recons, _errs = [], []
                    for _k in range(N_imgs):
                        _r = _ab_t.new_zeros(H, W, 3)
                        _r.reshape(-1, 3)[flat_mask] = shade_phong_env(
                            view_m, N_m, _ab_m, env_pix_all[_k], env_dirs_t, env_dw_t,
                            ka, kd, _ks_m, _shin_m, sbatch=cfg.get("sbatch", 64))
                        _r *= mask_t
                        _recons.append(wandb.Image(_r.cpu().numpy()))
                        _errs.append(wandb.Image((_r - imgs_t[_k]).abs().mul(mask_t).cpu().numpy()))
                wandb_run.log({
                    "loss": float(loss), "loss_data": float(l_d),
                    "loss_sparse": float(l_s), "loss_white": float(l_w), "loss_tv": float(l_tv),
                    "elapsed_s": elapsed,
                    "pred_albedo":    wandb.Image(
                        _fwd_albedo(albedo_param, tr_ab).clamp(0, 1).detach().cpu().numpy()),
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
        albedo_out   = _fwd_albedo(albedo_param, tr_ab).clamp(0, 1).cpu().numpy()
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
