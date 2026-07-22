"""Building the synthetic study scenes: meshes, light sets, and procedural textures."""
from __future__ import annotations

from functools import partial

import numpy as np
import torch

from idr.config import (LIGHT_ANGLES_DEG, LIGHT_COLOR, LIGHT_INTENSITY,
                        MATERIAL_CONFIGS, PHONG_MATERIAL_CONFIGS)
from idr.paths import ASSETS_DIR
from idr.render import (EnvMap, SHLighting, generate_mesh, load_obj, build_sh_basis,
                        SHLight, EnvMapLightGPU)

def _load_mesh(name: str):
    if name == "sphere":
        return generate_mesh("sphere")
    if name == "suzanne":
        return load_obj(str(ASSETS_DIR / "obj" / "suzanne.obj"))
    if name == "bunny":
        return load_obj(str(ASSETS_DIR / "obj" / "stanford-bunny.obj"))
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


def _scatter(flat: torch.Tensor, flat_mask: torch.Tensor, H: int, W: int) -> torch.Tensor:
    """Scatter foreground pixels (M, C) back to (H, W, C)."""
    C   = flat.shape[-1] if flat.dim() > 1 else 1
    buf = torch.zeros(H * W, C, device=flat.device, dtype=torch.float32)
    buf[flat_mask] = flat.reshape(-1, C).float()
    return buf.reshape(H, W, C)
