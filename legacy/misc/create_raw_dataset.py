"""
Ground-truth dataset generation for intrinsic decomposition experiments.

create_scenes(n_scenes)  — per-object rasterisation + depth compositing → GT maps
create_tiny_raw(seed)    — 3×6 px normal maps + albedo

Run with:
    python create_raw_dataset.py
"""

import json
import os

import numpy as np
import torch
from PIL import Image

from raw_renderer import Camera, generate_mesh
from raw_renderer.mesh import Mesh
from raw_renderer_gpu import rasterize_geometry


# ── constants ─────────────────────────────────────────────────────────────────

_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "raw_dataset")
_W, _H = 200, 200
_CAM = Camera(
    position=np.array([0.0, 0.0, 3.0], dtype=np.float32),
    target=np.array([0.0, 0.0, 0.0], dtype=np.float32),
)

_OBJECT_COLORS = [
    [255,   0,   0],
    [  0, 255,   0],
    [  0,   0, 255],
    [255, 255,   0],
    [  0, 255, 255],
    [255,   0, 255],
]


# ── mesh helper ───────────────────────────────────────────────────────────────

def _sphere_mesh(radius: float, position: np.ndarray,
                 n_lat: int = 24, n_lon: int = 48) -> Mesh:
    base = generate_mesh("sphere", n_lat=n_lat, n_lon=n_lon)
    return Mesh(
        vertices=base.vertices * radius + position,
        faces=base.faces,
        normals=base.normals,
        vertex_normals=base.vertex_normals,
    )


# ── random parameter generators ───────────────────────────────────────────────

def _rand_surface_normal(rng) -> np.ndarray:
    d = rng.standard_normal(3).astype(np.float32)
    d[2] = abs(d[2])
    return d / (np.linalg.norm(d) + 1e-8)


def _rand_color(rng) -> np.ndarray:
    c = rng.uniform(0.7, 1.0, 3).astype(np.float32)
    if rng.random() > 0.5:
        c[0] = min(1.0, c[0] * 1.15)
    else:
        c[2] = min(1.0, c[2] * 1.15)
    return c


def _rand_sh_coeffs(rng) -> np.ndarray:
    """Random order-2 SH coefficients (9, 3) for plausible diffuse lighting."""
    coeffs = np.zeros((9, 3), dtype=np.float32)
    color = _rand_color(rng)
    intensity = float(rng.uniform(0.5, 2.0))
    coeffs[0] = color * intensity
    scale1 = 0.5 * intensity
    scale2 = 0.3 * intensity
    coeffs[1:4] = rng.uniform(-scale1, scale1, (3, 3)).astype(np.float32)
    coeffs[4:9] = rng.uniform(-scale2, scale2, (5, 3)).astype(np.float32)
    return coeffs


def _rand_distinct_albedos(rng, min_dist: float = 0.5) -> tuple:
    a = rng.random(3).astype(np.float32)
    for _ in range(200):
        b = rng.random(3).astype(np.float32)
        if np.linalg.norm(a - b) >= min_dist:
            return a, b
    return a, (1.0 - a).astype(np.float32)


def _rand_ct_params(rng) -> dict:
    return {
        "metallic":  0.0,
        "roughness": float(rng.uniform(0.05, 1.0)),
    }


def _rand_lighting_config(rng) -> dict:
    return {"sh": {"coeffs": _rand_sh_coeffs(rng).tolist()}}


# ── GT map generation ─────────────────────────────────────────────────────────

def create_scenes(n_scenes: int = 1, seed: int = 42) -> None:
    """
    Generate ground-truth maps for n_scenes using per-object rasterisation.

    Each object is rasterised individually.  Pixels are assigned to the
    closest object by distance to camera (front-wins depth compositing),
    which is exact for any mesh geometry.

    Output per scene (raw_dataset/raw/<scene_id>/):
        normal_map.png   — world normals encoded (N*0.5+0.5)*255
        albedo_map.png   — per-pixel GT albedo
        mask.png         — colour-ID mask (background = black)
        params.json      — objects list + 10 SH lighting configs
    """
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(seed)

    for i in range(n_scenes):
        scene_id = f"scene_{i:04d}"
        out_dir = os.path.join(_ROOT, "raw", scene_id)
        os.makedirs(out_dir, exist_ok=True)

        albedo_a, albedo_b = _rand_distinct_albedos(rng)
        objs = []
        for side_idx, (side, albedo) in enumerate(zip((-1, +1), (albedo_a, albedo_b))):
            radius = float(rng.uniform(0.4, 0.80))
            offset_x = float(rng.uniform(0.35, 0.50))
            position = np.array([
                side * offset_x + float(rng.uniform(-0.05, 0.05)),
                float(rng.uniform(-0.20, 0.20)),
                0.0,
            ], dtype=np.float32)
            objs.append({
                "mask_color": _OBJECT_COLORS[side_idx],
                "radius":     radius,
                "position":   position.tolist(),
                "albedo":     albedo.tolist(),
                "ct":         _rand_ct_params(rng),
            })

        # ── depth-composite GT maps ───────────────────────────────────────────
        H, W = _H, _W
        best_depth = torch.full((H, W), float('inf'), device=dev)
        normals_c  = torch.zeros(H, W, 3, device=dev)
        albedo_c   = torch.zeros(H, W, 3, device=dev)
        mask_c     = torch.zeros(H, W, 3, device=dev)

        for obj in objs:
            mesh = _sphere_mesh(obj["radius"],
                                np.array(obj["position"], dtype=np.float32))
            normals_i, frag_pos_i, hit_i, cam_t = rasterize_geometry(
                mesh, _CAM, width=W, height=H, smooth=True, device=dev,
            )
            depth_i = (frag_pos_i - cam_t).norm(dim=-1)   # (H, W)
            update  = hit_i & (depth_i < best_depth)

            best_depth[update] = depth_i[update]
            normals_c[update]  = normals_i[update]
            albedo_c[update]   = torch.tensor(
                np.array(obj["albedo"], dtype=np.float32), device=dev)
            mask_c[update]     = torch.tensor(
                np.array(obj["mask_color"], dtype=np.float32) / 255.0, device=dev)

        # ── save ──────────────────────────────────────────────────────────────
        def _u8(t: torch.Tensor) -> np.ndarray:
            return (t.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)

        Image.fromarray(_u8(normals_c * 0.5 + 0.5)).save(
            os.path.join(out_dir, "normal_map.png"))
        Image.fromarray(_u8(albedo_c)).save(
            os.path.join(out_dir, "albedo_map.png"))
        Image.fromarray(_u8(mask_c)).save(
            os.path.join(out_dir, "mask.png"))

        params = {
            "scene_id": scene_id,
            "objects":  objs,
            "lighting": [_rand_lighting_config(rng) for _ in range(10)],
        }
        with open(os.path.join(out_dir, "params.json"), "w") as f:
            json.dump(params, f, indent=2)

        for oi, o in enumerate(objs):
            print(f"  [{scene_id}] obj{oi}: r={o['radius']:.2f}  "
                  f"pos={np.array(o['position'])[:2].round(2)}  "
                  f"albedo={np.array(o['albedo']).round(2)}")


# ── tiny raw data ─────────────────────────────────────────────────────────────

def create_tiny_raw(seed: int = 0) -> None:
    """
    Create 3×6 pixel ground-truth images (one pixel per unique surface patch).

    Output (raw_dataset/raw_tiny/):
        normals_a.png   — 18 pixels, 8 unique random normals
        normals_b.png   — 18 pixels, 9 unique random normals
        albedo_tiny.png — 18 pixels, 18 unique random albedo colors
    Normals encoded as (N*0.5+0.5)*255 uint8.
    """
    rng = np.random.default_rng(seed)
    out_dir = os.path.join(_ROOT, "raw_tiny")
    os.makedirs(out_dir, exist_ok=True)

    albedo_flat = rng.random((18, 3)).astype(np.float32)
    Image.fromarray(
        (albedo_flat.reshape(3, 6, 3) * 255).astype(np.uint8)
    ).save(os.path.join(out_dir, "albedo_tiny.png"))

    def _make_normal_img(n_unique: int) -> np.ndarray:
        unique = np.stack([_rand_surface_normal(rng) for _ in range(n_unique)])
        assign = rng.integers(0, n_unique, size=18)
        for j in range(n_unique):
            if j not in assign:
                assign[rng.integers(0, 18)] = j
        normals = unique[assign]
        encoded = ((normals * 0.5 + 0.5) * 255).clip(0, 255).astype(np.uint8)
        return encoded.reshape(3, 6, 3)

    Image.fromarray(_make_normal_img(8)).save(os.path.join(out_dir, "normals_a.png"))
    Image.fromarray(_make_normal_img(9)).save(os.path.join(out_dir, "normals_b.png"))
    print(f"Tiny raw data → {out_dir}")


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== create_scenes ===")
    create_scenes(n_scenes=1)
    print("=== create_tiny_raw ===")
    create_tiny_raw()
