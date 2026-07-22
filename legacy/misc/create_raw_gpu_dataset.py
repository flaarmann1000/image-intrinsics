"""
GPU-accelerated dataset creation for intrinsic decomposition experiments.

Functions
---------
create_scenes()       – ground-truth maps + param JSON  (CPU; re-exports from create_raw_dataset)
render_scenes_gpu()   – render CT+SH variants by loading precomputed GT maps and calling shade_ct_sh
create_tiny_raw()     – 3×6 px ground-truth maps        (CPU; re-exports from create_raw_dataset)
render_tiny_gpu()     – shade tiny maps with shade_ct_sh on GPU

Run with:
    python create_raw_gpu_dataset.py
"""

import json
import os

import numpy as np
import torch
from PIL import Image

from raw_renderer_gpu import shade_ct_sh

from create_raw_dataset import (
    create_scenes,
    create_tiny_raw,
    _ROOT,
    _rand_sh_coeffs,
)


# ── render scenes ─────────────────────────────────────────────────────────────

def render_scenes_gpu() -> None:
    """
    Render all CT+SH variants for every scene in raw_dataset/raw/.

    Loads the precomputed GT normal_map.png, albedo_map.png and mask.png,
    then calls shade_ct_sh for each of the 10 SH lighting configs stored in
    params.json — the same function used by the optimizer.

    Output: raw_dataset/rendered_gpu/ct/sh/<scene_id>/variant_<k>.png
    """
    raw_root  = os.path.join(_ROOT, "raw")
    scene_ids = sorted(
        d for d in os.listdir(raw_root)
        if os.path.isdir(os.path.join(raw_root, d))
    )
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    for scene_id in scene_ids:
        with open(os.path.join(raw_root, scene_id, "params.json")) as f:
            cfg = json.load(f)

        # Load GT maps saved by create_scenes()
        normals_u8 = np.array(
            Image.open(os.path.join(raw_root, scene_id, "normal_map.png")),
            dtype=np.float32,
        )
        normals_np = normals_u8 / 255.0 * 2.0 - 1.0
        normals_np /= np.linalg.norm(normals_np, axis=-1, keepdims=True).clip(min=1e-8)

        albedo_np = np.array(
            Image.open(os.path.join(raw_root, scene_id, "albedo_map.png")),
            dtype=np.float32,
        ) / 255.0

        mask_np = np.array(Image.open(os.path.join(raw_root, scene_id, "mask.png")))
        fg_mask = mask_np.any(axis=-1)    # (H, W) bool

        normals_t = torch.from_numpy(normals_np).to(dev)   # (H, W, 3)
        albedo_t  = torch.from_numpy(albedo_np).to(dev)    # (H, W, 3)
        mask_t    = torch.from_numpy(fg_mask).to(dev).unsqueeze(-1)  # (H, W, 1)

        out_dir = os.path.join(_ROOT, "rendered_gpu", "ct", "sh", scene_id)
        os.makedirs(out_dir, exist_ok=True)

        for k, light_cfg in enumerate(cfg["lighting"]):
            coeffs_t = torch.tensor(
                np.array(light_cfg["sh"]["coeffs"], dtype=np.float32), device=dev
            )
            shading = shade_ct_sh(normals_t, albedo_t, coeffs_t, metallic=0.0)
            shading = shading * mask_t

            img_u8 = (shading.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            out_path = os.path.join(out_dir, f"variant_{k:02d}.png")
            Image.fromarray(img_u8).save(out_path)
            print(f"  ct/sh/{scene_id}/variant_{k:02d}")


# ── render tiny ───────────────────────────────────────────────────────────────

def render_tiny_gpu(seed: int = 0) -> None:
    """
    Shade the 3×6 tiny normal maps with shade_ct_sh on GPU.
    Generates 10 random SH lighting variants.

    Output: raw_dataset/rendered_tiny_gpu/<normals_a|b>_variant_<k>.png
    """
    rng     = np.random.default_rng(seed)
    raw_dir = os.path.join(_ROOT, "raw_tiny")
    out_dir = os.path.join(_ROOT, "rendered_tiny_gpu")
    os.makedirs(out_dir, exist_ok=True)

    dev = "cuda" if torch.cuda.is_available() else "cpu"

    albedo_np = (
        np.array(Image.open(os.path.join(raw_dir, "albedo_tiny.png")), dtype=np.float32)
        / 255.0
    ).reshape(18, 3)
    albedo_t = torch.tensor(albedo_np, device=dev)

    sh_coeffs_list = [_rand_sh_coeffs(rng) for _ in range(10)]

    for normal_name in ("normals_a", "normals_b"):
        normals_np = (
            np.array(
                Image.open(os.path.join(raw_dir, f"{normal_name}.png")), dtype=np.float32
            ) / 255.0
        ).reshape(18, 3) * 2.0 - 1.0
        normals_np /= np.linalg.norm(normals_np, axis=1, keepdims=True).clip(min=1e-8)
        normals_t = torch.tensor(normals_np, device=dev)

        for k, coeffs in enumerate(sh_coeffs_list):
            coeffs_t  = torch.tensor(coeffs, device=dev)
            pixels_t  = shade_ct_sh(normals_t, albedo_t, coeffs_t, metallic=0.0)
            pixels_u8 = (pixels_t.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            out_path  = os.path.join(out_dir, f"{normal_name}_variant_{k:02d}.png")
            Image.fromarray(pixels_u8.reshape(3, 6, 3)).save(out_path)
            print(f"  {normal_name}  variant_{k:02d}")


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== create_scenes ===");     create_scenes(n_scenes=1)
    print("=== render_scenes_gpu ==="); render_scenes_gpu()
    print("=== create_tiny_raw ===");   create_tiny_raw()
    print("=== render_tiny_gpu ===");   render_tiny_gpu()
