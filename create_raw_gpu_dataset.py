"""
GPU-accelerated dataset creation for intrinsic decomposition experiments.

Functions
---------
create_scenes()       – ground-truth maps + param JSON  (CPU; re-exports from create_raw_dataset)
render_scenes_gpu()   – render scenes on GPU with a configurable shader × light combo list
create_tiny_raw()     – 3×6 px ground-truth maps        (CPU; re-exports from create_raw_dataset)
render_tiny_gpu()     – shade tiny maps with CT+SH on GPU via shade_ct_sh

Default render config: [("ct", "sh")]

Run with:
    python create_raw_gpu_dataset.py
"""

import contextlib
import io
import json
import os
import tempfile

import numpy as np
import torch
from PIL import Image

from raw_renderer import EnvMap, PhongMaterial, PBRMaterial, PointLight, SHLighting
from raw_renderer_gpu import render as _render_gpu
from raw_renderer_gpu.rasterizer import shade_ct_sh

from create_raw_dataset import (
    create_scenes,
    create_tiny_raw,
    _ROOT, _W, _H, _CAM, _OBJECT_COLORS,
    _sphere_mesh,
    _build_light,
    _build_material,
    _rand_sh_coeffs,
)

_COMBOS_DEFAULT = [("ct", "sh")]


@contextlib.contextmanager
def _silent():
    """Suppress stdout for noisy per-sphere render calls."""
    with contextlib.redirect_stdout(io.StringIO()):
        yield


# ── render scenes ─────────────────────────────────────────────────────────────

def render_scenes_gpu(
    combos: list = _COMBOS_DEFAULT,
) -> None:
    """
    Render all scenes in raw_dataset/raw/ using the GPU renderer.

    Each scene's multi-sphere image is built by rendering every sphere
    individually, then compositing with the saved per-object mask.

    Parameters
    ----------
    combos : list of (shader, light_type) pairs
        shader     ∈ {"ct", "phong"}
        light_type ∈ {"sh", "point", "envmap"}

    Output: raw_dataset/rendered_gpu/<shader>/<light_type>/<scene_id>/variant_<k>.png
    """
    raw_root  = os.path.join(_ROOT, "raw")
    scene_ids = sorted(
        d for d in os.listdir(raw_root)
        if os.path.isdir(os.path.join(raw_root, d))
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = os.path.join(tmpdir, "sphere.png")

        for scene_id in scene_ids:
            with open(os.path.join(raw_root, scene_id, "params.json")) as f:
                cfg = json.load(f)

            objs   = cfg["objects"]
            meshes = [
                _sphere_mesh(o["radius"], np.array(o["position"], dtype=np.float32))
                for o in objs
            ]

            # Pixel masks — one bool array (H*W,) per object, from saved mask.png
            mask_img  = np.array(Image.open(os.path.join(raw_root, scene_id, "mask.png")))
            obj_masks = [
                np.all(mask_img == np.array(_OBJECT_COLORS[i], dtype=np.uint8), axis=-1).reshape(-1)
                for i in range(len(objs))
            ]

            for shader, light_type in combos:
                mats    = [_build_material(o, shader) for o in objs]
                out_dir = os.path.join(_ROOT, "rendered_gpu", shader, light_type, scene_id)
                os.makedirs(out_dir, exist_ok=True)

                for k, light_cfg in enumerate(cfg["lighting"]):
                    light  = _build_light(light_cfg, light_type)
                    canvas = np.zeros((_H * _W, 3), dtype=np.uint8)

                    for i, (mesh, mat) in enumerate(zip(meshes, mats)):
                        with _silent():
                            img = _render_gpu(mesh, _CAM, mat, light,
                                              smooth=True, width=_W, height=_H,
                                              output_path=tmp_path)
                        canvas[obj_masks[i]] = img.reshape(-1, 3)[obj_masks[i]]

                    out_path = os.path.join(out_dir, f"variant_{k:02d}.png")
                    Image.fromarray(canvas.reshape(_H, _W, 3)).save(out_path)
                    print(f"  {shader}/{light_type}/{scene_id}/variant_{k:02d}")


# ── render tiny ───────────────────────────────────────────────────────────────

def render_tiny_gpu(seed: int = 0) -> None:
    """
    Shade the 3×6 tiny normal maps with Cook-Torrance + SH on GPU.
    Material: metallic=0 (pure diffuse Lambertian; roughness doesn't enter
    the SH diffuse path).  Generates 10 random SH lighting variants.

    Output: raw_dataset/rendered_tiny_gpu/<normals_a|b>_variant_<k>.png
    """
    rng     = np.random.default_rng(seed)
    raw_dir = os.path.join(_ROOT, "raw_tiny")
    out_dir = os.path.join(_ROOT, "rendered_tiny_gpu")
    os.makedirs(out_dir, exist_ok=True)

    albedo_np = (
        np.array(Image.open(os.path.join(raw_dir, "albedo_tiny.png")), dtype=np.float32)
        / 255.0
    ).reshape(18, 3)

    lights = [SHLighting(_rand_sh_coeffs(rng)) for _ in range(10)]

    dev      = "cuda"
    albedo_t = torch.tensor(albedo_np, device=dev)

    for normal_name in ("normals_a", "normals_b"):
        normals_np = (
            np.array(
                Image.open(os.path.join(raw_dir, f"{normal_name}.png")), dtype=np.float32
            ) / 255.0
        ).reshape(18, 3) * 2.0 - 1.0

        lengths    = np.linalg.norm(normals_np, axis=1, keepdims=True)
        normals_np = normals_np / np.maximum(lengths, 1e-8)
        normals_t  = torch.tensor(normals_np, device=dev)

        for k, light in enumerate(lights):
            coeffs_t  = torch.tensor(light.coeffs, device=dev)
            pixels_t  = shade_ct_sh(albedo_t, normals_t, coeffs_t, metallic=0.0)
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
