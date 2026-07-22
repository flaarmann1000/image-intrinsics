"""Reading and writing the synthetic dataset tree (renders, components, metadata)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np

from idr.config import LIGHT_ANGLES_DEG, LIGHT_COLOR, LIGHT_INTENSITY
from idr.paths import DATASET_ROOT
from .synthetic_scene import _scatter
import torch
from PIL import Image

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
