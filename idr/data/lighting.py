"""Constructing the SH light sets a dataset is rendered under."""
from __future__ import annotations

from pathlib import Path

import numpy as np

import json

from PIL import Image

from idr.render import EnvMap, SHLighting
from idr.config import LIGHT_COLOR, LIGHT_INTENSITY
def _make_directional_sh(az_deg: float = 45.0, el_deg: float = 0.0,
                          color: np.ndarray = LIGHT_COLOR,
                          intensity: float = LIGHT_INTENSITY) -> SHLighting:
    """Create SH lighting from a single directional light.

    az_deg: azimuth in the XZ plane, measured from +Z (0=frontal, 90=right)
    el_deg: elevation above the XZ plane (positive = upward)
    """
    az = np.radians(az_deg)
    el = np.radians(el_deg)
    direction = np.array([
        np.sin(az) * np.cos(el),
        np.sin(el),
        np.cos(az) * np.cos(el),
    ], dtype=np.float32)
    return SHLighting.directional(direction, color, intensity=intensity)


def make_dc_lifted_sh_lighting(
    src_dir: Path,
    out_dir: Path,
    resolution: int = 64,
    eps: float = 1e-4,
    normalize: bool = True,
) -> None:
    """Convert a ref-SH-lighting dir (sh_NNN.npy) into a DC-LIFTED variant.

    Rectified env maps (max(SH,0), as in EnvMap.from_sh) contain >order-2
    content that the SH decomposition model cannot represent. DC-lifting
    instead raises the DC coefficient per channel until the map is
    non-negative, so the map stays EXACTLY order-2 and the saved sh_NNN.npy
    are its true GT coefficients.

    Writes per light: sh_NNN.npy (lifted coeffs, peak-normalized to map max=1
    if normalize), sh_env_map_NNN.png (8-bit, for BlenderProc), and
    sh_env_map_NNN.exr (lossless float — preferred by render_front3d_multipass
    when present, eliminating the 8-bit env quantization).
    """
    import os
    os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
    import cv2  # noqa: PLC0415

    src_dir, out_dir = Path(src_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sh_files = sorted(src_dir.glob("sh_*.npy"))
    if not sh_files:
        raise FileNotFoundError(f"No sh_*.npy in {src_dir}")
    lifts = []
    for f in sh_files:
        coeffs = np.load(f).astype(np.float32)
        env, lifted = EnvMap.from_sh_dc_lifted(SHLighting(coeffs), resolution=resolution, eps=eps)
        img = env.image
        lifts.append(float(lifted[0].mean() - coeffs[0].mean()))
        if normalize:
            s = 1.0 / max(float(img.max()), 1e-8)
            img, lifted = img * s, lifted * s
        idx = f.stem.split("_")[-1]
        np.save(out_dir / f"sh_{idx}.npy", lifted.astype(np.float32))
        Image.fromarray((np.clip(img, 0, 1) * 255).round().astype(np.uint8)).save(
            out_dir / f"sh_env_map_{idx}.png")
        cv2.imwrite(str(out_dir / f"sh_env_map_{idx}.exr"), img[:, :, ::-1].astype(np.float32))
    with open(out_dir / "config.json", "w") as fh:
        json.dump({"src_dir": str(src_dir), "dc_lifted": True, "eps": eps,
                   "resolution": resolution, "normalize": normalize,
                   "n_lights": len(sh_files)}, fh, indent=2)
    print(f"DC-lifted {len(sh_files)} lights -> {out_dir}  "
          f"(mean DC lift before normalization: {np.mean(lifts):.4f})")
