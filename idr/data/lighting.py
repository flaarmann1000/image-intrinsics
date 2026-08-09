"""Constructing the SH light sets a dataset is rendered under."""
from __future__ import annotations

from pathlib import Path

import numpy as np

import json

from PIL import Image

from idr.render import EnvMap, SHLighting, build_sh_basis
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


def _project_env_to_sh(img: np.ndarray, order: int,
                       n_samples: int = 200_000, seed: int = 42) -> np.ndarray:
    """Monte-Carlo project an equirect env map to real SH of the given order.

    Same estimator/seed as SHLighting.from_env_map, but for arbitrary order so it
    can reach band 3. Returns ((order+1)^2, 3)."""
    env = EnvMap(img)
    rng = np.random.default_rng(seed)
    phi   = rng.uniform(0, 2 * np.pi, n_samples).astype(np.float32)
    cos_t = rng.uniform(-1, 1, n_samples).astype(np.float32)
    sin_t = np.sqrt(1 - cos_t ** 2)
    dirs  = np.stack([sin_t * np.cos(phi), cos_t, sin_t * np.sin(phi)], axis=1)
    Y = build_sh_basis(dirs, order=order)
    L = env.sample(dirs)
    return (4 * np.pi / n_samples) * np.einsum("ni,nj->ij", Y, L)


def make_sh3_upgraded_lighting(
    dclift_dir: Path,
    env_src_dir: Path,
    out_dir: Path,
    resolution: int = 256,
) -> None:
    """Order-3 (16-coeff) "strict upgrade" of an order-2 DC-lifted SH light set.

    For each light NNN, this keeps the existing dclift order-2 coefficients EXACTLY
    (bands 0-2) and *appends* a band-3 block (7 coeffs × 3 RGB) estimated by projecting
    that light's env map (``env_src_dir/sh_env_map_NNN.png``) to order 3 and rescaling
    band 3 per channel to dclift's band-0..2 magnitude. Consequences:

    * the order-2 truncation of every output is bit-identical to ``dclift_dir`` — so an
      SH2 dataset is EXACTLY the truncation of the SH3 one, isolating the band-3 effect;
    * band 3 carries the env map's real ~10% high-frequency content, placed at the same
      scale as the low-frequency lighting already in use.

    No DC-lift/renormalisation is applied (that would perturb bands 0-2); the CT-SH
    shader clamps any small negative lobes at render time, exactly as it does for the
    order-2 maps, so the inverse crime is preserved on both sides.

    Writes per light: ``sh_NNN.npy`` (16,3), ``sh_env_map_NNN.png`` (order-3 raster,
    for inspection), and ``config.json``.
    """
    dclift_dir, env_src_dir, out_dir = Path(dclift_dir), Path(env_src_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sh_files = sorted(dclift_dir.glob("sh_*.npy"),
                      key=lambda p: int(p.stem.split("_")[-1]))
    if not sh_files:
        raise FileNotFoundError(f"No sh_*.npy in {dclift_dir}")
    grid = build_sh_basis(EnvMap._sh_grid_dirs(resolution), order=3)   # (H,W,16)
    ratios = []
    for f in sh_files:
        idx = f.stem.split("_")[-1]
        c2 = np.load(f).astype(np.float32)                            # (9,3) dclift
        env_png = env_src_dir / f"sh_env_map_{idx}.png"
        if not env_png.exists():
            raise FileNotFoundError(f"env map for band-3 estimate missing: {env_png}")
        img = np.array(Image.open(env_png).convert("RGB"), np.float32) / 255.0
        cp = _project_env_to_sh(img, order=3)                         # (16,3)
        b3 = np.zeros((7, 3), np.float32)
        for ch in range(3):
            s = np.linalg.norm(c2[:, ch]) / (np.linalg.norm(cp[:9, ch]) + 1e-9)
            b3[:, ch] = cp[9:16, ch] * s
        c3 = np.concatenate([c2, b3], axis=0).astype(np.float32)      # (16,3)
        np.save(out_dir / f"sh_{idx}.npy", c3)
        ratios.append(float(np.linalg.norm(b3) / (np.linalg.norm(c2) + 1e-9)))
        raster = np.clip(grid @ c3, 0.0, None)                        # (H,W,3)
        vmax = max(float(raster.max()), 1e-8)
        Image.fromarray((np.clip(raster / vmax, 0, 1) * 255).round().astype(np.uint8)).save(
            out_dir / f"sh_env_map_{idx}.png")
    with open(out_dir / "config.json", "w") as fh:
        json.dump({"src_dir": str(dclift_dir), "band3_src": str(env_src_dir),
                   "method": "strict_upgrade_band3", "order": 3, "resolution": resolution,
                   "n_lights": len(sh_files)}, fh, indent=2)
    print(f"SH3-upgraded {len(sh_files)} lights -> {out_dir}  "
          f"(mean band3/band2 ratio: {np.mean(ratios):.1%})")
