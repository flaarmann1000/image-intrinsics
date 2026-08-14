#!/usr/bin/env python
"""
scripts/convert_mit_datasets.py — convert the MIT multi-illumination TEST set into the
canonical scene-dir structure (light_NNN.png / GT .npy+.png / config.json) that
load_scene / decompose_scene consume, writing to local_datasets/canonical/.

Source per scene (flat dir), e.g. multi_illumination_test_mip2_jpg/everett_dining1/:
    dir_0_mip2.jpg .. dir_24_mip2.jpg           25 sRGB LDR observations (varying light)
    marigold_normal_dir_0_mip2.png              Marigold-predicted normals  (camera frame)
    marigold_albedo_dir_0_mip2.png              Marigold-predicted albedo
    marigold_roughness_dir_0_mip2.png           Marigold-predicted roughness (grayscale)
    marigold_metallicity_dir_0_mip2.png         Marigold-predicted metallic  (grayscale)
    meta.json, probes/, materials_mip2.png, thumb.jpg   (not used)

There is NO ground-truth lighting (so no sh_NNN.npy is written — load_scene returns
sh_coeffs=None, which only disables the SH relighting metric), and NO ground-truth
intrinsics: normals + albedo + roughness + metallic all come from the Marigold prediction
(one prediction per scene, from dir_0), used as the fixed GT-substitute maps.

Conventions:
  * Observations are sRGB 8-bit JPGs, so they map to the canonical "srgb" variant:
    saved as light_NNN.png (values unchanged); load_scene linearises them at decode time.
  * Marigold normals are already in the canonical camera frame (front-facing -> +Z, +Y up,
    matching idr.pipelines.mit and make_proxy_geometry's frontal proxy), decoded RGB->[-1,1]
    and re-normalised. No rotation/flip needed (there is no depth map to recover a camera
    from — the proxy geometry's frontal view is the convention here, as in mit.py).
  * Marigold albedo is linearised (sRGB->linear) so it shares the linear space the observations
    are decoded into (toggle with --no_linearize_albedo). roughness/metallic are grayscale.

Re-runnable: a scene is rebuilt only when its observation count changes, or with --overwrite.

Usage:
    python scripts/convert_mit_datasets.py                       # all test scenes
    python scripts/convert_mit_datasets.py --scenes everett_dining1 --overwrite
    python scripts/convert_mit_datasets.py --downsample 2        # 1000x1500 -> 500x750
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # repo root for `idr`
sys.path.insert(0, str(Path(__file__).resolve().parent))       # sibling scripts for reuse

import numpy as np
from PIL import Image

from idr.paths import REPO_ROOT
from idr.data.scene_io import srgb_to_linear
from build_canonical_datasets import write_gt, _stride    # identical GT encoding + striding

DEFAULT_ROOT = REPO_ROOT / "local_datasets" / "MIT" / "multi_illumination_test_mip2_jpg"
# Grouped under a per-dataset subdir of canonical/, alongside INFINITE/ (the indoor set).
DEFAULT_OUT = REPO_ROOT / "local_datasets" / "canonical" / "MIT-train"


def _obs_files(scene_dir: Path):
    return sorted(scene_dir.glob("dir_*_mip2.jpg"),
                  key=lambda p: int(re.search(r"dir_(\d+)_mip2", p.name).group(1)))


def _marigold(scene_dir: Path, name: str) -> Path:
    """The single Marigold map of a given kind (predicted from dir_0)."""
    hits = sorted(scene_dir.glob(f"marigold_{name}_dir_*_mip2.png"))
    if not hits:
        raise FileNotFoundError(f"missing marigold_{name}_* in {scene_dir}")
    return hits[0]


def _read_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), np.float32) / 255.0


def load_marigold_gt(scene_dir: Path, linearize_albedo: bool) -> dict:
    """Marigold predictions as the canonical GT dict. Normals decoded to the camera frame."""
    n = _read_rgb(_marigold(scene_dir, "normal")) * 2.0 - 1.0
    nlen = np.linalg.norm(n, axis=-1, keepdims=True)
    mask = nlen[:, :, 0] > 0.5
    normals = np.where(mask[:, :, None], n / np.clip(nlen, 1e-6, None), 0.0).astype(np.float32)

    albedo = _read_rgb(_marigold(scene_dir, "albedo"))
    if linearize_albedo:
        albedo = srgb_to_linear(albedo)
    rough = _read_rgb(_marigold(scene_dir, "roughness"))[:, :, :1].astype(np.float32)
    metal = _read_rgb(_marigold(scene_dir, "metallicity"))[:, :, :1].astype(np.float32)
    return dict(albedo=albedo.astype(np.float32), normals=normals, mask=mask,
                roughness=rough, metallic=metal)


def _up_to_date(out_dir: Path, n_src: int) -> bool:
    cfg_p = out_dir / "config.json"
    if not cfg_p.exists():
        return False
    try:
        return int(json.loads(cfg_p.read_text()).get("n_lights", -1)) == n_src
    except Exception:
        return False


def convert_scene(scene_dir: Path, out_dir: Path, ds: int, hl_mode: str,
                  linearize_albedo: bool, n_lights: int | None) -> int:
    gt = _stride(load_marigold_gt(scene_dir, linearize_albedo), ds)
    H, W = gt["mask"].shape

    obs = _obs_files(scene_dir)
    if n_lights is not None:
        obs = obs[:n_lights]

    shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_gt(out_dir, gt)

    for i, p in enumerate(obs):
        arr = np.asarray(Image.open(p).convert("RGB"), np.uint8)
        if arr.shape[:2] != (gt["mask"].shape[0] * ds, gt["mask"].shape[1] * ds):
            arr = np.asarray(Image.fromarray(arr).resize(
                (gt["mask"].shape[1] * ds, gt["mask"].shape[0] * ds), Image.LANCZOS), np.uint8)
        if ds > 1:
            arr = np.ascontiguousarray(arr[::ds, ::ds])
        Image.fromarray(arr).save(out_dir / f"light_{i:03d}.png")       # sRGB, linearised at load

    (out_dir / "config.json").write_text(json.dumps({
        "variant": "srgb", "n_lights": len(obs), "light_type": "light", "lighting": "unknown",
        "prereduced_downsample": ds, "hl_mode": hl_mode, "diffuse_fresnel": True,
        "marigold_intrinsics": True, "albedo_linearized": bool(linearize_albedo),
        "src": str(scene_dir),
        "note": "MIT multi-illumination test; normals + albedo/roughness/metallic are Marigold "
                "predictions; no GT lighting; observations sRGB (linearised by load_scene)"},
        indent=1))
    print(f"      {len(obs)} lights  {H}x{W}  ds={ds} -> {out_dir.name}")
    return len(obs)


def discover(root: Path):
    scenes = []
    for sd in sorted(p for p in root.iterdir() if p.is_dir()):
        if any(sd.glob("dir_*_mip2.jpg")) and any(sd.glob("marigold_normal_dir_*_mip2.png")):
            scenes.append(sd)
    return scenes


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--scenes", nargs="+", default=None, help="scene-name prefixes to include")
    ap.add_argument("--hl_mode", choices=["analytic", "lut"], default="analytic")
    ap.add_argument("--downsample", type=int, default=1)
    ap.add_argument("--no_linearize_albedo", dest="linearize_albedo", action="store_false",
                    help="store the Marigold albedo as-is instead of sRGB->linear")
    ap.add_argument("--n_lights", type=int, default=None, help="cap observations per scene (testing)")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    if not args.root.exists():
        raise SystemExit(f"MIT test set not found: {args.root}")
    scenes = discover(args.root)
    if args.scenes:
        scenes = [s for s in scenes if any(s.name.startswith(p) for p in args.scenes)]
    if not scenes:
        raise SystemExit(f"no convertible scenes under {args.root}")
    print(f"root={args.root}\nout={args.out}\n{len(scenes)} scene(s)  "
          f"downsample={args.downsample}  linearize_albedo={args.linearize_albedo}\n")

    n_built = n_skipped = 0
    for i, sd in enumerate(scenes, 1):
        out_dir = args.out / sd.name
        n_src = len(_obs_files(sd)) if args.n_lights is None else min(len(_obs_files(sd)), args.n_lights)
        print(f"[{i}/{len(scenes)}] {sd.name}")
        if _up_to_date(out_dir, n_src) and not args.overwrite:
            print(f"      up to date ({n_src} lights) — skip"); n_skipped += 1; continue
        convert_scene(sd, out_dir, args.downsample, args.hl_mode, args.linearize_albedo, args.n_lights)
        n_built += 1

    print(f"\nDONE -> {args.out}   built={n_built} skipped={n_skipped}")


if __name__ == "__main__":
    main()
