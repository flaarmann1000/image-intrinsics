#!/usr/bin/env python
"""
scripts/sort_point_datasets.py — package the POINT-light BlenderProc renders into the
canonical dataset tree, as the `point_shadow` and `point_no_shadow` lighting variants.

The env/SH renders were packaged by scripts/build_canonical_datasets.py into
    local_datasets/26-08-09-datasets/sh3/<scene8>_v<view>/{blender_env*, ct-*}/
This script does the same for the point-light re-render, whose per-view layout is

    <root>/<scene-uuid>/<view>/point_shadow_<i>.exr      Cycles renders, hard shadows
    <root>/<scene-uuid>/<view>/point_no_shadow_<i>.exr   Cycles renders, shadows off
    <root>/<scene-uuid>/<view>/{albedo,normals}.exr + {roughness,metallic}.png   GT gbuffers
    <root>/<scene-uuid>/<view>/accepted_setup.json       per-light point specs + cam2world

producing, per (view, variant):

    <out>/<scene8>_v<view>/point_shadow/          light_NNN.npy  {GT}.png+.npy  config.json
    <out>/<scene8>_v<view>/point_no_shadow/       point_lights.json

Point lights are NOT spherical harmonics, so — unlike the env variants — NO sh_NNN.npy is
written (load_scene then returns sh_coeffs=None, which only disables the SH relighting
metric; the images + GT decompose fine). Instead the per-light point specs
(location/energy/color/size/brightness) and the camera pose from accepted_setup.json are
saved to point_lights.json for later use.

Observations come from the EXRs (the source of truth for what was rendered); the point
metadata is aligned to them best-effort by index. Because the render is ONGOING, the script
is re-runnable: a variant is rebuilt when its source EXR count changes (or with --overwrite),
and skipped when already up to date. Views missing GT gbuffers (not yet written) are skipped.

Usage:
    python scripts/sort_point_datasets.py                 # default root + out
    python scripts/sort_point_datasets.py --scenes 1f19c3ef --overwrite
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # repo root for `idr`
sys.path.insert(0, str(Path(__file__).resolve().parent))       # sibling scripts for reuse
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
os.environ.setdefault("WANDB_MODE", "disabled")

import numpy as np
from PIL import Image

from idr.paths import REPO_ROOT
from idr.data.build import build_3dfront_dataset, _sorted_lighting_files
from idr.data.scene_io import load_exr
# GT display/float encoding + striding, identical to the env variants so the point leaves
# are an exact GT match. (build_canonical's own load_gt reads EXR via cv2; we read GT through
# idr.data.scene_io.load_exr instead, which also has the OpenEXR fallback.)
from build_canonical_datasets import write_gt, _stride


def load_gt(gt_dir: Path) -> dict:
    """albedo/normals from EXR (float), roughness/metallic from 16-bit PNG. Mask = |n| > 0.5.
    Mirrors build_canonical_datasets.load_gt but reads EXR through idr.data.scene_io.load_exr."""
    albedo = load_exr(gt_dir / "albedo.exr")[:, :, :3]
    normals = load_exr(gt_dir / "normals.exr")[:, :, :3]
    nlen = np.linalg.norm(normals, axis=-1, keepdims=True)
    mask = nlen[:, :, 0] > 0.5
    normals = np.where(mask[:, :, None], normals / np.clip(nlen, 1e-6, None), 0.0).astype(np.float32)
    rough = np.asarray(Image.open(gt_dir / "roughness.png"), np.float32) / 65535.0
    metal = np.asarray(Image.open(gt_dir / "metallic.png"), np.float32) / 65535.0
    rough = (rough[:, :, None] if rough.ndim == 2 else rough[:, :, :1]).astype(np.float32)
    metal = (metal[:, :, None] if metal.ndim == 2 else metal[:, :, :1]).astype(np.float32)
    return dict(albedo=albedo, normals=normals, mask=mask, roughness=rough, metallic=metal)

DEFAULT_ROOT = Path("/run/media/felix/My Passport/3D-Front/260807_output/point")
DEFAULT_OUT = REPO_ROOT / "local_datasets" / "26-08-09-datasets" / "sh3"

# variant dir name -> source EXR stem
VARIANTS = {
    "point_shadow":    "point_shadow",
    "point_no_shadow": "point_no_shadow",
}

_GT_REQUIRED = ("albedo.exr", "normals.exr", "roughness.png", "metallic.png",
                "albedo.png", "normals.png")


# ── point-light metadata ──────────────────────────────────────────────────────
def _point_meta(view_dir: Path, n: int) -> dict:
    """Per-light point specs + camera pose from accepted_setup.json, aligned to the first
    `n` rendered lights. Best-effort: entries beyond the accepted list get an index only."""
    setup_p = view_dir / "accepted_setup.json"
    if not setup_p.exists():
        return {"cam2world": None, "n_accepted": 0, "lights": [{"index": i} for i in range(n)]}
    setup = json.loads(setup_p.read_text())
    pts = setup.get("point", [])
    lights = []
    for i in range(n):
        if i < len(pts) and pts[i].get("light_spec"):
            ls = pts[i]["light_spec"][0]
            lights.append({"index": i,
                           **{k: ls.get(k) for k in ("type", "color", "location", "energy", "size")},
                           "brightness": pts[i].get("brightness")})
        else:
            lights.append({"index": i})
    return {"cam2world": setup.get("cam2world"), "n_accepted": len(pts), "lights": lights}


# ── one (view, variant) ───────────────────────────────────────────────────────
def _source_exr_count(view_dir: Path, stem: str) -> int:
    return len(_sorted_lighting_files(view_dir, f"{stem}_*.exr"))


def _up_to_date(out_dir: Path, n_src: int) -> bool:
    cfg_p = out_dir / "config.json"
    if not cfg_p.exists():
        return False
    try:
        return int(json.loads(cfg_p.read_text()).get("n_lights", -1)) == n_src
    except Exception:
        return False


def build_point_variant(view_dir: Path, out_dir: Path, stem: str, ds: int,
                        hl_mode: str, n_lights: int | None) -> int:
    """Package <stem>_<i>.exr as light_NNN.npy + float GT + point_lights.json + config.json.
    Returns the number of lights written."""
    shutil.rmtree(out_dir, ignore_errors=True)
    # Observations + exr_scale + a first-cut config (light_type=<stem>).
    build_3dfront_dataset(view_dir, out_dir, variant="exr", lighting=stem)
    cfg = json.loads((out_dir / "config.json").read_text())

    light_files = sorted(out_dir.glob("light_*.npy"))
    if n_lights is not None:
        for f in light_files[n_lights:]:
            f.unlink(missing_ok=True)
        light_files = light_files[:n_lights]
    if ds > 1:
        for f in light_files:
            np.save(f, np.ascontiguousarray(np.load(f)[::ds, ::ds]))
    for f in out_dir.glob("light_*_preview.png"):          # drop the preview PNGs
        f.unlink()

    # Lossless float GT (albedo/normals from EXR, roughness/metallic from 16-bit PNG),
    # encoded exactly like the env variants so gt_npy=True reads the same maps.
    write_gt(out_dir, _stride(load_gt(view_dir), ds))

    n = len(light_files)
    (out_dir / "point_lights.json").write_text(json.dumps(_point_meta(view_dir, n), indent=1))

    cfg.update(n_lights=n, light_type="point", shadow=("no_shadow" not in stem),
               prereduced_downsample=ds, hl_mode=hl_mode, src=str(view_dir))
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=1))
    return n


# ── discovery + driver ────────────────────────────────────────────────────────
def discover(root: Path):
    """(scene-uuid, view-index, view_dir) for every view with GT gbuffers written."""
    views = []
    for sd in sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith("_")):
        for vd in sorted((p for p in sd.iterdir() if p.is_dir() and p.name.isdigit()),
                         key=lambda p: int(p.name)):
            if all((vd / f).exists() for f in _GT_REQUIRED):
                views.append((sd.name, int(vd.name), vd))
    return views


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="point render root")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT, help="dataset tree root")
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=list(VARIANTS))
    ap.add_argument("--scenes", nargs="+", default=None, help="scene-name prefixes to include")
    ap.add_argument("--hl_mode", choices=["analytic", "lut"], default="analytic")
    ap.add_argument("--downsample", type=int, default=1)
    ap.add_argument("--n_lights", type=int, default=None, help="cap lights per view (testing)")
    ap.add_argument("--overwrite", action="store_true",
                    help="rebuild even when the leaf is already up to date")
    args = ap.parse_args()

    if not args.root.exists():
        raise SystemExit(f"point render root not found: {args.root}\n"
                         f"(is the external drive mounted?)")
    views = discover(args.root)
    if args.scenes:
        views = [v for v in views if any(v[0].startswith(p) for p in args.scenes)]
    print(f"root={args.root}\nout={args.out}\n{len(views)} view(s) with GT ready, "
          f"variants={args.variants}  downsample={args.downsample}\n")

    n_built = n_skipped = n_empty = 0
    for vi, (scene, view, vd) in enumerate(views, 1):
        key = f"{scene[:8]}_v{view}"
        print(f"[{vi}/{len(views)}] {key}")
        for var in args.variants:
            stem = VARIANTS[var]
            out_dir = args.out / key / var
            n_src = _source_exr_count(vd, stem)
            if n_src == 0:
                print(f"      {var:16s} no {stem}_*.exr yet — skip"); n_empty += 1; continue
            n_expect = n_src if args.n_lights is None else min(n_src, args.n_lights)
            if _up_to_date(out_dir, n_expect) and not args.overwrite:
                print(f"      {var:16s} up to date ({n_expect} lights) — skip"); n_skipped += 1; continue
            n = build_point_variant(vd, out_dir, stem, args.downsample, args.hl_mode, args.n_lights)
            print(f"      {var:16s} {n} lights -> {out_dir.name}"); n_built += 1

    print(f"\nDONE -> {args.out}   built={n_built} skipped={n_skipped} "
          f"absent={n_empty}")


if __name__ == "__main__":
    main()
