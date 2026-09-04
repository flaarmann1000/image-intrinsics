#!/usr/bin/env python
"""
scripts/mit_method_relight_videos.py — for every MIT scene that has ALL FOUR decomposition
"versions", render one SINGLE-PANEL relight VIDEO PER METHOD in which a hard directional light (a
narrow env-map lobe) ROTATES in azimuth. The four methods are:

    Flux.2 Klein | GPT | Marigold | RGB->X

Flux.2 Klein / GPT are the regularized CT decompositions run on the generated datasets
(results/generated_config_study/MIT/{neg_shadow_light,GPT}/<scene>/<reg_cfg>); Marigold / RGB->X are
the single-image predictors' `marigold/` and `rgbx/` subfolders under canonical/MIT-train/<scene>.
Each method's ESTIMATED intrinsics are relit through the SHARED GT normals (MIT-train Marigold
normals), at the decomposition downsample (ds=4). Predictor albedos are sRGB->linearised before
relighting (the CT albedos are already linear).

Only scenes containing all four versions are processed (others are skipped). Output ->
results/mit_method_relight/<scene>/<method>_relight.mp4  (one per method per scene; mp4 via ffmpeg,
else GIF). Resumable (skips existing unless --overwrite).

Usage:
    python scripts/mit_method_relight_videos.py
    python scripts/mit_method_relight_videos.py --scenes marlborough_kitchen1 --n_frames 72
    python scripts/mit_method_relight_videos.py --el 30 --sigma 3 --fps 24
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch

from idr.paths import REPO_ROOT, RESULTS_DIR
from idr.data.scene_io import linear_to_srgb, srgb_to_linear
from canonical_relight_render import (Relighter, load_est_intrinsics, load_gt_intrinsics,
                                      _write_mp4, _upscale, _label)

MIT_TRAIN = REPO_ROOT / "local_datasets" / "canonical" / "MIT-train"
GEN_ROOT = RESULTS_DIR / "generated_config_study" / "MIT"
DEFAULT_REG_CFG = "metallic_l1_seg_tv_all_light_mono"
# (label, kind, path-relative-to-scene builder). kind: "decomp" (CT run dir) | "pred" (subfolder).
def _method_specs(scene, reg_cfg):
    return [
        ("Flux.2 Klein", "decomp", GEN_ROOT / "neg_shadow_light" / scene / reg_cfg),
        ("GPT",          "decomp", GEN_ROOT / "GPT" / scene / reg_cfg),
        ("Marigold",     "pred",   MIT_TRAIN / scene / "marigold"),
        ("RGB->X",       "pred",   MIT_TRAIN / scene / "rgbx"),
    ]


_SLUG = {"Flux.2 Klein": "flux2_klein", "GPT": "gpt", "Marigold": "marigold", "RGB->X": "rgbx"}


def _present(kind, path):
    return (path / "results.json").exists() if kind == "decomp" else (path / "albedo.npy").exists()


def _load_intr(kind, path, ds):
    """albedo (H,W,3) LINEAR, roughness/metallic (H,W,1) at the ds grid."""
    if kind == "decomp":
        e = load_est_intrinsics(path)                       # already at ds; albedo_scaled linear
        return dict(albedo=e["albedo"], roughness=e["roughness"], metallic=e["metallic"])
    a = srgb_to_linear(np.load(path / "albedo.npy").astype(np.float32))[::ds, ::ds]
    r = np.load(path / "roughness.npy").astype(np.float32)[::ds, ::ds]
    m = np.load(path / "metallic.npy").astype(np.float32)[::ds, ::ds]
    return dict(albedo=a, roughness=np.ascontiguousarray(r), metallic=np.ascontiguousarray(m))


def _decomp_ds(specs, fallback):
    for _, kind, path in specs:
        if kind == "decomp" and (path / "results.json").exists():
            try:
                return int(json.loads((path / "results.json").read_text())["cfg"].get("downsample", fallback) or fallback)
            except Exception:
                pass
    return fallback


def render_scene(scene, reg_cfg, ds_default, az_from, az_to, n_frames, el, sigma, fps, device,
                 out_dir, min_px, overwrite):
    """One SINGLE-PANEL rotating-hard-light video PER METHOD (scenes are pre-filtered to have all
    four, so `present` is the full set). Videos land in out_dir/<scene>/<method_slug>_relight.mp4."""
    scene_dir = MIT_TRAIN / scene
    specs = _method_specs(scene, reg_cfg)
    present = [(lab, kind, p) for lab, kind, p in specs if _present(kind, p)]
    ds = _decomp_ds(specs, ds_default)
    gt = load_gt_intrinsics(scene_dir, ds)                  # shared normals + mask at ds
    rel = Relighter(gt["normals"], gt["mask"], device, torch.float32, True)
    mask = gt["mask"]

    def tm(x):
        return np.clip(linear_to_srgb(np.clip(x, 0, 1)), 0, 1)

    az = np.linspace(az_from, az_to, n_frames, endpoint=False)   # full rotation loops seamlessly
    made = []
    for lab, kind, p in present:
        out = out_dir / scene / f"{_SLUG.get(lab, lab)}_relight"
        if (out.with_suffix(".mp4").exists() or out.with_suffix(".gif").exists()) and not overwrite:
            made.append(f"{_SLUG.get(lab, lab)}(cached)"); continue
        try:
            intr = _load_intr(kind, p, ds)
        except Exception as e:
            made.append(f"{_SLUG.get(lab, lab)}(ERR {type(e).__name__})"); continue
        frames = []
        for a in az:
            im = tm(rel.render_env(intr, float(a), el, sigma_deg=sigma))
            im8 = (im * 255).astype(np.uint8); im8[~mask] = 0
            frames.append(_label(_upscale(im8, min_px), f"az={a:+.0f}"))
        out.parent.mkdir(parents=True, exist_ok=True)
        made.append(Path(_write_mp4(frames, str(out), fps=fps, ping_pong=False)).name)
    del rel
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()
    return f"{scene}: ds={ds} {n_frames}f -> {made}"


def _complete(scene, reg_cfg):
    """True iff the scene has MIT-train geometry AND all four method versions."""
    scene_dir = MIT_TRAIN / scene
    if not (scene_dir / "normals.npy").exists() or not (scene_dir / "config.json").exists():
        return False
    return all(_present(kind, p) for _, kind, p in _method_specs(scene, reg_cfg))


def discover(scenes_filter, reg_cfg):
    """Only scenes that contain ALL four methods (optionally restricted by scenes_filter)."""
    return [p.name for p in sorted(MIT_TRAIN.iterdir())
            if p.is_dir() and (not scenes_filter or p.name in scenes_filter)
            and _complete(p.name, reg_cfg)]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenes", nargs="+", default=None, help="scene-name filter (default: all MIT-train)")
    ap.add_argument("--reg_cfg", default=DEFAULT_REG_CFG, help="CT config name for the Flux/GPT decomps")
    ap.add_argument("--ds", type=int, default=4, help="fallback downsample if not in results.json")
    ap.add_argument("--az_from", type=float, default=-45.0)
    ap.add_argument("--az_to", type=float, default=45.0)
    ap.add_argument("--n_frames", type=int, default=60)
    ap.add_argument("--el", type=float, default=35.0, help="light elevation (deg)")
    ap.add_argument("--sigma", type=float, default=4.0, help="env lobe half-width (deg); smaller = harder")
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--min_px", type=int, default=320, help="upscale each panel to >= this height")
    ap.add_argument("--out_dir", type=Path, default=RESULTS_DIR / "mit_method_relight")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    scenes = discover(set(args.scenes) if args.scenes else None, args.reg_cfg)
    if not scenes:
        raise SystemExit(f"no MIT-train scene has all 4 methods (Flux/GPT decomps + marigold/ + rgbx/)")
    print(f"complete scenes (all 4 methods): {len(scenes)} {scenes}  reg_cfg={args.reg_cfg}  "
          f"az {args.az_from:.0f}->{args.az_to:.0f} "
          f"x{args.n_frames}  el={args.el} sigma={args.sigma}  device={args.device}\n  -> {args.out_dir}")
    for i, scene in enumerate(scenes, 1):
        msg = render_scene(scene, args.reg_cfg, args.ds, args.az_from, args.az_to, args.n_frames,
                           args.el, args.sigma, args.fps, args.device, args.out_dir, args.min_px,
                           args.overwrite)
        print(f"[{i}/{len(scenes)}] {msg}", flush=True)
    print(f"\nDONE -> {args.out_dir}")


if __name__ == "__main__":
    main()
