#!/usr/bin/env python
"""
scripts/marigold_normals_sh3.py — analyse the influence of ESTIMATED (Marigold) normals on
decomposition, on the self-rendered SH3 dataset (ct-ct_sh-frOn_env).

For each scene, 16 Marigold normal maps (normals_marigold_000..015.png, one per training image)
sit next to the GT normal map. This study:
  * computes the RMSE (+ mean angular error) of each Marigold map vs the GT normal map,
  * computes the RMSE of the AVERAGED (mean of 16, re-normalised) Marigold map vs GT,
  * decomposes with the estimated normals as fixed geometry — for BOTH the single
    `normals_marigold_000` map and the averaged map — using the SAME config the inconsistency
    study uses (default `light_mono`), matching the canonical batch's ct_sh_env / light_mono
    setup (GT intrinsics, 16 train + 16 val) so the ONLY difference is the normal source.

Compare in the notebook against the canonical batch's GT-normal runs to isolate the effect.
Metrics are vs the TRUE GT intrinsics + the held-out relight set. Output ->
    results/marigold_normal_study/<config>/<scene>/<normal_source>/   (normal_000 | normal_avg)

Resumable per results.json. Usage:
    python scripts/marigold_normals_sh3.py
    python scripts/marigold_normals_sh3.py --config light_mono --views 1c349305_v0
"""
import argparse
import json
import os
import shutil
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import torch
from PIL import Image

from idr.paths import REPO_ROOT, RESULTS_DIR
from canonical_decomp_batch import CONFIGS, BASE

_SH3_ROOT = REPO_ROOT / "local_datasets" / "26-08-09-datasets" / "sh3"
VARIANT = "ct-ct_sh-frOn_env"
DEFAULT_CONFIG = "light_mono"            # the config the inconsistency study currently uses
N_MAPS = 16                              # Marigold maps = first 16 training images
SOURCES = ["normal_000", "normal_avg"]   # single 1st map, and the averaged-16 map
_WANDB_PROJECT, _WANDB_ENTITY = "sh3-marigold-normals", "DLVC-intrinsics"


# ── normal maps ───────────────────────────────────────────────────────────────
def _decode_normal(png_path):
    """Marigold normal PNG -> unit (H,W,3) in [-1,1], camera frame (as-is; verified aligned)."""
    n = np.asarray(Image.open(png_path).convert("RGB"), np.float32) / 255.0 * 2.0 - 1.0
    nl = np.linalg.norm(n, axis=-1, keepdims=True)
    return np.where(nl > 1e-6, n / np.clip(nl, 1e-6, None), 0.0).astype(np.float32)


def marigold_maps(scene_dir, n=N_MAPS):
    scene_dir = Path(scene_dir)
    return [_decode_normal(scene_dir / f"normals_marigold_{i:03d}.png") for i in range(n)]


def averaged_map(maps):
    avg = np.mean(maps, 0)
    nl = np.linalg.norm(avg, axis=-1, keepdims=True)
    return np.where(nl > 1e-6, avg / np.clip(nl, 1e-6, None), 0.0).astype(np.float32)


def normal_rmse(est, gt, mask):
    d = (est - gt)[mask]
    rmse = float(np.sqrt((d ** 2).mean()))
    dot = (est[mask] * gt[mask]).sum(-1).clip(-1, 1)
    return rmse, float(np.degrees(np.arccos(dot)).mean())


def normal_analysis(scene_dir):
    """Per-image + averaged-map RMSE/angle vs GT (scene-level, shared by both runs)."""
    from idr.data.scene_io import load_scene
    sc = load_scene(scene_dir, gt_npy=True)
    gt = sc["normals_np"]; mask = np.linalg.norm(gt, axis=-1) > 0.5
    maps = marigold_maps(scene_dir)
    per = [normal_rmse(m, gt, mask) for m in maps]
    avg = averaged_map(maps)
    a_rmse, a_ang = normal_rmse(avg, gt, mask)
    return dict(per_image_rmse=[p[0] for p in per], per_image_angle=[p[1] for p in per],
                per_image_rmse_mean=float(np.mean([p[0] for p in per])),
                per_image_angle_mean=float(np.mean([p[1] for p in per])),
                avg_map_rmse=a_rmse, avg_map_angle=a_ang), maps, avg


# ── build a scene dir with substituted normals ───────────────────────────────
def build_scene_with_normals(scene_dir, tmp_dir, normals, n_total):
    scene_dir, tmp_dir = Path(scene_dir), Path(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(scene_dir / "config.json", tmp_dir / "config.json")
    for f in ("albedo", "roughness", "metallic"):                 # GT intrinsics (eval targets)
        for ext in ("npy", "png"):
            src = scene_dir / f"{f}.{ext}"
            if src.exists():
                shutil.copy(src, tmp_dir / f"{f}.{ext}")
    for i in range(n_total):
        shutil.copy(scene_dir / f"light_{i:03d}.npy", tmp_dir / f"light_{i:03d}.npy")
        shutil.copy(scene_dir / f"sh_{i:03d}.npy", tmp_dir / f"sh_{i:03d}.npy")
    # substitute the normal map (this is the whole point)
    mask = np.linalg.norm(normals, axis=-1) > 0.5
    np.save(tmp_dir / "normals.npy", normals.astype(np.float32))
    nrm = ((normals + 1) / 2 * 255).astype(np.uint8); nrm[~mask] = 0
    Image.fromarray(nrm).save(tmp_dir / "normals.png")


def extra_metrics(scene_dir, out_dir):
    from idr.data.scene_io import load_scene
    sc = load_scene(Path(scene_dir), gt_npy=True); mask = sc["mask_np"]
    out = {}
    for key in ("roughness", "metallic"):
        est_p = Path(out_dir) / f"{key}_est.npy"
        if est_p.exists():
            est = np.load(est_p).astype(np.float32)[..., 0][mask]
            gt = sc[f"{key}_np"][..., 0][mask]
            out[f"{key}_rmse"] = float(np.sqrt(((est - gt) ** 2).mean()))
    return out


# ── one (scene, source) ───────────────────────────────────────────────────────
def _write_atomic(p, obj):
    tmp = Path(str(p) + ".tmp"); tmp.write_text(json.dumps(obj, indent=1)); os.replace(tmp, p)


def _row(scene, source, m, status):
    return dict(scene=scene, source=source,
                normal_rmse=m.get("normal_map_rmse"), normal_angle=m.get("normal_map_angle"),
                recon_rmse=m.get("recon_rmse"), albedo_rmse=m.get("albedo_rmse"),
                roughness_rmse=m.get("roughness_rmse"), metallic_rmse=m.get("metallic_rmse"),
                relighting_rmse=m.get("relighting_rmse"),
                elapsed_s=m.get("elapsed_s"), status=status)


def run_scene(task):
    """Both normal sources for one scene, sharing the normal analysis + temp maps."""
    os.environ.setdefault("WANDB_MODE", "disabled")
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from idr.pipelines.decompose import decompose_scene

    scene = task["scene"]; scene_dir = Path(task["scene_dir"]); device = task["device"]
    cfg = dict(task["cfg"]); n_train, n_val = task["n_train"], task["n_val"]; n_total = n_train + n_val
    rows = []
    analysis = None
    for source in task["sources"]:
        out_dir = Path(task["out_root"]) / scene / source
        results_p = out_dir / "results.json"
        if results_p.exists() and not task["force"]:
            try:
                rows.append(_row(scene, source, json.loads(results_p.read_text())["metrics"], "cached")); continue
            except Exception:
                pass
        try:
            if analysis is None:
                analysis, maps, avg = normal_analysis(scene_dir)
            normals = maps[0] if source == "normal_000" else avg
            nm_rmse = analysis["per_image_rmse"][0] if source == "normal_000" else analysis["avg_map_rmse"]
            nm_ang = analysis["per_image_angle"][0] if source == "normal_000" else analysis["avg_map_angle"]
            tmp = Path(tempfile.mkdtemp(prefix="mgnrm_"))
            try:
                build_scene_with_normals(scene_dir, tmp, normals, n_total)
                overrides = {**cfg, "n_images": n_total, "val_images": n_val}
                if out_dir.exists():
                    shutil.rmtree(out_dir, ignore_errors=True)
                m = decompose_scene(tmp, out_dir, cfg_overrides=overrides, device=device,
                                    wandb_project=_WANDB_PROJECT, wandb_entity=_WANDB_ENTITY)
            finally:
                shutil.rmtree(tmp, ignore_errors=True)
            metrics = {**{k: m.get(k) for k in ("recon_rmse", "recon_mae", "albedo_rmse", "albedo_mae",
                        "final_loss", "elapsed_s", "albedo_scale", "n_train_images", "n_val_images",
                        "relight_rmse")}, "relighting_rmse": m.get("relight_rmse"),
                       "normal_map_rmse": nm_rmse, "normal_map_angle": nm_ang,
                       **extra_metrics(scene_dir, out_dir), **analysis}
            res = dict(scene=scene, source=source, config=task["config"], scene_dir=str(scene_dir),
                       out_dir=str(out_dir), n_train=n_train, n_val=n_val, cfg=cfg,
                       metrics=metrics, status="ok")
            _write_atomic(results_p, res)
            rows.append(_row(scene, source, metrics, "ok"))
        except Exception as e:
            import traceback
            rows.append(dict(scene=scene, source=source, status="error",
                             error=f"{type(e).__name__}: {e}", traceback=traceback.format_exc()[-1500:]))
    return rows


# ── discovery + driver ────────────────────────────────────────────────────────
def discover(views):
    items = []
    if not _SH3_ROOT.exists():
        return items
    for vd in sorted(p for p in _SH3_ROOT.iterdir() if p.is_dir()):
        sd = vd / VARIANT
        if sd.is_dir() and (sd / "normals_marigold_000.png").exists() and any(sd.glob("light_*.npy")):
            if not views or vd.name in views:
                items.append((vd.name, sd))
    return items


def build_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=DEFAULT_CONFIG, choices=list(CONFIGS))
    p.add_argument("--sources", nargs="+", default=SOURCES, choices=SOURCES)
    p.add_argument("--views", nargs="*", default=None)
    p.add_argument("--out_root", type=Path, default=RESULTS_DIR / "marigold_normal_study")
    p.add_argument("--n_train", type=int, default=16)
    p.add_argument("--n_val", type=int, default=16, help="held-out GT-lighting relight test set")
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--force", action="store_true")
    return p


def main():
    args = build_parser().parse_args()
    cfg = {**dict(double=False, diffuse_fresnel=True, gt_npy=True, use_npy=True,
                  downsample=1, log_every=100), **BASE, **CONFIGS[args.config]}
    items = discover(set(args.views) if args.views else None)
    if not items:
        raise SystemExit(f"no {VARIANT} scenes with normals_marigold_000.png under {_SH3_ROOT}")
    out_root = args.out_root / args.config
    tasks = [dict(scene=s, scene_dir=str(sd), sources=args.sources, out_root=str(out_root),
                  cfg=cfg, config=args.config, n_train=args.n_train, n_val=args.n_val,
                  device=args.device, force=args.force) for s, sd in items]
    workers = max(1, min(args.workers, len(tasks)))
    print(f"{len(items)} scene(s) x {len(args.sources)} normal source(s) = {len(items)*len(args.sources)} run(s)")
    print(f"  config={args.config}  sources={args.sources}  n_train={args.n_train} n_val={args.n_val}"
          f"  workers={workers} device={args.device}\n  compare vs canonical batch: "
          f"results/canonical_ablation/ct_sh_env/<scene>/{args.config}  ->  {out_root}")
    out_root.mkdir(parents=True, exist_ok=True)
    try:
        import multiprocessing as mp
        ctx = mp.get_context("spawn")
    except Exception:
        ctx = None
    rows, done = [], 0
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
        futs = {ex.submit(run_scene, t): t for t in tasks}
        for fut in as_completed(futs):
            done += 1
            for r in fut.result():
                rows.append(r)
                tmp = out_root / "summary.csv.tmp"; pd.DataFrame(rows).to_csv(tmp, index=False)
                os.replace(tmp, out_root / "summary.csv")
                tag = f"{r['scene']}/{r['source']}"
                if r.get("status") == "error":
                    print(f"[scene {done}/{len(tasks)}] {tag}  ERROR: {r.get('error')}", flush=True)
                else:
                    print(f"[scene {done}/{len(tasks)}] {tag}  n_rmse={_f(r.get('normal_rmse'))} "
                          f"albedo={_f(r.get('albedo_rmse'))} recon={_f(r.get('recon_rmse'))} "
                          f"relight={_f(r.get('relighting_rmse'))}  ({r['status']})", flush=True)
    df = pd.DataFrame(rows); ok = df[df["status"].isin(["ok", "cached"])] if "status" in df else df
    if len(ok):
        cols = [c for c in ("normal_rmse", "recon_rmse", "albedo_rmse", "roughness_rmse",
                            "metallic_rmse", "relighting_rmse") if c in ok]
        print("\n=== mean over scenes (by normal source) ===")
        print(ok.groupby("source")[cols].mean().round(4).to_string())
    print(f"\nsummary -> {out_root / 'summary.csv'}")


def _f(v):
    return f"{v:.4f}" if isinstance(v, (int, float)) else "n/a"


if __name__ == "__main__":
    main()
