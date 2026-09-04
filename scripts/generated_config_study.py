#!/usr/bin/env python
"""
scripts/generated_config_study.py — decompose one GENERATED dataset leaf (default
canonical/MIT_GENERATED/neg_shadow_light, built by build_generated_dataset.py) across an ARRAY OF
CONFIGS and compare them. Sibling of generated_decomp_study.py, but the sweep axis is the config
(base / metallic_l1 / seg / seg_tv / ...) instead of ±source.

Each scene leaf is decomposed DIRECTLY (its 16 relit light_*.png are sRGB, linearised by load_scene;
Marigold GT maps are the fixed geometry + eval targets; segmentation.png, if present, drives the
`seg*` cohesion configs). There is no GT lighting for the artistic named lightings, so val_images=0
and metrics are vs the (Marigold) GT: albedo / roughness / metallic / recon RMSE. Relighting is
compared in the notebook against GT intrinsics via a synthetic sweep.

Configs come from canonical_decomp_batch.CONFIGS (so all named configs, incl the new seg ones, are
available). Output -> results/generated_config_study/<domain>/<gen_cfg>/<scene>/<config>/
Resumable per results.json. Usage:
    python scripts/generated_config_study.py
    python scripts/generated_config_study.py --configs base seg seg_tv metallic_l1_1e-2_tv1e-3
    python scripts/generated_config_study.py --dataset_dir local_datasets/canonical/MIT_GENERATED/neg_shadow
"""
import argparse
import json
import os
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import torch

from idr.paths import REPO_ROOT, RESULTS_DIR
from canonical_decomp_batch import CONFIGS, BASE

# DEFAULT_DATASET = REPO_ROOT / "local_datasets" / "canonical" / "MIT_GENERATED" / "neg_shadow_light"
DEFAULT_DATASET = REPO_ROOT / "local_datasets" / "canonical" / "MIT_GENERATED" / "GPT"
# DEFAULT_DATASET = REPO_ROOT / "local_datasets" / "canonical" / "INFINITE"
DEFAULT_CONFIGS = [
"metallic_l1_seg_tv_all_light_mono",
"metallic_l1_seg_tv_albedo_light_mono",
"metallic_l1_seg_tv_metallic_roughness_light_mono",
"seg_tv_all_light_mono",
"metallic_l1_seg_tv_all",
"metallic_l1_tv_all_light_mono"
]
DEFAULT_DOWNSAMPLE = 4                                   # MIT-scale (1000x1500) -> 250x375
# DEFAULT_DOWNSAMPLE = 2                                     # Infinite
_WANDB_PROJECT, _WANDB_ENTITY = "generated-config", "DLVC-intrinsics"


def extra_metrics(scene_dir, out_dir, ds=1):
    """roughness/metallic RMSE vs GT (strided by ds to match the downsampled estimates)."""
    from idr.data.scene_io import load_scene
    ds = int(ds or 1)
    sc = load_scene(Path(scene_dir), gt_npy=True)
    mask = sc["mask_np"][::ds, ::ds] if ds > 1 else sc["mask_np"]
    out = {}
    for key in ("roughness", "metallic"):
        est_p = Path(out_dir) / f"{key}_est.npy"
        if est_p.exists():
            est = np.load(est_p).astype(np.float32)[..., 0][mask]
            gt = (sc[f"{key}_np"][::ds, ::ds] if ds > 1 else sc[f"{key}_np"])[..., 0][mask]
            out[f"{key}_rmse"] = float(np.sqrt(((est - gt) ** 2).mean()))
    return out


def _write_atomic(p, obj):
    tmp = Path(str(p) + ".tmp"); tmp.write_text(json.dumps(obj, indent=1)); os.replace(tmp, p)


def _row(scene, config, m, status, ds=None):
    return dict(scene=scene, config=config, ds=ds,
                recon_rmse=m.get("recon_rmse"), albedo_rmse=m.get("albedo_rmse"),
                roughness_rmse=m.get("roughness_rmse"), metallic_rmse=m.get("metallic_rmse"),
                n_images=m.get("n_train_images"), elapsed_s=m.get("elapsed_s"), status=status)


def run_one(task):
    os.environ.setdefault("WANDB_MODE", "disabled")
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from idr.pipelines.decompose import decompose_scene

    scene_dir = Path(task["scene_dir"]); out_dir = Path(task["out_dir"])
    config = task["config"]; ds = int(task["ds"])
    results_p = out_dir / "results.json"
    if results_p.exists() and not task["force"]:
        try:
            r = json.loads(results_p.read_text())
            return _row(task["scene"], config, r["metrics"], "cached", r.get("ds", ds))
        except Exception:
            pass
    try:
        cfg = {**dict(double=False, diffuse_fresnel=True, gt_npy=True, use_npy=False,
                      downsample=ds, log_every=100), **BASE, **CONFIGS[config],
               "n_images": task["n_images"], "val_images": 0}
        if task["img_batch"]:
            cfg["img_batch"] = task["img_batch"]
        if out_dir.exists():
            shutil.rmtree(out_dir, ignore_errors=True)
        m = decompose_scene(scene_dir, out_dir, cfg_overrides=dict(cfg), device=task["device"],
                            wandb_project=_WANDB_PROJECT, wandb_entity=_WANDB_ENTITY)
        metrics = {**{k: m.get(k) for k in ("recon_rmse", "recon_mae", "albedo_rmse", "albedo_mae",
                    "final_loss", "elapsed_s", "albedo_scale", "n_train_images")},
                   **extra_metrics(scene_dir, out_dir, ds)}
        res = dict(scene=task["scene"], config=config, ds=ds, domain=task["domain"],
                   gen_cfg=task["gen_cfg"], scene_dir=str(scene_dir), out_dir=str(out_dir),
                   cfg=cfg, metrics=metrics, status="ok")
        _write_atomic(results_p, res)
        return _row(task["scene"], config, metrics, "ok", ds)
    except Exception as e:
        import traceback
        return dict(scene=task["scene"], config=config, status="error",
                    error=f"{type(e).__name__}: {e}", traceback=traceback.format_exc()[-1500:])


def discover_scenes(dataset_dir, views):
    return [(p.name, p) for p in sorted(dataset_dir.iterdir())
            if p.is_dir() and (p / "config.json").exists() and (not views or p.name in views)]


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset_dir", type=Path, default=DEFAULT_DATASET,
                   help="a <gen_cfg> dir holding scene leaves (default: MIT_GENERATED/neg_shadow_light)")
    p.add_argument("--configs", nargs="+", default=DEFAULT_CONFIGS, choices=list(CONFIGS))
    p.add_argument("--views", nargs="*", default=None, help="scene-name filter")
    p.add_argument("--out_root", type=Path, default=RESULTS_DIR / "generated_config_study")
    p.add_argument("--downsample", type=int, default=DEFAULT_DOWNSAMPLE)
    p.add_argument("--n_images", type=int, default=16, help="training lights per scene")
    p.add_argument("--img_batch", type=int, default=0, help="grad-accum chunk (0=off; use for ct_env OOM)")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--force", action="store_true")
    return p


def _f(v):
    return f"{v:.4f}" if isinstance(v, (int, float)) else "n/a"


def main():
    args = build_parser().parse_args()
    ds_dir = args.dataset_dir
    if not ds_dir.is_dir():
        raise SystemExit(f"dataset dir not found: {ds_dir}")
    scenes = discover_scenes(ds_dir, set(args.views) if args.views else None)
    if not scenes:
        raise SystemExit(f"no scene leaves under {ds_dir}")
    cfg0 = json.loads((scenes[0][1] / "config.json").read_text())
    domain = cfg0.get("domain", "?"); gen_cfg = cfg0.get("gen_config", ds_dir.name)
    out_root = args.out_root / domain / gen_cfg
    tasks = [dict(scene=name, scene_dir=str(sd), config=c, ds=args.downsample, n_images=args.n_images,
                  img_batch=args.img_batch, domain=domain, gen_cfg=gen_cfg,
                  out_dir=str(out_root / name / c), device=args.device, force=args.force)
             for name, sd in scenes for c in args.configs]
    workers = max(1, min(args.workers, len(tasks)))
    print(f"{len(scenes)} scene(s) x {len(args.configs)} config(s) = {len(tasks)} run(s)  "
          f"[{domain}/{gen_cfg}]")
    print(f"  configs={args.configs}  downsample={args.downsample}  img_batch={args.img_batch or 'off'}"
          f"  workers={workers} device={args.device}\n  -> {out_root}")
    out_root.mkdir(parents=True, exist_ok=True)
    try:
        import multiprocessing as mp
        ctx = mp.get_context("spawn")
    except Exception:
        ctx = None
    rows, done = [], 0
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
        futs = {ex.submit(run_one, t): t for t in tasks}
        for fut in as_completed(futs):
            done += 1; r = fut.result(); rows.append(r)
            tmp = out_root / "summary.csv.tmp"; pd.DataFrame(rows).to_csv(tmp, index=False)
            os.replace(tmp, out_root / "summary.csv")
            tag = f"{r['scene']}/{r['config']}"
            if r.get("status") == "error":
                print(f"[{done}/{len(tasks)}] {tag}  ERROR: {r.get('error')}", flush=True)
            else:
                print(f"[{done}/{len(tasks)}] {tag}  albedo={_f(r.get('albedo_rmse'))} "
                      f"rough={_f(r.get('roughness_rmse'))} metal={_f(r.get('metallic_rmse'))} "
                      f"recon={_f(r.get('recon_rmse'))}  ({r['status']})", flush=True)
    df = pd.DataFrame(rows); ok = df[df["status"].isin(["ok", "cached"])] if "status" in df else df
    if len(ok):
        cols = [c for c in ("albedo_rmse", "roughness_rmse", "metallic_rmse", "recon_rmse") if c in ok]
        print("\n=== mean over scenes (per config) ===")
        print(ok.groupby("config")[cols].mean().round(4).to_string())
    print(f"\nsummary -> {out_root / 'summary.csv'}")


if __name__ == "__main__":
    main()
