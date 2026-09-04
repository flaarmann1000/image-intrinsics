#!/usr/bin/env python
"""
scripts/fov_study.py — effect of FIELD-OF-VIEW misspecification. Decompose the SH3 self-rendered
dataset (ct_sh_env = ct-ct_sh-frOn_env, which was RENDERED at fov 60) with a DEVIATED proxy fov
(default 90), no regularization, GT normals — everything identical to the ct_sh_env `base` baseline
in canonical_decomp_batch except `fov_deg`. So the degradation vs the matched fov-60 baseline
(results/canonical_ablation/ct_sh_env/*/base) isolates the fov-misspecification model gap.

`decompose_scene` takes `fov_deg` as a direct kwarg (it is NOT a cfg key), so this study passes it
through per run. Metrics vs GT: recon / albedo / roughness / metallic RMSE, lighting-SH RMSE, and the
held-out relighting RMSE (ct_sh_env has GT SH). Reuses the batch's DATASETS / discover / resolve_split
/ compute_metrics so the split matches the baseline exactly.

Output -> results/fov_study/<dataset>/fov<fov>/<scene>/results.json   (+ estimates). Resumable.
Usage:
    python scripts/fov_study.py                       # ct_sh_env at fov 90 (vs the fov-60 baseline)
    python scripts/fov_study.py --fovs 45 75 90 --views 1c349305_v0
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

import pandas as pd
import torch

from idr.paths import RESULTS_DIR
from canonical_decomp_batch import DATASETS, BASE, discover, resolve_split, compute_metrics, _write_atomic

DEFAULT_DATASET = "ct_sh_env"        # SH3 self-render, rendered at fov 60 (the matched baseline)
DEFAULT_FOVS = [90.0]                # deviated fov(s); 60 is already the canonical_ablation baseline
_WANDB_PROJECT, _WANDB_ENTITY = "fov-study", "DLVC-intrinsics"


class _NS:                            # picklable namespace for resolve_split
    def __init__(self, n_images, n_val, downsample):
        self.n_images, self.n_val, self.downsample = n_images, n_val, downsample


def _row(dataset, scene, fov, m, status):
    return dict(dataset=dataset, scene=scene, fov=fov,
                recon_rmse=m.get("recon_rmse"), albedo_rmse=m.get("albedo_rmse"),
                roughness_rmse=m.get("roughness_rmse"), metallic_rmse=m.get("metallic_rmse"),
                lighting_sh_rmse=m.get("lighting_sh_rmse"), relighting_rmse=m.get("relighting_rmse"),
                elapsed_s=m.get("elapsed_s"), status=status)


def run_one(task):
    os.environ.setdefault("WANDB_MODE", "disabled")
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from idr.pipelines.decompose import decompose_scene

    spec = DATASETS[task["dataset"]]
    scene_dir = Path(task["scene_dir"]); out_dir = Path(task["out_dir"]); fov = float(task["fov"])
    results_p = out_dir / "results.json"
    if results_p.exists() and not task["force"]:
        try:
            return _row(task["dataset"], task["scene"], fov,
                        json.loads(results_p.read_text())["metrics"], "cached")
        except Exception:
            pass
    n_images_sc, val_images_sc = resolve_split(scene_dir, spec, task["args_ns"])
    # no-reg baseline cfg (BASE only), matched to canonical_ablation ct_sh_env/base
    cfg = {**dict(double=False, diffuse_fresnel=True, gt_npy=True, use_npy=spec["use_npy"],
                  downsample=task["ds"], log_every=100), **BASE,
           "n_images": n_images_sc, "val_images": val_images_sc}
    try:
        if out_dir.exists():
            shutil.rmtree(out_dir, ignore_errors=True)
        m = decompose_scene(scene_dir, out_dir, cfg_overrides=dict(cfg), device=task["device"],
                            fov_deg=fov, wandb_project=_WANDB_PROJECT, wandb_entity=_WANDB_ENTITY)
        extra = compute_metrics(scene_dir, out_dir, cfg, spec, task["device"])
        metrics = {**{k: m.get(k) for k in ("recon_rmse", "recon_mae", "albedo_rmse", "albedo_mae",
                    "final_loss", "elapsed_s", "albedo_scale", "n_train_images", "n_val_images")},
                   **extra, "relighting_rmse": m.get("relight_rmse"),
                   "relighting_mae": m.get("relight_mae")}
        res = dict(dataset=task["dataset"], scene=task["scene"], fov=fov, config="noreg",
                   fov_deg=fov, scene_dir=str(scene_dir), out_dir=str(out_dir),
                   has_gt_lighting=spec["has_gt_lighting"], cfg=cfg, metrics=metrics, status="ok")
        _write_atomic(results_p, res)
        return _row(task["dataset"], task["scene"], fov, metrics, "ok")
    except Exception as e:
        import traceback
        return dict(dataset=task["dataset"], scene=task["scene"], fov=fov, status="error",
                    error=f"{type(e).__name__}: {e}", traceback=traceback.format_exc()[-1500:])


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default=DEFAULT_DATASET, choices=list(DATASETS))
    p.add_argument("--fovs", nargs="+", type=float, default=DEFAULT_FOVS,
                   help="deviated proxy fov(s) in degrees (default 90; the dataset was rendered at 60)")
    p.add_argument("--views", nargs="*", default=None, help="restrict to these scene/view names")
    p.add_argument("--out_root", type=Path, default=RESULTS_DIR / "fov_study")
    p.add_argument("--downsample", type=int, default=0, help="0 = the dataset's default ds")
    p.add_argument("--n_images", type=int, default=16, help="training lights per scene")
    p.add_argument("--n_val", type=int, default=16, help="held-out relight test set (GT-lighting only)")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--force", action="store_true")
    return p


def _f(v):
    return f"{v:.4f}" if isinstance(v, (int, float)) else "n/a"


def main():
    args = build_parser().parse_args()
    spec = DATASETS[args.dataset]
    ds = args.downsample or spec["ds"]
    items = discover(args.dataset, spec, set(args.views) if args.views else None)
    if not items:
        raise SystemExit(f"no scenes for dataset={args.dataset}")
    args_ns = _NS(None if args.n_images == 0 else args.n_images, args.n_val, ds)
    out_root = args.out_root / args.dataset
    tasks = [dict(dataset=args.dataset, scene=name, scene_dir=str(sd), fov=fov, ds=ds,
                  out_dir=str(out_root / f"fov{fov:g}" / name), args_ns=args_ns,
                  device=args.device, force=args.force)
             for fov in args.fovs for name, sd in items]
    workers = max(1, min(args.workers, len(tasks)))
    print(f"{len(items)} scene(s) x fovs={args.fovs} = {len(tasks)} run(s)  dataset={args.dataset} "
          f"ds={ds}\n  (baseline fov=60 is results/canonical_ablation/{args.dataset}/*/base)"
          f"\n  workers={workers} device={args.device}  -> {out_root}")
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
            tag = f"{r['scene']}/fov{r['fov']:g}"
            if r.get("status") == "error":
                print(f"[{done}/{len(tasks)}] {tag}  ERROR: {r.get('error')}", flush=True)
            else:
                print(f"[{done}/{len(tasks)}] {tag}  recon={_f(r.get('recon_rmse'))} "
                      f"albedo={_f(r.get('albedo_rmse'))} relight={_f(r.get('relighting_rmse'))}  "
                      f"({r['status']})", flush=True)
    df = pd.DataFrame(rows); ok = df[df["status"].isin(["ok", "cached"])] if "status" in df else df
    if len(ok):
        cols = [c for c in ("recon_rmse", "albedo_rmse", "roughness_rmse", "metallic_rmse",
                            "relighting_rmse") if c in ok]
        print("\n=== mean over scenes (per fov) ===")
        print(ok.groupby("fov")[cols].mean().round(4).to_string())
    print(f"\nsummary -> {out_root / 'summary.csv'}")


if __name__ == "__main__":
    main()
