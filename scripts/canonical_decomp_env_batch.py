#!/usr/bin/env python
"""
scripts/canonical_decomp_env_batch.py — the canonical-decomposition batch, but with per-image
ENV-MAP lighting (shader="ct_env") instead of SH3 (`canonical_decomp_batch.py`, shader="ct_sh").

Same datasets, discovery, train/val split, resume, and intrinsic-map metrics as the SH batch — it
reuses those helpers directly. The only change is the lighting model: each image's illumination is
an optimized environment map (32x64 texel grid) rather than order-3 SH. Specular is evaluated by
GGX half-vector importance sampling by default (`spec_importance=True`), which stays accurate at all
roughness (no texel-grid aliasing below ~0.3); `--configs grid` uses the faster Riemann sum instead.

Per run (results.json): recon_rmse, albedo_rmse, roughness_rmse, metallic_rmse, and — on datasets
with GT lighting (infinite) — relighting_rmse (decompose_scene's held-out val relight: est
intrinsics under the val lights' GT SH vs the observed val image, so it is directly comparable to
the SH batch). There is no lighting-SH RMSE (lighting is an env map, not SH); the estimated env
maps are saved as env_map_*.png / env_maps_est.npy and shown in the notebook.

Output -> results/canonical_env/<dataset>/<scene>/<config>/  (results.json + estimates + env maps)
Resumable. Usage:
    python scripts/canonical_decomp_env_batch.py
    python scripts/canonical_decomp_env_batch.py --datasets infinite mit --configs base grid
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
# reuse the SH batch's datasets + machinery unchanged
from canonical_decomp_batch import (DATASETS, BASE, discover, resolve_split, compute_metrics,
                                     _write_atomic)

DEFAULT_DATASETS = ["infinite", "mit"]
# Per-dataset gradient-accumulation chunk (img_batch): hold only this many images' autograd
# graphs at once instead of all n_images, to fit big-resolution scenes in GPU memory WITHOUT
# changing n_images or downsample (result-identical grad accumulation, not mini-batching). MIT
# is 1000x1500 (ds=4 -> 250x375, ~2.5x INFINITE's pixels) and OOMs with the S-sample env
# specular; chunk it. 0 = off (whole batch). Overridable with --img_batch.
DATASET_IMG_BATCH = {"mit": 4}

# ── env-map BASE + ablations (mirrors the SH batch's BASE/CONFIGS shape) ───────
# BASE_ENV = SH BASE with the env shader + GGX half-vector importance sampling for the specular term
# (spec_importance=True — accurate at ALL roughness, no texel-grid aliasing below ~0.3). tr_env=
# "softplus" keeps the env radiance non-negative; material transforms match the SH BASE (tr="none")
# so the two batches are comparable.
#
# The importance sampler USED to diverge to NaN a few LBFGS steps in (sqrt of a clamp(min=0)
# quantity → infinite gradient at grazing angles); fixed at the source with clamp(min=1e-8) in
# idr/render/shade_ct.py::_spec_ggx_importance, so it is stable now and the default again. `grid`
# (spec_importance=False, texel-grid Riemann sum) remains available as a faster, lower-fidelity
# alternative.
BASE_ENV = {**BASE, "shader": "ct_env", "tr_env": "softplus",
            "spec_importance": True, "spec_samples": 32, "n_iter" : 100, "lbfgs_max_iter" : 20 }
CONFIGS = {
    # "base":              {},                                   # GGX importance-sampled specular, softplus env
    # "grid":              dict(spec_importance=False),          # texel-grid Riemann sum (faster; aliases <0.3)
    # "samples128":        dict(spec_samples=128),               # finer importance sampling
    # "sigmoid":           dict(tr_metallic="sigmoid", tr_roughness="sigmoid"),   # bounded materials
    "metallic_l1_1e-2_tv1e-3": dict(lambda_metallic_l1=1e-2, lambda_tv=1e-3),
    # "metallic_l1_1e-2_tv1e-3_sigmoid": dict(lambda_metallic_l1=1e-2, lambda_tv=1e-3,
                                            # tr_metallic="sigmoid", tr_roughness="sigmoid"),
}

_WANDB_PROJECT, _WANDB_ENTITY = "canonical-env", "DLVC-intrinsics"


def _resolve_img_batch(ds_name, args):
    """--img_batch overrides for all datasets; else the per-dataset default (0 = off)."""
    if getattr(args, "img_batch", None) is not None:
        return args.img_batch
    return DATASET_IMG_BATCH.get(ds_name, 0)


def build_cfg(config, spec, args):
    common = dict(double=False, diffuse_fresnel=True, gt_npy=True, use_npy=spec["use_npy"],
                  downsample=(args.downsample or spec["ds"]), log_every=10)
    cfg = {**common, **BASE_ENV, **CONFIGS[config]}
    # optional budget overrides (do NOT touch n_images/ds -> stays comparable with the SH runs)
    for k in ("n_iter", "lbfgs_max_iter", "log_every", "spec_samples"):
        v = getattr(args, k, None)
        if v is not None:
            cfg[k] = v
    return cfg


# ── one (scene, config) ───────────────────────────────────────────────────────
def _row(ds_name, scene, config, m, status):
    return dict(dataset=ds_name, scene=scene, config=config,
                recon_rmse=m.get("recon_rmse"), albedo_rmse=m.get("albedo_rmse"),
                roughness_rmse=m.get("roughness_rmse"), metallic_rmse=m.get("metallic_rmse"),
                relighting_rmse=m.get("relighting_rmse"), elapsed_s=m.get("elapsed_s"),
                status=status)


def run_scene(task):
    """Run every requested config for one (dataset, scene). No varpro warm-start (env optimizer is
    LBFGS-only here). Returns one summary row per config."""
    os.environ.setdefault("WANDB_MODE", "disabled")
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from idr.pipelines.decompose import decompose_scene

    spec = DATASETS[task["dataset"]]
    scene_dir = Path(task["scene_dir"])
    base_out = Path(task["out_root"]) / task["dataset"] / task["scene"]
    device, force = task["device"], task["force"]
    n_images_sc, val_images_sc = resolve_split(scene_dir, spec, task["args_ns"])
    rows = []
    for config in task["configs"]:
        out_dir = base_out / config
        results_p = out_dir / "results.json"
        if results_p.exists() and not force:
            try:
                res = json.loads(results_p.read_text())
                rows.append(_row(task["dataset"], task["scene"], config, res["metrics"], "cached"))
                continue
            except Exception:
                pass
        cfg = build_cfg(config, spec, task["args_ns"])
        cfg["n_images"], cfg["val_images"] = n_images_sc, val_images_sc
        cfg["img_batch"] = _resolve_img_batch(task["dataset"], task["args_ns"])
        try:
            if (out_dir / "metrics.json").exists() and not force:
                m = json.loads((out_dir / "metrics.json").read_text())
            else:
                if out_dir.exists():
                    shutil.rmtree(out_dir, ignore_errors=True)
                m = decompose_scene(scene_dir, out_dir, cfg_overrides=dict(cfg), device=device,
                                    wandb_project=_WANDB_PROJECT, wandb_entity=_WANDB_ENTITY)
            extra = compute_metrics(scene_dir, out_dir, cfg, spec, device)   # roughness/metallic RMSE
            metrics = {**{k: m.get(k) for k in ("recon_rmse", "recon_mae", "albedo_rmse",
                        "albedo_mae", "final_loss", "elapsed_s", "albedo_scale",
                        "n_train_images", "n_val_images")}, **extra,
                       "relighting_rmse": m.get("relight_rmse"),
                       "relighting_mae": m.get("relight_mae"),
                       "relighting_rmse_per_light": m.get("relight_rmse_per_light")}
            res = dict(dataset=task["dataset"], scene=task["scene"], config=config, shader="ct_env",
                       scene_dir=str(scene_dir), out_dir=str(out_dir),
                       has_gt_lighting=spec["has_gt_lighting"], cfg=cfg, metrics=metrics, status="ok")
            _write_atomic(results_p, res)
            rows.append(_row(task["dataset"], task["scene"], config, metrics, "ok"))
        except Exception as e:
            import traceback
            rows.append(dict(dataset=task["dataset"], scene=task["scene"], config=config,
                             status="error", error=f"{type(e).__name__}: {e}",
                             traceback=traceback.format_exc()[-1500:]))
    return rows


# ── driver ────────────────────────────────────────────────────────────────────
def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS, choices=list(DATASETS))
    p.add_argument("--configs", nargs="+", default=list(CONFIGS), choices=list(CONFIGS))
    p.add_argument("--views", nargs="*", default=None, help="restrict to these scene/view names")
    p.add_argument("--out_root", type=Path, default=RESULTS_DIR / "canonical_env")
    p.add_argument("--downsample", type=int, default=0, help="override every dataset's default ds")
    p.add_argument("--n_images", type=int, default=16, help="cap TRAINING lights per scene (0 = all)")
    p.add_argument("--n_val", type=int, default=16, help="held-out relighting test-set size (GT-lighting datasets only)")
    p.add_argument("--img_batch", type=int, default=None,
                   help="gradient-accumulation chunk = images held on the graph at once "
                        "(result-identical; lowers peak GPU mem). None = per-dataset default "
                        "(%s); 0 = off." % DATASET_IMG_BATCH)
    p.add_argument("--n_iter", type=int, default=None,
                   help="outer LBFGS steps (default 500). Env converges early; lower it to cut runtime.")
    p.add_argument("--lbfgs_max_iter", type=int, default=None,
                   help="inner LBFGS iters per step (default 40). Biggest per-step-time lever; try 20.")
    p.add_argument("--spec_samples", type=int, default=None,
                   help="GGX importance samples (default 64). 32 ~halves the specular cost (env-only).")
    p.add_argument("--log_every", type=int, default=None, help="log interval (default 25).")
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--force", action="store_true")
    return p


def _f(v):
    return f"{v:.4f}" if isinstance(v, (int, float)) else "n/a"


def main():
    args = build_parser().parse_args()
    if args.n_images == 0:
        args.n_images = None
    configs = list(args.configs)
    tasks = []
    for ds_name in args.datasets:
        items = discover(ds_name, DATASETS[ds_name], set(args.views) if args.views else None)
        for scene, scene_dir in items:
            tasks.append(dict(dataset=ds_name, scene=scene, scene_dir=str(scene_dir),
                              configs=configs, out_root=str(args.out_root), device=args.device,
                              force=args.force, args_ns=args))
    if not tasks:
        raise SystemExit(f"no scenes for datasets={args.datasets} (check {DATASETS})")
    per_ds = {d: sum(t["dataset"] == d for t in tasks) for d in args.datasets}
    total = len(tasks) * len(configs)
    workers = max(1, min(args.workers, len(tasks)))
    print(f"{len(tasks)} scene(s) [{per_ds}] x {len(configs)} config(s) = {total} run(s)  (shader=ct_env)")
    print(f"  configs={configs}\n  downsample={'override ' + str(args.downsample) if args.downsample else 'per-dataset'}"
          f"  workers={workers} device={args.device}"
          f"  img_batch={ {d: _resolve_img_batch(d, args) for d in args.datasets} }"
          f"\n  -> {args.out_root}")
    args.out_root.mkdir(parents=True, exist_ok=True)
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
                tag = f"{r['dataset']}/{r['scene']}/{r['config']}"
                if r.get("status") == "error":
                    print(f"  {tag}  ERROR: {r.get('error')}", flush=True)
                else:
                    print(f"  {tag}  albedo={_f(r.get('albedo_rmse'))} rough={_f(r.get('roughness_rmse'))} "
                          f"metal={_f(r.get('metallic_rmse'))} recon={_f(r.get('recon_rmse'))} "
                          f"relight={_f(r.get('relighting_rmse'))}  ({r['status']})", flush=True)
            tmp = args.out_root / "summary.csv.tmp"; pd.DataFrame(rows).to_csv(tmp, index=False)
            os.replace(tmp, args.out_root / "summary.csv")
            print(f"[{done}/{len(tasks)} scenes]", flush=True)
    df = pd.DataFrame(rows); ok = df[df["status"].isin(["ok", "cached"])] if "status" in df else df
    if len(ok):
        cols = [c for c in ("albedo_rmse", "roughness_rmse", "metallic_rmse", "recon_rmse",
                            "relighting_rmse") if c in ok]
        print("\n=== mean over scenes (dataset x config) ===")
        print(ok.groupby(["dataset", "config"])[cols].mean().round(4).to_string())
    print(f"\nsummary -> {args.out_root / 'summary.csv'}")


if __name__ == "__main__":
    main()
