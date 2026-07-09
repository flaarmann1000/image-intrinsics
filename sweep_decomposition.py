#!/usr/bin/env python
"""
sweep_decomposition.py — parallel hyperparameter search for CT-SH decomposition.

Sibling to run_decomposition.py. Takes the notebook's self-contained `datasets/`
tree, picks the first N views that have a ct_sh CT dataset, and sweeps a small
grid on each — by default:

    lambda_tv                 in {1e-3, 1e-4, 1e-5}
    lambda_metallic_binarize  in {1e-3, 1e-4, 1e-5}
    lbfgs_max_iter            in {30, 40}

with SH order 2. That is 3*3*2 = 18 combos per view; 10 views -> 180 runs.

All runs are launched across a process pool (one CUDA context per worker) so a
GPU with spare memory chews through the whole grid concurrently. Everything else
(the base cfg, dataset discovery, per-run artifacts, resume-on-metrics.json) is
kept identical to run_decomposition.py so results are directly comparable.

Per run: its own out_dir with metrics.json / report.json / intrinsics.png /
relight panels, plus a wandb run whose name+config carry the swept values.
Two CSVs land in --runs_root: sweep_summary.csv (one row per run) and
sweep_by_hp.csv (mean over views for each HP combo), sorted best-first.

Example (VM, run the whole grid 8-wide):
  WANDB_API_KEY=... python sweep_decomposition.py \
      --datasets_root /data/decomp/datasets --runs_root /data/decomp/sweep \
      --num_views 10 --workers 10 \
      --wandb_project 3dfront-hpsweep-gc --wandb_entity DLVC-intrinsics
"""
import argparse
import itertools
import json
import os
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")            # headless
import matplotlib.pyplot as plt
from PIL import Image

# Top-level so spawned workers re-import cleanly (argparse stays under __main__).
sys.path.insert(0, str(Path(__file__).resolve().parent))
from raw_optimizer.dfront_ct import decompose_scene   # noqa: E402


# ───────────────────────── plotting (mirrors run_decomposition.py) ────────────
def save_intrinsics_plot(run_dir, m, save_path):
    gt = Path(m["scene"])
    panels = [("albedo GT", gt / "albedo.png", None),
              ("albedo est", run_dir / "albedo_scaled.png", None),
              ("albedo err", run_dir / "albedo_err.npy", f"RMSE={m['albedo_rmse']:.3f}"),
              ("rough GT", gt / "roughness.png", None),
              ("rough est", run_dir / "roughness_est.png", None),
              ("rough err", run_dir / "roughness_err.npy", f"MAE={m['roughness_err_mean']:.3f}"),
              ("metal GT", gt / "metallic.png", None),
              ("metal est", run_dir / "metallic_est.png", None),
              ("metal err", run_dir / "metallic_err.npy", f"MAE={m['metallic_err_mean']:.3f}")]
    fig, ax = plt.subplots(3, 3, figsize=(9.5, 9.5))
    for a, (t, p, sub) in zip(ax.flat, panels):
        if Path(p).exists():
            im = np.load(p) if str(p).endswith(".npy") else np.array(Image.open(p))
            if str(p).endswith(".npy"):
                im = im.squeeze()
                if im.ndim == 3 and im.shape[-1] == 3:
                    im = im.mean(-1)
            a.imshow(im, cmap="gray" if im.ndim == 2 else None)
        a.set_title(t + (f"\n{sub}" if sub else ""), fontsize=8); a.axis("off")
    fig.suptitle(run_dir.name[:70], fontsize=8); plt.tight_layout()
    fig.savefig(save_path, dpi=80); plt.close(fig)


def save_relight_plots(run_dir, m, ds_dir, downsample):
    keys = m.get("relight_keys", [])
    rmses, maes = m.get("relight_rmse_per_light", []), m.get("relight_mae_per_light", [])
    pdir = run_dir / "relight" / "plots"; pdir.mkdir(parents=True, exist_ok=True)
    for k, key in enumerate(keys):
        relit_p = run_dir / "relight" / f"relit_{key}.npy"
        tgt_p = ds_dir / f"{key}.npy"
        if not relit_p.exists() or not tgt_p.exists():
            continue
        relit = np.load(relit_p)
        tgt = np.load(tgt_p)[::downsample, ::downsample]
        resid = np.abs(relit - tgt).mean(-1)
        fig, ax = plt.subplots(1, 3, figsize=(10, 3.4))
        ax[0].imshow(np.clip(tgt / 2, 0, 1)); ax[0].set_title(f"target {key}", fontsize=8)
        ax[1].imshow(np.clip(relit / 2, 0, 1)); ax[1].set_title("relit (est intrinsics + GT light)", fontsize=8)
        im = ax[2].imshow(resid, cmap="inferno")
        rm = rmses[k] if k < len(rmses) else float("nan")
        ma = maes[k] if k < len(maes) else float("nan")
        ax[2].set_title(f"residual  RMSE={rm:.4f}  MAE={ma:.4f}", fontsize=8)
        plt.colorbar(im, ax=ax[2], fraction=0.046)
        for a in ax:
            a.axis("off")
        plt.tight_layout(); fig.savefig(pdir / f"relight_{key}.png", dpi=80); plt.close(fig)


# ───────────────────────── one run (executed in a worker process) ─────────────
def run_one(task: dict) -> dict:
    """Decompose one (view, hp) combo. Self-contained + picklable so it can run
    in a spawned process. Returns a summary row (+ status/error)."""
    if task["wandb_mode"]:
        os.environ["WANDB_MODE"] = task["wandb_mode"]

    out_dir = Path(task["out_dir"])
    ds_dir  = Path(task["dir"])
    row_base = {k: task[k] for k in
                ("view", "dataset", "lambda_tv", "lambda_metallic_binarize",
                 "lbfgs_max_iter", "sh_order")}

    # Resume: metrics.json is written only on success, so its presence = done and
    # its absence in an existing dir = a partial/preempted run to clear + redo.
    if (out_dir / "metrics.json").exists() and not task["force"]:
        m = json.loads((out_dir / "metrics.json").read_text())
        return {**row_base, **_metrics_row(m), "status": "cached"}
    if out_dir.exists():
        shutil.rmtree(out_dir, ignore_errors=True)

    try:
        m = decompose_scene(ds_dir, out_dir, cfg_overrides=task["cfg"], device=task["device"],
                            wandb_entity=task["wandb_entity"], wandb_project=task["wandb_project"])
        if not task["no_plots"]:
            save_intrinsics_plot(out_dir, m, out_dir / "intrinsics.png")
            save_relight_plots(out_dir, m, ds_dir, task["eff_ds"])
        (out_dir / "report.json").write_text(json.dumps({**row_base, **_metrics_row(m)}, indent=1))
        return {**row_base, **_metrics_row(m), "status": "ok"}
    except Exception as e:                       # keep the sweep alive on one bad run
        import traceback
        return {**row_base, "status": "error", "error": f"{type(e).__name__}: {e}",
                "traceback": traceback.format_exc()[-2000:]}


def _metrics_row(m: dict) -> dict:
    return dict(
        albedo_rmse=m["albedo_rmse"], albedo_mae=m.get("albedo_mae"),
        roughness_mae=m["roughness_err_mean"], metallic_mae=m["metallic_err_mean"],
        train_recon_rmse=m["recon_rmse"], train_recon_mae=m.get("recon_mae"),
        val_relight_rmse=m.get("relight_rmse"), val_relight_mae=m.get("relight_mae"),
        final_loss=m["final_loss"])


# ───────────────────────── discovery: one ct_sh dataset per view ──────────────
def discover_views(datasets_root: Path, dataset_filter, view_filter, num_views, downsample):
    items = []
    for view_dir in sorted(p for p in datasets_root.iterdir() if p.is_dir()):
        if view_filter and not any(view_dir.name.startswith(f) for f in view_filter):
            continue
        # first dataset in this view matching the ct_sh filter (frOn env by default)
        match = None
        for ds_dir in sorted(p for p in view_dir.iterdir() if p.is_dir()):
            if not any(ds_dir.name.startswith(f) for f in dataset_filter):
                continue
            if not (ds_dir / "config.json").exists() or not any(ds_dir.glob("light_*.npy")):
                continue
            match = ds_dir
            break
        if match is None:
            continue
        cfg = json.loads((match / "config.json").read_text())
        eff_ds = 1 if cfg.get("prereduced_downsample", 0) and cfg["prereduced_downsample"] > 1 \
            else (1 if match.name.startswith("ct-") else downsample)
        items.append(dict(view=view_dir.name, name=match.name, dir=match, eff_ds=eff_ds))
        if len(items) >= num_views:
            break
    return items


def build_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--datasets_root", type=Path, default=Path("results/3dfront-batch/datasets"))
    p.add_argument("--runs_root", type=Path, default=Path("results/3dfront-batch/sweep"))
    p.add_argument("--num_views", type=int, default=10, help="First N views to sweep.")
    p.add_argument("--dataset_filter", nargs="+", default=["ct-ct_sh-frOn"],
                   help="Per view, the first dataset dir starting with one of these is swept "
                        "(default the Fresnel-ON ct_sh dataset).")
    p.add_argument("--view_filter", nargs="*", default=None,
                   help="Only view_keys starting with one of these prefixes.")
    p.add_argument("--sh_order", type=int, default=2, choices=[2, 3])
    # ── swept grids ──
    p.add_argument("--lambda_tv", type=float, nargs="+", default=[1e-3, 1e-4, 1e-5])
    p.add_argument("--lambda_metallic_binarize", type=float, nargs="+", default=[1e-3, 1e-4, 1e-5])
    p.add_argument("--lbfgs_max_iter", type=int, nargs="+", default=[30, 40])
    # ── fixed base cfg (mirrors run_decomposition.py) ──
    p.add_argument("--n_train", type=int, default=100)
    p.add_argument("--n_val", type=int, default=28, help="Last N lights held out for relighting.")
    p.add_argument("--n_iter", type=int, default=300)
    p.add_argument("--log_every", type=int, default=25)
    p.add_argument("--downsample", type=int, default=4,
                   help="Decompose downsample for datasets NOT already pre-reduced.")
    p.add_argument("--diffuse_fresnel", type=lambda s: s.lower() in ("1", "true", "on", "yes"),
                   default=True)
    p.add_argument("--double", type=lambda s: s.lower() in ("1", "true", "on", "yes"), default=False,
                   help="float64 (slow on T4/L4; float32 is fine for ct_sh — default here).")
    p.add_argument("--wandb_max_images", type=int, default=6)
    p.add_argument("--log_gt_recon_images", action="store_true")
    # ── execution ──
    p.add_argument("--workers", type=int, default=0,
                   help="Parallel worker processes (0 = min(#runs, CPU count)). The 31^2 LBFGS "
                        "workload is CPU/launch-bound, so ~one per core saturates a spare GPU.")
    p.add_argument("--device", default="cuda")
    p.add_argument("--wandb_project", default="3dfront-hpsweep-gc")
    p.add_argument("--wandb_entity", default="DLVC-intrinsics")
    p.add_argument("--wandb_mode", default=None, choices=[None, "online", "offline", "disabled"],
                   help="Force a wandb mode for every run (default: env/online).")
    p.add_argument("--force", action="store_true", help="Redo runs even if metrics.json exists.")
    p.add_argument("--no_plots", action="store_true")
    p.add_argument("--dry_run", action="store_true", help="Print the run plan and exit.")
    return p


def main():
    args = build_parser().parse_args()

    views = discover_views(args.datasets_root, args.dataset_filter,
                           args.view_filter, args.num_views, args.downsample)
    if not views:
        raise SystemExit(
            f"No views with a dataset matching {args.dataset_filter} under {args.datasets_root}")

    grid = list(itertools.product(args.lambda_tv, args.lambda_metallic_binarize, args.lbfgs_max_iter))
    tasks = []
    for it in views:
        for tv, mb, lbfgs in grid:
            hp = f"tv{tv:g}-mb{mb:g}-it{lbfgs}"
            tag = f"{it['view']}__{it['name']}__SH{args.sh_order}__{hp}"
            cfg = {
                "n_iter": args.n_iter, "log_every": args.log_every,
                "optimizer": "LBFGS", "lbfgs_max_iter": lbfgs,
                "lambda_tv": tv, "lambda_metallic_binarize": mb,
                "tr_metallic": "sigmoid", "tr_roughness": "sigmoid", "tr_albedo": "sigmoid",
                "init_roughness_zero": True, "double": args.double,
                "wandb_max_images": args.wandb_max_images, "diffuse_fresnel": args.diffuse_fresnel,
                "log_gt_recon_images": args.log_gt_recon_images,
                "shader": "ct_sh", "sh_order": args.sh_order, "downsample": it["eff_ds"],
                "n_images": args.n_train + args.n_val, "val_images": args.n_val,
                "hp": hp,                       # rides into the wandb run name for scanning
            }
            tasks.append(dict(
                view=it["view"], dataset=it["name"], dir=str(it["dir"]), eff_ds=it["eff_ds"],
                lambda_tv=tv, lambda_metallic_binarize=mb, lbfgs_max_iter=lbfgs,
                sh_order=args.sh_order, cfg=cfg, out_dir=str(args.runs_root / tag),
                device=args.device, wandb_entity=args.wandb_entity,
                wandb_project=args.wandb_project, wandb_mode=args.wandb_mode,
                force=args.force, no_plots=args.no_plots))

    workers = args.workers or min(len(tasks), os.cpu_count() or 4)
    workers = max(1, min(workers, len(tasks)))
    print(f"{len(views)} view(s) x {len(grid)} combo(s) = {len(tasks)} run(s)  "
          f"| SH{args.sh_order}  | workers={workers}  | device={args.device}")
    print(f"  grid: lambda_tv={args.lambda_tv}  lambda_metallic_binarize="
          f"{args.lambda_metallic_binarize}  lbfgs_max_iter={args.lbfgs_max_iter}")
    print(f"  views: {', '.join(it['view'] for it in views)}")
    if args.dry_run:
        for t in tasks:
            print(f"    - {Path(t['out_dir']).name}")
        return

    args.runs_root.mkdir(parents=True, exist_ok=True)
    rows, done = [], 0
    ctx = None
    try:
        import multiprocessing as mp
        ctx = mp.get_context("spawn")
    except Exception:
        ctx = None

    def _flush_csv():
        df = pd.DataFrame(rows)
        _tmp = args.runs_root / "sweep_summary.csv.tmp"
        df.to_csv(_tmp, index=False)
        os.replace(_tmp, args.runs_root / "sweep_summary.csv")

    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
        futs = {ex.submit(run_one, t): t for t in tasks}
        for fut in as_completed(futs):
            done += 1
            r = fut.result()
            rows.append(r)
            _flush_csv()
            if r.get("status") == "error":
                print(f"[{done}/{len(tasks)}] {r['view']} {_hp_str(r)}  ERROR: {r.get('error')}",
                      flush=True)
            else:
                print(f"[{done}/{len(tasks)}] {r['view']} {_hp_str(r)}  "
                      f"alb_rmse={r.get('albedo_rmse'):.4f}  relight={_f(r.get('val_relight_rmse'))}  "
                      f"recon={_f(r.get('train_recon_rmse'))}  ({r.get('status')})", flush=True)

    # ── aggregate: mean over views for each HP combo, best (lowest relight) first ─
    df = pd.DataFrame(rows)
    ok = df[df["status"].isin(["ok", "cached"])] if "status" in df else df
    if len(ok):
        keys = ["lambda_tv", "lambda_metallic_binarize", "lbfgs_max_iter"]
        agg = (ok.groupby(keys)[["albedo_rmse", "roughness_mae", "metallic_mae",
                                 "train_recon_rmse", "val_relight_rmse"]]
               .agg(["mean", "std"]))
        agg.columns = [f"{a}_{b}" for a, b in agg.columns]
        sort_key = "val_relight_rmse_mean" if ok["val_relight_rmse"].notna().any() else "albedo_rmse_mean"
        agg = agg.sort_values(sort_key)
        agg.to_csv(args.runs_root / "sweep_by_hp.csv")
        print(f"\n=== mean over {ok['view'].nunique()} view(s), best {sort_key} first ===")
        show = ["albedo_rmse_mean", "roughness_mae_mean", "metallic_mae_mean",
                "train_recon_rmse_mean", "val_relight_rmse_mean"]
        print(agg[show].round(4).to_string())
        best = agg.index[0]
        print(f"\nbest combo: lambda_tv={best[0]:g}  lambda_metallic_binarize={best[1]:g}  "
              f"lbfgs_max_iter={best[2]}  ({sort_key}={agg[sort_key].iloc[0]:.4f})")
    n_err = int((df["status"] == "error").sum()) if "status" in df else 0
    print(f"\nsummary -> {args.runs_root / 'sweep_summary.csv'}"
          f"   by-hp -> {args.runs_root / 'sweep_by_hp.csv'}"
          + (f"   ({n_err} run(s) errored)" if n_err else ""))


def _f(v):
    return "n/a" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.4f}"


def _hp_str(r):
    return f"tv{r['lambda_tv']:g} mb{r['lambda_metallic_binarize']:g} it{r['lbfgs_max_iter']}"


if __name__ == "__main__":
    main()
