#!/usr/bin/env python
"""
improve_decomposition_study.py — curriculum / init tricks for CT-SH decomposition.

Runs, on ONE 512^2 ct_sh view, a set of strategies meant to help the decomposition
escape the diffuse/roughness unidentifiability, all against a plain baseline:

  baseline              no tricks
  noisy_spec_init       Gaussian noise on the metallic+roughness init (natural space)
  freeze_spec_warmup    phase 1 optimizes albedo+SH only (metallic/roughness frozen)
  freeze_albedo_warmup  phase 1 optimizes SH+metallic+roughness only (albedo frozen)
  sh_only_warmup        phase 1 optimizes SH only (fit the lighting first)
  sh1_then_sh2          phase 1 uses order-1 lighting (l=2 band frozen), then full SH2
  pixel_subsample_warmup phase 1 fits SH on a random pixel subset, cheaply

The 7 strategies run with BOTH regularizers on (lambda_tv = lambda_metallic_binarize
= --reg_lambda). Two extra runs isolate the regularizer effect on the baseline
(tv-only, binarize-only). That is 9 runs at full 512^2.

Then the plain, UN-regularized baseline is re-run at --downsamples (2 4 8 16) to
show how resolution alone moves the metrics.

Everything uses the lossless float32 GT maps (gt_npy=True) and the .npy observations.
Optimizer is LBFGS (--n_iter 300, --lbfgs_max_iter 40). Warm-up phases spend the
first --warmup steps, so a curriculum run still totals --n_iter steps.

Output (resumable — a run with metrics.json is skipped unless --force):
  <out>/
    summary.csv
    compare_fullres.png        grouped-bar comparison of the 9 full-res runs
    compare_downsample.png     baseline metrics vs resolution
    runs/<run>/                metrics.json, report.json, intrinsics.png,
                               loss_curve.png, relight/, config.json

Example (VM):
  python improve_decomposition_study.py \
      --scene results/3dfront-batch/datasets/1f19c3ef_v2/ct-ct_sh-frOn_env \
      --out results/improve_study --device cuda
"""
import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from raw_optimizer.dfront_ct import decompose_scene            # noqa: E402
from run_decomposition import save_intrinsics_plot, save_relight_plots   # noqa: E402


# ───────────────────────── run specs ─────────────────────────────────────────
def build_runs(args):
    """Return a list of run dicts: name, downsample, cfg-extra (incl. reg)."""
    R = args.reg_lambda
    WARM, TOTAL = args.warmup, args.n_iter
    FINAL = max(1, TOTAL - WARM)
    both = dict(lambda_tv=R, lambda_metallic_binarize=R)

    strategies = {
        "baseline": {},
        "noisy_spec_init": dict(init_spec_noise_std=args.noise_std, init_seed=args.seed),
        "freeze_spec_warmup": dict(
            n_iter=FINAL, curriculum=[dict(n_iter=WARM, opt_params=["albedo", "sh"])]),
        "freeze_albedo_warmup": dict(
            n_iter=FINAL, curriculum=[dict(n_iter=WARM, opt_params=["sh", "metallic", "roughness"])]),
        "sh_only_warmup": dict(
            n_iter=FINAL, curriculum=[dict(n_iter=WARM, opt_params=["sh"])]),
        "sh1_then_sh2": dict(
            n_iter=FINAL, curriculum=[dict(n_iter=WARM, sh_active_order=1)]),
        "pixel_subsample_warmup": dict(
            n_iter=FINAL, curriculum=[dict(n_iter=WARM, opt_params=["sh"], pixel_frac=args.pixel_frac)]),
    }

    fd = args.fullres_downsample
    runs = []
    # 7 strategies, full-res, both regularizers
    for name, extra in strategies.items():
        runs.append(dict(name=name, group="fullres", downsample=fd,
                         reg="both", cfg={**both, **extra}))
    # regularizer isolation on the baseline
    runs.append(dict(name="baseline_tv_only", group="fullres", downsample=fd,
                     reg="tv", cfg=dict(lambda_tv=R, lambda_metallic_binarize=0.0)))
    runs.append(dict(name="baseline_binarize_only", group="fullres", downsample=fd,
                     reg="binarize", cfg=dict(lambda_tv=0.0, lambda_metallic_binarize=R)))
    # UN-regularized baseline at each downsample
    for ds in args.downsamples:
        runs.append(dict(name=f"baseline_ds{ds}", group="downsample", downsample=ds,
                         reg="none", cfg=dict(lambda_tv=0.0, lambda_metallic_binarize=0.0)))
    return runs


def base_cfg(args):
    return {
        "shader": "ct_sh", "sh_order": 2, "optimizer": "LBFGS",
        "n_iter": args.n_iter, "lbfgs_max_iter": args.lbfgs_max_iter, "lr": 1.0,
        "log_every": args.log_every, "loss": "L2", "double": args.double,
        "tr_albedo": "sigmoid", "tr_metallic": "sigmoid", "tr_roughness": "sigmoid",
        "init_roughness_zero": True, "img_batch": args.img_batch,
        "n_images": args.n_train + args.n_val, "val_images": args.n_val,
        "wandb_max_images": 0,
    }


# ───────────────────────── plotting ──────────────────────────────────────────
def save_loss_curve(run_dir, m):
    h = m.get("loss_history") or []
    if not h:
        return
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.semilogy(np.arange(len(h)) * m.get("log_every", 1), h, lw=1.6)
    ax.set_xlabel("iteration"); ax.set_ylabel("data loss"); ax.grid(alpha=0.3, which="both")
    ax.set_title(run_dir.name[:60], fontsize=8)
    plt.tight_layout(); fig.savefig(run_dir / "loss_curve.png", dpi=80); plt.close(fig)


METRICS = ["albedo_mae", "roughness_mae", "metallic_mae", "train_recon_rmse", "val_relight_rmse"]


def _row(m):
    return dict(albedo_rmse=m["albedo_rmse"], albedo_mae=m.get("albedo_mae"),
                roughness_mae=m["roughness_err_mean"], metallic_mae=m["metallic_err_mean"],
                train_recon_rmse=m["recon_rmse"], val_relight_rmse=m.get("relight_rmse"),
                val_relight_mae=m.get("relight_mae"), final_loss=m["final_loss"],
                elapsed_s=m.get("elapsed_s"))


def plot_fullres(df, path):
    d = df[df["group"] == "fullres"].set_index("name")
    if not len(d):
        return
    order = [n for n in d.index]
    x = np.arange(len(order))
    fig, axes = plt.subplots(1, len(METRICS), figsize=(3.3 * len(METRICS), 4.2))
    for ax, k in zip(axes, METRICS):
        vals = d[k].reindex(order).values
        base = d.loc["baseline", k] if "baseline" in d.index else np.nan
        colors = ["#55A868" if (np.isfinite(base) and v < base) else "#C44E52" for v in vals]
        ax.bar(x, vals, color=colors)
        if np.isfinite(base):
            ax.axhline(base, color="0.3", ls="--", lw=1, label="baseline")
        ax.set_xticks(x); ax.set_xticklabels(order, rotation=60, ha="right", fontsize=6)
        ax.set_title(k, fontsize=9); ax.grid(axis="y", alpha=0.3); ax.legend(fontsize=6)
    fig.suptitle("full-res strategies (green = beats baseline)", fontsize=11)
    plt.tight_layout(); fig.savefig(path, dpi=90); plt.close(fig)


def plot_downsample(df, path):
    d = df[df["group"] == "downsample"].copy()
    if not len(d):
        return
    d = d.sort_values("resolution")
    fig, ax = plt.subplots(1, 3, figsize=(13, 4))
    for a, k in zip(ax, ["train_recon_rmse", "val_relight_rmse", "albedo_mae"]):
        a.plot(d["resolution"], d[k], "o-", lw=1.8)
        a.set_xlabel("resolution (px/side)"); a.set_title(k, fontsize=10)
        if (d["resolution"] > 0).all() and d["resolution"].nunique() > 1:
            a.set_xscale("log", base=2)
        a.grid(alpha=0.3, which="both")
    fig.suptitle("un-regularized baseline vs resolution", fontsize=11)
    plt.tight_layout(); fig.savefig(path, dpi=90); plt.close(fig)


# ───────────────────────── main ──────────────────────────────────────────────
def discover_scene(datasets_root: Path):
    for vd in sorted(p for p in datasets_root.iterdir() if p.is_dir()):
        ds = vd / "ct-ct_sh-frOn_env"
        if (ds / "config.json").exists():
            return ds
    raise SystemExit(f"no ct-ct_sh-frOn_env view under {datasets_root}; pass --scene")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scene", type=Path, default=None,
                   help="A ct-ct_sh-frOn_env dataset dir (default: first under --datasets_root).")
    p.add_argument("--datasets_root", type=Path,
                   default=Path("results/3dfront-batch/datasets"))
    p.add_argument("--out", type=Path, default=Path("results/improve_study"))
    p.add_argument("--n_train", type=int, default=100)
    p.add_argument("--n_val", type=int, default=28)
    p.add_argument("--n_iter", type=int, default=300)
    p.add_argument("--lbfgs_max_iter", type=int, default=40)
    p.add_argument("--warmup", type=int, default=60, help="Curriculum phase-1 steps.")
    p.add_argument("--reg_lambda", type=float, default=1e-5)
    p.add_argument("--downsamples", type=int, nargs="+", default=[2, 4, 8, 16])
    p.add_argument("--fullres_downsample", type=int, default=1,
                   help="Downsample for the 9 'full-res' strategy runs (1 = true 512^2). "
                        "Raise to run the whole study cheaper, or for a quick smoke test.")
    p.add_argument("--noise_std", type=float, default=0.1)
    p.add_argument("--pixel_frac", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--img_batch", type=int, default=8,
                   help="Images per grad-accum chunk (bounds 512^2 memory; safe under LBFGS).")
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--double", action="store_true", help="fp64 (slow at 512^2; default fp32).")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--force", action="store_true")
    p.add_argument("--wandb_mode", default="disabled",
                   choices=["disabled", "offline", "online"])
    p.add_argument("--wandb_project", default="3dfront-improve-study")
    args = p.parse_args()
    os.environ["WANDB_MODE"] = args.wandb_mode

    scene = args.scene or discover_scene(args.datasets_root)
    assert Path(scene).exists(), f"missing scene {scene}"
    runs = build_runs(args)
    runs_root = args.out / "runs"
    runs_root.mkdir(parents=True, exist_ok=True)
    print(f"scene: {scene}\n{len(runs)} run(s): "
          f"{sum(r['group']=='fullres' for r in runs)} full-res + "
          f"{sum(r['group']=='downsample' for r in runs)} downsampled  | device={args.device}")

    rows = []
    for i, r in enumerate(runs, 1):
        out_dir = runs_root / r["name"]
        cfg = {**base_cfg(args), **r["cfg"], "downsample": r["downsample"]}
        tag = f"[{i}/{len(runs)}] {r['name']} (ds{r['downsample']}, reg={r['reg']})"
        if (out_dir / "metrics.json").exists() and not args.force:
            m = json.loads((out_dir / "metrics.json").read_text())
            print(f"{tag}: cached")
        else:
            if out_dir.exists():
                shutil.rmtree(out_dir, ignore_errors=True)
            print(f"{tag}: running …", flush=True)
            t0 = time.perf_counter()
            m = decompose_scene(scene, out_dir, cfg_overrides=cfg, device=args.device,
                                gt_npy=True, wandb_project=args.wandb_project)
            m.setdefault("elapsed_s", time.perf_counter() - t0)
            m["log_every"] = args.log_every
            save_intrinsics_plot(out_dir, m, out_dir / "intrinsics.png")
            save_relight_plots(out_dir, m, scene, r["downsample"])
            save_loss_curve(out_dir, m)
            (out_dir / "report.json").write_text(json.dumps(
                {"name": r["name"], "group": r["group"], "downsample": r["downsample"],
                 "reg": r["reg"], **_row(m)}, indent=1))
        H = m.get("H", 0)
        rows.append(dict(name=r["name"], group=r["group"], reg=r["reg"],
                         downsample=r["downsample"], resolution=H, **_row(m)))
        # atomic incremental summary
        args.out.mkdir(parents=True, exist_ok=True)
        _tmp = args.out / "summary.csv.tmp"
        pd.DataFrame(rows).to_csv(_tmp, index=False)
        os.replace(_tmp, args.out / "summary.csv")

    df = pd.DataFrame(rows)
    plot_fullres(df, args.out / "compare_fullres.png")
    plot_downsample(df, args.out / "compare_downsample.png")

    fr = df[df["group"] == "fullres"]
    if len(fr):
        print("\n=== full-res strategies (sorted by val_relight_rmse) ===")
        cols = ["name", "reg"] + METRICS
        print(fr[cols].sort_values("val_relight_rmse").round(5).to_string(index=False))
    print(f"\nsummary -> {args.out/'summary.csv'}   plots -> {args.out}")


if __name__ == "__main__":
    main()
