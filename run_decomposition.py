#!/usr/bin/env python
"""
run_decomposition.py — standalone batch CT decomposition, for headless / VM runs
with wandb logging.

Consumes the SELF-CONTAINED dataset folder produced by
batch_dataset_decomposition.ipynb (its `datasets/` tree). Each dataset dir holds
everything decompose_scene needs — light_*.npy, sh_*.npy (GT lighting), the GT
albedo/normals/roughness/metallic .png, and config.json — so this script does no
re-rendering, no scene discovery, and has no dependency on the original
BlenderProc output or the (Windows-absolute) env-map paths. Just build the
datasets locally (notebook Part B up to the decompose loop), upload `datasets/`,
and run this.

Layout expected under --datasets_root:
  <view_key>/<dataset_name>/{light_*.npy, sh_*.npy, albedo|normals|roughness|metallic.png, config.json}
  dataset_name is 'ct-<shader>-frOn/frOff_<cond>' (pre-reduced; decompose at
  downsample=1) or 'blender_<cond>' (full-res; decompose at --downsample).
  The effective downsample is read from config ('prereduced_downsample' present
  => 1) and falls back to the ct-/blender_ name prefix.

Per run writes: metrics.json (decompose_scene), report.json, intrinsics.png, and
one relight panel per val light. A summary CSV is written incrementally. Scalar
metrics stream to wandb. Resumable: runs with an existing metrics.json are
skipped unless --force.

Example (GCE VM):
  WANDB_API_KEY=... python run_decomposition.py \
      --datasets_root /data/decomp/datasets --runs_root /data/decomp/runs \
      --sh_orders 2 3 --n_train 100 --n_val 28 \
      --wandb_project 3dfront-batch-decomposition --wandb_entity DLVC-intrinsics
"""
import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import matplotlib
matplotlib.use("Agg")            # headless
import matplotlib.pyplot as plt
from PIL import Image


def build_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--datasets_root", type=Path, default=Path("results/3dfront-batch/datasets"),
                   help="Root of the notebook's self-contained dataset tree.")
    p.add_argument("--runs_root", type=Path, default=Path("results/3dfront-batch/runs"),
                   help="Where per-run outputs + the summary CSV are written.")
    p.add_argument("--view_filter", nargs="*", default=None,
                   help="Only view_keys starting with one of these prefixes.")
    p.add_argument("--dataset_filter", nargs="*", default=None,
                   help="Only dataset dirs whose name starts with one of these (e.g. 'blender_' 'ct-ct_sh').")
    p.add_argument("--downsample", type=int, default=4,
                   help="Decompose downsample for datasets NOT already pre-reduced (blender).")
    p.add_argument("--n_train", type=int, default=100)
    p.add_argument("--n_val", type=int, default=28,
                   help="Last N lights held out for the relighting metric.")
    p.add_argument("--sh_orders", type=int, nargs="+", default=[2, 3], choices=[2, 3])
    p.add_argument("--decomp_shaders", nargs="+", default=["ct_sh"],
                   choices=["ct_sh", "ct_env", "ct_env_imp"])
    p.add_argument("--diffuse_fresnel", type=lambda s: s.lower() in ("1", "true", "on", "yes"),
                   default=True, help="Optimizer diffuse-Fresnel (default True = always ON).")
    p.add_argument("--n_iter", type=int, default=600)
    p.add_argument("--lbfgs_max_iter", type=int, default=20)
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--lambda_tv", type=float, default=1e-3)
    p.add_argument("--lambda_metallic_binarize", type=float, default=1e-3)
    p.add_argument("--double", type=lambda s: s.lower() in ("1", "true", "on", "yes"),
                   default=True, help="float64 (fp64 is slow on T4/L4; float32 is fine for ct_sh).")
    p.add_argument("--spec_samples", type=int, default=128)
    p.add_argument("--wandb_max_images", type=int, default=6,
                   help="Cap per-image wandb previews (scalars still use all images).")
    p.add_argument("--wandb_project", default="3dfront-batch-decomposition")
    p.add_argument("--wandb_entity", default="DLVC-intrinsics")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--force", action="store_true", help="Redo runs even if metrics.json exists.")
    p.add_argument("--no_plots", action="store_true", help="Skip PNG artifact plotting.")
    return p


args = build_parser().parse_args()
DEVICE = args.device
sys.path.insert(0, str(Path(__file__).resolve().parent))
from raw_optimizer.dfront_ct import decompose_scene   # noqa: E402


# ───────────────────────── plotting ──────────────────────────────────────────
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


# ───────────────────────── discovery of dataset dirs ─────────────────────────
def discover_datasets():
    items = []
    for view_dir in sorted(p for p in args.datasets_root.iterdir() if p.is_dir()):
        if args.view_filter and not any(view_dir.name.startswith(f) for f in args.view_filter):
            continue
        for ds_dir in sorted(p for p in view_dir.iterdir() if p.is_dir()):
            if args.dataset_filter and not any(ds_dir.name.startswith(f) for f in args.dataset_filter):
                continue
            cfg_p = ds_dir / "config.json"
            if not cfg_p.exists() or not any(ds_dir.glob("light_*.npy")):
                continue
            cfg = json.loads(cfg_p.read_text())
            # already pre-reduced (CT) -> decompose at 1; else (blender) -> --downsample
            eff_ds = 1 if cfg.get("prereduced_downsample", 0) and cfg["prereduced_downsample"] > 1 \
                else (1 if ds_dir.name.startswith("ct-") else args.downsample)
            items.append(dict(view=view_dir.name, name=ds_dir.name, dir=ds_dir, eff_ds=eff_ds))
    return items


def main():
    items = discover_datasets()
    if not items:
        raise SystemExit(f"No dataset dirs under {args.datasets_root}")
    n_runs = len(items) * len(args.sh_orders) * len(args.decomp_shaders)
    print(f"{len(items)} dataset(s) × {args.sh_orders} × {args.decomp_shaders} = {n_runs} run(s); device={DEVICE}")

    base_cfg = {
        "n_iter": args.n_iter, "lbfgs_max_iter": args.lbfgs_max_iter, "log_every": args.log_every,
        "lambda_tv": args.lambda_tv, "lambda_metallic_binarize": args.lambda_metallic_binarize,
        "tr_metallic": "sigmoid", "tr_roughness": "sigmoid", "tr_albedo": "sigmoid",
        "init_roughness_zero": True, "double": args.double,
        "wandb_max_images": args.wandb_max_images, "diffuse_fresnel": args.diffuse_fresnel,
    }

    rows, done = [], 0
    for it in items:
        for order in args.sh_orders:
            for shader in args.decomp_shaders:
                done += 1
                cfg = dict(base_cfg)
                cfg.update(shader={"ct_sh": "ct_sh", "ct_env": "ct_env", "ct_env_imp": "ct_env"}[shader],
                           sh_order=order, downsample=it["eff_ds"],
                           n_images=args.n_train + args.n_val, val_images=args.n_val)
                if shader == "ct_env_imp":
                    cfg["spec_importance"] = True
                    cfg["spec_samples"] = args.spec_samples
                tag = f"{it['view']}__{it['name']}__{shader}_SH{order}_N{args.n_train}"
                out_dir = args.runs_root / tag
                if (out_dir / "metrics.json").exists() and not args.force:
                    m = json.loads((out_dir / "metrics.json").read_text())
                    print(f"[{done}/{n_runs}] {tag}: cached")
                else:
                    # metrics.json is written only on successful completion, so an
                    # out_dir WITHOUT it is a partially-written (preempted) run —
                    # clear it for a clean redo on spot-restart.
                    if out_dir.exists():
                        shutil.rmtree(out_dir, ignore_errors=True)
                    print(f"[{done}/{n_runs}] {tag}: decomposing …", flush=True)
                    m = decompose_scene(it["dir"], out_dir, cfg_overrides=cfg, device=DEVICE,
                                        wandb_entity=args.wandb_entity, wandb_project=args.wandb_project)
                    if not args.no_plots:
                        save_intrinsics_plot(out_dir, m, out_dir / "intrinsics.png")
                        save_relight_plots(out_dir, m, it["dir"], it["eff_ds"])
                    (out_dir / "report.json").write_text(json.dumps({
                        "view": it["view"], "dataset": it["name"], "decomp_shader": shader,
                        "sh_order": order,
                        "albedo_rmse": m["albedo_rmse"], "albedo_mae": m.get("albedo_mae"),
                        "roughness_mae": m["roughness_err_mean"], "metallic_mae": m["metallic_err_mean"],
                        "train_recon_rmse": m["recon_rmse"], "train_recon_mae": m.get("recon_mae"),
                        "val_relight_rmse": m.get("relight_rmse"), "val_relight_mae": m.get("relight_mae"),
                        "final_loss": m["final_loss"], "n_train": m.get("n_train_images"),
                        "n_val": m.get("n_val_images")}, indent=1))
                rows.append(dict(
                    view=it["view"], dataset=it["name"], shader=shader, sh_order=order,
                    albedo_rmse=m["albedo_rmse"], albedo_mae=m.get("albedo_mae"),
                    roughness_mae=m["roughness_err_mean"], metallic_mae=m["metallic_err_mean"],
                    train_recon_rmse=m["recon_rmse"], val_relight_rmse=m.get("relight_rmse"),
                    val_relight_mae=m.get("relight_mae"), final_loss=m["final_loss"]))
                # atomic write: a preemption mid-write can't corrupt the summary
                args.runs_root.mkdir(parents=True, exist_ok=True)
                _tmp = args.runs_root / "decomposition_summary.csv.tmp"
                pd.DataFrame(rows).to_csv(_tmp, index=False)
                os.replace(_tmp, args.runs_root / "decomposition_summary.csv")

    df = pd.DataFrame(rows)
    if len(df):
        print("\n=== mean over views ===")
        print(df.groupby(["dataset", "sh_order"])[
            ["albedo_rmse", "roughness_mae", "metallic_mae", "train_recon_rmse", "val_relight_rmse"]
        ].mean().round(4).to_string())
        print(f"\nsummary -> {args.runs_root / 'decomposition_summary.csv'}")


if __name__ == "__main__":
    main()
