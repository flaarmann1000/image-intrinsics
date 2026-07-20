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
  dataset_name is 'ct-<shader>-frOn/frOff_<cond>' or 'blender_<cond>'.
  The effective decompose downsample comes from config.json's
  'prereduced_downsample': >1 means the images were already shrunk at build time
  (decompose at 1), ==1 means they are full-res (apply --downsample). Only when
  the key is absent does it fall back to the ct-/blender_ name prefix. Startup
  prints the resulting resolution per dataset — check it before a long batch.

Per run writes: metrics.json (decompose_scene), report.json, intrinsics.png, and
one relight panel per val light. A summary CSV is written incrementally. Scalar
metrics stream to wandb. Resumable: runs with an existing metrics.json are
skipped unless --force.

With --relight_sweep / --relight_video it additionally relights the estimate under
a directional light swept across azimuth (see raw_optimizer/relight_sweep.py) and
writes <run>/relight_sweep/. That is a harder test than the built-in relight metric,
which re-lights under the val set's GT env SH — the same lighting family the
optimizer was fit on. The sweep only reads the saved *_est.npy, so it also
back-fills onto CACHED runs: adding the flag to an already-finished batch produces
the plots without redoing any optimization.

Example (GCE VM):
  WANDB_API_KEY=... python run_decomposition.py \
      --datasets_root /data/decomp/datasets --runs_root /data/decomp/runs \
      --sh_orders 2 3 --n_train 100 --n_val 28 \
      --wandb_project 3dfront-batch-decomposition-gc --wandb_entity DLVC-intrinsics

Add the relighting sweep + video to a batch that already ran (no re-optimization):
  python run_decomposition.py --runs_root /data/decomp/runs \
      --datasets_root /data/decomp/datasets --relight_video
"""
import argparse
import json
import os
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
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
    p.add_argument("--dataset_filter", nargs="*",
                   default='ct-ct_sh',
                   help="Only dataset dirs whose name starts with one of these (e.g. 'blender_' 'ct-ct_sh').")
    p.add_argument("--ground", default="both", choices=["both", "only", "exclude"],
                   help="Ground-plane condition to decompose. A dataset is 'ground' if its dir "
                        "name contains '_ground' (ct-*_env_ground, blender_env_ground_*). "
                        "both = every match (default), only = ground-plane sets, "
                        "exclude = no-ground sets. Orthogonal to --dataset_filter, which is a "
                        "PREFIX match and so cannot separate 'ct-ct_sh-frOn_env' from "
                        "'ct-ct_sh-frOn_env_ground'.")
    p.add_argument("--downsample", type=int, default=2,
                   help="Decompose downsample for datasets NOT already pre-reduced (blender).")
    p.add_argument("--n_train", type=int, default=100)
    p.add_argument("--n_val", type=int, default=28,
                   help="Last N lights held out for the relighting metric.")
    p.add_argument("--sh_orders", type=int, nargs="+", default=[2], choices=[2, 3])
    p.add_argument("--decomp_shaders", nargs="+", default=["ct_sh"],
                   choices=["ct_sh", "ct_env", "ct_env_imp"])
    p.add_argument("--diffuse_fresnel", type=lambda s: s.lower() in ("1", "true", "on", "yes"),
                   default=True, help="Optimizer diffuse-Fresnel (default True = always ON).")
    p.add_argument("--gt_npy", type=lambda s: s.lower() in ("1", "true", "on", "yes"),
                   default=True,
                   help="Read the GT intrinsic maps from the lossless float32 "
                        "{normals,albedo,roughness,metallic}.npy instead of the truncated "
                        "8/16-bit PNGs (default True). The PNGs impose a quantization floor "
                        "on albedo_rmse and feed quantized roughness/metallic/normals into "
                        "the forward model, since those are fixed inputs, not estimated.")
    p.add_argument("--use_npy", type=lambda s: s.lower() in ("1", "true", "on", "yes"),
                   default=True,
                   help="Read the observations from light_*.npy when the dataset variant is "
                        "png-based (default True). No effect on 'exr' datasets, which always "
                        "load .npy.")
    p.add_argument("--optimizer", default="LBFGS", choices=["LBFGS", "Adam", "LM"],
                   help="LM = Levenberg-Marquardt (ct_sh only). Needs far fewer iterations "
                        "than LBFGS, but each one builds + solves the normal equations.")
    p.add_argument("--lm_batch_size", type=int, default=0,
                   help="LM: images per step. 0 = full batch (deterministic, recommended).")
    p.add_argument("--lm_jacobian_max_num_rows", type=int, default=0,
                   help="LM: cap Jacobian rows held at once; slices the J^T J accumulation "
                        "down to O(P^2) memory. 0 = no slicing.")
    p.add_argument("--lm_damping", default="standard", choices=["standard", "fletcher"])
    p.add_argument("--lm_solver", default="cholesky", choices=["cholesky", "qr", "lstsq", "solve"])
    p.add_argument("--lm_jacobian_mode", default="auto", choices=["auto", "forward", "reverse"])
    p.add_argument("--lm_linear_solver", default="auto",
                   choices=["auto", "dense", "cg", "schur"],
                   help="How to solve the LM step. auto: dense below --lm_dense_max_params, "
                        "else schur if the regularizers are pixel-separable (no tv/sparse/white), "
                        "else cg. cg is matrix-free (O(P) memory) and scales to 512^2.")
    p.add_argument("--lm_dense_max_params", type=int, default=20000,
                   help="auto switches off the dense P x P solver above this P.")
    p.add_argument("--lm_image_chunk", type=int, default=8,
                   help="CG: images per jvp/vjp chunk (bounds the autograd graph).")
    p.add_argument("--lm_cg_tol", type=float, default=1e-4,
                   help="CG: relative residual tolerance for the inner solve.")
    p.add_argument("--lm_cg_maxiter", type=int, default=50)
    p.add_argument("--lm_schur_max_gb", type=float, default=4.0,
                   help="Schur: refuse if the cross block B would exceed this (then use cg).")
    p.add_argument("--lm_structured", action="store_true",
                   help="LM: build the normal equations from block-sparse per-pixel jacobians "
                        "(ct_sh only). Much faster than a dense Jacobian; same solution.")
    p.add_argument("--n_iter", type=int, default=500)
    p.add_argument("--lbfgs_max_iter", type=int, default=20)
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--lambda_tv", type=float, default=1e-5)
    p.add_argument("--lambda_metallic_binarize", type=float, default=0)
    p.add_argument("--double", type=lambda s: s.lower() in ("1", "true", "on", "yes"),
                   default=True, help="float64 (fp64 is slow on T4/L4; float32 is fine for ct_sh).")
    p.add_argument("--spec_samples", type=int, default=128)
    p.add_argument("--wandb_max_images", type=int, default=6,
                   help="Cap per-image wandb previews (scalars still use all images).")
    p.add_argument("--log_gt_recon_images", action="store_true",
                   help="Also log GT input images + recon error maps to wandb (final step only).")
    p.add_argument("--wandb_project", default="3dfront-batch-decomposition-gc")
    p.add_argument("--wandb_entity", default="DLVC-intrinsics")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--force", action="store_true", help="Redo runs even if metrics.json exists.")
    p.add_argument("--no_plots", action="store_true", help="Skip PNG artifact plotting.")
    # ── directional relighting sweep (opt-in; see raw_optimizer/relight_sweep.py) ──
    p.add_argument("--relight_sweep", action="store_true",
                   help="Also relight the estimate under a directional light swept across "
                        "azimuth and compare against GT intrinsics through identical "
                        "geometry. Harder than the built-in relight metric, which re-lights "
                        "under the val set's GT env SH — the same lighting family that was "
                        "fit. Writes <run>/relight_sweep/{sweep_panels,sweep_error}.png.")
    p.add_argument("--relight_video", action="store_true",
                   help="Also write the sweep as a video (GT | EST | error, plus EST alone). "
                        "Implies --relight_sweep. mp4 if imageio-ffmpeg is installed, else GIF.")
    p.add_argument("--sweep_az", type=float, nargs=2, default=[-45.0, 45.0],
                   metavar=("FROM", "TO"), help="Azimuth range in degrees.")
    p.add_argument("--sweep_frames", type=int, default=61)
    p.add_argument("--sweep_elev", type=float, default=30.0,
                   help="Light elevation above the horizon, degrees.")
    p.add_argument("--sweep_fps", type=int, default=20)
    p.add_argument("--sweep_no_ping_pong", action="store_true",
                   help="Play the sweep once instead of forward-then-back.")
    p.add_argument("--sweep_save_frames", action="store_true",
                   help="Also keep the individual PNG frames next to the video.")
    p.add_argument("--workers", type=int, default=2,
                   help="Parallel worker processes (0 = min(#runs, CPU count)). The 31^2 LBFGS "
                        "workload is CPU/launch-bound, so ~one per core saturates a spare GPU.")
    p.add_argument("--wandb_mode", default=None, choices=[None, "online", "offline", "disabled"],
                   help="Force a wandb mode for every run (default: env/online).")
    return p


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


# ───────────────────────── one run (executed in a worker process) ────────────
def _metrics_row(m):
    return dict(
        albedo_rmse=m["albedo_rmse"], albedo_mae=m.get("albedo_mae"),
        roughness_mae=m["roughness_err_mean"], metallic_mae=m["metallic_err_mean"],
        train_recon_rmse=m["recon_rmse"], train_recon_mae=m.get("recon_mae"),
        val_relight_rmse=m.get("relight_rmse"), val_relight_mae=m.get("relight_mae"),
        final_loss=m["final_loss"])


_SWEEP_ROW_KEYS = ("sweep_rmse_scaled", "sweep_rmse_raw", "sweep_mae_scaled",
                   "sweep_scale_ratio")


def maybe_relight_sweep(task, out_dir, ds_dir):
    """Run the directional sweep if requested. Returns summary-row fields.

    Deliberately independent of whether the decomposition just ran: it only needs the
    saved *_est.npy, so it also back-fills the sweep for CACHED runs. That means you can
    add --relight_sweep to an existing batch and get the plots without redoing any
    optimization. Never raises — a broken sweep must not fail an otherwise good run."""
    sw = task.get("sweep")
    if not sw:
        return {}
    prev_p = out_dir / "relight_sweep" / "sweep.json"
    if prev_p.exists() and not task["force"]:
        try:
            prev = json.loads(prev_p.read_text())
            # only reuse if it already has everything this invocation asks for
            if not (sw["video"] and "video" not in prev):
                return {**{k: prev[k] for k in _SWEEP_ROW_KEYS if k in prev},
                        "sweep_status": "cached"}
        except Exception:
            pass                                  # unreadable -> just recompute
    try:
        # The sweep runs in the same process as the decomposition that just finished, so
        # torch's caching allocator is still holding that run's blocks. Release them
        # first: otherwise peak = decomposition + sweep, which OOMs a shared GPU once
        # several workers hit their sweep at the same time.
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        from raw_optimizer.relight_sweep import relight_sweep
        r = relight_sweep(out_dir, ds_dir, downsample=task["eff_ds"], **sw)
        return {**{k: r[k] for k in _SWEEP_ROW_KEYS if k in r}, "sweep_status": "ok"}
    except Exception as e:
        return {"sweep_status": f"error: {type(e).__name__}: {e}"}


def run_one(task):
    """Decompose one (dataset, sh_order, shader) run. Self-contained + picklable
    so it runs in a spawned process. Returns a summary row (+ status/error)."""
    if task["wandb_mode"]:
        os.environ["WANDB_MODE"] = task["wandb_mode"]
    out_dir = Path(task["out_dir"]); ds_dir = Path(task["dir"])
    row_base = dict(view=task["view"], dataset=task["dataset"],
                    shader=task["shader"], sh_order=task["sh_order"])

    # metrics.json is written only on success: present => done; a dir without it
    # is a partial/preempted run to clear + redo.
    if (out_dir / "metrics.json").exists() and not task["force"]:
        m = json.loads((out_dir / "metrics.json").read_text())
        sweep = maybe_relight_sweep(task, out_dir, ds_dir)
        # a back-filled sweep must also land in report.json, not just sweep.json + the CSV
        rp = out_dir / "report.json"
        if sweep.get("sweep_status") == "ok" and rp.exists():
            try:
                rp.write_text(json.dumps({**json.loads(rp.read_text()), **sweep}, indent=1))
            except Exception:
                pass
        return {**row_base, **_metrics_row(m), "status": "cached", **sweep}
    if out_dir.exists():
        shutil.rmtree(out_dir, ignore_errors=True)

    try:
        m = decompose_scene(ds_dir, out_dir, cfg_overrides=task["cfg"], device=task["device"],
                            wandb_entity=task["wandb_entity"], wandb_project=task["wandb_project"])
        if not task["no_plots"]:
            save_intrinsics_plot(out_dir, m, out_dir / "intrinsics.png")
            save_relight_plots(out_dir, m, ds_dir, task["eff_ds"])
        sweep = maybe_relight_sweep(task, out_dir, ds_dir)
        (out_dir / "report.json").write_text(json.dumps({
            **row_base, "decomp_shader": task["shader"], **_metrics_row(m),
            "n_train": m.get("n_train_images"), "n_val": m.get("n_val_images"),
            **sweep}, indent=1))
        return {**row_base, **_metrics_row(m), "status": "ok", **sweep}
    except Exception as e:                       # keep the batch alive on one bad run
        import traceback
        return {**row_base, "status": "error", "error": f"{type(e).__name__}: {e}",
                "traceback": traceback.format_exc()[-2000:]}


# ───────────────────────── discovery of dataset dirs ─────────────────────────
def discover_datasets(args):
    items = []
    for view_dir in sorted(p for p in args.datasets_root.iterdir() if p.is_dir()):
        if args.view_filter and not any(view_dir.name.startswith(f) for f in args.view_filter):
            continue
        for ds_dir in sorted(p for p in view_dir.iterdir() if p.is_dir()):
            if args.dataset_filter and not any(ds_dir.name.startswith(f) for f in args.dataset_filter):
                continue
            is_ground = "_ground" in ds_dir.name
            if (args.ground == "only" and not is_ground) or \
               (args.ground == "exclude" and is_ground):
                continue
            cfg_p = ds_dir / "config.json"
            if not cfg_p.exists() or not any(ds_dir.glob("light_*.npy")):
                continue
            cfg = json.loads(cfg_p.read_text())
            # Effective decompose downsample. config.json is AUTHORITATIVE when it records
            # prereduced_downsample: >1 means the images were already shrunk at build time
            # (decompose at 1, they are small already), ==1 means they were written at full
            # resolution (so --downsample must still be applied). Only fall back to the
            # ct-/blender_ name prefix when the key is absent — a 'ct-' name does NOT imply
            # pre-reduction, and assuming it did silently ignored --downsample and ran
            # 512^2 decompositions.
            pre = cfg.get("prereduced_downsample")
            if pre is not None:
                eff_ds = 1 if pre > 1 else args.downsample
            else:
                eff_ds = 1 if ds_dir.name.startswith("ct-") else args.downsample
            items.append(dict(view=view_dir.name, name=ds_dir.name, dir=ds_dir, eff_ds=eff_ds))
    return items


def build_tasks(args, items):
    base_cfg = {
        "n_iter": args.n_iter, "lbfgs_max_iter": args.lbfgs_max_iter, "log_every": args.log_every,
        "lambda_tv": args.lambda_tv, "lambda_metallic_binarize": args.lambda_metallic_binarize,
        "tr_metallic": "sigmoid", "tr_roughness": "sigmoid", "tr_albedo": "sigmoid",
        "init_roughness_zero": True, "double": args.double,
        "wandb_max_images": args.wandb_max_images, "diffuse_fresnel": args.diffuse_fresnel,
        # decompose_scene pops these out of cfg_overrides and forwards them to load_scene
        "gt_npy": args.gt_npy, "use_npy": args.use_npy,
        "log_gt_recon_images": args.log_gt_recon_images,
        "optimizer": args.optimizer,
        "lm_batch_size": args.lm_batch_size, "lm_damping": args.lm_damping,
        "lm_solver": args.lm_solver, "lm_jacobian_mode": args.lm_jacobian_mode,
        "lm_jacobian_max_num_rows": args.lm_jacobian_max_num_rows,
        "lm_structured": args.lm_structured,
        "lm_linear_solver": args.lm_linear_solver,
        "lm_dense_max_params": args.lm_dense_max_params,
        "lm_image_chunk": args.lm_image_chunk, "lm_cg_tol": args.lm_cg_tol,
        "lm_cg_maxiter": args.lm_cg_maxiter, "lm_schur_max_gb": args.lm_schur_max_gb,
    }
    # --relight_video implies the sweep; None disables it entirely (checked in run_one)
    sweep = None
    if args.relight_sweep or args.relight_video:
        sweep = dict(az_from=args.sweep_az[0], az_to=args.sweep_az[1],
                     n_frames=args.sweep_frames, elev=args.sweep_elev,
                     diffuse_fresnel=args.diffuse_fresnel, device=args.device,
                     video=args.relight_video, plots=True, fps=args.sweep_fps,
                     ping_pong=not args.sweep_no_ping_pong,
                     save_frames=args.sweep_save_frames)
    tasks = []
    for it in items:
        for order in args.sh_orders:
            for shader in args.decomp_shaders:
                cfg = dict(base_cfg)
                cfg.update(shader={"ct_sh": "ct_sh", "ct_env": "ct_env", "ct_env_imp": "ct_env"}[shader],
                           sh_order=order, downsample=it["eff_ds"],
                           n_images=args.n_train + args.n_val, val_images=args.n_val)
                if shader == "ct_env_imp":
                    cfg["spec_importance"] = True
                    cfg["spec_samples"] = args.spec_samples
                tag = f"{it['view']}__{it['name']}__{shader}_SH{order}_N{args.n_train}"
                tasks.append(dict(
                    view=it["view"], dataset=it["name"], dir=str(it["dir"]), eff_ds=it["eff_ds"],
                    shader=shader, sh_order=order, cfg=cfg,
                    out_dir=str(args.runs_root / tag), device=args.device,
                    wandb_entity=args.wandb_entity, wandb_project=args.wandb_project,
                    wandb_mode=args.wandb_mode, force=args.force, no_plots=args.no_plots,
                    sweep=sweep))
    return tasks


def main():
    args = build_parser().parse_args()
    items = discover_datasets(args)
    if not items:
        raise SystemExit(
            f"No dataset dirs under {args.datasets_root} matching "
            f"dataset_filter={args.dataset_filter} view_filter={args.view_filter} "
            f"ground={args.ground}")
    tasks = build_tasks(args, items)
    workers = args.workers or min(len(tasks), os.cpu_count() or 4)
    workers = max(1, min(workers, len(tasks)))
    n_g = sum("_ground" in it["name"] for it in items)
    print(f"{len(items)} dataset(s) ({n_g} ground / {len(items)-n_g} no-ground, "
          f"--ground {args.ground}) × {args.sh_orders} × {args.decomp_shaders} "
          f"= {len(tasks)} run(s)  | workers={workers}  | device={args.device}")
    # Print the resolution each dataset will actually be decomposed at. eff_ds comes from
    # config.json, not --downsample, so a silent mismatch here is what turns a 256^2 run
    # into a 512^2 one and OOMs the GPU.
    n_png_gt = 0
    for it in items:
        try:
            import numpy as _np
            h = _np.load(next(iter(sorted(Path(it["dir"]).glob("light_*.npy")))),
                         mmap_mode="r").shape[0]
            eff = f"{h // it['eff_ds']}^2"
        except Exception:
            eff = "?"
        # load_scene falls back to the quantized PNGs silently if ANY of the four GT .npy
        # is missing, so surface which source each dataset will actually use.
        missing = [m for m in ("normals", "albedo", "roughness", "metallic")
                   if not (Path(it["dir"]) / f"{m}.npy").exists()]
        gt_src = "npy" if (args.gt_npy and not missing) else "PNG (quantized)"
        n_png_gt += gt_src != "npy"
        print(f"  {it['view']}/{it['name']}: downsample={it['eff_ds']} -> {eff}, GT={gt_src}"
              + (f"  [missing {','.join(missing)}]" if args.gt_npy and missing else ""))
    if n_png_gt:
        print(f"  ! {n_png_gt} dataset(s) will use quantized PNG GT maps — this puts a floor "
              f"on albedo_rmse and feeds quantized roughness/metallic into the forward model.")
    if tasks[0]["sweep"]:
        s = tasks[0]["sweep"]
        print(f"relight sweep: {s['az_from']:+.0f}..{s['az_to']:+.0f} deg azimuth, "
              f"elev {s['elev']:.0f} deg, {s['n_frames']} frames"
              + (f", video @ {s['fps']} fps" if s["video"] else ", plots only"))

    args.runs_root.mkdir(parents=True, exist_ok=True)
    try:
        import multiprocessing as mp
        ctx = mp.get_context("spawn")
    except Exception:
        ctx = None

    rows, done = [], 0
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
        futs = {ex.submit(run_one, t): t for t in tasks}
        for fut in as_completed(futs):
            done += 1
            r = fut.result()
            rows.append(r)
            # atomic write: a preemption mid-write can't corrupt the summary
            _tmp = args.runs_root / "decomposition_summary.csv.tmp"
            pd.DataFrame(rows).to_csv(_tmp, index=False)
            os.replace(_tmp, args.runs_root / "decomposition_summary.csv")
            if r.get("status") == "error":
                print(f"[{done}/{len(tasks)}] {r['view']} {r['dataset']} {r['shader']} SH{r['sh_order']}"
                      f"  ERROR: {r.get('error')}", flush=True)
            else:
                sw = ""
                if r.get("sweep_status", "").startswith("error"):
                    sw = f"  sweep FAILED ({r['sweep_status'][7:]})"
                elif r.get("sweep_rmse_scaled") is not None:
                    sw = f"  sweep={r['sweep_rmse_scaled']:.4f}"
                print(f"[{done}/{len(tasks)}] {r['view']} {r['dataset']} {r['shader']} SH{r['sh_order']}"
                      f"  alb_rmse={r.get('albedo_rmse'):.4f}  recon={r.get('train_recon_rmse'):.4f}"
                      f"{sw}  ({r.get('status')})", flush=True)

    df = pd.DataFrame(rows)
    ok = df[df["status"].isin(["ok", "cached"])] if "status" in df else df
    if len(ok):
        cols = ["albedo_rmse", "roughness_mae", "metallic_mae", "train_recon_rmse",
                "val_relight_rmse"]
        cols += [c for c in ("sweep_rmse_scaled", "sweep_rmse_raw") if c in ok]
        print("\n=== mean over views ===")
        print(ok.groupby(["dataset", "sh_order"])[cols].mean().round(4).to_string())
    if "sweep_status" in df:
        n_sw_err = int(df["sweep_status"].astype(str).str.startswith("error").sum())
        if n_sw_err:
            print(f"\n! {n_sw_err} relight sweep(s) failed; the decompositions themselves "
                  f"are unaffected. First: "
                  f"{df.loc[df['sweep_status'].astype(str).str.startswith('error'), 'sweep_status'].iloc[0]}")
    n_err = int((df["status"] == "error").sum()) if "status" in df else 0
    print(f"\nsummary -> {args.runs_root / 'decomposition_summary.csv'}"
          + (f"   ({n_err} run(s) errored)" if n_err else ""))


if __name__ == "__main__":
    main()
