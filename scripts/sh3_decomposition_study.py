#!/usr/bin/env python
"""
scripts/sh3_decomposition_study.py — the SH3 decomposition study.

Decomposes the sh3 dataset's two ground-plane variants under two optimizer recipes,
records reconstruction / parameter / lighting / relighting error, saves the estimated
intrinsics + lighting for later re-rendering, and writes a per-run results.json plus an
incremental summary CSV. Fully resumable: a finished run is skipped on its results.json.

For each of the 67 views it runs:
  variants : blender_env_ground_shadow, ct-ct_sh-frOn_env_ground
  setups   : lbfgs, varpro   (see SETUPS below)
  => 67 x 2 x 2 = 268 runs.

Common to every run: SH3 lighting, full resolution (downsample 1), fp32, ct_sh shader,
16 training lights + 16 held-out reference lights for relighting, lambda_light_mono on.

Each run writes into <out_root>/<view>__<variant>__<setup>/:
  metrics.json          decompose_scene's own metrics (its completion marker)
  results.json          THIS study's marker: identifiers + all metrics + cfg + artifact map
  albedo_est.npy/png, albedo_scaled.npy, metallic_est.npy, roughness_est.npy,
  sh_coeffs_est.npy, gt/{albedo,metallic,roughness}.npy, reconstructions/recon_*.npy,
  relight/relit_*.npy, and (default) relight_sweep/{compare,est}.mp4|gif + panels.

Beyond decompose_scene's metrics this study adds three things it does not compute:
  * lighting_sh_rmse   — RMSE of the estimated per-image SH vs the GT SH (training lights),
                         raw and scale-invariant (the albedo<->light scale is unobservable).
  * roughness_rmse / metallic_rmse — RMSE (decompose reports only the MAEs).
  * faithful_relight_rmse — render EST intrinsics vs GT intrinsics through IDENTICAL
                         geometry under the 16 held-out reference SH lights and compare.
                         Unlike decompose's built-in relight_rmse (which compares to the
                         OBSERVED val image, and so is polluted by cast shadows the CT model
                         cannot reproduce for the blender variant), this isolates intrinsics
                         quality and is the fair cross-variant number.

Examples:
  # smoke: one view, one setup, no video
  python scripts/sh3_decomposition_study.py --views 3e038beb_v0 \
      --variants ct-ct_sh-frOn_env_ground --setups varpro --no_video --workers 1

  # full study (resumable)
  python scripts/sh3_decomposition_study.py --workers 4
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from idr.paths import SH3_STUDY_DIR, REPO_ROOT           # noqa: E402

# The two target variants and the two optimizer recipes this study sweeps.
VARIANTS = ["blender_env_ground_shadow", "ct-ct_sh-frOn_env_ground"]

SETUPS = ["lbfgs", "varpro"]

# The single shared LBFGS run (500 steps). It is computed ONCE per (view, variant): its
# saved estimates both stand alone as the "lbfgs" result AND warm-start the VarPro polish,
# so the 500 LBFGS steps are never repeated.
LBFGS_CFG = {"optimizer": "LBFGS", "n_iter": 500, "lbfgs_max_iter": 40}

# VarPro polish. Warm-started from the saved LBFGS maps via init_maps (no curriculum).
VARPRO_CFG = {
    "optimizer": "VARPRO", "n_iter": 200, "varpro_space": "natural",
    "varpro_lam_init": 1e-4, "varpro_lam_ceiling": 1e10, "varpro_n_inner_rho": 0,
}

# Fallback ONLY when no saved LBFGS run is available to warm-start from (e.g. `--setups
# varpro` with no prior lbfgs run on disk): fold the identical LBFGS(500) back in as a
# curriculum phase so a standalone VarPro run still works. Same LBFGS config as LBFGS_CFG,
# so warm and cold paths are equivalent.
VARPRO_CURRICULUM = [{"optimizer": "LBFGS", "n_iter": 500, "lbfgs_max_iter": 40}]

DEFAULT_DATASET_ROOT = REPO_ROOT / "local_datasets" / "26-08-09-datasets" / "sh3"

_WANDB_PROJECT = "sh3-study"
_WANDB_ENTITY = "DLVC-intrinsics"


# ───────────────────────── config ────────────────────────────────────────────
def build_base_cfg(lambda_light_mono, n_train, n_relight):
    """Shared cfg for every run; per-run setup dict is merged on top.

    Mirrors scripts/decompose_batch.py's base_cfg (sigmoid transforms, TV, diffuse
    Fresnel), plus the study-specific SH3 / full-res / fp32 / 16+16 split and the
    monochrome-light prior. n_images = n_train + n_relight so decompose_scene's
    'hold out the last val_images' leaves the first n_train as training lights.
    """
    return {
        "shader": "ct_sh",
        "sh_order": 3,
        "downsample": 1,
        "double": False,
        "use_npy": True,
        "gt_npy": True,
        "n_images": n_train + n_relight,
        "val_images": n_relight,
        "tr_metallic": "sigmoid", "tr_roughness": "sigmoid", "tr_albedo": "sigmoid",
        "init_roughness_zero": True,
        "diffuse_fresnel": True,
        "lambda_tv": 1e-5,
        "lambda_light_mono": lambda_light_mono,
        "log_every": 25,
    }


# ───────────────────────── discovery / tasks ─────────────────────────────────
def _scene_ready(scene_dir):
    return ((scene_dir / "config.json").exists()
            and any(scene_dir.glob("light_*.npy"))
            and any(scene_dir.glob("sh_*.npy")))


def discover_tasks(args):
    """One task per (view, variant); each runs its requested setups in order (lbfgs first,
    so VarPro can warm-start from it within the same worker)."""
    base_cfg = build_base_cfg(args.lambda_light_mono, args.n_train, args.n_relight)
    setups = [s for s in SETUPS if s in args.setups]        # canonical order: lbfgs, varpro
    tasks = []
    for view_dir in sorted(p for p in args.dataset_root.iterdir() if p.is_dir()):
        if args.views and view_dir.name not in args.views:
            continue
        for variant in args.variants:
            scene_dir = view_dir / variant
            if not (scene_dir.is_dir() and _scene_ready(scene_dir)):
                continue
            tasks.append(dict(
                view=view_dir.name, variant=variant, setups=setups,
                scene_dir=str(scene_dir), out_root=str(args.out_root), base_cfg=base_cfg,
                n_train=args.n_train, n_relight=args.n_relight,
                device=args.device, force=args.force,
                relight_video=args.relight_video,
                sweep_frames=args.sweep_frames, sweep_elev=args.sweep_elev,
                sweep_az=args.sweep_az))
    return tasks


# ───────────────────────── extra metrics ─────────────────────────────────────
def _stride(a, ds):
    return np.ascontiguousarray(a[::ds, ::ds]) if ds > 1 else a


def _pad_sh(s, n_sh):
    """Zero-pad an (m,3) SH array up the coefficient axis to (n_sh,3) (m<=n_sh)."""
    s = np.asarray(s, np.float32)
    if s.shape[0] < n_sh:
        s = np.concatenate([s, np.zeros((n_sh - s.shape[0], 3), np.float32)], axis=0)
    return s[:n_sh]


def compute_extra_metrics(scene_dir, out_dir, cfg, n_train, n_relight, device):
    """Lighting-SH RMSE, roughness/metallic RMSE, and faithful est-vs-GT relight.

    Reads only saved artifacts + the dataset GT, so it also back-fills onto a run whose
    decompose_scene finished earlier (metrics.json present) but that has no results.json.
    """
    from idr.data.scene_io import load_scene
    from idr.eval.relight_sweep import Relighter

    scene_dir, out_dir = Path(scene_dir), Path(out_dir)
    ds = int(cfg.get("downsample", 1) or 1)
    sh_order = int(cfg.get("sh_order", 2))
    n_sh = (sh_order + 1) ** 2
    diffuse_fresnel = bool(cfg.get("diffuse_fresnel", True))

    sc = load_scene(scene_dir, gt_npy=True)
    mask = _stride(sc["mask_np"], ds)                       # (H,W) bool
    gt_albedo = _stride(sc["albedo_np"], ds)
    gt_metallic = _stride(sc["metallic_np"], ds)
    gt_roughness = _stride(sc["roughness_np"], ds)
    gt_normals = _stride(sc["normals_np"], ds)
    gt_sh = sc["sh_coeffs"]                                  # list aligned to light_keys
    light_keys = sc["light_keys"]

    out = {}

    # ── lighting SH RMSE over the training lights ─────────────────────────────
    sh_est_p = out_dir / "sh_coeffs_est.npy"
    if sh_est_p.exists() and gt_sh is not None:
        sh_est = np.load(sh_est_p).astype(np.float32)       # (K, n_sh, 3), inv_scale-rescaled
        K = sh_est.shape[0]
        sh_gt = np.stack([_pad_sh(gt_sh[i], sh_est.shape[1]) for i in range(K)])  # (K,n_sh,3)
        d = sh_est - sh_gt
        out["lighting_sh_rmse"] = float(np.sqrt((d ** 2).mean()))
        # scale-invariant: fit one global scalar alpha (the residual albedo<->light scale)
        denom = float((sh_est ** 2).sum())
        alpha = float((sh_gt * sh_est).sum() / denom) if denom > 1e-12 else 1.0
        d2 = alpha * sh_est - sh_gt
        out["lighting_sh_rmse_scaleinv"] = float(np.sqrt((d2 ** 2).mean()))

    # ── roughness / metallic RMSE (decompose reports only the MAEs) ───────────
    m_gt = gt_metallic[..., 0][mask]
    r_gt = gt_roughness[..., 0][mask]
    if (out_dir / "metallic_est.npy").exists():
        m_est = np.load(out_dir / "metallic_est.npy").astype(np.float32)[..., 0][mask]
        out["metallic_rmse"] = float(np.sqrt(((m_est - m_gt) ** 2).mean()))
    if (out_dir / "roughness_est.npy").exists():
        r_est = np.load(out_dir / "roughness_est.npy").astype(np.float32)[..., 0][mask]
        out["roughness_rmse"] = float(np.sqrt(((r_est - r_gt) ** 2).mean()))

    # ── faithful relight: EST vs GT intrinsics under the 16 reference lights ──
    # Both are rendered through identical geometry, so they differ only in the intrinsic
    # maps — shadow-free, hence unaffected by the blender variant's baked shadows.
    ab_scaled_p = out_dir / "albedo_scaled.npy"
    ab_raw_p = out_dir / "albedo_est.npy"
    if (ab_raw_p.exists() and (out_dir / "metallic_est.npy").exists()
            and (out_dir / "roughness_est.npy").exists() and gt_sh is not None):
        met = np.load(out_dir / "metallic_est.npy").astype(np.float32)
        rou = np.load(out_dir / "roughness_est.npy").astype(np.float32)
        est_raw = dict(albedo=np.load(ab_raw_p).astype(np.float32), metallic=met, roughness=rou)
        est_scaled = dict(est_raw)
        if ab_scaled_p.exists():
            est_scaled = dict(albedo=np.load(ab_scaled_p).astype(np.float32),
                              metallic=met, roughness=rou)
        gt_intr = dict(albedo=gt_albedo, metallic=gt_metallic, roughness=gt_roughness)

        rel = Relighter(gt_normals, mask, device=device, dtype=torch.float32,
                        diffuse_fresnel=diffuse_fresnel,
                        hl_mode=str(cfg.get("hl_mode", "analytic")))
        ref_idx = range(n_train, n_train + n_relight)
        per_scaled, per_raw, ref_keys = [], [], []
        for i in ref_idx:
            if i >= len(gt_sh):
                break
            sh = _pad_sh(gt_sh[i], n_sh)
            g = rel.render(gt_intr, sh)[mask]
            e_s = rel.render(est_scaled, sh)[mask]
            e_r = rel.render(est_raw, sh)[mask]
            per_scaled.append(float(np.sqrt(((g - e_s) ** 2).mean())))
            per_raw.append(float(np.sqrt(((g - e_r) ** 2).mean())))
            ref_keys.append(light_keys[i] if i < len(light_keys) else f"ref_{i:03d}")
        del rel
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()
        if per_scaled:
            out["faithful_relight_rmse"] = float(np.mean(per_scaled))
            out["faithful_relight_rmse_raw"] = float(np.mean(per_raw))
            out["faithful_relight_rmse_per_light"] = per_scaled
            out["faithful_relight_keys"] = ref_keys
    return out


# ───────────────────────── one run (in a worker process) ─────────────────────
def _summary_row(task, results):
    m = results["metrics"]
    return dict(
        view=task["view"], variant=task["variant"], setup=task["setup"],
        recon_rmse=m.get("recon_rmse"), albedo_rmse=m.get("albedo_rmse"),
        roughness_rmse=m.get("roughness_rmse"), metallic_rmse=m.get("metallic_rmse"),
        lighting_sh_rmse=m.get("lighting_sh_rmse"),
        faithful_relight_rmse=m.get("faithful_relight_rmse"),
        builtin_relight_rmse=m.get("builtin_relight_rmse"),
        final_loss=m.get("final_loss"), elapsed_s=m.get("elapsed_s"),
        status=results.get("status"))


def _write_json_atomic(path, obj):
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1))
    os.replace(tmp, path)


def _build_results(task, m, extra, status):
    """Assemble the per-run results.json payload."""
    metrics = dict(
        recon_rmse=m.get("recon_rmse"), recon_mae=m.get("recon_mae"),
        albedo_rmse=m.get("albedo_rmse"), albedo_mae=m.get("albedo_mae"),
        roughness_mae=m.get("roughness_err_mean"), metallic_mae=m.get("metallic_err_mean"),
        roughness_gt=m.get("roughness_gt"), metallic_gt=m.get("metallic_gt"),
        roughness_est_mean=m.get("roughness_est_mean"), metallic_est_mean=m.get("metallic_est_mean"),
        builtin_relight_rmse=m.get("relight_rmse"), builtin_relight_mae=m.get("relight_mae"),
        final_loss=m.get("final_loss"), elapsed_s=m.get("elapsed_s"),
        albedo_scale=m.get("albedo_scale"),
        n_train_images=m.get("n_train_images"), n_val_images=m.get("n_val_images"),
        **extra)
    return dict(
        schema_version=1,
        view=task["view"], variant=task["variant"], setup=task["setup"],
        scene_dir=str(task["scene_dir"]), out_dir=str(task["out_dir"]),
        n_train=task["n_train"], n_relight=task["n_relight"],
        sh_order=int(task["cfg"].get("sh_order", 2)),
        cfg=task["cfg"], warm_start_from=task.get("warm_start_from"),
        artifacts=dict(
            albedo_est="albedo_est.npy", albedo_scaled="albedo_scaled.npy",
            metallic_est="metallic_est.npy", roughness_est="roughness_est.npy",
            sh_coeffs_est="sh_coeffs_est.npy", gt_dir="gt",
            reconstructions_dir="reconstructions", relight_dir="relight",
            relight_sweep_dir="relight_sweep"),
        metrics=metrics, status=status)


def _maybe_relight_video(task, out_dir, scene_dir):
    """Directional relight sweep + video (elev 45, azimuth orbit). Never raises."""
    if not task["relight_video"]:
        return
    try:
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        from idr.eval.relight_sweep import relight_sweep
        relight_sweep(out_dir, scene_dir, downsample=int(task["cfg"].get("downsample", 1) or 1),
                      az_from=task["sweep_az"][0], az_to=task["sweep_az"][1],
                      n_frames=task["sweep_frames"], elev=task["sweep_elev"],
                      diffuse_fresnel=bool(task["cfg"].get("diffuse_fresnel", True)),
                      device=task["device"], video=True, plots=True)
    except Exception as e:                       # a broken sweep must not fail a good run
        print(f"  [relight_video] {task['view']}/{task['variant']}/{task['setup']} "
              f"failed: {type(e).__name__}: {e}", flush=True)


def _run_is_done(out_dir, force):
    return (out_dir / "results.json").exists() and not force


def _has_estimates(out_dir):
    """True if a run dir holds the artifacts + metrics needed to warm-start from it."""
    return all((out_dir / f).exists() for f in (
        "metrics.json", "albedo_est.npy", "metallic_est.npy",
        "roughness_est.npy", "sh_coeffs_est.npy"))


def load_init_maps(lbfgs_dir):
    """Reconstruct the optimizer's natural-space warm-start maps from a saved LBFGS run.

    decompose_scene saves albedo/metallic/roughness in natural space already, but the SH is
    rescaled by inv_scale (=1/albedo_scale) on the way out; undo that so `sh` matches the raw
    optimized coefficients a curriculum phase would have handed off (sh = sh_est * scale).
    """
    lbfgs_dir = Path(lbfgs_dir)
    m = json.loads((lbfgs_dir / "metrics.json").read_text())
    scale = np.asarray(m["albedo_scale"], np.float32)              # (3,)
    sh_est = np.load(lbfgs_dir / "sh_coeffs_est.npy").astype(np.float32)   # (K, n_sh, 3)
    sh_nat = sh_est * scale[None, None, :]
    return dict(
        albedo=np.load(lbfgs_dir / "albedo_est.npy").astype(np.float32),
        metallic=np.load(lbfgs_dir / "metallic_est.npy").astype(np.float32),
        roughness=np.load(lbfgs_dir / "roughness_est.npy").astype(np.float32),
        sh=[sh_nat[k] for k in range(sh_nat.shape[0])])


def _run_setup(subtask, init_maps=None):
    """Decompose one (view, variant, setup); compute extra metrics; write results.json.

    Two-tier resume: results.json (this study's marker) -> skip entirely; metrics.json alone
    -> decompose already ran, only (re)compute the extra metrics + video; neither -> full run.
    init_maps, when given, is a natural-space warm-start (VarPro polishing from a saved LBFGS
    run) forwarded to decompose_scene via cfg_overrides['init_maps'].
    """
    from idr.pipelines.decompose import decompose_scene

    out_dir = Path(subtask["out_dir"]); scene_dir = Path(subtask["scene_dir"])
    row_base = dict(view=subtask["view"], variant=subtask["variant"], setup=subtask["setup"])
    results_p = out_dir / "results.json"
    metrics_p = out_dir / "metrics.json"

    if _run_is_done(out_dir, subtask["force"]):
        try:
            res = json.loads(results_p.read_text())
            row = _summary_row(subtask, res)
            row["status"] = "cached"                       # distinguish from a fresh run
            return row
        except Exception:
            pass                                          # unreadable -> recompute below

    try:
        if metrics_p.exists() and not subtask["force"]:
            m = json.loads(metrics_p.read_text())          # decompose done; back-fill extras
        else:
            if out_dir.exists():
                shutil.rmtree(out_dir, ignore_errors=True)  # partial/preempted -> wipe + redo
            cfg_overrides = dict(subtask["cfg"])
            if init_maps is not None:
                cfg_overrides["init_maps"] = init_maps      # warm-start; popped by decompose
            m = decompose_scene(scene_dir, out_dir, cfg_overrides=cfg_overrides,
                                device=subtask["device"],
                                wandb_project=_WANDB_PROJECT, wandb_entity=_WANDB_ENTITY)
        extra = compute_extra_metrics(scene_dir, out_dir, subtask["cfg"],
                                      subtask["n_train"], subtask["n_relight"], subtask["device"])
        _maybe_relight_video(subtask, out_dir, scene_dir)
        results = _build_results(subtask, m, extra, status="ok")
        _write_json_atomic(results_p, results)
        return _summary_row(subtask, results)
    except Exception as e:
        import traceback
        return {**row_base, "status": "error", "error": f"{type(e).__name__}: {e}",
                "traceback": traceback.format_exc()[-2000:]}


def run_one(task):
    """Run all requested setups for one (view, variant), chaining the warm-start.

    LBFGS(500) runs once; its saved maps warm-start the VarPro polish (no curriculum, so the
    500 LBFGS steps are not repeated). Returns a LIST of summary rows (one per setup).
    Picklable + self-contained so it runs in a spawned process.
    """
    os.environ.setdefault("WANDB_MODE", "disabled")
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    out_root = Path(task["out_root"])
    base = task["base_cfg"]

    def _subtask(setup, cfg, warm_from=None):
        return dict(view=task["view"], variant=task["variant"], setup=setup,
                    scene_dir=task["scene_dir"],
                    out_dir=str(out_root / f"{task['view']}__{task['variant']}__{setup}"),
                    cfg=cfg, warm_start_from=warm_from,
                    n_train=task["n_train"], n_relight=task["n_relight"],
                    device=task["device"], force=task["force"],
                    relight_video=task["relight_video"], sweep_frames=task["sweep_frames"],
                    sweep_elev=task["sweep_elev"], sweep_az=task["sweep_az"])

    lbfgs_dir = out_root / f"{task['view']}__{task['variant']}__lbfgs"
    rows = []

    if "lbfgs" in task["setups"]:
        rows.append(_run_setup(_subtask("lbfgs", {**base, **LBFGS_CFG})))

    if "varpro" in task["setups"]:
        vdir = out_root / f"{task['view']}__{task['variant']}__varpro"
        if _run_is_done(vdir, task["force"]):
            rows.append(_run_setup(_subtask("varpro", {**base, **VARPRO_CFG})))  # returns cached
        elif _has_estimates(lbfgs_dir):
            # Warm-start the VarPro polish from the saved LBFGS maps — no repeated LBFGS steps.
            init_maps = load_init_maps(lbfgs_dir)
            rows.append(_run_setup(_subtask("varpro", {**base, **VARPRO_CFG},
                                            warm_from=str(lbfgs_dir)), init_maps=init_maps))
        else:
            # No saved LBFGS to warm from: fall back to the self-contained curriculum.
            rows.append(_run_setup(_subtask("varpro",
                                            {**base, **VARPRO_CFG, "curriculum": VARPRO_CURRICULUM})))
    return rows


# ───────────────────────── driver ────────────────────────────────────────────
def build_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset_root", type=Path, default=DEFAULT_DATASET_ROOT)
    p.add_argument("--out_root", type=Path, default=SH3_STUDY_DIR)
    p.add_argument("--views", nargs="*", default=None,
                   help="Subset of view dir names (default: all present).")
    p.add_argument("--variants", nargs="*", default=VARIANTS, choices=VARIANTS)
    p.add_argument("--setups", nargs="*", default=list(SETUPS), choices=list(SETUPS))
    p.add_argument("--lambda_light_mono", type=float, default=1e-3)
    p.add_argument("--n_train", type=int, default=16)
    p.add_argument("--n_relight", type=int, default=16,
                   help="Held-out reference lights for the relighting metric.")
    p.add_argument("--workers", type=int, default=2,
                   help="Parallel worker processes (0 = min(#runs, CPU count)).")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--force", action="store_true", help="Redo runs even if results.json exists.")
    p.add_argument("--relight_video", dest="relight_video", action="store_true", default=True,
                   help="Write the relight sweep video per run (default on).")
    p.add_argument("--no_video", dest="relight_video", action="store_false",
                   help="Skip the relight sweep video.")
    p.add_argument("--sweep_frames", type=int, default=61)
    p.add_argument("--sweep_elev", type=float, default=45.0,
                   help="Relight light elevation, degrees.")
    p.add_argument("--sweep_az", type=float, nargs=2, default=[-45.0, 45.0],
                   metavar=("FROM", "TO"))
    return p


def main():
    args = build_parser().parse_args()
    tasks = discover_tasks(args)
    if not tasks:
        raise SystemExit(
            f"No runs under {args.dataset_root} for views={args.views} "
            f"variants={args.variants} setups={args.setups}")
    workers = args.workers or min(len(tasks), os.cpu_count() or 4)
    workers = max(1, min(workers, len(tasks)))
    n_views = len({t["view"] for t in tasks})
    n_runs = sum(len(t["setups"]) for t in tasks)
    print(f"{len(tasks)} task(s) / {n_runs} run(s): {n_views} view(s) x {args.variants} x "
          f"{args.setups}  | workers={workers} | device={args.device} | video={args.relight_video}")
    print(f"  lambda_light_mono={args.lambda_light_mono}  "
          f"n_train={args.n_train} n_relight={args.n_relight}  "
          f"(LBFGS runs once per view/variant; VarPro warm-starts from it)  -> {args.out_root}")

    args.out_root.mkdir(parents=True, exist_ok=True)
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
            for r in fut.result():                         # one task -> a row per setup
                rows.append(r)
                tmp = args.out_root / "sh3_study_summary.csv.tmp"
                pd.DataFrame(rows).to_csv(tmp, index=False)
                os.replace(tmp, args.out_root / "sh3_study_summary.csv")
                tag = f"{r['view']} {r['variant']} {r['setup']}"
                if r.get("status") == "error":
                    print(f"[task {done}/{len(tasks)}] {tag}  ERROR: {r.get('error')}", flush=True)
                else:
                    print(f"[task {done}/{len(tasks)}] {tag}  recon={r.get('recon_rmse'):.4f} "
                          f"albedo={r.get('albedo_rmse'):.4f} "
                          f"relight={_fmt(r.get('faithful_relight_rmse'))}  ({r.get('status')})",
                          flush=True)

    df = pd.DataFrame(rows)
    ok = df[df["status"].isin(["ok", "cached"])] if "status" in df else df
    if len(ok):
        cols = [c for c in ("recon_rmse", "albedo_rmse", "roughness_rmse", "metallic_rmse",
                            "lighting_sh_rmse", "faithful_relight_rmse") if c in ok]
        print("\n=== mean over views ===")
        print(ok.groupby(["variant", "setup"])[cols].mean().round(4).to_string())
    n_err = int((df["status"] == "error").sum()) if "status" in df else 0
    print(f"\nsummary -> {args.out_root / 'sh3_study_summary.csv'}"
          + (f"   ({n_err} run(s) errored)" if n_err else ""))


def _fmt(v):
    return f"{v:.4f}" if isinstance(v, (int, float)) else "n/a"


if __name__ == "__main__":
    main()
