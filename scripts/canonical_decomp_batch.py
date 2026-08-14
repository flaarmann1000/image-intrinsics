#!/usr/bin/env python
"""
scripts/canonical_decomp_batch.py — batch CT decomposition of the canonical datasets under a
set of config ABLATIONS, writing per-(scene,config) metrics + saved estimates to results/.

Datasets (choose with --datasets; each carries its own default downsample):
  point     local_datasets/26-08-09-datasets/sh3/<view>/point_shadow   real GT intrinsics, NO GT lighting
  infinite  local_datasets/canonical/INFINITE/<scene>                  real GT intrinsics + GT SH lighting
  mit       local_datasets/canonical/MIT-train/<scene>                 Marigold intrinsics, NO GT lighting

Configs — a BASE and one-change ABLATIONS off it (choose with --configs):
  base                LBFGS 500 (lbfgs_max_iter 40), SH3, NO transforms, lambda_box=1e-1
  sigmoid             + sigmoid transforms on albedo/metallic/roughness
  sh2                 + SH order 2
  light_mono          + lambda_light_mono=1e-3
  varpro              base LBFGS  ->  VarPro polish (n_iter 200, lam_init 1e-4, ceiling 1e10,
                      n_inner_rho 0), WARM-STARTED from the base run (no repeated LBFGS)
  tv1e-5 / tv1e-4     + lambda_tv
  metallic_binarize   + lambda_metallic_binarize=1e-4
  sparse              + lambda_sparse=1e-4

Per run (results.json) records: recon_rmse, albedo_rmse, roughness_rmse, metallic_rmse,
lighting_sh_rmse and relighting_rmse — the last two ONLY where GT SH lighting exists
(infinite). relighting_rmse is the HELD-OUT TEST-SET relight: the last --n_val (default 16)
frames are held out, the estimate is relit under their GT lighting and compared to the
observations (decompose_scene's own val relight). Datasets without GT lighting (point, mit)
get no relighting_rmse. The full decomposition (albedo/metallic/roughness/sh estimates) is
saved by decompose_scene for the separate render script (canonical_relight_render.py) to make
the comparison panels + videos.

Resumable (per-run results.json marker) with a clear [done/total] progress line per config.

Usage:
  python scripts/canonical_decomp_batch.py --datasets infinite mit --configs base sigmoid varpro
  python scripts/canonical_decomp_batch.py                       # all datasets, all configs
"""
import argparse
import json
import os
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch

from idr.paths import REPO_ROOT, RESULTS_DIR

# ── datasets ──────────────────────────────────────────────────────────────────
_SH3_ROOT = REPO_ROOT / "local_datasets" / "26-08-09-datasets" / "sh3"
_CANON = REPO_ROOT / "local_datasets" / "canonical"
DATASETS = {
    "ct_sh_env":       dict(disc="view_variant", root=_SH3_ROOT, variant="ct-ct_sh-frOn_env",
                            ds=1, has_gt_lighting=True, use_npy=True),   # CT-SH inverse crime + GT SH
    "point":           dict(disc="view_variant", root=_SH3_ROOT, variant="point_shadow",
                            ds=1, has_gt_lighting=False, use_npy=True),
    "point_no_shadow": dict(disc="view_variant", root=_SH3_ROOT, variant="point_no_shadow",
                            ds=1, has_gt_lighting=False, use_npy=True),
    "infinite":        dict(disc="flat", root=_CANON / "INFINITE",
                            ds=2, has_gt_lighting=True, use_npy=True),
    "infinite-fullRes":dict(disc="flat", root=_CANON / "INFINITE",
                            ds=1, has_gt_lighting=True, use_npy=True),
    "mit":             dict(disc="flat", root=_CANON / "MIT-train",
                            ds=4, has_gt_lighting=False, use_npy=False),
}
# Default set to run (point_no_shadow is available but opt-in via --datasets).
# DEFAULT_DATASETS = ["ct_sh_env", "point", "point_no_shadow", "infinite", "mit"]
DEFAULT_DATASETS = ["infinite"]

# ── configs (BASE + one-change ablations) ─────────────────────────────────────
BASE = dict(shader="ct_sh", optimizer="LBFGS", n_iter=500, lbfgs_max_iter=40, sh_order=3,
            tr_albedo="none", tr_metallic="none", tr_roughness="none", lambda_box=1e-1)
VARPRO_POLISH = dict(optimizer="VARPRO", n_iter=200, varpro_space="natural",
                     varpro_lam_init=1e-4, varpro_lam_ceiling=1e10, varpro_n_inner_rho=0)
CONFIGS = {
    "base":              {},
    "sigmoid":           dict(tr_albedo="sigmoid", tr_metallic="sigmoid", tr_roughness="sigmoid"),
    "sh2":               dict(sh_order=2),
    "light_mono":        dict(lambda_light_mono=1e-3),
    #"varpro":            "VARPRO",                         # special: warm-start from base
    "tv1e-5":            dict(lambda_tv=1e-5),
    "tv1e-4":            dict(lambda_tv=1e-4),
    "metallic_binarize": dict(lambda_metallic_binarize=1e-4),
    "metallic_l1_1e-3":  dict(lambda_metallic_l1=1e-3),
    "metallic_l1_1e-2":  dict(lambda_metallic_l1=1e-2),
    "metallic_l1_1e-2_tv1e-3":         dict(lambda_metallic_l1=1e-2, lambda_tv=1e-3),
    "metallic_l1_1e-2_tv1e-3_light_mono":         dict(lambda_metallic_l1=1e-2, lambda_tv=1e-3, lambda_light_mono=1e-3),
    "metallic_l1_1e-2_tv1e-3_light_mono_sh2":         dict(lambda_metallic_l1=1e-2, lambda_tv=1e-3, lambda_light_mono=1e-3, sh_order=2),
    "metallic_l1_1e-2_tv1e-3_sigmoid": dict(lambda_metallic_l1=1e-2, lambda_tv=1e-3,
                                            tr_albedo="sigmoid", tr_metallic="sigmoid",
                                            tr_roughness="sigmoid"),
    "sparse":            dict(lambda_sparse=1e-4),
}

_WANDB_PROJECT, _WANDB_ENTITY = "canonical-decomp", "DLVC-intrinsics"


# ── discovery ─────────────────────────────────────────────────────────────────
def _scene_ready(d):
    return (d / "config.json").exists() and (
        any(d.glob("light_*.npy")) or any(d.glob("light_*.png")))


def discover(ds_name, spec, views_filter):
    items = []
    root = spec["root"]
    if not root.exists():
        return items
    if spec["disc"] == "flat":
        for sd in sorted(p for p in root.iterdir() if p.is_dir()):
            if _scene_ready(sd) and (not views_filter or sd.name in views_filter):
                items.append((sd.name, sd))
    else:                                                  # view_variant (point)
        for vd in sorted(p for p in root.iterdir() if p.is_dir()):
            sd = vd / spec["variant"]
            if sd.is_dir() and _scene_ready(sd) and (not views_filter or vd.name in views_filter):
                items.append((vd.name, sd))
    return items


# ── per-run cfg ───────────────────────────────────────────────────────────────
def build_cfg(config, spec, args):
    common = dict(double=False, diffuse_fresnel=True, gt_npy=True, use_npy=spec["use_npy"],
                  downsample=(args.downsample or spec["ds"]), log_every=100)
    if config == "varpro":
        return {**common, **BASE, **VARPRO_POLISH}
    return {**common, **BASE, **CONFIGS[config]}


def _count_lights(scene_dir):
    n = len(list(Path(scene_dir).glob("light_*.npy")))
    return n or len(list(Path(scene_dir).glob("light_*.png")))


def resolve_split(scene_dir, spec, args):
    """(n_images, val_images) for a scene. Datasets WITH GT lighting hold out the last
    `--n_val` frames as a relighting TEST set (capped at half the scene so training survives);
    the rest, up to `--n_images`, are training. Datasets without GT lighting hold out nothing
    (relighting error is undefined there)."""
    avail = _count_lights(scene_dir)
    cap = args.n_images or avail
    if spec["has_gt_lighting"] and avail >= 2:
        val = min(args.n_val, avail // 2)
        n_train = min(cap, avail - val)
        return n_train + val, val
    return min(cap, avail), 0


# ── metrics ───────────────────────────────────────────────────────────────────
def _pad_sh(s, n):
    s = np.asarray(s, np.float32)
    if s.shape[0] < n:
        s = np.concatenate([s, np.zeros((n - s.shape[0], 3), np.float32)], 0)
    return s[:n]


def compute_metrics(scene_dir, out_dir, cfg, spec, device):
    """Intrinsic-map metrics computed from the saved estimates. The RELIGHTING error is NOT
    here — it is the held-out test-set relight that decompose_scene computes (its relight_rmse,
    from cfg['val_images']) and is surfaced in run_scene."""
    from idr.data.scene_io import load_scene

    ds = int(cfg.get("downsample", 1) or 1)
    sc = load_scene(Path(scene_dir), gt_npy=True)
    mask = sc["mask_np"][::ds, ::ds] if ds > 1 else sc["mask_np"]
    out = {}

    # roughness / metallic RMSE (masked); albedo_rmse comes from decompose metrics.json
    for key in ("roughness", "metallic"):
        est_p = Path(out_dir) / f"{key}_est.npy"
        if est_p.exists():
            est = np.load(est_p).astype(np.float32)[..., 0][mask]
            gt = (sc[f"{key}_np"][::ds, ::ds] if ds > 1 else sc[f"{key}_np"])[..., 0][mask]
            out[f"{key}_rmse"] = float(np.sqrt(((est - gt) ** 2).mean()))

    # lighting SH RMSE — only where GT SH exists (infinite). Compares est SH for the TRAINING
    # lights (sh_coeffs_est has one per training image) against the GT SH of those same lights.
    sh_est_p = Path(out_dir) / "sh_coeffs_est.npy"
    if spec["has_gt_lighting"] and sh_est_p.exists() and sc.get("sh_coeffs") is not None:
        sh_est = np.load(sh_est_p).astype(np.float32)
        K = sh_est.shape[0]
        sh_gt = np.stack([_pad_sh(sc["sh_coeffs"][i], sh_est.shape[1]) for i in range(K)])
        out["lighting_sh_rmse"] = float(np.sqrt(((sh_est - sh_gt) ** 2).mean()))
    return out


# ── init-maps warm-start (varpro from base) ───────────────────────────────────
def _has_estimates(d):
    d = Path(d)
    return all((d / f).exists() for f in ("metrics.json", "albedo_est.npy", "metallic_est.npy",
                                          "roughness_est.npy", "sh_coeffs_est.npy"))


def load_init_maps(base_dir):
    base_dir = Path(base_dir)
    m = json.loads((base_dir / "metrics.json").read_text())
    scale = np.asarray(m["albedo_scale"], np.float32)
    sh = np.load(base_dir / "sh_coeffs_est.npy").astype(np.float32) * scale[None, None, :]
    return dict(albedo=np.load(base_dir / "albedo_est.npy").astype(np.float32),
                metallic=np.load(base_dir / "metallic_est.npy").astype(np.float32),
                roughness=np.load(base_dir / "roughness_est.npy").astype(np.float32),
                sh=[sh[k] for k in range(sh.shape[0])])


# ── one (scene, config) ───────────────────────────────────────────────────────
def _row(ds_name, scene, config, m, status):
    return dict(dataset=ds_name, scene=scene, config=config,
                recon_rmse=m.get("recon_rmse"), albedo_rmse=m.get("albedo_rmse"),
                roughness_rmse=m.get("roughness_rmse"), metallic_rmse=m.get("metallic_rmse"),
                lighting_sh_rmse=m.get("lighting_sh_rmse"),
                relighting_rmse=m.get("relighting_rmse"),
                elapsed_s=m.get("elapsed_s"), status=status)


def _write_atomic(p, obj):
    tmp = Path(str(p) + ".tmp"); tmp.write_text(json.dumps(obj, indent=1)); os.replace(tmp, p)


def run_scene(task):
    """Run every requested config for one (dataset, scene); base first so varpro warm-starts
    from it. Returns a list of summary rows (one per config)."""
    os.environ.setdefault("WANDB_MODE", "disabled")
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from idr.pipelines.decompose import decompose_scene

    spec = DATASETS[task["dataset"]]
    scene_dir = Path(task["scene_dir"]); base_out = Path(task["out_root"]) / task["dataset"] / task["scene"]
    device, force = task["device"], task["force"]
    # per-scene train/val split (val = relighting test set, lighting datasets only)
    n_images_sc, val_images_sc = resolve_split(scene_dir, spec, task["args_ns"])
    # base first, then the rest (varpro needs base's saved maps)
    order = [c for c in ["base"] if c in task["configs"]] + \
            [c for c in task["configs"] if c != "base"]
    rows = []
    for config in order:
        out_dir = base_out / config
        results_p = out_dir / "results.json"
        if results_p.exists() and not force:
            try:
                res = json.loads(results_p.read_text())
                rows.append({**_row(task["dataset"], task["scene"], config, res["metrics"], "cached")})
                continue
            except Exception:
                pass
        cfg = build_cfg(config, spec, task["args_ns"])
        cfg["n_images"], cfg["val_images"] = n_images_sc, val_images_sc
        init_maps = None
        try:
            if config == "varpro":
                base_dir = base_out / "base"
                if _has_estimates(base_dir):
                    init_maps = load_init_maps(base_dir)
                else:                                     # no base to warm from -> fold LBFGS back in
                    cfg = {**cfg, "curriculum": [{**BASE, "optimizer": "LBFGS", "n_iter": 500,
                                                  "lbfgs_max_iter": 40}]}
            if (out_dir / "metrics.json").exists() and not force:
                m = json.loads((out_dir / "metrics.json").read_text())
            else:
                if out_dir.exists():
                    shutil.rmtree(out_dir, ignore_errors=True)
                overrides = dict(cfg)
                if init_maps is not None:
                    overrides["init_maps"] = init_maps
                m = decompose_scene(scene_dir, out_dir, cfg_overrides=overrides, device=device,
                                    wandb_project=_WANDB_PROJECT, wandb_entity=_WANDB_ENTITY)
            extra = compute_metrics(scene_dir, out_dir, cfg, spec, device)
            metrics = {**{k: m.get(k) for k in ("recon_rmse", "recon_mae", "albedo_rmse",
                        "albedo_mae", "final_loss", "elapsed_s", "albedo_scale",
                        "n_train_images", "n_val_images")}, **extra,
                       # held-out test-set relighting error (decompose_scene's own val relight;
                       # present only where GT lighting held out a test set — i.e. infinite)
                       "relighting_rmse": m.get("relight_rmse"),
                       "relighting_mae": m.get("relight_mae"),
                       "relighting_rmse_per_light": m.get("relight_rmse_per_light")}
            res = dict(dataset=task["dataset"], scene=task["scene"], config=config,
                       scene_dir=str(scene_dir), out_dir=str(out_dir),
                       has_gt_lighting=spec["has_gt_lighting"], cfg=cfg, metrics=metrics,
                       status="ok")
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
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS, choices=list(DATASETS))
    p.add_argument("--configs", nargs="+", default=list(CONFIGS), choices=list(CONFIGS))
    p.add_argument("--views", nargs="*", default=None, help="restrict to these scene/view names")
    p.add_argument("--out_root", type=Path, default=RESULTS_DIR / "canonical_ablation")
    p.add_argument("--downsample", type=int, default=0, help="override every dataset's default ds")
    p.add_argument("--n_images", type=int, default=16, help="cap TRAINING lights per scene (0 = all)")
    p.add_argument("--n_val", type=int, default=16,
                   help="held-out relighting test-set size (datasets with GT lighting only; "
                        "capped at half the scene's lights so training survives)")
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--force", action="store_true")
    return p


def main():
    args = build_parser().parse_args()
    if args.n_images == 0:
        args.n_images = None
    # ensure base is included when varpro is requested (varpro warm-starts from it)
    configs = list(args.configs)
    if "varpro" in configs and "base" not in configs:
        configs = ["base"] + configs
        print("note: added 'base' (varpro warm-starts from it)")

    tasks = []
    for ds_name in args.datasets:
        items = discover(ds_name, DATASETS[ds_name], set(args.views) if args.views else None)
        for scene, scene_dir in items:
            tasks.append(dict(dataset=ds_name, scene=scene, scene_dir=str(scene_dir),
                              configs=configs, out_root=str(args.out_root),
                              device=args.device, force=args.force, args_ns=args))
    if not tasks:
        raise SystemExit(f"no scenes for datasets={args.datasets} views={args.views}")
    total_runs = sum(len(t["configs"]) for t in tasks)
    per_ds = {d: sum(t["dataset"] == d for t in tasks) for d in args.datasets}
    workers = max(1, min(args.workers, len(tasks)))
    print(f"{len(tasks)} scene(s) [{per_ds}] x {len(configs)} config(s) = {total_runs} run(s)")
    print(f"  configs={configs}\n  downsample={'override '+str(args.downsample) if args.downsample else 'per-dataset'}"
          f"  n_images={args.n_images}  n_val={args.n_val} (relight test set, GT-lighting datasets)"
          f"  workers={workers}  device={args.device}\n  -> {args.out_root}")

    args.out_root.mkdir(parents=True, exist_ok=True)
    try:
        import multiprocessing as mp
        ctx = mp.get_context("spawn")
    except Exception:
        ctx = None

    rows, done_runs, done_scenes = [], 0, 0
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
        futs = {ex.submit(run_scene, t): t for t in tasks}
        for fut in as_completed(futs):
            done_scenes += 1
            for r in fut.result():
                done_runs += 1
                rows.append(r)
                tmp = args.out_root / "summary.csv.tmp"
                pd.DataFrame(rows).to_csv(tmp, index=False)
                os.replace(tmp, args.out_root / "summary.csv")
                tag = f"{r['dataset']}/{r['scene']}/{r['config']}"
                if r.get("status") == "error":
                    print(f"[{done_runs}/{total_runs}] (scene {done_scenes}/{len(tasks)}) {tag}"
                          f"  ERROR: {r.get('error')}", flush=True)
                else:
                    rec, rel = r.get("recon_rmse"), r.get("relighting_rmse")
                    print(f"[{done_runs}/{total_runs}] (scene {done_scenes}/{len(tasks)}) {tag}"
                          f"  recon={_f(rec)} alb={_f(r.get('albedo_rmse'))} relight={_f(rel)}"
                          f"  ({r['status']})", flush=True)

    df = pd.DataFrame(rows)
    ok = df[df["status"].isin(["ok", "cached"])] if "status" in df else df
    if len(ok):
        cols = [c for c in ("recon_rmse", "albedo_rmse", "roughness_rmse", "metallic_rmse",
                            "lighting_sh_rmse", "relighting_rmse") if c in ok]
        print("\n=== mean over scenes (dataset x config) ===")
        print(ok.groupby(["dataset", "config"])[cols].mean().round(4).to_string())
    n_err = int((df["status"] == "error").sum()) if "status" in df else 0
    print(f"\nsummary -> {args.out_root / 'summary.csv'}" + (f"  ({n_err} errored)" if n_err else ""))


def _f(v):
    return f"{v:.4f}" if isinstance(v, (int, float)) else "n/a"


if __name__ == "__main__":
    main()
