#!/usr/bin/env python
"""
scripts/sh3_inconsistency_study.py — measure how INCONSISTENT training data degrades
decomposition, on the self-rendered SH3 dataset (ct-ct_sh-frOn_env, a CT-SH inverse crime).

Idea: the base dataset is perfectly self-consistent — its GT intrinsics + GT per-image SH
reproduce every render exactly. We deliberately break that consistency: for the N training
images and the 3 material parameters (albedo, roughness, metallic) we draw a FIXED 16x3
pattern of constant shifts (Uniform(-1,1), seeded once so it is identical everywhere), scale
each parameter's shifts by a configurable strength `m_<param>`, add them to the GT maps
(clipped to [0,1]) — a DIFFERENT constant per training image — and RE-RENDER each training
image under its GT SH. The decomposition then has to explain N images that were each rendered
with slightly different materials with a SINGLE material map + per-image lighting, so it can't
fit them consistently. The decomposition CONFIG is held fixed; only the manipulation varies.

A manipulation set is a list of strength dicts, e.g.:
    [{}, {"m_albedo": 0.1}, {"m_albedo": 0.3},
     {"m_albedo": 0.1, "m_metallic": 0.1, "m_roughness": 0.1}]
{} is the unmanipulated control (reproduces the base dataset). Params omitted from a dict get
strength 0. The same seeded pattern is reused across strengths, so only the SCALE changes.

Metrics compare the recovered maps against the TRUE (unmanipulated) GT, plus the held-out
relighting test set (last --n_val GT lights, unmanipulated). Results ->
    results/inconsistency_study/<config>/<scene>/<manip>/   (results.json + saved estimates)

Reproducible + resumable (per-run results.json). Usage:
    python scripts/sh3_inconsistency_study.py                       # default manips + config
    python scripts/sh3_inconsistency_study.py --config base --views 1c349305_v0
    python scripts/sh3_inconsistency_study.py --manipulations '[{}, {"m_albedo":0.2}]'
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

from idr.paths import REPO_ROOT, RESULTS_DIR
from canonical_decomp_batch import CONFIGS, BASE            # reuse the config definitions

_SH3_ROOT = REPO_ROOT / "local_datasets" / "26-08-09-datasets" / "sh3"
VARIANT = "ct-ct_sh-frOn_env"
PARAMS = ["albedo", "roughness", "metallic"]                # the 3 manipulated parameters

#DEFAULT_CONFIG = "metallic_l1_1e-2_tv1e-3_light_mono"
DEFAULT_CONFIG = "light_mono"

DEFAULT_MANIPULATIONS = [
    {},
    {"m_albedo": 0.2}, {"m_albedo": 0.4}, {"m_albedo": 0.6},
    {"m_roughness": 0.4},
    {"m_metallic": 0.4},
    {"m_albedo": 0.2, "m_roughness": 0.2, "m_metallic": 0.2},
    {"m_albedo": 0.4, "m_roughness": 0.4, "m_metallic": 0.4},
]
_WANDB_PROJECT, _WANDB_ENTITY = "sh3-inconsistency", "DLVC-intrinsics"


# ── manipulation ──────────────────────────────────────────────────────────────
def base_pattern(n_train, seed):
    """Fixed 16x3 unit shift pattern ~ Uniform(-1,1). Seeded once; identical for every scene
    so the manipulation is fully reproducible."""
    return np.random.default_rng(seed).uniform(-1.0, 1.0, size=(n_train, len(PARAMS))).astype(np.float32)


def shifts_for(manip, pattern):
    """(n_train, 3) constant shifts = pattern * per-parameter strength."""
    m = np.array([float(manip.get(f"m_{p}", 0.0)) for p in PARAMS], np.float32)
    return (pattern * m[None, :]).astype(np.float32)


def manip_name(manip):
    if not manip:
        return "unmanip"
    return "_".join(f"{k.replace('m_', '')}{float(v):g}" for k, v in sorted(manip.items()))


# ── build a manipulated (temporary) scene dir ─────────────────────────────────
def build_manip_scene(scene_dir, tmp_dir, shifts, n_train, n_val, device, diffuse_fresnel):
    """Re-render the first n_train images with per-image-shifted GT intrinsics; keep the next
    n_val GT renders (unmanipulated) as the relight test set; copy the TRUE GT maps + GT SH.
    Returns the RMSE of the manipulated training maps vs GT (per param) for logging."""
    from idr.data.scene_io import load_scene
    from canonical_relight_render import Relighter

    scene_dir, tmp_dir = Path(scene_dir), Path(tmp_dir)
    sc = load_scene(scene_dir, gt_npy=True)
    gt = dict(albedo=sc["albedo_np"], roughness=sc["roughness_np"], metallic=sc["metallic_np"],
              normals=sc["normals_np"], mask=sc["mask_np"])
    rel = Relighter(gt["normals"], gt["mask"], device, torch.float32, diffuse_fresnel)

    tmp_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(scene_dir / "config.json", tmp_dir / "config.json")
    for f in ("albedo", "normals", "roughness", "metallic"):     # TRUE GT maps (for metrics)
        for ext in ("npy", "png"):
            src = scene_dir / f"{f}.{ext}"
            if src.exists():
                shutil.copy(src, tmp_dir / f"{f}.{ext}")
    for i in range(n_train + n_val):                             # GT SH for train + val
        shutil.copy(scene_dir / f"sh_{i:03d}.npy", tmp_dir / f"sh_{i:03d}.npy")
    for i in range(n_train, n_train + n_val):                    # val lights: unmanipulated
        shutil.copy(scene_dir / f"light_{i:03d}.npy", tmp_dir / f"light_{i:03d}.npy")

    manip_rmse = np.zeros(len(PARAMS), np.float32)
    mask = gt["mask"]
    for i in range(n_train):                                     # train lights: manipulated
        intr = dict(albedo=np.clip(gt["albedo"] + shifts[i, 0], 0, 1).astype(np.float32),
                    roughness=np.clip(gt["roughness"] + shifts[i, 1], 0, 1).astype(np.float32),
                    metallic=np.clip(gt["metallic"] + shifts[i, 2], 0, 1).astype(np.float32))
        img = rel.render_sh(intr, sc["sh_coeffs"][i])
        np.save(tmp_dir / f"light_{i:03d}.npy", img.astype(np.float32))
        for pi, p in enumerate(PARAMS):
            d = (intr[p] - gt[p])[mask]
            manip_rmse[pi] += float(np.sqrt((d ** 2).mean())) / n_train
    del rel
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()
    return {f"manip_rmse_{p}": float(manip_rmse[pi]) for pi, p in enumerate(PARAMS)}


# ── metrics (est vs TRUE GT) ──────────────────────────────────────────────────
def extra_metrics(scene_dir, out_dir):
    from idr.data.scene_io import load_scene
    sc = load_scene(Path(scene_dir), gt_npy=True)
    mask = sc["mask_np"]
    out = {}
    for key in ("roughness", "metallic"):
        est_p = Path(out_dir) / f"{key}_est.npy"
        if est_p.exists():
            est = np.load(est_p).astype(np.float32)[..., 0][mask]
            gt = sc[f"{key}_np"][..., 0][mask]
            out[f"{key}_rmse"] = float(np.sqrt(((est - gt) ** 2).mean()))
    return out


# ── one (scene, manipulation) ─────────────────────────────────────────────────
def _write_atomic(p, obj):
    tmp = Path(str(p) + ".tmp"); tmp.write_text(json.dumps(obj, indent=1)); os.replace(tmp, p)


def _row(scene, manip, m, status):
    return dict(scene=scene, manip=manip_name(manip),
                recon_rmse=m.get("recon_rmse"), albedo_rmse=m.get("albedo_rmse"),
                roughness_rmse=m.get("roughness_rmse"), metallic_rmse=m.get("metallic_rmse"),
                relighting_rmse=m.get("relighting_rmse"),
                manip_rmse_albedo=m.get("manip_rmse_albedo"),
                manip_rmse_roughness=m.get("manip_rmse_roughness"),
                manip_rmse_metallic=m.get("manip_rmse_metallic"),
                elapsed_s=m.get("elapsed_s"), status=status)


def run_one(task):
    os.environ.setdefault("WANDB_MODE", "disabled")
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from idr.pipelines.decompose import decompose_scene

    scene_dir = Path(task["scene_dir"]); out_dir = Path(task["out_dir"])
    manip = task["manip"]; device = task["device"]
    results_p = out_dir / "results.json"
    if results_p.exists() and not task["force"]:
        try:
            res = json.loads(results_p.read_text())
            r = _row(task["scene"], manip, res["metrics"], "cached"); return r
        except Exception:
            pass
    cfg = dict(task["cfg"]); n_train, n_val = task["n_train"], task["n_val"]
    try:
        pattern = base_pattern(n_train, task["seed"])
        shifts = shifts_for(manip, pattern)
        tmp_scene = Path(tempfile.mkdtemp(prefix="manip_"))
        try:
            mstat = build_manip_scene(scene_dir, tmp_scene, shifts, n_train, n_val,
                                      device, bool(cfg.get("diffuse_fresnel", True)))
            overrides = {**cfg, "n_images": n_train + n_val, "val_images": n_val}
            if out_dir.exists():
                shutil.rmtree(out_dir, ignore_errors=True)
            m = decompose_scene(tmp_scene, out_dir, cfg_overrides=overrides, device=device,
                                wandb_project=_WANDB_PROJECT, wandb_entity=_WANDB_ENTITY)
        finally:
            shutil.rmtree(tmp_scene, ignore_errors=True)
        extra = extra_metrics(scene_dir, out_dir)
        metrics = {**{k: m.get(k) for k in ("recon_rmse", "recon_mae", "albedo_rmse", "albedo_mae",
                    "final_loss", "elapsed_s", "albedo_scale", "n_train_images", "n_val_images",
                    "relight_rmse", "relight_mae")},
                   "relighting_rmse": m.get("relight_rmse"), **extra, **mstat}
        res = dict(scene=task["scene"], manip=manip, manip_name=manip_name(manip),
                   config=task["config"], seed=task["seed"], n_train=n_train, n_val=n_val,
                   scene_dir=str(scene_dir), out_dir=str(out_dir),
                   shifts=shifts.tolist(), params=PARAMS, cfg=cfg, metrics=metrics, status="ok")
        _write_atomic(results_p, res)
        # also drop a small shifts.json for the notebook (regenerates manipulated maps)
        _write_atomic(out_dir / "shifts.json",
                      dict(manip=manip, seed=task["seed"], n_train=n_train, params=PARAMS,
                           shifts=shifts.tolist(), scene_dir=str(scene_dir)))
        return _row(task["scene"], manip, metrics, "ok")
    except Exception as e:
        import traceback
        return dict(scene=task["scene"], manip=manip_name(manip), status="error",
                    error=f"{type(e).__name__}: {e}", traceback=traceback.format_exc()[-1500:])


# ── discovery + driver ────────────────────────────────────────────────────────
def discover(views):
    root = _SH3_ROOT
    items = []
    if not root.exists():
        return items
    for vd in sorted(p for p in root.iterdir() if p.is_dir()):
        sd = vd / VARIANT
        if sd.is_dir() and (sd / "config.json").exists() and any(sd.glob("light_*.npy")):
            if not views or vd.name in views:
                items.append((vd.name, sd))
    return items


def build_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=DEFAULT_CONFIG, choices=list(CONFIGS),
                   help="decomposition config (held fixed across manipulations)")
    p.add_argument("--manipulations", default=None,
                   help="JSON list of strength dicts; default = DEFAULT_MANIPULATIONS")
    p.add_argument("--views", nargs="*", default=None, help="restrict to these view names")
    p.add_argument("--out_root", type=Path, default=RESULTS_DIR / "inconsistency_study")
    p.add_argument("--n_train", type=int, default=16)
    p.add_argument("--n_val", type=int, default=16, help="held-out unmanipulated relight test set")
    p.add_argument("--seed", type=int, default=0, help="seed for the fixed shift pattern")
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--force", action="store_true")
    return p


def main():
    args = build_parser().parse_args()
    manips = json.loads(args.manipulations) if args.manipulations else DEFAULT_MANIPULATIONS
    cfg = {**dict(double=False, diffuse_fresnel=True, gt_npy=True, use_npy=True,
                  downsample=1, log_every=100), **BASE, **CONFIGS[args.config]}
    items = discover(set(args.views) if args.views else None)
    if not items:
        raise SystemExit(f"no {VARIANT} scenes under {_SH3_ROOT} for views={args.views}")
    out_root = args.out_root / args.config
    tasks = []
    for scene, scene_dir in items:
        for manip in manips:
            tasks.append(dict(scene=scene, scene_dir=str(scene_dir), manip=manip,
                              out_dir=str(out_root / scene / manip_name(manip)),
                              cfg=cfg, config=args.config, n_train=args.n_train,
                              n_val=args.n_val, seed=args.seed, device=args.device,
                              force=args.force))
    workers = max(1, min(args.workers, len(tasks)))
    print(f"{len(items)} scene(s) x {len(manips)} manip(s) = {len(tasks)} run(s)")
    print(f"  config={args.config}  seed={args.seed}  n_train={args.n_train} n_val={args.n_val}"
          f"  workers={workers}  device={args.device}\n  manips={[manip_name(m) for m in manips]}"
          f"\n  -> {out_root}")
    out_root.mkdir(parents=True, exist_ok=True)
    # record the study-level config (config used, seed, manip list, shift pattern)
    _write_atomic(out_root / "study.json",
                  dict(config=args.config, cfg=cfg, seed=args.seed, params=PARAMS,
                       manipulations=manips, base_pattern=base_pattern(args.n_train, args.seed).tolist(),
                       n_train=args.n_train, n_val=args.n_val, variant=VARIANT))
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
            tag = f"{r['scene']}/{r['manip']}"
            if r.get("status") == "error":
                print(f"[{done}/{len(tasks)}] {tag}  ERROR: {r.get('error')}", flush=True)
            else:
                print(f"[{done}/{len(tasks)}] {tag}  recon={_f(r.get('recon_rmse'))} "
                      f"albedo={_f(r.get('albedo_rmse'))} relight={_f(r.get('relighting_rmse'))}"
                      f"  ({r['status']})", flush=True)
    df = pd.DataFrame(rows)
    ok = df[df["status"].isin(["ok", "cached"])] if "status" in df else df
    if len(ok):
        cols = [c for c in ("recon_rmse", "albedo_rmse", "roughness_rmse", "metallic_rmse",
                            "relighting_rmse") if c in ok]
        print("\n=== mean over scenes (by manipulation) ===")
        print(ok.groupby("manip")[cols].mean().round(4).to_string())
    print(f"\nsummary -> {out_root / 'summary.csv'}")


def _f(v):
    return f"{v:.4f}" if isinstance(v, (int, float)) else "n/a"


if __name__ == "__main__":
    main()
