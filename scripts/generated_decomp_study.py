#!/usr/bin/env python
"""
scripts/generated_decomp_study.py — decompose the GENERATED relighting datasets
(canonical/INFINITE_GENERATED/<gen_cfg>/<scene>, built by build_generated_dataset.py) and
compare INCLUDING the source image as an extra observation vs NOT.

For each (domain in {INFINITE, MIT}) x (generation config gen_cfg in {cfg1, cfg5}) x (scene) x
(source in {no_source, with_source}): build a temp scene whose observations are the relit images linearised to
float (variant=exr), plus — for `with_source` — the linear source as a 17th observation; copy
the TRUE GT maps (albedo/normals/roughness/metallic from INFINITE) as fixed geometry + eval
targets; decompose with a FIXED config (default `light_mono`). Metrics are vs the TRUE GT (there
is no GT lighting for the artistic named lightings, so no relight metric — relighting is compared
in the notebook against GT INTRINSICS via a synthetic sweep).

Output -> results/generated_study/<config>/<domain>/<gen_cfg>/<scene>/<source>/  (results.json + est)
Resumable per results.json. Usage:
    python scripts/generated_decomp_study.py
    python scripts/generated_decomp_study.py --domains MIT --gen_configs cfg1 --config light_mono
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
from PIL import Image

from idr.paths import REPO_ROOT, RESULTS_DIR
from canonical_decomp_batch import CONFIGS, BASE

_CANON = REPO_ROOT / "local_datasets" / "canonical"
# DEFAULT_CONFIG = "light_mono"
DEFAULT_CONFIG = "metallic_l1_1e-2_tv1e-3_light_mono"
SOURCES = ["no_source", "with_source"]
# Per-domain default downsample, mirroring canonical_decomp_batch.py's per-dataset `ds`
# (INFINITE=2, MIT-train=4). --downsample overrides for every domain.
DOMAIN_DS = {"INFINITE": 2, "MIT": 4}
_WANDB_PROJECT, _WANDB_ENTITY = "generated-decomp", "DLVC-intrinsics"


# ── temp scene (relits linearised to float; optional source) ──────────────────
def build_study_scene(ds_dir, tmp_dir, with_source):
    """Relit PNGs -> linear light_NNN.npy (variant=exr), + source as an extra light when
    with_source; copy TRUE GT maps. Returns n_images."""
    from idr.data.scene_io import srgb_to_linear
    ds_dir, tmp_dir = Path(ds_dir), Path(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    relits = sorted(ds_dir.glob("light_*.png"))
    for i, p in enumerate(relits):
        lin = srgb_to_linear(np.asarray(Image.open(p).convert("RGB"), np.float32) / 255.0)
        np.save(tmp_dir / f"light_{i:03d}.npy", lin.astype(np.float32))
    n = len(relits)
    if with_source:
        np.save(tmp_dir / f"light_{n:03d}.npy", np.load(ds_dir / "source.npy").astype(np.float32))
        n += 1
    for f in ("albedo", "normals", "roughness", "metallic"):       # TRUE GT (geometry + targets)
        for ext in ("npy", "png"):
            src = ds_dir / f"{f}.{ext}"
            if src.exists():
                shutil.copy(src, tmp_dir / f"{f}.{ext}")
    cfg = json.loads((ds_dir / "config.json").read_text())
    cfg["variant"] = "exr"; cfg["n_lights"] = n
    (tmp_dir / "config.json").write_text(json.dumps(cfg, indent=1))
    return n


def extra_metrics(ds_dir, out_dir, ds=1):
    """roughness/metallic RMSE from saved estimates. GT is strided by `ds` to match the
    downsampled estimates (same as canonical_decomp_batch.compute_metrics)."""
    from idr.data.scene_io import load_scene
    ds = int(ds or 1)
    sc = load_scene(Path(ds_dir), gt_npy=True)
    mask = sc["mask_np"][::ds, ::ds] if ds > 1 else sc["mask_np"]
    out = {}
    for key in ("roughness", "metallic"):
        est_p = Path(out_dir) / f"{key}_est.npy"
        if est_p.exists():
            est = np.load(est_p).astype(np.float32)[..., 0][mask]
            gt = (sc[f"{key}_np"][::ds, ::ds] if ds > 1 else sc[f"{key}_np"])[..., 0][mask]
            out[f"{key}_rmse"] = float(np.sqrt(((est - gt) ** 2).mean()))
    return out


# ── one (gen_cfg, scene, source) ──────────────────────────────────────────────
def _write_atomic(p, obj):
    tmp = Path(str(p) + ".tmp"); tmp.write_text(json.dumps(obj, indent=1)); os.replace(tmp, p)


def _row(domain, gen_cfg, scene, source, m, status, ds=None):
    return dict(domain=domain, gen_cfg=gen_cfg, scene=scene, source=source, ds=ds,
                recon_rmse=m.get("recon_rmse"), albedo_rmse=m.get("albedo_rmse"),
                roughness_rmse=m.get("roughness_rmse"), metallic_rmse=m.get("metallic_rmse"),
                n_images=m.get("n_train_images"), elapsed_s=m.get("elapsed_s"), status=status)


def run_one(task):
    os.environ.setdefault("WANDB_MODE", "disabled")
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from idr.pipelines.decompose import decompose_scene

    ds_dir = Path(task["ds_dir"]); out_dir = Path(task["out_dir"]); source = task["source"]
    domain = task["domain"]; ds = int(task["cfg"].get("downsample", 1) or 1)
    results_p = out_dir / "results.json"
    if results_p.exists() and not task["force"]:
        try:
            r = json.loads(results_p.read_text())
            return _row(domain, task["gen_cfg"], task["scene"], source, r["metrics"], "cached",
                        r.get("ds", ds))
        except Exception:
            pass
    try:
        tmp = Path(tempfile.mkdtemp(prefix="gen_"))
        try:
            n = build_study_scene(ds_dir, tmp, with_source=(source == "with_source"))
            overrides = {**dict(task["cfg"]), "n_images": n, "val_images": 0}
            if out_dir.exists():
                shutil.rmtree(out_dir, ignore_errors=True)
            m = decompose_scene(tmp, out_dir, cfg_overrides=overrides, device=task["device"],
                                wandb_project=_WANDB_PROJECT, wandb_entity=_WANDB_ENTITY)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        metrics = {**{k: m.get(k) for k in ("recon_rmse", "recon_mae", "albedo_rmse", "albedo_mae",
                    "final_loss", "elapsed_s", "albedo_scale", "n_train_images")},
                   **extra_metrics(ds_dir, out_dir, ds)}
        res = dict(domain=domain, gen_cfg=task["gen_cfg"], scene=task["scene"], source=source,
                   config=task["config"], ds=ds, ds_dir=str(ds_dir), out_dir=str(out_dir),
                   n_images=n, cfg=task["cfg"], metrics=metrics, status="ok")
        _write_atomic(results_p, res)
        return _row(domain, task["gen_cfg"], task["scene"], source, metrics, "ok", ds)
    except Exception as e:
        import traceback
        return dict(domain=domain, gen_cfg=task["gen_cfg"], scene=task["scene"], source=source,
                    status="error", error=f"{type(e).__name__}: {e}",
                    traceback=traceback.format_exc()[-1500:])


# ── discovery + driver ────────────────────────────────────────────────────────
def discover_domains():
    return sorted(p.name[:-len("_GENERATED")] for p in _CANON.glob("*_GENERATED") if p.is_dir())


def discover(domains, gen_configs, views):
    items = []
    for dom in domains:
        droot = _CANON / f"{dom}_GENERATED"
        for gc in (gen_configs or [p.name for p in sorted(droot.iterdir()) if p.is_dir()]):
            root = droot / gc
            if not root.exists():
                continue
            for sd in sorted(p for p in root.iterdir() if p.is_dir()):
                if (sd / "config.json").exists() and (not views or sd.name in views):
                    items.append((dom, gc, sd.name, sd))
    return items


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=DEFAULT_CONFIG, choices=list(CONFIGS))
    p.add_argument("--domains", nargs="+", default=None, help="default: all <DOMAIN>_GENERATED dirs")
    p.add_argument("--gen_configs", nargs="+", default=None, help="default: all under each domain")
    p.add_argument("--sources", nargs="+", default=SOURCES, choices=SOURCES)
    p.add_argument("--views", nargs="*", default=None)
    p.add_argument("--out_root", type=Path, default=RESULTS_DIR / "generated_study")
    p.add_argument("--downsample", type=int, default=0,
                   help="override every domain's default ds (0 = per-domain: %s)" % DOMAIN_DS)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--force", action="store_true")
    return p


def main():
    args = build_parser().parse_args()
    base_cfg = {**dict(double=False, diffuse_fresnel=True, gt_npy=True, use_npy=True,
                       downsample=1, log_every=100), **BASE, **CONFIGS[args.config]}
    domains = args.domains or discover_domains()
    items = discover(domains, args.gen_configs, set(args.views) if args.views else None)
    if not items:
        raise SystemExit(f"no leaves under {_CANON}/<DOMAIN>_GENERATED for domains={domains} "
                         f"gen_configs={args.gen_configs} (run build_generated_dataset.py first)")
    out_root = args.out_root / args.config

    def _ds(dom):                       # per-domain default ds, overridable by --downsample
        return args.downsample or DOMAIN_DS.get(dom, 1)
    tasks = [dict(domain=dom, gen_cfg=gc, scene=scene, ds_dir=str(sd), source=src,
                  out_dir=str(out_root / dom / gc / scene / src),
                  cfg={**base_cfg, "downsample": _ds(dom)}, config=args.config,
                  device=args.device, force=args.force)
             for dom, gc, scene, sd in items for src in args.sources]
    workers = max(1, min(args.workers, len(tasks)))
    ds_summary = {dom: _ds(dom) for dom in domains}
    print(f"{len(items)} (domain,gen_cfg,scene) x {len(args.sources)} source variant(s) = {len(tasks)} run(s)")
    print(f"  config={args.config}  domains={domains}  sources={args.sources}  downsample={ds_summary}"
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
            tag = f"{r['domain']}/{r['gen_cfg']}/{r['scene']}/{r['source']}"
            if r.get("status") == "error":
                print(f"[{done}/{len(tasks)}] {tag}  ERROR: {r.get('error')}", flush=True)
            else:
                print(f"[{done}/{len(tasks)}] {tag}  ds={r.get('ds')} albedo={_f(r.get('albedo_rmse'))} "
                      f"rough={_f(r.get('roughness_rmse'))} metal={_f(r.get('metallic_rmse'))} "
                      f"recon={_f(r.get('recon_rmse'))} n={r.get('n_images')}  ({r['status']})", flush=True)
    df = pd.DataFrame(rows); ok = df[df["status"].isin(["ok", "cached"])] if "status" in df else df
    if len(ok):
        cols = [c for c in ("albedo_rmse", "roughness_rmse", "metallic_rmse", "recon_rmse") if c in ok]
        print("\n=== mean over scenes (domain x gen_cfg x source) ===")
        print(ok.groupby(["domain", "gen_cfg", "source"])[cols].mean().round(4).to_string())
    print(f"\nsummary -> {out_root / 'summary.csv'}")


def _f(v):
    return f"{v:.4f}" if isinstance(v, (int, float)) else "n/a"


if __name__ == "__main__":
    main()
