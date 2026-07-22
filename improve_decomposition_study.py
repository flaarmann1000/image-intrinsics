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

The strategy study runs at --study_downsample (default 2 = 256^2); full resolution
is reserved for the resolution sweep. Each strategy is run under every mode in
--reg_modes (default `both none`: lambda_tv=lambda_metallic_binarize=--reg_lambda,
and both off). Run dirs are `<strategy>_<reg>_ds<D>` — the downsample is IN the
name, so a re-run at a different --study_downsample never stale-caches an earlier
run, and different reg modes never overwrite each other. Two extra runs (only when
'both' is requested) isolate the regularizer on the baseline (tv-only, binarize-only).

Then the plain, UN-regularized baseline is run at --downsamples (default 1 2 4 8 16,
i.e. including full res) to show how resolution alone moves the metrics.

Analysis + plots are built from EVERY `..._ds<D>` run found under <out>/runs
(previous + new), so incrementally adding a reg mode folds straight into the
comparison. (Pre-fix orphan dirs without the `_ds<D>` suffix are ignored — their
curriculum warm-ups had pinned frozen intrinsics at GT.)

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
import re
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from idr.pipelines.decompose import decompose_scene# noqa: E402
from run_decomposition import save_intrinsics_plot, save_relight_plots   # noqa: E402


# ───────────────────────── run specs ─────────────────────────────────────────
REG_CONFIGS = {
    "both":     lambda R: dict(lambda_tv=R, lambda_metallic_binarize=R),
    "none":     lambda R: dict(lambda_tv=0.0, lambda_metallic_binarize=0.0),
    "tv":       lambda R: dict(lambda_tv=R, lambda_metallic_binarize=0.0),
    "binarize": lambda R: dict(lambda_tv=0.0, lambda_metallic_binarize=R),
}


def build_runs(args):
    """Return a list of run dicts: name, strategy, group, downsample, reg, cfg-extra.

    The strategy study runs at --study_downsample (default 2). Its run dirs are
    `<strategy>_<reg>_ds<D>` — the downsample is IN the name so changing --study_downsample
    can never stale-cache an earlier run at a different resolution. The resolution
    sweep (un-regularized baseline) keeps the `baseline_ds<D>` names so earlier
    sweep runs on disk are still reused.
    """
    R = args.reg_lambda
    WARM, TOTAL = args.warmup, args.n_iter
    FINAL = max(1, TOTAL - WARM)
    sd = args.study_downsample

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

    runs = []
    for mode in args.reg_modes:                      # e.g. ["both", "none"]
        regcfg = REG_CONFIGS[mode](R)
        for strat, extra in strategies.items():
            runs.append(dict(name=f"{strat}_{mode}_ds{sd}", strategy=strat, group="fullres",
                             downsample=sd, reg=mode, cfg={**regcfg, **extra}))
    # regularizer isolation on the baseline (grouped under 'baseline' for plotting).
    if "both" in args.reg_modes:
        for iso in ("tv", "binarize"):
            runs.append(dict(name=f"baseline_{iso}_ds{sd}", strategy="baseline", group="fullres",
                             downsample=sd, reg=iso, cfg=REG_CONFIGS[iso](R)))
    # UN-regularized baseline at each downsample -> resolution curve (full res = ds 1)
    for ds in args.downsamples:
        runs.append(dict(name=f"baseline_ds{ds}", strategy="baseline", group="downsample",
                         downsample=ds, reg="none", cfg=REG_CONFIGS["none"](R)))
    # dedup by dir name (e.g. strategy-baseline-under-tv == the tv isolation baseline)
    seen, uniq = set(), []
    for r in runs:
        if r["name"] not in seen:
            seen.add(r["name"]); uniq.append(r)
    return uniq


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

# the study's own per-run artifacts (written AFTER decompose_scene's metrics.json)
_ARTIFACTS = ("report.json", "intrinsics.png", "loss_curve.png")


def run_one(task):
    """Execute (or resume) ONE study run. Top-level + picklable so it can run in a
    spawned process. Carries the full resume logic:
      valid metrics.json  -> never re-run the optimize; just finish missing artifacts
      corrupt / absent    -> wipe the partial dir and redo
    Returns a small status dict; the parent rebuilds the summary from disk.
    """
    os.environ["WANDB_MODE"] = task["wandb_mode"]
    out_dir = Path(task["out_dir"])
    scene, r, base_res = Path(task["scene"]), task["run"], task["base_res"]

    m = None
    if not task["force"] and (out_dir / "metrics.json").exists():
        try:
            m = json.loads((out_dir / "metrics.json").read_text())
        except (json.JSONDecodeError, OSError):
            m = None
    try:
        if m is not None:
            m.setdefault("log_every", task["log_every"])
            if all((out_dir / f).exists() for f in _ARTIFACTS):
                return {"name": r["name"], "status": "cached"}
            write_artifacts(out_dir, m, scene, r, base_res)
            return {"name": r["name"], "status": "cached+artifacts"}
        if out_dir.exists():                     # partial/preempted (or corrupt) -> clean
            shutil.rmtree(out_dir, ignore_errors=True)
        t0 = time.perf_counter()
        m = decompose_scene(scene, out_dir, cfg_overrides=task["cfg"], device=task["device"],
                            gt_npy=True, wandb_project=task["wandb_project"])
        m.setdefault("elapsed_s", time.perf_counter() - t0)
        m["log_every"] = task["log_every"]
        write_artifacts(out_dir, m, scene, r, base_res)
        return {"name": r["name"], "status": "ok", "elapsed_s": m["elapsed_s"]}
    except Exception as e:                       # keep the batch alive on one bad run
        import traceback
        return {"name": r["name"], "status": "error", "error": f"{type(e).__name__}: {e}",
                "traceback": traceback.format_exc()[-1500:]}


def write_artifacts(out_dir, m, scene, r, base_res):
    """Regenerate the study's per-run artifacts from a (cached or fresh) metrics
    dict. Cheap — never re-runs the decomposition — so a run interrupted between
    decompose_scene's marker and here is completed on resume, not redone."""
    save_intrinsics_plot(out_dir, m, out_dir / "intrinsics.png")
    save_relight_plots(out_dir, m, scene, r["downsample"])
    save_loss_curve(out_dir, m)
    (out_dir / "report.json").write_text(json.dumps(
        {"name": r["name"], "strategy": r["strategy"], "group": r["group"],
         "downsample": r["downsample"], "reg": r["reg"],
         "resolution": base_res // r["downsample"], **_row(m)}, indent=1))


def _row(m):
    return dict(albedo_rmse=m["albedo_rmse"], albedo_mae=m.get("albedo_mae"),
                roughness_mae=m["roughness_err_mean"], metallic_mae=m["metallic_err_mean"],
                train_recon_rmse=m["recon_rmse"], val_relight_rmse=m.get("relight_rmse"),
                val_relight_mae=m.get("relight_mae"), final_loss=m["final_loss"],
                elapsed_s=m.get("elapsed_s"))


def load_all_runs(runs_root: Path, base_res: int = 0) -> pd.DataFrame:
    """Every valid run on disk (previous + this invocation), so the analysis and
    plots always include earlier reg modes. Only dirs following the `..._ds<D>`
    naming are read — this deliberately skips pre-fix orphan dirs (whose curriculum
    warm-ups pinned frozen intrinsics at GT). Resolution is derived from the base
    image size / downsample (the stored metrics H was 0)."""
    rows = []
    for d in sorted(p for p in runs_root.iterdir() if p.is_dir()):
        if not re.search(r"_ds\d+$", d.name) or not (d / "metrics.json").exists():
            continue
        try:                                    # skip a corrupt/partly-written run
            m = json.loads((d / "metrics.json").read_text())
            meta = json.loads((d / "report.json").read_text()) if (d / "report.json").exists() else {}
        except (json.JSONDecodeError, OSError):
            continue
        name = meta.get("name", d.name)
        ds = int(meta.get("downsample") or re.search(r"_ds(\d+)$", name).group(1))
        # strategy / reg: explicit if present, else parsed from `<strat>_<reg>_ds<D>`
        base = re.sub(r"_ds\d+$", "", name)
        reg_guess, had_reg = "none", False
        for rr in ("both", "none", "tv", "binarize"):
            if base.endswith("_" + rr):
                reg_guess, base, had_reg = rr, base[: -(len(rr) + 1)], True
                break
        strat = meta.get("strategy") or (base or "baseline")
        # strategy runs carry a reg token in the name (fullres); the resolution
        # sweep is `baseline_ds<D>` with no reg token (downsample).
        group = meta.get("group") or ("fullres" if had_reg else "downsample")
        res = int(meta.get("resolution") or (base_res // ds if base_res else 0))
        rows.append(dict(name=name, strategy=strat, group=group,
                         reg=meta.get("reg", reg_guess), downsample=ds,
                         resolution=res, **_row(m)))
    return pd.DataFrame(rows)


REG_COLOR = {"both": "#4C72B0", "none": "#DD8452", "tv": "#55A868", "binarize": "#8172B3",
             "unknown": "#999999"}


def plot_fullres(df, path, study_ds=None):
    d = df[df["group"] == "fullres"].copy()
    if study_ds is not None:                 # one resolution only — never average
        d = d[d["downsample"] == study_ds]   # strategy runs across study_downsamples
    if not len(d):
        return
    # x-axis = strategy, grouped bars = reg mode. Only column-ize reg modes that
    # span >1 strategy (a full sweep); baseline-only isolation modes (e.g. binarize
    # run only on the baseline) would otherwise be a column of empty slots — they
    # stay in the CSV / printed table instead.
    strategies = sorted(d["strategy"].unique(),
                        key=lambda s: (s != "baseline", s))     # baseline first
    span = {r: d[d.reg == r]["strategy"].nunique() for r in set(d["reg"])}
    regs = [r for r in ("none", "both", "tv", "binarize") if span.get(r, 0) > 1]
    x = np.arange(len(strategies))
    w = 0.8 / max(len(regs), 1)
    fig, axes = plt.subplots(1, len(METRICS), figsize=(3.5 * len(METRICS), 4.6))
    for ax, k in zip(axes, METRICS):
        for i, reg in enumerate(regs):
            vals = [d[(d.strategy == s) & (d.reg == reg)][k].mean() for s in strategies]
            ax.bar(x + (i - (len(regs) - 1) / 2) * w, vals, w,
                   color=REG_COLOR.get(reg, "#999"), label=reg)
        ax.set_xticks(x); ax.set_xticklabels(strategies, rotation=60, ha="right", fontsize=6)
        ax.set_title(k, fontsize=9); ax.grid(axis="y", alpha=0.3); ax.legend(fontsize=6, title="reg")
    res = int(d["resolution"].iloc[0]) if len(d) and d["resolution"].iloc[0] else None
    fig.suptitle(f"strategies by regularization{f' @ {res}px' if res else ''} "
                 f"(lower = better)", fontsize=11)
    plt.tight_layout(); fig.savefig(path, dpi=90); plt.close(fig)


def plot_downsample(df, path):
    d = df[df["group"] == "downsample"].copy()
    # add the full-res un-regularized baseline as the top-resolution point, if present
    top = df[(df["group"] == "fullres") & (df["strategy"] == "baseline") & (df["reg"] == "none")]
    if len(top):
        d = pd.concat([d, top], ignore_index=True)
    if not len(d):
        return
    d = d[d["resolution"] > 0].drop_duplicates("resolution").sort_values("resolution")
    if not len(d):
        return
    fig, ax = plt.subplots(1, 3, figsize=(13, 4.2))
    for a, k in zip(ax, ["train_recon_rmse", "val_relight_rmse", "albedo_mae"]):
        a.plot(d["resolution"], d[k], "o-", lw=1.8, ms=6)
        for xx, yy in zip(d["resolution"], d[k]):
            a.annotate(f"{yy:.3g}", (xx, yy), fontsize=6, ha="center", va="bottom",
                       xytext=(0, 4), textcoords="offset points")
        # mark the best resolution
        best = d.loc[d[k].idxmin()]
        a.plot(best["resolution"], best[k], "*", ms=15, color="#C44E52",
               label=f"best @ {int(best['resolution'])}px")
        a.set_xlabel("resolution (px/side)"); a.set_title(k, fontsize=10)
        a.set_xticks(d["resolution"]); a.set_xticklabels([int(v) for v in d["resolution"]])
        if d["resolution"].nunique() > 1:
            a.set_xscale("log", base=2)
        a.grid(alpha=0.3, which="both"); a.legend(fontsize=7)
    fig.suptitle("un-regularized baseline vs resolution (down-sampling sweep)", fontsize=11)
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
    p.add_argument("--n_iter", type=int, default=500)
    p.add_argument("--lbfgs_max_iter", type=int, default=40)
    p.add_argument("--warmup", type=int, default=100, help="Curriculum phase-1 steps.")
    p.add_argument("--reg_lambda", type=float, default=1e-5)
    p.add_argument("--reg_modes", nargs="+", default=["both", "none", "tv"],
                   choices=["both", "none", "tv", "binarize"],
                   help="Which regularization settings to run the strategies under. "
                        "'both' keeps legacy (unsuffixed) run dirs; others are suffixed so "
                        "they never overwrite earlier runs. Default runs both + none.")
    p.add_argument("--downsamples", type=int, nargs="+", default=[1, 2, 4, 8, 16],
                   help="Resolution sweep for the un-regularized baseline (1 = full 512^2). "
                        "This is the ONLY place full resolution is used.")
    p.add_argument("--study_downsample", type=int, default=4,
                   help="Downsample for the strategy study (default 2 = 256^2). Full res is "
                        "reserved for the resolution sweep only.")
    p.add_argument("--noise_std", type=float, default=0.1)
    p.add_argument("--pixel_frac", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--img_batch", type=int, default=8,
                   help="Images per grad-accum chunk (bounds 512^2 memory; safe under LBFGS).")
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--double", action="store_true", help="fp64 (slow at 512^2; default fp32).")
    p.add_argument("--workers", type=int, default=0,
                   help="Parallel worker processes (0 = min(#runs, CPU count)); each gets its "
                        "own CUDA context. The runs are independent, so this scales well — but "
                        "at 256^2/512^2 each worker holds real GPU memory, so lower it if you OOM. "
                        "1 = sequential (in-process).")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--force", action="store_true")
    p.add_argument("--wandb_mode", default="online",
                   choices=["disabled", "offline", "online"])
    p.add_argument("--wandb_project", default="3dfront-improve-study")
    args = p.parse_args()
    os.environ["WANDB_MODE"] = args.wandb_mode

    scene = args.scene or discover_scene(args.datasets_root)
    assert Path(scene).exists(), f"missing scene {scene}"
    base_res = int(np.load(next(Path(scene).glob("light_*.npy"))).shape[0])
    runs = build_runs(args)
    runs_root = args.out / "runs"
    runs_root.mkdir(parents=True, exist_ok=True)
    print(f"scene: {scene}  (base {base_res}^2)  study at ds{args.study_downsample} "
          f"({base_res // args.study_downsample}^2)\n{len(runs)} run(s): "
          f"{sum(r['group']=='fullres' for r in runs)} strategy + "
          f"{sum(r['group']=='downsample' for r in runs)} resolution-sweep  | device={args.device}")

    args.out.mkdir(parents=True, exist_ok=True)
    tasks = [dict(run=r, out_dir=str(runs_root / r["name"]), scene=str(scene),
                  cfg={**base_cfg(args), **r["cfg"], "downsample": r["downsample"]},
                  base_res=base_res, device=args.device, force=args.force,
                  log_every=args.log_every, wandb_mode=args.wandb_mode,
                  wandb_project=args.wandb_project)
             for r in runs]
    workers = args.workers or min(len(tasks), os.cpu_count() or 4)
    workers = max(1, min(workers, len(tasks)))
    print(f"workers={workers}"
          + ("  (sequential)" if workers == 1 else
             f"  — each holds its own CUDA context; lower --workers if the GPU OOMs"))

    try:
        import multiprocessing as mp
        ctx = mp.get_context("spawn")
    except Exception:
        ctx = None

    def _flush_summary():
        _tmp = args.out / "summary.csv.tmp"
        load_all_runs(runs_root, base_res).to_csv(_tmp, index=False)
        os.replace(_tmp, args.out / "summary.csv")

    done, errs = 0, []
    if workers == 1:                             # keep a simple in-process path
        for t in tasks:
            res = run_one(t)
            done += 1
            print(f"[{done}/{len(tasks)}] {res['name']}: {res['status']}"
                  + (f" — {res.get('error')}" if res["status"] == "error" else ""), flush=True)
            if res["status"] == "error":
                errs.append(res)
            _flush_summary()
    else:
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
            futs = {ex.submit(run_one, t): t for t in tasks}
            for fut in as_completed(futs):
                res = fut.result()
                done += 1
                print(f"[{done}/{len(tasks)}] {res['name']}: {res['status']}"
                      + (f" — {res.get('error')}" if res["status"] == "error" else ""), flush=True)
                if res["status"] == "error":
                    errs.append(res)
                # incremental summary from disk (survives interruption on a long run)
                _flush_summary()
    if errs:
        print(f"\n{len(errs)} run(s) FAILED:")
        for e in errs:
            print(f"  {e['name']}: {e['error']}")

    # ── analysis + plots over ALL valid runs on disk (all reg modes) ──────────
    df = load_all_runs(runs_root, base_res)
    df.sort_values(["group", "downsample", "strategy", "reg"]).to_csv(args.out / "summary.csv", index=False)
    # strategy graphic is scoped to THIS study resolution (never averages across
    # study_downsamples) and named per-ds so different resolutions don't overwrite.
    sd = args.study_downsample
    plot_fullres(df, args.out / f"compare_strategies_ds{sd}.png", study_ds=sd)
    plot_downsample(df, args.out / "compare_downsample.png")

    fr = df[(df["group"] == "fullres") & (df["downsample"] == sd)]
    if len(fr):
        print(f"\n=== strategies @ ds{sd} ({base_res // sd}px), sorted by val_relight_rmse ===")
        print(fr[["name", "reg"] + METRICS].sort_values("val_relight_rmse")
              .round(5).to_string(index=False))
    other = sorted(set(df[df["group"] == "fullres"]["downsample"]) - {sd})
    if other:
        print(f"\n(note: strategy runs also exist at ds {other} — see compare_strategies_ds*.png)")
    print(f"\n{len(df)} run(s) on disk -> summary {args.out/'summary.csv'}   plots -> {args.out}")


if __name__ == "__main__":
    main()
