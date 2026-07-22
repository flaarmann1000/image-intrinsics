#!/usr/bin/env python
"""
scripts/profile_perf.py — where does the decomposition's wall time actually go?

Standalone profiler for the LBFGS ct_sh decomposition, sized to the REAL study config
(512^2, 100 train / 28 val by default) so the numbers transfer to the VM.

It answers, in order:
  0  task breakdown   forward vs backward vs LBFGS-internal vs reporting vs setup
  0b downsample       cost vs resolution (1/2/4 = 512/256/128) and how it scales
  1  closure cost     how many fwd+bwd passes one outer step really costs (line search)
  2  img_batch        does a bigger batch help, and what does it cost in memory?
  3  history_size     LBFGS's two-loop recursion is a launch-bound tax
  4  logging          cost of a logged step (recomputes the held-out relight metric)
  5  workers          does running N decompositions in parallel help on ONE GPU?
  6  summary          measured numbers + a recommended config

Why not torch.profiler: `Optimizer.step#LBFGS.step` is a *scope* that swallows the nested
forward/backward, so it reports impossible things (>100% GPU busy, ~0s backward). This
times the real phases with explicit cuda.synchronize() instead.

Every section is optional and every budget is a flag, because at 512^2 x 100 images a
single fwd+bwd pass is ~1s. The script measures one closure first and prints a runtime
estimate before doing the expensive parts.

Examples
  # the real thing, on the VM
  python scripts/profile_perf.py --scene <view> --downsample 1 --n_train 100 --n_val 28

  # quick sanity pass first (a few minutes)
  python scripts/profile_perf.py --downsample 4 --n_train 32 --n_val 8 --quick

  # just the resolution sweep (512 vs 256 vs 128)
  python scripts/profile_perf.py --skip breakdown closures plateau img_batch history \\
      logging workers --downsamples 1 2 4
"""
import argparse
import copy
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("WANDB_MODE", "disabled")

import idr.optim.steps as S                      # noqa: E402
from idr.data.geometry import make_proxy_geometry     # noqa: E402
from idr.data.scene_io import load_scene
from idr.config import DEFAULT_CFG  # noqa: E402
from idr.optim.models.ct_sh import _optimize_ct_sh
OOM = torch.cuda.OutOfMemoryError if hasattr(torch.cuda, "OutOfMemoryError") else RuntimeError


# ───────────────────────────── harness ───────────────────────────────────────
def sync(dev):
    if dev == "cuda":
        torch.cuda.synchronize()


class Timers:
    """Phase accumulators filled by the instrumented _opt_step."""
    def __init__(self): self.reset()

    def reset(self):
        self.fwd = self.bwd = self.step = self.extra = 0.0
        self.nclo = self.nout = 0


T = Timers()
_ORIG_OPT_STEP = S._opt_step
_ORIG_MAKE_OPT = S._make_optimizer
_HIST_OVERRIDE = [None]

# Count LBFGS closure evaluations GLOBALLY (one closure = one fwd+bwd over the batch).
# Must not depend on the phase-timing wrapper, since some sections run untimed — and
# the closure count is the headline number the whole extrapolation rests on.
_CLO = {"n": 0}
_ORIG_LBFGS_STEP = torch.optim.LBFGS.step


def _counting_lbfgs_step(self, closure):
    def wrapped():
        _CLO["n"] += 1
        return closure()
    return _ORIG_LBFGS_STEP(self, wrapped)


torch.optim.LBFGS.step = _counting_lbfgs_step


def _timed_opt_step(opt, forward_fn, cfg):
    """Drop-in for S._opt_step (full-batch path) that times each phase."""
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    def closure():
        opt.zero_grad()
        sync(dev); a = time.perf_counter()
        loss, *_ = forward_fn()
        sync(dev); b = time.perf_counter()
        loss.backward()
        sync(dev); c = time.perf_counter()
        T.fwd += b - a; T.bwd += c - b; T.nclo += 1
        return loss

    sync(dev); s0 = time.perf_counter()
    try:
        opt.step(closure)
    except (IndexError, TypeError):
        opt.state.clear()
    sync(dev); T.step += time.perf_counter() - s0; T.nout += 1
    with torch.no_grad():                       # _opt_step's extra reporting forward
        sync(dev); f0 = time.perf_counter()
        r = forward_fn()
        sync(dev); T.extra += time.perf_counter() - f0
    return r


def _patched_make_optimizer(params, cfg):
    """Honour --history_sizes without editing the repo's _make_optimizer."""
    if _HIST_OVERRIDE[0] is not None and str(cfg.get("optimizer", "")).upper() == "LBFGS":
        import inspect
        src = inspect.getsource(_ORIG_MAKE_OPT)
        kw = dict(lr=cfg["lr"], max_iter=cfg["lbfgs_max_iter"],
                  line_search_fn="strong_wolfe", history_size=_HIST_OVERRIDE[0])
        if "tolerance_grad" in src:            # mirror the repo's early-stop setting
            kw.update(tolerance_grad=0, tolerance_change=0)
        return torch.optim.LBFGS(params, **kw)
    return _ORIG_MAKE_OPT(params, cfg)


def load_inputs(args, quiet=False):
    scene = Path(args.scene)
    # Datasets built by batch_dataset_decomposition.ipynb store the GT maps as lossless
    # float32 .npy (normals/albedo/roughness/metallic) next to the 8/16-bit PNGs, and the
    # observations as light_*.npy (config variant "exr"). load_scene(gt_npy=True) prefers
    # the .npy maps but silently falls back to PNG if ANY of the four is missing — which
    # would quietly reintroduce the quantization we are trying to avoid. Report which
    # path is actually taken, and fail loudly if neither is complete.
    want_npy = not args.no_gt_npy
    maps = ("normals", "albedo", "roughness", "metallic")
    have_npy = {m: (scene / f"{m}.npy").exists() for m in maps}
    have_png = {m: (scene / f"{m}.png").exists() for m in maps}
    if want_npy and not all(have_npy.values()):
        missing = [m for m, ok in have_npy.items() if not ok]
        if all(have_png.values()):
            if not quiet:
                print(f"  ! GT .npy incomplete (missing {missing}) -> falling back to PNG maps "
                      f"(quantized). Pass --no_gt_npy to silence.")
        else:
            raise SystemExit(
                f"scene has neither complete .npy nor .png GT maps "
                f"(missing npy: {missing}; missing png: "
                f"{[m for m, ok in have_png.items() if not ok]}) — {scene}")
    elif not want_npy and not all(have_png.values()):
        raise SystemExit(f"--no_gt_npy given but PNG GT maps are missing in {scene}")

    sc = load_scene(scene, gt_npy=want_npy)
    src = "npy (lossless)" if (want_npy and all(have_npy.values())) else "png (quantized)"
    n_lights = len(sc["images"])
    if not quiet:
        print(f"  GT maps : {src}   observations: {n_lights} light_* "
              f"({'npy' if (scene / 'light_000.npy').exists() else 'png'})")
    if sc.get("sh_coeffs") is None:
        raise SystemExit(f"no GT sh_*.npy lighting in {scene} — needed for the val "
                         f"relight metric in the logging section")
    ds = args.downsample
    st = lambda a: np.ascontiguousarray(a[::ds, ::ds])
    n_tot = args.n_train + args.n_val
    avail = len(sc["images"])
    if avail < n_tot:
        if not quiet:
            print(f"  ! scene has only {avail} lights; using n_train={max(avail - args.n_val, 1)}")
        args.n_train = max(avail - args.n_val, 1)
        n_tot = args.n_train + args.n_val
    imgs = [st(im) for im in sc["images"][:n_tot]]
    shs = [np.asarray(x, np.float32) for x in sc["sh_coeffs"][:n_tot]]
    N, mask = st(sc["normals_np"]), st(sc["mask_np"])
    Nhw, frag, mhw, cam = make_proxy_geometry(N, mask, 60.0, 2.0, args.device, torch.float32)
    M = int(mask.sum())
    return dict(
        train_imgs=imgs[:args.n_train], train_sh=shs[:args.n_train],
        val_imgs=imgs[args.n_train:], val_sh=shs[args.n_train:],
        Nhw=Nhw, frag=frag, mhw=mhw, cam=cam,
        met=st(sc["metallic_np"]), rou=st(sc["roughness_np"]), alb=st(sc["albedo_np"]),
        M=M, res=N.shape[0], P=M * 5 + args.n_train * 27)


def run(inp, args, n_iter, max_iter, img_batch=0, log_every=10 ** 9,
        history=None, with_val=False, timed=False):
    """One real _optimize_ct_sh. Returns wall/closures/peak-mem/final-loss (or OOM)."""
    cfg = {**DEFAULT_CFG, "optimizer": "LBFGS", "n_iter": n_iter,
           "lbfgs_max_iter": max_iter, "log_every": log_every, "loss": "L2",
           "sh_order": 2, "double": args.double, "tr_albedo": "sigmoid",
           "tr_metallic": "sigmoid", "tr_roughness": "sigmoid",
           "init_roughness_zero": True, "lambda_tv": 0.0,
           "lambda_metallic_binarize": 0.0, "img_batch": img_batch}
    _HIST_OVERRIDE[0] = history
    S._opt_step = _timed_opt_step if timed else _ORIG_OPT_STEP
    T.reset(); _CLO["n"] = 0
    if args.device == "cuda":
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    sync(args.device); t0 = time.perf_counter()
    try:
        out = _optimize_ct_sh(
            inp["train_imgs"], inp["Nhw"], inp["frag"], inp["mhw"], inp["cam"],
            inp["met"], inp["rou"], cfg, gt_sh_coeffs=inp["train_sh"], gt_albedo=inp["alb"],
            val_images=inp["val_imgs"] if with_val else None,
            val_sh_coeffs=inp["val_sh"] if with_val else None)
        sync(args.device)
        res = dict(wall=time.perf_counter() - t0, oom=False,
                   peak_gb=(torch.cuda.max_memory_allocated() / 1e9
                            if args.device == "cuda" else 0.0),
                   final_loss=float(out[5][-1]), history=out[5])
    except OOM as e:
        if "out of memory" not in str(e).lower():
            raise
        res = dict(wall=float("nan"), oom=True, peak_gb=float("nan"),
                   final_loss=float("nan"), history=[])
        torch.cuda.empty_cache()
    finally:
        S._opt_step = _ORIG_OPT_STEP
        _HIST_OVERRIDE[0] = None
    res["closures"] = _CLO["n"]                 # global count: works timed or untimed
    res["phases"] = dict(fwd=T.fwd, bwd=T.bwd, step=T.step, extra=T.extra,
                         nclo=T.nclo or _CLO["n"], nout=T.nout)
    return res


def hdr(t):
    print(f"\n{'=' * 78}\n{t}\n{'=' * 78}", flush=True)


# ───────────────────────────── sections ──────────────────────────────────────
def sec_breakdown(inp, args, R):
    hdr("0. TASK BREAKDOWN — where the wall time goes")
    r = run(inp, args, args.breakdown_outer, args.max_iter,
            img_batch=args.breakdown_img_batch, timed=True)
    if r["oom"]:
        print(f"  OOM at img_batch={args.breakdown_img_batch}. Re-run with "
              f"--breakdown_img_batch <n> (see section 2 for what fits).")
        return
    p, wall = r["phases"], r["wall"]
    # The phase timings only exist if S._opt_step was actually swapped for the timed
    # wrapper. That patch works by rebinding a module attribute, so it is silently
    # defeated if a caller does `from idr.optim.steps import _opt_step` (binding the
    # original at import time) instead of calling steps._opt_step(). When that happened
    # the profiler happily reported 0 ms forward over 46 closures, so fail loudly here.
    if p["nclo"] and p["fwd"] <= 0.0:
        raise SystemExit(
            f"phase timing is dead: {p['nclo']} closures ran but forward measured "
            f"{p['fwd']:.3f}s.\n"
            f"    idr.optim.steps._opt_step was patched but the model did not go through "
            f"it.\n    Check that idr/optim/models/*.py call `steps._opt_step(...)` rather "
            f"than importing\n    the name directly.")
    internal = p["step"] - p["fwd"] - p["bwd"]
    misc = wall - p["step"] - p["extra"]
    rows = [("forward  (in closure)", p["fwd"], p["nclo"]),
            ("backward (in closure)", p["bwd"], p["nclo"]),
            ("LBFGS internal (history/line-search)", internal, p["nout"]),
            ("extra reporting forward (no_grad)", p["extra"], p["nout"]),
            ("setup + final shadings + misc", misc, 1)]
    print(f"  {args.breakdown_outer} outer x max_iter={args.max_iter}, "
          f"img_batch={args.breakdown_img_batch or 'full'}\n")
    print(f"  {'task':38} {'sec':>8} {'% wall':>7} {'calls':>7} {'ms/call':>10}")
    for n, v, c in rows:
        print(f"  {n:38} {v:8.2f} {100 * v / wall:7.1f} {c:7d} {v / max(c, 1) * 1000:10.1f}")
    print(f"  {'TOTAL wall':38} {wall:8.2f} {100.0:7.1f}")
    print(f"\n  closures={p['nclo']} ({p['nclo'] / max(p['nout'],1):.1f}/outer)   "
          f"backward/forward={p['bwd'] / max(p['fwd'], 1e-9):.2f}x   "
          f"fwd+bwd={100 * (p['fwd'] + p['bwd']) / wall:.0f}% of wall")
    R["breakdown"] = dict(wall=wall, fwd=p["fwd"], bwd=p["bwd"], lbfgs_internal=internal,
                          extra_forward=p["extra"], misc=misc, closures=p["nclo"],
                          outer=p["nout"])


def sec_downsample(args, R):
    """Cost vs resolution. Must re-load inputs per factor: load_inputs() strides the
    images/maps and builds the proxy geometry at args.downsample, so the working set is
    baked in and cannot be resized after the fact."""
    hdr("0b. DOWNSAMPLE — cost vs resolution")
    rows = []
    for d in args.downsamples:
        a2 = copy.copy(args)                     # shallow: run() only reads device/double
        a2.downsample = d
        inp2 = None
        try:
            inp2 = load_inputs(a2, quiet=True)
            res, M, P = inp2["res"], inp2["M"], inp2["P"]
            r = run(inp2, a2, args.ds_outer, args.ds_max_iter,
                    img_batch=args.breakdown_img_batch, timed=True)
            if r["oom"]:
                print(f"  ds={d:<2} {res:>4}^2  OOM at img_batch="
                      f"{args.breakdown_img_batch or 'full'}", flush=True)
                rows.append(dict(downsample=d, res=res, masked_px=M, P=P, oom=True))
                continue
            p = r["phases"]
            rows.append(dict(
                downsample=d, res=res, masked_px=M, P=P, oom=False,
                wall=r["wall"], s_per_outer=r["wall"] / args.ds_outer,
                closures=r["closures"],
                ms_per_closure=r["wall"] / max(r["closures"], 1) * 1000,
                fwd_bwd_s=p["fwd"] + p["bwd"],
                lbfgs_internal_s=p["step"] - p["fwd"] - p["bwd"],
                peak_gb=r["peak_gb"], final_loss=r["final_loss"]))
        finally:
            del inp2                              # free the GPU working set before the
            if args.device == "cuda":             # next (possibly larger) resolution
                torch.cuda.empty_cache()

    ok = [x for x in rows if not x["oom"]]
    if not ok:
        print("  every resolution OOM'd — lower --breakdown_img_batch")
        R["downsample"] = rows
        return

    print(f"  {args.ds_outer} outer x max_iter={args.ds_max_iter}, "
          f"img_batch={args.breakdown_img_batch or 'full'}, "
          f"{args.n_train} train images\n")
    print(f"  {'ds':>3} {'res':>7} {'masked px':>10} {'params':>9} {'closures':>9} "
          f"{'wall s':>8} {'ms/clo':>8} {'peak GB':>8} {'fwd+bwd%':>9}")
    for x in ok:
        print(f"  {x['downsample']:>3} {str(x['res'])+'^2':>7} {x['masked_px']:>10,} "
              f"{x['P']:>9,} {x['closures']:>9} {x['wall']:>8.2f} "
              f"{x['ms_per_closure']:>8.1f} {x['peak_gb']:>8.3f} "
              f"{100*x['fwd_bwd_s']/x['wall']:>9.1f}")

    # Scale relative to the CHEAPEST (largest ds), and compare against the pixel ratio.
    base = max(ok, key=lambda x: x["downsample"])
    print(f"\n  relative to ds={base['downsample']} ({base['res']}^2):")
    print(f"  {'ds':>3} {'px ratio':>9} {'time ratio':>11} {'mem ratio':>10} {'efficiency':>11}")
    for x in ok:
        px = x["masked_px"] / base["masked_px"]
        tr = x["ms_per_closure"] / base["ms_per_closure"]
        mr = x["peak_gb"] / base["peak_gb"] if base["peak_gb"] else float("nan")
        print(f"  {x['downsample']:>3} {px:>9.2f}x {tr:>10.2f}x {mr:>9.2f}x "
              f"{px/tr:>10.2f}x")

    # log-log fit: ms/closure ~ M^alpha. alpha~1 => compute-bound (time tracks pixels),
    # alpha<<1 => launch/overhead-bound (small resolutions cost far more than their pixels).
    if len(ok) >= 2:
        lm = np.log([x["masked_px"] for x in ok])
        lt = np.log([x["ms_per_closure"] for x in ok])
        alpha = float(np.polyfit(lm, lt, 1)[0])
        verdict = ("compute-bound: cost tracks pixel count, so downsampling buys "
                   "close to its full quadratic saving" if alpha > 0.8 else
                   "OVERHEAD-bound: cost grows far slower than pixels, so the small "
                   "resolutions are launch-limited and downsampling saves much less "
                   "than the {}x pixel reduction suggests".format(
                       f"{ok[0]['masked_px']/base['masked_px']:.0f}"))
        print(f"\n  ms/closure ~ masked_px^{alpha:.2f}  ->  {verdict}")
        R_alpha = alpha
    else:
        R_alpha = None

    full = min(ok, key=lambda x: x["downsample"])
    if full is not base:
        print(f"  ds={full['downsample']} costs {full['ms_per_closure']/base['ms_per_closure']:.1f}x "
              f"the time and {full['peak_gb']/max(base['peak_gb'],1e-9):.1f}x the memory "
              f"of ds={base['downsample']}")
    print("  NOTE: params P scale with masked px too, so the LBFGS two-loop recursion "
          "grows with\n        resolution as well — this is not purely a shading cost.")
    R["downsample"] = dict(rows=rows, alpha=R_alpha, base_ds=base["downsample"])


def sec_closures(inp, args, R):
    hdr("1. CLOSURE COST — the n_iter x max_iter multiplier")
    print(f"  {'max_iter':>9} {'closures':>9} {'per outer':>10} {'ls factor':>10} "
          f"{'wall s':>8} {'ms/closure':>11}")
    rows = []
    for mi in args.max_iters:
        r = run(inp, args, args.closure_outer, mi, img_batch=args.breakdown_img_batch)
        if r["oom"]:
            print(f"  {mi:>9} {'OOM':>9}"); continue
        per = r["closures"] / args.closure_outer
        rows.append(dict(max_iter=mi, closures=r["closures"], per_outer=per,
                         ls_factor=per / mi, wall=r["wall"],
                         ms_per_closure=r["wall"] / max(r["closures"], 1) * 1000))
        print(f"  {mi:>9} {r['closures']:>9} {per:>10.1f} {per/mi:>10.2f} "
              f"{r['wall']:>8.2f} {rows[-1]['ms_per_closure']:>11.1f}")
    if not rows:
        return
    ls = float(np.mean([x["ls_factor"] for x in rows]))
    ms = float(np.median([x["ms_per_closure"] for x in rows]))
    tot = args.study_n_iter * args.study_max_iter * ls
    print(f"\n  line search adds ~{ls:.2f}x on top of max_iter")
    print(f"  => study config (n_iter={args.study_n_iter}, max_iter={args.study_max_iter}) "
          f"= ~{tot:,.0f} fwd+bwd passes/run")
    print(f"  => at {ms:.0f} ms/closure that is ~{tot * ms / 1000 / 60:.1f} min/run "
          f"of pure optimization")
    R["closures"] = dict(rows=rows, ls_factor=ls, ms_per_closure=ms,
                         study_closures=tot, study_min_per_run=tot * ms / 1000 / 60)


def sec_plateau(inp, args, R):
    hdr("1b. CONVERGENCE — is the closure budget wasted?")
    r = run(inp, args, args.plateau_outer, args.max_iter,
            img_batch=args.breakdown_img_batch, log_every=1)
    if r["oom"] or not r["history"]:
        print("  OOM / no history"); return
    h = np.asarray(r["history"], float)
    cpo = r["closures"] / args.plateau_outer
    final = h[-1]
    print(f"  {r['closures']:,} closures over {args.plateau_outer} outer steps "
          f"-> final loss {final:.4e}\n")
    print(f"  {'% of budget':>12} {'closures':>10} {'loss':>12} {'x final':>9}")
    for f in (0.05, 0.10, 0.25, 0.50, 1.00):
        i = min(int(f * (len(h) - 1)), len(h) - 1)
        print(f"  {f:>11.0%} {i * cpo:>10,.0f} {h[i]:>12.4e} {h[i] / final:>9.2f}")
    still = h[int(0.9 * (len(h) - 1))] / final
    verdict = ("still descending — plateau NOT reached, run longer to size n_iter"
               if still > 1.02 else "flat — the tail of the budget is wasted")
    print(f"\n  loss over the LAST 10% of the budget improved {still:.3f}x ({verdict})")
    R["plateau"] = dict(closures=r["closures"], final_loss=final,
                        last10pct_ratio=float(still),
                        loss_curve=[float(v) for v in h])


def sec_img_batch(inp, args, R):
    hdr("2. IMG_BATCH — bigger batch vs memory (the main hypothesis)")
    K = len(inp["train_imgs"])
    cands = [b for b in args.img_batches if 0 < b < K] + [0]
    print(f"  K={K} train images. img_batch=0 means one full-batch pass.\n")
    print(f"  {'img_batch':>10} {'chunks':>7} {'wall s':>8} {'s/outer':>9} "
          f"{'peak GB':>9} {'vs full':>8}")
    rows = []
    for b in cands:
        r = run(inp, args, args.batch_outer, args.batch_max_iter, img_batch=b)
        ch = 1 if b == 0 else int(np.ceil(K / b))
        lbl = "full" if b == 0 else str(b)
        if r["oom"]:
            print(f"  {lbl:>10} {ch:>7} {'OOM':>8}")
            rows.append(dict(img_batch=lbl, chunks=ch, oom=True)); continue
        rows.append(dict(img_batch=lbl, chunks=ch, wall=r["wall"],
                         s_per_outer=r["wall"] / args.batch_outer, peak_gb=r["peak_gb"],
                         oom=False))
        print(f"  {lbl:>10} {ch:>7} {r['wall']:>8.2f} "
              f"{r['wall']/args.batch_outer:>9.2f} {r['peak_gb']:>9.3f}", flush=True)
    ok = [x for x in rows if not x.get("oom")]
    if ok:
        best = min(ok, key=lambda x: x["s_per_outer"])
        worst = max(ok, key=lambda x: x["s_per_outer"])
        for x in ok:
            x["vs_best"] = x["s_per_outer"] / best["s_per_outer"]
        print(f"\n  fastest: img_batch={best['img_batch']}  "
              f"({best['s_per_outer']:.2f}s/outer, {best['peak_gb']:.2f} GB peak)")
        print(f"  slowest ({worst['img_batch']}) is {worst['s_per_outer']/best['s_per_outer']:.2f}x "
              f"slower for {worst['peak_gb']:.2f} GB — chunking is launch overhead, "
              f"not a memory win")
        R["img_batch"] = dict(rows=rows, best=best["img_batch"],
                              best_peak_gb=best["peak_gb"])


def sec_history(inp, args, R):
    hdr("3. LBFGS history_size — launch-bound two-loop recursion")
    print(f"  {'history':>8} {'wall s':>8} {'lbfgs int s':>12} {'% wall':>7} {'final loss':>13}")
    rows = []
    for h in args.history_sizes:
        r = run(inp, args, args.history_outer, args.max_iter,
                img_batch=args.breakdown_img_batch, history=h, timed=True)
        if r["oom"]:
            print(f"  {h:>8} {'OOM':>8}"); continue
        p = r["phases"]; internal = p["step"] - p["fwd"] - p["bwd"]
        rows.append(dict(history=h, wall=r["wall"], internal=internal,
                         pct=100 * internal / r["wall"], final_loss=r["final_loss"]))
        print(f"  {h:>8} {r['wall']:>8.2f} {internal:>12.2f} "
              f"{100*internal/r['wall']:>7.1f} {r['final_loss']:>13.4e}", flush=True)
    if rows:
        base = rows[0]
        print(f"\n  {'history':>8} {'speedup vs ' + str(base['history']):>18} {'loss delta':>13}")
        for x in rows:
            print(f"  {x['history']:>8} {base['wall']/x['wall']:>18.2f}x "
                  f"{x['final_loss']/base['final_loss']:>12.4f}x")
        print("\n  pick the smallest history whose final loss is unchanged")
        R["history"] = rows


def sec_logging(inp, args, R):
    hdr("4. LOGGING — cost of a logged step (recomputes held-out relight)")
    n = args.log_outer
    a = run(inp, args, n, args.max_iter, img_batch=args.breakdown_img_batch,
            log_every=10 ** 9, with_val=True)
    b = run(inp, args, n, args.max_iter, img_batch=args.breakdown_img_batch,
            log_every=1, with_val=True)
    if a["oom"] or b["oom"]:
        print("  OOM"); return
    per = (b["wall"] - a["wall"]) / n
    print(f"  {len(inp['val_imgs'])} val images")
    print(f"  log_every=inf : {a['wall']:.2f}s")
    print(f"  log_every=1   : {b['wall']:.2f}s")
    if per <= 0 or per * 1000 < 5:
        print(f"  => ~{per*1000:.0f} ms per logged step — within run-to-run noise here; "
              f"re-measure at the real size (val relight scales with resolution x n_val)")
    else:
        n_logged = max(args.study_n_iter // max(args.study_log_every, 1), 1)
        print(f"  => ~{per*1000:.0f} ms per logged step")
        print(f"  at the study's log_every={args.study_log_every} over {args.study_n_iter} "
              f"iters = {n_logged} logged steps -> ~{per * n_logged:.1f} s/run "
              f"({100 * per * n_logged / max(R.get('closures', {}).get('study_min_per_run', 1) * 60, 1e-9):.1f}% "
              f"of the optimize time)")
    R["logging"] = dict(per_logged_step_s=per, no_log_s=a["wall"], log_s=b["wall"])


def sec_workers(inp, args, R):
    hdr("5. WORKERS — does parallelism help on ONE GPU?")
    print("  NOTE: each worker holds its own CUDA context + full working set.\n")
    script = tempfile.mktemp(suffix="_pw.py")
    Path(script).write_text(f"""
import sys, os, time, numpy as np, torch
os.environ['WANDB_MODE']='disabled'
sys.path.insert(0, r'{Path(__file__).resolve().parent}')
from pathlib import Path
from idr.data.geometry import make_proxy_geometry
from idr.data.scene_io import load_scene
from idr.config import DEFAULT_CFG
from idr.optim.models.ct_sh import _optimize_ct_sh
ds, K = {args.downsample}, {args.n_train}
sc = load_scene(Path(r'{args.scene}'), gt_npy={not args.no_gt_npy})
s = lambda a: np.ascontiguousarray(a[::ds, ::ds])
Nhw, frag, mhw, cam = make_proxy_geometry(s(sc['normals_np']), s(sc['mask_np']),
                                          60, 2, 'cuda', torch.float32)
cfg = {{**DEFAULT_CFG, 'optimizer':'LBFGS','n_iter':{args.worker_outer},
       'lbfgs_max_iter':{args.batch_max_iter},'log_every':10**9,'loss':'L2','sh_order':2,
       'double':{args.double},'tr_albedo':'sigmoid','tr_metallic':'sigmoid',
       'tr_roughness':'sigmoid','init_roughness_zero':True,'lambda_tv':0,
       'lambda_metallic_binarize':0,'img_batch':{args.breakdown_img_batch}}}
_optimize_ct_sh([s(i) for i in sc['images'][:K]], Nhw, frag, mhw, cam,
                s(sc['metallic_np']), s(sc['roughness_np']), cfg,
                gt_sh_coeffs=[np.asarray(x,np.float32) for x in sc['sh_coeffs'][:K]],
                gt_albedo=s(sc['albedo_np']))
torch.cuda.synchronize()
""")
    rows = []
    for n in args.workers:
        t0 = time.perf_counter()
        ps = [subprocess.Popen([sys.executable, script], stdout=subprocess.DEVNULL,
                               stderr=subprocess.PIPE,
                               cwd=str(Path(__file__).resolve().parent)) for _ in range(n)]
        fails = sum(1 for p in ps if p.wait() != 0)
        el = time.perf_counter() - t0
        tp = (n - fails) / el * 60
        rows.append(dict(workers=n, wall=el, ok=n - fails, runs_per_min=tp))
        print(f"  workers={n}: {el:7.1f}s  {n-fails}/{n} ok  -> {tp:5.2f} runs/min"
              + ("   (some FAILED — likely GPU OOM)" if fails else ""), flush=True)
    os.unlink(script)
    if rows:
        b = rows[0]["runs_per_min"]
        print(f"\n  {'workers':>8} {'runs/min':>10} {'speedup':>9}")
        for x in rows:
            print(f"  {x['workers']:>8} {x['runs_per_min']:>10.2f} "
                  f"{x['runs_per_min']/b:>8.2f}x")
        best = max(rows, key=lambda x: x["runs_per_min"])
        print(f"\n  best throughput at workers={best['workers']} "
              f"({best['runs_per_min']/b:.2f}x vs 1)")
        R["workers"] = rows


def sec_summary(args, R):
    hdr("6. SUMMARY / RECOMMENDED CONFIG")
    bd = R.get("breakdown")
    if bd:
        w = bd["wall"]
        print(f"  fwd+bwd is {100*(bd['fwd']+bd['bwd'])/w:.0f}% of wall "
              f"(bwd {100*bd['bwd']/w:.0f}%, fwd {100*bd['fwd']/w:.0f}%), "
              f"LBFGS internal {100*bd['lbfgs_internal']/w:.0f}%")
    cl = R.get("closures")
    if cl:
        print(f"  study budget = ~{cl['study_closures']:,.0f} fwd+bwd passes "
              f"(~{cl['study_min_per_run']:.1f} min/run at {cl['ms_per_closure']:.0f} ms each)")
    pl = R.get("plateau")
    if pl:
        verdict = ("STILL DESCENDING — do not cut n_iter blindly"
                   if pl["last10pct_ratio"] > 1.02
                   else "FLAT at the end — n_iter can be cut substantially")
        print(f"  convergence: {verdict} "
              f"(last 10% of budget improved {pl['last10pct_ratio']:.3f}x)")
    ds = R.get("downsample")
    if ds:
        ok = [x for x in ds["rows"] if not x["oom"]]
        if ok and ds.get("alpha") is not None:
            print(f"  resolution: ms/closure ~ masked_px^{ds['alpha']:.2f} "
                  + ("(compute-bound — downsampling pays off near-quadratically)"
                     if ds["alpha"] > 0.8 else
                     "(overhead-bound — downsampling saves less than the pixel count implies)"))
        # project the study budget onto each resolution (needs section 1's closure count)
        if ok and cl:
            print(f"  projected study run time (n_iter={args.study_n_iter}, "
                  f"max_iter={args.study_max_iter} = {cl['study_closures']:,.0f} closures):")
            for x in ok:
                print(f"    ds={x['downsample']} ({x['res']}^2): "
                      f"~{cl['study_closures'] * x['ms_per_closure'] / 1000 / 60:6.1f} min/run  "
                      f"(peak {x['peak_gb']:.2f} GB)")
    ib = R.get("img_batch")
    if ib:
        print(f"  img_batch  -> use {ib['best']}  (peak {ib['best_peak_gb']:.2f} GB)")
    hs = R.get("history")
    if hs:
        same = [x for x in hs if abs(x["final_loss"] / hs[0]["final_loss"] - 1) < 1e-3]
        if same:
            print(f"  history_size -> {min(x['history'] for x in same)} "
                  f"(smallest with an unchanged final loss)")
    wk = R.get("workers")
    if wk:
        best = max(wk, key=lambda x: x["runs_per_min"])
        print(f"  workers    -> {best['workers']} "
              f"({best['runs_per_min']/wk[0]['runs_per_min']:.2f}x vs 1 worker)")
    print("\n  Also check `_make_optimizer` in idr/optim/steps.py:")
    print("    tolerance_grad=0 / tolerance_change=0 DISABLE LBFGS early stopping, so every")
    print("    outer step burns the full max_iter inner iterations even after convergence.")


# ───────────────────────────── main ──────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scene",
                   default=r"results/3dfront-batch/datasets/1f19c3ef_v2")
                #    default=r"E:\DLVC-backups\260720_results\3dfront-batch\datasets"
                #            r"\1f19c3ef_v2\ct-ct_sh-frOn_env")
    p.add_argument("--downsample", type=int, default=1, help="1 = full 512^2")
    p.add_argument("--n_train", type=int, default=100)
    p.add_argument("--n_val", type=int, default=28)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--double", action="store_true", help="fp64 (study uses fp32)")
    p.add_argument("--no_gt_npy", action="store_true",
                   help="Load the quantized PNG GT maps instead of the lossless .npy ones.")
    p.add_argument("--dataset_name", default="ct-ct_sh-frOn_env",
                   help="Which dataset leaf to pick when --scene is a datasets root/view dir.")
    p.add_argument("--out", default="profile_results.json")
    # the config being profiled, for extrapolation
    p.add_argument("--study_n_iter", type=int, default=500)
    p.add_argument("--study_max_iter", type=int, default=40)
    p.add_argument("--study_log_every", type=int, default=50)
    # budgets (kept small: one closure at 512^2 x 100 imgs is ~1s)
    p.add_argument("--max_iter", type=int, default=20, help="max_iter for most sections")
    p.add_argument("--breakdown_outer", type=int, default=3)
    p.add_argument("--breakdown_img_batch", type=int, default=0, help="0 = full batch")
    p.add_argument("--downsamples", type=int, nargs="+", default=[1, 2, 4],
                   help="downsample factors to compare (1/2/4 = 512/256/128 at native 512)")
    p.add_argument("--ds_outer", type=int, default=2)
    p.add_argument("--ds_max_iter", type=int, default=10)
    p.add_argument("--closure_outer", type=int, default=3)
    p.add_argument("--max_iters", type=int, nargs="+", default=[5, 10, 20, 40])
    p.add_argument("--plateau_outer", type=int, default=40)
    p.add_argument("--batch_outer", type=int, default=2)
    p.add_argument("--batch_max_iter", type=int, default=10)
    p.add_argument("--img_batches", type=int, nargs="+", default=[4, 8, 16, 32, 50])
    p.add_argument("--history_outer", type=int, default=3)
    p.add_argument("--history_sizes", type=int, nargs="+", default=[100, 50, 20, 10])
    p.add_argument("--log_outer", type=int, default=5)
    p.add_argument("--worker_outer", type=int, default=3)
    p.add_argument("--workers", type=int, nargs="+", default=[1, 2])
    # section switches
    p.add_argument("--skip", nargs="*", default=["workers"],
                   choices=["breakdown", "downsample", "closures", "plateau", "img_batch",
                            "history", "logging", "workers"],
                   help="sections to skip (workers is skipped by default: it spawns "
                        "full extra processes and can OOM the GPU at 512^2)")
    p.add_argument("--quick", action="store_true",
                   help="tiny budgets for a fast smoke test")
    args = p.parse_args()

    if args.quick:
        args.breakdown_outer = args.closure_outer = args.history_outer = 2
        args.plateau_outer, args.log_outer, args.batch_outer = 6, 2, 1
        args.max_iters = [5, 10]; args.history_sizes = [100, 10]
        args.img_batches = [8, 32]; args.workers = [1, 2]
        args.ds_outer, args.ds_max_iter = 1, 5

    scene = Path(args.scene)
    if not scene.exists():
        raise SystemExit(f"scene not found: {scene}")
    # Convenience: accept a datasets ROOT or a view dir and descend to a dataset leaf
    # (a leaf is what holds light_*.npy). Lets you point at the tree that
    # batch_dataset_decomposition.ipynb produced without naming the leaf.
    if not any(scene.glob("light_*.npy")) and not any(scene.glob("light_*.png")):
        cand = sorted(p for p in scene.rglob(args.dataset_name)
                      if p.is_dir() and any(p.glob("light_*.npy")))
        if not cand:
            raise SystemExit(f"no '{args.dataset_name}' dataset leaf with light_*.npy "
                             f"under {scene}")
        scene = cand[0]
        print(f"(descended to dataset leaf: {scene})")
    args.scene = str(scene)

    print(f"scene   : {args.scene}")
    if args.device == "cuda":
        d = torch.cuda.get_device_properties(0)
        print(f"GPU     : {d.name}  {d.total_memory/1e9:.1f} GB   CPUs: {os.cpu_count()}")
    inp = load_inputs(args)
    print(f"config  : {inp['res']}^2  M={inp['M']:,} masked px  "
          f"{len(inp['train_imgs'])} train / {len(inp['val_imgs'])} val  "
          f"P={inp['P']:,} params  dtype={'fp64' if args.double else 'fp32'}")

    # calibrate: one cheap run -> per-closure cost -> runtime estimate
    print("\ncalibrating (warm-up + 1 timed run) …", flush=True)
    warm = run(inp, args, 1, 3, img_batch=args.breakdown_img_batch)
    if warm["oom"]:
        print(f"  ! OOM even at the warm-up with img_batch={args.breakdown_img_batch}.\n"
              f"    Re-run with e.g. --breakdown_img_batch 25 (or a larger --downsample).")
        sys.exit(1)
    ms = warm["wall"] / max(warm["closures"], 1) * 1000
    todo = set(["breakdown", "downsample", "closures", "plateau", "img_batch", "history",
                "logging", "workers"]) - set(args.skip)
    est = 0.0
    if "breakdown" in todo: est += args.breakdown_outer * args.max_iter * 1.3
    if "downsample" in todo:
        # ms was calibrated at args.downsample; assume cost ~ pixel count for the estimate
        est += (args.ds_outer * args.ds_max_iter * 1.3
                * sum((args.downsample / d) ** 2 for d in args.downsamples))
    if "closures" in todo:  est += args.closure_outer * sum(args.max_iters) * 1.3
    if "plateau" in todo:   est += args.plateau_outer * args.max_iter * 1.3
    if "img_batch" in todo: est += args.batch_outer * args.batch_max_iter * 1.3 * (len(args.img_batches) + 1) * 2
    if "history" in todo:   est += args.history_outer * args.max_iter * 1.3 * len(args.history_sizes)
    if "logging" in todo:   est += 2 * args.log_outer * args.max_iter * 1.3
    print(f"  ~{ms:.0f} ms per fwd+bwd closure  (peak {warm['peak_gb']:.2f} GB)")
    print(f"  sections: {', '.join(sorted(todo))}")
    print(f"  rough estimate: ~{est * ms / 1000 / 60:.0f} min "
          f"(workers section excluded from estimate)")

    R = dict(meta=dict(scene=args.scene, res=inp["res"], masked_px=inp["M"],
                       n_train=len(inp["train_imgs"]), n_val=len(inp["val_imgs"]),
                       P=inp["P"], device=args.device, fp64=args.double,
                       ms_per_closure=ms,
                       gpu=(torch.cuda.get_device_properties(0).name
                            if args.device == "cuda" else "cpu"),
                       cpus=os.cpu_count()))
    t0 = time.perf_counter()
    if "breakdown" in todo: sec_breakdown(inp, args, R)
    if "downsample" in todo: sec_downsample(args, R)
    if "closures" in todo:  sec_closures(inp, args, R)
    if "plateau" in todo:   sec_plateau(inp, args, R)
    if "img_batch" in todo: sec_img_batch(inp, args, R)
    if "history" in todo:   sec_history(inp, args, R)
    if "logging" in todo:   sec_logging(inp, args, R)
    if "workers" in todo:   sec_workers(inp, args, R)
    sec_summary(args, R)
    R["meta"]["total_profile_s"] = time.perf_counter() - t0
    Path(args.out).write_text(json.dumps(R, indent=1))
    print(f"\nprofile took {R['meta']['total_profile_s']/60:.1f} min -> {args.out}")


if __name__ == "__main__":
    main()
