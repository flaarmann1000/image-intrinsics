#!/usr/bin/env python
"""
golden.py — behaviour-preservation harness for the restructuring refactor.

A refactor that changes numbers is a failed refactor. This records reference outputs
BEFORE any restructuring and re-checks them after each stage.

Two layers, because they fail for different reasons:

  optimizer level  calls _optimize_{ct_sh,ct_env,phong_sh,phong_env} directly on fixed
                   inputs. Covers all four models — including the Phong pair, which
                   decompose_scene cannot reach — and isolates the optimizer from IO.
  pipeline level   runs decompose_scene end to end. Covers scene loading, downsampling,
                   metrics and artifact writing.

TOLERANCE. LBFGS on a GPU is not bitwise reproducible: reduction order varies between
launches. So `record` runs everything TWICE and stores the observed run-to-run delta as
the per-case noise floor. `check` then requires the refactored code to land within
max(noise_floor * SLACK, ATOL) — i.e. it must be no less reproducible than the original
code was against itself. A case whose two baseline runs agree bitwise gets ATOL, which
is tight.

Usage
  python tests/golden.py record            # once, on the pre-refactor tree
  python tests/golden.py check             # after each stage
  python tests/golden.py check --case A    # a single case
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
os.environ.setdefault("WANDB_MODE", "disabled")

GOLDEN_DIR = Path(__file__).resolve().parent / "golden"
# Scratch dataset synthesized by build_scene() — kept out of the repo tree.
SCENE = GOLDEN_DIR / "scene"

# Absolute floor: below this, two runs count as identical regardless of measured noise.
ATOL = 1e-9
# Metric keys that measure wall time rather than the computation's result.
TIMING_KEY = lambda k: k.endswith(("_s", "_time", "_sec")) or "elapsed" in k
# How much the refactor may exceed the baseline's own run-to-run spread.
SLACK = 4.0

# Small enough to run the whole matrix in ~2 min, large enough to exercise real code
# paths (multiple lights, real geometry, all regularizers live).
DOWNSAMPLE = 8
N_TRAIN = 12
N_VAL = 3
N_ITER = 8
MAX_ITER = 10


def _base_cfg(**over):
    from idr.config import DEFAULT_CFG
    cfg = {**DEFAULT_CFG,
           "optimizer": "LBFGS", "n_iter": N_ITER, "lbfgs_max_iter": MAX_ITER,
           "log_every": 10 ** 9, "loss": "L2", "double": False,
           "tr_albedo": "sigmoid", "tr_metallic": "sigmoid", "tr_roughness": "sigmoid",
           "init_roughness_zero": True, "lambda_tv": 1e-5,
           "lambda_metallic_binarize": 0.0, "sh_order": 2}
    cfg.update(over)
    return cfg


# ── synthetic scene ──────────────────────────────────────────────────────────
# The inputs are SYNTHESIZED rather than read from a dataset dir, deliberately: the
# harness then has no external-drive or dataset dependency and produces the same
# geometry on any machine (including the VM). It is rendered with the project's own
# shade_ct_sh, so the observations are exactly the forward model the optimizer inverts.
# RES is deliberately small. LM builds an (N_res x P) Jacobian and, below
# lm_dense_max_params, a dense P x P normal-equation solve; at RES=48 that is ~8.6k
# params and >12 GB resident. RES=32 keeps every code path live at a fraction of the
# cost — the goal is to exercise the branches, not to stress them.
RES = 32
SEED = 20260722


def _sphere_normals(res):
    """Unit sphere normal field + its mask — smooth, with real normal variation."""
    ys, xs = np.mgrid[0:res, 0:res].astype(np.float32)
    x = (xs - (res - 1) / 2) / (res / 2.4)
    y = ((res - 1) / 2 - ys) / (res / 2.4)
    r2 = x ** 2 + y ** 2
    mask = r2 < 0.92
    z = np.sqrt(np.clip(1.0 - r2, 0.0, None))
    n = np.stack([x, y, z], -1).astype(np.float32)
    n /= np.linalg.norm(n, axis=-1, keepdims=True).clip(1e-6)
    n[~mask] = 0.0
    return n, mask


def build_scene(device="cuda"):
    """Deterministic (images, geometry, GT maps, SH lights). Pure function of SEED."""
    from idr.data.geometry import make_proxy_geometry
    from idr.render import shade_ct_sh
    from idr.render.brdf import _get_ggx_sh_lut

    rng = np.random.default_rng(SEED)
    normals, mask = _sphere_normals(RES)

    # Spatially varying GT maps: constant maps would hide per-pixel indexing bugs.
    yy, xx = np.mgrid[0:RES, 0:RES].astype(np.float32) / RES
    albedo = np.stack([0.25 + 0.5 * xx, 0.25 + 0.5 * yy,
                       0.35 + 0.3 * ((xx + yy) % 0.5)], -1).astype(np.float32)
    rough = (0.15 + 0.6 * yy).astype(np.float32)[:, :, None]
    metal = (0.05 + 0.5 * xx).astype(np.float32)[:, :, None]

    Nhw, frag, mhw, cam = make_proxy_geometry(normals, mask, 60.0, 2.0, device,
                                              torch.float32)
    fm = mhw.reshape(-1)
    N_m = Nhw.reshape(-1, 3)[fm]
    V_m = torch.nn.functional.normalize(cam[None] - frag.reshape(-1, 3)[fm], dim=-1)
    to_m = lambda a, c: torch.from_numpy(np.ascontiguousarray(a)).to(
        device, torch.float32).reshape(-1, c)[fm]
    lut = _get_ggx_sh_lut(device, n_bands=3).to(torch.float32)

    # SH2 lights, order-2 zero-padded to 9 — matches the datasets' sh_*.npy layout.
    shs, images = [], []
    for _ in range(N_TRAIN + N_VAL):
        c = np.zeros((9, 3), np.float32)
        c[0] = rng.uniform(0.6, 1.6, 3)
        c[1:4] = rng.uniform(-0.5, 0.5, (3, 3))
        c[4:9] = rng.uniform(-0.25, 0.25, (5, 3))
        with torch.no_grad():
            px = shade_ct_sh(V_m, N_m, to_m(albedo, 3),
                             torch.from_numpy(c).to(device),
                             to_m(metal, 1), to_m(rough, 1), lut=lut)
        img = np.zeros((RES * RES, 3), np.float32)
        img[fm.cpu().numpy()] = px.float().cpu().numpy()
        images.append(img.reshape(RES, RES, 3))
        shs.append(c)

    return dict(normals=normals, mask=mask, albedo=albedo,
                roughness=rough, metallic=metal, images=images, sh=shs)


def write_scene(dest=SCENE):
    """Materialize the synthetic scene as a dataset dir for the pipeline case."""
    from PIL import Image
    dest.mkdir(parents=True, exist_ok=True)
    sc = build_scene()
    for i, (im, c) in enumerate(zip(sc["images"], sc["sh"])):
        np.save(dest / f"light_{i:03d}.npy", im.astype(np.float32))
        np.save(dest / f"sh_{i:03d}.npy", c.astype(np.float32))
    # load_scene(gt_npy=True) wants the .npy maps; the PNGs satisfy its fallback path
    # and keep the dir shaped like a real dataset.
    for name, arr in (("normals", sc["normals"]), ("albedo", sc["albedo"]),
                      ("roughness", sc["roughness"]), ("metallic", sc["metallic"])):
        np.save(dest / f"{name}.npy", arr.astype(np.float32))
    Image.fromarray(((sc["normals"] * 0.5 + 0.5) * 255).astype(np.uint8)).save(
        dest / "normals.png")
    Image.fromarray((sc["albedo"] * 255).astype(np.uint8)).save(dest / "albedo.png")
    for n in ("roughness", "metallic"):
        Image.fromarray((sc[n][:, :, 0] * 65535).astype(np.uint16)).save(dest / f"{n}.png")
    (dest / "config.json").write_text(json.dumps(
        {"variant": "exr", "prereduced_downsample": 1, "synthetic_golden": True,
         "seed": SEED, "resolution": RES}, indent=1))
    return dest


def load_inputs(device="cuda"):
    """Geometry + images shared by every optimizer-level case."""
    from idr.data.geometry import make_proxy_geometry
    sc = build_scene(device)
    Nhw, frag, mhw, cam = make_proxy_geometry(sc["normals"], sc["mask"], 60.0, 2.0,
                                              device, torch.float32)
    return dict(images=sc["images"][:N_TRAIN], sh=sc["sh"][:N_TRAIN],
                Nhw=Nhw, frag=frag, mhw=mhw, cam=cam,
                met=sc["metallic"], rou=sc["roughness"], alb=sc["albedo"])


def _env_geometry():
    from idr.render import EnvMap, SHLighting
    from idr.config import LIGHT_COLOR, LIGHT_INTENSITY
    ref = EnvMap.from_sh(SHLighting.directional(
        np.array([0, 0, 1], np.float32), LIGHT_COLOR, intensity=LIGHT_INTENSITY),
        resolution=32)
    return ref._dirs, ref._solid_angles, ref.image.shape[0], ref.image.shape[1]


# ── the case matrix ──────────────────────────────────────────────────────────
def case_ct_sh(inp, optimizer="LBFGS", sh_order=2, solver=None, n_img=None,
               varpro_space=None):
    from idr.optim.models.ct_sh import _optimize_ct_sh
    cfg = _base_cfg(optimizer=optimizer, sh_order=sh_order)
    if varpro_space is not None:
        # VarPro eliminates the lighting each iteration, so a handful of outer steps
        # already exercises the whole path (design -> active set -> Woodbury -> line
        # search); more only costs time.
        cfg.update(n_iter=4, varpro_space=varpro_space, varpro_chunk=2048)
    imgs, shs = inp["images"], inp["sh"]
    if optimizer == "LM":
        # 3 outer steps on a few images is enough to move every parameter through the
        # residual/Jacobian/solve path; more only costs time.
        cfg.update(n_iter=3, lm_batch_size=0, lm_jacobian_mode="forward",
                   lm_linear_solver=solver or "dense")
        if solver == "schur":
            # schur eliminates the block-diagonal per-pixel Hessian exactly, which
            # requires pixel-separable regularizers -- TV couples neighbours.
            cfg["lambda_tv"] = 0.0
        n_img = n_img or 4
    if n_img:
        imgs, shs = imgs[:n_img], shs[:n_img]
    a, light, ma, mb, _sh, hist, _t = _optimize_ct_sh(
        imgs, inp["Nhw"], inp["frag"], inp["mhw"], inp["cam"],
        inp["met"], inp["rou"], cfg,
        gt_sh_coeffs=shs, gt_albedo=inp["alb"])
    return dict(albedo=a, light=np.asarray(light), mat_a=ma, mat_b=mb,
                history=np.asarray(hist, np.float64))


def case_ct_env(inp):
    from idr.optim.models.ct_env import _optimize_ct_env
    d, dw, eH, eW = _env_geometry()
    a, light, ma, mb, _sh, hist, _t = _optimize_ct_env(
        inp["images"], inp["Nhw"], inp["frag"], inp["mhw"], inp["cam"],
        inp["met"], inp["rou"], env_dirs=d, env_dw=dw, cfg=_base_cfg(),
        env_H=eH, env_W=eW, gt_sh_coeffs=inp["sh"], gt_albedo=inp["alb"])
    return dict(albedo=a, light=np.asarray(light), mat_a=ma, mat_b=mb,
                history=np.asarray(hist, np.float64))


def case_phong_sh(inp):
    from idr.optim.models.phong_sh import _optimize_phong_sh
    a, light, ma, mb, _sh, hist, _t = _optimize_phong_sh(
        inp["images"], inp["Nhw"], inp["frag"], inp["mhw"], inp["cam"],
        gt_shininess=32.0, gt_ks=0.5, ka=0.0, kd=1.0, cfg=_base_cfg(),
        gt_sh_coeffs=None, gt_albedo=None)
    return dict(albedo=a, light=np.asarray(light), mat_a=ma, mat_b=mb,
                history=np.asarray(hist, np.float64))


def case_phong_env(inp):
    from idr.optim.models.phong_env import _optimize_phong_env
    d, dw, eH, eW = _env_geometry()
    a, light, ma, mb, _sh, hist, _t = _optimize_phong_env(
        inp["images"], inp["Nhw"], inp["frag"], inp["mhw"], inp["cam"],
        gt_shininess=32.0, gt_ks=0.5, ka=0.0, kd=1.0,
        env_dirs=d, env_dw=dw, cfg=_base_cfg(), env_H=eH, env_W=eW,
        gt_sh_coeffs=None, gt_albedo=None)
    return dict(albedo=a, light=np.asarray(light), mat_a=ma, mat_b=mb,
                history=np.asarray(hist, np.float64))


def case_pipeline(_inp):
    """Full decompose_scene: loading, downsample, metrics, artifacts."""
    import shutil
    import tempfile
    from idr.pipelines.decompose import decompose_scene
    if not (SCENE / "config.json").exists():
        write_scene()
    out = Path(tempfile.mkdtemp(prefix="golden_pipe_"))
    try:
        m = decompose_scene(
            SCENE, out,
            cfg_overrides=_base_cfg(downsample=1,
                                    n_images=N_TRAIN + N_VAL, val_images=N_VAL,
                                    gt_npy=True, use_npy=True),
            device="cuda", gt_npy=True, wandb_project="golden")
        res = {k: np.asarray(np.load(out / f"{k}.npy"))
               for k in ("albedo_est", "metallic_est", "roughness_est")}
        # Wall-clock keys (elapsed_s) are not reproducible by construction and would
        # inflate the tolerance for every other scalar, so they are excluded.
        keys = sorted(k for k, v in m.items()
                      if isinstance(v, (int, float)) and not isinstance(v, bool)
                      and not TIMING_KEY(k))
        res["metrics"] = np.array([float(m[k]) for k in keys], dtype=np.float64)
        return res
    finally:
        shutil.rmtree(out, ignore_errors=True)


CASES = {
    "A": ("ct_sh LBFGS SH2   (main path)", lambda i: case_ct_sh(i)),
    "B": ("ct_sh LM  dense   (lm residuals + dense normal eqs)",
          lambda i: case_ct_sh(i, optimizer="LM", solver="dense")),
    "C": ("ct_sh LM  cg      (lm matrix-free path, used at 512^2)",
          lambda i: case_ct_sh(i, optimizer="LM", solver="cg")),
    "I": ("ct_sh LM  schur   (exact per-pixel elimination)",
          lambda i: case_ct_sh(i, optimizer="LM", solver="schur")),
    "D": ("ct_sh LBFGS SH3   (order-3 basis + LUT band 3)", lambda i: case_ct_sh(i, sh_order=3)),
    "E": ("ct_env LBFGS      (env branch)", case_ct_env),
    "F": ("phong_sh LBFGS    (phong branch)", case_phong_sh),
    "G": ("phong_env LBFGS   (phong env branch)", case_phong_env),
    "J": ("ct_sh VARPRO natural      (lighting eliminated in closed form)",
          lambda i: case_ct_sh(i, optimizer="VARPRO", varpro_space="natural")),
    "K": ("ct_sh VARPRO transformed  (same, Jacobian chained through _fwd_*)",
          lambda i: case_ct_sh(i, optimizer="VARPRO", varpro_space="transformed")),
    "H": ("decompose_scene   (pipeline: IO+metrics+artifacts)", case_pipeline),
}


# ── compare ──────────────────────────────────────────────────────────────────
def _arrays(d):
    return {k: v for k, v in d.items() if isinstance(v, np.ndarray)}


def delta(a, b):
    """Max abs difference per key. Shape/keys mismatch is a hard failure."""
    out = {}
    ka, kb = set(_arrays(a)), set(_arrays(b))
    if ka != kb:
        raise AssertionError(f"key mismatch: only-ref={ka-kb} only-new={kb-ka}")
    for k in sorted(ka):
        x, y = np.asarray(a[k], np.float64), np.asarray(b[k], np.float64)
        if x.shape != y.shape:
            raise AssertionError(f"{k}: shape {x.shape} != {y.shape}")
        out[k] = float(np.abs(x - y).max()) if x.size else 0.0
    return out


def run_case(key, inp):
    t0 = time.perf_counter()
    r = CASES[key][1](inp)
    return r, time.perf_counter() - t0


def cmd_record(args):
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    inp = load_inputs()
    # Merge, don't overwrite: `record --case H` must not wipe the other cases' entries.
    np_ = GOLDEN_DIR / "noise.json"
    noise = json.loads(np_.read_text()) if np_.exists() else {}
    for key in args.cases:
        print(f"[{key}] {CASES[key][0]}", flush=True)
        r1, t1 = run_case(key, inp)
        r2, _ = run_case(key, inp)           # second, identical run -> noise floor
        d = delta(r1, r2)
        noise[key] = d
        np.savez(GOLDEN_DIR / f"{key}.npz", **_arrays(r1))
        worst = max(d.values()) if d else 0.0
        print(f"      {t1:5.1f}s  run-to-run noise: "
              + "  ".join(f"{k}={v:.2e}" for k, v in d.items()))
        if worst == 0.0:
            print("      bitwise reproducible -> ATOL will apply")
    (GOLDEN_DIR / "noise.json").write_text(json.dumps(noise, indent=1))
    print(f"\nrecorded {len(args.cases)} case(s) -> {GOLDEN_DIR}")


def cmd_check(args):
    noise = json.loads((GOLDEN_DIR / "noise.json").read_text())
    inp = load_inputs()
    bad = []
    for key in args.cases:
        gp = GOLDEN_DIR / f"{key}.npz"
        if not gp.exists():
            print(f"[{key}] SKIP (no golden recorded)"); continue
        ref = dict(np.load(gp))
        try:
            new, t = run_case(key, inp)
            d = delta(ref, new)
        except Exception as e:
            print(f"[{key}] FAIL {type(e).__name__}: {e}")
            bad.append(key); continue
        fails = []
        for k, v in d.items():
            tol = max(noise.get(key, {}).get(k, 0.0) * SLACK, ATOL)
            if v > tol:
                fails.append(f"{k}: {v:.3e} > tol {tol:.3e}")
        status = "OK  " if not fails else "FAIL"
        print(f"[{key}] {status} {t:5.1f}s  max_delta="
              f"{max(d.values()) if d else 0:.2e}   {CASES[key][0]}")
        for f in fails:
            print(f"        {f}")
        if fails:
            bad.append(key)
    print()
    if bad:
        print(f"REGRESSION in {len(bad)} case(s): {', '.join(bad)}")
        return 1
    print(f"all {len(args.cases)} case(s) match the golden baseline")
    return 0


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("cmd", choices=["record", "check"])
    p.add_argument("--case", nargs="*", dest="cases", default=list(CASES),
                   choices=list(CASES))
    args = p.parse_args()
    sys.exit(cmd_record(args) if args.cmd == "record" else cmd_check(args))


if __name__ == "__main__":
    main()
