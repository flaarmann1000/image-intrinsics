#!/usr/bin/env python
"""
scripts/convert_indoor_datasets.py — convert the `indoor_render_sh_opaque` dataset into the
canonical scene-dir structure (light_NNN.npy / sh_NNN.npy / GT .npy+.png / config.json) that
load_scene / decompose_scene consume, writing to local_datasets/canonical/.

Source per scene (flat dir):
    render_000.exr .. render_015.exr    16 LINEAR HDR observations (one per SH condition)
    albedo0001.exr normal0001.exr roughness0001.exr metallic0001.exr position0001.exr   GT
    lights.json    {"n":16, "sh":[(3,9) order-2 world env SH per condition, ...]}
    metadata.json  {"eye":[..], "resolution_x/y":.., "preset":.., note: view=normalize(eye-position)}

Convention bridging (the important part):
  The indoor maps are WORLD-space (+Z up), rendered by a REAL perspective camera and shipping
  a true per-pixel position map. The canonical format instead assumes a depthless PROXY camera
  at (0,0,+cam_dist) looking -Z, +Y up (idr.data.geometry.make_proxy_geometry), so both the
  normals AND the SH lighting must be rotated out of world space into that camera frame, or the
  specular term (which mixes normal, proxy view and lighting in one frame) is inconsistent.

    1. Recover the camera rotation R (world->cam) from the position map: the per-pixel world ray
       normalize(position - eye) is Kabsch-fit to the canonical NDC rays, sweeping the vertical
       FOV and keeping the best (residual is ~0.004 rad, i.e. the recovery is essentially exact).
    2. Normals: N_cam = R @ N_world.
    3. SH lighting: rotate by resample-and-refit through idr.render.sh.build_sh_basis — evaluate
       L_world at R^T-rotated sample directions and re-fit order-2 SH on the canonical basis, so
       the result is guaranteed to use idr's own SH convention (verified: the rotated diffuse
       still reconstructs the render, corr ~0.95).

Observations become light_NNN.npy = render/exr_scale (99th-pct normalisation, as the 3D-Front
exr variant); sh_NNN.npy = (rotated SH)/exr_scale, so shade_ct_sh(sh) matches the normalised
images. GT is written with the same encoding as build_canonical_datasets (float .npy + display
PNG, background zeroed on the |normal|>0.5 mask). The recovered camera (eye, fov, R, residual)
is recorded in config.json for provenance.

Re-runnable: a scene is rebuilt only when its render count changes or with --overwrite.

Usage:
    python scripts/convert_indoor_datasets.py                      # all scenes
    python scripts/convert_indoor_datasets.py --scenes bathroom_8289 --overwrite
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # repo root for `idr`
sys.path.insert(0, str(Path(__file__).resolve().parent))       # sibling scripts for reuse
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
os.environ.setdefault("WANDB_MODE", "disabled")

import numpy as np

from idr.paths import REPO_ROOT
from idr.data.scene_io import load_exr
from idr.render.sh import build_sh_basis
from build_canonical_datasets import write_gt, _stride    # identical GT encoding + striding

DEFAULT_ROOT = REPO_ROOT / "local_datasets" / "indoor_render_sh_opaque"
DEFAULT_OUT = REPO_ROOT / "local_datasets" / "canonical" / "INFINITE"
N_SH = 9                                                  # order-2

# Fixed sample directions for the SH resample-refit rotation (order-2 is exactly recoverable;
# 256 >> 9 samples keep the pseudo-inverse well conditioned). Deterministic across scenes.
_SH_DIRS = None


def _sh_sample_dirs():
    global _SH_DIRS
    if _SH_DIRS is None:
        g = np.random.default_rng(0).normal(size=(256, 3))
        _SH_DIRS = (g / np.linalg.norm(g, axis=1, keepdims=True)).astype(np.float32)
    return _SH_DIRS


# ── camera recovery ───────────────────────────────────────────────────────────
def _kabsch(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Rotation R (3x3) with B ≈ A @ R.T (maps rows of A onto rows of B)."""
    H = A.T @ B
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    return Vt.T @ np.diag([1.0, 1.0, d]) @ U.T


def recover_camera(position: np.ndarray, eye: np.ndarray, mask: np.ndarray):
    """Recover R_world_to_cam by fitting per-pixel world rays to the canonical NDC rays,
    sweeping the vertical FOV. Returns (R_wc, fov_deg, residual)."""
    H, W = position.shape[:2]
    dw = position - eye[None, None, :]
    dw /= np.clip(np.linalg.norm(dw, axis=-1, keepdims=True), 1e-6, None)
    ys = np.linspace(1.0, -1.0, H, dtype=np.float32)
    xs = np.linspace(-1.0, 1.0, W, dtype=np.float32)
    xg, yg = np.meshgrid(xs, ys)
    aspect = W / H
    A_w = dw[mask]

    def _fit(fov):
        t = np.tan(np.radians(fov / 2))
        dv = np.stack([xg * t * aspect, yg * t, -np.ones_like(xg)], -1)
        dv = (dv / np.linalg.norm(dv, axis=-1, keepdims=True))[mask]
        R_cw = _kabsch(dv, A_w)                           # dw ≈ dv @ R_cw.T
        return float(np.linalg.norm(A_w - dv @ R_cw.T, axis=1).mean()), R_cw

    # Coarse sweep over a wide FOV range (indoor cameras here span ~30–76 deg), then refine
    # around the best to 0.2 deg. Narrow FOVs sit well below a naive 40-deg floor.
    best = None
    for fov in np.arange(15.0, 140.1, 2.0):
        res, R_cw = _fit(fov)
        if best is None or res < best[0]:
            best = (res, fov, R_cw)
    for fov in np.arange(best[1] - 2.0, best[1] + 2.01, 0.2):
        res, R_cw = _fit(fov)
        if res < best[0]:
            best = (res, fov, R_cw)
    res, fov, R_cw = best
    return R_cw.T, float(fov), res                        # R_wc, fov, residual


# ── SH rotation (world -> cam), resample & refit on idr's basis ────────────────
def rotate_sh(sh_world_9x3: np.ndarray, R_wc: np.ndarray) -> np.ndarray:
    """Rotate order-2 SH from world into the camera frame. L_cam(w)=L_world(R_cw@w)."""
    dirs = _sh_sample_dirs()
    B_cam = build_sh_basis(dirs)                          # (M,9) canonical basis
    B_world = build_sh_basis((dirs @ R_wc).astype(np.float32))  # R_cw @ w = w @ R_wc
    return (np.linalg.pinv(B_cam) @ (B_world @ sh_world_9x3)).astype(np.float32)


# ── GT loading (world) ────────────────────────────────────────────────────────
# Normals below this length are treated as empty (true background). The renderer emits SHORT
# (anti-aliased) normals at texture / geometry edges — |n| ~ 0.1-0.5, ~2% of pixels on textured
# walls — which are VALID surface pixels (their position/albedo/render are all present). A
# |n|>0.5 threshold wrongly dropped them, leaving black speckle in the normal map; keep them and
# re-normalise, dropping only genuinely empty pixels (|n| ~ 0, of which these scenes have none).
_NORMAL_EPS = 0.1


def load_indoor_gt(scene_dir: Path) -> dict:
    """albedo/normals from EXR, roughness/metallic from EXR (single channel). World-space."""
    albedo = load_exr(scene_dir / "albedo0001.exr")[:, :, :3].astype(np.float32)
    normals = load_exr(scene_dir / "normal0001.exr")[:, :, :3].astype(np.float32)
    rough = load_exr(scene_dir / "roughness0001.exr")[:, :, :1].astype(np.float32)
    metal = load_exr(scene_dir / "metallic0001.exr")[:, :, :1].astype(np.float32)
    position = load_exr(scene_dir / "position0001.exr")[:, :, :3].astype(np.float32)
    nlen = np.linalg.norm(normals, axis=-1, keepdims=True)
    mask = nlen[:, :, 0] > _NORMAL_EPS
    # normalise masked normals to unit (the short AA-edge normals must be re-normalised, not
    # left at length ~0.3, or shading/geometry is wrong); empty pixels -> 0.
    normals = np.where(mask[:, :, None], normals / np.clip(nlen, 1e-6, None), 0.0).astype(np.float32)
    return dict(albedo=albedo, normals=normals, roughness=rough, metallic=metal,
                position=position, mask=mask)


# ── one scene ─────────────────────────────────────────────────────────────────
def _render_files(scene_dir: Path):
    return sorted(scene_dir.glob("render_*.exr"),
                  key=lambda p: int(p.stem.rsplit("_", 1)[-1]))


def _up_to_date(out_dir: Path, n_src: int) -> bool:
    cfg_p = out_dir / "config.json"
    if not cfg_p.exists():
        return False
    try:
        return int(json.loads(cfg_p.read_text()).get("n_lights", -1)) == n_src
    except Exception:
        return False


def convert_scene(scene_dir: Path, out_dir: Path, ds: int, hl_mode: str,
                  n_lights: int | None) -> int:
    meta = json.loads((scene_dir / "metadata.json").read_text())
    lights = json.loads((scene_dir / "lights.json").read_text())
    eye = np.asarray(meta["eye"], np.float32)
    sh_world = np.asarray(lights["sh"], np.float32)       # (n, 3, 9)

    gt = load_indoor_gt(scene_dir)
    R_wc, fov, res = recover_camera(gt["position"], eye, gt["mask"])
    if res > 0.05:
        print(f"      ! camera residual {res:.3f} (>0.05) — recovery may be poor")

    # Normals -> camera frame (rotation preserves length + the |n|>0.5 mask).
    gt["normals"] = (gt["normals"] @ R_wc.T).astype(np.float32)

    render_files = _render_files(scene_dir)
    if n_lights is not None:
        render_files = render_files[:n_lights]
    # Drop conditions whose GT SH is non-finite (HDRI/level-3), keeping renders aligned.
    keep = [i for i in range(len(render_files))
            if i < len(sh_world) and np.isfinite(sh_world[i]).all()]
    if len(keep) < len(render_files):
        print(f"      dropping {len(render_files) - len(keep)} condition(s) with non-finite SH")

    shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    # exr_scale over the KEPT renders (99th percentile of positive values), as the 3D-Front
    # exr variant. light = render / exr_scale.
    raw = [load_exr(render_files[i]).astype(np.float32) for i in keep]
    allpos = np.concatenate([im.reshape(-1) for im in raw]); allpos = allpos[allpos > 0]
    exr_scale = float(np.percentile(allpos, 99)) if allpos.size else 1.0

    for out_i, (src_i, im) in enumerate(zip(keep, raw)):
        img = (im / exr_scale).astype(np.float32)
        if ds > 1:
            img = np.ascontiguousarray(img[::ds, ::ds])
        np.save(out_dir / f"light_{out_i:03d}.npy", img)
        sh_cam = rotate_sh(sh_world[src_i].T, R_wc) / exr_scale   # (9,3) cam, image-normalised
        np.save(out_dir / f"sh_{out_i:03d}.npy", sh_cam.astype(np.float32))

    write_gt(out_dir, _stride({k: gt[k] for k in ("albedo", "normals", "mask",
                                                  "roughness", "metallic")}, ds))

    n = len(keep)
    (out_dir / "config.json").write_text(json.dumps({
        "variant": "exr", "n_lights": n, "light_type": "light", "lighting": "sh",
        "sh_order": 2, "exr_scale": exr_scale, "prereduced_downsample": ds,
        "hl_mode": hl_mode, "diffuse_fresnel": True,
        "camera": {"eye": eye.tolist(), "fov_deg": fov,
                   "R_world_to_cam": R_wc.tolist(), "recovery_residual": res,
                   "resolution": [meta.get("resolution_x"), meta.get("resolution_y")]},
        "preset": meta.get("preset"), "src": str(scene_dir),
        "note": "normals + sh rotated from world into the canonical proxy-camera frame"},
        indent=1))
    print(f"      {n} lights  {gt['normals'].shape[0]}x{gt['normals'].shape[1]}  "
          f"fov={fov:.0f} res={res:.4f}  exr_scale={exr_scale:.3f} -> {out_dir.name}")
    return n


# ── driver ────────────────────────────────────────────────────────────────────
def discover(root: Path):
    scenes = []
    for sd in sorted(p for p in root.iterdir() if p.is_dir()):
        if all((sd / f).exists() for f in ("metadata.json", "lights.json",
                                           "normal0001.exr", "position0001.exr")) \
           and any(sd.glob("render_*.exr")):
            scenes.append(sd)
    return scenes


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--scenes", nargs="+", default=None, help="scene-name prefixes to include")
    ap.add_argument("--hl_mode", choices=["analytic", "lut"], default="analytic")
    ap.add_argument("--downsample", type=int, default=1)
    ap.add_argument("--n_lights", type=int, default=None, help="cap conditions per scene (testing)")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    if not args.root.exists():
        raise SystemExit(f"indoor dataset not found: {args.root}")
    scenes = discover(args.root)
    if args.scenes:
        scenes = [s for s in scenes if any(s.name.startswith(p) for p in args.scenes)]
    if not scenes:
        raise SystemExit(f"no convertible scenes under {args.root}")
    print(f"root={args.root}\nout={args.out}\n{len(scenes)} scene(s)  "
          f"downsample={args.downsample}\n")

    n_built = n_skipped = 0
    for i, sd in enumerate(scenes, 1):
        out_dir = args.out / sd.name
        n_src = len(_render_files(sd)) if args.n_lights is None else \
            min(len(_render_files(sd)), args.n_lights)
        print(f"[{i}/{len(scenes)}] {sd.name}")
        if _up_to_date(out_dir, n_src) and not args.overwrite:
            print(f"      up to date ({n_src} lights) — skip"); n_skipped += 1; continue
        convert_scene(sd, out_dir, args.downsample, args.hl_mode, args.n_lights)
        n_built += 1

    print(f"\nDONE -> {args.out}   built={n_built} skipped={n_skipped}")


if __name__ == "__main__":
    main()
