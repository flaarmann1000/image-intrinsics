"""Batch-build canonical dataset leaves from BlenderProc multipass output.

Turns a re-rendered scene tree into the self-contained ``exr``-variant dataset dirs that
``load_scene`` / ``decompose_scene`` / ``run_decomposition.py`` / ``pixel_diag`` consume.

Input (one SH order per ``--root``):

    <root>/<scene>/<view>/{albedo,normals,roughness,metallic}.exr|png   GT gbuffers
    <root>/<scene>/<view>/env_<i>.exr                                   Cycles env renders
    <root>/<scene>/<view>/accepted_setup.json                          per-light SH (-> dclift[_sh3])
    <root>/<scene>/<view>/ground/{gbuffers, env_ground_{noshadow,shadow}_<i>.exr}

Output (one leaf per (view, variant)):

    <out>/<scene8>_v<view>/<ds_name>/   light_NNN.npy  sh_NNN.npy  {GT}.png+.npy  config.json

    ct-ct_sh-frOn_env / ct-ct_sh-frOn_env_ground   shade_ct_sh inverse crime (records hl_mode)
    blender_env / blender_env_ground_noshadow / blender_env_ground_shadow   the Cycles renders

GT gbuffers are read as FLOAT (albedo/normals from .exr, roughness/metallic from 16-bit .png),
so the albedo is not clipped to 8 bits — the same maps feed the CT and Blender leaves, so the
two are an exact GT match. Run once per order:

    python scripts/build_canonical_datasets.py --root E:\3D-Front\260807_output\sh2 --order 2
    python scripts/build_canonical_datasets.py --root E:\3D-Front\260807_output\sh3 --order 3
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # repo root for `idr`
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
os.environ.setdefault("WANDB_MODE", "disabled")

import numpy as np
import torch
from PIL import Image

from idr.render import shade_ct_sh
from idr.data.geometry import make_proxy_geometry
from idr.data.build import build_3dfront_dataset

# proxy-geometry params MUST match the reconstruction side (pixel_diag / decompose_scene)
FOV_DEG, CAM_DIST = 60.0, 2.0
ALL_VARIANTS = ["ct_env", "ct_env_ground",
                "blender_env", "blender_env_ground_noshadow", "blender_env_ground_shadow"]


# ── GT gbuffers (float) ───────────────────────────────────────────────────────
def _load_exr_rgb(path: Path) -> np.ndarray:
    import cv2
    im = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if im is None:
        raise FileNotFoundError(f"could not read EXR: {path}")
    if im.ndim == 2:
        im = np.repeat(im[:, :, None], 3, axis=2)
    return np.ascontiguousarray(im[:, :, ::-1]).astype(np.float32)          # BGR -> RGB


def load_gt(gt_dir: Path) -> dict:
    """albedo/normals from EXR, roughness/metallic from 16-bit PNG. Mask = |normal| > 0.5."""
    albedo = _load_exr_rgb(gt_dir / "albedo.exr")[:, :, :3]
    normals = _load_exr_rgb(gt_dir / "normals.exr")[:, :, :3]
    nlen = np.linalg.norm(normals, axis=-1, keepdims=True)
    mask = nlen[:, :, 0] > 0.5
    normals = np.where(mask[:, :, None], normals / np.clip(nlen, 1e-6, None), 0.0).astype(np.float32)
    rough = np.asarray(Image.open(gt_dir / "roughness.png"), np.float32) / 65535.0
    metal = np.asarray(Image.open(gt_dir / "metallic.png"), np.float32) / 65535.0
    rough = (rough[:, :, None] if rough.ndim == 2 else rough[:, :, :1]).astype(np.float32)
    metal = (metal[:, :, None] if metal.ndim == 2 else metal[:, :, :1]).astype(np.float32)
    return dict(albedo=albedo, normals=normals, mask=mask, roughness=rough, metallic=metal)


def _stride(gt: dict, ds: int) -> dict:
    if ds <= 1:
        return gt
    return {k: (np.ascontiguousarray(v[::ds, ::ds]) if isinstance(v, np.ndarray) else v)
            for k, v in gt.items()}


def write_gt(out_dir: Path, gt: dict) -> None:
    """Write GT as display PNGs + lossless float .npy, background = zero normal (matches
    load_scene's mask). Identical encoding for CT and Blender leaves."""
    nrm = ((gt["normals"] + 1) / 2 * 255).astype(np.uint8)
    nrm[~gt["mask"]] = 0
    Image.fromarray(nrm).save(out_dir / "normals.png")
    Image.fromarray((np.clip(gt["albedo"][:, :, :3], 0, 1) * 255).astype(np.uint8)).save(out_dir / "albedo.png")
    Image.fromarray((np.clip(gt["roughness"].squeeze(), 0, 1) * 65535).astype(np.uint16)).save(out_dir / "roughness.png")
    Image.fromarray((np.clip(gt["metallic"].squeeze(), 0, 1) * 65535).astype(np.uint16)).save(out_dir / "metallic.png")
    nrm_f = gt["normals"].astype(np.float32).copy(); nrm_f[~gt["mask"]] = 0.0
    np.save(out_dir / "normals.npy", nrm_f)
    np.save(out_dir / "albedo.npy", gt["albedo"][:, :, :3].astype(np.float32))
    np.save(out_dir / "roughness.npy", gt["roughness"].astype(np.float32))
    np.save(out_dir / "metallic.npy", gt["metallic"].astype(np.float32))


# ── per-light SH from the accepted setup ──────────────────────────────────────
def sh_from_setup(setup: dict, key: str, n_lights: int | None) -> list:
    """Load each light's SH coeffs from the paths in accepted_setup[key], scaled by strength.
    Resolves sh_<idx>.npy next to the referenced sh_env_map_<idx>.* (dclift / dclift_sh3)."""
    entries = setup[key]
    if n_lights is not None:
        entries = entries[:n_lights]
    out = []
    for e in entries:
        spec = e["light_spec"][0]
        p = Path(spec["path"]); idx = p.stem.split("_")[-1]
        coeffs = np.load(p.parent / f"sh_{idx}.npy").astype(np.float32)
        out.append(coeffs * float(spec.get("strength", 1.0)))
    return out


# ── builders ──────────────────────────────────────────────────────────────────
def build_ct(gt_dir: Path, sh_list: list, out_dir: Path, hl_mode: str,
             ds: int, device: str, diffuse_fresnel: bool = True) -> None:
    """Render the GT gbuffers under every light with shade_ct_sh — an exact inverse crime
    when decomposed with the same hl_mode/diffuse_fresnel."""
    gt = _stride(load_gt(gt_dir), ds)
    H, W = gt["mask"].shape
    Nhw, frag, mhw, cam = make_proxy_geometry(gt["normals"], gt["mask"], FOV_DEG, CAM_DIST,
                                              device, torch.float32)
    fm = mhw.reshape(-1)
    N_m = Nhw.reshape(-1, 3)[fm]
    V_m = torch.nn.functional.normalize(cam[None] - frag.reshape(-1, 3)[fm], dim=-1)
    to_m = lambda a, c: torch.from_numpy(np.ascontiguousarray(a.reshape(-1, c))).to(
        device, torch.float32)[fm.cpu()]
    A, R, Mt = to_m(gt["albedo"], 3), to_m(gt["roughness"], 1), to_m(gt["metallic"], 1)
    mflat = fm.cpu().numpy()

    shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_gt(out_dir, gt)
    obs_max = 0.0
    with torch.no_grad():
        for i, sh in enumerate(sh_list):
            sh_t = torch.from_numpy(np.asarray(sh, np.float32)).to(device)
            flat = shade_ct_sh(V_m, N_m, A, sh_t, Mt, R, diffuse_fresnel=diffuse_fresnel,
                               hl_mode=hl_mode)
            img = np.zeros((H * W, 3), np.float32)
            img[mflat] = flat.float().cpu().numpy()
            np.save(out_dir / f"light_{i:03d}.npy", img.reshape(H, W, 3))
            np.save(out_dir / f"sh_{i:03d}.npy", np.asarray(sh, np.float32))
            obs_max = max(obs_max, float(flat.max()))
    (out_dir / "config.json").write_text(json.dumps(
        {"variant": "exr", "n_lights": len(sh_list), "light_type": "light",
         "ct_shader": "ct_sh", "hl_mode": hl_mode, "diffuse_fresnel": diffuse_fresnel,
         "prereduced_downsample": ds, "src": str(gt_dir)}, indent=1))
    print(f"      ct  {out_dir.name:28s} {len(sh_list)} lights  {H}x{W}  obs_max={obs_max:.3f}")


def build_blender(img_dir: Path, gt_dir: Path, stem: str, sh_list: list,
                  out_dir: Path, ds: int, hl_mode: str = "analytic") -> None:
    """Package the Cycles <stem>_<i>.exr renders as observations; attach the same float GT
    and the per-light GT SH (scaled by the exr normalisation).

    `hl_mode` does NOT touch the observations (they are Cycles renders) — it is recorded so
    the decomposition / pixel_diag use the intended specular model on this dataset."""
    shutil.rmtree(out_dir, ignore_errors=True)
    build_3dfront_dataset(img_dir, out_dir, variant="exr", lighting=stem)   # images + exr_scale
    cfg = json.loads((out_dir / "config.json").read_text())
    exr_scale = float(cfg.get("exr_scale", 1.0) or 1.0)

    light_files = sorted(out_dir.glob("light_*.npy"))
    for f in light_files[len(sh_list):]:                                    # trim extras
        f.unlink(missing_ok=True)
    light_files = light_files[:len(sh_list)]
    if ds > 1:
        for f in light_files:
            np.save(f, np.ascontiguousarray(np.load(f)[::ds, ::ds]))
    for f in out_dir.glob("light_*_preview.png"):
        f.unlink()

    write_gt(out_dir, _stride(load_gt(gt_dir), ds))
    for i, sh in enumerate(sh_list):
        np.save(out_dir / f"sh_{i:03d}.npy", (np.asarray(sh, np.float32) / exr_scale))
    cfg.update(n_lights=len(sh_list), prereduced_downsample=ds, light_type="light",
               hl_mode=hl_mode)
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=1))
    print(f"      blender {out_dir.name:24s} {len(sh_list)} lights  exr_scale={exr_scale:.3f}")


# ── discovery + driver ─────────────────────────────────────────────────────────
def discover(root: Path) -> list:
    views = []
    for sd in sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith("_")):
        for vd in sorted((p for p in sd.iterdir() if p.is_dir() and p.name.isdigit()),
                         key=lambda p: int(p.name)):
            if (vd / "accepted_setup.json").exists() and (vd / "albedo.exr").exists():
                views.append((sd.name, int(vd.name), vd))
    return views


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help=r"order root, e.g. E:\3D-Front\260807_output\sh2")
    ap.add_argument("--order", type=int, choices=[2, 3], required=True,
                    help="SH order of this root (2 -> 9 coeffs, 3 -> 16); metadata only")
    ap.add_argument("--out", default=None,
                    help="output root (default <root>/../datasets/sh<order>)")
    ap.add_argument("--variants", nargs="+", default=ALL_VARIANTS, choices=ALL_VARIANTS)
    ap.add_argument("--hl_mode", choices=["analytic", "lut"], default="analytic")
    ap.add_argument("--downsample", type=int, default=1)
    ap.add_argument("--n_lights", type=int, default=None, help="cap lights per view (testing)")
    ap.add_argument("--scenes", nargs="+", default=None, help="scene-name prefixes to include")
    ap.add_argument("--overwrite", action="store_true", help="rebuild leaves that already exist")
    args = ap.parse_args()

    root = Path(args.root)
    out_root = Path(args.out) if args.out else root.parent / "datasets" / f"sh{args.order}"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    views = discover(root)
    if args.scenes:
        views = [v for v in views if any(v[0].startswith(p) for p in args.scenes)]
    print(f"root={root}\nout={out_root}\ndevice={device}  order={args.order}  hl_mode={args.hl_mode}  "
          f"downsample={args.downsample}\n{len(views)} (scene,view) pair(s), variants={args.variants}\n")

    # (variant -> how to source images + GT + which setup key for the SH)
    for vi, (scene, view, vd) in enumerate(views, 1):
        setup = json.loads((vd / "accepted_setup.json").read_text())
        key = f"{scene[:8]}_v{view}"
        gdir = vd / "ground"
        sh_env = sh_from_setup(setup, "env", args.n_lights) if "env" in setup else None
        sh_grd = sh_from_setup(setup, "env_ground", args.n_lights) if "env_ground" in setup else None
        print(f"[{vi}/{len(views)}] {key}")
        for var in args.variants:
            out_dir = out_root / key / {
                "ct_env": "ct-ct_sh-frOn_env", "ct_env_ground": "ct-ct_sh-frOn_env_ground",
                "blender_env": "blender_env",
                "blender_env_ground_noshadow": "blender_env_ground_noshadow",
                "blender_env_ground_shadow": "blender_env_ground_shadow"}[var]
            if out_dir.exists() and not args.overwrite:
                print(f"      skip {out_dir.name} (exists)"); continue
            if var == "ct_env":
                build_ct(vd, sh_env, out_dir, args.hl_mode, args.downsample, device)
            elif var == "ct_env_ground":
                build_ct(gdir, sh_grd, out_dir, args.hl_mode, args.downsample, device)
            elif var == "blender_env":
                build_blender(vd, vd, "env", sh_env, out_dir, args.downsample, args.hl_mode)
            elif var == "blender_env_ground_noshadow":
                build_blender(gdir, gdir, "env_ground_noshadow", sh_grd, out_dir, args.downsample, args.hl_mode)
            elif var == "blender_env_ground_shadow":
                build_blender(gdir, gdir, "env_ground_shadow", sh_grd, out_dir, args.downsample, args.hl_mode)
    print(f"\nDONE -> {out_root}")


if __name__ == "__main__":
    main()
