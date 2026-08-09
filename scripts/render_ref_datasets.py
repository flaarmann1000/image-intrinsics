"""Render shade_ct_sh (inverse-crime) datasets for 3D-Front views under a reference
SH light bank, at a chosen SH order.

For each (view, order) it loads the view's GT gbuffers, renders all N lights with
``shade_ct_sh`` under the reference lighting (``dclift`` for order 2, ``dclift_sh3`` for
order 3, scaled by ``strength``), and writes an EXR-variant dataset that
``idr.data.scene_io.load_scene`` reads directly:

    <out_root>/<scene8>_v<view>/sh<order>/
        albedo.npy normals.npy roughness.npy metallic.npy   (float32 GT maps)
        light_000.npy .. light_127.npy                       (float32 observations)
        sh_000.npy   .. sh_127.npy                           (effective SH = ref x strength)
        config.json                                          (variant=exr, sh_order, ...)

Because the same shader + LUT render and (later) reconstruct the data, each dataset is a
true inverse crime: r(GT-map, GT-light) is ~0 up to fp rounding.

Usage:
    python scripts/render_ref_datasets.py            # defaults below (views 0,2; SH2,SH3)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root for `idr`
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
os.environ.setdefault("WANDB_MODE", "disabled")

import numpy as np
import torch
from PIL import Image

from idr.render import shade_ct_sh
from idr.render.brdf import _get_ggx_sh_lut
from idr.data.geometry import make_proxy_geometry
from idr.paths import RESULTS_DIR

# proxy-geometry params MUST match the reconstruction side (pixel_diagnostics load_any)
FOV_DEG, CAM_DIST = 60.0, 2.0

LIGHTING = {2: RESULTS_DIR / "ref_lighting" / "dclift",
            3: RESULTS_DIR / "ref_lighting" / "dclift_sh3"}


def _load_exr_rgb(path: Path) -> np.ndarray:
    import cv2
    im = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if im is None:
        raise FileNotFoundError(f"could not read EXR: {path}")
    if im.ndim == 2:
        im = np.repeat(im[:, :, None], 3, axis=2)
    return np.ascontiguousarray(im[:, :, ::-1]).astype(np.float32)   # BGR -> RGB


def load_gt_maps(view_dir: Path) -> dict:
    """Float GT gbuffers: albedo/normals from EXR, roughness/metallic from 16-bit PNG."""
    albedo = _load_exr_rgb(view_dir / "albedo.exr")[:, :, :3]
    normals = _load_exr_rgb(view_dir / "normals.exr")[:, :, :3]
    nlen = np.linalg.norm(normals, axis=-1, keepdims=True)
    mask = nlen[:, :, 0] > 0.5
    normals = np.where(mask[:, :, None], normals / np.clip(nlen, 1e-6, None), 0.0).astype(np.float32)
    rough = (np.asarray(Image.open(view_dir / "roughness.png"), np.float32) / 65535.0)
    metal = (np.asarray(Image.open(view_dir / "metallic.png"), np.float32) / 65535.0)
    rough = rough[:, :, None] if rough.ndim == 2 else rough[:, :, :1]
    metal = metal[:, :, None] if metal.ndim == 2 else metal[:, :, :1]
    return dict(albedo=albedo, normals=normals, mask=mask,
                roughness=rough.astype(np.float32), metallic=metal.astype(np.float32))


def render_dataset(view_dir: Path, order: int, out_dir: Path, n_lights: int,
                   strength: float, diffuse_fresnel: bool, device: str,
                   hl_mode: str = "analytic") -> None:
    gt = load_gt_maps(view_dir)
    H, W = gt["mask"].shape
    Nhw, frag, mhw, cam = make_proxy_geometry(gt["normals"], gt["mask"], FOV_DEG, CAM_DIST,
                                              device, torch.float32)
    fm = mhw.reshape(-1)
    N_m = Nhw.reshape(-1, 3)[fm]
    V_m = torch.nn.functional.normalize(cam[None] - frag.reshape(-1, 3)[fm], dim=-1)
    to_m = lambda a, c: torch.from_numpy(a.reshape(-1, c)).to(device, torch.float32)[fm.cpu()]
    A = to_m(gt["albedo"], 3); R = to_m(gt["roughness"], 1)[:, 0]; Mt = to_m(gt["metallic"], 1)[:, 0]
    # hl_mode must match the decomposition run for the inverse crime to hold; "analytic"
    # (default) renders the closed-form specular, "lut" the shipped table.
    lut = _get_ggx_sh_lut(device, n_bands=order + 1).to(torch.float32) if hl_mode == "lut" else None

    sh_dir = LIGHTING[order]
    sh_files = sorted(sh_dir.glob("sh_*.npy"), key=lambda p: int(p.stem.split("_")[-1]))[:n_lights]
    if not sh_files:
        raise FileNotFoundError(f"no sh_*.npy in {sh_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "albedo.npy", gt["albedo"].astype(np.float32))
    np.save(out_dir / "normals.npy", gt["normals"].astype(np.float32))
    np.save(out_dir / "roughness.npy", gt["roughness"].astype(np.float32))
    np.save(out_dir / "metallic.npy", gt["metallic"].astype(np.float32))
    Image.fromarray((np.clip(gt["albedo"], 0, 1) * 255).astype(np.uint8)).save(out_dir / "albedo.png")

    mflat = fm.cpu().numpy()
    obs_max = 0.0
    with torch.no_grad():
        for f in sh_files:
            idx = f.stem.split("_")[-1]
            sh = torch.from_numpy(np.load(f).astype(np.float32) * strength).to(device)   # (9|16,3)
            flat = shade_ct_sh(V_m, N_m, A, sh, Mt[:, None], R[:, None],
                               lut=lut, diffuse_fresnel=diffuse_fresnel, hl_mode=hl_mode)
            img = np.zeros((H * W, 3), np.float32)
            img[mflat] = flat.float().cpu().numpy()
            np.save(out_dir / f"light_{idx}.npy", img.reshape(H, W, 3))
            np.save(out_dir / f"sh_{idx}.npy", (np.load(f).astype(np.float32) * strength))
            obs_max = max(obs_max, float(flat.max()))

    with open(out_dir / "config.json", "w") as fh:
        json.dump({"variant": "exr", "n_lights": len(sh_files), "light_type": "env",
                   "ct_shader": "ct_sh", "sh_order": order, "diffuse_fresnel": diffuse_fresnel,
                   "hl_mode": hl_mode,
                   "prereduced_downsample": 1, "strength": strength,
                   "ref_lighting": str(sh_dir), "src": str(view_dir)}, fh, indent=2)
    print(f"  [{out_dir.name}] {len(sh_files)} lights, {H}x{W}, order {order}, obs_max={obs_max:.3f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene_dir", default=r"E:\3D-Front\output\1c349305-5c5f-4329-aab8-2744f7b75379")
    ap.add_argument("--views", type=int, nargs="+", default=[0, 2])
    ap.add_argument("--orders", type=int, nargs="+", default=[2, 3])
    ap.add_argument("--out_root", default=str(RESULTS_DIR / "sh_order_study"))
    ap.add_argument("--n_lights", type=int, default=128)
    ap.add_argument("--strength", type=float, default=2.0)
    ap.add_argument("--no_diffuse_fresnel", action="store_true")
    ap.add_argument("--hl_mode", choices=["analytic", "lut"], default="analytic",
                    help="specular band source; must match the decomposition run for the "
                         "inverse crime (default analytic)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    scene_dir = Path(args.scene_dir)
    scene8 = scene_dir.name[:8]
    out_root = Path(args.out_root)
    print(f"scene {scene8}  views {args.views}  orders {args.orders}  -> {out_root}")
    for view in args.views:
        vdir = scene_dir / str(view)
        for order in args.orders:
            out = out_root / f"{scene8}_v{view}" / f"sh{order}"
            render_dataset(vdir, order, out, args.n_lights, args.strength,
                           not args.no_diffuse_fresnel, device, args.hl_mode)


if __name__ == "__main__":
    main()
