"""
test_env_roundtrip.py
=====================
Validate that the torch shaders (shade_ct_env / shade_ct_sh) reproduce a
BlenderProc render of the SAME scene lit by the SAME environment map.

Pipeline being checked
-----------------------
1. BlenderProc (test_sphere_pbr.py --env_map <png>) renders a sphere with a
   given equirectangular env map as the world background  ->  GROUND TRUTH.
2. The torch shader loads that scene's G-buffers + the same env map and renders
   it via shade_ct_env (and shade_ct_sh from the matching SH coeffs).
3. We report raw RMSE, the per-channel least-squares scale between shader and GT
   (which absorbs the Cook-Torrance-vs-Principled-BSDF brightness gap), the
   scale-invariant RMSE, and an orientation sanity check (identity must beat
   every flip/roll of the env map).

What the investigation established
----------------------------------
* The shader's env integration is energy-correct: a dense Monte-Carlo
  integration of the identical BRDF matches shade_ct_env to ~1%.
* The env orientation convention matches BlenderProc (identity wins the
  orientation search).
* BlenderProc's Principled BSDF is intrinsically ~1.5-1.7x brighter than this
  single-scatter Cook-Torrance for glossy/metallic surfaces (a near-gray
  additive surplus), so a residual scaled-RMSE remains by design. This is a
  BRDF-model gap, not a bug.
* render_3dfront_dataset must light from the MAX-NORMALISED env (normalize_env
  =True), because the sh_env_map PNGs BlenderProc consumes are max-normalised.

Usage
-----
  # Render the GT first (needs the torch311 env that has blenderproc):
  blenderproc run <bp_dir>/test_sphere_pbr.py output_sphere_env \
      --env_map results/ref_sh_lighting/sh_env_map_000.png --env_map_strength 1.0

  # Then validate:
  python test_env_roundtrip.py \
      --scene <bp_dir>/output_sphere_env \
      --env_map results/ref_sh_lighting/sh_env_map_000.png \
      --sh_npy results/ref_sh_lighting/sh_000.npy
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from raw_optimizer.dfront_ct import load_scene, make_proxy_geometry
from raw_renderer_gpu import EnvMap, SHLighting, shade_ct_env, shade_ct_sh
from raw_renderer_gpu.rasterizer import _get_ggx_sh_lut


def _flat_masked(arr_hw, flat_mask, device):
    t = torch.from_numpy(np.ascontiguousarray(arr_hw.astype(np.float32))).to(device)
    return t.reshape(-1, t.shape[-1])[flat_mask]


def _scatter(vec, flat_mask_np, H, W):
    out = np.zeros((H * W, 3), np.float32)
    out[flat_mask_np] = vec.detach().float().cpu().numpy()
    return out.reshape(H, W, 3)


def _per_channel_ls_scale(shader_img, gt_img, mask):
    """Least-squares per-channel scale s minimising ||s*shader - gt|| over fg."""
    s = np.empty(3, np.float32)
    for c in range(3):
        a = shader_img[mask][:, c]
        b = gt_img[mask][:, c]
        s[c] = (a * b).sum() / max((a * a).sum(), 1e-8)
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True, help="BlenderProc output dir (GT: rendering_linear.png + G-buffers)")
    ap.add_argument("--env_map", required=True, help="The equirectangular env PNG BlenderProc was lit with")
    ap.add_argument("--sh_npy", default=None, help="Optional (9,3) SH coeffs .npy for the ct_sh path")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--fov", type=float, default=60.0)
    args = ap.parse_args()

    scene_dir = Path(args.scene)
    gt = np.array(Image.open(scene_dir / "rendering_linear.png")).astype(np.float32)[:, :, :3] / 255.0
    mask = gt.sum(-1) > 0  # crude; refined by the real fg mask below

    scene = load_scene(scene_dir)
    H, W = scene["H"], scene["W"]
    mask = scene["mask_np"]
    nh, fh, mh, cam = make_proxy_geometry(scene["normals_np"], scene["mask_np"],
                                          fov_deg=args.fov, device=args.device)
    fm = mh.reshape(-1)
    fm_np = fm.cpu().numpy()
    al = _flat_masked(scene["albedo_np"], fm, args.device)
    ro = _flat_masked(scene["roughness_np"], fm, args.device)
    me = _flat_masked(scene["metallic_np"], fm, args.device)
    N = nh.reshape(-1, 3)[fm]
    fr = fh.reshape(-1, 3)[fm]
    V = torch.nn.functional.normalize(cam.unsqueeze(0) - fr, dim=-1)

    def report(name, render):
        raw = float(np.sqrt(((np.clip(render, 0, 1) - gt)[mask] ** 2).mean()))
        s = _per_channel_ls_scale(render, gt, mask)
        scaled = float(np.sqrt(((s[None, None, :] * render - gt)[mask] ** 2).mean()))
        ratio = render[mask].mean(0) / gt[mask].mean(0)
        print(f"{name:24s} raw_RMSE={raw:.4f}  scaled_RMSE={scaled:.4f}  "
              f"shader/GT={np.round(ratio,3)}  LS_scale={np.round(s,3)}")

    print(f"GT mean(fg) = {np.round(gt[mask].mean(0),4)}   ({int(mask.sum())} fg px)\n")

    # --- ct_env from the exact env PNG ---
    env = EnvMap.from_file(str(args.env_map))
    ep = torch.from_numpy(env._image_flat.copy()).to(args.device)
    ed = torch.from_numpy(env._dirs.copy()).to(args.device)
    edw = torch.from_numpy(env._solid_angles.copy()).to(args.device)
    r_env = _scatter(shade_ct_env(V, N, al, ep, ed, edw, metallic=me, roughness=ro), fm_np, H, W)
    report("ct_env(png)", r_env)

    # --- ct_sh from normalised SH coeffs (matches the normalised PNG) ---
    if args.sh_npy:
        coeffs = np.load(args.sh_npy).astype(np.float32)
        env_max = float(EnvMap.from_sh(SHLighting(coeffs)).image.max())
        cn = coeffs / max(env_max, 1e-8)
        lut = _get_ggx_sh_lut(args.device)
        r_sh = _scatter(shade_ct_sh(V, N, al, torch.from_numpy(cn).to(args.device),
                                    metallic=me, roughness=ro, lut=lut), fm_np, H, W)
        report("ct_sh(coeffs/env_max)", r_sh)

    # --- orientation sanity check: identity must beat flips/rolls ---
    base = np.array(Image.open(args.env_map).convert("RGB")).astype(np.float32) / 255.0
    Wn = base.shape[1]
    variants = {
        "identity": base,
        "flip_lr": base[:, ::-1],
        "flip_ud": base[::-1],
        "roll_W/2": np.roll(base, Wn // 2, 1),
        "roll_W/4": np.roll(base, Wn // 4, 1),
    }
    print("\norientation search (scale-invariant RMSE; identity should win):")
    best, best_v = None, 1e9
    for k, v in variants.items():
        e = EnvMap(np.ascontiguousarray(v))
        rr = _scatter(shade_ct_env(V, N, al,
                                   torch.from_numpy(e._image_flat.copy()).to(args.device),
                                   torch.from_numpy(e._dirs.copy()).to(args.device),
                                   torch.from_numpy(e._solid_angles.copy()).to(args.device),
                                   metallic=me, roughness=ro), fm_np, H, W)
        s = _per_channel_ls_scale(rr, gt, mask)
        sc = float(np.sqrt(((s[None, None, :] * rr - gt)[mask] ** 2).mean()))
        print(f"  {k:12s} scaled_RMSE={sc:.4f}")
        if sc < best_v:
            best, best_v = k, sc
    print(f"  -> best: {best}" + ("  [OK: orientation convention matches]"
                                  if best == "identity" else "  [WARNING: non-identity won!]"))


if __name__ == "__main__":
    main()
