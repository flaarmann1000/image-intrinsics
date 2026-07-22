"""Phase 2 of the synthetic study: decompose every rendered scene.

Named `run_study` rather than `run_decomposition`: the old name collided with the
batch driver script run_decomposition.py, which does something different (it drives
decompose_scene over a tree of real 3D-Front datasets).
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import wandb
from PIL import Image

from idr.config import (DEFAULT_CFG, MATERIAL_CONFIGS, PHONG_MATERIAL_CONFIGS,
                        NAMED_TRANSFORMS, SHININESS_RANGE, LIGHT_COLOR, LIGHT_INTENSITY)
from idr.paths import DATASET_ROOT, RESULTS_ROOT
from idr.config import DEFAULT_CAMERA, LIGHT_ANGLES_DEG
from idr.render import EnvMap, SHLighting, build_sh_basis, rasterize_geometry
from idr.data.synthetic_scene import _load_mesh
from idr.data.scene_io import load_scene
from idr.data.geometry import make_proxy_geometry
from idr.data.synthetic_scene import _get_light_entries, _scene_suffix, _scatter, _scatter_np
from idr.data.synthetic_io import _read_dataset_meta
from idr.optim.transforms import _parse_transforms, _transforms_folder, _fwd_albedo
from idr.optim.registry import optimize
from idr.optim.result import EnvGrid
from idr.eval.metrics import _albedo_lighting_scale, _rescale_albedo_lighting
from idr.track.wandb_log import _sh_coeffs_to_env_img, _env_flat_to_img
from raw_optimizer.helper import _albedo_rmse

_ALL_SHADERS = ["ct_sh", "ct_env", "phong_sh", "phong_env"]


def run_study(
    mesh_name:          str            = "sphere",
    width:              int            = 128,
    height:             int            = 128,
    shader:             str            = "ct_sh",
    mat_filter:         Optional[str]  = None,
    cfg_overrides:      Optional[dict] = None,
    device:             str            = "cuda",
    opt_params:         Optional[frozenset] = None,
    skip_existing:      bool           = False,
    mat_configs_filter: Optional[set]  = None,
    transforms:         Optional[dict] = None,
    light_mode:         str            = "directional",
    n_lights:           int            = 6,
    full_circle:        bool           = False,
    init_from_gt:       bool           = False,
    log_gradients:      bool           = False,
) -> None:
    cfg = {**DEFAULT_CFG, **(cfg_overrides or {})}
    dev = device
    tr  = transforms if transforms is not None else NAMED_TRANSFORMS["none"]
    transform_folder = _transforms_folder(tr)
    tr_dir = RESULTS_ROOT / transform_folder
    tr_dir.mkdir(parents=True, exist_ok=True)
    tr_json = tr_dir / "transforms.json"
    if not tr_json.exists():
        with open(tr_json, "w") as fh:
            json.dump(tr, fh, indent=2)

    is_phong = shader.startswith("phong")
    _base_configs = PHONG_MATERIAL_CONFIGS if is_phong else MATERIAL_CONFIGS
    mat_configs = {k: v for k, v in _base_configs.items()
                   if mat_configs_filter is None or k in mat_configs_filter}

    # Folder/run suffix for selective optimization and regularization
    if opt_params is not None:
        result_shader = shader + "_op=" + ",".join(sorted(opt_params))
    else:
        result_shader = shader
    ls = cfg.get("lambda_sparse", 0.0)
    lw = cfg.get("lambda_white",  0.0)
    lt = cfg.get("lambda_tv",     0.0)
    if ls:
        result_shader += f"_ls={ls}"
    if lw:
        result_shader += f"_lw={lw}"
    if lt:
        result_shader += f"_lt={lt}"
    if init_from_gt:
        result_shader += "_gtinit"
    if log_gradients:
        result_shader += "_gradlog"

    suffix = _scene_suffix(light_mode, n_lights, full_circle)

    mesh = _load_mesh(mesh_name)
    normals_hw, frag_pos_hw, mask_hw, cam_pos = rasterize_geometry(
        mesh, DEFAULT_CAMERA, width=width, height=height, smooth=True, device=dev,
    )

    _sh_ref  = SHLighting.directional(np.array([0, 0, 1], dtype=np.float32),
                                      LIGHT_COLOR, intensity=LIGHT_INTENSITY)
    _env_ref = EnvMap.from_sh(_sh_ref)
    env_dirs, env_dw = _env_ref._dirs, _env_ref._solid_angles
    env_H, env_W     = _env_ref.image.shape[:2]

    def _save_gray(arr_hw1: np.ndarray, path: Path) -> None:
        Image.fromarray((arr_hw1.squeeze(-1) * 255).clip(0, 255).astype(np.uint8)).save(path)

    for mat_id, mat_cfg in mat_configs.items():
        base_prefix  = f"{mesh_name}_phong_{mat_id}" if is_phong else f"{mesh_name}_{mat_id}"
        scene_prefix = f"{base_prefix}{suffix}"
        if mat_filter and scene_prefix != mat_filter and base_prefix != mat_filter:
            continue

        # Read light keys from dataset_meta.json; fall back to LIGHT_ANGLES_DEG for old data
        meta = _read_dataset_meta(scene_prefix)
        if meta is not None:
            light_keys = meta["light_keys"]
        else:
            light_keys = [f"light_{int(a):02d}deg" for a in LIGHT_ANGLES_DEG]

        out_dir_check = tr_dir / scene_prefix / result_shader
        if skip_existing and (out_dir_check / "metrics.json").exists():
            print(f"\n[Phase 2] {scene_prefix}  ({result_shader})  skipped (metrics.json exists)")
            continue

        print(f"\n[Phase 2] {scene_prefix}  ({shader})")

        images, gt_sh_list = [], []
        for light_id in light_keys:
            img_path = DATASET_ROOT / scene_prefix / shader / light_id / "render.png"
            cfg_path = DATASET_ROOT / scene_prefix / shader / light_id / "config.json"
            if not img_path.exists():
                raise FileNotFoundError(f"{img_path} — run --phase 1 --shader {shader} first.")
            npy_path = img_path.with_suffix(".npy")
            if npy_path.exists():
                images.append(np.load(npy_path).astype(np.float32))
            else:
                images.append(np.array(Image.open(img_path), dtype=np.float32) / 255.0)
            with open(cfg_path) as fh:
                gt_sh_list.append(np.array(
                    json.load(fh)["light"]["sh_coeffs"], dtype=np.float32))

        mask_np = mask_hw.cpu().numpy()
        _gt_alb_npy = DATASET_ROOT / scene_prefix / "gt" / "albedo.npy"
        if _gt_alb_npy.exists():
            gt_color = np.load(_gt_alb_npy)               # (H, W, 3), already masked
        else:
            gt_color = (np.tile(np.array(mat_cfg["albedo"], dtype=np.float32), (height, width, 1))
                        * mask_np[:, :, None])
        gt_albedo_img = gt_color

        run = wandb.init(
            entity  ="DLVC-intrinsics",
            project ="synthetic_ct_decomp",
            config  =dict(**cfg, mesh_name=mesh_name, mat_id=mat_id,
                          material=mat_cfg, shader=shader,
                          opt_params=sorted(opt_params) if opt_params is not None else "all",
                          transforms=tr, transform_folder=transform_folder,
                          width=width, height=height, n_images=len(images),
                          light_mode=light_mode, init_from_gt=init_from_gt),
            name    =f"{scene_prefix}_{result_shader}",
            reinit  =True,
        )

        # GT SH env maps — shared across both CT and Phong SH/env shaders
        gt_sh_env_imgs = [_sh_coeffs_to_env_img(gt_sh) for gt_sh in gt_sh_list]

        if is_phong:
            _gt_shin_npy = DATASET_ROOT / scene_prefix / "gt" / "shininess.npy"
            _gt_ks_npy   = DATASET_ROOT / scene_prefix / "gt" / "ks.npy"
            gt_shin_map  = (np.load(_gt_shin_npy) if _gt_shin_npy.exists()
                            else np.full((height, width, 1), mat_cfg["shininess"], dtype=np.float32)
                                 * mask_np[:, :, None])
            gt_ks_map    = (np.load(_gt_ks_npy) if _gt_ks_npy.exists()
                            else np.full((height, width, 1), mat_cfg["ks"], dtype=np.float32)
                                 * mask_np[:, :, None])
            gt_shin = float(gt_shin_map[mask_np].mean())
            gt_ks   = float(gt_ks_map[mask_np].mean())
            gt_shin_img = gt_shin_map[:, :, 0] / SHININESS_RANGE[1]
            gt_ks_img   = gt_ks_map[:, :, 0]
            run.log({"gt_images":       [wandb.Image(img) for img in images],
                     "gt_albedo":       wandb.Image(gt_albedo_img),
                     "gt_shininess":    wandb.Image(gt_shin_img),
                     "gt_ks":           wandb.Image(gt_ks_img),
                     "gt_sh_env_maps":  [wandb.Image(e) for e in gt_sh_env_imgs]}, step=0)
        else:
            _gt_met_npy   = DATASET_ROOT / scene_prefix / "gt" / "metallic.npy"
            _gt_rough_npy = DATASET_ROOT / scene_prefix / "gt" / "roughness.npy"
            gt_met_map    = (np.load(_gt_met_npy) if _gt_met_npy.exists()
                             else np.full((height, width, 1),
                                          mat_cfg["metallic"], dtype=np.float32) * mask_np[:, :, None])
            gt_rough_map  = (np.load(_gt_rough_npy) if _gt_rough_npy.exists()
                             else np.full((height, width, 1),
                                          mat_cfg["roughness"], dtype=np.float32) * mask_np[:, :, None])
            gt_metallic      = float(gt_met_map[mask_np].mean())
            gt_roughness     = float(gt_rough_map[mask_np].mean())
            gt_metallic_img  = gt_met_map[:, :, 0]
            gt_roughness_img = gt_rough_map[:, :, 0]
            run.log({"gt_images":       [wandb.Image(img) for img in images],
                     "gt_albedo":       wandb.Image(gt_albedo_img),
                     "gt_metallic":     wandb.Image(gt_metallic_img),
                     "gt_roughness":    wandb.Image(gt_roughness_img),
                     "gt_sh_env_maps":  [wandb.Image(e) for e in gt_sh_env_imgs]}, step=0)

        # gradient log dir (created lazily by _optimize_* when log_gradients=True)
        grad_log_dir = tr_dir / scene_prefix / result_shader / "gradient_flow"

        # ── dispatch ──────────────────────────────────────────────────────────
        sh_out:   np.ndarray = np.empty(0)
        env_maps: np.ndarray = np.empty(0)

        _is_ph = shader.startswith("phong")
        _res = optimize(
            shader, images, normals_hw, frag_pos_hw, mask_hw, cam_pos,
            (gt_shin_map if _is_ph else gt_met_map),
            (gt_ks_map   if _is_ph else gt_rough_map),
            cfg,
            ka=(mat_cfg["ka"] if _is_ph else None),
            kd=(mat_cfg["kd"] if _is_ph else None),
            env=(EnvGrid(env_dirs, env_dw, env_H, env_W)
                 if shader.endswith("_env") else None),
            wandb_run=run, gt_sh_coeffs=gt_sh_list, gt_albedo=gt_color,
            opt_params=opt_params, transforms=tr, init_from_gt=init_from_gt,
            log_gradients=log_gradients, grad_log_dir=grad_log_dir,
        )
        albedo, mat_a, mat_b = _res.albedo, _res.mat_a, _res.mat_b
        shadings, history, elapsed = _res.shadings, _res.history, _res.elapsed
        sh_out   = _res.sh       if _res.sh       is not None else np.empty(0)
        env_maps = _res.env_maps if _res.env_maps is not None else np.empty(0)

        # mat_a = metallic or shininess (H,W,1), mat_b = roughness or ks (H,W,1)
        gt_a = gt_shin   if is_phong else gt_metallic   # type: ignore[possibly-undefined]
        gt_b = gt_ks     if is_phong else gt_roughness  # type: ignore[possibly-undefined]
        a_label, b_label = ("shininess", "ks") if is_phong else ("metallic", "roughness")

        # ── albedo RMSE ───────────────────────────────────────────────────────
        est_px = torch.from_numpy(albedo[mask_np])
        gt_px  = torch.from_numpy(gt_color[mask_np])          # (M, 3) — handles flat and spatial GT
        rmse_t, scale_t = _albedo_rmse(est_px, gt_px)
        rmse   = float(rmse_t)
        scale  = scale_t.numpy()                                  # (3,) per-channel

        # Rescale lighting by 1/scale to match the albedo correction
        inv_scale = 1.0 / np.maximum(scale[None, None, :], 1e-8)  # (1,1,3)
        if shader in ("ct_sh", "phong_sh") and sh_out.size:
            sh_out_rescaled = sh_out * inv_scale
        else:
            sh_out_rescaled = sh_out
        if shader in ("ct_env", "phong_env") and env_maps.size:
            env_maps_rescaled = env_maps * inv_scale
        else:
            env_maps_rescaled = env_maps

        if is_phong:
            mat_a_err = np.abs(mat_a - gt_shin_map)    # type: ignore[possibly-undefined]
            mat_b_err = np.abs(mat_b - gt_ks_map)      # type: ignore[possibly-undefined]
        else:
            mat_a_err = np.abs(mat_a - gt_met_map)     # type: ignore[possibly-undefined]
            mat_b_err = np.abs(mat_b - gt_rough_map)   # type: ignore[possibly-undefined]
        mat_a_mean = float(mat_a[mask_np].mean())
        mat_b_mean = float(mat_b[mask_np].mean())

        albedo_scaled = (albedo * scale).clip(0, 1)
        albedo_err    = np.abs(albedo_scaled - gt_color) * mask_np[:, :, None]

        recon_err  = [np.abs(s - img) * mask_np[:, :, None]
                      for s, img in zip(shadings, images)]
        recon_rmse = float(np.mean([e[mask_np].mean() for e in recon_err]))

        metrics = dict(
            albedo_rmse=rmse, final_loss=float(history[-1]),
            recon_rmse=recon_rmse,
            albedo_scale=scale.tolist(), loss_history=history,
            **{f"{a_label}_est_mean": mat_a_mean,
               f"{a_label}_gt":       gt_a,
               f"{a_label}_err_mean": float(mat_a_err[mask_np].mean()),
               f"{b_label}_est_mean": mat_b_mean,
               f"{b_label}_gt":       gt_b,
               f"{b_label}_err_mean": float(mat_b_err[mask_np].mean())},
        )

        # Build final SH/env map images for wandb summary (scale-corrected)
        if shader in ("ct_sh", "phong_sh"):
            final_light_imgs = [wandb.Image(_sh_coeffs_to_env_img(sh_out_rescaled[k]))
                                for k in range(len(images))]
            light_img_key = "est_sh_env_maps"
        else:
            final_light_imgs = [wandb.Image(_env_flat_to_img(env_maps_rescaled[k], env_H, env_W))
                                for k in range(len(images))]
            light_img_key = "est_env_maps"

        run.log({
            "albedo_est":      wandb.Image(albedo.clip(0, 1)),
            "albedo_scaled":   wandb.Image(albedo_scaled),
            "albedo_err":      wandb.Image(albedo_err.mean(-1)),
            f"{a_label}_est":  wandb.Image(
                mat_a.squeeze(-1) / (SHININESS_RANGE[1] if is_phong else 1.0)),
            f"{b_label}_est":  wandb.Image(mat_b.squeeze(-1)),
            f"{a_label}_err":  wandb.Image(mat_a_err.squeeze(-1) * mask_np),
            f"{b_label}_err":  wandb.Image(mat_b_err.squeeze(-1) * mask_np),
            "reconstructions": [wandb.Image(s.clip(0, 1)) for s in shadings],
            "recon_errors":    [wandb.Image(e.mean(-1)) for e in recon_err],
            light_img_key:     final_light_imgs,
            "albedo_rmse":     rmse,
            "recon_rmse":      recon_rmse,
            "final_loss":      history[-1],
            "elapsed_s":       elapsed,
        }, step=cfg["n_iter"])
        run.finish()

        # ── save to disk ──────────────────────────────────────────────────────
        out_dir = tr_dir / scene_prefix / result_shader
        out_dir.mkdir(parents=True, exist_ok=True)
        recon_dir = out_dir / "reconstructions"
        recon_dir.mkdir(exist_ok=True)

        Image.fromarray((albedo.clip(0, 1) * 255).astype(np.uint8)).save(
            out_dir / "albedo_est.png")
        np.save(out_dir / "albedo_est.npy", albedo.astype(np.float32))   # raw, unscaled

        # shininess is in [s_min, s_max] — normalize to [0, 1] for 8-bit images
        a_norm = SHININESS_RANGE[1] if (is_phong and a_label == "shininess") else 1.0
        _save_gray(mat_a / a_norm, out_dir / f"{a_label}_est.png")
        np.save(out_dir / f"{a_label}_est.npy", mat_a.astype(np.float32))   # actual units
        _save_gray(mat_b,          out_dir / f"{b_label}_est.png")
        np.save(out_dir / f"{b_label}_est.npy", mat_b.astype(np.float32))
        _save_gray(mat_a_err / a_norm * mask_np[:, :, None], out_dir / f"{a_label}_err.png")
        np.save(out_dir / f"{a_label}_err.npy",
                (mat_a_err * mask_np[:, :, None]).astype(np.float32))       # actual units
        _save_gray(mat_b_err * mask_np[:, :, None],          out_dir / f"{b_label}_err.png")
        np.save(out_dir / f"{b_label}_err.npy",
                (mat_b_err * mask_np[:, :, None]).astype(np.float32))

        for k, (s, e) in enumerate(zip(shadings, recon_err)):
            lk = light_keys[k]
            Image.fromarray((s.clip(0, 1) * 255).astype(np.uint8)).save(
                recon_dir / f"recon_{lk}.png")
            np.save(recon_dir / f"recon_{lk}.npy", s.astype(np.float32))
            Image.fromarray((e.mean(-1) * 255).clip(0, 255).astype(np.uint8)).save(
                recon_dir / f"recon_err_{lk}.png")
            np.save(recon_dir / f"recon_err_{lk}.npy", e.astype(np.float32))

        if shader in ("ct_sh", "phong_sh"):
            np.save(out_dir / "sh_coeffs_est.npy", sh_out_rescaled)
            for k, sh_k in enumerate(sh_out_rescaled):
                sh_env_img = _sh_coeffs_to_env_img(sh_k)
                Image.fromarray((sh_env_img * 255).astype(np.uint8)).save(
                    out_dir / f"sh_env_map_{light_keys[k]}.png")
                np.save(out_dir / f"sh_env_map_{light_keys[k]}.npy", sh_env_img.astype(np.float32))
        else:
            np.save(out_dir / "env_maps_est.npy", env_maps_rescaled)
            for k, env_k in enumerate(env_maps_rescaled):
                env_img = _env_flat_to_img(env_k, env_H, env_W)
                Image.fromarray((env_img * 255).astype(np.uint8)).save(
                    out_dir / f"env_map_{light_keys[k]}.png")
                np.save(out_dir / f"env_map_{light_keys[k]}.npy", env_img.astype(np.float32))
            env_avg_img = _env_flat_to_img(env_maps_rescaled.mean(0), env_H, env_W)
            Image.fromarray((env_avg_img * 255).astype(np.uint8)).save(
                out_dir / "env_map_avg.png")

        with open(out_dir / "material_est.json", "w") as fh:
            json.dump({
                f"{a_label}_est_mean": mat_a_mean,
                f"{a_label}_gt":       gt_a,
                f"{a_label}_err_mean": float(mat_a_err[mask_np].mean()),
                f"{b_label}_est_mean": mat_b_mean,
                f"{b_label}_gt":       gt_b,
                f"{b_label}_err_mean": float(mat_b_err[mask_np].mean()),
            }, fh, indent=2)
        with open(out_dir / "metrics.json", "w") as fh:
            json.dump(metrics, fh, indent=2)

        print(f"  {elapsed:.1f}s  albedo RMSE={rmse:.4f}"
              f"  {a_label}={mat_a_mean:.3f}(GT={gt_a:.3f})"
              f"  {b_label}={mat_b_mean:.3f}(GT={gt_b:.3f})  -> {out_dir}")

    print("[Phase 2] Complete.")
