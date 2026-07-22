"""Phase 1 of the synthetic study: render the dataset."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image

from idr.config import (DEFAULT_CAMERA, DEFAULT_CFG, LIGHT_ANGLES_DEG, LIGHT_COLOR,
                        LIGHT_INTENSITY, MATERIAL_CONFIGS, PHONG_MATERIAL_CONFIGS,
                        SHININESS_RANGE)
from idr.paths import DATASET_ROOT, RESULTS_ROOT, SYNTHETIC_ROOT
from idr.render import (EnvMap, SHLighting, rasterize_geometry, shade_ct_sh,
                        shade_ct_env, shade_phong_sh, shade_phong_env)
from idr.render.brdf import _get_ggx_sh_lut
from idr.render.ops import _norm
from idr.track.wandb_log import _sh_coeffs_to_env_img, _env_flat_to_img
from idr.data.synthetic_scene import (
    _load_mesh, _make_lights, _make_lights_random_sh, _make_lights_circular,
    _get_light_angles, _scene_suffix, _get_light_entries, _checker_uv,
    _make_checker_map, _make_random_patch_map, _scatter_np, _scatter,
)
from idr.data.synthetic_io import (
    _write_dataset_meta, _read_dataset_meta, _all_renders_exist,
    _save_component_images, _save_render, _save_config_json,
)

def generate_dataset(
    mesh_name:          str            = "sphere",
    width:              int            = 128,
    height:             int            = 128,
    shader:             str            = "all",
    device:             str            = "cuda",
    skip_existing:      bool           = False,
    mat_configs_filter: Optional[set]  = None,
    light_mode:         str            = "directional",
    n_lights:           int            = 6,
    full_circle:        bool           = False,
) -> None:
    """Render material × light × shader combinations.

    shader      : "ct_sh" | "ct_env" | "phong_sh" | "phong_env" | "all"
    light_mode  : "directional" | "random_sh" | "circular"
    n_lights    : number of light configurations per scene
    full_circle : for directional mode — span [0°, 360°) instead of [0°, 90°]
    """
    dev          = device
    mesh         = _load_mesh(mesh_name)
    light_entries = _get_light_entries(light_mode, n_lights, full_circle)
    light_keys    = [k for k, _ in light_entries]
    suffix        = _scene_suffix(light_mode, n_lights, full_circle)

    ct_mat_cfgs    = {k: v for k, v in MATERIAL_CONFIGS.items()
                      if mat_configs_filter is None or k in mat_configs_filter}
    phong_mat_cfgs = {k: v for k, v in PHONG_MATERIAL_CONFIGS.items()
                      if mat_configs_filter is None or k in mat_configs_filter}

    normals_hw, frag_pos_hw, mask_hw, cam_pos = rasterize_geometry(
        mesh, DEFAULT_CAMERA, width=width, height=height, smooth=True, device=dev,
    )
    flat_mask  = mask_hw.reshape(-1)
    normals_m  = normals_hw.reshape(-1, 3)[flat_mask]
    frag_pos_m = frag_pos_hw.reshape(-1, 3)[flat_mask]
    view_m     = _norm(cam_pos.unsqueeze(0) - frag_pos_m)
    M          = int(flat_mask.sum())

    normals_vis = ((normals_hw.cpu().numpy() * 0.5 + 0.5) * 255).clip(0, 255).astype(np.uint8)
    mask_np_gen = mask_hw.cpu().numpy()

    do_ct_sh    = shader in ("ct_sh",    "all")
    do_ct_env   = shader in ("ct_env",   "all")
    do_phong_sh = shader in ("phong_sh", "all")
    do_phong_env= shader in ("phong_env","all")

    # ── CT scenes ────────────────────────────────────────────────────────────
    if do_ct_sh or do_ct_env:
        lut = _get_ggx_sh_lut(torch.device(dev))
        for mat_id, mat_cfg in ct_mat_cfgs.items():
            scene_name   = f"{mesh_name}_{mat_id}{suffix}"
            need_ct_sh   = do_ct_sh  and not (skip_existing and _all_renders_exist(scene_name, "ct_sh",  light_keys))
            need_ct_env  = do_ct_env and not (skip_existing and _all_renders_exist(scene_name, "ct_env", light_keys))
            if not need_ct_sh and not need_ct_env:
                print(f"[Phase 1] {scene_name}  skipped (all renders exist)")
                continue
            n_tiles      = mat_cfg.get("n_tiles", 16)
            seed         = mat_cfg.get("seed", 42)
            normals_m_np = normals_m.cpu().numpy()
            flat_mask_np = flat_mask.cpu().numpy()

            # ── albedo ───────────────────────────────────────────────────────
            if "albedo_range" in mat_cfg:
                low, high   = mat_cfg["albedo_range"]
                albedo_flat = _make_random_patch_map(normals_m_np, low, high, n_tiles, seed)
                albedo_t    = torch.from_numpy(albedo_flat).to(dev)
                albedo_hw   = _scatter_np(albedo_flat, flat_mask_np, height, width)
            elif "albedo_checker" in mat_cfg:
                col_a, col_b = mat_cfg["albedo_checker"]
                albedo_flat  = _make_checker_map(normals_m_np, col_a, col_b, n_tiles)
                albedo_t     = torch.from_numpy(albedo_flat).to(dev)
                albedo_hw    = _scatter_np(albedo_flat, flat_mask_np, height, width)
            else:
                albedo_t  = torch.tensor(mat_cfg["albedo"], device=dev,
                                         dtype=torch.float32).expand(M, 3).contiguous()
                albedo_hw = (np.tile(np.array(mat_cfg["albedo"], dtype=np.float32),
                                     (height, width, 1)) * mask_np_gen[:, :, None])

            # ── metallic ─────────────────────────────────────────────────────
            if "metallic_range" in mat_cfg:
                m_low, m_high  = mat_cfg["metallic_range"]
                met_flat       = _make_random_patch_map(normals_m_np, m_low, m_high, n_tiles, seed + 1)
                metallic_t     = torch.from_numpy(met_flat).to(dev)   # (M, 1)
                metallic_hw    = _scatter_np(met_flat, flat_mask_np, height, width)
                metallic       = float(met_flat.mean())
            else:
                metallic       = mat_cfg["metallic"]
                metallic_t     = None
                metallic_hw    = np.full((height, width, 1), metallic, dtype=np.float32) * mask_np_gen[:, :, None]

            # ── roughness ────────────────────────────────────────────────────
            if "roughness_range" in mat_cfg:
                r_low, r_high  = mat_cfg["roughness_range"]
                rough_flat     = _make_random_patch_map(normals_m_np, r_low, r_high, n_tiles, seed + 2)
                roughness_t    = torch.from_numpy(rough_flat).to(dev)  # (M, 1)
                roughness_hw   = _scatter_np(rough_flat, flat_mask_np, height, width)
                roughness      = float(rough_flat.mean())
            else:
                roughness      = mat_cfg["roughness"]
                roughness_t    = None
                roughness_hw   = np.full((height, width, 1), roughness, dtype=np.float32) * mask_np_gen[:, :, None]

            gt_dir = DATASET_ROOT / scene_name / "gt"
            gt_dir.mkdir(parents=True, exist_ok=True)
            gt_albedo_img = (albedo_hw * 255).clip(0, 255).astype(np.uint8)
            Image.fromarray(gt_albedo_img).save(gt_dir / "albedo.png")
            np.save(gt_dir / "albedo.npy", albedo_hw.astype(np.float32))
            for name, hw in [("metallic", metallic_hw), ("roughness", roughness_hw)]:
                gray = (hw[:, :, 0] * 255).clip(0, 255).astype(np.uint8)
                Image.fromarray(gray, mode="L").save(gt_dir / f"{name}.png")
                np.save(gt_dir / f"{name}.npy", hw.astype(np.float32))
            Image.fromarray(normals_vis).save(gt_dir / "normals.png")

            for light_id, make_fn in light_entries:
                sh_light, env_light, direction, sh_coeffs_np, env_raw = make_fn()
                angle_deg = float(np.degrees(np.arctan2(float(direction[0]), float(direction[2])))) if direction is not None else None

                with torch.no_grad():
                    if need_ct_sh:
                        sh_dir = DATASET_ROOT / scene_name / "ct_sh" / light_id
                        (sh_dir / "components").mkdir(parents=True, exist_ok=True)
                        composite, comps = shade_ct_sh(  # type: ignore[misc]
                            view_m, normals_m, albedo_t,
                            sh_light.coeffs.to(dev),
                            metallic_t if metallic_t is not None else metallic,
                            roughness_t if roughness_t is not None else roughness,
                            lut=lut, return_components=True,
                        )
                        _save_render(composite, flat_mask, height, width, sh_dir / "render.png")
                        _save_component_images(comps, flat_mask, height, width, sh_dir / "components")
                        _save_config_json(sh_dir / "config.json",
                                          mesh_name=mesh_name, mat_cfg=mat_cfg,
                                          angle_deg=angle_deg, direction=direction,
                                          sh_coeffs_np=sh_coeffs_np,
                                          width=width, height=height, light_type="ct_sh",
                                          light_mode=light_mode)
                        sh_env_img = _sh_coeffs_to_env_img(sh_coeffs_np)
                        Image.fromarray((sh_env_img * 255).astype(np.uint8)).save(
                            sh_dir / "sh_env_map.png")
                        np.save(sh_dir / "sh_env_map.npy", sh_env_img.astype(np.float32))

                    if need_ct_env:
                        env_dir = DATASET_ROOT / scene_name / "ct_env" / light_id
                        (env_dir / "components").mkdir(parents=True, exist_ok=True)
                        env_pix_t  = env_light.image_flat.to(dev)
                        env_dirs_t = env_light.dirs.to(dev)
                        env_dw_t   = env_light.solid_angles.to(dev)
                        composite_env, comps_env = shade_ct_env(  # type: ignore[misc]
                            view_m, normals_m, albedo_t,
                            env_pix_t, env_dirs_t, env_dw_t,
                            metallic_t if metallic_t is not None else metallic,
                            roughness_t if roughness_t is not None else roughness,
                            return_components=True,
                        )
                        _save_render(composite_env, flat_mask, height, width, env_dir / "render.png")
                        _save_component_images(comps_env, flat_mask, height, width, env_dir / "components")
                        _save_config_json(env_dir / "config.json",
                                          mesh_name=mesh_name, mat_cfg=mat_cfg,
                                          angle_deg=angle_deg, direction=direction,
                                          sh_coeffs_np=sh_coeffs_np,
                                          width=width, height=height, light_type="ct_env",
                                          light_mode=light_mode)
                        env_img = _env_flat_to_img(env_raw._image_flat,
                                                   *env_raw.image.shape[:2])
                        Image.fromarray((env_img * 255).astype(np.uint8)).save(
                            env_dir / "env_map.png")
                        np.save(env_dir / "env_map.npy", env_img.astype(np.float32))

            _write_dataset_meta(scene_name, light_mode, n_lights, full_circle, light_keys)
            print(f"[Phase 1] {scene_name}  done")

    # ── Phong scenes ─────────────────────────────────────────────────────────
    if do_phong_sh or do_phong_env:
        for mat_id, mat_cfg in phong_mat_cfgs.items():
            scene_name     = f"{mesh_name}_phong_{mat_id}{suffix}"
            need_phong_sh  = do_phong_sh  and not (skip_existing and _all_renders_exist(scene_name, "phong_sh",  light_keys))
            need_phong_env = do_phong_env and not (skip_existing and _all_renders_exist(scene_name, "phong_env", light_keys))
            if not need_phong_sh and not need_phong_env:
                print(f"[Phase 1] {scene_name}  skipped (all renders exist)")
                continue
            n_tiles      = mat_cfg.get("n_tiles", 16)
            seed         = mat_cfg.get("seed", 42)
            normals_m_np = normals_m.cpu().numpy()
            flat_mask_np = flat_mask.cpu().numpy()
            ka, kd = mat_cfg["ka"], mat_cfg["kd"]

            # ── albedo ───────────────────────────────────────────────────────
            if "albedo_range" in mat_cfg:
                low, high   = mat_cfg["albedo_range"]
                albedo_flat = _make_random_patch_map(normals_m_np, low, high, n_tiles, seed)
                albedo_t    = torch.from_numpy(albedo_flat).to(dev)
                albedo_hw   = _scatter_np(albedo_flat, flat_mask_np, height, width)
            elif "albedo_checker" in mat_cfg:
                col_a, col_b = mat_cfg["albedo_checker"]
                albedo_flat  = _make_checker_map(normals_m_np, col_a, col_b, n_tiles)
                albedo_t     = torch.from_numpy(albedo_flat).to(dev)
                albedo_hw    = _scatter_np(albedo_flat, flat_mask_np, height, width)
            else:
                albedo_t  = torch.tensor(mat_cfg["albedo"], device=dev,
                                         dtype=torch.float32).expand(M, 3).contiguous()
                albedo_hw = (np.tile(np.array(mat_cfg["albedo"], dtype=np.float32),
                                     (height, width, 1)) * mask_np_gen[:, :, None])

            # ── shininess ────────────────────────────────────────────────────
            if "shininess_range" in mat_cfg:
                s_low, s_high = mat_cfg["shininess_range"]
                shin_flat = _make_random_patch_map(normals_m_np, s_low, s_high, n_tiles, seed + 1)
                shin_t    = torch.from_numpy(shin_flat).to(dev)            # (M, 1)
                shin_hw   = _scatter_np(shin_flat, flat_mask_np, height, width)
                shin = float(shin_flat.mean())
            elif "shininess_checker" in mat_cfg:
                shin_a, shin_b = mat_cfg["shininess_checker"]
                shin_flat = _make_checker_map(normals_m_np, shin_a, shin_b, n_tiles)
                shin_t    = torch.from_numpy(shin_flat).to(dev)
                shin_hw   = _scatter_np(shin_flat, flat_mask_np, height, width)
                shin = float(np.mean([shin_a, shin_b]))
            else:
                shin   = mat_cfg["shininess"]
                shin_t = None
                shin_hw = np.full((height, width, 1), shin, dtype=np.float32) * mask_np_gen[:, :, None]

            # ── ks ───────────────────────────────────────────────────────────
            if "ks_range" in mat_cfg:
                k_low, k_high = mat_cfg["ks_range"]
                ks_flat = _make_random_patch_map(normals_m_np, k_low, k_high, n_tiles, seed + 2)
                ks_t    = torch.from_numpy(ks_flat).to(dev)                # (M, 1)
                ks_hw   = _scatter_np(ks_flat, flat_mask_np, height, width)
                ks = float(ks_flat.mean())
            elif "ks_checker" in mat_cfg:
                ks_a, ks_b = mat_cfg["ks_checker"]
                ks_flat = _make_checker_map(normals_m_np, ks_a, ks_b, n_tiles)
                ks_t    = torch.from_numpy(ks_flat).to(dev)
                ks_hw   = _scatter_np(ks_flat, flat_mask_np, height, width)
                ks = float(np.mean([ks_a, ks_b]))
            else:
                ks   = mat_cfg["ks"]
                ks_t = None
                ks_hw = np.full((height, width, 1), ks, dtype=np.float32) * mask_np_gen[:, :, None]

            gt_dir = DATASET_ROOT / scene_name / "gt"
            gt_dir.mkdir(parents=True, exist_ok=True)
            Image.fromarray((albedo_hw * 255).clip(0, 255).astype(np.uint8)).save(
                gt_dir / "albedo.png")
            np.save(gt_dir / "albedo.npy", albedo_hw.astype(np.float32))

            Image.fromarray(
                (shin_hw[:, :, 0] / SHININESS_RANGE[1] * 255).clip(0, 255).astype(np.uint8),
                mode="L").save(gt_dir / "shininess.png")
            np.save(gt_dir / "shininess.npy", shin_hw.astype(np.float32))

            Image.fromarray(
                (ks_hw[:, :, 0] * 255).clip(0, 255).astype(np.uint8),
                mode="L").save(gt_dir / "ks.png")
            np.save(gt_dir / "ks.npy", ks_hw.astype(np.float32))
            Image.fromarray(normals_vis).save(gt_dir / "normals.png")

            for light_id, make_fn in light_entries:
                sh_light, env_light, direction, sh_coeffs_np, env_raw = make_fn()
                angle_deg = float(np.degrees(np.arctan2(float(direction[0]), float(direction[2])))) if direction is not None else None

                with torch.no_grad():
                    if need_phong_sh:
                        ps_dir = DATASET_ROOT / scene_name / "phong_sh" / light_id
                        (ps_dir / "components").mkdir(parents=True, exist_ok=True)
                        composite_ps, comps_ps = shade_phong_sh(  # type: ignore[misc]
                            view_m, normals_m,
                            ka, kd,
                            ks_t   if ks_t   is not None else ks,
                            shin_t if shin_t is not None else shin,
                            albedo_t, sh_light.coeffs.to(dev),
                            return_components=True,
                        )
                        _save_render(composite_ps, flat_mask, height, width, ps_dir / "render.png")
                        _save_component_images(comps_ps, flat_mask, height, width, ps_dir / "components")
                        _save_config_json(ps_dir / "config.json",
                                          mesh_name=mesh_name, mat_cfg=mat_cfg,
                                          angle_deg=angle_deg, direction=direction,
                                          sh_coeffs_np=sh_coeffs_np,
                                          width=width, height=height, light_type="phong_sh",
                                          light_mode=light_mode)
                        sh_env_img = _sh_coeffs_to_env_img(sh_coeffs_np)
                        Image.fromarray((sh_env_img * 255).astype(np.uint8)).save(
                            ps_dir / "sh_env_map.png")
                        np.save(ps_dir / "sh_env_map.npy", sh_env_img.astype(np.float32))

                    if need_phong_env:
                        pe_dir = DATASET_ROOT / scene_name / "phong_env" / light_id
                        (pe_dir / "components").mkdir(parents=True, exist_ok=True)
                        env_pix_t  = env_light.image_flat.to(dev)
                        env_dirs_t = env_light.dirs.to(dev)
                        env_dw_t   = env_light.solid_angles.to(dev)
                        composite_pe, comps_pe = shade_phong_env(  # type: ignore[misc]
                            view_m, normals_m, albedo_t,
                            env_pix_t, env_dirs_t, env_dw_t,
                            ka, kd,
                            ks_t   if ks_t   is not None else ks,
                            shin_t if shin_t is not None else shin,
                            return_components=True,
                        )
                        _save_render(composite_pe, flat_mask, height, width, pe_dir / "render.png")
                        _save_component_images(comps_pe, flat_mask, height, width, pe_dir / "components")
                        _save_config_json(pe_dir / "config.json",
                                          mesh_name=mesh_name, mat_cfg=mat_cfg,
                                          angle_deg=angle_deg, direction=direction,
                                          sh_coeffs_np=sh_coeffs_np,
                                          width=width, height=height, light_type="phong_env",
                                          light_mode=light_mode)
                        env_img = _env_flat_to_img(env_raw._image_flat,
                                                   *env_raw.image.shape[:2])
                        Image.fromarray((env_img * 255).astype(np.uint8)).save(
                            pe_dir / "env_map.png")
                        np.save(pe_dir / "env_map.npy", env_img.astype(np.float32))

            _write_dataset_meta(scene_name, light_mode, n_lights, full_circle, light_keys)
            print(f"[Phase 1] {scene_name}  done")

    print("[Phase 1] Complete.")
