"""
Synthetic CT + Phong dataset: generation (Phase 1) + decomposition (Phase 2).

Phase 1 renders a sphere (or Suzanne / Stanford bunny) under material configs
× 6 light setups × shader types.

  Shaders:
    ct_sh    — Cook-Torrance + SH lighting
    ct_env   — Cook-Torrance + env-map lighting
    phong_sh — Phong + SH lighting
    phong_env— Phong + env-map lighting

Phase 2 runs intrinsic decomposition recovering per-pixel material + per-image lighting:
    ct_sh    : albedo, metallic, roughness, SH coefficients
    ct_env   : albedo, metallic, roughness, env-map pixels
    phong_sh : albedo, shininess, ks, SH coefficients
    phong_env: albedo, shininess, ks, env-map pixels

Usage
-----
    python raw_optimizer/synthetic_ct_dataset.py --phase 1 --shader ct_sh --mesh sphere
    python raw_optimizer/synthetic_ct_dataset.py --phase 2 --shader phong_sh --mesh sphere
    python raw_optimizer/synthetic_ct_dataset.py --phase 2 --shader all --mat sphere_default
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import wandb
from PIL import Image

_REPO_ROOT = Path(__file__).parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from raw_renderer_gpu import (
    rasterize_geometry, shade_ct_sh, shade_ct_env, shade_phong_sh, shade_phong_env,
    SHLight, EnvMapLightGPU,
    Camera, EnvMap, SHLighting, generate_mesh, load_obj,
)
from raw_renderer_gpu.rasterizer import _norm, _get_ggx_sh_lut, _sh_irradiance
from raw_optimizer.optimizer import _tv
from raw_optimizer.helper import _albedo_rmse


# ─────────────────────────────────────── constants ───────────────────────────

SYNTHETIC_ROOT = _REPO_ROOT / "synthetic_ct"
DATASET_ROOT   = SYNTHETIC_ROOT / "dataset"
RESULTS_ROOT   = SYNTHETIC_ROOT / "results"

MATERIAL_CONFIGS: dict[str, dict] = {
    "default":     dict(albedo=[0.5, 0.5, 0.5], metallic=0.1, roughness=0.4),
    "albedo_0":    dict(albedo=[0.8, 0.3, 0.2], metallic=0.1, roughness=0.4),
    "albedo_1":    dict(albedo=[0.2, 0.5, 0.8], metallic=0.1, roughness=0.4),
    "metallic_0":  dict(albedo=[0.5, 0.5, 0.5], metallic=0.0, roughness=0.4),
    "metallic_1":  dict(albedo=[0.5, 0.5, 0.5], metallic=0.8, roughness=0.4),
    "roughness_0": dict(albedo=[0.5, 0.5, 0.5], metallic=0.1, roughness=0.1),
    "roughness_1": dict(albedo=[0.5, 0.5, 0.5], metallic=0.1, roughness=0.8),
}

PHONG_MATERIAL_CONFIGS: dict[str, dict] = {
    "default":     dict(albedo=[0.5, 0.5, 0.5], shininess=32.0,  ks=0.5, ka=0.0, kd=1.0),
    "albedo_0":    dict(albedo=[0.8, 0.3, 0.2], shininess=32.0,  ks=0.5, ka=0.0, kd=1.0),
    "albedo_1":    dict(albedo=[0.2, 0.5, 0.8], shininess=32.0,  ks=0.5, ka=0.0, kd=1.0),
    "shininess_0": dict(albedo=[0.5, 0.5, 0.5], shininess=4.0,   ks=0.5, ka=0.0, kd=1.0),
    "shininess_1": dict(albedo=[0.5, 0.5, 0.5], shininess=128.0, ks=0.5, ka=0.0, kd=1.0),
    "ks_0":        dict(albedo=[0.5, 0.5, 0.5], shininess=32.0,  ks=0.1, ka=0.0, kd=1.0),
    "ks_1":        dict(albedo=[0.5, 0.5, 0.5], shininess=32.0,  ks=0.9, ka=0.0, kd=1.0),
}

SHININESS_RANGE = (1.0, 256.0)

LIGHT_ANGLES_DEG = [0, 18, 36, 54, 72, 90]
LIGHT_COLOR      = np.array([1.0, 0.9, 0.8], dtype=np.float32)
LIGHT_INTENSITY  = 2.0

DEFAULT_CAMERA = Camera(  # type: ignore[call-arg]
    position=np.array([0.0, 0.0, 3.0], dtype=np.float32),
    target  =np.array([0.0, 0.0, 0.0], dtype=np.float32),
)

DEFAULT_CFG = dict(
    optimizer      = "LBFGS",
    n_iter         = 50,
    lbfgs_max_iter = 20,
    lr             = 1.0,
    lambda_sparse  = 0.0,
    lambda_white   = 0.0,
    log_every      = 20,
    loss           = "L2",
    shininess_min  = SHININESS_RANGE[0],
    shininess_max  = SHININESS_RANGE[1],
)


# ─────────────────────────────────────── helpers ─────────────────────────────

def _load_mesh(name: str):
    if name == "sphere":
        return generate_mesh("sphere")
    if name == "suzanne":
        return load_obj(str(_REPO_ROOT / "assets" / "obj" / "suzanne.obj"))
    if name == "bunny":
        return load_obj(str(_REPO_ROOT / "assets" / "obj" / "stanford-bunny.obj"))
    raise ValueError(f"Unknown mesh {name!r}. Choose sphere / suzanne / bunny.")


def _make_lights(angle_deg: float):
    """Build SH and env-map lights for a single XZ-plane rotation angle.

    0° = frontal (+Z toward camera); 90° = side (+X).
    Returns (sh_light, env_light, direction_np, sh_coeffs_np, env_raw).
    """
    theta     = np.radians(angle_deg)
    direction = np.array([np.sin(theta), 0.0, np.cos(theta)], dtype=np.float32)
    sh_raw    = SHLighting.directional(direction, LIGHT_COLOR, intensity=LIGHT_INTENSITY)
    sh_light  = SHLight(coeffs=torch.from_numpy(sh_raw.coeffs))
    env_raw   = EnvMap.from_sh(sh_raw)
    env_light = EnvMapLightGPU(
        dirs        =torch.from_numpy(env_raw._dirs),
        image_flat  =torch.from_numpy(env_raw._image_flat),
        solid_angles=torch.from_numpy(env_raw._solid_angles),
    )
    return sh_light, env_light, direction, sh_raw.coeffs, env_raw


def _sh_coeffs_to_env_img(coeffs: np.ndarray, resolution: int = 64) -> np.ndarray:
    """(9,3) SH coefficients → (H,W,3) float32 image normalized to [0,1]."""
    img = EnvMap.from_sh(SHLighting(coeffs), resolution=resolution).image  # (H,W,3)
    mx = float(img.max())
    return img / max(mx, 1e-8)


def _env_flat_to_img(env_flat: np.ndarray, env_H: int, env_W: int) -> np.ndarray:
    """(P,3) flat env-map → (H,W,3) float32 image normalized to [0,1]."""
    img = env_flat.reshape(env_H, env_W, 3)
    mx = float(img.max())
    return img / max(mx, 1e-8)


def _scatter(flat: torch.Tensor, flat_mask: torch.Tensor, H: int, W: int) -> torch.Tensor:
    """Scatter foreground pixels (M, C) back to (H, W, C)."""
    C   = flat.shape[-1] if flat.dim() > 1 else 1
    buf = torch.zeros(H * W, C, device=flat.device, dtype=torch.float32)
    buf[flat_mask] = flat.reshape(-1, C).float()
    return buf.reshape(H, W, C)


def _save_component_images(
    components: dict[str, torch.Tensor],
    flat_mask:  torch.Tensor,
    H: int, W: int,
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    scalar_names    = {"NdotV", "G1"}
    direction_names = {"R"}

    def _to_u8(arr):
        return (arr.clip(0, 1) * 255).astype(np.uint8)

    for name, comp in components.items():
        comp_cpu = comp.detach().float().cpu()
        if name in scalar_names:
            full = _scatter(comp_cpu, flat_mask, H, W).numpy()
            fg   = full[flat_mask.reshape(H, W).cpu().numpy()]
            vmin, vmax = float(fg.min()), float(fg.max())
            normed = (full - vmin) / max(vmax - vmin, 1e-8)
            gray = (normed.squeeze(-1) * 255).clip(0, 255).astype(np.uint8)
            Image.fromarray(gray, mode="L").save(out_dir / f"{name}.png")
        elif name in direction_names:
            full = _scatter(comp_cpu, flat_mask, H, W).numpy()
            Image.fromarray(_to_u8(full * 0.5 + 0.5)).save(out_dir / f"{name}.png")
        else:
            C    = comp_cpu.shape[-1] if comp_cpu.dim() > 1 else 1
            full = _scatter(comp_cpu, flat_mask, H, W).numpy()
            if C == 1:
                full = np.repeat(full, 3, axis=-1)
            Image.fromarray(_to_u8(full)).save(out_dir / f"{name}.png")


def _save_render(composite: torch.Tensor, flat_mask: torch.Tensor,
                 H: int, W: int, path: Path) -> None:
    img = _scatter(composite.detach(), flat_mask, H, W).cpu().numpy()
    Image.fromarray((img.clip(0, 1) * 255).astype(np.uint8)).save(path)


def _save_config_json(path: Path, *, mesh_name, mat_cfg, angle_deg, direction,
                      sh_coeffs_np, width, height, light_type) -> None:
    with open(path, "w") as fh:
        json.dump({
            "mesh_name": mesh_name,
            "material":  mat_cfg,
            "light": {
                "angle_deg":  angle_deg,
                "direction":  direction.tolist(),
                "color":      LIGHT_COLOR.tolist(),
                "intensity":  LIGHT_INTENSITY,
                "sh_coeffs":  sh_coeffs_np.tolist(),
            },
            "render_resolution": [width, height],
            "light_type": light_type,
        }, fh, indent=2)


# ─────────────────────────────────────── Phase 1 ─────────────────────────────

def generate_dataset(
    mesh_name: str = "sphere",
    width:     int = 128,
    height:    int = 128,
    shader:    str = "all",
    device:    str = "cuda",
) -> None:
    """Render material × light × shader combinations.

    shader : "ct_sh" | "ct_env" | "phong_sh" | "phong_env" | "all"
    """
    dev  = device
    mesh = _load_mesh(mesh_name)

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
        for mat_id, mat_cfg in MATERIAL_CONFIGS.items():
            scene_name = f"{mesh_name}_{mat_id}"
            albedo_t   = torch.tensor(mat_cfg["albedo"], device=dev,
                                      dtype=torch.float32).expand(M, 3).contiguous()
            metallic   = mat_cfg["metallic"]
            roughness  = mat_cfg["roughness"]

            gt_dir = DATASET_ROOT / scene_name / "gt"
            gt_dir.mkdir(parents=True, exist_ok=True)
            gt_albedo_img = np.tile(
                np.array(mat_cfg["albedo"], dtype=np.float32), (height, width, 1))
            gt_albedo_img = (gt_albedo_img * mask_np_gen[:, :, None] * 255).clip(0, 255).astype(np.uint8)
            Image.fromarray(gt_albedo_img).save(gt_dir / "albedo.png")
            for name, val in [("metallic", metallic), ("roughness", roughness)]:
                gray = (np.full((height, width), val, dtype=np.float32)
                        * mask_np_gen * 255).clip(0, 255).astype(np.uint8)
                Image.fromarray(gray, mode="L").save(gt_dir / f"{name}.png")
            Image.fromarray(normals_vis).save(gt_dir / "normals.png")

            for angle_deg in LIGHT_ANGLES_DEG:
                sh_light, env_light, direction, sh_coeffs_np, env_raw = _make_lights(angle_deg)
                light_id = f"light_{int(angle_deg):02d}deg"

                with torch.no_grad():
                    if do_ct_sh:
                        sh_dir = DATASET_ROOT / scene_name / "ct_sh" / light_id
                        (sh_dir / "components").mkdir(parents=True, exist_ok=True)
                        composite, comps = shade_ct_sh(  # type: ignore[misc]
                            view_m, normals_m, albedo_t,
                            sh_light.coeffs.to(dev), metallic, roughness,
                            lut=lut, return_components=True,
                        )
                        _save_render(composite, flat_mask, height, width, sh_dir / "render.png")
                        _save_component_images(comps, flat_mask, height, width, sh_dir / "components")
                        _save_config_json(sh_dir / "config.json",
                                          mesh_name=mesh_name, mat_cfg=mat_cfg,
                                          angle_deg=angle_deg, direction=direction,
                                          sh_coeffs_np=sh_coeffs_np,
                                          width=width, height=height, light_type="ct_sh")
                        sh_env_img = _sh_coeffs_to_env_img(sh_coeffs_np)
                        Image.fromarray((sh_env_img * 255).astype(np.uint8)).save(
                            sh_dir / "sh_env_map.png")

                    if do_ct_env:
                        env_dir = DATASET_ROOT / scene_name / "ct_env" / light_id
                        (env_dir / "components").mkdir(parents=True, exist_ok=True)
                        env_pix_t  = env_light.image_flat.to(dev)
                        env_dirs_t = env_light.dirs.to(dev)
                        env_dw_t   = env_light.solid_angles.to(dev)
                        composite_env, comps_env = shade_ct_env(  # type: ignore[misc]
                            view_m, normals_m, albedo_t,
                            env_pix_t, env_dirs_t, env_dw_t,
                            metallic, roughness, return_components=True,
                        )
                        _save_render(composite_env, flat_mask, height, width, env_dir / "render.png")
                        _save_component_images(comps_env, flat_mask, height, width, env_dir / "components")
                        _save_config_json(env_dir / "config.json",
                                          mesh_name=mesh_name, mat_cfg=mat_cfg,
                                          angle_deg=angle_deg, direction=direction,
                                          sh_coeffs_np=sh_coeffs_np,
                                          width=width, height=height, light_type="ct_env")
                        env_img = _env_flat_to_img(env_raw._image_flat,
                                                   *env_raw.image.shape[:2])
                        Image.fromarray((env_img * 255).astype(np.uint8)).save(
                            env_dir / "env_map.png")

            print(f"[Phase 1] {scene_name}  done")

    # ── Phong scenes ─────────────────────────────────────────────────────────
    if do_phong_sh or do_phong_env:
        for mat_id, mat_cfg in PHONG_MATERIAL_CONFIGS.items():
            scene_name = f"{mesh_name}_phong_{mat_id}"
            albedo_t   = torch.tensor(mat_cfg["albedo"], device=dev,
                                      dtype=torch.float32).expand(M, 3).contiguous()
            ka, kd     = mat_cfg["ka"],  mat_cfg["kd"]
            ks, shin   = mat_cfg["ks"],  mat_cfg["shininess"]

            gt_dir = DATASET_ROOT / scene_name / "gt"
            gt_dir.mkdir(parents=True, exist_ok=True)
            gt_albedo_img = np.tile(
                np.array(mat_cfg["albedo"], dtype=np.float32), (height, width, 1))
            gt_albedo_img = (gt_albedo_img * mask_np_gen[:, :, None] * 255).clip(0, 255).astype(np.uint8)
            Image.fromarray(gt_albedo_img).save(gt_dir / "albedo.png")
            shin_gray = (np.full((height, width), shin / SHININESS_RANGE[1], dtype=np.float32)
                         * mask_np_gen * 255).clip(0, 255).astype(np.uint8)
            Image.fromarray(shin_gray, mode="L").save(gt_dir / "shininess.png")
            ks_gray = (np.full((height, width), ks, dtype=np.float32)
                       * mask_np_gen * 255).clip(0, 255).astype(np.uint8)
            Image.fromarray(ks_gray, mode="L").save(gt_dir / "ks.png")
            Image.fromarray(normals_vis).save(gt_dir / "normals.png")

            for angle_deg in LIGHT_ANGLES_DEG:
                sh_light, env_light, direction, sh_coeffs_np, env_raw = _make_lights(angle_deg)
                light_id = f"light_{int(angle_deg):02d}deg"

                with torch.no_grad():
                    if do_phong_sh:
                        ps_dir = DATASET_ROOT / scene_name / "phong_sh" / light_id
                        (ps_dir / "components").mkdir(parents=True, exist_ok=True)
                        composite_ps, comps_ps = shade_phong_sh(  # type: ignore[misc]
                            view_m, normals_m,
                            ka, kd, ks, shin,
                            albedo_t, sh_light.coeffs.to(dev),
                            return_components=True,
                        )
                        _save_render(composite_ps, flat_mask, height, width, ps_dir / "render.png")
                        _save_component_images(comps_ps, flat_mask, height, width, ps_dir / "components")
                        _save_config_json(ps_dir / "config.json",
                                          mesh_name=mesh_name, mat_cfg=mat_cfg,
                                          angle_deg=angle_deg, direction=direction,
                                          sh_coeffs_np=sh_coeffs_np,
                                          width=width, height=height, light_type="phong_sh")
                        sh_env_img = _sh_coeffs_to_env_img(sh_coeffs_np)
                        Image.fromarray((sh_env_img * 255).astype(np.uint8)).save(
                            ps_dir / "sh_env_map.png")

                    if do_phong_env:
                        pe_dir = DATASET_ROOT / scene_name / "phong_env" / light_id
                        (pe_dir / "components").mkdir(parents=True, exist_ok=True)
                        env_pix_t  = env_light.image_flat.to(dev)
                        env_dirs_t = env_light.dirs.to(dev)
                        env_dw_t   = env_light.solid_angles.to(dev)
                        composite_pe, comps_pe = shade_phong_env(  # type: ignore[misc]
                            view_m, normals_m, albedo_t,
                            env_pix_t, env_dirs_t, env_dw_t,
                            ka, kd, ks, shin,
                            return_components=True,
                        )
                        _save_render(composite_pe, flat_mask, height, width, pe_dir / "render.png")
                        _save_component_images(comps_pe, flat_mask, height, width, pe_dir / "components")
                        _save_config_json(pe_dir / "config.json",
                                          mesh_name=mesh_name, mat_cfg=mat_cfg,
                                          angle_deg=angle_deg, direction=direction,
                                          sh_coeffs_np=sh_coeffs_np,
                                          width=width, height=height, light_type="phong_env")
                        env_img = _env_flat_to_img(env_raw._image_flat,
                                                   *env_raw.image.shape[:2])
                        Image.fromarray((env_img * 255).astype(np.uint8)).save(
                            pe_dir / "env_map.png")

            print(f"[Phase 1] {scene_name}  done")

    print("[Phase 1] Complete.")


# ─────────────────────────────────────── Phase 2 helpers ─────────────────────

def _make_optimizer(params, cfg):
    if cfg["optimizer"] == "LBFGS":
        return torch.optim.LBFGS(
            params, lr=cfg["lr"],
            max_iter=cfg["lbfgs_max_iter"],
            line_search_fn="strong_wolfe",
        )
    return torch.optim.Adam(params, lr=cfg["lr"])


def _loss_fn(recon, target, mask_t, mode):
    diff = (recon - target).abs() if mode == "L1" else (recon - target) ** 2
    return diff[mask_t.expand_as(diff)].mean()


def _opt_step(opt, forward_fn, cfg):
    """Single optimizer step; returns (total_loss, loss_data, loss_sparse, loss_white)."""
    if cfg["optimizer"] == "LBFGS":
        def closure():
            opt.zero_grad()
            loss, *_ = forward_fn()
            loss.backward()
            return loss
        opt.step(closure)
        with torch.no_grad():
            return forward_fn()
    else:
        opt.zero_grad()
        result = forward_fn()
        result[0].backward()
        opt.step()
        return result


# ─────────────────────────────────────── Phase 2: SH optimizer ───────────────

def _optimize_ct_sh(
    images:       list,
    normals_hw:   torch.Tensor,
    frag_pos_hw:  torch.Tensor,
    mask_hw:      torch.Tensor,
    cam_pos:      torch.Tensor,
    gt_metallic:  float,
    gt_roughness: float,
    cfg:          dict,
    wandb_run=None,
    gt_sh_coeffs: Optional[list] = None,   # list of N (9,3) GT arrays
) -> tuple:
    """
    Full CT+SH decomposition: recovers albedo, per-image SH, metallic, roughness.
    Returns (albedo [H,W,3], sh [N,9,3], metallic float, roughness float, shadings, history).
    """
    dev    = normals_hw.device
    H, W   = normals_hw.shape[:2]
    N_imgs = len(images)

    def _t(x):
        return torch.from_numpy(np.asarray(x, np.float32)).to(dev) \
            if not isinstance(x, torch.Tensor) else x.to(dev, torch.float32)

    imgs_t    = torch.stack([_t(img) for img in images])
    flat_mask = mask_hw.reshape(-1)
    N_m       = normals_hw.reshape(-1, 3)[flat_mask]
    fp_m      = frag_pos_hw.reshape(-1, 3)[flat_mask]
    view_m    = _norm(cam_pos.unsqueeze(0) - fp_m)
    lut       = _get_ggx_sh_lut(dev)
    mask_t    = mask_hw.unsqueeze(-1).to(dev)

    # Learnable params
    log_albedo = torch.log(imgs_t.mean(0).clamp(0.05, 0.95)).requires_grad_(True)
    sh_init         = torch.zeros(N_imgs, 9, 3, device=dev)
    sh_init[:, 0, :] = 1.5
    sh_coeffs = sh_init.clone().requires_grad_(True)
    # per-pixel metallic/roughness; sigmoid maps to (0,1), roughness clamped ≥ 0.05
    metallic_raw  = torch.zeros(H, W, 1, device=dev).requires_grad_(True)
    roughness_raw = torch.zeros(H, W, 1, device=dev).requires_grad_(True)

    opt = _make_optimizer([log_albedo, sh_coeffs, metallic_raw, roughness_raw], cfg)

    def _forward():
        albedo      = torch.exp(log_albedo)
        albedo_m    = albedo.reshape(-1, 3)[flat_mask]
        metallic    = torch.sigmoid(metallic_raw)                 # (H,W,1)
        roughness   = 0.05 + 0.9 * torch.sigmoid(roughness_raw)  # (H,W,1)
        metallic_m  = metallic.reshape(-1, 1)[flat_mask]          # (M,1)
        roughness_m = roughness.reshape(-1, 1)[flat_mask]         # (M,1)
        loss_data = albedo.new_zeros(())
        for k in range(N_imgs):
            recon_m = shade_ct_sh(view_m, N_m, albedo_m, sh_coeffs[k],
                                  metallic_m, roughness_m, lut=lut)
            recon = albedo.new_zeros(H, W, 3)
            recon.reshape(-1, 3)[flat_mask] = recon_m
            loss_data = loss_data + _loss_fn(recon, imgs_t[k], mask_t, cfg["loss"])
        loss_sparse = cfg["lambda_sparse"] * _tv(log_albedo.permute(2, 0, 1))
        loss_white  = cfg["lambda_white"]  * (torch.exp(log_albedo).mean() - 0.5) ** 2
        return loss_data + loss_sparse + loss_white, loss_data, loss_sparse, loss_white

    history = []
    t0 = time.perf_counter()
    for i in range(cfg["n_iter"]):
        loss, l_d, l_s, l_w = _opt_step(opt, _forward, cfg)
        if i % cfg["log_every"] == 0:
            elapsed = time.perf_counter() - t0
            with torch.no_grad():
                met_map = torch.sigmoid(metallic_raw).detach()               # (H,W,1)
                rou_map = (0.05 + 0.9 * torch.sigmoid(roughness_raw)).detach()
                met = float(met_map[mask_hw].mean())
                rou = float(rou_map[mask_hw].mean())
            history.append(float(loss))
            print(f"  [{i:4d}] {elapsed:6.1f}s  loss={float(loss):.3e}  data={float(l_d):.3e}"
                  f"  metallic={met:.3f}  roughness={rou:.3f}")
            if wandb_run is not None:
                est_sh_np = sh_coeffs.detach().cpu().numpy()  # (N, 9, 3)
                wandb_run.log({
                    "loss": float(loss), "loss_data": float(l_d),
                    "loss_sparse": float(l_s), "loss_white": float(l_w),
                    "elapsed_s": elapsed,
                    "pred_albedo":     wandb.Image(
                        torch.exp(log_albedo).clamp(0, 1).detach().cpu().numpy()),
                    "pred_metallic":   wandb.Image(met_map.squeeze(-1).cpu().numpy()),
                    "pred_roughness":  wandb.Image(rou_map.squeeze(-1).cpu().numpy()),
                    "est_sh_env_maps": [wandb.Image(_sh_coeffs_to_env_img(est_sh_np[k]))
                                        for k in range(N_imgs)],
                    "metallic_mean":      met,
                    "roughness_mean":     rou,
                    "metallic_err_mean":  abs(met - gt_metallic),
                    "roughness_err_mean": abs(rou - gt_roughness),
                }, step=i)
    total_time = time.perf_counter() - t0

    with torch.no_grad():
        albedo_out = torch.exp(log_albedo).clamp(0, 1).cpu().numpy()
        sh_out     = sh_coeffs.cpu().numpy()
        met_out    = torch.sigmoid(metallic_raw).cpu().numpy()               # (H,W,1)
        rou_out    = (0.05 + 0.9 * torch.sigmoid(roughness_raw)).cpu().numpy()
        met_m      = torch.from_numpy(met_out).to(dev).reshape(-1, 1)[flat_mask]
        rou_m      = torch.from_numpy(rou_out).to(dev).reshape(-1, 1)[flat_mask]
        albedo_t2  = torch.exp(log_albedo)
        shadings   = []
        for k in range(N_imgs):
            albedo_m = albedo_t2.reshape(-1, 3)[flat_mask]
            recon_m  = shade_ct_sh(view_m, N_m, albedo_m, sh_coeffs[k],
                                   met_m, rou_m, lut=lut)
            s = albedo_t2.new_zeros(H, W, 3)
            s.reshape(-1, 3)[flat_mask] = recon_m
            s *= mask_t
            shadings.append(s.cpu().numpy())

    return albedo_out, sh_out, met_out, rou_out, shadings, history, total_time


# ─────────────────────────────────────── Phase 2: env-map optimizer ──────────

def _optimize_ct_env(
    images:      list,
    normals_hw:  torch.Tensor,
    frag_pos_hw: torch.Tensor,
    mask_hw:     torch.Tensor,
    cam_pos:     torch.Tensor,
    gt_metallic: float,
    gt_roughness: float,
    env_dirs:    np.ndarray,
    env_dw:      np.ndarray,
    cfg:         dict,
    wandb_run=None,
    env_H: int = 64,
    env_W: int = 128,
) -> tuple:
    """
    Full CT env-map decomposition: recovers albedo, per-image env-map pixels,
    metallic, roughness via shade_ct_env.
    Returns (albedo [H,W,3], env_maps [N,P,3], metallic float, roughness float,
             shadings, history).
    """
    dev    = normals_hw.device
    H, W   = normals_hw.shape[:2]
    N_imgs = len(images)
    P      = env_dirs.shape[0]

    def _t(x):
        return torch.from_numpy(np.asarray(x, np.float32)).to(dev) \
            if not isinstance(x, torch.Tensor) else x.to(dev, torch.float32)

    imgs_t     = torch.stack([_t(img) for img in images])
    flat_mask  = mask_hw.reshape(-1)
    N_m        = normals_hw.reshape(-1, 3)[flat_mask]
    fp_m       = frag_pos_hw.reshape(-1, 3)[flat_mask]
    view_m     = _norm(cam_pos.unsqueeze(0) - fp_m)
    env_dirs_t = _t(env_dirs)
    env_dw_t   = _t(env_dw)
    mask_t     = mask_hw.unsqueeze(-1).to(dev)

    # Learnable params
    log_albedo     = torch.log(imgs_t.mean(0).clamp(0.05, 0.95)).requires_grad_(True)
    # env map pixels: per-image, softplus ensures positivity
    env_raw_params = torch.zeros(N_imgs, P, 3, device=dev).requires_grad_(True)
    # per-pixel metallic/roughness maps
    metallic_raw   = torch.zeros(H, W, 1, device=dev).requires_grad_(True)
    roughness_raw  = torch.zeros(H, W, 1, device=dev).requires_grad_(True)

    opt = _make_optimizer(
        [log_albedo, env_raw_params, metallic_raw, roughness_raw], cfg)

    def _forward():
        import torch.nn.functional as F
        albedo      = torch.exp(log_albedo)
        albedo_m    = albedo.reshape(-1, 3)[flat_mask]
        metallic    = torch.sigmoid(metallic_raw)                 # (H,W,1)
        roughness   = 0.05 + 0.9 * torch.sigmoid(roughness_raw)  # (H,W,1)
        metallic_m  = metallic.reshape(-1, 1)[flat_mask]          # (M,1)
        roughness_m = roughness.reshape(-1, 1)[flat_mask]         # (M,1)
        loss_data = albedo.new_zeros(())
        for k in range(N_imgs):
            env_pix_k = F.softplus(env_raw_params[k])
            recon_m   = shade_ct_env(view_m, N_m, albedo_m,
                                     env_pix_k, env_dirs_t, env_dw_t,
                                     metallic_m, roughness_m)
            recon = albedo.new_zeros(H, W, 3)
            recon.reshape(-1, 3)[flat_mask] = recon_m
            loss_data = loss_data + _loss_fn(recon, imgs_t[k], mask_t, cfg["loss"])
        loss_sparse = cfg["lambda_sparse"] * _tv(log_albedo.permute(2, 0, 1))
        loss_white  = cfg["lambda_white"]  * (torch.exp(log_albedo).mean() - 0.5) ** 2
        return loss_data + loss_sparse + loss_white, loss_data, loss_sparse, loss_white

    history = []
    t0 = time.perf_counter()
    for i in range(cfg["n_iter"]):
        loss, l_d, l_s, l_w = _opt_step(opt, _forward, cfg)
        if i % cfg["log_every"] == 0:
            import torch.nn.functional as F
            elapsed = time.perf_counter() - t0
            with torch.no_grad():
                met_map = torch.sigmoid(metallic_raw).detach()
                rou_map = (0.05 + 0.9 * torch.sigmoid(roughness_raw)).detach()
                met = float(met_map[mask_hw].mean())
                rou = float(rou_map[mask_hw].mean())
            history.append(float(loss))
            print(f"  [{i:4d}] {elapsed:6.1f}s  loss={float(loss):.3e}  data={float(l_d):.3e}"
                  f"  metallic={met:.3f}  roughness={rou:.3f}")
            if wandb_run is not None:
                env_pix_all = F.softplus(env_raw_params).detach().cpu().numpy()  # (N,P,3)
                env_imgs_k  = [_env_flat_to_img(env_pix_all[k], env_H, env_W)
                               for k in range(N_imgs)]
                env_avg_img = _env_flat_to_img(env_pix_all.mean(0), env_H, env_W)
                wandb_run.log({
                    "loss": float(loss), "loss_data": float(l_d),
                    "loss_sparse": float(l_s), "loss_white": float(l_w),
                    "elapsed_s": elapsed,
                    "pred_albedo":    wandb.Image(
                        torch.exp(log_albedo).clamp(0, 1).detach().cpu().numpy()),
                    "pred_metallic":  wandb.Image(met_map.squeeze(-1).cpu().numpy()),
                    "pred_roughness": wandb.Image(rou_map.squeeze(-1).cpu().numpy()),
                    "est_env_maps":   [wandb.Image(img) for img in env_imgs_k],
                    "env_map_avg":    wandb.Image(env_avg_img),
                    "metallic_mean":      met,
                    "roughness_mean":     rou,
                    "metallic_err_mean":  abs(met - gt_metallic),
                    "roughness_err_mean": abs(rou - gt_roughness),
                }, step=i)

    total_time = time.perf_counter() - t0

    import torch.nn.functional as F
    with torch.no_grad():
        albedo_out   = torch.exp(log_albedo).clamp(0, 1).cpu().numpy()
        env_maps_out = F.softplus(env_raw_params).cpu().numpy()
        met_out      = torch.sigmoid(metallic_raw).cpu().numpy()               # (H,W,1)
        rou_out      = (0.05 + 0.9 * torch.sigmoid(roughness_raw)).cpu().numpy()
        met_m        = torch.from_numpy(met_out).to(dev).reshape(-1, 1)[flat_mask]
        rou_m        = torch.from_numpy(rou_out).to(dev).reshape(-1, 1)[flat_mask]
        albedo_t2    = torch.exp(log_albedo)
        shadings = []
        for k in range(N_imgs):
            albedo_m  = albedo_t2.reshape(-1, 3)[flat_mask]
            env_pix_k = torch.from_numpy(env_maps_out[k]).to(dev)
            recon_m   = shade_ct_env(view_m, N_m, albedo_m,
                                     env_pix_k, env_dirs_t, env_dw_t,
                                     met_m, rou_m)
            s = albedo_t2.new_zeros(H, W, 3)
            s.reshape(-1, 3)[flat_mask] = recon_m
            s *= mask_t
            shadings.append(s.cpu().numpy())

    return albedo_out, env_maps_out, met_out, rou_out, shadings, history, total_time


# ─────────────────────────────────────── Phase 2: Phong SH optimizer ─────────

def _optimize_phong_sh(
    images:       list,
    normals_hw:   torch.Tensor,
    frag_pos_hw:  torch.Tensor,
    mask_hw:      torch.Tensor,
    cam_pos:      torch.Tensor,
    gt_shininess: float,
    gt_ks:        float,
    ka:           float,
    kd:           float,
    cfg:          dict,
    wandb_run=None,
    gt_sh_coeffs: Optional[list] = None,   # list of N (9,3) GT arrays
) -> tuple:
    """
    Phong+SH decomposition: recovers albedo, per-image SH, per-pixel shininess/ks.
    Returns (albedo [H,W,3], sh [N,9,3], shininess [H,W,1], ks [H,W,1], shadings, history, elapsed).
    """
    dev    = normals_hw.device
    H, W   = normals_hw.shape[:2]
    N_imgs = len(images)
    s_min  = cfg.get("shininess_min", DEFAULT_CFG["shininess_min"])
    s_max  = cfg.get("shininess_max", DEFAULT_CFG["shininess_max"])

    def _t(x):
        return torch.from_numpy(np.asarray(x, np.float32)).to(dev) \
            if not isinstance(x, torch.Tensor) else x.to(dev, torch.float32)

    imgs_t    = torch.stack([_t(img) for img in images])
    flat_mask = mask_hw.reshape(-1)
    N_m       = normals_hw.reshape(-1, 3)[flat_mask]
    fp_m      = frag_pos_hw.reshape(-1, 3)[flat_mask]
    view_m    = _norm(cam_pos.unsqueeze(0) - fp_m)
    mask_t    = mask_hw.unsqueeze(-1).to(dev)

    log_albedo    = torch.log(imgs_t.mean(0).clamp(0.05, 0.95)).requires_grad_(True)
    sh_init       = torch.zeros(N_imgs, 9, 3, device=dev)
    sh_init[:, 0, :] = 1.5
    sh_coeffs     = sh_init.clone().requires_grad_(True)
    shininess_raw = torch.zeros(H, W, 1, device=dev).requires_grad_(True)
    ks_raw        = torch.zeros(H, W, 1, device=dev).requires_grad_(True)

    opt = _make_optimizer([log_albedo, sh_coeffs, shininess_raw, ks_raw], cfg)

    def _forward():
        albedo      = torch.exp(log_albedo)
        albedo_m    = albedo.reshape(-1, 3)[flat_mask]
        shininess_m = (s_min + (s_max - s_min) * torch.sigmoid(shininess_raw)).reshape(-1, 1)[flat_mask]
        ks_m        = torch.sigmoid(ks_raw).reshape(-1, 1)[flat_mask]
        loss_data   = albedo.new_zeros(())
        for k in range(N_imgs):
            recon_m = shade_phong_sh(view_m, N_m, ka, kd, ks_m, shininess_m,
                                     albedo_m, sh_coeffs[k])
            recon   = albedo.new_zeros(H, W, 3)
            recon.reshape(-1, 3)[flat_mask] = recon_m
            loss_data = loss_data + _loss_fn(recon, imgs_t[k], mask_t, cfg["loss"])
        loss_sparse = cfg["lambda_sparse"] * _tv(log_albedo.permute(2, 0, 1))
        loss_white  = cfg["lambda_white"]  * (torch.exp(log_albedo).mean() - 0.5) ** 2
        return loss_data + loss_sparse + loss_white, loss_data, loss_sparse, loss_white

    history = []
    t0 = time.perf_counter()
    for i in range(cfg["n_iter"]):
        loss, l_d, l_s, l_w = _opt_step(opt, _forward, cfg)
        if i % cfg["log_every"] == 0:
            elapsed = time.perf_counter() - t0
            with torch.no_grad():
                shin_map = (s_min + (s_max - s_min) * torch.sigmoid(shininess_raw)).detach()
                ks_map   = torch.sigmoid(ks_raw).detach()
                shin_val = float(shin_map[mask_hw].mean())
                ks_val   = float(ks_map[mask_hw].mean())
            history.append(float(loss))
            print(f"  [{i:4d}] {elapsed:6.1f}s  loss={float(loss):.3e}  data={float(l_d):.3e}"
                  f"  shininess={shin_val:.1f}  ks={ks_val:.3f}")
            if wandb_run is not None:
                est_sh_np = sh_coeffs.detach().cpu().numpy()  # (N, 9, 3)
                wandb_run.log({
                    "loss": float(loss), "loss_data": float(l_d),
                    "loss_sparse": float(l_s), "loss_white": float(l_w),
                    "elapsed_s": elapsed,
                    "pred_albedo":     wandb.Image(
                        torch.exp(log_albedo).clamp(0, 1).detach().cpu().numpy()),
                    "pred_shininess":  wandb.Image(
                        (shin_map / s_max).squeeze(-1).cpu().numpy()),
                    "pred_ks":         wandb.Image(ks_map.squeeze(-1).cpu().numpy()),
                    "est_sh_env_maps": [wandb.Image(_sh_coeffs_to_env_img(est_sh_np[k]))
                                        for k in range(N_imgs)],
                    "shininess_mean":     shin_val,
                    "ks_mean":            ks_val,
                    "shininess_err_mean": abs(shin_val - gt_shininess),
                    "ks_err_mean":        abs(ks_val   - gt_ks),
                }, step=i)
    total_time = time.perf_counter() - t0

    with torch.no_grad():
        albedo_out = torch.exp(log_albedo).clamp(0, 1).cpu().numpy()
        sh_out     = sh_coeffs.cpu().numpy()
        shin_out   = (s_min + (s_max - s_min) * torch.sigmoid(shininess_raw)).cpu().numpy()
        ks_out     = torch.sigmoid(ks_raw).cpu().numpy()
        shin_m_t   = torch.from_numpy(shin_out).to(dev).reshape(-1, 1)[flat_mask]
        ks_m_t     = torch.from_numpy(ks_out).to(dev).reshape(-1, 1)[flat_mask]
        albedo_t2  = torch.exp(log_albedo)
        shadings   = []
        for k in range(N_imgs):
            albedo_m = albedo_t2.reshape(-1, 3)[flat_mask]
            recon_m  = shade_phong_sh(view_m, N_m, ka, kd, ks_m_t, shin_m_t,
                                      albedo_m, sh_coeffs[k])
            s = albedo_t2.new_zeros(H, W, 3)
            s.reshape(-1, 3)[flat_mask] = recon_m
            s *= mask_t
            shadings.append(s.cpu().numpy())

    return albedo_out, sh_out, shin_out, ks_out, shadings, history, total_time


# ─────────────────────────────────────── Phase 2: Phong env-map optimizer ────

def _optimize_phong_env(
    images:       list,
    normals_hw:   torch.Tensor,
    frag_pos_hw:  torch.Tensor,
    mask_hw:      torch.Tensor,
    cam_pos:      torch.Tensor,
    gt_shininess: float,
    gt_ks:        float,
    ka:           float,
    kd:           float,
    env_dirs:     np.ndarray,
    env_dw:       np.ndarray,
    cfg:          dict,
    wandb_run=None,
    env_H:        int = 64,
    env_W:        int = 128,
) -> tuple:
    """
    Phong+env-map decomposition: recovers albedo, per-image env-map, per-pixel shininess/ks.
    Returns (albedo [H,W,3], env_maps [N,P,3], shininess [H,W,1], ks [H,W,1], shadings, history, elapsed).
    """
    dev    = normals_hw.device
    H, W   = normals_hw.shape[:2]
    N_imgs = len(images)
    P      = env_dirs.shape[0]
    s_min  = cfg.get("shininess_min", DEFAULT_CFG["shininess_min"])
    s_max  = cfg.get("shininess_max", DEFAULT_CFG["shininess_max"])

    def _t(x):
        return torch.from_numpy(np.asarray(x, np.float32)).to(dev) \
            if not isinstance(x, torch.Tensor) else x.to(dev, torch.float32)

    imgs_t     = torch.stack([_t(img) for img in images])
    flat_mask  = mask_hw.reshape(-1)
    N_m        = normals_hw.reshape(-1, 3)[flat_mask]
    fp_m       = frag_pos_hw.reshape(-1, 3)[flat_mask]
    view_m     = _norm(cam_pos.unsqueeze(0) - fp_m)
    env_dirs_t = _t(env_dirs)
    env_dw_t   = _t(env_dw)
    mask_t     = mask_hw.unsqueeze(-1).to(dev)

    log_albedo     = torch.log(imgs_t.mean(0).clamp(0.05, 0.95)).requires_grad_(True)
    env_raw_params = torch.zeros(N_imgs, P, 3, device=dev).requires_grad_(True)
    shininess_raw  = torch.zeros(H, W, 1, device=dev).requires_grad_(True)
    ks_raw         = torch.zeros(H, W, 1, device=dev).requires_grad_(True)

    opt = _make_optimizer([log_albedo, env_raw_params, shininess_raw, ks_raw], cfg)

    def _forward():
        import torch.nn.functional as F
        albedo      = torch.exp(log_albedo)
        albedo_m    = albedo.reshape(-1, 3)[flat_mask]
        shininess_m = (s_min + (s_max - s_min) * torch.sigmoid(shininess_raw)).reshape(-1, 1)[flat_mask]
        ks_m        = torch.sigmoid(ks_raw).reshape(-1, 1)[flat_mask]
        loss_data   = albedo.new_zeros(())
        for k in range(N_imgs):
            env_pix_k = F.softplus(env_raw_params[k])
            recon_m   = shade_phong_env(view_m, N_m, albedo_m,
                                        env_pix_k, env_dirs_t, env_dw_t,
                                        ka, kd, ks_m, shininess_m)
            recon = albedo.new_zeros(H, W, 3)
            recon.reshape(-1, 3)[flat_mask] = recon_m
            loss_data = loss_data + _loss_fn(recon, imgs_t[k], mask_t, cfg["loss"])
        loss_sparse = cfg["lambda_sparse"] * _tv(log_albedo.permute(2, 0, 1))
        loss_white  = cfg["lambda_white"]  * (torch.exp(log_albedo).mean() - 0.5) ** 2
        return loss_data + loss_sparse + loss_white, loss_data, loss_sparse, loss_white

    history = []
    t0 = time.perf_counter()
    for i in range(cfg["n_iter"]):
        loss, l_d, l_s, l_w = _opt_step(opt, _forward, cfg)
        if i % cfg["log_every"] == 0:
            import torch.nn.functional as F
            elapsed = time.perf_counter() - t0
            with torch.no_grad():
                shin_map = (s_min + (s_max - s_min) * torch.sigmoid(shininess_raw)).detach()
                ks_map   = torch.sigmoid(ks_raw).detach()
                shin_val = float(shin_map[mask_hw].mean())
                ks_val   = float(ks_map[mask_hw].mean())
            history.append(float(loss))
            print(f"  [{i:4d}] {elapsed:6.1f}s  loss={float(loss):.3e}  data={float(l_d):.3e}"
                  f"  shininess={shin_val:.1f}  ks={ks_val:.3f}")
            if wandb_run is not None:
                env_pix_all = F.softplus(env_raw_params).detach().cpu().numpy()  # (N,P,3)
                env_imgs_k  = [_env_flat_to_img(env_pix_all[k], env_H, env_W)
                               for k in range(N_imgs)]
                env_avg_img = _env_flat_to_img(env_pix_all.mean(0), env_H, env_W)
                wandb_run.log({
                    "loss": float(loss), "loss_data": float(l_d),
                    "loss_sparse": float(l_s), "loss_white": float(l_w),
                    "elapsed_s": elapsed,
                    "pred_albedo":    wandb.Image(
                        torch.exp(log_albedo).clamp(0, 1).detach().cpu().numpy()),
                    "pred_shininess": wandb.Image(
                        (shin_map / s_max).squeeze(-1).cpu().numpy()),
                    "pred_ks":        wandb.Image(ks_map.squeeze(-1).cpu().numpy()),
                    "est_env_maps":   [wandb.Image(img) for img in env_imgs_k],
                    "env_map_avg":    wandb.Image(env_avg_img),
                    "shininess_mean":     shin_val,
                    "ks_mean":            ks_val,
                    "shininess_err_mean": abs(shin_val - gt_shininess),
                    "ks_err_mean":        abs(ks_val   - gt_ks),
                }, step=i)
    total_time = time.perf_counter() - t0

    import torch.nn.functional as F
    with torch.no_grad():
        albedo_out   = torch.exp(log_albedo).clamp(0, 1).cpu().numpy()
        env_maps_out = F.softplus(env_raw_params).cpu().numpy()
        shin_out     = (s_min + (s_max - s_min) * torch.sigmoid(shininess_raw)).cpu().numpy()
        ks_out       = torch.sigmoid(ks_raw).cpu().numpy()
        shin_m_t     = torch.from_numpy(shin_out).to(dev).reshape(-1, 1)[flat_mask]
        ks_m_t       = torch.from_numpy(ks_out).to(dev).reshape(-1, 1)[flat_mask]
        albedo_t2    = torch.exp(log_albedo)
        shadings     = []
        for k in range(N_imgs):
            albedo_m  = albedo_t2.reshape(-1, 3)[flat_mask]
            env_pix_k = torch.from_numpy(env_maps_out[k]).to(dev)
            recon_m   = shade_phong_env(view_m, N_m, albedo_m,
                                        env_pix_k, env_dirs_t, env_dw_t,
                                        ka, kd, ks_m_t, shin_m_t)
            s = albedo_t2.new_zeros(H, W, 3)
            s.reshape(-1, 3)[flat_mask] = recon_m
            s *= mask_t
            shadings.append(s.cpu().numpy())

    return albedo_out, env_maps_out, shin_out, ks_out, shadings, history, total_time


# ─────────────────────────────────────── Phase 2: main loop ──────────────────

def run_decomposition(
    mesh_name:     str  = "sphere",
    width:         int  = 128,
    height:        int  = 128,
    shader:        str  = "ct_sh",
    mat_filter:    Optional[str] = None,
    cfg_overrides: Optional[dict] = None,
    device:        str  = "cuda",
) -> None:
    """
    Run intrinsic decomposition for all scenes matching shader + mat_filter.

    shader : "ct_sh" | "ct_env" | "phong_sh" | "phong_env"
    """
    cfg = {**DEFAULT_CFG, **(cfg_overrides or {})}
    dev = device

    is_phong = shader.startswith("phong")
    mat_configs = PHONG_MATERIAL_CONFIGS if is_phong else MATERIAL_CONFIGS

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
        scene_prefix = f"{mesh_name}_phong_{mat_id}" if is_phong else f"{mesh_name}_{mat_id}"
        if mat_filter and scene_prefix != mat_filter:
            continue

        print(f"\n[Phase 2] {scene_prefix}  ({shader})")

        images, gt_sh_list = [], []
        for angle_deg in LIGHT_ANGLES_DEG:
            light_id = f"light_{int(angle_deg):02d}deg"
            img_path = DATASET_ROOT / scene_prefix / shader / light_id / "render.png"
            cfg_path = DATASET_ROOT / scene_prefix / shader / light_id / "config.json"
            if not img_path.exists():
                raise FileNotFoundError(f"{img_path} — run --phase 1 --shader {shader} first.")
            images.append(np.array(Image.open(img_path), dtype=np.float32) / 255.0)
            with open(cfg_path) as fh:
                gt_sh_list.append(np.array(
                    json.load(fh)["light"]["sh_coeffs"], dtype=np.float32))

        gt_albedo_img = np.tile(np.array(mat_cfg["albedo"], dtype=np.float32),
                                (height, width, 1))
        mask_np = mask_hw.cpu().numpy()
        gt_color = np.array(mat_cfg["albedo"], dtype=np.float32)

        run = wandb.init(
            entity  ="DLVC-intrinsics",
            project ="synthetic_ct_decomp",
            config  =dict(**cfg, mesh_name=mesh_name, mat_id=mat_id,
                          material=mat_cfg, shader=shader,
                          width=width, height=height, n_images=len(images)),
            name    =f"{scene_prefix}_{shader}",
            reinit  =True,
        )

        # GT SH env maps — shared across both CT and Phong SH/env shaders
        gt_sh_env_imgs = [_sh_coeffs_to_env_img(gt_sh) for gt_sh in gt_sh_list]

        if is_phong:
            gt_shin = mat_cfg["shininess"]
            gt_ks   = mat_cfg["ks"]
            gt_shin_img = np.full((height, width),
                                  gt_shin / SHININESS_RANGE[1], dtype=np.float32)
            gt_ks_img   = np.full((height, width), gt_ks, dtype=np.float32)
            run.log({"gt_images":       [wandb.Image(img) for img in images],
                     "gt_albedo":       wandb.Image(gt_albedo_img),
                     "gt_shininess":    wandb.Image(gt_shin_img),
                     "gt_ks":           wandb.Image(gt_ks_img),
                     "gt_sh_env_maps":  [wandb.Image(e) for e in gt_sh_env_imgs]}, step=0)
        else:
            gt_metallic  = mat_cfg["metallic"]
            gt_roughness = mat_cfg["roughness"]
            gt_metallic_img  = np.full((height, width), gt_metallic,  dtype=np.float32)
            gt_roughness_img = np.full((height, width), gt_roughness, dtype=np.float32)
            run.log({"gt_images":       [wandb.Image(img) for img in images],
                     "gt_albedo":       wandb.Image(gt_albedo_img),
                     "gt_metallic":     wandb.Image(gt_metallic_img),
                     "gt_roughness":    wandb.Image(gt_roughness_img),
                     "gt_sh_env_maps":  [wandb.Image(e) for e in gt_sh_env_imgs]}, step=0)

        # ── dispatch ──────────────────────────────────────────────────────────
        sh_out:   np.ndarray = np.empty(0)
        env_maps: np.ndarray = np.empty(0)

        if shader == "ct_sh":
            albedo, sh_out, mat_a, mat_b, shadings, history, elapsed = _optimize_ct_sh(
                images, normals_hw, frag_pos_hw, mask_hw, cam_pos,
                gt_metallic, gt_roughness, cfg,  # type: ignore[possibly-undefined]
                wandb_run=run, gt_sh_coeffs=gt_sh_list,
            )
        elif shader == "ct_env":
            albedo, env_maps, mat_a, mat_b, shadings, history, elapsed = _optimize_ct_env(
                images, normals_hw, frag_pos_hw, mask_hw, cam_pos,
                gt_metallic, gt_roughness, env_dirs, env_dw, cfg,  # type: ignore[possibly-undefined]
                wandb_run=run, env_H=env_H, env_W=env_W,
            )
        elif shader == "phong_sh":
            albedo, sh_out, mat_a, mat_b, shadings, history, elapsed = _optimize_phong_sh(
                images, normals_hw, frag_pos_hw, mask_hw, cam_pos,
                gt_shin, gt_ks,  # type: ignore[possibly-undefined]
                mat_cfg["ka"], mat_cfg["kd"], cfg,
                wandb_run=run, gt_sh_coeffs=gt_sh_list,
            )
        else:  # phong_env
            albedo, env_maps, mat_a, mat_b, shadings, history, elapsed = _optimize_phong_env(
                images, normals_hw, frag_pos_hw, mask_hw, cam_pos,
                gt_shin, gt_ks,  # type: ignore[possibly-undefined]
                mat_cfg["ka"], mat_cfg["kd"],
                env_dirs, env_dw, cfg, wandb_run=run, env_H=env_H, env_W=env_W,
            )

        # mat_a = metallic or shininess (H,W,1), mat_b = roughness or ks (H,W,1)
        gt_a = gt_shin   if is_phong else gt_metallic   # type: ignore[possibly-undefined]
        gt_b = gt_ks     if is_phong else gt_roughness  # type: ignore[possibly-undefined]
        a_label, b_label = ("shininess", "ks") if is_phong else ("metallic", "roughness")

        # ── albedo RMSE ───────────────────────────────────────────────────────
        est_px = torch.from_numpy(albedo[mask_np])
        gt_px  = torch.from_numpy(gt_color).unsqueeze(0).expand_as(est_px)
        rmse_t, scale_t = _albedo_rmse(est_px, gt_px)
        rmse   = float(rmse_t)
        scale  = scale_t.numpy()

        mat_a_err  = np.abs(mat_a - gt_a)
        mat_b_err  = np.abs(mat_b - gt_b)
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

        # Build final SH/env map images for wandb summary
        if shader in ("ct_sh", "phong_sh"):
            final_light_imgs = [wandb.Image(_sh_coeffs_to_env_img(sh_out[k]))
                                for k in range(len(images))]
            light_img_key = "est_sh_env_maps"
        else:
            final_light_imgs = [wandb.Image(_env_flat_to_img(env_maps[k], env_H, env_W))
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
        out_dir = RESULTS_ROOT / scene_prefix / shader
        out_dir.mkdir(parents=True, exist_ok=True)
        recon_dir = out_dir / "reconstructions"
        recon_dir.mkdir(exist_ok=True)

        Image.fromarray((albedo.clip(0, 1) * 255).astype(np.uint8)).save(
            out_dir / "albedo_est.png")
        # shininess is in [s_min, s_max] — normalize to [0, 1] for 8-bit images
        a_norm = SHININESS_RANGE[1] if (is_phong and a_label == "shininess") else 1.0
        _save_gray(mat_a / a_norm, out_dir / f"{a_label}_est.png")
        _save_gray(mat_b,          out_dir / f"{b_label}_est.png")
        _save_gray(mat_a_err / a_norm * mask_np[:, :, None], out_dir / f"{a_label}_err.png")
        _save_gray(mat_b_err * mask_np[:, :, None],          out_dir / f"{b_label}_err.png")

        for k, (s, e) in enumerate(zip(shadings, recon_err)):
            angle = LIGHT_ANGLES_DEG[k]
            Image.fromarray((s.clip(0, 1) * 255).astype(np.uint8)).save(
                recon_dir / f"recon_{angle:02d}deg.png")
            Image.fromarray((e.mean(-1) * 255).clip(0, 255).astype(np.uint8)).save(
                recon_dir / f"recon_err_{angle:02d}deg.png")

        if shader in ("ct_sh", "phong_sh"):
            np.save(out_dir / "sh_coeffs_est.npy", sh_out)
            for k, sh_k in enumerate(sh_out):
                img = (_sh_coeffs_to_env_img(sh_k) * 255).astype(np.uint8)
                Image.fromarray(img).save(
                    out_dir / f"sh_env_map_{LIGHT_ANGLES_DEG[k]:02d}deg.png")
        else:
            np.save(out_dir / "env_maps_est.npy", env_maps)
            for k, env_k in enumerate(env_maps):
                img = (_env_flat_to_img(env_k, env_H, env_W) * 255).astype(np.uint8)
                Image.fromarray(img).save(
                    out_dir / f"env_map_{LIGHT_ANGLES_DEG[k]:02d}deg.png")
            env_avg_img = _env_flat_to_img(env_maps.mean(0), env_H, env_W)
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


# ─────────────────────────────────────── CLI ─────────────────────────────────

_ALL_SHADERS = ["ct_sh", "ct_env", "phong_sh", "phong_env"]


def _build_parser():
    p = argparse.ArgumentParser(description="Synthetic CT + Phong dataset + decomposer")
    p.add_argument("--mesh",     default="sphere",
                   choices=["sphere", "suzanne", "bunny", "all"])
    p.add_argument("--width",    type=int, default=128)
    p.add_argument("--height",   type=int, default=128)
    p.add_argument("--phase",    type=int, default=1, choices=[1, 2])
    p.add_argument("--shader",   default="ct_sh",
                   choices=_ALL_SHADERS + ["all"],
                   help="Shader type (default: ct_sh)")
    p.add_argument("--optimizer", default=None, choices=["LBFGS", "Adam"])
    p.add_argument("--n-iter",   type=int,   default=None)
    p.add_argument("--lr",       type=float, default=None)
    p.add_argument("--mat",      default=None,
                   help="Single material config, e.g. sphere_default")
    p.add_argument("--device",   default=None,
                   help="torch device, e.g. cuda, cuda:1, cpu (default: cuda if available)")
    return p


def main():
    args = _build_parser().parse_args()
    overrides = {k: v for k, v in [
        ("optimizer", args.optimizer), ("n_iter", args.n_iter), ("lr", args.lr)
    ] if v is not None}

    device  = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    meshes  = ["sphere", "suzanne", "bunny"] if args.mesh == "all" else [args.mesh]
    shaders = _ALL_SHADERS if args.shader == "all" else [args.shader]

    if args.phase == 1:
        for mesh in meshes:
            generate_dataset(mesh_name=mesh, width=args.width, height=args.height,
                             shader=args.shader, device=device)
    else:
        for mesh in meshes:
            for sh in shaders:
                run_decomposition(
                    mesh_name     =mesh,
                    width         =args.width,
                    height        =args.height,
                    shader        =sh,
                    mat_filter    =args.mat,
                    cfg_overrides =overrides or None,
                    device        =device,
                )


if __name__ == "__main__":
    main()
