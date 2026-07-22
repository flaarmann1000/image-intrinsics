"""Cook-Torrance + explicit environment-map lighting.

Moved here whole in Stage 4; Stage 5 reduces it to a model adapter over the
shared driver.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
import wandb

from idr.config import DEFAULT_CFG, NAMED_TRANSFORMS, LIGHT_COLOR, LIGHT_INTENSITY
from idr.optim.transforms import (
    _softplus_inv, _parse_transforms, _fwd_albedo, _fwd_metallic, _fwd_roughness,
    _fwd_shininess, _fwd_ks, _fwd_env, _init_albedo, _init_scalar, _init_map, _init_env,
)
from idr.optim.losses import _tv, _loss_fn, _sqrt_res, _tv_residuals
# Imported as a MODULE, not by name: profile_decomposition.py swaps
# idr.optim.steps._opt_step / ._make_optimizer at runtime to time each phase. A
# `from ... import _opt_step` would bind the original at import time and silently
# defeat that patch (it reported 0 ms/forward until this was changed).
import idr.optim.steps as steps
from idr.eval.metrics import _albedo_lighting_scale, _rescale_albedo_lighting
from idr.track.wandb_log import (
    _sh_coeffs_to_env_img, _env_flat_to_img, _save_grad_step, _structured_scalar_log,
)
from idr.data.synthetic_scene import _scatter, _scatter_np
from raw_optimizer.helper import _albedo_rmse
from idr.config import _CT_ENV_PARAMS
from idr.render import shade_ct_env, EnvMap, SHLighting, build_sh_basis
from idr.render.ops import _norm
from idr.render.brdf import _get_ggx_sh_lut


def _optimize_ct_env(
    images:       list,
    normals_hw:   torch.Tensor,
    frag_pos_hw:  torch.Tensor,
    mask_hw:      torch.Tensor,
    cam_pos:      torch.Tensor,
    gt_metallic:  Union[float, np.ndarray],
    gt_roughness: Union[float, np.ndarray],
    env_dirs:     np.ndarray,
    env_dw:       np.ndarray,
    cfg:          dict,
    wandb_run=None,
    env_H:        int = 64,
    env_W:        int = 128,
    gt_sh_coeffs: Optional[list] = None,
    gt_albedo:    Optional[np.ndarray] = None,
    opt_params:   Optional[frozenset] = None,
    transforms:   Optional[dict] = None,
    init_from_gt: bool = False,
    log_gradients: bool = False,
    grad_log_dir: Optional[Path] = None,
    val_images:   Optional[list] = None,
    val_sh_coeffs: Optional[list] = None,
) -> tuple:
    dev    = normals_hw.device
    ftype  = normals_hw.dtype
    H, W   = normals_hw.shape[:2]
    N_imgs = len(images)
    P      = env_dirs.shape[0]
    _n_log = N_imgs if cfg.get("wandb_max_images") is None else min(N_imgs, int(cfg["wandb_max_images"]))
    op = opt_params if opt_params is not None else _CT_ENV_PARAMS
    _diffuse_fresnel = bool(cfg.get("diffuse_fresnel", True))
    if transforms is not None:
        tr_ab, tr_met, tr_rou, tr_env = transforms["albedo"], transforms["metallic"], transforms["roughness"], transforms["env"]
    else:
        tr_ab  = cfg.get("tr_albedo",   "none")
        tr_met = cfg.get("tr_metallic",  "none")
        tr_rou = cfg.get("tr_roughness", "none")
        tr_env = cfg.get("tr_env",       "none")

    def _t(x):
        return torch.from_numpy(np.asarray(x, np.float32)).to(dev, ftype) \
            if not isinstance(x, torch.Tensor) else x.to(dev, ftype)

    imgs_t     = torch.stack([_t(img) for img in images])
    flat_mask  = mask_hw.reshape(-1)
    N_m        = normals_hw.reshape(-1, 3)[flat_mask]
    fp_m       = frag_pos_hw.reshape(-1, 3)[flat_mask]
    view_m     = _norm(cam_pos.unsqueeze(0) - fp_m)
    env_dirs_t = _t(env_dirs)
    env_dw_t   = _t(env_dw)
    mask_t     = mask_hw.unsqueeze(-1).to(dev)

    learnable    = []
    named_params = {}

    if "albedo" in op:
        if init_from_gt and gt_albedo is not None:
            base = torch.from_numpy(
                np.broadcast_to(np.asarray(gt_albedo, np.float32), (H, W, 3)).copy()
            ).to(dev, ftype)
        else:
            base = imgs_t.mean(0)
        albedo_param = _init_albedo(base, tr_ab).requires_grad_(True)
        learnable.append(albedo_param)
        named_params["albedo"] = albedo_param
    else:
        base = torch.from_numpy(
            np.broadcast_to(np.asarray(gt_albedo, np.float32), (H, W, 3)).copy()
        ).to(dev, ftype)
        albedo_param = _init_albedo(base, tr_ab)

    if "env" in op:
        if init_from_gt and gt_sh_coeffs is not None:
            gt_ef = np.stack([
                EnvMap.from_sh(SHLighting(gt_sh_coeffs[k]))._image_flat
                for k in range(N_imgs)
            ]).astype(np.float32)
            gt_ef_t = torch.from_numpy(gt_ef).to(dev, ftype)
            env_raw_params = (_softplus_inv(gt_ef_t) if tr_env == "softplus" else gt_ef_t.clone()).requires_grad_(True)
        else:
            env_raw_params = torch.zeros(N_imgs, P, 3, device=dev, dtype=ftype).requires_grad_(True)
        learnable.append(env_raw_params)
        named_params["env"] = env_raw_params
    else:
        gt_ef = np.stack([
            EnvMap.from_sh(SHLighting(gt_sh_coeffs[k]))._image_flat
            for k in range(N_imgs)
        ]).astype(np.float32)
        gt_ef_t = torch.from_numpy(gt_ef).to(dev, ftype)
        env_raw_params = _softplus_inv(gt_ef_t) if tr_env == "softplus" else gt_ef_t.clone()

    _gt_met_np = np.asarray(gt_metallic, np.float32)
    _gt_rou_np = np.asarray(gt_roughness, np.float32)
    _flat_mask_s = flat_mask.cpu().numpy()
    _gt_met_scalar = float(_gt_met_np.reshape(-1, 1)[_flat_mask_s].mean() if _gt_met_np.ndim > 0
                           else float(_gt_met_np))
    _gt_rou_scalar = float(_gt_rou_np.reshape(-1, 1)[_flat_mask_s].mean() if _gt_rou_np.ndim > 0
                           else float(_gt_rou_np))

    if "metallic" in op:
        if init_from_gt:
            metallic_raw = _init_map(_gt_met_np.reshape(H, W, 1), tr_met, dev).to(ftype).requires_grad_(True)
        else:
            _mv = 0.0 if cfg.get("init_spec_zero", False) else 0.5
            metallic_raw = _init_scalar(_mv, H, W, tr_met, dev=dev).to(ftype).requires_grad_(True)
        learnable.append(metallic_raw)
        named_params["metallic"] = metallic_raw
    else:
        if _gt_met_np.ndim > 0:
            metallic_raw = _init_map(_gt_met_np.reshape(H, W, 1), tr_met, dev).to(ftype)
        else:
            metallic_raw = _init_scalar(_gt_met_scalar, H, W, tr_met, dev=dev).to(ftype)

    if "roughness" in op:
        if init_from_gt:
            roughness_raw = _init_map(_gt_rou_np.reshape(H, W, 1), tr_rou, dev).to(ftype).requires_grad_(True)
        else:
            _rv = 1.0 if cfg.get("init_spec_zero", False) else (0.1 if cfg.get("init_roughness_zero", False) else 0.5)
            roughness_raw = _init_scalar(_rv, H, W, tr_rou, dev=dev).to(ftype).requires_grad_(True)
        learnable.append(roughness_raw)
        named_params["roughness"] = roughness_raw
    else:
        if _gt_rou_np.ndim > 0:
            roughness_raw = _init_map(_gt_rou_np.reshape(H, W, 1), tr_rou, dev).to(ftype)
        else:
            roughness_raw = _init_scalar(_gt_rou_scalar, H, W, tr_rou, dev=dev).to(ftype)

    def _get_met(): return _fwd_metallic(metallic_raw, tr_met)
    def _get_rou(): return _fwd_roughness(roughness_raw, tr_rou)

    opt   = steps._make_optimizer(learnable, cfg) if learnable else None
    sched = steps._make_scheduler(opt, cfg, cfg["n_iter"]) if opt is not None else None
    _step = [0]
    _loss_ml = [torch.zeros((), device=dev, dtype=ftype)]
    if steps._optimizer_name(cfg) == "LM":
        # Each env map is 2048x3 free params PER IMAGE, so P explodes (>600k at 100
        # images) and J^T J is not formable. Use LBFGS/Adam for ct_env, or ct_sh.
        raise NotImplementedError(
            "optimizer='LM' is only supported for shader='ct_sh' (ct_env's per-image "
            "env-map parameters make the P x P normal equations intractable)")

    _loss_mb = [torch.zeros((), device=dev, dtype=ftype)]
    _recon_rmse = [0.0]           # per-image recon RMSE/MAE, stashed by _forward
    _recon_mae  = [0.0]

    def _forward(img_indices=None):
        if img_indices is None:
            img_indices = range(N_imgs)
        img_indices = list(img_indices)
        frac = len(img_indices) / N_imgs
        _spec_warmup = _step[0] < cfg.get("spec_warmup_steps", 0)
        albedo      = _fwd_albedo(albedo_param, tr_ab)
        albedo_m    = albedo.reshape(-1, 3)[flat_mask]
        metallic    = _fwd_metallic(metallic_raw,  tr_met)
        roughness   = _fwd_roughness(roughness_raw, tr_rou)
        metallic_m  = metallic.reshape(-1, 1)[flat_mask]
        roughness_m = roughness.reshape(-1, 1)[flat_mask]
        if _spec_warmup:
            metallic_m  = albedo_m.new_zeros(albedo_m.shape[0], 1)
            roughness_m = albedo_m.new_ones(albedo_m.shape[0], 1)
        if _step[0] < cfg.get("min_metallic_steps", 0):
            metallic_m = metallic_m.clamp(min=0.1)
        loss_data = albedo.new_zeros(())
        _rr_sum = 0.0
        _ra_sum = 0.0
        for k in img_indices:
            env_pix_k = _fwd_env(env_raw_params[k], tr_env)
            recon_m   = shade_ct_env(view_m, N_m, albedo_m,
                                     env_pix_k, env_dirs_t, env_dw_t,
                                     metallic_m, roughness_m,
                                     sbatch=cfg.get("sbatch", 64),
                                     spec_importance=cfg.get("spec_importance", False),
                                     spec_samples=cfg.get("spec_samples", 64), diffuse_fresnel=_diffuse_fresnel)
            recon = albedo.new_zeros(H, W, 3)
            recon.reshape(-1, 3)[flat_mask] = recon_m
            loss_data = loss_data + _loss_fn(recon, imgs_t[k], mask_t, cfg["loss"],
                                             cfg.get("huber_delta", 0.05))
            with torch.no_grad():
                _rd = (recon_m - imgs_t[k].reshape(-1, 3)[flat_mask]).detach()
                _rr_sum += float(_rd.pow(2).mean().sqrt())
                _ra_sum += float(_rd.abs().mean())
        # mean over images (chunks sum to the full-batch value): loss_data is
        # comparable across dataset sizes. Regularizer lambdas stay absolute.
        loss_data = loss_data / N_imgs
        _n_sel = max(len(img_indices), 1)
        _recon_rmse[0] = _rr_sum / _n_sel
        _recon_mae[0]  = _ra_sum / _n_sel
        loss_sparse = frac * cfg["lambda_sparse"] * _tv(albedo_param.permute(2, 0, 1))
        loss_white  = frac * cfg["lambda_white"]  * (_fwd_albedo(albedo_param, tr_ab).mean() - 0.5) ** 2
        loss_tv     = frac * cfg["lambda_tv"] * (
            _tv(albedo_param.permute(2, 0, 1)) +
            _tv(metallic_raw.permute(2, 0, 1)) +
            _tv(roughness_raw.permute(2, 0, 1))
        )
        met_m = metallic.reshape(-1, 1)[flat_mask]
        loss_metallic_l1       = frac * cfg.get("lambda_metallic_l1",       0.0) * met_m.abs().mean()
        loss_metallic_binarize = frac * cfg.get("lambda_metallic_binarize",  0.0) * (met_m * (1.0 - met_m)).mean()
        _loss_ml[0] = loss_metallic_l1.detach()
        _loss_mb[0] = loss_metallic_binarize.detach()
        return loss_data + loss_sparse + loss_white + loss_tv + loss_metallic_l1 + loss_metallic_binarize, loss_data, loss_sparse, loss_white, loss_tv

    def _forward_components():
        with torch.no_grad():
            albedo_m    = _fwd_albedo(albedo_param, tr_ab).reshape(-1, 3)[flat_mask]
            metallic_m  = _fwd_metallic(metallic_raw, tr_met).reshape(-1, 1)[flat_mask]
            roughness_m = _fwd_roughness(roughness_raw, tr_rou).reshape(-1, 1)[flat_mask]
            result = []
            for k in range(N_imgs):
                env_pix_k = _fwd_env(env_raw_params[k], tr_env)
                _, comps = shade_ct_env(view_m, N_m, albedo_m,
                                        env_pix_k, env_dirs_t, env_dw_t,
                                        metallic_m, roughness_m,
                                        sbatch=cfg.get("sbatch", 64),
                                     spec_importance=cfg.get("spec_importance", False),
                                     spec_samples=cfg.get("spec_samples", 64), return_components=True)
                result.append(comps)
        return result

    gt_ef_np = np.stack([
        EnvMap.from_sh(SHLighting(gt_sh_coeffs[k]))._image_flat
        for k in range(N_imgs)
    ]).astype(np.float32) if gt_sh_coeffs is not None else None
    gt_map_grad = {
        "albedo":    np.broadcast_to(np.asarray(gt_albedo, np.float32), (H, W, 3)).copy() if gt_albedo is not None else None,
        "env":       gt_ef_np,
        "metallic":  np.broadcast_to(np.asarray(gt_metallic,  np.float32), (H, W, 1)).copy(),
        "roughness": np.broadcast_to(np.asarray(gt_roughness, np.float32), (H, W, 1)).copy(),
    }
    fwd_map_grad = {
        "albedo":    lambda p: _fwd_albedo(p, tr_ab),
        "env":       lambda p: _fwd_env(p, tr_env),
        "metallic":  lambda p: _fwd_metallic(p, tr_met),
        "roughness": lambda p: _fwd_roughness(p, tr_rou),
    }

    if log_gradients and grad_log_dir is not None:
        grad_log_dir.mkdir(parents=True, exist_ok=True)

    # ── GT RMSE helpers (mirrors _optimize_ct_sh) ─────────────────────────────
    _flat_mask_np = flat_mask.cpu().numpy()
    _gt_ab_m = (torch.from_numpy(
                    np.asarray(gt_albedo, np.float32).reshape(-1, 3)[_flat_mask_np]
                ).to(dev, ftype) if gt_albedo is not None else None)
    _gt_met_arr = np.asarray(gt_metallic, np.float32)
    _gt_met_m = torch.from_numpy(
        _gt_met_arr.reshape(-1, 1)[_flat_mask_np] if _gt_met_arr.ndim > 0
        else np.full((int(flat_mask.sum()), 1), float(_gt_met_arr), np.float32)
    ).to(dev, ftype)
    _gt_rou_arr = np.asarray(gt_roughness, np.float32)
    _gt_rou_m = torch.from_numpy(
        _gt_rou_arr.reshape(-1, 1)[_flat_mask_np] if _gt_rou_arr.ndim > 0
        else np.full((int(flat_mask.sum()), 1), float(_gt_rou_arr), np.float32)
    ).to(dev, ftype)

    def _gt_rmse_metrics(ab_m, met_m, rou_m):
        out = {}
        if _gt_ab_m is not None:
            scale = (_gt_ab_m * ab_m).sum(0) / (ab_m * ab_m).sum(0).clamp(1e-8)
            out["albedo_rmse"] = float((ab_m * scale - _gt_ab_m).pow(2).mean().sqrt())
            out["albedo_mae"]  = float((ab_m * scale - _gt_ab_m).abs().mean())
        out["metallic_rmse"]  = float((met_m - _gt_met_m).pow(2).mean().sqrt())
        out["roughness_rmse"] = float((rou_m - _gt_rou_m).pow(2).mean().sqrt())
        out["metallic_mae"]   = float((met_m - _gt_met_m).abs().mean())
        out["roughness_mae"]  = float((rou_m - _gt_rou_m).abs().mean())
        return out

    # ── held-out relighting metric ────────────────────────────────────────────
    # GT lighting expressed on the optimizer's env grid (rectified SH radiance,
    # the representation this shader lights with).
    _val_imgs_m = _val_env = None
    if val_images and val_sh_coeffs is not None:
        _val_imgs_m = torch.stack([_t(v) for v in val_images]) \
            .reshape(len(val_images), -1, 3)[:, flat_mask, :]              # (V, M, 3)
        _val_env = torch.stack([
            torch.from_numpy(np.maximum(
                build_sh_basis(env_dirs) @ np.asarray(s, np.float32), 0.0)
            ).to(dev, ftype)
            for s in val_sh_coeffs])                                       # (V, P, 3)

    def _relight_metrics(ab_m, met_m, rou_m):
        """Render the held-out images with CURRENT intrinsics + their GT
        lighting; report the per-image error vs the observed val images."""
        if _val_imgs_m is None:
            return {}
        rs, ms = [], []
        with torch.no_grad():
            for k in range(_val_imgs_m.shape[0]):
                recon = shade_ct_env(view_m, N_m, ab_m, _val_env[k],
                                     env_dirs_t, env_dw_t, met_m, rou_m,
                                     sbatch=cfg.get("sbatch", 64),
                                     spec_importance=cfg.get("spec_importance", False),
                                     spec_samples=cfg.get("spec_samples", 64), diffuse_fresnel=_diffuse_fresnel)
                d = recon - _val_imgs_m[k]
                rs.append(float(d.pow(2).mean().sqrt()))
                ms.append(float(d.abs().mean()))
        return {"relight_rmse": float(np.mean(rs)), "relight_mae": float(np.mean(ms))}

    history = []
    with torch.no_grad():
        _il, _ild, _ils, _ilw, _iltv = _forward()
        _im = float(_get_met()[mask_hw].mean())
        _ir = float(_get_rou()[mask_hw].mean())
    print(f"  [init]          loss={float(_il):.3e}  data={float(_ild):.3e}"
          f"  metallic={_im:.3f}  roughness={_ir:.3f}")
    if wandb_run is not None:
        with torch.no_grad():
            _ab_t  = _fwd_albedo(albedo_param, tr_ab).detach()
            _ab_m  = _ab_t.reshape(-1, 3)[flat_mask]
            _met_m = _get_met().detach().reshape(-1, 1)[flat_mask]
            _rou_m = _get_rou().detach().reshape(-1, 1)[flat_mask]
            _init_scale = (_albedo_lighting_scale(albedo_param, tr_ab, flat_mask, _gt_ab_m)
                           if _gt_ab_m is not None else None)
        wandb_run.log(_structured_scalar_log(
            loss=_il, l_d=_ild, l_s=_ils, l_w=_ilw, l_tv=_iltv,
            loss_ml=_loss_ml[0], loss_mb=_loss_mb[0], scale3=_init_scale,
            gt_metrics=_gt_rmse_metrics(_ab_m, _met_m, _rou_m),
            relight=_relight_metrics(_ab_m, _met_m, _rou_m),
            recon_rmse=_recon_rmse[0], recon_mae=_recon_mae[0],
            lr=opt.param_groups[0]["lr"] if opt is not None else 0.0,
        ), step=-1)
    _rescale_every = cfg.get("rescale_every", 0)
    _img_batch = int(cfg.get("img_batch", 0) or 0)
    _use_img_batch = 0 < _img_batch < N_imgs
    if _use_img_batch and log_gradients:
        print("  [img_batch] disabled: incompatible with log_gradients")
        _use_img_batch = False
    t0 = time.perf_counter()
    for i in range(cfg["n_iter"]):
        _step[0] = i
        if opt is not None:
            if log_gradients:
                pre_raw = {n: p.data.clone() for n, p in named_params.items()}
            if _use_img_batch:
                loss, l_d, l_s, l_w, l_tv = steps._opt_step_img_batched(
                    opt, _forward, cfg, N_imgs, _img_batch)
            else:
                loss, l_d, l_s, l_w, l_tv = steps._opt_step(opt, _forward, cfg)
            if sched is not None:
                sched.step()
            if (_rescale_every > 0 and (i + 1) % _rescale_every == 0
                    and "albedo" in op and "env" in op and _gt_ab_m is not None):
                with torch.no_grad():
                    _rescale_albedo_lighting(
                        albedo_param, [env_raw_params], tr_ab, flat_mask, _gt_ab_m)
            if log_gradients and grad_log_dir is not None:
                _save_grad_step(i, named_params, pre_raw, gt_map_grad, fwd_map_grad,
                                _forward_components,
                                (float(loss), float(l_d), float(l_s), float(l_w), float(l_tv)),
                                grad_log_dir, flat_mask, H, W)
        else:
            with torch.no_grad():
                loss, l_d, l_s, l_w, l_tv = _forward()
        if i % cfg["log_every"] == 0:
            elapsed = time.perf_counter() - t0
            with torch.no_grad():
                met_map = _get_met().detach()
                rou_map = _get_rou().detach()
                met = float(met_map[mask_hw].mean())
                rou = float(rou_map[mask_hw].mean())
            history.append(float(loss))
            print(f"  [{i:4d}] {elapsed:6.1f}s  loss={float(loss):.3e}  data={float(l_d):.3e}"
                  f"  metallic={met:.3f}  roughness={rou:.3f}")
            if wandb_run is not None:
                with torch.no_grad():
                    _ab_t  = _fwd_albedo(albedo_param, tr_ab).detach()
                    _ab_m  = _ab_t.reshape(-1, 3)[flat_mask]
                    _met_m = met_map.reshape(-1, 1)[flat_mask]
                    _rou_m = rou_map.reshape(-1, 1)[flat_mask]
                    _step_scale = (_albedo_lighting_scale(albedo_param, tr_ab, flat_mask, _gt_ab_m)
                                   if _gt_ab_m is not None else None)
                _payload = _structured_scalar_log(
                    loss=loss, l_d=l_d, l_s=l_s, l_w=l_w, l_tv=l_tv,
                    loss_ml=_loss_ml[0], loss_mb=_loss_mb[0], scale3=_step_scale,
                    gt_metrics=_gt_rmse_metrics(_ab_m, _met_m, _rou_m),
                    relight=_relight_metrics(_ab_m, _met_m, _rou_m),
                    recon_rmse=_recon_rmse[0], recon_mae=_recon_mae[0],
                    lr=opt.param_groups[0]["lr"] if opt is not None else 0.0)
                _payload["elapsed_s"] = elapsed
                wandb_run.log(_payload, step=i)

    total_time = time.perf_counter() - t0

    with torch.no_grad():
        albedo_out   = _fwd_albedo(albedo_param, tr_ab).cpu().numpy()
        env_maps_out = _fwd_env(env_raw_params, tr_env).cpu().numpy()
        met_out      = _fwd_metallic(metallic_raw, tr_met).cpu().numpy()
        rou_out      = _fwd_roughness(roughness_raw, tr_rou).cpu().numpy()
        met_m        = torch.from_numpy(met_out).to(dev).reshape(-1, 1)[flat_mask]
        rou_m        = torch.from_numpy(rou_out).to(dev).reshape(-1, 1)[flat_mask]
        albedo_t2    = _fwd_albedo(albedo_param, tr_ab)
        shadings = []
        for k in range(N_imgs):
            albedo_m  = albedo_t2.reshape(-1, 3)[flat_mask]
            env_pix_k = torch.from_numpy(env_maps_out[k]).to(dev)
            recon_m   = shade_ct_env(view_m, N_m, albedo_m,
                                     env_pix_k, env_dirs_t, env_dw_t,
                                     met_m, rou_m,
                                     sbatch=cfg.get("sbatch", 64),
                                     spec_importance=cfg.get("spec_importance", False),
                                     spec_samples=cfg.get("spec_samples", 64),
                                     diffuse_fresnel=_diffuse_fresnel)
            s = albedo_t2.new_zeros(H, W, 3)
            s.reshape(-1, 3)[flat_mask] = recon_m
            s *= mask_t
            shadings.append(s.cpu().numpy())

    return albedo_out, env_maps_out, met_out, rou_out, shadings, history, total_time
