"""Phong + explicit environment-map lighting.

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
from idr.eval.metrics import _albedo_rmse
from idr.config import _PHONG_ENV_PARAMS, SHININESS_RANGE
from idr.render import shade_phong_env, EnvMap, SHLighting
from idr.render.ops import _norm


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
    gt_sh_coeffs: Optional[list] = None,
    gt_albedo:    Optional[np.ndarray] = None,
    opt_params:   Optional[frozenset] = None,
    transforms:   Optional[dict] = None,
    init_from_gt: bool = False,
    log_gradients: bool = False,
    grad_log_dir: Optional[Path] = None,
) -> tuple:
    dev    = normals_hw.device
    H, W   = normals_hw.shape[:2]
    N_imgs = len(images)
    _n_log = N_imgs if cfg.get("wandb_max_images") is None else min(N_imgs, int(cfg["wandb_max_images"]))
    P      = env_dirs.shape[0]
    s_min  = cfg.get("shininess_min", DEFAULT_CFG["shininess_min"])
    s_max  = cfg.get("shininess_max", DEFAULT_CFG["shininess_max"])
    op     = opt_params if opt_params is not None else _PHONG_ENV_PARAMS
    tr     = transforms if transforms is not None else NAMED_TRANSFORMS["none"]
    tr_ab, tr_shin, tr_ks, tr_env = tr["albedo"], tr["shininess"], tr["ks"], tr["env"]

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

    learnable    = []
    named_params = {}

    if "albedo" in op:
        if init_from_gt and gt_albedo is not None:
            base = torch.from_numpy(
                np.broadcast_to(np.asarray(gt_albedo, np.float32), (H, W, 3)).copy()
            ).to(dev)
        else:
            base = imgs_t.mean(0)
        albedo_param = _init_albedo(base, tr_ab).requires_grad_(True)
        learnable.append(albedo_param)
        named_params["albedo"] = albedo_param
    else:
        base = torch.from_numpy(
            np.broadcast_to(np.asarray(gt_albedo, np.float32), (H, W, 3)).copy()
        ).to(dev)
        albedo_param = _init_albedo(base, tr_ab)

    if "env" in op:
        if init_from_gt and gt_sh_coeffs is not None:
            gt_ef = np.stack([
                EnvMap.from_sh(SHLighting(gt_sh_coeffs[k]))._image_flat
                for k in range(N_imgs)
            ]).astype(np.float32)
            gt_ef_t = torch.from_numpy(gt_ef).to(dev)
            env_raw_params = (_softplus_inv(gt_ef_t) if tr_env == "softplus" else gt_ef_t.clone()).requires_grad_(True)
        else:
            env_raw_params = torch.zeros(N_imgs, P, 3, device=dev).requires_grad_(True)
        learnable.append(env_raw_params)
        named_params["env"] = env_raw_params
    else:
        gt_ef = np.stack([
            EnvMap.from_sh(SHLighting(gt_sh_coeffs[k]))._image_flat
            for k in range(N_imgs)
        ]).astype(np.float32)
        gt_ef_t = torch.from_numpy(gt_ef).to(dev)
        env_raw_params = _softplus_inv(gt_ef_t) if tr_env == "softplus" else gt_ef_t.clone()

    if "shininess" in op:
        if init_from_gt:
            _gt_shin = np.broadcast_to(np.asarray(gt_shininess, np.float32), (H, W, 1)).copy()
            if tr_shin == "sigmoid":
                sv = np.clip((_gt_shin - s_min) / (s_max - s_min), 1e-6, 1 - 1e-6)
                raw_arr = np.log(sv / (1 - sv))
            elif tr_shin == "log":
                raw_arr = np.log(_gt_shin.clip(1e-7))
            else:
                raw_arr = _gt_shin
            shininess_raw = torch.from_numpy(raw_arr.astype(np.float32)).to(dev).requires_grad_(True)
        else:
            if tr_shin == "sigmoid":
                s0 = 0.0
            elif tr_shin == "log":
                s0 = float(np.log(0.5 * (s_min + s_max)))
            else:
                s0 = 0.5 * (s_min + s_max)
            shininess_raw = torch.full((H, W, 1), s0, dtype=torch.float32, device=dev).requires_grad_(True)
        learnable.append(shininess_raw)
        named_params["shininess"] = shininess_raw
    else:
        _gt_shin_s = float(np.asarray(gt_shininess).mean())
        if tr_shin == "sigmoid":
            sv = float(np.clip((_gt_shin_s - s_min) / (s_max - s_min), 1e-6, 1 - 1e-6))
            shin_raw_val = float(np.log(sv / (1 - sv)))
        elif tr_shin == "log":
            shin_raw_val = float(np.log(_gt_shin_s))
        else:
            shin_raw_val = float(_gt_shin_s)
        shininess_raw = torch.full((H, W, 1), shin_raw_val, dtype=torch.float32, device=dev)

    if "ks" in op:
        if init_from_gt:
            _gt_ks = np.broadcast_to(np.asarray(gt_ks, np.float32), (H, W, 1)).copy()
            if tr_ks == "sigmoid":
                kv = np.clip(_gt_ks, 1e-6, 1 - 1e-6)
                raw_arr = np.log(kv / (1 - kv))
            else:
                raw_arr = _gt_ks
            ks_raw = torch.from_numpy(raw_arr.astype(np.float32)).to(dev).requires_grad_(True)
        else:
            ks0 = (-10.0 if tr_ks == "sigmoid" else 0.0) if cfg.get("init_spec_zero", False) \
                  else (0.0 if tr_ks == "sigmoid" else 0.5)
            ks_raw = torch.full((H, W, 1), ks0, dtype=torch.float32, device=dev).requires_grad_(True)
        learnable.append(ks_raw)
        named_params["ks"] = ks_raw
    else:
        _gt_ks_s = float(np.asarray(gt_ks).mean())
        if tr_ks == "sigmoid":
            kv = float(np.clip(_gt_ks_s, 1e-6, 1 - 1e-6))
            ks_raw_val = float(np.log(kv / (1 - kv)))
        else:
            ks_raw_val = float(_gt_ks_s)
        ks_raw = torch.full((H, W, 1), ks_raw_val, dtype=torch.float32, device=dev)

    opt   = steps._make_optimizer(learnable, cfg)
    sched = steps._make_scheduler(opt, cfg, cfg["n_iter"])
    _step = [0]

    def _forward(img_indices=None):
        if img_indices is None:
            img_indices = range(N_imgs)
        img_indices = list(img_indices)
        frac = len(img_indices) / N_imgs
        _spec_warmup = _step[0] < cfg.get("spec_warmup_steps", 0)
        albedo      = _fwd_albedo(albedo_param, tr_ab)
        albedo_m    = albedo.reshape(-1, 3)[flat_mask]
        shininess_m = _fwd_shininess(shininess_raw, tr_shin, s_min, s_max).reshape(-1, 1)[flat_mask]
        ks_m        = _fwd_ks(ks_raw, tr_ks).reshape(-1, 1)[flat_mask]
        if _spec_warmup:
            ks_m = albedo_m.new_zeros(albedo_m.shape[0], 1)
        loss_data   = albedo.new_zeros(())
        for k in img_indices:
            env_pix_k = _fwd_env(env_raw_params[k], tr_env)
            recon_m   = shade_phong_env(view_m, N_m, albedo_m,
                                        env_pix_k, env_dirs_t, env_dw_t,
                                        ka, kd, ks_m, shininess_m,
                                        sbatch=cfg.get("sbatch", 64))
            recon = albedo.new_zeros(H, W, 3)
            recon.reshape(-1, 3)[flat_mask] = recon_m
            loss_data = loss_data + _loss_fn(recon, imgs_t[k], mask_t, cfg["loss"])
        loss_sparse = frac * cfg["lambda_sparse"] * _tv(albedo_param.permute(2, 0, 1))
        loss_white  = frac * cfg["lambda_white"]  * (_fwd_albedo(albedo_param, tr_ab).mean() - 0.5) ** 2
        loss_tv     = frac * cfg["lambda_tv"] * (
            _tv(albedo_param.permute(2, 0, 1)) +
            _tv(shininess_raw.permute(2, 0, 1)) +
            _tv(ks_raw.permute(2, 0, 1))
        )
        return loss_data + loss_sparse + loss_white + loss_tv, loss_data, loss_sparse, loss_white, loss_tv

    def _forward_components():
        with torch.no_grad():
            albedo_m    = _fwd_albedo(albedo_param, tr_ab).reshape(-1, 3)[flat_mask]
            shininess_m = _fwd_shininess(shininess_raw, tr_shin, s_min, s_max).reshape(-1, 1)[flat_mask]
            ks_m        = _fwd_ks(ks_raw, tr_ks).reshape(-1, 1)[flat_mask]
            result = []
            for k in range(N_imgs):
                env_pix_k = _fwd_env(env_raw_params[k], tr_env)
                _, comps = shade_phong_env(view_m, N_m, albedo_m,
                                           env_pix_k, env_dirs_t, env_dw_t,
                                           ka, kd, ks_m, shininess_m,
                                           sbatch=cfg.get("sbatch", 64),
                                           return_components=True)
                result.append(comps)
        return result

    gt_ef_np = np.stack([
        EnvMap.from_sh(SHLighting(gt_sh_coeffs[k]))._image_flat
        for k in range(N_imgs)
    ]).astype(np.float32) if gt_sh_coeffs is not None else None
    gt_map_grad = {
        "albedo":    np.broadcast_to(np.asarray(gt_albedo, np.float32), (H, W, 3)).copy() if gt_albedo is not None else None,
        "env":       gt_ef_np,
        "shininess": np.broadcast_to(np.asarray(gt_shininess, np.float32), (H, W, 1)).copy(),
        "ks":        np.broadcast_to(np.asarray(gt_ks, np.float32), (H, W, 1)).copy(),
    }
    fwd_map_grad = {
        "albedo":    lambda p: _fwd_albedo(p, tr_ab),
        "env":       lambda p: _fwd_env(p, tr_env),
        "shininess": lambda p: _fwd_shininess(p, tr_shin, s_min, s_max),
        "ks":        lambda p: _fwd_ks(p, tr_ks),
    }

    if log_gradients and grad_log_dir is not None:
        grad_log_dir.mkdir(parents=True, exist_ok=True)

    _gt_shin_mean = float(np.asarray(gt_shininess).mean())
    _gt_ks_mean   = float(np.asarray(gt_ks).mean())
    history = []
    with torch.no_grad():
        _il, _ild, _ils, _ilw, _iltv = _forward()
        _is = float(_fwd_shininess(shininess_raw, tr_shin, s_min, s_max)[mask_hw].mean())
        _ik = float(_fwd_ks(ks_raw, tr_ks)[mask_hw].mean())
    print(f"  [init]          loss={float(_il):.3e}  data={float(_ild):.3e}"
          f"  shininess={_is:.1f}  ks={_ik:.3f}")
    if wandb_run is not None:
        with torch.no_grad():
            _env_pix_all = _fwd_env(env_raw_params, tr_env).detach()
            _env_imgs_k  = [_env_flat_to_img(_env_pix_all[k].cpu().numpy(), env_H, env_W) for k in range(_n_log)]
            _env_avg_img = _env_flat_to_img(_env_pix_all.mean(0).cpu().numpy(), env_H, env_W)
            _shin_map = _fwd_shininess(shininess_raw, tr_shin, s_min, s_max).detach()
            _ks_map   = _fwd_ks(ks_raw, tr_ks).detach()
            _ab_t   = _fwd_albedo(albedo_param, tr_ab).detach()
            _ab_m   = _ab_t.reshape(-1, 3)[flat_mask]
            _shin_m = _shin_map.reshape(-1, 1)[flat_mask]
            _ks_m   = _ks_map.reshape(-1, 1)[flat_mask]
            _recons, _errs = [], []
            for _k in range(_n_log):
                _r = _ab_t.new_zeros(H, W, 3)
                _r.reshape(-1, 3)[flat_mask] = shade_phong_env(
                    view_m, N_m, _ab_m, _env_pix_all[_k], env_dirs_t, env_dw_t,
                    ka, kd, _ks_m, _shin_m, sbatch=cfg.get("sbatch", 64))
                _r *= mask_t
                _recons.append(wandb.Image(_r.float().cpu().numpy()))
                _errs.append(wandb.Image((_r - imgs_t[_k]).abs().mul(mask_t).float().cpu().numpy()))
        wandb_run.log({
            "loss": float(_il), "loss_data": float(_ild), "data_rmse": float(_ild) ** 0.5,
            "loss_sparse": float(_ils), "loss_white": float(_ilw), "loss_tv": float(_iltv),
            "pred_albedo":    wandb.Image(_ab_t.float().cpu().numpy()),
            "pred_shininess": wandb.Image((_shin_map / s_max).squeeze(-1).cpu().numpy()),
            "pred_ks":        wandb.Image(_ks_map.squeeze(-1).cpu().numpy()),
            "est_env_maps":   [wandb.Image(img) for img in _env_imgs_k],
            "env_map_avg":    wandb.Image(_env_avg_img),
            "recons":         _recons,
            "recon_err_maps": _errs,
            "shininess_mean":     _is,
            "ks_mean":            _ik,
            "shininess_err_mean": abs(_is - _gt_shin_mean),
            "ks_err_mean":        abs(_ik - _gt_ks_mean),
            "lr": opt.param_groups[0]["lr"] if opt is not None else 0.0,
        }, step=-1)
    t0 = time.perf_counter()
    img_batch = cfg.get("img_batch", N_imgs) or N_imgs
    for i in range(cfg["n_iter"]):
        _step[0] = i
        if log_gradients:
            pre_raw = {n: p.data.clone() for n, p in named_params.items()}
        if img_batch >= N_imgs or log_gradients:
            loss, l_d, l_s, l_w, l_tv = steps._opt_step(opt, _forward, cfg)
        else:
            totals = [0.0] * 5

            def _accum():
                opt.zero_grad()
                totals[:] = [0.0] * 5
                for _b in range(0, N_imgs, img_batch):
                    _vals = _forward(range(_b, min(_b + img_batch, N_imgs)))
                    _vals[0].backward()
                    for _j, _v in enumerate(_vals):
                        totals[_j] += float(_v.detach())
                return torch.tensor(totals[0])

            if cfg.get("optimizer") == "LBFGS":
                opt.step(_accum)
            else:
                _accum()
                opt.step()
            loss, l_d, l_s, l_w, l_tv = totals
        if sched is not None:
            sched.step()
        if log_gradients and grad_log_dir is not None:
            _save_grad_step(i, named_params, pre_raw, gt_map_grad, fwd_map_grad,
                            _forward_components,
                            (float(loss), float(l_d), float(l_s), float(l_w), float(l_tv)),
                            grad_log_dir, flat_mask, H, W)
        if i % cfg["log_every"] == 0:
            elapsed = time.perf_counter() - t0
            with torch.no_grad():
                shin_map = _fwd_shininess(shininess_raw, tr_shin, s_min, s_max).detach()
                ks_map   = _fwd_ks(ks_raw, tr_ks).detach()
                shin_val = float(shin_map[mask_hw].mean())
                ks_val   = float(ks_map[mask_hw].mean())
            history.append(float(loss))
            print(f"  [{i:4d}] {elapsed:6.1f}s  loss={float(loss):.3e}  data={float(l_d):.3e}"
                  f"  shininess={shin_val:.1f}  ks={ks_val:.3f}")
            if wandb_run is not None:
                with torch.no_grad():
                    env_pix_all = _fwd_env(env_raw_params, tr_env).detach()
                    env_imgs_k  = [_env_flat_to_img(env_pix_all[k].cpu().numpy(), env_H, env_W)
                                   for k in range(_n_log)]
                    env_avg_img = _env_flat_to_img(env_pix_all.mean(0).cpu().numpy(), env_H, env_W)
                    _ab_t   = _fwd_albedo(albedo_param, tr_ab).detach()
                    _ab_m   = _ab_t.reshape(-1, 3)[flat_mask]
                    _shin_m = shin_map.reshape(-1, 1)[flat_mask]
                    _ks_m   = ks_map.reshape(-1, 1)[flat_mask]
                    _recons, _errs = [], []
                    for _k in range(_n_log):
                        _r = _ab_t.new_zeros(H, W, 3)
                        _r.reshape(-1, 3)[flat_mask] = shade_phong_env(
                            view_m, N_m, _ab_m, env_pix_all[_k], env_dirs_t, env_dw_t,
                            ka, kd, _ks_m, _shin_m, sbatch=cfg.get("sbatch", 64))
                        _r *= mask_t
                        _recons.append(wandb.Image(_r.float().cpu().numpy()))
                        _errs.append(wandb.Image((_r - imgs_t[_k]).abs().mul(mask_t).float().cpu().numpy()))
                wandb_run.log({
                    "loss": float(loss), "loss_data": float(l_d), "data_rmse": float(l_d) ** 0.5,
                    "loss_sparse": float(l_s), "loss_white": float(l_w), "loss_tv": float(l_tv),
                    "elapsed_s": elapsed,
                    "pred_albedo":    wandb.Image(
                        _fwd_albedo(albedo_param, tr_ab).detach().float().cpu().numpy()),
                    "pred_shininess": wandb.Image(
                        (shin_map / s_max).squeeze(-1).cpu().numpy()),
                    "pred_ks":        wandb.Image(ks_map.squeeze(-1).cpu().numpy()),
                    "est_env_maps":   [wandb.Image(img) for img in env_imgs_k],
                    "env_map_avg":    wandb.Image(env_avg_img),
                    "recons":         _recons,
                    "recon_err_maps": _errs,
                    "shininess_mean":     shin_val,
                    "ks_mean":            ks_val,
                    "shininess_err_mean": abs(shin_val - _gt_shin_mean),
                    "ks_err_mean":        abs(ks_val   - _gt_ks_mean),
                    "lr": opt.param_groups[0]["lr"] if opt is not None else 0.0,
                }, step=i)
    total_time = time.perf_counter() - t0

    with torch.no_grad():
        albedo_out   = _fwd_albedo(albedo_param, tr_ab).cpu().numpy()
        env_maps_out = _fwd_env(env_raw_params, tr_env).cpu().numpy()
        shin_out     = _fwd_shininess(shininess_raw, tr_shin, s_min, s_max).cpu().numpy()
        ks_out       = _fwd_ks(ks_raw, tr_ks).cpu().numpy()
        shin_m_t     = torch.from_numpy(shin_out).to(dev).reshape(-1, 1)[flat_mask]
        ks_m_t       = torch.from_numpy(ks_out).to(dev).reshape(-1, 1)[flat_mask]
        albedo_t2    = _fwd_albedo(albedo_param, tr_ab)
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
