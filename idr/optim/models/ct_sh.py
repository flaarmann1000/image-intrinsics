"""Cook-Torrance + spherical-harmonics lighting.

The main model. Also the only one with a Levenberg-Marquardt path.

Moved here whole in Stage 4; Stage 5 reduces it to a model adapter over the
shared driver.
"""
from __future__ import annotations

import math
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
from idr.optim.lm.problem import build_lm_solver
from idr.optim.varpro.problem import build_varpro_solver
from idr.config import _CT_SH_PARAMS
from idr.render import shade_ct_sh
from idr.render.brdf import _get_ggx_sh_lut, _lut_lookup, ggx_sh_bands
from idr.render.sh import _sh_basis, _sh_irradiance
from idr.render.ops import _norm


def _optimize_ct_sh(
    images:       list,
    normals_hw:   torch.Tensor,
    frag_pos_hw:  torch.Tensor,
    mask_hw:      torch.Tensor,
    cam_pos:      torch.Tensor,
    gt_metallic:  Union[float, np.ndarray],
    gt_roughness: Union[float, np.ndarray],
    cfg:          dict,
    wandb_run=None,
    gt_sh_coeffs: Optional[list] = None,
    gt_albedo:    Optional[np.ndarray] = None,
    opt_params:   Optional[frozenset] = None,
    transforms:   Optional[dict] = None,
    init_from_gt: bool = False,
    log_gradients: bool = False,
    grad_log_dir: Optional[Path] = None,
    val_images:   Optional[list] = None,
    val_sh_coeffs: Optional[list] = None,
    init_maps:    Optional[dict] = None,
    light_prior:  Optional[np.ndarray] = None,
    wandb_step_offset: int = 0,
    wandb_phase:  Optional[int] = None,
) -> tuple:
    dev    = normals_hw.device
    ftype  = normals_hw.dtype
    H, W   = normals_hw.shape[:2]
    N_imgs = len(images)
    op = opt_params if opt_params is not None else _CT_SH_PARAMS
    if transforms is not None:
        tr_ab, tr_met, tr_rou = transforms["albedo"], transforms["metallic"], transforms["roughness"]
    else:
        tr_ab  = cfg.get("tr_albedo",   "none")
        tr_met = cfg.get("tr_metallic",  "none")
        tr_rou = cfg.get("tr_roughness", "none")

    def _t(x):
        return torch.from_numpy(np.asarray(x, np.float32)).to(dev, ftype) \
            if not isinstance(x, torch.Tensor) else x.to(dev, ftype)

    _sh_ord = int(cfg.get("sh_order", 2))
    if _sh_ord not in (2, 3):
        raise ValueError(f"sh_order must be 2 or 3, got {_sh_ord}")
    n_sh = (_sh_ord + 1) ** 2                     # 9 or 16
    _diffuse_fresnel = bool(cfg.get("diffuse_fresnel", True))
    _hl_mode = str(cfg.get("hl_mode", "analytic"))
    _n_log = N_imgs if cfg.get("wandb_max_images") is None else min(N_imgs, int(cfg["wandb_max_images"]))

    def _pad_sh(arr):
        """(9,3) GT coefficients → (n_sh,3), zero-padding band 3 if needed."""
        arr = np.asarray(arr, np.float32)
        if arr.shape[0] < n_sh:
            arr = np.concatenate([arr, np.zeros((n_sh - arr.shape[0], 3), np.float32)])
        return arr

    imgs_t    = torch.stack([_t(img) for img in images])
    flat_mask = mask_hw.reshape(-1)
    N_m       = normals_hw.reshape(-1, 3)[flat_mask]
    fp_m      = frag_pos_hw.reshape(-1, 3)[flat_mask]
    view_m    = _norm(cam_pos.unsqueeze(0) - fp_m)
    lut       = _get_ggx_sh_lut(dev, n_bands=_sh_ord + 1).to(ftype)
    mask_t    = mask_hw.unsqueeze(-1).to(dev)

    learnable    = []
    named_params = {}   # for gradient logging

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
        # frozen albedo: GT only when init_from_gt, else the SAME init a normal run
        # would use (mean image). A frozen param must never silently become GT —
        # that turns a warm-up (e.g. sh_only) into an inverse crime.
        if init_from_gt and gt_albedo is not None:
            base = torch.from_numpy(
                np.broadcast_to(np.asarray(gt_albedo, np.float32), (H, W, 3)).copy()
            ).to(dev, ftype)
        else:
            base = imgs_t.mean(0)
        albedo_param = _init_albedo(base, tr_ab)

    if "sh" in op:
        if init_from_gt and gt_sh_coeffs is not None:
            sh_coeffs = torch.stack([
                torch.from_numpy(_pad_sh(gt_sh_coeffs[k])).to(dev, ftype) for k in range(N_imgs)
            ]).requires_grad_(True)
        else:
            sh_init = torch.zeros(N_imgs, n_sh, 3, device=dev, dtype=ftype)
            sh_init[:, 0, :] = 1.5
            sh_coeffs = sh_init.clone().requires_grad_(True)
        learnable.append(sh_coeffs)
        named_params["sh"] = sh_coeffs
    else:
        sh_coeffs = torch.stack([
            torch.from_numpy(_pad_sh(gt_sh_coeffs[k])).to(dev, ftype) for k in range(N_imgs)
        ])

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
            # init_metallic overrides the flag-derived default outright, so a paper
            # recipe that starts at e.g. 0.05 (mostly dielectric) is expressible
            # without inventing another boolean.
            _mv = cfg.get("init_metallic")
            if _mv is None:
                _mv = 0.0 if cfg.get("init_spec_zero", False) else 0.5
            metallic_raw = _init_scalar(_mv, H, W, tr_met, dev=dev).to(ftype).requires_grad_(True)
        learnable.append(metallic_raw)
        named_params["metallic"] = metallic_raw
    else:
        # frozen metallic: GT only when init_from_gt, else the normal init.
        if init_from_gt:
            metallic_raw = (_init_map(_gt_met_np.reshape(H, W, 1), tr_met, dev).to(ftype)
                            if _gt_met_np.ndim > 0
                            else _init_scalar(_gt_met_scalar, H, W, tr_met, dev=dev).to(ftype))
        else:
            # init_metallic overrides the flag-derived default outright, so a paper
            # recipe that starts at e.g. 0.05 (mostly dielectric) is expressible
            # without inventing another boolean.
            _mv = cfg.get("init_metallic")
            if _mv is None:
                _mv = 0.0 if cfg.get("init_spec_zero", False) else 0.5
            metallic_raw = _init_scalar(_mv, H, W, tr_met, dev=dev).to(ftype)

    if "roughness" in op:
        if init_from_gt:
            roughness_raw = _init_map(_gt_rou_np.reshape(H, W, 1), tr_rou, dev).to(ftype).requires_grad_(True)
        else:
            _rv = cfg.get("init_roughness")
            if _rv is None:
                _rv = (1.0 if cfg.get("init_spec_zero", False)
                       else (0.1 if cfg.get("init_roughness_zero", False) else 0.5))
            roughness_raw = _init_scalar(_rv, H, W, tr_rou, dev=dev).to(ftype).requires_grad_(True)
        learnable.append(roughness_raw)
        named_params["roughness"] = roughness_raw
    else:
        # frozen roughness: GT only when init_from_gt, else the normal init.
        if init_from_gt:
            roughness_raw = (_init_map(_gt_rou_np.reshape(H, W, 1), tr_rou, dev).to(ftype)
                             if _gt_rou_np.ndim > 0
                             else _init_scalar(_gt_rou_scalar, H, W, tr_rou, dev=dev).to(ftype))
        else:
            _rv = cfg.get("init_roughness")
            if _rv is None:
                _rv = (1.0 if cfg.get("init_spec_zero", False)
                       else (0.1 if cfg.get("init_roughness_zero", False) else 0.5))
            roughness_raw = _init_scalar(_rv, H, W, tr_rou, dev=dev).to(ftype)

    # ── warm-start overrides (natural-space maps from a previous phase) ────────
    # init_maps["albedo"|"sh"|"metallic"|"roughness"] override the init in-place
    # (values are natural, i.e. post-transform); used for curriculum chaining.
    if init_maps is not None:
        with torch.no_grad():
            if init_maps.get("albedo") is not None:
                _am = np.broadcast_to(np.asarray(init_maps["albedo"], np.float32), (H, W, 3)).copy()
                albedo_param.data.copy_(_init_albedo(torch.from_numpy(_am).to(dev, ftype), tr_ab))
            if init_maps.get("sh") is not None:
                sh_coeffs.data.copy_(torch.stack([
                    torch.from_numpy(_pad_sh(init_maps["sh"][k])).to(dev, ftype)
                    for k in range(N_imgs)]))
            if init_maps.get("metallic") is not None:
                _mm = np.broadcast_to(np.asarray(init_maps["metallic"], np.float32), (H, W, 1)).copy()
                metallic_raw.data.copy_(_init_map(_mm, tr_met, dev).to(ftype))
            if init_maps.get("roughness") is not None:
                _rm = np.broadcast_to(np.asarray(init_maps["roughness"], np.float32), (H, W, 1)).copy()
                roughness_raw.data.copy_(_init_map(_rm, tr_rou, dev).to(ftype))

    # ── coarse-to-fine lighting: freeze SH bands above `sh_active_order` ───────
    # Keeps the full order-`sh_order` machinery but only lets the first (a+1)^2
    # coefficients move (a = sh_active_order), by zeroing the higher bands at init
    # and masking their gradient. Used for the "SH1 first, then SH2" curriculum.
    _sh_active = cfg.get("sh_active_order")
    if _sh_active is not None and "sh" in op:
        _na = (int(_sh_active) + 1) ** 2
        if _na < n_sh:
            with torch.no_grad():
                sh_coeffs[:, _na:, :] = 0.0
            _sh_band_mask = torch.zeros(1, n_sh, 1, device=dev, dtype=ftype)
            _sh_band_mask[:, :_na, :] = 1.0
            sh_coeffs.register_hook(lambda g, _m=_sh_band_mask: g * _m)

    # ── optional Gaussian noise on the metallic/roughness init (natural space) ─
    _spec_noise = float(cfg.get("init_spec_noise_std", 0.0) or 0.0)
    if _spec_noise > 0:
        if cfg.get("init_seed") is not None:
            torch.manual_seed(int(cfg["init_seed"]))
        with torch.no_grad():
            if "metallic" in op:
                _mn = (_fwd_metallic(metallic_raw, tr_met)
                       + torch.randn_like(metallic_raw) * _spec_noise).clamp(1e-4, 1 - 1e-4)
                metallic_raw.data.copy_(_init_map(_mn.cpu().numpy(), tr_met, dev).to(ftype))
            if "roughness" in op:
                _rn = (_fwd_roughness(roughness_raw, tr_rou)
                       + torch.randn_like(roughness_raw) * _spec_noise).clamp(1e-4, 1 - 1e-4)
                roughness_raw.data.copy_(_init_map(_rn.cpu().numpy(), tr_rou, dev).to(ftype))

    def _get_met(): return _fwd_metallic(metallic_raw, tr_met)
    def _get_rou(): return _fwd_roughness(roughness_raw, tr_rou)

    # ── Precompute geometry-only terms (never change during optimisation) ─────
    with torch.no_grad():
        _A_vals = [
            torch.pi,
            2*torch.pi/3, 2*torch.pi/3, 2*torch.pi/3,
            torch.pi/4,   torch.pi/4,   torch.pi/4, torch.pi/4, torch.pi/4,
        ]
        if _sh_ord >= 3:
            _A_vals += [0.0] * 7        # Lambertian ZH weight of band 3 is zero
        _A = N_m.new_tensor(_A_vals)
        _AY        = _A * _sh_basis(N_m, order=_sh_ord)               # (M, n_sh)
        _NdotV_raw = (N_m * view_m).sum(-1, keepdim=True)             # (M, 1)
        _NdotV     = _NdotV_raw.clamp(min=0.0)                        # (M, 1)
        _R         = _norm(2.0 * _NdotV_raw * N_m - view_m)           # (M, 3)
        _Y_R       = _sh_basis(_R, order=_sh_ord)                     # (M, n_sh)
        _front     = (_NdotV_raw > 0).to(ftype)                       # (M, 1)
        _imgs_m    = imgs_t.reshape(N_imgs, -1, 3)[:, flat_mask, :]   # (N, M, 3)

    opt   = steps._make_optimizer(learnable, cfg) if learnable else None
    sched = steps._make_scheduler(opt, cfg, cfg["n_iter"]) if opt is not None else None

    # ── Levenberg-Marquardt (alternative to LBFGS / Adam) ─────────────────────
    # LM minimises sum(r^2) directly, so _forward's scalar loss is re-expressed as
    # residuals: the data term exactly for L2 (r = recon-target), and every
    # non-negative penalty (L1/huber data, TV, metallic L1/binarize) via the
    # square-root trick. Scaling matches _forward, so sum(r^2) == its loss.
    _lm = None
    _lm_frac = [1.0]                       # chunk fraction, mirrors _forward's `frac`
    _lm = _lm_bs = None
    _lm_full = True
    if steps._optimizer_name(cfg) == "LM" and learnable:
        _lm, _lm_bs, _lm_full = build_lm_solver(
            cfg, learnable, named_params, albedo_param, sh_coeffs, metallic_raw,
            roughness_raw, flat_mask, N_imgs, N_m, view_m, _imgs_m, lut,
            _diffuse_fresnel, tr_ab, tr_met, tr_rou, _lm_frac, dev, ftype)

    # ── Variable projection (ct_sh only, like LM) ─────────────────────────────
    # Eliminates the SH lighting in closed form -- the render is linear in it once the
    # material is fixed -- and takes a lighting-projected Gauss-Newton step over the
    # per-pixel material. It therefore OWNS sh_coeffs: the elimination overwrites it
    # each iteration, so `sh` must not also be in opt_params.
    _vp = None
    if steps._optimizer_name(cfg) == "VARPRO":
        _vp = build_varpro_solver(
            cfg, albedo_param, sh_coeffs, metallic_raw, roughness_raw, flat_mask,
            _imgs_m, _AY, _Y_R, _NdotV, _front, lut, _sh_ord, _diffuse_fresnel,
            tr_ab, tr_met, tr_rou, dev, ftype)


    _step = [0]
    _loss_ml = [torch.zeros((), device=dev, dtype=ftype)]
    _loss_mb = [torch.zeros((), device=dev, dtype=ftype)]
    _loss_box = [torch.zeros((), device=dev, dtype=ftype)]
    _loss_lp = [torch.zeros((), device=dev, dtype=ftype)]
    # Lighting-prior reference (padded per image to n_sh), if one was supplied — either as
    # the `light_prior` keyword or via cfg["light_prior"] (so it rides along in a cfg dict).
    _lp_ref = light_prior if light_prior is not None else cfg.get("light_prior", None)
    _light_prior_t = None
    if _lp_ref is not None:
        _light_prior_t = torch.stack([
            torch.from_numpy(_pad_sh(np.asarray(_lp_ref[k], np.float32))).to(dev, ftype)
            for k in range(N_imgs)])
    _recon_rmse = [0.0]           # per-image recon RMSE/MAE, stashed by _forward
    _recon_mae  = [0.0]

    def _forward(img_indices=None):
        # img_indices: iterable of image indices for gradient-accumulation
        # chunks (None = all images). Image-independent loss terms are scaled
        # by the chunk fraction so the chunk losses sum to the full-batch loss.
        idx   = None if img_indices is None else list(img_indices)
        n_sel = N_imgs if idx is None else len(idx)
        frac  = n_sel / N_imgs
        _spec_warmup = _step[0] < cfg.get("spec_warmup_steps", 0)
        albedo      = _fwd_albedo(albedo_param, tr_ab)                # (H, W, 3)
        albedo_m    = albedo.reshape(-1, 3)[flat_mask]                 # (M, 3)
        metallic    = _get_met()                                       # (H, W, 1)
        roughness   = _get_rou()                                       # (H, W, 1)
        metallic_m  = metallic.reshape(-1, 1)[flat_mask]               # (M, 1)
        roughness_m = roughness.reshape(-1, 1)[flat_mask]               # (M, 1)
        met_m_true  = metallic_m                                       # for regularisation
        rou_m_true  = roughness_m                                      # (pre-warmup, for box penalty)
        if _spec_warmup:
            metallic_m  = albedo_m.new_zeros(albedo_m.shape[0], 1)
            roughness_m = albedo_m.new_ones(albedo_m.shape[0], 1)
        if _step[0] < cfg.get("min_metallic_steps", 0):
            metallic_m = metallic_m.clamp(min=0.1)

        # ── material terms (identical for all N images) ───────────────────
        f0   = 0.04 * (1.0 - metallic_m) + albedo_m * metallic_m     # (M, 3)
        F    = f0 + (1.0 - f0) * (1.0 - _NdotV).pow(5)               # (M, 3)
        alpha = roughness_m ** 2
        G1    = _NdotV / (_NdotV * (1.0 - alpha**2/2.0) + alpha**2/2.0 + 1e-6)  # (M, 1)
        # Diffuse weight. Must match the data generator (shade_ct_sh) and the
        # final shadings/relight, or recon_rmse decouples from the data loss and
        # the inverse crime can't reach 0. diffuse_fresnel=True multiplies by
        # (1-F) (energy taken by specular); default True = shade_ct_sh default.
        k_d   = 1.0 - metallic_m                                       # (M, 1)
        if _diffuse_fresnel:
            k_d = (1.0 - F) * k_d                                      # (M, 3)

        # ── specular SH filter B (roughness-dependent, recomputed each step) ─
        # Band source follows cfg["hl_mode"]; must match the data generator + final
        # shadings/relight so recon_rmse tracks the data loss (the inverse crime).
        Bvals = ggx_sh_bands(roughness_m.squeeze(-1), _hl_mode, lut,
                             n_bands=_sh_ord + 1)                      # (M, n_bands)
        _bp = [Bvals[..., 0:1],
               Bvals[..., 1:2].expand(-1, 3),
               Bvals[..., 2:3].expand(-1, 5)]
        if _sh_ord >= 3:
            _bp.append(Bvals[..., 3:4].expand(-1, 7))
        BY = torch.cat(_bp, dim=-1) * _Y_R                             # (M, n_sh)

        # ── batch SH projection over the selected images ───────────────────
        # (1, M, 9) @ (n, 9, 3) → (n, M, 3) via broadcast matmul
        sh_sel   = sh_coeffs if idx is None else sh_coeffs[idx]       # (n, 9, 3)
        imgs_sel = _imgs_m   if idx is None else _imgs_m[idx]         # (n, M, 3)
        irr_all    = (_AY.unsqueeze(0) @ sh_sel).clamp(min=0)         # (n, M, 3)
        L_spec_all = (BY.unsqueeze(0)  @ sh_sel).clamp(min=0)         # (n, M, 3)

        diff_m  = k_d * albedo_m / torch.pi * irr_all                 # (n, M, 3)
        spec_m  = F * G1 * L_spec_all / 4.0                           # (n, M, 3)
        recon_m = (diff_m + spec_m) * _front                          # (n, M, 3)

        # ── loss in masked pixel space (scatter back not needed) ──────────
        resid = recon_m - imgs_sel
        with torch.no_grad():                        # per-image recon RMSE/MAE (for logging)
            _rd = resid.detach()
            _recon_rmse[0] = float(_rd.pow(2).mean(dim=(-2, -1)).sqrt().mean())
            _recon_mae[0]  = float(_rd.abs().mean(dim=(-2, -1)).mean())
        if cfg["loss"] == "L1":
            err = resid.abs()
        elif cfg["loss"] == "huber":
            _d = cfg.get("huber_delta", 0.05)
            _a = resid.abs()
            err = torch.where(_a <= _d, 0.5 * resid ** 2, _d * (_a - 0.5 * _d))
        else:
            err = resid ** 2
        # mean over images (n_sel/N_imgs lets gradient-accumulation chunks sum
        # to the full-batch value): loss_data is comparable across dataset
        # sizes. Regularizer lambdas are absolute (not divided by N).
        loss_data = err.mean() * (n_sel / N_imgs)

        loss_sparse = frac * cfg["lambda_sparse"] * _tv(albedo.permute(2, 0, 1))
        loss_white  = frac * cfg["lambda_white"]  * (albedo.mean() - 0.5) ** 2
        loss_tv     = frac * cfg["lambda_tv"] * (
            _tv(albedo.permute(2, 0, 1)) +
            _tv(metallic.permute(2, 0, 1)) +
            _tv(roughness.permute(2, 0, 1))
        )
        loss_metallic_l1       = frac * cfg.get("lambda_metallic_l1",       0.0) * met_m_true.abs().mean()
        loss_metallic_binarize = frac * cfg.get("lambda_metallic_binarize",  0.0) * (met_m_true * (1.0 - met_m_true)).mean()
        # Soft box constraint: squared hinge penalty for material values outside [0, 1].
        # Keeps natural-space (identity-transform) albedo/metallic/roughness physical
        # without a sigmoid's vanishing gradient at the bounds. Zero under sigmoid.
        _lam_box = cfg.get("lambda_box", 0.0)
        def _box(x):
            return (torch.relu(-x) ** 2 + torch.relu(x - 1.0) ** 2).mean()
        loss_box = frac * _lam_box * (_box(albedo_m) + _box(met_m_true) + _box(rou_m_true))
        # Lighting prior: ||sh - reference||^2 if a reference was given, else an SH-smoothness
        # prior shrinking the non-DC (directional) coefficients toward 0.
        _lam_lp = cfg.get("lambda_light_prior", 0.0)
        if _lam_lp:
            if _light_prior_t is not None:
                _ref = _light_prior_t if idx is None else _light_prior_t[idx]
                loss_light = frac * _lam_lp * ((sh_sel - _ref) ** 2).mean()
            else:
                loss_light = frac * _lam_lp * (sh_sel[:, 1:, :] ** 2).mean()
        else:
            loss_light = sh_sel.new_zeros(())
        # Monochrome-light prior: push each image's SH toward achromatic (R=G=B) so colour
        # is explained by albedo, not the lighting (nvdiffrec-style white-balance).
        _lam_mono = cfg.get("lambda_light_mono", 0.0)
        if _lam_mono:
            loss_light = loss_light + frac * _lam_mono * (
                (sh_sel - sh_sel.mean(dim=-1, keepdim=True)) ** 2).mean()
        _loss_ml[0] = loss_metallic_l1.detach()
        _loss_mb[0] = loss_metallic_binarize.detach()
        _loss_box[0] = loss_box.detach()
        _loss_lp[0] = loss_light.detach()
        return (loss_data + loss_sparse + loss_white + loss_tv
                + loss_metallic_l1 + loss_metallic_binarize + loss_box + loss_light,
                loss_data, loss_sparse, loss_white, loss_tv)

    def _forward_components():
        with torch.no_grad():
            albedo_m    = _fwd_albedo(albedo_param, tr_ab).reshape(-1, 3)[flat_mask]
            metallic_m  = _fwd_metallic(metallic_raw, tr_met).reshape(-1, 1)[flat_mask]
            roughness_m = _fwd_roughness(roughness_raw, tr_rou).reshape(-1, 1)[flat_mask]
            result = []
            for k in range(N_imgs):
                _, comps = shade_ct_sh(view_m, N_m, albedo_m, sh_coeffs[k],
                                       metallic_m, roughness_m, lut=lut,
                                       diffuse_fresnel=_diffuse_fresnel, hl_mode=_hl_mode,
                                       return_components=True)
                result.append(comps)
        return result

    gt_map_grad = {
        "albedo":    np.broadcast_to(np.asarray(gt_albedo,   np.float32), (H, W, 3)).copy() if gt_albedo is not None else None,
        "sh":        np.stack([_pad_sh(s) for s in gt_sh_coeffs]).astype(np.float32) if gt_sh_coeffs is not None else None,
        "metallic":  np.broadcast_to(np.asarray(gt_metallic,  np.float32), (H, W, 1)).copy(),
        "roughness": np.broadcast_to(np.asarray(gt_roughness, np.float32), (H, W, 1)).copy(),
    }
    fwd_map_grad = {
        "albedo":    lambda p: _fwd_albedo(p, tr_ab),
        "sh":        lambda p: p,
        "metallic":  lambda p: _fwd_metallic(p, tr_met),
        "roughness": lambda p: _fwd_roughness(p, tr_rou),
    }

    if log_gradients and grad_log_dir is not None:
        grad_log_dir.mkdir(parents=True, exist_ok=True)

    history = []

    # Precompute flat GT tensors for per-step RMSE logging
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
        """Pixel-level RMSE/MAE against GT intrinsics (only when GT is available)."""
        out = {}
        if _gt_ab_m is not None:
            scale = (_gt_ab_m * ab_m).sum(0) / (ab_m * ab_m).sum(0).clamp(1e-8)
            out["albedo_rmse"] = float((ab_m * scale - _gt_ab_m).pow(2).mean().sqrt())
            out["albedo_mae"]  = float((ab_m * scale - _gt_ab_m).abs().mean())
        out["metallic_rmse"]  = float((met_m - _gt_met_m).pow(2).mean().sqrt())
        out["metallic_mae"]   = float((met_m - _gt_met_m).abs().mean())
        out["roughness_rmse"] = float((rou_m - _gt_rou_m).pow(2).mean().sqrt())
        out["roughness_mae"]  = float((rou_m - _gt_rou_m).abs().mean())
        return out

    # ── held-out relighting metric ────────────────────────────────────────────
    _val_imgs_m = _val_sh = None
    if val_images and val_sh_coeffs is not None:
        _val_imgs_m = torch.stack([_t(v) for v in val_images]) \
            .reshape(len(val_images), -1, 3)[:, flat_mask, :]              # (V, M, 3)
        _val_sh = torch.stack([
            torch.from_numpy(_pad_sh(s)).to(dev, ftype)
            for s in val_sh_coeffs])                                       # (V, n_sh, 3)

    def _relight_metrics(ab_m, met_m, rou_m):
        """Render the held-out images with CURRENT intrinsics + their GT
        lighting; report the per-image error vs the observed val images."""
        if _val_imgs_m is None:
            return {}
        rs, ms = [], []
        with torch.no_grad():
            for k in range(_val_imgs_m.shape[0]):
                recon = shade_ct_sh(view_m, N_m, ab_m, _val_sh[k],
                                    met_m, rou_m, lut=lut, diffuse_fresnel=_diffuse_fresnel,
                                    hl_mode=_hl_mode)
                d = recon - _val_imgs_m[k]
                rs.append(float(d.pow(2).mean().sqrt()))
                ms.append(float(d.abs().mean()))
        return {"relight_rmse": float(np.mean(rs)), "relight_mae": float(np.mean(ms))}

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
        # The step=-1 "before optimization" baseline only makes sense once. In a
        # curriculum every phase shares one wandb run on a single step timeline
        # (wandb_step_offset), so later phases skip this -- their starting state is just
        # the previous phase's final logged step.
        if wandb_step_offset == 0:
            _init_payload = _structured_scalar_log(
                loss=_il, l_d=_ild, l_s=_ils, l_w=_ilw, l_tv=_iltv,
                loss_ml=_loss_ml[0], loss_mb=_loss_mb[0], scale3=_init_scale,
                gt_metrics=_gt_rmse_metrics(_ab_m, _met_m, _rou_m),
                relight=_relight_metrics(_ab_m, _met_m, _rou_m),
                recon_rmse=_recon_rmse[0], recon_mae=_recon_mae[0],
                lr=opt.param_groups[0]["lr"] if opt is not None else 0.0,
            )
            if wandb_phase is not None:
                _init_payload["phase"] = wandb_phase
            wandb_run.log(_init_payload, step=-1)
    _rescale_every = cfg.get("rescale_every", 0)
    _img_batch = int(cfg.get("img_batch", 0) or 0)
    _use_img_batch = 0 < _img_batch < N_imgs
    if _use_img_batch and log_gradients:
        print("  [img_batch] disabled: incompatible with log_gradients")
        _use_img_batch = False
    _lm_info = {}
    t0 = time.perf_counter()
    for i in range(cfg["n_iter"]):
        _step[0] = i
        if _vp is not None:
            _vp_info = _vp.step()
            with torch.no_grad():                       # report the full-batch loss
                loss, l_d, l_s, l_w, l_tv = _forward()
        elif _lm is not None:
            # full batch (idx=None) or a random image mini-batch of `lm_batch_size`
            if _lm_full:
                _idx, _lm_frac[0] = None, 1.0
            else:
                _idx = torch.randperm(N_imgs, device=dev)[:_lm_bs]
                _lm_frac[0] = _lm_bs / N_imgs
            _lm_info = _lm.step(_idx)
            with torch.no_grad():                       # report the full-batch loss
                loss, l_d, l_s, l_w, l_tv = _forward()
        elif opt is not None:
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
                    and "albedo" in op and "sh" in op and _gt_ab_m is not None):
                with torch.no_grad():
                    _rescale_albedo_lighting(
                        albedo_param, [sh_coeffs], tr_ab, flat_mask, _gt_ab_m)
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
                if wandb_phase is not None:
                    # Numeric so it charts as a step function: the boundary between
                    # curriculum phases (e.g. LBFGS -> VarPro) is then visible on any
                    # metric, and metrics can be grouped/filtered by phase in the UI.
                    _payload["phase"] = wandb_phase
                if _lm_info:
                    _payload["lm/damping"]  = _lm_info["damping"]
                    _payload["lm/accepted"] = float(_lm_info["accepted"])
                    _payload["lm/attempts"] = _lm_info["attempts"]
                # In a curriculum the phases lay end to end on one step timeline; the
                # offset is 0 for a standalone run, so this is a no-op there.
                wandb_run.log(_payload, step=i + wandb_step_offset)
    total_time = time.perf_counter() - t0

    with torch.no_grad():
        albedo_out = _fwd_albedo(albedo_param, tr_ab).cpu().numpy()
        sh_out     = sh_coeffs.cpu().numpy()
        met_out    = _get_met().cpu().numpy()
        rou_out    = _get_rou().cpu().numpy()
        met_m      = torch.from_numpy(met_out).to(dev).reshape(-1, 1)[flat_mask]
        rou_m      = torch.from_numpy(rou_out).to(dev).reshape(-1, 1)[flat_mask]
        albedo_t2  = _fwd_albedo(albedo_param, tr_ab)
        shadings   = []
        for k in range(N_imgs):
            albedo_m = albedo_t2.reshape(-1, 3)[flat_mask]
            recon_m  = shade_ct_sh(view_m, N_m, albedo_m, sh_coeffs[k],
                                   met_m, rou_m, lut=lut, diffuse_fresnel=_diffuse_fresnel,
                                   hl_mode=_hl_mode)
            s = albedo_t2.new_zeros(H, W, 3)
            s.reshape(-1, 3)[flat_mask] = recon_m
            s *= mask_t
            shadings.append(s.cpu().numpy())

    return albedo_out, sh_out, met_out, rou_out, shadings, history, total_time
