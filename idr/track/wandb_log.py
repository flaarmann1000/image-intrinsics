"""Experiment logging: scalar payloads and image conversions for wandb."""
from __future__ import annotations

import numpy as np
import torch

from pathlib import Path

from idr.render import build_sh_basis, EnvMap

def _sh_coeffs_to_env_img(coeffs: np.ndarray, resolution: int = 64) -> np.ndarray:
    """(9|16,3) SH coefficients → (H,W,3) float32 image normalized to [0,1]."""
    coeffs = np.asarray(coeffs, np.float32)
    order = 3 if coeffs.shape[0] == 16 else 2
    dirs = EnvMap._sh_grid_dirs(resolution)
    img = np.maximum(build_sh_basis(dirs, order=order) @ coeffs, 0.0)
    mx = float(img.max())
    return img / max(mx, 1e-8)


def _env_flat_to_img(env_flat: np.ndarray, env_H: int, env_W: int) -> np.ndarray:
    """(P,3) flat env-map → (H,W,3) float32 image normalized to [0,1]."""
    img = env_flat.reshape(env_H, env_W, 3)
    mx = float(img.max())
    return img / max(mx, 1e-8)


def _save_grad_step(
    step:       int,
    named_params: dict,        # {name: tensor} — learnable raw params
    pre_raw:    dict,          # {name: tensor} — param.data clone before opt step
    gt_map:     dict,          # {name: np.ndarray} — GT values in physical space, broadcast to param shape
    fwd_map:    dict,          # {name: callable(raw) -> physical}
    fwd_comps_fn,              # callable() -> list[dict] of shade components per image
    losses:     tuple,         # (total, data, sparse, white, tv)
    grad_log_dir: Path,
    flat_mask:  torch.Tensor,
    H: int, W: int,
) -> None:
    """Save one gradient-flow snapshot as a compressed npz file."""
    data = {}
    data["loss_total"]  = np.float32(losses[0])
    data["loss_data"]   = np.float32(losses[1])
    data["loss_sparse"] = np.float32(losses[2])
    data["loss_white"]  = np.float32(losses[3])
    data["loss_tv"]     = np.float32(losses[4])
    with torch.no_grad():
        for name, p in named_params.items():
            raw_np  = p.data.detach().cpu().numpy()
            data[f"{name}_raw"]    = raw_np
            data[f"{name}_update"] = raw_np - pre_raw[name].cpu().numpy()
            if p.grad is not None:
                data[f"{name}_grad"] = p.grad.detach().cpu().numpy()
            val = fwd_map[name](p.data)
            val_np = val.detach().cpu().numpy()
            data[f"{name}_value"] = val_np
            if name in gt_map and gt_map[name] is not None:
                gt_np = np.broadcast_to(gt_map[name], val_np.shape)
                data[f"{name}_gt_error"] = val_np - gt_np

        comps_per_img = fwd_comps_fn()
        for k, comps in enumerate(comps_per_img):
            for cname, ctensor in comps.items():
                c_np = ctensor.detach().float().cpu()
                C = c_np.shape[-1] if c_np.dim() > 1 else 1
                full = torch.zeros(H * W, C, dtype=torch.float32)
                full[flat_mask.cpu()] = c_np.reshape(-1, C)
                data[f"shade_k{k:02d}_{cname}"] = full.reshape(H, W, C).numpy()

    np.savez_compressed(grad_log_dir / f"step_{step:05d}.npz", **data)


def _structured_scalar_log(*, loss, l_d, l_s, l_w, l_tv, loss_ml, loss_mb,
                           scale3, gt_metrics, relight, recon_rmse, recon_mae, lr):
    """Per-step wandb scalars grouped into three sections by name prefix:
      loss/*  — loss terms + albedo scales
      rmse/*  — intrinsics (albedo/roughness/metallic) + recon + relight RMSE
      mae/*   — the same, MAE
    """
    d = {
        "loss/total": float(loss), "loss/data": float(l_d),
        "loss/tv": float(l_tv), "loss/sparse": float(l_s), "loss/white": float(l_w),
        "loss/metallic_l1": float(loss_ml), "loss/metallic_binarize": float(loss_mb),
        "lr": float(lr),
    }
    if recon_rmse is not None:
        d["rmse/recon"] = float(recon_rmse)
    if recon_mae is not None:
        d["mae/recon"] = float(recon_mae)
    if scale3 is not None:
        d["loss/albedo_scale_r"] = float(scale3[0])
        d["loss/albedo_scale_g"] = float(scale3[1])
        d["loss/albedo_scale_b"] = float(scale3[2])
        d["loss/albedo_scale_mean"] = float(sum(float(x) for x in scale3) / 3.0)
    for k, v in (gt_metrics or {}).items():          # e.g. albedo_rmse -> rmse/albedo
        base, kind = k.rsplit("_", 1)
        d[f"{kind}/{base}"] = v
    for k, v in (relight or {}).items():             # relight_rmse -> rmse/relight
        base, kind = k.rsplit("_", 1)
        d[f"{kind}/{base}"] = v
    return d
