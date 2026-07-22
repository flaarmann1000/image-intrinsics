"""Metrics over recovered intrinsics.

The albedo/lighting scale is unobservable from the images -- the two trade off by a
global factor -- so these fit and apply that factor before comparing to GT.
"""
from __future__ import annotations

import numpy as np
import torch

from idr.optim.transforms import _fwd_albedo

def _albedo_lighting_scale(
    albedo_param: torch.Tensor,
    tr_ab: str,
    flat_mask: torch.Tensor,
    gt_ab_m: torch.Tensor,
) -> torch.Tensor:
    """Per-channel LS scale aligning estimated albedo to GT. No side effects."""
    ab_m = _fwd_albedo(albedo_param, tr_ab).detach().reshape(-1, 3)[flat_mask]
    return (gt_ab_m * ab_m).sum(0) / (ab_m * ab_m).sum(0).clamp(1e-8)  # (3,)


def _rescale_albedo_lighting(
    albedo_param: torch.Tensor,
    lighting_params: list,
    tr_ab: str,
    flat_mask: torch.Tensor,
    gt_ab_m: torch.Tensor,
) -> torch.Tensor:
    """Rescale albedo and lighting in-place to align estimated albedo with GT.

    Returns the applied scale (3,) for logging.
    lighting_params: list of tensors with shape (..., 3) — sh_coeffs or env_maps.
    """
    scale = _albedo_lighting_scale(albedo_param, tr_ab, flat_mask, gt_ab_m)  # (3,)
    cur_out = _fwd_albedo(albedo_param, tr_ab)                           # (H, W, 3)
    new_out = (cur_out * scale[None, None, :])
    if tr_ab == "log":
        albedo_param.data.copy_(torch.log(new_out))
    elif tr_ab == "sigmoid":
        albedo_param.data.copy_(torch.logit(new_out.clamp(1e-6, 1 - 1e-6)))
    else:
        albedo_param.data.copy_(new_out)
    for lp in lighting_params:
        lp.data /= scale  # (..., 3) / (3,) — broadcasts over all leading dims
    return scale
