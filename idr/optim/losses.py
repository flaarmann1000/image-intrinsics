"""Data-fit losses and regularizers.

Each penalty appears twice by necessity: once as a scalar (`_tv`, `_loss_fn`) for
the gradient-based optimizers, and once as residuals (`_tv_residuals`, `_sqrt_res`)
for Levenberg-Marquardt, which needs r such that sum(r^2) is the same objective.
"""
from __future__ import annotations

import torch

def _tv(x: torch.Tensor) -> torch.Tensor:
    """Isotropic total variation. x: [..., H, W] (last two dims are spatial).

    Was imported from raw_optimizer/optimizer.py, whose only other export
    (`optimize`) is a superseded pre-CT entry point; it now lives in legacy/misc/.
    Companion of `_tv_residuals` below, which is the same penalty expressed as LM
    residuals rather than a scalar."""
    dh = x[..., 1:, :] - x[..., :-1, :]
    dw = x[..., :, 1:] - x[..., :, :-1]
    return (dh**2 + 1e-8).sqrt().mean() + (dw**2 + 1e-8).sqrt().mean()


def _loss_fn(recon, target, mask_t, mode, huber_delta=0.05):
    resid = recon - target
    if mode == "L1":
        diff = resid.abs()
    elif mode == "huber":
        a = resid.abs()
        diff = torch.where(a <= huber_delta, 0.5 * resid ** 2,
                           huber_delta * (a - 0.5 * huber_delta))
    else:
        diff = resid ** 2
    return diff[mask_t.expand_as(diff)].mean()


def _sqrt_res(x, eps=1e-12):
    """Square-root trick: a non-negative loss term c_i is represented by the
    residual sqrt(c_i), so sum_i r_i^2 == sum_i c_i exactly. eps keeps the
    derivative of sqrt finite at 0."""
    return (x + eps).sqrt()


def _tv_residuals(x, scale):
    """Residuals whose squared sum equals `scale * _tv(x)` (isotropic TV)."""
    dh = x[..., 1:, :] - x[..., :-1, :]
    dw = x[..., :, 1:] - x[..., :, :-1]
    rh = _sqrt_res(scale * (dh**2 + 1e-8).sqrt() / dh.numel())
    rw = _sqrt_res(scale * (dw**2 + 1e-8).sqrt() / dw.numel())
    return torch.cat([rh.reshape(-1), rw.reshape(-1)])
