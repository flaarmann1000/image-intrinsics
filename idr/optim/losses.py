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


# ── segmentation cohesion prior (SAM masks) ───────────────────────────────────
def build_seg_groups(seg_labels_hw, flat_mask):
    """Group the FOREGROUND pixels by SAM segment for the cohesion prior.

    seg_labels_hw : (H, W) int array — class id >= 0, or -1 for unlabeled (black in
                    segmentation.png). flat_mask : (H*W,) bool tensor (foreground).
    Returns (seg_ids_m, classified_m, n_seg, counts) with labels remapped to contiguous
    0..n_seg-1 over the classified foreground pixels, or None if nothing is classified.
    """
    dev = flat_mask.device
    lab = torch.as_tensor(seg_labels_hw, dtype=torch.long, device=dev).reshape(-1)[flat_mask]  # (M,)
    classified = lab >= 0
    if not bool(classified.any()):
        return None
    uniq = torch.unique(lab[classified])
    remap = torch.full((int(uniq.max().item()) + 1,), -1, dtype=torch.long, device=dev)
    remap[uniq] = torch.arange(uniq.numel(), device=dev)
    seg_ids = torch.zeros_like(lab)
    seg_ids[classified] = remap[lab[classified]]
    n_seg = int(uniq.numel())
    counts = torch.bincount(seg_ids[classified], minlength=n_seg).clamp(min=1)
    return seg_ids, classified, n_seg, counts


def segment_cohesion(vals_m, groups):
    """Mean squared deviation of each classified pixel from its OWN segment's current
    mean (a soft "one value per object" pull). vals_m: (M,) or (M,1); groups from
    build_seg_groups. Recomputed from live estimates every call, so pixels may disagree
    with their object mean if the data demands it. Scaled as a mean (like `_tv`)."""
    seg_ids, classified, n_seg, counts = groups
    v = vals_m.reshape(-1)[classified]
    ids = seg_ids[classified]
    sums = torch.zeros(n_seg, dtype=v.dtype, device=v.device).scatter_add(0, ids, v)
    means = sums / counts.to(v.dtype)
    return ((v - means[ids]) ** 2).mean()
