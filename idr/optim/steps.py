"""Optimizer construction and one optimization step.

`_opt_step` and `_make_optimizer` are patched by name in profile_decomposition.py to
time the phases, so they are deliberately kept as module-level functions.
"""
from __future__ import annotations

import numpy as np
import torch

def _optimizer_name(cfg) -> str:
    return str(cfg.get("optimizer", "LBFGS")).upper()


def _make_optimizer(params, cfg):
    name = _optimizer_name(cfg)
    if name in ("LM", "VARPRO"):
        # Neither is a torch.optim.Optimizer: LM drives itself (idr/optim/lm),
        # VarPro eliminates the lighting and takes its own reduced step
        # (idr/optim/varpro). Falling through to the Adam default below would
        # silently build an optimizer nobody steps.
        return None
    if name == "LBFGS":
        return torch.optim.LBFGS(
            params, 
            lr=cfg["lr"],            
            max_iter=cfg["lbfgs_max_iter"],
            line_search_fn="strong_wolfe",
            tolerance_grad=1e-9,
            tolerance_change=1e-11,
            history_size=10

        )
    return torch.optim.Adam(params, lr=cfg["lr"])


def _make_scheduler(opt, cfg, n_steps):
    """Return an LR scheduler for Adam, or None (LBFGS / 'none').

    cfg keys used:
        lr_schedule       : "none" | "cosine" | "step" | "linear" | "exponential"
        lr_end            : target LR at the end of training (default 0).
                            Used by cosine (eta_min), linear (end_factor), exponential (gamma).
                            Ignored by step (use lr_schedule_gamma instead).
        lr_schedule_step  : step-size in iters for "step" mode (default 50)
        lr_schedule_gamma : per-step decay factor for "step" mode (default 0.5)
    """
    if _optimizer_name(cfg) in ("LBFGS", "LM"):
        return None
    mode    = cfg.get("lr_schedule", "none")
    lr_0    = cfg["lr"]
    lr_end  = cfg.get("lr_end", 0.0)
    if mode == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=n_steps, eta_min=lr_end,
        )
    if mode == "step":
        return torch.optim.lr_scheduler.StepLR(
            opt, step_size=cfg.get("lr_schedule_step", 50),
            gamma=cfg.get("lr_schedule_gamma", 0.5),
        )
    if mode == "linear":
        end_factor = lr_end / lr_0 if lr_0 > 0 else 0.0
        return torch.optim.lr_scheduler.LinearLR(
            opt, start_factor=1.0, end_factor=end_factor, total_iters=n_steps,
        )
    if mode == "exponential":
        # compute per-step gamma so that lr_0 * gamma^n_steps == lr_end
        floor     = max(lr_end, 1e-12)
        gamma     = (floor / lr_0) ** (1.0 / max(n_steps, 1)) if lr_0 > 0 else 1.0
        return torch.optim.lr_scheduler.ExponentialLR(opt, gamma=gamma)
    return None


def _opt_step(opt, forward_fn, cfg):
    """Single optimizer step; returns (total_loss, loss_data, loss_sparse, loss_white, loss_tv)."""
    if cfg["optimizer"] == "LBFGS":
        def closure():
            opt.zero_grad()
            loss, *_ = forward_fn()
            loss.backward()
            return loss
        try:
            opt.step(closure)
        except (IndexError, TypeError):
            opt.state.clear()  # line search failed; reset LBFGS state so next iter starts fresh
        with torch.no_grad():
            return forward_fn()
    else:
        opt.zero_grad()
        result = forward_fn()
        result[0].backward()
        opt.step()
        return result


def _opt_step_img_batched(opt, forward_fn, cfg, n_imgs, img_batch):
    """Single optimizer step with gradient accumulation over image chunks.

    Behaves like _opt_step, but each closure evaluation runs forward+backward
    on `img_batch` images at a time, so the autograd graph only ever holds one
    chunk. The summed loss/gradients equal the full-batch ones up to float
    summation order — NOT stochastic mini-batching, so it is safe under LBFGS.

    forward_fn must accept an iterable of image indices (or None for all) and
    scale its image-independent loss terms by len(indices)/n_imgs.
    """
    def closure():
        opt.zero_grad()
        total = 0.0
        for b in range(0, n_imgs, img_batch):
            loss_b, *_ = forward_fn(range(b, min(b + img_batch, n_imgs)))
            loss_b.backward()
            total += float(loss_b.detach())
        return torch.tensor(total)

    if cfg["optimizer"] == "LBFGS":
        try:
            opt.step(closure)
        except (IndexError, TypeError):
            opt.state.clear()  # line search failed; reset LBFGS state so next iter starts fresh
    else:
        closure()
        opt.step()
    # re-evaluate at the accepted parameters for accurate logging (graph-free,
    # so memory stays bounded regardless of n_imgs)
    with torch.no_grad():
        return forward_fn(None)
