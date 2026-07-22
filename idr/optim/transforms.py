"""Domain transforms between raw (unconstrained) parameters and physical values.

The optimizer works in an unconstrained space; these map to/from it. `_fwd_*` go
raw -> physical during the forward pass, `_init_*` invert a physical starting value
back into raw space.
"""
from __future__ import annotations

import numpy as np
import torch

from idr.config import NAMED_TRANSFORMS

def _softplus_inv(x: torch.Tensor) -> torch.Tensor:
    """Inverse of softplus: softplus(result) ≈ x for x > 0."""
    return torch.log(torch.expm1(x))


def _transforms_folder(tr: dict) -> str:
    if tr == NAMED_TRANSFORMS["none"]: return "no_transforms"
    if tr == NAMED_TRANSFORMS["all"]:  return "all_transforms"
    if tr == NAMED_TRANSFORMS["only_softplus"]:  return "only_softplus_transforms"
    if tr == NAMED_TRANSFORMS["only_shininess"]:  return "only_shininess_transforms"
    parts = [f"{k}={v}" for k, v in sorted(tr.items()) if v != "none"]
    return "tr_" + ",".join(parts)


def _parse_transforms(spec: str) -> dict:
    if spec in NAMED_TRANSFORMS:
        return dict(NAMED_TRANSFORMS[spec])
    base = dict(NAMED_TRANSFORMS["none"])
    for part in spec.split(","):
        k, v = part.strip().split("=")
        base[k] = v
    return base


def _fwd_albedo(p: torch.Tensor, t: str) -> torch.Tensor:
    if t == "sigmoid": return torch.sigmoid(p) 
    if t == "log": return torch.exp(p) 
    return  p


def _fwd_metallic(p: torch.Tensor, t: str) -> torch.Tensor:
    if t == "sigmoid":    return torch.sigmoid(p)
    if t == "sigmoid_sq": return torch.sigmoid(p) ** 2
    return p


def _fwd_roughness(p: torch.Tensor, t: str) -> torch.Tensor:
    if t == "sigmoid":    return torch.sigmoid(p)
    if t == "sigmoid_sq": return torch.sigmoid(p) ** 2
    return p


def _fwd_shininess(p: torch.Tensor, t: str, s_min: float, s_max: float) -> torch.Tensor:
    if t == "sigmoid":
        return s_min + (s_max - s_min) * torch.sigmoid(p)
    elif t == "log":
        return torch.exp(p)
    else:
        return p


def _fwd_ks(p: torch.Tensor, t: str) -> torch.Tensor:
    if t == "sigmoid":    return torch.sigmoid(p)
    if t == "sigmoid_sq": return torch.sigmoid(p) ** 2
    return p


def _fwd_env(p: torch.Tensor, t: str) -> torch.Tensor:
    import torch.nn.functional as F
    return F.softplus(p) if t == "softplus" else p


def _init_albedo(base: torch.Tensor, t: str) -> torch.Tensor:
    """base: (H,W,3). Returns raw param in transform space."""
    if t == "log":     return torch.log(base)
    if t == "sigmoid": return torch.logit(base.clamp(1e-6, 1 - 1e-6))
    return base.clone()


def _init_scalar(val: float, H: int, W: int, t: str,
                 squeeze_fn=None, dev=None) -> torch.Tensor:
    """Initialize a (H,W,1) scalar param for a fixed value."""
    dtype = torch.float32
    if t in ("sigmoid", "sigmoid_r"):
        v = np.clip(val, 1e-6, 1-1e-6)
        raw = float(np.log(v / (1 - v)))
    elif t == "sigmoid_sq":
        v = np.clip(np.sqrt(np.clip(val, 0, 1)), 1e-6, 1-1e-6)
        raw = float(np.log(v / (1 - v)))
    else:
        raw = float(val)
    return torch.full((H, W, 1), raw, dtype=dtype, device=dev)


def _init_map(arr: np.ndarray, t: str, dev) -> torch.Tensor:
    """Initialize a (H, W, 1) raw param from a spatial GT map."""
    x = torch.from_numpy(arr.astype(np.float32)).to(dev)
    if t in ("sigmoid", "sigmoid_r"):
        return torch.logit(x)
    elif t == "sigmoid_sq":
        return torch.logit(x.clamp(1e-6, 1 - 1e-6).sqrt())
    else:
        return x.clone()


def _init_env(gt_flat: np.ndarray, t: str, dev) -> torch.Tensor:
    gt_t = torch.from_numpy(gt_flat.astype(np.float32)).to(dev)
    return _softplus_inv(gt_t) if t == "softplus" else gt_t.clone()


def pad_sh(arr, n_sh):
    """(9,3) GT coefficients -> (n_sh,3), zero-padding band 3 if needed.

    Was a closure inside _optimize_ct_sh, which made it invisible to the other
    models -- _optimize_phong_sh referenced it and raised NameError whenever
    gt_sh_coeffs was supplied. Module-level so every model can use it.
    """
    arr = np.asarray(arr, np.float32)
    if arr.shape[0] < n_sh:
        arr = np.concatenate([arr, np.zeros((n_sh - arr.shape[0], 3), np.float32)])
    return arr
