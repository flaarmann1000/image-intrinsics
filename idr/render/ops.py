"""Tiny tensor helpers shared by the rasteriser and the shading modules.

These two live apart from both so that `shade_ct` / `shade_phong` do not have to
import `raster` (which imports them back).
"""
from __future__ import annotations

import numpy as np
import torch


def _cuda(x, dev, dtype=torch.float32):
    return torch.from_numpy(np.ascontiguousarray(x)).to(dev, dtype=dtype)


def _norm(x, dim=-1):
    return x / (x.norm(dim=dim, keepdim=True) + 1e-8)
