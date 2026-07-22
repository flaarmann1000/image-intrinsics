"""Spherical-harmonics basis, irradiance and radiance evaluation.

`build_sh_basis` (numpy, used when building env maps) and the torch `_sh_basis`
(used in the shading inner loop) are the same basis in two dtypes; they live
together so the ordering convention is defined in exactly one file.
"""
from __future__ import annotations

import math

import numpy as np
import torch

def build_sh_basis(dirs: np.ndarray, order: int = 2) -> np.ndarray:
    """Real SH basis. dirs: (..., 3) → (..., 9) for order 2, (..., 16) for order 3.
    Band ordering matches raw_renderer_gpu.rasterizer._sh_basis."""
    dirs = np.asarray(dirs, dtype=np.float32)
    x, y, z = dirs[..., 0], dirs[..., 1], dirs[..., 2]
    terms = [
        np.full_like(x, 0.282095),
        0.488603 * y, 0.488603 * z, 0.488603 * x,
        1.092548 * x * y, 1.092548 * y * z,
        0.315392 * (3.0 * z * z - 1.0),
        1.092548 * x * z,
        0.546274 * (x * x - y * y),
    ]
    if order >= 3:
        terms += [
            0.590044 * y * (3.0 * x * x - y * y),
            2.890611 * x * y * z,
            0.457046 * y * (5.0 * z * z - 1.0),
            0.373176 * z * (5.0 * z * z - 3.0),
            0.457046 * x * (5.0 * z * z - 1.0),
            1.445306 * z * (x * x - y * y),
            0.590044 * x * (x * x - 3.0 * y * y),
        ]
    return np.stack(terms, axis=-1).astype(np.float32)


def _sh_basis(d, order: int = 2):
    """d: (..., 3) → (..., 9) for order 2, (..., 16) for order 3.

    Band ordering follows the Ramamoorthi convention used throughout:
    [DC | y,z,x | xy,yz,3z²−1,xz,x²−y² | band-3 m=−3..3].
    """
    x, y, z = d[..., 0], d[..., 1], d[..., 2]
    terms = [
        torch.ones_like(x) * 0.282095,
        0.488603*y, 0.488603*z, 0.488603*x,
        1.092548*x*y, 1.092548*y*z,
        0.315392*(3*z**2 - 1),
        1.092548*x*z, 0.546274*(x**2 - y**2),
    ]
    if order >= 3:
        terms += [
            0.590044 * y * (3*x**2 - y**2),
            2.890611 * x * y * z,
            0.457046 * y * (5*z**2 - 1),
            0.373176 * z * (5*z**2 - 3),
            0.457046 * x * (5*z**2 - 1),
            1.445306 * z * (x**2 - y**2),
            0.590044 * x * (x**2 - 3*y**2),
        ]
    return torch.stack(terms, dim=-1)              # (..., 9|16)


def _sh_order_of(coeffs_t) -> int:
    """Infer the SH order from the coefficient count (9 → 2, 16 → 3)."""
    n = coeffs_t.shape[0]
    if n == 9:
        return 2
    if n == 16:
        return 3
    raise ValueError(f"Expected 9 (order 2) or 16 (order 3) SH coefficients, got {n}")


def _sh_irradiance(coeffs_t, N):
    """ZH band-limiting weights for diffuse — coeffs_t: (9|16,3); N: (...,3) → (...,3).

    Band 3 has A₃ = 0 (odd Lambertian ZH bands above 1 vanish), so order-3
    lighting affects the diffuse term only through its lower bands.
    """
    order = _sh_order_of(coeffs_t)
    Y = _sh_basis(N, order=order)                  # (..., 9|16)
    A_vals = [
        torch.pi,
        2*torch.pi/3, 2*torch.pi/3, 2*torch.pi/3,
        torch.pi/4,   torch.pi/4,   torch.pi/4, torch.pi/4, torch.pi/4,
    ]
    if order >= 3:
        A_vals += [0.0] * 7
    A = N.new_tensor(A_vals)
    return ((A * Y) @ coeffs_t).clamp(min=0)       # (..., 3)


def _sh_phong_filtered_radiance(coeffs_t, dirs, shininess):
    """Phong-lobe SH filter — coeffs_t: (9,3); dirs: (...,3) → (...,3)"""
    Y = _sh_basis(dirs)
    # accept plain float or 0-d tensor as well as per-pixel (..., 1) tensor
    if not torch.is_tensor(shininess):
        shininess = torch.tensor(shininess, dtype=Y.dtype, device=Y.device)
    if shininess.dim() == 0:
        shininess = shininess.unsqueeze(-1)   # → (1,), broadcasts over pixels
    B_0 = 2 * torch.pi / (shininess + 1)
    B_1 = 2 * torch.pi / (shininess + 2)
    B_2 = torch.pi * (3.0 / (shininess + 3) - 1.0 / (shininess + 1))
    norm = (shininess + 2) / (2 * torch.pi)
    B = torch.cat([B_0, B_1, B_1, B_1, B_2, B_2, B_2, B_2, B_2], dim=-1) * norm
    return ((B * Y) @ coeffs_t).clamp(min=0)       # (..., 3)
