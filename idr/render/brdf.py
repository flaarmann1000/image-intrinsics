"""Cook-Torrance BRDF terms and the GGX-SH lookup table.

The LUT is the precomputed convolution of the GGX lobe against the SH basis; it is
cached per (device, dtype, n_bands) by `get_ggx_sh_lut`.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import numpy as np
import torch

# Precomputed LUTs are cached on disk next to this module. The .npz files moved here
# from raw_renderer_gpu/ along with the code, so `Path(__file__).parent` still
# resolves to the directory that holds them — otherwise the first call would silently
# recompute the 512 x 8192 integration instead of loading it.
_LUT_PATH = Path(__file__).parent / "ggx_sh_lut.npz"
_LUT_N_ROUGH = 512    # roughness resolution (uniform grid over [0, 1])
_LUT_N_THETA = 8192   # integration resolution

_LUT_PATH_O3 = Path(__file__).parent / "ggx_sh_lut_o3.npz"
_LUT_CACHE: dict = {}                       # n_bands -> (N, n_bands) tensor

def _ggx_D(NdH, alpha2):
    d = NdH**2 * (alpha2 - 1.0) + 1.0
    return alpha2 / (torch.pi * d**2 + 1e-7)


def _schlick_F(VdH, F0):
    """VdH: (...); F0: (3,) → (..., 3)"""
    return F0 + (1.0 - F0) * (1.0 - VdH.unsqueeze(-1)) ** 5


def _smith_G(NdV, NdL, k):
    return (NdV/(NdV*(1-k)+k+1e-7)) * (NdL/(NdL*(1-k)+k+1e-7))


def _f0_mat(albedo, metallic):
    return 0.04*(1-metallic) + albedo*metallic


def _compute_ggx_sh_lut(
    n_roughness: int = _LUT_N_ROUGH,
    n_theta:     int = _LUT_N_THETA,
    n_bands:     int = 3,
) -> np.ndarray:
    """
    Numerically integrate GGX zonal SH band weights for bands 0..n_bands-1.

    Kernel:   w(θ) = D_GGX(n·h = cos(θ/2), α)        ← no cos θ factor
    Bands:    h_l = 2π ∫₀^{π/2} w(θ) P_l(cosθ) sinθ dθ
    Stored:   raw h_l (no h_1 normalisation)

    Limit:  α → 0  ⇒  D → δ  ⇒  ∫D dω_l = 4  ⇒  h_l = 4 for all l.
    """

    theta = np.linspace(0.0, np.pi / 2.0, n_theta, dtype=np.float64)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    cos_th2 = np.cos(theta / 2.0)**2

    Pl = [
        np.ones_like(cos_t),
        cos_t,
        0.5 * (3.0 * cos_t**2 - 1.0),
        0.5 * (5.0 * cos_t**3 - 3.0 * cos_t),
    ][:n_bands]

    lut = np.zeros((n_roughness, n_bands), dtype=np.float32)

    for i in range(n_roughness):
        r = float(i) / max(n_roughness - 1, 1)
        a = r ** 2
        a2 = a ** 2

        if a2 < 1e-12:                       # delta limit
            lut[i] = [4.0] * n_bands
            continue

        D = a2 / (np.pi * (cos_th2 * (a2 - 1.0) + 1.0) ** 2)
        # NO cos_t factor in the kernel
        h = np.array([
            2.0 * np.pi * np.trapezoid(D * p * sin_t, theta)
            for p in Pl
        ])
        lut[i] = h                            # raw, unnormalised

    return lut


def _get_ggx_sh_lut(
    device:     torch.device,
    cache_path: Optional[Path] = None,
    n_bands:    int = 3,
) -> torch.Tensor:
    """
    Return the GGX SH LUT on `device`, with `n_bands` zonal bands (3 for
    order-2 SH, 4 for order-3).

    First call:  computes the LUT, saves it to its cache file, caches in memory.
    Later calls: returns the in-memory tensor (moved to `device` if needed).
    """
    if n_bands not in _LUT_CACHE:
        path = cache_path if cache_path is not None else (
            _LUT_PATH if n_bands == 3 else _LUT_PATH_O3)
        if path.exists():
            data = np.load(path)
            _LUT_CACHE[n_bands] = torch.from_numpy(data["lut"])
        else:
            print(f"[ggx_sh] {n_bands}-band LUT not found — computing … ", end="", flush=True)
            lut_np = _compute_ggx_sh_lut(n_bands=n_bands)
            np.savez_compressed(path, lut=lut_np)
            _LUT_CACHE[n_bands] = torch.from_numpy(lut_np)
            print(f"done  →  saved to {path}")

    return _LUT_CACHE[n_bands].to(device)


def _lut_lookup(lut: torch.Tensor, roughness: torch.Tensor) -> torch.Tensor:
    """
    Differentiable linear interpolation into a uniform 1-D LUT.

    Parameters
    ----------
    lut       : (N, C)   table indexed uniformly over roughness ∈ [0, 1]
    roughness : (...)    values in [0, 1], any shape

    Returns
    -------
    (..., C)  — gradients flow through `roughness` via the lerp weight `t`.
    """
    N = lut.shape[0]
    idx_f = roughness.clamp(0.0, 1.0) * (N - 1)
    idx_lo = idx_f.long().clamp(0, N - 2)              # integer, no grad needed
    t = (idx_f - idx_lo.float()).unsqueeze(-1)     # (..., 1)  differentiable
    return lut[idx_lo] + t * (lut[idx_lo + 1] - lut[idx_lo])  # (..., C)


# `_get_ggx_sh_lut` is imported directly by the optimizer and by several notebooks, so
# it is public API in practice; expose it under a public name and keep the old one as
# an alias so existing call sites keep working.
get_ggx_sh_lut = _get_ggx_sh_lut
