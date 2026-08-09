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


def ggx_sh_bands_analytic(
    roughness: torch.Tensor,
    n_bands:   int = 3,
) -> torch.Tensor:
    """Closed-form GGX zonal-SH band weights h_0..h_{n_bands-1}. (...,) -> (..., n_bands).

    A drop-in, LUT-free replacement for `_lut_lookup(_compute_ggx_sh_lut(...), r)`. It
    evaluates the SAME integral

        h_l = 2π ∫₀^{π/2} D_GGX(cos(θ/2), α) P_l(cosθ) sinθ dθ,   α = roughness²,

    in CLOSED FORM instead of interpolating a uniform-in-roughness table of numerically
    integrated knots. The table is under-resolved where it matters most: the GGX lobe has
    width ~α = r², so the 512-knot roughness grid (and the uniform-θ trapezoid that built
    each knot) both miss the lobe below r ~ 0.08 — a knot at r = 0.002 evaluates to ~0.02
    instead of the correct ~4, a >100x cliff between adjacent knots that the lerp then
    blends through. The lerp also makes h piecewise-LINEAR, so dh/dr is a staircase in the
    very variable being optimized. The closed form has neither defect (verified to 1.3e-4
    of a 2^20-sample GGX-importance-sampled reference across r ∈ [0, 1], all four bands).

    Substituting u = cosθ turns each band into  h ∝ ∫₀¹ P_l(u)/(b + c u)² du  with b = 1+e,
    c = e-1, e = α². Those integrals have elementary closed forms; band 0 is stable
    everywhere (h_0 = 4/(1+e) → 2 at r=1, → 4 at r=0), while bands 1..3 hit a removable 0/0
    as c → 0 (r → 1) and are switched to their Taylor series in x = 1-e for x < 1e-2.

    FLOAT64 IS DELIBERATE. The closed form is not fp32-safe near r = 1 (catastrophic
    cancellation in the 1/c^k terms) nor near r = 0 (1/e in the r→0 branch). Evaluating in
    float64 and casting back — including through the backward pass — fixes both at
    negligible cost (h_l is (..., n_bands); the render einsums are far larger).
    """
    r = roughness.double()
    e = (r * r) * (r * r)                       # α² = r⁴
    x = 1.0 - e                                 # → 0 as r → 1
    lo = e < 1e-12                              # r → 0: delta limit, all bands → 4
    hi = x < 1e-2                               # r → 1: closed form is 0/0, use series
    # Clamp e so the dead branches of the torch.where below never divide by zero — a
    # nan/inf in an unused branch survives the backward pass and poisons the gradient.
    ec = e.clamp(min=1e-12, max=1.0 - 1e-2)
    b, c = 1.0 + ec, ec - 1.0
    bc = b + c                                  # 2·ec
    Lg = torch.log(bc) - torch.log(b)           # ln(2e/(1+e))
    J0 = 1.0 / (b * bc)
    J1 = (Lg - c / bc) / (c * c)
    four = torch.full_like(e, 4.0)

    bands = [torch.where(lo, four, 4.0 / (1.0 + e))]                  # h_0 (exact, stable)
    if n_bands >= 2:
        h1c = 8.0 * ec * J1
        h1s = 1.0 + x * (2.0 / 3 + x * (11.0 / 24 + x * (13.0 / 40 + x * 19.0 / 80)))
        bands.append(torch.where(lo, four, torch.where(hi, h1s, h1c)))  # h_1
    if n_bands >= 3:
        J2 = (1.0 - (2.0 * b / c) * Lg + b / bc) / (c * c)
        h2c = 4.0 * ec * (3.0 * J2 - J0)
        h2s = x * (1.0 / 4 + x * (13.0 / 40 + x * (13.0 / 40 + x * 83.0 / 280)))
        bands.append(torch.where(lo, four, torch.where(hi, h2s, h2c)))  # h_2
    if n_bands >= 4:
        J3 = (-2.0 * b + 0.5 * c + (3.0 * b * b / c) * Lg - b * b / bc) / (c ** 3)
        h3c = 4.0 * ec * (5.0 * J3 - 3.0 * J1)
        h3s = -0.25 + x * x * (1.0 / 8 + x * (51.0 / 280 + x * 909.0 / 4480))
        bands.append(torch.where(lo, four, torch.where(hi, h3s, h3c)))  # h_3
    return torch.stack(bands, dim=-1).to(roughness.dtype)


def ggx_sh_bands(
    roughness: torch.Tensor,     # (...,)  perceptual roughness in [0, 1]
    hl_mode:   str = "analytic",
    lut:       Optional[torch.Tensor] = None,
    n_bands:   int = 3,
) -> torch.Tensor:
    """GGX zonal-SH band weights h_l via `hl_mode`, as (..., n_bands).

    "analytic" — closed form (`ggx_sh_bands_analytic`); correct at every roughness. Default.
    "lut"      — the shipped uniform table (`_lut_lookup`); reproduces the previous shading
                 bit-for-bit. Requires a `lut` (fetched from `_get_ggx_sh_lut` if None).
    """
    if hl_mode == "analytic":
        return ggx_sh_bands_analytic(roughness, n_bands=n_bands)
    if hl_mode == "lut":
        if lut is None:
            lut = _get_ggx_sh_lut(roughness.device, n_bands=n_bands)
        return _lut_lookup(lut, roughness)
    raise ValueError(f"unknown hl_mode {hl_mode!r} (expected 'analytic' or 'lut')")


# `_get_ggx_sh_lut` is imported directly by the optimizer and by several notebooks, so
# it is public API in practice; expose it under a public name and keep the old one as
# an alias so existing call sites keep working.
get_ggx_sh_lut = _get_ggx_sh_lut
