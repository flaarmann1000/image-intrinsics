"""
Cook-Torrance microfacet BRDF (GGX NDF · Schlick Fresnel · Smith geometry).

Supports all light source types via the unified samples() interface:
  PointLight / EnvMap  — per-sample GGX specular + Lambertian diffuse
  SHLighting           — analytical SH diffuse; specular omitted (too low-freq)

PBRMaterial uses the standard metallic-roughness parameterisation:
  F0    = lerp(0.04, albedo, metallic)   — reflectance at normal incidence
  alpha = roughness²                      — GGX width parameter
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Union

from .lighting import PointLight, EnvMap, SHLighting

Light = Union[PointLight, EnvMap, SHLighting]


# ---------------------------------------------------------------------------
# Material
# ---------------------------------------------------------------------------

@dataclass
class PBRMaterial:
    albedo:    np.ndarray = field(
        default_factory=lambda: np.array([0.8, 0.8, 0.8], dtype=np.float32)
    )
    metallic:  float = 0.0   # 0 = dielectric, 1 = metal
    roughness: float = 0.5   # perceptual roughness; remapped to alpha = roughness²


# ---------------------------------------------------------------------------
# BRDF micro-terms  (vectorised)
# ---------------------------------------------------------------------------

def _f0(albedo: np.ndarray, metallic: float) -> np.ndarray:
    """Fresnel reflectance at normal incidence (metallic-roughness convention)."""
    return 0.04 * (1.0 - metallic) + albedo * metallic


def _D_v(NdH: np.ndarray, alpha2: float) -> np.ndarray:
    """Vectorised GGX NDF. NdH: (N,) → (N,)."""
    d = (NdH ** 2 * (alpha2 - 1.0) + 1.0) ** 2
    return alpha2 / (np.pi * d + 1e-7)


def _F_v(VdH: np.ndarray, F0: np.ndarray) -> np.ndarray:
    """Vectorised Schlick Fresnel. VdH: (N,), F0: (3,) → (N, 3)."""
    return F0[None] + (1.0 - F0[None]) * (1.0 - VdH[:, None]) ** 5


def _G_v(NdV: float, NdL: np.ndarray, k: float) -> np.ndarray:
    """Vectorised Smith geometry. NdL: (N,) → (N,)."""
    g_v = NdV / (NdV * (1.0 - k) + k + 1e-7)
    g_l = NdL / (NdL * (1.0 - k) + k + 1e-7)
    return g_v * g_l


def _F_scalar(VdH: float, F0: np.ndarray) -> np.ndarray:
    """Scalar Schlick Fresnel (used for SH path). Returns (3,)."""
    return F0 + (1.0 - F0) * (1.0 - VdH) ** 5


# ---------------------------------------------------------------------------
# Cook-Torrance shader
# ---------------------------------------------------------------------------

def cook_torrance_shader(
    frag_pos: np.ndarray,   # (3,) world-space fragment position
    normal:   np.ndarray,   # (3,) unit surface normal
    cam_pos:  np.ndarray,   # (3,) camera/eye world position
    mat:      PBRMaterial,
    light:    Light,
) -> np.ndarray:
    """
    Cook-Torrance (GGX / Schlick / Smith) microfacet BRDF.
    Returns RGB in [0, 1].

    For sampled lights (PointLight, EnvMap):
        spec = Σ D·F·G / (4·NdV) · radiance · dω
        diff = k_d · albedo/π · Σ radiance · NdL · dω

        Smith k: (roughness+1)²/8 for single-sample (punctual), α²/2 for IBL.

    For SHLighting (samples() returns None):
        diff = k_d · albedo/π · irradiance(N)
        spec = 0
    """
    N   = normal / (np.linalg.norm(normal) + 1e-8)
    V   = cam_pos - frag_pos;  V /= np.linalg.norm(V) + 1e-8
    NdV = max(float(np.dot(N, V)), 1e-4)
    F0  = _f0(mat.albedo, mat.metallic)
    alpha  = mat.roughness ** 2
    alpha2 = alpha ** 2

    samps = light.samples(frag_pos)
    if samps is not None:
        dirs, rad, dw = samps

        # Smith k: IBL variant for area lights, direct variant for punctual
        k = alpha2 / 2.0 if len(dw) > 1 else (mat.roughness + 1) ** 2 / 8

        NdL_all = dirs @ N
        valid   = NdL_all > 1e-4
        dirs_v  = dirs[valid]             # (V, 3)
        NdL_v   = NdL_all[valid]          # (V,)
        dw_v    = dw[valid]               # (V,)
        rad_v   = rad[valid]              # (V, 3)

        if not np.any(valid):
            return np.zeros(3, dtype=np.float32)

        HL    = dirs_v + V[None]
        H_v   = HL / (np.linalg.norm(HL, axis=1, keepdims=True) + 1e-8)
        NdH_v = np.clip(H_v @ N, 0.0, 1.0)    # (V,)
        VdH_v = np.clip(H_v @ V, 0.0, 1.0)    # (V,)

        D_val = _D_v(NdH_v, alpha2)            # (V,)
        F_val = _F_v(VdH_v, F0)               # (V, 3)
        G_val = _G_v(NdV, NdL_v, k)           # (V,)

        # Rendering equation: Σ f_r · L_i · NdL · dω
        # BRDF denominator 4·NdV·NdL cancels with NdL in numerator
        weight = D_val * G_val * dw_v / (4.0 * NdV + 1e-7)   # (V,)
        spec   = (F_val * weight[:, None] * rad_v).sum(0)     # (3,)

        diff_irr = (rad_v * NdL_v[:, None] * dw_v[:, None]).sum(0)   # (3,)
        k_d      = (1.0 - F_val.mean(0)) * (1.0 - mat.metallic)
        diff     = k_d * mat.albedo / np.pi * diff_irr

    else:
        irr    = light.irradiance(N)  # type: ignore[union-attr]  — only SHLighting returns None
        F_approx = _F_scalar(NdV, F0)
        k_d    = (1.0 - F_approx) * (1.0 - mat.metallic)
        diff   = k_d * mat.albedo / np.pi * irr
        spec   = np.zeros(3, dtype=np.float32)

    return np.clip(diff + spec, 0.0, 1.0)
