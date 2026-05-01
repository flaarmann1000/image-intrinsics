"""
Cook-Torrance microfacet BRDF (GGX NDF · Schlick Fresnel · Smith geometry).

Supports three light source types:
  PointLight  — analytic single-sample BRDF evaluation
  EnvMap      — GGX importance-sampled IBL (specular) + cosine MC (diffuse)
  SHLighting  — analytical SH diffuse; specular omitted (SH too low-frequency)

PBRMaterial uses the standard metallic-roughness parameterisation:
  F0 = lerp(0.04, albedo, metallic)   — reflectance at normal incidence
  alpha = roughness²                   — GGX width parameter
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Union

from .lighting import PointLight, EnvMap, SHLighting

Light = Union[PointLight, EnvMap, SHLighting]

# Shared RNG for importance sampling — sequential calls give per-fragment variety.
_rng = np.random.default_rng(42)


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
# BRDF micro-terms  (scalar — used for point lights)
# ---------------------------------------------------------------------------

def _f0(albedo: np.ndarray, metallic: float) -> np.ndarray:
    """Fresnel reflectance at normal incidence (metallic-roughness convention)."""
    return 0.04 * (1.0 - metallic) + albedo * metallic


def _D(NdH: float, alpha2: float) -> float:
    """GGX Normal Distribution Function."""
    d = (NdH * NdH * (alpha2 - 1.0) + 1.0) ** 2
    return alpha2 / (np.pi * d + 1e-7)


def _F(VdH: float, F0: np.ndarray) -> np.ndarray:
    """Schlick Fresnel approximation. Returns (3,)."""
    return F0 + (1.0 - F0) * (1.0 - VdH) ** 5


def _G(NdV: float, NdL: float, k: float) -> float:
    """Smith masking-shadowing with GGX Schlick approximation."""
    def g1(x: float) -> float:
        return x / (x * (1.0 - k) + k + 1e-7)
    return g1(max(NdV, 1e-4)) * g1(max(NdL, 1e-4))


# ---------------------------------------------------------------------------
# BRDF micro-terms  (vectorised — used for env-map IBL)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# GGX importance sampling
# ---------------------------------------------------------------------------

def _build_tbn(N: np.ndarray):
    """Return tangent T and bitangent B orthonormal to N."""
    up = np.array([0, 1, 0], dtype=np.float32)
    if abs(float(np.dot(N, up))) > 0.99:
        up = np.array([1, 0, 0], dtype=np.float32)
    T = np.cross(up, N); T /= np.linalg.norm(T)
    B = np.cross(N, T)
    return T, B


def _sample_ggx_halfvectors(N: np.ndarray, n: int, alpha2: float) -> np.ndarray:
    """
    Draw n half-vectors in world space from the GGX distribution around N.
    Returns (n, 3) unit vectors.

    Inversion sampling (Walter et al. 2007):
      cos_theta_m = sqrt((1-xi) / (1 + (alpha²-1)*xi))
    """
    T, B = _build_tbn(N)
    xi1 = _rng.random(n).astype(np.float32)
    xi2 = _rng.uniform(0, 2 * np.pi, n).astype(np.float32)

    cos_t = np.sqrt(np.clip((1 - xi1) / (1 + (alpha2 - 1) * xi1 + 1e-8), 0, 1))
    sin_t = np.sqrt(1 - cos_t ** 2)

    H_local = np.stack([sin_t * np.cos(xi2), sin_t * np.sin(xi2), cos_t], axis=1)
    TBN = np.stack([T, B, N], axis=0)          # (3, 3) — rows are world-space basis
    return H_local @ TBN                        # (n, 3) world-space half-vectors


# ---------------------------------------------------------------------------
# Cook-Torrance shader
# ---------------------------------------------------------------------------

def cook_torrance_shader(
    frag_pos:      np.ndarray,   # (3,) world-space fragment position
    normal:        np.ndarray,   # (3,) unit surface normal
    cam_pos:       np.ndarray,   # (3,) camera/eye world position
    mat:           PBRMaterial,
    light:         Light,
    n_env_samples: int = 32,     # samples for env-map IBL (specular + diffuse)
) -> np.ndarray:
    """
    Cook-Torrance (GGX / Schlick / Smith) microfacet BRDF.
    Returns RGB in [0, 1].

    Point light:
        L_o = (k_d * albedo/π + D·F·G / (4·NdV·NdL)) · L_i · NdL

    Env map:
        specular via GGX importance sampling,
        diffuse  via cosine-weighted MC hemisphere integration.

    SH lighting:
        diffuse  = SH irradiance evaluated analytically at N,
        specular = omitted (SH is too low-frequency for glossy highlights).
    """
    N = normal / (np.linalg.norm(normal) + 1e-8)
    V = cam_pos - frag_pos;  V /= np.linalg.norm(V) + 1e-8
    NdV   = max(float(np.dot(N, V)), 1e-4)
    F0    = _f0(mat.albedo, mat.metallic)
    alpha = mat.roughness ** 2              # GGX α
    alpha2 = alpha ** 2                     # GGX α²
    k     = (mat.roughness + 1) ** 2 / 8   # Smith k (direct-light variant)

    # -------------------------------------------------------------------------
    # Point light — analytic single-sample evaluation
    # -------------------------------------------------------------------------
    if isinstance(light, PointLight):
        L = light.position - frag_pos;  L /= np.linalg.norm(L) + 1e-8
        H = V + L;                       H /= np.linalg.norm(H) + 1e-8

        NdL = max(float(np.dot(N, L)), 0.0)
        NdH = max(float(np.dot(N, H)), 0.0)
        VdH = max(float(np.dot(V, H)), 0.0)

        D_val = _D(NdH, alpha2)
        F_val = _F(VdH, F0)
        G_val = _G(NdV, NdL, k)

        specular = (D_val * F_val * G_val) / (4.0 * NdV * NdL + 1e-7)
        k_d      = (1.0 - F_val) * (1.0 - mat.metallic)
        Lo       = (k_d * mat.albedo / np.pi + specular) * light.color * NdL
        return np.clip(Lo, 0.0, 1.0)

    # -------------------------------------------------------------------------
    # Environment map — GGX importance-sampled IBL
    # -------------------------------------------------------------------------
    elif isinstance(light, EnvMap):
        k_ibl = alpha2 / 2.0               # Smith k for IBL variant

        # --- Specular: importance-sample GGX half-vectors --------------------
        H_world = _sample_ggx_halfvectors(N, n_env_samples, alpha2)     # (n, 3)
        VdH_arr = np.clip((H_world * V[None]).sum(1), 0, 1)             # (n,)
        L_world = 2 * VdH_arr[:, None] * H_world - V[None]             # (n, 3)
        NdL_arr = np.clip((L_world * N[None]).sum(1), 0, 1)             # (n,)
        NdH_arr = np.clip((H_world * N[None]).sum(1), 0, 1)             # (n,)
        valid   = NdL_arr > 1e-4

        radiance = light.sample(L_world)                                 # (n, 3)
        F_arr    = _F_v(VdH_arr, F0)                                    # (n, 3)
        G_arr    = _G_v(NdV, NdL_arr, k_ibl)                           # (n,)

        # Monte Carlo weight: F·G·VdH / (NdH·NdV) — D and pdf cancel
        w      = (G_arr * VdH_arr) / (NdH_arr * NdV + 1e-7)            # (n,)
        n_valid = valid.sum()
        if n_valid > 0:
            spec = (F_arr * w[:, None] * radiance * valid[:, None]).sum(0) / n_valid
        else:
            spec = np.zeros(3, dtype=np.float32)

        # --- Diffuse: cosine MC hemisphere integration ----------------------
        diff_irr = light.diffuse_irradiance(N, n_samples=n_env_samples * 4)
        k_d_avg  = (1.0 - F_arr.mean(0)) * (1.0 - mat.metallic)
        diff     = k_d_avg * mat.albedo / np.pi * diff_irr

        return np.clip(diff + spec, 0.0, 1.0)

    # -------------------------------------------------------------------------
    # SH lighting — analytical diffuse, specular omitted
    # -------------------------------------------------------------------------
    else:  # SHLighting
        irr      = light.irradiance(N)
        F_approx = _F(NdV, F0)
        k_d      = (1.0 - F_approx) * (1.0 - mat.metallic)
        diff     = k_d * mat.albedo / np.pi * irr
        return np.clip(diff, 0.0, 1.0)
