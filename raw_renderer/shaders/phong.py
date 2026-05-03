"""
Phong illumination model.

Supports all light source types via the unified samples() interface:
  PointLight / EnvMap  — per-sample integration over (dirs, radiance, weights)
  SHLighting           — analytical irradiance; specular omitted (too low-freq)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Union

from .lighting import PointLight, EnvMap, SHLighting

Light = Union[PointLight, EnvMap, SHLighting]


@dataclass
class PhongMaterial:
    base_color: np.ndarray = field(
        default_factory=lambda: np.array([0.7, 0.7, 0.7], dtype=np.float32)
    )
    ka:        float = 0.05   # ambient weight
    kd:        float = 0.80   # diffuse weight
    ks:        float = 0.30   # specular weight
    shininess: float = 32.0


def phong_shader(
    frag_pos:  np.ndarray,   # (3,) world-space fragment position
    normal:    np.ndarray,   # (3,) unit surface normal
    cam_pos:   np.ndarray,   # (3,) camera/eye world position
    mat:       PhongMaterial,
    light:     Light,
) -> np.ndarray:
    """
    Compute RGB colour in [0, 1] using the Phong model.

    For sampled lights (PointLight, EnvMap):
        diffuse  = kd * Σ (N·L) * radiance * base_color/π * dω
        specular = ks * Σ (R·V)^shininess * radiance * dω
        ambient  = ka * mean(radiance) * base_color

    For SHLighting (samples() returns None):
        diffuse  = kd * irradiance(N) * base_color/π
        specular = 0  (SH too low-frequency for glossy highlights)
    """
    N = normal / (np.linalg.norm(normal) + 1e-8)
    # view direction from cam to fragment in world space
    V = cam_pos - frag_pos
    V /= np.linalg.norm(V) + 1e-8

    samps = light.samples(frag_pos)
    if samps is not None:
        dirs, rad, dw = samps
        # hemisphere above face normal
        valid = (dirs @ N) > 1e-4
        dirs_v = dirs[valid]                       # (V, 3)
        rad_v = rad[valid]                        # (V, 3)
        dw_v = dw[valid]                         # (V,)
        NdL_v = dirs_v @ N                        # (V,)

        if len(dirs_v) == 0:
            return np.zeros(3, dtype=np.float32)

        diffuse = mat.kd * (rad_v * (NdL_v * dw_v)
                            [:, None]).sum(0) * mat.base_color / np.pi

        R_v = 2 * NdL_v[:, None] * N[None] - \
            dirs_v        # (V, 3) reflected light dirs
        RdV = np.clip(R_v @ V, 0.0, 1.0)                    # (V,)
        spec_sum = ((RdV ** mat.shininess)
                    [:, None] * rad_v * dw_v[:, None]).sum(0)
        # Riemann sum Σ (R·V)^s dω integrates to 2π/(s+2) over the hemisphere.
        # Normalize area lights so their peak matches the point-light convention (dw=1).
        if len(dw) > 1:
            spec_sum *= (mat.shininess + 2) / (2 * np.pi)
        specular = mat.ks * spec_sum

        ambient = mat.ka * rad.mean(0) * mat.base_color
    else:
        # type: ignore[union-attr]  — only SHLighting returns None from samples()
        irr = light.irradiance(N)
        diffuse = mat.kd * irr * mat.base_color / np.pi
        # specular = np.zeros(3, dtype=np.float32)
        NdV = np.dot(N, V)

        if NdV <= 0.0:
            specular = np.zeros(3, dtype=np.float32)

        else:
            R = 2 * NdV * N - V  # reflection of view about normal
            R /= np.linalg.norm(R) + 1e-8

            # R is the direction a perfect mirror would sample —
            # look up how much radiance comes from there
            L_R = np.maximum(light.radiance(R), 0.0)

            # (R·V) is always 1.0 at the specular peak by construction,
            # so use the radiance magnitude to modulate, but still need
            # a directional falloff term. Extract dominant light dir from SH:
            # L_band1 coeffs (indices 1,2,3) encode the mean light direction.
            dominant_L = np.array([
                light.coeffs[3, :].mean(),  # x  (L[3] = 0.488603 * x)
                light.coeffs[1, :].mean(),  # y
                light.coeffs[2, :].mean(),  # z
            ])
            dom_norm = np.linalg.norm(dominant_L)
            if dom_norm > 1e-6:
                dominant_L /= dom_norm
                NdL = np.clip(dominant_L @ N, 0.0, 1.0)
                # standard Phong: reflect L about N, dot with V
                RdV = np.clip(R @ dominant_L, 0.0, 1.0)
                # but R is already refl(V), so RdV = dot(refl(V), L) = dot(V, refl(L)) ✓
                specular = mat.ks * (RdV ** mat.shininess) * L_R * NdL
            else:
                specular = np.zeros(3, dtype=np.float32)

        ambient = np.zeros(3, dtype=np.float32)

    return np.clip(ambient + diffuse + specular, 0.0, 1.0)
