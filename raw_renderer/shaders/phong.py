"""
Phong illumination model.

Supports three light source types:
  PointLight  — classic per-fragment diffuse + specular
  EnvMap      — diffuse from hemisphere MC integral, specular from reflection ray
  SHLighting  — diffuse from SH irradiance (no high-freq specular)
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

    I = ambient + diffuse + specular

    ambient  = ka * light_color * base_color
    diffuse  = kd * max(N·L, 0) * light_color * base_color
    specular = ks * max(R·V, 0)^shininess * light_color
    """
    N = normal / (np.linalg.norm(normal) + 1e-8)
    V = cam_pos - frag_pos;  V /= np.linalg.norm(V) + 1e-8

    if isinstance(light, PointLight):
        L = light.position - frag_pos;  L /= np.linalg.norm(L) + 1e-8
        R = 2 * np.dot(N, L) * N - L

        ambient  = mat.ka * light.color * mat.base_color
        diffuse  = mat.kd * max(float(np.dot(N, L)), 0.0) * light.color * mat.base_color
        specular = mat.ks * max(float(np.dot(R, V)), 0.0) ** mat.shininess * light.color

    elif isinstance(light, EnvMap):
        irr     = light.diffuse_irradiance(N)
        R_dir   = 2 * np.dot(N, V) * N - V           # perfect mirror reflection of V
        spec_c  = light.sample(R_dir[None, :])[0]     # look up env at reflection direction

        ambient  = np.zeros(3, dtype=np.float32)
        diffuse  = mat.kd * irr * mat.base_color / np.pi
        specular = mat.ks * spec_c

    else:  # SHLighting — analytical diffuse, no high-frequency specular
        irr = light.irradiance(N)
        ambient  = np.zeros(3, dtype=np.float32)
        diffuse  = mat.kd * irr * mat.base_color / np.pi
        specular = np.zeros(3, dtype=np.float32)

    return np.clip(ambient + diffuse + specular, 0.0, 1.0)
