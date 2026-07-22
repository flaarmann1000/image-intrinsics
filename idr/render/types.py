"""Plain data holders shared by the renderer: material, light and scene types.

Split out of rasterizer.py / scene.py so the shading modules can import the types
without pulling in the shading code (and vice versa).
"""
from __future__ import annotations

import numpy as np
import torch
from dataclasses import dataclass, field
from typing import Optional, Union

@dataclass
class SHLight:
    """SH lighting: 9 coefficients per RGB channel."""
    coeffs: torch.Tensor          # (9, 3)


@dataclass
class PBRMat:
    """Cook-Torrance metallic-roughness material."""
    albedo:    torch.Tensor                       # (3,) or (..., 3)
    metallic:  Union[float, torch.Tensor] = 0.0
    roughness: float = 0.5


@dataclass
class PhongMat:
    """Phong material."""
    base_color: torch.Tensor      # (3,)
    ka:         float = 0.05
    kd:         float = 0.80
    ks:         float = 0.30
    shininess:  float = 32.0


@dataclass
class PointLightGPU:
    """Point light source."""
    position: torch.Tensor        # (3,)
    color:    torch.Tensor        # (3,)


@dataclass
class EnvMapLightGPU:
    """Pre-processed environment map (flat samples)."""
    dirs:         torch.Tensor    # (P, 3)
    image_flat:   torch.Tensor    # (P, 3)
    solid_angles: torch.Tensor    # (P,)


@dataclass
class Camera:
    position: np.ndarray = field(default_factory=lambda: np.array([0, 0, 3], dtype=np.float32))
    target:   np.ndarray = field(default_factory=lambda: np.array([0, 0, 0], dtype=np.float32))
    up:       np.ndarray = field(default_factory=lambda: np.array([0, 1, 0], dtype=np.float32))
    fov_deg:  float = 60.0


@dataclass
class Mesh:
    vertices:       np.ndarray   # (V, 3) float32
    faces:          np.ndarray   # (F, 3) int32
    normals:        np.ndarray   # (F, 3) float32 — flat face normals
    vertex_normals: np.ndarray   # (V, 3) float32 — averaged per-vertex normals
