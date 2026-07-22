"""The value every optimizer returns, and the env-map sampling grid they consume."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, NamedTuple, Optional

import numpy as np


class EnvGrid(NamedTuple):
    """Sampling grid for explicit env-map lighting.

    `dirs` (P,3) and `dw` (P,) come from an EnvMap's `_dirs` / `_solid_angles`; H and W
    are the map's pixel shape, needed to reshape the flat P back into an image.
    """
    dirs: np.ndarray
    dw: np.ndarray
    H: int
    W: int

    @classmethod
    def from_envmap(cls, env) -> "EnvGrid":
        return cls(env._dirs, env._solid_angles, env.image.shape[0], env.image.shape[1])


@dataclass
class OptimResult:
    """Outcome of one decomposition.

    Replaces the positional 7-tuple every optimizer used to return. The tuple was
    identical in shape across all four models but its *meaning* was not: slot 1 is SH
    coefficients for the sh models and env-map pixels for the env models, and slots 2/3
    are (metallic, roughness) for Cook-Torrance but (shininess, ks) for Phong. Reading
    a call site therefore required knowing which optimizer had produced it. The named
    fields plus `is_env` / `is_phong` make that explicit.
    """
    albedo: np.ndarray
    light: np.ndarray          # (K, n_sh, 3) SH coefficients, or (K, P, 3) env pixels
    mat_a: np.ndarray          # metallic (Cook-Torrance) | shininess (Phong)
    mat_b: np.ndarray          # roughness (Cook-Torrance) | ks       (Phong)
    shadings: Any
    history: Any
    elapsed: float
    shader: str

    @property
    def is_env(self) -> bool:
        """True when `light` holds env-map pixels rather than SH coefficients."""
        return self.shader.endswith("_env")

    @property
    def is_phong(self) -> bool:
        """True when mat_a/mat_b mean (shininess, ks) rather than (metallic, roughness)."""
        return self.shader.startswith("phong")

    @property
    def sh(self) -> Optional[np.ndarray]:
        """The SH coefficients, or None for an env model."""
        return None if self.is_env else self.light

    @property
    def env_maps(self) -> Optional[np.ndarray]:
        """The env-map pixels, or None for an SH model."""
        return self.light if self.is_env else None

    def as_tuple(self):
        """The legacy 7-tuple, for call sites not yet migrated."""
        return (self.albedo, self.light, self.mat_a, self.mat_b,
                self.shadings, self.history, self.elapsed)
