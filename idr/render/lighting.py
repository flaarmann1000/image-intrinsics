"""EnvMap and SHLighting — the light representations the shaders consume."""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from PIL import Image
from typing import Optional, Union

from .sh import build_sh_basis

class EnvMap:
    """Equirectangular (lat-long) environment map."""

    def __init__(self, image: np.ndarray):
        self.image = image.astype(np.float32)
        self.H, self.W = self.image.shape[:2]
        self._precompute()

    def _precompute(self) -> None:
        theta = np.pi * (np.arange(self.H) + 0.5) / self.H
        phi   = 2 * np.pi * (np.arange(self.W) + 0.5) / self.W - np.pi
        sin_t = np.sin(theta)[:, None]
        cos_t = np.cos(theta)[:, None]
        cos_p = np.cos(phi)[None, :]
        sin_p = np.sin(phi)[None, :]
        self._dirs = np.stack([
            (sin_t * cos_p).reshape(-1),
            np.broadcast_to(cos_t, (self.H, self.W)).reshape(-1),
            (sin_t * sin_p).reshape(-1),
        ], axis=-1).astype(np.float32)
        self._solid_angles = (
            sin_t * (np.pi / self.H) * (2 * np.pi / self.W) * np.ones((1, self.W))
        ).reshape(-1).astype(np.float32)
        self._image_flat = self.image.reshape(-1, 3)

    @classmethod
    def from_file(cls, path: str) -> "EnvMap":
        img = np.array(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
        return cls(img)

    @classmethod
    def constant(cls, color: tuple = (0.5, 0.5, 0.5), resolution: int = 128) -> "EnvMap":
        img = np.full((resolution // 2, resolution, 3), color, dtype=np.float32)
        return cls(img)

    @staticmethod
    def _sh_grid_dirs(resolution: int) -> np.ndarray:
        H, W = resolution // 2, resolution
        theta = np.pi * (np.arange(H, dtype=np.float32) + 0.5) / H
        phi   = 2 * np.pi * (np.arange(W, dtype=np.float32) + 0.5) / W - np.pi
        sin_t = np.sin(theta)[:, None]
        return np.stack([
            sin_t * np.cos(phi)[None, :],
            np.broadcast_to(np.cos(theta)[:, None], (H, W)).copy(),
            sin_t * np.sin(phi)[None, :],
        ], axis=-1)                                          # (H, W, 3)

    @classmethod
    def from_sh(cls, sh: "SHLighting", resolution: int = 64) -> "EnvMap":
        raw = build_sh_basis(cls._sh_grid_dirs(resolution)) @ sh.coeffs  # (H, W, 3)
        return cls(np.maximum(raw, 0.0).astype(np.float32))

    @classmethod
    def from_sh_dc_lifted(cls, sh: "SHLighting", resolution: int = 64,
                          eps: float = 1e-4) -> tuple:
        """SH → non-negative equirect env map WITHOUT rectification: instead of
        clipping negative lobes (max(·,0), which injects >order-2 content the
        SH lighting model cannot represent), the DC coefficient is lifted per
        channel until the whole map is ≥ eps. The result stays EXACTLY
        order-2, so the returned lifted coefficients are the true GT SH of
        the map.

        Returns (EnvMap, lifted_coeffs (9,3)). The minimum is taken on a dense
        512-wide grid so coarser render grids are safely non-negative.
        """
        coeffs = sh.coeffs.astype(np.float32).copy()
        dense = build_sh_basis(cls._sh_grid_dirs(512)) @ coeffs
        lift = np.maximum(0.0, eps - dense.min(axis=(0, 1))) / 0.282095  # per-channel DC lift
        coeffs[0] += lift.astype(np.float32)
        img = (build_sh_basis(cls._sh_grid_dirs(resolution)) @ coeffs).astype(np.float32)
        return cls(img), coeffs

    @classmethod
    def point_like(cls, direction: np.ndarray, color: tuple = (1.0, 1.0, 1.0),
                   resolution: int = 64) -> "EnvMap":
        H, W = resolution // 2, resolution
        d = np.asarray(direction, dtype=np.float32)
        d /= np.linalg.norm(d) + 1e-8
        theta = float(np.arccos(np.clip(d[1], -1.0, 1.0)))
        phi   = float(np.arctan2(d[2], d[0]))
        i = int(np.clip(round(theta * H / np.pi - 0.5), 0, H - 1))
        j = int(round((phi + np.pi) * W / (2 * np.pi) - 0.5)) % W
        img = np.zeros((H, W, 3), dtype=np.float32)
        img[i, j] = np.asarray(color, dtype=np.float32)
        return cls(img)

    def sample(self, dirs: np.ndarray) -> np.ndarray:
        x, y, z = dirs[..., 0], dirs[..., 1], dirs[..., 2]
        u = (np.arctan2(z, x) / (2 * np.pi) + 0.5).clip(0, 1)
        v = (np.arccos(np.clip(y, -1.0, 1.0)) / np.pi).clip(0, 1)
        px = (u * (self.W - 1)).astype(int)
        py = (v * (self.H - 1)).astype(int)
        return self.image[py, px]

    def diffuse_irradiance(self, normal: np.ndarray) -> np.ndarray:
        cos_theta = self._dirs @ normal
        weights = np.maximum(cos_theta, 0.0) * self._solid_angles
        return (self._image_flat * weights[:, None]).sum(axis=0)


class SHLighting:
    """Order-2 SH irradiance (9 coefficients × 3 RGB channels)."""

    def __init__(self, coeffs: np.ndarray):
        if coeffs.shape != (9, 3):
            raise ValueError(f"Expected (9, 3) coefficients, got {coeffs.shape}")
        self.coeffs = coeffs.astype(np.float32)

    @classmethod
    def from_env_map(cls, env: EnvMap, n_samples: int = 50_000) -> "SHLighting":
        rng = np.random.default_rng(42)
        phi   = rng.uniform(0, 2 * np.pi, n_samples).astype(np.float32)
        cos_t = rng.uniform(-1, 1, n_samples).astype(np.float32)
        sin_t = np.sqrt(1 - cos_t ** 2)
        dirs  = np.stack([sin_t * np.cos(phi), cos_t, sin_t * np.sin(phi)], axis=1)
        Y     = build_sh_basis(dirs)
        L     = env.sample(dirs)
        coeffs = (4 * np.pi / n_samples) * np.einsum("ni,nj->ij", Y, L)
        return cls(coeffs)

    @classmethod
    def white_ambient(cls, intensity: float = 1.0) -> "SHLighting":
        coeffs = np.zeros((9, 3), dtype=np.float32)
        coeffs[0] = float(np.pi * intensity / 0.886227)
        return cls(coeffs)

    @classmethod
    def directional(cls, direction: np.ndarray, color: np.ndarray,
                    intensity: float = 2.0) -> "SHLighting":
        d = np.asarray(direction, dtype=np.float32)
        d /= np.linalg.norm(d)
        c = np.asarray(color, dtype=np.float32) * intensity
        coeffs = build_sh_basis(d)[:, None] * c[None, :]    # (9, 3)
        return cls(coeffs)

    def irradiance(self, normal: np.ndarray) -> np.ndarray:
        A = np.array([
            np.pi,
            2*np.pi/3, 2*np.pi/3, 2*np.pi/3,
            np.pi/4, np.pi/4, np.pi/4, np.pi/4, np.pi/4,
        ], dtype=np.float32)
        return np.maximum((A * build_sh_basis(normal)) @ self.coeffs, 0.0)
