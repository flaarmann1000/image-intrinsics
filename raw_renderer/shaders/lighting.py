"""
Light source representations.

PointLight  — a single point emitter in world space
EnvMap      — equirectangular (lat-long) image-based lighting
SHLighting  — order-2 spherical-harmonics irradiance (9 coeffs, RGB)
              Based on Ramamoorthi & Hanrahan 2001.
"""

import numpy as np
from dataclasses import dataclass, field
from PIL import Image


# ---------------------------------------------------------------------------
# Point light
# ---------------------------------------------------------------------------

@dataclass
class PointLight:
    position: np.ndarray = field(default_factory=lambda: np.array([2, 4, 2], dtype=np.float32))
    color:    np.ndarray = field(default_factory=lambda: np.array([1, 1, 1], dtype=np.float32))


# ---------------------------------------------------------------------------
# Environment map
# ---------------------------------------------------------------------------

class EnvMap:
    """
    Equirectangular (lat-long) environment map.

    Load from file with EnvMap.from_file(path) or create a solid-colour
    placeholder with EnvMap.constant(color).
    """

    def __init__(self, image: np.ndarray):
        """image: (H, W, 3) float32, linear-light values in [0, ∞)."""
        self.image = image.astype(np.float32)
        self.H, self.W = self.image.shape[:2]

    @classmethod
    def from_file(cls, path: str) -> "EnvMap":
        """Load an LDR image and normalise to [0, 1]."""
        img = np.array(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
        return cls(img)

    @classmethod
    def constant(cls, color: tuple = (0.5, 0.5, 0.5), resolution: int = 128) -> "EnvMap":
        """Uniform-colour env map — useful as a placeholder."""
        img = np.full((resolution // 2, resolution, 3), color, dtype=np.float32)
        return cls(img)

    @classmethod
    def sky_ground(
        cls,
        sky:    tuple = (0.4, 0.6, 1.0),
        ground: tuple = (0.2, 0.15, 0.1),
        resolution: int = 128,
    ) -> "EnvMap":
        """
        Simple gradient env map: sky colour at the top hemisphere, ground colour
        at the bottom. Faces with different normals pick up noticeably different
        irradiance, so shape reads clearly without a real HDR file.
        """
        H, W = resolution // 2, resolution
        # v ∈ [0,1] maps theta ∈ [0, π]: 0 = straight up, 1 = straight down
        v = np.linspace(0.0, 1.0, H, dtype=np.float32)[:, None, None]
        sky_arr    = np.array(sky,    dtype=np.float32)
        ground_arr = np.array(ground, dtype=np.float32)
        img = (1.0 - v) * sky_arr + v * ground_arr   # (H, 1, 3) broadcast to (H, W, 3)
        img = np.broadcast_to(img, (H, W, 3)).copy()
        return cls(img)

    def sample(self, dirs: np.ndarray) -> np.ndarray:
        """
        Sample radiance for unit direction vectors.
        dirs: (..., 3)  →  returns (..., 3) RGB via nearest-neighbour lookup.
        Equirectangular: phi ∈ [-π, π], theta ∈ [0, π].
        """
        x, y, z = dirs[..., 0], dirs[..., 1], dirs[..., 2]
        phi   = np.arctan2(x, z)                             # [-π, π]
        theta = np.arccos(np.clip(y, -1.0, 1.0))            # [0, π]
        u = (phi / (2 * np.pi) + 0.5).clip(0, 1)
        v = (theta / np.pi).clip(0, 1)
        px = (u * (self.W - 1)).astype(int)
        py = (v * (self.H - 1)).astype(int)
        return self.image[py, px]

    def diffuse_irradiance(self, normal: np.ndarray, n_samples: int = 128) -> np.ndarray:
        """
        Cosine-weighted hemisphere Monte Carlo integral of incoming radiance
        (the Lambertian irradiance) at surface normal `normal`.
        Returns RGB in [0, ∞).
        """
        n = normal / (np.linalg.norm(normal) + 1e-8)
        up = np.array([0, 1, 0], dtype=np.float32)
        if abs(float(np.dot(n, up))) > 0.99:
            up = np.array([1, 0, 0], dtype=np.float32)
        t = np.cross(up, n); t /= np.linalg.norm(t)
        b = np.cross(n, t)

        rng = np.random.default_rng(0)
        # Cosine-weighted sampling: pdf = cos_theta / π
        cos_t = np.sqrt(rng.random(n_samples)).astype(np.float32)
        phi   = rng.uniform(0, 2 * np.pi, n_samples).astype(np.float32)
        sin_t = np.sqrt(1 - cos_t ** 2)
        lx, ly, lz = sin_t * np.cos(phi), cos_t, sin_t * np.sin(phi)
        dirs = lx[:, None] * t + ly[:, None] * n + lz[:, None] * b  # (N, 3)
        # E[f/pdf] = E[π * radiance] — cosine and pdf cancel
        return np.pi * self.sample(dirs).mean(axis=0)


# ---------------------------------------------------------------------------
# Spherical harmonics lighting
# ---------------------------------------------------------------------------

class SHLighting:
    """
    Order-2 SH irradiance for Lambertian surfaces.
    9 coefficients per RGB channel encode incoming radiance from all directions.
    Irradiance is evaluated analytically at any surface normal in O(1).
    """

    # Ramamoorthi & Hanrahan 2001, Table 1
    _C1, _C2, _C3, _C4, _C5 = 0.429043, 0.511664, 0.743125, 0.886227, 0.247708

    # Real SH basis weights at direction (x, y, z)
    # Ordering: Y_0^0, Y_1^-1, Y_1^0, Y_1^1, Y_2^-2, Y_2^-1, Y_2^0, Y_2^1, Y_2^2
    _BASIS_SCALE = np.array(
        [0.282095, 0.488603, 0.488603, 0.488603,
         1.092548, 1.092548, 0.315392, 1.092548, 0.546274],
        dtype=np.float32,
    )

    def __init__(self, coeffs: np.ndarray):
        """coeffs: (9, 3) float32 — one SH coefficient vector per RGB channel."""
        if coeffs.shape != (9, 3):
            raise ValueError(f"Expected (9, 3) coefficients, got {coeffs.shape}")
        self.coeffs = coeffs.astype(np.float32)

    # --- Constructors -------------------------------------------------------

    @classmethod
    def from_env_map(cls, env: EnvMap, n_samples: int = 50_000) -> "SHLighting":
        """Monte Carlo projection of an env map onto the order-2 SH basis."""
        rng = np.random.default_rng(42)
        phi   = rng.uniform(0, 2 * np.pi, n_samples).astype(np.float32)
        cos_t = rng.uniform(-1, 1, n_samples).astype(np.float32)
        sin_t = np.sqrt(1 - cos_t ** 2)
        x = sin_t * np.cos(phi)
        y = cos_t
        z = sin_t * np.sin(phi)
        dirs = np.stack([x, y, z], axis=1)       # (N, 3)
        L    = env.sample(dirs)                   # (N, 3)

        # Real SH basis evaluated at each sample direction
        Y = np.stack([
            np.ones(n_samples) * 0.282095,   # Y_0^0
            0.488603 * y,                     # Y_1^-1
            0.488603 * z,                     # Y_1^0
            0.488603 * x,                     # Y_1^1
            1.092548 * x * y,                # Y_2^-2
            1.092548 * y * z,                # Y_2^-1
            0.315392 * (3 * z ** 2 - 1),     # Y_2^0
            1.092548 * x * z,                # Y_2^1
            0.546274 * (x ** 2 - y ** 2),    # Y_2^2
        ], axis=1)  # (N, 9)

        # L_lm ≈ (4π/N) Σ_i L(d_i) * Y_lm(d_i)
        coeffs = (4 * np.pi / n_samples) * np.einsum("ni,nj->ij", Y, L)  # (9, 3)
        return cls(coeffs)

    @classmethod
    def white_ambient(cls, intensity: float = 1.0) -> "SHLighting":
        """
        Uniform ambient — irradiance = π·intensity on all surfaces.
        Only L_0^0 is non-zero.
        """
        coeffs = np.zeros((9, 3), dtype=np.float32)
        # c4 * L00 = π·intensity  →  L00 = π·intensity / c4
        coeffs[0] = float(np.pi * intensity / cls._C4)
        return cls(coeffs)

    @classmethod
    def directional(
        cls,
        direction: np.ndarray,
        color: np.ndarray,
        intensity: float = 2.0,
    ) -> "SHLighting":
        """
        Approximate a directional light as an SH-projected delta function.
        `direction` points FROM the light TO the scene (i.e. the light direction
        is -direction when computing N·L, but we project the emission here).
        """
        d = np.asarray(direction, dtype=np.float32)
        d /= np.linalg.norm(d)
        x, y, z = float(d[0]), float(d[1]), float(d[2])
        c = np.asarray(color, dtype=np.float32) * intensity

        Y_at_d = np.array([
            0.282095,
            0.488603 * y,
            0.488603 * z,
            0.488603 * x,
            1.092548 * x * y,
            1.092548 * y * z,
            0.315392 * (3 * z ** 2 - 1),
            1.092548 * x * z,
            0.546274 * (x ** 2 - y ** 2),
        ], dtype=np.float32)

        # For a delta function: c_lm = ∫ I·δ(ω-d)·Y_lm(ω) dω = I·Y_lm(d)
        # No 4π factor here — that belongs only to the MC estimator in from_env_map.
        coeffs = Y_at_d[:, None] * c[None, :]  # (9, 3)
        return cls(coeffs)

    # --- Evaluation ---------------------------------------------------------

    def irradiance(self, normal: np.ndarray) -> np.ndarray:
        """
        Analytically evaluate Lambertian irradiance at unit surface `normal`.
        Returns RGB in [0, ∞).

        Formula: Ramamoorthi & Hanrahan 2001, Eq. 12.
        """
        L = self.coeffs
        x, y, z = float(normal[0]), float(normal[1]), float(normal[2])
        c1, c2, c3, c4, c5 = self._C1, self._C2, self._C3, self._C4, self._C5

        irr = (
            c4 * L[0]
            + 2 * c2 * (L[1] * y  + L[2] * z  + L[3] * x)
            + 2 * c1 * (L[4] * x*y + L[5] * y*z + L[7] * x*z)
            + c3 * L[6] * z ** 2
            - c5 * L[6]
            + c1 * L[8] * (x ** 2 - y ** 2)
        )
        return np.maximum(irr, 0.0)
