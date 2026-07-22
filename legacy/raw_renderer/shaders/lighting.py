"""
Light source representations.

PointLight  — a single point emitter in world space
EnvMap      — equirectangular (lat-long) image-based lighting
SHLighting  — order-2 spherical-harmonics irradiance (9 coeffs, RGB)
              Based on Ramamoorthi & Hanrahan 2001.

Unified interface
-----------------
Every light type exposes:

    samples(frag_pos) -> (dirs, radiance, weights) | None

  dirs:     (N, 3) unit vectors pointing FROM surface TO light
  radiance: (N, 3) incoming radiance per sample
  weights:  (N,)   integration weights (solid angle dω or 1.0)

SHLighting returns None — shaders fall back to light.irradiance(normal).
"""

import numpy as np
from dataclasses import dataclass, field
from PIL import Image


def build_SH_basis(dirs: np.ndarray) -> np.ndarray:
    """
    Evaluate order-2 real SH basis.

    Input:
        dirs: (..., 3)

    Output:
        Y: (..., 9)

    Ordering:
        Y_0^0,
        Y_1^-1, Y_1^0, Y_1^1,
        Y_2^-2, Y_2^-1, Y_2^0, Y_2^1, Y_2^2
    """
    dirs = np.asarray(dirs, dtype=np.float32)

    if dirs.shape[-1] != 3:
        raise ValueError(f"Expected dirs shape (..., 3), got {dirs.shape}")

    x = dirs[..., 0]
    y = dirs[..., 1]
    z = dirs[..., 2]

    return np.stack([
        np.full_like(x, 0.282095),
        0.488603 * y,
        0.488603 * z,
        0.488603 * x,
        1.092548 * x * y,
        1.092548 * y * z,
        0.315392 * (3.0 * z * z - 1.0),
        1.092548 * x * z,
        0.546274 * (x * x - y * y),
    ], axis=-1).astype(np.float32)

# ---------------------------------------------------------------------------
# Point light
# ---------------------------------------------------------------------------


@dataclass
class PointLight:
    position: np.ndarray = field(
        default_factory=lambda: np.array([2, 4, 2], dtype=np.float32))
    color:    np.ndarray = field(
        default_factory=lambda: np.array([1, 1, 1], dtype=np.float32))

    def samples(self, frag_pos: np.ndarray):
        L = self.position - frag_pos
        L = (L / (np.linalg.norm(L) + 1e-8)).astype(np.float32)
        return (L[None], self.color[None].copy(), np.ones(1, dtype=np.float32))


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
        self._precompute()

    def _precompute(self) -> None:
        """
        Build a flat list of directions and solid-angle weights for every pixel,
        used by diffuse_irradiance and cook_torrance_shader for Riemann sums.

        Equirectangular mapping (pixel centres):
            theta[i] = π * (i + 0.5) / H        — polar angle  [0, π]
            phi[j]   = 2π * (j + 0.5) / W - π   — azimuth      [-π, π]
            dω       = sin(theta) * dtheta * dphi
        """
        theta = np.pi * (np.arange(self.H) + 0.5) / self.H   # (H,)
        phi = 2*np.pi * (np.arange(self.W) + 0.5) / self.W - np.pi  # (W,)

        sin_t = np.sin(theta)[:, None]   # (H, 1)
        cos_t = np.cos(theta)[:, None]   # (H, 1)
        cos_p = np.cos(phi)[None, :]     # (1, W)
        sin_p = np.sin(phi)[None, :]     # (1, W)

        dirs_x = sin_t * cos_p                                    # (H, W)
        dirs_y = np.broadcast_to(cos_t, (self.H, self.W)).copy()  # (H, W)
        dirs_z = sin_t * sin_p                                    # (H, W)

        # (P, 3) and (P,) — P = H*W
        self._dirs = np.stack([dirs_x, dirs_y, dirs_z],
                              axis=-1).reshape(-1, 3).astype(np.float32)
        self._solid_angles = (sin_t * (np.pi / self.H) * (2*np.pi / self.W)
                              * np.ones((1, self.W))).reshape(-1).astype(np.float32)
        self._image_flat = self.image.reshape(-1, 3)

    @classmethod
    def from_file(cls, path: str) -> "EnvMap":
        """Load an LDR image and normalise to [0, 1]."""
        img = np.array(Image.open(path).convert(
            "RGB"), dtype=np.float32) / 255.0
        return cls(img)

    @classmethod
    def constant(cls, color: tuple = (0.5, 0.5, 0.5), resolution: int = 128) -> "EnvMap":
        """Uniform-colour env map — useful as a placeholder."""
        img = np.full((resolution // 2, resolution, 3),
                      color, dtype=np.float32)
        return cls(img)

    @classmethod
    def from_sh(cls, sh: "SHLighting", resolution: int = 64) -> "EnvMap":
        """
        Bake an SHLighting into an equirectangular env map.

        Uses the raw SH reconstruction Σ L_lm · Y_lm(d) — no cosine-lobe
        convolution — so point_like() produces the smallest spot that
        order-2 SH can represent.
        """
        H, W = resolution // 2, resolution
        theta = np.pi * (np.arange(H, dtype=np.float32) + 0.5) / H
        phi = 2*np.pi * (np.arange(W, dtype=np.float32) + 0.5) / W - np.pi

        sin_t = np.sin(theta)[:, None]
        x3 = (sin_t * np.cos(phi)[None, :])[...,
                                            # (H, W, 1)
                                            None]
        y3 = np.broadcast_to(
            np.cos(theta)[:, None, None], (H, W, 1)).copy()  # (H, W, 1)
        z3 = (sin_t * np.sin(phi)[None, :])[...,
                                            # (H, W, 1)
                                            None]

        L = sh.coeffs   # (9, 3)
        Y = build_SH_basis(np.concatenate([x3, y3, z3], axis=-1))
        raw = Y @ L

        return cls(np.maximum(raw, 0.0).astype(np.float32))

    @classmethod
    def point_like(
        cls,
        direction: np.ndarray,
        color: tuple = (1.0, 1.0, 1.0),
        resolution: int = 64,
    ) -> "EnvMap":
        """
        Env map with a single pixel lit at `direction` — a perfectly sharp
        directional light with no SH bandlimiting.

        The inverse of _precompute's mapping:
            y = cos(θ)  →  θ = arccos(y)
            x = sin(θ)·cos(φ), z = sin(θ)·sin(φ)  →  φ = arctan2(z, x)
        """
        H, W = resolution // 2, resolution
        d = np.asarray(direction, dtype=np.float32)
        d = d / (np.linalg.norm(d) + 1e-8)
        x, y, z = float(d[0]), float(d[1]), float(d[2])

        theta = float(np.arccos(np.clip(y, -1.0, 1.0)))
        phi = float(np.arctan2(z, x))

        i = int(np.clip(round(theta * H / np.pi - 0.5), 0, H - 1))
        j = int(round((phi + np.pi) * W / (2 * np.pi) - 0.5)) % W

        img = np.zeros((H, W, 3), dtype=np.float32)
        img[i, j] = np.asarray(color, dtype=np.float32)
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
        sky_arr = np.array(sky,    dtype=np.float32)
        ground_arr = np.array(ground, dtype=np.float32)
        # (H, 1, 3) broadcast to (H, W, 3)
        img = (1.0 - v) * sky_arr + v * ground_arr
        img = np.broadcast_to(img, (H, W, 3)).copy()
        return cls(img)

    def sample(self, dirs: np.ndarray) -> np.ndarray:
        """
        Sample radiance for unit direction vectors.
        dirs: (..., 3)  →  returns (..., 3) RGB via nearest-neighbour lookup.
        Equirectangular: phi ∈ [-π, π], theta ∈ [0, π].
        """
        x, y, z = dirs[..., 0], dirs[..., 1], dirs[..., 2]
        phi = np.arctan2(z, x)                             # [-π, π]
        theta = np.arccos(np.clip(y, -1.0, 1.0))            # [0, π]
        u = (phi / (2 * np.pi) + 0.5).clip(0, 1)
        v = (theta / np.pi).clip(0, 1)
        px = (u * (self.W - 1)).astype(int)
        py = (v * (self.H - 1)).astype(int)
        return self.image[py, px]

    def samples(self, _frag_pos: np.ndarray):
        return (self._dirs, self._image_flat, self._solid_angles)

    def diffuse_irradiance(self, normal: np.ndarray) -> np.ndarray:
        """
        Lambertian irradiance E(N) = ∫ L(ω)·max(N·ω, 0) dω

        Computed as a deterministic Riemann sum over all env-map pixels:
            E ≈ Σ_k  L_k · max(N·d_k, 0) · dω_k
        where dω_k = sin(θ_k) · dθ · dφ is the solid angle of pixel k.
        No noise, no sample count to tune.
        """
        cos_theta = self._dirs @ normal                            # (P,)
        weights = np.maximum(cos_theta, 0.0) * self._solid_angles  # (P,)
        return (self._image_flat * weights[:, None]).sum(axis=0)   # (3,)


# ---------------------------------------------------------------------------
# Spherical harmonics lighting
# ---------------------------------------------------------------------------

class SHLighting:
    """
    Order-2 SH irradiance for Lambertian surfaces.
    9 coefficients per RGB channel encode incoming radiance from all directions.
    Irradiance is evaluated analytically at any surface normal in O(1).
    """

    def __init__(self, coeffs: np.ndarray):
        """coeffs: (9, 3) float32 — one SH coefficient vector per RGB channel."""
        if coeffs.shape != (9, 3):
            raise ValueError(
                f"Expected (9, 3) coefficients, got {coeffs.shape}")
        self.coeffs = coeffs.astype(np.float32)

    # --- Constructors -------------------------------------------------------

    @classmethod
    def from_env_map(cls, env: EnvMap, n_samples: int = 50_000) -> "SHLighting":
        """Monte Carlo projection of an env map onto the order-2 SH basis."""
        rng = np.random.default_rng(42)
        phi = rng.uniform(0, 2 * np.pi, n_samples).astype(np.float32)
        cos_t = rng.uniform(-1, 1, n_samples).astype(np.float32)
        sin_t = np.sqrt(1 - cos_t ** 2)
        x = sin_t * np.cos(phi)
        y = cos_t
        z = sin_t * np.sin(phi)
        dirs = np.stack([x, y, z], axis=1)       # (N, 3)
        L = env.sample(dirs)                   # (N, 3)

        # Real SH basis evaluated at each sample direction
        Y = build_SH_basis(dirs)

        # L_lm ≈ (4π/N) Σ_i L(d_i) * Y_lm(d_i)
        coeffs = (4 * np.pi / n_samples) * \
            np.einsum("ni,nj->ij", Y, L)  # (9, 3)
        return cls(coeffs)

    @classmethod
    def white_ambient(cls, intensity: float = 1.0) -> "SHLighting":
        """
        Uniform ambient — irradiance = π·intensity on all surfaces.
        Only L_0^0 is non-zero.
        """
        coeffs = np.zeros((9, 3), dtype=np.float32)
        # c4 * L00 = π·intensity  →  L00 = π·intensity / c4
        # Ramamoorthi & Hanrahan 2001, Table 1
        # _C1, _C2, _C3, _C4, _C5 = 0.429043, 0.511664, 0.743125, 0.886227, 0.247708
        _C4 = 0.886227
        coeffs[0] = float(np.pi * intensity / _C4)
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
        c = np.asarray(color, dtype=np.float32) * intensity

        Y_at_d = build_SH_basis(d)

        # For a delta function: c_lm = ∫ I·δ(ω-d)·Y_lm(ω) dω = I·Y_lm(d)
        # No 4π factor here — that belongs only to the MC estimator in from_env_map.
        coeffs = Y_at_d[:, None] * c[None, :]  # (9, 3)
        return cls(coeffs)

    def samples(self, _frag_pos: np.ndarray):
        return None

    # --- Evaluation ---------------------------------------------------------

    def irradiance(self, normal: np.ndarray) -> np.ndarray:
        """
        Analytically evaluate Lambertian irradiance at unit surface `normal`.
        Returns RGB in [0, ∞).        
        """
        L = self.coeffs
        Y = build_SH_basis(normal)

        A = np.array([
            np.pi,
            2*np.pi/3, 2*np.pi/3, 2*np.pi/3,
            np.pi/4, np.pi/4, np.pi/4, np.pi/4, np.pi/4,
        ], dtype=np.float32)

        return np.maximum((A * Y) @ L, 0.0)

    def phong_filtered_radiance(self, direction: np.ndarray, shininess: float) -> np.ndarray:
        L = self.coeffs
        Y = build_SH_basis(direction)
        B_0 = 2 * np.pi / (shininess + 1)
        B_1 = 2 * np.pi / (shininess + 2)
        B_2 = np.pi * (3/(shininess + 3)-1/(shininess + 1))
        norm = (shininess + 2) / (2 * np.pi)
        B = np.array([B_0,
                      B_1, B_1, B_1,
                      B_2, B_2, B_2, B_2, B_2]) * norm
        return np.maximum((B * Y) @ L, 0.0)
