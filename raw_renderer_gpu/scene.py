"""
Scene primitives for raw_renderer_gpu: Camera, Mesh utilities, and lighting.

These replace the equivalent classes from raw_renderer so callers only need
to import from raw_renderer_gpu.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from PIL import Image


# ─────────────────────────────────────────── Camera ──────────────────────────

@dataclass
class Camera:
    position: np.ndarray = field(default_factory=lambda: np.array([0, 0, 3], dtype=np.float32))
    target:   np.ndarray = field(default_factory=lambda: np.array([0, 0, 0], dtype=np.float32))
    up:       np.ndarray = field(default_factory=lambda: np.array([0, 1, 0], dtype=np.float32))
    fov_deg:  float = 60.0


# ─────────────────────────────────────────── Mesh ────────────────────────────

@dataclass
class Mesh:
    vertices:       np.ndarray   # (V, 3) float32
    faces:          np.ndarray   # (F, 3) int32
    normals:        np.ndarray   # (F, 3) float32 — flat face normals
    vertex_normals: np.ndarray   # (V, 3) float32 — averaged per-vertex normals


def _face_normal(v0: np.ndarray, v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    n = np.cross(v1 - v0, v2 - v0).astype(np.float32)
    length = np.linalg.norm(n)
    return n / length if length > 1e-8 else n


def _vertex_normals(vertices: np.ndarray, faces: np.ndarray, face_normals: np.ndarray) -> np.ndarray:
    vn = np.zeros((len(vertices), 3), dtype=np.float32)
    np.add.at(vn, faces[:, 0], face_normals)
    np.add.at(vn, faces[:, 1], face_normals)
    np.add.at(vn, faces[:, 2], face_normals)
    lengths = np.linalg.norm(vn, axis=1, keepdims=True)
    return vn / np.maximum(lengths, 1e-8)


def _sphere_mesh(n_lat: int, n_lon: int) -> tuple:
    verts = [[0.0, 1.0, 0.0]]
    for i in range(1, n_lat):
        theta = np.pi * i / n_lat
        for j in range(n_lon):
            phi = 2.0 * np.pi * j / n_lon
            verts.append([
                np.sin(theta) * np.cos(phi),
                np.cos(theta),
                np.sin(theta) * np.sin(phi),
            ])
    verts.append([0.0, -1.0, 0.0])
    v = np.array(verts, dtype=np.float32)
    south = len(v) - 1
    faces = []

    def ring_start(i: int) -> int:
        return 1 + (i - 1) * n_lon

    for j in range(n_lon):
        faces.append([0, ring_start(1) + (j + 1) % n_lon, ring_start(1) + j])
    for i in range(1, n_lat - 1):
        rs, rs2 = ring_start(i), ring_start(i + 1)
        for j in range(n_lon):
            nj = (j + 1) % n_lon
            faces.append([rs + j, rs2 + nj, rs2 + j])
            faces.append([rs + j, rs + nj, rs2 + nj])
    rs_last = ring_start(n_lat - 1)
    for j in range(n_lon):
        faces.append([south, rs_last + j, rs_last + (j + 1) % n_lon])

    return v, np.array(faces, dtype=np.int32)


def generate_mesh(shape: str = "cube", **kwargs) -> Mesh:
    if shape == "cube":
        v = np.array([
            [-0.5, -0.5, -0.5], [0.5, -0.5, -0.5],
            [0.5,  0.5, -0.5], [-0.5,  0.5, -0.5],
            [-0.5, -0.5,  0.5], [0.5, -0.5,  0.5],
            [0.5,  0.5,  0.5], [-0.5,  0.5,  0.5],
        ], dtype=np.float32)
        faces = np.array([
            [0, 2, 1], [0, 3, 2],
            [4, 5, 6], [4, 6, 7],
            [0, 1, 5], [0, 5, 4],
            [3, 6, 2], [3, 7, 6],
            [0, 4, 7], [0, 7, 3],
            [1, 2, 6], [1, 6, 5],
        ], dtype=np.int32)
    elif shape == "plane":
        v = np.array([[-1, 0, -1], [1, 0, -1], [1, 0, 1], [-1, 0, 1]], dtype=np.float32)
        faces = np.array([[0, 2, 1], [0, 3, 2]], dtype=np.int32)
    elif shape == "sphere":
        n_lat = int(kwargs.get("n_lat", 16))
        n_lon = int(kwargs.get("n_lon", 32))
        v, faces = _sphere_mesh(n_lat, n_lon)
    else:
        raise ValueError(f"Unknown shape {shape!r}. Choose 'cube', 'plane', or 'sphere'.")

    normals = np.array([_face_normal(v[f[0]], v[f[1]], v[f[2]]) for f in faces], dtype=np.float32)
    return Mesh(vertices=v, faces=faces, normals=normals,
                vertex_normals=_vertex_normals(v, faces, normals))


def load_obj(path: str, normalize: bool = True) -> Mesh:
    verts, faces = [], []
    with open(path, "r") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if parts[0] == "v":
                verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif parts[0] == "f":
                indices = [int(tok.split("/")[0]) for tok in parts[1:]]
                n = len(verts)
                indices = [i - 1 if i > 0 else n + i for i in indices]
                for k in range(1, len(indices) - 1):
                    faces.append([indices[0], indices[k], indices[k + 1]])

    v = np.array(verts, dtype=np.float32)
    if normalize:
        v -= v.mean(axis=0)
        v /= np.linalg.norm(v, axis=1).max() + 1e-8
    f = np.array(faces, dtype=np.int32)
    normals = np.array([_face_normal(v[t[0]], v[t[1]], v[t[2]]) for t in f], dtype=np.float32)
    return Mesh(vertices=v, faces=f, normals=normals,
                vertex_normals=_vertex_normals(v, f, normals))


# ─────────────────────────────────────────── SH helpers ──────────────────────

def build_sh_basis(dirs: np.ndarray, order: int = 2) -> np.ndarray:
    """Real SH basis. dirs: (..., 3) → (..., 9) for order 2, (..., 16) for order 3.
    Band ordering matches raw_renderer_gpu.rasterizer._sh_basis."""
    dirs = np.asarray(dirs, dtype=np.float32)
    x, y, z = dirs[..., 0], dirs[..., 1], dirs[..., 2]
    terms = [
        np.full_like(x, 0.282095),
        0.488603 * y, 0.488603 * z, 0.488603 * x,
        1.092548 * x * y, 1.092548 * y * z,
        0.315392 * (3.0 * z * z - 1.0),
        1.092548 * x * z,
        0.546274 * (x * x - y * y),
    ]
    if order >= 3:
        terms += [
            0.590044 * y * (3.0 * x * x - y * y),
            2.890611 * x * y * z,
            0.457046 * y * (5.0 * z * z - 1.0),
            0.373176 * z * (5.0 * z * z - 3.0),
            0.457046 * x * (5.0 * z * z - 1.0),
            1.445306 * z * (x * x - y * y),
            0.590044 * x * (x * x - 3.0 * y * y),
        ]
    return np.stack(terms, axis=-1).astype(np.float32)


# ─────────────────────────────────────────── EnvMap ──────────────────────────

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


# ─────────────────────────────────────────── SHLighting ──────────────────────

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
