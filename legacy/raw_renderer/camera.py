import numpy as np
from dataclasses import dataclass, field


@dataclass
class Camera:
    position: np.ndarray = field(default_factory=lambda: np.array([0, 0, 3],  dtype=np.float32))
    target:   np.ndarray = field(default_factory=lambda: np.array([0, 0, 0],  dtype=np.float32))
    up:       np.ndarray = field(default_factory=lambda: np.array([0, 1, 0],  dtype=np.float32))
    fov_deg:  float = 60.0


def look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    """4×4 view matrix: world → camera space."""
    f = target - eye;  f /= np.linalg.norm(f)
    r = np.cross(f, up); r /= np.linalg.norm(r)
    u = np.cross(r, f)
    M = np.eye(4, dtype=np.float32)
    M[0, :3] = r;  M[0, 3] = -r.dot(eye)
    M[1, :3] = u;  M[1, 3] = -u.dot(eye)
    M[2, :3] = -f; M[2, 3] =  f.dot(eye)
    return M


def perspective(fov_deg: float, aspect: float, near: float = 0.1, far: float = 100.0) -> np.ndarray:
    """4×4 perspective projection matrix (OpenGL/NDC convention)."""
    t = np.tan(np.radians(fov_deg) / 2)
    M = np.zeros((4, 4), dtype=np.float32)
    M[0, 0] = 1 / (aspect * t)
    M[1, 1] = 1 / t
    M[2, 2] = -(far + near) / (far - near)
    M[2, 3] = -2 * far * near / (far - near)
    M[3, 2] = -1
    return M
