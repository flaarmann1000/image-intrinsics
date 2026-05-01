import numpy as np
from dataclasses import dataclass


@dataclass
class Mesh:
    vertices:       np.ndarray  # (V, 3) float32
    faces:          np.ndarray  # (F, 3) int32  — triangle indices into vertices
    normals:        np.ndarray  # (F, 3) float32 — outward unit normal per face (flat shading)
    vertex_normals: np.ndarray  # (V, 3) float32 — averaged face normals per vertex (smooth shading)


def _vertex_normals(vertices: np.ndarray, faces: np.ndarray, face_normals: np.ndarray) -> np.ndarray:
    """Average surrounding face normals at each vertex and normalise."""
    vn = np.zeros((len(vertices), 3), dtype=np.float32)
    np.add.at(vn, faces[:, 0], face_normals)
    np.add.at(vn, faces[:, 1], face_normals)
    np.add.at(vn, faces[:, 2], face_normals)
    lengths = np.linalg.norm(vn, axis=1, keepdims=True)
    return vn / np.maximum(lengths, 1e-8)


def _face_normal(v0: np.ndarray, v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    n = np.cross(v1 - v0, v2 - v0).astype(np.float32)
    length = np.linalg.norm(n)
    return n / length if length > 1e-8 else n


def _sphere_mesh(n_lat: int, n_lon: int) -> tuple:
    """
    UV (lat-long) sphere with outward-facing face normals.

    Vertices
    --------
    Index 0                        — north pole (0, 1, 0)
    Indices 1 … (n_lat-1)*n_lon   — (n_lat-1) latitude rings, n_lon verts each
    Index (n_lat-1)*n_lon + 1     — south pole (0, -1, 0)

    Winding (verified to give outward cross-product normals)
    -------
    North cap : [north, ring1[j+1], ring1[j]]
    Body      : [top_j, bot_{j+1}, bot_j]  and  [top_j, top_{j+1}, bot_{j+1}]
    South cap : [south, last[j],   last[j+1]]
    """
    verts = [[0.0, 1.0, 0.0]]                         # north pole

    for i in range(1, n_lat):                          # latitude rings
        theta = np.pi * i / n_lat                      # 0 < theta < π
        for j in range(n_lon):
            phi = 2.0 * np.pi * j / n_lon
            verts.append([
                np.sin(theta) * np.cos(phi),
                np.cos(theta),
                np.sin(theta) * np.sin(phi),
            ])

    verts.append([0.0, -1.0, 0.0])                    # south pole
    v = np.array(verts, dtype=np.float32)

    south = len(v) - 1
    faces = []

    def ring_start(i: int) -> int:                     # index of first vertex in ring i (1-based)
        return 1 + (i - 1) * n_lon

    # North cap
    for j in range(n_lon):
        faces.append([0, ring_start(1) + (j + 1) % n_lon, ring_start(1) + j])

    # Body
    for i in range(1, n_lat - 1):
        rs  = ring_start(i)
        rs2 = ring_start(i + 1)
        for j in range(n_lon):
            nj = (j + 1) % n_lon
            faces.append([rs + j,  rs2 + nj, rs2 + j ])
            faces.append([rs + j,  rs  + nj, rs2 + nj])

    # South cap
    rs_last = ring_start(n_lat - 1)
    for j in range(n_lon):
        faces.append([south, rs_last + j, rs_last + (j + 1) % n_lon])

    return v, np.array(faces, dtype=np.int32)


def generate_mesh(shape: str = "cube", **kwargs) -> Mesh:
    """
    Build a primitive mesh.

    Supported shapes
    ----------------
    'cube'   — unit cube centred at the origin
    'plane'  — flat 2×2 quad in the XZ plane (y = 0)
    'sphere' — UV sphere of radius 1, centred at the origin
               kwargs: n_lat (default 16), n_lon (default 32)
    """
    if shape == "cube":
        v = np.array([
            [-0.5, -0.5, -0.5], [ 0.5, -0.5, -0.5],  # 0 1
            [ 0.5,  0.5, -0.5], [-0.5,  0.5, -0.5],  # 2 3
            [-0.5, -0.5,  0.5], [ 0.5, -0.5,  0.5],  # 4 5
            [ 0.5,  0.5,  0.5], [-0.5,  0.5,  0.5],  # 6 7
        ], dtype=np.float32)
        faces = np.array([
            [0, 2, 1], [0, 3, 2],  # back   (-Z)
            [4, 5, 6], [4, 6, 7],  # front  (+Z)
            [0, 1, 5], [0, 5, 4],  # bottom (-Y)
            [3, 6, 2], [3, 7, 6],  # top    (+Y)
            [0, 4, 7], [0, 7, 3],  # left   (-X)
            [1, 2, 6], [1, 6, 5],  # right  (+X)
        ], dtype=np.int32)
    elif shape == "plane":
        v = np.array([
            [-1, 0, -1], [ 1, 0, -1],
            [ 1, 0,  1], [-1, 0,  1],
        ], dtype=np.float32)
        faces = np.array([[0, 2, 1], [0, 3, 2]], dtype=np.int32)
    elif shape == "sphere":
        n_lat = int(kwargs.get("n_lat", 16))
        n_lon = int(kwargs.get("n_lon", 32))
        v, faces = _sphere_mesh(n_lat, n_lon)
    else:
        raise ValueError(f"Unknown shape {shape!r}. Choose 'cube', 'plane', or 'sphere'.")

    normals = np.array(
        [_face_normal(v[f[0]], v[f[1]], v[f[2]]) for f in faces],
        dtype=np.float32,
    )
    return Mesh(vertices=v, faces=faces, normals=normals,
                vertex_normals=_vertex_normals(v, faces, normals))
