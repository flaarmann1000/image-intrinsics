"""
Software rasterizer.

render() projects a Mesh through a Camera and calls a user-supplied
shader_fn for every visible fragment. The shader is a plain callable:

    shader_fn(frag_pos, normal, cam_pos) -> np.ndarray shape (3,), RGB in [0,1]

Build the shader as a closure so it captures material and light:

    mat   = PBRMaterial(albedo=np.array([0.8, 0.2, 0.1]), roughness=0.3)
    light = PointLight(position=np.array([3, 5, 3]))
    fn    = lambda pos, n, cam: cook_torrance_shader(pos, n, cam, mat, light)
    render(mesh, camera, fn, output_path="out.png")

Pipeline per triangle
─────────────────────
1. Project all vertices: model → clip → NDC → screen space
2. Bounding-box rasterisation: iterate over the pixel tile covering the triangle
3. Barycentric test: keep pixels inside the triangle
4. Z-test: keep pixels closer than current z-buffer value
5. Shade: interpolate world-space position, call shader_fn
"""

import numpy as np
from typing import Callable
from PIL import Image

from .mesh import Mesh
from .camera import Camera, look_at, perspective


# ---------------------------------------------------------------------------
# Projection helpers
# ---------------------------------------------------------------------------

def _project_vertices(
    mesh: Mesh, view: np.ndarray, proj: np.ndarray, W: int, H: int
) -> np.ndarray:
    """
    Return (V, 3) array of screen-space positions.
    Channel 0/1 = pixel x/y (origin top-left), channel 2 = NDC depth in [-1,1].
    """
    homo = np.hstack([mesh.vertices, np.ones((len(mesh.vertices), 1), dtype=np.float32)])
    clip = homo @ (proj @ view).T            # (V, 4)
    w    = clip[:, 3:4]
    ndc  = clip[:, :3] / w                  # (V, 3)
    screen = np.empty_like(ndc)
    screen[:, 0] = (ndc[:, 0] + 1) * 0.5 * W   # x: left→right
    screen[:, 1] = (1 - ndc[:, 1]) * 0.5 * H   # y: top→bottom (flip Y)
    screen[:, 2] = ndc[:, 2]
    return screen


# ---------------------------------------------------------------------------
# Rasterisation helpers
# ---------------------------------------------------------------------------

def _barycentric(s0, s1, s2, px, py):
    """
    Barycentric coordinates (w0, w1, w2) for pixel grid arrays (px, py)
    with respect to triangle (s0, s1, s2) in screen space.
    Returns (None, None, None) for degenerate triangles.
    """
    denom = (s1[1] - s2[1]) * (s0[0] - s2[0]) + (s2[0] - s1[0]) * (s0[1] - s2[1])
    if abs(denom) < 1e-6:
        return None, None, None
    w0 = ((s1[1] - s2[1]) * (px - s2[0]) + (s2[0] - s1[0]) * (py - s2[1])) / denom
    w1 = ((s2[1] - s0[1]) * (px - s2[0]) + (s0[0] - s2[0]) * (py - s2[1])) / denom
    w2 = 1 - w0 - w1
    return w0, w1, w2


# ---------------------------------------------------------------------------
# Main render function
# ---------------------------------------------------------------------------

def render(
    mesh:        Mesh,
    camera:      Camera,
    shader_fn:   Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
    width:       int = 512,
    height:      int = 512,
    smooth:      bool = False,
    output_path: str = "render.png",
) -> np.ndarray:
    """
    Rasterise `mesh` as seen by `camera`, shade every fragment with `shader_fn`.
    Saves a PNG to `output_path` and returns the (H, W, 3) uint8 image array.

    smooth=False  — flat shading: one face normal per triangle (hard edges)
    smooth=True   — Phong interpolation: vertex normals blended barycentrically
                    (smooth silhouette, good for curved surfaces like spheres)
    """
    view   = look_at(camera.position, camera.target, camera.up)
    proj   = perspective(camera.fov_deg, width / height)
    sverts = _project_vertices(mesh, view, proj, width, height)

    fb   = np.zeros((height, width, 3), dtype=np.float32)
    zbuf = np.full((height, width), np.inf, dtype=np.float32)

    for fi, face in enumerate(mesh.faces):
        i0, i1, i2 = face
        s0, s1, s2 = sverts[i0], sverts[i1], sverts[i2]

        # Screen-space bounding box (clamped to image)
        x_min = max(0,         int(np.floor(min(s0[0], s1[0], s2[0]))))
        x_max = min(width - 1, int(np.ceil( max(s0[0], s1[0], s2[0]))))
        y_min = max(0,         int(np.floor(min(s0[1], s1[1], s2[1]))))
        y_max = min(height- 1, int(np.ceil( max(s0[1], s1[1], s2[1]))))

        if x_max < x_min or y_max < y_min:
            continue

        # Pixel grid (sample at pixel centres)
        px, py = np.meshgrid(
            np.arange(x_min, x_max + 1, dtype=np.float32) + 0.5,
            np.arange(y_min, y_max + 1, dtype=np.float32) + 0.5,
        )

        w0, w1, w2 = _barycentric(s0, s1, s2, px, py)
        if w0 is None:
            continue
        assert w0 is not None and w1 is not None and w2 is not None

        inside = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
        z      = w0 * s0[2] + w1 * s1[2] + w2 * s2[2]

        tile_z  = zbuf[y_min:y_max+1, x_min:x_max+1]
        visible = inside & (z < tile_z)

        p0, p1, p2 = mesh.vertices[i0], mesh.vertices[i1], mesh.vertices[i2]
        flat_normal = mesh.normals[fi]
        vn0 = mesh.vertex_normals[i0]
        vn1 = mesh.vertex_normals[i1]
        vn2 = mesh.vertex_normals[i2]

        ys, xs = np.where(visible)
        for k in range(len(ys)):
            row, col      = ys[k] + y_min, xs[k] + x_min
            bw0, bw1, bw2 = w0[ys[k], xs[k]], w1[ys[k], xs[k]], w2[ys[k], xs[k]]
            frag_pos      = bw0 * p0 + bw1 * p1 + bw2 * p2

            if smooth:
                raw_n  = bw0 * vn0 + bw1 * vn1 + bw2 * vn2
                normal = raw_n / (np.linalg.norm(raw_n) + 1e-8)
            else:
                normal = flat_normal

            fb[row, col]   = shader_fn(frag_pos, normal, camera.position)
            zbuf[row, col] = z[ys[k], xs[k]]

    img_u8 = (fb * 255).clip(0, 255).astype(np.uint8)
    Image.fromarray(img_u8).save(output_path)
    print(f"Saved {output_path}  ({width}×{height})")
    return img_u8
