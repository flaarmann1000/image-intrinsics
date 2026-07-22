"""Projection, face-batched rasterisation, attribute interpolation, and render().

Rasterisation is face-batched and env-map shading sample-batched so GPU memory
stays bounded regardless of mesh or env-map size.
"""
from __future__ import annotations

import math

import numpy as np
import torch
from PIL import Image
from typing import Optional, Union

from .types import SHLight, PBRMat, PhongMat, PointLightGPU, EnvMapLightGPU
from .ops import _cuda, _norm
from .shade_ct import _ct_point, _ct_envmap, shade_ct_sh
from .shade_phong import _phong_point, _phong_envmap, shade_phong_sh

def _look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    """4×4 view matrix: world → camera space."""
    f = target - eye
    f /= np.linalg.norm(f)
    r = np.cross(f, up)
    r /= np.linalg.norm(r)
    u = np.cross(r, f)
    M = np.eye(4, dtype=np.float32)
    M[0, :3] = r
    M[0, 3] = -r.dot(eye)
    M[1, :3] = u
    M[1, 3] = -u.dot(eye)
    M[2, :3] = -f
    M[2, 3] = f.dot(eye)
    return M


def _perspective(fov_deg: float, aspect: float, near: float = 0.1, far: float = 100.0) -> np.ndarray:
    """4×4 perspective projection matrix (OpenGL/NDC convention)."""
    t = np.tan(np.radians(fov_deg) / 2)
    M = np.zeros((4, 4), dtype=np.float32)
    M[0, 0] = 1 / (aspect * t)
    M[1, 1] = 1 / t
    M[2, 2] = -(far + near) / (far - near)
    M[2, 3] = -2 * far * near / (far - near)
    M[3, 2] = -1
    return M






def _project(verts, MVP, W, H):
    homo = torch.cat([verts, verts.new_ones(len(verts), 1)], 1)  # (V, 4)
    clip = homo @ MVP.T                                           # (V, 4)
    ndc = clip[:, :3] / clip[:, 3:4]                            # (V, 3)
    s = verts.new_empty(ndc.shape)
    s[:, 0] = (ndc[:, 0] + 1.0) * 0.5 * W
    s[:, 1] = (1.0 - ndc[:, 1]) * 0.5 * H
    s[:, 2] = ndc[:, 2]
    return s                                                       # (V, 3)


def _rasterize(sverts, faces, W, H, face_batch=32):
    """
    Per-pixel closest-triangle test, processed in face batches to limit VRAM.
    Returns face_ids (H*W,) int64  [−1 = background]
            bary     (H*W, 3) float32
    """
    dev = sverts.device
    P = W * H
    F_n = faces.shape[0]

    px = torch.arange(W, device=dev, dtype=torch.float32) + 0.5
    py = torch.arange(H, device=dev, dtype=torch.float32) + 0.5
    py_g, px_g = torch.meshgrid(py, px, indexing='ij')
    pcx = px_g.reshape(P)        # (P,)
    pcy = py_g.reshape(P)

    best_z = sverts.new_full((P,),   float('inf'))
    face_ids = torch.full((P,), -1, device=dev, dtype=torch.long)
    bary = sverts.new_zeros(P, 3)
    aP = torch.arange(P, device=dev)

    for fi in range(0, F_n, face_batch):
        fb = faces[fi: fi + face_batch]          # (B, 3)
        ax = sverts[fb[:, 0], 0]
        ay = sverts[fb[:, 0], 1]
        az = sverts[fb[:, 0], 2]
        bx = sverts[fb[:, 1], 0]
        by = sverts[fb[:, 1], 1]
        bz = sverts[fb[:, 1], 2]
        cx = sverts[fb[:, 2], 0]
        cy = sverts[fb[:, 2], 1]
        cz = sverts[fb[:, 2], 2]

        denom = (by - cy)*(ax - cx) + (cx - bx)*(ay - cy)   # (B,)
        ok = denom.abs() > 1e-6

        # barycentric coords for all pixels × batch faces  →  (P, B)
        w0 = ((by-cy)*(pcx[:, None]-cx) + (cx-bx)
              * (pcy[:, None]-cy)) / (denom + 1e-10)
        w1 = ((cy-ay)*(pcx[:, None]-cx) + (ax-cx)
              * (pcy[:, None]-cy)) / (denom + 1e-10)
        w2 = 1.0 - w0 - w1

        inside = (w0 >= 0) & (w1 >= 0) & (w2 >= 0) & ok    # (P, B)
        z = torch.where(inside, w0*az + w1*bz + w2*cz,
                        sverts.new_full((), float('inf')))

        bz_best, bl = z.min(1)                               # (P,)
        imp = bz_best < best_z
        best_z = torch.where(imp, bz_best, best_z)
        face_ids = torch.where(imp, fi + bl, face_ids)
        bary = torch.where(imp[:, None],
                           torch.stack(
                               [w0[aP, bl], w1[aP, bl], w2[aP, bl]], 1),
                           bary)

    return face_ids, bary


def _interp(verts, fn, vn, faces, face_ids, bary, hit, smooth):
    fi = face_ids[hit]
    bw = bary[hit]                               # (M, 3)
    i0, i1, i2 = faces[fi, 0], faces[fi, 1], faces[fi, 2]

    frag_pos = bw[:, 0:1]*verts[i0] + bw[:, 1:2] * \
        verts[i1] + bw[:, 2:3]*verts[i2]

    if smooth:
        raw_n = bw[:, 0:1]*vn[i0] + bw[:, 1:2]*vn[i1] + bw[:, 2:3]*vn[i2]
        N = _norm(raw_n)
    else:
        N = fn[fi]

    return frag_pos, N                             # (M, 3), (M, 3)


def rasterize_geometry(
    mesh,
    camera,
    width:  int = 512,
    height: int = 512,
    smooth: bool = False,
    device: str = "cuda",
) -> tuple:
    """
    Rasterise mesh and return per-pixel geometry tensors without shading.

    Returns
    -------
    normals  : Tensor [H, W, 3]  float32 – unit normals (zero on background)
    frag_pos : Tensor [H, W, 3]  float32 – world-space fragment positions
    mask     : Tensor [H, W]     bool    – True on foreground pixels
    cam_pos  : Tensor [3]        float32 – camera position
    """
    W, H = width, height

    verts = _cuda(mesh.vertices,       device)
    faces = _cuda(mesh.faces,          device, dtype=torch.long)
    fn = _cuda(mesh.normals,        device)
    vn = _cuda(mesh.vertex_normals, device)
    cam_t = _cuda(camera.position.astype(np.float32), device)
    MVP = _cuda((_perspective(camera.fov_deg, W/H) @
                 _look_at(camera.position, camera.target, camera.up)
                 ).astype(np.float32), device)

    sverts = _project(verts, MVP, W, H)
    face_ids, bary = _rasterize(sverts, faces, W, H)

    hit = face_ids >= 0                          # (H*W,)
    normals_flat = verts.new_zeros(H * W, 3)
    frag_flat = verts.new_zeros(H * W, 3)

    if hit.any():
        fp, N = _interp(verts, fn, vn, faces, face_ids, bary, hit, smooth)
        normals_flat[hit] = N
        frag_flat[hit] = fp

    return (
        normals_flat.reshape(H, W, 3),
        frag_flat.reshape(H, W, 3),
        hit.reshape(H, W),
        cam_t,
    )


def render(
    mesh,
    camera,
    material,              # PhongMat | PBRMat
    light,                 # PointLightGPU | EnvMapLightGPU | SHLight
    width:       int = 512,
    height:      int = 512,
    smooth:      bool = False,
    output_path: str = "render_gpu.png",
    # amp: float = 1,
    amp: float = 5,
) -> np.ndarray:
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    W, H = width, height

    verts = _cuda(mesh.vertices,       dev)
    faces = _cuda(mesh.faces,          dev, dtype=torch.long)
    fn = _cuda(mesh.normals,        dev)
    vn = _cuda(mesh.vertex_normals, dev)
    cam_t = _cuda(camera.position.astype(np.float32), dev)
    MVP = _cuda((_perspective(camera.fov_deg, W/H) @
                 _look_at(camera.position, camera.target, camera.up)
                 ).astype(np.float32), dev)

    sverts = _project(verts, MVP, W, H)
    face_ids, bary = _rasterize(sverts, faces, W, H)

    hit = face_ids >= 0                          # (H*W,)
    # fb = verts.new_zeros(H*W, 3)
    fb = verts.new_ones(H*W, 3)

    if hit.any():
        frag_pos, N = _interp(verts, fn, vn, faces,
                              face_ids, bary, hit, smooth)

        def _dev(t):
            return t.to(dev) if isinstance(t, torch.Tensor) else t

        if isinstance(material, PhongMat):
            mat = PhongMat(base_color=_dev(material.base_color),
                           ka=material.ka, kd=material.kd,
                           ks=material.ks, shininess=material.shininess)
            if isinstance(light, PointLightGPU):
                col = _phong_point(frag_pos, N, cam_t, mat,
                                   PointLightGPU(_dev(light.position), _dev(light.color)))
            elif isinstance(light, EnvMapLightGPU):
                col = _phong_envmap(frag_pos, N, cam_t, mat,
                                    EnvMapLightGPU(_dev(light.dirs), _dev(light.image_flat), _dev(light.solid_angles)))
            elif isinstance(light, SHLight):
                # def shade_phong_sh(frag_pos, N, cam_pos, ka, kd, ks, shininess, base_color, coeffs):
                V = _norm(cam_t - frag_pos)
                col = shade_phong_sh(V, N, mat.ka, mat.kd,
                                     mat.ks, mat.shininess, mat.base_color, _dev(light.coeffs))
            else:
                raise TypeError(type(light))

        elif isinstance(material, PBRMat):
            mat = PBRMat(albedo=_dev(material.albedo),
                         metallic=material.metallic, roughness=material.roughness)
            if isinstance(light, PointLightGPU):
                col = _ct_point(frag_pos, N, cam_t, mat,
                                PointLightGPU(_dev(light.position), _dev(light.color)))
            elif isinstance(light, EnvMapLightGPU):
                col = _ct_envmap(frag_pos, N, cam_t, mat,
                                 EnvMapLightGPU(_dev(light.dirs), _dev(light.image_flat), _dev(light.solid_angles)))
            elif isinstance(light, SHLight):
                V = _norm(cam_t - frag_pos)
                col = shade_ct_sh(V, N, _dev(material.albedo), _dev(
                    light.coeffs), material.metallic, material.roughness)
            else:
                raise TypeError(type(light))

        else:
            raise TypeError(type(material))

        fb[hit] = col * amp

    img_u8 = (fb.reshape(H, W, 3)*255).clamp(0, 255).byte().cpu().numpy()
    Image.fromarray(img_u8).save(output_path)
    print(f"Saved {output_path}  ({W}×{H})")
    return img_u8
