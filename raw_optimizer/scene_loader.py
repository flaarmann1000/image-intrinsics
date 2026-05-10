"""
Helpers for loading scenes from raw_dataset.

Directory layout (relative to the repo root):
    raw_dataset/raw/<scene_id>/
        normal_map.png   — world normals encoded (N*0.5+0.5)*255
        albedo_map.png   — per-pixel GT albedo (different colour per object)
        mask.png         — colour-ID mask: object i uses _OBJECT_COLORS[i], background = black
        params.json      — objects list (mask_color, radius, position, albedo, phong, ct)
                           + 10 lighting configs with "sh": {"coeffs": [[9×3]]}

    raw_dataset/rendered/<shader>/<light_type>/<scene_id>/
        variant_00.png … variant_09.png
"""

import json
import os

import numpy as np
from PIL import Image

from raw_renderer import Camera, SHLighting, generate_mesh
from raw_renderer.mesh import Mesh
from raw_renderer_gpu import rasterize_geometry

_DATASET_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "raw_dataset",
)

_DEFAULT_CAM = Camera(
    position=np.array([0.0, 0.0, 3.0], dtype=np.float32),
    target=np.array([0.0, 0.0, 0.0], dtype=np.float32),
)


# ── private helpers ───────────────────────────────────────────────────────────

def _decode_normals(img_np: np.ndarray) -> np.ndarray:
    """uint8 [H,W,3] → float32 [H,W,3] unit normals."""
    n = img_np.astype(np.float32) / 255.0 * 2.0 - 1.0
    norm = np.linalg.norm(n, axis=-1, keepdims=True).clip(1e-8)
    return (n / norm).astype(np.float32)


def _load_png_float(path: str) -> np.ndarray:
    """Load PNG → float32 [H,W,3] in [0,1]."""
    return np.array(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def _load_png_int(path: str) -> np.ndarray:
    """Load PNG [H,W,3] in [0,255]."""
    return np.array(Image.open(path).convert("RGB"), dtype=np.uint8)


def _sphere_mesh(radius: float, position: np.ndarray,
                 n_lat: int = 24, n_lon: int = 48) -> Mesh:
    base = generate_mesh("sphere", n_lat=n_lat, n_lon=n_lon)
    return Mesh(
        vertices=base.vertices * radius + position,
        faces=base.faces,
        normals=base.normals,
        vertex_normals=base.vertex_normals,
    )


def _merge_meshes(meshes: list) -> Mesh:
    V_offset = 0
    all_verts, all_faces, all_normals, all_vn = [], [], [], []
    for m in meshes:
        all_verts.append(m.vertices)
        all_faces.append(m.faces + V_offset)
        all_normals.append(m.normals)
        all_vn.append(m.vertex_normals)
        V_offset += len(m.vertices)
    return Mesh(
        vertices=np.concatenate(all_verts),
        faces=np.concatenate(all_faces),
        normals=np.concatenate(all_normals),
        vertex_normals=np.concatenate(all_vn),
    )


def _decode_object_mask(mask_img: np.ndarray, objects: list) -> np.ndarray:
    """
    Decode colour-ID mask PNG to per-pixel object index.

    mask_img : uint8 [H, W, 3]
    objects  : list of object dicts, each with "mask_color": [R, G, B]
    Returns  : int8 [H, W]  — −1 = background, i = object i
    """
    H, W = mask_img.shape[:2]
    obj_mask = np.full((H, W), -1, dtype=np.int8)
    for i, obj in enumerate(objects):
        color = np.array(obj["mask_color"], dtype=np.uint8)
        hit = np.all(mask_img == color[None, None, :], axis=-1)
        obj_mask[hit] = i
    return obj_mask


def _read_sh_coeffs(light_cfg: dict) -> np.ndarray:
    """Read (9,3) SH coefficients stored directly in the lighting config."""
    return np.array(light_cfg["sh"]["coeffs"], dtype=np.float32)


# ── public API ────────────────────────────────────────────────────────────────

def list_scenes() -> list:
    """Return sorted list of scene IDs available in raw_dataset/raw/."""
    raw_root = os.path.join(_DATASET_ROOT, "raw")
    if not os.path.isdir(raw_root):
        return []
    return sorted(
        d for d in os.listdir(raw_root)
        if os.path.isdir(os.path.join(raw_root, d))
    )


def load_scene(
    scene_id: str,
    shader: str = "ct",
    light_type: str = "sh",
    variant_indices=None,           # list[int] or None → all 10
    width: int = 200,
    height: int = 200,
    use_mesh_normals: bool = True,  # True → rasterize_geometry; False → load PNG
    device: str = "cuda",
) -> dict:
    """
    Load one scene from raw_dataset.

    Returns a dict with:
        images       : list of [H, W, 3] float32 in [0, 1]
        normals      : Tensor [H, W, 3]  unit normals
        frag_pos     : Tensor [H, W, 3]  (only if use_mesh_normals=True, else None)
        cam_pos      : Tensor [3]        (only if use_mesh_normals=True, else None)
        mask         : Tensor or ndarray [H, W] bool  — foreground pixels
        object_mask  : ndarray [H, W] int8  — per-pixel object index (−1 = bg)
        gt_albedo    : ndarray [H, W, 3] float32  per-pixel GT albedo
        gt_sh_coeffs : list of ndarray [9, 3]  GT SH coefficients per variant
        scene_id     : str
        params       : dict  — raw params.json content
    """
    if variant_indices is None:
        variant_indices = list(range(10))

    raw_dir = os.path.join(_DATASET_ROOT, "raw", scene_id)
    with open(os.path.join(raw_dir, "params.json")) as f:
        params = json.load(f)

    # ── Rendered input images ─────────────────────────────────────────────────
    rendered_dir = os.path.join(
        # _DATASET_ROOT, "rendered", shader, light_type, scene_id
        _DATASET_ROOT, "rendered_gpu", shader, light_type, scene_id
    )
    images = [
        _load_png_float(os.path.join(rendered_dir, f"variant_{k:02d}.png"))
        for k in variant_indices
    ]

    # ── Geometry ──────────────────────────────────────────────────────────────
    objs = params["objects"]
    meshes = [
        _sphere_mesh(o["radius"], np.array(o["position"], dtype=np.float32))
        for o in objs
    ]
    combined = _merge_meshes(meshes)

    if use_mesh_normals:
        normals, frag_pos, mask, cam_pos = rasterize_geometry(
            combined, _DEFAULT_CAM, width=width, height=height,
            smooth=True, device=device,
        )
    else:
        normals_img = np.array(
            Image.open(os.path.join(raw_dir, "normal_map.png")).convert("RGB")
        )
        normals = _decode_normals(normals_img)
        mask_raw = np.array(
            Image.open(os.path.join(raw_dir, "mask.png")).convert("RGB")
        )
        # Foreground = any non-black pixel
        mask = np.any(mask_raw > 0, axis=-1)
        frag_pos = None
        cam_pos = None

    # ── Colour-ID mask ────────────────────────────────────────────────────────
    mask_img = np.array(Image.open(
        os.path.join(raw_dir, "mask.png")).convert("RGB"))
    object_mask = _decode_object_mask(mask_img, objs)   # [H,W] int8, -1=bg

    # Binary foreground mask (used by optimizer)
    if use_mesh_normals:
        # mask already comes from rasterize_geometry as a bool Tensor
        pass
    else:
        mask = object_mask >= 0   # ndarray bool

    # ── Ground-truth maps ─────────────────────────────────────────────────────
    gt_albedo = _load_png_int(os.path.join(
        raw_dir, "albedo_map.png"))  # [H,W,3]

    # ── Ground-truth SH coefficients ─────────────────────────────────────────
    gt_sh_coeffs = [
        _read_sh_coeffs(params["lighting"][k])
        for k in variant_indices
    ]

    return dict(
        images=images,
        normals=normals,
        frag_pos=frag_pos,
        cam_pos=cam_pos,
        mask=mask,
        object_mask=object_mask,
        gt_albedo=gt_albedo,
        gt_sh_coeffs=gt_sh_coeffs,
        scene_id=scene_id,
        params=params,
    )
