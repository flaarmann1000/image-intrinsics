"""
Dataset generation for intrinsic decomposition experiments.

Four public functions
---------------------
create_scenes(n_scenes)   – ground-truth normal / albedo / mask maps + param JSON
render_scenes()           – render every scene × shader × lighting × variant
create_tiny_raw()         – 3×6 px normal maps (8 or 9 unique normals) + albedo
render_tiny()             – shade the tiny maps with CT + SH (metallic=0, roughness=0)

Run all four with:
    python dataset.py

Mask encoding
-------------
mask.png uses per-object colour IDs from _OBJECT_COLORS.
Background pixels are black [0, 0, 0].
Each object i is painted with _OBJECT_COLORS[i].
params.json stores "mask_color": [R, G, B] per object for look-up.

SH lighting
-----------
Random order-2 SH coefficients (9 × 3 RGB) are stored directly in params.json
under "sh": {"coeffs": [...]}. This avoids the directional-light sign bug and
gives richer environment-like lighting.
"""

import json
import os

import numpy as np
from PIL import Image

from raw_renderer import (
    Camera, EnvMap, PhongMaterial, PBRMaterial,
    PointLight, SHLighting, generate_mesh, render,
)
from raw_renderer.mesh import Mesh
from raw_renderer.shaders.phong         import phong_shader
from raw_renderer.shaders.cook_torrance import cook_torrance_shader


# ── constants ─────────────────────────────────────────────────────────────────

_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "raw_dataset")
_W, _H = 200, 200
_CAM   = Camera(
    position=np.array([0.0, 0.0, 3.0], dtype=np.float32),
    target  =np.array([0.0, 0.0, 0.0], dtype=np.float32),
)

# Unique per-object mask colours (background = [0, 0, 0])
_OBJECT_COLORS = [
    [255,   0,   0],   # object 0 → red
    [  0, 255,   0],   # object 1 → green
    [  0,   0, 255],   # object 2 → blue
    [255, 255,   0],   # object 3 → yellow
    [  0, 255, 255],   # object 4 → cyan
    [255,   0, 255],   # object 5 → magenta
]


# ── mesh helpers ──────────────────────────────────────────────────────────────

def _sphere_mesh(radius: float, position: np.ndarray, n_lat=24, n_lon=48) -> Mesh:
    base = generate_mesh("sphere", n_lat=n_lat, n_lon=n_lon)
    return Mesh(
        vertices      = base.vertices * radius + position,
        faces         = base.faces,
        normals       = base.normals,
        vertex_normals= base.vertex_normals,
    )


def _merge_meshes(meshes: list) -> Mesh:
    """Concatenate multiple Mesh objects into one (with correct face index offsets)."""
    V_offset = 0
    all_verts, all_faces, all_normals, all_vn = [], [], [], []
    for m in meshes:
        all_verts.append(m.vertices)
        all_faces.append(m.faces + V_offset)
        all_normals.append(m.normals)
        all_vn.append(m.vertex_normals)
        V_offset += len(m.vertices)
    return Mesh(
        vertices      = np.concatenate(all_verts),
        faces         = np.concatenate(all_faces),
        normals       = np.concatenate(all_normals),
        vertex_normals= np.concatenate(all_vn),
    )


# ── dispatch shaders (position-based material assignment) ─────────────────────

def _make_albedo_shader(positions: list, albedos: list):
    pos_arr = [np.asarray(p, dtype=np.float32) for p in positions]
    alb_arr = [np.asarray(a, dtype=np.float32) for a in albedos]
    def shader(frag_pos, normal, cam_pos):
        idx = int(np.argmin([np.linalg.norm(frag_pos - p) for p in pos_arr]))
        return alb_arr[idx]
    return shader


def _make_object_id_shader(positions: list, mask_colors: list):
    """Returns the object's mask colour (normalised to [0,1]) per fragment."""
    pos_arr    = [np.asarray(p, dtype=np.float32) for p in positions]
    colors_f   = [np.array(c, dtype=np.float32) / 255.0 for c in mask_colors]
    def shader(frag_pos, normal, cam_pos):
        idx = int(np.argmin([np.linalg.norm(frag_pos - p) for p in pos_arr]))
        return colors_f[idx]
    return shader


def _make_multi_shader(positions: list, mats: list, light, shader_type: str):
    pos_arr  = [np.asarray(p, dtype=np.float32) for p in positions]
    shade_fn = phong_shader if shader_type == "phong" else cook_torrance_shader
    def shader(frag_pos, normal, cam_pos):
        idx = int(np.argmin([np.linalg.norm(frag_pos - p) for p in pos_arr]))
        return shade_fn(frag_pos, normal, cam_pos, mats[idx], light)
    return shader


# ── random parameter generators ───────────────────────────────────────────────

def _rand_light_dir(rng) -> np.ndarray:
    d = rng.standard_normal(3).astype(np.float32)
    d[1] = abs(d[1])
    return d / (np.linalg.norm(d) + 1e-8)


def _rand_surface_normal(rng) -> np.ndarray:
    d = rng.standard_normal(3).astype(np.float32)
    d[2] = abs(d[2])
    return d / (np.linalg.norm(d) + 1e-8)


def _rand_distinct_albedos(rng, min_dist: float = 0.5) -> tuple:
    """
    Return two albedo colours (each [3] float32 in [0,1]) whose L2 distance
    is at least min_dist.  Falls back to complementary colours after 200 tries.
    """
    a = rng.random(3).astype(np.float32)
    for _ in range(200):
        b = rng.random(3).astype(np.float32)
        if np.linalg.norm(a - b) >= min_dist:
            return a, b
    return a, (1.0 - a).astype(np.float32)


def _rand_color(rng) -> np.ndarray:
    c = rng.uniform(0.7, 1.0, 3).astype(np.float32)
    if rng.random() > 0.5:
        c[0] = min(1.0, c[0] * 1.15)   # warm
    else:
        c[2] = min(1.0, c[2] * 1.15)   # cool
    return c


def _rand_sh_coeffs(rng) -> np.ndarray:
    """
    Random order-2 SH coefficients (9, 3) for plausible diffuse lighting.

    Band 0 (ambient): per-channel positive value → always-visible base irradiance.
    Bands 1–2: random directional / quadratic variation, bounded so irradiance
               stays positive on front-facing surfaces.
    """
    coeffs = np.zeros((9, 3), dtype=np.float32)
    color     = _rand_color(rng)
    intensity = float(rng.uniform(0.5, 2.0))
    coeffs[0] = color * intensity                                    # ambient
    scale1    = 0.5 * intensity
    scale2    = 0.3 * intensity
    coeffs[1:4] = rng.uniform(-scale1, scale1, (3, 3)).astype(np.float32)
    coeffs[4:9] = rng.uniform(-scale2, scale2, (5, 3)).astype(np.float32)
    return coeffs


def _rand_phong_params(rng) -> dict:
    return {
        "ka":        float(rng.uniform(0.02, 0.10)),
        "kd":        float(rng.uniform(0.40, 1.00)),
        "ks":        float(rng.uniform(0.10, 0.60)),
        "shininess": float(rng.choice([8, 16, 32, 64, 128, 256])),
    }


def _rand_ct_params(rng) -> dict:
    return {
        "metallic":  float(rng.uniform(0.0, 1.0)),
        "roughness": float(rng.uniform(0.05, 1.0)),
    }


def _rand_lighting_config(rng) -> dict:
    color = _rand_color(rng)
    return {
        "point": {
            "position": (_rand_light_dir(rng) * float(rng.uniform(1.5, 4.0))).tolist(),
            "color":    color.tolist(),
        },
        "sh": {
            # Full 9×3 RGB SH coefficients stored directly — avoids the
            # directional-light sign ambiguity in SHLighting.directional().
            "coeffs": _rand_sh_coeffs(rng).tolist(),
        },
        "envmap": {
            "direction": _rand_light_dir(rng).tolist(),
            "color":     color.tolist(),
        },
    }


def _build_light(cfg: dict, light_type: str):
    if light_type == "point":
        p = cfg["point"]
        return PointLight(
            position=np.array(p["position"], dtype=np.float32),
            color   =np.array(p["color"],    dtype=np.float32),
        )
    if light_type == "sh":
        return SHLighting(np.array(cfg["sh"]["coeffs"], dtype=np.float32))
    if light_type == "envmap":
        e = cfg["envmap"]
        return EnvMap.point_like(
            direction=np.array(e["direction"], dtype=np.float32),
            color    =tuple(e["color"]),
        )
    raise ValueError(f"Unknown light type: {light_type!r}")


def _build_material(obj: dict, shader: str):
    albedo = np.array(obj["albedo"], dtype=np.float32)
    if shader == "phong":
        p = obj["phong"]
        return PhongMaterial(
            base_color=albedo,
            ka=p["ka"], kd=p["kd"], ks=p["ks"], shininess=p["shininess"],
        )
    if shader == "ct":
        c = obj["ct"]
        return PBRMaterial(
            albedo   =albedo,
            metallic =c["metallic"],
            roughness=c["roughness"],
        )
    raise ValueError(f"Unknown shader: {shader!r}")


# ── Function 1 ────────────────────────────────────────────────────────────────

def create_scenes(n_scenes: int = 1, seed: int = 42) -> None:
    """
    Render ground-truth maps and generate param JSONs for n_scenes.
    Each scene contains two spheres with distinct albedo colours.

    Per scene output (raw_dataset/raw/<scene_id>/):
        normal_map.png   — world normals encoded (N*0.5+0.5)*255
        albedo_map.png   — per-pixel albedo (different colour per sphere)
        mask.png         — colour-ID mask: object i painted with _OBJECT_COLORS[i]
        params.json      — objects list + 10 lighting configs
    """
    rng = np.random.default_rng(seed)

    for i in range(n_scenes):
        scene_id = f"scene_{i:04d}"
        out_dir  = os.path.join(_ROOT, "raw", scene_id)
        os.makedirs(out_dir, exist_ok=True)

        albedo_a, albedo_b = _rand_distinct_albedos(rng)
        objs = []
        for side_idx, (side, albedo) in enumerate(zip((-1, +1), (albedo_a, albedo_b))):
            radius   = float(rng.uniform(0.15, 0.30))
            offset_x = float(rng.uniform(0.35, 0.50))
            position = np.array([
                side * offset_x + rng.uniform(-0.05, 0.05),
                rng.uniform(-0.20, 0.20),
                0.0,
            ], dtype=np.float32)
            objs.append({
                "mask_color": _OBJECT_COLORS[side_idx],
                "radius":     radius,
                "position":   position.tolist(),
                "albedo":     albedo.tolist(),
                "phong":      _rand_phong_params(rng),
                "ct":         _rand_ct_params(rng),
            })

        meshes       = [_sphere_mesh(o["radius"], np.array(o["position"], dtype=np.float32)) for o in objs]
        combined     = _merge_meshes(meshes)
        positions_3d = [np.array(o["position"],   dtype=np.float32) for o in objs]
        albedos_3d   = [np.array(o["albedo"],      dtype=np.float32) for o in objs]
        mask_colors  = [o["mask_color"] for o in objs]

        render(combined, _CAM,
               lambda p, n, c: (n * 0.5 + 0.5).astype(np.float32),
               smooth=True, width=_W, height=_H,
               output_path=os.path.join(out_dir, "normal_map.png"))

        render(combined, _CAM,
               _make_albedo_shader(positions_3d, albedos_3d),
               smooth=True, width=_W, height=_H,
               output_path=os.path.join(out_dir, "albedo_map.png"))

        render(combined, _CAM,
               _make_object_id_shader(positions_3d, mask_colors),
               smooth=True, width=_W, height=_H,
               output_path=os.path.join(out_dir, "mask.png"))

        params = {
            "scene_id": scene_id,
            "objects":  objs,
            "lighting": [_rand_lighting_config(rng) for _ in range(10)],
        }
        with open(os.path.join(out_dir, "params.json"), "w") as f:
            json.dump(params, f, indent=2)

        for oi, o in enumerate(objs):
            print(f"  [{scene_id}] obj{oi}: r={o['radius']:.2f}  "
                  f"pos={np.array(o['position'])[:2].round(2)}  "
                  f"albedo={np.array(o['albedo']).round(2)}")


# ── Function 2 ────────────────────────────────────────────────────────────────

def render_scenes() -> None:
    """
    Render all scenes found in raw_dataset/raw/ for every combination of
    shader (phong | ct) × lighting (point | sh | envmap) × 10 variants.

    Output: raw_dataset/rendered/<shader>/<lighting>/<scene_id>/variant_<k>.png
    """
    raw_root  = os.path.join(_ROOT, "raw")
    scene_ids = sorted(
        d for d in os.listdir(raw_root)
        if os.path.isdir(os.path.join(raw_root, d))
    )

    for scene_id in scene_ids:
        with open(os.path.join(raw_root, scene_id, "params.json")) as f:
            cfg = json.load(f)

        objs         = cfg["objects"]
        meshes       = [_sphere_mesh(o["radius"], np.array(o["position"], dtype=np.float32)) for o in objs]
        combined     = _merge_meshes(meshes)
        positions_3d = [np.array(o["position"], dtype=np.float32) for o in objs]

        for shader in ("phong", "ct"):
            mats = [_build_material(o, shader) for o in objs]

            for light_type in ("point", "sh", "envmap"):
                out_dir = os.path.join(_ROOT, "rendered", shader, light_type, scene_id)
                os.makedirs(out_dir, exist_ok=True)

                for k, light_cfg in enumerate(cfg["lighting"]):
                    light = _build_light(light_cfg, light_type)
                    fn    = _make_multi_shader(positions_3d, mats, light, shader)

                    render(combined, _CAM, fn,
                           smooth=True, width=_W, height=_H,
                           output_path=os.path.join(out_dir, f"variant_{k:02d}.png"))

                    print(f"  {shader}/{light_type}/{scene_id}/variant_{k:02d}")


# ── Function 3 ────────────────────────────────────────────────────────────────

def create_tiny_raw(seed: int = 0) -> None:
    """
    Create 3×6 pixel ground-truth images (one pixel per unique surface patch).

    Output (raw_dataset/raw_tiny/):
        normals_a.png   — 18 pixels, 8 unique random normals
        normals_b.png   — 18 pixels, 9 unique random normals
        albedo_tiny.png — 18 pixels, 18 unique random albedo colors
    Normals encoded as (N*0.5+0.5)*255 uint8.
    """
    rng     = np.random.default_rng(seed)
    out_dir = os.path.join(_ROOT, "raw_tiny")
    os.makedirs(out_dir, exist_ok=True)

    albedo_flat = rng.random((18, 3)).astype(np.float32)
    Image.fromarray(
        (albedo_flat.reshape(3, 6, 3) * 255).astype(np.uint8)
    ).save(os.path.join(out_dir, "albedo_tiny.png"))

    def _make_normal_img(n_unique: int) -> np.ndarray:
        unique = np.stack([_rand_surface_normal(rng) for _ in range(n_unique)])
        assign = rng.integers(0, n_unique, size=18)
        for j in range(n_unique):
            if j not in assign:
                assign[rng.integers(0, 18)] = j
        normals = unique[assign]
        encoded = ((normals * 0.5 + 0.5) * 255).clip(0, 255).astype(np.uint8)
        return encoded.reshape(3, 6, 3)

    Image.fromarray(_make_normal_img(8)).save(os.path.join(out_dir, "normals_a.png"))
    Image.fromarray(_make_normal_img(9)).save(os.path.join(out_dir, "normals_b.png"))
    print(f"Tiny raw data → {out_dir}")


# ── Function 4 ────────────────────────────────────────────────────────────────

def render_tiny(seed: int = 0) -> None:
    """
    Shade the 3×6 tiny normal maps with Cook-Torrance + SH.
    Material: metallic=0, roughness=0 (pure diffuse Lambertian limit).
    Generates 10 random SH lighting variants.

    Output: raw_dataset/rendered_tiny/<normals_a|b>_variant_<k>.png
    """
    rng     = np.random.default_rng(seed)
    raw_dir = os.path.join(_ROOT, "raw_tiny")
    out_dir = os.path.join(_ROOT, "rendered_tiny")
    os.makedirs(out_dir, exist_ok=True)

    albedo_flat = (
        np.array(Image.open(os.path.join(raw_dir, "albedo_tiny.png")), dtype=np.float32) / 255.0
    ).reshape(18, 3)

    lights = [SHLighting(_rand_sh_coeffs(rng)) for _ in range(10)]

    frag_pos = np.zeros(3, dtype=np.float32)
    cam_pos  = _CAM.position

    for normal_name in ("normals_a", "normals_b"):
        normals_raw = (
            np.array(Image.open(os.path.join(raw_dir, f"{normal_name}.png")), dtype=np.float32) / 255.0
        ).reshape(18, 3) * 2.0 - 1.0

        lengths = np.linalg.norm(normals_raw, axis=1, keepdims=True)
        normals = normals_raw / np.maximum(lengths, 1e-8)

        for k, light in enumerate(lights):
            pixels = np.zeros((18, 3), dtype=np.float32)
            for idx in range(18):
                mat = PBRMaterial(albedo=albedo_flat[idx], metallic=0.0, roughness=0.0)
                pixels[idx] = cook_torrance_shader(frag_pos, normals[idx], cam_pos, mat, light)

            out_path = os.path.join(out_dir, f"{normal_name}_variant_{k:02d}.png")
            Image.fromarray(
                (pixels.reshape(3, 6, 3) * 255).clip(0, 255).astype(np.uint8)
            ).save(out_path)
            print(f"  {normal_name}  variant_{k:02d}")


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== create_scenes ===");  create_scenes(n_scenes=1)
    print("=== render_scenes ===");  render_scenes()
    print("=== create_tiny_raw ==="); create_tiny_raw()
    print("=== render_tiny ===");    render_tiny()
