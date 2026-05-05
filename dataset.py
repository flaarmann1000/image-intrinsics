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


# ── private helpers ───────────────────────────────────────────────────────────

def _sphere_mesh(radius: float, position: np.ndarray, n_lat=24, n_lon=48) -> Mesh:
    base = generate_mesh("sphere", n_lat=n_lat, n_lon=n_lon)
    return Mesh(
        vertices      = base.vertices * radius + position,
        faces         = base.faces,
        normals       = base.normals,          # unit normals invariant under uniform scale
        vertex_normals= base.vertex_normals,
    )


def _rand_light_dir(rng) -> np.ndarray:
    """Random unit vector on upper hemisphere (y ≥ 0)."""
    d = rng.standard_normal(3).astype(np.float32)
    d[1] = abs(d[1])
    return d / (np.linalg.norm(d) + 1e-8)


def _rand_surface_normal(rng) -> np.ndarray:
    """Random unit normal loosely facing the camera (+z)."""
    d = rng.standard_normal(3).astype(np.float32)
    d[2] = abs(d[2])
    return d / (np.linalg.norm(d) + 1e-8)


def _rand_color(rng) -> np.ndarray:
    """Warm- or cool-tinted white."""
    c = rng.uniform(0.7, 1.0, 3).astype(np.float32)
    if rng.random() > 0.5:
        c[0] = min(1.0, c[0] * 1.15)   # warm
    else:
        c[2] = min(1.0, c[2] * 1.15)   # cool
    return c


def _rand_phong_params(rng, albedo: np.ndarray) -> dict:
    return {
        "base_color": albedo.tolist(),
        "ka":         float(rng.uniform(0.02, 0.10)),
        "kd":         float(rng.uniform(0.40, 1.00)),
        "ks":         float(rng.uniform(0.10, 0.60)),
        "shininess":  float(rng.choice([8, 16, 32, 64, 128, 256])),
    }


def _rand_ct_params(rng, albedo: np.ndarray) -> dict:
    return {
        "albedo":    albedo.tolist(),
        "metallic":  float(rng.uniform(0.0, 1.0)),
        "roughness": float(rng.uniform(0.05, 1.0)),
    }


def _rand_lighting_config(rng) -> dict:
    """One lighting config carrying point / SH / envmap variants."""
    color = _rand_color(rng)
    return {
        "point": {
            "position": (_rand_light_dir(rng) * float(rng.uniform(1.5, 4.0))).tolist(),
            "color":    color.tolist(),
        },
        "sh": {
            "direction": _rand_light_dir(rng).tolist(),
            "color":     color.tolist(),
            "intensity": float(rng.uniform(0.5, 2.0)),
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
        s = cfg["sh"]
        return SHLighting.directional(
            direction=np.array(s["direction"], dtype=np.float32),
            color    =np.array(s["color"],     dtype=np.float32),
            intensity=s["intensity"],
        )
    if light_type == "envmap":
        e = cfg["envmap"]
        return EnvMap.point_like(
            direction=np.array(e["direction"], dtype=np.float32),
            color    =tuple(e["color"]),
        )
    raise ValueError(f"Unknown light type: {light_type!r}")


def _build_material(cfg: dict, shader: str):
    if shader == "phong":
        p = cfg["phong"]
        return PhongMaterial(
            base_color=np.array(p["base_color"], dtype=np.float32),
            ka=p["ka"], kd=p["kd"], ks=p["ks"], shininess=p["shininess"],
        )
    if shader == "ct":
        c = cfg["ct"]
        return PBRMaterial(
            albedo   =np.array(c["albedo"], dtype=np.float32),
            metallic =c["metallic"],
            roughness=c["roughness"],
        )
    raise ValueError(f"Unknown shader: {shader!r}")


# ── Function 1 ────────────────────────────────────────────────────────────────

def create_scenes(n_scenes: int = 1, seed: int = 42) -> None:
    """
    Render ground-truth maps and generate param JSONs for n_scenes spheres.

    Per scene output (raw_dataset/raw/<scene_id>/):
        normal_map.png   — world normals, encoded (N*0.5+0.5)*255
        albedo_map.png   — flat sphere albedo
        mask.png         — white on sphere, black elsewhere
        params.json      — sphere geometry, Phong/CT material, 10 lighting configs
    """
    rng = np.random.default_rng(seed)

    for i in range(n_scenes):
        scene_id = f"scene_{i:04d}"
        out_dir  = os.path.join(_ROOT, "raw", scene_id)
        os.makedirs(out_dir, exist_ok=True)

        radius   = float(rng.uniform(0.3, 0.7))
        position = np.array([
            rng.uniform(-0.3, 0.3),
            rng.uniform(-0.3, 0.3),
            0.0,
        ], dtype=np.float32)
        albedo = rng.random(3).astype(np.float32)
        mesh   = _sphere_mesh(radius, position)

        render(mesh, _CAM,
               lambda p, n, c: (n * 0.5 + 0.5).astype(np.float32),
               smooth=True, width=_W, height=_H,
               output_path=os.path.join(out_dir, "normal_map.png"))

        alb = albedo.copy()
        render(mesh, _CAM,
               lambda p, n, c, a=alb: a,
               smooth=True, width=_W, height=_H,
               output_path=os.path.join(out_dir, "albedo_map.png"))

        render(mesh, _CAM,
               lambda p, n, c: np.ones(3, dtype=np.float32),
               smooth=True, width=_W, height=_H,
               output_path=os.path.join(out_dir, "mask.png"))

        params = {
            "scene_id": scene_id,
            "sphere":   {"radius": radius, "position": position.tolist(), "albedo": albedo.tolist()},
            "phong":    _rand_phong_params(rng, albedo),
            "ct":       _rand_ct_params(rng, albedo),
            "lighting": [_rand_lighting_config(rng) for _ in range(10)],
        }
        with open(os.path.join(out_dir, "params.json"), "w") as f:
            json.dump(params, f, indent=2)

        print(f"[{scene_id}]  r={radius:.2f}  pos={position[:2].round(2)}  albedo={albedo.round(2)}")


# ── Function 2 ────────────────────────────────────────────────────────────────

def render_scenes() -> None:
    """
    Render all scenes found in raw_dataset/raw/ for every combination of
    shader (phong | ct) × lighting (point | sh | envmap) × 10 variants.

    Output: raw_dataset/rendered/<shader>/<lighting>/<scene_id>/variant_<k>.png
    """
    raw_root = os.path.join(_ROOT, "raw")
    scene_ids = sorted(
        d for d in os.listdir(raw_root)
        if os.path.isdir(os.path.join(raw_root, d))
    )

    for scene_id in scene_ids:
        with open(os.path.join(raw_root, scene_id, "params.json")) as f:
            cfg = json.load(f)

        radius   = cfg["sphere"]["radius"]
        position = np.array(cfg["sphere"]["position"], dtype=np.float32)
        mesh     = _sphere_mesh(radius, position)

        for shader in ("phong", "ct"):
            mat = _build_material(cfg, shader)

            for light_type in ("point", "sh", "envmap"):
                out_dir = os.path.join(_ROOT, "rendered", shader, light_type, scene_id)
                os.makedirs(out_dir, exist_ok=True)

                for k, light_cfg in enumerate(cfg["lighting"]):
                    light = _build_light(light_cfg, light_type)

                    if shader == "phong":
                        fn = lambda p, n, c, _m=mat, _l=light: phong_shader(p, n, c, _m, _l)
                    else:
                        fn = lambda p, n, c, _m=mat, _l=light: cook_torrance_shader(p, n, c, _m, _l)

                    render(mesh, _CAM, fn,
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

    # 18 unique albedo colors
    albedo_flat = rng.random((18, 3)).astype(np.float32)
    Image.fromarray(
        (albedo_flat.reshape(3, 6, 3) * 255).astype(np.uint8)
    ).save(os.path.join(out_dir, "albedo_tiny.png"))

    def _make_normal_img(n_unique: int) -> np.ndarray:
        unique = np.stack([_rand_surface_normal(rng) for _ in range(n_unique)])  # (U, 3)
        assign = rng.integers(0, n_unique, size=18)
        # guarantee every unique normal appears at least once
        for j in range(n_unique):
            if j not in assign:
                assign[rng.integers(0, 18)] = j
        normals = unique[assign]                                       # (18, 3)
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
    ).reshape(18, 3)                                               # (18, 3)

    # 10 random SH lights
    lights = [
        SHLighting.directional(
            direction=_rand_light_dir(rng),
            color    =_rand_color(rng),
            intensity=float(rng.uniform(0.5, 2.0)),
        )
        for _ in range(10)
    ]

    frag_pos = np.zeros(3, dtype=np.float32)   # all patches at origin
    cam_pos  = _CAM.position

    for normal_name in ("normals_a", "normals_b"):
        normals_raw = (
            np.array(Image.open(os.path.join(raw_dir, f"{normal_name}.png")), dtype=np.float32) / 255.0
        ).reshape(18, 3) * 2.0 - 1.0                              # decode → (18, 3)

        # re-normalise (encoding round-trips lose precision)
        lengths = np.linalg.norm(normals_raw, axis=1, keepdims=True)
        normals = normals_raw / np.maximum(lengths, 1e-8)

        for k, light in enumerate(lights):
            pixels = np.zeros((18, 3), dtype=np.float32)
            for idx in range(18):
                mat = PBRMaterial(
                    albedo   =albedo_flat[idx],
                    metallic =0.0,
                    roughness=0.0,
                )
                pixels[idx] = cook_torrance_shader(
                    frag_pos, normals[idx], cam_pos, mat, light
                )

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
