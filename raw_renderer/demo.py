"""
Demo: render a sphere with every shader × light combination.

Run from anywhere:
    python raw_renderer/demo.py
    python -m raw_renderer.demo

Output (all 512×512 unless noted) in renders/:
    phong_point.png   — Phong + point light
    phong_sh.png      — Phong + spherical harmonics
    phong_envmap.png  — Phong + sky/ground env map  (256×256)
    ct_point.png      — Cook-Torrance + point light
    ct_sh.png         — Cook-Torrance + SH
    ct_envmap.png     — Cook-Torrance + env map IBL  (256×256)
"""

import os

import numpy as np

from raw_renderer import (
    Camera,
    EnvMap,
    PhongMaterial,
    PBRMaterial,
    PointLight,
    SHLighting,
    cook_torrance_shader,
    generate_mesh,
    phong_shader,
    render,
)

mesh = generate_mesh("sphere")
cam = Camera(
    position=np.array([1.5, 1.5, 3.0], dtype=np.float32),
    target=np.array([0.0, 0.0, 0.0], dtype=np.float32),
)

# ---------------------------------------------------------------------------
# Lights
# ---------------------------------------------------------------------------
pt_light = PointLight(
    position=np.array([3.0, 5.0, 3.0], dtype=np.float32),
    color=np.array([1.0, 0.95, 0.9], dtype=np.float32),
)

sh_light = SHLighting.directional(
    direction=np.array([0.5, 1.0, 0.3], dtype=np.float32),
    color=np.array([1.0, 0.9, 0.8], dtype=np.float32),
    intensity=2.0,
)

env_map = EnvMap.sky_ground(sky=(0.4, 0.6, 1.0), ground=(0.2, 0.15, 0.1))

out = "render/"
os.makedirs(out, exist_ok=True)

# ---------------------------------------------------------------------------
# Phong
# ---------------------------------------------------------------------------
phong_mat = PhongMaterial(
    base_color=np.array([0.5, 0.6, 0.9], dtype=np.float32),
    shininess=64.0,
)

render(mesh, cam,
       lambda p, n, c: phong_shader(p, n, c, phong_mat, pt_light),
       #    smooth=True,
       output_path=out + "phong_point.png")

render(mesh, cam,
       lambda p, n, c: phong_shader(p, n, c, phong_mat, sh_light),
       output_path=out + "phong_sh.png")

render(mesh, cam,
       lambda p, n, c: phong_shader(p, n, c, phong_mat, env_map),
       width=256, height=256,
       output_path=out + "phong_envmap.png")

# ---------------------------------------------------------------------------
# Cook-Torrance
# ---------------------------------------------------------------------------
pbr_mat = PBRMaterial(
    albedo=np.array([0.8, 0.3, 0.15], dtype=np.float32),
    roughness=0.5,
    metallic=0.0,
)

render(mesh, cam,
       lambda p, n, c: cook_torrance_shader(p, n, c, pbr_mat, pt_light),
       output_path=out + "ct_point.png")

render(mesh, cam,
       lambda p, n, c: cook_torrance_shader(p, n, c, pbr_mat, sh_light),
       output_path=out + "ct_sh.png")

render(mesh, cam,
       lambda p, n, c: cook_torrance_shader(
           p, n, c, pbr_mat, env_map, n_env_samples=16),
       width=256, height=256,
       output_path=out + "ct_envmap.png")
