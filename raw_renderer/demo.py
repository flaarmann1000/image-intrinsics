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

import matplotlib.pyplot as plt
import os

import numpy as np
from PIL import Image

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
    position=np.array([0.0, 0.0, 3.0], dtype=np.float32),
    target=np.array([0.0, 0.0, 0.0], dtype=np.float32),
)

# ---------------------------------------------------------------------------
# Lights
# ---------------------------------------------------------------------------
pt_light = PointLight(
    # position=np.array([1000.0, 0.0, 1000.0], dtype=np.float32),
    position=np.array([2.0, 0.0, 2.0], dtype=np.float32),
    color=np.array([1.0, 0.9, 0.8], dtype=np.float32),
)


sh_light = SHLighting.directional(
    direction=np.array([1.0, 0.0, 1.0]),
    color=np.array([1.0, 0.9, 0.8], dtype=np.float32),
    intensity=1.0,
)


# env_map = EnvMap.sky_ground(sky=(0.4, 0.6, 1.0), ground=(0.2, 0.15, 0.1))

env_map = EnvMap.from_sh(sh_light)
# env_map = EnvMap.point_like(direction=np.array(
#     [1.0, 0.0, 1.0]), color=(1.0, 0.9, 0.8))

out = "render/"
os.makedirs(out, exist_ok=True)

env_map_u8 = (env_map.image * 255).clip(0, 255).astype(np.uint8)
Image.fromarray(env_map_u8).save(out+"env_map.png")

# ---------------------------------------------------------------------------
# Phong
# ---------------------------------------------------------------------------
phong_mat = PhongMaterial(
    base_color=np.array([0.5, 0.6, 0.9], dtype=np.float32),
    shininess=64.0,
    # shininess=256.0,
    # shininess=1.0,
    # ka=0.05, kd=0.8,  ks=0.3
    ka=0.5, kd=80,  ks=0
)

# render(mesh, cam,
#        lambda p, n, c: phong_shader(p, n, c, phong_mat, pt_light),
#        smooth=True,
#        width=256, height=256,
#        output_path=out + "phong_point.png")

# render(mesh, cam,
#        lambda p, n, c: phong_shader(p, n, c, phong_mat, sh_light),
#        smooth=True,
#        width=256, height=256,
#        output_path=out + "phong_sh.png")

# render(mesh, cam,
#        lambda p, n, c: phong_shader(p, n, c, phong_mat, env_map),
#        smooth=True,
#        width=256, height=256,
#        output_path=out + "phong_envmap.png")

# ---------------------------------------------------------------------------
# Cook-Torrance
# ---------------------------------------------------------------------------
pbr_mat = PBRMaterial(
    albedo=np.array([0.5, 0.6, 0.9], dtype=np.float32),
    roughness=0.2,
    metallic=0.0,
)

# render(mesh, cam,
#        lambda p, n, c: cook_torrance_shader(p, n, c, pbr_mat, pt_light),
#        smooth=True,
#        width=256, height=256,
#        output_path=out + "ct_point.png")

# render(mesh, cam,
#        lambda p, n, c: cook_torrance_shader(p, n, c, pbr_mat, sh_light),
#        smooth=True,
#        width=256, height=256,
#        output_path=out + "ct_sh.png")

render(mesh, cam,
       lambda p, n, c: cook_torrance_shader(p, n, c, pbr_mat, env_map),
       smooth=True,
       width=256, height=256,
       output_path=out + "ct_envmap.png")
