"""
GPU demo — mirrors raw_renderer/demo.py but calls render() once with a GPU kernel.

Run:
    python raw_renderer_gpu/demo.py
    python -m raw_renderer_gpu.demo
"""

import os
import time
import numpy as np
from PIL import Image

from raw_renderer import (
    Camera, EnvMap, PhongMaterial, PBRMaterial, PointLight, SHLighting,
    generate_mesh,
)
from raw_renderer_gpu import render

mesh = generate_mesh("sphere")
cam  = Camera(
    position=np.array([0.0, 0.0, 3.0], dtype=np.float32),
    target  =np.array([0.0, 0.0, 0.0], dtype=np.float32),
)

# ── Lights ────────────────────────────────────────────────────────────────────
pt_light = PointLight(
    position=np.array([2.0, 0.0, 2.0], dtype=np.float32),
    color   =np.array([1.0, 0.9, 0.8], dtype=np.float32),
)

sh_light = SHLighting.directional(
    direction=np.array([1.0, 0.0, 1.0]),
    color    =np.array([1.0, 0.9, 0.8], dtype=np.float32),
    intensity=1.0,
)

env_map = EnvMap.from_sh(sh_light)

out = "render_gpu/"
os.makedirs(out, exist_ok=True)

# Save the env map for reference
Image.fromarray(
    (env_map.image * 255).clip(0, 255).astype(np.uint8)
).save(out + "env_map.png")

# ── Phong ─────────────────────────────────────────────────────────────────────
phong_mat = PhongMaterial(
    base_color=np.array([0.5, 0.6, 0.9], dtype=np.float32),
    shininess=64.0, ka=0.05, kd=0.8, ks=0.3,
)

t0 = time.perf_counter()
render(mesh, cam, phong_mat, pt_light,
       smooth=True, width=512, height=512,
       output_path=out + "phong_point.png")
print(f"  phong+point  : {(time.perf_counter()-t0)*1e3:.1f} ms")

t0 = time.perf_counter()
render(mesh, cam, phong_mat, sh_light,
       smooth=True, width=512, height=512,
       output_path=out + "phong_sh.png")
print(f"  phong+sh     : {(time.perf_counter()-t0)*1e3:.1f} ms")

t0 = time.perf_counter()
render(mesh, cam, phong_mat, env_map,
       smooth=True, width=256, height=256,
       output_path=out + "phong_envmap.png")
print(f"  phong+envmap : {(time.perf_counter()-t0)*1e3:.1f} ms")

# ── Cook-Torrance ─────────────────────────────────────────────────────────────
pbr_mat = PBRMaterial(
    albedo   =np.array([0.5, 0.6, 0.9], dtype=np.float32),
    roughness=0.2,
    metallic =0.0,
)

t0 = time.perf_counter()
render(mesh, cam, pbr_mat, pt_light,
       smooth=True, width=512, height=512,
       output_path=out + "ct_point.png")
print(f"  CT+point     : {(time.perf_counter()-t0)*1e3:.1f} ms")

t0 = time.perf_counter()
render(mesh, cam, pbr_mat, sh_light,
       smooth=True, width=512, height=512,
       output_path=out + "ct_sh.png")
print(f"  CT+sh        : {(time.perf_counter()-t0)*1e3:.1f} ms")

t0 = time.perf_counter()
render(mesh, cam, pbr_mat, env_map,
       smooth=True, width=256, height=256,
       output_path=out + "ct_envmap.png")
print(f"  CT+envmap    : {(time.perf_counter()-t0)*1e3:.1f} ms")
