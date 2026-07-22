"""
GPU demo — mirrors raw_renderer/demo.py but calls render() once with a GPU kernel.

Run:
    python raw_renderer_gpu/demo.py
    python -m raw_renderer_gpu.demo
"""

import os
import time
import numpy as np
import torch
from PIL import Image

from raw_renderer import Camera, EnvMap, SHLighting, generate_mesh, load_obj
from raw_renderer_gpu import (
    render,
    SHLight, PBRMat, PhongMat, PointLightGPU, EnvMapLightGPU,
)

# mesh = generate_mesh("sphere")
# mesh = load_obj(r"assets\obj\suzanne.obj")
mesh = load_obj(r"assets\obj\stanford-bunny.obj")
cam = Camera(
    position=np.array([0.0, 0.0, 3.0], dtype=np.float32),
    target=np.array([0.0, 0.0, 0.0], dtype=np.float32),
)

width = 512
height = 512

# ── Lights ────────────────────────────────────────────────────────────────────
pt_light = PointLightGPU(
    position=torch.tensor([2.0, 0.0, 2.0]),
    color=torch.tensor([1.0, 0.9, 0.8]),
)

sh_raw = SHLighting.directional(
    direction=np.array([1.0, 0.0, 1.0]),
    color=np.array([1.0, 0.9, 0.8], dtype=np.float32),
    intensity=1.0,
)
sh_light = SHLight(coeffs=torch.from_numpy(sh_raw.coeffs))

env_raw = EnvMap.from_sh(sh_raw)
# env_raw = EnvMap.point_like(direction=np.array(
#     [1.0, 0.0, 1.0]), color=(100, 90, 80))
env_light = EnvMapLightGPU(
    dirs=torch.from_numpy(env_raw._dirs),
    image_flat=torch.from_numpy(env_raw._image_flat),
    solid_angles=torch.from_numpy(env_raw._solid_angles),
)

out = "render_gpu/"
os.makedirs(out, exist_ok=True)

# Save the env map for reference
Image.fromarray(
    (env_raw.image * 255).clip(0, 255).astype(np.uint8)
).save(out + "env_map.png")

# ── Phong ─────────────────────────────────────────────────────────────────────
phong_mat = PhongMat(
    base_color=torch.tensor([0.5, 0.6, 0.9]),
    shininess=64.0, ka=0.00, kd=0.8, ks=0.3,
)

t0 = time.perf_counter()
render(mesh, cam, phong_mat, pt_light,
       smooth=True, width=width, height=height,
       output_path=out + "phong_point.png")
print(f"  phong+point  : {(time.perf_counter()-t0)*1e3:.1f} ms")

t0 = time.perf_counter()
render(mesh, cam, phong_mat, sh_light,
       smooth=True, width=width, height=height,
       output_path=out + "phong_sh.png")
print(f"  phong+sh     : {(time.perf_counter()-t0)*1e3:.1f} ms")

t0 = time.perf_counter()
render(mesh, cam, phong_mat, env_light,
       smooth=True, width=width, height=height,
       output_path=out + "phong_envmap.png")
print(f"  phong+envmap : {(time.perf_counter()-t0)*1e3:.1f} ms")

# ── Cook-Torrance ─────────────────────────────────────────────────────────────
pbr_mat = PBRMat(
    # albedo=torch.tensor([0.8, 0.6, 0.9]),
    roughness=0.3,
    metallic=0.3,
    # albedo=torch.tensor([0.5, 0.5, 0.5]),
    albedo=torch.tensor([0.5, 0.6, 0.9]),
    # metallic=0.0,
)

t0 = time.perf_counter()
render(mesh, cam, pbr_mat, pt_light,
       smooth=True, width=width, height=height,
       output_path=out + "ct_point.png")
print(f"  CT+point     : {(time.perf_counter()-t0)*1e3:.1f} ms")

t0 = time.perf_counter()
render(mesh, cam, pbr_mat, sh_light,
       smooth=True, width=width, height=height,
       output_path=out + "ct_sh.png")
print(f"  CT+sh        : {(time.perf_counter()-t0)*1e3:.1f} ms")

t0 = time.perf_counter()
render(mesh, cam, pbr_mat, env_light,
       smooth=True, width=width, height=height,
       output_path=out + "ct_envmap.png")
print(f"  CT+envmap    : {(time.perf_counter()-t0)*1e3:.1f} ms")
