"""GPU renderer — pure PyTorch, no C++ compilation needed.

Split out of the former single-file `raw_renderer_gpu.rasterizer` (1165 lines):

    types.py        material / light / scene dataclasses
    sh.py           SH basis, irradiance, filtered radiance
    brdf.py         Cook-Torrance terms + the GGX-SH lookup table
    shade_ct.py     shade_ct_sh / shade_ct_env
    shade_phong.py  shade_phong_sh / shade_phong_env
    raster.py       projection, face-batched rasterisation, render()
    mesh.py         generate_mesh / load_obj
    lighting.py     EnvMap / SHLighting

This re-exports exactly the names the old package exposed, so
`from idr.render import shade_ct_sh, EnvMap` matches the previous
`from raw_renderer_gpu import ...` surface.
"""
from .types import (
    SHLight, PBRMat, PhongMat, PointLightGPU, EnvMapLightGPU, Camera, Mesh,
)
from .sh import build_sh_basis
from .brdf import get_ggx_sh_lut, _get_ggx_sh_lut
from .shade_ct import shade_ct_sh, shade_ct_env
from .shade_phong import shade_phong_sh, shade_phong_env
from .raster import render, rasterize_geometry
from .mesh import generate_mesh, load_obj
from .lighting import EnvMap, SHLighting

__all__ = [
    "render", "rasterize_geometry", "shade_ct_sh", "shade_ct_env",
    "shade_phong_sh", "shade_phong_env",
    "SHLight", "PBRMat", "PhongMat", "PointLightGPU", "EnvMapLightGPU",
    "Camera", "Mesh", "generate_mesh", "load_obj",
    "EnvMap", "SHLighting", "build_sh_basis",
    # de-facto public: notebooks and the optimizer import the LUT directly
    "get_ggx_sh_lut",
]
