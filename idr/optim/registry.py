"""One dispatch point for the four {Cook-Torrance, Phong} x {SH, env} optimizers.

The same four-way if/elif was written out at four call sites -- decompose_scene, the
synthetic study, the MIT driver and the real-scene driver -- each rebuilding the same
branch and unpacking the same positional 7-tuple. They differ only in

    material args   CT wants (metallic, roughness); Phong wants (shininess, ks, ka, kd)
    lighting args   env models take (dirs, dw) positionally and (H, W) as keywords

so the branch is fully described by the table below, and callers can just say

    result = optimize("ct_sh", images, *geom, mat_a, mat_b, cfg, ...)
"""
from __future__ import annotations

import time
from typing import Optional

from .result import EnvGrid, OptimResult

__all__ = ["SHADERS", "optimize", "is_env_shader", "is_phong_shader"]

# shader -> (needs ka/kd, needs an EnvGrid)
SHADERS = {
    "ct_sh":     (False, False),
    "ct_env":    (False, True),
    "phong_sh":  (True,  False),
    "phong_env": (True,  True),
}


def is_env_shader(shader: str) -> bool:
    return SHADERS[shader][1]


def is_phong_shader(shader: str) -> bool:
    return SHADERS[shader][0]


def _impl(shader: str):
    """Import the optimizer lazily: importing all four pulls in every shading path."""
    if shader == "ct_sh":
        from .models.ct_sh import _optimize_ct_sh as f
    elif shader == "ct_env":
        from .models.ct_env import _optimize_ct_env as f
    elif shader == "phong_sh":
        from .models.phong_sh import _optimize_phong_sh as f
    elif shader == "phong_env":
        from .models.phong_env import _optimize_phong_env as f
    else:
        raise ValueError(f"unknown shader {shader!r}; expected one of {sorted(SHADERS)}")
    return f


def optimize(shader, images, normals_hw, frag_pos_hw, mask_hw, cam_pos,
             mat_a, mat_b, cfg, *, ka=None, kd=None,
             env: Optional[EnvGrid] = None, **common) -> OptimResult:
    """Run one decomposition.

    mat_a / mat_b are the two fixed material inputs: (metallic, roughness) for the
    Cook-Torrance models, (shininess, ks) for the Phong ones. `ka`/`kd` are required by
    the Phong models and rejected by the CT ones; `env` is required by the env models
    and rejected by the SH ones. Everything else (wandb_run, gt_sh_coeffs, gt_albedo,
    opt_params, transforms, init_from_gt, log_gradients, grad_log_dir, val_images,
    val_sh_coeffs, init_maps) passes straight through as keywords.
    """
    if shader not in SHADERS:
        raise ValueError(f"unknown shader {shader!r}; expected one of {sorted(SHADERS)}")
    wants_kakd, wants_env = SHADERS[shader]

    if wants_kakd and (ka is None or kd is None):
        raise ValueError(f"{shader} needs ka and kd")
    if not wants_kakd and (ka is not None or kd is not None):
        raise ValueError(f"{shader} does not take ka/kd (those are Phong-only)")
    if wants_env and env is None:
        raise ValueError(f"{shader} needs env=EnvGrid(dirs, dw, H, W)")
    if not wants_env and env is not None:
        raise ValueError(f"{shader} does not take env (it solves for SH coefficients)")

    geom = (images, normals_hw, frag_pos_hw, mask_hw, cam_pos)
    args = (*geom, mat_a, mat_b)
    if wants_kakd:
        args += (ka, kd)
    if wants_env:
        args += (env.dirs, env.dw)
        common = {**common, "env_H": env.H, "env_W": env.W}

    out = _impl(shader)(*args, cfg, **common)
    albedo, light, a, b, shadings, history, elapsed = out
    return OptimResult(albedo=albedo, light=light, mat_a=a, mat_b=b,
                       shadings=shadings, history=history, elapsed=elapsed,
                       shader=shader)
