"""Phong shading entry points: SH lighting and explicit env-map lighting."""
from __future__ import annotations

import math

import numpy as np
import torch
from typing import Optional, Union

from .ops import _norm
from .types import PhongMat, PointLightGPU, EnvMapLightGPU
from .sh import _sh_basis, _sh_irradiance, _sh_phong_filtered_radiance

def _phong_point(frag_pos, N, cam_pos, mat: PhongMat, light: PointLightGPU):
    V = _norm(cam_pos - frag_pos)
    L = _norm(light.position - frag_pos)
    NdL = (N*L).sum(1, keepdim=True).clamp(min=0)
    diff = mat.kd * light.color * mat.base_color / torch.pi * NdL
    R = _norm(2*NdL*N - L)
    RdV = (R*V).sum(1, keepdim=True).clamp(min=0)
    spec = mat.ks * light.color * RdV**mat.shininess
    amb = mat.ka * light.color * mat.base_color
    return (amb + diff + spec).clamp(0, 1)


def _phong_envmap(frag_pos, N, cam_pos, mat: PhongMat, light: EnvMapLightGPU, sbatch=128):
    S = light.dirs.shape[0]
    norm_f = (mat.shininess + 2.0) / (2.0 * torch.pi)
    V = _norm(cam_pos - frag_pos)
    NdV = (N*V).sum(1, keepdim=True)             # (M, 1)
    diff = frag_pos.new_zeros(frag_pos.shape)
    spec = frag_pos.new_zeros(frag_pos.shape)
    for si in range(0, S, sbatch):
        L_b = light.dirs[si:si+sbatch]           # (B, 3)
        r_b = light.image_flat[si:si+sbatch]
        dw_b = light.solid_angles[si:si+sbatch]  # (B,)
        NdL = (N @ L_b.T).clamp(min=0)           # (M, B)
        mask = (NdL > 1e-4).float()
        diff += (NdL * dw_b * mask) @ r_b         # (M, 3)
        LdV = V @ L_b.T                           # (M, B)
        RdV = (2.0*NdL*NdV - LdV).clamp(min=0) * mask
        spec += (RdV**mat.shininess * dw_b) @ r_b
    mean_rad = light.image_flat.mean(0)
    return (mat.ka*mean_rad*mat.base_color + mat.kd*diff*mat.base_color/torch.pi + mat.ks*spec*norm_f).clamp(0, 1)


def shade_phong_sh(V, N, ka, kd, ks, shininess, base_color, coeffs,
                   return_components: bool = False):
    irr  = _sh_irradiance(coeffs, N)
    diff = kd * irr * base_color / torch.pi
    NdV  = (N * V).sum(1, keepdim=True).clamp(min=0)
    R    = _norm(2 * NdV * N - V)
    ks_pos = (ks > 0) if not torch.is_tensor(ks) else torch.any(ks > 0)
    if ks_pos:
        L_R  = _sh_phong_filtered_radiance(coeffs, R, shininess)
        spec = ks * L_R
    else:
        L_R  = diff.new_zeros(diff.shape)
        spec = diff.new_zeros(diff.shape)
    composite = diff + spec
    if not return_components:
        return composite
    return composite, {"irr": irr, "diff": diff, "R": R, "L_spec": L_R, "spec": spec}


def shade_phong_env(
    view:              torch.Tensor,
    normals:           torch.Tensor,
    albedo:            torch.Tensor,
    env_pixels:        torch.Tensor,       # (P, 3) — may be learnable
    env_dirs:          torch.Tensor,       # (P, 3)
    env_dw:            torch.Tensor,       # (P,)
    ka:                float = 0.0,
    kd:                float = 1.0,
    ks:                Union[float, torch.Tensor] = 0.5,
    shininess:         Union[float, torch.Tensor] = 32.0,
    sbatch:            int = 128,
    return_components: bool = False,
) -> torch.Tensor:
    """
    Differentiable Phong shading with explicit env-map integration.

    Companion to shade_phong_sh: same flat (M, 3) interface, integrates over
    explicit (P, 3) env-map samples. Gradients flow through env_pixels, albedo,
    ks, and shininess.

    Parameters
    ----------
    view, normals, albedo : (M, 3)
    env_pixels            : (P, 3)  env-map radiance (learnable)
    env_dirs              : (P, 3)  sample directions
    env_dw                : (P,)    solid angles
    ka, kd                : scene-level ambient / diffuse scalars
    ks                    : scalar or (M, 1) specular coefficient
    shininess             : scalar or (M, 1) Phong exponent
    return_components     : if True, return (composite, dict)
    """
    device = albedo.device

    def _as_tensor(x):
        if torch.is_tensor(x):
            return x.to(device)
        return torch.full(
            albedo.shape[:-1] + (1,), float(x),
            dtype=albedo.dtype, device=device,
        )

    ks_t        = _as_tensor(ks)
    shininess_t = _as_tensor(shininess)
    norm_f      = (shininess_t + 2.0) / (2.0 * torch.pi)  # (M,1) or scalar

    P           = env_pixels.shape[0]
    NdotV_raw   = (normals * view).sum(-1, keepdim=True)   # (M, 1)
    front       = (NdotV_raw > 0).to(albedo.dtype)

    diff_irr    = albedo.new_zeros(albedo.shape)
    spec        = albedo.new_zeros(albedo.shape)

    for si in range(0, P, sbatch):
        L_b  = env_dirs[si:si + sbatch]    # (B, 3)
        r_b  = env_pixels[si:si + sbatch]  # (B, 3)
        dw_b = env_dw[si:si + sbatch]      # (B,)

        NdL_raw = normals @ L_b.T          # (M, B)
        mf      = (NdL_raw > 1e-4).float()
        LdV     = view @ L_b.T             # (M, B)

        # R·V = 2*(N·L)*(N·V) - L·V, clamped and masked to front-facing lights
        RdV     = (2.0 * NdL_raw.clamp(min=0) * NdotV_raw - LdV).clamp(min=0) * mf

        # spec_contrib: (M, B)  —  shininess_t (M,1) broadcasts over B
        spec_contrib = RdV ** shininess_t * norm_f * dw_b * mf
        spec     += spec_contrib @ r_b                            # (M, 3)
        diff_irr += (NdL_raw.clamp(min=0) * dw_b * mf) @ r_b    # (M, 3)

    mean_rad  = env_pixels.mean(0)
    diff      = kd * albedo / torch.pi * diff_irr
    spec_out  = ks_t * spec
    amb       = ka * mean_rad * albedo
    composite = (amb + diff + spec_out) * front

    if not return_components:
        return composite
    return composite, {
        "NdotV": NdotV_raw,
        "irr":   diff_irr,
        "diff":  diff,
        "spec":  spec_out,
    }
