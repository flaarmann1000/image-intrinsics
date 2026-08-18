"""Cook-Torrance shading entry points: SH lighting and explicit env-map lighting."""
from __future__ import annotations

import math

import numpy as np
import torch
from typing import Optional, Union

from .ops import _norm
from .types import PBRMat, PointLightGPU, EnvMapLightGPU
from .brdf import (_ggx_D, _schlick_F, _smith_G, _f0_mat, _get_ggx_sh_lut,
                   _lut_lookup, ggx_sh_bands)
from .sh import _sh_basis, _sh_order_of, _sh_irradiance

def _ct_point(frag_pos, N, cam_pos, mat: PBRMat, light: PointLightGPU):
    F0 = _f0_mat(mat.albedo, mat.metallic)
    alpha2 = mat.roughness**4
    k = (mat.roughness + 1)**2 / 8.0
    V = _norm(cam_pos - frag_pos)
    NdV = (N*V).sum(1).clamp(min=1e-4)            # (M,)
    L = _norm(light.position - frag_pos)
    NdL = (N*L).sum(1)                            # (M,)
    hit = NdL > 1e-4
    out = frag_pos.new_zeros(frag_pos.shape)
    if not hit.any():
        return out
    N_h, V_h, L_h = N[hit], V[hit], L[hit]
    NdV_h = NdV[hit]
    NdL_h = NdL[hit]
    H_v = _norm(L_h + V_h)
    NdH = (N_h*H_v).sum(1).clamp(0, 1)
    VdH = (V_h*H_v).sum(1).clamp(0, 1)
    D = _ggx_D(NdH, alpha2)
    F = _schlick_F(VdH, F0)                       # (M_h, 3)
    G = _smith_G(NdV_h, NdL_h, k)
    spec = F * (D*G / (4*NdV_h + 1e-7))[:, None] * light.color
    k_d = (1 - F) * (1 - mat.metallic)
    diff = k_d * mat.albedo / torch.pi * light.color * NdL_h[:, None]
    out[hit] = (diff + spec).clamp(0, 1)
    return out


def _ct_envmap(frag_pos, N, cam_pos, mat: PBRMat, light: EnvMapLightGPU, sbatch=64):
    F0 = _f0_mat(mat.albedo, mat.metallic)

    # roughness = max(float(mat.roughness), 0.12)    
    roughness = float(mat.roughness)
    alpha2 = roughness ** 4
    k = alpha2 / 2.0

    S = light.dirs.shape[0]
    V = _norm(cam_pos - frag_pos)
    NdV = (N * V).sum(1).clamp(min=1e-4)

    spec = frag_pos.new_zeros(frag_pos.shape)
    diff_irr = frag_pos.new_zeros(frag_pos.shape)
    F_sum = frag_pos.new_zeros(frag_pos.shape)
    n_valid = frag_pos.new_zeros(frag_pos.shape[0])

    for si in range(0, S, sbatch):
        L_b = light.dirs[si:si + sbatch]
        r_b = light.image_flat[si:si + sbatch]
        dw_b = light.solid_angles[si:si + sbatch]

        NdL_raw = N @ L_b.T
        mask = NdL_raw > 1e-4
        mf = mask.float()
        NdL = NdL_raw.clamp(min=1e-4)

        NdV_e = NdV[:, None]
        LdV = V @ L_b.T

        H_len = (2.0 + 2.0 * LdV).clamp(min=1e-8).sqrt()
        NdH = ((NdL_raw + NdV_e) / H_len).clamp(0, 1)
        VdH = ((LdV + 1.0) / H_len).clamp(0, 1)

        D = _ggx_D(NdH, alpha2)
        F = _schlick_F(VdH, F0)
        G = _smith_G(NdV_e, NdL, k)

        w = (D * G * dw_b / (4 * NdV_e + 1e-7)) * mf

        spec += (F * w[:, :, None] * r_b).sum(1)
        diff_irr += ((NdL_raw.clamp(min=0) * dw_b * mf) @ r_b)

        F_sum += (F * mf[:, :, None]).sum(1)
        n_valid += mf.sum(1)

    F_mean = F_sum / n_valid[:, None].clamp(min=1)
    k_d = (1 - F_mean) * (1 - mat.metallic)
    diff = k_d * mat.albedo / torch.pi * diff_irr

    return (diff + spec).clamp(0, 1)


def _sh_ggx_filtered_radiance(
    coeffs_t:  torch.Tensor,   # (9|16, 3)
    dirs:      torch.Tensor,   # (..., 3)  unit reflection directions
    roughness: torch.Tensor,   # (..., 1)  in [0, 1]
    lut:       Optional[torch.Tensor] = None,   # (N, 3|4); only used when hl_mode="lut"
    hl_mode:   str = "analytic",
) -> torch.Tensor:
    """
    GGX-lobe SH filter — analogue of _sh_phong_filtered_radiance.

    Convolves the SH-encoded environment with the GGX lobe centred on `dirs`,
    with per-element width controlled by `roughness`. The SH order is inferred
    from the coefficient count; order-3 needs a 4th zonal band.

    The zonal band weights h_l come from `hl_mode`: "analytic" (closed form, correct at
    every roughness — the default) or "lut" (the shipped uniform table, bit-identical to
    the previous behaviour, which is under-resolved below roughness ~0.08). See
    `idr.render.brdf.ggx_sh_bands`.

    Returns (..., 3), clamped to ≥ 0.
    """
    order = _sh_order_of(coeffs_t)
    if hl_mode == "lut" and lut is not None and lut.shape[1] < order + 1:
        raise ValueError(f"order-{order} SH needs a {order + 1}-band GGX LUT, "
                         f"got {lut.shape[1]} bands")
    Y = _sh_basis(dirs, order=order)             # (..., 9|16)
    Bvals = ggx_sh_bands(roughness.squeeze(-1), hl_mode, lut, n_bands=order + 1)  # (..., n_bands)

    parts = [Bvals[..., 0:1],                    # band 0 (1 coeff)
             Bvals[..., 1:2].expand(*Bvals.shape[:-1], 3),
             Bvals[..., 2:3].expand(*Bvals.shape[:-1], 5)]
    if order >= 3:
        parts.append(Bvals[..., 3:4].expand(*Bvals.shape[:-1], 7))
    B = torch.cat(parts, dim=-1)                 # (..., 9|16)

    return ((B * Y) @ coeffs_t).clamp(min=0.0)   # (..., 3)


def shade_ct_sh(
    view:             torch.Tensor,
    normals:          torch.Tensor,
    albedo:           torch.Tensor,
    sh_coeffs:        torch.Tensor,
    metallic:         Union[float, torch.Tensor] = 0.0,
    roughness:        Union[float, torch.Tensor] = 0.5,
    lut:              Optional[torch.Tensor] = None,
    diffuse_fresnel:  bool = True,
    return_components: bool = False,
    hl_mode:          str = "analytic",
) -> torch.Tensor:
    """
    Differentiable Cook-Torrance shading with SH-environment specular.

    Parameters
    ----------
    view      : (..., 3)  unit view directions (camera → fragment, or reversed)
    normals   : (..., 3)  unit surface normals
    albedo    : (..., 3)  base colour in [0, 1]
    sh_coeffs : (9, 3)    SH lighting coefficients
    metallic  : scalar or (..., 1)  metallic factor in [0, 1]
    roughness : scalar or (..., 1)  perceptual roughness in [0, 1]
    lut       : optional pre-loaded GGX SH LUT; only consulted when hl_mode="lut"
                (fetched/computed if None), ignored under the analytic default
    diffuse_fresnel : if True, multiply the diffuse by (1-F) on top of
                (1-metallic). Default False to match Blender's Principled BSDF.
    hl_mode   : specular zonal-band source, "analytic" (closed form, correct at every
                roughness — default) or "lut" (the shipped table, bit-identical to the
                previous behaviour but under-resolved below roughness ~0.08). See
                `idr.render.brdf.ggx_sh_bands`.

    Returns
    -------
    (..., 3)  shaded RGB
    """
    device = albedo.device

    def _as_tensor(x: Union[float, torch.Tensor]) -> torch.Tensor:
        if torch.is_tensor(x):
            return x.to(device)
        return torch.full(
            albedo.shape[:-1] + (1,), float(x),
            dtype=albedo.dtype, device=device,
        )

    roughness = _as_tensor(roughness)
    metallic = _as_tensor(metallic)

    if lut is None and hl_mode == "lut":
        lut = _get_ggx_sh_lut(device, n_bands=_sh_order_of(sh_coeffs) + 1)

    # ── Fresnel-Schlick ───────────────────────────────────────────────────────
    # F0: dielectrics reflect ~4%; metals reflect with their albedo tint
    f0 = 0.04 * (1.0 - metallic) + albedo * metallic          # (..., 3)
    NdotV_raw = (normals * view).sum(-1, keepdim=True)
    NdotV = NdotV_raw.clamp(min=0.0)
    F = f0 + (1.0 - f0) * (1.0 - NdotV).pow(5)              # (..., 3)

    # ── Smith G1 — view term  (IBL variant: k = α⁴/2) ────────────────────────
    alpha = roughness ** 2                          # α = roughness²
    k = alpha ** 2 / 2.0                        # k = α²/2
    G1 = NdotV / (NdotV * (1.0 - k) + k + 1e-6)  # (..., 1)

    # ── Diffuse (Lambertian, energy-conserving) ───────────────────────────────
    irr = _sh_irradiance(sh_coeffs, normals)       # (..., 3)
    # Diffuse weight. Default matches Blender's Principled BSDF, which weights the
    # diffuse layer by (1-metallic) only. The (1-F) "energy taken by specular"
    # cross-term darkens the diffuse and was the main cause of the shader being
    # too dark vs BlenderProc; enable diffuse_fresnel=True to restore it.
    k_d = (1.0 - metallic)
    if diffuse_fresnel:
        k_d = (1.0 - F) * k_d
    diff = k_d * albedo / torch.pi * irr

    # ── Specular (GGX SH) ─────────────────────────────────────────────────────
    R = _norm(2.0 * NdotV_raw * normals - view)
    L_spec = _sh_ggx_filtered_radiance(sh_coeffs, R, roughness, lut, hl_mode=hl_mode)
    spec = F * G1 * L_spec / 4.0
    # spec = G1 * L_spec / 4.0

    # import matplotlib.pyplot as plt
    # n = normals[NdotV.squeeze() <= 0][:1].numpy()   # shape: (N, 3)
    # v = view[NdotV.squeeze() <= 0][:1].numpy()   # shape: (N, 3)

    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')

    # ax.quiver(
    #     np.zeros(len(n)),  # x origins
    #     np.zeros(len(n)),  # y origins
    #     np.zeros(len(n)),  # z origins
    #     n[:, 0],            # dx
    #     n[:, 1],            # dy
    #     n[:, 2],            # dz
    #     color="red",
    # )

    # ax.quiver(
    #     np.zeros(len(v)),  # x origins
    #     np.zeros(len(v)),  # y origins
    #     np.zeros(len(v)),  # z origins
    #     v[:, 0],            # dx
    #     v[:, 1],            # dy
    #     v[:, 2],            # dz
    # )

    # plt.show()

    front = (NdotV_raw > 0).to(albedo.dtype)
    composite = (diff + spec) * front
    if not return_components:
        return composite
    return composite, {
        "NdotV":  NdotV,
        "F":      F,
        "G1":     G1,
        "irr":    irr,
        "k_d":    k_d,
        "R":      R,
        "L_spec": L_spec,
        "diff":   diff,
        "spec":   spec,
    }


def _env_bilinear(env_pixels: torch.Tensor, dirs: torch.Tensor) -> torch.Tensor:
    """Differentiable bilinear lookup of a flat (H*W, 3) equirect env map
    (EnvMap layout: W = 2H, texel centers at half-integers, pole = +Y,
    u = atan2(z, x)) at arbitrary unit directions (..., 3).

    Gradients flow into env_pixels (gather) AND into dirs (through the
    bilinear weights), so lobe-shape parameters like roughness receive
    gradients from the env content."""
    P = env_pixels.shape[0]
    H = int(math.sqrt(P / 2))
    W = 2 * H
    assert H * W == P, f"env_pixels ({P}) is not an equirect W=2H grid"
    x, y, z = dirs[..., 0], dirs[..., 1], dirs[..., 2]
    u = (torch.atan2(z, x) / (2 * math.pi) + 0.5) * W - 0.5      # texel coords
    # clamp INSIDE the poles (not [-1, 1]): d/dy acos(y) = -1/sqrt(1-y²) is -inf at y=±1, so a
    # sampled direction hitting the pole backprops NaN (even through masked samples: 0 * nan = nan).
    v = (torch.acos(y.clamp(-1.0 + 1e-6, 1.0 - 1e-6)) / math.pi) * H - 0.5
    u0 = torch.floor(u);  wu = (u - u0).unsqueeze(-1)
    v0 = torch.floor(v);  wv = (v - v0).unsqueeze(-1)
    iu0 = (u0.long() % W); iu1 = (iu0 + 1) % W                    # wrap in phi
    iv0 = v0.long().clamp(0, H - 1); iv1 = (iv0 + 1).clamp(0, H - 1)  # clamp at poles
    p00 = env_pixels[iv0 * W + iu0]
    p01 = env_pixels[iv0 * W + iu1]
    p10 = env_pixels[iv1 * W + iu0]
    p11 = env_pixels[iv1 * W + iu1]
    return ((p00 * (1 - wu) + p01 * wu) * (1 - wv)
            + (p10 * (1 - wu) + p11 * wu) * wv)


def _spec_ggx_importance(
    view:        torch.Tensor,   # (M, 3)
    normals:     torch.Tensor,   # (M, 3)
    env_pixels:  torch.Tensor,   # (H*W, 3)
    F0:          torch.Tensor,   # (M, 3)
    NdotV_raw:   torch.Tensor,   # (M, 1)
    alpha2:      torch.Tensor,   # (M, 1) or scalar tensor — α² with α = roughness²
    k_smith:     torch.Tensor,   # (M, 1) — Smith k, α²/2 (matches quadrature path)
    n_samples:   int = 64,
) -> torch.Tensor:
    """GGX half-vector importance sampling of the specular env integral.

    Deterministic stratified samples (identical across pixels and calls, so
    L-BFGS closures see a noise-free objective): xi1 stratified, xi2 golden-
    ratio sequence. The D term cancels against the pdf, leaving the classic
    estimator  spec = mean_s F(VdH) G(NdV, NdL) VdH / (NdH NdV) * L_env(L).
    Valid at all roughness values (no texel-grid aliasing)."""
    M      = normals.shape[0]
    device = normals.device
    dtype  = normals.dtype
    NdotV  = NdotV_raw.clamp(min=1e-4)                       # (M, 1)
    alpha  = alpha2.clamp(min=1e-12).sqrt()                  # (M, 1) — GGX α

    # tangent frame around N (branchless up-vector selection)
    up   = torch.where(normals[:, 2:3].abs() < 0.999,
                       torch.tensor([0.0, 0.0, 1.0], device=device, dtype=dtype).expand_as(normals),
                       torch.tensor([1.0, 0.0, 0.0], device=device, dtype=dtype).expand_as(normals))
    t1 = _norm(torch.cross(up, normals, dim=-1))             # (M, 3)
    t2 = torch.cross(normals, t1, dim=-1)                    # (M, 3)

    s   = torch.arange(n_samples, device=device, dtype=dtype)
    xi1 = (s + 0.5) / n_samples                              # (S,) stratified
    xi2 = torch.frac(s * 0.6180339887498949)                 # (S,) golden ratio

    # GGX D sampling: cosθ_h = sqrt((1-ξ)/(1+(α²-1)ξ))
    # clamp(min=1e-8) — NOT min=0: d/dx sqrt(x) is +inf at x=0, so a sqrt of a clamp-to-0 quantity
    # backprops NaN at grazing angles (arg→0 as ξ→1 / cosθ_h→1); 1e-8 keeps the gradient finite.
    cos_th = ((1.0 - xi1) / (1.0 + (alpha2 - 1.0) * xi1)).clamp(min=1e-8).sqrt()  # (M, S)
    sin_th = (1.0 - cos_th ** 2).clamp(min=1e-8).sqrt()
    phi    = 2 * math.pi * xi2                                # (S,)
    Hvec = (sin_th * torch.cos(phi)).unsqueeze(-1) * t1.unsqueeze(1) \
         + (sin_th * torch.sin(phi)).unsqueeze(-1) * t2.unsqueeze(1) \
         + cos_th.unsqueeze(-1) * normals.unsqueeze(1)        # (M, S, 3)

    VdH = (view.unsqueeze(1) * Hvec).sum(-1)                  # (M, S)
    L   = 2.0 * VdH.unsqueeze(-1) * Hvec - view.unsqueeze(1)  # (M, S, 3)
    NdL_raw = (normals.unsqueeze(1) * L).sum(-1)              # (M, S)
    mf  = ((NdL_raw > 1e-4) & (VdH > 1e-4)).to(dtype)
    NdL = NdL_raw.clamp(min=1e-4)
    NdH = (normals.unsqueeze(1) * Hvec).sum(-1).clamp(min=1e-4)

    F = _schlick_F(VdH.clamp(0, 1), F0.unsqueeze(1))          # (M, S, 3)
    G = _smith_G(NdotV, NdL, k_smith)                         # (M, S)

    radiance = _env_bilinear(env_pixels, L)                   # (M, S, 3)
    w = (G * VdH / (NdH * NdotV + 1e-7)) * mf                 # (M, S)
    return (F * w.unsqueeze(-1) * radiance).mean(1)           # (M, 3)


def shade_ct_env(
    view:              torch.Tensor,
    normals:           torch.Tensor,
    albedo:            torch.Tensor,
    env_pixels:        torch.Tensor,       # (P, 3) — may be learnable
    env_dirs:          torch.Tensor,       # (P, 3)
    env_dw:            torch.Tensor,       # (P,)
    metallic:          Union[float, torch.Tensor] = 0.0,
    roughness:         Union[float, torch.Tensor] = 0.5,
    sbatch:            int = 64,
    diffuse_fresnel:   bool = True,
    return_components: bool = False,
    spec_importance:   bool = False,
    spec_samples:      int = 64,
) -> torch.Tensor:
    """
    Differentiable Cook-Torrance shading with explicit env-map integration.

    Drop-in companion to shade_ct_sh: same flat (M, 3) interface, same front-face
    masking, but integrates over explicit (P, 3) env-map samples instead of SH.
    Gradients flow through env_pixels, albedo, metallic, and roughness.

    Parameters
    ----------
    view, normals, albedo : (M, 3)  flat foreground-pixel arrays
    env_pixels            : (P, 3)  env-map radiance (learnable)
    env_dirs              : (P, 3)  sample directions (unit sphere)
    env_dw                : (P,)    solid angles
    metallic, roughness   : scalar or (M, 1) Tensor
    sbatch                : env-map samples per memory batch
    diffuse_fresnel       : if True, multiply diffuse by (1-F) on top of
                            (1-metallic). Default False (matches Blender).
    return_components     : if True, return (composite, dict)
    spec_importance       : if True, compute the SPECULAR term by GGX
                            half-vector importance sampling (spec_samples
                            deterministic stratified samples, bilinear env
                            lookup) instead of the texel-grid Riemann sum.
                            The Riemann sum aliases for lobes narrower than a
                            texel (roughness ≲ 0.3 on the 32x64 grid, with a
                            float32 instability peaking near roughness 0.1);
                            importance sampling stays valid at ALL roughness.
                            Requires env_pixels on an equirect grid with
                            W = 2H (the EnvMap layout). Diffuse is unchanged.

    Returns
    -------
    (M, 3) shaded RGB, optionally paired with a components dict containing
    NdotV (M,1), F_avg (M,3), irr (M,3), k_d (M,3), diff (M,3), spec (M,3).
    """
    device = albedo.device

    def _as_tensor(x):
        if torch.is_tensor(x):
            return x.to(device)
        return torch.full(
            albedo.shape[:-1] + (1,), float(x),
            dtype=albedo.dtype, device=device,
        )

    metallic_t  = _as_tensor(metallic)
    # roughness_t = _as_tensor(roughness).clamp(min=0.05)
    roughness_t = _as_tensor(roughness)
    alpha2      = roughness_t ** 4
    k_smith     = alpha2 / 2.0

    F0        = _f0_mat(albedo, metallic_t)                   # (M, 3)
    NdotV_raw = (normals * view).sum(-1, keepdim=True)        # (M, 1)
    NdotV     = NdotV_raw.clamp(min=1e-4)                     # (M, 1)

    P         = env_pixels.shape[0]
    spec      = albedo.new_zeros(albedo.shape)
    diff_irr  = albedo.new_zeros(albedo.shape)
    F_sum     = albedo.new_zeros(albedo.shape)
    n_valid   = albedo.new_zeros(albedo.shape[0])

    for si in range(0, P, sbatch):
        L_b  = env_dirs[si:si + sbatch]             # (B, 3)
        r_b  = env_pixels[si:si + sbatch]           # (B, 3)
        dw_b = env_dw[si:si + sbatch]               # (B,)

        NdL_raw = normals @ L_b.T                   # (M, B)
        mf      = (NdL_raw > 1e-4).float()
        NdL     = NdL_raw.clamp(min=1e-4)

        LdV   = view @ L_b.T                        # (M, B)
        H_len = (2.0 + 2.0 * LdV).clamp(min=1e-8).sqrt()
        NdH   = ((NdL_raw + NdotV) / H_len).clamp(0, 1)
        VdH   = ((LdV + 1.0)       / H_len).clamp(0, 1)

        F = _schlick_F(VdH, F0.unsqueeze(1))                  # (M, B, 3)

        if not spec_importance:
            D = _ggx_D(NdH, alpha2)                           # (M, B)
            G = _smith_G(NdotV, NdL, k_smith)                 # (M, B)
            w = (D * G * dw_b / (4 * NdotV + 1e-7)) * mf      # (M, B)
            spec += (F * w[:, :, None] * r_b).sum(1)
        diff_irr += (NdL_raw.clamp(min=0) * dw_b * mf) @ r_b
        F_sum    += (F * mf[:, :, None]).sum(1)
        n_valid  += mf.sum(1)

    if spec_importance:
        spec = _spec_ggx_importance(view, normals, env_pixels, F0,
                                    NdotV_raw, alpha2, k_smith, spec_samples)

    F_mean    = F_sum / n_valid[:, None].clamp(min=1)         # (M, 3)
    # Diffuse weight: (1-metallic) by default (matches Blender's Principled BSDF);
    # pass diffuse_fresnel=True to also apply the (1-F) cross-term.
    k_d       = (1.0 - metallic_t)                            # (M, 3)
    if diffuse_fresnel:
        k_d = (1.0 - F_mean) * k_d
    diff      = k_d * albedo / torch.pi * diff_irr            # (M, 3)
    front     = (NdotV_raw > 0).to(albedo.dtype)
    composite = (diff + spec) * front
    if not return_components:
        return composite
    return composite, {
        "NdotV": NdotV,
        "F_avg": F_mean,
        "irr":   diff_irr,
        "k_d":   k_d,
        "diff":  diff,
        "spec":  spec,
    }
