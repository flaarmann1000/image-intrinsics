"""Conditional refinement of albedo alone, with lighting and the rest of the material fixed.

VarPro eliminates the lighting because the render is *linear* in it. Albedo is the next
best candidate — but not linear: `F0 = 0.04(1-m) + rho*m` makes the specular term depend
on albedo too, so the render is **quadratic** in albedo whenever metallic > 0, and there
is no single closed-form solve. What there is instead is a very well-behaved sub-problem:

* it is separable per pixel (pixel p's residual sees only pixel p's albedo), and
* separable per channel (albedo_c only enters channel c),

so it reduces to independent 3x3 damped Gauss-Newton solves, warm-started from the
current estimate. A handful of steps gets essentially to the conditional optimum.

This is deliberately *not* a trust region: there is no accept/reject, only a small
relative ridge for conditioning. The sub-problem is smooth and the starting point is
already good, so the extra machinery would cost more than it saves.

Natural space only. The profiling exploits the render's structure in albedo directly;
under a sigmoid reparameterisation that structure is gone, which is why
`varpro_space="transformed"` rejects `varpro_profile_rho` rather than silently skipping.
"""
from __future__ import annotations

import torch

from idr.render.brdf import _lut_lookup
from .core import _band_expand_1d

__all__ = ["refine_rho"]


def _px_render_rho_only(rho_p, rough_p, metal_p, AY_p, YR_p, NdotV_p, front_p,
                        ad_p, as_p, sh, lut, sh_order, dfres):
    """Single-pixel render as a function of albedo ONLY.

    Roughness and metallic are per-pixel constants here — the point of the sub-problem.
    Returns `(out, out)` so `jacfwd(..., has_aux=True)` hands back the primal for free:
    forward-mode AD computes it anyway, so a separate forward pass would be pure waste.
    """
    f0 = 0.04 * (1.0 - metal_p) + rho_p * metal_p          # (3,)
    F = f0 + (1.0 - f0) * (1.0 - NdotV_p).pow(5)           # (3,)
    alpha = rough_p ** 2
    G1 = NdotV_p / (NdotV_p * (1.0 - alpha ** 2 / 2.0) + alpha ** 2 / 2.0 + 1e-6)
    k_d = 1.0 - metal_p
    if dfres:
        k_d = (1.0 - F) * k_d
    diff_w = front_p * k_d * rho_p / torch.pi              # (3,)
    spec_w = front_p * F * G1 / 4.0                        # (3,)

    BY = _band_expand_1d(_lut_lookup(lut, rough_p).reshape(-1), sh_order) * YR_p
    raw_d = torch.einsum('i,nic->nc', AY_p, sh)            # (n, 3)
    raw_s = torch.einsum('i,nic->nc', BY, sh)
    out = diff_w[None, :] * raw_d * ad_p + spec_w[None, :] * raw_s * as_p
    return out, out


def refine_rho(geom, material, sh, active, observations, n_steps=10, ridge_rel=1e-6,
               lower=0.0, upper=1.0, chunk=0):
    """-> refined albedo (M, 3). `material` is (M,5) NATURAL [albedo(3), roughness, metallic]."""
    if n_steps <= 0:
        return material[:, :3].clone()

    M = material.shape[0]
    a_d, a_s = active
    rho = material[:, :3].clone()
    rough, metal = material[:, 3], material[:, 4]
    lut, sh_order, dfres = geom.lut, geom.sh_order, geom.diffuse_fresnel
    step = chunk if chunk and chunk > 0 else M

    jac = torch.func.jacfwd(_px_render_rho_only, argnums=0, has_aux=True)
    jac_v = torch.func.vmap(
        jac, in_dims=(0, 0, 0, 0, 0, 0, 0, 0, 0, None, None, None, None))

    for _ in range(n_steps):
        for lo in range(0, M, step):
            hi = min(lo + step, M)
            sl = slice(lo, hi)
            # Shapes must match core.build_px_render's: roughness/metallic/NdotV/front
            # arrive as (1,), NOT 0-dim. `_lut_lookup` indexes its table with the
            # roughness, and under vmap a 0-dim index trips the data-dependent-control-flow
            # guard; keeping the trailing dim also makes the (3,)-broadcast of the
            # Fresnel term identical to the main renderer's.
            D_rho, recon = jac_v(
                rho[sl], material[sl, 3:4], material[sl, 4:5], geom.AY[sl], geom.Y_R[sl],
                geom.NdotV[sl], geom.front[sl],
                a_d[:, sl, :].permute(1, 0, 2).contiguous(),
                a_s[:, sl, :].permute(1, 0, 2).contiguous(),
                sh, lut, sh_order, dfres)                       # (b,n,3,3), (b,n,3)
            r = recon - observations[:, sl, :].permute(1, 0, 2)
            g = torch.einsum('pnck,pnc->pk', D_rho, r)          # (b,3)
            H = torch.einsum('pnck,pncl->pkl', D_rho, D_rho)    # (b,3,3)
            ridge = ridge_rel * H.diagonal(dim1=1, dim2=2).clamp_min(1e-12)
            K = H + torch.diag_embed(ridge)
            delta = torch.linalg.solve(K, (-g).unsqueeze(-1)).squeeze(-1)
            rho[sl] = torch.clamp(rho[sl] + delta, lower, upper)
    return rho
