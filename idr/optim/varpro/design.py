"""The per-pixel linear design of the CT+SH forward model in the SH coefficients.

Variable projection needs the render written as a matrix acting on the lighting. With
the material held fixed, `shade_ct_sh` is linear in the SH coefficients:

    recon[n,p,c] = front[p] * ( diff_w[p,c] * clamp(<AY[p], sh[n,:,c]>, 0)
                              + spec_w[p,c] * clamp(<BY[p], sh[n,:,c]>, 0) )

so with the two clamps replaced by 0/1 active indicators it becomes a plain contraction

    recon[n,p,c] = < a_d[n,p,c]*D[p,c,:] + a_s[n,p,c]*S[p,c,:] ,  sh[n,:,c] >

with the PRE-CLAMP designs

    D[p,c,i] = front[p] * k_d[p,c] * albedo[p,c]/pi * AY[p,i]        (diffuse)
    S[p,c,i] = front[p] * F[p,c] * G1[p] / 4       * BY[p,i]         (specular)

`AY`, `Y_R`, `NdotV` and `front` are geometry-only and are already computed once per run
by idr/optim/models/ct_sh.py under "Precompute geometry-only terms" -- this module takes
them rather than recomputing, so there is exactly one definition of the geometry.

NOTE the roughness floor is deliberately NOT applied here. It is a bound on the
optimizer's iterate, not part of the model; clamping inside the design would make it
disagree with what `shade_ct_sh` actually renders, and the equivalence test in
tests/test_varpro_design.py exists to catch precisely that class of drift.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from idr.render.brdf import _lut_lookup

__all__ = ["VarProGeometry", "build_design_split", "contract", "forward_from_design"]


@dataclass
class VarProGeometry:
    """Geometry-only terms, constant for the whole optimisation.

    AY    (M, n_sh)  A_l * Y_lm(N)   diffuse SH basis, Lambertian ZH-weighted
    Y_R   (M, n_sh)  Y_lm(R)         specular SH basis along the reflection vector
    NdotV (M, 1)     clamped N.V
    front (M, 1)     front-facing mask, 1.0 / 0.0
    lut   (L, bands) GGX-SH lookup table
    """
    AY: torch.Tensor
    Y_R: torch.Tensor
    NdotV: torch.Tensor
    front: torch.Tensor
    lut: torch.Tensor
    sh_order: int = 2
    diffuse_fresnel: bool = True

    @property
    def n_sh(self) -> int:
        return self.AY.shape[-1]

    @property
    def M(self) -> int:
        return self.AY.shape[0]


def _band_expand(Bvals: torch.Tensor, sh_order: int) -> torch.Tensor:
    """(M, n_bands) zonal weights -> (M, n_sh), each band repeated 2l+1 times.

    Mirrors the expansion in ct_sh._forward and _sh_ggx_filtered_radiance; the reference
    implementation hardcodes three bands, this follows the configured SH order.
    """
    parts = [Bvals[..., 0:1],
             Bvals[..., 1:2].expand(-1, 3),
             Bvals[..., 2:3].expand(-1, 5)]
    if sh_order >= 3:
        parts.append(Bvals[..., 3:4].expand(-1, 7))
    return torch.cat(parts, dim=-1)


def build_design_split(geom: VarProGeometry,
                       albedo: torch.Tensor,       # (M, 3)
                       metallic: torch.Tensor,     # (M, 1)
                       roughness: torch.Tensor):   # (M, 1)
    """-> (D, S), each (M, 3, n_sh): the pre-clamp linear coefficient on each SH basis
    function, split into the diffuse and specular contributions."""
    NdotV = geom.NdotV
    f0 = 0.04 * (1.0 - metallic) + albedo * metallic                  # (M, 3)
    F = f0 + (1.0 - f0) * (1.0 - NdotV).pow(5)                        # (M, 3)

    alpha = roughness ** 2
    G1 = NdotV / (NdotV * (1.0 - alpha ** 2 / 2.0) + alpha ** 2 / 2.0 + 1e-6)   # (M, 1)

    k_d = 1.0 - metallic                                              # (M, 1)
    if geom.diffuse_fresnel:
        k_d = (1.0 - F) * k_d                                         # (M, 3)
    diff_w = geom.front * k_d * albedo / torch.pi                     # (M, 3)
    spec_w = geom.front * F * G1 / 4.0                                # (M, 3)

    BY = _band_expand(_lut_lookup(geom.lut, roughness.squeeze(-1)),
                      geom.sh_order) * geom.Y_R                       # (M, n_sh)

    D = diff_w[:, :, None] * geom.AY[:, None, :]                      # (M, 3, n_sh)
    S = spec_w[:, :, None] * BY[:, None, :]                           # (M, 3, n_sh)
    return D, S


def contract(design: torch.Tensor, sh: torch.Tensor) -> torch.Tensor:
    """(M, 3, n_sh) x (n, n_sh, 3) -> (n, M, 3).

    `sh` follows this repo's coefficient-major (n, n_sh, 3) convention, not the
    reference implementation's (n, 3, n_sh).
    """
    return torch.einsum('pci,nic->npc', design, sh)


def forward_from_design(geom: VarProGeometry,
                        albedo: torch.Tensor,
                        metallic: torch.Tensor,
                        roughness: torch.Tensor,
                        sh: torch.Tensor,
                        active: Optional[tuple] = None):
    """Reconstruct through the design matrices. -> (recon, (a_d, a_s))

    With `active=None` the clamps are evaluated from the current contraction, which
    reproduces `shade_ct_sh` exactly. Passing a fixed `(a_d, a_s)` instead freezes the
    active set, which is what makes the lighting sub-problem linear.
    """
    D, S = build_design_split(geom, albedo, metallic, roughness)
    raw_d = contract(D, sh)                                           # (n, M, 3)
    raw_s = contract(S, sh)
    if active is None:
        a_d = (raw_d >= 0).to(raw_d.dtype)
        a_s = (raw_s >= 0).to(raw_s.dtype)
    else:
        a_d, a_s = active
    return raw_d * a_d + raw_s * a_s, (a_d, a_s)
