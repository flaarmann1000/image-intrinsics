#!/usr/bin/env python
"""The design matrix must reproduce the renderer, exactly.

Variable projection replaces `shade_ct_sh` with a contraction of a per-pixel design
matrix against the SH coefficients. If that contraction is not the same function, every
result downstream -- the closed-form lighting solve, the Woodbury Jacobian, the fitted
materials -- is a solution to the wrong problem, and nothing else in the pipeline would
notice: the optimiser would converge happily to the wrong answer.

So this checks

    forward_from_design(geom, material, sh)  ==  shade_ct_sh(..., material, sh)

over SH order 2 and 3, `diffuse_fresnel` on and off, and materials chosen to exercise the
clamps, the front-facing mask, and the metallic/dielectric Fresnel branches.

    python tests/test_varpro_design.py
"""
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from idr.optim.varpro.design import VarProGeometry, forward_from_design   # noqa: E402
from idr.render import shade_ct_sh                                        # noqa: E402
from idr.render.brdf import _get_ggx_sh_lut                               # noqa: E402
from idr.render.ops import _norm                                          # noqa: E402
from idr.render.sh import _sh_basis                                       # noqa: E402

TOL = 1e-6
SEED = 20260722


def make_geometry(M, sh_order, diffuse_fresnel, device, dtype):
    """Random view/normal geometry, built exactly as ct_sh._forward precomputes it."""
    g = torch.Generator(device="cpu").manual_seed(SEED)
    N = _norm(torch.randn(M, 3, generator=g).to(device, dtype))
    V = _norm(torch.randn(M, 3, generator=g).to(device, dtype))

    A_vals = [torch.pi,
              2 * torch.pi / 3, 2 * torch.pi / 3, 2 * torch.pi / 3,
              torch.pi / 4, torch.pi / 4, torch.pi / 4, torch.pi / 4, torch.pi / 4]
    if sh_order >= 3:
        A_vals += [0.0] * 7
    A = N.new_tensor(A_vals)

    NdotV_raw = (N * V).sum(-1, keepdim=True)
    NdotV = NdotV_raw.clamp(min=0.0)
    R = _norm(2.0 * NdotV_raw * N - V)
    n_bands = sh_order + 2 if sh_order >= 3 else 3
    lut = _get_ggx_sh_lut(device, n_bands=n_bands).to(dtype)

    geom = VarProGeometry(
        AY=A * _sh_basis(N, order=sh_order),
        Y_R=_sh_basis(R, order=sh_order),
        NdotV=NdotV,
        front=(NdotV_raw > 0).to(dtype),
        lut=lut,
        sh_order=sh_order,
        diffuse_fresnel=diffuse_fresnel,
    )
    return geom, N, V


def reference(N, V, albedo, metallic, roughness, sh, lut, diffuse_fresnel, front):
    """shade_ct_sh per image, assembled to (n, M, 3) and masked like ct_sh._forward."""
    outs = []
    for k in range(sh.shape[0]):
        outs.append(shade_ct_sh(V, N, albedo, sh[k], metallic, roughness,
                                lut=lut, diffuse_fresnel=diffuse_fresnel))
    return torch.stack(outs, 0) * front


def case(name, M, n_img, sh_order, diffuse_fresnel, metallic_mode, device, dtype):
    geom, N, V = make_geometry(M, sh_order, diffuse_fresnel, device, dtype)
    g = torch.Generator(device="cpu").manual_seed(SEED + 1)
    n_sh = 9 if sh_order == 2 else 16

    albedo = torch.rand(M, 3, generator=g).to(device, dtype) * 0.8 + 0.1
    roughness = torch.rand(M, 1, generator=g).to(device, dtype) * 0.9 + 0.05
    if metallic_mode == "zero":
        metallic = torch.zeros(M, 1, device=device, dtype=dtype)
    elif metallic_mode == "one":
        metallic = torch.ones(M, 1, device=device, dtype=dtype)
    else:
        metallic = torch.rand(M, 1, generator=g).to(device, dtype)

    # Lighting with a strong negative lobe, so both clamps are genuinely exercised
    # rather than being no-ops on smooth positive illumination.
    sh = torch.randn(n_img, n_sh, 3, generator=g).to(device, dtype)
    sh[:, 0, :] = sh[:, 0, :].abs() * 0.6

    got, (a_d, a_s) = forward_from_design(geom, albedo, metallic, roughness, sh)
    want = reference(N, V, albedo, metallic, roughness, sh, geom.lut,
                     diffuse_fresnel, geom.front)

    err = (got - want).abs().max().item()
    scale = want.abs().max().item()
    clamped_d = 1.0 - a_d.mean().item()
    clamped_s = 1.0 - a_s.mean().item()
    ok = err <= TOL * max(scale, 1.0)
    print(f"  {name:52} max|d|={err:.3e}  clamped d/s={clamped_d:5.1%}/{clamped_s:5.1%}"
          f"  {'OK' if ok else 'FAIL'}")
    if clamped_d < 0.01 and clamped_s < 0.01:
        print("      ! neither clamp fired -- this case does not test the active set")
    return ok


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float64          # exactness check, not a performance check
    print(f"  device={device} dtype={dtype}  tol={TOL}\n")
    results = []
    for sh_order in (2, 3):
        for df in (True, False):
            for mm in ("zero", "mixed", "one"):
                results.append(case(
                    f"order={sh_order} diffuse_fresnel={str(df):5} metallic={mm}",
                    M=512, n_img=4, sh_order=sh_order, diffuse_fresnel=df,
                    metallic_mode=mm, device=device, dtype=dtype))
    print()
    if all(results):
        print(f"  design matrix reproduces shade_ct_sh in all {len(results)} cases")
        return 0
    print(f"  {results.count(False)}/{len(results)} case(s) FAILED")
    return 1


if __name__ == "__main__":
    sys.exit(main())
