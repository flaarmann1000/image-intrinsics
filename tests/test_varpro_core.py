#!/usr/bin/env python
"""The chunked Woodbury step must equal a dense, independently-built reduced solve.

`core.woodbury_step` is the part of VarPro with no obvious failure mode: a wrong
projector, a transposed inverse or a mis-ordered index all produce a plausible-looking
step that simply converges somewhere else. Two such bugs were found by exactly this
test and would not have been visible any other way:

  * `A^T r` was not zero, because on a non-converged active set the lighting solve
    returned `sh*` computed at the PREVIOUS mask while reporting the NEW one.
  * `U` carried `L^-1 L^-T` instead of `L^-T L^-1` — an index transposition that leaves
    `U U^T` symmetric and positive semi-definite, hence entirely innocuous-looking.

So the reference here is built from scratch: assemble the full Jacobian `J`, the block
projector `P_A` per (image, channel), and solve `(J^T(I-P)J + lam·diag)δ = -J^T r`
densely. Small `M` keeps that tractable.

    python tests/test_varpro_core.py
"""
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from idr.optim.varpro.core import (build_px_render, woodbury_step,       # noqa: E402
                                   _chol_inv_lower)
from idr.optim.varpro.design import build_design_split, forward_from_design  # noqa: E402
from idr.optim.varpro.lighting import solve_lighting_active_set          # noqa: E402
from tests.test_varpro_design import make_geometry                       # noqa: E402

RTOL = 1e-8
LAM = 1e-3


def setup(M, n, sh_order, diffuse_fresnel, device, dtype, seed=3):
    geom, _, _ = make_geometry(M, sh_order, diffuse_fresnel, device, dtype)
    g = torch.Generator(device="cpu").manual_seed(seed)
    n_sh = 9 if sh_order == 2 else 16
    alb = (torch.rand(M, 3, generator=g) * .7 + .15).to(device, dtype)
    rou = (torch.rand(M, 1, generator=g) * .7 + .15).to(device, dtype)
    met = (torch.rand(M, 1, generator=g) * .3).to(device, dtype)
    sh_t = torch.randn(n, n_sh, 3, generator=g).to(device, dtype)
    sh_t[:, 0, :] = sh_t[:, 0, :].abs() * .9
    obs, _ = forward_from_design(geom, alb, met, rou, sh_t)
    obs = obs + 0.02 * torch.randn(obs.shape, generator=g).to(device, dtype)
    # start away from the truth, so the step is not trivially zero
    mat = torch.cat([(alb * .8).clamp(0, 1), (rou * 1.3).clamp(.03, 1), met], dim=-1)
    return geom, mat, obs


def dense_reference(geom, mat, sh, active, obs, lam):
    """Assemble J and P densely and solve the reduced system directly."""
    M = mat.shape[0]
    n, n_sh = sh.shape[0], sh.shape[1]
    dev, dt = mat.device, mat.dtype
    ad, asx = active
    px = build_px_render(geom, "natural", None)
    jv = torch.func.vmap(torch.func.jacfwd(px, argnums=0),
                         in_dims=(0, 0, 0, 0, 0, 0, 0, None))
    fv = torch.func.vmap(px, in_dims=(0, 0, 0, 0, 0, 0, 0, None))
    args = (mat, geom.AY, geom.Y_R, geom.NdotV, geom.front,
            ad.permute(1, 0, 2).contiguous(), asx.permute(1, 0, 2).contiguous())
    Dj = jv(*args, sh)
    r = fv(*args, sh) - obs.permute(1, 0, 2)

    D_, S_ = build_design_split(geom, mat[:, :3], mat[:, 4:5], mat[:, 3:4])
    A = (ad.permute(1, 0, 2)[:, :, :, None] * D_[:, None, :, :] +
         asx.permute(1, 0, 2)[:, :, :, None] * S_[:, None, :, :])       # (M,n,3,n_sh)

    NR = n * 3 * M
    J = torch.zeros(NR, 5 * M, dtype=dt, device=dev)
    for p in range(M):
        blk = Dj[p].reshape(n * 3, 5)
        for k in range(n * 3):
            J[k * M + p, 5 * p:5 * p + 5] = blk[k]
    rv = torch.zeros(NR, dtype=dt, device=dev)
    for p in range(M):
        rv[torch.arange(n * 3, device=dev) * M + p] = r[p].reshape(-1)

    P = torch.zeros(NR, NR, dtype=dt, device=dev)
    for i in range(n):
        for c in range(3):
            Ab = A[:, i, c, :]
            P_blk = Ab @ torch.linalg.pinv(Ab.T @ Ab) @ Ab.T
            idx = (i * 3 + c) * M + torch.arange(M, device=dev)
            P[idx[:, None], idx[None, :]] = P_blk

    JPJ = J.T @ P @ J
    H = J.T @ J - JPJ
    grad = J.T @ rv
    Hd = H + torch.diag(lam * H.diagonal().clamp_min(1e-12))
    delta = torch.linalg.solve(Hd, -grad).reshape(M, 5)
    return delta, A, Dj, r, JPJ, Hd, grad


def case(name, M, n, sh_order, diffuse_fresnel, chunks, device, dtype):
    geom, mat, obs = setup(M, n, sh_order, diffuse_fresnel, device, dtype)
    sh, active, info = solve_lighting_active_set(
        geom, mat[:, :3], mat[:, 4:5], mat[:, 3:4], obs)

    # The lighting solve must return a self-consistent triple: sh* is the least-squares
    # solution for the design implied by the returned masks, so the residual is
    # orthogonal to the lighting column space. Everything downstream assumes this.
    d_ref, A, Dj, r, JPJ, Hd, grad = dense_reference(geom, mat, sh, active, obs, LAM)
    orth = max(float((A[:, i, c, :].T @ r[:, i, c]).abs().max())
               for i in range(n) for c in range(3))
    r_scale = float(r.abs().max())
    orth_ok = orth <= 1e-10 * max(r_scale, 1.0)

    # U U^T must equal J^T P J — the actual content of the low-rank correction. This one
    # IS tight: it is an algebraic identity, unaffected by the conditioning below.
    n_sh = sh.shape[1]
    Li = _chol_inv_lower(info["gram"], n_sh).reshape(n, 3, n_sh, n_sh)
    AL = torch.einsum('bncj,ncij->bnci', A, Li)
    U = torch.einsum('bnca,bnci->banci', Dj, AL).reshape(M, 5, n * 3 * n_sh)
    UUt = torch.einsum('pam,qbm->paqb', U, U).reshape(5 * M, 5 * M)
    u_err = float((UUt - JPJ).abs().max()) / max(float(JPJ.abs().max()), 1e-30)

    cond = float(torch.linalg.cond(Hd))
    gn = float(grad.norm())
    ok = orth_ok and u_err < 1e-10
    print(f"  {name:44} |A^T r|={orth:.1e}  |UU^T-J^TPJ|={u_err:.1e}  cond(H)={cond:.1e}")
    for ck in chunks:
        delta, _ = woodbury_step(geom, mat, sh, active, info["gram"], obs, LAM, chunk=ck)
        # The reduced Hessian is near-singular BY CONSTRUCTION -- projecting out the
        # lighting leaves flat directions (the albedo/lighting scale ambiguity among
        # them), and cond(H) ~ 1e15 here. Comparing raw deltas is then meaningless: two
        # correct solvers disagree freely in the null space. What must hold is that the
        # step SOLVES the system, so score it by the normal-equation residual, and report
        # the raw difference for information only.
        res = float((Hd @ delta.reshape(-1) + grad).norm()) / max(gn, 1e-30)
        raw = float((delta - d_ref).abs().max()) / max(float(d_ref.abs().max()), 1e-30)
        good = res < RTOL
        ok = ok and good
        print(f"      chunk={ck:<4} normal-eq residual = {res:.2e}  {'OK' if good else 'FAIL'}"
              f"   (raw delta diff {raw:.1e}, null-space)")
    return ok


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float64
    print(f"  device={device} dtype={dtype}\n")
    res = [
        case("order=2 diffuse_fresnel=True", 48, 3, 2, True, (16, 48), device, dtype),
        case("order=2 diffuse_fresnel=False", 48, 3, 2, False, (16, 48), device, dtype),
        case("order=3 diffuse_fresnel=True", 40, 2, 3, True, (13, 40), device, dtype),
    ]
    print()
    if all(res):
        print("  chunked Woodbury matches the dense reduced solve")
        return 0
    print(f"  {res.count(False)}/{len(res)} case(s) FAILED")
    return 1


if __name__ == "__main__":
    sys.exit(main())
