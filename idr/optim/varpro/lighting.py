"""Closed-form SH lighting solve — the block variable projection eliminates.

With the material fixed the render is linear in the SH coefficients, so the optimal
lighting is a least-squares solution. Two wrinkles:

**The clamps.** `shade_ct_sh` clamps both the diffuse irradiance and the specular
radiance at zero, which destroys linearity. Handled by an active-set fixed point: hold
the 0/1 indicators fixed, solve the resulting linear problem, recompute the indicators,
repeat until the set stops changing (a handful of iterations in practice). The reference
implementation needs this only for the specular clamp because its diffuse term is
unclamped; this repo clamps both, so both are in the active set.

**Scale.** The reference builds the full design `(n_images, n_pixels, 9)` and calls
`torch.linalg.lstsq`. That is fine at its scene size but not here: 100 images x 50k
masked pixels x 9 coefficients in fp64 is over a terabyte. Since the unknown per
(image, channel) is only 9 or 16 numbers, the normal equations are instead accumulated
over image chunks -- (n_sh, n_sh) per system, kilobytes total.

Forming A^T A squares the condition number, so the solve wants fp64. It does NOT want a
blanket ridge: measured on exact data, ridge 0 recovers the lighting to 1.5e-14 while a
relative ridge of 1e-8 only reaches 3.0e-7 -- seven digits paid for protection that
almost every system does not need. The ridge is therefore applied per-system, only where
the Cholesky factorisation actually fails.
"""
from __future__ import annotations

import torch

from .design import build_design_split, contract

__all__ = ["solve_lighting_active_set"]


def _solve_normal(AtA: torch.Tensor, Atb: torch.Tensor, ridge: float):
    """Batched SPD solve. AtA (B,k,k), Atb (B,k) -> ((B,k), n_ridged).

    The ridge is applied ONLY to systems that are not positive definite, because it is
    not free: a relative ridge of 1e-8 costs about seven digits in the recovered
    lighting (measured -- ridge 0 recovers exact-data SH to 1.5e-14, ridge 1e-8 to
    3.0e-7). Most systems are well conditioned, so ridging them all to protect the few
    degenerate ones throws away most of fp64's headroom for nothing.

    `cholesky_ex` reports per-system failure rather than raising, which is what makes
    the split cheap: factor once, ridge only the systems that need it.
    """
    L, info = torch.linalg.cholesky_ex(AtA)
    bad = info != 0
    n_bad = int(bad.sum())
    if n_bad == 0:
        return torch.cholesky_solve(Atb.unsqueeze(-1), L).squeeze(-1), 0

    k = AtA.shape[-1]
    eye = torch.eye(k, dtype=AtA.dtype, device=AtA.device)
    # Trace-relative, so the ridge means the same thing at any radiometric scale.
    scale = AtA.diagonal(dim1=-2, dim2=-1).sum(-1).clamp_min(1e-30) / k
    A = AtA + torch.where(bad, ridge * scale, torch.zeros_like(scale))[:, None, None] * eye
    L2, info2 = torch.linalg.cholesky_ex(A)
    if int((info2 != 0).sum()) == 0:
        return torch.cholesky_solve(Atb.unsqueeze(-1), L2).squeeze(-1), n_bad
    # Still singular: an unlit or degenerate pixel set leaves SH directions
    # unobservable. lstsq returns the minimum-norm solution, which is the right answer
    # for an unconstrained direction.
    return torch.linalg.lstsq(A, Atb.unsqueeze(-1)).solution.squeeze(-1), n_bad


def solve_lighting_active_set(geom,
                              albedo, metallic, roughness,
                              observations,                 # (n, M, 3)
                              max_iters: int = 8,
                              ridge: float = 1e-10,
                              flip_tol: float = 1e-9,
                              img_chunk: int = 0,
                              return_active: bool = True):
    """Least-squares SH lighting for the given material, respecting both clamps.

    Returns `sh_star` (n, n_sh, 3) and, when `return_active`, the converged
    `(a_d, a_s)` indicators — the caller needs them to build a Jacobian that is
    consistent with the lighting it was handed.
    """
    D, S = build_design_split(geom, albedo, metallic, roughness)       # (M,3,n_sh)
    n_img, M, _ = observations.shape
    n_sh = geom.n_sh
    dev, dt = D.device, D.dtype
    chunk = img_chunk if img_chunk and img_chunk > 0 else n_img

    # Start with every entry active, i.e. "no clamping" — the same initial guess the
    # reference uses, and the correct one whenever the lighting is mostly positive.
    a_d = torch.ones((n_img, M, 3), dtype=dt, device=dev)
    a_s = torch.ones_like(a_d)

    # A^T A alongside the solution, kept for the Woodbury correction: the reduced Hessian
    # needs a projector onto the lighting column space, and (A^T A)^-1 is exactly that
    # projector's inverse Gram. Reusing it is what lets core.py stay exact under the
    # active set -- the reference instead reuses an UNCLAMPED orthonormal basis, leaving
    # its projector slightly inconsistent with the lighting it was given.
    def _solve_at(ad, asx):
        sh = torch.zeros((n_img, n_sh, 3), dtype=dt, device=dev)
        gm = torch.zeros((n_img, 3, n_sh, n_sh), dtype=dt, device=dev)
        nr = 0
        for lo in range(0, n_img, chunk):
            hi = min(lo + chunk, n_img)
            # Q[k,p,c,i] = a_d*D + a_s*S — the per-image design under the active set.
            Q = (ad[lo:hi, :, :, None] * D[None] +
                 asx[lo:hi, :, :, None] * S[None])                     # (b,M,3,n_sh)
            AtA = torch.einsum('bpci,bpcj->bcij', Q, Q)                # (b,3,n_sh,n_sh)
            Atb = torch.einsum('bpci,bpc->bci', Q, observations[lo:hi])  # (b,3,n_sh)
            b = hi - lo
            sol, nb = _solve_normal(AtA.reshape(b * 3, n_sh, n_sh),
                                    Atb.reshape(b * 3, n_sh), ridge)
            nr += nb
            gm[lo:hi] = AtA
            sh[lo:hi] = sol.reshape(b, 3, n_sh).permute(0, 2, 1)
        return sh, gm, nr

    # The loop is arranged so `sh_star`/`gram` are ALWAYS the solve at the CURRENT
    # (a_d, a_s) — on early exit and on exhaustion alike. That matters more than reaching
    # a true fixed point: the Woodbury correction only requires that sh* be the
    # least-squares solution for the design it is handed (so that A^T r = 0). Returning a
    # solution computed at the *previous* active set silently breaks that orthogonality,
    # which is exactly the failure this ordering prevents.
    sh_star, gram, n_ridged = _solve_at(a_d, a_s)
    converged = False
    for it in range(max_iters):
        raw_d, raw_s = contract(D, sh_star), contract(S, sh_star)
        # Deadband: entries sitting essentially at zero keep their current state rather
        # than flipping on sign noise. Measured, this does NOT stop the set from cycling
        # (no convergence at 30 iterations for any flip_tol from 0 to 1e-6) -- it is kept
        # only because holding numerically-meaningless entries steady is the right
        # behaviour regardless. Cycling is tolerable here precisely because the loop
        # above guarantees the returned triple is self-consistent; the two sets it
        # alternates between give the same residual to 3 significant figures.
        td = flip_tol * raw_d.abs().amax().clamp_min(1e-30)
        ts = flip_tol * raw_s.abs().amax().clamp_min(1e-30)
        new_d = torch.where(raw_d > td, torch.ones_like(a_d),
                            torch.where(raw_d < -td, torch.zeros_like(a_d), a_d))
        new_s = torch.where(raw_s > ts, torch.ones_like(a_s),
                            torch.where(raw_s < -ts, torch.zeros_like(a_s), a_s))
        if torch.equal(new_d, a_d) and torch.equal(new_s, a_s):
            converged = True
            break
        a_d, a_s = new_d, new_s
        sh_star, gram, n_ridged = _solve_at(a_d, a_s)

    info = {"iters": it + 1, "converged": converged, "n_ridged": n_ridged, "gram": gram}
    if return_active:
        return sh_star, (a_d, a_s), info
    return sh_star
