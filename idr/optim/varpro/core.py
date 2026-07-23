"""The reduced Gauss-Newton step over the material, with the lighting projected out.

After `lighting.solve_lighting_active_set` eliminates the SH coefficients, the remaining
problem is over the per-pixel material only. Its Gauss-Newton Hessian is *not* simply
`J^T J`: because `sh*` moves when the material moves, the correct reduced Hessian
projects the Jacobian orthogonally to the lighting column space,

    H = J^T (I - P_A) J = blockdiag(J_p^T J_p) - U U^T,     P_A = A (A^T A)^-1 A^T

The first term is block diagonal (pixel p's residual depends only on pixel p's material);
the second is low rank, of rank `m = n_images * 3 * n_sh` — the number of lighting
degrees of freedom. Woodbury turns the solve into per-pixel 5x5 systems plus one m x m
system:

    (K - U U^T)^-1 = K^-1 + K^-1 U (I - U^T K^-1 U)^-1 U^T K^-1,   K = blockdiag(Dbd + Lam)

Two departures from the reference implementation, both forced:

**U is never materialised.** The reference forms `U` of shape `(P, 5, m)` — 5-10 GB at a
realistic 50k pixels x 100 images, the same OOM it hit and solved by chunking. `U` is
only ever used in sums over pixels, so it is rebuilt per chunk from the stored Jacobian.

**The projector uses the active-set Gram, not an unclamped basis.** The reference builds
its orthonormal basis from the *unclamped* design even though the lighting came from the
clamped solve, leaving its projector slightly inconsistent with the lighting it corrects.
Factoring `P_A = A (A^T A)^-1 A^T` keeps the active masks inside `A`, and `A^T A` is
already computed by the lighting solve — so exactness costs nothing here.

**The reduced Hessian is near-singular by construction, so the damping is not optional.**
Measured `cond(H + Lam) ~ 1e15` even with Marquardt damping at lam=1e-3: projecting out
the lighting leaves genuinely flat directions, the global albedo/lighting scale ambiguity
among them. Two correct solvers therefore produce steps that differ freely in that null
space — `tests/test_varpro_core.py` scores the step by how well it *solves* the normal
equations rather than by comparing it to a reference step, because the latter is not a
well-posed comparison at this conditioning.
"""
from __future__ import annotations

import torch

from idr.render.brdf import _lut_lookup
from .design import build_design_split

__all__ = ["build_px_render", "to_natural", "woodbury_step", "estimate_step_memory"]


def _band_expand_1d(Bv: torch.Tensor, sh_order: int) -> torch.Tensor:
    """(n_bands,) -> (n_sh,), each zonal weight repeated 2l+1 times."""
    parts = [Bv[0:1], Bv[1:2].expand(3), Bv[2:3].expand(5)]
    if sh_order >= 3:
        parts.append(Bv[3:4].expand(7))
    return torch.cat(parts)


def to_natural(material, space="natural", transforms=None):
    """(M,5) in `space` -> (M,5) natural [albedo(3), roughness, metallic].

    The design matrix is always a function of the natural values; only the Jacobian
    differs between spaces.
    """
    if space == "natural":
        return material
    from idr.optim.transforms import _fwd_albedo, _fwd_metallic, _fwd_roughness
    tr_ab, tr_met, tr_rou = transforms
    return torch.cat([_fwd_albedo(material[:, :3], tr_ab),
                      _fwd_roughness(material[:, 3:4], tr_rou),
                      _fwd_metallic(material[:, 4:5], tr_met)], dim=-1)


def build_px_render(geom, space="natural", transforms=None):
    """Return `f(m5, AY_p, YR_p, NdotV_p, front_p, ad_p, as_p, sh) -> (n, 3)`.

    Single-pixel forward built on the *design*, so the frozen active set is respected
    exactly. Differentiating `shade_ct_sh` directly would re-derive the clamps and yield
    the Jacobian of a different, moving-active-set problem.

    With `space="transformed"`, `m5` holds the raw pre-transform parameters and `_fwd_*`
    is applied inside, so `jacfwd` returns the chain rule through the sigmoids for free.
    `transforms` is then the `(tr_albedo, tr_metallic, tr_roughness)` triple.
    """
    lut, sh_order, dfres = geom.lut, geom.sh_order, geom.diffuse_fresnel
    if space == "transformed":
        from idr.optim.transforms import _fwd_albedo, _fwd_metallic, _fwd_roughness
        tr_ab, tr_met, tr_rou = transforms

    def f(m5, AY_p, YR_p, NdotV_p, front_p, ad_p, as_p, sh):
        if space == "transformed":
            alb = _fwd_albedo(m5[None, :3], tr_ab)[0]
            rough = _fwd_roughness(m5[None, 3:4], tr_rou)[0]
            metal = _fwd_metallic(m5[None, 4:5], tr_met)[0]
        else:
            alb, rough, metal = m5[:3], m5[3:4], m5[4:5]

        f0 = 0.04 * (1.0 - metal) + alb * metal                  # (3,)
        F = f0 + (1.0 - f0) * (1.0 - NdotV_p).pow(5)             # (3,)
        alpha = rough ** 2
        G1 = NdotV_p / (NdotV_p * (1.0 - alpha ** 2 / 2.0) + alpha ** 2 / 2.0 + 1e-6)
        k_d = 1.0 - metal
        if dfres:
            k_d = (1.0 - F) * k_d
        diff_w = front_p * k_d * alb / torch.pi                  # (3,)
        spec_w = front_p * F * G1 / 4.0                          # (3,)

        BY = _band_expand_1d(_lut_lookup(lut, rough).reshape(-1), sh_order) * YR_p
        raw_d = torch.einsum('i,nic->nc', AY_p, sh)              # (n, 3)
        raw_s = torch.einsum('i,nic->nc', BY, sh)
        return diff_w[None, :] * raw_d * ad_p + spec_w[None, :] * raw_s * as_p

    return f


def estimate_step_memory(M, n_img, n_sh, chunk, itemsize=8):
    """Bytes for the largest tensors in one step, so a run can refuse up front rather
    than OOM halfway through."""
    m = n_img * 3 * n_sh
    return {
        "m": m,
        "jac": M * n_img * 3 * 5 * itemsize,        # stored once, reused by all passes
        "U_chunk": chunk * 5 * m * itemsize,
        "Msmall": m * m * itemsize,
        "U_full_if_unchunked": M * 5 * m * itemsize,
    }


def _chol_inv_lower(gram, n_sh, ridge=1e-10):
    """L^-1 for each (image, channel) Gram, so U can carry A L^-T."""
    flat = gram.reshape(-1, n_sh, n_sh)
    L, info = torch.linalg.cholesky_ex(flat)
    if int((info != 0).sum()):
        eye = torch.eye(n_sh, dtype=flat.dtype, device=flat.device)
        sc = flat.diagonal(dim1=-2, dim2=-1).sum(-1).clamp_min(1e-30) / n_sh
        L, _ = torch.linalg.cholesky_ex(flat + (ridge * sc)[:, None, None] * eye)
    eye = torch.eye(n_sh, dtype=flat.dtype, device=flat.device).expand_as(flat)
    return torch.linalg.solve_triangular(L, eye, upper=False)     # (n*3, n_sh, n_sh)


def woodbury_step(geom, material, sh, active, gram, observations, lam,
                  space="natural", transforms=None, chunk=4096, max_bytes=None,
                  store_jac=True, lam_rel_init=1e-3):
    """One damped reduced Gauss-Newton step.

    material     (M,5) in `space`: [albedo(3), roughness, metallic]
    sh           (n, n_sh, 3)   lighting from the elimination
    active       (a_d, a_s), each (n, M, 3)
    gram         (n, 3, n_sh, n_sh)  A^T A at the converged active set
    observations (n, M, 3)

    -> (delta (M,5), info)
    """
    M = material.shape[0]
    n_img, n_sh = sh.shape[0], sh.shape[1]
    dev, dt = material.device, material.dtype
    m = n_img * 3 * n_sh
    a_d, a_s = active
    chunk = min(chunk, M)

    mem = estimate_step_memory(M, n_img, n_sh, chunk, material.element_size())
    need = mem["U_chunk"] + mem["Msmall"] + (mem["jac"] if store_jac else 0)
    if max_bytes is not None and need > max_bytes:
        raise MemoryError(
            f"VarPro step needs ~{need / 1e9:.2f} GB (jac {mem['jac'] / 1e9:.2f} + "
            f"U chunk {mem['U_chunk'] / 1e9:.2f} + Msmall {mem['Msmall'] / 1e9:.2f}) "
            f"> max_bytes {max_bytes / 1e9:.2f} GB. Reduce varpro_chunk (now {chunk}), "
            f"use fewer images (m = n_img*3*n_sh = {m}), or set store_jac=False.")

    nat = to_natural(material, space, transforms)
    D_des, S_des = build_design_split(geom, nat[:, :3], nat[:, 4:5], nat[:, 3:4])
    Linv = _chol_inv_lower(gram, n_sh).reshape(n_img, 3, n_sh, n_sh)

    px = build_px_render(geom, space, transforms)
    jac_v = torch.func.vmap(torch.func.jacfwd(px, argnums=0),
                            in_dims=(0, 0, 0, 0, 0, 0, 0, None))
    fwd_v = torch.func.vmap(px, in_dims=(0, 0, 0, 0, 0, 0, 0, None))

    def _args(lo, hi):
        sl = slice(lo, hi)
        return (material[sl], geom.AY[sl], geom.Y_R[sl], geom.NdotV[sl], geom.front[sl],
                a_d[:, sl, :].permute(1, 0, 2).contiguous(),
                a_s[:, sl, :].permute(1, 0, 2).contiguous())

    def _U(lo, hi, Dj):
        """(b, 5, m) — A under the active set, whitened by L^-T, contracted with the
        material Jacobian."""
        sl = slice(lo, hi)
        ad = a_d[:, sl, :].permute(1, 0, 2)                      # (b, n, 3)
        asx = a_s[:, sl, :].permute(1, 0, 2)
        A_act = (ad[:, :, :, None] * D_des[sl][:, None, :, :] +
                 asx[:, :, :, None] * S_des[sl][:, None, :, :])  # (b, n, 3, n_sh)
        # Contract A against Linv's SECOND index: we need (A^T A)^-1 = L^-T L^-1, so
        # sum_i AL[p,i]AL[q,i] must give A[p] (Linv^T Linv) A[q]^T. Contracting the
        # first index instead yields Linv Linv^T -- a different matrix, and the step
        # it produces is silently wrong rather than obviously broken.
        AL = torch.einsum('bncj,ncij->bnci', A_act, Linv)
        return torch.einsum('bnca,bnci->banci', Dj, AL).reshape(hi - lo, 5, m)

    slices = [(lo, min(lo + chunk, M)) for lo in range(0, M, chunk)]

    # ── pass 1: Jacobian, gradient, block Hessian, and diag(U U^T) ──────────
    jac_store = [] if store_jac else None
    g = torch.zeros((M, 5), dtype=dt, device=dev)
    Dbd = torch.zeros((M, 5, 5), dtype=dt, device=dev)
    diagUU = torch.zeros((M, 5), dtype=dt, device=dev)
    loss = 0.0
    for lo, hi in slices:
        Dj = jac_v(*_args(lo, hi), sh)                           # (b, n, 3, 5)
        r = fwd_v(*_args(lo, hi), sh) - observations[:, lo:hi, :].permute(1, 0, 2)
        loss += float((r ** 2).sum().detach())
        g[lo:hi] = torch.einsum('bnca,bnc->ba', Dj, r)
        Dbd[lo:hi] = torch.einsum('bncx,bncy->bxy', Dj, Dj)
        diagUU[lo:hi] = (_U(lo, hi, Dj) ** 2).sum(-1)
        if store_jac:
            jac_store.append(Dj)

    # Marquardt scaling on the diagonal of the REDUCED Hessian. `lam` may be a scalar or
    # one value PER PIXEL: pixels differ enormously in conditioning (a shadowed or
    # near-grazing pixel constrains its material far more weakly than a well-lit one), so
    # a single global trust region is set by the worst pixel and throttles all the rest.
    diagH = (Dbd.diagonal(dim1=1, dim2=2) - diagUU).clamp_min(1e-12)
    if lam is None:
        # Seed per-pixel damping from the curvature, as the reference does
        # (lam0 = 1e-3 * max diag H). It has to happen here: the caller cannot know the
        # Hessian before the step that computes it. The seeded value comes back in
        # info["lam"] so the caller can carry it forward.
        lam = lam_rel_init * diagH.max(dim=1).values
    _lam = lam if not torch.is_tensor(lam) else (
        lam[:, None] if lam.dim() == 1 else lam)
    Lam = _lam * diagH
    K = Dbd + torch.diag_embed(Lam)
    Kf, info_k = torch.linalg.cholesky_ex(K)
    if int((info_k != 0).sum()):
        K = K + 1e-10 * torch.eye(5, dtype=dt, device=dev)
        Kf, _ = torch.linalg.cholesky_ex(K)
    z = torch.cholesky_solve((-g).unsqueeze(-1), Kf).squeeze(-1)  # (M,5)

    # ── pass 2: the m x m system ────────────────────────────────────────────
    Msmall = torch.zeros((m, m), dtype=dt, device=dev)
    rhs = torch.zeros(m, dtype=dt, device=dev)
    for i, (lo, hi) in enumerate(slices):
        Dj = jac_store[i] if store_jac else jac_v(*_args(lo, hi), sh)
        U = _U(lo, hi, Dj)                                       # (b,5,m)
        Y = torch.cholesky_solve(U, Kf[lo:hi])                   # (b,5,m)
        Msmall += torch.einsum('bam,bak->mk', U, Y)
        rhs += torch.einsum('bam,ba->m', U, z[lo:hi])
    Msmall = torch.eye(m, dtype=dt, device=dev) - Msmall
    w = torch.linalg.solve(Msmall, rhs.unsqueeze(-1)).squeeze(-1)

    # ── pass 3: apply the correction ────────────────────────────────────────
    delta = z.clone()
    for i, (lo, hi) in enumerate(slices):
        Dj = jac_store[i] if store_jac else jac_v(*_args(lo, hi), sh)
        Y = torch.cholesky_solve(_U(lo, hi, Dj), Kf[lo:hi])
        delta[lo:hi] += torch.einsum('bam,m->ba', Y, w)

    # Lam and g are returned because the caller's trust-region gain ratio needs the
    # predicted reduction  0.5 * step^T (Lam*step - g), which cannot be reconstructed
    # from delta alone. diagH seeds a per-pixel lam0.
    return delta, {"loss": loss, "m": m, "chunks": len(slices),
                   "grad_norm": float(g.norm()), "Lam": Lam, "g": g, "diagH": diagH,
                   "lam": lam}
