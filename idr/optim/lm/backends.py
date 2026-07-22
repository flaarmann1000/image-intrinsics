"""Matrix-free linear solvers for the Levenberg-Marquardt normal equations.

Dense P x P factorisation is only viable for small problems. `_CGBackend` is
matrix-free (O(P) memory) and is what runs at 512^2; `_SchurBackend` exploits
pixel-separable regularizers to eliminate the per-pixel block first.
"""
from __future__ import annotations

import numpy as np
import torch

from idr.optim.losses import _sqrt_res, _tv_residuals
from .core import pcg

class _CGBackend:
    """Matrix-free LM step: solve (J^T J + lam D) d = J^T r by preconditioned CG.

    Never forms the P x P matrix (P = 1.3e6 at 512^2), so memory is O(P). Every
    matvec is  J^T (J v)  evaluated with torch.func jvp/vjp, chunked over images
    so the autograd graph only ever holds `img_chunk` images. Handles *all*
    regularizers exactly, including the pixel-coupling ones (TV, sparse, white).

    Preconditioner: block-Jacobi from the exact Gauss-Newton diagonal blocks —
    a 5x5 block per pixel and a (3*n_sh)^2 block per image — which are cheap to
    build from the per-pixel jacobians and dramatically cut the CG count.
    """

    def __init__(self, *, unflatten, data_res_fn, reg_res_fn, blocks_fn,
                 pix_idx, sh_off, n_sh3, n_imgs, img_chunk, tol, maxiter, dev, dtype):
        self.unflatten, self.data_res_fn, self.reg_res_fn = unflatten, data_res_fn, reg_res_fn
        self.blocks_fn = blocks_fn
        self.pix_idx, self.sh_off, self.n_sh3 = pix_idx, sh_off, n_sh3
        self.n_imgs, self.img_chunk = n_imgs, max(1, int(img_chunk))
        self.tol, self.maxiter = float(tol), int(maxiter)
        self.dev, self.dtype = dev, dtype
        self.last_cg_iters = 0

    def _chunks(self, idx):
        for s in range(0, idx.numel(), self.img_chunk):
            yield idx[s:s + self.img_chunk]

    def build(self, params, idx):
        self._theta = torch.cat([p.detach().reshape(-1) for p in params])
        self._idx = idx
        rhs = torch.zeros_like(self._theta)
        loss = 0.0
        for sub in self._chunks(idx):
            f = lambda th, _s=sub: self.data_res_fn(self.unflatten(th), _s)
            r, vjp_fn = torch.func.vjp(f, self._theta)
            rhs += vjp_fn(r)[0]
            loss += float(r.pow(2).sum())
            del vjp_fn
        if self.reg_res_fn is not None:
            fr = lambda th: self.reg_res_fn(self.unflatten(th))
            r, vjp_fn = torch.func.vjp(fr, self._theta)
            rhs += vjp_fn(r)[0]
            loss += float(r.pow(2).sum())
            del vjp_fn
        self.rhs = rhs
        self._A, self._C = self.blocks_fn(params, idx)     # (M,npix,npix), (K,n_sh3,n_sh3)
        return loss

    def _JtJv(self, v):
        out = torch.zeros_like(v)
        for sub in self._chunks(self._idx):
            f = lambda th, _s=sub: self.data_res_fn(self.unflatten(th), _s)
            _, Jv = torch.func.jvp(f, (self._theta,), (v,))
            _, vjp_fn = torch.func.vjp(f, self._theta)
            out += vjp_fn(Jv)[0]
            del vjp_fn, Jv
        if self.reg_res_fn is not None:
            fr = lambda th: self.reg_res_fn(self.unflatten(th))
            _, Jv = torch.func.jvp(fr, (self._theta,), (v,))
            _, vjp_fn = torch.func.vjp(fr, self._theta)
            out += vjp_fn(Jv)[0]
            del vjp_fn, Jv
        return out

    def solve(self, damping, kind):
        if kind == "fletcher":
            d = torch.zeros_like(self.rhs)
            if self.pix_idx is not None:
                d[self.pix_idx.reshape(-1)] = torch.diagonal(
                    self._A, dim1=-2, dim2=-1).reshape(-1)
            if self._C is not None:
                d[self.sh_off:self.sh_off + self._C.shape[0] * self.n_sh3] = torch.diagonal(
                    self._C, dim1=-2, dim2=-1).reshape(-1)
            d = d.clamp_min(1e-12)
        else:
            d = None

        def matvec(v):
            damp = damping * (d * v if d is not None else v)
            return self._JtJv(v) + damp

        # block-Jacobi preconditioner, refreshed for this damping value
        eye_p = torch.eye(self._A.shape[-1], device=self.dev, dtype=self.dtype) \
            if self.pix_idx is not None else None
        Ainv = Cinv = None
        if self.pix_idx is not None:
            Ad = self._A + damping * (torch.diag_embed(torch.diagonal(self._A, dim1=-2, dim2=-1))
                                      if kind == "fletcher" else eye_p)
            Ainv = torch.linalg.inv(Ad + 1e-12 * eye_p)
        if self._C is not None:
            eye_s = torch.eye(self.n_sh3, device=self.dev, dtype=self.dtype)
            Cd = self._C + damping * (torch.diag_embed(torch.diagonal(self._C, dim1=-2, dim2=-1))
                                      if kind == "fletcher" else eye_s)
            Cinv = torch.linalg.inv(Cd + 1e-12 * eye_s)

        def precond(v):
            out = v.clone()
            if Ainv is not None:
                vp = v[self.pix_idx]                                  # (M, npix)
                out[self.pix_idx] = torch.einsum('mij,mj->mi', Ainv, vp)
            if Cinv is not None:
                n = Cinv.shape[0] * self.n_sh3
                vs = v[self.sh_off:self.sh_off + n].view(-1, self.n_sh3)
                out[self.sh_off:self.sh_off + n] = torch.einsum(
                    'kij,kj->ki', Cinv, vs).reshape(-1)
            return out

        x, iters = pcg(matvec, self.rhs, precond, tol=self.tol, maxiter=self.maxiter)
        self.last_cg_iters = iters
        if not torch.isfinite(x).all():
            return None
        return x


class _SchurBackend:
    """Exact LM step via Schur complement over the per-pixel blocks.

    Each pixel's 5 raw params touch only its own residuals, so the pixel Hessian A
    is block-diagonal (5x5 per pixel) *provided no regularizer couples pixels*
    (no TV / sparse / white). Eliminate it and solve the tiny reduced SH system:

        S      = C - B^T (A+lam)^-1 B                  (K*3*n_sh squared, 29 MB at K=100)
        d_sh   = S^-1 (g_s - B^T (A+lam)^-1 g_p)
        d_pix  = (A+lam)^-1 (g_p - B d_sh)

    Exact, and fast: the reduced system is only (K*3*n_sh)^2 (2700^2 at K=100).
    The cost is memory for the cross block B, which is (M*npix) x (K*3*n_sh):
    ~0.8 GB at 124^2/K=100, ~3.5 GB at 256^2, ~14 GB at 512^2. Above
    `lm_schur_max_gb` this backend refuses to run and tells you to use 'cg'
    (which is O(P) and is the auto choice when a pixel-coupling regularizer is on).
    """

    def __init__(self, *, blocks_full_fn, pix_idx, sh_off, n_sh3, dev, dtype):
        self.blocks_full_fn = blocks_full_fn
        self.pix_idx, self.sh_off, self.n_sh3 = pix_idx, sh_off, n_sh3
        self.dev, self.dtype = dev, dtype
        self.last_cg_iters = 0

    def build(self, params, idx):
        self._params, self._idx = params, idx
        self._P = sum(p.numel() for p in params)
        A, C, B, g_p, g_s, loss = self.blocks_full_fn(params, idx)
        self._A, self._C, self._B, self._gp, self._gs = A, C, B, g_p, g_s
        return loss

    def solve(self, damping, kind):
        A, C, B, gp, gs = self._A, self._C, self._B, self._gp, self._gs
        M, npix = A.shape[0], A.shape[-1]
        K = C.shape[0]
        eye_p = torch.eye(npix, device=self.dev, dtype=self.dtype)
        eye_s = torch.eye(self.n_sh3, device=self.dev, dtype=self.dtype)
        if kind == "fletcher":
            Ad = A + damping * torch.diag_embed(torch.diagonal(A, dim1=-2, dim2=-1))
            Cd = C + damping * torch.diag_embed(torch.diagonal(C, dim1=-2, dim2=-1))
        else:
            Ad, Cd = A + damping * eye_p, C + damping * eye_s
        try:
            Ainv = torch.linalg.inv(Ad + 1e-12 * eye_p)                  # (M, npix, npix)
        except RuntimeError:
            return None
        # T[p] = Ainv[p] @ B[p]  -> (M, npix, K*n_sh3)
        T = torch.einsum('mij,mja->mia', Ainv, B)
        S = torch.block_diag(*Cd) - torch.einsum('mia,mib->ab', B, T)    # (K*n_sh3, K*n_sh3)
        rhs_s = gs.reshape(-1) - torch.einsum('mia,mi->a', T, gp)
        try:
            L = torch.linalg.cholesky(S)
            d_sh = torch.cholesky_solve(rhs_s.unsqueeze(-1), L).squeeze(-1)
        except RuntimeError:
            return None
        d_pix = torch.einsum('mij,mj->mi', Ainv,
                             gp - torch.einsum('mia,a->mi', B, d_sh))    # (M, npix)
        delta = torch.zeros(self._P, device=self.dev, dtype=self.dtype)
        delta[self.pix_idx] = d_pix
        delta[self.sh_off:self.sh_off + K * self.n_sh3] = d_sh
        if not torch.isfinite(delta).all():
            return None
        return delta


def _lm_cfg_kwargs(cfg) -> dict:
    return dict(
        solver            = cfg.get("lm_solver", "cholesky"),
        damping           = cfg.get("lm_damping", "standard"),
        damping_init      = float(cfg.get("lm_damping_init", 1e-3)),
        damping_factor    = float(cfg.get("lm_damping_factor", 10.0)),
        damping_min       = float(cfg.get("lm_damping_min", 1e-12)),
        damping_max       = float(cfg.get("lm_damping_max", 1e10)),
        adaptive_damping  = bool(cfg.get("lm_adaptive_damping", True)),
        learning_rate     = float(cfg.get("lm_learning_rate", 1.0)),
        attempts_per_step = int(cfg.get("lm_attempts_per_step", 5)),
        jacobian_max_num_rows = int(cfg.get("lm_jacobian_max_num_rows", 0) or 0),        
        jacobian_mode     = cfg.get("lm_jacobian_mode", "auto"),
    )
