"""
levenberg_marquardt.py — batched Levenberg-Marquardt for the CT decomposition.

A self-contained LM optimizer in the spirit of fabiodimarco/torch-levenberg-marquardt,
adapted to this project: the caller supplies a *residual* function instead of a
scalar loss, and LM minimises S(w) = sum_i r_i(w)^2 via

    (J^T J + lambda * D) delta = J^T r        w <- w - lr * delta

with an adaptive damping lambda that shrinks on an accepted step and grows on a
rejected one (the step is rolled back and retried, up to `attempts_per_step`).

Two things make this usable on the decomposition problem, where the parameter
count P (albedo+metallic+roughness maps + per-image SH) is a few thousand and the
residual count N (images x masked px x 3) is a few hundred thousand:

* **Sliced Jacobian.** The dense J is N x P (8.6 GB at 100 images / 31^2). We never
  form it: residuals are grouped by *sample* (= image), and J^T J (P x P) and J^T r
  are accumulated over sample chunks sized by `jacobian_max_num_rows`. Memory drops
  from O(N P) to O(P^2) (225 MB at P=7505, fp32).
* **Over/under-determined switch.** With few samples per step N can fall below P
  (e.g. batch_size=1 => N=2883 < P=4832). Then the minimum-norm form
  delta = J^T (J J^T + lambda I)^-1 r is solved instead, which is also cheaper.

Non-quadratic penalties (L1/huber data terms, TV, metallic-L1/binarize) enter via
the **square-root trick**: any non-negative loss term c_i >= 0 is represented by the
residual r_i = sqrt(c_i), so sum_i r_i^2 reproduces the term exactly. `sqrt` has an
infinite derivative at 0, so a small epsilon is added inside the root.

Usage
-----
    lm = LevenbergMarquardt(params, residuals_fn, n_samples=N_imgs,
                            reg_residuals_fn=reg_fn, jacobian_max_num_rows=60_000)
    info = lm.step()                 # full batch
    info = lm.step(idx)              # mini-batch over sample indices

`residuals_fn(params, idx) -> (rows,)` and `reg_residuals_fn(params) -> (rows,)`
must be pure functions of `params` (a tuple of tensors) so `torch.func` can
differentiate them. Regularizer residuals are sample-independent and are added
once per step, not per chunk.
"""
from __future__ import annotations

import math
from typing import Callable, Optional, Sequence

import torch
from torch.func import jacfwd, jacrev


def _jacobian(f, theta: torch.Tensor, n_rows: int, mode: str = "auto") -> torch.Tensor:
    """Jacobian of f at theta, shape (n_rows, P).

    Mode matters enormously here. Reverse mode (`jacrev`) costs one VJP per
    *residual*; forward mode (`jacfwd`) costs one JVP per *parameter*. This
    problem is strongly overdetermined (N ~ 3e5 residuals vs P ~ 5e3 params), so
    forward mode is the right default — reverse mode was ~100x slower on the
    SH-only subproblem (P=162, N=17298). 'auto' picks whichever dimension is
    smaller, mirroring the over/under-determined split of the normal equations.
    """
    P = theta.numel()
    mode = (mode or "auto").lower()
    if mode == "auto":
        mode = "forward" if P <= n_rows else "reverse"
    if mode == "forward":
        return jacfwd(f)(theta)
    return jacrev(f)(theta)


# ───────────────────────────── damping strategies ────────────────────────────
class DampingStrategy:
    """Adaptive lambda. `apply` builds the damped Gauss-Newton matrix."""

    def __init__(self, init: float = 1e-3, factor: float = 10.0,
                 min_value: float = 1e-12, max_value: float = 1e10,
                 adaptive: bool = True):
        self.init, self.factor = float(init), float(factor)
        self.min_value, self.max_value = float(min_value), float(max_value)
        self.adaptive = bool(adaptive)
        self.damping = float(init)

    def reset(self):
        self.damping = self.init

    def on_success(self):
        if self.adaptive:
            self.damping = max(self.damping / self.factor, self.min_value)

    def on_failure(self):
        if not self.adaptive:
            return False
        if self.damping >= self.max_value:
            return False                       # cannot damp further -> give up
        self.damping = min(self.damping * self.factor, self.max_value)
        return True

    def apply(self, JJ: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class StandardDamping(DampingStrategy):
    """J^T J + lambda * I — isotropic."""

    def apply(self, JJ):
        eye = torch.eye(JJ.shape[0], device=JJ.device, dtype=JJ.dtype)
        return JJ + self.damping * eye


class FletcherDamping(DampingStrategy):
    """J^T J + lambda * diag(J^T J) — scale-invariant (Fletcher / Marquardt)."""

    def apply(self, JJ):
        d = torch.diagonal(JJ).clamp_min(1e-12)
        return JJ + self.damping * torch.diag(d)


def make_damping(name: str = "standard", **kw) -> DampingStrategy:
    name = (name or "standard").lower()
    if name == "fletcher":
        return FletcherDamping(**kw)
    if name == "standard":
        return StandardDamping(**kw)
    raise ValueError(f"unknown damping strategy {name!r} (standard|fletcher)")


# ───────────────────────── matrix-free preconditioned CG ─────────────────────
def pcg(matvec: Callable[[torch.Tensor], torch.Tensor], b: torch.Tensor,
        precond: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        tol: float = 1e-4, maxiter: int = 100) -> tuple:
    """Solve A x = b for SPD A given only `matvec(v) = A v`.

    Used for the LM step at resolutions where the P x P matrix cannot be formed
    (P = 1.3e6 at 512^2). `tol` is relative to ||b||; LM tolerates an inexact
    inner solve (it becomes a truncated-Newton method) because the accept/reject
    test still guards every step.
    """
    x = torch.zeros_like(b)
    r = b.clone()
    z = precond(r) if precond is not None else r
    p = z.clone()
    rz = torch.dot(r, z)
    b_norm = torch.linalg.vector_norm(b).clamp_min(1e-30)
    it = 0
    for it in range(1, maxiter + 1):
        Ap = matvec(p)
        pAp = torch.dot(p, Ap)
        if not torch.isfinite(pAp) or pAp <= 0:      # loss of positive-definiteness
            break
        alpha = rz / pAp
        x = x + alpha * p
        r = r - alpha * Ap
        if torch.linalg.vector_norm(r) / b_norm < tol:
            break
        z = precond(r) if precond is not None else r
        rz_new = torch.dot(r, z)
        p = z + (rz_new / rz.clamp_min(1e-30)) * p
        rz = rz_new
    return x, it


class LinearBackend:
    """Strategy for forming + solving the LM normal equations.

    build(params, idx) -> loss      prepare rhs / operators at the current iterate
    solve(damping, kind) -> delta   solve (J^T J + damping * D) delta = J^T r
                                    kind is 'standard' (D=I) or 'fletcher' (D=diag)
                                    return None to signal "raise the damping and retry"
    """

    def build(self, params: tuple, idx: torch.Tensor) -> float:
        raise NotImplementedError

    def solve(self, damping: float, kind: str) -> Optional[torch.Tensor]:
        raise NotImplementedError


# ───────────────────────────────── solvers ───────────────────────────────────
def _solve(A: torch.Tensor, b: torch.Tensor, method: str) -> Optional[torch.Tensor]:
    """Solve A x = b for SPD-ish A. Returns None if the factorisation fails, so the
    caller can raise the damping and retry (a failed Cholesky means A is not PD)."""
    method = (method or "cholesky").lower()
    try:
        if method == "cholesky":
            L = torch.linalg.cholesky(A)
            return torch.cholesky_solve(b.unsqueeze(-1), L).squeeze(-1)
        if method == "qr":
            Q, R = torch.linalg.qr(A)
            return torch.linalg.solve_triangular(
                R, (Q.transpose(-2, -1) @ b.unsqueeze(-1)), upper=True).squeeze(-1)
        if method == "lstsq":
            return torch.linalg.lstsq(A, b.unsqueeze(-1)).solution.squeeze(-1)
        return torch.linalg.solve(A, b.unsqueeze(-1)).squeeze(-1)
    except RuntimeError:                       # singular / not positive definite
        return None


# ───────────────────────────────── the optimizer ─────────────────────────────
class LevenbergMarquardt:
    def __init__(
        self,
        params: Sequence[torch.Tensor],
        residuals_fn: Callable[[tuple, torch.Tensor], torch.Tensor],
        n_samples: int,
        *,
        reg_residuals_fn: Optional[Callable[[tuple], torch.Tensor]] = None,
        solver: str = "cholesky",
        damping: str = "standard",
        damping_init: float = 1e-3,
        damping_factor: float = 10.0,
        damping_min: float = 1e-12,
        damping_max: float = 1e10,
        adaptive_damping: bool = True,
        learning_rate: float = 1.0,
        attempts_per_step: int = 5,
        jacobian_max_num_rows: int = 0,
        jacobian_mode: str = "auto",
        structured_gn_fn: Optional[Callable[[tuple, torch.Tensor], tuple]] = None,
        linear_backend: Optional[LinearBackend] = None,
    ):
        self.params = [p for p in params]
        if not self.params:
            raise ValueError("LevenbergMarquardt: no parameters to optimize")
        self.residuals_fn = residuals_fn
        self.reg_residuals_fn = reg_residuals_fn
        self.n_samples = int(n_samples)
        self.solver = solver
        self.lr = float(learning_rate)
        self.attempts_per_step = max(1, int(attempts_per_step))
        self.jacobian_max_num_rows = int(jacobian_max_num_rows or 0)
        self.jacobian_mode = jacobian_mode
        # Optional fast path: (params, idx) -> (JtJ_data, Jtr_data, loss_data).
        # Lets a caller that knows the residual's sparsity build the normal
        # equations from small per-pixel blocks instead of a dense Jacobian.
        self.structured_gn_fn = structured_gn_fn
        # Optional: never form the P x P matrix at all (matrix-free CG, or an
        # exact Schur complement over the block-diagonal pixel params).
        self.linear_backend = linear_backend
        self.damping_strategy = make_damping(
            damping, init=damping_init, factor=damping_factor,
            min_value=damping_min, max_value=damping_max, adaptive=adaptive_damping)

        self._shapes = [tuple(p.shape) for p in self.params]
        self._numels = [p.numel() for p in self.params]
        self.n_params = int(sum(self._numels))
        self._dev, self._dtype = self.params[0].device, self.params[0].dtype
        self._rows_per_sample: Optional[int] = None

    # ── flat <-> tuple ────────────────────────────────────────────────────────
    def _unflatten(self, theta: torch.Tensor) -> tuple:
        out, i = [], 0
        for shp, n in zip(self._shapes, self._numels):
            out.append(theta[i:i + n].view(shp))
            i += n
        return tuple(out)

    def _flat(self) -> torch.Tensor:
        return torch.cat([p.detach().reshape(-1) for p in self.params])

    @torch.no_grad()
    def _write(self, theta: torch.Tensor):
        i = 0
        for p, n in zip(self.params, self._numels):
            p.copy_(theta[i:i + n].view_as(p))
            i += n

    # ── loss (no jacobian) ────────────────────────────────────────────────────
    @torch.no_grad()
    def loss(self, theta: torch.Tensor, idx: torch.Tensor) -> float:
        pt = self._unflatten(theta)
        total = 0.0
        chunk = self._sample_chunk()
        for s in range(0, idx.numel(), chunk):
            total += float(self.residuals_fn(pt, idx[s:s + chunk]).pow(2).sum())
        if self.reg_residuals_fn is not None:
            total += float(self.reg_residuals_fn(pt).pow(2).sum())
        return total

    def _sample_chunk(self) -> int:
        if self.jacobian_max_num_rows <= 0 or not self._rows_per_sample:
            return self.n_samples
        return max(1, self.jacobian_max_num_rows // max(self._rows_per_sample, 1))

    # ── normal equations, accumulated over sample chunks ──────────────────────
    def _gauss_newton(self, theta: torch.Tensor, idx: torch.Tensor):
        """Return (JJ, rhs, loss, overdetermined, J_or_None).

        Overdetermined: JJ = J^T J (P,P), rhs = J^T r  -> memory O(P^2), J discarded.
        Underdetermined: JJ = J J^T (N,N), rhs = r, J kept for delta = J^T JJ^-1 r.
        """
        P = self.n_params
        if self._rows_per_sample is None:                    # probe once
            with torch.no_grad():
                self._rows_per_sample = int(
                    self.residuals_fn(self._unflatten(theta), idx[:1]).numel())

        n_rows = self._rows_per_sample * idx.numel()
        n_reg = 0
        if self.reg_residuals_fn is not None:
            with torch.no_grad():
                n_reg = int(self.reg_residuals_fn(self._unflatten(theta)).numel())
        overdet = (n_rows + n_reg) >= P

        # ── structured fast path: caller assembles the data-term normal equations
        # from block-sparse per-pixel jacobians; regularizers stay generic. ──────
        if self.structured_gn_fn is not None:
            JJ, rhs, loss = self.structured_gn_fn(self._unflatten(theta), idx)
            if self.reg_residuals_fn is not None and n_reg:
                fr = lambda th: self.reg_residuals_fn(self._unflatten(th))
                Jr = _jacobian(fr, theta, n_reg, self.jacobian_mode)
                with torch.no_grad():
                    rr = fr(theta)
                    JJ = JJ + Jr.T @ Jr
                    rhs = rhs + Jr.T @ rr
                    loss += float(rr.pow(2).sum())
                del Jr
            return JJ, rhs, loss, True, None

        chunk = self._sample_chunk()
        loss = 0.0
        if overdet:
            JJ = torch.zeros(P, P, device=self._dev, dtype=self._dtype)
            rhs = torch.zeros(P, device=self._dev, dtype=self._dtype)
            for s in range(0, idx.numel(), chunk):
                sub = idx[s:s + chunk]
                f = lambda th, _sub=sub: self.residuals_fn(self._unflatten(th), _sub)
                rows = self._rows_per_sample * sub.numel()
                J = _jacobian(f, theta, rows, self.jacobian_mode)      # (rows, P)
                with torch.no_grad():
                    r = f(theta)
                    JJ += J.T @ J
                    rhs += J.T @ r
                    loss += float(r.pow(2).sum())
                del J
            if self.reg_residuals_fn is not None and n_reg:
                fr = lambda th: self.reg_residuals_fn(self._unflatten(th))
                Jr = _jacobian(fr, theta, n_reg, self.jacobian_mode)
                with torch.no_grad():
                    rr = fr(theta)
                    JJ += Jr.T @ Jr
                    rhs += Jr.T @ rr
                    loss += float(rr.pow(2).sum())
                del Jr
            return JJ, rhs, loss, True, None

        # underdetermined: build the (small) full J once
        # N < P: reverse mode is cheapest here (cost ~ n_rows), and J is small.
        Js, rs = [], []
        for s in range(0, idx.numel(), chunk):
            sub = idx[s:s + chunk]
            f = lambda th, _sub=sub: self.residuals_fn(self._unflatten(th), _sub)
            Js.append(jacrev(f)(theta))
            with torch.no_grad():
                rs.append(f(theta))
        if self.reg_residuals_fn is not None and n_reg:
            fr = lambda th: self.reg_residuals_fn(self._unflatten(th))
            Js.append(jacrev(fr)(theta))
            with torch.no_grad():
                rs.append(fr(theta))
        J = torch.cat(Js, 0)
        r = torch.cat(rs, 0)
        with torch.no_grad():
            loss = float(r.pow(2).sum())
            return (J @ J.T), r, loss, False, J

    # ── one LM step (with accept/reject + damping adaptation) ─────────────────
    def step(self, idx: Optional[torch.Tensor] = None) -> dict:
        if idx is None:
            idx = torch.arange(self.n_samples, device=self._dev)
        elif not torch.is_tensor(idx):
            idx = torch.as_tensor(list(idx), device=self._dev, dtype=torch.long)

        theta0 = self._flat()

        # ── matrix-free / Schur path: never forms a P x P matrix ──────────────
        if self.linear_backend is not None:
            kind = "fletcher" if isinstance(self.damping_strategy, FletcherDamping) else "standard"
            loss0 = self.linear_backend.build(self._unflatten(theta0), idx)
            accepted, attempts, loss_new = False, 0, loss0
            for attempts in range(1, self.attempts_per_step + 1):
                delta = self.linear_backend.solve(self.damping_strategy.damping, kind)
                if delta is None:
                    if not self.damping_strategy.on_failure():
                        break
                    continue
                theta_try = theta0 - self.lr * delta
                loss_try = self.loss(theta_try, idx)
                if loss_try < loss0 and math.isfinite(loss_try):
                    self._write(theta_try)
                    self.damping_strategy.on_success()
                    accepted, loss_new = True, loss_try
                    break
                if not self.damping_strategy.on_failure():
                    break
            if not accepted:
                self._write(theta0)
            return dict(loss=loss_new, loss_before=loss0, accepted=accepted,
                        attempts=attempts, damping=self.damping_strategy.damping,
                        overdetermined=True, n_params=self.n_params,
                        cg_iters=getattr(self.linear_backend, "last_cg_iters", 0))

        JJ, rhs, loss0, overdet, J = self._gauss_newton(theta0, idx)

        accepted, attempts = False, 0
        loss_new = loss0
        for attempts in range(1, self.attempts_per_step + 1):
            A = self.damping_strategy.apply(JJ)
            sol = _solve(A, rhs, self.solver)
            if sol is None:                                   # not PD -> damp harder
                if not self.damping_strategy.on_failure():
                    break
                continue

            # overdetermined: delta solves (J^T J + lam D) delta = J^T r
            # underdetermined: delta = J^T (J J^T + lam I)^-1 r   (minimum norm)
            delta = sol if overdet else (J.T @ sol)

            theta_try = theta0 - self.lr * delta
            loss_try = self.loss(theta_try, idx)
            if loss_try < loss0 and math.isfinite(loss_try):
                self._write(theta_try)
                self.damping_strategy.on_success()
                accepted, loss_new = True, loss_try
                break
            if not self.damping_strategy.on_failure():
                break

        if not accepted:
            self._write(theta0)               # roll back

        return dict(loss=loss_new, loss_before=loss0, accepted=accepted,
                    attempts=attempts, damping=self.damping_strategy.damping,
                    overdetermined=overdet, n_params=self.n_params, cg_iters=0)
