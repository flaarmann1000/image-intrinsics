"""Assemble the VarPro solver for the Cook-Torrance + SH model.

Mirrors `idr/optim/lm/problem.build_lm_solver`: takes the model's already-computed
geometry and parameter tensors and returns a stepper the main loop drives, so
`ct_sh.py` gains one branch rather than an algorithm.

One iteration is:

  1. eliminate the lighting in closed form for the current material (`lighting`)
  2. build the reduced, lighting-projected Gauss-Newton step (`core`)
  3. backtrack along it until the objective actually decreases
  4. write the accepted material and the solved lighting back into the model's params

Steps 3-4 are what make this usable rather than merely correct. The reduced Hessian is
near-singular (see `core`), so an undamped or unchecked step can leave the basin
entirely; the line search plus Marquardt adaptation is the guard.

`varpro_space` selects the parameter space. In `"natural"` the optimiser works on
physical values in a box and the model's `tr_*` transforms are inverted on write-back;
in `"transformed"` it works on the raw parameters directly and the transforms are
honoured, so the write-back is a plain copy.
"""
from __future__ import annotations

import torch

from idr.optim.transforms import (_fwd_albedo, _fwd_metallic, _fwd_roughness,
                                  _init_albedo)
from .core import to_natural, woodbury_step
from .design import VarProGeometry, forward_from_design
from .lighting import solve_lighting_active_set
from .profile_rho import refine_rho

__all__ = ["build_varpro_solver", "VarProSolver"]

# Physical bounds. The roughness floor is a conditioning guard: the GGX-SH lookup is
# worst-behaved as the lobe narrows, and the reference uses the same 0.03.
LOWER = (0.0, 0.0, 0.0, 0.03, 0.0)
UPPER = (1.0, 1.0, 1.0, 1.0, 1.0)
LINE_SEARCH = (1.0, 0.5, 0.25, 0.125)


def _inv_unit(x: torch.Tensor, t: str) -> torch.Tensor:
    """Invert the unit-interval transforms used by metallic/roughness."""
    if t == "sigmoid":
        return torch.logit(x.clamp(1e-6, 1 - 1e-6))
    if t == "sigmoid_sq":
        return torch.logit(x.clamp(1e-6, 1 - 1e-6).sqrt())
    if t in ("none", "", None):
        return x
    raise ValueError(
        f"varpro_space='natural' must write results back through the inverse of "
        f"tr_metallic/tr_roughness, and {t!r} has no inverse implemented. Use "
        f"varpro_space='transformed' (which needs no inverse) or add one.")


def _inv_albedo(x: torch.Tensor, t: str) -> torch.Tensor:
    if t in ("sigmoid", "log", "none", "", None):
        return _init_albedo(x.clamp(1e-6, 1 - 1e-6) if t == "sigmoid" else x, t)
    raise ValueError(f"no inverse implemented for tr_albedo={t!r}")


class VarProSolver:
    """Holds the geometry and parameter handles; `step()` advances one iteration."""

    def __init__(self, geom, albedo_param, sh_coeffs, metallic_raw, roughness_raw,
                 flat_mask, observations, transforms, cfg, dev, dtype):
        self.geom = geom
        self.albedo_param = albedo_param
        self.sh_coeffs = sh_coeffs
        self.metallic_raw = metallic_raw
        self.roughness_raw = roughness_raw
        self.flat_mask = flat_mask
        self.obs = observations                      # (N, M, 3)
        self.tr = transforms                         # (tr_ab, tr_met, tr_rou)
        self.dev, self.dtype = dev, dtype

        self.space = str(cfg.get("varpro_space", "natural")).lower()
        if self.space not in ("natural", "transformed"):
            raise ValueError(f"varpro_space must be 'natural' or 'transformed', "
                             f"got {self.space!r}")
        if int(cfg.get("varpro_n_inner_rho", 0)) > 0 and self.space == "transformed":
            # refine_rho works because the render is LINEAR in albedo; under a sigmoid
            # it is not. Fail loudly rather than silently skip the refinement.
            raise ValueError(
                "varpro_n_inner_rho > 0 requires varpro_space='natural'. The albedo "
                "profiling exploits the render's structure in albedo directly; a sigmoid "
                "reparameterisation destroys it. Use natural space, or set "
                "varpro_n_inner_rho=0 to run VarPro without profiling.")
        # lam is PER PIXEL and seeded from the curvature on the first step, matching the
        # reference's lam0 = 1e-3 * max diag(H). It cannot be built here because the
        # Hessian is not known until the first Woodbury step, hence the None sentinel.
        self.lam = None
        self.lam_init_rel = float(cfg.get("varpro_lam_init", 1e-3))
        self.lam_ceiling = float(cfg.get("varpro_lam_ceiling", 1e10))
        self.n_inner_rho = int(cfg.get("varpro_n_inner_rho", 0))
        self._state = None
        M = int(flat_mask.sum())
        # Nielsen's failure multiplier, per pixel: doubles on each consecutive rejection
        # so a pixel that keeps failing backs off geometrically.
        self.v = torch.full((M,), 2.0, dtype=dtype, device=dev)
        self.active_iters = int(cfg.get("varpro_active_iters", 8))
        self.ridge = float(cfg.get("varpro_lighting_ridge", 1e-10))
        self.chunk = int(cfg.get("varpro_chunk", 4096))
        self.max_bytes = cfg.get("varpro_max_bytes", None)
        self.fractions = tuple(cfg.get("varpro_line_search", LINE_SEARCH))
        self.lower = torch.tensor(cfg.get("varpro_lower", LOWER), dtype=dtype, device=dev)
        self.upper = torch.tensor(cfg.get("varpro_upper", UPPER), dtype=dtype, device=dev)

    # ── parameter <-> (M,5) material ────────────────────────────────────────
    def _gather(self):
        """Current material as (M,5) in `self.space`: [albedo(3), roughness, metallic]."""
        fm = self.flat_mask
        tr_ab, tr_met, tr_rou = self.tr
        ab = self.albedo_param.reshape(-1, 3)[fm]
        me = self.metallic_raw.reshape(-1, 1)[fm]
        ro = self.roughness_raw.reshape(-1, 1)[fm]
        if self.space == "natural":
            ab = _fwd_albedo(ab, tr_ab)
            me = _fwd_metallic(me, tr_met)
            ro = _fwd_roughness(ro, tr_rou)
        # DETACH. The model's parameters require grad, but VarPro never backpropagates
        # through them -- it builds its own Jacobian with torch.func.jacfwd. Without this
        # the whole active-set iteration and every line-search evaluation get recorded
        # into an autograd graph that is never used: measured 3.25 GB of graph on a
        # 128^2 x 75-image scene, enough to OOM a 4 GB card, against 0.51 GB for the
        # actual VarPro tensors.
        return torch.cat([ab, ro, me], dim=-1).detach()

    def _scatter(self, mat):
        """Write (M,5) back into the model's raw parameter tensors."""
        fm = self.flat_mask
        tr_ab, tr_met, tr_rou = self.tr
        ab, ro, me = mat[:, :3], mat[:, 3:4], mat[:, 4:5]
        if self.space == "natural":
            ab = _inv_albedo(ab, tr_ab)
            ro = _inv_unit(ro, tr_rou)
            me = _inv_unit(me, tr_met)
        with torch.no_grad():
            self.albedo_param.reshape(-1, 3)[fm] = ab
            self.roughness_raw.reshape(-1, 1)[fm] = ro
            self.metallic_raw.reshape(-1, 1)[fm] = me

    # ── objective ───────────────────────────────────────────────────────────
    def _eval(self, mat):
        """-> (material, F_p, sh, active, info).

        Eliminates the lighting, optionally profiles albedo, and returns the objective
        PER PIXEL so the caller can accept or reject each pixel independently.
        `F_p = 0.5 * sum(r^2)` over images and channels, matching the convention in which
        `g = J^T r` is the gradient.

        The returned material is not necessarily the one passed in: with
        `varpro_profile_rho` the albedo is refined here, and that refined value is the
        state the outer step continues from.

        Only the data term — VarPro's objective is the projected residual, and the
        regularizers `_forward` adds are not part of the eliminated problem.
        """
        with torch.no_grad():
            nat = to_natural(mat, self.space, self.tr)
            sh, active, info = solve_lighting_active_set(
                self.geom, nat[:, :3], nat[:, 4:5], nat[:, 3:4], self.obs,
                max_iters=self.active_iters, ridge=self.ridge)

            if self.n_inner_rho > 0:
                # Profile albedo toward its conditional optimum, then RE-solve the
                # lighting for it: the two are coupled, and stopping after the albedo
                # refinement would leave a lighting fitted to the pre-refinement albedo.
                rho = refine_rho(self.geom, nat, sh, active, self.obs,
                                 n_steps=self.n_inner_rho, chunk=self.chunk)
                nat = torch.cat([rho, nat[:, 3:5]], dim=1)
                mat = nat if self.space == "natural" else torch.cat(
                    [_inv_albedo(rho, self.tr[0]), mat[:, 3:5]], dim=1)
                sh, active, info = solve_lighting_active_set(
                    self.geom, nat[:, :3], nat[:, 4:5], nat[:, 3:4], self.obs,
                    max_iters=self.active_iters, ridge=self.ridge)

            recon, _ = forward_from_design(self.geom, nat[:, :3], nat[:, 4:5],
                                           nat[:, 3:4], sh, active=active)
            # `forward_from_design` and `self.obs` are both (n, M, 3) — no permute.
            r = recon - self.obs
            F_p = 0.5 * (r * r).sum(dim=(0, 2))          # (M,) per-pixel objective
        return mat, F_p, sh, active, info

    # ── one iteration ───────────────────────────────────────────────────────
    def step(self):
        # The evaluated state carries over between iterations: the state settled at the
        # end of one step is exactly the top-of-loop state of the next, so caching it
        # saves a full lighting solve per iteration.
        if self._state is None:
            self._state = self._eval(self._gather())
        mat, F_p, sh, active, linfo = self._state

        delta, sinfo = woodbury_step(
            self.geom, mat, sh, active, linfo["gram"], self.obs, self.lam,
            space=self.space, transforms=self.tr, chunk=self.chunk,
            max_bytes=self.max_bytes, lam_rel_init=self.lam_init_rel)
        Lam, g = sinfo["Lam"], sinfo["g"]
        self.lam = sinfo["lam"]        # seeded from curvature on the first step

        # PER-PIXEL line search. Pixels differ in how far the quadratic model stays
        # trustworthy, so each keeps the first fraction that yields a positive gain
        # ratio rather than the whole image sharing one fraction. Note each trial still
        # moves ALL pixels (the lighting solve is global), so a pixel's F_try is measured
        # with its neighbours also displaced -- an approximation the reference makes too.
        best_mat = mat.clone()
        best_F = F_p.clone()
        best_ratio = torch.full_like(F_p, -1.0)
        accepted = torch.zeros(F_p.shape[0], dtype=torch.bool, device=F_p.device)
        n_tried = 0
        for f in self.fractions:
            n_tried += 1
            stepv = f * delta
            trial = mat + stepv
            if self.space == "natural":
                trial = torch.max(torch.min(trial, self.upper), self.lower)
            _, F_try, _, _, _ = self._eval(trial)
            # Predicted reduction of the damped quadratic model, per pixel.
            predicted = 0.5 * (stepv * (Lam * stepv - g)).sum(dim=1)
            ratio = torch.where(predicted > 0, (F_p - F_try) / predicted,
                                torch.full_like(predicted, -1.0))
            newly = (ratio > 0) & (~accepted)
            best_mat = torch.where(newly[:, None], trial, best_mat)
            best_F = torch.where(newly, F_try, best_F)
            best_ratio = torch.where(newly, ratio, best_ratio)
            accepted |= newly
            if bool(accepted.all()):
                break

        # Nielsen damping update: the shrink factor is continuous in the gain ratio
        # (a near-perfect model shrinks lam by ~1/3, a marginal one barely at all),
        # while a rejected pixel grows by v and doubles v -- so repeated failure backs
        # off geometrically instead of at a fixed rate.
        lam_ok = self.lam * torch.clamp_min(
            1.0 - (2.0 * best_ratio.clamp(max=1.0) - 1.0) ** 3, 1.0 / 3.0)
        self.lam = torch.where(accepted, lam_ok, self.lam * self.v).clamp(
            max=self.lam_ceiling)
        self.v = torch.where(accepted, torch.full_like(self.v, 2.0), self.v * 2.0)

        n_acc = int(accepted.sum())
        if n_acc == 0:
            return {"accepted": 0, "frac_pixels": 0.0, "loss": float(F_p.sum()),
                    "fractions_tried": n_tried, "active_converged": linfo["converged"]}

        new_mat = torch.where(accepted[:, None], best_mat, mat)
        # Re-evaluate once at the settled (mixed-fraction) material: this both writes a
        # lighting that matches the material actually kept, and becomes the next step's
        # cached state -- so it is not an extra solve.
        self._state = self._eval(new_mat)
        self._scatter(self._state[0])
        with torch.no_grad():
            self.sh_coeffs.copy_(self._state[2])
        return {"accepted": n_acc, "frac_pixels": n_acc / F_p.shape[0],
                "loss": float(self._state[1].sum()), "loss_prev": float(F_p.sum()),
                "fractions_tried": n_tried, "active_converged": linfo["converged"],
                "lam_median": float(self.lam.median())}


def build_varpro_solver(cfg, albedo_param, sh_coeffs, metallic_raw, roughness_raw,
                        flat_mask, imgs_m, AY, Y_R, NdotV, front, lut, sh_order,
                        diffuse_fresnel, tr_ab, tr_met, tr_rou, dev, ftype):
    """Build the stepper from tensors `ct_sh` has already computed.

    `AY`/`Y_R`/`NdotV`/`front` come straight from ct_sh's "Precompute geometry-only
    terms" block, so the geometry has exactly one definition in the codebase.
    """
    geom = VarProGeometry(AY=AY, Y_R=Y_R, NdotV=NdotV, front=front, lut=lut,
                          sh_order=sh_order, diffuse_fresnel=diffuse_fresnel)
    return VarProSolver(geom, albedo_param, sh_coeffs, metallic_raw, roughness_raw,
                        flat_mask, imgs_m, (tr_ab, tr_met, tr_rou), cfg, dev, ftype)
