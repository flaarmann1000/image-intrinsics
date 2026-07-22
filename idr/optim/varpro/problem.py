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
        if cfg.get("varpro_profile_rho", False) and self.space == "transformed":
            # refine_rho works because the render is LINEAR in albedo; under a sigmoid
            # it is not. Fail loudly rather than silently skip the refinement.
            raise ValueError("varpro_profile_rho requires varpro_space='natural' — the "
                             "albedo profiling step relies on linearity in albedo, which "
                             "a sigmoid transform destroys.")
        self.lam = float(cfg.get("varpro_lam_init", 1e-3))
        self.lam_ceiling = float(cfg.get("varpro_lam_ceiling", 1e10))
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
        return torch.cat([ab, ro, me], dim=-1)

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
        """Eliminate the lighting for this material and return (loss, sh, active, info).

        The loss is the data term only — VarPro's objective is the projected residual,
        and the regularizers `_forward` adds are not part of the eliminated problem.
        """
        nat = to_natural(mat, self.space, self.tr)
        sh, active, info = solve_lighting_active_set(
            self.geom, nat[:, :3], nat[:, 4:5], nat[:, 3:4], self.obs,
            max_iters=self.active_iters, ridge=self.ridge)
        recon, _ = forward_from_design(self.geom, nat[:, :3], nat[:, 4:5], nat[:, 3:4],
                                       sh, active=active)
        # `forward_from_design` and `self.obs` are both (n, M, 3) — no permute.
        loss = float(((recon - self.obs) ** 2).sum().detach())
        return loss, sh, active, info

    # ── one iteration ───────────────────────────────────────────────────────
    def step(self):
        mat = self._gather()
        loss0, sh, active, linfo = self._eval(mat)

        delta, _sinfo = woodbury_step(
            self.geom, mat, sh, active, linfo["gram"], self.obs, self.lam,
            space=self.space, transforms=self.tr, chunk=self.chunk,
            max_bytes=self.max_bytes)

        # Backtrack. The reduced Hessian is near-singular, so a full step can overshoot
        # badly; accept the first fraction that actually lowers the objective.
        best = None
        for f in self.fractions:
            trial = mat + f * delta
            if self.space == "natural":
                trial = torch.max(torch.min(trial, self.upper), self.lower)
            l_try, sh_try, act_try, _ = self._eval(trial)
            if l_try < loss0:
                best = (trial, l_try, sh_try, f)
                break
        if best is None:
            # Nothing helped: shrink the trust region and leave the state untouched.
            self.lam = min(self.lam * 10.0, self.lam_ceiling)
            return {"accepted": False, "loss": loss0, "lam": self.lam, "frac": 0.0,
                    "active_converged": linfo["converged"]}

        trial, l_new, sh_new, frac = best
        self._scatter(trial)
        with torch.no_grad():
            self.sh_coeffs.copy_(sh_new)
        # Adapt the trust region to the step length that was actually usable. Accepting
        # only a heavily shortened step means the quadratic model is over-reaching, so
        # damping should RISE even though the iteration succeeded -- otherwise lam decays
        # on every success and the line search pays for it again next iteration.
        if frac >= self.fractions[0]:
            self.lam = max(self.lam / 3.0, 1e-12)
        elif frac <= self.fractions[-1]:
            self.lam = min(self.lam * 2.0, self.lam_ceiling)
        return {"accepted": True, "loss": l_new, "loss_prev": loss0, "lam": self.lam,
                "frac": frac, "active_converged": linfo["converged"]}


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
