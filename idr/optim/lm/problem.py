"""Assembling the Levenberg-Marquardt problem for the Cook-Torrance + SH model.

LM needs the objective expressed as residuals r with sum(r^2) equal to the loss the
gradient path minimises, plus (for the structured solvers) the per-pixel Jacobian
blocks. That machinery is ~325 lines and applies only to ct_sh -- the other three
models have no LM path -- so it lives here rather than inside the model.

The nested helpers are kept as closures over `build_lm_solver`'s arguments, exactly as
they were as closures over _optimize_ct_sh's locals. That list of arguments is not
incidental: it is the problem definition (geometry, observations, parameters,
transforms, settings).
"""
from __future__ import annotations

import math

import numpy as np
import torch

from idr.optim.losses import _sqrt_res, _tv_residuals
from idr.optim.transforms import _fwd_albedo, _fwd_metallic, _fwd_roughness
from idr.render import shade_ct_sh
from idr.render.brdf import _lut_lookup
from idr.render.sh import _sh_basis, _sh_irradiance
from .backends import _CGBackend, _SchurBackend, _lm_cfg_kwargs
from .core import LevenbergMarquardt

__all__ = ["build_lm_solver"]


def build_lm_solver(
        cfg, learnable, named_params, albedo_param, sh_coeffs, metallic_raw, roughness_raw,
        flat_mask, N_imgs, N_m, view_m, _imgs_m, lut, _diffuse_fresnel,
        tr_ab, tr_met, tr_rou, _lm_frac, dev, ftype):
    """Return (lm, batch_size, is_full_batch) for the LM path.

    `_lm_frac` is a one-element list shared with the model's `_forward`: it carries the
    current chunk fraction so the residual scaling tracks the loss scaling.
    """
    _lm_names = [n for n in ("albedo", "sh", "metallic", "roughness") if n in named_params]
    _hl_mode = str(cfg.get("hl_mode", "analytic"))
    _M = int(flat_mask.sum())
    _denom = float(N_imgs * _M * 3)    # so sum(r^2) over ALL images == loss_data

    def _lm_unpack(pt):
        d = dict(zip(_lm_names, pt))
        return (d.get("albedo",    albedo_param.detach()),
                d.get("sh",        sh_coeffs.detach()),
                d.get("metallic",  metallic_raw.detach()),
                d.get("roughness", roughness_raw.detach()))

    def _lm_maps(pt):
        ab_p, sh_c, met_r, rou_r = _lm_unpack(pt)
        ab = _fwd_albedo(ab_p, tr_ab)
        return (ab, ab.reshape(-1, 3)[flat_mask], sh_c,
                _fwd_metallic(met_r, tr_met), _fwd_roughness(rou_r, tr_rou))

    def _lm_data_residuals(pt, idx):
        ab_hw, ab_m, sh_c, met_hw, rou_hw = _lm_maps(pt)
        met_m = met_hw.reshape(-1, 1)[flat_mask]
        rou_m = rou_hw.reshape(-1, 1)[flat_mask]
        rec = torch.stack([
            shade_ct_sh(view_m, N_m, ab_m, sh_c[k], met_m, rou_m, lut=lut, hl_mode=_hl_mode,
                        diffuse_fresnel=_diffuse_fresnel)
            for k in idx.tolist()])                       # (n, M, 3)
        resid = rec - _imgs_m[idx]
        if cfg["loss"] == "L2":
            return (resid / math.sqrt(_denom)).reshape(-1)
        if cfg["loss"] == "L1":
            return _sqrt_res(resid.abs() / _denom).reshape(-1)
        _d = cfg.get("huber_delta", 0.05)
        _a = resid.abs()
        hub = torch.where(_a <= _d, 0.5 * resid**2, _d * (_a - 0.5 * _d))
        return _sqrt_res(hub / _denom).reshape(-1)

    def _lm_reg_residuals(pt):
        ab_hw, _, _, met_hw, rou_hw = _lm_maps(pt)
        fr = _lm_frac[0]
        out = []
        if cfg["lambda_sparse"]:
            out.append(_tv_residuals(ab_hw.permute(2, 0, 1), fr * cfg["lambda_sparse"]))
        if cfg["lambda_tv"]:
            s = fr * cfg["lambda_tv"]
            out.append(_tv_residuals(ab_hw.permute(2, 0, 1), s))
            out.append(_tv_residuals(met_hw.permute(2, 0, 1), s))
            out.append(_tv_residuals(rou_hw.permute(2, 0, 1), s))
        if cfg["lambda_white"]:                            # already quadratic
            out.append((math.sqrt(fr * cfg["lambda_white"])
                        * (ab_hw.mean() - 0.5)).reshape(1))
        _m = met_hw.reshape(-1, 1)[flat_mask]
        if cfg.get("lambda_metallic_l1", 0.0):
            out.append(_sqrt_res(fr * cfg["lambda_metallic_l1"] * _m.abs() / _m.numel()).reshape(-1))
        if cfg.get("lambda_metallic_binarize", 0.0):
            out.append(_sqrt_res(fr * cfg["lambda_metallic_binarize"]
                                 * (_m * (1.0 - _m)).clamp(min=0) / _m.numel()).reshape(-1))
        if not out:
            return ab_hw.new_zeros(0)
        return torch.cat(out)

    # ── block-sparse (structured) normal equations ────────────────────────
    # The data residual r[k,p,:] depends ONLY on pixel p's own 5 raw params
    # (albedo rgb, metallic, roughness) and on image k's SH coeffs. So J is
    # block-sparse and a dense jacrev/jacfwd wastes ~99.9% of its work. Build
    # J^T J directly from per-pixel blocks:
    #     H_pp[p]   = sum_k  Jp[k,p]^T Jp[k,p]        (npix x npix)
    #     H_ss[k]   = sum_p  Js[k,p]^T Js[k,p]        (3*n_sh x 3*n_sh)
    #     H_ps[p,k] =        Jp[k,p]^T Js[k,p]        (npix x 3*n_sh)
    # Each per-pixel jacobian costs only 3 VJPs (output dim 3), vmapped over
    # pixels — independent of the residual count.
    # Shared layout + per-pixel jacobian (used by the structured-dense, CG and
    # Schur paths alike).
    _gidx = torch.nonzero(flat_mask, as_tuple=False).squeeze(-1)   # grid idx of masked px
    _nsh  = sh_coeffs.shape[1]
    _nsh3 = _nsh * 3
    _inv_sd = 1.0 / math.sqrt(_denom)
    _offs, _o = {}, 0
    for _nm, _par in zip(_lm_names, learnable):   # NB: don't shadow the _t() helper
        _offs[_nm] = _o
        _o += _par.numel()
    _P = _o
    _cols, _local = [], []
    if "albedo" in _offs:
        _cols.append(_offs["albedo"] + _gidx[:, None] * 3
                     + torch.arange(3, device=dev, dtype=torch.long))
        _local += [0, 1, 2]
    if "metallic" in _offs:
        _cols.append((_offs["metallic"] + _gidx)[:, None]); _local += [3]
    if "roughness" in _offs:
        _cols.append((_offs["roughness"] + _gidx)[:, None]); _local += [4]
    _pix_idx = torch.cat(_cols, 1) if _cols else None          # (M, npix)
    _local_t = torch.tensor(_local, device=dev, dtype=torch.long)
    _npix = len(_local)
    _has_sh = "sh" in _offs
    _ar_nsh3 = torch.arange(_nsh3, device=dev, dtype=torch.long)
    _lm_shapes = [tuple(p.shape) for p in learnable]
    _lm_numels = [p.numel() for p in learnable]

    def _lm_unflat(theta):
        out, i = [], 0
        for shp, n in zip(_lm_shapes, _lm_numels):
            out.append(theta[i:i + n].view(shp)); i += n
        return tuple(out)

    def _px_fn(ab_r, met_r, rou_r, sh_k, v, n):
        ab = _fwd_albedo(ab_r, tr_ab)
        me = _fwd_metallic(met_r, tr_met)
        ro = _fwd_roughness(rou_r, tr_rou)
        return shade_ct_sh(v[None], n[None], ab[None], sh_k, me[None], ro[None],
                           lut=lut, diffuse_fresnel=_diffuse_fresnel, hl_mode=_hl_mode)[0] * _inv_sd

    _jac_px = torch.func.vmap(
        torch.func.jacrev(_px_fn, argnums=(0, 1, 2, 3)),
        in_dims=(0, 0, 0, None, 0, 0))

    def _lm_pixsh_maps(pt):
        d = dict(zip(_lm_names, pt))
        return (d.get("albedo",    albedo_param.detach()).reshape(-1, 3)[flat_mask],
                d.get("sh",        sh_coeffs.detach()),
                d.get("metallic",  metallic_raw.detach()).reshape(-1, 1)[flat_mask],
                d.get("roughness", roughness_raw.detach()).reshape(-1, 1)[flat_mask])

    def _lm_jac_and_resid(pt, k):
        """Per-pixel jacobian blocks + residual for one image."""
        ab_m, sh_c, me_m, ro_m = _lm_pixsh_maps(pt)
        with torch.no_grad():
            rec = shade_ct_sh(view_m, N_m, _fwd_albedo(ab_m, tr_ab), sh_c[k],
                              _fwd_metallic(me_m, tr_met), _fwd_roughness(ro_m, tr_rou),
                              lut=lut, diffuse_fresnel=_diffuse_fresnel, hl_mode=_hl_mode)
            r_k = (rec - _imgs_m[k]) * _inv_sd                       # (M, 3)
        Ja, Jm, Jr, Js = _jac_px(ab_m, me_m, ro_m, sh_c[k], view_m, N_m)
        Jpix = torch.cat([Ja, Jm, Jr], dim=-1)[..., _local_t]        # (M, 3, npix)
        return Jpix, Js.reshape(_gidx.numel(), 3, _nsh3), r_k

    def _lm_blocks(pt, idx):
        """Gauss-Newton diagonal blocks only: A (M,npix,npix), C (K,n_sh3,n_sh3).
        Used as the CG block-Jacobi preconditioner (data term only — a
        preconditioner never needs to be exact)."""
        A = torch.zeros(_gidx.numel(), _npix, _npix, device=dev, dtype=ftype) if _npix else None
        C = torch.zeros(len(idx), _nsh3, _nsh3, device=dev, dtype=ftype) if _has_sh else None
        for j, k in enumerate(idx.tolist()):
            Jpix, Jsf, _ = _lm_jac_and_resid(pt, k)
            if _npix:
                A += torch.einsum('mci,mcj->mij', Jpix, Jpix)
            if _has_sh:
                C[j] = torch.einsum('mca,mcb->ab', Jsf, Jsf)
        return A, C

    def _reg_px_fn(ab_r, met_r, rou_r):
        """Per-pixel regularizer residuals — only the pixel-SEPARABLE ones, which
        is all Schur permits. Squared-summed they reproduce the scalar terms."""
        me = _fwd_metallic(met_r, tr_met)
        fr = _lm_frac[0]
        out = []
        if cfg.get("lambda_metallic_l1", 0.0):
            out.append(_sqrt_res(fr * cfg["lambda_metallic_l1"] * me.abs() / _M))
        if cfg.get("lambda_metallic_binarize", 0.0):
            out.append(_sqrt_res(fr * cfg["lambda_metallic_binarize"]
                                 * (me * (1.0 - me)).clamp(min=0) / _M))
        if not out:
            return ab_r.new_zeros(0)
        return torch.cat(out)

    _jac_reg_px = torch.func.vmap(torch.func.jacrev(_reg_px_fn, argnums=(0, 1, 2)))
    _has_px_reg = bool(cfg.get("lambda_metallic_l1", 0.0)
                       or cfg.get("lambda_metallic_binarize", 0.0))

    def _lm_blocks_full(pt, idx):
        """A, C, B, g_p, g_s, loss  — everything the Schur complement needs.
        B is (M, npix, K*n_sh3): the memory bound of this backend.
        Includes the pixel-separable regularizers (they add to A and g_p only)."""
        K = len(idx)
        nb = _gidx.numel() * _npix * K * _nsh3 * (8 if ftype == torch.float64 else 4)
        cap = float(cfg.get("lm_schur_max_gb", 4.0)) * 1e9
        if nb > cap:
            raise MemoryError(
                f"Schur cross-block B needs {nb/1e9:.1f} GB (> lm_schur_max_gb="
                f"{cap/1e9:.1f}). Use lm_linear_solver='cg' (O(P) memory), or "
                f"lower the light count / resolution.")
        A = torch.zeros(_gidx.numel(), _npix, _npix, device=dev, dtype=ftype)
        C = torch.zeros(K, _nsh3, _nsh3, device=dev, dtype=ftype)
        B = torch.zeros(_gidx.numel(), _npix, K * _nsh3, device=dev, dtype=ftype)
        gp = torch.zeros(_gidx.numel(), _npix, device=dev, dtype=ftype)
        gs = torch.zeros(K, _nsh3, device=dev, dtype=ftype)
        loss = 0.0
        for j, k in enumerate(idx.tolist()):
            Jpix, Jsf, r_k = _lm_jac_and_resid(pt, k)
            loss += float(r_k.pow(2).sum())
            A  += torch.einsum('mci,mcj->mij', Jpix, Jpix)
            gp += torch.einsum('mci,mc->mi',  Jpix, r_k)
            C[j] = torch.einsum('mca,mcb->ab', Jsf, Jsf)
            gs[j] = torch.einsum('mca,mc->a',  Jsf, r_k)
            B[:, :, j * _nsh3:(j + 1) * _nsh3] = torch.einsum('mci,mca->mia', Jpix, Jsf)

        # Pixel-separable regularizers: they touch only each pixel's own params,
        # so they add to A and g_p (never to B or C). Omitting them would make
        # the Schur step solve a DIFFERENT system than the loss LM accepts on.
        if _has_px_reg:
            ab_m, _, me_m, ro_m = _lm_pixsh_maps(pt)
            with torch.no_grad():
                r_reg = torch.func.vmap(_reg_px_fn)(ab_m, me_m, ro_m)       # (M, nr)
                loss += float(r_reg.pow(2).sum())
            Ra, Rm, Rr = _jac_reg_px(ab_m, me_m, ro_m)
            Jreg = torch.cat([Ra, Rm, Rr], dim=-1)[..., _local_t]           # (M, nr, npix)
            A  += torch.einsum('mri,mrj->mij', Jreg, Jreg)
            gp += torch.einsum('mri,mr->mi',  Jreg, r_reg)
        return A, C, B, gp, gs, loss

    # ── block-sparse (structured) DENSE normal equations ──────────────────
    _lm_structured = bool(cfg.get("lm_structured", False))
    _structured_gn = None
    if _lm_structured:

        def _structured_gn(pt, idx):
            d = dict(zip(_lm_names, pt))
            ab_p  = d.get("albedo",    albedo_param.detach())
            sh_c  = d.get("sh",        sh_coeffs.detach())
            met_r = d.get("metallic",  metallic_raw.detach())
            rou_r = d.get("roughness", roughness_raw.detach())
            ab_m = ab_p.reshape(-1, 3)[flat_mask]
            me_m = met_r.reshape(-1, 1)[flat_mask]
            ro_m = rou_r.reshape(-1, 1)[flat_mask]
            ab_f = _fwd_albedo(ab_m, tr_ab)
            me_f = _fwd_metallic(me_m, tr_met)
            ro_f = _fwd_roughness(ro_m, tr_rou)

            JJ  = torch.zeros(_P, _P, device=dev, dtype=ftype)
            rhs = torch.zeros(_P, device=dev, dtype=ftype)
            flat = JJ.view(-1)
            loss = 0.0
            Hpp = gp = None
            for k in idx.tolist():
                with torch.no_grad():
                    rec = shade_ct_sh(view_m, N_m, ab_f, sh_c[k], me_f, ro_f,
                                      lut=lut, diffuse_fresnel=_diffuse_fresnel, hl_mode=_hl_mode)
                    r_k = (rec - _imgs_m[k]) * _inv_sd                  # (M, 3)
                    loss += float(r_k.pow(2).sum())
                Ja, Jm, Jr, Js = _jac_px(ab_m, me_m, ro_m, sh_c[k], view_m, N_m)
                Jpix = torch.cat([Ja, Jm, Jr], dim=-1)[..., _local_t]   # (M, 3, npix)
                Jsf  = Js.reshape(_gidx.numel(), 3, _nsh3)              # (M, 3, 3*n_sh)
                if _npix:
                    _hpp = torch.einsum('mci,mcj->mij', Jpix, Jpix)
                    _gp  = torch.einsum('mci,mc->mi',  Jpix, r_k)
                    Hpp = _hpp if Hpp is None else Hpp + _hpp
                    gp  = _gp  if gp  is None else gp  + _gp
                if _has_sh:
                    sh_i = _offs["sh"] + k * _nsh3 + _ar_nsh3           # (3*n_sh,)
                    _hss = torch.einsum('mca,mcb->ab', Jsf, Jsf)
                    _gs  = torch.einsum('mca,mc->a',  Jsf, r_k)
                    flat.index_add_(0, (sh_i[:, None] * _P + sh_i[None, :]).reshape(-1),
                                    _hss.reshape(-1))
                    rhs.index_add_(0, sh_i, _gs)
                    if _npix:
                        _hps = torch.einsum('mci,mca->mia', Jpix, Jsf)  # (M, npix, 3*n_sh)
                        rws = _pix_idx[:, :, None].expand(-1, -1, _nsh3)
                        cls = sh_i.view(1, 1, -1).expand(rws.shape[0], _npix, -1)
                        flat.index_add_(0, (rws * _P + cls).reshape(-1), _hps.reshape(-1))
                        flat.index_add_(0, (cls * _P + rws).reshape(-1), _hps.reshape(-1))
            if _npix:
                rws = _pix_idx[:, :, None].expand(-1, -1, _npix)
                cls = _pix_idx[:, None, :].expand(-1, _npix, -1)
                flat.index_add_(0, (rws * _P + cls).reshape(-1), Hpp.reshape(-1))
                rhs.index_add_(0, _pix_idx.reshape(-1), gp.reshape(-1))
            return JJ, rhs, loss

    _has_reg = any(cfg.get(k, 0.0) for k in
                   ("lambda_sparse", "lambda_tv", "lambda_white",
                    "lambda_metallic_l1", "lambda_metallic_binarize"))
    # Regularizers that couple DIFFERENT pixels: they destroy the block-diagonal
    # pixel Hessian that the exact Schur complement relies on.
    _pix_coupled = any(cfg.get(k, 0.0) for k in
                       ("lambda_tv", "lambda_sparse", "lambda_white"))

    # ── choose the linear solver ──────────────────────────────────────────
    # P = M*5 + K*3*n_sh. A dense P x P matrix is 25 GB at 124^2 and 6.9 TB at
    # 512^2, so above ~20k params we must never form it:
    #   schur : exact, eliminates the block-diagonal per-pixel 5x5 blocks and
    #           solves the tiny reduced SH system. Needs pixel-separable regs.
    #   cg    : matrix-free preconditioned CG. O(P) memory, keeps every
    #           regularizer, inner solve is inexact (truncated Newton).
    _lin = str(cfg.get("lm_linear_solver", "auto")).lower()
    if _lin == "auto":
        if _P <= int(cfg.get("lm_dense_max_params", 20000)):
            _lin = "dense"
        else:
            _lin = "cg" if _pix_coupled else "schur"
    if _lin == "schur" and _pix_coupled:
        raise ValueError(
            "lm_linear_solver='schur' needs pixel-separable regularizers: set "
            "lambda_tv=lambda_sparse=lambda_white=0 (metallic_l1/binarize are fine), "
            "or use lm_linear_solver='cg'.")
    if _lin == "dense" and _P > 40000:
        print(f"  [LM] WARNING: dense solver with P={_P} -> JtJ is "
              f"{_P**2*(8 if ftype==torch.float64 else 4)/1e9:.1f} GB")

    _lm_backend = None
    if _lin == "cg":
        _lm_backend = _CGBackend(
            unflatten=_lm_unflat, data_res_fn=_lm_data_residuals,
            reg_res_fn=_lm_reg_residuals if _has_reg else None,
            blocks_fn=_lm_blocks, pix_idx=_pix_idx,
            sh_off=_offs.get("sh", 0), n_sh3=_nsh3, n_imgs=N_imgs,
            img_chunk=int(cfg.get("lm_image_chunk", 8)),
            tol=float(cfg.get("lm_cg_tol", 1e-4)),
            maxiter=int(cfg.get("lm_cg_maxiter", 50)),
            dev=dev, dtype=ftype)
    elif _lin == "schur":
        _lm_backend = _SchurBackend(
            blocks_full_fn=_lm_blocks_full, pix_idx=_pix_idx,
            sh_off=_offs.get("sh", 0), n_sh3=_nsh3, dev=dev, dtype=ftype)

    _lm = LevenbergMarquardt(
        learnable, _lm_data_residuals, n_samples=N_imgs,
        reg_residuals_fn=_lm_reg_residuals if _has_reg else None,
        structured_gn_fn=_structured_gn if _lin == "dense" else None,
        linear_backend=_lm_backend,
        **_lm_cfg_kwargs(cfg))
    _lm_bs = int(cfg.get("lm_batch_size", 0) or 0)
    _lm_full = not (0 < _lm_bs < N_imgs)
    _jac_desc = ('block-sparse' if (_lin == 'dense' and _lm_structured)
                 else ('blocks+autograd' if _lin in ('cg', 'schur')
                       else cfg.get('lm_jacobian_mode', 'auto')))
    print(f"  [LM] P={_lm.n_params}  samples={N_imgs}  "
          f"{'full batch' if _lm_full else f'batch={_lm_bs}'}  "
          f"linear={_lin}  damping={cfg.get('lm_damping','standard')}  jacobian={_jac_desc}")

    return _lm, _lm_bs, _lm_full
