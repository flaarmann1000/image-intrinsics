"""Shared machinery for the ``pixel_diagnostics*`` notebooks.

The notebooks (``pixel_diagnostics``, ``…_ADAM``, ``…_no_transform``,
``sh_order_diagnostics``) differ only in their config; the analysis is identical.
This module holds that analysis so each notebook is just config + thin calls:

    from pixel_diag import PixelDiag, gt_render_floor, compare_floor
    D = PixelDiag.load(SCENE, CFG, N_IMAGES, DOWNSAMPLE, DEVICE, N_WORST, SWEEP_N)
    D.fit()
    D.show_worst(); D.show_landscape("loss"); D.show_landscape("spec")
    D.show_grad_breakdown(); D.show_residual_attribution(); D.show_demand()
    D.show_basins(); D.show_varpro()

Every renderer path goes through ``shade_ct_sh`` and the proxy geometry is built
full-res then strided (``make_proxy_geometry(..., stride=downsample)``), so the view
vectors match a full-res render at any downsample.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

from idr.render import shade_ct_sh
from idr.render.brdf import _get_ggx_sh_lut
from idr.render.sh import _sh_basis
from idr.data.geometry import make_proxy_geometry
from idr.optim.registry import optimize

__all__ = ["PixelDiag", "load_bundle", "gt_render_floor", "compare_floor"]


# ──────────────────────────────────────────────────────────────────────────────
# Scene loading (synthetic golden OR a rendered dataset leaf)
# ──────────────────────────────────────────────────────────────────────────────
def load_bundle(scene, n_images, downsample, device, sh_order):
    """-> dict with GT maps, masked geometry, observations, and GT SH lighting."""
    # The reconstruction must use the SAME specular band source the dataset was rendered
    # with, or r(GT,GT) no longer bottoms out at ~0 (the inverse crime). The synthetic
    # golden scene is analytic; a real dataset records its mode in config.json (datasets
    # rendered before the analytic default lack the key and were LUT-rendered).
    hl_mode = "analytic"
    if str(scene) == "synthetic":
        from tests.golden import build_scene
        sc = build_scene(device)
        maps = dict(normals=sc["normals"], mask=sc["mask"], albedo=sc["albedo"],
                    roughness=sc["roughness"], metallic=sc["metallic"])
        images = sc["images"][:n_images]
        sh_gt = [np.asarray(s, np.float32) for s in sc["sh"][:n_images]]
        geo_normals, geo_mask, geo_stride = maps["normals"], maps["mask"], 1
    else:
        _cfgp = Path(scene) / "config.json"
        if _cfgp.exists():
            import json
            hl_mode = json.loads(_cfgp.read_text()).get("hl_mode", "lut")
        else:
            hl_mode = "lut"
        from idr.data.scene_io import load_scene
        s = load_scene(Path(scene), gt_npy=True)
        st = lambda a: np.ascontiguousarray(a[::downsample, ::downsample])
        maps = dict(normals=st(s["normals_np"]), mask=st(s["mask_np"]),
                    albedo=st(s["albedo_np"]), roughness=st(s["roughness_np"]),
                    metallic=st(s["metallic_np"]))
        images = [st(im) for im in s["images"][:n_images]]
        if s.get("sh_coeffs") is None:
            raise ValueError("scene has no GT sh_*.npy — needed to render observations")
        sh_gt = [np.asarray(c, np.float32) for c in s["sh_coeffs"][:n_images]]
        # Build geometry from the FULL-res normals/mask, strided, so the view vectors
        # match a full-res render instead of the coarse grid (see make_proxy_geometry).
        geo_normals, geo_mask, geo_stride = s["normals_np"], s["mask_np"], downsample

    H, W = maps["normals"].shape[:2]
    Nhw, frag, mhw, cam = make_proxy_geometry(geo_normals, geo_mask, 60.0, 2.0,
                                              device, torch.float32, stride=geo_stride)
    fm = mhw.reshape(-1)
    N_m = Nhw.reshape(-1, 3)[fm]
    V_m = torch.nn.functional.normalize(cam[None] - frag.reshape(-1, 3)[fm], dim=-1)
    lut = _get_ggx_sh_lut(device, n_bands=int(sh_order) + 1).to(torch.float32)
    obs = torch.stack([torch.from_numpy(im).to(device, torch.float32)
                       for im in images]).reshape(len(images), -1, 3)[:, fm, :]
    # Pad (or truncate) the GT SH to the fit's order so sh_t matches sh_est — e.g. fitting a
    # dataset rendered at order 2 (9 coeffs) with cfg sh_order=3 (16) pads band 3 with zeros
    # (which is the correct GT: an order-2 scene has no band-3 lighting).
    _n_sh = (int(sh_order) + 1) ** 2
    def _pad_sh(a):
        a = np.asarray(a, np.float32)
        if a.shape[0] < _n_sh:
            a = np.concatenate([a, np.zeros((_n_sh - a.shape[0], 3), np.float32)], axis=0)
        return a[:_n_sh]
    sh_t = torch.from_numpy(np.stack([_pad_sh(c) for c in sh_gt])).to(device, torch.float32)

    def _gt(k):
        a = torch.from_numpy(maps[k].reshape(-1, 3 if k == "albedo" else 1)).to(device, torch.float32)[fm]
        return a if k == "albedo" else a[:, 0]
    gt_px = {k: _gt(k) for k in ("albedo", "roughness", "metallic")}
    return dict(maps=maps, H=H, W=W, fm=fm, N_m=N_m, V_m=V_m, lut=lut, obs=obs,
                sh_t=sh_t, images=images, gt_px=gt_px, geom=(Nhw, frag, mhw, cam),
                hl_mode=hl_mode)


# ──────────────────────────────────────────────────────────────────────────────
# The diagnostics context
# ──────────────────────────────────────────────────────────────────────────────
class PixelDiag:
    def __init__(self, S, cfg, device, n_worst=4, sweep_n=121):
        self.S, self.cfg, self.device = S, dict(cfg), device
        # Reconstruct with the band source the dataset was rendered with. `hl_mode` is a
        # property so that assigning `D.hl_mode = "..."` also updates cfg["hl_mode"] — every
        # diagnostic (which reads self.hl_mode) AND fit() (which reads the cfg) then agree.
        self.hl_mode = S.get("hl_mode", "analytic")
        self.n_worst, self.sweep_n = int(n_worst), int(sweep_n)
        self.N_imgs = S["obs"].shape[0]
        self.est = self.sh_est = self.res = None
        self._err = self._worst = None
        self._scene = None; self._downsample = 1     # stashed by load() for compare_images

    @property
    def hl_mode(self):
        return self._hl_mode

    @hl_mode.setter
    def hl_mode(self, v):
        # keep the fit cfg in lock-step, so `D.hl_mode = "analytic"; D.fit()` actually fits
        # analytic instead of silently reusing the cfg's stale value.
        self._hl_mode = v
        self.cfg["hl_mode"] = v

    # ---- construction --------------------------------------------------------
    @classmethod
    def load(cls, scene, cfg, n_images, downsample, device, n_worst=4, sweep_n=121):
        sh_order = int(cfg.get("sh_order", 2))
        S = load_bundle(scene, n_images, downsample, device, sh_order)
        print(f"{S['H']}x{S['W']}, {int(S['fm'].sum())} masked px, "
              f"{S['obs'].shape[0]} images, SH order {sh_order}, hl_mode={S.get('hl_mode', 'analytic')}")
        d = cls(S, cfg, device, n_worst, sweep_n)
        d._scene, d._downsample = scene, downsample
        return d

    def fit(self, cfg=None):
        """Run the decomposition; sets self.est / self.sh_est (returns None so the
        notebook cell shows only the fit log, not a repr).

        A ``curriculum`` key is honoured (``optimize()`` alone ignores it): each phase runs in
        turn, threading its maps into the next via init_maps, and the base cfg is the FINAL
        stage warm-started from them — same semantics as decompose_scene's curriculum."""
        S, cfg = self.S, dict(cfg or self.cfg)
        cfg.setdefault("hl_mode", self.hl_mode)     # match the dataset's render mode
        Nhw, frag, mhw, cam = S["geom"]
        gt_sh = [c.cpu().numpy() for c in S["sh_t"]]; gt_alb = S["maps"]["albedo"]
        curr = cfg.pop("curriculum", None)
        im = None
        for pi, ph in enumerate(curr or []):
            print(f"  [curriculum {pi + 1}/{len(curr)}] "
                  f"{ph.get('optimizer', cfg.get('optimizer', 'LBFGS'))} "
                  f"n_iter={ph.get('n_iter')}", flush=True)
            rp = optimize("ct_sh", S["images"], Nhw, frag, mhw, cam, S["maps"]["metallic"],
                          S["maps"]["roughness"], {**cfg, **ph},
                          gt_sh_coeffs=gt_sh, gt_albedo=gt_alb, init_maps=im)
            im = {"albedo": rp.albedo, "sh": rp.light, "metallic": rp.mat_a, "roughness": rp.mat_b}
        res = optimize("ct_sh", S["images"], Nhw, frag, mhw, cam,
                       S["maps"]["metallic"], S["maps"]["roughness"], cfg,
                       gt_sh_coeffs=gt_sh, gt_albedo=gt_alb, init_maps=im)
        fm = S["fm"]
        self.est = {
            "albedo":    torch.from_numpy(res.albedo).to(self.device).reshape(-1, 3)[fm],
            "roughness": torch.from_numpy(res.mat_b).to(self.device).reshape(-1, 1)[fm, 0],
            "metallic":  torch.from_numpy(res.mat_a).to(self.device).reshape(-1, 1)[fm, 0],
        }
        self.sh_est = torch.as_tensor(np.asarray(res.light), dtype=torch.float32, device=self.device)
        self.res = res
        self._err = self._worst = None
        print("fitted:", {k: tuple(v.shape) for k, v in self.est.items()},
              "| sh_est", tuple(self.sh_est.shape))

    # ---- render primitives ---------------------------------------------------
    def _bands_exact(self, roughness, n_theta=4096):
        """GGX zonal-SH band coefficients h_l(roughness), computed LUT-FREE by fine
        quadrature at the exact roughness (same integral as brdf._compute_ggx_sh_lut,
        but no roughness-grid discretisation or linear interpolation). -> (n_bands,).

        Used to overlay an *analytic* specular on the LUT-based one, to check whether the
        LUT's linear interpolation introduces kinks / spurious minima in the roughness axis.
        """
        n_bands = int(self.cfg.get("sh_order", 2)) + 1
        a2 = max((float(roughness) ** 2) ** 2, 1e-12)
        th = torch.linspace(0.0, float(np.pi / 2), n_theta, device=self.device, dtype=torch.float64)
        ct, st = th.cos(), th.sin()
        cth2 = (th / 2).cos() ** 2
        D = a2 / (np.pi * (cth2 * (a2 - 1.0) + 1.0) ** 2)          # GGX NDF kernel, no cosθ
        Pl = [torch.ones_like(ct), ct, 0.5 * (3 * ct ** 2 - 1),
              0.5 * (5 * ct ** 3 - 3 * ct)][:n_bands]              # Legendre / zonal
        h = torch.stack([2 * np.pi * torch.trapezoid(D * p * st, th) for p in Pl])
        return h.to(torch.float32)                                # (n_bands,)

    def _spec_from_bands(self, comp, sh_k, B, order):
        """Specular radiance from explicit band coeffs B (n_bands,), reusing the LUT
        render's F, G1 and reflection vector R. Mirrors shade_ct_sh's specular exactly
        except the band coefficients come from B instead of the LUT lookup."""
        Y = _sh_basis(comp["R"], order=order)                     # (1, n_sh)
        parts = [B[0:1], B[1:2].expand(3), B[2:3].expand(5)]
        if order >= 3:
            parts.append(B[3:4].expand(7))
        Bexp = torch.cat(parts)                                   # (n_sh,)
        L = ((Bexp * Y) @ sh_k).clamp(min=0)                      # (1, 3)
        return comp["F"] * comp["G1"] * L / 4.0                   # (1, 3)

    def render_px(self, pi, albedo, roughness, metallic, sh=None, want_spec=False, analytic=False):
        """Render pixel `pi` under every image. `analytic=True` swaps the LUT specular
        for the on-the-fly quadrature bands (masked pixels are front-facing, so front=1)."""
        S = self.S
        sh = self.sh_est if sh is None else sh
        v, n = S["V_m"][pi:pi + 1], S["N_m"][pi:pi + 1]
        a = albedo[None] if albedo.ndim == 1 else albedo
        order = int(self.cfg.get("sh_order", 2))
        Bx = self._bands_exact(roughness) if analytic else None
        need = want_spec or analytic
        recon, spec = [], []
        for k in range(self.N_imgs):
            out = shade_ct_sh(v, n, a, sh[k], metallic[None, None], roughness[None, None],
                              lut=S["lut"], hl_mode=self.hl_mode, return_components=need)
            if need:
                out, comp = out
                spec_k = comp["spec"]
                if analytic:
                    spec_k = self._spec_from_bands(comp, sh[k], Bx, order)
                    out = out - comp["spec"] + spec_k             # replace LUT specular
                if want_spec:
                    spec.append(spec_k)
            recon.append(out)
        recon = torch.cat(recon, 0)
        return (recon, torch.cat(spec, 0)) if want_spec else recon

    def loss_at(self, pi, mat, sh):
        S = self.S
        a, r, m = mat
        rec = torch.cat([shade_ct_sh(S["V_m"][pi:pi + 1], S["N_m"][pi:pi + 1], a[None], sh[k],
                                     m[None, None], r[None, None], lut=S["lut"], hl_mode=self.hl_mode)
                         for k in range(self.N_imgs)], 0)
        return float(((rec - S["obs"][:, pi, :]) ** 2).sum())

    def mat_fit(self, pi):
        return (self.est["albedo"][pi], self.est["roughness"][pi], self.est["metallic"][pi])

    def mat_gt(self, pi):
        g = self.S["gt_px"]
        return (g["albedo"][pi], g["roughness"][pi], g["metallic"][pi])

    # ---- worst pixels (cached) -----------------------------------------------
    @property
    def err(self):
        if self._err is None:
            self._compute_worst()
        return self._err

    @property
    def worst(self):
        if self._worst is None:
            self._compute_worst()
        return self._worst

    def _compute_worst(self):
        S, est = self.S, self.est
        with torch.no_grad():
            M = int(S["fm"].sum())
            err = torch.empty(M, device=self.device)
            for pi in range(M):
                r = self.render_px(pi, est["albedo"][pi], est["roughness"][pi], est["metallic"][pi])
                err[pi] = ((r - S["obs"][:, pi, :]) ** 2).mean().sqrt()
        self._err = err
        self._worst = torch.argsort(err, descending=True)[:self.n_worst].tolist()

    # ── §4 reconstruction error → worst pixels ────────────────────────────────
    def show_worst(self):
        S, est, err, worst = self.S, self.est, self.err, self.worst
        fm = S["fm"]
        print("worst-pixel RMSE:", [round(float(err[p]), 4) for p in worst])
        emap = np.full(S["H"] * S["W"], np.nan, np.float32)
        emap[fm.cpu().numpy()] = err.cpu().numpy()
        emap = emap.reshape(S["H"], S["W"])
        fg_idx = np.flatnonzero(fm.cpu().numpy())
        ys, xs = np.divmod(fg_idx[worst], S["W"])
        print("\nintrinsics for worst pixels (fit  |  GT):")
        print(f"  {'#':>2} {'px':>5} {'row,col':>9} {'RMSE':>7}   "
              f"{'albedo (RGB)':>22}   {'rough':>12}   {'metal':>12}")
        for i, pi in enumerate(worst):
            a = est["albedo"][pi].tolist(); ag = S["gt_px"]["albedo"][pi].tolist()
            r = float(est["roughness"][pi]); rg = float(S["gt_px"]["roughness"][pi])
            m = float(est["metallic"][pi]); mg = float(S["gt_px"]["metallic"][pi])
            print(f"  #{i:<1} {fg_idx[worst[i]]:>5} ({int(ys[i]):>2},{int(xs[i]):>2}) "
                  f"{float(err[pi]):7.4f}   "
                  f"[{a[0]:.2f} {a[1]:.2f} {a[2]:.2f}]|[{ag[0]:.2f} {ag[1]:.2f} {ag[2]:.2f}]   "
                  f"{r:.3f}|{rg:.3f}   {m:.3f}|{mg:.3f}")
        fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
        im = ax[0].imshow(emap, cmap="inferno"); ax[0].set_title("per-pixel recon RMSE")
        ax[0].scatter(xs, ys, s=80, facecolors="none", edgecolors="cyan", linewidths=1.8)
        for i, (x, y) in enumerate(zip(xs, ys)):
            ax[0].annotate(f"#{i}", (x, y), color="cyan", fontsize=9,
                           xytext=(4, 4), textcoords="offset points")
        plt.colorbar(im, ax=ax[0], fraction=0.046)
        ax[1].hist(err.cpu().numpy(), bins=40, color="steelblue")
        for p in worst:
            ax[1].axvline(float(err[p]), color="crimson", lw=1)
        ax[1].set_title("RMSE distribution (worst marked)"); ax[1].set_xlabel("RMSE")
        ax[0].axis("off"); plt.tight_layout(); plt.show()

    # ── §5/§6 per-parameter sweeps ────────────────────────────────────────────
    PARAMS = [("albedo R", 0), ("albedo G", 1), ("albedo B", 2),
              ("roughness", "roughness"), ("metallic", "metallic")]
    RANGES = {"albedo R": (0.0, 1.0), "albedo G": (0.0, 1.0), "albedo B": (0.0, 1.0),
              "roughness": (0.03, 1.0), "metallic": (0.0, 1.0)}

    def sweep_pixel(self, pi, key, grid, metric="loss", analytic=False):
        est, S = self.est, self.S
        a0 = est["albedo"][pi].clone(); r0 = est["roughness"][pi].clone(); m0 = est["metallic"][pi].clone()
        obs = S["obs"][:, pi, :]
        out = np.empty(len(grid), np.float32)
        with torch.no_grad():
            for j, val in enumerate(grid):
                a, r, m = a0.clone(), r0.clone(), m0.clone()
                if key in (0, 1, 2): a[key] = val
                elif key == "roughness": r = torch.tensor(float(val), device=self.device)
                else: m = torch.tensor(float(val), device=self.device)
                if metric == "loss":
                    out[j] = float(((self.render_px(pi, a, r, m, analytic=analytic) - obs) ** 2).sum())
                else:
                    _, sp = self.render_px(pi, a, r, m, want_spec=True, analytic=analytic)
                    out[j] = float(sp.abs().mean())
        return out

    def pred_gt(self, pi, key):
        est, g = self.est, self.S["gt_px"]
        if key in (0, 1, 2):
            return float(est["albedo"][pi][key]), float(g["albedo"][pi][key])
        k = "roughness" if key == "roughness" else "metallic"
        return float(est[k][pi]), float(g[k][pi])

    def show_landscape(self, metric="loss", analytic=False):
        """Per-parameter sweeps. ``analytic=True`` overlays a second curve computed with
        the LUT-free (on-the-fly quadrature) specular — if the LUT (solid) wiggles where the
        analytic (dashed orange) is smooth, the roughness-axis artefact is the LUT's."""
        ylabel = "data loss" if metric == "loss" else "mean |spec|"
        title = ("Loss landscape per parameter" if metric == "loss"
                 else "Specular magnitude per parameter")
        title += "  (dashed red=pred, solid green=GT" + (
            "; blue=LUT, orange=analytic)" if analytic else ")")
        worst = self.worst
        fig, ax = plt.subplots(self.n_worst, len(self.PARAMS),
                               figsize=(3.0 * len(self.PARAMS), 2.4 * self.n_worst), squeeze=False)
        for i, pi in enumerate(worst):
            for j, (name, key) in enumerate(self.PARAMS):
                lo, hi = self.RANGES[name]; grid = np.linspace(lo, hi, self.sweep_n)
                y = self.sweep_pixel(pi, key, grid, metric=metric)
                pred, gt = self.pred_gt(pi, key)
                a = ax[i][j]
                a.plot(grid, y, color="steelblue", lw=1.6, label="LUT")
                if analytic:
                    ya = self.sweep_pixel(pi, key, grid, metric=metric, analytic=True)
                    a.plot(grid, ya, color="darkorange", ls="--", lw=1.4, label="analytic")
                a.axvline(pred, color="crimson", ls="--", lw=1.4, label="pred")
                a.axvline(gt, color="green", ls="-", lw=1.4, label="GT")
                if i == 0: a.set_title(name, fontsize=10)
                if j == 0: a.set_ylabel(f"px #{i}\n{ylabel}", fontsize=8)
                a.tick_params(labelsize=7)
        ax[0][-1].legend(fontsize=7, loc="best")
        fig.suptitle(title, fontsize=12); plt.tight_layout(); plt.show()

    # ── §7 gradient breakdown ─────────────────────────────────────────────────
    def grad_split(self, pi):
        S, est, sh_est = self.S, self.est, self.sh_est
        obs = S["obs"][:, pi, :]
        v, n = S["V_m"][pi:pi + 1], S["N_m"][pi:pi + 1]

        def render_terms(a, r, m):
            diff, spec = [], []
            for k in range(self.N_imgs):
                _, comp = shade_ct_sh(v, n, a[None], sh_est[k], m[None, None], r[None, None],
                                      lut=S["lut"], hl_mode=self.hl_mode, return_components=True)
                diff.append(comp["diff"]); spec.append(comp["spec"])
            return torch.cat(diff, 0), torch.cat(spec, 0)

        with torch.no_grad():
            d0, s0 = render_terms(est["albedo"][pi], est["roughness"][pi], est["metallic"][pi])
            r_k = (d0 + s0) - obs

        def path_grad(term):
            a = est["albedo"][pi].clone().requires_grad_(True)
            r = est["roughness"][pi].clone().requires_grad_(True)
            m = est["metallic"][pi].clone().requires_grad_(True)
            diff, spec = render_terms(a, r, m)
            comp = diff if term == "diff" else spec
            surrogate = (2.0 * r_k.detach() * comp).sum()
            ga, gr, gm = torch.autograd.grad(surrogate, (a, r, m), allow_unused=True)
            z = lambda t: 0.0 if t is None else float(t)
            ga = np.zeros(3, np.float32) if ga is None else ga.detach().cpu().numpy()
            return ga, z(gr), z(gm)

        gd = path_grad("diff"); gs = path_grad("spec")
        out = {}
        for idx, nm in [(0, "albedo R"), (1, "albedo G"), (2, "albedo B")]:
            out[nm] = (gd[0][idx] + gs[0][idx], gd[0][idx], gs[0][idx])
        out["roughness"] = (gd[1] + gs[1], gd[1], gs[1])
        out["metallic"] = (gd[2] + gs[2], gd[2], gs[2])
        return out

    def show_grad_breakdown(self):
        worst, err = self.worst, self.err
        names = ["albedo R", "albedo G", "albedo B", "roughness", "metallic"]
        fig, ax = plt.subplots(1, self.n_worst, figsize=(3.4 * self.n_worst, 3.6), squeeze=False)
        for i, pi in enumerate(worst):
            g = self.grad_split(pi)
            xloc = np.arange(len(names)); w = 0.4
            diff = [g[nm][1] for nm in names]; spec = [g[nm][2] for nm in names]
            a = ax[0][i]
            a.bar(xloc - w / 2, diff, w, label="diffuse path", color="#4c78a8")
            a.bar(xloc + w / 2, spec, w, label="specular path", color="#e45756")
            a.axhline(0, color="k", lw=0.6)
            a.set_xticks(xloc); a.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
            a.set_title(f"px #{i}  (RMSE {float(err[worst[i]]):.3f})", fontsize=9)
            if i == 0: a.set_ylabel("d(loss)/d(param)")
        ax[0][0].legend(fontsize=8)
        fig.suptitle("Gradient composition: diffuse vs specular path", fontsize=12)
        plt.tight_layout(); plt.show()

        print(f"pixel #0  (flat masked index {worst[0]}):")
        g = self.grad_split(worst[0])
        print(f"  {'param':10} {'total':>12} {'diffuse':>12} {'specular':>12}   pred / GT")
        for nm, key in [("albedo R", 0), ("albedo G", 1), ("albedo B", 2),
                        ("roughness", "roughness"), ("metallic", "metallic")]:
            tot, gdif, gsp = g[nm]
            pr, gt = self.pred_gt(worst[0], key)
            print(f"  {nm:10} {tot:12.4e} {gdif:12.4e} {gsp:12.4e}   {pr:.3f} / {gt:.3f}")

    # ── §8 residual attribution + unmet demand ────────────────────────────────
    def render_all(self, sh):
        S, est = self.S, self.est
        return torch.stack([shade_ct_sh(S["V_m"], S["N_m"], est["albedo"], sh[k],
                                        est["metallic"][:, None], est["roughness"][:, None],
                                        lut=S["lut"], hl_mode=self.hl_mode)
                            for k in range(self.N_imgs)], 0)

    def show_residual_attribution(self):
        S, worst = self.S, self.worst
        sh_gt, sh_est = S["sh_t"], self.sh_est
        fig, ax = plt.subplots(1, self.n_worst, figsize=(3.0 * self.n_worst, 3.0), squeeze=False)
        with torch.no_grad():
            for i, pi in enumerate(worst):
                A = np.array([[self.loss_at(pi, self.mat_gt(pi), sh_gt), self.loss_at(pi, self.mat_fit(pi), sh_gt)],
                              [self.loss_at(pi, self.mat_gt(pi), sh_est), self.loss_at(pi, self.mat_fit(pi), sh_est)]])
                a = ax[0][i]
                a.imshow(np.log10(A + 1e-16), cmap="viridis")
                a.set_xticks([0, 1]); a.set_xticklabels(["GT mat", "fit mat"], fontsize=8)
                a.set_yticks([0, 1]); a.set_yticklabels(["GT light", "fit light"], fontsize=8)
                for (yy, xx), v in np.ndenumerate(A):
                    a.text(xx, yy, f"{v:.1e}", ha="center", va="center",
                           color="w" if np.log10(v + 1e-16) < np.log10(A + 1e-16).mean() else "k", fontsize=8)
                a.set_title(f"px #{i}", fontsize=9)
        fig.suptitle("Residual attribution  log10 loss  (bottom-right = the fit)", fontsize=11)
        plt.tight_layout(); plt.show()

    def show_demand(self):
        S, worst, sh_est = self.S, self.worst, self.sh_est
        fm = S["fm"]
        she = sh_est.clone().requires_grad_(True)
        G_sh, = torch.autograd.grad(((self.render_all(she) - S["obs"]) ** 2).sum(), she)
        g_glob = float(G_sh.norm()) / int(fm.sum())
        g_light, g_mat = [], []
        for pi in worst:
            a, r, m = self.mat_fit(pi)
            sh2 = sh_est.clone().requires_grad_(True)
            Lp = ((torch.cat([shade_ct_sh(S["V_m"][pi:pi + 1], S["N_m"][pi:pi + 1], a[None], sh2[k],
                                          m[None, None], r[None, None], lut=S["lut"], hl_mode=self.hl_mode)
                              for k in range(self.N_imgs)], 0) - S["obs"][:, pi, :]) ** 2).sum()
            gL, = torch.autograd.grad(Lp, sh2); g_light.append(float(gL.norm()))
            aa = a.clone().requires_grad_(True); rr = r.clone().requires_grad_(True); mm = m.clone().requires_grad_(True)
            Lp2 = ((torch.cat([shade_ct_sh(S["V_m"][pi:pi + 1], S["N_m"][pi:pi + 1], aa[None], sh_est[k],
                                           mm[None, None], rr[None, None], lut=S["lut"], hl_mode=self.hl_mode)
                               for k in range(self.N_imgs)], 0) - S["obs"][:, pi, :]) ** 2).sum()
            ga, gr, gm = torch.autograd.grad(Lp2, (aa, rr, mm))
            g_mat.append(float(torch.cat([ga, gr.reshape(1), gm.reshape(1)]).norm()))
        x = np.arange(len(worst)); w = 0.38
        fig, ax = plt.subplots(figsize=(6.2, 3.4))
        ax.bar(x - w / 2, g_mat, w, label="|grad| w.r.t. material", color="#4c78a8")
        ax.bar(x + w / 2, g_light, w, label="|grad| w.r.t. this pixel's lighting", color="#e45756")
        ax.axhline(g_glob, color="k", ls="--", lw=1.2, label="global light force |G|/n_px (~0)")
        ax.set_xticks(x); ax.set_xticklabels([f"#{i}" for i in range(len(worst))])
        ax.set_ylabel("gradient norm"); ax.set_yscale("log")
        ax.set_title("Unmet demand: does the pixel still pull on material / lighting?")
        ax.legend(fontsize=8); plt.tight_layout(); plt.show()
        for i, pi in enumerate(worst):
            tag = ("lighting-limited (wants a shared-light move it can't get)"
                   if g_light[i] > 5 * g_glob and g_light[i] > g_mat[i]
                   else "material-limited / at a bound" if g_mat[i] > g_light[i]
                   else "near-stationary on both")
            print(f"  px #{i}: |grad_mat|={g_mat[i]:.2e}  |grad_light|={g_light[i]:.2e}"
                  f"  (global {g_glob:.2e})  ->  {tag}")

    # ── §9 the two basins (joint fit→GT slice) ────────────────────────────────
    def _ts(self):
        ts = np.round(np.arange(-0.3, 1.3 + 1e-9, 0.05), 4)
        i0 = int(np.argmin(np.abs(ts))); i1 = int(np.argmin(np.abs(ts - 1.0)))
        assert abs(ts[i0]) < 1e-9 and abs(ts[i1] - 1.0) < 1e-9, "grid must hit fit(0) and GT(1)"
        return ts, i0, i1

    def _interp_mat(self, pi, t):
        af, rf, mf = self.mat_fit(pi); ag, rg, mg = self.mat_gt(pi)
        ft = self.sh_est.dtype
        a = ((1.0 - t) * af + t * ag).clamp(0.0, 1.0)
        r = torch.as_tensor((1.0 - t) * float(rf) + t * float(rg), dtype=ft, device=self.device)
        m = torch.as_tensor((1.0 - t) * float(mf) + t * float(mg), dtype=ft, device=self.device)
        return a, r, m

    def diag_fit_to_gt(self, est_x, sh_x, pi, ts):
        """Loss along the joint est_x→GT line (est_x = (albedo,roughness,metallic) getters)."""
        af, rf, mf = est_x["albedo"][pi], est_x["roughness"][pi], est_x["metallic"][pi]
        ag, rg, mg = self.mat_gt(pi); ft = self.sh_est.dtype; sh_gt_t = self.S["sh_t"]
        out = np.empty(len(ts), np.float32)
        with torch.no_grad():
            for k, t in enumerate(ts):
                t = float(t)
                a = ((1.0 - t) * af + t * ag).clamp(0.0, 1.0)
                r = torch.as_tensor((1.0 - t) * float(rf) + t * float(rg), dtype=ft, device=self.device)
                m = torch.as_tensor((1.0 - t) * float(mf) + t * float(mg), dtype=ft, device=self.device)
                out[k] = self.loss_at(pi, (a, r, m), (1.0 - t) * sh_x + t * sh_gt_t)
        return out

    def surface_fit_to_gt(self, pi, ts):
        sh_gt_t, sh_est = self.S["sh_t"], self.sh_est
        Z = np.empty((len(ts), len(ts)), np.float32)
        with torch.no_grad():
            for iy, b in enumerate(ts):
                shb = (1.0 - float(b)) * sh_est + float(b) * sh_gt_t
                for ix, t in enumerate(ts):
                    Z[iy, ix] = self.loss_at(pi, self._interp_mat(pi, float(t)), shb)
        return Z

    def show_basins(self):
        worst = self.worst
        ts, i0, i1 = self._ts()
        Z0 = self.surface_fit_to_gt(worst[0], ts)
        diags = [np.diag(Z0)] + [self.diag_fit_to_gt(self.est, self.sh_est, pi, ts) for pi in worst[1:]]
        fig, (axS, axC) = plt.subplots(1, 2, figsize=(13, 5.0))
        c = axS.contourf(ts, ts, np.log10(Z0 + 1e-16), levels=30, cmap="viridis")
        axS.contour(ts, ts, np.log10(Z0 + 1e-16), levels=12, colors="w", linewidths=0.4, alpha=0.5)
        axS.plot(0, 0, "o", ms=13, mfc="cyan", mec="k", label="fit")
        axS.plot(1, 1, "*", ms=20, mfc="red", mec="k", label="GT")
        axS.plot([ts[0], ts[-1]], [ts[0], ts[-1]], "r--", lw=1, alpha=0.7, label="fit->GT diagonal")
        axS.set_xlabel("material   fit(0) -> GT(1)"); axS.set_ylabel("lighting   fit(0) -> GT(1)")
        axS.set_title(f"px #0 (idx {worst[0]}): joint loss surface (log10)\nplot edges = the 1D sweeps of section 5")
        axS.legend(loc="lower right", fontsize=8); fig.colorbar(c, ax=axS, shrink=0.85)
        for i, d in enumerate(diags):
            axC.semilogy(ts, d, lw=2.0, label=f"diagonal px #{i}")
        axC.semilogy(ts, Z0[i0, :], "--", color="steelblue", lw=1.0, label="material axis, #0 (light=fit)")
        axC.semilogy(ts, Z0[:, i0], "--", color="green", lw=1.0, label="lighting axis, #0 (mat=fit)")
        axC.axvline(0, ls=":", c="cyan"); axC.axvline(1, ls=":", c="red")
        axC.set_xlabel("interpolation t   (0=fit, 1=GT)"); axC.set_ylabel("pixel loss")
        axC.set_title("cross-sections: axis slices stay on the plateau; only the diagonal finds GT")
        axC.legend(fontsize=7); axC.grid(alpha=0.3)
        plt.tight_layout(); plt.show()
        for i, d in enumerate(diags):
            barrier = float(d[i0:i1 + 1].max()); fit0 = float(d[i0]); floor = float(d[i1])
            verdict = ("separate basins (barrier on the joint fit->GT path)"
                       if barrier > 1.15 * max(fit0, floor) else "connected downhill valley to GT")
            print(f"  px #{i}: fit={fit0:.2e}  GT/model-floor={floor:.2e}  "
                  f"diagonal barrier={barrier:.2e} ({barrier / max(fit0, floor):.1f}x)  ->  {verdict}")

    # ── §10 does a second optimiser (VarPro) land in the GT funnel? ────────────
    def show_varpro(self, curriculum=None, **overrides):
        """Refit with VarPro (fair init) and overlay each optimiser's fit→GT diagonal.

        curriculum: optional list of GD phase dicts run BEFORE VarPro to warm-start it,
          e.g. ``[{"optimizer":"Adam","n_iter":20000,"lr":0.05}]``. Each phase inherits
          ``self.cfg``, is overridden by the phase dict, and threads its result into the next
          via ``init_maps`` — exactly like decompose_scene's curriculum. (``optimize()`` on its
          own runs a SINGLE optimiser and ignores a ``curriculum`` key, which is why passing it
          as a plain override did nothing.)
        overrides: keys for the FINAL VarPro phase (n_iter, varpro_lam_init,
          varpro_n_inner_rho, varpro_lam_ceiling, ...)."""
        S = self.S
        Nhw, frag, mhw, cam = S["geom"]; fm = S["fm"]
        gt_sh = [c.cpu().numpy() for c in S["sh_t"]]; gt_alb = S["maps"]["albedo"]

        def _run(cfg_phase, init_maps=None):
            cfg_phase = dict(cfg_phase); cfg_phase.pop("curriculum", None)
            return optimize("ct_sh", S["images"], Nhw, frag, mhw, cam,
                            S["maps"]["metallic"], S["maps"]["roughness"], cfg_phase,
                            gt_sh_coeffs=gt_sh, gt_albedo=gt_alb, init_maps=init_maps)

        init_maps = None
        for _pi, ph in enumerate(curriculum or []):
            print(f"  [curriculum {_pi + 1}/{len(curriculum)}] "
                  f"{ph.get('optimizer', 'LBFGS')} n_iter={ph.get('n_iter')} "
                  f"lr={ph.get('lr', '-')}", flush=True)
            r = _run({**self.cfg, **ph}, init_maps)
            init_maps = {"albedo": r.albedo, "sh": r.light, "metallic": r.mat_a, "roughness": r.mat_b}

        cfg_vp = {**self.cfg, "optimizer": "VARPRO", "n_iter": 200, "varpro_space": "natural",
                  "double": False, "init_roughness": 0.5, "init_metallic": 0.05}
        cfg_vp.pop("init_roughness_zero", None)
        cfg_vp.update(overrides)
        res_vp = _run(cfg_vp, init_maps)
        vp_tag = "GD->VarPro" if curriculum else "VarPro"
        est_vp = {"albedo": torch.from_numpy(res_vp.albedo).to(self.device).reshape(-1, 3)[fm],
                  "roughness": torch.from_numpy(res_vp.mat_b).to(self.device).reshape(-1, 1)[fm, 0],
                  "metallic": torch.from_numpy(res_vp.mat_a).to(self.device).reshape(-1, 1)[fm, 0]}
        sh_vp = torch.as_tensor(np.asarray(res_vp.light), dtype=torch.float32, device=self.device)
        ts, i0, i1 = self._ts()
        runs = [("fit", self.est, self.sh_est, "#4c78a8"), (vp_tag, est_vp, sh_vp, "#e45756")]
        worst = self.worst
        fig, ax = plt.subplots(1, self.n_worst, figsize=(3.2 * self.n_worst, 3.4), squeeze=False)
        for i, pi in enumerate(worst):
            for tag, ex, sx, col in runs:
                d = self.diag_fit_to_gt(ex, sx, pi, ts)
                a = ax[0][i]
                a.semilogy(ts, d, color=col, lw=2.0, label=f"{tag} fit->GT")
                a.plot(0, d[i0], "o", color=col, ms=7, mec="k")
            a.axvline(1, ls=":", c="green"); a.axvline(0, ls=":", c="0.6")
            a.set_title(f"px #{i}", fontsize=9); a.set_xlabel("t  (0=fit, 1=GT)")
            if i == 0:
                a.set_ylabel("pixel loss"); a.legend(fontsize=7)
        fig.suptitle("Fit->GT diagonal per optimiser: does VarPro start inside the GT funnel?", fontsize=11)
        plt.tight_layout(); plt.show()
        arm = lambda ex: float(((ex["albedo"] - S["gt_px"]["albedo"]) ** 2).mean().sqrt())
        print(f"global albedo RMSE:  LBFGS={arm(self.est):.4f}   VarPro={arm(est_vp):.4f}")
        print(f"  {'':5} {'px':>4} {'loss@fit':>11} {'barrier':>11} {'x_barrier':>10} {'|albedo-GT|':>12}")
        for tag, ex, sx, _ in runs:
            for i, pi in enumerate(worst):
                d = self.diag_fit_to_gt(ex, sx, pi, ts); b = float(d[i0:i1 + 1].max()); f0 = float(d[i0])
                da = float((ex["albedo"][pi] - S["gt_px"]["albedo"][pi]).norm())
                print(f"  {tag:5} {i:>4} {f0:11.3e} {b:11.3e} {b / max(f0, float(d[i1])):9.1f}x {da:12.3f}")
        self.est_vp, self.sh_vp = est_vp, sh_vp    # kept on self for further poking; no repr

    # ── convergence check: natural vs optimizer-space gradient ────────────────
    @staticmethod
    def _dphys_dx(p, tr):
        """dphysical/dx of the forward transform at the fitted physical value p — the
        factor that turns a natural-space gradient into the one the optimizer descends."""
        if tr in ("sigmoid", "sigmoid_r"): return p * (1.0 - p)          # p=σ(x)
        if tr == "sigmoid_sq":
            s = p.clamp(1e-6, 1.0).sqrt(); return 2.0 * s * s * (1.0 - s)  # p=σ(x)²
        if tr == "log":                    return p                       # p=exp(x)
        return torch.ones_like(p)                                         # none / identity

    def optimizer_grad(self, est=None, sh=None, space=None, tag="fit"):
        """Total-loss gradient at a solution, in BOTH the natural (physical) space and the
        optimizer's transformed space (∂loss/∂x, physical=fwd(x)) — the transformed one is
        what the optimizer actually descends. Reads out, per worst pixel:

          * |g_opt| ~ 0  while |g_nat| > 0  → **converged** for the optimizer; the fit is a
            reparameterisation stationary point (sigmoid-saturated at a box bound). More
            iterations / a GD tail will NOT move it.
          * |g_opt| still large           → **stalled**; the optimizer could still descend
            (a GD tail or better settings would help).
          * both ~ 0                       → genuine interior local optimum.

        `space="natural"` forces identity transforms — pass it with `est=D.est_vp, sh=D.sh_vp`
        to check the VarPro solution (VarPro optimises material in the natural box)."""
        S = self.S; fm = S["fm"]
        est = self.est if est is None else est
        sh = self.sh_est if sh is None else sh
        trA = "none" if space == "natural" else self.cfg.get("tr_albedo", "none")
        trR = "none" if space == "natural" else self.cfg.get("tr_roughness", "none")
        trM = "none" if space == "natural" else self.cfg.get("tr_metallic", "none")
        A = est["albedo"].clone().requires_grad_(True)
        R = est["roughness"].clone().requires_grad_(True)
        M = est["metallic"].clone().requires_grad_(True)
        shp = sh.clone().requires_grad_(True)
        rec = torch.stack([shade_ct_sh(S["V_m"], S["N_m"], A, shp[k], M[:, None], R[:, None],
                                       lut=S["lut"], hl_mode=self.hl_mode)
                           for k in range(self.N_imgs)], 0)
        gA, gR, gM, gSH = torch.autograd.grad(((rec - S["obs"]) ** 2).sum(), (A, R, M, shp))
        jA, jR, jM = self._dphys_dx(A.detach(), trA), self._dphys_dx(R.detach(), trR), self._dphys_dx(M.detach(), trM)
        rms = lambda t: float((t ** 2).mean().sqrt())
        g_light = float(gSH.norm()) / int(fm.sum())
        print(f"[{tag}] transforms: albedo={trA} roughness={trR} metallic={trM}")
        print(f"  GLOBAL material grad RMS  natural={rms(torch.cat([gA, gR[:,None], gM[:,None]],1)):.2e}"
              f"  optimizer(x)={rms(torch.cat([gA*jA, (gR*jR)[:,None], (gM*jM)[:,None]],1)):.2e}"
              f"   | lighting grad/px={g_light:.2e}")
        print(f"  {'#':>2} {'|g_nat|':>10} {'|g_opt|':>10} {'ratio':>7}   verdict")
        gn, go = [], []
        for i, pi in enumerate(self.worst):
            nat = torch.cat([gA[pi], gR[pi:pi+1], gM[pi:pi+1]])
            opt = torch.cat([(gA*jA)[pi], (gR*jR)[pi:pi+1], (gM*jM)[pi:pi+1]])
            n, o = float(nat.norm()), float(opt.norm()); gn.append(n); go.append(o)
            verdict = ("stalled - optimizer can still descend" if o > 1e-2
                       else "converged at a box bound (g_opt~0, g_nat>0)" if n > 5 * max(o, 1e-9)
                       else "genuine local optimum (both ~0)")
            print(f"  #{i:>1} {n:10.2e} {o:10.2e} {o/max(n,1e-12):7.2f}   {verdict}")
        x = np.arange(len(self.worst)); w = 0.38
        fig, ax = plt.subplots(figsize=(6.2, 3.4))
        ax.bar(x - w/2, gn, w, label="|grad| natural (∂loss/∂physical)", color="#4c78a8")
        ax.bar(x + w/2, go, w, label="|grad| optimizer (∂loss/∂x)", color="#e45756")
        ax.axhline(1e-2, color="k", ls="--", lw=1, label="stalled/converged guide")
        ax.set_xticks(x); ax.set_xticklabels([f"#{i}" for i in range(len(self.worst))])
        ax.set_yscale("log"); ax.set_ylabel("gradient norm"); ax.legend(fontsize=8)
        ax.set_title(f"[{tag}] material gradient: natural vs what the optimizer descends")
        plt.tight_layout(); plt.show()

    # ── which parameter matters? one-at-a-time est→GT interpolation ───────────
    def show_param_interp(self, step=0.05):
        """Interpolate ONE parameter group est→GT with ALL OTHERS pinned at GT — the
        opposite decomposition of §9's diagonal (which moves everything at once). At t=1
        every curve is the full-GT point (the sharp spike); at t=0 the curve's height is the
        loss contributed by THAT parameter's estimate alone. The highest/steepest curve is the
        parameter whose error dominates. Groups: albedo R/G/B, roughness, metallic, lighting.

        The grid includes t=0 and t=1 EXACTLY (step must divide 1) — otherwise the nearest
        sample sits ~step up the needle-thin GT funnel and every curve reads a different
        (spurious) depth at the spike instead of the shared model floor r(GT,GT)."""
        ts = np.round(np.arange(-0.3, 1.3 + 1e-9, step), 6)
        i0 = int(np.argmin(np.abs(ts))); i1 = int(np.argmin(np.abs(ts - 1.0)))
        assert abs(ts[i0]) < 1e-9 and abs(ts[i1] - 1.0) < 1e-9, "grid must hit t=0 and t=1"
        n = len(ts)
        S = self.S; sh_gt = S["sh_t"]; ft = self.sh_est.dtype
        groups = ["albedo R", "albedo G", "albedo B", "roughness", "metallic", "lighting"]
        colors = plt.cm.tab10(np.arange(len(groups)))
        fig, ax = plt.subplots(1, self.n_worst, figsize=(3.6 * self.n_worst, 3.8), squeeze=False)
        print("loss with ONLY this parameter at its estimate, all others at GT (t=0):")
        print(f"  {'#':>2} " + "".join(f"{g:>11}" for g in groups) + "   dominant")
        for i, pi in enumerate(self.worst):
            af, rf, mf = self.mat_fit(pi); ag, rg, mg = self.mat_gt(pi)
            at0 = []
            for gi, g in enumerate(groups):
                ys = np.empty(n, np.float32)
                with torch.no_grad():
                    for j, t in enumerate(ts):
                        a = ag.clone(); r = float(rg); m = float(mg); sh = sh_gt
                        if g[:6] == "albedo":
                            c = "RGB".index(g[-1]); a[c] = (1 - t) * af[c] + t * ag[c]
                        elif g == "roughness": r = (1 - t) * float(rf) + t * float(rg)
                        elif g == "metallic":  m = (1 - t) * float(mf) + t * float(mg)
                        else:                  sh = (1 - t) * self.sh_est + t * sh_gt
                        rr = torch.as_tensor(r, dtype=ft, device=self.device)
                        mm = torch.as_tensor(m, dtype=ft, device=self.device)
                        ys[j] = self.loss_at(pi, (a, rr, mm), sh)
                ax[0][i].semilogy(ts, ys, color=colors[gi], lw=1.8, label=g)
                at0.append(float(ys[i0]))
            ax[0][i].axvline(0, ls=":", c="0.6"); ax[0][i].axvline(1, ls=":", c="green")
            ax[0][i].set_title(f"px #{i}", fontsize=9); ax[0][i].set_xlabel("t  (0=est, 1=GT)")
            if i == 0:
                ax[0][i].set_ylabel("pixel loss"); ax[0][i].legend(fontsize=6, loc="upper left")
            print(f"  #{i:>1} " + "".join(f"{v:>11.2e}" for v in at0)
                  + f"   {groups[int(np.argmax(at0))]}")
        fig.suptitle("Per-parameter est→GT interpolation (others at GT): which error dominates?",
                     fontsize=11)
        plt.tight_layout(); plt.show()

    # ── sensitivity AT GT: sweep each parameter around the true optimum ───────
    def show_gt_sensitivity(self, n=61):
        """Sweep each parameter around its GT value with ALL OTHERS pinned at GT, so every
        curve bottoms at the model floor r(GT,GT) and its **width/curvature is the parameter's
        identifiability at the true solution** — a sharp well is well-constrained, a flat trough
        is not. Columns: albedo R/G/B, roughness, metallic, an overall lighting **scale** (sh·s),
        and the coupled **albedo↔lighting scale** (albedo·s, sh/s) — the classic gauge direction,
        which should be far flatter than any single-parameter sweep."""
        S = self.S; sh_gt = S["sh_t"]; ft = self.sh_est.dtype
        specs = [("albedo R", ("a", 0), (0.0, 1.0)), ("albedo G", ("a", 1), (0.0, 1.0)),
                 ("albedo B", ("a", 2), (0.0, 1.0)), ("roughness", ("r", None), (0.03, 1.0)),
                 ("metallic", ("m", None), (0.0, 1.0)), ("light x s", ("Ls", None), (0.5, 1.5)),
                 ("alb*s, light/s", ("cpl", None), (0.5, 1.5))]
        fig, ax = plt.subplots(self.n_worst, len(specs),
                               figsize=(2.6 * len(specs), 2.3 * self.n_worst), squeeze=False)
        for i, pi in enumerate(self.worst):
            ag, rg, mg = self.mat_gt(pi)
            for j, (name, (kind, c), (lo, hi)) in enumerate(specs):
                grid = np.linspace(lo, hi, n); ys = np.empty(n, np.float32)
                with torch.no_grad():
                    for k, v in enumerate(grid):
                        a = ag.clone(); r = float(rg); m = float(mg); sh = sh_gt
                        if kind == "a":   a[c] = v
                        elif kind == "r": r = v
                        elif kind == "m": m = v
                        elif kind == "Ls": sh = sh_gt * float(v)
                        else:             a = (ag * float(v)).clamp(0.0, 1.0); sh = sh_gt / float(v)
                        rr = torch.as_tensor(r, dtype=ft, device=self.device)
                        mm = torch.as_tensor(m, dtype=ft, device=self.device)
                        ys[k] = self.loss_at(pi, (a, rr, mm), sh)
                gtv = {"a": (float(ag[c]) if c is not None else 0), "r": float(rg),
                       "m": float(mg), "Ls": 1.0, "cpl": 1.0}[kind]
                a_ = ax[i][j]; a_.semilogy(grid, ys, color="steelblue", lw=1.6)
                a_.axvline(gtv, color="green", lw=1.4)
                if i == 0: a_.set_title(name, fontsize=9)
                if j == 0: a_.set_ylabel(f"px #{i}\nloss", fontsize=8)
                a_.tick_params(labelsize=7)
        fig.suptitle("Loss around GT (others at GT): sharp well = well-identified, flat = not",
                     fontsize=11)
        plt.tight_layout(); plt.show()

    # ── strategies to escape local optima: run and compare ────────────────────
    def compare_strategies(self, strategies):
        """Run one full ct_sh fit per strategy (each dict overrides self.cfg) and tabulate
        intrinsics error vs GT. A ``curriculum`` key warm-starts (GD→…), threaded like
        show_varpro. Reports global albedo RMSE / roughness MAE / metallic MAE and a bar
        chart. EXPENSIVE — one fit per strategy; keep n_iter modest for a screen, scale the
        winner up. Judge by intrinsics error, not recon loss (they decouple here)."""
        S = self.S; Nhw, frag, mhw, cam = S["geom"]; fm = S["fm"]
        gt_sh = [c.cpu().numpy() for c in S["sh_t"]]; gt_alb = S["maps"]["albedo"]; gp = S["gt_px"]

        _PRIOR_KEYS = ("lambda_light_prior", "lambda_light_mono", "lambda_box")

        def _fit(ov):
            ov = dict(ov); curr = ov.pop("curriculum", None); im = None
            # VarPro solves lighting in closed form and IGNORES these penalty priors, so a prior
            # under a VarPro (final) phase is a silent no-op — warn rather than mislead.
            _final_opt = ov.get("optimizer", self.cfg.get("optimizer", "LBFGS"))
            if _final_opt == "VARPRO" and any(ov.get(k) for k in _PRIOR_KEYS):
                print("    [warn] VarPro ignores lambda_light_prior/mono/box — run priors under LBFGS/Adam")
            for ph in (curr or []):
                r = optimize("ct_sh", S["images"], Nhw, frag, mhw, cam, S["maps"]["metallic"],
                             S["maps"]["roughness"], {**self.cfg, **ph},
                             gt_sh_coeffs=gt_sh, gt_albedo=gt_alb, init_maps=im)
                im = {"albedo": r.albedo, "sh": r.light, "metallic": r.mat_a, "roughness": r.mat_b}
            return optimize("ct_sh", S["images"], Nhw, frag, mhw, cam, S["maps"]["metallic"],
                            S["maps"]["roughness"], {**self.cfg, **ov},
                            gt_sh_coeffs=gt_sh, gt_albedo=gt_alb, init_maps=im)

        rows = []
        for name, ov in strategies.items():
            print(f"  running [{name}] ...", flush=True)
            r = _fit(ov)
            A = torch.from_numpy(r.albedo).to(self.device).reshape(-1, 3)[fm]
            R = torch.from_numpy(r.mat_b).to(self.device).reshape(-1, 1)[fm, 0]
            M = torch.from_numpy(r.mat_a).to(self.device).reshape(-1, 1)[fm, 0]
            arm = float(((A - gp["albedo"]) ** 2).mean().sqrt())
            rmae = float((R - gp["roughness"]).abs().mean()); mmae = float((M - gp["metallic"]).abs().mean())
            rows.append((name, arm, rmae, mmae))
        print(f"\n  {'strategy':22} {'albedo_rmse':>12} {'rough_mae':>10} {'metal_mae':>10}")
        for name, arm, rmae, mmae in sorted(rows, key=lambda x: x[1]):
            print(f"  {name:22} {arm:12.4f} {rmae:10.4f} {mmae:10.4f}")
        fig, ax = plt.subplots(figsize=(max(6, 1.3 * len(rows)), 3.6))
        ax.bar(range(len(rows)), [x[1] for x in rows], color="steelblue")
        ax.set_xticks(range(len(rows))); ax.set_xticklabels([x[0] for x in rows], rotation=30, ha="right", fontsize=8)
        ax.set_ylabel("global albedo RMSE"); ax.set_title("Strategies vs intrinsics error (lower = better)")
        plt.tight_layout(); plt.show()
        return rows

    # ── oracle: fix the lighting, optimise material only ──────────────────────
    def fit_material_given_lighting(self, ref="gt", **cfg_over):
        """Hold the lighting FIXED at a reference and optimise material only — the oracle for
        'if the lighting were known, is the material recoverable?'. ``ref="gt"`` uses S['sh_t']
        (the reference bank the data was rendered under); or pass a (K, n_sh, 3) array. Prints
        albedo RMSE vs the joint fit; a large drop means lighting estimation was the whole
        problem (material has no separate degeneracy)."""
        S = self.S; Nhw, frag, mhw, cam = S["geom"]; fm = S["fm"]; gp = S["gt_px"]
        if isinstance(ref, str) and ref == "gt":
            ref_arr = np.stack([c.cpu().numpy() for c in S["sh_t"]]); ref_t = S["sh_t"]
        else:
            ref_arr = np.asarray(ref, np.float32)
            ref_t = torch.from_numpy(ref_arr).to(self.device, torch.float32)
        mat_cfg = {**self.cfg, **cfg_over}
        if mat_cfg.get("optimizer", "LBFGS") == "VARPRO":
            # VarPro re-SOLVES the lighting internally, so it can't be held fixed — the oracle
            # needs a gradient optimiser over material with the SH frozen.
            mat_cfg["optimizer"] = "LBFGS"
            print("  (optimizer forced LBFGS: VarPro would re-solve the lighting)")
        r = optimize("ct_sh", S["images"], Nhw, frag, mhw, cam,
                     S["maps"]["metallic"], S["maps"]["roughness"], mat_cfg,
                     gt_sh_coeffs=[c.cpu().numpy() for c in S["sh_t"]], gt_albedo=S["maps"]["albedo"],
                     init_maps={"sh": ref_arr}, opt_params=frozenset({"albedo", "metallic", "roughness"}))
        A = torch.from_numpy(r.albedo).to(self.device).reshape(-1, 3)[fm]
        R = torch.from_numpy(r.mat_b).to(self.device).reshape(-1, 1)[fm, 0]
        M = torch.from_numpy(r.mat_a).to(self.device).reshape(-1, 1)[fm, 0]
        arm = float(((A - gp["albedo"]) ** 2).mean().sqrt())

        def _recon_rmse(a, r_, m_):
            with torch.no_grad():
                rec = torch.stack([shade_ct_sh(S["V_m"], S["N_m"], a, ref_t[k], m_[:, None], r_[:, None],
                                               lut=S["lut"], hl_mode=self.hl_mode)
                                   for k in range(self.N_imgs)], 0)
                return float(((rec - S["obs"]) ** 2).mean().sqrt())
        oracle_recon = _recon_rmse(A, R, M)                                    # does oracle material fit obs?
        floor_recon = _recon_rmse(gp["albedo"], gp["roughness"], gp["metallic"])  # does GT material fit obs under ref?

        print(f"albedo RMSE (lighting fixed at ref): {arm:.4f}")
        print(f"  recon RMSE:  oracle material={oracle_recon:.3e}   GT material under ref={floor_recon:.3e}")
        if floor_recon > 1e-2:
            v = "the REFERENCE lighting is wrong (GT material doesn't fit obs under it) — not the true lighting / scale mismatch"
        elif arm < 0.05:
            v = "material fully recovers -> lighting estimation was the whole problem"
        elif oracle_recon < 5 * max(floor_recon, 1e-9):
            v = "material AMBIGUOUS: a different material fits obs equally well under the known lighting (per-pixel gauge)"
        else:
            v = "material optimizer STALLED (recon still high) -> try a stronger material config (natural+box / more iters)"
        print("  ->", v)
        return r

    # ── staged optimisation: estimate lighting, freeze it, fit material ───────
    def fit_lighting_then_material(self, lighting_cfg=None, material_cfg=None):
        """Two-stage (nvdiffrec-style staged optimisation, and your idea): (1) estimate the
        lighting with a strong joint optimiser, (2) FREEZE the ESTIMATED lighting and do a
        material-only LBFGS fit. This is the realistic version of the §15 oracle — it uses the
        *estimated* lighting, not GT, so it measures how close a good lighting estimate gets the
        material. `lighting_cfg` defaults to GD→VarPro; a 'curriculum' key is threaded."""
        S = self.S; Nhw, frag, mhw, cam = S["geom"]
        gt_sh = [c.cpu().numpy() for c in S["sh_t"]]; gt_alb = S["maps"]["albedo"]
        lcfg = dict(lighting_cfg or {"optimizer": "VARPRO", "varpro_space": "natural",
                                     "n_iter": 100, "varpro_lam_init": 1e-4, "varpro_n_inner_rho": 10,
                                     "curriculum": [{"optimizer": "Adam", "n_iter": 20000, "lr": 0.05}]})
        curr = lcfg.pop("curriculum", None); im = None
        for ph in (curr or []):
            print(f"  [stage 1 curriculum] {ph.get('optimizer')} n_iter={ph.get('n_iter')}", flush=True)
            r = optimize("ct_sh", S["images"], Nhw, frag, mhw, cam, S["maps"]["metallic"],
                         S["maps"]["roughness"], {**self.cfg, **ph}, gt_sh_coeffs=gt_sh,
                         gt_albedo=gt_alb, init_maps=im)
            im = {"albedo": r.albedo, "sh": r.light, "metallic": r.mat_a, "roughness": r.mat_b}
        r1 = optimize("ct_sh", S["images"], Nhw, frag, mhw, cam, S["maps"]["metallic"],
                      S["maps"]["roughness"], {**self.cfg, **lcfg}, gt_sh_coeffs=gt_sh,
                      gt_albedo=gt_alb, init_maps=im)
        est_sh = np.asarray(r1.light)
        lrm = float(((torch.from_numpy(est_sh).to(self.device, torch.float32) - S["sh_t"]) ** 2).mean().sqrt())
        print(f"stage 1: estimated-lighting RMSE vs GT = {lrm:.4f}")
        print("stage 2: freeze estimated lighting, fit material only ->")
        return self.fit_material_given_lighting(
            ref=est_sh, **(material_cfg or dict(optimizer="LBFGS", n_iter=500, lbfgs_max_iter=40)))

    # ── coarse-to-fine over colour: grayscale first, then split colour ────────
    def fit_grayscale_then_color(self, gray_cfg=None, color_cfg=None, freeze_light=False,
                                 mono_gray=1e-1, mono_color=1e-3):
        """Two-stage coarse-to-fine over COLOUR:
          (1) fit on LUMINANCE observations with a strong monochrome-light prior — a low-DOF
              grayscale decomposition that estimates the lighting *shape* in a better-conditioned
              landscape;
          (2) recover colour, warm-started from stage 1. ``freeze_light=False`` (default) re-opens
              the full RGB lighting from the grayscale basin with a mild monochrome prior
              (`mono_color`) so its shape doesn't re-drift; ``freeze_light=True`` holds the
              grayscale lighting fixed and only fits per-channel material (the white-light split,
              ~ the monochrome ceiling). Reports albedo / lighting RMSE."""
        S = self.S; Nhw, frag, mhw, cam = S["geom"]; fm = S["fm"]
        gt_sh = [c.cpu().numpy() for c in S["sh_t"]]; gt_alb = S["maps"]["albedo"]; gp = S["gt_px"]
        LUM = np.array([0.2126, 0.7152, 0.0722], np.float32)
        imgs_gray = [np.repeat((im * LUM).sum(-1, keepdims=True), 3, axis=-1).astype(np.float32)
                     for im in S["images"]]
        g1 = {**self.cfg, "optimizer": "LBFGS", "n_iter": 300, "lbfgs_max_iter": 40,
              "lambda_light_mono": mono_gray}
        g1.update(gray_cfg or {})
        r1 = optimize("ct_sh", imgs_gray, Nhw, frag, mhw, cam, S["maps"]["metallic"],
                      S["maps"]["roughness"], g1, gt_sh_coeffs=gt_sh, gt_albedo=gt_alb)
        lrm1 = float(((torch.from_numpy(np.asarray(r1.light)).to(self.device, torch.float32)
                       - S["sh_t"]) ** 2).mean().sqrt())
        init = {"albedo": r1.albedo, "sh": r1.light, "metallic": r1.mat_a, "roughness": r1.mat_b}
        print(f"  stage 1 (grayscale): lighting-shape RMSE vs GT = {lrm1:.4f}")

        g2 = {**self.cfg, "optimizer": "LBFGS", "n_iter": 300, "lbfgs_max_iter": 40,
              "lambda_light_mono": mono_color}
        g2.update(color_cfg or {})
        kw = dict(gt_sh_coeffs=gt_sh, gt_albedo=gt_alb, init_maps=init)
        if freeze_light:
            kw["opt_params"] = frozenset({"albedo", "metallic", "roughness"})
        r2 = optimize("ct_sh", S["images"], Nhw, frag, mhw, cam, S["maps"]["metallic"],
                      S["maps"]["roughness"], g2, **kw)
        A = torch.from_numpy(r2.albedo).to(self.device).reshape(-1, 3)[fm]
        arm = float(((A - gp["albedo"]) ** 2).mean().sqrt())
        lrm = float(((torch.from_numpy(np.asarray(r2.light)).to(self.device, torch.float32)
                      - S["sh_t"]) ** 2).mean().sqrt())
        print(f"  stage 2 (colour, freeze_light={freeze_light}): albedo RMSE={arm:.4f}  "
              f"lighting RMSE={lrm:.4f}")
        return r2

    # ── does more images cure the lighting-dominated error? ───────────────────
    def compare_images(self, image_counts, cfg=None):
        """Reload the scene at several image counts and refit, testing whether more
        observations fix the lighting-dominated error (§12). Lighting is SHARED across
        images, so more lightings over-determine it (and the per-pixel material must explain
        many lightings — photometric-stereo). Reports albedo RMSE and lighting RMSE
        (||sh_est − sh_GT||) vs N. One full fit per N — expensive."""
        cfg = cfg or self.cfg
        sh_order = int(cfg.get("sh_order", 2))
        rows = []
        for N in image_counts:
            S = load_bundle(self._scene, N, self._downsample, self.device, sh_order)
            d = PixelDiag(S, cfg, self.device, self.n_worst, self.sweep_n)
            d._scene, d._downsample = self._scene, self._downsample
            d.fit()
            arm = float(((d.est["albedo"] - S["gt_px"]["albedo"]) ** 2).mean().sqrt())
            lrm = float(((d.sh_est - S["sh_t"]) ** 2).mean().sqrt())
            rows.append((N, arm, lrm))
            print(f"  N={N:>4}: albedo_rmse={arm:.4f}   lighting_rmse={lrm:.4f}")
        Ns = [r[0] for r in rows]
        fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
        ax[0].plot(Ns, [r[1] for r in rows], "o-", color="#4c78a8")
        ax[0].set_xlabel("# images"); ax[0].set_ylabel("albedo RMSE"); ax[0].set_title("material error vs # images")
        ax[1].plot(Ns, [r[2] for r in rows], "o-", color="#e45756")
        ax[1].set_xlabel("# images"); ax[1].set_ylabel("lighting RMSE  (sh_est vs GT)")
        ax[1].set_title("lighting error vs # images")
        for a in ax: a.grid(alpha=0.3)
        plt.tight_layout(); plt.show()
        return rows


# ──────────────────────────────────────────────────────────────────────────────
# §11 GT-render floor (dataset-level, no fit needed)
# ──────────────────────────────────────────────────────────────────────────────
def gt_render_floor(scene_path, sh_order, downsample=8, n_images=24, device="cuda",
                    hl_mode=None):
    """r(GT,GT): render GT maps under GT SH, compare to stored obs. -> per-pixel RMSE.

    `hl_mode` defaults to the dataset's own render mode (config.json, "lut" for legacy
    datasets) so the floor bottoms out at ~0 rather than picking up an analytic-vs-LUT gap."""
    import json
    from idr.data.scene_io import load_scene
    if hl_mode is None:
        _cfgp = Path(scene_path) / "config.json"
        hl_mode = json.loads(_cfgp.read_text()).get("hl_mode", "lut") if _cfgp.exists() else "lut"
    s = load_scene(Path(scene_path), gt_npy=True)
    st = lambda a: np.ascontiguousarray(a[::downsample, ::downsample])
    mask = st(s["mask_np"])
    albedo, rough, metal = st(s["albedo_np"]), st(s["roughness_np"]), st(s["metallic_np"])
    imgs = [st(im) for im in s["images"][:n_images]]
    sh = np.stack([np.asarray(c, np.float32) for c in s["sh_coeffs"][:n_images]])
    Nhw, frag, mhw, cam = make_proxy_geometry(s["normals_np"], s["mask_np"], 60., 2.,
                                              device, torch.float32, stride=downsample)
    fm = mhw.reshape(-1); N_m = Nhw.reshape(-1, 3)[fm]
    V_m = torch.nn.functional.normalize(cam[None] - frag.reshape(-1, 3)[fm], dim=-1)
    lut = _get_ggx_sh_lut(device, n_bands=int(sh_order) + 1).to(torch.float32)
    obs = torch.stack([torch.from_numpy(im).to(device, torch.float32)
                       for im in imgs]).reshape(len(imgs), -1, 3)[:, fm, :]
    A = torch.from_numpy(albedo.reshape(-1, 3)).to(device)[fm.cpu()]
    R = torch.from_numpy(rough.reshape(-1, 1)).to(device)[fm.cpu(), 0]
    Mt = torch.from_numpy(metal.reshape(-1, 1)).to(device)[fm.cpu(), 0]
    sht = torch.from_numpy(sh).to(device, torch.float32)
    with torch.no_grad():
        rec = torch.stack([shade_ct_sh(V_m, N_m, A, sht[k], Mt[:, None], R[:, None],
                                       lut=lut, hl_mode=hl_mode)
                           for k in range(len(imgs))], 0)
        err = ((rec - obs) ** 2).mean(dim=(0, 2)).sqrt()
    H, W = mask.shape
    emap = np.full(H * W, np.nan, np.float32); emap[fm.cpu().numpy()] = err.cpu().numpy()
    return dict(err=err.cpu().numpy(), emap=emap.reshape(H, W), M=int(fm.sum()))


def compare_floor(runs, k=30):
    """runs: list of (label, floor_dict from gt_render_floor). Plots maps + worst-pixel bars."""
    fig, ax = plt.subplots(1, len(runs) + 1, figsize=(4.7 * (len(runs) + 1) / 2 + 4, 4.2))
    vmax = max(float(np.nanmax(r["emap"])) for _, r in runs)
    for i, (tag, r) in enumerate(runs):
        im = ax[i].imshow(r["emap"], cmap="inferno", vmin=0, vmax=vmax)
        ax[i].set_title(f"{tag}\nr(GT,GT) RMSE"); ax[i].axis("off")
        plt.colorbar(im, ax=ax[i], fraction=0.046)
    for tag, r in runs:
        ax[-1].plot(np.sort(r["err"])[::-1][:k], lw=2, label=tag)
    ax[-1].set_yscale("log"); ax[-1].set_xlabel(f"worst {k} pixels (ranked)")
    ax[-1].set_ylabel("recon RMSE"); ax[-1].set_title("worst-pixel GT-render floor")
    ax[-1].legend(fontsize=8); ax[-1].grid(alpha=0.3)
    plt.tight_layout(); plt.show()
    for tag, r in runs:
        e = np.sort(r["err"])[::-1]
        print(f"{tag:22} px={r['M']:5d}  median={np.median(r['err']):.2e}  "
              f"p99={np.quantile(r['err'], 0.99):.2e}  max={e[0]:.2e}  worst4={e[:4].round(6)}")
