"""
relight_sweep.py — relight recovered intrinsics under a MOVING DIRECTIONAL light.

The held-out `relight_rmse` that decompose_scene reports re-lights the estimate under
the val set's GT *environment* SH — same lighting family the optimizer was fit on. This
module is the harder test: a single directional light swept across azimuth, which is a
lighting family the decomposition never saw.

Geometry (normals + mask) always comes from the DATASET, for both the GT and the
estimated intrinsics: the decomposition never estimates normals, so the two are relit
through identical geometry and differ only in albedo/metallic/roughness. That is what
makes the error map meaningful.

Conventions — make_proxy_geometry puts the camera at (0, 0, +cam_dist) looking down -Z
with +Y up, so azimuth rotates about Y (up) and az=0 means "lit from the camera":

    d(az, el) = [cos(el)*sin(az), sin(el), cos(el)*cos(az)]

Albedo scale: albedo and lighting trade off by a global scale that is unobservable from
the images, so both variants are reported —
    raw    (albedo_est.npy)    what decompose_scene's own relight_rmse uses
    scaled (albedo_scaled.npy) scale fitted to the GT albedo; what is left is structural
A large raw/scaled ratio means the error is dominated by that unobservable scale rather
than by a genuinely wrong decomposition.

Used by run_decomposition.py (--relight_sweep / --relight_video) and usable directly:

    from idr.eval.relight_sweep import relight_sweep
    relight_sweep(run_dir, ds_dir, downsample=2, video=True)
"""
import math
import os
from pathlib import Path

import numpy as np
import torch

from idr.data.geometry import make_proxy_geometry

from idr.data.scene_io import load_scene

from idr.config import LIGHT_COLOR, LIGHT_INTENSITY
from idr.render import shade_ct_sh, SHLighting
from idr.render.brdf import _get_ggx_sh_lut

__all__ = ["dir_sh", "Relighter", "relight_sweep"]


def dir_sh(az_deg, elev_deg=30.0, color=LIGHT_COLOR, intensity=LIGHT_INTENSITY):
    """SH2 coefficients (9,3) for a directional light at (azimuth, elevation), degrees."""
    az, el = math.radians(az_deg), math.radians(elev_deg)
    d = np.array([math.cos(el) * math.sin(az),
                  math.sin(el),
                  math.cos(el) * math.cos(az)], np.float32)
    d /= np.linalg.norm(d)
    return SHLighting.directional(d, np.asarray(color, np.float32), intensity).coeffs


class Relighter:
    """Fixed geometry + GGX LUT; relights any (albedo, metallic, roughness) under any SH.

    Build once per scene — make_proxy_geometry and the LUT are the expensive parts, and
    they are shared by every frame and every intrinsics variant."""

    def __init__(self, normals_np, mask_np, device="cuda", dtype=torch.float32,
                 diffuse_fresnel=True, hl_mode="analytic"):
        self.H, self.W = normals_np.shape[:2]
        self.mask_np = mask_np
        self.diffuse_fresnel = diffuse_fresnel
        self.hl_mode = hl_mode
        n, f, m, cam = make_proxy_geometry(normals_np, mask_np, 60.0, 2.0, device, dtype)
        self.fm = m.reshape(-1)
        self.N = n.reshape(-1, 3)[self.fm]
        self.V = torch.nn.functional.normalize(cam[None] - f.reshape(-1, 3)[self.fm], dim=-1)
        self.lut = _get_ggx_sh_lut(device, n_bands=3).to(dtype)
        self.device, self.dtype = device, dtype
        self._fm_np = self.fm.detach().cpu().numpy()

    def _m(self, a, c):
        return torch.from_numpy(np.ascontiguousarray(a)).to(self.device, self.dtype) \
                    .reshape(-1, c)[self.fm]

    def render(self, intr, sh):
        """-> (H, W, 3) float32, background 0."""
        with torch.no_grad():
            px = shade_ct_sh(self.V, self.N, self._m(intr["albedo"], 3),
                             torch.from_numpy(np.asarray(sh, np.float32)).to(self.device, self.dtype),
                             self._m(intr["metallic"], 1), self._m(intr["roughness"], 1),
                             lut=self.lut, diffuse_fresnel=self.diffuse_fresnel,
                             hl_mode=self.hl_mode)
        out = np.zeros((self.H * self.W, 3), np.float32)
        out[self._fm_np] = px.float().cpu().numpy()
        return out.reshape(self.H, self.W, 3)


# ───────────────────────────── loading ───────────────────────────────────────
def _stride(a, ds):
    return np.ascontiguousarray(a[::ds, ::ds]) if ds > 1 else a


def _load_gt(ds_dir, ds):
    sc = load_scene(Path(ds_dir), gt_npy=True)
    return dict(albedo=_stride(sc["albedo_np"], ds), metallic=_stride(sc["metallic_np"], ds),
                roughness=_stride(sc["roughness_np"], ds),
                normals=_stride(sc["normals_np"], ds), mask=_stride(sc["mask_np"], ds))


def _load_est(run_dir, scaled):
    run_dir = Path(run_dir)
    alb = run_dir / ("albedo_scaled.npy" if scaled else "albedo_est.npy")
    if not alb.exists():                       # scaled is optional; raw always exists
        alb = run_dir / "albedo_est.npy"
    return dict(albedo=np.load(alb).astype(np.float32),
                metallic=np.load(run_dir / "metallic_est.npy").astype(np.float32),
                roughness=np.load(run_dir / "roughness_est.npy").astype(np.float32))


# ───────────────────────────── video ─────────────────────────────────────────
def _tonemap_fn(frames, pct=99.5):
    """One exposure for the whole sweep, so the video does not flicker."""
    s = float(np.percentile(np.stack(frames), pct))
    s = s if s > 1e-8 else 1.0
    return lambda im: (np.clip(im / s, 0, 1) ** (1 / 2.2) * 255).astype(np.uint8)


def _upscale(img, min_px):
    """Nearest-neighbour upscale so a 64^2 decomposition is still watchable."""
    from PIL import Image
    h, w = img.shape[:2]
    f = max(1, int(np.ceil(min_px / max(h, 1))))
    if f == 1:
        return img
    return np.asarray(Image.fromarray(img).resize((w * f, h * f), Image.NEAREST))


def _write_video(frames_u8, stem, fps=20, ping_pong=True, save_frames=False):
    """mp4 when an ffmpeg backend is available, else animated GIF. Returns the path."""
    from PIL import Image
    seq = list(frames_u8) + list(frames_u8)[-2:0:-1] if ping_pong else list(frames_u8)
    if save_frames:
        fdir = Path(f"{stem}_frames"); fdir.mkdir(parents=True, exist_ok=True)
        for i, f in enumerate(frames_u8):
            Image.fromarray(f).save(fdir / f"frame_{i:03d}.png")
    try:
        import imageio.v2 as iio
        try:
            iio.mimsave(f"{stem}.mp4", seq, fps=fps, macro_block_size=1)
            return f"{stem}.mp4"
        except Exception:
            # imageio opens the container before it discovers ffmpeg is missing, leaving
            # a few-byte stub. Remove it, or a corrupt .mp4 sits next to the good .gif.
            try:
                os.unlink(f"{stem}.mp4")
            except OSError:
                pass
            iio.mimsave(f"{stem}.gif", seq, duration=1 / fps, loop=0)
            return f"{stem}.gif"
    except ImportError:
        ims = [Image.fromarray(f) for f in seq]
        ims[0].save(f"{stem}.gif", save_all=True, append_images=ims[1:],
                    duration=int(1000 / fps), loop=0)
        return f"{stem}.gif"


def _label(img, txt):
    from PIL import Image, ImageDraw
    im = Image.fromarray(img)
    ImageDraw.Draw(im).text((3, 2), txt, fill=(255, 255, 255))
    return np.asarray(im)


# ───────────────────────────── main entry point ──────────────────────────────
def relight_sweep(run_dir, ds_dir, downsample=1, az_from=-45.0, az_to=45.0, n_frames=61,
                  elev=30.0, diffuse_fresnel=True, device=None, dtype=torch.float32,
                  video=True, plots=True, fps=20, ping_pong=True, min_panel_px=192,
                  save_frames=False, n_panels=5, dpi=90, out_name="relight_sweep"):
    """Sweep a directional light across azimuth and compare GT vs estimated intrinsics.

    Writes into <run_dir>/<out_name>/:
      sweep_panels.png     GT / EST / |err| at n_panels azimuths
      sweep_error.png      RMSE vs azimuth, raw and scaled albedo
      compare.(mp4|gif)    GT | EST | |err| side by side   (video=True)
      est.(mp4|gif)        the estimate alone              (video=True)
      sweep.json           the numbers

    Returns a metrics dict (also merged into the caller's summary row).
    """
    run_dir, ds_dir = Path(run_dir), Path(ds_dir)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = run_dir / out_name
    out_dir.mkdir(parents=True, exist_ok=True)

    est_scaled = _load_est(run_dir, scaled=True)
    est_raw = _load_est(run_dir, scaled=False)
    gt = _load_gt(ds_dir, downsample)
    if gt["albedo"].shape[:2] != est_scaled["albedo"].shape[:2]:
        raise ValueError(f"resolution mismatch: GT {gt['albedo'].shape[:2]} vs EST "
                         f"{est_scaled['albedo'].shape[:2]} at downsample={downsample}")

    az = np.linspace(az_from, az_to, n_frames)
    shs = [dir_sh(a, elev) for a in az]
    rel = Relighter(gt["normals"], gt["mask"], device, dtype, diffuse_fresnel)

    mask = gt["mask"]
    # Only the GT and scaled frames are kept: they feed the plots and the video. The raw
    # variant is needed for its error curve alone, so it is streamed one frame at a time
    # rather than materialised — at 61 frames x 512^2 a third frame list is ~190 MB of
    # host RAM per worker, which matters when several workers run side by side.
    gt_f = [rel.render(gt, s) for s in shs]
    est_f = [rel.render(est_scaled, s) for s in shs]

    def _frame_err(g, e):
        d = (g - e)[mask]
        return float(np.sqrt((d ** 2).mean())), float(np.abs(d).mean())

    rm_s, ma_s = map(np.array, zip(*[_frame_err(g, e) for g, e in zip(gt_f, est_f)]))
    rm_r, ma_r = map(np.array, zip(*[_frame_err(g, rel.render(est_raw, s))
                                     for g, s in zip(gt_f, shs)]))
    err_maps = [np.abs(g - e).mean(-1) for g, e in zip(gt_f, est_f)]
    emax = float(np.percentile(np.stack(err_maps), 99.5)) or 1.0

    res = dict(sweep_rmse_scaled=float(rm_s.mean()), sweep_mae_scaled=float(ma_s.mean()),
               sweep_rmse_raw=float(rm_r.mean()), sweep_mae_raw=float(ma_r.mean()),
               sweep_rmse_scaled_min=float(rm_s.min()), sweep_rmse_scaled_max=float(rm_s.max()),
               sweep_scale_ratio=float(rm_r.mean() / max(rm_s.mean(), 1e-12)),
               az_from=az_from, az_to=az_to, n_frames=n_frames, elev=elev,
               diffuse_fresnel=diffuse_fresnel, downsample=downsample,
               rmse_scaled_per_az=[float(v) for v in rm_s],
               rmse_raw_per_az=[float(v) for v in rm_r],
               azimuths=[float(v) for v in az])

    tm = _tonemap_fn(gt_f)                      # GT sets the exposure for every panel

    if plots:
        import matplotlib.pyplot as plt
        idx = np.linspace(0, n_frames - 1, min(n_panels, n_frames)).astype(int)
        fig, axes = plt.subplots(3, len(idx), figsize=(3.1 * len(idx), 9), squeeze=False)
        for c, i in enumerate(idx):
            for r, (im, t, cm) in enumerate([(tm(gt_f[i]), "GT", None),
                                             (tm(est_f[i]), "EST", None),
                                             (err_maps[i], "|err|", "inferno")]):
                a = axes[r][c]
                h = a.imshow(im, cmap=cm, vmin=0 if cm else None, vmax=emax if cm else None)
                a.axis("off")
                if r == 0:
                    a.set_title(f"az={az[i]:+.0f}deg", fontsize=9)
                if c == 0:
                    a.text(-0.08, 0.5, t, transform=a.transAxes, rotation=90,
                           va="center", ha="center", fontsize=10)
                if cm and c == len(idx) - 1:
                    plt.colorbar(h, ax=a, fraction=0.046)
        fig.suptitle(f"{run_dir.name[:70]}\ndirectional sweep {az_from:+.0f}..{az_to:+.0f}deg, "
                     f"elev {elev:.0f}deg", fontsize=9)
        plt.tight_layout(); fig.savefig(out_dir / "sweep_panels.png", dpi=dpi); plt.close(fig)

        fig, a = plt.subplots(figsize=(7.5, 4))
        a.plot(az, rm_r, lw=1.8, label=f"raw albedo_est (mean {rm_r.mean():.4f})")
        a.plot(az, rm_s, lw=1.8, label=f"scaled albedo (mean {rm_s.mean():.4f})")
        a.set_xlabel("light azimuth (deg)"); a.set_ylabel("relight RMSE vs GT")
        a.set_title(f"{run_dir.name[:60]}\nrelighting error across the sweep", fontsize=9)
        a.grid(alpha=0.3); a.legend(fontsize=8)
        plt.tight_layout(); fig.savefig(out_dir / "sweep_error.png", dpi=dpi); plt.close(fig)
        res["plots"] = [str(out_dir / "sweep_panels.png"), str(out_dir / "sweep_error.png")]

    if video:
        import matplotlib.pyplot as plt
        cmap = plt.get_cmap("inferno")

        def err_rgb(e):
            rgb = (cmap(np.clip(e / emax, 0, 1))[..., :3] * 255).astype(np.uint8)
            rgb[~mask] = 0
            return rgb

        combo, est_only = [], []
        for g, e, er, a_ in zip(gt_f, est_f, err_maps, az):
            g8, e8, r8 = (_upscale(tm(g), min_panel_px), _upscale(tm(e), min_panel_px),
                          _upscale(err_rgb(er), min_panel_px))
            combo.append(np.concatenate([_label(g8, "GT"), _label(e8, "EST"),
                                         _label(r8, f"|err| az={a_:+.0f}")], axis=1))
            est_only.append(e8)
        res["video"] = _write_video(combo, str(out_dir / "compare"), fps, ping_pong, save_frames)
        res["video_est"] = _write_video(est_only, str(out_dir / "est"), fps, ping_pong, False)
        res["error_scale"] = emax

    (out_dir / "sweep.json").write_text(__import__("json").dumps(res, indent=1))
    del rel                                     # drop the geometry + LUT before returning
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()
    return res
