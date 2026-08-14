#!/usr/bin/env python
"""
scripts/canonical_relight_render.py — relight recovered intrinsics under a fresh light and
compare against the GT (or Marigold) intrinsics through identical geometry.

Two lighting models, both moving a single light of the SAME direction/colour so the ONLY
difference is band-limiting:
  * "sh3"       — order-3 SH directional light via shade_ct_sh. Diffuse + soft specular; a
                  band-limited light cannot carry a sharp highlight.
  * "env_sharp" — a narrow bright lobe on an equirect env map via shade_ct_env, so the
                  specular highlight stays crisp (what SH throws away).

Used two ways:
  * imported by scripts/canonical_decomp_batch.py for the scalar relighting_rmse metric
    (mean over a small reference-light set, sh3);
  * imported by the overview notebook (or run as a CLI) to render, for chosen (scene, config):
      - a comparison panel  GT-relit | EST-relit | |err|   (sh3 and/or env_sharp), and
      - a relight VIDEO sweeping azimuth az_from..az_to at a fixed elevation (default the
        requested -45..45 at elevation 45), GT | EST | |err| side by side, as mp4 (ffmpeg)
        with a GIF fallback.

Everything is relit through make_proxy_geometry (fov 60, cam_dist 2) — the same proxy the
decomposition was fit under — so GT and EST differ only in albedo/metallic/roughness.
"""
from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # repo root for `idr`

import numpy as np
import torch
from PIL import Image

from idr.data.geometry import make_proxy_geometry
from idr.data.scene_io import load_scene
from idr.render import shade_ct_sh, shade_ct_env, EnvMap
from idr.render.sh import build_sh_basis
from idr.render.brdf import _get_ggx_sh_lut
from idr.config import LIGHT_COLOR, LIGHT_INTENSITY
from idr.eval.relight_sweep import _tonemap_fn, _upscale, _label


def _write_mp4(frames, stem, fps=20, ping_pong=True):
    """Encode uint8 frames to mp4 via the installed ffmpeg (no imageio dependency, which has
    no wheel on this Python). Falls back to an animated GIF if ffmpeg is missing or fails."""
    seq = list(frames) + list(frames)[-2:0:-1] if ping_pong else list(frames)
    ff = shutil.which("ffmpeg")
    if ff:
        with tempfile.TemporaryDirectory() as td:
            for i, f in enumerate(seq):
                h, w = f.shape[:2]
                if h % 2 or w % 2:                        # yuv420p needs even dimensions
                    f = np.pad(f, ((0, h % 2), (0, w % 2), (0, 0)))
                Image.fromarray(f).save(f"{td}/f_{i:05d}.png")
            out = f"{stem}.mp4"
            r = subprocess.run([ff, "-y", "-framerate", str(fps), "-i", f"{td}/f_%05d.png",
                                "-c:v", "libx264", "-pix_fmt", "yuv420p",
                                "-movflags", "+faststart", out], capture_output=True)
            if r.returncode == 0 and os.path.exists(out):
                return out
    ims = [Image.fromarray(f) for f in seq]
    ims[0].save(f"{stem}.gif", save_all=True, append_images=ims[1:],
                duration=int(1000 / fps), loop=0)
    return f"{stem}.gif"

__all__ = ["dir_sh3", "dir_unit", "sharp_env_pixels", "Relighter",
           "reference_relight_rmse", "render_comparison", "render_relight_video"]


# ── light construction ────────────────────────────────────────────────────────
def dir_unit(az_deg: float, elev_deg: float) -> np.ndarray:
    """Unit light direction. Convention matches idr.eval.relight_sweep.dir_sh / make_proxy_
    geometry: camera at (0,0,+2) looking -Z, +Y up; az about Y, az=0 = lit from the camera."""
    az, el = math.radians(az_deg), math.radians(elev_deg)
    d = np.array([math.cos(el) * math.sin(az), math.sin(el), math.cos(el) * math.cos(az)],
                 np.float32)
    return d / (np.linalg.norm(d) + 1e-8)


def dir_sh3(az_deg: float, elev_deg: float, color=LIGHT_COLOR,
            intensity: float = LIGHT_INTENSITY) -> np.ndarray:
    """Order-3 SH coefficients (16,3) for a directional light."""
    d = dir_unit(az_deg, elev_deg)
    c = np.asarray(color, np.float32) * intensity
    return (build_sh_basis(d[None, :], order=3)[0][:, None] * c[None, :]).astype(np.float32)


def sharp_env_pixels(env_dirs: np.ndarray, az_deg: float, elev_deg: float,
                     color=LIGHT_COLOR, intensity: float = LIGHT_INTENSITY,
                     sigma_deg: float = 4.0, ambient: float = 0.0) -> np.ndarray:
    """A narrow bright Gaussian lobe on the env grid — a sharp, controllable highlight.

    env_dirs: (P,3) unit directions of the env sampling grid. Returns (P,3) radiance. The
    lobe integrates to roughly `intensity` regardless of sigma so brightness is comparable
    across sharpness settings."""
    d = dir_unit(az_deg, elev_deg)
    cos = np.clip(env_dirs @ d, -1.0, 1.0)
    ang = np.arccos(cos)                                     # radians from the light dir
    sig = math.radians(sigma_deg)
    lobe = np.exp(-0.5 * (ang / sig) ** 2)
    lobe = lobe / (lobe.sum() + 1e-8)                        # normalise total energy
    scale = float(intensity) * env_dirs.shape[0] / (4 * np.pi)  # ~match a unit directional
    px = (lobe[:, None] * (np.asarray(color, np.float32)[None, :] * scale)).astype(np.float32)
    return px + float(ambient)


# ── relighter (both models) ───────────────────────────────────────────────────
class Relighter:
    """Fixed proxy geometry + GGX LUT + env grid; relights any intrinsics under sh3 or a
    sharp env light. Build once per scene (geometry/LUT are the expensive parts)."""

    def __init__(self, normals_np, mask_np, device="cuda", dtype=torch.float32,
                 diffuse_fresnel=True, env_resolution=96):
        self.H, self.W = normals_np.shape[:2]
        self.mask_np = mask_np
        self.device, self.dtype, self.diffuse_fresnel = device, dtype, diffuse_fresnel
        n, f, m, cam = make_proxy_geometry(normals_np, mask_np, 60.0, 2.0, device, dtype)
        self.fm = m.reshape(-1)
        self._fm_np = self.fm.detach().cpu().numpy()
        self.N = n.reshape(-1, 3)[self.fm]
        self.V = torch.nn.functional.normalize(cam[None] - f.reshape(-1, 3)[self.fm], dim=-1)
        self.lut = _get_ggx_sh_lut(device, n_bands=4).to(dtype)   # order-3 bands
        # env sampling grid (equirect) shared by every env-sharp frame
        env = EnvMap.constant(resolution=env_resolution)
        self.env_dirs_np = env._dirs.astype(np.float32)
        self.env_dirs = torch.from_numpy(self.env_dirs_np).to(device, dtype)
        self.env_dw = torch.from_numpy(env._solid_angles.astype(np.float32)).to(device, dtype)

    def _m(self, a, c):
        return torch.from_numpy(np.ascontiguousarray(a)).to(self.device, self.dtype) \
                    .reshape(-1, c)[self.fm]

    def _scatter(self, px):
        out = np.zeros((self.H * self.W, 3), np.float32)
        out[self._fm_np] = px.float().cpu().numpy()
        return out.reshape(self.H, self.W, 3)

    def render_sh(self, intr, sh):
        """Render under GIVEN SH coefficients (n_sh, 3) — e.g. a training light's GT or
        estimated lighting, rather than a synthetic directional light."""
        sh_t = torch.from_numpy(np.asarray(sh, np.float32)).to(self.device, self.dtype)
        with torch.no_grad():
            px = shade_ct_sh(self.V, self.N, self._m(intr["albedo"], 3), sh_t,
                             self._m(intr["metallic"], 1), self._m(intr["roughness"], 1),
                             lut=self.lut, diffuse_fresnel=self.diffuse_fresnel)
        return self._scatter(px)

    def render_sh3(self, intr, az, el, color=LIGHT_COLOR, intensity=LIGHT_INTENSITY):
        return self.render_sh(intr, dir_sh3(az, el, color, intensity))

    def render_env(self, intr, az, el, color=LIGHT_COLOR, intensity=LIGHT_INTENSITY,
                   sigma_deg=4.0):
        px_env = sharp_env_pixels(self.env_dirs_np, az, el, color, intensity, sigma_deg)
        with torch.no_grad():
            px = shade_ct_env(self.V, self.N, self._m(intr["albedo"], 3),
                              torch.from_numpy(px_env).to(self.device, self.dtype),
                              self.env_dirs, self.env_dw,
                              self._m(intr["metallic"], 1), self._m(intr["roughness"], 1),
                              diffuse_fresnel=self.diffuse_fresnel)
        return self._scatter(px)

    def render(self, intr, mode, az, el, **kw):
        return self.render_env(intr, az, el, **kw) if mode == "env_sharp" \
            else self.render_sh3(intr, az, el, **kw)


# ── loading est / gt intrinsics ───────────────────────────────────────────────
def _stride(a, ds):
    return np.ascontiguousarray(a[::ds, ::ds]) if ds > 1 else a


def load_est_intrinsics(run_dir, scaled=True):
    """Load estimated intrinsics. For the scaled albedo we RECOMPUTE albedo_est * albedo_scale
    (clipped) from metrics.json rather than reading albedo_scaled.npy, because older runs saved
    a corrupted albedo_scaled.npy (the signed albedo error) — see the decompose_scene fix. Raw
    albedo_est.npy has always been correct, so this renders old runs correctly without redoing
    the decomposition."""
    run_dir = Path(run_dir)
    metallic = np.load(run_dir / "metallic_est.npy").astype(np.float32)
    roughness = np.load(run_dir / "roughness_est.npy").astype(np.float32)
    raw = np.load(run_dir / "albedo_est.npy").astype(np.float32)
    albedo = raw
    if scaled:
        try:
            scale = np.asarray(json.loads((run_dir / "metrics.json").read_text())["albedo_scale"],
                               np.float32)
            albedo = np.clip(raw * scale, 0.0, 1.0)
        except Exception:                                     # no metrics -> fall back to raw
            albedo = raw
    return dict(albedo=albedo.astype(np.float32), metallic=metallic, roughness=roughness)


def load_gt_intrinsics(scene_dir, ds=1):
    sc = load_scene(Path(scene_dir), gt_npy=True)
    return dict(albedo=_stride(sc["albedo_np"], ds), metallic=_stride(sc["metallic_np"], ds),
                roughness=_stride(sc["roughness_np"], ds),
                normals=_stride(sc["normals_np"], ds), mask=_stride(sc["mask_np"], ds))


# ── scalar metric: est-vs-GT-intrinsics relight over a small reference set ─────
_REF_LIGHTS = [(az, 45.0) for az in (-45, -22.5, 0, 22.5, 45)] + [(0.0, 20.0), (0.0, 70.0)]


def reference_relight_rmse(run_dir, scene_dir, ds=1, device="cuda", diffuse_fresnel=True,
                           lights=_REF_LIGHTS, scaled=True):
    """Mean RMSE between EST-intrinsics and GT-intrinsics renders over a fixed sh3 reference
    light set (masked). Both rendered through identical geometry, so this isolates intrinsics."""
    gt = load_gt_intrinsics(scene_dir, ds)
    est = load_est_intrinsics(run_dir, scaled=scaled)
    if gt["albedo"].shape[:2] != est["albedo"].shape[:2]:
        raise ValueError(f"resolution mismatch GT {gt['albedo'].shape[:2]} vs EST "
                         f"{est['albedo'].shape[:2]} at ds={ds}")
    rel = Relighter(gt["normals"], gt["mask"], device, torch.float32, diffuse_fresnel)
    mask = gt["mask"]
    rs = []
    for az, el in lights:
        g = rel.render_sh3(gt, az, el)[mask]
        e = rel.render_sh3(est, az, el)[mask]
        rs.append(float(np.sqrt(((g - e) ** 2).mean())))
    del rel
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()
    return float(np.mean(rs)), [float(x) for x in rs]


# ── comparison panel + video ──────────────────────────────────────────────────
def render_comparison(run_dir, scene_dir, mode="sh3", az=0.0, el=45.0, ds=1,
                      device="cuda", diffuse_fresnel=True, out_path=None, dpi=110,
                      scaled=True, sigma_deg=4.0):
    """GT-relit | EST-relit | |err| panel under one light. Returns the saved path.

    Does NOT switch the matplotlib backend — the figure is saved and closed (never shown), so
    this stays notebook-safe (calling it must not break subsequent inline `plt.show()` cells)."""
    import matplotlib.pyplot as plt
    gt = load_gt_intrinsics(scene_dir, ds); est = load_est_intrinsics(run_dir, scaled=scaled)
    rel = Relighter(gt["normals"], gt["mask"], device, torch.float32, diffuse_fresnel)
    kw = dict(sigma_deg=sigma_deg) if mode == "env_sharp" else {}
    g = rel.render(gt, mode, az, el, **kw); e = rel.render(est, mode, az, el, **kw)
    err = np.abs(g - e).mean(-1) * gt["mask"]
    tm = _tonemap_fn([g])
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.6))
    for a, im, t, cm in [(ax[0], tm(g), "GT relit", None), (ax[1], tm(e), "EST relit", None),
                         (ax[2], err, "|err|", "inferno")]:
        h = a.imshow(im, cmap=cm); a.set_title(t, fontsize=10); a.axis("off")
        if cm:
            plt.colorbar(h, ax=a, fraction=0.046)
    fig.suptitle(f"{Path(run_dir).name}  |  {mode}  az={az:+.0f} el={el:.0f}", fontsize=10)
    out_path = Path(out_path) if out_path else Path(run_dir) / f"compare_{mode}_az{int(az)}_el{int(el)}.png"
    plt.tight_layout(); fig.savefig(out_path, dpi=dpi); plt.close(fig)
    del rel
    return str(out_path)


def render_relight_video(run_dir, scene_dir, mode="env_sharp", az_from=-45.0, az_to=45.0,
                         elev=45.0, n_frames=61, ds=1, device="cuda", diffuse_fresnel=True,
                         fps=20, out_stem=None, ping_pong=True, min_px=256, scaled=True,
                         sigma_deg=4.0):
    """Sweep azimuth az_from..az_to at fixed elevation; write GT | EST | |err| video (mp4 via
    ffmpeg, else GIF). Returns the written path."""
    gt = load_gt_intrinsics(scene_dir, ds); est = load_est_intrinsics(run_dir, scaled=scaled)
    rel = Relighter(gt["normals"], gt["mask"], device, torch.float32, diffuse_fresnel)
    mask = gt["mask"]; kw = dict(sigma_deg=sigma_deg) if mode == "env_sharp" else {}
    az = np.linspace(az_from, az_to, n_frames)
    gt_f = [rel.render(gt, mode, a, elev, **kw) for a in az]
    est_f = [rel.render(est, mode, a, elev, **kw) for a in az]
    errs = [np.abs(g - e).mean(-1) for g, e in zip(gt_f, est_f)]
    emax = float(np.percentile(np.stack(errs), 99.5)) or 1.0
    tm = _tonemap_fn(gt_f)
    import matplotlib.pyplot as plt
    cmap = plt.get_cmap("inferno")

    def err_rgb(e):
        rgb = (cmap(np.clip(e / emax, 0, 1))[..., :3] * 255).astype(np.uint8); rgb[~mask] = 0
        return rgb
    frames = []
    for g, e, er, a_ in zip(gt_f, est_f, errs, az):
        g8, e8, r8 = (_upscale(tm(g), min_px), _upscale(tm(e), min_px), _upscale(err_rgb(er), min_px))
        frames.append(np.concatenate([_label(g8, "GT"), _label(e8, "EST"),
                                       _label(r8, f"|err| az={a_:+.0f}")], axis=1))
    out_stem = str(out_stem) if out_stem else str(Path(run_dir) / f"relight_{mode}")
    path = _write_mp4(frames, out_stem, fps=fps, ping_pong=ping_pong)
    del rel
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()
    return path


# ── CLI ───────────────────────────────────────────────────────────────────────
def _build_parser():
    import argparse
    p = argparse.ArgumentParser(description="Relight comparison plots + videos for one run.")
    p.add_argument("--run_dir", required=True, help="a per-(scene,config) decomposition run dir")
    p.add_argument("--scene_dir", required=True, help="the source scene dir (for GT intrinsics)")
    p.add_argument("--ds", type=int, default=1, help="downsample the run was decoded at")
    p.add_argument("--what", nargs="+", default=["panel_sh3", "panel_env", "video_env", "video_sh3"],
                   choices=["panel_sh3", "panel_env", "video_env", "video_sh3"])
    p.add_argument("--az", type=float, default=0.0); p.add_argument("--el", type=float, default=45.0)
    p.add_argument("--az_range", type=float, nargs=2, default=[-45.0, 45.0])
    p.add_argument("--elev", type=float, default=45.0); p.add_argument("--frames", type=int, default=61)
    p.add_argument("--fps", type=int, default=20); p.add_argument("--sigma_deg", type=float, default=4.0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p


def main():
    a = _build_parser().parse_args()
    rd, sd = a.run_dir, a.scene_dir
    if "panel_sh3" in a.what:
        print(render_comparison(rd, sd, "sh3", a.az, a.el, a.ds, a.device))
    if "panel_env" in a.what:
        print(render_comparison(rd, sd, "env_sharp", a.az, a.el, a.ds, a.device, sigma_deg=a.sigma_deg))
    if "video_env" in a.what:
        print(render_relight_video(rd, sd, "env_sharp", a.az_range[0], a.az_range[1], a.elev,
                                   a.frames, a.ds, a.device, fps=a.fps, sigma_deg=a.sigma_deg))
    if "video_sh3" in a.what:
        print(render_relight_video(rd, sd, "sh3", a.az_range[0], a.az_range[1], a.elev,
                                   a.frames, a.ds, a.device, fps=a.fps))


if __name__ == "__main__":
    main()
