"""
CT decomposition and rendering for 3D-Front scenes.

The scene directory must contain:
  albedo.png    uint8 RGB
  normals.png   uint8 RGB  (world-space, encoded as (n+1)/2*255)
  roughness.png uint16 grayscale  (stored as linear roughness * 65535)
  metallic.png  uint16 grayscale  (stored as metallic * 65535)
  light_000.png ... light_N.png   uint8 RGB rendered images

Usage (CLI):
  python -m raw_optimizer.dfront_ct render \
      --scene datasets/3D-Front/00ad8345-45e0-45b3-867d-4a3c88c2517a/0 \
      --out 3dfront_results/00ad8345/rendered \
      --az 45 --shader ct_sh

  python -m raw_optimizer.dfront_ct decompose \
      --scene datasets/3D-Front/00ad8345-45e0-45b3-867d-4a3c88c2517a/0 \
      --out 3dfront_results/00ad8345/ct_sh \
      --shader ct_sh --n_iter 100 --lambda_tv 0.001
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import wandb
from PIL import Image

from raw_renderer_gpu import (
    shade_ct_sh, shade_ct_env,
    SHLight, EnvMapLightGPU,
    EnvMap, SHLighting,
)
from raw_renderer_gpu.rasterizer import _get_ggx_sh_lut
from raw_optimizer.synthetic_ct_dataset import (
    _optimize_ct_sh,
    _optimize_ct_env,
    _sh_coeffs_to_env_img,
    _env_flat_to_img,
    _make_lights_random_sh,
    DEFAULT_CFG,
    LIGHT_COLOR,
    LIGHT_INTENSITY,
)
from raw_optimizer.helper import _albedo_rmse

_WANDB_ENTITY  = "DLVC-intrinsics"
_WANDB_PROJECT = "3dfront_ct_decomp"

_REPO_ROOT = Path(__file__).parent.parent

# ─────────────────────────────────────── image helpers ──────────────────────


def srgb_to_linear(img: np.ndarray) -> np.ndarray:
    """float32 [0,1] sRGB → float32 [0,1] linear light (IEC 61966-2-1 EOTF)."""
    img = np.clip(img, 0.0, 1.0).astype(np.float32)
    return np.where(img <= 0.04045, img / 12.92, ((img + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(img: np.ndarray) -> np.ndarray:
    """float32 [0,1] linear light → float32 [0,1] sRGB (IEC 61966-2-1 OETF)."""
    img = np.clip(img, 0.0, 1.0).astype(np.float32)
    return np.where(img <= 0.0031308, img * 12.92, 1.055 * img ** (1.0 / 2.4) - 0.055)


def load_exr(path: Path) -> np.ndarray:
    """Load an EXR file as (H, W, 3) float32 linear-light RGB.

    Tries imageio (requires imageio-freeimage or OpenEXR) then falls back to
    OpenCV (cv2), which supports EXR natively on most platforms.
    """
    path = Path(path)
    try:
        import imageio
        img = np.asarray(imageio.imread(str(path))).astype(np.float32)
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)
        return img[:, :, :3]
    except Exception:
        import cv2  # noqa: PLC0415
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED | cv2.IMREAD_ANYDEPTH)
        if img is None:
            raise IOError(f"Could not read EXR (tried imageio and cv2): {path}")
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)
        return img[:, :, :3][:, :, ::-1].astype(np.float32)  # BGR→RGB


# ─────────────────────────────────────── data loading ───────────────────────


def load_scene(scene_dir: Path, no_shadow: bool = False) -> dict:
    """Load all maps from a 3D-Front scene directory.

    Reads config.json (if present) to determine the image variant:
      "linear" (default) — light_NNN.png loaded as float32 [0,1]
      "srgb"             — light_NNN.png loaded and sRGB-linearized
      "exr"              — light_NNN.npy loaded as float32 (may be HDR)

    Returns a dict with keys:
      normals_np   (H, W, 3) float32, unit length
      mask_np      (H, W)    bool  — True for foreground pixels
      albedo_np    (H, W, 3) float32 [0,1]
      metallic_np  (H, W, 1) float32 [0,1]
      roughness_np (H, W, 1) float32 [0,1]
      images       list of (H, W, 3) float32
      light_keys   list of str  (e.g. ["light_000", ...])
      H, W         ints

    no_shadow: if True and light_NNN_no_shadow.png files exist, load those
               instead of the standard light_NNN.png renders.
               (Ignored for the "exr" variant.)
    """
    scene_dir = Path(scene_dir)

    # Auto-detect variant from config.json
    cfg_path = scene_dir / "config.json"
    variant = "linear"
    if cfg_path.exists():
        with open(cfg_path) as fh:
            _ds_cfg = json.load(fh)
        variant = _ds_cfg.get("variant", "linear")

    # Normals: uint8 → float [-1,1], then renormalize
    norm_raw = np.array(Image.open(scene_dir / "normals.png"), dtype=np.float32)
    normals = norm_raw / 255.0 * 2.0 - 1.0
    nlen = np.linalg.norm(normals, axis=-1, keepdims=True).clip(1e-6, None)
    normals = normals / nlen  # (H, W, 3), unit length

    # Mask: pixels where the raw normal is non-zero (background is all-black)
    mask = (norm_raw.sum(axis=-1) > 0)  # (H, W) bool

    # Albedo: uint8 → float [0,1]
    albedo = np.array(Image.open(scene_dir / "albedo.png"), dtype=np.float32)
    albedo = albedo[:, :, :3] / 255.0

    # Roughness / metallic: uint16 → float [0,1]
    rough = np.array(Image.open(scene_dir / "roughness.png"), dtype=np.float32)
    rough = (rough / 65535.0)[:, :, None]  # (H, W, 1)

    metal = np.array(Image.open(scene_dir / "metallic.png"), dtype=np.float32)
    metal = (metal / 65535.0)[:, :, None]  # (H, W, 1)

    # Rendered images
    if variant == "exr":
        # EXR datasets store images as float32 .npy (possibly HDR, already normalized)
        npy_files = sorted(
            (f for f in scene_dir.glob("light_*.npy")),
            key=lambda p: int(p.stem.split("_")[-1]),
        )
        images     = [np.load(f).astype(np.float32) for f in npy_files]
        light_keys = [f.stem for f in npy_files]
    else:
        # PNG datasets (linear or sRGB)
        base_files = sorted(
            (f for f in scene_dir.glob("light_*.png")
             if "_no_shadow" not in f.stem and "_preview" not in f.stem),
            key=lambda p: int(p.stem.split("_")[-1]),
        )
        if no_shadow:
            img_files = []
            for bf in base_files:
                ns = bf.with_stem(bf.stem + "_no_shadow")
                img_files.append(ns if ns.exists() else bf)
        else:
            img_files = base_files

        images = [
            np.array(Image.open(f), dtype=np.float32)[:, :, :3] / 255.0
            for f in img_files
        ]
        if variant == "srgb":
            images = [srgb_to_linear(img) for img in images]
        # Always use the base key (light_NNN) regardless of which file was loaded
        light_keys = [bf.stem for bf in base_files]

    H, W = normals.shape[:2]
    return dict(
        normals_np=normals,
        mask_np=mask,
        albedo_np=albedo,
        metallic_np=metal,
        roughness_np=rough,
        images=images,
        light_keys=light_keys,
        H=H, W=W,
    )


# ────────────────────────────────── proxy geometry ──────────────────────────


def make_proxy_geometry(
    normals_np: np.ndarray,
    mask_np: np.ndarray,
    fov_deg: float = 60.0,
    cam_dist: float = 2.0,
    device: str = "cuda",
) -> tuple:
    """Return (normals_hw, frag_pos_hw, mask_hw, cam_pos) tensors on device.

    Since the 3D-Front scene has no depth map, fragment positions are
    approximated by placing the camera at (0, 0, cam_dist) and distributing
    pixel positions on the z=0 plane according to fov_deg (perspective).
    This gives correct view-direction variation across the image without
    actual depth.
    """
    H, W = normals_np.shape[:2]

    # Pixel grid in NDC [-1, 1] (y flipped: top row = +1)
    ys = np.linspace(1.0, -1.0, H, dtype=np.float32)
    xs = np.linspace(-1.0,  1.0, W, dtype=np.float32)
    xg, yg = np.meshgrid(xs, ys)  # (H, W)

    tan_half = float(np.tan(np.radians(fov_deg / 2)))
    frag_pos = np.stack([
        xg * tan_half,
        yg * tan_half,
        np.zeros((H, W), dtype=np.float32),
    ], axis=-1)  # (H, W, 3)

    cam_pos = np.array([0.0, 0.0, cam_dist], dtype=np.float32)

    def _t(x):
        return torch.from_numpy(x).to(device)

    return (
        _t(normals_np),
        _t(frag_pos),
        _t(mask_np),
        _t(cam_pos),
    )


# ──────────────────────────────────── rendering ─────────────────────────────


def _make_directional_sh(az_deg: float = 45.0, el_deg: float = 0.0,
                          color: np.ndarray = LIGHT_COLOR,
                          intensity: float = LIGHT_INTENSITY) -> SHLighting:
    """Create SH lighting from a single directional light.

    az_deg: azimuth in the XZ plane, measured from +Z (0=frontal, 90=right)
    el_deg: elevation above the XZ plane (positive = upward)
    """
    az = np.radians(az_deg)
    el = np.radians(el_deg)
    direction = np.array([
        np.sin(az) * np.cos(el),
        np.sin(el),
        np.cos(az) * np.cos(el),
    ], dtype=np.float32)
    return SHLighting.directional(direction, color, intensity=intensity)


def render_scene(
    scene_dir: Path,
    out_dir: Path,
    az_deg: float = 45.0,
    el_deg: float = 0.0,
    intensity: float = LIGHT_INTENSITY,
    color: np.ndarray | None = None,
    shader: str = "ct_sh",
    fov_deg: float = 60.0,
    cam_dist: float = 2.0,
    device: str = "cuda",
) -> np.ndarray:
    """Render the 3D-Front scene using GT material maps and a single directional light.

    Saves render to out_dir/render.png (+.npy) and the SH env map to
    out_dir/sh_env_map.png.  Returns the render as (H,W,3) float32.

    shader: "ct_sh" renders with SH diffuse+specular;
            "ct_env" converts SH to an env map first (same result, slower).
    """
    scene_dir = Path(scene_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    scene = load_scene(scene_dir)
    H, W = scene["H"], scene["W"]
    normals_hw, frag_pos_hw, mask_hw, cam_pos = make_proxy_geometry(
        scene["normals_np"], scene["mask_np"],
        fov_deg=fov_deg, cam_dist=cam_dist, device=device,
    )

    # Build lighting
    _color = LIGHT_COLOR if color is None else np.asarray(color, dtype=np.float32)
    sh_light = _make_directional_sh(az_deg, el_deg, color=_color, intensity=intensity)
    sh_coeffs_t = torch.from_numpy(sh_light.coeffs).to(device)  # (9, 3)

    # Flatten to masked pixels
    flat_mask = mask_hw.reshape(-1)
    N_m = int(flat_mask.sum())
    def _flat(arr_hw):
        t = torch.from_numpy(arr_hw).to(device)
        return t.reshape(-1, t.shape[-1])[flat_mask]

    albedo_m  = _flat(scene["albedo_np"].astype(np.float32))     # (M, 3)
    rough_m   = _flat(scene["roughness_np"].astype(np.float32))  # (M, 1)
    metal_m   = _flat(scene["metallic_np"].astype(np.float32))   # (M, 1)
    normals_m = normals_hw.reshape(-1, 3)[flat_mask]
    frag_m    = frag_pos_hw.reshape(-1, 3)[flat_mask]
    view_m    = torch.nn.functional.normalize(cam_pos.unsqueeze(0) - frag_m, dim=-1)

    # Render
    lut = _get_ggx_sh_lut(device)
    if shader == "ct_sh":
        render_m = shade_ct_sh(
            view_m, normals_m, albedo_m,
            sh_coeffs_t,  # (9, 3) — shared across all pixels
            metallic=metal_m, roughness=rough_m, lut=lut,
        )
    elif shader == "ct_env_direct":
        # Direct env map: borrow the grid layout from the SH env map, then
        # place all light energy in the single pixel closest to the light
        # direction.  Dividing by solid_angle makes the integral equal to
        # the desired color × intensity (same energy as the SH version).
        az = np.radians(az_deg)
        el = np.radians(el_deg)
        light_dir = np.array([
            np.sin(az) * np.cos(el),
            np.sin(el),
            np.cos(az) * np.cos(el),
        ], dtype=np.float32)
        env_raw    = EnvMap.from_sh(sh_light)
        best_idx   = int((env_raw._dirs @ light_dir).argmax())
        direct_pix = np.zeros_like(env_raw._image_flat)
        direct_pix[best_idx] = _color * intensity / env_raw._solid_angles[best_idx]
        env_dirs_t = torch.from_numpy(env_raw._dirs).to(device)
        env_dw_t   = torch.from_numpy(env_raw._solid_angles).to(device)
        env_pix_t  = torch.from_numpy(direct_pix).to(device)
        render_m = shade_ct_env(
            view_m, normals_m, albedo_m,
            env_pix_t, env_dirs_t, env_dw_t,
            metallic=metal_m, roughness=rough_m,
        )
    else:  # ct_env — SH reconstructed env map
        env_raw = EnvMap.from_sh(sh_light)
        env_dirs_t = torch.from_numpy(env_raw._dirs).to(device)
        env_dw_t   = torch.from_numpy(env_raw._solid_angles).to(device)
        env_pix_t  = torch.from_numpy(env_raw._image_flat).to(device)
        render_m = shade_ct_env(
            view_m, normals_m, albedo_m,
            env_pix_t,  # (P, 3) — shared across all pixels
            env_dirs_t, env_dw_t, metallic=metal_m, roughness=rough_m,
        )

    # Scatter back to full image
    render_np = np.zeros((H * W, 3), dtype=np.float32)
    render_np[flat_mask.cpu().numpy()] = render_m.clamp(0, 1).float().cpu().numpy()
    render_np = render_np.reshape(H, W, 3)

    # Save
    Image.fromarray((render_np * 255).astype(np.uint8)).save(out_dir / "render.png")
    np.save(out_dir / "render.npy", render_np)

    sh_env_img = _sh_coeffs_to_env_img(sh_light.coeffs)
    Image.fromarray((sh_env_img * 255).astype(np.uint8)).save(out_dir / "sh_env_map.png")

    light_cfg = {
        "az_deg": az_deg, "el_deg": el_deg,
        "intensity": intensity, "color": _color.tolist(),
        "sh_coeffs": sh_light.coeffs.tolist(),
    }
    with open(out_dir / "light_config.json", "w") as fh:
        json.dump(light_cfg, fh, indent=2)

    print(f"Rendered {H}x{W} -> {out_dir / 'render.png'}")
    return render_np


def render_3dfront_dataset(
    scene_dir: Path,
    out_dir: Path,
    n_lights: int = 16,
    seed: int = 0,
    fov_deg: float = 60.0,
    cam_dist: float = 2.0,
    device: str = "cuda",
) -> None:
    """Render 3D-Front GT material maps under n_lights random SH conditions.

    Uses the same random SH distribution as the synthetic dataset's
    random_sh light mode (_make_lights_random_sh with sequential seeds).

    Saves:
      out_dir/albedo.png          — GT albedo (copied from scene_dir)
      out_dir/normals.png         — GT normals (copied from scene_dir)
      out_dir/roughness.png       — GT roughness (copied from scene_dir)
      out_dir/metallic.png        — GT metallic (copied from scene_dir)
      out_dir/light_{i:03d}.png   — rendered uint8 RGB image
      out_dir/light_{i:03d}.npy   — rendered float32 array
      out_dir/sh_{i:03d}.npy      — SH coefficients (9, 3) float32
      out_dir/sh_env_map_{i:03d}.png — SH env map visualization
      out_dir/config.json         — dataset metadata

    The output directory doubles as a valid scene_dir for decompose_scene.
    """
    scene_dir = Path(scene_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Copy GT material maps so the dataset folder is self-contained
    for fname in ("albedo.png", "normals.png", "roughness.png", "metallic.png"):
        shutil.copy2(scene_dir / fname, out_dir / fname)

    scene = load_scene(scene_dir)
    H, W = scene["H"], scene["W"]
    normals_hw, frag_pos_hw, mask_hw, cam_pos = make_proxy_geometry(
        scene["normals_np"], scene["mask_np"],
        fov_deg=fov_deg, cam_dist=cam_dist, device=device,
    )

    flat_mask = mask_hw.reshape(-1)

    def _flat(arr_hw):
        t = torch.from_numpy(arr_hw.astype(np.float32)).to(device)
        return t.reshape(-1, t.shape[-1])[flat_mask]

    albedo_m  = _flat(scene["albedo_np"])     # (M, 3)
    rough_m   = _flat(scene["roughness_np"])  # (M, 1)
    metal_m   = _flat(scene["metallic_np"])   # (M, 1)
    normals_m = normals_hw.reshape(-1, 3)[flat_mask]
    frag_m    = frag_pos_hw.reshape(-1, 3)[flat_mask]
    view_m    = torch.nn.functional.normalize(cam_pos.unsqueeze(0) - frag_m, dim=-1)

    lut = _get_ggx_sh_lut(device)

    for i in range(n_lights):
        sh_light, _, _, coeffs_np, _ = _make_lights_random_sh(seed=seed + i)
        sh_coeffs_t = torch.from_numpy(coeffs_np).to(device)  # (9, 3)

        with torch.no_grad():
            render_m = shade_ct_sh(
                view_m, normals_m, albedo_m,
                sh_coeffs_t,
                metallic=metal_m, roughness=rough_m, lut=lut,
            )

        render_np = np.zeros((H * W, 3), dtype=np.float32)
        render_np[flat_mask.cpu().numpy()] = render_m.clamp(0, 1).float().cpu().numpy()
        render_np = render_np.reshape(H, W, 3)

        Image.fromarray((render_np * 255).astype(np.uint8)).save(
            out_dir / f"light_{i:03d}.png")
        np.save(out_dir / f"light_{i:03d}.npy", render_np)
        np.save(out_dir / f"sh_{i:03d}.npy", coeffs_np)

        sh_env_img = _sh_coeffs_to_env_img(coeffs_np)
        Image.fromarray((sh_env_img * 255).astype(np.uint8)).save(
            out_dir / f"sh_env_map_{i:03d}.png")

        print(f"  light {i:03d}/{n_lights} done")

    cfg = {
        "scene_dir": str(scene_dir),
        "n_lights": n_lights,
        "seed": seed,
        "fov_deg": fov_deg,
        "cam_dist": cam_dist,
    }
    with open(out_dir / "config.json", "w") as fh:
        json.dump(cfg, fh, indent=2)

    print(f"3D-Front dataset rendered: {n_lights} images -> {out_dir}")


# ──────────────────────────────── sun dataset builder ───────────────────────


def _sorted_sun_files(src_dir: Path, pattern: str, exclude_suffix: str = "") -> list:
    """Glob sun render files and sort by numeric index (handles sun_10 > sun_9)."""
    files = [f for f in src_dir.glob(pattern)
             if not (exclude_suffix and f.stem.endswith(exclude_suffix))]
    def _idx(p: Path) -> int:
        stem = p.stem.replace("_srgb", "")
        return int(stem.rsplit("_", 1)[-1])
    return sorted(files, key=_idx)


def build_3dfront_sun_dataset(
    src_dir: Path,
    out_dir: Path,
    variant: str = "linear",
) -> None:
    """Build a CT decomposition dataset from 3D-Front sun renderings.

    Copies GT material maps (PNG) and converts sun renders to the standard
    light_NNN format expected by load_scene / decompose_scene.

    variant:
      "linear"  — sun_N.png   → light_NNN.png  (uint8 linear-light)
      "srgb"    — sun_N_srgb.png → light_NNN.png  (uint8 sRGB;
                  linearized automatically by load_scene at decomp time)
      "exr"     — sun_N.exr   → light_NNN.npy  (float32, HDR;
                  normalized to 99th-percentile ≈ 1.0)

    The output directory is a valid scene_dir for decompose_scene.
    """
    src_dir = Path(src_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # GT material maps always come from PNG
    for fname in ("albedo.png", "normals.png", "roughness.png", "metallic.png"):
        shutil.copy2(src_dir / fname, out_dir / fname)

    # Collect sun render files sorted numerically
    if variant == "linear":
        sun_files = _sorted_sun_files(src_dir, "sun_*.png", exclude_suffix="_srgb")
    elif variant == "srgb":
        sun_files = _sorted_sun_files(src_dir, "sun_*_srgb.png")
    elif variant == "exr":
        sun_files = _sorted_sun_files(src_dir, "sun_*.exr")
    else:
        raise ValueError(f"variant must be 'linear', 'srgb', or 'exr'; got {variant!r}")

    exr_scale = None
    if variant == "exr":
        # Load all EXR images, compute a single normalization scale so that
        # the 99th-percentile foreground value maps to ~1.0
        raw_imgs = [load_exr(f) for f in sun_files]
        all_pos = np.concatenate([img.reshape(-1) for img in raw_imgs])
        all_pos = all_pos[all_pos > 0]
        exr_scale = float(np.percentile(all_pos, 99)) if len(all_pos) else 1.0
        for i, img in enumerate(raw_imgs):
            img_norm = (img / exr_scale).astype(np.float32)
            np.save(out_dir / f"light_{i:03d}.npy", img_norm)
            # Preview PNG (clamped to [0,1]) for quick inspection
            Image.fromarray((img_norm.clip(0, 1) * 255).astype(np.uint8)).save(
                out_dir / f"light_{i:03d}_preview.png")
    else:
        for i, f in enumerate(sun_files):
            shutil.copy2(f, out_dir / f"light_{i:03d}.png")

    cfg: dict = {
        "src_dir": str(src_dir),
        "variant": variant,
        "n_lights": len(sun_files),
        "light_type": "sun",
    }
    if exr_scale is not None:
        cfg["exr_scale"] = exr_scale

    with open(out_dir / "config.json", "w") as fh:
        json.dump(cfg, fh, indent=2)

    print(f"Built '{variant}' sun dataset: {len(sun_files)} images -> {out_dir}")


# ─────────────────────────────────── decomposition ──────────────────────────


def decompose_scene(
    scene_dir: Path,
    out_dir: Path,
    shader: str = "ct_sh",
    cfg_overrides: Optional[dict] = None,
    fov_deg: float = 60.0,
    cam_dist: float = 2.0,
    no_shadow: bool = False,
    log_gradients: bool = False,
    device: str = "cuda",
    wandb_entity: str = _WANDB_ENTITY,
    wandb_project: str = _WANDB_PROJECT,
) -> dict:
    """Run CT intrinsic decomposition on a 3D-Front scene.

    Uses the pre-rendered light_*.png images as input observations and
    estimates albedo + per-image SH or env-map lighting.  GT metallic and
    roughness are fixed at the provided values (not optimized).

    Saves results in the same format as run_decomposition() so that
    show_results() and plotting helpers work unchanged.

    Returns the metrics dict.
    """
    scene_dir = Path(scene_dir)
    out_dir   = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = {**DEFAULT_CFG, **(cfg_overrides or {})}

    scene = load_scene(scene_dir, no_shadow=no_shadow)
    H, W = scene["H"], scene["W"]
    images     = scene["images"]
    light_keys = scene["light_keys"]
    mask_np    = scene["mask_np"]

    normals_hw, frag_pos_hw, mask_hw, cam_pos = make_proxy_geometry(
        scene["normals_np"], mask_np,
        fov_deg=fov_deg, cam_dist=cam_dist, device=device,
    )

    gt_metallic  = scene["metallic_np"]   # (H, W, 1)
    gt_roughness = scene["roughness_np"]  # (H, W, 1)
    gt_albedo    = scene["albedo_np"]     # (H, W, 3)

    grad_log_dir = out_dir / "gradient_flow" if log_gradients else None

    # ── build env-map sampling grid (needed for ct_env) ───────────────────────
    _sh_ref  = SHLighting.directional(
        np.array([0, 0, 1], dtype=np.float32), LIGHT_COLOR, intensity=LIGHT_INTENSITY
    )
    _env_ref = EnvMap.from_sh(_sh_ref)
    env_dirs, env_dw = _env_ref._dirs, _env_ref._solid_angles
    env_H, env_W     = _env_ref.image.shape[:2]

    # ── wandb run ─────────────────────────────────────────────────────────────
    scene_name = Path(scene_dir).parent.name + "/" + Path(scene_dir).name
    _SKIP = frozenset({
        "n_iter", "lbfgs_max_iter", "log_every", "sbatch",
        "lr", "lr_end", "lr_schedule", "lr_schedule_step", "lr_schedule_gamma",
        "loss", "optimizer",
        "shininess_min", "shininess_max",
    })
    def _fmt(v):
        return f"{v:g}" if isinstance(v, float) else str(v)
    override_tags = "_".join(
        f"{k}={_fmt(v)}"
        for k, v in (cfg_overrides or {}).items()
        if k not in _SKIP and v != DEFAULT_CFG.get(k)
    )
    run_name = (f"{scene_name}_{shader}"
                + ("_noshadow" if no_shadow else "")
                + (f"_{override_tags}" if override_tags else ""))
    run = wandb.init(
        entity  =wandb_entity,
        project =wandb_project,
        config  =dict(**cfg, shader=shader, scene=str(scene_dir),
                      fov_deg=fov_deg, cam_dist=cam_dist, no_shadow=no_shadow,
                      n_images=len(images), H=H, W=W),
        name    =run_name,
        reinit  =True,
    )
    run.log({
        "gt_images":    [wandb.Image(img) for img in images],
        "gt_albedo":    wandb.Image(gt_albedo),
        "gt_metallic":  wandb.Image(gt_metallic[:, :, 0]),
        "gt_roughness": wandb.Image(gt_roughness[:, :, 0]),
    }, step=0)

    # ── optimize ──────────────────────────────────────────────────────────────
    t0 = time.time()
    if shader == "ct_sh":
        albedo, sh_out, mat_a, mat_b, shadings, history, elapsed = _optimize_ct_sh(
            images, normals_hw, frag_pos_hw, mask_hw, cam_pos,
            gt_metallic, gt_roughness, cfg,
            wandb_run=run,
            gt_albedo=gt_albedo,
            log_gradients=log_gradients,
            grad_log_dir=grad_log_dir,
        )
    elif shader == "ct_env":
        albedo, env_maps_out, mat_a, mat_b, shadings, history, elapsed = _optimize_ct_env(
            images, normals_hw, frag_pos_hw, mask_hw, cam_pos,
            gt_metallic, gt_roughness,
            env_dirs, env_dw, cfg,
            wandb_run=run,
            gt_albedo=gt_albedo,
            log_gradients=log_gradients,
            grad_log_dir=grad_log_dir,
        )
    else:
        raise ValueError(f"shader must be 'ct_sh' or 'ct_env', got {shader!r}")

    # ── albedo RMSE + scale ────────────────────────────────────────────────────
    mask_flat  = mask_np.reshape(-1)
    est_px  = torch.from_numpy(albedo[mask_np])
    gt_px   = torch.from_numpy(gt_albedo[mask_np])
    rmse_t, scale_t = _albedo_rmse(est_px, gt_px)
    rmse    = float(rmse_t)
    scale   = scale_t.numpy()

    inv_scale = 1.0 / np.maximum(scale[None, None, :], 1e-8)
    if shader == "ct_sh":
        sh_out_rescaled   = sh_out * inv_scale
        env_maps_rescaled = np.empty(0)
    else:
        sh_out_rescaled   = np.empty(0)
        env_maps_rescaled = env_maps_out * inv_scale

    albedo_scaled = (albedo * scale).clip(0, 1)
    albedo_err    = np.abs(albedo_scaled - gt_albedo) * mask_np[:, :, None]

    mat_a_err = np.abs(mat_a - gt_metallic)
    mat_b_err = np.abs(mat_b - gt_roughness)

    recon_err  = [np.abs(s - img) * mask_np[:, :, None]
                  for s, img in zip(shadings, images)]
    recon_rmse = float(np.mean([e[mask_np].mean() for e in recon_err]))

    metrics = dict(
        albedo_rmse=rmse, final_loss=float(history[-1]),
        recon_rmse=recon_rmse,
        albedo_scale=scale.tolist(), loss_history=history,
        metallic_est_mean=float(mat_a[mask_np].mean()),
        metallic_gt=float(gt_metallic[mask_np].mean()),
        metallic_err_mean=float(mat_a_err[mask_np].mean()),
        roughness_est_mean=float(mat_b[mask_np].mean()),
        roughness_gt=float(gt_roughness[mask_np].mean()),
        roughness_err_mean=float(mat_b_err[mask_np].mean()),
        elapsed_s=elapsed,
        shader=shader,
        scene=str(scene_dir),
    )

    # ── save to disk ──────────────────────────────────────────────────────────
    def _save_gray(arr_hw1, path):
        Image.fromarray(
            (arr_hw1.squeeze(-1).clip(0, 1) * 255).astype(np.uint8)
        ).save(path)

    recon_dir = out_dir / "reconstructions"
    recon_dir.mkdir(exist_ok=True)

    # GT maps (so show_results can find them in the expected location)
    gt_dir = out_dir / "gt"
    gt_dir.mkdir(exist_ok=True)
    np.save(gt_dir / "albedo.npy",    gt_albedo.astype(np.float32))
    np.save(gt_dir / "metallic.npy",  gt_metallic.astype(np.float32))
    np.save(gt_dir / "roughness.npy", gt_roughness.astype(np.float32))

    Image.fromarray((albedo.clip(0, 1) * 255).astype(np.uint8)).save(
        out_dir / "albedo_est.png")
    np.save(out_dir / "albedo_est.npy", albedo.astype(np.float32))
    Image.fromarray((albedo_scaled * 255).astype(np.uint8)).save(
        out_dir / "albedo_scaled.png")
    np.save(out_dir / "albedo_scaled.npy", albedo_scaled.astype(np.float32))

    _save_gray(mat_a, out_dir / "metallic_est.png")
    np.save(out_dir / "metallic_est.npy", mat_a.astype(np.float32))
    _save_gray(mat_b, out_dir / "roughness_est.png")
    np.save(out_dir / "roughness_est.npy", mat_b.astype(np.float32))

    _save_gray(mat_a_err * mask_np[:, :, None], out_dir / "metallic_err.png")
    np.save(out_dir / "metallic_err.npy", (mat_a_err * mask_np[:, :, None]).astype(np.float32))
    _save_gray(mat_b_err * mask_np[:, :, None], out_dir / "roughness_err.png")
    np.save(out_dir / "roughness_err.npy", (mat_b_err * mask_np[:, :, None]).astype(np.float32))

    for k, (s, e, lk) in enumerate(zip(shadings, recon_err, light_keys)):
        Image.fromarray((s.clip(0, 1) * 255).astype(np.uint8)).save(
            recon_dir / f"recon_{lk}.png")
        np.save(recon_dir / f"recon_{lk}.npy", s.astype(np.float32))
        Image.fromarray((e.mean(-1) * 255).clip(0, 255).astype(np.uint8)).save(
            recon_dir / f"recon_err_{lk}.png")

    if shader == "ct_sh":
        np.save(out_dir / "sh_coeffs_est.npy", sh_out_rescaled)
        sh_env_imgs = []
        for k, sh_k in enumerate(sh_out_rescaled):
            sh_env_img = _sh_coeffs_to_env_img(sh_k)
            sh_env_imgs.append(sh_env_img)
            Image.fromarray((sh_env_img * 255).astype(np.uint8)).save(
                out_dir / f"sh_env_map_{light_keys[k]}.png")
            np.save(out_dir / f"sh_env_map_{light_keys[k]}.npy", sh_env_img.astype(np.float32))
        light_img_key   = "est_sh_env_maps"
        light_imgs_wandb = [wandb.Image(e) for e in sh_env_imgs]
    else:
        np.save(out_dir / "env_maps_est.npy", env_maps_rescaled)
        env_imgs = []
        for k, env_k in enumerate(env_maps_rescaled):
            env_img = _env_flat_to_img(env_k, env_H, env_W)
            env_imgs.append(env_img)
            Image.fromarray((env_img * 255).astype(np.uint8)).save(
                out_dir / f"env_map_{light_keys[k]}.png")
            np.save(out_dir / f"env_map_{light_keys[k]}.npy", env_img.astype(np.float32))
        light_img_key   = "est_env_maps"
        light_imgs_wandb = [wandb.Image(e) for e in env_imgs]

    # ── wandb final summary ────────────────────────────────────────────────────
    run.log({
        "albedo_est":      wandb.Image(albedo.clip(0, 1)),
        "albedo_scaled":   wandb.Image(albedo_scaled),
        "albedo_err":      wandb.Image(albedo_err.mean(-1)),
        "metallic_est":    wandb.Image(mat_a.squeeze(-1)),
        "roughness_est":   wandb.Image(mat_b.squeeze(-1)),
        "metallic_err":    wandb.Image(mat_a_err.squeeze(-1) * mask_np),
        "roughness_err":   wandb.Image(mat_b_err.squeeze(-1) * mask_np),
        "reconstructions": [wandb.Image(s.clip(0, 1)) for s in shadings],
        "recon_errors":    [wandb.Image(e.mean(-1)) for e in recon_err],
        light_img_key:     light_imgs_wandb,
        "albedo_rmse":     rmse,
        "recon_rmse":      recon_rmse,
        "final_loss":      history[-1],
        "elapsed_s":       elapsed,
    }, step=cfg["n_iter"])
    run.finish()

    with open(out_dir / "metrics.json", "w") as fh:
        json.dump(metrics, fh, indent=2)

    print(
        f"  {elapsed:.1f}s  albedo RMSE={rmse:.4f}"
        f"  metallic={metrics['metallic_est_mean']:.3f}(GT={metrics['metallic_gt']:.3f})"
        f"  roughness={metrics['roughness_est_mean']:.3f}(GT={metrics['roughness_gt']:.3f})"
        f"  -> {out_dir}"
    )
    return metrics


# ─────────────────────────────────────── CLI ────────────────────────────────


def _build_parser():
    p = argparse.ArgumentParser(description="CT decomp/render for 3D-Front scenes")
    sub = p.add_subparsers(dest="cmd", required=True)

    # render sub-command
    r = sub.add_parser("render", help="Render 3D-Front scene from GT maps + directional SH")
    r.add_argument("--scene",  required=True)
    r.add_argument("--out",    required=True)
    r.add_argument("--az",     type=float, default=45.0, help="azimuth degrees (0=frontal)")
    r.add_argument("--el",     type=float, default=0.0,  help="elevation degrees")
    r.add_argument("--intensity", type=float, default=LIGHT_INTENSITY)
    r.add_argument("--shader", choices=["ct_sh", "ct_env"], default="ct_sh")
    r.add_argument("--fov",    type=float, default=60.0)
    r.add_argument("--device", default="cuda")

    # dataset sub-command
    ds = sub.add_parser("dataset", help="Render 3D-Front GT with random SH lighting dataset")
    ds.add_argument("--scene",    required=True)
    ds.add_argument("--out",      required=True)
    ds.add_argument("--n_lights", type=int,   default=16)
    ds.add_argument("--seed",     type=int,   default=0)
    ds.add_argument("--fov",      type=float, default=60.0)
    ds.add_argument("--device",   default="cuda")

    # decompose sub-command
    d = sub.add_parser("decompose", help="Decompose 3D-Front scene: estimate albedo + lighting")
    d.add_argument("--scene",      required=True)
    d.add_argument("--out",        required=True)
    d.add_argument("--shader",     choices=["ct_sh", "ct_env"], default="ct_sh")
    d.add_argument("--n_iter",     type=int,   default=100)
    d.add_argument("--lambda_tv",  type=float, default=0.0)
    d.add_argument("--fov",        type=float, default=60.0)
    d.add_argument("--no_shadow",  action="store_true")
    d.add_argument("--log_grads",  action="store_true")
    d.add_argument("--device",     default="cuda")
    return p


def main():
    args = _build_parser().parse_args()
    if args.cmd == "render":
        render_scene(
            Path(args.scene), Path(args.out),
            az_deg=args.az, el_deg=args.el, intensity=args.intensity,
            shader=args.shader, fov_deg=args.fov, device=args.device,
        )
    elif args.cmd == "dataset":
        render_3dfront_dataset(
            Path(args.scene), Path(args.out),
            n_lights=args.n_lights, seed=args.seed,
            fov_deg=args.fov, device=args.device,
        )
    else:
        cfg = {"n_iter": args.n_iter, "lambda_tv": args.lambda_tv}
        decompose_scene(
            Path(args.scene), Path(args.out),
            shader=args.shader, cfg_overrides=cfg,
            fov_deg=args.fov, no_shadow=args.no_shadow,
            log_gradients=args.log_grads, device=args.device,
        )


if __name__ == "__main__":
    main()
