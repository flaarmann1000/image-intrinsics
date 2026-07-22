"""Rendering a scene and building a self-contained dataset directory from it."""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image

from idr.paths import DEFAULT_SH_LIGHTS_DIR as _DEFAULT_SH_LIGHTS_DIR
from idr.render import EnvMap, SHLighting, build_sh_basis, shade_ct_sh, shade_ct_env
from idr.render.brdf import _get_ggx_sh_lut
from .scene_io import load_scene, linear_to_srgb, load_exr
from idr.config import LIGHT_COLOR, LIGHT_INTENSITY
from idr.data.synthetic_scene import _make_lights_random_sh
from idr.track.wandb_log import _sh_coeffs_to_env_img
from .geometry import make_proxy_geometry
from .lighting import _make_directional_sh, make_dc_lifted_sh_lighting

def _sorted_lighting_files(src_dir: Path, pattern: str, exclude_suffix: str = "") -> list:
    """Glob render files and sort by numeric index (handles sun_10 > sun_9)."""
    files = [f for f in src_dir.glob(pattern)
             if not (exclude_suffix and f.stem.endswith(exclude_suffix))]
    def _idx(p: Path) -> int:
        stem = p.stem.replace("_srgb", "")
        return int(stem.rsplit("_", 1)[-1])
    return sorted(files, key=_idx)


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
    render_np[flat_mask.cpu().numpy()] = render_m.float().cpu().numpy()
    render_np = render_np.reshape(H, W, 3)

    # Save. PNG is 8-bit: clip to [0,1] (raw render may fall outside, which would
    # overflow-wrap uint8) and round rather than truncate (halves quantization
    # error and removes the ~0.5/255 dark bias). The .npy keeps the raw float.
    Image.fromarray((np.clip(render_np, 0, 1) * 255).round().astype(np.uint8)).save(out_dir / "render.png")
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
    n_lights: Optional[int] = None,
    seed: int = 0,
    fov_deg: float = 60.0,
    cam_dist: float = 2.0,
    device: str = "cuda",
    sh_lights_dir: Optional[Path] = _DEFAULT_SH_LIGHTS_DIR,
    shader: str = "ct_sh",
    normalize_env: bool = True,
) -> None:
    """Render 3D-Front GT material maps under SH lighting conditions.

    n_lights: total number of lights to render.
      - None (default): use all lights found in sh_lights_dir, no random supplement.
      - int: load up to n_lights from sh_lights_dir, then fill the remainder with
        randomly generated SH coefficients (seeds: seed, seed+1, ...).

    normalize_env: if True (default), each light's SH coefficients are scaled so
        that its reconstructed env-map peak equals 1.0 — exactly the per-map
        max-normalisation that ``_sh_coeffs_to_env_img`` applies when writing the
        ``sh_env_map_NNN.png`` files. BlenderProc's ``render_front3d_multipass.py``
        ``env`` condition consumes those normalised PNGs as the world background,
        so this makes the shader's lighting INPUT identical to BlenderProc's.
        Leaving it False lights from the raw (un-normalised) coefficients, which
        is brighter and does NOT match a BlenderProc env render. (Verified on the
        sphere: normalised reproduces the PNG path to RMSE ~1e-3; raw is ~15% off.)

    SH lights source:
      1. sh_lights_dir — directory of precomputed .npy SH coefficient files,
         used in sorted order (pass None to skip).
      2. Random SH generation (_make_lights_random_sh) with sequential seeds,
         used only when n_lights exceeds the number of precomputed files.

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
    out_dir   = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load precomputed lights
    precomputed: list[np.ndarray] = []
    if sh_lights_dir is not None and Path(sh_lights_dir).is_dir():
        npy_files = sorted(Path(sh_lights_dir).glob("*.npy"))
        cap = n_lights if n_lights is not None else len(npy_files)
        npy_files = npy_files[:cap]
        if not npy_files:
            raise FileNotFoundError(f"No .npy files found in {sh_lights_dir}")
        precomputed = [np.load(p).astype(np.float32) for p in npy_files]
        print(f"  Loaded {len(precomputed)} precomputed SH lights from {sh_lights_dir}")

    # Supplement with random lights if a target was given and not yet reached
    n_random = max(0, n_lights - len(precomputed)) if n_lights is not None else 0
    random_lights: list[np.ndarray] = [
        _make_lights_random_sh(seed=seed + i)[3]
        for i in range(n_random)
    ]
    if n_random:
        print(f"  Generating {n_random} random SH lights (seed {seed}–{seed + n_random - 1})")

    sh_coeffs_list = precomputed + random_lights
    if not sh_coeffs_list:
        raise ValueError("No lights available: sh_lights_dir is empty or None and n_lights is None.")
    n_lights = len(sh_coeffs_list)

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

    # Pre-build env map structure once (same topology for all lights)
    env_dirs_t = env_dw_t = None
    if shader == "ct_env":
        _env_proto = EnvMap.from_sh(SHLighting(np.zeros((9, 3), dtype=np.float32)))
        env_dirs_t = torch.from_numpy(_env_proto._dirs).to(device)
        env_dw_t   = torch.from_numpy(_env_proto._solid_angles).to(device)

    for i, coeffs_np in enumerate(sh_coeffs_list):
        # Match the per-map max-normalisation baked into the sh_env_map PNGs that
        # BlenderProc renders from (see _sh_coeffs_to_env_img). Scaling the SH
        # coefficients is equivalent and works for both the ct_sh and ct_env paths.
        if normalize_env:
            env_max = float(EnvMap.from_sh(SHLighting(coeffs_np)).image.max())
            coeffs_np = coeffs_np / max(env_max, 1e-8)
        with torch.no_grad():
            if shader == "ct_env":
                assert env_dirs_t is not None and env_dw_t is not None
                env_pix = EnvMap.from_sh(SHLighting(coeffs_np))._image_flat
                env_pix_t = torch.from_numpy(env_pix.astype(np.float32)).to(device)
                render_m = shade_ct_env(
                    view_m, normals_m, albedo_m,
                    env_pix_t, env_dirs_t, env_dw_t,
                    metallic=metal_m, roughness=rough_m,
                )
            else:
                sh_coeffs_t = torch.from_numpy(coeffs_np).to(device)  # (9, 3)
                render_m = shade_ct_sh(
                    view_m, normals_m, albedo_m,
                    sh_coeffs_t,
                    metallic=metal_m, roughness=rough_m, lut=lut,
                )

        render_np = np.zeros((H * W, 3), dtype=np.float32)
        render_np[flat_mask.cpu().numpy()] = render_m.float().cpu().numpy()
        render_np = render_np.reshape(H, W, 3)

        # PNG is 8-bit/display only: clip to [0,1] (raw render may fall outside,
        # which would overflow-wrap uint8) and round instead of truncate. The
        # .npy below keeps the raw float values and is the lossless target.
        Image.fromarray((np.clip(render_np, 0, 1) * 255).round().astype(np.uint8)).save(
            out_dir / f"light_{i:03d}.png")
        np.save(out_dir / f"light_{i:03d}.npy", render_np)
        np.save(out_dir / f"sh_{i:03d}.npy", coeffs_np)

        sh_env_img = _sh_coeffs_to_env_img(coeffs_np)
        Image.fromarray((sh_env_img * 255).astype(np.uint8)).save(
            out_dir / f"sh_env_map_{i:03d}.png")

        print(f"  light {i:03d}/{n_lights} done")

    cfg = {
        "scene_dir":    str(scene_dir),
        "n_lights":     n_lights,
        "sh_lights_dir": str(sh_lights_dir) if sh_lights_dir is not None else None,
        "seed":         seed,
        "fov_deg":      fov_deg,
        "cam_dist":     cam_dist,
    }
    with open(out_dir / "config.json", "w") as fh:
        json.dump(cfg, fh, indent=2)

    print(f"3D-Front dataset rendered: {n_lights} images -> {out_dir}")


def build_3dfront_dataset(
    src_dir: Path,
    out_dir: Path,
    variant: str = "linear",
    lighting: str = "sun",
    sh_src_dir: Optional[Path] = None,
    sh_scale: float = 1.0,
) -> None:
    """Build a CT decomposition dataset from 3D-Front renderings.

    Copies GT material maps (PNG) and converts renders to the standard
    light_NNN format expected by load_scene / decompose_scene.

    variant:
      "linear"  — sun_N.png   → light_NNN.png  (uint8 linear-light)
      "srgb"    — sun_N_srgb.png → light_NNN.png  (uint8 sRGB;
                  linearized automatically by load_scene at decomp time)
      "exr"     — sun_N.exr   → light_NNN.npy  (float32, HDR;
                  normalized to 99th-percentile ≈ 1.0)

    sh_src_dir: optional dir with sh_NNN.npy GT SH coefficients (e.g.
      results/ref_sh_lighting for env renders, cycled by index). They are
      copied into the dataset as sh_NNN.npy — required for the val_images /
      relighting metric of decompose_scene — scaled by the EFFECTIVE lighting
      scale of the images: sh_scale (e.g. the BlenderProc env strength 2.0),
      additionally divided by the exr normalization for the "exr" variant.

    The output directory is a valid scene_dir for decompose_scene.
    """
    src_dir = Path(src_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # GT material maps always come from PNG
    for fname in ("albedo.png", "normals.png", "roughness.png", "metallic.png"):
        shutil.copy2(src_dir / fname, out_dir / fname)

    # Collect render files sorted numerically
    if variant == "linear":
        lighting_files = _sorted_lighting_files(src_dir, f"{lighting}_*.png", exclude_suffix="_srgb")
    elif variant == "srgb":
        lighting_files = _sorted_lighting_files(src_dir, f"{lighting}_*_srgb.png")
    elif variant == "exr":
        lighting_files = _sorted_lighting_files(src_dir, f"{lighting}_*.exr")
    else:
        raise ValueError(f"variant must be 'linear', 'srgb', or 'exr'; got {variant!r}")

    exr_scale = None
    if variant == "exr":
        # Load all EXR images, compute a single normalization scale so that
        # the 99th-percentile foreground value maps to ~1.0
        raw_imgs = [load_exr(f) for f in lighting_files]
        all_pos = np.concatenate([img.reshape(-1) for img in raw_imgs])
        all_pos = all_pos[all_pos > 0]
        exr_scale = float(np.percentile(all_pos, 99)) if len(all_pos) else 1.0
        for i, img in enumerate(raw_imgs):
            img_norm = (img / exr_scale).astype(np.float32)
            np.save(out_dir / f"light_{i:03d}.npy", img_norm)            
            Image.fromarray((img_norm.clip(0, 1) * 255).astype(np.uint8)).save(
                out_dir / f"light_{i:03d}_preview.png")
    else:
        for i, f in enumerate(lighting_files):
            shutil.copy2(f, out_dir / f"light_{i:03d}.png")

    if sh_src_dir is not None:
        sh_files = sorted(Path(sh_src_dir).glob("sh_*.npy"))
        if not sh_files:
            raise FileNotFoundError(f"No sh_*.npy files in {sh_src_dir}")
        eff_scale = sh_scale / exr_scale if exr_scale else sh_scale
        for i in range(len(lighting_files)):
            sh = np.load(sh_files[i % len(sh_files)]).astype(np.float32) * eff_scale
            np.save(out_dir / f"sh_{i:03d}.npy", sh)

    cfg: dict = {
        "src_dir": str(src_dir),
        "variant": variant,
        "n_lights": len(lighting_files),
        "light_type": lighting,
    }
    if exr_scale is not None:
        cfg["exr_scale"] = exr_scale
    if sh_src_dir is not None:
        cfg["sh_src_dir"] = str(sh_src_dir)
        cfg["sh_scale"] = sh_scale
        cfg["sh_eff_scale"] = float(sh_scale / exr_scale if exr_scale else sh_scale)

    with open(out_dir / "config.json", "w") as fh:
        json.dump(cfg, fh, indent=2)

    print(f"Built '{variant}' sun dataset: {len(lighting_files)} images -> {out_dir}")
