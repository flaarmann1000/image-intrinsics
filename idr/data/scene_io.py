"""Reading a scene directory: images, GT intrinsic maps, and colour-space helpers."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

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

    Tries imageio (requires imageio-freeimage or OpenEXR), then the OpenEXR v3 bindings
    (`pip install OpenEXR` — the only EXR reader that ships wheels for very new CPythons,
    where opencv-python is built without the EXR codec), then OpenCV (cv2). Raises only if
    every reader is missing or fails.
    """
    path = Path(path)
    try:
        import imageio
        img = np.asarray(imageio.imread(str(path))).astype(np.float32)
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)
        return img[:, :, :3]
    except Exception:
        pass
    try:
        import OpenEXR  # v3 numpy bindings; channels are grouped, e.g. {"RGB": ...}
        ch = OpenEXR.File(str(path)).channels()
        if "RGB" in ch:
            img = np.asarray(ch["RGB"].pixels, np.float32)
        elif "RGBA" in ch:
            img = np.asarray(ch["RGBA"].pixels, np.float32)[:, :, :3]
        elif all(k in ch for k in ("R", "G", "B")):
            img = np.stack([np.asarray(ch[k].pixels, np.float32) for k in ("R", "G", "B")], -1)
        else:                                            # single/unknown channel -> broadcast
            img = np.asarray(next(iter(ch.values())).pixels, np.float32)
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)
        return img[:, :, :3]
    except Exception:
        pass
    import cv2  # noqa: PLC0415
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED | cv2.IMREAD_ANYDEPTH)
    if img is None:
        raise IOError(f"Could not read EXR (tried imageio, OpenEXR, cv2): {path}")
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    return img[:, :, :3][:, :, ::-1].astype(np.float32)  # BGR→RGB


def load_scene(scene_dir: Path, no_shadow: bool = False, use_npy: bool = False,
               gt_npy: bool = False) -> dict:
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

    gt_npy: if True, load the GT intrinsic maps from the lossless float32
            {normals,albedo,roughness,metallic}.npy copies (when all present)
            instead of the truncated 8/16-bit PNGs; falls back to PNG otherwise.
    """
    scene_dir = Path(scene_dir)

    # Auto-detect variant from config.json
    cfg_path = scene_dir / "config.json"
    variant = "linear"
    if cfg_path.exists():
        with open(cfg_path) as fh:
            _ds_cfg = json.load(fh)
        variant = _ds_cfg.get("variant", "linear")

    # ── GT intrinsic maps ────────────────────────────────────────────────────
    # gt_npy: load the lossless float32 .npy copies written alongside the PNGs
    # (normals/albedo/roughness/metallic .npy) instead of the truncated 8/16-bit
    # PNGs — but only if all four are present, otherwise fall back to PNG.
    _gt_npy_ok = gt_npy and all(
        (scene_dir / f"{m}.npy").exists()
        for m in ("normals", "albedo", "roughness", "metallic")
    )
    if _gt_npy_ok:
        # Normals: float32, background stored as all-zero → mask from norm > 0
        normals = np.load(scene_dir / "normals.npy").astype(np.float32)
        mask = (np.linalg.norm(normals, axis=-1) > 0)  # (H, W) bool
        nlen = np.linalg.norm(normals, axis=-1, keepdims=True).clip(1e-6, None)
        normals = normals / nlen  # (H, W, 3), unit length

        albedo = np.load(scene_dir / "albedo.npy").astype(np.float32)[:, :, :3]

        rough = np.load(scene_dir / "roughness.npy").astype(np.float32)
        rough = rough if rough.ndim == 3 else rough[:, :, None]  # (H, W, 1)

        metal = np.load(scene_dir / "metallic.npy").astype(np.float32)
        metal = metal if metal.ndim == 3 else metal[:, :, None]  # (H, W, 1)
    else:
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

        if use_npy:
            npy_files = [bf.with_suffix(".npy") for bf in base_files]
            if all(f.exists() for f in npy_files):
                images = [np.load(f).astype(np.float32) for f in npy_files]
            else:
                images = [
                    np.array(Image.open(f), dtype=np.float32)[:, :, :3] / 255.0
                    for f in img_files
                ]
                if variant == "srgb":
                    images = [srgb_to_linear(img) for img in images]
        else:
            images = [
                np.array(Image.open(f), dtype=np.float32)[:, :, :3] / 255.0
                for f in img_files
            ]
            if variant == "srgb":
                images = [srgb_to_linear(img) for img in images]
        # Always use the base key (light_NNN) regardless of which file was loaded
        light_keys = [bf.stem for bf in base_files]

    H, W = normals.shape[:2]

    # Load precomputed SH coefficients if present (one .npy per light)
    sh_coeffs_list = []
    for lk in light_keys:
        idx = int(lk.split("_")[-1])
        sh_path = scene_dir / f"sh_{idx:03d}.npy"
        if sh_path.exists():
            sh_coeffs_list.append(np.load(sh_path).astype(np.float32))
    sh_coeffs = sh_coeffs_list if len(sh_coeffs_list) == len(light_keys) else None

    return dict(
        normals_np=normals,
        mask_np=mask,
        albedo_np=albedo,
        metallic_np=metal,
        roughness_np=rough,
        images=images,
        light_keys=light_keys,
        H=H, W=W,
        sh_coeffs=sh_coeffs,
    )
