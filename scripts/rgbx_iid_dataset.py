#!/usr/bin/env python
"""
scripts/rgbx_iid_dataset.py — run RGB->X (Zeng et al. 2024, zheng95z/rgb-to-x) intrinsic
decomposition on one image per scene of a dataset and save the predicted channels into an `rgbx/`
subfolder next to that image. The RGB->X analogue of scripts/marigold_iid_dataset.py.

MUST RUN IN THE `rgbx` CONDA ENV (it imports the rgb2x pipeline + its deps):
    conda run -n rgbx python scripts/rgbx_iid_dataset.py [...]
    # or: /home/felix/miniconda3/envs/rgbx/bin/python scripts/rgbx_iid_dataset.py [...]

RGB->X predicts albedo / normal / roughness / metallic / irradiance (one AOV per diffusion pass).
Default saves the material intrinsics (albedo, roughness, metallic) to match the Marigold script;
add normal/irradiance via --aovs. Input space matches the rgb2x demo: light_NNN.npy is LINEAR HDR
-> tonemapped exactly like rgb2x's load_exr_image (Yxy key-0.18 exposure + clamp); a .png is read
as sRGB and converted to linear (load_ldr_image, **2.2). The image is resized to multiples of 8
(max side 1000) for the net, then predictions are resized back to the original resolution.

Works BOTH with a per-scene variant subdir (sh3: <scene>/ct-ct_sh-frOn_env/light_000) and without
one (light directly in the scene dir, e.g. canonical/INFINITE/<scene>/light_000).

Output per scene -> <image_dir>/rgbx/{albedo,roughness,metallic}.{png,npy}
albedo.npy (H,W,3) in [0,1]; roughness/metallic.npy (H,W,1) in [0,1].

Usage:
    conda run -n rgbx python scripts/rgbx_iid_dataset.py                       # sh3 default
    conda run -n rgbx python scripts/rgbx_iid_dataset.py --root local_datasets/canonical/INFINITE
    conda run -n rgbx python scripts/rgbx_iid_dataset.py --scenes 1c349305_v0 --aovs albedo roughness metallic normal
"""
import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import numpy as np
import torch
import torchvision
from PIL import Image

_REPO_ROOT = Path(__file__).resolve().parents[1]
_RGBX_DIR = Path("/home/felix/Projects/rgbx/rgb2x")
sys.path.insert(0, str(_RGBX_DIR))

DEFAULT_ROOT = _REPO_ROOT / "local_datasets" / "26-08-09-datasets" / "sh3"
DEFAULT_VARIANT = "ct-ct_sh-frOn_env"
DEFAULT_LIGHT = "light_000"
DEFAULT_AOVS = ["albedo", "roughness", "metallic"]           # + normal / irradiance optionally
MAX_SIDE = 1000

_PROMPTS = {
    "albedo": "Albedo (diffuse basecolor)",
    "normal": "Camera-space Normal",
    "roughness": "Roughness",
    "metallic": "Metallicness",
    "irradiance": "Irradiance (diffuse lighting)",
}
_GRAY = {"roughness", "metallic"}                            # saved as (H,W,1)


def tonemap_linear(rgb_np):
    """Linear HDR (H,W,3) -> CHW tensor in [0,1], VERBATIM rgb2x load_exr_image(tonemapping,clamp)."""
    from load_image import convert_rgb_2_Yxy, convert_Yxy_2_rgb
    image = torch.from_numpy(np.ascontiguousarray(rgb_np[..., :3].astype("float32")))
    image[~torch.isfinite(image)] = 0
    Yxy = convert_rgb_2_Yxy(image)
    lum = image[:, :, 0:1] * 0.2125 + image[:, :, 1:2] * 0.7154 + image[:, :, 2:3] * 0.0721
    lum = torch.log(torch.clamp(lum, min=1e-6))
    lum_mean = torch.exp(torch.mean(lum))
    lp = Yxy[:, :, 0:1] * 0.18 / torch.clamp(lum_mean, min=1e-6)
    Yxy[:, :, 0:1] = lp
    image = convert_Yxy_2_rgb(Yxy)
    return torch.clamp(image, 0.0, 1.0).permute(2, 0, 1)     # (C,H,W)


def load_photo(image_dir: Path, light: str):
    """light_NNN.npy (linear -> tonemapped) or .png/.jpg (sRGB -> linear); CHW tensor, or None."""
    from load_image import load_ldr_image
    npy = image_dir / f"{light}.npy"
    for ext in (".png", ".jpg", ".jpeg"):
        png = image_dir / f"{light}{ext}"
        if npy.exists():
            break
        if png.exists():
            return load_ldr_image(str(png), from_srgb=True)
    if npy.exists():
        return tonemap_linear(np.load(npy))
    return None


def _resize_for_net(photo, max_side, allow_upscale):
    """Resize the input to a net-friendly size (both dims multiples of 8, >=8).

    By default we DO NOT upscale: RGB->X collapses to flat maps when a small image is blown up to a
    large size (an 8x upscale of a 124px render -> near-flat albedo; verified). We only downscale
    when the longest side exceeds `max_side`. Pass allow_upscale=True to restore the demo's behavior
    (longest side forced to max_side), which only helps already-detailed large photos."""
    _, h, w = photo.shape
    long = max(h, w)
    scale = 1.0
    if long > max_side or (allow_upscale and long < max_side):
        scale = max_side / long
    nh = max(8, int(round(h * scale)) // 8 * 8)
    nw = max(8, int(round(w * scale)) // 8 * 8)
    if (nh, nw) == (h, w):
        return photo, (h, w), (h, w)
    return torchvision.transforms.Resize((nh, nw))(photo), (h, w), (nh, nw)


def process_scene(pipe, scene, image_dir: Path, light, aovs, steps, seed, overwrite, device,
                  max_side, allow_upscale):
    out_dir = image_dir / "rgbx"
    if all((out_dir / f"{a}.npy").exists() for a in aovs) and not overwrite:
        return f"skip {scene}: rgbx/ has all requested AOVs"
    photo = load_photo(image_dir, light)
    if photo is None:
        return f"skip {scene}: no {light}.npy/.png"
    photo = photo.to(device)
    photo, (old_h, old_w), (nh, nw) = _resize_for_net(photo, max_side, allow_upscale)
    gen = torch.Generator(device=device).manual_seed(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    back = torchvision.transforms.Resize((old_h, old_w))
    for aov in aovs:
        img = pipe(prompt=_PROMPTS[aov], photo=photo, num_inference_steps=steps,
                   height=nh, width=nw, generator=gen, required_aovs=[aov]).images[0][0]
        img = back(img)                                      # PIL, back to native resolution
        img.save(out_dir / f"{aov}.png")
        arr = np.asarray(img).astype(np.float32) / 255.0
        if aov in _GRAY:
            arr = arr[..., :1] if arr.ndim == 3 else arr[..., None]
        np.save(out_dir / f"{aov}.npy", arr)
    return f"{scene}: {'+'.join(aovs)} {old_h}x{old_w} -> {out_dir}"


def discover(root: Path, variant: str, light: str, scenes):
    """(scene_name, image_dir): image_dir = <scene>/<variant> if that subdir exists, else the
    scene dir itself (variant-less datasets like canonical/INFINITE)."""
    out = []
    for sd in sorted(p for p in root.iterdir() if p.is_dir()):
        if scenes and sd.name not in scenes:
            continue
        vd = sd / variant if variant and (sd / variant).is_dir() else sd
        if (vd / f"{light}.npy").exists() or any((vd / f"{light}{e}").exists()
                                                 for e in (".png", ".jpg", ".jpeg")):
            out.append((sd.name, vd))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--variant", default=DEFAULT_VARIANT,
                    help="per-scene subdir holding the light image; falls back to the scene dir if "
                         "absent (pass '' for variant-less datasets like canonical/INFINITE)")
    ap.add_argument("--light", default=DEFAULT_LIGHT)
    ap.add_argument("--scenes", nargs="+", default=None, help="scene-name filter")
    ap.add_argument("--aovs", nargs="+", default=DEFAULT_AOVS, choices=list(_PROMPTS))
    ap.add_argument("--steps", type=int, default=50, help="diffusion steps (demo default 50)")
    ap.add_argument("--max-side", type=int, default=MAX_SIDE,
                    help="downscale so the longest side <= this (mult of 8); never upscales by default")
    ap.add_argument("--upscale", action="store_true",
                    help="restore the demo's behavior: force longest side to --max-side even when "
                         "upscaling (flattens small/synthetic inputs — NOT recommended)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    scene_dirs = discover(args.root, args.variant, args.light, set(args.scenes) if args.scenes else None)
    if not scene_dirs:
        raise SystemExit(f"no scenes with {args.light}.npy/.png under {args.root} "
                         f"(variant='{args.variant}' subdir or scene dir directly)")
    print(f"root={args.root}  variant={args.variant or '(none)'}  light={args.light}  "
          f"scenes={len(scene_dirs)}\n  aovs={args.aovs}  steps={args.steps} max_side={args.max_side} "
          f"upscale={args.upscale} device={args.device}\n")

    from diffusers import DDIMScheduler
    from pipeline_rgb2x import StableDiffusionAOVMatEstPipeline
    dtype = torch.float16 if str(args.device).startswith("cuda") else torch.float32
    pipe = StableDiffusionAOVMatEstPipeline.from_pretrained(
        "zheng95z/rgb-to-x", torch_dtype=dtype,
        cache_dir=str(_RGBX_DIR / "model_cache")).to(args.device)
    pipe.scheduler = DDIMScheduler.from_config(
        pipe.scheduler.config, rescale_betas_zero_snr=True, timestep_spacing="trailing")
    pipe.set_progress_bar_config(disable=True)

    for i, (scene, vd) in enumerate(scene_dirs, 1):
        msg = process_scene(pipe, scene, vd, args.light, args.aovs, args.steps, args.seed,
                            args.overwrite, args.device, args.max_side, args.upscale)
        print(f"[{i}/{len(scene_dirs)}] {msg}", flush=True)
    print("\nDONE -> rgbx/ under each scene's image dir")


if __name__ == "__main__":
    main()
