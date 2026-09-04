#!/usr/bin/env python
"""
scripts/marigold_iid_dataset.py — run Marigold intrinsic-image-decomposition (the APPEARANCE model,
prs-eth/marigold-iid-appearance-v1-1) on one image per scene of a dataset and save the predicted
ALBEDO / ROUGHNESS / METALLIC into a `marigold/` subfolder next to that image. (Normals are not
produced by this model / not wanted here — use estimate_normals / marigold-normals for those.)

Works for datasets WITH a per-scene variant subdir (sh3):
    local_datasets/26-08-09-datasets/sh3/<scene>/ct-ct_sh-frOn_env/light_000.{npy|png}
and WITHOUT one (light stored directly in the scene dir, e.g. canonical/INFINITE):
    local_datasets/canonical/INFINITE/<scene>/light_000.{npy|png}
The image dir is <scene>/<variant> when that subdir exists, else the scene dir itself.
The scene image is `light_000` (the first light). It is stored as .npy (linear HDR) in these datasets,
but a .png (sRGB) is also accepted; either is converted to an sRGB PIL image for Marigold (linear
npy -> linear_to_srgb; png taken as-is).

Marigold-IID appearance predicts two targets: "albedo" (sRGB) and "material" (R=roughness,
G=metallicity — see Marigold's interiorverse_dataset.py). We save, per scene, into
    <scene>/<variant>/marigold/{albedo,roughness,metallic}.{npy,png}
albedo.npy is (H,W,3) in [0,1] (sRGB, as predicted); roughness/metallic.npy are (H,W,1) in [0,1].

Usage:
    python scripts/marigold_iid_dataset.py
    python scripts/marigold_iid_dataset.py --scenes 1c349305_v0 --overwrite
    python scripts/marigold_iid_dataset.py --root local_datasets/26-08-09-datasets/sh3 \
        --variant ct-ct_sh-frOn_env --light light_000 --ensemble-size 5 --steps 4
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from PIL import Image

from idr.paths import REPO_ROOT
from idr.data.scene_io import linear_to_srgb

DEFAULT_ROOT = REPO_ROOT / "local_datasets" / "26-08-09-datasets" / "sh3"
DEFAULT_VARIANT = "ct-ct_sh-frOn_env"
DEFAULT_LIGHT = "light_000"
MODEL = "prs-eth/marigold-iid-appearance-v1-1"


def load_srgb_image(variant_dir: Path, light: str) -> Image.Image | None:
    """light_NNN.npy (linear HDR -> sRGB) or .png (sRGB as-is) -> PIL RGB uint8, or None."""
    npy, png = variant_dir / f"{light}.npy", variant_dir / f"{light}.png"
    if npy.exists():
        lin = np.load(npy)[..., :3].astype(np.float32)
        arr = (linear_to_srgb(np.clip(lin, 0, 1)) * 255.0 + 0.5).clip(0, 255).astype(np.uint8)
        return Image.fromarray(arr)
    if png.exists():
        return Image.open(png).convert("RGB")
    return None


def _target(out, pipe, name: str) -> np.ndarray:
    """Pull one named target (H,W,3) float32 in [0,1] from a MarigoldIntrinsicsOutput."""
    names = list(pipe.target_properties["target_names"])
    pred = np.asarray(out.prediction, dtype=np.float32)      # (targets,H,W,3) or (1,targets,H,W,3)
    if pred.ndim == 5:
        pred = pred[0]
    return np.clip(pred[names.index(name)], 0.0, 1.0)


def _save_gray(arr01: np.ndarray, stem: Path):
    np.save(stem.with_suffix(".npy"), arr01[..., None].astype(np.float32))       # (H,W,1)
    Image.fromarray((arr01 * 255.0 + 0.5).clip(0, 255).astype(np.uint8)).save(stem.with_suffix(".png"))


def process_scene(pipe, scene: str, image_dir: Path, light: str, steps: int, ens: int, overwrite: bool) -> str:
    out_dir = image_dir / "marigold"
    if (out_dir / "albedo.npy").exists() and not overwrite:
        return f"skip {scene}: marigold/ exists"
    img = load_srgb_image(image_dir, light)
    if img is None:
        return f"skip {scene}: no {light}.npy/.png"
    out = pipe(img, num_inference_steps=steps, ensemble_size=ens, output_type="np")
    albedo = _target(out, pipe, "albedo")                    # (H,W,3) sRGB
    material = _target(out, pipe, "material")                # R=roughness, G=metallicity
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "albedo.npy", albedo.astype(np.float32))
    Image.fromarray((albedo * 255.0 + 0.5).clip(0, 255).astype(np.uint8)).save(out_dir / "albedo.png")
    _save_gray(material[..., 0], out_dir / "roughness")
    _save_gray(material[..., 1], out_dir / "metallic")
    return f"{scene}: albedo{albedo.shape} + roughness + metallic -> {out_dir}"


def discover(root: Path, variant: str, light: str, scenes):
    """(scene_name, image_dir) per scene. image_dir = <scene>/<variant> when that subdir exists,
    else the scene dir itself (datasets like canonical/INFINITE store light_NNN directly)."""
    out = []
    for sd in sorted(p for p in root.iterdir() if p.is_dir()):
        if scenes and sd.name not in scenes:
            continue
        vd = sd / variant if variant and (sd / variant).is_dir() else sd
        if (vd / f"{light}.npy").exists() or (vd / f"{light}.png").exists():
            out.append((sd.name, vd))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--variant", default=DEFAULT_VARIANT,
                    help="per-scene subdir holding the light image; falls back to the scene dir if "
                         "absent (pass '' for variant-less datasets like canonical/INFINITE)")
    ap.add_argument("--light", default=DEFAULT_LIGHT, help="which light to run Marigold on")
    ap.add_argument("--scenes", nargs="+", default=None, help="scene-name filter")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--ensemble-size", type=int, default=5, help="more = less noisy, slower")
    ap.add_argument("--steps", type=int, default=4, help="diffusion denoising steps")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    scene_dirs = discover(args.root, args.variant, args.light, set(args.scenes) if args.scenes else None)
    if not scene_dirs:
        raise SystemExit(f"no scenes with {args.light}.npy/.png under {args.root} "
                         f"(variant='{args.variant}' subdir or scene dir directly)")
    print(f"root={args.root}  variant={args.variant or '(none)'}  light={args.light}  scenes={len(scene_dirs)}\n"
          f"model={args.model}  ensemble={args.ensemble_size} steps={args.steps} device={args.device}\n")

    from diffusers import MarigoldIntrinsicsPipeline
    dtype = torch.float16 if str(args.device).startswith("cuda") else torch.float32
    kw = dict(torch_dtype=dtype)
    if dtype == torch.float16:
        kw["variant"] = "fp16"
    pipe = MarigoldIntrinsicsPipeline.from_pretrained(args.model, **kw).to(args.device)
    pipe.set_progress_bar_config(disable=True)
    print("targets:", list(pipe.target_properties["target_names"]), "\n")

    for i, (scene, vd) in enumerate(scene_dirs, 1):
        msg = process_scene(pipe, scene, vd, args.light, args.steps, args.ensemble_size, args.overwrite)
        print(f"[{i}/{len(scene_dirs)}] {msg}", flush=True)
    print("\nDONE -> marigold/ under each scene's image dir")


if __name__ == "__main__":
    main()
