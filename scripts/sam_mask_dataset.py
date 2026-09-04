#!/usr/bin/env python3
"""Run the SAM2 auto-mask ComfyUI workflow over a dataset's scenes.

For a dataset path (e.g. local_datasets/canonical/INFINITE), find each scene's
`light_000` frame (PNG or, for the 'exr' variant, the linear float32 NPY), push
it through the SAM2 automatic-mask-generator graph on the local ComfyUI server,
and write the coloured segmentation back into the scene folder as segmentation.png.

Flat API-format equivalent of ComfyUI/user/default/workflows/SAM3.json
(DownloadAndLoadSAM2Model -> LoadImage -> Sam2AutoSegmentation -> segmented_image).

    /home/felix/myenv/bin/python scripts/sam_mask_dataset.py \
        local_datasets/canonical/INFINITE
    /home/felix/myenv/bin/python scripts/sam_mask_dataset.py <dataset> --light light_003 --overwrite

Nested layout (3D-Front / sh3: <root>/<scene>/<variant>/): generate the mask from ONE variant's
frame and copy segmentation.png into every sibling variant of that scene:
    /home/felix/myenv/bin/python scripts/sam_mask_dataset.py \
        local_datasets/26-08-09-datasets/sh3 --source-variant ct-ct_sh-frOn_env
"""
from __future__ import annotations

import argparse
import io
import sys
import uuid
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from idr.data.scene_io import linear_to_srgb  # noqa: E402
# reuse the ComfyUI HTTP helpers from the relight batch runner
from scripts.comfy_relight_batch import (  # noqa: E402
    upload_image, queue_prompt, wait_for, fetch_outputs,
)
import requests  # noqa: E402

# SAM2 settings copied verbatim from SAM3.json (node 150 + node 151 widgets)
SAM2_MODEL = "sam2.1_hiera_large.safetensors"
SEGMENTOR = "automaskgenerator"
DEVICE = "cuda"
PRECISION = "fp16"
AUTOSEG = dict(  # Sam2AutoSegmentation widget values, in schema order
    points_per_side=10, points_per_batch=160, pred_iou_thresh=0.9,
    stability_score_thresh=0.95, stability_score_offset=1.0, mask_threshold=0.0,
    crop_n_layers=0, box_nms_thresh=0.6, crop_nms_thresh=0.4,
    crop_overlap_ratio=0.34, crop_n_points_downscale_factor=1,
    min_mask_region_area=1.0, use_m2m=True, keep_model_loaded=True,
)


def light_to_png_bytes(path: Path) -> bytes:
    """PNG bytes for upload. PNG source used as-is; NPY treated as linear -> sRGB."""
    if path.suffix.lower() == ".npy":
        a = np.load(path)
        if a.ndim == 2:
            a = a[..., None].repeat(3, -1)
        a = a[..., :3].astype(np.float32)
        # exr-variant frames are linear radiance; may exceed 1.0 (HDR) -> clip first
        arr = (np.clip(linear_to_srgb(np.clip(a, 0.0, 1.0)), 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
        img = Image.fromarray(arr)
    else:
        img = Image.open(path).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def build_graph(image_name: str, filename_prefix: str) -> dict:
    return {
        "150": {"class_type": "DownloadAndLoadSAM2Model",
                "inputs": {"model": SAM2_MODEL, "segmentor": SEGMENTOR,
                           "device": DEVICE, "precision": PRECISION}},
        "152": {"class_type": "LoadImage", "inputs": {"image": image_name}},
        "151": {"class_type": "Sam2AutoSegmentation",
                "inputs": {"sam2_model": ["150", 0], "image": ["152", 0], **AUTOSEG}},
        # segmented_image is output slot 1 (0=mask, 1=segmented_image, 2=bbox)
        "154": {"class_type": "SaveImage",
                "inputs": {"images": ["151", 1], "filename_prefix": filename_prefix}},
    }


def _light_in(d: Path, light: str) -> Path | None:
    for ext in (".png", ".npy"):
        p = d / f"{light}{ext}"
        if p.exists():
            return p
    return None


def find_scenes(root: Path, light: str) -> list[tuple[Path, Path]]:
    """(scene_dir, light_file) for every dir under root holding <light>.png/.npy."""
    found: dict[Path, Path] = {}
    for ext in (".png", ".npy"):
        for p in sorted(root.rglob(f"{light}{ext}")):
            found.setdefault(p.parent, p)  # prefer .png (globbed first)
    return sorted(found.items())


def find_nested_scenes(root: Path, source_variant: str, light: str) -> list[tuple[Path, Path]]:
    """Nested <root>/<scene>/<variant>/ layout: return (scene_dir, source_light_file) using the
    <source_variant>'s frame. The mask is generated once per scene and copied to every variant."""
    out = []
    for scene_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        lf = _light_in(scene_dir / source_variant, light)
        if lf is not None:
            out.append((scene_dir, lf))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset", type=Path,
                    help="dataset root, e.g. local_datasets/canonical/INFINITE")
    ap.add_argument("--light", default="light_000",
                    help="frame stem to segment per scene (default light_000)")
    ap.add_argument("--source-variant", default=None,
                    help="nested <root>/<scene>/<variant>/ layout (e.g. sh3): run SAM on THIS "
                         "variant's frame per scene and copy segmentation.png to ALL sibling "
                         "variants. e.g. --source-variant ct-ct_sh-frOn_env")
    ap.add_argument("--out-name", default="segmentation.png",
                    help="filename written into each scene dir")
    ap.add_argument("--url", default="http://127.0.0.1:8188")
    ap.add_argument("--limit", type=int, default=0, help="max scenes (0 = all)")
    ap.add_argument("--overwrite", action="store_true",
                    help="regenerate even if the segmentation already exists")
    args = ap.parse_args()

    base = args.url.rstrip("/")
    root = args.dataset
    if not root.is_dir():
        sys.exit(f"not a directory: {root}")
    try:
        requests.get(f"{base}/system_stats", timeout=5).raise_for_status()
    except Exception as e:
        sys.exit(f"ComfyUI not reachable at {base}: {e}")

    nested = args.source_variant is not None
    scenes = (find_nested_scenes(root, args.source_variant, args.light) if nested
              else find_scenes(root, args.light))
    if args.limit:
        scenes = scenes[:args.limit]
    if not scenes:
        where = f"<scene>/{args.source_variant}/" if nested else ""
        sys.exit(f"no '{args.light}.(png|npy)' found under {root}/{where}")

    client_id = uuid.uuid4().hex
    print(f"{len(scenes)} scenes under {root}"
          f"{' (source variant '+args.source_variant+', copied to siblings)' if nested else ''} "
          f"| server {base}\n")
    ok = fail = skip = 0
    for i, (scene_dir, light_file) in enumerate(scenes, 1):
        name = scene_dir.name
        # where the mask lands + where to copy it: the source dir, plus every sibling variant dir
        src_dir = light_file.parent
        targets = ([d for d in sorted(scene_dir.iterdir()) if d.is_dir()] if nested else [scene_dir])
        tag = f"[{i}/{len(scenes)}] {name}"
        exists_at = src_dir / args.out_name
        if exists_at.exists() and not args.overwrite:
            # already generated — just back-fill any variants missing the copy
            missing = [d for d in targets if not (d / args.out_name).exists()]
            for d in missing:
                (d / args.out_name).write_bytes(exists_at.read_bytes())
            print(f"{tag} -> skip (exists){f', copied to {len(missing)} variant(s)' if missing else ''}")
            skip += 1; continue
        try:
            png = light_to_png_bytes(light_file)
            uploaded = upload_image(base, png, f"sam_src_{name}.png")
            pid = queue_prompt(base, build_graph(uploaded, f"sam_seg/{name}"), client_id)
            imgs = fetch_outputs(base, wait_for(base, pid))
        except Exception as e:
            print(f"{tag} -> FAILED: {e}"); fail += 1; continue
        if not imgs:
            print(f"{tag} -> no image returned"); fail += 1; continue
        for d in targets:
            (d / args.out_name).write_bytes(imgs[0])
        print(f"{tag} -> {args.out_name} in {len(targets)} dir(s) ({light_file.suffix})")
        ok += 1

    print(f"\ndone. {ok} written, {skip} skipped, {fail} failed.")


if __name__ == "__main__":
    main()
