#!/usr/bin/env python
"""
scripts/build_generated_dataset.py — assemble canonical scene dirs from the GENERATED relighting
datasets (ComfyUI relights of a source image), pairing them with the domain's canonical GT maps.

Inputs (per DOMAIN in gen_relit_sorted/, e.g. INFINITE, MIT):
  local_datasets/gen_relit_sorted/<DOMAIN>/generation_relit_cfg{1,5}/<scene>/<scene>__<Light>.png
  local_datasets/generation_source/[out/]<scene>.{exr|jpg|png}   the source (EXR=linear, else sRGB)
  local_datasets/canonical/<DOMAIN>[-train]/<scene>/  albedo|normals|roughness|metallic (.npy+.png)

Output (per-domain top-level, one leaf per generation-config + scene):
  local_datasets/canonical/<DOMAIN>_GENERATED/<gen_cfg>/<scene>/
    light_000..NNN.png     the relit observations (sRGB; load_scene linearises them; variant=srgb)
    source.npy / source.png   the source image (linear .npy + display .png), named "source"
    albedo|normals|roughness|metallic .npy+.png   GT (copied from the domain's canonical dir)
    lights.json            index -> lighting name;   config.json (domain, gt_root, camera, ...)

INFINITE GT is true GT; MIT GT is Marigold predictions (variant srgb, no GT lighting) — either way
there is NO GT SH for the artistic named lightings, so no sh_*.npy is written. Relits/source are
resized to the canonical GT resolution if they differ. --downsample strides everything alike.

Usage:
    python scripts/build_generated_dataset.py                 # all domains
    python scripts/build_generated_dataset.py --domains MIT --configs cfg1 --downsample 2
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from PIL import Image

from idr.paths import REPO_ROOT
from idr.data.scene_io import load_exr, linear_to_srgb, srgb_to_linear

_LOCAL = REPO_ROOT / "local_datasets"
GEN_SORTED = _LOCAL / "gen_relit_sorted"
SOURCE_ROOT = _LOCAL / "generation_source"
CANONICAL = _LOCAL / "canonical"
_SRC_EXTS = (".exr", ".jpg", ".jpeg", ".png")


def _stride(a, ds):
    return np.ascontiguousarray(a[::ds, ::ds]) if ds > 1 else a


def _resize_to(arr, hw):
    """Resize (H,W[,C]) to hw; uint8 or float32, bilinear per channel."""
    H, W = int(hw[0]), int(hw[1])
    if arr.shape[:2] == (H, W):
        return arr
    if arr.dtype == np.uint8:
        return np.asarray(Image.fromarray(arr).resize((W, H), Image.BILINEAR))
    chs = [np.asarray(Image.fromarray(arr[..., c], mode="F").resize((W, H), Image.BILINEAR))
           for c in range(arr.shape[2])]
    return np.stack(chs, -1).astype(np.float32)


def _lighting_name(png_path, scene):
    stem = png_path.stem
    return stem[len(scene) + 2:] if stem.startswith(scene + "__") else stem


def _gt_root(domain):
    for cand in (CANONICAL / domain, CANONICAL / f"{domain}-train"):
        if cand.is_dir():
            return cand
    return None


def _find_source(scene):
    for base in (SOURCE_ROOT, SOURCE_ROOT / "out"):
        for ext in _SRC_EXTS:
            p = base / f"{scene}{ext}"
            if p.exists():
                return p
    return None


def _load_source_linear(p):
    """Return an (H,W,3) linear float32 image. EXR is already linear; sRGB (jpg/png) is linearised."""
    if p.suffix.lower() == ".exr":
        return load_exr(p)[:, :, :3].astype(np.float32)
    return srgb_to_linear(np.asarray(Image.open(p).convert("RGB"), np.float32) / 255.0)


def build_leaf(domain, gen_cfg, scene, relit_dir, out_dir, ds):
    gt = _gt_root(domain)
    if gt is None or not (gt / scene).is_dir():
        return f"skip {domain}/{scene}: no canonical GT"
    inf = gt / scene
    src_p = _find_source(scene)
    if src_p is None:
        return f"skip {domain}/{scene}: no source image"
    relits = sorted(relit_dir.glob("*.png"))
    if not relits:
        return f"skip {domain}/{scene}: no relit PNGs"

    gt_hw = np.load(inf / "albedo.npy").shape[:2]        # canonical resolution
    shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    # relit observations (sRGB PNG; load_scene linearises via variant=srgb) — resized to GT, strided
    names = []
    for i, p in enumerate(relits):
        arr = _stride(_resize_to(np.asarray(Image.open(p).convert("RGB"), np.uint8), gt_hw), ds)
        Image.fromarray(arr).save(out_dir / f"light_{i:03d}.png")
        names.append({"index": i, "name": _lighting_name(p, scene)})
    (out_dir / "lights.json").write_text(json.dumps(names, indent=1))

    # source (linear .npy + display .png) — resized to GT, strided
    src = _stride(_resize_to(_load_source_linear(src_p), gt_hw), ds).astype(np.float32)
    np.save(out_dir / "source.npy", src)
    Image.fromarray((linear_to_srgb(np.clip(src, 0, 1)) * 255).astype(np.uint8)).save(out_dir / "source.png")

    # GT maps from the canonical dir (strided to match)
    for f in ("albedo", "normals", "roughness", "metallic"):
        a = np.load(inf / f"{f}.npy")
        np.save(out_dir / f"{f}.npy", _stride(a, ds).astype(np.float32))
        pf = inf / f"{f}.png"
        if pf.exists():
            Image.fromarray(_stride(np.asarray(Image.open(pf)), ds)).save(out_dir / f"{f}.png")

    inf_cfg = json.loads((inf / "config.json").read_text())
    (out_dir / "config.json").write_text(json.dumps({
        "variant": "srgb", "n_lights": len(relits), "light_type": "generated", "lighting": "named",
        "prereduced_downsample": ds, "hl_mode": inf_cfg.get("hl_mode", "analytic"),
        "diffuse_fresnel": inf_cfg.get("diffuse_fresnel", True), "camera": inf_cfg.get("camera"),
        "domain": domain, "gen_config": gen_cfg, "src_scene": scene,
        "gt_root": str(inf), "source_img": str(src_p),
        "gt_kind": "marigold" if inf_cfg.get("marigold_intrinsics") else "true",
        "note": "generated relights (sRGB) + canonical GT intrinsics; no GT lighting; 'source' is "
                "the linear relighting-model input"}, indent=1))
    h, w = src.shape[:2]
    return f"{domain}/{gen_cfg}/{scene}: {len(relits)} relit lights + source, {h}x{w} -> {out_dir}"


def discover_domains(root):
    return sorted(p.name for p in root.iterdir() if p.is_dir()) if root.exists() else []


def discover_configs(domain_root):
    """map short gen-cfg name (e.g. 'cfg1') -> relit config dir."""
    out = {}
    for p in sorted(domain_root.iterdir()) if domain_root.exists() else []:
        if p.is_dir() and p.name.startswith("generation_relit_"):
            out[p.name[len("generation_relit_"):]] = p
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--domains", nargs="+", default=None, help="default: all under gen_relit_sorted")
    ap.add_argument("--configs", nargs="+", default=None, help="gen-cfg filter, e.g. cfg1 cfg5")
    ap.add_argument("--scenes", nargs="+", default=None, help="scene-name prefix filter")
    ap.add_argument("--out_canonical", type=Path, default=CANONICAL,
                    help="parent for <DOMAIN>_GENERATED outputs")
    ap.add_argument("--downsample", type=int, default=1)
    args = ap.parse_args()
    domains = args.domains or discover_domains(GEN_SORTED)
    print(f"gen_relit_sorted={GEN_SORTED}\ndomains={domains}  configs={args.configs or 'all'}  "
          f"downsample={args.downsample}\n")
    n = 0
    for domain in domains:
        cfgs = discover_configs(GEN_SORTED / domain)
        for gen_cfg, cfg_dir in cfgs.items():
            if args.configs and gen_cfg not in args.configs:
                continue
            for scene_dir in sorted(p for p in cfg_dir.iterdir() if p.is_dir()):
                scene = scene_dir.name
                if args.scenes and not any(scene.startswith(s) for s in args.scenes):
                    continue
                out_dir = args.out_canonical / f"{domain}_GENERATED" / gen_cfg / scene
                msg = build_leaf(domain, gen_cfg, scene, scene_dir, out_dir, args.downsample)
                print("  " + msg); n += ("->" in msg)
    print(f"\nDONE   ({n} leaf/leaves built under {args.out_canonical}/<DOMAIN>_GENERATED)")


if __name__ == "__main__":
    main()
