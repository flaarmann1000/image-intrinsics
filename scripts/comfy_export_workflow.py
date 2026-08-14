#!/usr/bin/env python3
"""Emit a flat, debuggable ComfyUI *UI* workflow for the img2img structure-lock
relighting graph, so it can be opened and poked at in the ComfyUI editor.

Rather than synthesise node JSON from scratch (fragile across frontend versions),
this reuses the real node dicts from the existing working workflow
    ComfyUI/user/default/workflows/relighting.json
by flattening its two nested subgraphs (Image Edit + Reference Conditioning),
then applies the structure-lock surgery:
  * drop EmptyFlux2LatentImage; sample from the source's VAE latent instead
  * insert SplitSigmasDenoise(denoise) between Flux2Scheduler and the sampler
  * turn the promoted boundary inputs (prompt text, seed) into plain widgets
  * end in a PreviewImage so nothing is written to disk while debugging

The chosen source EXR is tonemapped + uploaded to ComfyUI so the LoadImage node
resolves immediately when the graph opens.

    /home/felix/myenv/bin/python scripts/comfy_export_workflow.py \
        --denoise 0.7 --variant warm_sun_left
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts.comfy_relight_batch import (  # noqa: E402
    CFG, STEPS, exr_to_png_bytes, image_to_png_bytes, upload_image,
)

SRC_WORKFLOW = Path.home() / "Projects/ComfyUI/user/default/workflows/relighting.json"
OUT_WORKFLOW = Path.home() / "Projects/ComfyUI/user/default/workflows/relight_structlock.json"

# nodes to lift out of the two subgraphs. EmptyFlux2LatentImage(66) is added only
# in full-regen mode; SplitSigmasDenoise(200) only in img2img mode; the scale node
# ImageScaleToTotalPixels(80) only when an explicit --megapixels target is given.
IMAGE_EDIT_KEEP = [61, 62, 65, 74, 67, 72, 81, 73, 70, 71, 64, 63]
REFCOND_KEEP = [76, 77, 78]

# edges common to both topologies, as (from_id, out_name, to_id, in_name, type).
# slot indices are resolved from each node's real inputs/outputs arrays. The image
# source (84 native, or 80 scaled) is wired in main() so the graph self-configures.
EDGES_COMMON = [
    (70, "MODEL", 63, "model", "MODEL"),
    (72, "VAE", 78, "vae", "VAE"),
    (72, "VAE", 65, "vae", "VAE"),
    (71, "CLIP", 74, "clip", "CLIP"),
    (71, "CLIP", 67, "clip", "CLIP"),
    (74, "CONDITIONING", 77, "conditioning", "CONDITIONING"),
    (67, "CONDITIONING", 76, "conditioning", "CONDITIONING"),
    (78, "LATENT", 77, "latent", "LATENT"),
    (78, "LATENT", 76, "latent", "LATENT"),
    (77, "CONDITIONING", 63, "positive", "CONDITIONING"),
    (76, "CONDITIONING", 63, "negative", "CONDITIONING"),
    (81, "width", 62, "width", "INT"),
    (81, "height", 62, "height", "INT"),
    (63, "GUIDER", 64, "guider", "GUIDER"),
    (61, "SAMPLER", 64, "sampler", "SAMPLER"),
    (73, "NOISE", 64, "noise", "NOISE"),
    (64, "output", 65, "samples", "LATENT"),
    (65, "IMAGE", 202, "images", "IMAGE"),
]
# img2img structure-lock: sample from the source latent through a denoise split
EDGES_IMG2IMG = [
    (78, "LATENT", 64, "latent_image", "LATENT"),
    (62, "SIGMAS", 200, "sigmas", "SIGMAS"),
    (200, "low_sigmas", 64, "sigmas", "SIGMAS"),
]
# full-regen (denoise>=1.0): empty latent + scheduler sigmas straight to sampler
EDGES_FULLREGEN = [
    (81, "width", 66, "width", "INT"),
    (81, "height", 66, "height", "INT"),
    (66, "LATENT", 64, "latent_image", "LATENT"),
    (62, "SIGMAS", 64, "sigmas", "SIGMAS"),
]

# left-to-right layout: column x, then stacked y per column
LAYOUT = {
    70: (0, 0), 71: (0, 1), 72: (0, 2), 84: (0, 3),
    80: (1, 3), 74: (1, 0), 67: (1, 1),
    81: (2, 3), 78: (2, 2), 77: (2, 0), 76: (2, 1),
    62: (3, 3), 61: (3, 4), 73: (3, 5), 63: (3, 0),
    200: (4, 3), 66: (4, 4), 64: (4, 1),
    65: (5, 1), 202: (6, 1),
}
COL_W, ROW_H = 320, 210


def find_subgraph(wf: dict, name_part: str) -> dict:
    for sg in wf["definitions"]["subgraphs"]:
        if name_part in sg.get("name", ""):
            return sg
    raise KeyError(name_part)


def slot_by_name(arr: list, name: str) -> int:
    for i, e in enumerate(arr):
        if e.get("name") == name:
            return i
    raise KeyError(f"{name} not in {[e.get('name') for e in arr]}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path,
                    default=REPO / "local_datasets/generation_source")
    ap.add_argument("--image", default=None,
                    help="specific source filename to embed; default = first in --src")
    ap.add_argument("--prompts", type=Path, default=REPO / "scripts/relight_prompts.json")
    ap.add_argument("--variant", default="warm_sun_left",
                    help="which prompt variant to bake into the positive node")
    ap.add_argument("--denoise", type=float, default=0.7)
    ap.add_argument("--cfg", type=float, default=CFG)
    ap.add_argument("--megapixels", type=float, default=0.0,
                    help="0 (default) = native resolution, NO scale node, so the graph "
                         "self-configures to whatever image is loaded. A value inserts "
                         "ImageScaleToTotalPixels at that (baked, image-specific) target.")
    ap.add_argument("--seed", type=int, default=876568685160426)
    ap.add_argument("--url", default="http://127.0.0.1:8188")
    ap.add_argument("--out", type=Path, default=None,
                    help="output workflow path; default names by mode so the two "
                         "topologies don't overwrite each other")
    args = ap.parse_args()
    if args.out is None:
        stem = "relight_structlock" if args.denoise < 1.0 else "relight_fullregen"
        args.out = OUT_WORKFLOW.with_name(f"{stem}.json")

    wf = json.loads(SRC_WORKFLOW.read_text())
    edit = find_subgraph(wf, "Image Edit")
    refc = find_subgraph(wf, "Reference Conditioning")
    by_id = {n["id"]: n for n in edit["nodes"]}
    by_id.update({n["id"]: n for n in refc["nodes"]})
    top = {n["id"]: n for n in wf["nodes"]}

    # ---- pick + upload the source image so LoadImage resolves on open ----
    srcs = sorted(p for p in args.src.iterdir()
                  if p.is_file() and p.suffix.lower() in {".exr", ".png", ".jpg", ".jpeg"})
    src = next((p for p in srcs if p.name == args.image), None) if args.image else (srcs[0] if srcs else None)
    if src is None:
        sys.exit(f"no source image found in {args.src}")
    png, w, h = exr_to_png_bytes(src) if src.suffix.lower() == ".exr" else image_to_png_bytes(src)
    uploaded = upload_image(args.url.rstrip("/"), png, f"relight_src_{src.stem}.png")

    # ---- prompt text ----
    cfg_prompts = json.loads(args.prompts.read_text())
    preamble = cfg_prompts.get("preamble", "")
    variant = next((v for v in cfg_prompts["variants"] if v["name"] == args.variant), None)
    if variant is None:
        sys.exit(f"variant {args.variant!r} not in {args.prompts}")
    positive = preamble + variant["prompt"]

    img2img = args.denoise < 1.0
    scale = args.megapixels > 0
    img_src = 80 if scale else 84   # feeds GetImageSize + VAEEncode
    edges = list(EDGES_COMMON) + [
        (img_src, "IMAGE", 81, "image", "IMAGE"),
        (img_src, "IMAGE", 78, "pixels", "IMAGE"),
    ] + (EDGES_IMG2IMG if img2img else EDGES_FULLREGEN)
    if scale:
        edges.append((84, "IMAGE", 80, "image", "IMAGE"))

    # ---- collect flattened nodes (deep-copied so we can rewrite freely) ----
    nodes: dict[int, dict] = {}
    for nid in IMAGE_EDIT_KEEP + REFCOND_KEEP:
        nodes[nid] = copy.deepcopy(by_id[nid])
    nodes[84] = copy.deepcopy(top[84])          # source LoadImage
    nodes[202] = copy.deepcopy(top[85])         # reuse a PreviewImage template
    nodes[202]["id"] = 202
    nodes[202]["title"] = "relit preview"

    if scale:
        nodes[80] = copy.deepcopy(by_id[80])    # ImageScaleToTotalPixels

    if img2img:
        # synthesised node: SplitSigmasDenoise (trims the scheduler to the tail)
        nodes[200] = {
            "id": 200, "type": "SplitSigmasDenoise", "mode": 0, "flags": {},
            "inputs": [
                {"name": "sigmas", "type": "SIGMAS", "link": None},
                {"name": "denoise", "type": "FLOAT", "widget": {"name": "denoise"}, "link": None},
            ],
            "outputs": [
                {"name": "high_sigmas", "type": "SIGMAS", "links": []},
                {"name": "low_sigmas", "type": "SIGMAS", "links": []},
            ],
            "properties": {"Node name for S&R": "SplitSigmasDenoise"},
            "widgets_values": [args.denoise],
            "size": [230, 80],
        }
    else:
        # full-regen: empty latent, matching the batch script's denoise>=1.0 graph
        nodes[66] = copy.deepcopy(by_id[66])

    # ---- wipe all existing link references, then place + tidy ----
    for nid, n in nodes.items():
        for inp in n.get("inputs", []) or []:
            inp["link"] = None
        for o in n.get("outputs", []) or []:
            o["links"] = []
        col, row = LAYOUT[nid]
        n["pos"] = [col * COL_W, row * ROW_H]
        n["flags"] = n.get("flags", {})
        n["mode"] = 0

    # ---- bake widget values ----
    nodes[84]["widgets_values"] = [uploaded, "image"]
    if scale:
        nodes[80]["widgets_values"] = ["nearest-exact", args.megapixels, 1]
    nodes[74]["widgets_values"] = [positive]      # positive prompt
    nodes[67]["widgets_values"] = [""]            # negative
    nodes[62]["widgets_values"] = [STEPS, w, h]   # width/height linked but listed
    nodes[61]["widgets_values"] = ["euler"]
    nodes[73]["widgets_values"] = [args.seed, "fixed"]
    nodes[63]["widgets_values"] = [args.cfg]
    if img2img:
        nodes[200]["widgets_values"] = [args.denoise]
    else:
        nodes[66]["widgets_values"] = [w, h, 1]   # width/height linked but listed

    # ---- build links from the selected edge set ----
    links = []
    lid = 0
    for from_id, out_name, to_id, in_name, typ in edges:
        lid += 1
        fn, tn = nodes[from_id], nodes[to_id]
        os_ = slot_by_name(fn["outputs"], out_name)
        is_ = slot_by_name(tn["inputs"], in_name)
        fn["outputs"][os_].setdefault("links", []).append(lid)
        fn["outputs"][os_]["slot_index"] = os_
        tn["inputs"][is_]["link"] = lid
        links.append([lid, from_id, os_, to_id, is_, typ])

    node_list = list(nodes.values())
    for i, n in enumerate(node_list):
        n["order"] = i

    out = {
        "id": "relight-structlock",
        "revision": 0,
        "last_node_id": max(nodes) + 1,
        "last_link_id": lid,
        "nodes": node_list,
        "links": links,
        "groups": [],
        "config": {},
        "extra": {},
        "version": 0.4,
    }
    args.out.write_text(json.dumps(out, indent=1))
    res = f"scaled to {args.megapixels} MP" if scale else f"native {w}x{h} (self-configuring)"
    print(f"wrote {args.out}")
    print(f"  source   : {src.name}  ({w}x{h})  uploaded as {uploaded}")
    print(f"  size     : {res}")
    print(f"  variant  : {args.variant}")
    print(f"  denoise  : {args.denoise}   cfg: {args.cfg}   seed: {args.seed}")
    print(f"  nodes    : {len(node_list)}   links: {lid}")
    print(f"Open ComfyUI -> Workflows -> {args.out.stem} to debug.")


if __name__ == "__main__":
    main()
