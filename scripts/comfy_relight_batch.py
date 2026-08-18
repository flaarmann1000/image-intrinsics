#!/usr/bin/env python3
"""Batch relighting via a local ComfyUI Flux.2 Klein workflow.

For every image in --src, and every lighting variant in --prompts, this submits
the Flux.2 Klein "Image Edit" graph to a running ComfyUI server over its HTTP API
and saves the relit result. Source EXRs are tonemapped to sRGB PNG before upload;
PNG/JPG sources are uploaded as-is.

The graph here is the flattened (API-format) equivalent of
    ComfyUI/user/default/workflows/relighting.json
with its two nested subgraphs (Image Edit + Reference Conditioning) inlined.

Run with the project env (needs numpy + idr for the EXR reader):
    /home/felix/myenv/bin/python scripts/comfy_relight_batch.py

    # match native resolution (no upscaling), all 6 variants:
    /home/felix/myenv/bin/python scripts/comfy_relight_batch.py

    # smoke-test one image, one variant:
    /home/felix/myenv/bin/python scripts/comfy_relight_batch.py \
        --limit 1 --only warm_sun_left
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import time
import uuid
from pathlib import Path

import numpy as np
import requests
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from idr.data.scene_io import load_exr, linear_to_srgb  # noqa: E402

# --- model / sampler settings (from relighting.json) -----------------------
UNET_NAME = "flux-2-klein-base-4b-fp8.safetensors"
CLIP_NAME = "qwen_3_4b.safetensors"
VAE_NAME = "flux2-vae.safetensors"
STEPS = 20
CFG = 5.0
SAMPLER = "euler"
IMAGE_EXTS = {".exr", ".png", ".jpg", ".jpeg", ".webp"}

CANONICAL = REPO / "local_datasets/canonical"
GEN_SORTED = REPO / "local_datasets/gen_relit_sorted"


def resolve_domain(scene: str) -> str | None:
    """Domain (e.g. INFINITE / MIT) = the canonical GT set that contains <scene>.

    Mirrors build_generated_dataset.py: canonical/<DOMAIN>/<scene> or
    canonical/<DOMAIN>-train/<scene>. The '-train' suffix is stripped so the
    sorted-output folder matches the existing gen_relit_sorted/<DOMAIN> naming.
    """
    if not CANONICAL.is_dir():
        return None
    for cand in sorted(CANONICAL.iterdir()):
        if cand.is_dir() and (cand / scene).is_dir():
            name = cand.name
            return name[:-len("-train")] if name.endswith("-train") else name
    return None


def exr_to_png_bytes(path: Path) -> tuple[bytes, int, int]:
    """Load a linear-HDR EXR, tonemap to 8-bit sRGB PNG. Returns (png, w, h)."""
    lin = np.clip(load_exr(path), 0.0, 1.0)
    srgb = np.clip(linear_to_srgb(lin), 0.0, 1.0)
    arr = (srgb * 255.0 + 0.5).astype(np.uint8)
    h, w = arr.shape[:2]
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue(), w, h


def image_to_png_bytes(path: Path) -> tuple[bytes, int, int]:
    img = Image.open(path).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue(), img.width, img.height


def build_graph(image_name: str, positive: str, seed: int,
                megapixels: float | None, filename_prefix: str,
                cfg: float = CFG, denoise: float = 1.0, negative: str = "") -> dict:
    """Flattened API-format Flux.2 Klein Image-Edit graph.

    node ids mirror the original workflow's inner node ids for traceability.

    `megapixels`:
      * None (default) -- work at the source's NATIVE resolution. No scale node is
        emitted at all: the loaded image feeds GetImageSize/VAEEncode directly, so
        the graph self-configures to whatever image is loaded (Flux2 still aligns
        the latent to a multiple of 16, so output dims are the input floored to /16).
      * a float -- insert ImageScaleToTotalPixels to resample to that many
        mebipixels (megapixels * 1024*1024) before generating.

    Two structure-preservation regimes, selected by `denoise`:

    * denoise >= 1.0 (default) -- the original workflow. Sampling starts from an
      EMPTY latent (pure noise); the source enters only as ReferenceLatent
      context tokens, so the model regenerates the whole scene from scratch and
      objects can drift.

    * denoise < 1.0 -- img2img "structure lock". Sampling starts from the source's
      OWN VAE latent, noised only to the `denoise` fraction (via SplitSigmasDenoise),
      so composition/geometry are pinned by the starting latent while the last
      denoise steps restyle the illumination. Lower denoise = less object variation
      (but also less lighting change). ~0.55-0.8 is the useful band.
    """
    img2img = denoise < 1.0
    scale = megapixels is not None
    pix = ["80", 0] if scale else ["84", 0]   # pixel source for GetImageSize/VAEEncode
    g = {
        # loaders
        "70": {"class_type": "UNETLoader",
               "inputs": {"unet_name": UNET_NAME, "weight_dtype": "default"}},
        "71": {"class_type": "CLIPLoader",
               "inputs": {"clip_name": CLIP_NAME, "type": "flux2", "device": "default"}},
        "72": {"class_type": "VAELoader", "inputs": {"vae_name": VAE_NAME}},
        # source image (optionally resampled to a target pixel count) -> read its size
        "84": {"class_type": "LoadImage", "inputs": {"image": image_name}},
        "81": {"class_type": "GetImageSize", "inputs": {"image": pix}},
        # text conditioning (positive from prompt, negative empty)
        "74": {"class_type": "CLIPTextEncode",
               "inputs": {"text": positive, "clip": ["71", 0]}},
        "67": {"class_type": "CLIPTextEncode",
               "inputs": {"text": negative, "clip": ["71", 0]}},
        # reference conditioning (inlined subgraph): encode source, inject as ref latent
        "78": {"class_type": "VAEEncode",
               "inputs": {"pixels": pix, "vae": ["72", 0]}},
        "77": {"class_type": "ReferenceLatent",
               "inputs": {"conditioning": ["74", 0], "latent": ["78", 0]}},
        "76": {"class_type": "ReferenceLatent",
               "inputs": {"conditioning": ["67", 0], "latent": ["78", 0]}},
        # sampling
        "63": {"class_type": "CFGGuider",
               "inputs": {"model": ["70", 0], "positive": ["77", 0],
                          "negative": ["76", 0], "cfg": cfg}},
        "61": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": SAMPLER}},
        "62": {"class_type": "Flux2Scheduler",
               "inputs": {"steps": STEPS, "width": ["81", 0], "height": ["81", 1]}},
        "73": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
        "64": {"class_type": "SamplerCustomAdvanced",
               "inputs": {"noise": ["73", 0], "guider": ["63", 0], "sampler": ["61", 0],
                          "sigmas": ["62", 0], "latent_image": ["66", 0]}},
        "65": {"class_type": "VAEDecode",
               "inputs": {"samples": ["64", 0], "vae": ["72", 0]}},
        "99": {"class_type": "SaveImage",
               "inputs": {"images": ["65", 0], "filename_prefix": filename_prefix}},
    }
    if scale:
        # only present when a target size is requested; otherwise native resolution
        g["80"] = {"class_type": "ImageScaleToTotalPixels",
                   "inputs": {"image": ["84", 0], "upscale_method": "nearest-exact",
                              "megapixels": megapixels, "resolution_steps": 1}}
    if img2img:
        # start from the source latent, noised only to the denoise fraction
        g["68"] = {"class_type": "SplitSigmasDenoise",
                   "inputs": {"sigmas": ["62", 0], "denoise": denoise}}
        g["64"]["inputs"]["sigmas"] = ["68", 1]        # low_sigmas tail
        g["64"]["inputs"]["latent_image"] = ["78", 0]  # source VAE latent
    else:
        # full denoise from an empty latent (original behaviour)
        g["66"] = {"class_type": "EmptyFlux2LatentImage",
                   "inputs": {"width": ["81", 0], "height": ["81", 1], "batch_size": 1}}
    return g


def upload_image(base: str, png: bytes, name: str) -> str:
    r = requests.post(
        f"{base}/upload/image",
        files={"image": (name, png, "image/png")},
        data={"overwrite": "true", "type": "input"},
        timeout=60,
    )
    r.raise_for_status()
    j = r.json()
    sub = j.get("subfolder", "")
    return f"{sub}/{j['name']}" if sub else j["name"]


def queue_prompt(base: str, graph: dict, client_id: str) -> str:
    r = requests.post(f"{base}/prompt",
                      json={"prompt": graph, "client_id": client_id}, timeout=60)
    if r.status_code != 200:
        raise RuntimeError(f"/prompt rejected ({r.status_code}): {r.text[:800]}")
    return r.json()["prompt_id"]


def wait_for(base: str, prompt_id: str, timeout: float = 900.0) -> dict:
    """Poll /history until the prompt finishes; return its outputs dict."""
    t0 = time.time()
    while time.time() - t0 < timeout:
        h = requests.get(f"{base}/history/{prompt_id}", timeout=30).json()
        entry = h.get(prompt_id)
        if entry:
            status = entry.get("status", {})
            if status.get("completed") or status.get("status_str") == "success":
                return entry.get("outputs", {})
            if status.get("status_str") == "error":
                raise RuntimeError(f"execution error: {json.dumps(status)[:800]}")
        time.sleep(1.0)
    raise TimeoutError(f"prompt {prompt_id} did not finish in {timeout}s")


def fetch_outputs(base: str, outputs: dict) -> list[bytes]:
    imgs = []
    for node in outputs.values():
        for img in node.get("images", []):
            r = requests.get(f"{base}/view", params={
                "filename": img["filename"], "subfolder": img.get("subfolder", ""),
                "type": img.get("type", "output")}, timeout=60)
            r.raise_for_status()
            imgs.append(r.content)
    return imgs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path,
                    default=REPO / "local_datasets/generation_source")
    ap.add_argument("--out", type=Path,
                    default=REPO / "local_datasets/generation_relit",
                    help="output root when --postfix is NOT set: <out>/<scene>/")
    ap.add_argument("--postfix", default=None,
                    help="name the sorted export folder. Writes to gen_relit_sorted/"
                         "<DOMAIN>/generation_relit_<postfix>/<scene>/, with DOMAIN "
                         "resolved per scene from canonical/ (INFINITE, MIT, ...).")
    ap.add_argument("--prompts", type=Path, default=REPO / "scripts/relight_prompts.json")
    ap.add_argument("--url", default="http://127.0.0.1:8188")
    ap.add_argument("--megapixels", type=float, default=0.0,
                    help="target size for ImageScaleToTotalPixels; "
                         "0 = match each source's native pixel count (no upscaling)")
    ap.add_argument("--denoise", type=float, default=1.0,
                    help="1.0 (default) = original regime: full denoise from empty "
                         "latent, source enters only as reference tokens (objects can "
                         "drift). <1.0 = img2img structure-lock: start from the source "
                         "latent, only restyle lighting. Try 0.6-0.8 to vary objects less.")
    ap.add_argument("--cfg", type=float, default=CFG,
                    help=f"classifier-free guidance (default {CFG}); lower stays closer "
                         "to the source, higher follows the prompt harder")
    ap.add_argument("--tag", default=None,
                    help="suffix added to output filenames + SaveImage prefix, e.g. "
                         "'d70'; defaults to 'dNN' when --denoise<1 so regimes don't collide")
    ap.add_argument("--seed", type=int, default=876568685160426,
                    help="static noise seed used for EVERY image/variant, so outputs "
                         "differ only by the prompt (matches the exported UI workflow)")
    ap.add_argument("--random-seed", action="store_true",
                    help="use a fresh random seed per generation instead of --seed")
    ap.add_argument("--only", nargs="*", default=None,
                    help="restrict to these variant names")
    ap.add_argument("--limit", type=int, default=0, help="max source images (0 = all)")
    ap.add_argument("--overwrite", action="store_true",
                    help="regenerate even if the output PNG already exists")
    args = ap.parse_args()

    base = args.url.rstrip("/")
    cfg = json.loads(args.prompts.read_text())
    preamble = cfg.get("preamble", "")
    negative_preamble = cfg.get("negative_preamble", "")
    variants = cfg["variants"]
    if args.only:
        want = set(args.only)
        variants = [v for v in variants if v["name"] in want]
        if not variants:
            sys.exit(f"no variants matched {args.only}")

    try:
        requests.get(f"{base}/system_stats", timeout=5).raise_for_status()
    except Exception as e:
        sys.exit(f"ComfyUI not reachable at {base}: {e}")

    srcs = sorted(p for p in args.src.iterdir()
                  if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    if args.limit:
        srcs = srcs[:args.limit]
    if not srcs:
        sys.exit(f"no images found in {args.src}")

    # a filename suffix so the two regimes never overwrite each other
    if args.tag is not None:
        tag = f"_{args.tag}" if args.tag else ""
    elif args.denoise < 1.0:
        tag = f"_d{int(round(args.denoise * 100)):02d}"
    else:
        tag = ""

    client_id = uuid.uuid4().hex
    if not args.postfix:
        args.out.mkdir(parents=True, exist_ok=True)
    total = len(srcs) * len(variants)
    mode = (f"img2img structure-lock (denoise={args.denoise}, cfg={args.cfg})"
            if args.denoise < 1.0 else f"reference-token regen (cfg={args.cfg})")
    dest = (f"{GEN_SORTED}/<DOMAIN>/generation_relit_{args.postfix}/<scene>"
            if args.postfix else f"{args.out}/<scene>")
    print(f"{len(srcs)} images x {len(variants)} variants = {total} generations")
    print(f"mode: {mode}")
    print(f"server {base} | out {dest}\n")

    done = 0
    for src in srcs:
        stem = src.stem
        if src.suffix.lower() == ".exr":
            png, w, h = exr_to_png_bytes(src)
        else:
            png, w, h = image_to_png_bytes(src)
        # None => native resolution (no scale node); a value => resample to that many
        # mebipixels. Native was previously emulated with a no-op ImageScaleToTotalPixels
        # at megapixels = w*h/1024**2; dropping the node is bit-identical and lets the
        # exported UI graph self-configure to any image that gets loaded into it.
        mp = args.megapixels if args.megapixels > 0 else None
        if args.postfix:
            domain = resolve_domain(stem)
            if domain is None:
                print(f"[--/--] {stem} -> SKIP: no domain in {CANONICAL} for this scene")
                continue
            out_dir = GEN_SORTED / domain / f"generation_relit_{args.postfix}" / stem
        else:
            out_dir = args.out / stem
        uploaded = upload_image(base, png, f"relight_src_{stem}.png")
        out_dir.mkdir(parents=True, exist_ok=True)
        for vi, v in enumerate(variants):
            done += 1
            vname = v["name"].replace(" ", "_")
            label = f"[{done}/{total}] {stem} :: {v['name']}"
            out_path = out_dir / f"{stem}__{vname}{tag}.png"
            if out_path.exists() and not args.overwrite:
                print(f"{label} -> skip (exists)")
                continue
            # one static seed for every image/variant, so differences are purely the
            # prompt (best for debugging); --random-seed opts back into fresh noise.
            seed = np.random.randint(0, 2**48) if args.random_seed else args.seed
            graph = build_graph(
                image_name=uploaded,
                positive=preamble + v["prompt"],
                negative=negative_preamble + v.get("negative", ""),
                seed=int(seed),
                megapixels=mp,
                filename_prefix=f"relight/{stem}__{vname}{tag}",
                cfg=args.cfg,
                denoise=args.denoise,
            )
            t0 = time.time()
            try:
                pid = queue_prompt(base, graph, client_id)
                outs = wait_for(base, pid)
                imgs = fetch_outputs(base, outs)
            except Exception as e:
                print(f"{label} -> FAILED: {e}")
                continue
            if not imgs:
                print(f"{label} -> no image returned")
                continue
            out_path.write_bytes(imgs[0])
            print(f"{label} -> {out_path.relative_to(REPO)}  ({time.time()-t0:.1f}s)")

    print("\ndone.")


if __name__ == "__main__":
    main()
