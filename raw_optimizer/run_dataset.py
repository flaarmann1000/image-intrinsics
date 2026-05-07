"""
Run the CT+SH optimizer over raw_dataset scenes.

Edit the CONFIG dicts below to try different hyperparameter settings.
Each config is run for every scene and the results are saved under OUTPUT_DIR.

Usage:
    python raw_optimizer/run_dataset.py
    python -m raw_optimizer.run_dataset
"""

import json
import os

import numpy as np
from PIL import Image

from .scene_loader import list_scenes, load_scene
from .optimizer import optimize, optimize_tiny

# ── Mode ─────────────────────────────────────────────────────────────────────

MODE = "tiny"   # "scenes" | "tiny"

# ── Scene dataset selection ───────────────────────────────────────────────────

SCENES = None          # None → all scenes; or e.g. ["scene_0000"]
SHADER = "ct"
LIGHT_TYPE = "sh"
VARIANT_INDICES = list(range(10))   # which rendered variants to use as inputs
WIDTH = 200
HEIGHT = 200
USE_MESH_NORMALS = False         # True = rasterize_geometry; False = load PNG
DEVICE = "cuda"
NORMALIZE_SCALE = True   # per-channel scale normalization before computing albedo RMSE

# ── Tiny dataset selection ────────────────────────────────────────────────────

TINY_NORMAL_SETS = ["normals_a", "normals_b"]   # which normal maps to run
TINY_N_VARIANTS = None                          # None → all 10

# ── Configurations to try ─────────────────────────────────────────────────────
# Each entry is passed as **kwargs to optimizer.optimize().
# metallic / roughness override GT values when set; omit to use GT.

CONFIGS = [
    {
        "name":           "no reg",
        "n_iter":         2000,
        "lr":             5e-3,
        "lambda_sparse":  0.0,
        "lambda_white":   0.0,
    },
]

OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "raw_optimizer_results",
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _save_img(arr: np.ndarray, path: str) -> None:
    """Save float32 [H,W,3] in [0,1] as uint8 PNG."""
    Image.fromarray((arr.clip(0, 1) * 255).astype(np.uint8)).save(path)


def _albedo_rmse(est: np.ndarray, gt: np.ndarray,
                 mask: np.ndarray, normalize_scale: bool = True) -> float:
    """
    RMSE between estimated and GT albedo over foreground pixels.

    est, gt : [H, W, 3] float32
    mask    : [H, W] bool
    normalize_scale : if True, rescale each channel of est by (mean_gt / mean_est)
        before computing RMSE, removing the per-channel scale ambiguity that arises
        when lambda_white = 0 (SH and albedo can only be recovered up to scale).
    """
    fg_mask = mask if isinstance(mask, np.ndarray) else mask.cpu().numpy()
    fg_est = est[fg_mask]          # [N, 3]
    fg_gt = gt[fg_mask]           # [N, 3]

    if normalize_scale:
        # scale = fg_gt.mean(0) / (fg_est.mean(0) + 1e-8)   # [3] per-channel
        numerator = (fg_gt * fg_est).sum(0)   # (3,)
        denominator = (fg_est * fg_est).sum(0)  # (3,)
        scale = numerator / denominator
        fg_est = fg_est * scale

        print(f"normalized by scale: {scale}")

    return float(np.sqrt(((fg_est - fg_gt) ** 2).mean()))


# ── Main loop ─────────────────────────────────────────────────────────────────

def run() -> None:
    scenes = SCENES if SCENES is not None else list_scenes()
    if not scenes:
        print("No scenes found. Run dataset.py first to generate raw_dataset.")
        return

    print(f"Running {len(scenes)} scene(s) × {len(CONFIGS)} config(s)")

    all_metrics = {}

    for scene_id in scenes:
        print(f"\n{'='*60}")
        print(f"Scene: {scene_id}")
        print(f"{'='*60}")

        scene = load_scene(
            scene_id,
            shader=SHADER,
            light_type=LIGHT_TYPE,
            variant_indices=VARIANT_INDICES,
            width=WIDTH,
            height=HEIGHT,
            use_mesh_normals=USE_MESH_NORMALS,
            device=DEVICE,
        )
        objs = scene["params"]["objects"]
        for oi, o in enumerate(objs):
            print(f"  obj{oi}: albedo={np.array(o['albedo']).round(3)}  "
                  f"metallic={o['ct']['metallic']:.3f}  roughness={o['ct']['roughness']:.3f}")
        print(f"  images    : {len(scene['images'])} × {WIDTH}×{HEIGHT}")

        all_metrics[scene_id] = {}

        for cfg in CONFIGS:
            cfg_name = cfg["name"]
            print(f"\n  --- config: {cfg_name} ---")

            # metallic: config override or default to 0.0 (pure dielectric)
            metallic = cfg.get("metallic", 0.0)

            opt_kwargs = {k: v for k, v in cfg.items()
                          if k not in ("name", "metallic")}

            albedo, sh_coeffs_out, shadings, history = optimize(
                images=scene["images"],
                normals=scene["normals"],
                frag_pos=scene["frag_pos"],
                cam_pos=scene["cam_pos"],
                mask=scene["mask"],
                metallic=metallic,
                **opt_kwargs,
            )

            # ── Metrics ───────────────────────────────────────────────────────
            rmse = _albedo_rmse(albedo, scene["gt_albedo"], scene["mask"],
                                normalize_scale=NORMALIZE_SCALE)
            print(
                f"  albedo RMSE vs GT: {rmse:.4f}  (scale-norm={NORMALIZE_SCALE})")
            print(f"  final loss: {history[-1]:.5f}")

            all_metrics[scene_id][cfg_name] = {
                "albedo_rmse":    rmse,
                "final_loss":     history[-1],
                "normalize_scale": NORMALIZE_SCALE,
                "loss_history":   history,
            }

            # ── Save outputs ──────────────────────────────────────────────────
            out_dir = os.path.join(OUTPUT_DIR, scene_id, cfg_name)
            os.makedirs(out_dir, exist_ok=True)

            _save_img(albedo, os.path.join(out_dir, "albedo_est.png"))
            _save_img(scene["gt_albedo"], os.path.join(
                out_dir, "albedo_gt.png"))

            np.save(os.path.join(out_dir, "sh_coeffs_est.npy"), sh_coeffs_out)
            gt_sh = np.stack(scene["gt_sh_coeffs"])   # [N, 9, 3]
            np.save(os.path.join(out_dir, "sh_coeffs_gt.npy"), gt_sh)

            for k, shading in enumerate(shadings):
                _save_img(shading, os.path.join(
                    out_dir, f"shading_{k:02d}.png"))

            with open(os.path.join(out_dir, "metrics.json"), "w") as f:
                json.dump(all_metrics[scene_id][cfg_name], f, indent=2)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Summary (albedo RMSE)")
    header = f"{'scene':<15}" + "".join(f"{c['name']:>12}" for c in CONFIGS)
    print(header)
    for sid, results in all_metrics.items():
        row = f"{sid:<15}"
        for cfg in CONFIGS:
            v = results.get(cfg["name"], {}).get("albedo_rmse", float("nan"))
            row += f"{v:>12.4f}"
        print(row)

    summary_path = os.path.join(OUTPUT_DIR, "summary.json")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nFull results → {OUTPUT_DIR}")


def run_tiny() -> None:
    _ds_root = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "raw_dataset",
    )
    gt_albedo = (
        np.array(
            Image.open(os.path.join(_ds_root, "raw_tiny", "albedo_tiny.png")),
            dtype=np.float32,
        ) / 255.0
    )                                                      # (3, 6, 3)

    all_metrics = {}

    for normal_set in TINY_NORMAL_SETS:
        print(f"\n{'='*60}")
        print(f"Tiny: {normal_set}")
        print(f"{'='*60}")

        all_metrics[normal_set] = {}

        for cfg in CONFIGS:
            cfg_name = cfg["name"]
            print(f"\n  --- config: {cfg_name} ---")

            opt_kwargs = {k: v for k, v in cfg.items(
            ) if k not in ("name", "metallic")}

            albedo, sh_coeffs_out, shadings, history = optimize_tiny(
                normal_set=normal_set,
                n_variants=TINY_N_VARIANTS,
                **opt_kwargs,
            )

            mask = np.ones((3, 6), dtype=bool)   # all 18 pixels are foreground
            rmse = _albedo_rmse(albedo, gt_albedo, mask,
                                normalize_scale=NORMALIZE_SCALE)
            print(
                f"  albedo RMSE vs GT: {rmse:.4f}  (scale-norm={NORMALIZE_SCALE})")
            print(f"  final loss: {history[-1]:.5f}")

            all_metrics[normal_set][cfg_name] = {
                "albedo_rmse":     rmse,
                "final_loss":      history[-1],
                "normalize_scale": NORMALIZE_SCALE,
                "loss_history":    history,
            }

            out_dir = os.path.join(OUTPUT_DIR, "tiny", normal_set, cfg_name)
            os.makedirs(out_dir, exist_ok=True)

            _save_img(albedo,    os.path.join(out_dir, "albedo_est.png"))
            _save_img(gt_albedo, os.path.join(out_dir, "albedo_gt.png"))
            np.save(os.path.join(out_dir, "sh_coeffs_est.npy"), sh_coeffs_out)

            for k, shading in enumerate(shadings):
                _save_img(shading, os.path.join(
                    out_dir, f"shading_{k:02d}.png"))

            with open(os.path.join(out_dir, "metrics.json"), "w") as f:
                json.dump(all_metrics[normal_set][cfg_name], f, indent=2)

    print(f"\n{'='*60}")
    print("Summary (albedo RMSE)")
    header = f"{'normal_set':<15}" + \
        "".join(f"{c['name']:>12}" for c in CONFIGS)
    print(header)
    for ns, results in all_metrics.items():
        row = f"{ns:<15}"
        for cfg in CONFIGS:
            v = results.get(cfg["name"], {}).get("albedo_rmse", float("nan"))
            row += f"{v:>12.4f}"
        print(row)

    tiny_out = os.path.join(OUTPUT_DIR, "tiny")
    os.makedirs(tiny_out, exist_ok=True)
    with open(os.path.join(tiny_out, "summary.json"), "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nFull results → {tiny_out}")


if __name__ == "__main__":
    if MODE == "tiny":
        run_tiny()
    else:
        run()
