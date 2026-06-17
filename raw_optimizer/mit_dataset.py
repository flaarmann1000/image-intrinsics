"""
MIT multi-illumination dataset: intrinsic decomposition with CT/Phong × SH/Env.

Runs the same 4 optimizers from synthetic_ct_dataset.py on real multi-illumination
images, using Marigold-predicted normals as geometry proxy and a frontal (or
perspective) view direction.

Usage
-----
    # Smoke-test: one shader, no regularization, 50 iters
    python raw_optimizer/mit_dataset.py --dataset small --shader ct_sh --reg none --n-iter 50

    # Full 12-combination batch on small dataset
    python raw_optimizer/mit_dataset.py --dataset small --shader all --reg all --n-iter 100

    # With perspective camera (horizontal FOV 70°)
    python raw_optimizer/mit_dataset.py --dataset full --shader ct_env --hfov 70 --n-iter 200
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image

_REPO_ROOT = Path(__file__).parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from raw_renderer_gpu import EnvMap, SHLighting
from raw_optimizer.synthetic_ct_dataset import (
    _optimize_ct_sh, _optimize_ct_env,
    _optimize_phong_sh, _optimize_phong_env,
    DEFAULT_CFG, NAMED_TRANSFORMS, SHININESS_RANGE,
    LIGHT_COLOR, LIGHT_INTENSITY,
    _parse_transforms, _transforms_folder,
    _sh_coeffs_to_env_img, _env_flat_to_img,
)


# ─────────────────────────────────────── constants ───────────────────────────

MIT_ROOT     = _REPO_ROOT / "datasets" / "mit"
RESULTS_ROOT = _REPO_ROOT / "mit_results"

DATASET_PATHS: dict[str, Path] = {
    "full":  MIT_ROOT / "multi_ill_dataset",
    "small": MIT_ROOT / "multi_ill_dataset_small",
}
NORMAL_PATH = MIT_ROOT / "multi_ill_marigold" / "normal.png"

_ALL_SHADERS = ["ct_sh", "ct_env", "phong_sh", "phong_env"]

REG_CONFIGS: dict[str, dict] = {
    "none": dict(lambda_sparse=0.0, lambda_tv=0.0),
    "ls":   dict(lambda_sparse=0.1, lambda_tv=0.0),
    "lt":   dict(lambda_sparse=0.0, lambda_tv=0.1),
}


# ─────────────────────────────────────── data helpers ────────────────────────

def _load_images(dataset_dir: Path, width: int, height: int) -> list[np.ndarray]:
    """Load dir_N_mip2.jpg files sorted by N, return list of (H,W,3) float32 in [0,1]."""
    paths = sorted(
        dataset_dir.glob("dir_*_mip2.jpg"),
        key=lambda p: int(re.search(r"dir_(\d+)_mip2", p.name).group(1)),
    )
    if not paths:
        raise FileNotFoundError(f"No images found in {dataset_dir}")
    return [
        np.array(
            Image.open(p).convert("RGB").resize((width, height), Image.LANCZOS),
            dtype=np.float32,
        ) / 255.0
        for p in paths
    ]


def _load_normals(width: int, height: int, device: str) -> torch.Tensor:
    """Load Marigold normal.png, decode RGB→[-1,1], re-normalize, return (H,W,3) tensor."""
    raw = (
        np.array(
            Image.open(NORMAL_PATH).convert("RGB").resize((width, height), Image.NEAREST),
            dtype=np.float32,
        ) / 255.0 * 2.0 - 1.0
    )
    raw = raw / np.linalg.norm(raw, axis=-1, keepdims=True).clip(min=1e-8)
    return torch.from_numpy(raw).to(device)


def _make_view_inputs(
    H: int, W: int,
    fx: Optional[float] = None,
    fy: Optional[float] = None,
    cx: Optional[float] = None,
    cy: Optional[float] = None,
    device: str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Return (frag_pos_hw, cam_pos) for the existing optimizer functions.

    Frontal mode (no intrinsics):
        frag_pos_hw = zeros(H,W,3),  cam_pos = [0,0,1]
        → view = _norm([0,0,1] - 0) = [0,0,1] for every pixel.

    Perspective mode (intrinsics given):
        frag_pos_hw[v,u] = [(u-cx)/fx, -(v-cy)/fy, -1]
        cam_pos = [0,0,0]
        → view = _norm(0 - frag_pos) = _norm([-(u-cx)/fx, (v-cy)/fy, 1])
        which is the unit vector pointing from pixel toward the camera.
    """
    if fx is None:
        frag_pos = torch.zeros(H, W, 3, dtype=torch.float32, device=device)
        cam_pos  = torch.tensor([0., 0., 1.], dtype=torch.float32, device=device)
        return frag_pos, cam_pos

    fy = fy or fx
    cx = cx if cx is not None else W / 2.0
    cy = cy if cy is not None else H / 2.0

    u = torch.arange(W, dtype=torch.float32, device=device)
    v = torch.arange(H, dtype=torch.float32, device=device)
    vv, uu = torch.meshgrid(v, u, indexing="ij")       # (H, W)
    x_n =  (uu - cx) / fx
    y_n = -(vv - cy) / fy                               # flip: rows go down, Y goes up

    frag_pos = torch.stack([x_n, y_n, torch.full_like(x_n, -1.0)], dim=-1)
    cam_pos  = torch.zeros(3, dtype=torch.float32, device=device)
    return frag_pos, cam_pos


def _save_gray(arr_hw1: np.ndarray, path: Path) -> None:
    Image.fromarray((arr_hw1.squeeze(-1) * 255).clip(0, 255).astype(np.uint8)).save(path)


def _make_env_geometry() -> tuple[np.ndarray, np.ndarray, int, int]:
    """Build env-map sample directions and solid angles (same reference as synthetic script)."""
    ref = EnvMap.from_sh(SHLighting.directional(
        np.array([0, 0, 1], dtype=np.float32), LIGHT_COLOR, intensity=LIGHT_INTENSITY,
    ), resolution=32)
    return ref._dirs, ref._solid_angles, ref.image.shape[0], ref.image.shape[1]


def _result_shader_name(shader: str, reg_cfg: dict) -> str:
    name = shader
    ls = reg_cfg.get("lambda_sparse", 0.0)
    lt = reg_cfg.get("lambda_tv",     0.0)
    if ls:
        name += f"_ls={ls}"
    if lt:
        name += f"_lt={lt}"
    return name


# ─────────────────────────────────────── decomposition ───────────────────────

def run_mit_decomposition(
    dataset:       str            = "small",
    shader:        str            = "ct_sh",
    reg:           str            = "none",
    width:         int            = 384 // 2,
    height:        int            = 256 // 2,
    cfg_overrides: Optional[dict] = None,
    device:        str            = "cpu",
    skip_existing: bool           = True,
    transforms:    Optional[dict] = None,
    use_wandb:     bool           = True,
    fx: Optional[float] = None,
    fy: Optional[float] = None,
    cx: Optional[float] = None,
    cy: Optional[float] = None,
    view_suffix:   str            = "",
) -> None:
    import wandb as wb

    shaders   = _ALL_SHADERS if shader == "all" else [shader]
    reg_names = list(REG_CONFIGS.keys()) if reg == "all" else [reg]
    tr        = transforms if transforms is not None else NAMED_TRANSFORMS["none"]
    transform_folder = _transforms_folder(tr)

    print(f"[MIT] dataset={dataset}  shaders={shaders}  reg={reg_names}"
          f"  {width}×{height}  device={device}")

    # ── load shared inputs ────────────────────────────────────────────────────
    images     = _load_images(DATASET_PATHS[dataset], width, height)
    normals_hw = _load_normals(width, height, device)
    mask_hw    = torch.ones(height, width, dtype=torch.bool, device=device)
    frag_pos_hw, cam_pos = _make_view_inputs(height, width, fx, fy, cx, cy, device)

    print(f"  Loaded {len(images)} images  "
          f"{'perspective' if fx is not None else 'frontal'} view")

    # ── env-map geometry (shared for all env-shader runs) ─────────────────────
    env_dirs_np, env_dw_np, env_H, env_W = _make_env_geometry()

    for shader_name in shaders:
        is_phong = shader_name.startswith("phong")
        is_env   = shader_name.endswith("env")
        a_label  = "shininess" if is_phong else "metallic"
        b_label  = "ks"        if is_phong else "roughness"
        a_norm   = SHININESS_RANGE[1] if (is_phong and a_label == "shininess") else 1.0

        for reg_name in reg_names:
            reg_cfg       = REG_CONFIGS[reg_name]
            ov            = cfg_overrides or {}
            cfg_suffix    = ""
            if ov.get("init_spec_zero"):
                cfg_suffix += "_sz"
            if ov.get("spec_warmup_steps"):
                cfg_suffix += f"_sw{ov['spec_warmup_steps']}"
            if ov.get("optimizer") == "Adam":
                cfg_suffix += "_adam"
            if ov.get("lr") is not None:
                cfg_suffix += f"_lr{ov['lr']}"
            if ov.get("lr_schedule", "none") != "none":
                cfg_suffix += f"_{ov['lr_schedule']}"
                if ov.get("lr_end"):
                    cfg_suffix += f"_lre{ov['lr_end']}"
                if ov.get("lr_schedule") == "step":
                    if ov.get("lr_schedule_step") is not None:
                        cfg_suffix += f"_s{ov['lr_schedule_step']}"
                    if ov.get("lr_schedule_gamma") is not None:
                        cfg_suffix += f"_g{ov['lr_schedule_gamma']}"
            result_shader = _result_shader_name(shader_name, reg_cfg) + cfg_suffix + view_suffix
            out_dir       = RESULTS_ROOT / transform_folder / dataset / result_shader
            out_dir.mkdir(parents=True, exist_ok=True)

            if skip_existing and (out_dir / "metrics.json").exists():
                print(f"  [skip] {out_dir}")
                continue

            cfg      = {**DEFAULT_CFG, **reg_cfg, **(cfg_overrides or {})}
            run_name = f"mit_{dataset}_{result_shader}"
            print(f"\n  [{run_name}]  n_iter={cfg['n_iter']}  out={out_dir}")

            wandb_run = None
            if use_wandb:
                wandb_run = wb.init(
                    entity  ="DLVC-intrinsics",
                    project="intrinsic-mit", name=run_name, reinit=True,
                    config={"dataset": dataset, "shader": shader_name, "reg": reg_name,
                            "width": width, "height": height, **cfg},
                )
                wandb_run.log({
                    "gt_images":  [wb.Image(img) for img in images],
                    "gt_normals": wb.Image(
                        (normals_hw.cpu().numpy() * 0.5 + 0.5).clip(0, 1)),
                }, step=0)

            # ── dispatch ─────────────────────────────────────────────────────
            if device != "cpu":
                torch.cuda.empty_cache()
            sh_out   = np.zeros(0)
            env_maps = np.zeros(0)

            if shader_name == "ct_sh":
                albedo, sh_out, mat_a, mat_b, shadings, history, elapsed = _optimize_ct_sh(
                    images, normals_hw, frag_pos_hw, mask_hw, cam_pos,
                    gt_metallic=0.0, gt_roughness=0.5,
                    cfg=cfg, wandb_run=wandb_run,
                    gt_sh_coeffs=None, gt_albedo=None, opt_params=None, transforms=tr,
                )
            elif shader_name == "ct_env":
                albedo, env_maps, mat_a, mat_b, shadings, history, elapsed = _optimize_ct_env(
                    images, normals_hw, frag_pos_hw, mask_hw, cam_pos,
                    gt_metallic=0.0, gt_roughness=0.5,
                    env_dirs=env_dirs_np, env_dw=env_dw_np,
                    cfg=cfg, wandb_run=wandb_run,
                    env_H=env_H, env_W=env_W,
                    gt_sh_coeffs=None, gt_albedo=None, opt_params=None, transforms=tr,
                )
            elif shader_name == "phong_sh":
                albedo, sh_out, mat_a, mat_b, shadings, history, elapsed = _optimize_phong_sh(
                    images, normals_hw, frag_pos_hw, mask_hw, cam_pos,
                    gt_shininess=32.0, gt_ks=0.5, ka=0.0, kd=1.0,
                    cfg=cfg, wandb_run=wandb_run,
                    gt_sh_coeffs=None, gt_albedo=None, opt_params=None, transforms=tr,
                )
            else:  # phong_env
                albedo, env_maps, mat_a, mat_b, shadings, history, elapsed = _optimize_phong_env(
                    images, normals_hw, frag_pos_hw, mask_hw, cam_pos,
                    gt_shininess=32.0, gt_ks=0.5, ka=0.0, kd=1.0,
                    env_dirs=env_dirs_np, env_dw=env_dw_np,
                    cfg=cfg, wandb_run=wandb_run,
                    env_H=env_H, env_W=env_W,
                    gt_sh_coeffs=None, gt_albedo=None, opt_params=None, transforms=tr,
                )

            # ── metrics ───────────────────────────────────────────────────────
            mask_np    = mask_hw.cpu().numpy()
            recon_err  = [np.abs(s - img) for s, img in zip(shadings, images)]
            recon_rmse = float(np.mean([e.mean() for e in recon_err]))
            mat_a_mean = float(mat_a[mask_np].mean())
            mat_b_mean = float(mat_b[mask_np].mean())

            metrics = dict(
                recon_rmse=recon_rmse,
                final_loss=float(history[-1]),
                elapsed_s=elapsed,
                n_images=len(images),
                shader=shader_name,
                reg=reg_name,
                **{f"{a_label}_est_mean": mat_a_mean,
                   f"{b_label}_est_mean": mat_b_mean},
            )

            # ── wandb summary ─────────────────────────────────────────────────
            if wandb_run is not None:
                if not is_env:
                    light_imgs = [wb.Image(_sh_coeffs_to_env_img(sh_out[k]))
                                  for k in range(len(images))]
                    light_key  = "est_sh_env_maps"
                else:
                    light_imgs = [wb.Image(_env_flat_to_img(env_maps[k], env_H, env_W))
                                  for k in range(len(images))]
                    light_key  = "est_env_maps"
                wandb_run.log({
                    "albedo_est":      wb.Image(albedo.clip(0, 1)),
                    f"{a_label}_est":  wb.Image((mat_a.squeeze(-1) / a_norm).clip(0, 1)),
                    f"{b_label}_est":  wb.Image(mat_b.squeeze(-1).clip(0, 1)),
                    "reconstructions": [wb.Image(s.clip(0, 1)) for s in shadings],
                    "recon_errors":    [wb.Image(e.mean(-1)) for e in recon_err],
                    light_key:         light_imgs,
                    "recon_rmse":      recon_rmse,
                    "final_loss":      float(history[-1]),
                    "elapsed_s":       elapsed,
                }, step=cfg["n_iter"])
                wandb_run.finish()

            # ── save to disk ──────────────────────────────────────────────────
            recon_dir = out_dir / "reconstructions"
            recon_dir.mkdir(exist_ok=True)

            Image.fromarray((albedo.clip(0, 1) * 255).astype(np.uint8)).save(
                out_dir / "albedo_est.png")
            _save_gray(mat_a / a_norm, out_dir / f"{a_label}_est.png")
            _save_gray(mat_b,          out_dir / f"{b_label}_est.png")

            for k, (s, e) in enumerate(zip(shadings, recon_err)):
                Image.fromarray((s.clip(0, 1) * 255).astype(np.uint8)).save(
                    recon_dir / f"recon_{k:02d}.png")
                Image.fromarray((e.mean(-1) * 255).clip(0, 255).astype(np.uint8)).save(
                    recon_dir / f"recon_err_{k:02d}.png")

            if not is_env:
                np.save(out_dir / "sh_coeffs_est.npy", sh_out)
                for k, sh_k in enumerate(sh_out):
                    img = (_sh_coeffs_to_env_img(sh_k) * 255).astype(np.uint8)
                    Image.fromarray(img).save(out_dir / f"sh_env_map_{k:02d}.png")
            else:
                np.save(out_dir / "env_maps_est.npy", env_maps)
                for k, env_k in enumerate(env_maps):
                    img = (_env_flat_to_img(env_k, env_H, env_W) * 255).astype(np.uint8)
                    Image.fromarray(img).save(out_dir / f"env_map_{k:02d}.png")
                env_avg = _env_flat_to_img(env_maps.mean(0), env_H, env_W)
                Image.fromarray((env_avg * 255).astype(np.uint8)).save(
                    out_dir / "env_map_avg.png")

            with open(out_dir / "metrics.json", "w") as fh:
                json.dump(metrics, fh, indent=2)

            print(f"  {elapsed:.1f}s  recon_rmse={recon_rmse:.4f}"
                  f"  {a_label}={mat_a_mean:.3f}  {b_label}={mat_b_mean:.3f}"
                  f"  -> {out_dir}")

    print("[MIT] Done.")


# ─────────────────────────────────────── CLI ─────────────────────────────────

def _build_parser():
    p = argparse.ArgumentParser(description="MIT multi-illumination intrinsic decomposition")
    p.add_argument("--dataset",  default="small", choices=["full", "small"])
    p.add_argument("--shader",   default="ct_sh",
                   choices=_ALL_SHADERS + ["all"])
    p.add_argument("--reg",      default="none",
                   choices=list(REG_CONFIGS) + ["all"])
    p.add_argument("--width",    type=int,   default=384)
    p.add_argument("--height",   type=int,   default=256)
    p.add_argument("--n-iter",   type=int,   default=None)
    p.add_argument("--lr",        type=float, default=None)
    p.add_argument("--optimizer", default=None, choices=["Adam", "LBFGS"])
    p.add_argument("--sbatch",   type=int,   default=1024,
                   help="Env-map sample batch size per forward pass (default 64; "
                        "reduce if OOM, increase for speed)")
    p.add_argument("--img-batch", type=int,  default=None,
                   help="Images per gradient-accumulation step (default: all images at once); "
                        "set to 1 to process images one-by-one and avoid OOM on large datasets")
    p.add_argument("--spec-warmup", type=int, default=None,
                   help="Freeze specular contribution for this many steps (metallic=0/roughness=1 "
                        "for CT, ks=0 for Phong); useful to stabilise albedo before fitting specularity")
    p.add_argument("--init-spec-zero", action="store_true",
                   help="Init specular params to their 'off' state (metallic=0/roughness=1 for CT, "
                        "ks=0 for Phong) instead of the default mid-range values")
    p.add_argument("--lr-end", type=float, default=None,
                   help="Final LR for cosine / linear / exponential schedules (default 0)")
    p.add_argument("--lr-schedule", default=None,
                   choices=["cosine", "step", "linear", "exponential"],
                   help="LR schedule for Adam (ignored for LBFGS): cosine annealing, "
                        "step decay, linear decay, or exponential decay")
    p.add_argument("--lr-schedule-step", type=int, default=None,
                   help="Step size (iters) for --lr-schedule step (default 50)")
    p.add_argument("--lr-schedule-gamma", type=float, default=None,
                   help="Decay factor for --lr-schedule step (default 0.5)")
    p.add_argument("--device",   default=None)
    p.add_argument("--no-skip", action="store_true",
                   help="Re-run even if metrics.json already exists")
    p.add_argument("--transforms", default="none",
                   help="Parameter domain transforms: 'none', 'all', or 'k=v,...'")
    p.add_argument("--no-wandb", action="store_true")
    # Camera intrinsics (optional — omit for constant frontal view)
    p.add_argument("--hfov", type=float, nargs='+', default=None,
                   help="Horizontal FOV(s) in degrees; e.g. --hfov 70 or --hfov 30 50 70")
    p.add_argument("--fx",   type=float, default=None, help="Focal length in pixels (x)")
    p.add_argument("--fy",   type=float, default=None, help="Focal length in pixels (y)")
    p.add_argument("--cx",   type=float, default=None, help="Principal point x (default W/2)")
    p.add_argument("--cy",   type=float, default=None, help="Principal point y (default H/2)")
    return p


def main():
    args   = _build_parser().parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    tr     = _parse_transforms(args.transforms)

    overrides = {k: v for k, v in [
        ("n_iter",            args.n_iter),
        ("lr",                args.lr),
        ("sbatch",            args.sbatch),
        ("img_batch",         args.img_batch),
        ("spec_warmup_steps", args.spec_warmup),
        ("init_spec_zero",    True if args.init_spec_zero else None),
        ("lr_end",            args.lr_end),
        ("lr_schedule",       args.lr_schedule),
        ("lr_schedule_step",  args.lr_schedule_step),
        ("lr_schedule_gamma", args.lr_schedule_gamma),
        ("optimizer",         args.optimizer),
    ] if v is not None}

    hfov_list = args.hfov  # None, or list of one or more floats

    def _run(fx=None, fy=None, cx=None, cy=None, view_suffix=""):
        run_mit_decomposition(
            dataset=args.dataset, shader=args.shader, reg=args.reg,
            width=args.width, height=args.height,
            cfg_overrides=overrides or None, device=device,
            skip_existing=not args.no_skip, transforms=tr,
            use_wandb=not args.no_wandb,
            fx=fx, fy=fy, cx=cx, cy=cy, view_suffix=view_suffix,
        )

    if hfov_list is not None:
        for hfov in hfov_list:
            fx = fy = (args.width / 2.0) / math.tan(math.radians(hfov) / 2.0)
            cx, cy  = args.width / 2.0, args.height / 2.0
            _run(fx=fx, fy=fy, cx=cx, cy=cy, view_suffix=f"_hfov={int(hfov)}")
    else:
        fx, fy, cx, cy = args.fx, args.fy, args.cx, args.cy
        view_suffix = f"_perspective" if fx is not None else ""
        _run(fx=fx, fy=fy, cx=cx, cy=cy, view_suffix=view_suffix)


if __name__ == "__main__":
    main()
