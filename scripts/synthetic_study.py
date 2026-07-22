#!/usr/bin/env python
"""CLI for the synthetic study: render the dataset (phase 1), then decompose it (phase 2).

Recovered from raw_optimizer/synthetic_ct_dataset.py, which was split up in the
restructuring; the two phases now live in idr/pipelines/synthetic_{generate,run}.py.
`run_decomposition()` was renamed `run_study()` there to stop colliding with
scripts/decompose_batch.py (formerly run_decomposition.py), which drives the real
3D-Front datasets instead.

  python scripts/synthetic_study.py --phase 1 --mesh sphere
  python scripts/synthetic_study.py --phase 2 --mesh sphere --shader ct_sh
"""
import argparse
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from idr.config import MATERIAL_CONFIGS, PHONG_MATERIAL_CONFIGS   # noqa: E402
from idr.pipelines.synthetic_generate import generate_dataset      # noqa: E402
from idr.pipelines.synthetic_run import run_study                  # noqa: E402
from idr.optim.transforms import _parse_transforms                  # noqa: E402

_ALL_SHADERS = ["ct_sh", "ct_env", "phong_sh", "phong_env"]


def _build_parser():
    p = argparse.ArgumentParser(description="Synthetic CT + Phong dataset + decomposer")
    p.add_argument("--mesh",     default="sphere",
                   choices=["sphere", "suzanne", "bunny", "all"])
    p.add_argument("--width",    type=int, default=128)
    p.add_argument("--height",   type=int, default=128)
    p.add_argument("--phase",    type=int, default=1, choices=[1, 2])
    p.add_argument("--shader",   default="ct_sh",
                   choices=_ALL_SHADERS + ["all"],
                   help="Shader type (default: ct_sh)")
    p.add_argument("--optimizer", default=None, choices=["LBFGS", "Adam"])
    p.add_argument("--n-iter",        type=int,   default=None)
    p.add_argument("--lr",            type=float, default=None)
    p.add_argument("--lambda-sparse", type=float, default=None)
    p.add_argument("--lambda-white",  type=float, default=None)
    p.add_argument("--lambda-tv",     type=float, default=None)
    p.add_argument("--mat",         default=None,
                   help="Single full scene name, e.g. sphere_default")
    p.add_argument("--mat-configs", default=None,
                   help="Comma-separated config keys to include, e.g. 'albedo_0,metallic_1'. "
                        "Default: all configs.")
    p.add_argument("--device",    default=None,
                   help="torch device, e.g. cuda, cuda:1, cpu (default: cuda if available)")
    p.add_argument("--opt-params", default=None,
                   help="Comma-separated learnable params, e.g. 'albedo,sh'. Default: all. "
                        "Results written to <shader>_op=<params> subfolder.")
    p.add_argument("--skip-existing", action="store_true",
                   help="Skip dataset renders / optimization runs whose output already exists.")
    p.add_argument("--transforms", default="none",
                   help="Parameter domain transforms: 'none', 'all', or custom 'k=v,...' pairs. "
                        "Default: none (no transforms).")
    p.add_argument("--light-mode", default="directional",
                   choices=["directional", "random_sh", "circular"],
                   help="Lighting mode for dataset generation (default: directional).")
    p.add_argument("--n-lights",   type=int, default=6,
                   help="Number of light configurations per scene (default: 6).")
    p.add_argument("--full-circle", action="store_true",
                   help="Spread directional lights over full 360° instead of 0–90°.")
    p.add_argument("--init-from-gt", action="store_true",
                   help="Initialize optimizable parameters from GT values (phase 2).")
    p.add_argument("--log-gradients", action="store_true",
                   help="Log per-step gradient flow snapshots to gradient_flow/ (phase 2).")
    return p


def main():
    args = _build_parser().parse_args()
    overrides = {k: v for k, v in [
        ("optimizer",      args.optimizer),
        ("n_iter",         args.n_iter),
        ("lr",             args.lr),
        ("lambda_sparse",  args.lambda_sparse),
        ("lambda_white",   args.lambda_white),
        ("lambda_tv",      args.lambda_tv),
    ] if v is not None}

    device        = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    meshes        = ["sphere", "suzanne", "bunny"] if args.mesh == "all" else [args.mesh]
    shaders       = _ALL_SHADERS if args.shader == "all" else [args.shader]
    opt_params         = frozenset(args.opt_params.split(",")) if args.opt_params else None
    skip_existing      = args.skip_existing
    mat_configs_filter = set(args.mat_configs.split(",")) if args.mat_configs else None
    transforms         = _parse_transforms(args.transforms)

    if args.phase == 1:
        for mesh in meshes:
            generate_dataset(mesh_name=mesh, width=args.width, height=args.height,
                             shader=args.shader, device=device,
                             skip_existing=skip_existing,
                             mat_configs_filter=mat_configs_filter,
                             light_mode=args.light_mode,
                             n_lights=args.n_lights,
                             full_circle=args.full_circle)
    else:
        for mesh in meshes:
            for sh in shaders:
                run_study(
                    mesh_name          =mesh,
                    width              =args.width,
                    height             =args.height,
                    shader             =sh,
                    mat_filter         =args.mat,
                    cfg_overrides      =overrides or None,
                    device             =device,
                    opt_params         =opt_params,
                    skip_existing      =skip_existing,
                    mat_configs_filter =mat_configs_filter,
                    transforms         =transforms,
                    light_mode         =args.light_mode,
                    n_lights           =args.n_lights,
                    full_circle        =args.full_circle,
                    init_from_gt       =args.init_from_gt,
                    log_gradients      =args.log_gradients,
                )


if __name__ == "__main__":
    main()
