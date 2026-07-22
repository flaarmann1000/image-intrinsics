#!/usr/bin/env python
"""Smoke test: every CLI and every public entry point still imports and parses args.

Exists because the restructuring silently DROPPED the synthetic study's CLI: the
Stage-4 split checked that each new module was internally consistent, but nothing
checked that every top-level definition from the original file landed somewhere. The
loss was invisible for two stages -- `_build_parser`/`main` still existed under other
modules, so a name-based audit did not flag it either.

    python tests/test_entrypoints.py
"""
import importlib
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

CLIS = ["decompose_batch", "profile_perf", "sweep_cfg", "improve_study",
        "synthetic_study"]

MODULES = [
    "idr.config", "idr.paths",
    "idr.render", "idr.render.brdf", "idr.render.sh", "idr.render.raster",
    "idr.render.shade_ct", "idr.render.shade_phong", "idr.render.mesh",
    "idr.render.lighting", "idr.render.ops", "idr.render.types",
    "idr.data.scene_io", "idr.data.geometry", "idr.data.lighting", "idr.data.build",
    "idr.data.synthetic_scene", "idr.data.synthetic_io",
    "idr.optim.transforms", "idr.optim.losses", "idr.optim.steps",
    "idr.optim.registry", "idr.optim.result",
    "idr.optim.lm.core", "idr.optim.lm.backends", "idr.optim.lm.problem",
    "idr.optim.models.ct_sh", "idr.optim.models.ct_env",
    "idr.optim.models.phong_sh", "idr.optim.models.phong_env",
    "idr.eval.metrics", "idr.eval.plots", "idr.eval.relight_sweep",
    "idr.track.wandb_log",
    "idr.pipelines.decompose", "idr.pipelines.synthetic_generate",
    "idr.pipelines.synthetic_run", "idr.pipelines.mit", "idr.pipelines.real_scene",
]

# name -> module it must be reachable from (the API notebooks and scripts rely on)
PUBLIC = {
    "idr.render": ["shade_ct_sh", "shade_ct_env", "shade_phong_sh", "shade_phong_env",
                   "EnvMap", "SHLighting", "build_sh_basis", "get_ggx_sh_lut"],
    "idr.optim.registry": ["optimize", "SHADERS"],
    "idr.optim.result": ["OptimResult", "EnvGrid"],
    "idr.data.scene_io": ["load_scene", "load_exr"],
    "idr.data.geometry": ["make_proxy_geometry"],
    "idr.pipelines.decompose": ["decompose_scene", "make_run_name"],
    "idr.pipelines.synthetic_generate": ["generate_dataset"],
    "idr.pipelines.synthetic_run": ["run_study"],
    "idr.eval.relight_sweep": ["relight_sweep"],
    "idr.eval.metrics": ["albedo_rmse"],
    "idr.config": ["DEFAULT_CFG", "NAMED_TRANSFORMS"],
}


def main():
    bad = []
    for m in MODULES:
        try:
            importlib.import_module(m)
        except Exception as e:
            bad.append(f"import {m}: {type(e).__name__}: {e}")
    for m, names in PUBLIC.items():
        try:
            mod = importlib.import_module(m)
        except Exception:
            continue                                  # already reported above
        for n in names:
            if not hasattr(mod, n):
                bad.append(f"{m}.{n} is missing")
    for c in CLIS:
        p = REPO / "scripts" / f"{c}.py"
        if not p.exists():
            bad.append(f"scripts/{c}.py does not exist")
            continue
        r = subprocess.run([sys.executable, str(p), "--help"],
                           capture_output=True, text=True, cwd=str(REPO))
        if r.returncode != 0:
            bad.append(f"scripts/{c}.py --help exited {r.returncode}: "
                       f"{(r.stderr or '').strip()[-160:]}")

    print(f"  {len(MODULES)} modules, {sum(len(v) for v in PUBLIC.values())} public names, "
          f"{len(CLIS)} CLIs")
    if bad:
        print("\n".join(f"  FAIL {b}" for b in bad))
        return 1
    print("  all entry points OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
