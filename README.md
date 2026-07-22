# idr — intrinsic decomposition and relighting

Recovers per-pixel albedo, roughness and metallic (or Phong shininess/ks) plus per-image
lighting from a set of images of a fixed scene under varying illumination.

## Layout

```
idr/
  config.py        DEFAULT_CFG, transform presets, material presets
  paths.py         repo-relative paths (never Path(__file__).parents[N] in situ)
  render/          GPU renderer
    types  ops  sh  brdf  shade_ct  shade_phong  raster  mesh  lighting
  data/            scene IO, proxy geometry, dataset building
    scene_io  geometry  lighting  build  synthetic_scene  synthetic_io
  optim/           the optimizers
    config-driven: transforms  losses  steps
    registry.py    ONE dispatch point over the four models
    result.py      OptimResult (replaces the old positional 7-tuple)
    models/        ct_sh  ct_env  phong_sh  phong_env
    lm/            core  backends (CG/Schur)  problem (ct_sh's LM assembly)
  eval/            metrics  relight_sweep  plots
  track/           wandb logging
  pipelines/       decompose  synthetic_generate  synthetic_run  mit  real_scene
scripts/           command-line entry points
notebooks/         current notebooks; archive/ holds the dormant ones
tools/             nbstrip.py (git clean filter for notebook outputs)
tests/             golden.py + the recorded baseline, test_entrypoints.py
legacy/            superseded code, kept for reference; nothing live imports it

datasets/          SOURCE data — read, never written (3D-Front, MIT, office)
assets/            meshes and other static resources
results/           EVERY generated artifact, grouped by study
  3dfront-batch/     datasets/ runs/ sweep/     the main workflow
  synthetic/         dataset/ runs/ backup/     the synthetic study
  mit/  real_scene/  improve_study/  relight_video/
  ref_lighting/      dclift/                    reference SH light sets
  demos/             cpu/ gpu/                  renderer demo output
  archive/                                      superseded output
```

`idr/paths.py` is the single source of truth for all of these — import the constants
rather than writing repo-relative paths at the call site.

## Entry points

```bash
# decompose a tree of datasets (the main batch driver)
python scripts/decompose_batch.py --datasets_root <dir> --runs_root <dir>

# ... and additionally sweep a directional light across azimuth, with video
python scripts/decompose_batch.py --datasets_root <dir> --runs_root <dir> --relight_video

# where does the wall time go?
python scripts/profile_perf.py --scene <dataset-leaf> --quick

python scripts/sweep_cfg.py          # config sweep
python scripts/synthetic_study.py --phase 1   # synthetic study: render, then --phase 2
python scripts/improve_study.py      # study driver
```

The four `{Cook-Torrance, Phong} x {SH, env-map}` models are reached through one call:

```python
from idr.optim.registry import optimize
from idr.optim.result import EnvGrid

res = optimize("ct_sh", images, normals_hw, frag_pos_hw, mask_hw, cam_pos,
               metallic, roughness, cfg, gt_sh_coeffs=..., gt_albedo=...)
res.albedo, res.sh, res.mat_a          # named, not positional
```

`mat_a`/`mat_b` mean (metallic, roughness) for Cook-Torrance and (shininess, ks) for
Phong; `res.light` is SH coefficients for the `_sh` models and env-map pixels for the
`_env` ones. `res.is_phong` / `res.is_env` say which, so a call site does not have to
know which optimizer produced the result.

## Behaviour-preservation harness

`tests/golden.py` records reference outputs for nine cases covering every branch that
differs — LBFGS, LM with each of the dense/CG/Schur solvers, SH2 and SH3, all four
models, and the full `decompose_scene` pipeline — then re-checks them after a change.

```bash
python tests/golden.py check          # ~2 min, numerical
python tests/test_entrypoints.py       # modules, public names, CLIs still resolve
python tests/golden.py record         # re-baseline (only when a change is intended)
```

Its inputs are **synthesized**, not loaded from a dataset directory: a sphere normal
field with spatially varying maps, rendered through the project's own `shade_ct_sh`. So
it has no dataset or external-drive dependency and reproduces identically anywhere. All
nine cases are bitwise reproducible run-to-run, so the tolerance is `1e-9` rather than a
noise band — any real numeric drift shows up immediately.

## Notebooks

Live notebooks are in `notebooks/`; `notebooks/archive/` holds ones not touched since
before July, kept runnable but out of the way. Each starts by walking up from the CWD to
find the repo root and `chdir`-ing there, so they work whether Jupyter is launched from
the repo root or from `notebooks/`.

**Outputs are stripped from the committed blob.** Committing them had grown `.git` to
4.5 GB against 74 MB of notebooks on disk. Enable the filter once per clone:

```bash
python tools/nbstrip.py --install
```

`.gitattributes` then routes `*.ipynb` through it: git stores 0.4 MB instead of 74 MB
(196x), while your working copies keep their outputs. This only stops *future* growth —
the existing history still holds every past output, and shrinking that needs a history
rewrite (every commit hash changes), so it is a separate deliberate decision.

If the filter is not installed, notebooks commit with outputs and the repo starts growing
again — nothing breaks, but the point is lost.
