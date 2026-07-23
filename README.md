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

## Optimizers

`cfg["optimizer"]` selects one of four, all reached through `idr.optim.registry.optimize`:

| | what it does |
|---|---|
| `LBFGS` / `Adam` | gradient descent over every unknown jointly |
| `LM` | Levenberg-Marquardt, `ct_sh` only (`idr/optim/lm/`) |
| `VARPRO` | variable projection, `ct_sh` only (`idr/optim/varpro/`) |

**Variable projection** exploits the fact that, with the material fixed, the render is
*linear* in the SH lighting. It solves the lighting in closed form each iteration and
takes a lighting-projected Gauss-Newton step over the per-pixel material only, so the
lighting never has to be descended into.

```python
cfg["optimizer"]     = "VARPRO"
cfg["varpro_space"]  = "natural"      # or "transformed"
```

`natural` optimises physical values in a box `[0,1]³ × [0.03,1] × [0,1]`; `transformed`
optimises the raw parameters so `tr_albedo` / `tr_metallic` / `tr_roughness` are honoured.
Both are regression-locked as golden cases J and K.

Warm-starting VarPro from a gradient stage is pure config — the curriculum forwards any
cfg key to a phase, and `scripts/decompose_batch.py` takes it as JSON:

```bash
python scripts/decompose_batch.py --optimizer VARPRO --varpro_space natural --double 0 ...

python scripts/decompose_batch.py --double 0 --curriculum   '[{"optimizer":"LBFGS","n_iter":200},{"optimizer":"VARPRO","n_iter":30}]' ...
```

**Use `--double 0` with VarPro.** fp64 is ~7x slower here for no gain (measured: albedo
RMSE 0.0392 fp64 vs 0.0354 fp32 at the same budget) because VarPro's cost is dominated by
dense linear algebra. The caveat is that the reference implementation warns fp32 is
unverified in the low-roughness band; roughness recovered cleanly in these runs, but that
band is where to look first if a result seems off.

Measured on `1f19c3ef_v2/ct-ct_sh-frOn_env` (128² , 75 images, 40 iterations):

| | albedo RMSE | roughness MAE | metallic MAE | recon | relight |
|---|---|---|---|---|---|
| LBFGS | 0.1075 | 0.3057 | 0.2326 | 0.0083 | 0.2262 |
| VarPro | **0.0438** | **0.0742** | **0.0044** | 0.0052 | **0.1139** |
| LBFGS → VarPro | 0.0437 | 0.1055 | 0.0071 | **0.0028** | 0.1729 |

VarPro costs roughly 2x the wall time per run here and recovers metallic ~50x better —
LBFGS barely moves it off its initialisation.

### Reproducing the reference project's recipe

The parallel prototype (`Documents/intrinsic_decomposition`) settles on one "baseline
recipe", identical across `run_baseline_sweep.sh` and `run_baseline_sweep_extra.sh`: a
long Adam run, then a short VarPro polish from its checkpoint. Its settings, and how they
map here:

| reference | value | here |
|---|---|---|
| GD optimizer / lr | Adam, `0.05` | `optimizer=Adam`, `lr=0.05` |
| GD iterations | 60k (indoor) / 20k (3D-Front) | `n_iter` |
| images | 16 | `n_train` |
| dtype | float32 | `--double 0` |
| material parameterization | sigmoid reparam | `tr_*=sigmoid` |
| albedo init | mean image | (default) |
| roughness init | 0.5 | `init_roughness=0.5` |
| metallic init | 0.05 | `init_metallic=0.05` |
| SH init | DC 1.5, rest 0 | (default) |
| regularizers | none | all `lambda_*=0` |
| VarPro iterations | 20 | `n_iter` of the final stage |
| VarPro space | box `[0,0,0,0.03,0]..[1,1,1,1,1]` | `varpro_space=natural` |
| line search | `1.0, 0.5, 0.25, 0.125` | (default) |
| damping ceiling | `1e10` | `varpro_lam_ceiling` (default) |

```bash
python scripts/decompose_batch.py --datasets_root <dir> --runs_root <dir> \
    --double 0 --n_train 16 --optimizer VARPRO --varpro_space natural --n_iter 20 \
    --curriculum '[{"optimizer":"Adam","lr":0.05,"n_iter":20000,
                    "tr_albedo":"sigmoid","tr_metallic":"sigmoid","tr_roughness":"sigmoid",
                    "init_metallic":0.05,"init_roughness":0.5,
                    "lambda_tv":0,"lambda_sparse":0,"lambda_white":0}]'
```

**Three VarPro refinements are not implemented here**, so this reproduces the recipe's
shape and hyperparameters but not its polish step exactly:

- **profiled albedo** (`--n-inner-rho 10`): an inner loop re-solving albedo in closed form
  each iteration, exploiting the render being linear in it
- **per-pixel damping**: `lam0 = 1e-3 * diagH.max(dim=1)` per pixel, versus one scalar here
- **per-pixel accept/reject** with a Nielsen gain-ratio update
  (`lam * clamp_min(1 - (2*rho-1)^3, 1/3)`), versus a global "did the total loss drop"

They were deferred as "Stage B" because VarPro already beat LBFGS clearly without them.
Expect this configuration to track the reference qualitatively, not numerically.

One further difference: the reference sums its per-image loss over images where this repo
means over them, a factor of `n_images`. Adam is per-parameter scale-invariant, so
`lr=0.05` still transfers.

## Behaviour-preservation harness

`tests/golden.py` records reference outputs for eleven cases covering every branch that
differs — LBFGS, LM with each of the dense/CG/Schur solvers, VarPro in both parameter
spaces, SH2 and SH3, all four
models, and the full `decompose_scene` pipeline — then re-checks them after a change.

```bash
python tests/golden.py check          # ~2 min, numerical
python tests/test_entrypoints.py       # modules, public names, CLIs still resolve
python tests/golden.py record         # re-baseline (only when a change is intended)
```

Its inputs are **synthesized**, not loaded from a dataset directory: a sphere normal
field with spatially varying maps, rendered through the project's own `shade_ct_sh`. So
it has no dataset or external-drive dependency and reproduces identically anywhere. All
eleven cases are bitwise reproducible run-to-run, so the tolerance is `1e-9` rather than a
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
