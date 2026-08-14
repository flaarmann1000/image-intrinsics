"""Repository-relative paths — the single source of truth for where things live.

Defined once, here, rather than as `Path(__file__).parents[N]` inside whichever module
happens to need it: N depends on how deeply that module is nested, so those expressions
silently start pointing somewhere else the moment a file moves between directories.

The tree separates inputs from outputs:

    datasets/   source data (3D-Front scans, MIT, office scenes) — read, never written
    assets/     meshes and other static resources
    results/    EVERY generated artifact, grouped by study to mirror idr/pipelines/
"""
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# ── inputs ───────────────────────────────────────────────────────────────────
DATASETS_DIR = REPO_ROOT / "datasets"
ASSETS_DIR = REPO_ROOT / "assets"
MIT_DATA_DIR = DATASETS_DIR / "mit"

# ── outputs ──────────────────────────────────────────────────────────────────
RESULTS_DIR = REPO_ROOT / "results"

# 3D-Front batch study: the main workflow (scripts/decompose_batch.py).
# Name keeps its hyphen — it already appears in VM paths and CLI defaults.
BATCH_DIR = RESULTS_DIR / "3dfront-batch"
BATCH_DATASETS_DIR = BATCH_DIR / "datasets"
BATCH_RUNS_DIR = BATCH_DIR / "runs"
BATCH_SWEEP_DIR = BATCH_DIR / "sweep"

# Synthetic study: phase 1 renders into dataset/, phase 2 writes runs/.
SYNTHETIC_DIR = RESULTS_DIR / "synthetic"
SYNTHETIC_DATASET_DIR = SYNTHETIC_DIR / "dataset"
SYNTHETIC_RUNS_DIR = SYNTHETIC_DIR / "runs"

# SH3 decomposition study (scripts/sh3_decomposition_study.py): two optimizer setups
# over the sh3 dataset's blender/ct ground variants, with per-run results.json.
SH3_STUDY_DIR = RESULTS_DIR / "sh3_study"

MIT_DIR = RESULTS_DIR / "mit"
REAL_SCENE_DIR = RESULTS_DIR / "real_scene"
IMPROVE_STUDY_DIR = RESULTS_DIR / "improve_study"
RELIGHT_VIDEO_DIR = RESULTS_DIR / "relight_video"
DEMOS_DIR = RESULTS_DIR / "demos"
ARCHIVE_DIR = RESULTS_DIR / "archive"

# Reference SH lighting sets consumed by render_3dfront_dataset.
# Points at the DC-lifted set, which is the one that actually exists — the previous
# default (results/ref_sh_lighting) named a directory that had never been created, so
# any call omitting sh_lights_dir failed on a missing path.
REF_LIGHTING_DIR = RESULTS_DIR / "ref_lighting"
DEFAULT_SH_LIGHTS_DIR = REF_LIGHTING_DIR / "dclift"

# ── backwards-compatible aliases ─────────────────────────────────────────────
# The synthetic pipeline used these names before the results/ reorganisation.
SYNTHETIC_ROOT = SYNTHETIC_DIR
DATASET_ROOT = SYNTHETIC_DATASET_DIR
RESULTS_ROOT = SYNTHETIC_RUNS_DIR
