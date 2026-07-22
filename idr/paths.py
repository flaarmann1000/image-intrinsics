"""Repository-relative paths.

Defined once, here, rather than as `Path(__file__).parents[N]` inside whichever module
happens to need it: N depends on how deeply that module is nested, so those expressions
silently start pointing somewhere else the moment a file moves between directories.
"""
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

RESULTS_DIR = REPO_ROOT / "results"
DEFAULT_SH_LIGHTS_DIR = RESULTS_DIR / "ref_sh_lighting"
