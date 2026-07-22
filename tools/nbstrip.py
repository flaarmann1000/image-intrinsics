#!/usr/bin/env python
"""Strip outputs from Jupyter notebooks before they are committed.

Notebook outputs are embedded base64 PNGs. Committing them made .git 4.5 GB while the
working tree held only 71 MB of notebooks -- every save of a 36 MB notebook adds another
36 MB blob. This is a git *clean* filter: it rewrites the blob git stores, while your
working copy keeps its outputs, so nothing changes about how you use the notebooks.

Setup (once per clone -- filter definitions live in .git/config and cannot be committed):

    python tools/nbstrip.py --install

Then .gitattributes routes *.ipynb through it automatically.

To check what git would store for a notebook:

    python tools/nbstrip.py < notebooks/foo.ipynb | wc -c

Note this only stops FUTURE growth. The existing history still contains every past
output; shrinking that needs a history rewrite, which changes every commit hash and
should be a separate, deliberate decision.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

# Kernel/runtime metadata that changes on every execution and carries no information.
VOLATILE_NB_META = ("signature", "widgets")
VOLATILE_CELL_META = ("execution", "collapsed", "scrolled", "ExecuteTime")


def strip(nb: dict) -> dict:
    """Clear outputs, execution counts and volatile metadata, in place."""
    for key in VOLATILE_NB_META:
        nb.get("metadata", {}).pop(key, None)
    for cell in nb.get("cells", []):
        if cell.get("cell_type") == "code":
            cell["outputs"] = []
            cell["execution_count"] = None
        for key in VOLATILE_CELL_META:
            cell.get("metadata", {}).pop(key, None)
    return nb


def install() -> int:
    repo = Path(__file__).resolve().parent.parent
    script = Path("tools") / "nbstrip.py"          # repo-relative: works on any clone
    cmd = ["git", "config", "filter.nbstrip.clean", f"python {script.as_posix()}"]
    r = subprocess.run(cmd, cwd=repo)
    if r.returncode != 0:
        print("failed to configure the filter", file=sys.stderr)
        return r.returncode
    # `smudge` is intentionally left unset: git then passes content through unchanged on
    # checkout, which is what we want -- a checked-out notebook simply has no outputs.
    print(f"filter.nbstrip.clean configured in {repo}/.git/config\n"
          f"*.ipynb are now stripped on commit; your working copies keep their outputs.")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--install", action="store_true",
                   help="configure the git clean filter for this clone, then exit")
    args = p.parse_args()
    if args.install:
        return install()

    # git hands the file on stdin and expects the replacement on stdout. Anything that
    # is not a readable notebook must pass through untouched -- a filter that mangles
    # input on error would silently corrupt commits.
    raw = sys.stdin.buffer.read()
    try:
        nb = json.loads(raw.decode("utf-8"))
    except Exception:
        sys.stdout.buffer.write(raw)
        return 0
    out = json.dumps(strip(nb), indent=1, ensure_ascii=False, sort_keys=True)
    sys.stdout.buffer.write((out + "\n").encode("utf-8"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
