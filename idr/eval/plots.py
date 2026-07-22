"""Per-run diagnostic figures written after a decomposition.

Moved out of the batch driver: these render results, they do not orchestrate runs.
Matplotlib is forced to Agg here because these are always called headless (worker
processes, VM batches).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image


def save_intrinsics_plot(run_dir, m, save_path):
    gt = Path(m["scene"])
    panels = [("albedo GT", gt / "albedo.png", None),
              ("albedo est", run_dir / "albedo_scaled.png", None),
              ("albedo err", run_dir / "albedo_err.npy", f"RMSE={m['albedo_rmse']:.3f}"),
              ("rough GT", gt / "roughness.png", None),
              ("rough est", run_dir / "roughness_est.png", None),
              ("rough err", run_dir / "roughness_err.npy", f"MAE={m['roughness_err_mean']:.3f}"),
              ("metal GT", gt / "metallic.png", None),
              ("metal est", run_dir / "metallic_est.png", None),
              ("metal err", run_dir / "metallic_err.npy", f"MAE={m['metallic_err_mean']:.3f}")]
    fig, ax = plt.subplots(3, 3, figsize=(9.5, 9.5))
    for a, (t, p, sub) in zip(ax.flat, panels):
        if Path(p).exists():
            im = np.load(p) if str(p).endswith(".npy") else np.array(Image.open(p))
            if str(p).endswith(".npy"):
                im = im.squeeze()
                if im.ndim == 3 and im.shape[-1] == 3:
                    im = im.mean(-1)
            a.imshow(im, cmap="gray" if im.ndim == 2 else None)
        a.set_title(t + (f"\n{sub}" if sub else ""), fontsize=8); a.axis("off")
    fig.suptitle(run_dir.name[:70], fontsize=8); plt.tight_layout()
    fig.savefig(save_path, dpi=80); plt.close(fig)

def save_relight_plots(run_dir, m, ds_dir, downsample):
    keys = m.get("relight_keys", [])
    rmses, maes = m.get("relight_rmse_per_light", []), m.get("relight_mae_per_light", [])
    pdir = run_dir / "relight" / "plots"; pdir.mkdir(parents=True, exist_ok=True)
    for k, key in enumerate(keys):
        relit_p = run_dir / "relight" / f"relit_{key}.npy"
        tgt_p = ds_dir / f"{key}.npy"
        if not relit_p.exists() or not tgt_p.exists():
            continue
        relit = np.load(relit_p)
        tgt = np.load(tgt_p)[::downsample, ::downsample]
        resid = np.abs(relit - tgt).mean(-1)
        fig, ax = plt.subplots(1, 3, figsize=(10, 3.4))
        ax[0].imshow(np.clip(tgt / 2, 0, 1)); ax[0].set_title(f"target {key}", fontsize=8)
        ax[1].imshow(np.clip(relit / 2, 0, 1)); ax[1].set_title("relit (est intrinsics + GT light)", fontsize=8)
        im = ax[2].imshow(resid, cmap="inferno")
        rm = rmses[k] if k < len(rmses) else float("nan")
        ma = maes[k] if k < len(maes) else float("nan")
        ax[2].set_title(f"residual  RMSE={rm:.4f}  MAE={ma:.4f}", fontsize=8)
        plt.colorbar(im, ax=ax[2], fraction=0.046)
        for a in ax:
            a.axis("off")
        plt.tight_layout(); fig.savefig(pdir / f"relight_{key}.png", dpi=80); plt.close(fig)
