import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def plot_loss_curves(summary_path: Path):
    with open(summary_path) as f:
        summary = json.load(f)

    fig, ax = plt.subplots(figsize=(10, 6))

    for scene, variants in summary.items():
        for variant, data in variants.items():
            loss_history = data.get("loss_history", [])
            if not loss_history:
                continue
            label = f"{scene} / {variant}"
            ax.plot(loss_history, label=label)

    ax.set_yscale("log")
    ax.set_xlabel("Iteration x 200")
    ax.set_ylabel("Loss")
    ax.set_title(
        f"Loss curves — {summary_path.parent.name}/{summary_path.name}")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)

    out_path = summary_path.with_name("loss_curves.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot loss curves from a summary.json")
    parser.add_argument("summary", type=Path, help="Path to summary.json")
    args = parser.parse_args()
    plot_loss_curves(args.summary)
