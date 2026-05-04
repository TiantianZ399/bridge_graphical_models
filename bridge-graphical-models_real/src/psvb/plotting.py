"""Plotting helpers for toy diagnostics."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from numpy.typing import NDArray
import numpy as np

Array = NDArray[np.float64]


def plot_toy_projection_gap(
    z: Array,
    x_target: Array,
    times: Array,
    independent_gaps: Array,
    ot_gaps: Array,
    independent_auc: float,
    ot_auc: float,
    out_dir: str | Path,
    stem: str = "toy_projection_gap",
) -> tuple[Path, Path]:
    """Create the two-panel diagnostic figure used in the paper."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(6.5, 2.5))
    ax1 = fig.add_axes([0.08, 0.18, 0.34, 0.73])
    ax1.scatter(z[:, 0], z[:, 1], s=3, alpha=0.25, label="base")
    ax1.scatter(x_target[:, 0], x_target[:, 1], s=3, alpha=0.25, label="target")
    ax1.set_title("Toy source/target")
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_aspect("equal")
    ax1.legend(frameon=False, markerscale=2, fontsize=7, loc="upper right")

    ax2 = fig.add_axes([0.52, 0.18, 0.43, 0.73])
    ax2.plot(times, independent_gaps, marker="o", label=f"independent, AUC={independent_auc:.3f}")
    ax2.plot(times, ot_gaps, marker="s", label=f"minibatch OT, AUC={ot_auc:.3f}")
    ax2.set_xlabel("bridge time t")
    ax2.set_ylabel("estimated Markovization gap")
    ax2.set_title("Coupling changes causalization difficulty")
    ax2.legend(frameon=False, fontsize=7)
    ax2.grid(True, alpha=0.25)

    pdf_path = out_dir / f"{stem}.pdf"
    png_path = out_dir / f"{stem}.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return pdf_path, png_path
