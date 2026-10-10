"""Shared plotting plumbing.

The per-PDE figures stay custom -- each problem has its own story to tell -- but
two things are common to all of them and are wrong-by-default otherwise:

1. the backend must be non-interactive, or a headless run blocks forever on
   ``plt.show()``;
2. the loss history is a dict of equal-length lists and is drawn the same way
   everywhere.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")  # must be set before pyplot is imported anywhere

import matplotlib.pyplot as plt  # noqa: E402

__all__ = ["plt", "plot_loss_history", "save_figure"]


def plot_loss_history(
    ax,
    history: Dict[str, List[float]],
    title: str = "Training Losses",
    keys: Optional[List[str]] = None,
) -> None:
    """Log-scale curves for each loss term, with ``total`` drawn heaviest."""
    keys = keys or [k for k in history if k != "total"]
    ax.semilogy(history["total"], label="Total", color="black", lw=1.2)
    for k in keys:
        ax.semilogy(history[k], label=k.upper(), alpha=0.7)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (log scale)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)


def save_figure(fig, filename: str, dpi: int = 150) -> None:
    """Save and close. No ``plt.show()`` -- that is what hangs in CI."""
    fig.tight_layout()
    fig.savefig(filename, dpi=dpi)
    plt.close(fig)
    print("Saved: " + filename)
