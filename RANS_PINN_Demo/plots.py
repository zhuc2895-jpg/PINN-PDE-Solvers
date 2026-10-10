"""Figures for the closure demo.

Each figure's story is its own; the backend and the save-and-close rule come
from :mod:`pinn.plotting`, so nothing here can hang a headless run.
"""

from __future__ import annotations

from typing import Iterable, Optional, Sequence, Tuple

import torch

from pinn.plotting import plt, save_figure

from .closure import BaselineClosure, Closure, TruthClosure

__all__ = ["closure_comparison"]

DEFAULT_RE_TAUS = (180.0, 395.0, 590.0)


def closure_comparison(
    Re_taus: Sequence[float] = DEFAULT_RE_TAUS,
    truth: Optional[Closure] = None,
    n: int = 400,
):
    """Baseline vs truth ``nu_t`` against ``y_plus``, one panel per ``Re_tau``.

    Log-log on purpose: the whole point of the picture is that the two curves
    are straight lines near the wall with *different slopes*.  On linear axes
    they overlap and the deviation is invisible.
    """
    truth = TruthClosure() if truth is None else truth
    baseline = BaselineClosure()
    y = torch.linspace(1e-6, 1.0, n, dtype=torch.float64).reshape(-1, 1)

    fig, axes = plt.subplots(1, len(Re_taus), figsize=(4.2 * len(Re_taus), 3.8))
    axes = list(axes) if hasattr(axes, "__len__") else [axes]

    for ax, Re_tau in zip(axes, Re_taus):
        y_plus = (y.reshape(-1) * Re_tau).numpy()
        ax.loglog(y_plus, baseline(y, Re_tau).reshape(-1).numpy(), label="baseline")
        ax.loglog(y_plus, truth(y, Re_tau).reshape(-1).numpy(), label="truth")
        ax.set_title(r"$Re_\tau = %g$" % Re_tau)
        ax.set_xlabel(r"$y^+$")
        ax.grid(True, which="both", alpha=0.3)
    axes[0].set_ylabel(r"$\nu_t$")
    axes[0].legend()
    return fig, axes


if __name__ == "__main__":  # pragma: no cover - a demo hook, not logic
    import sys
    from pathlib import Path

    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("RANS_PINN_Demo/figures/closure_profiles.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    figure, _ = closure_comparison()
    save_figure(figure, str(out))