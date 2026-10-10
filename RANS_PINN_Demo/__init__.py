"""Coarse-grid RANS + physics-constrained neural closure correction.

A standalone feasibility demo.  It reuses ``pinn``'s MLP, trainer, device and
plotting plumbing, but the *problem* it poses is a different shape from the
forward PDE solvers in ``pinn/problems``: the network learns a correction to an
eddy-viscosity closure, and the velocity profile comes out of a RANS solve.

Everything turns on one seam -- :class:`~RANS_PINN_Demo.closure.Closure` -- so
that the baseline, the synthetic truth and the learned correction are
interchangeable under ablation, coupling and evaluation.
"""

from .ablations import A1, A2, A3, ABLATIONS, ALL_OFF, AblationConfig
from .channel import MODEL_INPUTS, ChannelClosure, ChannelProblem
from .closure import BaselineClosure, Closure, TruthClosure
from .solver import SolveResult, solve_channel, wall_clustered_grid

__all__ = [
    "Closure",
    "BaselineClosure",
    "TruthClosure",
    "SolveResult",
    "solve_channel",
    "wall_clustered_grid",
    "ChannelClosure",
    "ChannelProblem",
    "MODEL_INPUTS",
    "AblationConfig",
    "A1",
    "A2",
    "A3",
    "ALL_OFF",
    "ABLATIONS",
]
