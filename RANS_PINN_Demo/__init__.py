"""Coarse-grid RANS + physics-constrained neural closure correction.

A standalone feasibility demo.  It reuses ``pinn``'s MLP, trainer, device and
plotting plumbing, but the *problem* it poses is a different shape from the
forward PDE solvers in ``pinn/problems``: the network learns a correction to an
eddy-viscosity closure, and the velocity profile comes out of a RANS solve.

Everything turns on one seam -- :class:`~RANS_PINN_Demo.closure.Closure` -- so
that the baseline, the synthetic truth and the learned correction are
interchangeable under ablation, coupling and evaluation.
"""

from .closure import BaselineClosure, Closure, TruthClosure

__all__ = ["Closure", "BaselineClosure", "TruthClosure"]
