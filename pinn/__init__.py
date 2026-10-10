"""PINN-PDE-Solvers: one training loop, four PDEs behind a small seam.

The package is the shared implementation.  Each PDE lives in
``pinn.problems`` and only describes its mathematics (``sample`` / ``losses`` /
``loss_weights`` / ``exact``); the network, the autograd derivatives, the
sampling layouts and the training loop are all shared.
"""

from .derivatives import d, grad_all, laplacian
from .device import resolve_device
from .model import MLP, FourierFeatures
from .problem import Data, PDEProblem, PointSet, inputs_from
from .trainer import TrainConfig, train

__all__ = [
    "MLP",
    "FourierFeatures",
    "TrainConfig",
    "train",
    "resolve_device",
    "d",
    "grad_all",
    "laplacian",
    "Data",
    "PointSet",
    "PDEProblem",
    "inputs_from",
]
