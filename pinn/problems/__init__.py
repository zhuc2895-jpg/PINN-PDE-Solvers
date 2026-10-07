"""The four PDEs. Each one is only its mathematics behind the shared seam."""

from .burgers import Burgers1D
from .convection_diffusion import ConvectionDiffusion2D
from .navier_stokes import LidDrivenCavity
from .poisson import Poisson2D

__all__ = ["Burgers1D", "ConvectionDiffusion2D", "LidDrivenCavity", "Poisson2D"]
