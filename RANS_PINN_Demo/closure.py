"""The one seam: an eddy-viscosity closure as a callable.

Units.  The channel is non-dimensionalised with ``h = 1``, ``u_tau = 1`` and
``nu = 1 / Re_tau``.  The wall-normal coordinate ``y`` is therefore ``y / h`` in
``[0, 1]``, ``y_plus = y * Re_tau``, and the returned ``nu_t`` carries the same
units as ``nu``.

Everything the demo compares -- baseline closure, synthetic truth, and later the
learned correction -- implements :class:`Closure`, so ablations, the coupling
solve and evaluation all swap implementations rather than branching.

``shear`` is the local mean shear of the current iterate.  Prescribed closures
ignore it; a learned correction needs it, because a closure that depends only on
``y`` would make the RANS equation a single quadrature and the "fixed-point
iteration" a fiction (see ``docs/adr/0001-solver-in-the-loop-training.md``).
"""

from __future__ import annotations

import math
from typing import Optional, Protocol, runtime_checkable

import torch

__all__ = ["Closure", "BaselineClosure", "TruthClosure", "KAPPA", "A_PLUS", "C_CORE"]

#: von Karman constant.
KAPPA = 0.41
#: van Driest damping constant, in wall units.
A_PLUS = 26.0
#: Outer-layer plateau coefficient: keeps nu_t finite at the centreline, where
#: the mixing-length term ``y (1 - y)`` vanishes but the real profile does not.
C_CORE = 0.06

#: The truth's damping law differs from the baseline's in *both* of its
#: parameters.  The exponent is what breaks the ``y**3`` asymptote; the constant
#: sets how far out the damping keeps biting.  Changing only the exponent moves
#: a sliver of the profile and a fit could hide it.
TRUTH_DAMPING_A_PLUS = 45.0
TRUTH_DAMPING_EXPONENT = 2.6

#: Fractional drift of the log-layer slope per e-fold of ``Re_tau`` away from
#: ``TRUTH_KAPPA_REF``.
TRUTH_KAPPA_DRIFT = 0.35
TRUTH_KAPPA_REF = 300.0

#: The log layer, in wall units: ``[LOG_LAYER_LO, LOG_LAYER_FRACTION * Re_tau]``,
#: with a smooth taper of ``LOG_LAYER_TAPER`` wall units at each edge.
LOG_LAYER_LO = 30.0
LOG_LAYER_FRACTION = 0.2
LOG_LAYER_TAPER = 10.0


@runtime_checkable
class Closure(Protocol):
    """Eddy viscosity ``nu_t(y, Re_tau)`` on a half channel, wall at ``y = 0``."""

    def __call__(
        self,
        y: torch.Tensor,
        Re_tau: float,
        shear: Optional[torch.Tensor] = None,
    ) -> torch.Tensor: ...


def _damping(y_plus: torch.Tensor, exponent: float = 2.0, a_plus: float = A_PLUS) -> torch.Tensor:
    """van Driest damping ``(1 - exp(-y_plus / a_plus))**exponent``.

    At the baseline parameters (exponent 2) this is what turns the mixing
    length's single power of ``y`` into the ``y**3`` viscous-sub-layer asymptote.
    """
    return (1.0 - torch.exp(-y_plus / a_plus)) ** exponent


def _log_layer_window(
    y_plus: torch.Tensor,
    Re_tau: float,
    lo: float = LOG_LAYER_LO,
    fraction: float = LOG_LAYER_FRACTION,
    taper: float = LOG_LAYER_TAPER,
) -> torch.Tensor:
    """A smooth 0-to-1 window over the log layer.

    Localising the slope drift here is what stops it from being a constant
    factor: scaling ``kappa`` over the whole profile is just scaling ``nu_t``,
    which any correction can absorb for free.
    """
    return torch.sigmoid((y_plus - lo) / taper) * torch.sigmoid(
        (fraction * Re_tau - y_plus) / taper
    )


class BaselineClosure:
    """The coarse-grid closure the correction is measured against.

    Deliberately *not* the textbook ``kappa y (1 - y) u_tau``: that vanishes at
    the centreline, and a multiplicative correction cannot repair an error where
    its own factor is zero.  The ``C_CORE`` term supplies a finite centreline
    value while the mixing-length term still dominates the log layer.
    """

    def __call__(
        self,
        y: torch.Tensor,
        Re_tau: float,
        shear: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        y_hat = y
        shape = KAPPA * y_hat * (1.0 - y_hat) + C_CORE * y_hat**2
        return shape * _damping(y_hat * Re_tau)


class TruthClosure:
    """Synthetic ground truth, so H1 ("a learnable residual exists") is testable
    before any DNS data enters the picture.

    Two deviations from the baseline, each switchable, each deliberately one a
    constant multiplier cannot absorb:

    * ``near_wall_deviation`` gives the truth a different damping law, which
      breaks the ``y**3`` asymptote the baseline's exponent of 2 produces.
    * ``log_layer_deviation`` lets the log-layer slope drift with ``Re_tau``,
      windowed to the log layer so it is a change of *shape*, not of scale.

    Both off reproduces :class:`BaselineClosure` bit for bit.
    """

    def __init__(
        self,
        near_wall_deviation: bool = True,
        log_layer_deviation: bool = True,
        damping_a_plus: float = TRUTH_DAMPING_A_PLUS,
        damping_exponent: float = TRUTH_DAMPING_EXPONENT,
        kappa_drift: float = TRUTH_KAPPA_DRIFT,
        kappa_ref: float = TRUTH_KAPPA_REF,
    ):
        self.near_wall_deviation = near_wall_deviation
        self.log_layer_deviation = log_layer_deviation
        self.damping_a_plus = damping_a_plus
        self.damping_exponent = damping_exponent
        self.kappa_drift = kappa_drift
        self.kappa_ref = kappa_ref

    def kappa(self, Re_tau: float) -> float:
        """The log-layer slope the truth uses at this ``Re_tau``.

        Equals :data:`KAPPA` at :data:`TRUTH_KAPPA_REF`, so the deviation is
        required to vanish at one Reynolds number rather than merely differ.
        """
        if not self.log_layer_deviation:
            return KAPPA
        return KAPPA * (1.0 + self.kappa_drift * math.log(Re_tau / self.kappa_ref))

    def __call__(
        self,
        y: torch.Tensor,
        Re_tau: float,
        shear: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        y_hat = y
        y_plus = y_hat * Re_tau

        if self.near_wall_deviation:
            a_plus = self.damping_a_plus
            exponent = self.damping_exponent
        else:
            a_plus = A_PLUS
            exponent = 2.0

        kappa = self._effective_kappa(y_plus, Re_tau)
        shape = kappa * y_hat * (1.0 - y_hat) + C_CORE * y_hat**2
        return shape * _damping(y_plus, exponent=exponent, a_plus=a_plus)

    def _effective_kappa(self, y_plus: torch.Tensor, Re_tau: float):
        if not self.log_layer_deviation:
            return KAPPA
        k_true = self.kappa(Re_tau)
        return KAPPA + (k_true - KAPPA) * _log_layer_window(y_plus, Re_tau)