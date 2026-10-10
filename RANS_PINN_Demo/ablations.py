"""Fixed physics-axis switches for the T5 model.

The demo has three structural/physics axes:

* ``structure`` -- multiplicative ``exp`` correction and van Driest damping;
* ``rans_residual`` -- discrete momentum residual on the solved profile;
* ``smoothness`` -- penalty on ``d(delta)/dy``.

Turning all three off must reproduce the T4 data-only problem exactly.  The soft
amplitude bound belongs to the structural parameterisation: without the ``exp``
map there is no dimensionless ``delta`` to bound, so A1 and the all-off baseline
drop that term with it.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

__all__ = [
    "AblationConfig",
    "A1",
    "A2",
    "A3",
    "ALL_OFF",
    "ABLATIONS",
    "DATA_LOSS_WEIGHT",
    "RANS_LOSS_WEIGHT",
    "SMOOTH_LOSS_WEIGHT",
    "AMPLITUDE_LOSS_WEIGHT",
    "AMPLITUDE_LIMIT",
]


@dataclass(frozen=True)
class AblationConfig:
    """Which physics axes are active for one training arm."""

    structure: bool = True
    rans_residual: bool = True
    smoothness: bool = True

    @property
    def data_only(self) -> bool:
        return not (self.structure or self.rans_residual or self.smoothness)


#: A1: remove the structural map, leaving the additive closure ``nu_t_base + delta``.
A1 = AblationConfig(structure=False)
#: A2: keep structure and smoothness, remove the RANS residual.
A2 = AblationConfig(rans_residual=False)
#: A3: keep structure and RANS, remove the smoothness penalty.
A3 = AblationConfig(smoothness=False)
#: All physics axes off.  This is the T4 data-only arm, not a second implementation.
ALL_OFF = AblationConfig(structure=False, rans_residual=False, smoothness=False)

ABLATIONS: Mapping[str, AblationConfig] = MappingProxyType(
    {"A1": A1, "A2": A2, "A3": A3, "all_off": ALL_OFF}
)


#: Fixed loss weights.  They are deliberately module constants, not a learned or
#: adaptive weighting scheme: T6 may select values on validation, but each run
#: then freezes whatever it was handed.
DATA_LOSS_WEIGHT = 1.0
RANS_LOSS_WEIGHT = 1e-2
SMOOTH_LOSS_WEIGHT = 1e-3
AMPLITUDE_LOSS_WEIGHT = 1e-2
#: Soft upper bound on ``|delta|``; the penalty is zero inside this interval.
AMPLITUDE_LIMIT = 1.0
