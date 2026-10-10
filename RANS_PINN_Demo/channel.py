"""``ChannelProblem``: the demo posed behind the existing ``PDEProblem`` seam.

The network does not approximate the velocity field -- it corrects a closure.
``U`` comes out of the coupling solve in :mod:`RANS_PINN_Demo.solver`, so the
loss compares *solved* profiles against sparse observations.  That is why
``pinn.trainer.train`` can drive this unchanged: the four protocol methods are
still the whole contract, and the trainer never learns that the field it is
helping to fit is not the field the network emits.

The learned correction sees the local shear, which is what makes the inner
solve a real fixed point rather than a quadrature -- see
``docs/adr/0001-solver-in-the-loop-training.md``.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import torch

from pinn.problem import Data, PointSet

from .ablations import (
    ABLATIONS,
    A1,
    A2,
    A3,
    ALL_OFF,
    AMPLITUDE_LIMIT,
    AMPLITUDE_LOSS_WEIGHT,
    DATA_LOSS_WEIGHT,
    RANS_LOSS_WEIGHT,
    SMOOTH_LOSS_WEIGHT,
    AblationConfig,
)
from .closure import A_PLUS, BaselineClosure, Closure, TruthClosure
from .solver import SolveResult, shear_plus, solve_channel

__all__ = [
    "ChannelClosure",
    "ChannelProblem",
    "stratified_points",
    "MODEL_INPUTS",
    "TRAINING_RE_TAUS",
    "AblationConfig",
    "A1",
    "A2",
    "A3",
    "ALL_OFF",
    "ABLATIONS",
]

#: The network's input layout, in order.  This is the problem's ``coords`` too:
#: the protocol says ``coords`` describes the model's inputs, so
#: ``MLP(len(problem.coords), problem.out_dim)`` has to be the right shape.
MODEL_INPUTS: Tuple[str, ...] = ("y", "shear", "Re_tau")

#: Re_tau is divided by this before the network sees it, so the input is O(1).
RE_TAU_SCALE = 1000.0
#: Relative observation noise.  Applied to training points only.
NOISE_LEVEL = 0.01
#: Reynolds numbers trained on.  Held out: 590.
TRAINING_RE_TAUS = (180.0, 395.0)
#: Data budgets the nested point sets are defined for, per Reynolds number.
DATA_BUDGETS = (8, 12, 16)

#: Floor for the inner solve during training.  The correction is a float32
#: network, so its own output precision puts a floor under the fixed-point
#: residual -- asking for 1e-10 there is asking for something unreachable, and
#: the solve silently burns its whole budget failing to reach it.
TRAINING_SOLVE_TOL = 1e-8
SOLVE_MAX_ITER = 30

#: The additive arm needs a physical scale for its network output.  The
#: structured arm instead treats the same output as a dimensionless log
#: correction, so the two arms keep identical network capacity while respecting
#: their different parameterisations.
ADDITIVE_CORRECTION_SCALE = 1e-3


def _strata(Re_tau: float) -> Tuple[Tuple[float, float], ...]:
    """Four wall-normal bands in wall units, inner to outer.

    Sized as fractions of ``Re_tau``, not as fixed wall-unit edges: at
    ``Re_tau = 180`` a fixed log-layer band of ``(30, 0.15 * Re_tau)`` comes out
    as ``(30, 27)`` -- inverted, and quietly inside the buffer layer.  The
    innermost edge is a wall-unit value so the buffer layer is always resolved.
    """
    return (
        (3.0, 0.15 * Re_tau),
        (0.15 * Re_tau, 0.40 * Re_tau),
        (0.40 * Re_tau, 0.70 * Re_tau),
        (0.70 * Re_tau, 0.98 * Re_tau),
    )


def _band_points(lo: float, hi: float, interior: bool) -> torch.Tensor:
    """Four points across a band, log-spaced (every band edge is positive).

    Interior bands stay strictly inside their band.  Edge bands include their
    outer edge, which is either the wall or the outermost observation.  This is
    what keeps neighbouring bands from emitting the same point -- an earlier
    version shared band edges and spent 3 of its 16 observations on duplicates.
    """
    if interior:
        t = (torch.arange(4, dtype=torch.float64) + 0.5) / 4.0
    else:
        t = torch.linspace(0.0, 1.0, 4, dtype=torch.float64)
    return lo * (hi / lo) ** t


#: Which of each band's four points a budget keeps.  Chosen so the budgets are
#: *nested* -- 8 subset of 12 subset of 16 -- rather than three unrelated clouds.
#: Otherwise the data-volume curve would be measuring placement and volume at
#: the same time, and neither would be attributable.
_NESTED_KEEP = {2: (0, 3), 3: (0, 2, 3), 4: (0, 1, 2, 3)}


def stratified_points(n_per_Re_tau: int, Re_tau: float) -> torch.Tensor:
    """Observation points as ``y/h``, stratified and nested in the budget."""
    if n_per_Re_tau not in DATA_BUDGETS:
        raise ValueError(
            f"data budget must be one of {DATA_BUDGETS} so the sets stay nested, "
            f"got {n_per_Re_tau}"
        )
    keep = _NESTED_KEEP[n_per_Re_tau // 4]
    bands = _strata(Re_tau)
    pieces = [
        _band_points(lo, hi, interior=0 < index < len(bands) - 1)[list(keep)]
        for index, (lo, hi) in enumerate(bands)
    ]
    return (torch.cat(pieces) / Re_tau).reshape(-1, 1)


def _interpolate(x_new: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Piecewise-linear ``y(x_new)`` on a sorted grid; differentiable in ``y``."""
    idx = torch.searchsorted(x, x_new).clamp(1, x.numel() - 1)
    x0, x1 = x[idx - 1], x[idx]
    y0, y1 = y[idx - 1], y[idx]
    weight = (x_new - x0) / (x1 - x0)
    return y0 + weight * (y1 - y0)


def _profile_at(solved: SolveResult, y: torch.Tensor) -> torch.Tensor:
    """The solved profile sampled at ``y``."""
    return _interpolate(
        y.reshape(-1), solved.y.reshape(-1), solved.U.reshape(-1)
    ).reshape(-1, 1)


class ChannelClosure:
    """Ablation-aware learned closure.

    The structured path is ``nu_t = nu_t_base * exp(D * delta)`` with van Driest
    damping ``D = 1 - exp(-y_plus / A_plus)``.  The base closure already carries
    ``D**2``, so the learned factor preserves its zero wall value, its ``y**3``
    near-wall asymptote and its finite non-zero centreline value.  At ``delta = 0``
    the exponential is exactly one and the structured closure is bitwise equal to
    the baseline.

    A1 turns off the structural map and uses the additive arm
    ``nu_t = nu_t_base + scale * delta`` instead.  That arm is intentionally allowed
    to lose positivity; the point of the ablation is to test what the structure
    buys rather than to smuggle it back in through a clamp.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        base: Optional[Closure] = None,
        *,
        structure: bool = True,
    ):
        self.model = model
        self.base = base if base is not None else BaselineClosure()
        self.structure = structure

    def _features(self, y: torch.Tensor, Re_tau: float, shear: Optional[torch.Tensor]) -> torch.Tensor:
        zero_shear = torch.zeros_like(y)
        shear = zero_shear if shear is None else shear
        re_tau = torch.full_like(y, Re_tau / RE_TAU_SCALE)
        return torch.cat([y, shear, re_tau], dim=1).to(
            next(self.model.parameters()).dtype
        )

    def raw_delta(
        self,
        y: torch.Tensor,
        Re_tau: float,
        shear: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """The network output, in its native dimensionless units."""
        delta = self.model(self._features(y, Re_tau, shear))
        return delta.to(y.dtype)

    def delta(
        self,
        y: torch.Tensor,
        Re_tau: float,
        shear: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """The correction used by this parameterisation.

        The structured arm uses the raw log correction.  The additive arm scales
        it to eddy-viscosity units before adding it to the baseline.
        """
        delta = self.raw_delta(y, Re_tau, shear)
        return delta if self.structure else ADDITIVE_CORRECTION_SCALE * delta

    def damping(self, y: torch.Tensor, Re_tau: float) -> torch.Tensor:
        """Van Driest damping for the learned factor, zero at the wall."""
        return 1.0 - torch.exp(-(y * Re_tau) / A_PLUS)

    def __call__(
        self,
        y: torch.Tensor,
        Re_tau: float,
        shear: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        base = self.base(y, Re_tau)
        delta = self.delta(y, Re_tau, shear)
        if not self.structure:
            return base + delta
        return base * torch.exp(self.damping(y, Re_tau) * delta)


def _central_difference(field: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Second-order central difference on a non-uniform grid."""
    return (field[2:] - field[:-2]) / (y[2:] - y[:-2])


def _rans_residual(solved: SolveResult, Re_tau: float) -> torch.Tensor:
    """Mean-square discrete residual of the steady channel momentum equation.

    This intentionally differentiates the *solved* profile rather than reading
    ``_du_dy`` back out of the solver.  Reading the analytic stress relation
    would make the residual identically zero by construction; differentiating
    the quadrature exposes the coarse-grid momentum error that the RANS axis is
    supposed to constrain.
    """
    nu = 1.0 / Re_tau
    shear = _central_difference(solved.U, solved.y)
    flux = (nu + solved.nu_t[1:-1]) * shear
    residual = _central_difference(flux, solved.y[1:-1]) + 1.0
    return (residual**2).mean()


def _smoothness_penalty(delta: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """``mean(|d delta / dy|**2)`` on the non-uniform solve grid."""
    slope = torch.diff(delta, dim=0) / torch.diff(y, dim=0)
    return (slope**2).mean()


def _amplitude_penalty(delta: torch.Tensor, limit: float) -> torch.Tensor:
    """Soft bound: zero inside ``[-limit, limit]``, quadratic outside it."""
    excess = torch.relu(delta.abs() - limit)
    return (excess**2).mean()


def _correction_field(
    closure: ChannelClosure,
    y: torch.Tensor,
    Re_tau: float,
    shear: torch.Tensor,
) -> torch.Tensor:
    """The correction actually applied to the closure.

    The structured arm applies ``D * delta`` inside the exponential, so that is
    the field whose smoothness and amplitude are physically meaningful.  The
    additive A1 arm has no damping factor.
    """
    delta = closure.delta(y, Re_tau, shear)
    if closure.structure:
        return closure.damping(y, Re_tau) * delta
    return delta


class ChannelProblem:
    """A :class:`pinn.problem.PDEProblem`: closure correction on a 1-D channel."""

    name = "1d-channel-closure"
    coords = MODEL_INPUTS
    out_dim = 1

    def __init__(
        self,
        truth: Optional[Closure] = None,
        base: Optional[Closure] = None,
        Re_taus: Sequence[float] = TRAINING_RE_TAUS,
        noise: float = NOISE_LEVEL,
        seed: int = 0,
        solve_max_iter: int = SOLVE_MAX_ITER,
        solve_tol: float = TRAINING_SOLVE_TOL,
        ablation: AblationConfig = AblationConfig(),
        rans_weight: float = RANS_LOSS_WEIGHT,
        smooth_weight: float = SMOOTH_LOSS_WEIGHT,
        amplitude_weight: float = AMPLITUDE_LOSS_WEIGHT,
        amplitude_limit: float = AMPLITUDE_LIMIT,
    ):
        if rans_weight < 0.0 or smooth_weight < 0.0 or amplitude_weight < 0.0:
            raise ValueError("loss weights must be non-negative")
        if amplitude_limit <= 0.0:
            raise ValueError("amplitude_limit must be positive")

        self.truth = truth if truth is not None else TruthClosure()
        self.base = base if base is not None else BaselineClosure()
        self.Re_taus = tuple(Re_taus)
        self.noise = noise
        self.seed = seed
        self.solve_max_iter = solve_max_iter
        self.solve_tol = solve_tol
        self.ablation = ablation
        self.rans_weight = rans_weight
        self.smooth_weight = smooth_weight
        self.amplitude_weight = amplitude_weight
        self.amplitude_limit = amplitude_limit
        self._reference_cache: Dict[float, SolveResult] = {}

    def solved_truth(self, Re_tau: float) -> SolveResult:
        """The truth closure, solved -- cached, since it does not depend on the model."""
        if Re_tau not in self._reference_cache:
            self._reference_cache[Re_tau] = solve_channel(self.truth, Re_tau)
        return self._reference_cache[Re_tau]

    def reference(self, y: torch.Tensor, Re_tau: float) -> torch.Tensor:
        """The profile the observations are drawn from: the truth closure, solved."""
        return _profile_at(self.solved_truth(Re_tau), y)

    def sample(self, n_interior: int, n_boundary: int, n_initial: int, device: torch.device) -> Data:
        """``n_interior`` observations *per* Reynolds number, with noise added.

        The seam's boundary/initial slots are unused: this problem is driven by
        sparse data, not by boundary conditions on a field the network emits.
        """
        generator = torch.Generator().manual_seed(self.seed)
        ys, re_taus, shears, targets = [], [], [], []
        for Re_tau in self.Re_taus:
            solved = self.solved_truth(Re_tau)
            y = stratified_points(n_interior, Re_tau).to(device)
            observed = _profile_at(solved, y).to(device)
            if self.noise:
                observed = observed * (
                    1.0 + self.noise * torch.randn(y.shape, generator=generator, dtype=y.dtype)
                )
            ys.append(y)
            re_taus.append(torch.full_like(y, Re_tau))
            shears.append(_interpolate(
                y.reshape(-1),
                solved.y.reshape(-1),
                shear_plus(solved.nu_t, solved.y, Re_tau).reshape(-1),
            ).reshape(-1, 1))
            targets.append(observed)
        return {
            "interior": PointSet(
                {"y": torch.cat(ys), "shear": torch.cat(shears), "Re_tau": torch.cat(re_taus)},
                {"U": torch.cat(targets)},
            )
        }

    def losses(self, model: torch.nn.Module, data: Data) -> Dict[str, torch.Tensor]:
        device = next(model.parameters()).device
        if device.type != "cpu":
            raise ValueError(
                "this problem solves on the CPU, matching the repo default "
                "(see pinn/device.py); got a model on " + device.type
            )

        observations = data["interior"]
        y_obs = observations.coords["y"]
        Re_obs = observations.coords["Re_tau"]
        U_observed = observations.targets["U"]
        closure = ChannelClosure(model, self.base, structure=self.ablation.structure)

        data_terms = []
        rans_terms = []
        smooth_terms = []
        amplitude_terms = []

        for Re_tau in self.Re_taus:
            selected = (Re_obs == Re_tau).reshape(-1)
            solved = solve_channel(
                closure,
                Re_tau,
                max_iter=self.solve_max_iter,
                tol=self.solve_tol,
            )
            predicted = _profile_at(solved, y_obs[selected])
            data_terms.append(((predicted - U_observed[selected]) ** 2).mean())

            if self.ablation.rans_residual:
                rans_terms.append(_rans_residual(solved, Re_tau))

            if self.ablation.smoothness or self.ablation.structure:
                shear = shear_plus(solved.nu_t, solved.y, Re_tau)
                delta = _correction_field(closure, solved.y, Re_tau, shear)
                if self.ablation.smoothness:
                    smooth_terms.append(_smoothness_penalty(delta, solved.y))
                if self.ablation.structure:
                    amplitude_terms.append(_amplitude_penalty(delta, self.amplitude_limit))

        terms = {"data": torch.stack(data_terms).mean()}
        if self.ablation.rans_residual:
            terms["rans"] = torch.stack(rans_terms).mean()
        if self.ablation.smoothness:
            terms["smooth"] = torch.stack(smooth_terms).mean()
        if self.ablation.structure:
            terms["amplitude"] = torch.stack(amplitude_terms).mean()
        return terms

    def loss_weights(self) -> Dict[str, float]:
        weights = {"data": DATA_LOSS_WEIGHT}
        if self.ablation.rans_residual:
            weights["rans"] = self.rans_weight
        if self.ablation.smoothness:
            weights["smooth"] = self.smooth_weight
        if self.ablation.structure:
            weights["amplitude"] = self.amplitude_weight
        return weights

    @staticmethod
    def exact(coords: Dict[str, torch.Tensor]) -> None:
        """No closed form: the reference is the truth closure, solved."""
        return None
