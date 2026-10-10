"""T5: structural closure invariants, losses, and ablation configurations."""

from __future__ import annotations

import math

import pytest
import torch
from torch import nn

from pinn.model import MLP
from pinn.trainer import TrainConfig, train

from RANS_PINN_Demo.ablations import (
    A1,
    A2,
    A3,
    ALL_OFF,
    ABLATIONS,
    EXPERIMENT_ARMS,
    FULL,
)
from RANS_PINN_Demo.channel import (
    ADDITIVE_CORRECTION_SCALE,
    ChannelClosure,
    ChannelProblem,
    _amplitude_penalty,
    _rans_residual,
    _smoothness_penalty,
)
from RANS_PINN_Demo.closure import A_PLUS, BaselineClosure
from RANS_PINN_Demo.solver import solve_channel, wall_clustered_grid


class _ZeroModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros(x.shape[0], 1, dtype=x.dtype, device=x.device)


class _ConstantModel(nn.Module):
    def __init__(self, value: float):
        super().__init__()
        self.value = nn.Parameter(torch.tensor(value, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.value.expand(x.shape[0], 1).to(x.dtype)


def _closure(seed: int = 0, structure: bool = True) -> ChannelClosure:
    return ChannelClosure(MLP(3, 1, width=16, depth=2, seed=seed), structure=structure)


def _near_wall_slope(closure: ChannelClosure, Re_tau: float) -> float:
    y = torch.tensor([[1e-4], [2e-4]], dtype=torch.float64)
    nu_t = closure(y, Re_tau)
    return math.log(float(nu_t[1] / nu_t[0])) / math.log(2.0)


# --- structural invariants ----------------------------------------------------


def test_zero_delta_reproduces_the_baseline_bit_for_bit():
    y = wall_clustered_grid(n=101)
    base = BaselineClosure()
    learned = ChannelClosure(_ZeroModel(), base)
    assert torch.equal(learned(y, 395.0), base(y, 395.0))


def test_the_structured_closure_is_positive_inside_the_channel():
    y = wall_clustered_grid(n=801)
    for value in (-50.0, 0.0, 50.0):
        nu_t = ChannelClosure(_ConstantModel(value))(y, 395.0)
        assert nu_t[0].item() == 0.0  # the physical wall condition
        assert torch.all(nu_t[1:] > 0.0)


def test_the_structured_closure_preserves_the_near_wall_cubic_asymptote():
    assert abs(_near_wall_slope(_closure(), 395.0) - 3.0) < 0.05


def test_the_structured_closure_stays_finite_and_non_zero_at_the_centreline():
    y = wall_clustered_grid(n=401)
    nu_t = _closure()(y, 395.0)
    assert torch.isfinite(nu_t[-1])
    assert nu_t[-1].item() > 0.0


def test_the_structural_formula_is_exp_with_van_driest_damping():
    y = wall_clustered_grid(n=101)
    Re_tau = 395.0
    closure = _closure()
    raw = closure.raw_delta(y, Re_tau)
    damping = 1.0 - torch.exp(-(y * Re_tau) / A_PLUS)
    expected = BaselineClosure()(y, Re_tau) * torch.exp(damping * raw)
    assert torch.allclose(closure(y, Re_tau), expected)


def test_a1_uses_the_additive_formula_without_exp_or_damping():
    y = wall_clustered_grid(n=101)
    Re_tau = 395.0
    closure = _closure(structure=False)
    raw = closure.raw_delta(y, Re_tau)
    expected = BaselineClosure()(y, Re_tau) + ADDITIVE_CORRECTION_SCALE * raw
    assert torch.equal(closure(y, Re_tau), expected)


# --- loss terms ---------------------------------------------------------------


def test_the_default_problem_reports_all_four_fixed_loss_terms():
    problem = ChannelProblem()
    model = MLP(3, 1, width=8, depth=2, seed=0)
    data = problem.sample(8, 0, 0, torch.device("cpu"))
    terms = problem.losses(model, data)

    assert set(terms) == {"data", "rans", "smooth", "amplitude"}
    assert set(problem.loss_weights()) == set(terms)
    assert problem.loss_weights() == problem.loss_weights()  # fixed, not stateful
    assert all(torch.isfinite(value) and value >= 0.0 for value in terms.values())


def test_the_amplitude_bound_is_soft_and_quadratic_outside_its_limit():
    delta = torch.tensor([[-2.0], [-0.5], [0.5], [2.0]], dtype=torch.float64)
    expected = torch.tensor(0.5, dtype=torch.float64)  # (1**2 + 1**2) / 4
    assert torch.equal(_amplitude_penalty(delta, 1.0), expected)


def test_a_constant_correction_has_zero_smoothness_penalty():
    delta = torch.full((8, 1), 0.25, dtype=torch.float64)
    y = wall_clustered_grid(n=8)
    assert _smoothness_penalty(delta, y).item() == 0.0


def test_the_rans_term_is_a_live_momentum_residual():
    """The RANS axis must constrain the solved profile, not be a dead zero.
    
    The residual is built from finite differences of the solver's ``U`` rather
    than by reading the analytic stress relation back out, so it has a non-zero
    value and a gradient with respect to the closure network.
    """
    model = MLP(3, 1, width=8, depth=2, seed=0)
    closure = ChannelClosure(model)
    solved = solve_channel(closure, 395.0, max_iter=30, tol=1e-8)
    residual = _rans_residual(solved, 395.0)
    gradients = torch.autograd.grad(residual, model.parameters(), allow_unused=True)

    assert residual.item() > 0.0
    assert any(g is not None and torch.count_nonzero(g).item() > 0 for g in gradients)


# --- ablation configurations --------------------------------------------------


def test_ablation_and_experiment_arm_mappings_have_distinct_roles():
    assert set(ABLATIONS) == {"A1", "A2", "A3", "all_off"}
    assert set(EXPERIMENT_ARMS) == {"full", "A1", "A2", "A3", "all_off"}
    assert EXPERIMENT_ARMS["full"] == FULL


@pytest.mark.parametrize(
    ("config", "expected_terms"),
    [
        (FULL, {"data", "rans", "smooth", "amplitude"}),
        (A1, {"data", "rans", "smooth"}),
        (A2, {"data", "smooth", "amplitude"}),
        (A3, {"data", "rans", "amplitude"}),
        (ALL_OFF, {"data"}),
    ],
)
def test_each_ablation_removes_only_the_named_axis(config, expected_terms):
    problem = ChannelProblem(ablation=config)
    model = MLP(3, 1, width=8, depth=2, seed=0)
    data = problem.sample(8, 0, 0, torch.device("cpu"))
    assert set(problem.loss_weights()) == expected_terms
    assert set(problem.losses(model, data)) == expected_terms


def test_all_off_reuses_the_data_only_path(monkeypatch):
    problem = ChannelProblem(ablation=ALL_OFF)
    model = MLP(3, 1, width=8, depth=2, seed=0)
    data = problem.sample(8, 0, 0, torch.device("cpu"))

    def unexpected(*args, **kwargs):
        raise AssertionError("disabled physics term was evaluated")

    monkeypatch.setattr("RANS_PINN_Demo.channel._rans_residual", unexpected)
    monkeypatch.setattr("RANS_PINN_Demo.channel._smoothness_penalty", unexpected)
    monkeypatch.setattr("RANS_PINN_Demo.channel._amplitude_penalty", unexpected)

    assert problem.ablation.data_only
    assert problem.loss_weights() == {"data": 1.0}
    assert set(problem.losses(model, data)) == {"data"}


@pytest.mark.parametrize("name", list(ABLATIONS))
def test_every_ablation_config_can_actually_train(name):
    problem = ChannelProblem(ablation=ABLATIONS[name], solve_max_iter=20)
    model = MLP(3, 1, width=8, depth=2, seed=0)
    history = train(
        problem,
        model,
        TrainConfig(epochs=2, n_interior=8, device="cpu", verbose=False),
    )

    assert set(history) == set(problem.loss_weights()) | {"total"}
    assert all(math.isfinite(value) for values in history.values() for value in values)
