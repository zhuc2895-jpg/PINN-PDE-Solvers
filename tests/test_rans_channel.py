"""``ChannelProblem``: the demo posed behind the existing ``PDEProblem`` seam.

The point of this file is that ``pinn.trainer.train`` drives it *unchanged* --
the network corrects a closure, the velocity profile comes out of the coupling
solve, and the shared trainer never learns that anything unusual is going on.
"""

from typing import Optional

import torch

from pinn.model import MLP
from pinn.trainer import TrainConfig, train

from RANS_PINN_Demo.channel import DATA_BUDGETS, ChannelClosure, ChannelProblem, stratified_points
from RANS_PINN_Demo.solver import solve_channel as run_solver
from RANS_PINN_Demo.solver import wall_clustered_grid


def _model(in_dim: int = 3) -> MLP:
    return MLP(in_dim, 1, width=16, depth=2, seed=0)


def test_channel_problem_drives_the_shared_trainer_unchanged():
    """No adapter, no subclass of the trainer: the four protocol methods are the
    whole contract, exactly as they are for the PDE plugins in ``pinn/``."""
    problem = ChannelProblem()
    history = train(
        problem,
        _model(),
        TrainConfig(epochs=3, n_interior=8, device="cpu", verbose=False),
    )
    assert set(history) == {"data", "total"}
    assert len(history["total"]) == 3
    assert all(v == v for v in history["total"])  # no NaN


def test_channel_problem_reports_the_protocol_attributes():
    problem = ChannelProblem()
    assert problem.name
    assert problem.coords == ("y", "Re_tau")
    assert problem.out_dim == 1
    assert problem.loss_weights() == {"data": 1.0}
    assert problem.exact({}) is None

# --- observation placement ----------------------------------------------------


def test_data_budgets_are_nested_not_three_unrelated_clouds():
    """Otherwise the data-volume curve measures placement and volume at once,
    and a difference between budgets cannot be attributed to either."""
    for Re_tau in (180.0, 395.0, 590.0):
        eight = set(stratified_points(8, Re_tau).reshape(-1).tolist())
        twelve = set(stratified_points(12, Re_tau).reshape(-1).tolist())
        sixteen = set(stratified_points(16, Re_tau).reshape(-1).tolist())
        assert eight < twelve < sixteen


def test_observations_reach_from_the_buffer_layer_to_the_centreline():
    Re_tau = 590.0
    y_plus = stratified_points(16, Re_tau).reshape(-1) * Re_tau
    assert y_plus.min() <= 3.0 + 1e-9
    assert y_plus.max() >= 0.97 * Re_tau
    # every band between those ends is represented, not just the extremes
    for lo, hi in ((3.0, 30.0), (30.0, 0.15 * Re_tau), (0.15 * Re_tau, 0.6 * Re_tau)):
        inside = ((y_plus >= lo - 1e-9) & (y_plus <= hi + 1e-9)).sum()
        assert inside >= 2


def test_no_observation_point_lands_on_the_evaluation_grid():
    """Scoring happens on the dense grid; if an observation sat on it, the
    "prediction" error there would be partly a fitted quantity."""
    grid = wall_clustered_grid().reshape(-1)
    for Re_tau in (180.0, 395.0, 590.0):
        for budget in DATA_BUDGETS:
            for value in stratified_points(budget, Re_tau).reshape(-1):
                assert torch.all((grid - value).abs() > 1e-12)


def test_the_budget_has_to_be_one_that_keeps_the_sets_nested():
    import pytest

    with pytest.raises(ValueError, match="nested"):
        stratified_points(10, 395.0)


# --- noise -------------------------------------------------------------------


def test_noise_perturbs_observations_by_roughly_one_percent():
    problem = ChannelProblem(Re_taus=(395.0,), noise=0.01)
    data = problem.sample(16, 0, 0, torch.device("cpu"))
    y = data["interior"].coords["y"]
    observed = data["interior"].targets["U"]
    clean = problem.reference(y, 395.0)

    relative = ((observed - clean).abs() / clean.abs()).max().item()
    assert relative > 0.0
    assert relative < 0.05


def test_the_reference_is_noise_free_and_repeatable():
    problem = ChannelProblem(Re_taus=(395.0,), noise=0.01)
    y = stratified_points(16, 395.0)
    assert torch.equal(problem.reference(y, 395.0), problem.reference(y, 395.0))


def test_observation_noise_is_reproducible_for_a_seed():
    def draw():
        problem = ChannelProblem(Re_taus=(395.0,), seed=7)
        return problem.sample(8, 0, 0, torch.device("cpu"))["interior"].targets["U"]

    assert torch.equal(draw(), draw())

# --- the path has to be differentiable end to end -----------------------------


def test_the_learned_closure_makes_the_inner_solve_a_real_fixed_point():
    """ADR-0001's premise, pinned: because the correction sees the local shear,
    the profile is a fixed point rather than one quadrature.

    If this ever collapses to a single iteration, the coupling story in H3 has
    quietly stopped being about anything.
    """
    problem = ChannelProblem()
    model = MLP(3, 1, width=16, depth=2, seed=0)
    solved = run_solver(
        ChannelClosure(model, problem.base), 395.0, max_iter=problem.solve_max_iter
    )
    assert solved.iterations > 1


def test_data_only_training_reduces_the_observation_loss():
    """T4's acceptance: the seam is not merely drivable, it learns.

    Data loss only -- no RANS residual, no physics term.  Those arrive in T5.
    What is being shown here is that the path from a network parameter, through
    the fixed point, to a solved profile is differentiable end to end.
    """
    problem = ChannelProblem()
    model = MLP(3, 1, width=32, depth=2, seed=0)
    history = train(
        problem,
        model,
        TrainConfig(epochs=60, n_interior=16, device="cpu", verbose=False, lr=1e-2),
    )
    assert history["data"][-1] < 0.25 * history["data"][0]
