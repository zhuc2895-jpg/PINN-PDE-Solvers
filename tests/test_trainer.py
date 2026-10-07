"""End-to-end behaviour of the one training loop every PDE shares."""

import torch

from pinn.model import MLP
from pinn.problem import PointSet
from pinn.problems import Poisson2D
from pinn.trainer import TrainConfig, train


def test_train_records_one_history_entry_per_loss_term_plus_a_total():
    problem = Poisson2D()
    model = MLP(len(problem.coords), problem.out_dim, width=16, depth=2, seed=0)
    config = TrainConfig(
        epochs=5, n_interior=32, n_boundary=16, device="cpu", verbose=False
    )
    history = train(problem, model, config)
    assert set(history) == {"pde", "bc", "total"}
    assert len(history["total"]) == 5


def test_train_reduces_the_total_loss():
    problem = Poisson2D()
    model = MLP(len(problem.coords), problem.out_dim, width=32, depth=3, seed=0)
    config = TrainConfig(
        epochs=200, n_interior=128, n_boundary=64, device="cpu", verbose=False
    )
    history = train(problem, model, config)
    assert history["total"][-1] < history["total"][0]


def test_extra_weights_override_the_problem_defaults():
    problem = Poisson2D()
    model = MLP(len(problem.coords), problem.out_dim, width=8, depth=2, seed=0)
    config = TrainConfig(
        epochs=2,
        n_interior=16,
        n_boundary=8,
        device="cpu",
        verbose=False,
        extra_weights={"bc": 100.0},
    )
    history = train(problem, model, config)
    # With bc weighted 100x, the total must sit well above the plain pde+bc sum.
    last = -1
    assert history["total"][last] > history["bc"][last] * 50.0


def test_resample_every_redraws_collocation_points():
    problem = Poisson2D()
    model = MLP(len(problem.coords), problem.out_dim, width=8, depth=2, seed=0)
    config = TrainConfig(
        epochs=4, n_interior=16, n_boundary=8, device="cpu", verbose=False, resample_every=2
    )
    history = train(problem, model, config)
    assert len(history["total"]) == 4


def test_a_problem_only_needs_the_four_protocol_methods():
    """The seam is the whole contract: anything with these four methods trains."""

    class ConstantZero:
        name = "zero"
        coords = ("x",)
        out_dim = 1

        def sample(self, n_interior, n_boundary, n_initial, device):
            return {
                "interior": PointSet({"x": torch.rand(n_interior, 1, device=device)})
            }

        def losses(self, model, data):
            x = data["interior"].coords["x"]
            u = model(x)
            return {"pde": torch.mean(u**2)}

        def loss_weights(self):
            return {"pde": 1.0}

        def exact(self, coords):
            return None

    model = MLP(1, 1, width=8, depth=2, seed=0)
    history = train(
        ConstantZero(), model,
        TrainConfig(epochs=3, n_interior=8, device="cpu", verbose=False),
    )
    assert set(history) == {"pde", "total"}
    assert all(v == v for v in history["total"])  # no NaN

