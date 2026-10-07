"""Tests for the shared numeric core: derivatives, model, and the Data seam."""

import math

import torch
import torch.nn as nn

from pinn.derivatives import d, grad_all, laplacian
from pinn.model import MLP
from pinn.problem import PointSet, inputs_from


def test_d_matches_the_analytic_derivative_of_a_cubic():
    x = torch.linspace(-1.0, 1.0, 7).reshape(-1, 1).requires_grad_(True)
    y = x**3
    assert torch.allclose(d(y, x), 3.0 * x**2, atol=1e-6)


def test_laplacian_of_a_known_field_matches_hand_derivation():
    # u = x^3 + y^3  =>  d2u/dx2 + d2u/dy2 = 6x + 6y
    x = torch.rand(16, 1, requires_grad=True)
    y = torch.rand(16, 1, requires_grad=True)
    u = x**3 + y**3
    lap = laplacian(u, {"x": x, "y": y})
    assert torch.allclose(lap, 6.0 * x + 6.0 * y, atol=1e-5)


def test_grad_all_returns_one_entry_per_coordinate():
    x = torch.rand(5, 1, requires_grad=True)
    y = torch.rand(5, 1, requires_grad=True)
    u = torch.sin(x) * torch.cos(y)
    g = grad_all(u, {"x": x, "y": y})
    assert set(g) == {"x", "y"}
    assert torch.allclose(g["x"], torch.cos(x) * torch.cos(y), atol=1e-6)
    assert torch.allclose(g["y"], -torch.sin(x) * torch.sin(y), atol=1e-6)


def test_mlp_returns_the_requested_output_shape():
    model = MLP(2, 3, width=8, depth=2)
    assert model(torch.rand(5, 2)).shape == (5, 3)


def test_fourier_features_are_optional_and_change_the_input_width():
    plain = MLP(2, 1, width=8, depth=2)
    fourier = MLP(2, 1, width=8, depth=2, n_fourier=4)
    assert isinstance(plain.encoder, nn.Identity)
    assert fourier.encoder.out_dim == 8
    assert fourier(torch.rand(5, 2)).shape == (5, 1)


def test_pointset_detaches_targets_so_they_cannot_be_backwarded_twice():
    x = torch.rand(4, 1, requires_grad=True)
    ps = PointSet({"x": x}, {"u": 2.0 * x})
    assert not ps.targets["u"].requires_grad
    assert ps.targets["u"].grad_fn is None


def test_inputs_from_stacks_coordinates_in_the_declared_order():
    coords = {"x": torch.zeros(3, 1), "y": torch.ones(3, 1)}
    stacked = inputs_from(coords, ("x", "y"))
    assert stacked.shape == (3, 2)
    assert torch.all(stacked[:, 0] == 0) and torch.all(stacked[:, 1] == 1)
