"""The Closure seam: baseline and synthetic-truth eddy-viscosity closures.

These pin the *structural* properties the demo's argument rests on -- not "does
the number look right", but "does the closure have the asymptotics we claim".
A baseline with the wrong near-wall exponent would make H1 unfalsifiable before
any training happens, so the asymptotics are checked against the analytic limit
rather than against a recorded output.
"""

import math

import torch

from RANS_PINN_Demo.closure import KAPPA, BaselineClosure, TruthClosure

RE_TAU_TRAIN = (180.0, 395.0)
RE_TAU_TEST = 590.0


def _y(n: int = 200) -> torch.Tensor:
    return torch.linspace(0.0, 1.0, n).reshape(-1, 1)


def test_closure_returns_one_eddy_viscosity_per_point():
    y = _y(9)
    assert BaselineClosure()(y, 395.0).shape == (9, 1)


def test_baseline_is_zero_at_the_wall_and_positive_just_above_it():
    y = torch.tensor([[0.0], [1e-3], [0.5]])
    nu_t = BaselineClosure()(y, 395.0)
    assert nu_t[0].item() == 0.0
    assert torch.all(nu_t[1:] > 0)


def test_baseline_near_wall_mixing_length_grows_like_y_cubed():
    """``nu_t ~ y**3`` is the viscous-sub-layer asymptote.

    The mixing length contributes one power of ``y``; the van Driest damping
    supplies the other two.  Measured as a log-log slope between two points well
    inside ``y_plus << A_plus``, so the constant prefactor drops out.
    """
    y = torch.tensor([[1e-4], [2e-4]], dtype=torch.float64)
    nu_t = BaselineClosure()(y, 395.0)
    slope = (torch.log(nu_t[1]) - torch.log(nu_t[0])) / math.log(2.0)
    assert abs(slope.item() - 3.0) < 0.05


def test_baseline_does_not_collapse_at_the_centreline():
    """The textbook ``kappa*y*(1-y)*u_tau`` vanishes at the centre.

    That is exactly the error a *multiplicative* correction can never repair --
    multiplying zero by anything is still zero -- so the baseline carries a
    finite centreline value instead.
    """
    y = _y(401).double()
    nu_t = BaselineClosure()(y, 395.0)
    assert nu_t[-1].item() > 0.0
    # A quarter of the peak is a guard against "fixing" the collapse by adding
    # an epsilon, not a physical constant: the real ratio here is 0.5, the peak
    # sits at y/h ~ 0.55.
    assert nu_t[-1].item() > 0.25 * nu_t.max().item()

def test_truth_with_both_deviations_off_is_exactly_the_baseline():
    """Both deliberate deviations off must reproduce the baseline bit for bit.

    If this only held approximately, every ablation would be measuring the gap
    between two spellings of the same closure rather than the deviations we
    actually named.
    """
    y = _y(101)
    truth = TruthClosure(near_wall_deviation=False, log_layer_deviation=False)
    assert torch.equal(truth(y, 395.0), BaselineClosure()(y, 395.0))

# --- the two deliberate deviations, and why neither is a fitted constant -------


def test_truth_breaks_the_near_wall_y_cubed_asymptote():
    """The baseline's asymptote is ``y**3``; the truth's damping law is a
    different function, so its asymptote must differ.  A measured slope is the
    only way to pin this -- the two profiles look alike on a plot of y vs nu_t.
    """
    y = torch.tensor([[1e-4], [2e-4]], dtype=torch.float64)
    slope = {}
    for name, closure in (
        ("baseline", BaselineClosure()),
        ("truth", TruthClosure(log_layer_deviation=False)),
    ):
        nu_t = closure(y, 395.0)
        slope[name] = (torch.log(nu_t[1]) - torch.log(nu_t[0])).item() / math.log(2.0)
    assert abs(slope["baseline"] - 3.0) < 0.05
    assert slope["truth"] > slope["baseline"] + 0.3


def test_log_layer_deviation_is_inert_at_the_reference_reynolds_number():
    """``kappa(Re_tau_ref) == KAPPA``, so there the log-layer switch does nothing.

    This is what makes the Re_tau story falsifiable: the deviation has to vanish
    at one Reynolds number and grow away from it, not just be "different".
    """
    y = _y(64).double()
    drifting = TruthClosure(near_wall_deviation=False, log_layer_deviation=True)
    assert drifting.kappa(300.0) == KAPPA
    assert torch.equal(drifting(y, 300.0), BaselineClosure()(y, 300.0))
    assert drifting.kappa(590.0) > KAPPA > drifting.kappa(180.0)


def test_log_layer_deviation_lives_in_the_log_layer_only():
    """Scaling ``kappa`` globally would just be scaling ``nu_t`` by a constant.

    A constant is exactly what the correction could absorb for free, and it is
    what would make the whole "does physics buy extrapolation?" question
    vacuous.  So the drift is windowed to the log layer: outside it the truth
    and the baseline must agree exactly.
    """
    y = _y(401).double()
    grid = y.reshape(-1)
    truth = TruthClosure(near_wall_deviation=False, log_layer_deviation=True)(y, 590.0)
    base = BaselineClosure()(y, 590.0)

    outer = grid > 0.6
    outer_gap = ((truth[outer] - base[outer]).abs() / base[outer].abs()).max().item()
    assert outer_gap < 1e-6

    # ...and it must actually do something where the log layer sits.
    log_layer = (grid > 0.08) & (grid < 0.18)
    log_gap = ((truth[log_layer] - base[log_layer]).abs() / base[log_layer].abs()).max().item()
    assert log_gap > 1e-3


def test_truth_residual_survives_an_optimal_constant_rescaling():
    """The residual must not be in the one-parameter family ``k * nu_t_base``.

    The bar is a floor, not a claim about effect size: after the *best* constant
    is fitted, what is left is still orders of magnitude above float64 round-off
    (which is what a genuine ``k * nu_t_base`` would leave).
    """
    y = _y(401).double()
    base = BaselineClosure()(y, 400.0)
    truth = TruthClosure()(y, 590.0)
    base = BaselineClosure()(y, 590.0)
    k = (base * truth).sum() / (base * base).sum()
    leftover = (torch.norm(truth - k * base) / torch.norm(truth)).item()
    assert leftover > 0.02

def test_closure_comparison_draws_one_panel_per_reynolds_number():
    """The T1 deliverable is a picture someone can look at; this only pins that
    the picture gets made for every requested Re_tau."""
    from pinn.plotting import plt

    from RANS_PINN_Demo.plots import closure_comparison

    fig, axes = closure_comparison(Re_taus=(180.0, 590.0))
    try:
        assert len(axes) == 2
        assert all(len(ax.lines) == 2 for ax in axes)
    finally:
        plt.close(fig)