"""Coarse-grid RANS + physics-constrained neural closure correction.

A standalone feasibility demo.  It reuses ``pinn``'s MLP, trainer, device and
plotting plumbing, but the *problem* it poses is a different shape from the
forward PDE solvers in ``pinn/problems``: the network learns a correction to an
eddy-viscosity closure, and the velocity profile comes out of a RANS solve.

Everything turns on one seam -- :class:`~RANS_PINN_Demo.closure.Closure` -- so
that the baseline, the synthetic truth and the learned correction are
interchangeable under ablation, coupling and evaluation.
"""

from .ablations import (
    A1,
    A2,
    A3,
    ABLATIONS,
    ALL_OFF,
    EXPERIMENT_ARMS,
    FULL,
    AblationConfig,
)
from .channel import MODEL_INPUTS, ChannelClosure, ChannelProblem
from .closure import BaselineClosure, Closure, TruthClosure
from .experiment import (
    ABLATION_RELATIVE_ERROR_INCREASE_TARGET,
    EXTRAPOLATION_BUDGET,
    EXTRAPOLATION_IMPROVEMENT_TARGET,
    TEST_RE_TAU,
    ExperimentResult,
    Hyperparameters,
    ProfileMetrics,
    RunResult,
    RunSpec,
    run_experiment,
    save_experiment,
    plot_ablation_extrapolation,
)
from .inversion import (
    BandSensitivity,
    EddyViscosityInversion,
    ErrorDecomposition,
    IdentifiabilityReport,
    InversionAnalysis,
    TabulatedClosure,
    analyze_inversion,
    decompose_errors,
    identifiability_report,
    invert_eddy_viscosity,
    replay_error,
    save_inversion_artifacts,
)
from .solver import SolveResult, solve_channel, wall_clustered_grid

__all__ = [
    "Closure",
    "BaselineClosure",
    "TruthClosure",
    "SolveResult",
    "solve_channel",
    "wall_clustered_grid",
    "ChannelClosure",
    "ChannelProblem",
    "MODEL_INPUTS",
    "AblationConfig",
    "FULL",
    "A1",
    "A2",
    "A3",
    "ALL_OFF",
    "ABLATIONS",
    "EXPERIMENT_ARMS",
    "Hyperparameters",
    "RunSpec",
    "ProfileMetrics",
    "RunResult",
    "ExperimentResult",
    "TEST_RE_TAU",
    "EXTRAPOLATION_BUDGET",
    "EXTRAPOLATION_IMPROVEMENT_TARGET",
    "ABLATION_RELATIVE_ERROR_INCREASE_TARGET",
    "run_experiment",
    "save_experiment",
    "plot_ablation_extrapolation",
    "EddyViscosityInversion",
    "TabulatedClosure",
    "ErrorDecomposition",
    "BandSensitivity",
    "IdentifiabilityReport",
    "InversionAnalysis",
    "invert_eddy_viscosity",
    "replay_error",
    "decompose_errors",
    "identifiability_report",
    "analyze_inversion",
    "save_inversion_artifacts",
]
