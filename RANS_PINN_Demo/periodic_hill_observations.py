"""Frozen observation design and leakage isolation for the 2-D hill case.

Section 5 of ``docs/2d-periodic-hill-plan.md`` fixes the ten reference stations
into train, validation, and test groups; fixes nested B1/B2 budgets; and allows
1% noise on training observations only.  This module makes that contract
executable instead of leaving it as prose.

The important design choice is that one evaluation grid is frozen from the
union of all training observations across all seeds.  A raw profile point that
was visible to *any* training run is therefore absent from evaluation, even if
another seed happened not to select it.  Test and validation stations are never
sampled for observations, and their evaluation profiles stay clean.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Dict, FrozenSet, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np

from .periodic_hill import (
    REFERENCE_STATIONS,
    PeriodicHillProfile,
    load_reference,
)

__all__ = [
    "TRAIN_STATIONS",
    "VALIDATION_STATIONS",
    "TEST_STATIONS",
    "BUDGETS",
    "B1",
    "B2",
    "B1_TOTAL_POINTS",
    "B2_TOTAL_POINTS",
    "FROZEN_STATION_SPLIT",
    "NOISE_LEVEL",
    "OBSERVATION_SEEDS",
    "OBSERVATION_NOISE",
    "OBSERVATION_FIELDS",
    "FORBIDDEN_OBSERVATION_FIELDS",
    "HYPERPARAMETER_SELECTION_GROUP",
    "DEFAULT_REFERENCE_DIR",
    "DEFAULT_OUTPUT",
    "DEFAULT_LEAKAGE_OUTPUT",
    "ObservationPoint",
    "EvaluationPoint",
    "TrainingObservations",
    "EvaluationObservations",
    "ObservationSplitSet",
    "ObservationDesign",
    "FrozenEvaluationGrid",
    "LeakageCheck",
    "LeakageAudit",
    "BudgetSpec",
    "EvaluationIOPlan",
    "ObservationProtocol",
    "build_observation_design",
    "build_observation_splits",
    "build_evaluation_grid",
    "assert_nested_budgets",
    "assert_no_observation_evaluation_overlap",
    "assert_clean_validation_and_test",
    "assert_frozen_station_split",
    "audit_observation_protocol",
    "build_observation_protocol",
    "run_leakage_audit",
    "assert_leakage_isolation",
    "write_observation_protocol",
    "save_observation_design",
    "save_leakage_audit",
    "load_observation_protocol",
    "main",
]

TRAIN_STATIONS: Tuple[float, ...] = (0.5, 1.0, 2.0, 6.0)
VALIDATION_STATIONS: Tuple[float, ...] = (3.0, 7.0)
TEST_STATIONS: Tuple[float, ...] = (0.05, 4.0, 5.0, 8.0)

#: Observations per station for each nested budget.  B1 is a strict subset of B2.
BUDGETS: Mapping[str, int] = MappingProxyType({"B1": 8, "B2": 16})

#: The pre-registered seed set for the formal two-budget experiment.
OBSERVATION_SEEDS: Tuple[int, ...] = tuple(range(20))
OBSERVATION_NOISE = 0.01
NOISE_LEVEL = OBSERVATION_NOISE
OBSERVATION_FIELDS = ("u_over_ub", "v_over_ub")
FORBIDDEN_OBSERVATION_FIELDS = (
    "uu",
    "vv",
    "uv",
    "k",
    "uu_over_ub2",
    "vv_over_ub2",
    "uv_over_ub2",
    "k_over_ub2",
    "tau_w",
    "u_tau",
)
HYPERPARAMETER_SELECTION_GROUP = "validation"
B1_TOTAL_POINTS = len(TRAIN_STATIONS) * BUDGETS["B1"]
B2_TOTAL_POINTS = len(TRAIN_STATIONS) * BUDGETS["B2"]
FROZEN_STATION_SPLIT: Mapping[str, Tuple[float, ...]] = MappingProxyType(
    {
        "train": TRAIN_STATIONS,
        "validation": VALIDATION_STATIONS,
        "test": TEST_STATIONS,
    }
)

DEFAULT_REFERENCE_DIR = Path("RANS_PINN_Demo/reference/periodic_hill/extracted")
DEFAULT_OUTPUT = Path("RANS_PINN_Demo/results/periodic_hill_observation_protocol.json")
DEFAULT_LEAKAGE_OUTPUT = Path("RANS_PINN_Demo/results/periodic_hill_observation_leakage.json")

def _station_split_payload() -> Dict[str, Any]:
    return {
        "train": list(TRAIN_STATIONS),
        "validation": list(VALIDATION_STATIONS),
        "test": list(TEST_STATIONS),
    }


def _budget_specs_payload() -> Dict[str, Any]:
    return {
        "B1": {
            "name": "B1",
            "per_station": int(BUDGETS["B1"]),
            "total_points": int(B1_TOTAL_POINTS),
            "requested_total_points": int(B1_TOTAL_POINTS),
        },
        "B2": {
            "name": "B2",
            "per_station": int(BUDGETS["B2"]),
            "total_points": int(B2_TOTAL_POINTS),
            "requested_total_points": int(B2_TOTAL_POINTS),
        },
    }

PointKey = Tuple[float, int]


def _station_key(station: float) -> float:
    """Return the canonical float key for a published station."""

    matches = [candidate for candidate in REFERENCE_STATIONS if abs(candidate - station) <= 1e-12]
    if len(matches) != 1:
        raise ValueError(f"unknown periodic-hill station: {station!r}")
    return matches[0]


def _require_station_profiles(
    profiles: Mapping[float, PeriodicHillProfile],
    stations: Iterable[float],
) -> Dict[float, PeriodicHillProfile]:
    normalized: Dict[float, PeriodicHillProfile] = {}
    for station in stations:
        key = _station_key(float(station))
        try:
            profile = profiles[key]
        except KeyError as exc:
            raise KeyError(f"missing reference profile at x/h={key}") from exc
        normalized[key] = profile
    return normalized


def _profile_arrays(profile: PeriodicHillProfile) -> Tuple[np.ndarray, ...]:
    y = np.asarray(profile.y_over_h, dtype=float)
    eta = np.asarray(profile.wall_normal, dtype=float)
    u = np.asarray(profile.u_over_ub, dtype=float)
    v = np.asarray(profile.v_over_ub, dtype=float)
    shape = y.shape
    if eta.shape != shape or u.shape != shape or v.shape != shape:
        raise ValueError(f"profile at x/h={profile.x_over_h} has inconsistent array shapes")
    return y, eta, u, v


def _eligible_indices(profile: PeriodicHillProfile) -> np.ndarray:
    """Return fluid-side profile indices ordered by wall-normal coordinate."""

    _, eta, u, v = _profile_arrays(profile)
    valid = np.flatnonzero(
        np.isfinite(eta) & np.isfinite(u) & np.isfinite(v) & (eta >= 0.0)
    )
    if valid.size == 0:
        raise ValueError(f"profile at x/h={profile.x_over_h} has no fluid-side samples")
    return valid[np.lexsort((valid, eta[valid]))]


def _nested_stratified_indices(
    profile: PeriodicHillProfile,
    *,
    seed: int,
    station: float,
) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    """Select B1 inside B2 with wall-normal stratification.

    B1 is one point from each of ``BUDGETS["B1"]`` equal-count strata.  B2 adds
    one more point to a random subset of those strata, giving the fixed nested
    budget while retaining wall-normal coverage.  If a reference profile is too
    short for the requested budget, the actual budget is reduced rather than
    moving a validation/test station into training.
    """

    ordered = _eligible_indices(profile)
    count = int(ordered.size)
    b2_count = min(BUDGETS["B2"], count)
    # Keep B1 strictly nested even when a short profile forces a lower budget.
    b1_count = min(BUDGETS["B1"], max(1, b2_count - 1))
    if b1_count <= 0 or b2_count < b1_count:
        raise ValueError(f"cannot build a nested budget at x/h={profile.x_over_h}")

    station_seed = (int(seed), int(round(float(station) * 1_000_000.0)))
    rng = np.random.default_rng(np.random.SeedSequence(station_seed))
    strata = np.array_split(ordered, b1_count)

    b1_indices = []
    available_extra = []
    for stratum in strata:
        members = np.asarray(stratum, dtype=int).copy()
        rng.shuffle(members)
        b1_indices.append(int(members[0]))
        if members.size > 1:
            available_extra.append([int(value) for value in members[1:]])

    need_extra = b2_count - b1_count
    extra_indices = []
    while need_extra > 0:
        choices = [position for position, pool in enumerate(available_extra) if pool]
        if not choices:
            raise ValueError(
                f"profile at x/h={profile.x_over_h} cannot support B2={b2_count}"
            )
        position = int(choices[int(rng.integers(0, len(choices)))])
        extra_indices.append(available_extra[position].pop())
        need_extra -= 1

    def order_key(index: int) -> Tuple[float, int]:
        return (float(profile.wall_normal[index]), int(index))

    b1 = tuple(sorted(b1_indices, key=order_key))
    b2 = tuple(sorted(b1_indices + extra_indices, key=order_key))
    if len(set(b1)) != b1_count or len(set(b2)) != b2_count:
        raise RuntimeError("internal error: selected observation indices are not unique")
    return b1, b2

@dataclass(frozen=True)
class BudgetSpec:
    """One pre-registered observation budget."""

    name: str
    per_station: int
    total_points: int

    @property
    def requested_total_points(self) -> int:
        return int(self.total_points)


B1 = BudgetSpec("B1", BUDGETS["B1"], B1_TOTAL_POINTS)
B2 = BudgetSpec("B2", BUDGETS["B2"], B2_TOTAL_POINTS)


@dataclass(frozen=True)
class ObservationPoint:
    """One noisy training observation, with its clean value retained for audit."""

    station: float
    reference_index: int
    y_over_h: float
    wall_normal: float
    clean_u_over_ub: float
    clean_v_over_ub: float
    noisy_u_over_ub: float
    noisy_v_over_ub: float
    noise_relative: float = OBSERVATION_NOISE

    @property
    def key(self) -> PointKey:
        return (float(self.station), int(self.reference_index))

    @property
    def noise_applied(self) -> bool:
        return True

    @property
    def u_over_ub(self) -> float:
        """Training input value: the noisy streamwise velocity."""

        return float(self.noisy_u_over_ub)

    @property
    def v_over_ub(self) -> float:
        """Training input value: the noisy wall-normal velocity."""

        return float(self.noisy_v_over_ub)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "station": float(self.station),
            "reference_index": int(self.reference_index),
            "y_over_h": float(self.y_over_h),
            "wall_normal": float(self.wall_normal),
            "clean_u_over_ub": float(self.clean_u_over_ub),
            "clean_v_over_ub": float(self.clean_v_over_ub),
            "noisy_u_over_ub": float(self.noisy_u_over_ub),
            "noisy_v_over_ub": float(self.noisy_v_over_ub),
            "noise_relative": float(self.noise_relative),
        }


@dataclass(frozen=True)
class EvaluationPoint:
    """One clean, held-out evaluation sample."""

    station: float
    reference_index: int
    y_over_h: float
    wall_normal: float
    u_over_ub: float
    v_over_ub: float
    role: str

    @property
    def key(self) -> PointKey:
        return (float(self.station), int(self.reference_index))

    @property
    def noise_applied(self) -> bool:
        return False

    def as_dict(self) -> Dict[str, Any]:
        return {
            "station": float(self.station),
            "reference_index": int(self.reference_index),
            "y_over_h": float(self.y_over_h),
            "wall_normal": float(self.wall_normal),
            "u_over_ub": float(self.u_over_ub),
            "v_over_ub": float(self.v_over_ub),
            "role": self.role,
        }


@dataclass(frozen=True)
class ObservationDesign:
    """Nested B1/B2 observations for one random seed."""

    seed: int
    b1: Tuple[ObservationPoint, ...]
    b2: Tuple[ObservationPoint, ...]

    def budget(self, name: str) -> Tuple[ObservationPoint, ...]:
        if name == "B1":
            return self.b1
        if name == "B2":
            return self.b2
        raise KeyError(f"unknown observation budget {name!r}")

    def as_dict(self) -> Dict[str, Any]:
        return {
            "seed": int(self.seed),
            "B1": [point.as_dict() for point in self.b1],
            "B2": [point.as_dict() for point in self.b2],
        }


@dataclass(frozen=True)
class TrainingObservations:
    """Noisy training points plus the fields a trainer may consume."""

    points: Tuple[ObservationPoint, ...]
    input_fields: Tuple[str, ...] = ("x_over_h", "wall_normal")
    target_fields: Tuple[str, ...] = ("u_over_ub",)

    @property
    def stations(self) -> Tuple[float, ...]:
        return tuple(sorted({point.station for point in self.points}))

    def inputs(self) -> Dict[str, Tuple[float, ...]]:
        return {
            "x_over_h": tuple(point.station for point in self.points),
            "wall_normal": tuple(point.wall_normal for point in self.points),
        }

    def targets(self) -> Dict[str, Tuple[float, ...]]:
        return {"u_over_ub": tuple(point.u_over_ub for point in self.points)}

    def as_dict(self) -> Dict[str, Any]:
        return {
            "input_fields": list(self.input_fields),
            "target_fields": list(self.target_fields),
            "points": [point.as_dict() for point in self.points],
        }


@dataclass(frozen=True)
class EvaluationObservations:
    """Clean evaluation points for one role."""

    points: Tuple[EvaluationPoint, ...]

    @property
    def stations(self) -> Tuple[float, ...]:
        return tuple(sorted({point.station for point in self.points}))

    def as_dict(self) -> Dict[str, Any]:
        return {"points": [point.as_dict() for point in self.points]}


@dataclass(frozen=True)
class ObservationSplitSet:
    """Compatibility view of one nested budget for one seed."""

    budget: BudgetSpec
    seed: int
    train: TrainingObservations
    validation: EvaluationObservations
    test: EvaluationObservations
    same_station_extrapolation: EvaluationObservations

    @property
    def requested_total_points(self) -> int:
        return self.budget.total_points

    @property
    def actual_total_points(self) -> int:
        return len(self.train.points)

    @property
    def reduced(self) -> bool:
        return self.actual_total_points < self.requested_total_points

    @property
    def train_station_counts(self) -> Dict[float, int]:
        counts = {station: 0 for station in TRAIN_STATIONS}
        for point in self.train.points:
            counts[point.station] = counts.get(point.station, 0) + 1
        return counts

    @property
    def validation_points(self) -> int:
        return len(self.validation.points)

    @property
    def test_points(self) -> int:
        return len(self.test.points)

    @property
    def same_station_extrapolation_points(self) -> int:
        return len(self.same_station_extrapolation.points)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "budget": {
                "name": self.budget.name,
                "per_station": self.budget.per_station,
                "requested_total_points": self.requested_total_points,
                "total_points": self.budget.total_points,
            },
            "seed": int(self.seed),
            "requested_total_points": self.requested_total_points,
            "actual_total_points": self.actual_total_points,
            "reduced": self.reduced,
            "train_station_counts": {
                str(station): count for station, count in self.train_station_counts.items()
            },
            "validation_points": self.validation_points,
            "test_points": self.test_points,
            "same_station_extrapolation_points": self.same_station_extrapolation_points,
            "train": self.train.as_dict(),
            "validation": self.validation.as_dict(),
            "test": self.test.as_dict(),
            "same_station_extrapolation": self.same_station_extrapolation.as_dict(),
        }


@dataclass(frozen=True)
class FrozenEvaluationGrid:
    """Clean evaluation points and the immutable evaluation contract."""

    points: Tuple[EvaluationPoint, ...]

    @property
    def keys(self) -> FrozenSet[PointKey]:
        return frozenset(point.key for point in self.points)

    @property
    def retrain(self) -> bool:
        return False

    @property
    def reselect_points(self) -> bool:
        return False

    def role_counts(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for point in self.points:
            counts[point.role] = counts.get(point.role, 0) + 1
        return counts

    def as_dict(self) -> Dict[str, Any]:
        return {
            "retrain": self.retrain,
            "reselect_points": self.reselect_points,
            "role_counts": self.role_counts(),
            "points": [point.as_dict() for point in self.points],
        }


@dataclass(frozen=True)
class EvaluationIOPlan:
    """Declared I/O behavior of the final evaluation script."""

    reads: Tuple[str, ...] = ("frozen_model", "test_files")
    retrain: bool = False
    reselect_points: bool = False


@dataclass(frozen=True)
class LeakageCheck:
    key: str
    title: str
    passed: bool
    detail: str
    check_id: int = 0

    @property
    def name(self) -> str:
        return self.key

    @property
    def details(self) -> str:
        return self.detail

    def as_dict(self) -> Dict[str, Any]:
        return {
            "check_id": int(self.check_id),
            "key": self.key,
            "name": self.key,
            "title": self.title,
            "passed": bool(self.passed),
            "detail": self.detail,
            "details": self.detail,
        }


@dataclass(frozen=True)
class LeakageAudit:
    checks: Tuple[LeakageCheck, ...]

    @property
    def passed(self) -> bool:
        return bool(self.checks) and all(check.passed for check in self.checks)

    @property
    def all_passed(self) -> bool:
        return self.passed

    def as_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "checks": [check.as_dict() for check in self.checks],
        }


@dataclass(frozen=True)
class ObservationProtocol:
    designs: Tuple[ObservationDesign, ...]
    evaluation: FrozenEvaluationGrid
    leakage_audit: LeakageAudit

    @property
    def seeds(self) -> Tuple[int, ...]:
        return tuple(design.seed for design in self.designs)

    @property
    def observed_keys(self) -> FrozenSet[PointKey]:
        return frozenset(
            point.key
            for design in self.designs
            for budget in ("B1", "B2")
            for point in design.budget(budget)
        )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "schema": "periodic-hill-observation-protocol/v1",
            "station_split": _station_split_payload(),
            "observation_fields": list(OBSERVATION_FIELDS),
            "forbidden_observation_fields": list(FORBIDDEN_OBSERVATION_FIELDS),
            "budgets": {name: int(count) for name, count in BUDGETS.items()},
            "observation_noise": float(OBSERVATION_NOISE),
            "hyperparameter_selection_group": HYPERPARAMETER_SELECTION_GROUP,
            "seeds": list(self.seeds),
            "designs": [design.as_dict() for design in self.designs],
            "evaluation": self.evaluation.as_dict(),
            "leakage_audit": self.leakage_audit.as_dict(),
        }

def _ordered_points(points: Iterable[ObservationPoint]) -> Tuple[ObservationPoint, ...]:
    return tuple(
        sorted(points, key=lambda point: (point.station, point.wall_normal, point.reference_index))
    )


def build_observation_design(
    profiles: Mapping[float, PeriodicHillProfile],
    seed: int,
) -> ObservationDesign:
    """Build the frozen nested budgets for one pre-registered seed."""

    if int(seed) not in OBSERVATION_SEEDS:
        raise ValueError(f"seed {seed!r} is outside the frozen seed set")
    selected = _require_station_profiles(profiles, TRAIN_STATIONS)
    b1_points = []
    b2_points = []

    for station in TRAIN_STATIONS:
        profile = selected[station]
        b1_indices, b2_indices = _nested_stratified_indices(
            profile, seed=int(seed), station=station
        )
        _, _, u, v = _profile_arrays(profile)
        station_seed = (int(seed), int(round(station * 1_000_000.0)), 0xB1)
        rng = np.random.default_rng(np.random.SeedSequence(station_seed))
        point_by_index: Dict[int, ObservationPoint] = {}
        for index in b2_indices:
            clean_u = float(u[index])
            clean_v = float(v[index])
            noisy_u = clean_u + OBSERVATION_NOISE * abs(clean_u) * float(rng.normal())
            noisy_v = clean_v + OBSERVATION_NOISE * abs(clean_v) * float(rng.normal())
            point_by_index[index] = ObservationPoint(
                station=float(station),
                reference_index=int(index),
                y_over_h=float(np.asarray(profile.y_over_h)[index]),
                wall_normal=float(np.asarray(profile.wall_normal)[index]),
                clean_u_over_ub=clean_u,
                clean_v_over_ub=clean_v,
                noisy_u_over_ub=noisy_u,
                noisy_v_over_ub=noisy_v,
            )
        b1_points.extend(point_by_index[index] for index in b1_indices)
        b2_points.extend(point_by_index[index] for index in b2_indices)

    return ObservationDesign(
        seed=int(seed),
        b1=_ordered_points(b1_points),
        b2=_ordered_points(b2_points),
    )


def _evaluation_role(station: float) -> str:
    if station in TEST_STATIONS:
        return "test"
    if station in VALIDATION_STATIONS:
        return "validation"
    if station in TRAIN_STATIONS:
        return "same_station_remaining"
    raise ValueError(f"station {station!r} is not in the frozen split")


def build_evaluation_grid(
    profiles: Mapping[float, PeriodicHillProfile],
    *,
    observed_keys: Iterable[PointKey] = (),
) -> FrozenEvaluationGrid:
    """Build one clean grid shared by all seeds.

    ``observed_keys`` should be the union over every seed and budget.  Test and
    validation points always remain in the grid; only unobserved fluid-side
    training-station points are used for the same-station diagnostic.
    """

    selected = _require_station_profiles(profiles, REFERENCE_STATIONS)
    observed = frozenset((float(station), int(index)) for station, index in observed_keys)
    points = []
    for station in REFERENCE_STATIONS:
        profile = selected[station]
        y, eta, u, v = _profile_arrays(profile)
        for index in range(y.size):
            key = (float(station), int(index))
            if station in TRAIN_STATIONS and key in observed:
                continue
            points.append(
                EvaluationPoint(
                    station=float(station),
                    reference_index=int(index),
                    y_over_h=float(y[index]),
                    wall_normal=float(eta[index]),
                    u_over_ub=float(u[index]),
                    v_over_ub=float(v[index]),
                    role=_evaluation_role(station),
                )
            )
    return FrozenEvaluationGrid(points=tuple(points))


def build_observation_splits(
    profiles: Mapping[float, PeriodicHillProfile],
    seed: int = 0,
) -> Dict[str, ObservationSplitSet]:
    """Return consumable B1/B2 split objects for one seed.

    This is the seam used by training.  The stricter protocol keeps one
    evaluation grid for all seeds; here each budget's evaluation grid excludes
    exactly the points observed by that budget, matching the original data
    volume experiment.
    """

    design = build_observation_design(profiles, seed=seed)
    splits: Dict[str, ObservationSplitSet] = {}
    for name in ("B1", "B2"):
        train_points = design.budget(name)
        observed_keys = {point.key for point in train_points}
        evaluation = build_evaluation_grid(profiles, observed_keys=observed_keys)
        validation_points = tuple(
            point for point in evaluation.points if point.station in VALIDATION_STATIONS
        )
        test_points = tuple(point for point in evaluation.points if point.station in TEST_STATIONS)
        same_station = tuple(
            point for point in evaluation.points if point.role == "same_station_remaining"
        )
        splits[name] = ObservationSplitSet(
            budget=B1 if name == "B1" else B2,
            seed=int(seed),
            train=TrainingObservations(points=train_points),
            validation=EvaluationObservations(points=validation_points),
            test=EvaluationObservations(points=test_points),
            same_station_extrapolation=EvaluationObservations(points=same_station),
        )
    assert_nested_budgets(splits)
    for split in splits.values():
        assert_frozen_station_split(split)
        assert_no_observation_evaluation_overlap(split)
        assert_clean_validation_and_test(split)
    return splits


def _evaluation_points(split: ObservationSplitSet) -> Tuple[EvaluationPoint, ...]:
    return (
        split.validation.points
        + split.test.points
        + split.same_station_extrapolation.points
    )


def assert_nested_budgets(splits: Mapping[str, ObservationSplitSet]) -> None:
    if set(splits) != {"B1", "B2"}:
        raise AssertionError("expected exactly B1 and B2 observation splits")
    b1_keys = {point.key for point in splits["B1"].train.points}
    b2_keys = {point.key for point in splits["B2"].train.points}
    if not b1_keys < b2_keys:
        raise AssertionError("B1 observations must be a strict subset of B2")


def assert_no_observation_evaluation_overlap(split: ObservationSplitSet) -> None:
    observed = {point.key for point in split.train.points}
    evaluated = {point.key for point in _evaluation_points(split)}
    if observed.intersection(evaluated):
        raise AssertionError("training observations and evaluation points overlap")


def assert_clean_validation_and_test(split: ObservationSplitSet) -> None:
    if any(point.noise_applied for point in split.validation.points + split.test.points):
        raise AssertionError("validation and test evaluation points must remain clean")


def assert_frozen_station_split(split: ObservationSplitSet) -> None:
    if set(split.train.stations) != set(TRAIN_STATIONS):
        raise AssertionError("training stations do not match the frozen split")
    if set(split.validation.stations) != set(VALIDATION_STATIONS):
        raise AssertionError("validation stations do not match the frozen split")
    if set(split.test.stations) != set(TEST_STATIONS):
        raise AssertionError("test stations do not match the frozen split")

def _check(key: str, title: str, passed: bool, detail: str) -> LeakageCheck:
    return LeakageCheck(key=key, title=title, passed=bool(passed), detail=detail)


def audit_observation_protocol(
    designs: Sequence[ObservationDesign],
    evaluation: FrozenEvaluationGrid,
) -> LeakageAudit:
    """Audit the five section-5.4 leakage requirements and their contracts."""

    designs = tuple(designs)
    observations = tuple(
        point
        for design in designs
        for budget in ("B1", "B2")
        for point in design.budget(budget)
    )
    observation_keys = {point.key for point in observations}
    observed_stations = {point.station for point in observations}
    seed_set = {design.seed for design in designs}
    split_set = set(TRAIN_STATIONS) | set(VALIDATION_STATIONS) | set(TEST_STATIONS)

    partition_ok = (
        split_set == set(REFERENCE_STATIONS)
        and set(TRAIN_STATIONS).isdisjoint(VALIDATION_STATIONS)
        and set(TRAIN_STATIONS).isdisjoint(TEST_STATIONS)
        and set(VALIDATION_STATIONS).isdisjoint(TEST_STATIONS)
    )
    checks = [
        _check(
            "frozen_station_partition",
            "Station groups are a frozen partition",
            partition_ok,
            "train/validation/test station groups must be disjoint and cover all ten profiles",
        ),
        _check(
            "training_only_observations",
            "Only training stations are observed",
            observed_stations <= set(TRAIN_STATIONS),
            "observed stations: " + repr(sorted(observed_stations)),
        ),
        _check(
            "no_test_station_observations",
            "No test station enters observations",
            observed_stations.isdisjoint(TEST_STATIONS),
            "test stations observed: "
            + repr(sorted(observed_stations.intersection(TEST_STATIONS))),
        ),
        _check(
            "nested_budgets",
            "B1 is a strict subset of B2 for every seed",
            bool(designs)
            and all(
                {point.key for point in design.b1} < {point.key for point in design.b2}
                and len({point.key for point in design.b1}) == len(design.b1)
                and len({point.key for point in design.b2}) == len(design.b2)
                for design in designs
            ),
            "each seed must select B1 from the same profile points as its B2",
        ),
        _check(
            "budget_limits",
            "Budgets do not exceed the pre-registered per-station counts",
            all(
                1
                <= sum(1 for point in design.budget(name) if point.station == station)
                <= BUDGETS[name]
                for design in designs
                for name in ("B1", "B2")
                for station in TRAIN_STATIONS
            ),
            "B1 <= 8 and B2 <= 16 points per station",
        ),
        _check(
            "allowed_observation_fields",
            "Training fields are velocity-only",
            tuple(OBSERVATION_FIELDS) == ("u_over_ub", "v_over_ub")
            and not set(OBSERVATION_FIELDS).intersection(FORBIDDEN_OBSERVATION_FIELDS)
            and all(
                not any(hasattr(point, field) for field in FORBIDDEN_OBSERVATION_FIELDS)
                for point in observations
            ),
            "U and V are allowed; turbulence statistics, tau_w, and u_tau are forbidden",
        ),
        _check(
            "train_only_noise",
            "Noise is 1% and attached to training observations only",
            all(
                math.isfinite(point.noise_relative)
                and abs(point.noise_relative - OBSERVATION_NOISE) <= 1e-15
                and math.isfinite(point.noisy_u_over_ub)
                and math.isfinite(point.noisy_v_over_ub)
                for point in observations
            ),
            "evaluation points contain clean values only",
        ),
        _check(
            "evaluation_excludes_observations",
            "Evaluation points never overlap any seed's observations",
            evaluation.keys.isdisjoint(observation_keys),
            "overlap count: " + str(len(evaluation.keys.intersection(observation_keys))),
        ),
        _check(
            "evaluation_covers_frozen_sites",
            "Every validation and test station remains in evaluation",
            bool(evaluation.points)
            and {
                point.station
                for point in evaluation.points
                if point.station in VALIDATION_STATIONS
            }
            == set(VALIDATION_STATIONS)
            and {
                point.station
                for point in evaluation.points
                if point.station in TEST_STATIONS
            }
            == set(TEST_STATIONS),
            "validation and test stations cannot be dropped by observation sampling",
        ),
        _check(
            "hyperparameter_selection_is_validation",
            "Hyperparameter selection is validation-only",
            HYPERPARAMETER_SELECTION_GROUP == "validation",
            "test stations are never used for model or threshold selection",
        ),
        _check(
            "frozen_evaluation_contract",
            "Evaluation cannot retrain or reselect points",
            evaluation.retrain is False and evaluation.reselect_points is False,
            "the evaluator consumes the frozen artifact and model only",
        ),
        _check(
            "unique_registered_seeds",
            "Seed designs are unique and pre-registered",
            bool(seed_set)
            and len(seed_set) == len(designs)
            and seed_set <= set(OBSERVATION_SEEDS),
            "seeds: " + repr(sorted(seed_set)),
        ),
    ]
    return LeakageAudit(checks=tuple(checks))

def _is_wall_shear_term(field: str) -> bool:
    normalized = field.lower()
    return any(
        token in normalized
        for token in ("tau", "u_tau", "utau", "wall_shear", "wall_stress", "friction")
    )


def _is_forbidden_reference_field(field: str) -> bool:
    normalized = field.lower()
    return normalized in FORBIDDEN_OBSERVATION_FIELDS or _is_wall_shear_term(normalized)


def _declared_check(
    check_id: int,
    key: str,
    title: str,
    passed: bool,
    detail: str,
) -> LeakageCheck:
    return LeakageCheck(
        key=key,
        title=title,
        passed=bool(passed),
        detail=detail,
        check_id=int(check_id),
    )


def run_leakage_audit(
    profiles: Mapping[float, PeriodicHillProfile],
    *,
    seeds: Sequence[int] = OBSERVATION_SEEDS,
    require_all_seeds: bool = True,
    training_input_fields: Sequence[str] = ("x_over_h", "wall_normal", "u_over_ub"),
    training_loss_stations: Optional[Sequence[float]] = None,
    training_loss_terms: Sequence[str] = ("u_over_ub",),
    hyperparameter_selection_stations: Sequence[float] = VALIDATION_STATIONS,
    evaluation_io: Optional[EvaluationIOPlan] = None,
) -> LeakageAudit:
    """Run the five section-5.4 checks against explicit training declarations.

    The observation design is built from ``profiles`` first, so the audit sees
    the same nested budgets and clean evaluation grid that will be saved for
    training.  Callers must declare the fields and loss terms their training
    code actually uses; a design-only check cannot prove those declarations.
    """

    protocol = build_observation_protocol(
        profiles,
        seeds=seeds,
        require_all_seeds=require_all_seeds,
    )
    input_fields = tuple(str(field) for field in training_input_fields)
    loss_terms = tuple(str(term) for term in training_loss_terms)
    loss_stations = tuple(
        float(station)
        for station in (
            TRAIN_STATIONS if training_loss_stations is None else training_loss_stations
        )
    )
    hyper_stations = tuple(float(station) for station in hyperparameter_selection_stations)
    io_plan = EvaluationIOPlan() if evaluation_io is None else evaluation_io

    forbidden_inputs = [field for field in input_fields if _is_forbidden_reference_field(field)]
    test_loss_stations = [
        station
        for station in loss_stations
        if any(abs(station - test_station) <= 1e-12 for test_station in TEST_STATIONS)
    ]
    invalid_loss_stations = [
        station
        for station in loss_stations
        if not any(abs(station - known) <= 1e-12 for known in REFERENCE_STATIONS)
    ]
    bad_hyper_stations = [
        station
        for station in hyper_stations
        if not any(abs(station - valid) <= 1e-12 for valid in VALIDATION_STATIONS)
    ]
    forbidden_loss_terms = [
        term
        for term in loss_terms
        if _is_forbidden_reference_field(term) or _is_wall_shear_term(term)
    ]
    allowed_reads = {"frozen_model", "test_files", "observation_protocol"}
    reads = tuple(str(item) for item in io_plan.reads)
    io_ok = (
        bool(reads)
        and set(reads) <= allowed_reads
        and "frozen_model" in reads
        and "test_files" in reads
        and io_plan.retrain is False
        and io_plan.reselect_points is False
    )

    declared_checks = [
        _declared_check(
            1,
            "training_inputs_exclude_reference_turbulence",
            "Training inputs exclude reference turbulence",
            not forbidden_inputs,
            "clean fields: " + repr(input_fields)
            if not forbidden_inputs
            else "forbidden fields: " + repr(forbidden_inputs),
        ),
        _declared_check(
            2,
            "training_loss_uses_no_test_stations",
            "Training loss uses no test stations",
            not test_loss_stations and not invalid_loss_stations,
            "loss stations: " + repr(loss_stations)
            if not test_loss_stations and not invalid_loss_stations
            else "test/unknown loss stations: "
            + repr(sorted(test_loss_stations + invalid_loss_stations)),
        ),
        _declared_check(
            3,
            "hyperparameter_selection_uses_validation",
            "Hyperparameter selection uses validation stations",
            bool(hyper_stations) and not bad_hyper_stations,
            "hyperparameter stations: " + repr(hyper_stations)
            if hyper_stations and not bad_hyper_stations
            else "non-validation stations: " + repr(sorted(bad_hyper_stations)),
        ),
        _declared_check(
            4,
            "wall_shear_reference_is_isolated",
            "Wall-shear reference is isolated",
            not any(_is_wall_shear_term(field) for field in input_fields)
            and not forbidden_loss_terms,
            "wall-shear terms are absent from inputs and loss"
            if not any(_is_wall_shear_term(field) for field in input_fields)
            and not forbidden_loss_terms
            else "leaked wall-shear terms: "
            + repr(
                [field for field in input_fields if _is_wall_shear_term(field)]
                + forbidden_loss_terms
            ),
        ),
        _declared_check(
            5,
            "frozen_evaluation_io_is_isolated",
            "Frozen evaluation I/O is isolated",
            io_ok,
            "reads: " + repr(reads) + "; retrain=False; reselect_points=False"
            if io_ok
            else "evaluation I/O contract violated: " + repr(reads),
        ),
    ]
    return LeakageAudit(checks=tuple(declared_checks) + protocol.leakage_audit.checks)


def assert_leakage_isolation(audit: LeakageAudit) -> None:
    """Raise if any leakage-isolation check failed."""

    if audit.passed:
        return
    failed = [
        f"{check.check_id}:{check.key} ({check.detail})"
        for check in audit.checks
        if not check.passed
    ]
    raise AssertionError("leakage isolation failed: " + "; ".join(failed))

def build_observation_protocol(
    profiles: Mapping[float, PeriodicHillProfile],
    *,
    seeds: Sequence[int] = OBSERVATION_SEEDS,
    require_all_seeds: bool = True,
) -> ObservationProtocol:
    """Build all frozen seed designs and one clean seed-independent grid."""

    seeds = tuple(int(seed) for seed in seeds)
    if not seeds:
        raise ValueError("at least one pre-registered seed is required")
    if require_all_seeds and seeds != OBSERVATION_SEEDS:
        raise ValueError("the formal protocol requires the complete 20-seed set")
    designs = tuple(build_observation_design(profiles, seed=seed) for seed in seeds)
    observed_keys = frozenset(
        point.key
        for design in designs
        for budget in ("B1", "B2")
        for point in design.budget(budget)
    )
    evaluation = build_evaluation_grid(profiles, observed_keys=observed_keys)
    audit = audit_observation_protocol(designs, evaluation)
    if not audit.passed:
        failed = ", ".join(check.key for check in audit.checks if not check.passed)
        raise ValueError(f"observation protocol failed leakage audit: {failed}")
    return ObservationProtocol(
        designs=designs,
        evaluation=evaluation,
        leakage_audit=audit,
    )


def write_observation_protocol(path: Path, protocol: ObservationProtocol) -> None:
    if protocol.seeds != OBSERVATION_SEEDS:
        raise ValueError("refusing to freeze a protocol without the complete 20-seed set")
    audit = audit_observation_protocol(protocol.designs, protocol.evaluation)
    if not protocol.leakage_audit.passed or not audit.passed:
        raise ValueError("refusing to write an observation protocol that fails leakage isolation")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(protocol.as_dict(), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def save_observation_design(
    splits_or_path: Any,
    path_or_protocol: Any,
) -> Path:
    """Write either split objects or a frozen protocol as JSON.

    Both ``save_observation_design(splits, path)`` (training-facing) and
    ``save_observation_design(path, protocol)`` (compatibility with the first
    draft) are accepted.
    """

    if isinstance(splits_or_path, (str, Path)):
        destination = Path(splits_or_path)
        payload_object = path_or_protocol
    else:
        payload_object = splits_or_path
        destination = Path(path_or_protocol)

    if isinstance(payload_object, ObservationProtocol):
        write_observation_protocol(destination, payload_object)
        return destination

    if not isinstance(payload_object, Mapping):
        raise TypeError("observation design must be a B1/B2 mapping or a protocol")
    payload = {
        "schema": "periodic-hill-observation-splits/v1",
        "station_split": _station_split_payload(),
        "budget_specs": _budget_specs_payload(),
        "budgets": {
            name: split.as_dict() for name, split in payload_object.items()
        },
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return destination


def save_leakage_audit(audit_or_path: Any, path_or_audit: Any) -> Path:
    """Write a standalone leakage audit report.

    Both ``save_leakage_audit(audit, path)`` and the path-first argument order
    used by the CLI are accepted.
    """

    if isinstance(audit_or_path, LeakageAudit):
        audit = audit_or_path
        path = Path(path_or_audit)
    else:
        path = Path(audit_or_path)
        audit = path_or_audit
    if not isinstance(audit, LeakageAudit):
        raise TypeError("save_leakage_audit requires a LeakageAudit")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "periodic-hill-leakage-audit/v1",
        "passed": audit.passed,
        "all_passed": audit.all_passed,
        "station_split": _station_split_payload(),
        "budget_specs": _budget_specs_payload(),
        "checks": [check.as_dict() for check in audit.checks],
    }
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return path

def _observation_point_from_dict(payload: Mapping[str, Any]) -> ObservationPoint:
    return ObservationPoint(
        station=float(payload["station"]),
        reference_index=int(payload["reference_index"]),
        y_over_h=float(payload["y_over_h"]),
        wall_normal=float(payload["wall_normal"]),
        clean_u_over_ub=float(payload["clean_u_over_ub"]),
        clean_v_over_ub=float(payload["clean_v_over_ub"]),
        noisy_u_over_ub=float(payload["noisy_u_over_ub"]),
        noisy_v_over_ub=float(payload["noisy_v_over_ub"]),
        noise_relative=float(payload["noise_relative"]),
    )


def _evaluation_point_from_dict(payload: Mapping[str, Any]) -> EvaluationPoint:
    return EvaluationPoint(
        station=float(payload["station"]),
        reference_index=int(payload["reference_index"]),
        y_over_h=float(payload["y_over_h"]),
        wall_normal=float(payload["wall_normal"]),
        u_over_ub=float(payload["u_over_ub"]),
        v_over_ub=float(payload["v_over_ub"]),
        role=str(payload["role"]),
    )


def _assert_frozen_payload(payload: Mapping[str, Any]) -> None:
    if payload.get("schema") != "periodic-hill-observation-protocol/v1":
        raise ValueError("unsupported periodic-hill observation protocol schema")
    split = payload.get("station_split", {})
    expected = {
        "train": list(TRAIN_STATIONS),
        "validation": list(VALIDATION_STATIONS),
        "test": list(TEST_STATIONS),
    }
    if split != expected:
        raise ValueError("observation protocol station split does not match the frozen split")
    if payload.get("budgets") != BUDGETS:
        raise ValueError("observation protocol budgets do not match the frozen budgets")
    if abs(float(payload.get("observation_noise", float("nan"))) - OBSERVATION_NOISE) > 1e-15:
        raise ValueError("observation protocol noise does not match the frozen 1% policy")
    if tuple(payload.get("observation_fields", ())) != OBSERVATION_FIELDS:
        raise ValueError("observation protocol fields do not match the frozen velocity fields")


def load_observation_protocol(path: Path) -> ObservationProtocol:
    """Load and re-audit a frozen observation protocol artifact."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("observation protocol artifact must contain a JSON object")
    _assert_frozen_payload(payload)
    designs = []
    for design_payload in payload.get("designs", []):
        designs.append(
            ObservationDesign(
                seed=int(design_payload["seed"]),
                b1=tuple(
                    _observation_point_from_dict(point)
                    for point in design_payload.get("B1", [])
                ),
                b2=tuple(
                    _observation_point_from_dict(point)
                    for point in design_payload.get("B2", [])
                ),
            )
        )
    if tuple(int(seed) for seed in payload.get("seeds", ())) != OBSERVATION_SEEDS:
        raise ValueError("loaded observation protocol must contain all 20 frozen seeds")
    design_seeds = tuple(design.seed for design in designs)
    if design_seeds != OBSERVATION_SEEDS:
        raise ValueError(
            "loaded observation protocol designs must cover all 20 frozen seeds"
        )
    evaluation_payload = payload.get("evaluation", {})
    evaluation = FrozenEvaluationGrid(
        points=tuple(
            _evaluation_point_from_dict(point)
            for point in evaluation_payload.get("points", [])
        )
    )
    audit = audit_observation_protocol(designs, evaluation)
    stored_passed = bool(payload.get("leakage_audit", {}).get("passed", False))
    if not stored_passed or not audit.passed:
        failed = ", ".join(check.key for check in audit.checks if not check.passed)
        raise ValueError(f"loaded observation protocol failed leakage audit: {failed}")
    return ObservationProtocol(
        designs=tuple(designs),
        evaluation=evaluation,
        leakage_audit=audit,
    )


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference-dir",
        type=Path,
        default=DEFAULT_REFERENCE_DIR,
        help="extracted ERCOFTAC case-081 directory",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="frozen JSON observation protocol path",
    )
    parser.add_argument(
        "--leakage-output",
        type=Path,
        default=DEFAULT_LEAKAGE_OUTPUT,
        help="standalone JSON leakage-audit path",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    try:
        profiles = load_reference(args.reference_dir)
        protocol = build_observation_protocol(profiles)
        leakage_audit = run_leakage_audit(profiles)
        if not leakage_audit.passed:
            assert_leakage_isolation(leakage_audit)
        write_observation_protocol(args.output, protocol)
        save_leakage_audit(args.leakage_output, leakage_audit)
    except FileNotFoundError as exc:
        print(str(exc))
        print(
            "Fetch the independent reference first: "
            "python -m RANS_PINN_Demo.fetch_periodic_hill_reference"
        )
        return 2
    except (KeyError, ValueError) as exc:
        print(str(exc))
        return 1

    print(
        "periodic-hill observation protocol: PASS "
        f"({len(protocol.designs)} seeds, {len(protocol.evaluation.points)} evaluation points)"
    )
    for check in leakage_audit.checks[:5]:
        print(f"{check.key}: {'PASS' if check.passed else 'FAIL'} -- {check.detail}")
    print("JSON protocol: " + str(args.output))
    print("JSON leakage audit: " + str(args.leakage_output))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
