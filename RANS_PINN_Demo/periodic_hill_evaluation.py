"""Frozen-run evaluation for the 2-D periodic-hill observation protocol.

This module is deliberately split from training.  It loads the pre-registered
evaluation grid, requires a prediction for every frozen point, and reports the
same-station, cross-station, wall-shear, force-balance, separation/reattachment,
and run-failure fields required by section 5.3 of the 2-D plan.  A caller cannot
make it retrain or reselect points.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np

from .periodic_hill import REFERENCE_STATIONS
from .periodic_hill_observations import (
    OBSERVATION_SEEDS,
    EvaluationPoint,
    ObservationProtocol,
    load_observation_protocol,
)

__all__ = [
    "REQUIRED_EVALUATION_METRICS",
    "FrozenRunInputs",
    "evaluate_frozen_run",
    "evaluate_frozen_runs",
    "assert_evaluation_report_complete",
    "write_evaluation_report",
    "main",
]

REQUIRED_EVALUATION_METRICS: Tuple[str, ...] = (
    "same_station_relative_l2_U",
    "cross_station_relative_l2_U",
    "validation_relative_l2_U",
    "tau_w_relative_error",
    "force_balance_residual",
    "separation",
    "reattachment",
    "failure_rate",
)

PointKey = Tuple[float, int]


@dataclass(frozen=True)
class FrozenRunInputs:
    """Precomputed quantities from one frozen model/solver run."""

    seed: int
    run_status: str
    predictions: Mapping[PointKey, Tuple[float, float]]
    tau_w_rans: Mapping[float, float]
    tau_w_reference: Mapping[float, float]
    force_balance_residual: float
    separation: Optional[float]
    reattachment: Optional[float]


def _relative_l2(
    points: Sequence[EvaluationPoint],
    predictions: Mapping[PointKey, Tuple[float, float]],
    component: int,
) -> Optional[float]:
    if not points:
        return None
    reference = np.array(
        [
            point.u_over_ub if component == 0 else point.v_over_ub
            for point in points
        ],
        dtype=float,
    )
    predicted = np.array(
        [predictions[point.key][component] for point in points],
        dtype=float,
    )
    denominator = float(np.linalg.norm(reference))
    if not math.isfinite(denominator) or denominator <= 0.0:
        raise ValueError("reference profile has zero or non-finite L2 norm")
    return float(np.linalg.norm(predicted - reference) / denominator)


def _tau_errors(inputs: FrozenRunInputs) -> Dict[str, float]:
    missing = [
        station
        for station in REFERENCE_STATIONS
        if station not in inputs.tau_w_rans or station not in inputs.tau_w_reference
    ]
    if missing:
        raise ValueError(f"missing tau_w values at stations {missing}")
    errors: Dict[str, float] = {}
    for station in REFERENCE_STATIONS:
        reference = float(inputs.tau_w_reference[station])
        predicted = float(inputs.tau_w_rans[station])
        if not math.isfinite(reference) or not math.isfinite(predicted):
            raise ValueError(f"non-finite tau_w at x/h={station}")
        if abs(reference) <= 0.0:
            raise ValueError(f"zero reference tau_w at x/h={station}")
        errors[str(station)] = abs(predicted - reference) / abs(reference)
    return errors


def evaluate_frozen_run(
    protocol: ObservationProtocol,
    inputs: FrozenRunInputs,
) -> Dict[str, Any]:
    """Evaluate one frozen run without training or point reselection."""

    if int(inputs.seed) not in protocol.seeds:
        raise ValueError(f"seed {inputs.seed!r} is not part of the frozen protocol")
    evaluation_keys = protocol.evaluation.keys
    prediction_keys = {(float(x), int(i)) for x, i in inputs.predictions}
    if prediction_keys != evaluation_keys:
        missing = sorted(evaluation_keys - prediction_keys)[:5]
        extra = sorted(prediction_keys - evaluation_keys)[:5]
        raise ValueError(f"prediction keys do not match frozen grid; missing={missing} extra={extra}")

    points = protocol.evaluation.points
    by_role = {
        "same_station_remaining": [p for p in points if p.role == "same_station_remaining"],
        "test": [p for p in points if p.role == "test"],
        "validation": [p for p in points if p.role == "validation"],
    }
    tau_errors = _tau_errors(inputs)
    tau_values = np.array(list(tau_errors.values()), dtype=float)
    status = str(inputs.run_status)
    converged = status == "converged"
    failure_rate = 0.0 if converged else 1.0

    report: Dict[str, Any] = {
        "schema": "periodic-hill-frozen-evaluation/v1",
        "seed": int(inputs.seed),
        "run_status": status,
        "converged": converged,
        "eligible_for_accuracy": converged,
        "same_station_relative_l2_U": _relative_l2(
            by_role["same_station_remaining"], inputs.predictions, 0
        ),
        "same_station_relative_l2_V": _relative_l2(
            by_role["same_station_remaining"], inputs.predictions, 1
        ),
        "cross_station_relative_l2_U": _relative_l2(by_role["test"], inputs.predictions, 0),
        "cross_station_relative_l2_V": _relative_l2(by_role["test"], inputs.predictions, 1),
        "validation_relative_l2_U": _relative_l2(by_role["validation"], inputs.predictions, 0),
        "validation_relative_l2_V": _relative_l2(by_role["validation"], inputs.predictions, 1),
        "tau_w_relative_error": tau_errors,
        "tau_w_relative_error_mean": float(np.mean(tau_values)),
        "tau_w_relative_error_median": float(np.median(tau_values)),
        "force_balance_residual": float(inputs.force_balance_residual),
        "separation": None if inputs.separation is None else float(inputs.separation),
        "reattachment": None if inputs.reattachment is None else float(inputs.reattachment),
        "failure_rate": failure_rate,
        "retrained": False,
        "points_reselected": False,
    }
    if not all(
        math.isfinite(float(report[key]))
        for key in (
            "force_balance_residual",
        )
    ):
        raise ValueError("force_balance_residual must be finite")
    if report["separation"] is not None and not math.isfinite(report["separation"]):
        raise ValueError("separation must be finite or null")
    if report["reattachment"] is not None and not math.isfinite(report["reattachment"]):
        raise ValueError("reattachment must be finite or null")
    if converged and (report["separation"] is None or report["reattachment"] is None):
        raise ValueError(
            "converged runs must report both separation and reattachment positions"
        )
    return report


def _mean_or_none(values: Sequence[Optional[float]]) -> Optional[float]:
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return float(np.mean(finite)) if finite else None


def _median_or_none(values: Sequence[Optional[float]]) -> Optional[float]:
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return float(np.median(finite)) if finite else None


def evaluate_frozen_runs(
    protocol: ObservationProtocol,
    runs: Sequence[FrozenRunInputs],
) -> Dict[str, Any]:
    """Evaluate a frozen batch and aggregate the failure rate without hiding runs."""

    if not runs:
        raise ValueError("at least one frozen run is required")
    reports = [evaluate_frozen_run(protocol, run) for run in runs]
    eligible = [report for report in reports if report["eligible_for_accuracy"]]
    failure_count = len(reports) - len(eligible)
    tau_aggregate: Dict[str, Optional[float]] = {}
    for station in REFERENCE_STATIONS:
        key = str(station)
        tau_aggregate[key] = _median_or_none(
            [report["tau_w_relative_error"].get(key) for report in eligible]
        )
    report = {
        "schema": "periodic-hill-frozen-evaluation-batch/v1",
        "run_count": len(reports),
        "failure_count": failure_count,
        "failure_rate": float(failure_count) / float(len(reports)),
        "same_station_relative_l2_U": _mean_or_none(
            [item["same_station_relative_l2_U"] for item in eligible]
        ),
        "cross_station_relative_l2_U": _mean_or_none(
            [item["cross_station_relative_l2_U"] for item in eligible]
        ),
        "validation_relative_l2_U": _mean_or_none(
            [item["validation_relative_l2_U"] for item in eligible]
        ),
        "tau_w_relative_error": tau_aggregate,
        "force_balance_residual": _mean_or_none(
            [item["force_balance_residual"] for item in eligible]
        ),
        "separation": _median_or_none([item["separation"] for item in eligible]),
        "reattachment": _median_or_none([item["reattachment"] for item in eligible]),
        "runs": reports,
        "retrained": False,
        "points_reselected": False,
    }
    assert_evaluation_report_complete(report)
    return report


def assert_evaluation_report_complete(report: Mapping[str, Any]) -> None:
    missing = [key for key in REQUIRED_EVALUATION_METRICS if key not in report]
    if missing:
        raise AssertionError("evaluation report is missing metrics: " + repr(missing))
    if report.get("retrained") is not False or report.get("points_reselected") is not False:
        raise AssertionError("evaluation report indicates retraining or point reselection")


def write_evaluation_report(report: Mapping[str, Any], destination: Path) -> Path:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return destination


def _parse_predictions(payload: Mapping[str, Any]) -> Dict[PointKey, Tuple[float, float]]:
    predictions: Dict[PointKey, Tuple[float, float]] = {}
    for station, rows in payload.items():
        for index, values in rows.items():
            predictions[(float(station), int(index))] = (float(values[0]), float(values[1]))
    return predictions


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--frozen-model", type=Path, required=True)
    parser.add_argument(
        "--run",
        type=Path,
        required=True,
        nargs="+",
        help="one or more precomputed frozen-run JSON files",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    try:
        protocol = load_observation_protocol(args.protocol)
        if not args.frozen_model.is_file():
            raise FileNotFoundError(f"frozen model not found: {args.frozen_model}")
        expected_model_hash = _sha256(args.frozen_model)
        inputs = []
        for run_path in args.run:
            payload = json.loads(Path(run_path).read_text(encoding="utf-8"))
            if payload.get("frozen_model_sha256") != expected_model_hash:
                raise ValueError(
                    f"{run_path}: frozen_model_sha256 does not match --frozen-model"
                )
            inputs.append(
                FrozenRunInputs(
                    seed=int(payload["seed"]),
                    run_status=str(payload["run_status"]),
                    predictions=_parse_predictions(payload["predictions"]),
                    tau_w_rans={
                        float(k): float(v) for k, v in payload["tau_w_rans"].items()
                    },
                    tau_w_reference={
                        float(k): float(v)
                        for k, v in payload["tau_w_reference"].items()
                    },
                    force_balance_residual=float(payload["force_balance_residual"]),
                    separation=None
                    if payload.get("separation") is None
                    else float(payload["separation"]),
                    reattachment=None
                    if payload.get("reattachment") is None
                    else float(payload["reattachment"]),
                )
            )
        report = evaluate_frozen_runs(protocol, inputs)
        report["frozen_model_sha256"] = expected_model_hash
        write_evaluation_report(report, args.output)
    except (FileNotFoundError, KeyError, ValueError) as exc:
        print(str(exc))
        return 1
    print("periodic-hill frozen evaluation: PASS")
    print("JSON report: " + str(args.output))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
