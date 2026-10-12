"""Contracts for T10.4 observation design and leakage isolation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import numpy as np
import pytest

from RANS_PINN_Demo.periodic_hill import (
    CHANNEL_HEIGHT,
    REFERENCE_STATIONS,
    PeriodicHillProfile,
    hill_height,
)
from RANS_PINN_Demo.periodic_hill_observations import (
    BUDGETS,
    OBSERVATION_FIELDS,
    OBSERVATION_NOISE,
    OBSERVATION_SEEDS,
    TEST_STATIONS,
    TRAIN_STATIONS,
    VALIDATION_STATIONS,
    ObservationDesign,
    ObservationPoint,
    audit_observation_protocol,
    build_evaluation_grid,
    build_observation_design,
    build_observation_protocol,
    EvaluationIOPlan,
    assert_clean_validation_and_test,
    assert_frozen_station_split,
    assert_leakage_isolation,
    assert_nested_budgets,
    assert_no_observation_evaluation_overlap,
    build_observation_splits,
    load_observation_protocol,
    run_leakage_audit,
    save_leakage_audit,
    save_observation_design,
    write_observation_protocol,
)
from RANS_PINN_Demo.periodic_hill_evaluation import (
    FrozenRunInputs,
    assert_evaluation_report_complete,
    evaluate_frozen_run,
    evaluate_frozen_runs,
)


def _audit_check(audit, key: str):
    return next(check for check in audit.checks if check.key == key)

def _profiles(n: int = 32) -> Dict[float, PeriodicHillProfile]:
    profiles: Dict[float, PeriodicHillProfile] = {}
    for station in REFERENCE_STATIONS:
        wall = float(hill_height(np.array([station]))[0])
        y = np.linspace(wall + 0.01, CHANNEL_HEIGHT - 0.01, n)
        shape = np.linspace(0.0, 1.0, n)
        profiles[station] = PeriodicHillProfile(
            x_over_h=station,
            y_over_h=y,
            u_over_ub=1.25 * shape + 0.2,
            v_over_ub=-0.15 * shape + 0.03 * np.sin(3.0 * shape),
            uu_over_ub2=0.1 * shape,
            vv_over_ub2=0.08 * shape,
            uv_over_ub2=-0.02 * shape,
            k_over_ub2=0.13 * shape,
        )
    return profiles

def test_station_splits_are_frozen_and_partition_the_reference():
    assert TRAIN_STATIONS == (0.5, 1.0, 2.0, 6.0)
    assert VALIDATION_STATIONS == (3.0, 7.0)
    assert TEST_STATIONS == (0.05, 4.0, 5.0, 8.0)
    assert set(TRAIN_STATIONS).isdisjoint(VALIDATION_STATIONS)
    assert set(TRAIN_STATIONS).isdisjoint(TEST_STATIONS)
    assert set(VALIDATION_STATIONS).isdisjoint(TEST_STATIONS)
    assert set(TRAIN_STATIONS + VALIDATION_STATIONS + TEST_STATIONS) == set(
        REFERENCE_STATIONS
    )
    assert BUDGETS == {"B1": 8, "B2": 16}
    assert OBSERVATION_NOISE == pytest.approx(0.01)
    assert OBSERVATION_SEEDS == tuple(range(20))


def test_budgets_are_nested_unique_and_restricted_to_training_stations():
    design = build_observation_design(_profiles(), seed=0)
    b1 = design.budget("B1")
    b2 = design.budget("B2")
    assert len(b1) == len(TRAIN_STATIONS) * BUDGETS["B1"]
    assert len(b2) == len(TRAIN_STATIONS) * BUDGETS["B2"]
    assert {point.station for point in b1 + b2} == set(TRAIN_STATIONS)
    assert len({point.key for point in b1}) == len(b1)
    assert len({point.key for point in b2}) == len(b2)
    assert {point.key for point in b1} < {point.key for point in b2}

    for station in TRAIN_STATIONS:
        local_b1 = [point for point in b1 if point.station == station]
        local_b2 = [point for point in b2 if point.station == station]
        assert len(local_b1) == BUDGETS["B1"]
        assert len(local_b2) == BUDGETS["B2"]
        assert np.all(np.diff([point.wall_normal for point in local_b2]) >= 0.0)
        # Two observations per equal-count wall-normal stratum.
        strata = np.array_split(
            np.sort([point.wall_normal for point in local_b2]),
            BUDGETS["B1"],
        )
        for stratum in strata:
            assert len(stratum) == 2


def test_training_noise_is_reproducible_train_only_and_relative():
    profiles = _profiles()
    first = build_observation_design(profiles, seed=7)
    second = build_observation_design(profiles, seed=7)
    other = build_observation_design(profiles, seed=8)
    assert [point.noisy_u_over_ub for point in first.b2] == pytest.approx(
        [point.noisy_u_over_ub for point in second.b2]
    )
    assert [point.noisy_u_over_ub for point in first.b2] != pytest.approx(
        [point.noisy_u_over_ub for point in other.b2]
    )
    for point in first.b2:
        assert point.noise_relative == pytest.approx(OBSERVATION_NOISE)
        assert point.noisy_u_over_ub != pytest.approx(point.clean_u_over_ub)
    assert OBSERVATION_FIELDS == ("u_over_ub", "v_over_ub")

def test_evaluation_grid_is_clean_and_excludes_every_training_observation():
    profiles = _profiles()
    seeds = (0, 1, 2)
    protocol = build_observation_protocol(profiles, seeds=seeds, require_all_seeds=False)
    observed = {
        point.key
        for design in protocol.designs
        for budget in ("B1", "B2")
        for point in design.budget(budget)
    }
    evaluation_keys = {point.key for point in protocol.evaluation.points}
    assert evaluation_keys.isdisjoint(observed)
    assert len(evaluation_keys) == len(protocol.evaluation.points)
    assert all(not hasattr(point, "noisy_u_over_ub") for point in protocol.evaluation.points)

    test_keys = {
        (point.station, point.reference_index)
        for point in protocol.evaluation.points
        if point.station in TEST_STATIONS
    }
    expected_test_keys = {
        (station, index)
        for station, profile in profiles.items()
        if station in TEST_STATIONS
        for index in range(profile.y_over_h.size)
    }
    assert test_keys == expected_test_keys
    validation_keys = {
        (point.station, point.reference_index)
        for point in protocol.evaluation.points
        if point.station in VALIDATION_STATIONS
    }
    expected_validation_keys = {
        (station, index)
        for station, profile in profiles.items()
        if station in VALIDATION_STATIONS
        for index in range(profile.y_over_h.size)
    }
    assert validation_keys == expected_validation_keys
    assert protocol.leakage_audit.passed


def test_leakage_audit_rejects_a_test_station_observation():
    profiles = _profiles()
    design = build_observation_design(profiles, seed=0)
    source_point = design.b1[0]
    leaked = ObservationPoint(
        station=4.0,
        reference_index=source_point.reference_index,
        y_over_h=source_point.y_over_h,
        wall_normal=source_point.wall_normal,
        clean_u_over_ub=source_point.clean_u_over_ub,
        clean_v_over_ub=source_point.clean_v_over_ub,
        noisy_u_over_ub=source_point.noisy_u_over_ub,
        noisy_v_over_ub=source_point.noisy_v_over_ub,
        noise_relative=source_point.noise_relative,
    )
    bad_design = ObservationDesign(seed=0, b1=(leaked,) + design.b1[1:], b2=design.b2)
    evaluation = build_evaluation_grid(profiles, observed_keys={leaked.key})
    audit = audit_observation_protocol((bad_design,), evaluation)
    assert audit.passed is False
    assert any(
        check.key == "no_test_station_observations" and not check.passed
        for check in audit.checks
    )


def test_protocol_writer_records_frozen_configuration(tmp_path: Path):
    output = tmp_path / "protocol.json"
    protocol = build_observation_protocol(_profiles(), seeds=OBSERVATION_SEEDS)
    write_observation_protocol(output, protocol)
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["schema"] == "periodic-hill-observation-protocol/v1"
    assert payload["station_split"]["train"] == list(TRAIN_STATIONS)
    assert payload["station_split"]["validation"] == list(VALIDATION_STATIONS)
    assert payload["station_split"]["test"] == list(TEST_STATIONS)
    assert payload["budgets"] == BUDGETS
    assert payload["observation_noise"] == pytest.approx(OBSERVATION_NOISE)
    assert payload["evaluation"]["retrain"] is False
    assert payload["evaluation"]["reselect_points"] is False
    assert payload["leakage_audit"]["passed"] is True


def test_short_profiles_reduce_the_nested_budget_without_moving_stations():
    profiles = _profiles(n=6)
    design = build_observation_design(profiles, seed=0)
    for station in TRAIN_STATIONS:
        local_b1 = [point for point in design.b1 if point.station == station]
        local_b2 = [point for point in design.b2 if point.station == station]
        assert len(local_b1) == 5
        assert len(local_b2) == 6
        assert {point.key for point in local_b1} < {point.key for point in local_b2}
    protocol = build_observation_protocol(profiles, seeds=(0,), require_all_seeds=False)
    assert protocol.leakage_audit.passed


def test_protocol_loader_reaudits_the_frozen_artifact(tmp_path: Path):
    output = tmp_path / "protocol.json"
    protocol = build_observation_protocol(_profiles(), seeds=OBSERVATION_SEEDS)
    write_observation_protocol(output, protocol)
    loaded = load_observation_protocol(output)
    assert loaded.seeds == protocol.seeds
    assert loaded.evaluation.as_dict() == protocol.evaluation.as_dict()
    assert loaded.leakage_audit.passed


def test_declared_training_inputs_loss_and_tau_are_audited():
    profiles = _profiles()
    clean = run_leakage_audit(
        profiles,
        seeds=(0, 1),
        require_all_seeds=False,
        training_input_fields=("x_over_h", "wall_normal", "u_over_ub"),
    )
    assert clean.passed

    leaked_input = run_leakage_audit(
        profiles,
        seeds=(0, 1),
        require_all_seeds=False,
        training_input_fields=("uu_over_ub2",),
    )
    assert leaked_input.passed is False
    assert _audit_check(leaked_input, "training_inputs_exclude_reference_turbulence").passed is False
    with pytest.raises(AssertionError):
        assert_leakage_isolation(leaked_input)

    leaked_loss = run_leakage_audit(
        profiles,
        seeds=(0, 1),
        require_all_seeds=False,
        training_loss_stations=(4.0,),
    )
    assert leaked_loss.passed is False
    assert _audit_check(leaked_loss, "training_loss_uses_no_test_stations").passed is False

    leaked_tau = run_leakage_audit(
        profiles,
        seeds=(0, 1),
        require_all_seeds=False,
        training_loss_terms=("tau_w_reference",),
    )
    assert leaked_tau.passed is False
    assert _audit_check(leaked_tau, "wall_shear_reference_is_isolated").passed is False


def test_compatibility_split_surface_is_consumable_and_invariant():
    splits = build_observation_splits(_profiles(), seed=0)
    assert set(splits) == {"B1", "B2"}
    assert splits["B1"].requested_total_points == 32
    assert splits["B2"].requested_total_points == 64
    assert splits["B1"].actual_total_points == 32
    assert splits["B2"].actual_total_points == 64
    assert set(splits["B1"].train.inputs()) == {"x_over_h", "wall_normal"}
    assert set(splits["B1"].train.targets()) == {"u_over_ub"}
    assert all(point.noise_applied for point in splits["B1"].train.points)
    assert not any(point.noise_applied for point in splits["B1"].validation.points)
    assert not any(point.noise_applied for point in splits["B1"].test.points)
    assert_nested_budgets(splits)
    assert_no_observation_evaluation_overlap(splits["B1"])
    assert_clean_validation_and_test(splits["B1"])
    assert_frozen_station_split(splits["B1"])


def test_frozen_evaluation_io_rejects_training_reads_and_retraining():
    bad_io = EvaluationIOPlan(reads=("frozen_model", "training_files"), retrain=True)
    audit = run_leakage_audit(
        _profiles(),
        seeds=(0,),
        require_all_seeds=False,
        evaluation_io=bad_io,
    )
    assert audit.passed is False
    assert _audit_check(audit, "frozen_evaluation_io_is_isolated").passed is False


def test_audit_and_design_are_saved_as_json(tmp_path: Path):
    profiles = _profiles()
    splits = build_observation_splits(profiles, seed=0)
    audit = run_leakage_audit(
        profiles,
        seeds=(0,),
        require_all_seeds=False,
    )
    design_path = save_observation_design(splits, tmp_path / "observation_design.json")
    audit_path = save_leakage_audit(audit, tmp_path / "observation_leakage.json")
    design = json.loads(design_path.read_text(encoding="utf-8"))
    leakage = json.loads(audit_path.read_text(encoding="utf-8"))
    assert design["schema"] == "periodic-hill-observation-splits/v1"
    assert design["budgets"]["B1"]["actual_total_points"] == 32
    assert leakage["schema"] == "periodic-hill-leakage-audit/v1"
    assert leakage["all_passed"] is True







def test_formal_protocol_requires_the_complete_seed_set(tmp_path: Path):
    profiles = _profiles()
    with pytest.raises(ValueError):
        build_observation_protocol(profiles, seeds=(0,))
    partial = build_observation_protocol(
        profiles, seeds=(0,), require_all_seeds=False
    )
    with pytest.raises(ValueError):
        write_observation_protocol(tmp_path / "partial.json", partial)




def test_frozen_evaluation_reports_required_metrics_without_reselection():
    protocol = build_observation_protocol(
        _profiles(), seeds=(0,), require_all_seeds=False
    )
    predictions = {
        point.key: (point.u_over_ub, point.v_over_ub)
        for point in protocol.evaluation.points
    }
    tau = {station: 0.25 for station in REFERENCE_STATIONS}
    report = evaluate_frozen_run(
        protocol,
        FrozenRunInputs(
            seed=0,
            run_status="converged",
            predictions=predictions,
            tau_w_rans=tau,
            tau_w_reference=tau,
            force_balance_residual=0.0,
            separation=0.4,
            reattachment=4.2,
        ),
    )
    assert_evaluation_report_complete(report)
    assert report["same_station_relative_l2_U"] == pytest.approx(0.0)
    assert report["cross_station_relative_l2_U"] == pytest.approx(0.0)
    assert report["tau_w_relative_error_median"] == pytest.approx(0.0)
    assert report["failure_rate"] == pytest.approx(0.0)
    assert report["retrained"] is False
    assert report["points_reselected"] is False


def test_frozen_evaluation_marks_nonconverged_runs_without_averaging_them_in():
    protocol = build_observation_protocol(
        _profiles(), seeds=(0,), require_all_seeds=False
    )
    predictions = {
        point.key: (point.u_over_ub, point.v_over_ub)
        for point in protocol.evaluation.points
    }
    tau = {station: 0.25 for station in REFERENCE_STATIONS}
    report = evaluate_frozen_run(
        protocol,
        FrozenRunInputs(
            seed=0,
            run_status="max_iter",
            predictions=predictions,
            tau_w_rans=tau,
            tau_w_reference=tau,
            force_balance_residual=0.0,
            separation=0.4,
            reattachment=4.2,
        ),
    )
    assert report["eligible_for_accuracy"] is False
    assert report["failure_rate"] == pytest.approx(1.0)


def _frozen_run(protocol, status: str, *, separation=0.4, reattachment=4.2):
    predictions = {
        point.key: (point.u_over_ub, point.v_over_ub)
        for point in protocol.evaluation.points
    }
    tau = {station: 0.25 for station in REFERENCE_STATIONS}
    return FrozenRunInputs(
        seed=0,
        run_status=status,
        predictions=predictions,
        tau_w_rans=tau,
        tau_w_reference=tau,
        force_balance_residual=0.0,
        separation=separation,
        reattachment=reattachment,
    )


def test_converged_evaluation_requires_separation_and_reattachment():
    protocol = build_observation_protocol(
        _profiles(), seeds=(0,), require_all_seeds=False
    )
    with pytest.raises(ValueError):
        evaluate_frozen_run(
            protocol,
            _frozen_run(protocol, "converged", separation=None, reattachment=None),
        )


def test_batch_evaluation_reports_failure_rate_without_hiding_failed_runs():
    protocol = build_observation_protocol(
        _profiles(), seeds=(0,), require_all_seeds=False
    )
    report = evaluate_frozen_runs(
        protocol,
        [
            _frozen_run(protocol, "converged"),
            _frozen_run(protocol, "max_iter", separation=None, reattachment=None),
        ],
    )
    assert report["run_count"] == 2
    assert report["failure_count"] == 1
    assert report["failure_rate"] == pytest.approx(0.5)
    assert len(report["runs"]) == 2



def test_loader_rejects_seed_list_that_does_not_match_designs(tmp_path: Path):
    output = tmp_path / "protocol.json"
    protocol = build_observation_protocol(_profiles(), seeds=OBSERVATION_SEEDS)
    write_observation_protocol(output, protocol)
    payload = json.loads(output.read_text(encoding="utf-8"))
    payload["designs"] = payload["designs"][:1]
    tampered = tmp_path / "tampered.json"
    tampered.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError):
        load_observation_protocol(tampered)
