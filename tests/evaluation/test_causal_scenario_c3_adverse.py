from __future__ import annotations

from copy import deepcopy

import pytest

from trade_rl.artifacts.hashing import content_digest
from trade_rl.evaluation.causal_scenario_c3_adverse import (
    build_c3_adverse_thresholds,
    evaluate_c3_adverse_fold,
    selection_days_from_source_fold,
)


def _source_config() -> dict[str, object]:
    return {
        "candidates": [
            {
                "name": "residual-ppo-15m",
                "run": {"environment": {"decision_hours": 0.25}},
            }
        ],
        "execution_sensitivity": {
            "maximum_drawdown": 0.20,
            "minimum_baseline_uplift": 0.0,
            "minimum_selected_return": 0.0,
            "required_scenario": "joint_2x",
            "scenarios": [
                {"name": "nominal", "report_only": False},
                {"name": "joint_2x", "report_only": False},
            ],
            "schema_version": "execution_sensitivity_config_v1",
        },
        "maximum_selection_cost_fraction": 0.03,
        "maximum_selection_drawdown": 0.20,
        "maximum_selection_turnover_per_day": 1.0,
        "minimum_selection_uplift": 0.001,
        "schema_version": "market_walk_forward_config_v1",
    }


def _scenario_result() -> dict[str, object]:
    payload: dict[str, object] = {
        "baseline": {
            "cost_fraction": 0.01,
            "maximum_drawdown": 0.08,
            "total_return": 0.01,
            "turnover_per_day": 0.30,
        },
        "baseline_uplift": 0.002,
        "report_only": False,
        "scenario": {"name": "joint_2x", "report_only": False},
        "selected": {
            "cost_fraction": 0.02,
            "maximum_drawdown": 0.10,
            "total_return": 0.03,
            "turnover_per_day": 0.80,
        },
    }
    payload["scenario_result_digest"] = content_digest(payload)
    return payload


def test_adverse_thresholds_are_derived_from_published_run_config() -> None:
    thresholds = build_c3_adverse_thresholds(_source_config())
    assert thresholds.required_scenario == "joint_2x"
    assert thresholds.minimum_selected_return == 0.0
    assert thresholds.minimum_baseline_uplift == 0.001
    assert thresholds.maximum_cost_fraction == 0.03
    assert thresholds.maximum_turnover_per_day == 1.0
    assert thresholds.maximum_drawdown == 0.20
    assert thresholds.decision_hours == 0.25


def test_selection_days_are_derived_from_source_fold() -> None:
    thresholds = build_c3_adverse_thresholds(_source_config())
    assert (
        selection_days_from_source_fold(
            {"selection_range": [100, 2_980]},
            thresholds=thresholds,
        )
        == 30
    )


def test_required_adverse_gate_is_computed_not_self_asserted() -> None:
    thresholds = build_c3_adverse_thresholds(_source_config())
    evidence = evaluate_c3_adverse_fold(
        fold_index=0,
        scenario_result=_scenario_result(),
        thresholds=thresholds,
        source_artifact_digest="a" * 64,
    )
    assert evidence.passed is True
    assert evidence.failed_conditions == ()

    cases = {
        "cost_fraction": ("selected", "cost_fraction", 0.031),
        "turnover_per_day": ("selected", "turnover_per_day", 1.01),
        "maximum_drawdown": ("selected", "maximum_drawdown", 0.21),
        "baseline_uplift": (None, "baseline_uplift", 0.0009),
        "selected_return": ("selected", "total_return", 0.0),
    }
    for expected, (section, field, value) in cases.items():
        payload = deepcopy(_scenario_result())
        payload.pop("scenario_result_digest")
        if section is None:
            payload[field] = value
        else:
            nested = payload[section]
            assert isinstance(nested, dict)
            nested[field] = value
        payload["scenario_result_digest"] = content_digest(payload)
        failed = evaluate_c3_adverse_fold(
            fold_index=0,
            scenario_result=payload,
            thresholds=thresholds,
            source_artifact_digest="a" * 64,
        )
        assert expected in failed.failed_conditions
        assert failed.passed is False


def test_adverse_scenario_digest_tampering_fails_closed() -> None:
    thresholds = build_c3_adverse_thresholds(_source_config())
    payload = _scenario_result()
    payload["baseline_uplift"] = 0.5
    with pytest.raises(ValueError, match="digest"):
        evaluate_c3_adverse_fold(
            fold_index=0,
            scenario_result=payload,
            thresholds=thresholds,
            source_artifact_digest="a" * 64,
        )


def test_missing_predeclared_threshold_fails_closed() -> None:
    config = _source_config()
    config["maximum_selection_cost_fraction"] = None
    with pytest.raises(ValueError, match="maximum_selection_cost_fraction"):
        build_c3_adverse_thresholds(config)
