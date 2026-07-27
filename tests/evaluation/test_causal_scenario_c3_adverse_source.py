from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

from trade_rl.artifacts.codec import canonical_json_bytes
from trade_rl.artifacts.hashing import content_digest
from trade_rl.evaluation.causal_scenario_c3_adverse import (
    load_c3_source_adverse_evidence,
)


def _sha(char: str) -> str:
    return char * 64


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
            "returns": [0.001, 0.002],
            "total_return": 0.003002,
            "turnover_per_day": 0.30,
        },
        "baseline_uplift": 0.002,
        "report_only": False,
        "scenario": {"name": "joint_2x", "report_only": False},
        "selected": {
            "cost_fraction": 0.02,
            "maximum_drawdown": 0.10,
            "returns": [0.002, 0.003],
            "total_return": 0.005006,
            "turnover_per_day": 0.80,
        },
    }
    payload["scenario_result_digest"] = content_digest(payload)
    return payload


def _artifact() -> dict[str, object]:
    sensitivity = _source_config()["execution_sensitivity"]
    assert isinstance(sensitivity, dict)
    scenario_pack_digest = content_digest(sensitivity)
    access: dict[str, object] = {
        "base_access_digest": _sha("1"),
        "dataset_id": _sha("a"),
        "experiment_plan_digest": _sha("2"),
        "fold_index": 0,
        "purpose": "post_selection_execution_sensitivity",
        "scenario_pack_digest": scenario_pack_digest,
        "selected_configuration": "residual-ppo-15m",
        "selected_policy_digest": _sha("3"),
        "test_range": [3_000, 5_880],
    }
    access["access_digest"] = content_digest(access)
    payload: dict[str, object] = {
        "dataset_id": _sha("a"),
        "experiment_plan_digest": _sha("2"),
        "folds": [
            {
                "access": access,
                "fold_index": 0,
                "scenarios": [_scenario_result()],
            }
        ],
        "gate": {
            "baseline_total_return": 0.003002,
            "baseline_uplift": 0.002004,
            "maximum_drawdown_threshold": 0.20,
            "maximum_fold_drawdown": 0.10,
            "minimum_baseline_uplift": 0.0,
            "minimum_selected_return": 0.0,
            "passed": True,
            "required_scenario": "joint_2x",
            "selected_total_return": 0.005006,
        },
        "production_status": "NO-GO",
        "scenario_pack_digest": scenario_pack_digest,
        "schema_version": "execution_sensitivity_v1",
    }
    payload["artifact_digest"] = content_digest(payload)
    return payload


def _write_artifact(root: Path, payload: dict[str, object]) -> None:
    root.mkdir()
    (root / "execution-sensitivity.json").write_bytes(canonical_json_bytes(payload))


def test_source_adverse_evidence_is_bound_to_fold_and_config(tmp_path: Path) -> None:
    root = tmp_path / "run"
    payload = _artifact()
    _write_artifact(root, payload)
    evidence = load_c3_source_adverse_evidence(
        root,
        walk_forward_config=_source_config(),
        source_folds={
            0: {
                "fold_index": 0,
                "selection_range": [100, 2_980],
                "test_range": [3_000, 5_880],
            }
        },
        dataset_id=_sha("a"),
    )
    assert evidence.source_artifact_digest == payload["artifact_digest"]
    assert evidence.required_scenario == "joint_2x"
    assert evidence.selection_days_by_fold == {0: 30}
    assert evidence.by_fold_index[0].passed is True


def test_source_adverse_artifact_tampering_fails_closed(tmp_path: Path) -> None:
    root = tmp_path / "run"
    payload = _artifact()
    payload["dataset_id"] = _sha("b")
    _write_artifact(root, payload)
    with pytest.raises(ValueError, match="artifact digest"):
        load_c3_source_adverse_evidence(
            root,
            walk_forward_config=_source_config(),
            source_folds={
                0: {
                    "fold_index": 0,
                    "selection_range": [100, 2_980],
                    "test_range": [3_000, 5_880],
                }
            },
            dataset_id=_sha("a"),
        )


def test_source_adverse_access_range_mismatch_fails_closed(tmp_path: Path) -> None:
    root = tmp_path / "run"
    payload = _artifact()
    folds = payload["folds"]
    assert isinstance(folds, list)
    fold = folds[0]
    assert isinstance(fold, dict)
    access = fold["access"]
    assert isinstance(access, dict)
    access.pop("access_digest")
    access["test_range"] = [3_001, 5_880]
    access["access_digest"] = content_digest(access)
    payload.pop("artifact_digest")
    payload["artifact_digest"] = content_digest(payload)
    _write_artifact(root, payload)
    with pytest.raises(ValueError, match="test range"):
        load_c3_source_adverse_evidence(
            root,
            walk_forward_config=_source_config(),
            source_folds={
                0: {
                    "fold_index": 0,
                    "selection_range": [100, 2_980],
                    "test_range": [3_000, 5_880],
                }
            },
            dataset_id=_sha("a"),
        )


def test_source_adverse_required_scenario_is_not_substitutable(tmp_path: Path) -> None:
    root = tmp_path / "run"
    payload = deepcopy(_artifact())
    folds = payload["folds"]
    assert isinstance(folds, list)
    fold = folds[0]
    assert isinstance(fold, dict)
    scenarios = fold["scenarios"]
    assert isinstance(scenarios, list)
    scenario = scenarios[0]
    assert isinstance(scenario, dict)
    scenario.pop("scenario_result_digest")
    scenario_config = scenario["scenario"]
    assert isinstance(scenario_config, dict)
    scenario_config["name"] = "joint_5x"
    scenario["scenario_result_digest"] = content_digest(scenario)
    payload.pop("artifact_digest")
    payload["artifact_digest"] = content_digest(payload)
    _write_artifact(root, payload)
    with pytest.raises(ValueError, match="required scenario"):
        load_c3_source_adverse_evidence(
            root,
            walk_forward_config=_source_config(),
            source_folds={
                0: {
                    "fold_index": 0,
                    "selection_range": [100, 2_980],
                    "test_range": [3_000, 5_880],
                }
            },
            dataset_id=_sha("a"),
        )
