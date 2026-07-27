from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from trade_rl.artifacts.codec import canonical_json_bytes
from trade_rl.artifacts.hashing import content_digest
from trade_rl.evaluation.causal_scenario_c3_adverse import C3AdverseFoldEvidence
from trade_rl.evaluation.causal_scenario_c3_contracts import (
    C3ReplayIdentity,
    CausalScenarioC3Config,
)
from trade_rl.workflows.causal_scenario import c3 as batch_module
from trade_rl.workflows.causal_scenario import c3_evaluation as module


def _sha(char: str) -> str:
    return char * 64


def _identity() -> C3ReplayIdentity:
    return C3ReplayIdentity(
        dataset_id=_sha("a"),
        fold_digest=_sha("b"),
        environment_digest=_sha("c"),
        action_spec_digest=_sha("d"),
        observation_digest=_sha("e"),
        execution_policy_digest=_sha("f"),
        risk_digest=_sha("1"),
        initial_state_digest=_sha("2"),
        query_index=100,
        query_timestamp_ns=1,
        realized_stop_index=196,
        aum=100_000.0,
    )


def _query(*, scenario: str = "nominal") -> batch_module.C3BatchQuery:
    query = object.__new__(batch_module.C3BatchQuery)
    object.__setattr__(query, "fold_id", "fold-0")
    object.__setattr__(query, "decision_root", Path("decision"))
    object.__setattr__(query, "replay", SimpleNamespace(identity=_identity()))
    object.__setattr__(query, "ppo_mean_action", np.asarray([0.0]))
    object.__setattr__(
        query,
        "prediction_evidence",
        SimpleNamespace(validate_for_decision=lambda decision: None),
    )
    object.__setattr__(query, "execution_scenario", scenario)
    object.__setattr__(query, "perfect_information", SimpleNamespace())
    return query


def _adverse() -> C3AdverseFoldEvidence:
    return C3AdverseFoldEvidence(
        fold_index=0,
        source_artifact_digest=_sha("7"),
        thresholds_digest=_sha("8"),
        required_scenario="joint_2x",
        selected_return=0.02,
        baseline_uplift=0.01,
        cost_fraction=0.01,
        turnover_per_day=0.5,
        maximum_drawdown=0.1,
        failed_conditions=(),
    )


def test_batch_binds_source_adverse_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loaded = SimpleNamespace(
        decision=SimpleNamespace(
            decision_digest=_sha("4"),
            fold_digest=_sha("b"),
        )
    )
    comparison = SimpleNamespace(
        decision_digest=_sha("4"), execution_scenario="nominal"
    )
    fold_report = SimpleNamespace()
    report = SimpleNamespace(digest=_sha("5"))
    gate = SimpleNamespace(report_digest=_sha("5"), digest=_sha("6"), passed=True)
    captured: dict[str, object] = {}
    monkeypatch.setattr(batch_module, "load_c3_decision_artifact", lambda root: loaded)
    monkeypatch.setattr(
        batch_module,
        "run_c3_query_comparison",
        lambda *args, **kwargs: comparison,
    )
    monkeypatch.setattr(
        batch_module,
        "build_c3_fold_report",
        lambda **kwargs: captured.update(kwargs) or fold_report,
    )
    monkeypatch.setattr(
        batch_module,
        "build_c3_aggregate_report",
        lambda *args, **kwargs: report,
    )
    monkeypatch.setattr(
        batch_module,
        "evaluate_phase_a_entry_gate",
        lambda *args, **kwargs: gate,
    )
    monkeypatch.setattr(
        batch_module,
        "write_c3_aggregate_report_artifact",
        lambda *args, **kwargs: _sha("9"),
    )
    monkeypatch.setattr(
        batch_module,
        "write_phase_a_gate_artifact",
        lambda *args, **kwargs: _sha("a"),
    )
    config = CausalScenarioC3Config(
        required_folds=1,
        required_selection_days=30,
        bootstrap_resamples=128,
        bootstrap_block_days=7,
    )
    monkeypatch.setattr(
        batch_module,
        "C3BatchResult",
        lambda **kwargs: SimpleNamespace(production_status="NO-GO", **kwargs),
    )

    result = batch_module.execute_c3_batch(
        (_query(),),
        output_root=tmp_path,
        fold_selection_days={"fold-0": 30},
        required_adverse_evidence={"fold-0": _adverse()},
        config=config,
    )

    assert captured["selection_days"] == 30
    assert captured["required_adverse_passed"] is True
    assert captured["required_adverse_evidence_digest"] == _adverse().digest
    assert result.production_status == "NO-GO"


def test_batch_rejects_missing_adverse_fold_mapping(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="required_adverse_evidence"):
        batch_module.execute_c3_batch(
            (_query(),),
            output_root=tmp_path,
            fold_selection_days={"fold-0": 30},
            required_adverse_evidence={},
            config=CausalScenarioC3Config(required_folds=1, required_selection_days=30),
        )


def _request_payload() -> dict[str, object]:
    return {
        "config": {
            "bootstrap_block_days": 7,
            "bootstrap_resamples": 128,
            "horizon_decisions": 96,
            "random_comparator_count": 8,
            "ranking_tolerance": 1e-8,
            "required_folds": 1,
            "required_selection_days": 30,
            "scenario_count": 64,
            "schema_version": "causal_scenario_c3_config_v1",
        },
        "folds": [
            {
                "fold_digest": _sha("b"),
                "fold_id": "fold-0",
                "fold_index": 0,
                "library": "library",
                "queries": [
                    {
                        "execution_scenario": "nominal",
                        "outcomes": [],
                        "perfect_information": {
                            "bound_log_return": None,
                            "causal_log_return": None,
                            "compatibility_evidence_digest": None,
                            "gap": None,
                            "reason": "not_evaluated",
                            "status": "not_evaluated",
                        },
                        "ppo_mean_action": [0.0, 0.0],
                        "value_artifact": "value",
                    },
                    {
                        "execution_scenario": "joint_2x",
                        "outcomes": [],
                        "perfect_information": {
                            "bound_log_return": None,
                            "causal_log_return": None,
                            "compatibility_evidence_digest": None,
                            "gap": None,
                            "reason": "not_evaluated",
                            "status": "not_evaluated",
                        },
                        "ppo_mean_action": [0.0, 0.0],
                        "value_artifact": "value",
                    },
                ],
            }
        ],
        "schema_version": "causal_scenario_c3_evaluation_request_v2",
        "source_walk_forward_run": "source-run",
    }


def test_request_schema_rejects_self_reported_support(tmp_path: Path) -> None:
    payload = _request_payload()
    fold = payload["folds"][0]
    fold["selection_days"] = 30
    fold["required_adverse_passed"] = True
    request = tmp_path / "request.json"
    request.write_bytes(canonical_json_bytes(payload))

    with pytest.raises(ValueError, match="field closure"):
        module.execute_c3_evaluation_request(request, output_root=tmp_path / "output")


def test_request_uses_source_bound_support_and_adverse_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = tmp_path / "request.json"
    request.write_bytes(canonical_json_bytes(_request_payload()))
    for name in ("source-run", "library", "value"):
        (tmp_path / name).mkdir()
    config_payload = {"schema_version": "market_walk_forward_config_v1"}
    config_digest = content_digest(config_payload)
    manifest = SimpleNamespace(
        digest=_sha("9"),
        dataset_id=_sha("a"),
        evaluation_digest=_sha("8"),
        workflow_config_digest=config_digest,
        fold_count=1,
    )
    walk_forward = {
        "dataset_id": _sha("a"),
        "evaluation_digest": _sha("8"),
        "folds": [
            {
                "fold_index": 0,
                "selection_range": [0, 2_880],
                "train_range": [0, 90],
                "test_range": [90, 300],
            }
        ],
    }
    adverse = _adverse()
    source_evidence = SimpleNamespace(
        by_fold_index={0: adverse},
        selection_days_by_fold={0: 30},
        required_scenario="joint_2x",
    )
    library = SimpleNamespace(
        dataset_id=_sha("a"),
        train_start=0,
        train_stop=90,
        library_digest=_sha("3"),
        config=SimpleNamespace(horizon_decisions=96, scenario_count=64),
    )
    value = SimpleNamespace(
        dataset_id=_sha("a"),
        fold_digest=_sha("b"),
        train_start=0,
        train_stop=90,
        query_index=100,
        scenario_library_digest=_sha("3"),
        config=SimpleNamespace(
            horizon_decisions=96,
            scenario_count=64,
            action_dimension=2,
        ),
    )
    decision = SimpleNamespace(decision_digest=_sha("4"), replay_identity=_identity())
    batch = SimpleNamespace(production_status="NO-GO")
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        module, "validate_walk_forward_run_directory", lambda path: manifest
    )
    monkeypatch.setattr(
        module, "_walk_forward_payload", lambda root, loaded: walk_forward
    )
    monkeypatch.setattr(
        module,
        "_walk_forward_config_payload",
        lambda root, loaded: config_payload,
    )
    monkeypatch.setattr(
        module,
        "load_c3_source_adverse_evidence",
        lambda *args, **kwargs: source_evidence,
    )
    monkeypatch.setattr(
        module, "load_causal_scenario_library_artifact", lambda path: library
    )
    monkeypatch.setattr(
        module, "load_causal_scenario_value_artifact", lambda path: value
    )
    monkeypatch.setattr(
        module, "build_persisted_scenario_decision", lambda loaded: decision
    )
    monkeypatch.setattr(
        module, "write_c3_decision_artifact", lambda *args, **kwargs: _sha("4")
    )
    monkeypatch.setattr(
        module,
        "build_c3_prediction_evidence",
        lambda loaded: SimpleNamespace(validate_for_decision=lambda created: None),
    )
    monkeypatch.setattr(
        module, "_load_replay_outcomes", lambda *args, **kwargs: {_sha("7"): object()}
    )
    monkeypatch.setattr(
        module, "ArtifactBackedC3Replay", lambda *args, **kwargs: object()
    )
    monkeypatch.setattr(
        module, "C3BatchQuery", lambda **kwargs: SimpleNamespace(**kwargs)
    )
    monkeypatch.setattr(
        module,
        "execute_c3_batch",
        lambda queries, **kwargs: captured.update(kwargs) or batch,
    )
    monkeypatch.setattr(
        module,
        "C3EvaluationResult",
        lambda **kwargs: SimpleNamespace(production_status="NO-GO", **kwargs),
    )

    result = module.execute_c3_evaluation_request(
        request,
        output_root=tmp_path / "output",
    )

    assert result.batch is batch
    assert captured["fold_selection_days"] == {"fold-0": 30}
    assert captured["required_adverse_evidence"] == {"fold-0": adverse}
    assert result.production_status == "NO-GO"


def test_request_requires_source_declared_adverse_scenario(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _request_payload()
    payload["folds"][0]["queries"] = payload["folds"][0]["queries"][:1]
    request = tmp_path / "request.json"
    request.write_bytes(canonical_json_bytes(payload))
    with pytest.raises(ValueError, match="required adverse scenario"):
        module._require_execution_scenarios(
            {"nominal"},
            required_scenario="joint_2x",
        )
