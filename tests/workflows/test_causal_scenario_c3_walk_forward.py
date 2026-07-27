from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from trade_rl.artifacts.hashing import content_digest
from trade_rl.evaluation.causal_scenario_c3_contracts import (
    C3ReplayIdentity,
    RealizedPolicyOutcome,
)
from trade_rl.workflows.causal_scenario import c3_evaluation as module


def _sha(char: str) -> str:
    return char * 64


def _outcome(policy_kind: str, value: float) -> RealizedPolicyOutcome:
    payload = {
        "borrow_paid": 0.0,
        "cancel_replace_events": 0,
        "fees": 0.0001,
        "fill_count": 1,
        "fill_ratio": 1.0,
        "filled_turnover": 0.1,
        "funding_paid": 0.0,
        "gross_log_return": value,
        "impact_cost": 0.0001,
        "max_drawdown": 0.05,
        "pending_order_events": 0,
        "policy_kind": policy_kind,
        "schema_version": "causal_scenario_c3_realized_outcome_v1",
        "spread_cost": 0.0001,
        "terminal_equity": 100_000.0 * float(np.exp(value)),
        "termination_reason": "horizon",
    }
    return RealizedPolicyOutcome(
        policy_kind=policy_kind,
        gross_log_return=value,
        filled_turnover=0.1,
        fees=0.0001,
        spread_cost=0.0001,
        impact_cost=0.0001,
        funding_paid=0.0,
        borrow_paid=0.0,
        fill_ratio=1.0,
        fill_count=1,
        pending_order_events=0,
        cancel_replace_events=0,
        max_drawdown=0.05,
        terminal_equity=100_000.0 * float(np.exp(value)),
        termination_reason="horizon",
        outcome_digest=content_digest(payload),
    )


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


def test_artifact_backed_replay_is_exact_and_fail_closed() -> None:
    action = np.asarray([0.5, 0.0], dtype=np.float64)
    outcome = _outcome("ppo_mean", 0.02)
    replay = module.ArtifactBackedC3Replay(
        _identity(),
        {module._action_key("ppo_mean", action): outcome},
    )
    clone = replay.clone_for_replay()
    assert clone is not replay
    assert (
        clone.run(
            action,
            horizon_decisions=96,
            zero_residual_after_first=True,
            policy_kind="ppo_mean",
        ).outcome_digest
        == outcome.outcome_digest
    )
    with pytest.raises(ValueError, match="missing"):
        clone.run(
            np.asarray([0.0, 0.0]),
            horizon_decisions=96,
            zero_residual_after_first=True,
            policy_kind="trend",
        )
    with pytest.raises(ValueError, match="horizon"):
        clone.run(
            action,
            horizon_decisions=95,
            zero_residual_after_first=True,
            policy_kind="ppo_mean",
        )


def _request_payload() -> dict[str, object]:
    return {
        "schema_version": "causal_scenario_c3_evaluation_request_v1",
        "source_walk_forward_run": "source-run",
        "config": {
            "bootstrap_block_days": 7,
            "bootstrap_resamples": 128,
            "horizon_decisions": 96,
            "random_comparator_count": 8,
            "ranking_tolerance": 1e-8,
            "required_folds": 6,
            "required_selection_days": 180,
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
                        "execution_scenario": "cost_2x",
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
                "required_adverse_passed": True,
                "selection_days": 30,
            }
        ],
    }


def test_published_run_lifecycle_validates_and_publishes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = tmp_path / "request.json"
    request.write_text(json.dumps(_request_payload()), encoding="utf-8")
    for name in ("source-run", "library", "value"):
        (tmp_path / name).mkdir()

    manifest = SimpleNamespace(
        digest=_sha("9"),
        dataset_id=_sha("a"),
        evaluation_digest=_sha("8"),
        fold_count=1,
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
    decision = SimpleNamespace(
        decision_digest=_sha("4"),
        replay_identity=_identity(),
    )
    calls: list[str] = []
    monkeypatch.setattr(
        module,
        "validate_walk_forward_run_directory",
        lambda path: calls.append(f"run:{path.name}") or manifest,
    )
    monkeypatch.setattr(
        module,
        "_walk_forward_payload",
        lambda root, loaded: {
            "dataset_id": loaded.dataset_id,
            "evaluation_digest": loaded.evaluation_digest,
            "folds": [
                {
                    "fold_index": 0,
                    "train_range": [0, 90],
                    "test_range": [90, 300],
                }
            ],
        },
    )
    monkeypatch.setattr(
        module,
        "load_causal_scenario_library_artifact",
        lambda path: calls.append(f"library:{path.name}") or library,
    )
    monkeypatch.setattr(
        module,
        "load_causal_scenario_value_artifact",
        lambda path: calls.append(f"value:{path.name}") or value,
    )
    monkeypatch.setattr(
        module,
        "build_persisted_scenario_decision",
        lambda loaded: calls.append("decision") or decision,
    )
    monkeypatch.setattr(
        module,
        "write_c3_decision_artifact",
        lambda root, created: calls.append(f"write:{root.name}") or created.decision_digest,
    )
    monkeypatch.setattr(
        module,
        "build_c3_prediction_evidence",
        lambda loaded: SimpleNamespace(validate_for_decision=lambda created: None),
    )
    monkeypatch.setattr(
        module,
        "_load_replay_outcomes",
        lambda *args, **kwargs: {_sha("7"): _outcome("trend", 0.0)},
    )
    monkeypatch.setattr(module, "C3BatchQuery", lambda **kwargs: SimpleNamespace(**kwargs))
    batch = SimpleNamespace(production_status="NO-GO")
    monkeypatch.setattr(
        module,
        "execute_c3_batch",
        lambda queries, **kwargs: calls.append(f"batch:{len(queries)}") or batch,
    )

    result = module.execute_c3_evaluation_request(
        request,
        output_root=tmp_path / "output",
    )
    assert result.source_run_digest == manifest.digest
    assert result.batch is batch
    assert calls == [
        "run:source-run",
        "library:library",
        "value:value",
        "decision",
        f"write:{decision.decision_digest}",
        "value:value",
        "decision",
        f"write:{decision.decision_digest}",
        "batch:2",
    ]


def test_request_rejects_unsafe_source_path(tmp_path: Path) -> None:
    payload = _request_payload()
    payload["source_walk_forward_run"] = "../outside"
    request = tmp_path / "request.json"
    request.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="safe relative path"):
        module.execute_c3_evaluation_request(request, output_root=tmp_path / "output")
