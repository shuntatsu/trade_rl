from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import trade_rl.workflows.universal_causal_alpha_v10_stage_entry as stage_entry
from trade_rl.artifacts.hashing import content_digest
from trade_rl.learning.causal_alpha_v10 import CausalAlphaV10Candidate
from trade_rl.learning.rollout_evaluation import ActionPathExecutionTrace


def _execution_evidence(
    *, steps: int = 2
) -> tuple[dict[str, object], dict[str, object]]:
    if steps not in (1, 2):
        raise ValueError("test execution evidence supports one or two steps")
    trace = ActionPathExecutionTrace(
        pre_action_weights=np.asarray([[0.0], [0.1]])[:steps],
        risk_constrained_weights=np.asarray([[0.1], [0.0]])[:steps],
        post_step_weights=np.asarray([[0.1], [0.0]])[:steps],
        applied_risk_scales=np.asarray([1.0, 1.0])[:steps],
        strategy_intent_changes=np.asarray([True, True])[:steps],
        realized_state_follows=np.asarray([False, False])[:steps],
        rebalance_reassertions=np.asarray([False, False])[:steps],
        hard_risk_violations=np.asarray([False, False])[:steps],
    )
    evaluation = SimpleNamespace(
        execution_trace=trace,
        collapse_evidence=SimpleNamespace(hard_risk_violation=False),
    )
    trace_payload = stage_entry._execution_trace_payload(evaluation)
    diagnostics = stage_entry._execution_diagnostics(evaluation, trace_payload)
    return trace_payload, diagnostics


def _metric(
    candidate: CausalAlphaV10Candidate,
    final_target_digest: str,
    *,
    decision_count: int = 2,
) -> object:
    return SimpleNamespace(
        candidate=stage_entry.V8_CANDIDATE_BY_V10[candidate],
        v8_target_path_digest=final_target_digest,
        v8_config_digest="c" * 64,
        v6_metric=SimpleNamespace(
            symbol="BTCUSDT",
            episode_index=8,
            contract_digest="d" * 64,
            decision_count=decision_count,
            hard_risk_violation=False,
        ),
        calibration_fit_digest="f" * 64,
        digest="m" * 64,
    )


def _load_leaf(
    monkeypatch: pytest.MonkeyPatch,
    *,
    execution_trace: dict[str, object],
    execution_diagnostics: dict[str, object],
    decision_count: int = 2,
) -> None:
    candidate = CausalAlphaV10Candidate.HIERARCHICAL_WAVE
    final_target_digest = "t" * 64
    metric = _metric(candidate, final_target_digest, decision_count=decision_count)

    class MetricFactory:
        @staticmethod
        def from_payload(_payload: object) -> object:
            return metric

    monkeypatch.setattr(stage_entry, "CausalAlphaV8ReplayMetric", MetricFactory)
    input_digest = "n" * 64
    leaf = {
        "candidate": candidate.value,
        "candidate_input_digest": input_digest,
        "execution_trace": execution_trace,
        "execution_diagnostics": execution_diagnostics,
        "replay": {},
        "replay_digest": metric.digest,
        "target_path_digest": final_target_digest,
        "target": {
            "artifact_digest": final_target_digest,
            "candidate": candidate.value,
            "hierarchy_input_digest": input_digest,
        },
    }

    class Store:
        config_digest = "c" * 64

        def load_leaf(self, _path: Path, *, expected_schema: str) -> dict[str, object]:
            assert expected_schema == "causal_alpha_v10_replay_leaf_v3"
            return leaf

    stage_entry._load(
        Store(),
        path=Path("selection/replays/08/BTCUSDT/hierarchical_wave.json"),
        candidate=candidate,
        candidate_input_digest=input_digest,
        expected_fast_fit_digest="f" * 64,
        expected_target_digest=None,
        symbol="BTCUSDT",
        episode=8,
        contract_digest="d" * 64,
    )


def test_v10_resume_rejects_stale_hierarchy_policy_input_digest(monkeypatch) -> None:
    candidate = CausalAlphaV10Candidate.HIERARCHICAL_WAVE
    final_target_digest = "t" * 64
    metric = _metric(candidate, final_target_digest)

    class MetricFactory:
        @staticmethod
        def from_payload(_payload: object) -> object:
            return metric

    monkeypatch.setattr(stage_entry, "CausalAlphaV8ReplayMetric", MetricFactory)

    stale_input_digest = "o" * 64
    execution_trace, execution_diagnostics = _execution_evidence()
    leaf = {
        "candidate": candidate.value,
        "candidate_input_digest": stale_input_digest,
        "execution_trace": execution_trace,
        "execution_diagnostics": execution_diagnostics,
        "replay": {},
        "replay_digest": metric.digest,
        "target_path_digest": final_target_digest,
        "target": {
            "artifact_digest": final_target_digest,
            "candidate": candidate.value,
            "hierarchy_input_digest": stale_input_digest,
        },
    }

    class Store:
        config_digest = "c" * 64

        def load_leaf(self, _path: Path, *, expected_schema: str) -> dict[str, object]:
            assert expected_schema == "causal_alpha_v10_replay_leaf_v3"
            return leaf

    with pytest.raises(ValueError, match="resumed replay identity drifted"):
        stage_entry._load(
            Store(),
            path=Path("selection/replays/08/BTCUSDT/hierarchical_wave.json"),
            candidate=candidate,
            candidate_input_digest="n" * 64,
            expected_fast_fit_digest="f" * 64,
            expected_target_digest=None,
            symbol="BTCUSDT",
            episode=8,
            contract_digest="d" * 64,
        )


def test_v10_resume_rejects_tampered_execution_diagnostics(monkeypatch) -> None:
    execution_trace, execution_diagnostics = _execution_evidence()
    execution_diagnostics = dict(execution_diagnostics)
    execution_diagnostics["artifact_digest"] = "0" * 64

    with pytest.raises(ValueError, match="execution diagnostics digest drifted"):
        _load_leaf(
            monkeypatch,
            execution_trace=execution_trace,
            execution_diagnostics=execution_diagnostics,
        )


def test_v10_resume_rejects_semantically_tampered_execution_diagnostics(
    monkeypatch,
) -> None:
    execution_trace, execution_diagnostics = _execution_evidence()
    tampered = dict(execution_diagnostics)
    tampered["strategy_intent_change_count"] = 0
    body = {key: value for key, value in tampered.items() if key != "artifact_digest"}
    tampered["artifact_digest"] = content_digest(body)

    with pytest.raises(ValueError, match="do not reconcile with execution trace"):
        _load_leaf(
            monkeypatch,
            execution_trace=execution_trace,
            execution_diagnostics=tampered,
        )


def test_v10_resume_rejects_execution_trace_with_wrong_decision_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    execution_trace, execution_diagnostics = _execution_evidence(steps=1)

    with pytest.raises(ValueError, match="resumed replay identity drifted"):
        _load_leaf(
            monkeypatch,
            execution_trace=execution_trace,
            execution_diagnostics=execution_diagnostics,
            decision_count=2,
        )
