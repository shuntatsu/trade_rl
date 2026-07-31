from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from trade_rl.simulation.execution import ExecutionCostConfig
from trade_rl.simulation.execution_promotion import (
    EXECUTION_EVIDENCE_SCHEMA,
    ExecutionEvidence,
    ExecutionPromotionError,
    execution_evidence_from_cost,
    load_execution_evidence,
    validate_execution_promotion,
    write_execution_evidence,
)
from tests.evaluation.replay_support import (
    CANDIDATE_CONFIG_DIGEST,
    EVALUATION_RUN_DIGEST,
    FOLD,
    SEED,
    execution_artifact,
)
from trade_rl.simulation.execution_replay import write_execution_event_artifact

_DATASET_ID = "d" * 64
_COST = ExecutionCostConfig(path_mode="conservative")
_POLICY_DIGEST = _COST.execution_policy_digest


def _valid_evidence(
    tmp_path: Path,
    **changes: object,
) -> tuple[ExecutionEvidence, Path]:
    artifact = execution_artifact()
    event_path = write_execution_event_artifact(
        tmp_path / "order-events.json",
        artifact,
    )
    evidence = execution_evidence_from_cost(
        dataset_id=_DATASET_ID,
        cost=_COST,
        order_event_artifact_path=event_path,
        sensitivity_path_modes=("optimistic", "neutral", "conservative"),
    )
    return replace(evidence, **changes), event_path


def _validate(evidence: ExecutionEvidence, event_path: Path):
    return validate_execution_promotion(
        evidence,
        expected_policy_digest=_POLICY_DIGEST,
        event_artifact_path=event_path,
        expected_candidate_config_digest=CANDIDATE_CONFIG_DIGEST,
        expected_evaluation_run_digest=EVALUATION_RUN_DIGEST,
        expected_fold=FOLD,
        expected_seed=SEED,
    )


def test_valid_conservative_execution_evidence_promotes(tmp_path: Path) -> None:
    evidence, event_path = _valid_evidence(tmp_path)
    decision = _validate(evidence, event_path)
    assert decision.promotable is True
    assert decision.evidence_digest == evidence.digest
    assert decision.execution_policy_digest == _POLICY_DIGEST


@pytest.mark.parametrize("mode", ["neutral", "optimistic"])
def test_non_conservative_primary_evidence_cannot_promote(
    mode: str,
    tmp_path: Path,
) -> None:
    evidence, event_path = _valid_evidence(tmp_path, path_mode=mode)
    with pytest.raises(ExecutionPromotionError, match="conservative"):
        _validate(evidence, event_path)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"processing_bar_volume_capacity": False}, "processing-bar"),
        ({"partial_fill_carry": False}, "partial-fill"),
        ({"complete_order_evidence": False}, "complete order evidence"),
        ({"sensitivity_path_modes": ("optimistic", "neutral")}, "sensitivity"),
        ({"trigger_volume_fractions": (1.0, 0.75, 0.5, 0.0)}, "trigger volume"),
    ],
)
def test_incomplete_or_optimistic_execution_evidence_fails_closed(
    changes: dict[str, object],
    message: str,
    tmp_path: Path,
) -> None:
    evidence, event_path = _valid_evidence(tmp_path, **changes)
    with pytest.raises(ExecutionPromotionError, match=message):
        _validate(evidence, event_path)


def test_execution_policy_identity_must_match_experiment_plan(tmp_path: Path) -> None:
    evidence, event_path = _valid_evidence(tmp_path)
    with pytest.raises(ExecutionPromotionError, match="policy digest"):
        validate_execution_promotion(
            evidence,
            expected_policy_digest="f" * 64,
            event_artifact_path=event_path,
        )


def test_execution_evidence_round_trips_canonically(tmp_path: Path) -> None:
    evidence, _ = _valid_evidence(tmp_path)
    path = tmp_path / "execution-evidence.json"
    write_execution_evidence(path, evidence)
    assert load_execution_evidence(path) == evidence
    assert load_execution_evidence(path).schema_version == EXECUTION_EVIDENCE_SCHEMA
    with pytest.raises(FileExistsError, match="already exists"):
        write_execution_evidence(path, replace(evidence, order_event_count=13))


def test_execution_evidence_uses_the_complete_cost_policy_digest() -> None:
    cost = ExecutionCostConfig(path_mode="conservative")
    evidence = execution_evidence_from_cost(dataset_id=_DATASET_ID, cost=cost)

    assert evidence.execution_policy_digest == cost.execution_policy_digest
    assert (
        evidence.execution_policy_digest
        != replace(
            cost,
            fee_rate=cost.fee_rate + 0.001,
        ).execution_policy_digest
    )
