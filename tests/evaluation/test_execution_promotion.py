from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from trade_rl.simulation.execution_promotion import (
    EXECUTION_EVIDENCE_SCHEMA,
    ExecutionEvidence,
    ExecutionPromotionError,
    load_execution_evidence,
    validate_execution_promotion,
    write_execution_evidence,
)


def _valid_evidence(**changes: object) -> ExecutionEvidence:
    evidence = ExecutionEvidence(
        dataset_id="d" * 64,
        execution_policy_digest="e" * 64,
        path_mode="conservative",
        processing_bar_volume_capacity=True,
        partial_fill_carry=True,
        trigger_volume_fractions=(1.0, 0.5, 0.25, 0.0),
        order_event_count=12,
        complete_order_evidence=True,
        sensitivity_path_modes=("optimistic", "neutral", "conservative"),
    )
    return replace(evidence, **changes)


def test_valid_conservative_execution_evidence_promotes() -> None:
    evidence = _valid_evidence()
    decision = validate_execution_promotion(
        evidence,
        expected_policy_digest="e" * 64,
    )
    assert decision.promotable is True
    assert decision.evidence_digest == evidence.digest
    assert decision.execution_policy_digest == "e" * 64


@pytest.mark.parametrize("mode", ["neutral", "optimistic"])
def test_non_conservative_primary_evidence_cannot_promote(mode: str) -> None:
    with pytest.raises(ExecutionPromotionError, match="conservative"):
        validate_execution_promotion(
            _valid_evidence(path_mode=mode),
            expected_policy_digest="e" * 64,
        )


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
) -> None:
    with pytest.raises(ExecutionPromotionError, match=message):
        validate_execution_promotion(
            _valid_evidence(**changes),
            expected_policy_digest="e" * 64,
        )


def test_execution_policy_identity_must_match_experiment_plan() -> None:
    with pytest.raises(ExecutionPromotionError, match="policy digest"):
        validate_execution_promotion(
            _valid_evidence(),
            expected_policy_digest="f" * 64,
        )


def test_execution_evidence_round_trips_canonically(tmp_path: Path) -> None:
    evidence = _valid_evidence()
    path = tmp_path / "execution-evidence.json"
    write_execution_evidence(path, evidence)
    assert load_execution_evidence(path) == evidence
    assert load_execution_evidence(path).schema_version == EXECUTION_EVIDENCE_SCHEMA
    with pytest.raises(FileExistsError, match="already exists"):
        write_execution_evidence(path, replace(evidence, order_event_count=13))


def test_execution_evidence_uses_the_complete_cost_policy_digest() -> None:
    from trade_rl.simulation.execution import ExecutionCostConfig
    from trade_rl.simulation.execution_promotion import execution_evidence_from_cost

    cost = ExecutionCostConfig(path_mode="conservative")
    evidence = execution_evidence_from_cost(dataset_id="d" * 64, cost=cost)

    assert evidence.execution_policy_digest == cost.execution_policy_digest
    assert evidence.execution_policy_digest != replace(
        cost,
        fee_rate=cost.fee_rate + 0.001,
    ).execution_policy_digest
