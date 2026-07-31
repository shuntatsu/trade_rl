from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from trade_rl.simulation.execution import ExecutionCostConfig
from trade_rl.simulation.execution_promotion import (
    ExecutionEvidence,
    ExecutionPromotionError,
    execution_evidence_from_cost,
    validate_execution_event_artifact,
    validate_execution_promotion,
)
from trade_rl.simulation.execution_replay import write_execution_event_artifact
from tests.evaluation.replay_support import (
    execution_artifact,
)

_DATASET_ID = "d" * 64
_COST = ExecutionCostConfig(path_mode="conservative")
_POLICY_DIGEST = _COST.execution_policy_digest


def _evidence(tmp_path: Path) -> tuple[ExecutionEvidence, Path]:
    artifact = execution_artifact()
    path = write_execution_event_artifact(tmp_path / "order-events.json", artifact)
    evidence = execution_evidence_from_cost(
        dataset_id=_DATASET_ID,
        cost=_COST,
        order_event_artifact_path=path,
        sensitivity_path_modes=("conservative",),
    )
    return evidence, path


def test_zero_order_events_cannot_be_promoted_as_complete_evidence() -> None:
    evidence = ExecutionEvidence(
        dataset_id=_DATASET_ID,
        execution_policy_digest=_POLICY_DIGEST,
        path_mode="conservative",
        processing_bar_volume_capacity=True,
        partial_fill_carry=True,
        trigger_volume_fractions=(1.0, 0.5, 0.25, 0.0),
        order_event_count=0,
        complete_order_evidence=True,
        sensitivity_path_modes=("conservative",),
    )

    with pytest.raises(ExecutionPromotionError, match="order event"):
        validate_execution_promotion(
            evidence,
            expected_policy_digest=_POLICY_DIGEST,
        )


def test_event_artifact_rejects_forged_count_and_terminal_digests(
    tmp_path: Path,
) -> None:
    evidence, path = _evidence(tmp_path)
    validate_execution_event_artifact(evidence, path)

    with pytest.raises(ExecutionPromotionError, match="event count"):
        validate_execution_event_artifact(
            replace(evidence, order_event_count=evidence.order_event_count + 1),
            path,
        )
    with pytest.raises(ExecutionPromotionError, match="terminal book"):
        validate_execution_event_artifact(
            replace(evidence, terminal_book_digest="f" * 64),
            path,
        )
    with pytest.raises(ExecutionPromotionError, match="terminal order book"):
        validate_execution_event_artifact(
            replace(evidence, terminal_order_book_digest="1" * 64),
            path,
        )


def test_event_artifact_substitution_is_rejected(tmp_path: Path) -> None:
    evidence, path = _evidence(tmp_path)
    replacement_artifact = execution_artifact(reason="substituted", cash=899.0)
    path.unlink()
    write_execution_event_artifact(path, replacement_artifact)

    with pytest.raises(ExecutionPromotionError, match="artifact (size|digest)"):
        validate_execution_event_artifact(evidence, path)


def test_event_artifact_symlink_is_rejected(tmp_path: Path) -> None:
    evidence, path = _evidence(tmp_path)
    link = tmp_path / "order-events-link.json"
    try:
        link.symlink_to(path.name)
    except OSError:
        pytest.skip("symlinks are unavailable on this platform")

    with pytest.raises(ExecutionPromotionError, match="regular non-symlink"):
        validate_execution_event_artifact(evidence, link)
