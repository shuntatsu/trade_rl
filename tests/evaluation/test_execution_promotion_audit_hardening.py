from __future__ import annotations

import pytest

from trade_rl.simulation.execution_promotion import (
    ExecutionEvidence,
    ExecutionPromotionError,
    validate_execution_promotion,
)


def test_zero_order_events_cannot_be_promoted_as_complete_evidence() -> None:
    evidence = ExecutionEvidence(
        dataset_id="d" * 64,
        execution_policy_digest="e" * 64,
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
            expected_policy_digest="e" * 64,
        )
