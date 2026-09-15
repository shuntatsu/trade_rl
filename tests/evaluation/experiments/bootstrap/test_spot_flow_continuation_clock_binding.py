from __future__ import annotations

import math
from datetime import UTC, datetime

import pytest

from trade_rl.evaluation.experiments.bootstrap import (
    spot_flow_continuation_diagnostic as diagnostic,
)
from trade_rl.evaluation.experiments.bootstrap.spot_flow_continuation_prereg import (
    canonical_spot_flow_continuation_protocol,
)

QUARTER_HOUR_MS = 900_000


def test_label_builder_matches_sealed_completed_bar_clock() -> None:
    protocol = canonical_spot_flow_continuation_protocol()
    decision = int(
        datetime(2021, 1, 15, 0, 15, tzinfo=UTC).timestamp() * 1000
    )
    execution = decision + protocol.execution_raw_open_time_offset_minutes * 60_000
    endpoint = decision + protocol.endpoint_raw_open_time_offset_minutes * 60_000
    assert execution == decision
    assert endpoint == decision + 16 * QUARTER_HOUR_MS
    bars = tuple(
        diagnostic.TargetBar(
            open_time_ms=execution + index * QUARTER_HOUR_MS,
            open_price=100.0 + index,
        )
        for index in range(17)
    )
    label = diagnostic.build_four_hour_label(bars, decision_time_ms=decision)
    assert label == pytest.approx(math.log(116.0 / 100.0))


def test_decision_row_is_previous_raw_kline_under_sealed_clock() -> None:
    protocol = canonical_spot_flow_continuation_protocol()
    assert protocol.target_dataset_timestamp_semantics == "completed_bar_close_boundary"
    assert protocol.target_dataset_timestamp_formula == "dataset_timestamp=raw_open_time+15m"
    assert protocol.decision_target_row_semantics == "decision_t_equals_target_completed_row_timestamp"
    assert protocol.decision_row_raw_open_time_offset_minutes == -15
    assert protocol.execution_raw_open_time_offset_minutes == 0
    assert protocol.endpoint_raw_open_time_offset_minutes == 240
