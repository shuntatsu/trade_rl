from __future__ import annotations

import pytest

from trade_rl.evaluation.experiments.bootstrap.spot_flow_continuation_prereg import (
    SpotFlowContinuationProtocol,
    canonical_spot_flow_continuation_protocol,
)


def test_target_history_source_is_frozen_before_any_target_relation() -> None:
    protocol = canonical_spot_flow_continuation_protocol()

    assert protocol.target_source_family == "binance_vision_contract_klines"
    assert protocol.target_transport_mode == "VISION"
    assert protocol.target_source_fallback_allowed is False
    assert protocol.target_source_replacement_allowed is False
    assert protocol.target_source_checksum_required is True
    assert protocol.target_source_manifest_binding_required is True
    assert protocol.target_source_structural_preflight_required is True


def test_target_clock_mapping_is_content_addressed() -> None:
    protocol = canonical_spot_flow_continuation_protocol()

    assert protocol.target_dataset_timestamp_semantics == "completed_bar_close_boundary"
    assert protocol.target_dataset_timestamp_formula == "dataset_timestamp=raw_open_time+15m"
    assert protocol.decision_target_row_semantics == (
        "decision_t_equals_target_completed_row_timestamp"
    )
    assert protocol.decision_row_raw_open_time_offset_minutes == -15
    assert protocol.execution_raw_open_time_offset_minutes == 0
    assert protocol.endpoint_raw_open_time_offset_minutes == 240


def test_target_source_rescue_mutations_fail_closed() -> None:
    payload = canonical_spot_flow_continuation_protocol().to_payload()
    mutations: list[tuple[str, object]] = [
        ("target_source_family", "rest_klines"),
        ("target_transport_mode", "AUTO"),
        ("target_source_fallback_allowed", True),
        ("target_source_replacement_allowed", True),
        ("target_source_checksum_required", False),
        ("target_source_manifest_binding_required", False),
        ("target_source_structural_preflight_required", False),
        ("target_dataset_timestamp_semantics", "raw_open_boundary"),
        ("target_dataset_timestamp_formula", "dataset_timestamp=raw_open_time"),
        ("decision_target_row_semantics", "decision_t_equals_raw_open_time"),
        ("decision_row_raw_open_time_offset_minutes", 0),
        ("execution_raw_open_time_offset_minutes", 15),
        ("endpoint_raw_open_time_offset_minutes", 255),
    ]
    for field, value in mutations:
        changed = dict(payload)
        changed[field] = value
        with pytest.raises(ValueError):
            SpotFlowContinuationProtocol.from_payload(changed)
