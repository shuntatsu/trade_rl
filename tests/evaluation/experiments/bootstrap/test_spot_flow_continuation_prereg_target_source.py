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
    ]
    for field, value in mutations:
        changed = dict(payload)
        changed[field] = value
        with pytest.raises(ValueError):
            SpotFlowContinuationProtocol.from_payload(changed)
