from __future__ import annotations

import json

import pytest

from trade_rl.artifacts.canonical import canonical_json_bytes
from trade_rl.evaluation.experiments.bootstrap.spot_flow_continuation_prereg import (
    SpotFlowContinuationProtocol,
    canonical_spot_flow_continuation_protocol,
    load_spot_flow_continuation_protocol_bytes,
)

SYMBOLS = ("BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT")
DATES = ("2021-01-15", "2021-07-15", "2022-01-15", "2022-07-15")


def test_canonical_protocol_freezes_source_feature_target_and_gate() -> None:
    protocol = canonical_spot_flow_continuation_protocol()

    assert protocol.issue_number == 584
    assert protocol.symbols == SYMBOLS
    assert protocol.spot_dates == DATES
    assert protocol.spot_archive_count == 20
    assert protocol.spot_source_issue == 578
    assert protocol.spot_source_status == "PASS_SPOT_AGGTRADES_SOURCE"
    assert protocol.spot_source_protocol_head == "9932775127c79a85d81fb304850c3349460e4a2a"
    assert protocol.spot_source_protocol_digest == (
        "808f999915815259762067aaff381a88589a7c28ceb0383d8aa0626d5c67e8d4"
    )
    assert protocol.spot_source_result_sha256 == (
        "dfa2a5350ffdd416c1bc06fb48539cf882a009910929b9873fd7942ef53e6030"
    )
    assert protocol.spot_source_result_content_digest == (
        "06207c17f997ebd6c99db553cee07881a10e8b54d16af062a47e0fcb0a2cb51a"
    )

    assert protocol.feature_name == "15m__spot_aggressive_notional_imbalance"
    assert protocol.signal_window_minutes == 15
    assert protocol.candidate_intervals_per_day == 96
    assert protocol.candidate_decisions_per_symbol == 384
    assert protocol.buyer_aggressor_token == "False"
    assert protocol.seller_aggressor_token == "True"
    assert protocol.buyer_aggressor_sign == 1
    assert protocol.seller_aggressor_sign == -1
    assert protocol.weighting == "quote_notional_price_times_quantity"
    assert protocol.interval_semantics == "left_closed_right_open"
    assert protocol.empty_interval_is_unavailable is True
    assert protocol.sentinel_rows_are_feature_events is False
    assert protocol.use_best_price_match is False

    assert protocol.target_market == "binance_usdm_perpetual"
    assert protocol.target_timeframe == "15m"
    assert protocol.execution_open_offset_bars == 1
    assert protocol.endpoint_open_offset_bars == 17
    assert protocol.horizon_bars == 16
    assert protocol.horizon_minutes == 240
    assert protocol.expected_direction == "CONTINUATION"
    assert protocol.minimum_eligible_observations_per_symbol == 360
    assert protocol.required_positive_symbol_slopes == 4
    assert protocol.valid_status == "VALID_SPOT_FLOW_CONTINUATION"
    assert protocol.reject_status == "REJECT_SPOT_FLOW_HYPOTHESIS"
    assert protocol.invalid_coverage_status == "INVALID_SPOT_FLOW_COVERAGE"

    assert protocol.no_intercept is True
    assert protocol.fixed_reduction == "math.fsum_chronological"
    assert protocol.alternate_sign_allowed is False
    assert protocol.alternate_window_allowed is False
    assert protocol.alternate_horizon_allowed is False
    assert protocol.symbol_subset_allowed is False
    assert protocol.additional_spot_dates_allowed is False
    assert protocol.evaluation_pnl_inspected is False
    assert protocol.final_test_authorized is False
    assert protocol.shared_cash_profitability_established is False
    assert protocol.production_eligible is False
    assert protocol.live_trading_authorized is False

    assert protocol.canonical_json_bytes == canonical_json_bytes(protocol.to_payload())
    assert len(protocol.digest) == 64


def test_exact_candidate_clock_is_frozen() -> None:
    protocol = canonical_spot_flow_continuation_protocol()
    expected = tuple(range(15, 24 * 60 + 1, 15))
    assert protocol.decision_minutes_after_day_start == expected
    assert len(expected) == 96
    assert expected[0] == 15
    assert expected[-1] == 1440


def test_post_result_mutations_fail_closed() -> None:
    payload = canonical_spot_flow_continuation_protocol().to_payload()
    mutations: list[tuple[str, object]] = [
        ("symbols", list(reversed(SYMBOLS))),
        ("spot_dates", ["2021-01-16", *DATES[1:]]),
        ("feature_name", "15m__spot_flow"),
        ("signal_window_minutes", 60),
        ("buyer_aggressor_sign", -1),
        ("weighting", "base_quantity"),
        ("target_timeframe", "1h"),
        ("execution_open_offset_bars", 0),
        ("endpoint_open_offset_bars", 33),
        ("expected_direction", "REVERSAL"),
        ("minimum_eligible_observations_per_symbol", 300),
        ("required_positive_symbol_slopes", 3),
        ("alternate_sign_allowed", True),
        ("additional_spot_dates_allowed", True),
        ("evaluation_pnl_inspected", True),
        ("production_eligible", True),
    ]
    for field, value in mutations:
        changed = dict(payload)
        changed[field] = value
        with pytest.raises(ValueError):
            SpotFlowContinuationProtocol.from_payload(changed)

    changed = dict(payload)
    changed["decision_minutes_after_day_start"] = list(range(60, 1441, 60))
    with pytest.raises(ValueError):
        SpotFlowContinuationProtocol.from_payload(changed)


def test_loader_requires_exact_canonical_bytes_and_field_set() -> None:
    protocol = canonical_spot_flow_continuation_protocol()
    raw = protocol.canonical_json_bytes
    assert load_spot_flow_continuation_protocol_bytes(raw) == protocol

    decoded = json.loads(raw)
    decoded["unknown"] = 1
    with pytest.raises(ValueError):
        SpotFlowContinuationProtocol.from_payload(decoded)

    decoded = json.loads(raw)
    noncanonical = json.dumps(decoded, sort_keys=True, indent=2).encode() + b"\n"
    with pytest.raises(ValueError, match="canonical"):
        load_spot_flow_continuation_protocol_bytes(noncanonical)

    with pytest.raises(ValueError):
        load_spot_flow_continuation_protocol_bytes(raw + b" ")
