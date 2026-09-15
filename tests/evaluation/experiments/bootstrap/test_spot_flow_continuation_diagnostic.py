from __future__ import annotations

import json
import math
from datetime import UTC, datetime

import pytest

from trade_rl.artifacts.canonical import canonical_json_bytes
from trade_rl.evaluation.experiments.bootstrap import (
    spot_flow_continuation_diagnostic as diagnostic,
)
from trade_rl.evaluation.experiments.bootstrap.spot_flow_continuation_prereg import (
    canonical_spot_flow_continuation_protocol,
)

SYMBOLS = ("BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT")
DAY_MS = 86_400_000
QUARTER_HOUR_MS = 900_000
FOUR_HOURS_MS = 14_400_000


def _day_start_ms(date: str) -> int:
    return int(
        datetime.strptime(date, "%Y-%m-%d").replace(tzinfo=UTC).timestamp() * 1000
    )


def _trade(
    event_time_ms: int,
    *,
    price: float,
    quantity: float,
    buyer_is_maker: bool,
) -> diagnostic.SpotAggTrade:
    return diagnostic.SpotAggTrade(
        event_time_ms=event_time_ms,
        price=price,
        quantity=quantity,
        is_buyer_maker=buyer_is_maker,
    )


def _target_bar(
    open_time_ms: int,
    open_price: float,
    *,
    information_available: bool = True,
    active: bool = True,
    tradable: bool = True,
) -> diagnostic.TargetBar:
    return diagnostic.TargetBar(
        open_time_ms=open_time_ms,
        open_price=open_price,
        information_available=information_available,
        active=active,
        tradable=tradable,
    )


def _positive_observations(beta: float = 0.01) -> tuple[list[float], list[float]]:
    signals = [1.0] * 360
    labels = [beta] * 360
    return signals, labels


def _negative_observations(beta: float = -0.01) -> tuple[list[float], list[float]]:
    signals = [1.0] * 360
    labels = [beta] * 360
    return signals, labels


def _result_inputs(
    *, positive_symbols: int = 4
) -> dict[str, tuple[list[float], list[float]]]:
    result: dict[str, tuple[list[float], list[float]]] = {}
    for index, symbol in enumerate(SYMBOLS):
        result[symbol] = (
            _positive_observations()
            if index < positive_symbols
            else _negative_observations()
        )
    return result


def _build_result(*, positive_symbols: int = 4) -> diagnostic.SpotFlowDiagnosticResult:
    protocol = canonical_spot_flow_continuation_protocol()
    return diagnostic.build_spot_flow_diagnostic_result(
        _result_inputs(positive_symbols=positive_symbols),
        protocol,
        implementation_head="1" * 40,
        source_manifest_digest="2" * 64,
        target_manifest_digest="3" * 64,
        target_preflight_run_id=123,
        target_preflight_artifact_id=456,
        target_preflight_artifact_api_digest="4" * 64,
        execution_run_id=789,
    )


def test_spot_csv_parser_preserves_provider_semantics_and_excludes_exact_sentinel() -> (
    None
):
    date = "2021-01-15"
    start = _day_start_ms(date)
    payload = (
        f"1,100,2,10,10,{start + 1},False,True\n"
        f"2,0,0,-1,-1,{start + 2},True,False\n"
        f"3,200,1,11,11,{start + 3},True,True\n"
    ).encode()

    parsed = diagnostic.parse_spot_aggtrades_csv(payload, expected_date=date)

    assert [item.event_time_ms for item in parsed] == [start + 1, start + 3]
    assert parsed[0].is_buyer_maker is False
    assert parsed[1].is_buyer_maker is True
    assert parsed[0].price == 100.0
    assert parsed[0].quantity == 2.0

    near_sentinel = f"2,0,1,-1,-1,{start + 2},True,False\n".encode()
    with pytest.raises(ValueError, match="sentinel|price|quantity|malformed"):
        diagnostic.parse_spot_aggtrades_csv(near_sentinel, expected_date=date)


def test_spot_csv_parser_rejects_wrong_boolean_and_out_of_day_timestamp() -> None:
    date = "2021-01-15"
    start = _day_start_ms(date)
    wrong_boolean = f"1,100,2,10,10,{start + 1},false,True\n".encode()
    with pytest.raises(ValueError, match="boolean|maker"):
        diagnostic.parse_spot_aggtrades_csv(wrong_boolean, expected_date=date)

    next_day = f"1,100,2,10,10,{start + DAY_MS},False,True\n".encode()
    with pytest.raises(ValueError, match="date|timestamp"):
        diagnostic.parse_spot_aggtrades_csv(next_day, expected_date=date)


def test_interval_signal_uses_left_closed_right_open_quote_notional_and_provider_order() -> (
    None
):
    decision = 10 * QUARTER_HOUR_MS
    start = decision - QUARTER_HOUR_MS
    trades = (
        _trade(start - 1, price=999.0, quantity=1.0, buyer_is_maker=False),
        _trade(start, price=900.0, quantity=1.0, buyer_is_maker=False),
        _trade(start + 1, price=100.0, quantity=1.0, buyer_is_maker=True),
        _trade(decision - 1, price=100.0, quantity=1.0, buyer_is_maker=True),
        _trade(decision, price=999.0, quantity=1.0, buyer_is_maker=True),
    )

    signal = diagnostic.aggregate_spot_interval(trades, decision_time_ms=decision)

    assert signal == pytest.approx(700.0 / 1100.0)
    assert signal > 0.0


def test_interval_signal_is_unavailable_for_empty_window_and_prefix_causal() -> None:
    decision = 4 * QUARTER_HOUR_MS
    past = (_trade(decision - 1, price=100.0, quantity=1.0, buyer_is_maker=False),)
    future = (
        *past,
        _trade(decision, price=10_000.0, quantity=1.0, buyer_is_maker=True),
        _trade(decision + 1, price=20_000.0, quantity=1.0, buyer_is_maker=True),
    )
    assert diagnostic.aggregate_spot_interval((), decision_time_ms=decision) is None
    assert diagnostic.aggregate_spot_interval(past, decision_time_ms=decision) == (
        diagnostic.aggregate_spot_interval(future, decision_time_ms=decision)
    )


def test_interval_signal_observably_uses_math_fsum(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_fsum = math.fsum
    calls: list[tuple[float, ...]] = []

    def tracked(values: object) -> float:
        materialized = tuple(float(value) for value in values)  # type: ignore[union-attr]
        calls.append(materialized)
        return real_fsum(materialized)

    monkeypatch.setattr(diagnostic.math, "fsum", tracked)
    decision = QUARTER_HOUR_MS
    trades = (
        _trade(1, price=1e16, quantity=1.0, buyer_is_maker=False),
        _trade(2, price=1.0, quantity=1.0, buyer_is_maker=True),
        _trade(3, price=1.0, quantity=1.0, buyer_is_maker=False),
    )
    diagnostic.aggregate_spot_interval(trades, decision_time_ms=decision)
    assert len(calls) == 2


def test_target_csv_parser_requires_native_15m_grid_and_finite_positive_open() -> None:
    date = "2021-01-15"
    start = _day_start_ms(date)
    rows = []
    for index in range(3):
        open_time = start + index * QUARTER_HOUR_MS
        close_time = open_time + QUARTER_HOUR_MS - 1
        rows.append(
            f"{open_time},{100 + index},101,99,100,1,{close_time},100,1,0.5,50,0"
        )
    parsed = diagnostic.parse_usdm_15m_klines_csv(
        ("\n".join(rows) + "\n").encode(), expected_date=date
    )
    assert [item.open_time_ms for item in parsed] == [
        start,
        start + QUARTER_HOUR_MS,
        start + 2 * QUARTER_HOUR_MS,
    ]

    off_grid = rows[0].replace(str(start), str(start + 1), 1).encode() + b"\n"
    with pytest.raises(ValueError, match="grid|timestamp"):
        diagnostic.parse_usdm_15m_klines_csv(off_grid, expected_date=date)

    nonpositive = rows[0].replace(",100,101", ",0,101", 1).encode() + b"\n"
    with pytest.raises(ValueError, match="open|positive"):
        diagnostic.parse_usdm_15m_klines_csv(nonpositive, expected_date=date)


def test_four_hour_label_uses_execution_at_decision_and_endpoint_plus_16_bars() -> None:
    decision = _day_start_ms("2021-01-15") + QUARTER_HOUR_MS
    bars = tuple(
        _target_bar(decision + index * QUARTER_HOUR_MS, 100.0 + index)
        for index in range(17)
    )

    label = diagnostic.build_four_hour_label(bars, decision_time_ms=decision)

    assert label == pytest.approx(math.log(116.0 / 100.0))


def test_four_hour_label_fails_closed_on_gap_or_invalid_target_mask() -> None:
    decision = _day_start_ms("2021-01-15")
    complete = [
        _target_bar(decision + index * QUARTER_HOUR_MS, 100.0 + index)
        for index in range(17)
    ]
    assert (
        diagnostic.build_four_hour_label(complete, decision_time_ms=decision)
        is not None
    )

    missing = complete[:8] + complete[9:]
    assert diagnostic.build_four_hour_label(missing, decision_time_ms=decision) is None

    invalid = list(complete)
    invalid[8] = _target_bar(
        invalid[8].open_time_ms,
        invalid[8].open_price,
        information_available=False,
    )
    assert diagnostic.build_four_hour_label(invalid, decision_time_ms=decision) is None


def test_no_intercept_calibration_differs_from_centered_ols() -> None:
    result = diagnostic.calibrate_spot_flow_symbol(
        "BTCUSDT", [1.0, 2.0], [1.0, 1.0], minimum_observations=2
    )
    assert result.beta == pytest.approx(3.0 / 5.0)
    assert result.positive_slope is True


def test_frozen_four_of_five_positive_gate_and_coverage_gate() -> None:
    valid = _build_result(positive_symbols=4)
    assert valid.status == "VALID_SPOT_FLOW_CONTINUATION"
    assert valid.positive_slope_count == 4
    assert valid.failures == ()

    rejected = _build_result(positive_symbols=3)
    assert rejected.status == "REJECT_SPOT_FLOW_HYPOTHESIS"
    assert rejected.positive_slope_count == 3
    assert rejected.failures == ()

    protocol = canonical_spot_flow_continuation_protocol()
    observations = _result_inputs(positive_symbols=5)
    observations["BTCUSDT"] = ([1.0] * 359, [0.01] * 359)
    invalid = diagnostic.build_spot_flow_diagnostic_result(
        observations,
        protocol,
        implementation_head="1" * 40,
        source_manifest_digest="2" * 64,
        target_manifest_digest="3" * 64,
        target_preflight_run_id=123,
        target_preflight_artifact_id=456,
        target_preflight_artifact_api_digest="4" * 64,
        execution_run_id=789,
    )
    assert invalid.status == "INVALID_SPOT_FLOW_COVERAGE"
    assert invalid.symbol_results[0].beta is None
    assert invalid.failures == ("BTCUSDT:eligible_observations<360",)


def test_zero_denominator_is_invalid_not_a_sign_rescue() -> None:
    result = diagnostic.calibrate_spot_flow_symbol(
        "BTCUSDT", [0.0] * 360, [0.01] * 360, minimum_observations=360
    )
    assert result.beta is None
    assert result.positive_slope is False
    assert result.failures == ("BTCUSDT:denominator_not_finite_positive",)


def test_result_artifact_is_strict_canonical_and_authority_bound() -> None:
    result = _build_result()
    raw = diagnostic.canonical_spot_flow_result_bytes(result)
    loaded = diagnostic.load_spot_flow_result_bytes(raw)
    assert loaded == result
    assert result.protocol_digest == canonical_spot_flow_continuation_protocol().digest
    assert result.prereg_head == "89ec1092e438d69657840567e684739ca0c4e3d5"
    assert result.prereg_full_verify_run_id == 34932104475
    assert result.prereg_seal_run_id == 34932424904
    assert result.prereg_seal_artifact_id == 10381763665
    assert result.prereg_fresh_artifact_id == 10381379452
    assert result.training_relation_executed is True
    assert result.evaluation_pnl_inspected is False
    assert result.final_test_authorized is False
    assert result.shared_cash_profitability_established is False
    assert result.production_eligible is False
    assert result.live_trading_authorized is False

    decoded = json.loads(raw)
    decoded["unknown"] = 1
    with pytest.raises(ValueError, match="field|canonical"):
        diagnostic.load_spot_flow_result_bytes(canonical_json_bytes(decoded))

    decoded = json.loads(raw)
    decoded["implementation_head"] = "9" * 40
    decoded["content_digest"] = "0" * 64
    with pytest.raises(ValueError, match="digest|authority|canonical"):
        diagnostic.load_spot_flow_result_bytes(canonical_json_bytes(decoded))

    decoded = json.loads(raw)
    decoded["execution_run_id"] = True
    decoded["content_digest"] = "0" * 64
    with pytest.raises(ValueError, match="execution_run_id|integer|digest"):
        diagnostic.load_spot_flow_result_bytes(canonical_json_bytes(decoded))


def test_noncanonical_result_bytes_are_rejected() -> None:
    raw = diagnostic.canonical_spot_flow_result_bytes(_build_result())
    decoded = json.loads(raw)
    noncanonical = json.dumps(decoded, sort_keys=True, indent=2).encode() + b"\n"
    with pytest.raises(ValueError, match="canonical"):
        diagnostic.load_spot_flow_result_bytes(noncanonical)
