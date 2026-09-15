from __future__ import annotations

import json
import math
from datetime import UTC, datetime, timedelta

import pytest

from trade_rl.artifacts.hashing import content_digest
from trade_rl.evaluation.experiments.bootstrap.premium_pressure_calibration import (
    PremiumObservation,
    PremiumPressureSymbolCalibration,
    TargetOpenObservation,
    TrainingPair,
    build_premium_pressure_result,
    build_training_pairs,
    calibrate_symbol,
    canonical_premium_pressure_result_bytes,
    load_premium_pressure_result_bytes,
    parse_premium_index_1h_csv,
    parse_usdm_1h_target_csv,
)
from trade_rl.evaluation.experiments.bootstrap.premium_pressure_prereg import (
    canonical_premium_pressure_protocol,
)

_HOUR_MS = 60 * 60 * 1000
_HEADER = (
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
    "count",
    "taker_buy_volume",
    "taker_buy_quote_volume",
    "ignore",
)


def _ms(value: datetime) -> int:
    return int(value.timestamp() * 1000)


def _csv(rows: list[list[object]], *, header: bool = False) -> bytes:
    lines: list[str] = []
    if header:
        lines.append(",".join(_HEADER))
    lines.extend(",".join(str(value) for value in row) for row in rows)
    return ("\n".join(lines) + "\n").encode()


def _premium_row(
    raw_open_ms: int,
    *,
    open_value: str = "9",
    high: str = "8",
    low: str = "7",
    close: str = "0.001",
) -> list[object]:
    return [
        raw_open_ms,
        open_value,
        high,
        low,
        close,
        "0",
        raw_open_ms + _HOUR_MS - 1,
        "0",
        "0",
        "0",
        "0",
        "0",
    ]


def _target_row(raw_open_ms: int, *, open_price: str = "100") -> list[object]:
    return [
        raw_open_ms,
        open_price,
        open_price,
        open_price,
        open_price,
        "0",
        raw_open_ms + _HOUR_MS - 1,
        "0",
        "0",
        "0",
        "0",
        "0",
    ]


def _decision() -> datetime:
    return datetime(2021, 1, 1, 1, tzinfo=UTC)


def _premium_point(
    decision: datetime,
    value_bps: float = 10.0,
) -> PremiumObservation:
    dataset_ms = _ms(decision)
    return PremiumObservation(
        raw_open_time_ms=dataset_ms - _HOUR_MS,
        dataset_time_ms=dataset_ms,
        close_bps=value_bps,
    )


def _target_window(
    decision: datetime,
    *,
    execution_open: float = 100.0,
    endpoint_open: float = 90.0,
) -> tuple[TargetOpenObservation, ...]:
    result: list[TargetOpenObservation] = []
    decision_ms = _ms(decision)
    for offset in range(1, 26):
        dataset_ms = decision_ms + offset * _HOUR_MS
        price = endpoint_open if offset == 25 else execution_open
        result.append(
            TargetOpenObservation(
                raw_open_time_ms=dataset_ms - _HOUR_MS,
                dataset_time_ms=dataset_ms,
                open_price=price,
            )
        )
    return tuple(result)


def _calibration(symbol: str, beta: float) -> PremiumPressureSymbolCalibration:
    x_bar = 1.0
    y_bar = 2.0
    denominator = 2.0
    numerator = beta * denominator
    alpha = y_bar - beta * x_bar
    return PremiumPressureSymbolCalibration(
        symbol=symbol,
        eligible_observations=16_621,
        x_bar=x_bar,
        y_bar=y_bar,
        numerator=numerator,
        denominator=denominator,
        alpha=alpha,
        beta=beta,
        negative_slope=beta < 0.0,
        failures=(),
    )


def _valid_result(negative_count: int = 4):
    protocol = canonical_premium_pressure_protocol()
    calibrations = tuple(
        _calibration(symbol, -1.0 if index < negative_count else 1.0)
        for index, symbol in enumerate(protocol.symbols)
    )
    return build_premium_pressure_result(
        calibrations,
        implementation_head="9" * 40,
        implementation_verification_run_id=123,
        execution_run_id=456,
    )


def test_premium_parser_uses_close_only_fixed_bps_and_completed_bar_clock() -> None:
    decision = _decision()
    raw_open = _ms(decision - timedelta(hours=1))
    rows = [
        _premium_row(
            raw_open,
            open_value="123",
            high="456",
            low="-789",
            close="-0.00125",
        ),
        _premium_row(raw_open + _HOUR_MS, close="0"),
        _premium_row(raw_open + 2 * _HOUR_MS, close="0.0025"),
    ]
    points = parse_premium_index_1h_csv(
        _csv(rows, header=True), expected_month="2021-01"
    )
    assert tuple(point.close_bps for point in points) == (-12.5, 0.0, 25.0)
    assert points[0].raw_open_time_ms == raw_open
    assert points[0].dataset_time_ms == _ms(decision)
    assert points[0].dataset_time_ms - points[0].raw_open_time_ms == _HOUR_MS


def test_premium_parser_rejects_nonfinite_close_wrong_clock_and_bad_schema() -> None:
    raw_open = _ms(datetime(2021, 1, 1, tzinfo=UTC))
    bad_close = _premium_row(raw_open, close="nan")
    with pytest.raises(ValueError):
        parse_premium_index_1h_csv(_csv([bad_close]), expected_month="2021-01")

    off_grid = _premium_row(raw_open + 1)
    with pytest.raises(ValueError):
        parse_premium_index_1h_csv(_csv([off_grid]), expected_month="2021-01")

    wrong_close_time = _premium_row(raw_open)
    wrong_close_time[6] = int(wrong_close_time[6]) + 1
    with pytest.raises(ValueError):
        parse_premium_index_1h_csv(_csv([wrong_close_time]), expected_month="2021-01")

    wrong_fields = _premium_row(raw_open)[:-1]
    with pytest.raises(ValueError):
        parse_premium_index_1h_csv(_csv([wrong_fields]), expected_month="2021-01")


def test_target_parser_maps_raw_open_to_completed_dataset_and_rejects_bad_open() -> (
    None
):
    raw_open = _ms(_decision())
    points = parse_usdm_1h_target_csv(
        _csv([_target_row(raw_open, open_price="101.25")], header=True),
        expected_month="2021-01",
    )
    assert len(points) == 1
    assert points[0].raw_open_time_ms == raw_open
    assert points[0].dataset_time_ms == raw_open + _HOUR_MS
    assert points[0].open_price == 101.25

    for bad in ("0", "-1", "nan"):
        with pytest.raises(ValueError):
            parse_usdm_1h_target_csv(
                _csv([_target_row(raw_open, open_price=bad)]),
                expected_month="2021-01",
            )


def test_training_pair_uses_exact_t_tplus1_tplus25_clock() -> None:
    decision = _decision()
    premium = (_premium_point(decision, 12.0),)
    target = _target_window(decision, execution_open=100.0, endpoint_open=80.0)
    pairs = build_training_pairs(premium, target)
    assert len(pairs) == 1
    pair = pairs[0]
    assert pair.decision_time_ms == _ms(decision)
    assert pair.x == 12.0
    assert pair.y == math.log(80.0 / 100.0)

    shifted = list(target)
    shifted[0] = TargetOpenObservation(
        raw_open_time_ms=shifted[0].raw_open_time_ms,
        dataset_time_ms=shifted[0].dataset_time_ms,
        open_price=50.0,
    )
    shifted[-1] = TargetOpenObservation(
        raw_open_time_ms=shifted[-1].raw_open_time_ms,
        dataset_time_ms=shifted[-1].dataset_time_ms,
        open_price=125.0,
    )
    shifted_pair = build_training_pairs(premium, tuple(shifted))[0]
    assert shifted_pair.y == math.log(125.0 / 50.0)


def test_missing_premium_and_future_mutation_never_fill_or_rewrite_prior_pair() -> None:
    decision = _decision()
    target = _target_window(decision)
    assert build_training_pairs((), target) == ()

    base = build_training_pairs((_premium_point(decision),), target)
    future_decision = decision + timedelta(days=10)
    mutated = build_training_pairs(
        (_premium_point(decision), _premium_point(future_decision, -999.0)),
        target,
    )
    assert base == mutated


def test_any_missing_or_masked_target_row_invalidates_the_pair() -> None:
    decision = _decision()
    premium = (_premium_point(decision),)
    target = list(_target_window(decision))

    for index in (0, 7, 24):
        missing = tuple(item for pos, item in enumerate(target) if pos != index)
        assert build_training_pairs(premium, missing) == ()

    for field in ("information_available", "active", "tradable"):
        changed = list(target)
        original = changed[7]
        values = {
            "information_available": original.information_available,
            "active": original.active,
            "tradable": original.tradable,
        }
        values[field] = False
        changed[7] = TargetOpenObservation(
            raw_open_time_ms=original.raw_open_time_ms,
            dataset_time_ms=original.dataset_time_ms,
            open_price=original.open_price,
            **values,
        )
        assert build_training_pairs(premium, tuple(changed)) == ()


def test_endpoint_at_fit_cutoff_is_excluded_but_last_frozen_decision_is_eligible() -> (
    None
):
    protocol = canonical_premium_pressure_protocol()
    last = protocol.last_candidate_decision
    pairs = build_training_pairs((_premium_point(last),), _target_window(last))
    assert len(pairs) == 1

    too_late = last + timedelta(hours=1)
    target = _target_window(too_late)
    assert target[-1].dataset_time_ms == _ms(protocol.fit_cutoff)
    assert build_training_pairs((_premium_point(too_late),), target) == ()


def test_intercept_ols_is_canonical_and_differs_from_no_intercept_regression() -> None:
    protocol = canonical_premium_pressure_protocol()
    pattern = ((1.0, 8.0), (2.0, 6.0), (3.0, 4.0))
    pairs = tuple(
        TrainingPair(
            decision_time_ms=index * _HOUR_MS,
            x=pattern[index % len(pattern)][0],
            y=pattern[index % len(pattern)][1],
        )
        for index in range(protocol.minimum_eligible_observations_per_symbol)
    )
    result = calibrate_symbol("BTCUSDT", pairs)
    assert result.failures == ()
    assert result.beta == -2.0
    assert result.alpha == 10.0
    assert result.negative_slope is True

    no_intercept = math.fsum(pair.x * pair.y for pair in pairs) / math.fsum(
        pair.x * pair.x for pair in pairs
    )
    assert no_intercept > 0.0
    assert no_intercept != result.beta


def test_calibration_uses_fsum_and_invalidates_low_coverage_or_zero_denominator() -> (
    None
):
    protocol = canonical_premium_pressure_protocol()
    n = protocol.minimum_eligible_observations_per_symbol
    x_values = [1e16, 1.0, -1e16] + [0.0] * (n - 3)
    pairs = tuple(
        TrainingPair(
            decision_time_ms=index * _HOUR_MS,
            x=x,
            y=-x * 1e-16 + 3.0,
        )
        for index, x in enumerate(x_values)
    )
    result = calibrate_symbol("BTCUSDT", pairs)
    assert result.failures == ()
    assert result.x_bar == math.fsum(x_values) / n
    assert result.x_bar != sum(x_values) / n
    assert result.beta is not None and result.beta < 0.0

    low = calibrate_symbol("BTCUSDT", pairs[:-1])
    assert low.eligible_observations == n - 1
    assert low.beta is None
    assert low.alpha is None
    assert low.failures == (f"BTCUSDT:eligible_observations<{n}",)

    constant_pairs = tuple(
        TrainingPair(decision_time_ms=index * _HOUR_MS, x=1.0, y=float(index))
        for index in range(n)
    )
    constant = calibrate_symbol("BTCUSDT", constant_pairs)
    assert constant.beta is None
    assert constant.alpha is None
    assert constant.failures == ("BTCUSDT:denominator_not_finite_positive",)


def test_four_of_five_negative_is_valid_three_is_reject_and_failure_is_invalid() -> (
    None
):
    valid = _valid_result(negative_count=4)
    assert valid.status == "VALID_PREMIUM_PRESSURE_REVERSAL"
    assert valid.negative_slope_count == 4

    reject = _valid_result(negative_count=3)
    assert reject.status == "REJECT_PREMIUM_PRESSURE_HYPOTHESIS"
    assert reject.negative_slope_count == 3

    protocol = canonical_premium_pressure_protocol()
    bad = list(valid.symbol_results)
    bad[0] = PremiumPressureSymbolCalibration(
        symbol=protocol.symbols[0],
        eligible_observations=16_620,
        x_bar=None,
        y_bar=None,
        numerator=None,
        denominator=None,
        alpha=None,
        beta=None,
        negative_slope=False,
        failures=(f"{protocol.symbols[0]}:eligible_observations<16621",),
    )
    invalid = build_premium_pressure_result(
        tuple(bad),
        implementation_head="9" * 40,
        implementation_verification_run_id=123,
        execution_run_id=456,
    )
    assert invalid.status == "INVALID_PREMIUM_PRESSURE_COVERAGE"


def test_result_binds_all_authorities_is_deterministic_and_post_training_blind() -> (
    None
):
    result = _valid_result()
    raw = canonical_premium_pressure_result_bytes(result)
    assert raw == canonical_premium_pressure_result_bytes(result)
    assert load_premium_pressure_result_bytes(raw) == result
    payload = result.to_dict()
    assert json.loads(raw) == payload

    assert payload["protocol_head"] == "6ffc414baa10e91f34df258fe5cfabacce65fd77"
    assert payload["protocol_digest"] == (
        "18bf9625502eccbe475d157df90635f1c4645d3c4cced48ec0fc565723206731"
    )
    assert payload["premium_source_run_id"] == 34863522941
    assert payload["premium_source_report_sha256"] == (
        "059987f3692c0f249f2db5ef2e14f8cbaaefa5ca24dccf4c45bae009ca19ebe4"
    )
    assert payload["schema_version"] == "premium_pressure_calibration_result_v2"
    assert payload["protocol_full_verify_run_id"] == 34959430185
    assert payload["protocol_seal_run_id"] == 34959849347
    assert payload["protocol_seal_artifact_id"] == 10392936042
    assert payload["protocol_seal_artifact_api_digest"] == (
        "4b611462f8d4f9bb4d23ab15bb4d8a20525d4093e590c1afb62d1b74ca875e15"
    )
    assert payload["protocol_fresh_artifact_id"] == 10391714897
    assert payload["protocol_fresh_artifact_api_digest"] == (
        "858e930a03e442f047556801c3d9a536fbc89435b5f261cba621988dd8062acd"
    )
    assert payload["premium_source_issue"] == 570
    assert payload["premium_source_probe_head"] == (
        "80b65a773c622af530f074888eb01fff2f7f98df"
    )
    assert payload["target_source_issue"] == 601
    assert payload["target_source_validator_head"] == (
        "c707be853480e57362f6a3d64f116e4056c3553f"
    )
    assert payload["target_source_validator_verification_run_id"] == 34961165095
    assert payload["target_source_preflight_head"] == (
        "e64256dd40e76d9606edfbdebe1c37b86668d467"
    )
    assert payload["target_source_run_id"] == 34962227821
    assert payload["target_source_status"] == "PASS_USDM_1H_TARGET_SOURCE"
    assert payload["target_source_publisher_artifact_id"] == 10393693240
    assert payload["target_source_publisher_artifact_api_digest"] == (
        "0c7f240390a35174329890fa83e9615e304de5f1a84114007ead1fddaaaf37ff"
    )
    assert payload["target_source_fresh_artifact_id"] == 10393379932
    assert payload["target_source_fresh_artifact_api_digest"] == (
        "0c0c05b8f5256c918b90ae709ff05ce0f05f8e88a929f1820387cf60a12dfeff"
    )
    assert payload["target_source_report_sha256"] == (
        "6833311f14e45ce634dd4ba13fdf8defeaa0b1a2b90cf7ff0598ee2054bde151"
    )
    assert payload["target_source_report_content_digest"] == (
        "a9bfe64d5232747ac2ee543e3b65a4be22d4d034969603040391b7b0eabec988"
    )
    assert payload["target_source_manifest_sha256"] == (
        "e287f03bf18619834108b5b452c26cdffbc66a6be3d51d418c928daa2e77dd15"
    )
    assert payload["target_source_manifest_content_digest"] == (
        "9ced3fbab6e51fbd654768ec6a4ea28f98cc4b9f36c1d9d2a9a8f6540864934e"
    )
    assert payload["training_relation_executed"] is True
    assert payload["evaluation_pnl_inspected"] is False
    assert payload["evaluation_execution_authorized"] is False
    assert payload["final_test_authorized"] is False
    assert payload["shared_cash_profitability_established"] is False
    assert payload["production_eligible"] is False
    assert payload["live_trading_authorized"] is False
    assert len(payload["content_digest"]) == 64


def test_result_loader_rejects_summary_nested_authority_and_boundary_tampering() -> (
    None
):
    result = _valid_result()
    payload = result.to_dict()

    for mutate in (
        lambda value: value.__setitem__("negative_slope_count", 5),
        lambda value: value.__setitem__("status", "REJECT_PREMIUM_PRESSURE_HYPOTHESIS"),
        lambda value: value.__setitem__("protocol_head", "0" * 40),
        lambda value: value.__setitem__("evaluation_pnl_inspected", True),
    ):
        changed = json.loads(json.dumps(payload))
        mutate(changed)
        changed.pop("content_digest")
        changed["content_digest"] = content_digest(changed)
        with pytest.raises(ValueError):
            load_premium_pressure_result_bytes(
                json.dumps(changed, sort_keys=True, separators=(",", ":")).encode()
            )

    nested = json.loads(json.dumps(payload))
    nested["symbol_results"][0]["eligible_observations"] = True
    nested.pop("content_digest")
    nested["content_digest"] = content_digest(nested)
    with pytest.raises(ValueError):
        load_premium_pressure_result_bytes(
            json.dumps(nested, sort_keys=True, separators=(",", ":")).encode()
        )

    unknown = json.loads(json.dumps(payload))
    unknown["pnl"] = 1.0
    unknown.pop("content_digest")
    unknown["content_digest"] = content_digest(unknown)
    with pytest.raises(ValueError):
        load_premium_pressure_result_bytes(
            json.dumps(unknown, sort_keys=True, separators=(",", ":")).encode()
        )

    raw = canonical_premium_pressure_result_bytes(result)
    with pytest.raises(ValueError):
        load_premium_pressure_result_bytes(b"\n" + raw)


def test_missing_target_timestamp_invalidates_only_affected_disjoint_label() -> None:
    first = _decision()
    second = first + timedelta(hours=48)
    premium = (_premium_point(first), _premium_point(second))
    target = tuple(
        sorted(
            (*_target_window(first), *_target_window(second)),
            key=lambda item: item.dataset_time_ms,
        )
    )
    assert len(build_training_pairs(premium, target)) == 2

    removed_raw_open = _ms(first + timedelta(hours=7))
    missing = tuple(
        item for item in target if item.raw_open_time_ms != removed_raw_open
    )
    surviving = build_training_pairs(premium, missing)
    assert len(surviving) == 1
    assert surviving[0].decision_time_ms == _ms(second)


def test_coverage_boundary_16620_invalid_16621_coverage_valid() -> None:
    n = canonical_premium_pressure_protocol().minimum_eligible_observations_per_symbol
    assert n == 16_621
    valid_pairs = tuple(
        TrainingPair(
            decision_time_ms=index * _HOUR_MS,
            x=float(index % 7),
            y=-float(index % 7),
        )
        for index in range(n)
    )
    valid = calibrate_symbol("BTCUSDT", valid_pairs)
    assert valid.eligible_observations == 16_621
    assert valid.failures == ()

    invalid = calibrate_symbol("BTCUSDT", valid_pairs[:-1])
    assert invalid.eligible_observations == 16_620
    assert invalid.failures == ("BTCUSDT:eligible_observations<16621",)
    assert invalid.beta is None


def test_symbol_calibration_rejects_resigned_semantic_forgery_at_construction() -> None:
    with pytest.raises(ValueError):
        PremiumPressureSymbolCalibration(
            symbol="BTCUSDT",
            eligible_observations=16_621,
            x_bar=None,
            y_bar=None,
            numerator=None,
            denominator=None,
            alpha=None,
            beta=None,
            negative_slope=False,
            failures=("BTCUSDT:eligible_observations<16621",),
        )

    with pytest.raises(ValueError):
        PremiumPressureSymbolCalibration(
            symbol="BTCUSDT",
            eligible_observations=16_621,
            x_bar=1.0,
            y_bar=2.0,
            numerator=-2.0,
            denominator=2.0,
            alpha=3.0,
            beta=-1.0,
            negative_slope=False,
            failures=(),
        )


def test_loader_rejects_missing_field_digest_tamper_and_resigned_nested_forgery() -> (
    None
):
    payload = _valid_result().to_dict()

    missing = json.loads(json.dumps(payload))
    missing.pop("target_source_manifest_content_digest")
    missing.pop("content_digest")
    missing["content_digest"] = content_digest(missing)
    with pytest.raises(ValueError):
        load_premium_pressure_result_bytes(
            json.dumps(missing, sort_keys=True, separators=(",", ":")).encode()
        )

    bad_digest = json.loads(json.dumps(payload))
    bad_digest["content_digest"] = "0" * 64
    with pytest.raises(ValueError):
        load_premium_pressure_result_bytes(
            json.dumps(bad_digest, sort_keys=True, separators=(",", ":")).encode()
        )

    forged = json.loads(json.dumps(payload))
    forged["symbol_results"][0]["negative_slope"] = not forged["symbol_results"][0][
        "negative_slope"
    ]
    forged.pop("content_digest")
    forged["content_digest"] = content_digest(forged)
    with pytest.raises(ValueError):
        load_premium_pressure_result_bytes(
            json.dumps(forged, sort_keys=True, separators=(",", ":")).encode()
        )

    authority = json.loads(json.dumps(payload))
    authority["target_source_publisher_artifact_api_digest"] = "0" * 64
    authority.pop("content_digest")
    authority["content_digest"] = content_digest(authority)
    with pytest.raises(ValueError):
        load_premium_pressure_result_bytes(
            json.dumps(authority, sort_keys=True, separators=(",", ":")).encode()
        )
