"""Deterministic training-only implementation for the sealed Issue 600 premium study."""

from __future__ import annotations

import csv
import io
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from datetime import UTC, datetime
from typing import Any

from trade_rl.artifacts.canonical import canonical_json_bytes
from trade_rl.artifacts.hashing import content_digest
from trade_rl.evaluation.experiments.bootstrap.premium_pressure_prereg import (
    canonical_premium_pressure_protocol,
)

_HOUR_MS = 60 * 60 * 1000
_RESULT_SCHEMA = "premium_pressure_calibration_result_v2"

_PROTOCOL_HEAD = "6ffc414baa10e91f34df258fe5cfabacce65fd77"
_PROTOCOL_DIGEST = "18bf9625502eccbe475d157df90635f1c4645d3c4cced48ec0fc565723206731"
_PROTOCOL_FULL_VERIFY_RUN_ID = 34959430185
_PROTOCOL_SEAL_RUN_ID = 34959849347
_PROTOCOL_SEAL_ARTIFACT_ID = 10392936042
_PROTOCOL_SEAL_ARTIFACT_DIGEST = (
    "4b611462f8d4f9bb4d23ab15bb4d8a20525d4093e590c1afb62d1b74ca875e15"
)
_PROTOCOL_FRESH_ARTIFACT_ID = 10391714897
_PROTOCOL_FRESH_ARTIFACT_DIGEST = (
    "858e930a03e442f047556801c3d9a536fbc89435b5f261cba621988dd8062acd"
)

_PREMIUM_SOURCE_ISSUE = 570
_PREMIUM_SOURCE_RUN_ID = 34863522941
_PREMIUM_SOURCE_PROBE_HEAD = "80b65a773c622af530f074888eb01fff2f7f98df"
_PREMIUM_SOURCE_STATUS = "PASS_SPARSE_SOURCE"
_PREMIUM_SOURCE_ARTIFACT_ID = 10355913123
_PREMIUM_SOURCE_ARTIFACT_DIGEST = (
    "0076698414d3ac8197a676aecc0155853befc3828a2a9c62e92b76576ad2dc04"
)
_PREMIUM_SOURCE_FRESH_ARTIFACT_ID = 10355333076
_PREMIUM_SOURCE_FRESH_ARTIFACT_DIGEST = (
    "2212693404afefedc041b869a03dbe664d3b5cb1e25e28c6c3cdc04a89c1bd3f"
)
_PREMIUM_SOURCE_REPORT_SHA256 = (
    "059987f3692c0f249f2db5ef2e14f8cbaaefa5ca24dccf4c45bae009ca19ebe4"
)
_PREMIUM_SOURCE_REPORT_CONTENT_DIGEST = (
    "fc756962aa997cfb8beafdafe06e70e02602e23e09a5dce5a36d229ddb558f88"
)

_TARGET_SOURCE_ISSUE = 601
_TARGET_SOURCE_STATUS = "PASS_USDM_1H_TARGET_SOURCE"
_TARGET_SOURCE_VALIDATOR_HEAD = "c707be853480e57362f6a3d64f116e4056c3553f"
_TARGET_SOURCE_VALIDATOR_VERIFY_RUN_ID = 34961165095
_TARGET_SOURCE_PREFLIGHT_HEAD = "e64256dd40e76d9606edfbdebe1c37b86668d467"
_TARGET_SOURCE_RUN_ID = 34962227821
_TARGET_SOURCE_PUBLISHER_ARTIFACT_ID = 10393693240
_TARGET_SOURCE_PUBLISHER_ARTIFACT_DIGEST = (
    "0c7f240390a35174329890fa83e9615e304de5f1a84114007ead1fddaaaf37ff"
)
_TARGET_SOURCE_FRESH_ARTIFACT_ID = 10393379932
_TARGET_SOURCE_FRESH_ARTIFACT_DIGEST = (
    "0c0c05b8f5256c918b90ae709ff05ce0f05f8e88a929f1820387cf60a12dfeff"
)
_TARGET_SOURCE_REPORT_SHA256 = (
    "6833311f14e45ce634dd4ba13fdf8defeaa0b1a2b90cf7ff0598ee2054bde151"
)
_TARGET_SOURCE_REPORT_CONTENT_DIGEST = (
    "a9bfe64d5232747ac2ee543e3b65a4be22d4d034969603040391b7b0eabec988"
)
_TARGET_SOURCE_MANIFEST_SHA256 = (
    "e287f03bf18619834108b5b452c26cdffbc66a6be3d51d418c928daa2e77dd15"
)
_TARGET_SOURCE_MANIFEST_CONTENT_DIGEST = (
    "9ced3fbab6e51fbd654768ec6a4ea28f98cc4b9f36c1d9d2a9a8f6540864934e"
)

_KLINE_HEADER = (
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


def _strict_int(value: object, *, field: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{field} must be an integer >= {minimum}")
    return value


def _strict_bool(value: object, *, field: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{field} must be boolean")
    return value


def _strict_string(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _finite_number(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be finite")
    resolved = float(value)
    if not math.isfinite(resolved):
        raise ValueError(f"{field} must be finite")
    return resolved


def _optional_finite(value: object, *, field: str) -> float | None:
    if value is None:
        return None
    return _finite_number(value, field=field)


def _hex(value: object, *, length: int, field: str) -> str:
    if not isinstance(value, str) or len(value) != length:
        raise ValueError(f"{field} must be {length} lowercase hexadecimal characters")
    if value.lower() != value or any(char not in "0123456789abcdef" for char in value):
        raise ValueError(f"{field} must be {length} lowercase hexadecimal characters")
    return value


def _parse_int_token(value: str, *, field: str, minimum: int | None = None) -> int:
    if not value or value.strip() != value or value.startswith("+"):
        raise ValueError(f"{field} must be an integer token")
    try:
        parsed = int(value)
    except ValueError as error:
        raise ValueError(f"{field} must be an integer token") from error
    if str(parsed) != value and not (value == "-0" and parsed == 0):
        raise ValueError(f"{field} must be an integer token")
    if minimum is not None and parsed < minimum:
        raise ValueError(f"{field} must be >= {minimum}")
    return parsed


def _parse_float_token(value: str, *, field: str) -> float:
    if not value or value.strip() != value:
        raise ValueError(f"{field} must be finite")
    try:
        parsed = float(value)
    except ValueError as error:
        raise ValueError(f"{field} must be finite") from error
    if not math.isfinite(parsed):
        raise ValueError(f"{field} must be finite")
    return parsed


def _month_bounds_ms(month: str) -> tuple[int, int]:
    try:
        start = datetime.strptime(month + "-01", "%Y-%m-%d").replace(tzinfo=UTC)
    except ValueError as error:
        raise ValueError("expected_month must use YYYY-MM") from error
    if start.strftime("%Y-%m") != month:
        raise ValueError("expected_month must use YYYY-MM")
    if start.month == 12:
        end = datetime(start.year + 1, 1, 1, tzinfo=UTC)
    else:
        end = datetime(start.year, start.month + 1, 1, tzinfo=UTC)
    return int(start.timestamp() * 1000), int(end.timestamp() * 1000)


def _decode_kline_rows(payload: bytes) -> list[list[str]]:
    if not isinstance(payload, bytes):
        raise TypeError("CSV payload must be bytes")
    try:
        text = payload.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise ValueError("CSV payload is not UTF-8") from error
    rows = list(csv.reader(io.StringIO(text)))
    if rows and rows[0] and rows[0][0] == "open_time":
        if tuple(rows[0]) != _KLINE_HEADER:
            raise ValueError("kline header differs from frozen schema")
        rows = rows[1:]
    return rows


def _validate_common_kline_row(
    row: list[str],
    *,
    expected_month: str,
    previous_open_time: int | None,
) -> tuple[int, int]:
    if len(row) != len(_KLINE_HEADER):
        raise ValueError("kline row is malformed: expected twelve fields")
    open_time = _parse_int_token(row[0], field="open_time", minimum=0)
    close_time = _parse_int_token(row[6], field="close_time", minimum=0)
    month_start, month_end = _month_bounds_ms(expected_month)
    if not month_start <= open_time < month_end:
        raise ValueError("open_time is outside expected_month")
    if open_time % _HOUR_MS != 0:
        raise ValueError("open_time is off the native 1h grid")
    if close_time != open_time + _HOUR_MS - 1:
        raise ValueError("close_time violates native 1h semantics")
    if previous_open_time is not None and open_time <= previous_open_time:
        raise ValueError("open times must be strictly increasing and unique")
    for index, field in (
        (1, "open"),
        (2, "high"),
        (3, "low"),
        (4, "close"),
        (5, "volume"),
        (7, "quote_volume"),
        (9, "taker_buy_volume"),
        (10, "taker_buy_quote_volume"),
        (11, "ignore"),
    ):
        _parse_float_token(row[index], field=field)
    _parse_int_token(row[8], field="count", minimum=0)
    return open_time, close_time


@dataclass(frozen=True, slots=True)
class PremiumObservation:
    raw_open_time_ms: int
    dataset_time_ms: int
    close_bps: float

    def __post_init__(self) -> None:
        raw = _strict_int(self.raw_open_time_ms, field="raw_open_time_ms")
        dataset = _strict_int(self.dataset_time_ms, field="dataset_time_ms")
        if raw % _HOUR_MS != 0 or dataset != raw + _HOUR_MS:
            raise ValueError("Premium observation violates completed-bar 1h clock")
        _finite_number(self.close_bps, field="close_bps")


@dataclass(frozen=True, slots=True)
class TargetOpenObservation:
    raw_open_time_ms: int
    dataset_time_ms: int
    open_price: float
    information_available: bool = True
    active: bool = True
    tradable: bool = True

    def __post_init__(self) -> None:
        raw = _strict_int(self.raw_open_time_ms, field="raw_open_time_ms")
        dataset = _strict_int(self.dataset_time_ms, field="dataset_time_ms")
        if raw % _HOUR_MS != 0 or dataset != raw + _HOUR_MS:
            raise ValueError("target observation violates completed-bar 1h clock")
        price = _finite_number(self.open_price, field="open_price")
        if price <= 0.0:
            raise ValueError("open_price must be finite and positive")
        _strict_bool(self.information_available, field="information_available")
        _strict_bool(self.active, field="active")
        _strict_bool(self.tradable, field="tradable")


@dataclass(frozen=True, slots=True)
class TrainingPair:
    decision_time_ms: int
    x: float
    y: float

    def __post_init__(self) -> None:
        decision = _strict_int(self.decision_time_ms, field="decision_time_ms")
        if decision % _HOUR_MS != 0:
            raise ValueError("decision_time_ms must lie on the native 1h grid")
        _finite_number(self.x, field="x")
        _finite_number(self.y, field="y")


@dataclass(frozen=True, slots=True)
class PremiumPressureSymbolCalibration:
    symbol: str
    eligible_observations: int
    x_bar: float | None
    y_bar: float | None
    numerator: float | None
    denominator: float | None
    alpha: float | None
    beta: float | None
    negative_slope: bool
    failures: tuple[str, ...]

    def __post_init__(self) -> None:
        protocol = canonical_premium_pressure_protocol()
        symbol = _strict_string(self.symbol, field="symbol")
        if symbol not in protocol.symbols:
            raise ValueError("symbol is outside the frozen Premium Pressure roster")
        eligible = _strict_int(
            self.eligible_observations, field="eligible_observations"
        )
        _strict_bool(self.negative_slope, field="negative_slope")
        if any(not isinstance(item, str) or not item for item in self.failures):
            raise ValueError("failures must contain non-empty strings")
        stats = (
            self.x_bar,
            self.y_bar,
            self.numerator,
            self.denominator,
            self.alpha,
            self.beta,
        )
        minimum = protocol.minimum_eligible_observations_per_symbol
        if eligible < minimum:
            expected = (f"{symbol}:eligible_observations<{minimum}",)
            if self.failures != expected:
                raise ValueError("low-coverage failure does not match frozen threshold")
            if any(value is not None for value in stats):
                raise ValueError("low-coverage statistics must be null")
            if self.negative_slope:
                raise ValueError("negative_slope must be false on low coverage")
            return

        if self.failures:
            allowed = {
                f"{symbol}:calibration_reduction_not_finite",
                f"{symbol}:x_bar_not_finite",
                f"{symbol}:y_bar_not_finite",
                f"{symbol}:numerator_not_finite",
                f"{symbol}:denominator_not_finite_positive",
                f"{symbol}:alpha_or_beta_not_finite",
            }
            if len(self.failures) != 1 or self.failures[0] not in allowed:
                raise ValueError("numeric calibration failure is not canonical")
            if any(value is not None for value in stats):
                raise ValueError(
                    "calibration statistics must be null when failures exist"
                )
            if self.negative_slope:
                raise ValueError("negative_slope must be false when failures exist")
            return

        if any(value is None for value in stats):
            raise ValueError("successful calibration requires all statistics")
        x_bar = _finite_number(self.x_bar, field="x_bar")
        y_bar = _finite_number(self.y_bar, field="y_bar")
        numerator = _finite_number(self.numerator, field="numerator")
        denominator = _finite_number(self.denominator, field="denominator")
        alpha = _finite_number(self.alpha, field="alpha")
        beta = _finite_number(self.beta, field="beta")
        if denominator <= 0.0:
            raise ValueError("denominator must be finite and positive")
        expected_beta = numerator / denominator
        expected_alpha = y_bar - expected_beta * x_bar
        if beta != expected_beta:
            raise ValueError("beta must equal numerator / denominator")
        if alpha != expected_alpha:
            raise ValueError("alpha must equal y_bar - beta*x_bar")
        if self.negative_slope is not (beta < 0.0):
            raise ValueError("negative_slope must equal beta < 0")

    def to_dict(self) -> dict[str, object]:
        return {
            "symbol": self.symbol,
            "eligible_observations": self.eligible_observations,
            "x_bar": self.x_bar,
            "y_bar": self.y_bar,
            "numerator": self.numerator,
            "denominator": self.denominator,
            "alpha": self.alpha,
            "beta": self.beta,
            "negative_slope": self.negative_slope,
            "failures": list(self.failures),
        }

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, object]
    ) -> PremiumPressureSymbolCalibration:
        expected = {
            "symbol",
            "eligible_observations",
            "x_bar",
            "y_bar",
            "numerator",
            "denominator",
            "alpha",
            "beta",
            "negative_slope",
            "failures",
        }
        if set(payload) != expected:
            raise ValueError("symbol calibration fields differ from frozen schema")
        failures = payload["failures"]
        if not isinstance(failures, list) or any(
            not isinstance(item, str) for item in failures
        ):
            raise ValueError("failures must be a JSON string array")
        return cls(
            symbol=_strict_string(payload["symbol"], field="symbol"),
            eligible_observations=_strict_int(
                payload["eligible_observations"], field="eligible_observations"
            ),
            x_bar=_optional_finite(payload["x_bar"], field="x_bar"),
            y_bar=_optional_finite(payload["y_bar"], field="y_bar"),
            numerator=_optional_finite(payload["numerator"], field="numerator"),
            denominator=_optional_finite(payload["denominator"], field="denominator"),
            alpha=_optional_finite(payload["alpha"], field="alpha"),
            beta=_optional_finite(payload["beta"], field="beta"),
            negative_slope=_strict_bool(
                payload["negative_slope"], field="negative_slope"
            ),
            failures=tuple(failures),
        )


def parse_premium_index_1h_csv(
    payload: bytes,
    *,
    expected_month: str,
) -> tuple[PremiumObservation, ...]:
    """Parse frozen Premium Index 1h rows into causal completed-bar observations."""

    rows = _decode_kline_rows(payload)
    result: list[PremiumObservation] = []
    previous: int | None = None
    for row in rows:
        open_time, _ = _validate_common_kline_row(
            row, expected_month=expected_month, previous_open_time=previous
        )
        previous = open_time
        close = _parse_float_token(row[4], field="premium close")
        result.append(
            PremiumObservation(
                raw_open_time_ms=open_time,
                dataset_time_ms=open_time + _HOUR_MS,
                close_bps=10_000.0 * close,
            )
        )
    return tuple(result)


def parse_usdm_1h_target_csv(
    payload: bytes,
    *,
    expected_month: str,
) -> tuple[TargetOpenObservation, ...]:
    """Parse frozen standard-contract 1h rows into completed-bar open observations."""

    rows = _decode_kline_rows(payload)
    result: list[TargetOpenObservation] = []
    previous: int | None = None
    for row in rows:
        open_time, _ = _validate_common_kline_row(
            row, expected_month=expected_month, previous_open_time=previous
        )
        previous = open_time
        open_price = _parse_float_token(row[1], field="target open")
        if open_price <= 0.0:
            raise ValueError("target open must be finite and positive")
        result.append(
            TargetOpenObservation(
                raw_open_time_ms=open_time,
                dataset_time_ms=open_time + _HOUR_MS,
                open_price=open_price,
            )
        )
    return tuple(result)


def _ordered_unique_premium(
    observations: Sequence[PremiumObservation],
) -> tuple[PremiumObservation, ...]:
    result: list[PremiumObservation] = []
    previous: int | None = None
    for item in observations:
        if not isinstance(item, PremiumObservation):
            raise TypeError("premium observations must be PremiumObservation values")
        if previous is not None and item.dataset_time_ms <= previous:
            raise ValueError("premium Dataset timestamps must be strictly increasing")
        previous = item.dataset_time_ms
        result.append(item)
    return tuple(result)


def _target_by_dataset_time(
    observations: Sequence[TargetOpenObservation],
) -> dict[int, TargetOpenObservation]:
    result: dict[int, TargetOpenObservation] = {}
    previous: int | None = None
    for item in observations:
        if not isinstance(item, TargetOpenObservation):
            raise TypeError("target observations must be TargetOpenObservation values")
        if previous is not None and item.dataset_time_ms <= previous:
            raise ValueError("target Dataset timestamps must be strictly increasing")
        previous = item.dataset_time_ms
        if item.dataset_time_ms in result:
            raise ValueError("target Dataset timestamps must be unique")
        result[item.dataset_time_ms] = item
    return result


def build_training_pairs(
    premium_observations: Sequence[PremiumObservation],
    target_observations: Sequence[TargetOpenObservation],
) -> tuple[TrainingPair, ...]:
    """Build exact t -> (t+1,t+25) training pairs with no fill/asof fallback."""

    protocol = canonical_premium_pressure_protocol()
    premium = _ordered_unique_premium(premium_observations)
    target = _target_by_dataset_time(target_observations)
    fit_start_ms = int(protocol.fit_start.timestamp() * 1000)
    fit_cutoff_ms = int(protocol.fit_cutoff.timestamp() * 1000)
    last_decision_ms = int(protocol.last_candidate_decision.timestamp() * 1000)
    result: list[TrainingPair] = []

    for feature in premium:
        decision = feature.dataset_time_ms
        if decision < fit_start_ms or decision > last_decision_ms:
            continue
        required: list[TargetOpenObservation] = []
        missing = False
        for offset in range(1, protocol.label_endpoint_offset_bars + 1):
            timestamp = decision + offset * _HOUR_MS
            point = target.get(timestamp)
            if point is None:
                missing = True
                break
            if not (point.information_available and point.active and point.tradable):
                missing = True
                break
            required.append(point)
        if missing:
            continue
        if len(required) != protocol.label_endpoint_offset_bars:
            continue
        endpoint = required[-1]
        if endpoint.raw_open_time_ms >= fit_cutoff_ms:
            continue
        execution = required[0]
        ratio = endpoint.open_price / execution.open_price
        if not math.isfinite(ratio) or ratio <= 0.0:
            continue
        label = math.log(ratio)
        if not math.isfinite(label):
            continue
        result.append(
            TrainingPair(decision_time_ms=decision, x=feature.close_bps, y=label)
        )
    return tuple(result)


def calibrate_symbol(
    symbol: str,
    pairs: Sequence[TrainingPair],
) -> PremiumPressureSymbolCalibration:
    """Compute the sealed per-symbol OLS-with-intercept calibration."""

    protocol = canonical_premium_pressure_protocol()
    if symbol not in protocol.symbols:
        raise ValueError("symbol is outside the frozen Premium Pressure roster")
    ordered: list[TrainingPair] = []
    previous: int | None = None
    for pair in pairs:
        if not isinstance(pair, TrainingPair):
            raise TypeError("pairs must contain TrainingPair values")
        if previous is not None and pair.decision_time_ms <= previous:
            raise ValueError("training pairs must be strictly chronological and unique")
        previous = pair.decision_time_ms
        ordered.append(pair)

    n = len(ordered)
    minimum = protocol.minimum_eligible_observations_per_symbol
    if n < minimum:
        return PremiumPressureSymbolCalibration(
            symbol=symbol,
            eligible_observations=n,
            x_bar=None,
            y_bar=None,
            numerator=None,
            denominator=None,
            alpha=None,
            beta=None,
            negative_slope=False,
            failures=(f"{symbol}:eligible_observations<{minimum}",),
        )

    try:
        x_bar = math.fsum(pair.x for pair in ordered) / n
        y_bar = math.fsum(pair.y for pair in ordered) / n
        numerator = math.fsum((pair.x - x_bar) * (pair.y - y_bar) for pair in ordered)
        denominator = math.fsum((pair.x - x_bar) ** 2 for pair in ordered)
    except (OverflowError, ValueError):
        return PremiumPressureSymbolCalibration(
            symbol=symbol,
            eligible_observations=n,
            x_bar=None,
            y_bar=None,
            numerator=None,
            denominator=None,
            alpha=None,
            beta=None,
            negative_slope=False,
            failures=(f"{symbol}:calibration_reduction_not_finite",),
        )

    if not math.isfinite(x_bar):
        failure = "x_bar_not_finite"
    elif not math.isfinite(y_bar):
        failure = "y_bar_not_finite"
    elif not math.isfinite(numerator):
        failure = "numerator_not_finite"
    elif not math.isfinite(denominator) or denominator <= 0.0:
        failure = "denominator_not_finite_positive"
    else:
        failure = ""
    if failure:
        return PremiumPressureSymbolCalibration(
            symbol=symbol,
            eligible_observations=n,
            x_bar=None,
            y_bar=None,
            numerator=None,
            denominator=None,
            alpha=None,
            beta=None,
            negative_slope=False,
            failures=(f"{symbol}:{failure}",),
        )

    beta = numerator / denominator
    alpha = y_bar - beta * x_bar
    if not math.isfinite(beta) or not math.isfinite(alpha):
        return PremiumPressureSymbolCalibration(
            symbol=symbol,
            eligible_observations=n,
            x_bar=None,
            y_bar=None,
            numerator=None,
            denominator=None,
            alpha=None,
            beta=None,
            negative_slope=False,
            failures=(f"{symbol}:alpha_or_beta_not_finite",),
        )
    return PremiumPressureSymbolCalibration(
        symbol=symbol,
        eligible_observations=n,
        x_bar=x_bar,
        y_bar=y_bar,
        numerator=numerator,
        denominator=denominator,
        alpha=alpha,
        beta=beta,
        negative_slope=beta < 0.0,
        failures=(),
    )


@dataclass(frozen=True, slots=True)
class PremiumPressureResult:
    schema_version: str
    issue_number: int
    protocol_issue: int
    multiplicity_issue: int
    protocol_head: str
    protocol_digest: str
    protocol_full_verify_run_id: int
    protocol_seal_run_id: int
    protocol_seal_artifact_id: int
    protocol_seal_artifact_api_digest: str
    protocol_fresh_artifact_id: int
    protocol_fresh_artifact_api_digest: str
    premium_source_issue: int
    premium_source_probe_head: str
    premium_source_status: str
    premium_source_run_id: int
    premium_source_artifact_id: int
    premium_source_artifact_api_digest: str
    premium_source_fresh_artifact_id: int
    premium_source_fresh_artifact_api_digest: str
    premium_source_report_sha256: str
    premium_source_report_content_digest: str
    target_source_issue: int
    target_source_status: str
    target_source_validator_head: str
    target_source_validator_verification_run_id: int
    target_source_preflight_head: str
    target_source_run_id: int
    target_source_publisher_artifact_id: int
    target_source_publisher_artifact_api_digest: str
    target_source_fresh_artifact_id: int
    target_source_fresh_artifact_api_digest: str
    target_source_report_sha256: str
    target_source_report_content_digest: str
    target_source_manifest_sha256: str
    target_source_manifest_content_digest: str
    implementation_head: str
    implementation_verification_run_id: int
    execution_run_id: int
    minimum_eligible_observations_per_symbol: int
    required_negative_symbol_slopes: int
    symbol_results: tuple[PremiumPressureSymbolCalibration, ...]
    negative_slope_count: int
    status: str
    training_relation_executed: bool
    evaluation_pnl_inspected: bool
    evaluation_execution_authorized: bool
    final_test_authorized: bool
    shared_cash_profitability_established: bool
    production_eligible: bool
    live_trading_authorized: bool
    content_digest: str

    def __post_init__(self) -> None:
        protocol = canonical_premium_pressure_protocol()
        if protocol.digest != _PROTOCOL_DIGEST:
            raise ValueError("runtime protocol digest differs from sealed authority")
        for field_name in (
            "protocol_head",
            "premium_source_probe_head",
            "target_source_validator_head",
            "target_source_preflight_head",
            "implementation_head",
        ):
            _hex(getattr(self, field_name), length=40, field=field_name)
        for field_name in (
            "protocol_digest",
            "protocol_seal_artifact_api_digest",
            "protocol_fresh_artifact_api_digest",
            "premium_source_artifact_api_digest",
            "premium_source_fresh_artifact_api_digest",
            "premium_source_report_sha256",
            "premium_source_report_content_digest",
            "target_source_publisher_artifact_api_digest",
            "target_source_fresh_artifact_api_digest",
            "target_source_report_sha256",
            "target_source_report_content_digest",
            "target_source_manifest_sha256",
            "target_source_manifest_content_digest",
            "content_digest",
        ):
            _hex(getattr(self, field_name), length=64, field=field_name)
        for field_name in (
            "issue_number",
            "protocol_issue",
            "multiplicity_issue",
            "protocol_full_verify_run_id",
            "protocol_seal_run_id",
            "protocol_seal_artifact_id",
            "protocol_fresh_artifact_id",
            "premium_source_issue",
            "premium_source_run_id",
            "premium_source_artifact_id",
            "premium_source_fresh_artifact_id",
            "target_source_issue",
            "target_source_validator_verification_run_id",
            "target_source_run_id",
            "target_source_publisher_artifact_id",
            "target_source_fresh_artifact_id",
            "implementation_verification_run_id",
            "execution_run_id",
            "minimum_eligible_observations_per_symbol",
            "required_negative_symbol_slopes",
        ):
            _strict_int(getattr(self, field_name), field=field_name, minimum=1)
        _strict_int(self.negative_slope_count, field="negative_slope_count")
        _strict_string(self.schema_version, field="schema_version")
        _strict_string(self.premium_source_status, field="premium_source_status")
        _strict_string(self.target_source_status, field="target_source_status")
        _strict_string(self.status, field="status")
        for field_name in (
            "training_relation_executed",
            "evaluation_pnl_inspected",
            "evaluation_execution_authorized",
            "final_test_authorized",
            "shared_cash_profitability_established",
            "production_eligible",
            "live_trading_authorized",
        ):
            _strict_bool(getattr(self, field_name), field=field_name)

        fixed_authority: dict[str, object] = {
            "schema_version": _RESULT_SCHEMA,
            "issue_number": 603,
            "protocol_issue": 600,
            "multiplicity_issue": 599,
            "protocol_head": _PROTOCOL_HEAD,
            "protocol_digest": _PROTOCOL_DIGEST,
            "protocol_full_verify_run_id": _PROTOCOL_FULL_VERIFY_RUN_ID,
            "protocol_seal_run_id": _PROTOCOL_SEAL_RUN_ID,
            "protocol_seal_artifact_id": _PROTOCOL_SEAL_ARTIFACT_ID,
            "protocol_seal_artifact_api_digest": _PROTOCOL_SEAL_ARTIFACT_DIGEST,
            "protocol_fresh_artifact_id": _PROTOCOL_FRESH_ARTIFACT_ID,
            "protocol_fresh_artifact_api_digest": _PROTOCOL_FRESH_ARTIFACT_DIGEST,
            "premium_source_issue": _PREMIUM_SOURCE_ISSUE,
            "premium_source_probe_head": _PREMIUM_SOURCE_PROBE_HEAD,
            "premium_source_status": _PREMIUM_SOURCE_STATUS,
            "premium_source_run_id": _PREMIUM_SOURCE_RUN_ID,
            "premium_source_artifact_id": _PREMIUM_SOURCE_ARTIFACT_ID,
            "premium_source_artifact_api_digest": _PREMIUM_SOURCE_ARTIFACT_DIGEST,
            "premium_source_fresh_artifact_id": _PREMIUM_SOURCE_FRESH_ARTIFACT_ID,
            "premium_source_fresh_artifact_api_digest": _PREMIUM_SOURCE_FRESH_ARTIFACT_DIGEST,
            "premium_source_report_sha256": _PREMIUM_SOURCE_REPORT_SHA256,
            "premium_source_report_content_digest": _PREMIUM_SOURCE_REPORT_CONTENT_DIGEST,
            "target_source_issue": _TARGET_SOURCE_ISSUE,
            "target_source_status": _TARGET_SOURCE_STATUS,
            "target_source_validator_head": _TARGET_SOURCE_VALIDATOR_HEAD,
            "target_source_validator_verification_run_id": _TARGET_SOURCE_VALIDATOR_VERIFY_RUN_ID,
            "target_source_preflight_head": _TARGET_SOURCE_PREFLIGHT_HEAD,
            "target_source_run_id": _TARGET_SOURCE_RUN_ID,
            "target_source_publisher_artifact_id": _TARGET_SOURCE_PUBLISHER_ARTIFACT_ID,
            "target_source_publisher_artifact_api_digest": _TARGET_SOURCE_PUBLISHER_ARTIFACT_DIGEST,
            "target_source_fresh_artifact_id": _TARGET_SOURCE_FRESH_ARTIFACT_ID,
            "target_source_fresh_artifact_api_digest": _TARGET_SOURCE_FRESH_ARTIFACT_DIGEST,
            "target_source_report_sha256": _TARGET_SOURCE_REPORT_SHA256,
            "target_source_report_content_digest": _TARGET_SOURCE_REPORT_CONTENT_DIGEST,
            "target_source_manifest_sha256": _TARGET_SOURCE_MANIFEST_SHA256,
            "target_source_manifest_content_digest": _TARGET_SOURCE_MANIFEST_CONTENT_DIGEST,
            "minimum_eligible_observations_per_symbol": protocol.minimum_eligible_observations_per_symbol,
            "required_negative_symbol_slopes": protocol.required_negative_symbol_slopes,
        }
        for field_name, expected in fixed_authority.items():
            if getattr(self, field_name) != expected:
                raise ValueError(f"{field_name} differs from frozen authority")

        if tuple(item.symbol for item in self.symbol_results) != protocol.symbols:
            raise ValueError("symbol result roster/order differs from frozen protocol")
        observed_negative = sum(
            item.failures == () and item.negative_slope for item in self.symbol_results
        )
        if self.negative_slope_count != observed_negative:
            raise ValueError("negative_slope_count differs from symbol semantics")
        if any(item.failures for item in self.symbol_results):
            expected_status = protocol.invalid_coverage_status
        elif observed_negative >= protocol.required_negative_symbol_slopes:
            expected_status = protocol.valid_status
        else:
            expected_status = protocol.reject_status
        if self.status != expected_status:
            raise ValueError("status differs from frozen decision rule")
        if self.training_relation_executed is not True:
            raise ValueError("training_relation_executed must be true for a result")
        if any(
            (
                self.evaluation_pnl_inspected,
                self.evaluation_execution_authorized,
                self.final_test_authorized,
                self.shared_cash_profitability_established,
                self.production_eligible,
                self.live_trading_authorized,
            )
        ):
            raise ValueError("training-only result crossed a forbidden boundary")
        unsigned = self.to_dict()
        observed_digest = unsigned.pop("content_digest")
        if content_digest(unsigned) != observed_digest:
            raise ValueError("content_digest differs from canonical semantic payload")

    def to_dict(self) -> dict[str, object]:
        return {
            field.name: (
                [item.to_dict() for item in self.symbol_results]
                if field.name == "symbol_results"
                else getattr(self, field.name)
            )
            for field in fields(self)
        }


def _result_without_digest(
    calibrations: Sequence[PremiumPressureSymbolCalibration],
    *,
    implementation_head: str,
    implementation_verification_run_id: int,
    execution_run_id: int,
) -> dict[str, object]:
    protocol = canonical_premium_pressure_protocol()
    if protocol.digest != _PROTOCOL_DIGEST:
        raise ValueError("runtime protocol differs from sealed Issue 600 authority")
    by_symbol: dict[str, PremiumPressureSymbolCalibration] = {}
    for calibration in calibrations:
        if not isinstance(calibration, PremiumPressureSymbolCalibration):
            raise TypeError(
                "calibrations must contain PremiumPressureSymbolCalibration values"
            )
        if calibration.symbol in by_symbol:
            raise ValueError("duplicate symbol calibration")
        by_symbol[calibration.symbol] = calibration
    if set(by_symbol) != set(protocol.symbols):
        raise ValueError("calibration symbol roster differs from frozen protocol")
    ordered = tuple(by_symbol[symbol] for symbol in protocol.symbols)
    negative_count = sum(
        item.failures == () and item.negative_slope is True for item in ordered
    )
    if any(item.failures for item in ordered):
        status = protocol.invalid_coverage_status
    elif negative_count >= protocol.required_negative_symbol_slopes:
        status = protocol.valid_status
    else:
        status = protocol.reject_status
    head = _hex(implementation_head, length=40, field="implementation_head")
    verify_run = _strict_int(
        implementation_verification_run_id,
        field="implementation_verification_run_id",
        minimum=1,
    )
    run_id = _strict_int(execution_run_id, field="execution_run_id", minimum=1)
    return {
        "schema_version": _RESULT_SCHEMA,
        "issue_number": 603,
        "protocol_issue": 600,
        "multiplicity_issue": 599,
        "protocol_head": _PROTOCOL_HEAD,
        "protocol_digest": _PROTOCOL_DIGEST,
        "protocol_full_verify_run_id": _PROTOCOL_FULL_VERIFY_RUN_ID,
        "protocol_seal_run_id": _PROTOCOL_SEAL_RUN_ID,
        "protocol_seal_artifact_id": _PROTOCOL_SEAL_ARTIFACT_ID,
        "protocol_seal_artifact_api_digest": _PROTOCOL_SEAL_ARTIFACT_DIGEST,
        "protocol_fresh_artifact_id": _PROTOCOL_FRESH_ARTIFACT_ID,
        "protocol_fresh_artifact_api_digest": _PROTOCOL_FRESH_ARTIFACT_DIGEST,
        "premium_source_issue": _PREMIUM_SOURCE_ISSUE,
        "premium_source_probe_head": _PREMIUM_SOURCE_PROBE_HEAD,
        "premium_source_status": _PREMIUM_SOURCE_STATUS,
        "premium_source_run_id": _PREMIUM_SOURCE_RUN_ID,
        "premium_source_artifact_id": _PREMIUM_SOURCE_ARTIFACT_ID,
        "premium_source_artifact_api_digest": _PREMIUM_SOURCE_ARTIFACT_DIGEST,
        "premium_source_fresh_artifact_id": _PREMIUM_SOURCE_FRESH_ARTIFACT_ID,
        "premium_source_fresh_artifact_api_digest": _PREMIUM_SOURCE_FRESH_ARTIFACT_DIGEST,
        "premium_source_report_sha256": _PREMIUM_SOURCE_REPORT_SHA256,
        "premium_source_report_content_digest": _PREMIUM_SOURCE_REPORT_CONTENT_DIGEST,
        "target_source_issue": _TARGET_SOURCE_ISSUE,
        "target_source_status": _TARGET_SOURCE_STATUS,
        "target_source_validator_head": _TARGET_SOURCE_VALIDATOR_HEAD,
        "target_source_validator_verification_run_id": _TARGET_SOURCE_VALIDATOR_VERIFY_RUN_ID,
        "target_source_preflight_head": _TARGET_SOURCE_PREFLIGHT_HEAD,
        "target_source_run_id": _TARGET_SOURCE_RUN_ID,
        "target_source_publisher_artifact_id": _TARGET_SOURCE_PUBLISHER_ARTIFACT_ID,
        "target_source_publisher_artifact_api_digest": _TARGET_SOURCE_PUBLISHER_ARTIFACT_DIGEST,
        "target_source_fresh_artifact_id": _TARGET_SOURCE_FRESH_ARTIFACT_ID,
        "target_source_fresh_artifact_api_digest": _TARGET_SOURCE_FRESH_ARTIFACT_DIGEST,
        "target_source_report_sha256": _TARGET_SOURCE_REPORT_SHA256,
        "target_source_report_content_digest": _TARGET_SOURCE_REPORT_CONTENT_DIGEST,
        "target_source_manifest_sha256": _TARGET_SOURCE_MANIFEST_SHA256,
        "target_source_manifest_content_digest": _TARGET_SOURCE_MANIFEST_CONTENT_DIGEST,
        "implementation_head": head,
        "implementation_verification_run_id": verify_run,
        "execution_run_id": run_id,
        "minimum_eligible_observations_per_symbol": protocol.minimum_eligible_observations_per_symbol,
        "required_negative_symbol_slopes": protocol.required_negative_symbol_slopes,
        "symbol_results": [item.to_dict() for item in ordered],
        "negative_slope_count": int(negative_count),
        "status": status,
        "training_relation_executed": True,
        "evaluation_pnl_inspected": False,
        "evaluation_execution_authorized": False,
        "final_test_authorized": False,
        "shared_cash_profitability_established": False,
        "production_eligible": False,
        "live_trading_authorized": False,
    }


def build_premium_pressure_result(
    calibrations: Sequence[PremiumPressureSymbolCalibration],
    *,
    implementation_head: str,
    implementation_verification_run_id: int,
    execution_run_id: int,
) -> PremiumPressureResult:
    """Build the strict content-addressed training-only result."""

    payload = _result_without_digest(
        calibrations,
        implementation_head=implementation_head,
        implementation_verification_run_id=implementation_verification_run_id,
        execution_run_id=execution_run_id,
    )
    digest = content_digest(payload)
    symbol_payloads = payload.pop("symbol_results")
    if not isinstance(symbol_payloads, list):
        raise ValueError("symbol result payload is malformed")
    symbol_results = tuple(
        PremiumPressureSymbolCalibration.from_dict(item)
        for item in symbol_payloads
        if isinstance(item, dict)
    )
    if len(symbol_results) != len(symbol_payloads):
        raise ValueError("symbol result payload is malformed")
    return PremiumPressureResult(
        **payload,  # type: ignore[arg-type]
        symbol_results=symbol_results,
        content_digest=digest,
    )


def canonical_premium_pressure_result_bytes(result: PremiumPressureResult) -> bytes:
    """Validate complete semantic closure and return canonical result bytes."""

    if not isinstance(result, PremiumPressureResult):
        raise TypeError("result must be PremiumPressureResult")
    rebuilt = build_premium_pressure_result(
        result.symbol_results,
        implementation_head=result.implementation_head,
        implementation_verification_run_id=result.implementation_verification_run_id,
        execution_run_id=result.execution_run_id,
    )
    if rebuilt != result:
        raise ValueError("Premium Pressure result differs from semantic reconstruction")
    return canonical_json_bytes(result.to_dict())


def load_premium_pressure_result_bytes(payload: bytes) -> PremiumPressureResult:
    """Strictly load canonical, authority-bound Issue 603 result bytes."""

    if not isinstance(payload, bytes):
        raise TypeError("payload must be bytes")
    try:
        raw: Any = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("Premium Pressure result is not valid UTF-8 JSON") from error
    if not isinstance(raw, dict) or any(not isinstance(key, str) for key in raw):
        raise ValueError("Premium Pressure result must be a JSON object")
    if canonical_json_bytes(raw) != payload:
        raise ValueError("Premium Pressure result bytes are not canonical JSON")
    expected_fields = {field.name for field in fields(PremiumPressureResult)}
    if set(raw) != expected_fields:
        raise ValueError("Premium Pressure result fields differ from frozen schema")
    symbol_payloads = raw.get("symbol_results")
    if not isinstance(symbol_payloads, list):
        raise ValueError("symbol_results must be a list")
    calibrations = tuple(
        PremiumPressureSymbolCalibration.from_dict(item)
        for item in symbol_payloads
        if isinstance(item, dict)
    )
    if len(calibrations) != len(symbol_payloads):
        raise ValueError("symbol_results contains malformed entries")
    head = raw.get("implementation_head")
    verify_run = raw.get("implementation_verification_run_id")
    execution_run = raw.get("execution_run_id")
    if not isinstance(head, str):
        raise ValueError("implementation_head must be a string")
    if isinstance(verify_run, bool) or not isinstance(verify_run, int):
        raise ValueError("implementation_verification_run_id must be integer")
    if isinstance(execution_run, bool) or not isinstance(execution_run, int):
        raise ValueError("execution_run_id must be integer")
    rebuilt = build_premium_pressure_result(
        calibrations,
        implementation_head=head,
        implementation_verification_run_id=verify_run,
        execution_run_id=execution_run,
    )
    if rebuilt.to_dict() != raw:
        raise ValueError("Premium Pressure result semantic closure differs")
    return rebuilt


__all__ = [
    "PremiumObservation",
    "PremiumPressureResult",
    "PremiumPressureSymbolCalibration",
    "TargetOpenObservation",
    "TrainingPair",
    "build_premium_pressure_result",
    "build_training_pairs",
    "calibrate_symbol",
    "canonical_premium_pressure_result_bytes",
    "load_premium_pressure_result_bytes",
    "parse_premium_index_1h_csv",
    "parse_usdm_1h_target_csv",
]
