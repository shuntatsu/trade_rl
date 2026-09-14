"""Deterministic training-only calibration for the sealed signed taker-flow gate."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from trade_rl.artifacts.hashing import content_digest
from trade_rl.data.market import MarketDataset
from trade_rl.evaluation.experiments.bootstrap.signed_taker_flow_prereg import (
    SignedTakerFlowProtocol,
    canonical_signed_taker_flow_protocol,
)

_SCHEMA_VERSION = "signed_taker_flow_calibration_v1"
_PROTOCOL_SEAL_RUN_ID = 34844820406
_PROTOCOL_SEAL_ARTIFACT_ID = 10347597245
_PROTOCOL_SEAL_API_DIGEST = (
    "sha256:f3521a53df81828eb7b94fd23a6bdc8c47bcb064004ca848e0afc9844b1090a2"
)
_PROTOCOL_JSON_SHA256 = (
    "a94ba98b1905387892896a7e5e9078b1e7a06e1a67d62fc7c1fdef418cf557e0"
)
_PROTOCOL_FRESH_ARTIFACT_ID = 10348235185
_PROTOCOL_FRESH_API_DIGEST = (
    "sha256:95f97dc22946bd91f9b8abc6747f5f8ae739529220e144a7d4b0f6a52a291cb4"
)
_IMPLEMENTATION_HEAD = "adf835cf45936dbc40bf8d4f74879825ae211ef9"
_ONE_HOUR_NS = 3_600_000_000_000


def _datetime64(value: datetime) -> np.datetime64:
    return np.datetime64(value.astimezone(UTC).replace(tzinfo=None), "ns")


def _datetime_text(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _require_hex(value: object, *, length: int, field: str) -> str:
    if not isinstance(value, str) or len(value) != length:
        raise ValueError(f"{field} must be a {length}-character lowercase hex value")
    if value.lower() != value or any(char not in "0123456789abcdef" for char in value):
        raise ValueError(f"{field} must be a {length}-character lowercase hex value")
    return value


def _require_int(value: object, *, field: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{field} must be an integer >= {minimum}")
    return value


def _require_bool(value: object, *, field: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{field} must be boolean")
    return value


def _require_float_or_none(value: object, *, field: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be finite or null")
    resolved = float(value)
    if not math.isfinite(resolved):
        raise ValueError(f"{field} must be finite or null")
    return resolved


def _require_string(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _required_bool_matrix(value: np.ndarray | None, *, field: str) -> np.ndarray:
    if value is None:
        raise ValueError(f"Dataset {field} is required by the sealed calibration")
    array = np.asarray(value, dtype=np.bool_)
    if array.ndim != 2:
        raise ValueError(f"Dataset {field} must be two-dimensional")
    return array


def _safe_fsum(values: list[float]) -> float | None:
    try:
        resolved = math.fsum(values)
    except OverflowError:
        return None
    return resolved if math.isfinite(resolved) else None


@dataclass(frozen=True, slots=True)
class SignedTakerFlowSymbolCalibration:
    """One symbol's immutable fixed-order training regression evidence."""

    symbol: str
    eligible_observations: int
    numerator: float | None
    denominator: float | None
    beta: float | None
    positive_slope: bool

    def __post_init__(self) -> None:
        _require_string(self.symbol, field="symbol")
        _require_int(
            self.eligible_observations,
            field="eligible_observations",
            minimum=0,
        )
        numerator = _require_float_or_none(self.numerator, field="numerator")
        denominator = _require_float_or_none(self.denominator, field="denominator")
        beta = _require_float_or_none(self.beta, field="beta")
        _require_bool(self.positive_slope, field="positive_slope")
        if denominator is not None and denominator < 0.0:
            raise ValueError("denominator must be non-negative when present")
        expected_positive = beta is not None and beta > 0.0
        if self.positive_slope is not expected_positive:
            raise ValueError("positive_slope does not match beta sign")
        if beta is not None and (
            numerator is None or denominator is None or denominator <= 0.0
        ):
            raise ValueError("beta requires finite numerator and positive denominator")

    def to_payload(self) -> dict[str, object]:
        return {
            "symbol": self.symbol,
            "eligible_observations": self.eligible_observations,
            "numerator": self.numerator,
            "denominator": self.denominator,
            "beta": self.beta,
            "positive_slope": self.positive_slope,
        }


@dataclass(frozen=True, slots=True)
class SignedTakerFlowCalibrationResult:
    """Strict content-addressed training-only signed-flow result."""

    protocol_digest: str
    dataset_id: str
    symbols: tuple[str, ...]
    symbol_results: tuple[SignedTakerFlowSymbolCalibration, ...]
    positive_slope_count: int
    status: str
    failures: tuple[str, ...]
    calibration_head: str | None = None
    source_manifest_digest: str | None = None
    protocol_seal_run_id: int = _PROTOCOL_SEAL_RUN_ID
    protocol_seal_artifact_id: int = _PROTOCOL_SEAL_ARTIFACT_ID
    protocol_seal_api_digest: str = _PROTOCOL_SEAL_API_DIGEST
    protocol_json_sha256: str = _PROTOCOL_JSON_SHA256
    protocol_fresh_artifact_id: int = _PROTOCOL_FRESH_ARTIFACT_ID
    protocol_fresh_api_digest: str = _PROTOCOL_FRESH_API_DIGEST
    implementation_head: str = _IMPLEMENTATION_HEAD
    training_relation_executed: bool = True
    evaluation_pnl_inspected: bool = False
    evaluation_execution_authorized: bool = False
    final_test_authorized: bool = False
    shared_cash_profitability_established: bool = False
    production_eligible: bool = False
    live_trading_authorized: bool = False
    schema_version: str = _SCHEMA_VERSION

    def __post_init__(self) -> None:
        protocol = canonical_signed_taker_flow_protocol()
        if self.schema_version != _SCHEMA_VERSION:
            raise ValueError("calibration schema_version is not canonical")
        if self.protocol_digest != protocol.digest:
            raise ValueError("protocol_digest is not canonical")
        _require_hex(self.protocol_digest, length=64, field="protocol_digest")
        _require_hex(self.dataset_id, length=64, field="dataset_id")
        if self.calibration_head is not None:
            _require_hex(self.calibration_head, length=40, field="calibration_head")
        if self.source_manifest_digest is not None:
            _require_hex(
                self.source_manifest_digest,
                length=64,
                field="source_manifest_digest",
            )
        if self.symbols != protocol.symbols:
            raise ValueError("calibration symbol roster is not canonical")
        if tuple(item.symbol for item in self.symbol_results) != self.symbols:
            raise ValueError("symbol calibration roster does not match symbols")
        _require_int(self.positive_slope_count, field="positive_slope_count")
        observed_positive = sum(item.positive_slope for item in self.symbol_results)
        if self.positive_slope_count != observed_positive:
            raise ValueError(
                "canonical positive_slope_count does not match symbol results"
            )
        if self.status not in {
            protocol.valid_status,
            protocol.reject_status,
            protocol.invalid_coverage_status,
        }:
            raise ValueError("calibration status is not canonical")
        if any(not isinstance(item, str) or not item for item in self.failures):
            raise ValueError("calibration failures must be non-empty strings")
        if self.failures and self.status != protocol.invalid_coverage_status:
            raise ValueError("calibration failures require INVALID_FLOW_COVERAGE")
        if not self.failures:
            expected_status = (
                protocol.valid_status
                if self.positive_slope_count >= protocol.required_positive_symbol_slopes
                else protocol.reject_status
            )
            if self.status != expected_status:
                raise ValueError(
                    "canonical calibration decision does not match beta signs"
                )

        fixed_authority = {
            "protocol_seal_run_id": _PROTOCOL_SEAL_RUN_ID,
            "protocol_seal_artifact_id": _PROTOCOL_SEAL_ARTIFACT_ID,
            "protocol_seal_api_digest": _PROTOCOL_SEAL_API_DIGEST,
            "protocol_json_sha256": _PROTOCOL_JSON_SHA256,
            "protocol_fresh_artifact_id": _PROTOCOL_FRESH_ARTIFACT_ID,
            "protocol_fresh_api_digest": _PROTOCOL_FRESH_API_DIGEST,
            "implementation_head": _IMPLEMENTATION_HEAD,
        }
        for field_name, expected in fixed_authority.items():
            if getattr(self, field_name) != expected:
                raise ValueError(f"{field_name} is not canonical")

        for field_name in (
            "training_relation_executed",
            "evaluation_pnl_inspected",
            "evaluation_execution_authorized",
            "final_test_authorized",
            "shared_cash_profitability_established",
            "production_eligible",
            "live_trading_authorized",
        ):
            _require_bool(getattr(self, field_name), field=field_name)
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
            raise ValueError("training-only calibration crossed a forbidden boundary")

    def to_payload(self) -> dict[str, object]:
        protocol = canonical_signed_taker_flow_protocol()
        return {
            "schema_version": self.schema_version,
            "protocol_digest": self.protocol_digest,
            "protocol_seal_run_id": self.protocol_seal_run_id,
            "protocol_seal_artifact_id": self.protocol_seal_artifact_id,
            "protocol_seal_api_digest": self.protocol_seal_api_digest,
            "protocol_json_sha256": self.protocol_json_sha256,
            "protocol_fresh_artifact_id": self.protocol_fresh_artifact_id,
            "protocol_fresh_api_digest": self.protocol_fresh_api_digest,
            "implementation_head": self.implementation_head,
            "calibration_head": self.calibration_head,
            "source_manifest_digest": self.source_manifest_digest,
            "dataset_id": self.dataset_id,
            "fit_start": _datetime_text(protocol.fit_start),
            "fit_cutoff": _datetime_text(protocol.fit_cutoff),
            "symbols": list(self.symbols),
            "symbol_results": [item.to_payload() for item in self.symbol_results],
            "positive_slope_count": self.positive_slope_count,
            "status": self.status,
            "failures": list(self.failures),
            "training_relation_executed": self.training_relation_executed,
            "evaluation_pnl_inspected": self.evaluation_pnl_inspected,
            "evaluation_execution_authorized": self.evaluation_execution_authorized,
            "final_test_authorized": self.final_test_authorized,
            "shared_cash_profitability_established": (
                self.shared_cash_profitability_established
            ),
            "production_eligible": self.production_eligible,
            "live_trading_authorized": self.live_trading_authorized,
        }

    @property
    def digest(self) -> str:
        return content_digest(self.to_payload())

    def to_artifact_payload(self) -> dict[str, object]:
        return {**self.to_payload(), "content_digest": self.digest}


def _validate_dataset(
    dataset: MarketDataset,
    protocol: SignedTakerFlowProtocol,
) -> tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if tuple(dataset.symbols) != protocol.symbols:
        raise ValueError("Dataset symbol roster does not match the sealed protocol")
    matches = [
        index
        for index, name in enumerate(dataset.feature_names)
        if name == protocol.feature_name
    ]
    if len(matches) != 1:
        raise ValueError("Dataset feature identity does not match the sealed protocol")
    feature_index = matches[0]

    timestamps = np.asarray(dataset.timestamps, dtype="datetime64[ns]")
    if timestamps.ndim != 1 or len(timestamps) != dataset.n_bars:
        raise ValueError("Dataset timestamps have invalid shape")
    timestamp_ns = timestamps.astype(np.int64)
    if timestamp_ns.size < protocol.label_endpoint_offset_bars + 2:
        raise ValueError("Dataset does not contain a usable calibration clock")
    if np.any(np.diff(timestamp_ns) != _ONE_HOUR_NS):
        raise ValueError("Dataset calibration clock must be contiguous native 1h bars")

    expected_shape = (dataset.n_bars, dataset.n_symbols)
    features = np.asarray(dataset.features, dtype=np.float32)
    available = np.asarray(dataset.feature_available, dtype=np.bool_)
    opens = np.asarray(dataset.open, dtype=np.float64)
    active = _required_bool_matrix(dataset.asset_active, field="asset_active")
    tradable = np.asarray(dataset.tradable, dtype=np.bool_)
    information = _required_bool_matrix(
        dataset.information_available,
        field="information_available",
    )
    if features.shape[:2] != expected_shape or available.shape != features.shape:
        raise ValueError("Dataset feature shape is invalid")
    for field_name, array in (
        ("open", opens),
        ("asset_active", active),
        ("tradable", tradable),
        ("information_available", information),
    ):
        if array.shape != expected_shape:
            raise ValueError(f"Dataset {field_name} shape is invalid")
    return (
        feature_index,
        timestamps,
        features,
        available,
        opens,
        active,
        tradable & information,
    )


def calibrate_signed_taker_flow(
    dataset: MarketDataset,
    protocol: SignedTakerFlowProtocol,
    *,
    calibration_head: str | None = None,
    source_manifest_digest: str | None = None,
) -> SignedTakerFlowCalibrationResult:
    """Evaluate only the frozen pre-2023 no-intercept sign calibration."""

    canonical = canonical_signed_taker_flow_protocol()
    if protocol != canonical or protocol.digest != canonical.digest:
        raise ValueError(
            "protocol differs from the sealed signed taker-flow preregistration"
        )
    if calibration_head is not None:
        _require_hex(calibration_head, length=40, field="calibration_head")
    if source_manifest_digest is not None:
        _require_hex(
            source_manifest_digest,
            length=64,
            field="source_manifest_digest",
        )

    (
        feature_index,
        timestamps,
        features,
        feature_available,
        open_price,
        active,
        observable,
    ) = _validate_dataset(dataset, protocol)
    information = _required_bool_matrix(
        dataset.information_available,
        field="information_available",
    )
    tradable = np.asarray(dataset.tradable, dtype=np.bool_)
    fit_start = _datetime64(protocol.fit_start)
    fit_cutoff = _datetime64(protocol.fit_cutoff)
    endpoint_offset = protocol.label_endpoint_offset_bars
    execution_offset = protocol.label_execution_offset_bars

    symbol_results: list[SignedTakerFlowSymbolCalibration] = []
    failures: list[str] = []
    for symbol_index, symbol in enumerate(protocol.symbols):
        signals: list[float] = []
        labels: list[float] = []
        max_t = dataset.n_bars - endpoint_offset
        for t in range(max_t):
            if timestamps[t] < fit_start:
                continue
            endpoint = t + endpoint_offset
            if timestamps[endpoint] >= fit_cutoff:
                continue
            if not feature_available[t, symbol_index, feature_index]:
                continue
            if not (
                active[t, symbol_index]
                and tradable[t, symbol_index]
                and information[t, symbol_index]
            ):
                continue
            label_window = slice(t + execution_offset, endpoint + 1)
            if not np.all(active[label_window, symbol_index]):
                continue
            if not np.all(observable[label_window, symbol_index]):
                continue

            signal = float(features[t, symbol_index, feature_index])
            if not math.isfinite(signal):
                raise ValueError("available signed taker-flow feature must be finite")
            if (
                not protocol.feature_lower_bound
                <= signal
                <= protocol.feature_upper_bound
            ):
                raise ValueError(
                    "available signed taker-flow feature violates [-1, 1] bound"
                )
            start_open = float(open_price[t + execution_offset, symbol_index])
            end_open = float(open_price[endpoint, symbol_index])
            if (
                not math.isfinite(start_open)
                or not math.isfinite(end_open)
                or start_open <= 0.0
                or end_open <= 0.0
            ):
                continue
            ratio = end_open / start_open
            if not math.isfinite(ratio) or ratio <= 0.0:
                continue
            label = math.log(ratio)
            if not math.isfinite(label):
                continue
            signals.append(signal)
            labels.append(label)

        eligible = len(signals)
        numerator = _safe_fsum(
            [signal * label for signal, label in zip(signals, labels, strict=True)]
        )
        denominator = _safe_fsum([signal * signal for signal in signals])
        beta: float | None = None
        symbol_failures: list[str] = []
        if eligible < protocol.minimum_eligible_observations_per_symbol:
            symbol_failures.append(
                f"{symbol}:eligible_observations<{protocol.minimum_eligible_observations_per_symbol}"
            )
        if numerator is None:
            symbol_failures.append(f"{symbol}:numerator_not_finite")
        if denominator is None or denominator <= 0.0:
            symbol_failures.append(f"{symbol}:denominator_not_finite_positive")
        if not symbol_failures:
            assert numerator is not None
            assert denominator is not None
            beta = numerator / denominator
            if not math.isfinite(beta):
                beta = None
                symbol_failures.append(f"{symbol}:beta_not_finite")
        failures.extend(symbol_failures)
        symbol_results.append(
            SignedTakerFlowSymbolCalibration(
                symbol=symbol,
                eligible_observations=eligible,
                numerator=numerator,
                denominator=denominator,
                beta=beta,
                positive_slope=beta is not None and beta > 0.0,
            )
        )

    positive_count = sum(item.positive_slope for item in symbol_results)
    if failures:
        status = protocol.invalid_coverage_status
    elif positive_count >= protocol.required_positive_symbol_slopes:
        status = protocol.valid_status
    else:
        status = protocol.reject_status

    return SignedTakerFlowCalibrationResult(
        protocol_digest=protocol.digest,
        calibration_head=calibration_head,
        source_manifest_digest=source_manifest_digest,
        dataset_id=dataset.dataset_id,
        symbols=protocol.symbols,
        symbol_results=tuple(symbol_results),
        positive_slope_count=int(positive_count),
        status=status,
        failures=tuple(failures),
    )


def _symbol_result_from_payload(raw: object) -> SignedTakerFlowSymbolCalibration:
    if not isinstance(raw, dict) or any(not isinstance(key, str) for key in raw):
        raise ValueError("signed taker-flow symbol result is malformed")
    expected = {
        "symbol",
        "eligible_observations",
        "numerator",
        "denominator",
        "beta",
        "positive_slope",
    }
    if set(raw) != expected:
        raise ValueError(
            "signed taker-flow symbol result has unknown or missing fields"
        )
    return SignedTakerFlowSymbolCalibration(
        symbol=_require_string(raw["symbol"], field="symbol"),
        eligible_observations=_require_int(
            raw["eligible_observations"],
            field="eligible_observations",
        ),
        numerator=_require_float_or_none(raw["numerator"], field="numerator"),
        denominator=_require_float_or_none(raw["denominator"], field="denominator"),
        beta=_require_float_or_none(raw["beta"], field="beta"),
        positive_slope=_require_bool(raw["positive_slope"], field="positive_slope"),
    )


def load_signed_taker_flow_calibration_result(
    path: str | Path,
) -> SignedTakerFlowCalibrationResult:
    """Load a strict canonical result artifact and verify its content digest."""

    try:
        raw: Any = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("signed taker-flow calibration result is malformed") from error
    if not isinstance(raw, dict) or any(not isinstance(key, str) for key in raw):
        raise ValueError("signed taker-flow calibration result must be a JSON object")

    expected = {
        "schema_version",
        "protocol_digest",
        "protocol_seal_run_id",
        "protocol_seal_artifact_id",
        "protocol_seal_api_digest",
        "protocol_json_sha256",
        "protocol_fresh_artifact_id",
        "protocol_fresh_api_digest",
        "implementation_head",
        "calibration_head",
        "source_manifest_digest",
        "dataset_id",
        "fit_start",
        "fit_cutoff",
        "symbols",
        "symbol_results",
        "positive_slope_count",
        "status",
        "failures",
        "training_relation_executed",
        "evaluation_pnl_inspected",
        "evaluation_execution_authorized",
        "final_test_authorized",
        "shared_cash_profitability_established",
        "production_eligible",
        "live_trading_authorized",
        "content_digest",
    }
    if set(raw) != expected:
        raise ValueError(
            "signed taker-flow calibration result has unknown or missing fields"
        )

    protocol = canonical_signed_taker_flow_protocol()
    if raw["fit_start"] != _datetime_text(protocol.fit_start):
        raise ValueError("fit_start is not canonical")
    if raw["fit_cutoff"] != _datetime_text(protocol.fit_cutoff):
        raise ValueError("fit_cutoff is not canonical")
    symbols_raw = raw["symbols"]
    if not isinstance(symbols_raw, list) or any(
        not isinstance(item, str) for item in symbols_raw
    ):
        raise ValueError("symbols must be a string array")
    results_raw = raw["symbol_results"]
    if not isinstance(results_raw, list):
        raise ValueError("symbol_results must be an array")
    failures_raw = raw["failures"]
    if not isinstance(failures_raw, list) or any(
        not isinstance(item, str) or not item for item in failures_raw
    ):
        raise ValueError("failures must be a string array")

    calibration_head_raw = raw["calibration_head"]
    calibration_head = (
        None
        if calibration_head_raw is None
        else _require_hex(calibration_head_raw, length=40, field="calibration_head")
    )
    source_manifest_raw = raw["source_manifest_digest"]
    source_manifest_digest = (
        None
        if source_manifest_raw is None
        else _require_hex(
            source_manifest_raw,
            length=64,
            field="source_manifest_digest",
        )
    )

    result = SignedTakerFlowCalibrationResult(
        schema_version=_require_string(raw["schema_version"], field="schema_version"),
        protocol_digest=_require_string(
            raw["protocol_digest"], field="protocol_digest"
        ),
        protocol_seal_run_id=_require_int(
            raw["protocol_seal_run_id"], field="protocol_seal_run_id"
        ),
        protocol_seal_artifact_id=_require_int(
            raw["protocol_seal_artifact_id"], field="protocol_seal_artifact_id"
        ),
        protocol_seal_api_digest=_require_string(
            raw["protocol_seal_api_digest"], field="protocol_seal_api_digest"
        ),
        protocol_json_sha256=_require_string(
            raw["protocol_json_sha256"], field="protocol_json_sha256"
        ),
        protocol_fresh_artifact_id=_require_int(
            raw["protocol_fresh_artifact_id"], field="protocol_fresh_artifact_id"
        ),
        protocol_fresh_api_digest=_require_string(
            raw["protocol_fresh_api_digest"], field="protocol_fresh_api_digest"
        ),
        implementation_head=_require_string(
            raw["implementation_head"], field="implementation_head"
        ),
        calibration_head=calibration_head,
        source_manifest_digest=source_manifest_digest,
        dataset_id=_require_string(raw["dataset_id"], field="dataset_id"),
        symbols=tuple(symbols_raw),
        symbol_results=tuple(_symbol_result_from_payload(item) for item in results_raw),
        positive_slope_count=_require_int(
            raw["positive_slope_count"], field="positive_slope_count"
        ),
        status=_require_string(raw["status"], field="status"),
        failures=tuple(failures_raw),
        training_relation_executed=_require_bool(
            raw["training_relation_executed"], field="training_relation_executed"
        ),
        evaluation_pnl_inspected=_require_bool(
            raw["evaluation_pnl_inspected"], field="evaluation_pnl_inspected"
        ),
        evaluation_execution_authorized=_require_bool(
            raw["evaluation_execution_authorized"],
            field="evaluation_execution_authorized",
        ),
        final_test_authorized=_require_bool(
            raw["final_test_authorized"], field="final_test_authorized"
        ),
        shared_cash_profitability_established=_require_bool(
            raw["shared_cash_profitability_established"],
            field="shared_cash_profitability_established",
        ),
        production_eligible=_require_bool(
            raw["production_eligible"], field="production_eligible"
        ),
        live_trading_authorized=_require_bool(
            raw["live_trading_authorized"], field="live_trading_authorized"
        ),
    )
    expected_digest = _require_hex(
        raw["content_digest"], length=64, field="content_digest"
    )
    if result.digest != expected_digest:
        raise ValueError("signed taker-flow calibration content digest mismatch")
    return result


__all__ = [
    "SignedTakerFlowCalibrationResult",
    "SignedTakerFlowSymbolCalibration",
    "calibrate_signed_taker_flow",
    "load_signed_taker_flow_calibration_result",
]
