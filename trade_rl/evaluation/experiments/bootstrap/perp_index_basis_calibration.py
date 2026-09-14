"""Deterministic training-only calibration for sealed perpetual-index basis."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from trade_rl.artifacts.canonical import canonical_json_bytes
from trade_rl.artifacts.hashing import content_digest
from trade_rl.data.market import MarketDataset
from trade_rl.evaluation.experiments.bootstrap.perp_index_basis_prereg import (
    PerpIndexBasisProtocol,
    canonical_perp_index_basis_protocol,
)

_SCHEMA_VERSION = "perp_index_basis_calibration_v1"
_PREREG_HEAD = "795b2e9efc2e7d9d65a7131a160c7da16ba02043"
_PREREG_SEAL_RUN_ID = 34867687158
_PREREG_SEAL_ARTIFACT_ID = 10357004799
_PREREG_SEAL_API_DIGEST = (
    "sha256:e4b156b66d273bebe012b664ed0f72ef551d46cb48ca0a8fddb04ba5ff61c96b"
)
_PREREG_FRESH_ARTIFACT_ID = 10357836707
_PREREG_FRESH_API_DIGEST = (
    "sha256:ff4ce128b2b2103c309d94f2731505bd2c5e0029df99c47d9783b6e566c548f4"
)
_SOURCE_PREFLIGHT_RUN_ID = 34868358617
_SOURCE_PREFLIGHT_ARTIFACT_ID = 10358261087
_SOURCE_PREFLIGHT_ARTIFACT_API_DIGEST = (
    "sha256:63eaa8135aa42330809a5dceb00905e141a18594d6a61edc699e7a6e02d2dae6"
)
_SOURCE_PREFLIGHT_FRESH_ARTIFACT_ID = 10357523519
_SOURCE_PREFLIGHT_FRESH_API_DIGEST = (
    "sha256:f1399532a3a90b5c75e70cd06da931942a7c8b0370f7eba06b62916d3d4f16bf"
)
_SOURCE_PREFLIGHT_REPORT_SHA256 = (
    "9a8b4d88c48885eb2aba8becb1ae347c6a1c20774dfc9d47281fa042b065f7bf"
)
_SOURCE_PREFLIGHT_CONTENT_DIGEST = (
    "ccb22002ac28a0f2a1275e4c31fb5cd0cde59b72d14017e15d1f88141dcf0e65"
)
_SOURCE_IMPLEMENTATION_HEAD = "4911579874111bb6620481be8a4dbd7a8e664918"
_SOURCE_IMPLEMENTATION_CI_RUN_ID = 34871818525
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
class PerpIndexBasisSymbolCalibration:
    symbol: str
    eligible_observations: int
    numerator: float | None
    denominator: float | None
    beta: float | None
    negative_slope: bool

    def __post_init__(self) -> None:
        _require_string(self.symbol, field="symbol")
        _require_int(self.eligible_observations, field="eligible_observations")
        numerator = _require_float_or_none(self.numerator, field="numerator")
        denominator = _require_float_or_none(self.denominator, field="denominator")
        beta = _require_float_or_none(self.beta, field="beta")
        _require_bool(self.negative_slope, field="negative_slope")
        if denominator is not None and denominator < 0.0:
            raise ValueError("denominator must be non-negative when present")
        if self.negative_slope is not (beta is not None and beta < 0.0):
            raise ValueError("negative_slope does not match beta sign")
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
            "negative_slope": self.negative_slope,
        }


@dataclass(frozen=True, slots=True)
class PerpIndexBasisCalibrationResult:
    protocol_digest: str
    dataset_id: str
    symbols: tuple[str, ...]
    symbol_results: tuple[PerpIndexBasisSymbolCalibration, ...]
    negative_slope_count: int
    status: str
    failures: tuple[str, ...]
    calibration_head: str | None = None
    source_manifest_digest: str | None = None
    prereg_head: str = _PREREG_HEAD
    prereg_seal_run_id: int = _PREREG_SEAL_RUN_ID
    prereg_seal_artifact_id: int = _PREREG_SEAL_ARTIFACT_ID
    prereg_seal_api_digest: str = _PREREG_SEAL_API_DIGEST
    prereg_fresh_artifact_id: int = _PREREG_FRESH_ARTIFACT_ID
    prereg_fresh_api_digest: str = _PREREG_FRESH_API_DIGEST
    source_preflight_run_id: int = _SOURCE_PREFLIGHT_RUN_ID
    source_preflight_artifact_id: int = _SOURCE_PREFLIGHT_ARTIFACT_ID
    source_preflight_artifact_api_digest: str = _SOURCE_PREFLIGHT_ARTIFACT_API_DIGEST
    source_preflight_fresh_artifact_id: int = _SOURCE_PREFLIGHT_FRESH_ARTIFACT_ID
    source_preflight_fresh_api_digest: str = _SOURCE_PREFLIGHT_FRESH_API_DIGEST
    source_preflight_report_sha256: str = _SOURCE_PREFLIGHT_REPORT_SHA256
    source_preflight_content_digest: str = _SOURCE_PREFLIGHT_CONTENT_DIGEST
    source_implementation_head: str = _SOURCE_IMPLEMENTATION_HEAD
    source_implementation_ci_run_id: int = _SOURCE_IMPLEMENTATION_CI_RUN_ID
    training_relation_executed: bool = True
    evaluation_pnl_inspected: bool = False
    evaluation_execution_authorized: bool = False
    final_test_authorized: bool = False
    shared_cash_profitability_established: bool = False
    production_eligible: bool = False
    live_trading_authorized: bool = False
    schema_version: str = _SCHEMA_VERSION

    def __post_init__(self) -> None:
        protocol = canonical_perp_index_basis_protocol()
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
        _require_int(self.negative_slope_count, field="negative_slope_count")
        observed = sum(item.negative_slope for item in self.symbol_results)
        if self.negative_slope_count != observed:
            raise ValueError("negative_slope_count does not match symbol results")
        if self.status not in {
            protocol.valid_status,
            protocol.reject_status,
            protocol.invalid_coverage_status,
        }:
            raise ValueError("calibration status is not canonical")
        if any(not isinstance(item, str) or not item for item in self.failures):
            raise ValueError("calibration failures must be non-empty strings")
        if self.failures and self.status != protocol.invalid_coverage_status:
            raise ValueError("calibration failures require INVALID_BASIS_COVERAGE")
        if not self.failures:
            expected_status = (
                protocol.valid_status
                if self.negative_slope_count >= protocol.required_negative_symbol_slopes
                else protocol.reject_status
            )
            if self.status != expected_status:
                raise ValueError(
                    "calibration decision does not match frozen slope gate"
                )

        fixed_authority = {
            "prereg_head": _PREREG_HEAD,
            "prereg_seal_run_id": _PREREG_SEAL_RUN_ID,
            "prereg_seal_artifact_id": _PREREG_SEAL_ARTIFACT_ID,
            "prereg_seal_api_digest": _PREREG_SEAL_API_DIGEST,
            "prereg_fresh_artifact_id": _PREREG_FRESH_ARTIFACT_ID,
            "prereg_fresh_api_digest": _PREREG_FRESH_API_DIGEST,
            "source_preflight_run_id": _SOURCE_PREFLIGHT_RUN_ID,
            "source_preflight_artifact_id": _SOURCE_PREFLIGHT_ARTIFACT_ID,
            "source_preflight_artifact_api_digest": _SOURCE_PREFLIGHT_ARTIFACT_API_DIGEST,
            "source_preflight_fresh_artifact_id": _SOURCE_PREFLIGHT_FRESH_ARTIFACT_ID,
            "source_preflight_fresh_api_digest": _SOURCE_PREFLIGHT_FRESH_API_DIGEST,
            "source_preflight_report_sha256": _SOURCE_PREFLIGHT_REPORT_SHA256,
            "source_preflight_content_digest": _SOURCE_PREFLIGHT_CONTENT_DIGEST,
            "source_implementation_head": _SOURCE_IMPLEMENTATION_HEAD,
            "source_implementation_ci_run_id": _SOURCE_IMPLEMENTATION_CI_RUN_ID,
        }
        for field_name, expected in fixed_authority.items():
            if getattr(self, field_name) != expected:
                raise ValueError(f"{field_name} is not canonical authority")

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
        protocol = canonical_perp_index_basis_protocol()
        return {
            "schema_version": self.schema_version,
            "protocol_digest": self.protocol_digest,
            "prereg_head": self.prereg_head,
            "prereg_seal_run_id": self.prereg_seal_run_id,
            "prereg_seal_artifact_id": self.prereg_seal_artifact_id,
            "prereg_seal_api_digest": self.prereg_seal_api_digest,
            "prereg_fresh_artifact_id": self.prereg_fresh_artifact_id,
            "prereg_fresh_api_digest": self.prereg_fresh_api_digest,
            "source_preflight_run_id": self.source_preflight_run_id,
            "source_preflight_artifact_id": self.source_preflight_artifact_id,
            "source_preflight_artifact_api_digest": self.source_preflight_artifact_api_digest,
            "source_preflight_fresh_artifact_id": self.source_preflight_fresh_artifact_id,
            "source_preflight_fresh_api_digest": self.source_preflight_fresh_api_digest,
            "source_preflight_report_sha256": self.source_preflight_report_sha256,
            "source_preflight_content_digest": self.source_preflight_content_digest,
            "source_implementation_head": self.source_implementation_head,
            "source_implementation_ci_run_id": self.source_implementation_ci_run_id,
            "calibration_head": self.calibration_head,
            "source_manifest_digest": self.source_manifest_digest,
            "dataset_id": self.dataset_id,
            "fit_start": _datetime_text(protocol.fit_start),
            "fit_cutoff": _datetime_text(protocol.fit_cutoff),
            "symbols": list(self.symbols),
            "symbol_results": [item.to_payload() for item in self.symbol_results],
            "negative_slope_count": self.negative_slope_count,
            "status": self.status,
            "failures": list(self.failures),
            "training_relation_executed": self.training_relation_executed,
            "evaluation_pnl_inspected": self.evaluation_pnl_inspected,
            "evaluation_execution_authorized": self.evaluation_execution_authorized,
            "final_test_authorized": self.final_test_authorized,
            "shared_cash_profitability_established": self.shared_cash_profitability_established,
            "production_eligible": self.production_eligible,
            "live_trading_authorized": self.live_trading_authorized,
        }

    @property
    def digest(self) -> str:
        return content_digest(self.to_payload())

    def _require_publication_authority(self) -> None:
        if self.calibration_head is None:
            raise ValueError("calibration_head is required for artifact publication")
        if self.source_manifest_digest is None:
            raise ValueError(
                "source_manifest_digest is required for artifact publication"
            )

    def to_artifact_payload(self) -> dict[str, object]:
        self._require_publication_authority()
        return {**self.to_payload(), "content_digest": self.digest}


def _validate_dataset(
    dataset: MarketDataset,
    protocol: PerpIndexBasisProtocol,
) -> tuple[
    int,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
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
        dataset.information_available, field="information_available"
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
        tradable,
        information,
    )


def calibrate_perp_index_basis(
    dataset: MarketDataset,
    protocol: PerpIndexBasisProtocol,
    *,
    calibration_head: str | None = None,
    source_manifest_digest: str | None = None,
) -> PerpIndexBasisCalibrationResult:
    canonical = canonical_perp_index_basis_protocol()
    if protocol != canonical or protocol.digest != canonical.digest:
        raise ValueError(
            "protocol differs from sealed perpetual-index basis preregistration"
        )
    if calibration_head is not None:
        _require_hex(calibration_head, length=40, field="calibration_head")
    if source_manifest_digest is not None:
        _require_hex(source_manifest_digest, length=64, field="source_manifest_digest")

    (
        feature_index,
        timestamps,
        features,
        feature_available,
        opens,
        active,
        tradable,
        information,
    ) = _validate_dataset(dataset, protocol)
    fit_start = _datetime64(protocol.fit_start)
    fit_cutoff = _datetime64(protocol.fit_cutoff)
    execution_offset = protocol.label_execution_offset_bars
    endpoint_offset = protocol.label_endpoint_offset_bars

    symbol_results: list[PerpIndexBasisSymbolCalibration] = []
    failures: list[str] = []
    for symbol_index, symbol in enumerate(protocol.symbols):
        signals: list[float] = []
        labels: list[float] = []
        for t in range(dataset.n_bars - endpoint_offset):
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
            # Raw row_present is intentionally not persisted on MarketDataset.
            # The canonical builder folds it fail-closed into both tradable and
            # information_available, so requiring those masks preserves row presence.
            label_window = slice(t + execution_offset, endpoint + 1)
            if not np.all(active[label_window, symbol_index]):
                continue
            if not np.all(tradable[label_window, symbol_index]):
                continue
            if not np.all(information[label_window, symbol_index]):
                continue

            signal = float(features[t, symbol_index, feature_index])
            if not math.isfinite(signal):
                raise ValueError(
                    "available perpetual-index basis feature must be finite"
                )
            start_open = float(opens[t + execution_offset, symbol_index])
            end_open = float(opens[endpoint, symbol_index])
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
            PerpIndexBasisSymbolCalibration(
                symbol=symbol,
                eligible_observations=eligible,
                numerator=numerator,
                denominator=denominator,
                beta=beta,
                negative_slope=beta is not None and beta < 0.0,
            )
        )

    negative_count = sum(item.negative_slope for item in symbol_results)
    if failures:
        status = protocol.invalid_coverage_status
    elif negative_count >= protocol.required_negative_symbol_slopes:
        status = protocol.valid_status
    else:
        status = protocol.reject_status
    return PerpIndexBasisCalibrationResult(
        protocol_digest=protocol.digest,
        calibration_head=calibration_head,
        source_manifest_digest=source_manifest_digest,
        dataset_id=dataset.dataset_id,
        symbols=protocol.symbols,
        symbol_results=tuple(symbol_results),
        negative_slope_count=int(negative_count),
        status=status,
        failures=tuple(failures),
    )


def _symbol_result_from_payload(raw: object) -> PerpIndexBasisSymbolCalibration:
    if not isinstance(raw, dict) or any(not isinstance(key, str) for key in raw):
        raise ValueError("perpetual-index symbol result is malformed")
    expected = {
        "symbol",
        "eligible_observations",
        "numerator",
        "denominator",
        "beta",
        "negative_slope",
    }
    if set(raw) != expected:
        raise ValueError("perpetual-index symbol result has unknown or missing fields")
    return PerpIndexBasisSymbolCalibration(
        symbol=_require_string(raw["symbol"], field="symbol"),
        eligible_observations=_require_int(
            raw["eligible_observations"], field="eligible_observations"
        ),
        numerator=_require_float_or_none(raw["numerator"], field="numerator"),
        denominator=_require_float_or_none(raw["denominator"], field="denominator"),
        beta=_require_float_or_none(raw["beta"], field="beta"),
        negative_slope=_require_bool(raw["negative_slope"], field="negative_slope"),
    )


def load_perp_index_basis_calibration_result(
    path: str | Path,
) -> PerpIndexBasisCalibrationResult:
    source = Path(path)
    try:
        raw_bytes = source.read_bytes()
        raw_text = raw_bytes.decode("utf-8")
        raw: Any = json.loads(raw_text)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(
            "perpetual-index basis calibration result is malformed"
        ) from error
    if not isinstance(raw, dict) or any(not isinstance(key, str) for key in raw):
        raise ValueError(
            "perpetual-index basis calibration result must be a JSON object"
        )
    try:
        canonical_bytes = canonical_json_bytes(raw)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "perpetual-index basis calibration result is malformed"
        ) from error
    if raw_bytes != canonical_bytes:
        raise ValueError(
            "perpetual-index basis calibration result must use canonical JSON bytes"
        )

    expected = {
        "schema_version",
        "protocol_digest",
        "prereg_head",
        "prereg_seal_run_id",
        "prereg_seal_artifact_id",
        "prereg_seal_api_digest",
        "prereg_fresh_artifact_id",
        "prereg_fresh_api_digest",
        "source_preflight_run_id",
        "source_preflight_artifact_id",
        "source_preflight_artifact_api_digest",
        "source_preflight_fresh_artifact_id",
        "source_preflight_fresh_api_digest",
        "source_preflight_report_sha256",
        "source_preflight_content_digest",
        "source_implementation_head",
        "source_implementation_ci_run_id",
        "calibration_head",
        "source_manifest_digest",
        "dataset_id",
        "fit_start",
        "fit_cutoff",
        "symbols",
        "symbol_results",
        "negative_slope_count",
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
            "perpetual-index basis calibration result has unknown or missing keys"
        )
    protocol = canonical_perp_index_basis_protocol()
    if raw["fit_start"] != _datetime_text(protocol.fit_start) or raw[
        "fit_cutoff"
    ] != _datetime_text(protocol.fit_cutoff):
        raise ValueError("calibration fit clock is not canonical")
    symbols_raw = raw["symbols"]
    results_raw = raw["symbol_results"]
    failures_raw = raw["failures"]
    if not isinstance(symbols_raw, list) or any(
        not isinstance(item, str) for item in symbols_raw
    ):
        raise ValueError("symbols must be a string array")
    if not isinstance(results_raw, list):
        raise ValueError("symbol_results must be an array")
    if not isinstance(failures_raw, list) or any(
        not isinstance(item, str) or not item for item in failures_raw
    ):
        raise ValueError("failures must be a string array")

    result = PerpIndexBasisCalibrationResult(
        schema_version=_require_string(raw["schema_version"], field="schema_version"),
        protocol_digest=_require_string(
            raw["protocol_digest"], field="protocol_digest"
        ),
        prereg_head=_require_string(raw["prereg_head"], field="prereg_head"),
        prereg_seal_run_id=_require_int(
            raw["prereg_seal_run_id"], field="prereg_seal_run_id"
        ),
        prereg_seal_artifact_id=_require_int(
            raw["prereg_seal_artifact_id"], field="prereg_seal_artifact_id"
        ),
        prereg_seal_api_digest=_require_string(
            raw["prereg_seal_api_digest"], field="prereg_seal_api_digest"
        ),
        prereg_fresh_artifact_id=_require_int(
            raw["prereg_fresh_artifact_id"], field="prereg_fresh_artifact_id"
        ),
        prereg_fresh_api_digest=_require_string(
            raw["prereg_fresh_api_digest"], field="prereg_fresh_api_digest"
        ),
        source_preflight_run_id=_require_int(
            raw["source_preflight_run_id"], field="source_preflight_run_id"
        ),
        source_preflight_artifact_id=_require_int(
            raw["source_preflight_artifact_id"], field="source_preflight_artifact_id"
        ),
        source_preflight_artifact_api_digest=_require_string(
            raw["source_preflight_artifact_api_digest"],
            field="source_preflight_artifact_api_digest",
        ),
        source_preflight_fresh_artifact_id=_require_int(
            raw["source_preflight_fresh_artifact_id"],
            field="source_preflight_fresh_artifact_id",
        ),
        source_preflight_fresh_api_digest=_require_string(
            raw["source_preflight_fresh_api_digest"],
            field="source_preflight_fresh_api_digest",
        ),
        source_preflight_report_sha256=_require_string(
            raw["source_preflight_report_sha256"],
            field="source_preflight_report_sha256",
        ),
        source_preflight_content_digest=_require_string(
            raw["source_preflight_content_digest"],
            field="source_preflight_content_digest",
        ),
        source_implementation_head=_require_string(
            raw["source_implementation_head"], field="source_implementation_head"
        ),
        source_implementation_ci_run_id=_require_int(
            raw["source_implementation_ci_run_id"],
            field="source_implementation_ci_run_id",
        ),
        calibration_head=_require_hex(
            raw["calibration_head"], length=40, field="calibration_head"
        ),
        source_manifest_digest=_require_hex(
            raw["source_manifest_digest"], length=64, field="source_manifest_digest"
        ),
        dataset_id=_require_string(raw["dataset_id"], field="dataset_id"),
        symbols=tuple(symbols_raw),
        symbol_results=tuple(_symbol_result_from_payload(item) for item in results_raw),
        negative_slope_count=_require_int(
            raw["negative_slope_count"], field="negative_slope_count"
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
        raise ValueError("perpetual-index basis calibration content digest mismatch")
    return result


__all__ = [
    "PerpIndexBasisCalibrationResult",
    "PerpIndexBasisSymbolCalibration",
    "calibrate_perp_index_basis",
    "load_perp_index_basis_calibration_result",
]
