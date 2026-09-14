"""Deterministic training-only calibration for the sealed mean-reversion gate."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime

import numpy as np

from trade_rl.artifacts.hashing import content_digest
from trade_rl.data.market import MarketDataset
from trade_rl.evaluation.experiments.bootstrap.mean_reversion_economic_gate_prereg import (
    MeanReversionEconomicGateProtocol,
    canonical_mean_reversion_economic_gate_protocol,
)

_SCHEMA_VERSION = "mean_reversion_economic_gate_calibration_v1"
_VALID_STATUS = "VALID_CALIBRATION"


def _datetime64(value: datetime) -> np.datetime64:
    return np.datetime64(value.replace(tzinfo=None), "ns")


def _readonly_bool(value: np.ndarray | None, *, field: str) -> np.ndarray:
    if value is None:
        raise ValueError(f"{field} is required by the sealed calibration protocol")
    array = np.asarray(value, dtype=np.bool_)
    if array.ndim != 2:
        raise ValueError(f"{field} must be a two-dimensional array")
    return array


def _readonly_float(value: np.ndarray | None, *, field: str) -> np.ndarray:
    if value is None:
        raise ValueError(f"{field} is required by the sealed calibration protocol")
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"{field} must be a two-dimensional array")
    return array


def _finite_cost(value: float, *, expected: float, field: str) -> float:
    resolved = float(value)
    if not math.isfinite(resolved) or resolved < 0.0:
        raise ValueError(f"execution cost {field} must be finite and non-negative")
    if resolved != expected:
        raise ValueError(f"execution cost {field} drifted from the preregistered value")
    return resolved


@dataclass(frozen=True, slots=True)
class MeanReversionEconomicGateSymbolCalibration:
    """One symbol's fixed-order training-only regression evidence."""

    symbol: str
    eligible_observations: int
    numerator: float
    denominator: float
    beta: float | None
    fee_rate_min: float | None
    fee_rate_max: float | None
    taker_fee_rate_min: float | None
    taker_fee_rate_max: float | None
    spread_rate_min: float | None
    spread_rate_max: float | None

    def to_payload(self) -> dict[str, object]:
        return {
            "symbol": self.symbol,
            "eligible_observations": self.eligible_observations,
            "numerator": self.numerator,
            "denominator": self.denominator,
            "beta": self.beta,
            "fee_rate_min": self.fee_rate_min,
            "fee_rate_max": self.fee_rate_max,
            "taker_fee_rate_min": self.taker_fee_rate_min,
            "taker_fee_rate_max": self.taker_fee_rate_max,
            "spread_rate_min": self.spread_rate_min,
            "spread_rate_max": self.spread_rate_max,
        }


@dataclass(frozen=True, slots=True)
class MeanReversionEconomicGateCalibration:
    """Content-addressable output of the sealed training-only calibration."""

    protocol_digest: str
    dataset_id: str
    symbols: tuple[str, ...]
    symbol_results: tuple[MeanReversionEconomicGateSymbolCalibration, ...]
    negative_slope_count: int
    beta_gate: float | None
    status: str
    evaluation_pnl_inspected: bool = False
    strategy_execution_performed: bool = False
    production_eligible: bool = False
    evaluation_execution_authorized: bool = False
    schema_version: str = _SCHEMA_VERSION

    def __post_init__(self) -> None:
        if tuple(item.symbol for item in self.symbol_results) != self.symbols:
            raise ValueError("symbol calibration roster does not match symbols")
        if self.negative_slope_count < 0 or self.negative_slope_count > len(
            self.symbols
        ):
            raise ValueError("negative_slope_count is invalid")
        if self.beta_gate is not None and (
            not math.isfinite(self.beta_gate) or self.beta_gate >= 0.0
        ):
            raise ValueError("beta_gate must be finite and strictly negative")
        if type(self.evaluation_pnl_inspected) is not bool:
            raise ValueError("evaluation_pnl_inspected must be boolean")
        if type(self.strategy_execution_performed) is not bool:
            raise ValueError("strategy_execution_performed must be boolean")
        if type(self.production_eligible) is not bool:
            raise ValueError("production_eligible must be boolean")
        if type(self.evaluation_execution_authorized) is not bool:
            raise ValueError("evaluation_execution_authorized must be boolean")
        if any(
            (
                self.evaluation_pnl_inspected,
                self.strategy_execution_performed,
                self.production_eligible,
                self.evaluation_execution_authorized,
            )
        ):
            raise ValueError("training-only calibration crossed a forbidden boundary")

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "protocol_digest": self.protocol_digest,
            "dataset_id": self.dataset_id,
            "symbols": list(self.symbols),
            "symbol_results": [item.to_payload() for item in self.symbol_results],
            "negative_slope_count": self.negative_slope_count,
            "beta_gate": self.beta_gate,
            "status": self.status,
            "evaluation_pnl_inspected": self.evaluation_pnl_inspected,
            "strategy_execution_performed": self.strategy_execution_performed,
            "production_eligible": self.production_eligible,
            "evaluation_execution_authorized": self.evaluation_execution_authorized,
        }

    @property
    def digest(self) -> str:
        return content_digest(self.to_payload())


def _cost_range(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    return min(values), max(values)


def calibrate_mean_reversion_economic_gate(
    dataset: MarketDataset,
    protocol: MeanReversionEconomicGateProtocol,
) -> MeanReversionEconomicGateCalibration:
    """Calibrate the sealed gate using pre-cutoff Dataset evidence only."""

    canonical = canonical_mean_reversion_economic_gate_protocol()
    if protocol != canonical or protocol.digest != canonical.digest:
        raise ValueError("protocol differs from the sealed preregistration")
    if dataset.dataset_id != protocol.successor_dataset_id:
        raise ValueError(
            "dataset_id does not match the preregistered successor Dataset"
        )
    if tuple(dataset.symbols) != protocol.symbols:
        raise ValueError("dataset symbol roster does not match the preregistration")
    if not 0 <= protocol.signal_index < len(dataset.feature_names):
        raise ValueError("signal index is outside the Dataset feature roster")
    if dataset.feature_names[protocol.signal_index] != protocol.signal_name:
        raise ValueError("signal feature identity does not match the preregistration")

    timestamps = np.asarray(dataset.timestamps, dtype="datetime64[ns]")
    features = np.asarray(dataset.features, dtype=np.float32)
    feature_available = np.asarray(dataset.feature_available, dtype=np.bool_)
    open_price = np.asarray(dataset.open, dtype=np.float64)
    active = _readonly_bool(dataset.asset_active, field="asset_active")
    tradable = np.asarray(dataset.tradable, dtype=np.bool_)
    fee = _readonly_float(dataset.fee_rate, field="fee_rate")
    taker_fee = _readonly_float(dataset.taker_fee_rate, field="taker_fee_rate")
    spread = _readonly_float(dataset.spread_rate, field="spread_rate")

    expected_shape = (dataset.n_bars, dataset.n_symbols)
    if active.shape != expected_shape or tradable.shape != expected_shape:
        raise ValueError("Dataset active/tradable shape is invalid")
    for field_name, array in (
        ("fee_rate", fee),
        ("taker_fee_rate", taker_fee),
        ("spread_rate", spread),
    ):
        if array.shape != expected_shape:
            raise ValueError(f"Dataset {field_name} shape is invalid")
    if (
        features.shape[:2] != expected_shape
        or feature_available.shape != features.shape
    ):
        raise ValueError("Dataset feature shape is invalid")

    fit_start = _datetime64(protocol.fit_start)
    fit_cutoff = _datetime64(protocol.fit_cutoff)
    signal_index = protocol.signal_index
    label_end_offset = protocol.label_endpoint_offset_bars
    label_start_offset = protocol.label_window_start_offset_bars
    label_stop_offset = protocol.label_window_stop_offset_bars_inclusive
    execution_offset = protocol.cost_execution_offset_bars

    symbol_results: list[MeanReversionEconomicGateSymbolCalibration] = []
    coverage_invalid = False

    for symbol_index, symbol in enumerate(protocol.symbols):
        signals: list[float] = []
        labels: list[float] = []
        fee_values: list[float] = []
        taker_values: list[float] = []
        spread_values: list[float] = []

        max_t = dataset.n_bars - label_end_offset
        for t in range(max_t):
            if timestamps[t] < fit_start:
                continue
            label_end = t + label_end_offset
            if timestamps[label_end] >= fit_cutoff:
                continue
            if not feature_available[t, symbol_index, signal_index]:
                continue
            if not active[t, symbol_index] or not tradable[t, symbol_index]:
                continue
            window = slice(t + label_start_offset, t + label_stop_offset + 1)
            if not np.all(
                active[window, symbol_index] & tradable[window, symbol_index]
            ):
                continue

            signal = float(features[t, symbol_index, signal_index])
            start_open = float(
                open_price[t + protocol.label_execution_offset_bars, symbol_index]
            )
            end_open = float(open_price[label_end, symbol_index])
            if (
                not math.isfinite(signal)
                or not math.isfinite(start_open)
                or not math.isfinite(end_open)
                or start_open <= 0.0
                or end_open <= 0.0
            ):
                continue

            execution_row = t + execution_offset
            fee_value = _finite_cost(
                float(fee[execution_row, symbol_index]),
                expected=protocol.market_order_fee_rate,
                field="fee_rate",
            )
            taker_value = _finite_cost(
                float(taker_fee[execution_row, symbol_index]),
                expected=protocol.market_order_taker_fee_rate,
                field="taker_fee_rate",
            )
            spread_value = _finite_cost(
                float(spread[execution_row, symbol_index]),
                expected=protocol.market_order_spread_rate,
                field="spread_rate",
            )
            label = math.log(end_open / start_open)
            signals.append(signal)
            labels.append(label)
            fee_values.append(fee_value)
            taker_values.append(taker_value)
            spread_values.append(spread_value)

        numerator = math.fsum(
            signal * label for signal, label in zip(signals, labels, strict=True)
        )
        denominator = math.fsum(signal * signal for signal in signals)
        eligible = len(signals)
        beta: float | None
        if (
            eligible < protocol.minimum_eligible_observations_per_symbol
            or not math.isfinite(numerator)
            or not math.isfinite(denominator)
            or denominator <= 0.0
        ):
            beta = None
            coverage_invalid = True
        else:
            beta = numerator / denominator
            if not math.isfinite(beta):
                beta = None
                coverage_invalid = True

        fee_min, fee_max = _cost_range(fee_values)
        taker_min, taker_max = _cost_range(taker_values)
        spread_min, spread_max = _cost_range(spread_values)
        symbol_results.append(
            MeanReversionEconomicGateSymbolCalibration(
                symbol=symbol,
                eligible_observations=eligible,
                numerator=numerator,
                denominator=denominator,
                beta=beta,
                fee_rate_min=fee_min,
                fee_rate_max=fee_max,
                taker_fee_rate_min=taker_min,
                taker_fee_rate_max=taker_max,
                spread_rate_min=spread_min,
                spread_rate_max=spread_max,
            )
        )

    negative_count = sum(
        item.beta is not None and item.beta < 0.0 for item in symbol_results
    )
    beta_gate: float | None = None
    if coverage_invalid:
        status = protocol.invalid_coverage_status
    elif negative_count < protocol.required_negative_symbol_slopes:
        status = protocol.invalid_training_edge_status
    else:
        betas = sorted(item.beta for item in symbol_results if item.beta is not None)
        beta_gate = betas[protocol.beta_gate_order_statistic - 1]
        if beta_gate >= 0.0:
            raise ValueError(
                "preregistered beta order statistic is not strictly negative"
            )
        status = _VALID_STATUS

    return MeanReversionEconomicGateCalibration(
        protocol_digest=protocol.digest,
        dataset_id=dataset.dataset_id,
        symbols=protocol.symbols,
        symbol_results=tuple(symbol_results),
        negative_slope_count=int(negative_count),
        beta_gate=beta_gate,
        status=status,
    )


__all__ = [
    "MeanReversionEconomicGateCalibration",
    "MeanReversionEconomicGateSymbolCalibration",
    "calibrate_mean_reversion_economic_gate",
]
