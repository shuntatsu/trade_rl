"""Sealed pairwise evaluator for the Issue #616 ridge economic transition gate."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, fields, replace
from pathlib import Path
from statistics import median

import numpy as np

from trade_rl.artifacts.hashing import content_digest
from trade_rl.data.market import MarketDataset
from trade_rl.evaluation.comparison.strategies import compare_strategies_by_symbol
from trade_rl.simulation.execution import ExecutionCostConfig
from trade_rl.strategies.forecasts.ridge import (
    RidgeForecastModel,
    RidgeForecastStrategy,
    fit_ridge_forecast,
)
from trade_rl.strategies.forecasts.ridge_economic_gate import (
    RidgeEconomicGateConfig,
    RidgeEconomicGateStrategy,
)

_SYMBOLS = ("BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT")
_FEATURE_NAMES = (
    "1h__log_return_1bar",
    "1h__log_return_4bar",
    "1h__log_return_24bar",
    "1h__realized_volatility_24bar",
    "1h__volume_zscore_24bar",
    "1h__funding_bps",
    "1h__rsi_14bar",
    "1h__macd_histogram_12_26_9",
    "4h__log_return_4bar",
    "4h__realized_volatility_24bar",
    "1d__log_return_1bar",
    "1d__realized_volatility_24bar",
)
_FEATURE_INDICES = (0, 1, 2, 4, 5, 6, 7, 10, 60, 63, 114, 118)
_FIT_SYMBOL_INDICES = (0, 1, 2, 3, 4)
_CAPACITY_CAPS = (
    0.0021629560553901974,
    0.002044685341258238,
    0.002184898995567895,
    0.0020480213652913385,
    0.002346308308284808,
)

_CANONICAL_SPEC_VALUES: dict[str, object] = {
    "schema_version": "ridge_economic_gate_evaluation_spec_v1",
    "issue_number": 616,
    "protocol_digest": "167af235eeba44b9ade780c95ef7bab11c7b211d81018b82ece5d7044f7c4eb3",
    "prereg_head": "75999e53c70224c31a62b106e4a8d2caa4b920ac",
    "prereg_seal_run_id": 35126253392,
    "prereg_seal_artifact_id": 10459431804,
    "prereg_seal_artifact_digest": "603ca1c50acfcc734762c1059b315e3a31dd6a1068fb5c6481790bfe941714c7",
    "prereg_fresh_artifact_id": 10458759061,
    "prereg_fresh_artifact_digest": "4b00406c1b1ff587514256bd02cc7119940f35b7830829beb21c125f5bb9fa66",
    "dataset_id": "6c0b040d317a1bb73a9273f4135879b31691634aa837f30f0eec005ac7531518",
    "dataset_artifact_digest": "af481dd978db7d84cd3aa8ff4f5a35d8608ac44c755dd74f61e934105c02b6b7",
    "study_digest": "bfa2fcb307773f5384d7dcb884444164d6d3b575373d8f7dc1c810a61bf4c820",
    "execution_overlay": "zero_overlay_dataset_fields_authoritative_previous_completed_bar_capacity",
    "symbols": _SYMBOLS,
    "feature_names": _FEATURE_NAMES,
    "feature_indices": _FEATURE_INDICES,
    "fit_symbol_indices": _FIT_SYMBOL_INDICES,
    "fit_cutoff": "2023-01-01T00:00:00",
    "evaluation_start": "2023-01-01T00:00:00",
    "evaluation_stop_exclusive": "2025-01-01T00:00:00",
    "ridge_horizon_hours": 24,
    "ridge_alpha": 1.0,
    "forecast_entry_threshold": 0.0025,
    "forecast_exit_threshold": 0.0005,
    "gross_budget": 0.5,
    "initial_capital": 100_000.0,
    "market_order_fee_rate": 0.0005,
    "market_order_taker_fee_rate": 0.0,
    "market_order_spread_rate": 0.0002,
    "one_way_explicit_cost": 0.0007,
    "capacity_caps": _CAPACITY_CAPS,
    "evaluation_pnl_inspected": False,
    "evaluation_execution_authorized": False,
    "final_test_accessed": False,
    "final_test_authorized": False,
    "shared_cash_profitability_established": False,
    "operational_eligibility_established": False,
    "production_eligible": False,
    "live_trading_authorized": False,
    "merge_authorized": False,
}


def _require_hex(value: object, *, field: str, length: int = 64) -> str:
    if not isinstance(value, str) or len(value) != length:
        raise ValueError(f"{field} must be a {length}-character lowercase hex digest")
    if value.lower() != value or any(char not in "0123456789abcdef" for char in value):
        raise ValueError(f"{field} must be a {length}-character lowercase hex digest")
    return value


def _require_finite(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be finite")
    resolved = float(value)
    if not math.isfinite(resolved):
        raise ValueError(f"{field} must be finite")
    return resolved


def _require_nonnegative(value: object, *, field: str) -> float:
    resolved = _require_finite(value, field=field)
    if resolved < 0.0:
        raise ValueError(f"{field} must be non-negative")
    return resolved


def _require_count(value: object, *, field: str, maximum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    if maximum is not None and value > maximum:
        raise ValueError(f"{field} exceeds the symbol roster")
    return value


def _as_json_value(value: object) -> object:
    if isinstance(value, tuple):
        return [_as_json_value(item) for item in value]
    return value


@dataclass(frozen=True, slots=True)
class RidgeEconomicGateEvaluationSpec:
    """Immutable evaluator semantics bound to the sealed Issue #616 protocol."""

    schema_version: str
    issue_number: int
    protocol_digest: str
    prereg_head: str
    prereg_seal_run_id: int
    prereg_seal_artifact_id: int
    prereg_seal_artifact_digest: str
    prereg_fresh_artifact_id: int
    prereg_fresh_artifact_digest: str
    dataset_id: str
    dataset_artifact_digest: str
    study_digest: str
    execution_overlay: str
    symbols: tuple[str, ...]
    feature_names: tuple[str, ...]
    feature_indices: tuple[int, ...]
    fit_symbol_indices: tuple[int, ...]
    fit_cutoff: str
    evaluation_start: str
    evaluation_stop_exclusive: str
    ridge_horizon_hours: int
    ridge_alpha: float
    forecast_entry_threshold: float
    forecast_exit_threshold: float
    gross_budget: float
    initial_capital: float
    market_order_fee_rate: float
    market_order_taker_fee_rate: float
    market_order_spread_rate: float
    one_way_explicit_cost: float
    capacity_caps: tuple[float, ...]
    evaluation_pnl_inspected: bool
    evaluation_execution_authorized: bool
    final_test_accessed: bool
    final_test_authorized: bool
    shared_cash_profitability_established: bool
    operational_eligibility_established: bool
    production_eligible: bool
    live_trading_authorized: bool
    merge_authorized: bool

    def __post_init__(self) -> None:
        for field_name in (
            "protocol_digest",
            "prereg_seal_artifact_digest",
            "prereg_fresh_artifact_digest",
            "dataset_id",
            "dataset_artifact_digest",
            "study_digest",
        ):
            _require_hex(getattr(self, field_name), field=field_name)
        _require_hex(self.prereg_head, field="prereg_head", length=40)
        for field_name in (
            "issue_number",
            "prereg_seal_run_id",
            "prereg_seal_artifact_id",
            "prereg_fresh_artifact_id",
            "ridge_horizon_hours",
        ):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{field_name} must be a positive integer")
        for field_name in (
            "ridge_alpha",
            "forecast_entry_threshold",
            "forecast_exit_threshold",
            "gross_budget",
            "initial_capital",
            "market_order_fee_rate",
            "market_order_taker_fee_rate",
            "market_order_spread_rate",
            "one_way_explicit_cost",
        ):
            _require_nonnegative(getattr(self, field_name), field=field_name)
        if (
            self.ridge_alpha <= 0.0
            or self.gross_budget <= 0.0
            or self.initial_capital <= 0.0
        ):
            raise ValueError("preregistered model/capital parameters are invalid")
        if self.forecast_exit_threshold >= self.forecast_entry_threshold:
            raise ValueError("preregistered forecast thresholds are invalid")
        if not math.isclose(
            self.market_order_fee_rate
            + self.market_order_taker_fee_rate
            + self.market_order_spread_rate,
            self.one_way_explicit_cost,
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            raise ValueError(
                "one_way_explicit_cost differs from preregistered components"
            )
        for field_name in (
            "evaluation_pnl_inspected",
            "evaluation_execution_authorized",
            "final_test_accessed",
            "final_test_authorized",
            "shared_cash_profitability_established",
            "operational_eligibility_established",
            "production_eligible",
            "live_trading_authorized",
            "merge_authorized",
        ):
            if type(getattr(self, field_name)) is not bool:
                raise ValueError(f"{field_name} must be a boolean")
        for item in fields(self):
            if getattr(self, item.name) != _CANONICAL_SPEC_VALUES[item.name]:
                raise ValueError(
                    f"{item.name} differs from sealed Issue #616 semantics"
                )

    def to_payload(self) -> dict[str, object]:
        return {
            item.name: _as_json_value(getattr(self, item.name)) for item in fields(self)
        }

    @property
    def digest(self) -> str:
        return content_digest(self.to_payload())


def canonical_ridge_economic_gate_evaluation_spec() -> RidgeEconomicGateEvaluationSpec:
    """Return the only evaluator spec authorized by the sealed Issue #616 protocol."""

    return RidgeEconomicGateEvaluationSpec(**_CANONICAL_SPEC_VALUES)  # type: ignore[arg-type]


def ridge_model_sha256(model: RidgeForecastModel) -> str:
    """Hash the complete frozen Ridge model semantics in a portable fixed encoding."""

    if not isinstance(model, RidgeForecastModel):
        raise TypeError("model must be a RidgeForecastModel")
    metadata = {
        "feature_indices": list(model.feature_indices),
        "intercept": model.intercept,
        "horizon_hours": model.horizon_hours,
        "alpha": model.alpha,
        "n_samples": model.n_samples,
        "fit_cutoff_ns": int(np.datetime64(model.fit_cutoff, "ns").astype(np.int64)),
    }
    hasher = hashlib.sha256()
    hasher.update(
        json.dumps(
            metadata, allow_nan=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    )
    for array in (model.feature_mean, model.feature_scale, model.coefficients):
        values = np.asarray(array, dtype="<f8").reshape(-1)
        hasher.update(len(values).to_bytes(8, "big", signed=False))
        hasher.update(values.tobytes(order="C"))
    return hasher.hexdigest()


def research_status_from_counts(
    *,
    positive_effect_symbols: int,
    median_excess_total_return: float,
    cost_reduction_symbols: int,
    turnover_reduction_symbols: int,
    drawdown_nonworse_symbols: int,
    new_termination_symbols: int,
) -> str:
    """Apply the sealed #616 cross-symbol decision rule without reinterpretation."""

    for field_name, value in (
        ("positive_effect_symbols", positive_effect_symbols),
        ("cost_reduction_symbols", cost_reduction_symbols),
        ("turnover_reduction_symbols", turnover_reduction_symbols),
        ("drawdown_nonworse_symbols", drawdown_nonworse_symbols),
        ("new_termination_symbols", new_termination_symbols),
    ):
        _require_count(value, field=field_name, maximum=len(_SYMBOLS))
    median_value = _require_finite(
        median_excess_total_return, field="median_excess_total_return"
    )

    if (
        positive_effect_symbols >= 4
        and median_value > 0.0
        and cost_reduction_symbols >= 4
        and turnover_reduction_symbols >= 4
        and drawdown_nonworse_symbols >= 4
        and new_termination_symbols == 0
    ):
        return "PROMOTE_RESEARCH_REFERENCE"
    if (
        positive_effect_symbols <= 2
        or median_value <= 0.0
        or cost_reduction_symbols <= 2
        or turnover_reduction_symbols <= 2
        or drawdown_nonworse_symbols <= 2
        or new_termination_symbols > 0
    ):
        return "REJECT_MECHANISM"
    return "INCONCLUSIVE"


@dataclass(frozen=True, slots=True)
class RidgeEconomicGateSymbolResult:
    """One symbol's fully auditable baseline/candidate pairwise metrics."""

    symbol: str
    baseline_total_return: float
    candidate_total_return: float
    excess_total_return: float
    baseline_total_cost: float
    candidate_total_cost: float
    baseline_turnover_total: float
    candidate_turnover_total: float
    baseline_max_drawdown: float
    candidate_max_drawdown: float
    baseline_termination_count: int
    candidate_termination_count: int
    baseline_termination_reasons: tuple[str, ...]
    candidate_termination_reasons: tuple[str, ...]
    new_termination: bool
    baseline_n_periods: int
    candidate_n_periods: int
    baseline_return_sha256: str
    candidate_return_sha256: str

    def __post_init__(self) -> None:
        if self.symbol not in _SYMBOLS:
            raise ValueError("symbol is outside the sealed Issue #616 roster")
        for field_name in (
            "baseline_total_return",
            "candidate_total_return",
            "excess_total_return",
        ):
            _require_finite(getattr(self, field_name), field=field_name)
        for field_name in (
            "baseline_total_cost",
            "candidate_total_cost",
            "baseline_turnover_total",
            "candidate_turnover_total",
            "baseline_max_drawdown",
            "candidate_max_drawdown",
        ):
            _require_nonnegative(getattr(self, field_name), field=field_name)
        expected_excess = self.candidate_total_return - self.baseline_total_return
        if not math.isclose(
            self.excess_total_return,
            expected_excess,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(
                "excess_total_return does not match candidate minus baseline"
            )
        for field_name in (
            "baseline_termination_count",
            "candidate_termination_count",
            "baseline_n_periods",
            "candidate_n_periods",
        ):
            _require_count(getattr(self, field_name), field=field_name)
        if self.baseline_n_periods <= 0 or self.candidate_n_periods <= 0:
            raise ValueError("pairwise return series must contain at least one period")
        if self.baseline_n_periods != self.candidate_n_periods:
            raise ValueError("baseline/candidate n_periods must match")
        for field_name in (
            "baseline_termination_reasons",
            "candidate_termination_reasons",
        ):
            reasons = getattr(self, field_name)
            if not isinstance(reasons, tuple) or any(
                not isinstance(reason, str) or not reason for reason in reasons
            ):
                raise ValueError(f"{field_name} must be a tuple of non-empty strings")
        if self.baseline_termination_count != len(self.baseline_termination_reasons):
            raise ValueError("baseline termination count/reasons disagree")
        if self.candidate_termination_count != len(self.candidate_termination_reasons):
            raise ValueError("candidate termination count/reasons disagree")
        expected_new = (
            self.candidate_termination_count > self.baseline_termination_count
            or any(
                reason not in self.baseline_termination_reasons
                for reason in self.candidate_termination_reasons
            )
        )
        if (
            type(self.new_termination) is not bool
            or self.new_termination != expected_new
        ):
            raise ValueError(
                "new_termination does not match pairwise termination evidence"
            )
        _require_hex(self.baseline_return_sha256, field="baseline_return_sha256")
        _require_hex(self.candidate_return_sha256, field="candidate_return_sha256")

    def to_payload(self) -> dict[str, object]:
        return {
            item.name: _as_json_value(getattr(self, item.name)) for item in fields(self)
        }


def _return_sha256(values: tuple[float, ...]) -> str:
    array = np.asarray(values, dtype="<f8")
    if array.ndim != 1 or array.size == 0 or not np.isfinite(array).all():
        raise ValueError("return series must be non-empty and finite")
    hasher = hashlib.sha256()
    hasher.update(array.size.to_bytes(8, "big", signed=False))
    hasher.update(array.tobytes(order="C"))
    return hasher.hexdigest()


@dataclass(frozen=True, slots=True)
class RidgeEconomicGateEvaluation:
    """Strict pairwise result whose aggregate decision is recomputed on load/build."""

    spec_digest: str
    dataset_id: str
    model_sha256: str
    symbols: tuple[str, ...]
    by_symbol: tuple[RidgeEconomicGateSymbolResult, ...]
    positive_effect_symbols: int
    median_excess_total_return: float
    cost_reduction_symbols: int
    turnover_reduction_symbols: int
    drawdown_nonworse_symbols: int
    new_termination_symbols: int
    candidate_positive_total_return_symbols: int
    research_status: str
    evaluation_pnl_computed: bool = True
    evaluation_pnl_interpreted: bool = False
    final_test_accessed: bool = False
    final_test_authorized: bool = False
    shared_cash_profitability_established: bool = False
    operational_eligibility_established: bool = False
    production_eligible: bool = False
    live_trading_authorized: bool = False
    merge_authorized: bool = False

    def __post_init__(self) -> None:
        _require_hex(self.spec_digest, field="spec_digest")
        _require_hex(self.dataset_id, field="dataset_id")
        _require_hex(self.model_sha256, field="model_sha256")
        sealed_spec = canonical_ridge_economic_gate_evaluation_spec()
        if self.spec_digest != sealed_spec.digest:
            raise ValueError(
                "spec_digest differs from sealed Issue #616 evaluator spec"
            )
        if self.dataset_id != sealed_spec.dataset_id:
            raise ValueError(
                "dataset_id differs from sealed Issue #616 evaluator authority"
            )
        if self.symbols != _SYMBOLS:
            raise ValueError("symbols differ from sealed Issue #616 roster")
        if tuple(row.symbol for row in self.by_symbol) != self.symbols:
            raise ValueError("by_symbol order differs from sealed Issue #616 roster")
        if len(self.by_symbol) != len(_SYMBOLS):
            raise ValueError("by_symbol must contain all five sealed symbols")

        positive = sum(row.excess_total_return > 0.0 for row in self.by_symbol)
        median_excess = float(median(row.excess_total_return for row in self.by_symbol))
        cost_reduction = sum(
            row.candidate_total_cost < row.baseline_total_cost for row in self.by_symbol
        )
        turnover_reduction = sum(
            row.candidate_turnover_total < row.baseline_turnover_total
            for row in self.by_symbol
        )
        drawdown_nonworse = sum(
            row.candidate_max_drawdown <= row.baseline_max_drawdown
            for row in self.by_symbol
        )
        new_termination = sum(row.new_termination for row in self.by_symbol)
        candidate_positive = sum(
            row.candidate_total_return > 0.0 for row in self.by_symbol
        )
        expected_counts = {
            "positive_effect_symbols": positive,
            "cost_reduction_symbols": cost_reduction,
            "turnover_reduction_symbols": turnover_reduction,
            "drawdown_nonworse_symbols": drawdown_nonworse,
            "new_termination_symbols": new_termination,
            "candidate_positive_total_return_symbols": candidate_positive,
        }
        for field_name, expected in expected_counts.items():
            actual = getattr(self, field_name)
            _require_count(actual, field=field_name, maximum=len(_SYMBOLS))
            if actual != expected:
                raise ValueError(f"{field_name} does not match by_symbol evidence")
        if not math.isclose(
            self.median_excess_total_return,
            median_excess,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(
                "median_excess_total_return does not match by_symbol evidence"
            )
        expected_status = research_status_from_counts(
            positive_effect_symbols=positive,
            median_excess_total_return=median_excess,
            cost_reduction_symbols=cost_reduction,
            turnover_reduction_symbols=turnover_reduction,
            drawdown_nonworse_symbols=drawdown_nonworse,
            new_termination_symbols=new_termination,
        )
        if self.research_status != expected_status:
            raise ValueError("research_status does not match sealed decision rule")
        fixed_flags = {
            "evaluation_pnl_computed": True,
            "evaluation_pnl_interpreted": False,
            "final_test_accessed": False,
            "final_test_authorized": False,
            "shared_cash_profitability_established": False,
            "operational_eligibility_established": False,
            "production_eligible": False,
            "live_trading_authorized": False,
            "merge_authorized": False,
        }
        for field_name, expected in fixed_flags.items():
            value = getattr(self, field_name)
            if type(value) is not bool or value is not expected:
                raise ValueError(f"{field_name} violates the sealed research boundary")

    def to_payload(self) -> dict[str, object]:
        payload: dict[str, object] = {}
        for item in fields(self):
            value = getattr(self, item.name)
            if item.name == "by_symbol":
                payload[item.name] = [row.to_payload() for row in self.by_symbol]
            else:
                payload[item.name] = _as_json_value(value)
        return payload


def _exact_timestamp_index(
    dataset: MarketDataset, timestamp: str, *, field: str
) -> int:
    target = np.datetime64(timestamp, "ns")
    matches = np.flatnonzero(dataset.timestamps == target)
    if matches.size != 1:
        raise ValueError(f"{field} is not uniquely present in Dataset timestamps")
    return int(matches[0])


def _validate_dataset(
    dataset: MarketDataset,
    spec: RidgeEconomicGateEvaluationSpec,
) -> tuple[int, int]:
    if dataset.dataset_id != spec.dataset_id:
        raise ValueError("Dataset id differs from sealed Issue #616 authority")
    if tuple(dataset.symbols) != spec.symbols:
        raise ValueError("Dataset symbols differ from sealed Issue #616 authority")
    if max(spec.feature_indices) >= dataset.n_features:
        raise ValueError("sealed feature index is outside Dataset features")
    selected_names = tuple(
        dataset.feature_names[index] for index in spec.feature_indices
    )
    if selected_names != spec.feature_names:
        raise ValueError(
            "Dataset feature roster differs from sealed Issue #616 authority"
        )

    start_index = _exact_timestamp_index(
        dataset, spec.evaluation_start, field="evaluation_start"
    )
    stop_index = _exact_timestamp_index(
        dataset, spec.evaluation_stop_exclusive, field="evaluation_stop_exclusive"
    )
    if not 0 <= start_index < stop_index < dataset.n_bars:
        raise ValueError("sealed evaluation range is invalid for Dataset")

    scope = slice(start_index, stop_index + 1)
    for field_name, expected in (
        ("fee_rate", spec.market_order_fee_rate),
        ("taker_fee_rate", spec.market_order_taker_fee_rate),
        ("spread_rate", spec.market_order_spread_rate),
    ):
        values = np.asarray(dataset.resolved_array(field_name)[scope], dtype=np.float64)
        if not np.isfinite(values).all() or np.any(values < 0.0):
            raise ValueError(f"evaluation cost drift: {field_name} is invalid")
        if not bool(np.all(values == expected)):
            raise ValueError(
                f"evaluation cost drift: {field_name} differs from sealed value"
            )

    capacity = np.asarray(
        dataset.resolved_array("max_participation_rate")[scope], dtype=np.float64
    )
    expected_capacity = np.broadcast_to(
        np.asarray(spec.capacity_caps, dtype=np.float64), capacity.shape
    )
    if not np.isfinite(capacity).all() or not np.array_equal(
        capacity, expected_capacity
    ):
        raise ValueError(
            "Dataset causal capacity differs from sealed Issue #616 authority"
        )
    return start_index, stop_index


def _entry_by_name(symbol_comparison: object, name: str) -> object:
    entries = getattr(getattr(symbol_comparison, "comparison"), "entries")
    matches = [entry for entry in entries if getattr(entry, "name") == name]
    if len(matches) != 1:
        raise ValueError(f"pairwise comparison must contain exactly one {name} entry")
    return matches[0]


def _symbol_result(symbol_comparison: object) -> RidgeEconomicGateSymbolResult:
    symbol = getattr(symbol_comparison, "symbol")
    baseline = _entry_by_name(symbol_comparison, "baseline")
    candidate = _entry_by_name(symbol_comparison, "candidate")
    baseline_metrics = getattr(baseline, "metrics")
    candidate_metrics = getattr(candidate, "metrics")
    baseline_replay = getattr(baseline, "replay")
    candidate_replay = getattr(candidate, "replay")
    baseline_reasons = tuple(
        getattr(baseline_replay.diagnostics, "termination_reasons")
    )
    candidate_reasons = tuple(
        getattr(candidate_replay.diagnostics, "termination_reasons")
    )
    new_termination = (
        candidate_metrics.termination_count > baseline_metrics.termination_count
        or any(reason not in baseline_reasons for reason in candidate_reasons)
    )
    return RidgeEconomicGateSymbolResult(
        symbol=symbol,
        baseline_total_return=float(baseline_metrics.total_return),
        candidate_total_return=float(candidate_metrics.total_return),
        excess_total_return=float(
            candidate_metrics.total_return - baseline_metrics.total_return
        ),
        baseline_total_cost=float(baseline_metrics.total_cost),
        candidate_total_cost=float(candidate_metrics.total_cost),
        baseline_turnover_total=float(baseline_metrics.turnover_total),
        candidate_turnover_total=float(candidate_metrics.turnover_total),
        baseline_max_drawdown=float(baseline_metrics.max_drawdown),
        candidate_max_drawdown=float(candidate_metrics.max_drawdown),
        baseline_termination_count=int(baseline_metrics.termination_count),
        candidate_termination_count=int(candidate_metrics.termination_count),
        baseline_termination_reasons=baseline_reasons,
        candidate_termination_reasons=candidate_reasons,
        new_termination=new_termination,
        baseline_n_periods=int(baseline_metrics.n_periods),
        candidate_n_periods=int(candidate_metrics.n_periods),
        baseline_return_sha256=_return_sha256(tuple(baseline_replay.returns.values)),
        candidate_return_sha256=_return_sha256(tuple(candidate_replay.returns.values)),
    )


def evaluate_ridge_economic_gate(
    dataset: MarketDataset,
    spec: RidgeEconomicGateEvaluationSpec | None = None,
) -> RidgeEconomicGateEvaluation:
    """Fit canonical Ridge once, then replay baseline/candidate under one environment."""

    resolved_spec = spec or canonical_ridge_economic_gate_evaluation_spec()
    start_index, stop_index = _validate_dataset(dataset, resolved_spec)
    model = fit_ridge_forecast(
        dataset,
        feature_indices=resolved_spec.feature_indices,
        fit_symbol_indices=resolved_spec.fit_symbol_indices,
        fit_cutoff=np.datetime64(resolved_spec.fit_cutoff, "ns"),
        horizon_hours=resolved_spec.ridge_horizon_hours,
        alpha=resolved_spec.ridge_alpha,
    )
    baseline = RidgeForecastStrategy(
        model,
        entry_threshold=resolved_spec.forecast_entry_threshold,
        exit_threshold=resolved_spec.forecast_exit_threshold,
    )
    candidate = RidgeEconomicGateStrategy(
        RidgeEconomicGateConfig(
            model=model,
            entry_threshold=resolved_spec.forecast_entry_threshold,
            exit_threshold=resolved_spec.forecast_exit_threshold,
            one_way_explicit_cost=resolved_spec.one_way_explicit_cost,
        )
    )
    execution_cost = replace(
        ExecutionCostConfig.zero(),
        processing_bar_volume_capacity=False,
    )
    comparison = compare_strategies_by_symbol(
        dataset,
        {"baseline": baseline, "candidate": candidate},
        start_index=start_index,
        stop_index=stop_index,
        gross_budget=resolved_spec.gross_budget,
        initial_capital=resolved_spec.initial_capital,
        execution_cost=execution_cost,
        risk=None,
    )
    rows = tuple(_symbol_result(item) for item in comparison.by_symbol)
    if tuple(row.symbol for row in rows) != resolved_spec.symbols:
        raise ValueError(
            "comparison symbol order differs from sealed Issue #616 roster"
        )
    positive = sum(row.excess_total_return > 0.0 for row in rows)
    median_excess = float(median(row.excess_total_return for row in rows))
    cost_reduction = sum(
        row.candidate_total_cost < row.baseline_total_cost for row in rows
    )
    turnover_reduction = sum(
        row.candidate_turnover_total < row.baseline_turnover_total for row in rows
    )
    drawdown_nonworse = sum(
        row.candidate_max_drawdown <= row.baseline_max_drawdown for row in rows
    )
    new_termination = sum(row.new_termination for row in rows)
    candidate_positive = sum(row.candidate_total_return > 0.0 for row in rows)
    status = research_status_from_counts(
        positive_effect_symbols=positive,
        median_excess_total_return=median_excess,
        cost_reduction_symbols=cost_reduction,
        turnover_reduction_symbols=turnover_reduction,
        drawdown_nonworse_symbols=drawdown_nonworse,
        new_termination_symbols=new_termination,
    )
    return RidgeEconomicGateEvaluation(
        spec_digest=resolved_spec.digest,
        dataset_id=dataset.dataset_id,
        model_sha256=ridge_model_sha256(model),
        symbols=resolved_spec.symbols,
        by_symbol=rows,
        positive_effect_symbols=positive,
        median_excess_total_return=median_excess,
        cost_reduction_symbols=cost_reduction,
        turnover_reduction_symbols=turnover_reduction,
        drawdown_nonworse_symbols=drawdown_nonworse,
        new_termination_symbols=new_termination,
        candidate_positive_total_return_symbols=candidate_positive,
        research_status=status,
    )


_RESULT_SCHEMA_VERSION = "ridge_economic_gate_evaluation_result_v1"


def _canonical_json_bytes(payload: object) -> bytes:
    return (
        json.dumps(payload, allow_nan=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def canonical_ridge_economic_gate_evaluation_bytes(
    result: RidgeEconomicGateEvaluation,
) -> bytes:
    """Encode one strict immutable result document in canonical JSON bytes."""

    if not isinstance(result, RidgeEconomicGateEvaluation):
        raise TypeError("result must be a RidgeEconomicGateEvaluation")
    payload = result.to_payload()
    document = {
        "schema_version": _RESULT_SCHEMA_VERSION,
        "content_digest": content_digest(payload),
        "result": payload,
    }
    return _canonical_json_bytes(document)


def _require_object(value: object, *, field: str) -> dict[str, object]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{field} must be an object")
    return value


def _require_exact_keys(
    payload: dict[str, object],
    *,
    expected: set[str],
    field: str,
) -> None:
    if set(payload) != expected:
        raise ValueError(f"{field} keys differ from sealed schema")


def _tuple_strings(value: object, *, field: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError(f"{field} must be a string array")
    return tuple(value)


def _decode_symbol_result(value: object) -> RidgeEconomicGateSymbolResult:
    raw = _require_object(value, field="by_symbol item")
    expected = {item.name for item in fields(RidgeEconomicGateSymbolResult)}
    _require_exact_keys(raw, expected=expected, field="by_symbol item")
    converted = dict(raw)
    converted["baseline_termination_reasons"] = _tuple_strings(
        raw["baseline_termination_reasons"], field="baseline_termination_reasons"
    )
    converted["candidate_termination_reasons"] = _tuple_strings(
        raw["candidate_termination_reasons"], field="candidate_termination_reasons"
    )
    return RidgeEconomicGateSymbolResult(**converted)  # type: ignore[arg-type]


def load_ridge_economic_gate_evaluation(
    path: str | Path,
) -> RidgeEconomicGateEvaluation:
    """Load canonical result bytes and reject schema, digest, or semantic forgery."""

    raw_bytes = Path(path).read_bytes()
    try:
        document_value = json.loads(raw_bytes)
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ValueError("evaluation result must contain valid JSON") from error
    document = _require_object(document_value, field="evaluation result")
    _require_exact_keys(
        document,
        expected={"schema_version", "content_digest", "result"},
        field="evaluation result",
    )
    if document["schema_version"] != _RESULT_SCHEMA_VERSION:
        raise ValueError("evaluation result schema_version is not supported")
    digest = _require_hex(document["content_digest"], field="content_digest")
    result_raw = _require_object(document["result"], field="result")
    expected_result_keys = {item.name for item in fields(RidgeEconomicGateEvaluation)}
    _require_exact_keys(result_raw, expected=expected_result_keys, field="result")

    converted = dict(result_raw)
    converted["symbols"] = _tuple_strings(result_raw["symbols"], field="symbols")
    by_symbol_raw = result_raw["by_symbol"]
    if not isinstance(by_symbol_raw, list):
        raise ValueError("by_symbol must be an array")
    converted["by_symbol"] = tuple(
        _decode_symbol_result(item) for item in by_symbol_raw
    )
    result = RidgeEconomicGateEvaluation(**converted)  # type: ignore[arg-type]
    if content_digest(result.to_payload()) != digest:
        raise ValueError("evaluation result content_digest mismatch")
    canonical = canonical_ridge_economic_gate_evaluation_bytes(result)
    if raw_bytes != canonical:
        raise ValueError("evaluation result bytes are not canonical JSON")
    return result


__all__ = [
    "RidgeEconomicGateEvaluation",
    "RidgeEconomicGateEvaluationSpec",
    "RidgeEconomicGateSymbolResult",
    "canonical_ridge_economic_gate_evaluation_bytes",
    "canonical_ridge_economic_gate_evaluation_spec",
    "evaluate_ridge_economic_gate",
    "load_ridge_economic_gate_evaluation",
    "research_status_from_counts",
    "ridge_model_sha256",
]
