"""Result-blind pairwise evaluator for the sealed ridge24 economic gate."""

from __future__ import annotations

import hashlib
import math
from collections import Counter
from dataclasses import dataclass, fields
from statistics import median

import numpy as np

from trade_rl.artifacts.hashing import content_digest
from trade_rl.data.market import MarketDataset
from trade_rl.evaluation.comparison.strategies import compare_strategies_by_symbol
from trade_rl.evaluation.experiments.bootstrap.ridge_economic_gate_prereg import (
    canonical_ridge_economic_gate_protocol,
)
from trade_rl.evaluation.metrics import compound_return
from trade_rl.evaluation.runs import execution_cost_for_overlay
from trade_rl.strategies.forecasts.ridge import (
    RidgeForecastStrategy,
    fit_ridge_forecast,
)
from trade_rl.strategies.forecasts.ridge_economic_gate import (
    RidgeEconomicGateConfig,
    RidgeEconomicGateStrategy,
)
from trade_rl.strategies.interface import SingleSymbolStrategy

_SCHEMA_VERSION = "ridge_economic_gate_evaluation_v1"
_RESULT_SCHEMA_VERSION = "ridge_economic_gate_evaluation_result_v1"
_COST_AUTHORITY_SCHEMA_VERSION = "ridge_economic_gate_cost_constancy_v1"
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
_SPEC_VALUES: dict[str, object] = {
    "schema_version": _SCHEMA_VERSION,
    "issue_number": 618,
    "protocol_issue_number": 616,
    "protocol_head_sha": "75999e53c70224c31a62b106e4a8d2caa4b920ac",
    "protocol_module_blob": "72d532351238223a1c467d54addb8b6a8de829f5",
    "protocol_digest": "167af235eeba44b9ade780c95ef7bab11c7b211d81018b82ece5d7044f7c4eb3",
    "protocol_document_content_digest": "82c7ede14f088e54f7e0922c013f48870aff9f5b9d796c6053f23d1daee10d48",
    "protocol_seal_content_digest": "b2e1414e1224983fd96cbe309071b71c2ce17c75c4aea1ea69014b70ab04094e",
    "protocol_seal_run_id": 35126253392,
    "protocol_seal_artifact_id": 10459431804,
    "protocol_seal_artifact_digest": "603ca1c50acfcc734762c1059b315e3a31dd6a1068fb5c6481790bfe941714c7",
    "protocol_fresh_artifact_id": 10458759061,
    "protocol_fresh_artifact_digest": "4b00406c1b1ff587514256bd02cc7119940f35b7830829beb21c125f5bb9fa66",
    "successor_bundle_run_id": 34803217815,
    "successor_bundle_artifact_id": 10331899302,
    "successor_bundle_artifact_digest": "89e899427f23fa46929c8be1e71fd49abe0d1d465c7a7f796a0874426b885bce",
    "dataset_id": "6c0b040d317a1bb73a9273f4135879b31691634aa837f30f0eec005ac7531518",
    "dataset_artifact_digest": "af481dd978db7d84cd3aa8ff4f5a35d8608ac44c755dd74f61e934105c02b6b7",
    "study_digest": "bfa2fcb307773f5384d7dcb884444164d6d3b575373d8f7dc1c810a61bf4c820",
    "execution_overlay": "zero_overlay_dataset_fields_authoritative_previous_completed_bar_capacity",
    "symbols": _SYMBOLS,
    "feature_names": _FEATURE_NAMES,
    "feature_indices": _FEATURE_INDICES,
    "fit_symbol_names": _SYMBOLS,
    "fit_cutoff": "2023-01-01T00:00:00.000000000",
    "evaluation_start": "2023-01-01T00:00:00.000000000",
    "evaluation_stop_exclusive": "2025-01-01T00:00:00.000000000",
    "ridge_horizon_hours": 24,
    "ridge_alpha": 1.0,
    "forecast_entry_threshold": 0.0025,
    "forecast_exit_threshold": 0.0005,
    "gross_budget": 0.5,
    "initial_capital": 100_000.0,
    "fee_rate": 0.0005,
    "taker_fee_rate": 0.0,
    "spread_rate": 0.0002,
    "one_way_explicit_cost": 0.0007,
    "cost_constancy_authority_required": True,
    "promote_min_positive_effect_symbols": 4,
    "promote_min_cost_reduction_symbols": 4,
    "promote_min_turnover_reduction_symbols": 4,
    "promote_min_drawdown_nonworse_symbols": 4,
    "reject_max_positive_effect_symbols": 2,
    "reject_max_cost_reduction_symbols": 2,
    "reject_max_turnover_reduction_symbols": 2,
    "reject_max_drawdown_nonworse_symbols": 2,
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


def _strict_equal(left: object, right: object) -> bool:
    if type(left) is not type(right):
        return False
    if isinstance(left, tuple) and isinstance(right, tuple):
        return len(left) == len(right) and all(
            _strict_equal(a, b) for a, b in zip(left, right, strict=True)
        )
    return bool(left == right)


def _require_hex(value: object, *, field: str, length: int = 64) -> str:
    if not isinstance(value, str) or len(value) != length:
        raise ValueError(f"{field} must be a {length}-character lowercase hex digest")
    if value.lower() != value or any(char not in "0123456789abcdef" for char in value):
        raise ValueError(f"{field} must be a {length}-character lowercase hex digest")
    return value


def _json_value(value: object) -> object:
    if isinstance(value, tuple):
        return [_json_value(item) for item in value]
    return value


@dataclass(frozen=True, slots=True)
class RidgeEconomicGateEvaluationSpec:
    """Exact result-blind authorities for the future one-shot ridge evaluation."""

    schema_version: str
    issue_number: int
    protocol_issue_number: int
    protocol_head_sha: str
    protocol_module_blob: str
    protocol_digest: str
    protocol_document_content_digest: str
    protocol_seal_content_digest: str
    protocol_seal_run_id: int
    protocol_seal_artifact_id: int
    protocol_seal_artifact_digest: str
    protocol_fresh_artifact_id: int
    protocol_fresh_artifact_digest: str
    successor_bundle_run_id: int
    successor_bundle_artifact_id: int
    successor_bundle_artifact_digest: str
    dataset_id: str
    dataset_artifact_digest: str
    study_digest: str
    execution_overlay: str
    symbols: tuple[str, ...]
    feature_names: tuple[str, ...]
    feature_indices: tuple[int, ...]
    fit_symbol_names: tuple[str, ...]
    fit_cutoff: str
    evaluation_start: str
    evaluation_stop_exclusive: str
    ridge_horizon_hours: int
    ridge_alpha: float
    forecast_entry_threshold: float
    forecast_exit_threshold: float
    gross_budget: float
    initial_capital: float
    fee_rate: float
    taker_fee_rate: float
    spread_rate: float
    one_way_explicit_cost: float
    cost_constancy_authority_required: bool
    promote_min_positive_effect_symbols: int
    promote_min_cost_reduction_symbols: int
    promote_min_turnover_reduction_symbols: int
    promote_min_drawdown_nonworse_symbols: int
    reject_max_positive_effect_symbols: int
    reject_max_cost_reduction_symbols: int
    reject_max_turnover_reduction_symbols: int
    reject_max_drawdown_nonworse_symbols: int
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
        for item in fields(self):
            expected = _SPEC_VALUES[item.name]
            if not _strict_equal(getattr(self, item.name), expected):
                raise ValueError(
                    f"frozen evaluation field {item.name} does not match preregistration"
                )

    def to_payload(self) -> dict[str, object]:
        return {
            item.name: _json_value(getattr(self, item.name)) for item in fields(self)
        }

    @property
    def digest(self) -> str:
        return content_digest(self.to_payload())


@dataclass(frozen=True, slots=True)
class RidgeEconomicGateCostAuthority:
    """Separate PRE-P&L proof that the frozen explicit cost is constant."""

    source_run_id: int
    source_artifact_id: int
    source_artifact_api_digest: str
    dataset_id: str
    dataset_artifact_digest: str
    evaluation_start: str
    evaluation_stop_exclusive: str
    fee_rate: float
    taker_fee_rate: float
    spread_rate: float
    one_way_explicit_cost: float
    verified: bool
    schema_version: str = _COST_AUTHORITY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        spec = canonical_ridge_economic_gate_evaluation_spec()
        for field_name in ("source_run_id", "source_artifact_id"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{field_name} must be a positive integer")
        _require_hex(
            self.source_artifact_api_digest, field="source Artifact API digest"
        )
        if self.schema_version != _COST_AUTHORITY_SCHEMA_VERSION:
            raise ValueError("unsupported cost authority schema")
        if self.dataset_id != spec.dataset_id:
            raise ValueError("cost authority dataset does not match frozen evaluation")
        if self.dataset_artifact_digest != spec.dataset_artifact_digest:
            raise ValueError(
                "cost authority dataset artifact does not match frozen evaluation"
            )
        if (
            self.evaluation_start != spec.evaluation_start
            or self.evaluation_stop_exclusive != spec.evaluation_stop_exclusive
        ):
            raise ValueError(
                "cost authority evaluation clock does not match frozen evaluation"
            )
        for field_name in (
            "fee_rate",
            "taker_fee_rate",
            "spread_rate",
            "one_way_explicit_cost",
        ):
            value = getattr(self, field_name)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{field_name} must be finite and non-negative")
            if value != getattr(spec, field_name):
                raise ValueError(
                    f"cost authority {field_name} differs from frozen evaluation"
                )
        if (
            self.one_way_explicit_cost
            != self.fee_rate + self.taker_fee_rate + self.spread_rate
        ):
            raise ValueError(
                "one_way_explicit_cost must equal explicit cost components"
            )
        if type(self.verified) is not bool or not self.verified:
            raise ValueError("cost authority must be verified before evaluation")

    def to_payload(self) -> dict[str, object]:
        return {item.name: getattr(self, item.name) for item in fields(self)}

    @property
    def digest(self) -> str:
        return content_digest(self.to_payload())


@dataclass(frozen=True, slots=True)
class RidgeEconomicGateSymbolResult:
    """One symbol's independently replayed baseline/candidate evidence."""

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
        if not isinstance(self.symbol, str) or not self.symbol:
            raise ValueError("symbol must be non-empty")
        numeric = (
            self.baseline_total_return,
            self.candidate_total_return,
            self.excess_total_return,
            self.baseline_total_cost,
            self.candidate_total_cost,
            self.baseline_turnover_total,
            self.candidate_turnover_total,
            self.baseline_max_drawdown,
            self.candidate_max_drawdown,
        )
        if not all(math.isfinite(value) for value in numeric):
            raise ValueError("symbol evaluation metrics must be finite")
        if (
            self.excess_total_return
            != self.candidate_total_return - self.baseline_total_return
        ):
            raise ValueError("excess_total_return must equal candidate minus baseline")
        for field_name in (
            "baseline_total_cost",
            "candidate_total_cost",
            "baseline_turnover_total",
            "candidate_turnover_total",
            "baseline_max_drawdown",
            "candidate_max_drawdown",
        ):
            if getattr(self, field_name) < 0.0:
                raise ValueError(f"{field_name} must be non-negative")
        for field_name in (
            "baseline_termination_count",
            "candidate_termination_count",
            "baseline_n_periods",
            "candidate_n_periods",
        ):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{field_name} must be a non-negative integer")
        if self.baseline_termination_count != len(self.baseline_termination_reasons):
            raise ValueError("baseline termination count/reasons mismatch")
        if self.candidate_termination_count != len(self.candidate_termination_reasons):
            raise ValueError("candidate termination count/reasons mismatch")
        for reasons in (
            self.baseline_termination_reasons,
            self.candidate_termination_reasons,
        ):
            if any(not isinstance(reason, str) or not reason for reason in reasons):
                raise ValueError("termination reasons must contain non-empty strings")
        if type(self.new_termination) is not bool:
            raise ValueError("new_termination must be a bool")
        expected_new = _has_new_termination(
            self.baseline_termination_reasons,
            self.candidate_termination_reasons,
            baseline_count=self.baseline_termination_count,
            candidate_count=self.candidate_termination_count,
        )
        if self.new_termination is not expected_new:
            raise ValueError("new_termination does not match termination evidence")
        _require_hex(self.baseline_return_sha256, field="baseline return SHA-256")
        _require_hex(self.candidate_return_sha256, field="candidate return SHA-256")

    def to_payload(self) -> dict[str, object]:
        return {
            "symbol": self.symbol,
            "baseline_total_return": self.baseline_total_return,
            "candidate_total_return": self.candidate_total_return,
            "excess_total_return": self.excess_total_return,
            "baseline_total_cost": self.baseline_total_cost,
            "candidate_total_cost": self.candidate_total_cost,
            "baseline_turnover_total": self.baseline_turnover_total,
            "candidate_turnover_total": self.candidate_turnover_total,
            "baseline_max_drawdown": self.baseline_max_drawdown,
            "candidate_max_drawdown": self.candidate_max_drawdown,
            "baseline_termination_count": self.baseline_termination_count,
            "candidate_termination_count": self.candidate_termination_count,
            "baseline_termination_reasons": list(self.baseline_termination_reasons),
            "candidate_termination_reasons": list(self.candidate_termination_reasons),
            "new_termination": self.new_termination,
            "baseline_n_periods": self.baseline_n_periods,
            "candidate_n_periods": self.candidate_n_periods,
            "baseline_return_sha256": self.baseline_return_sha256,
            "candidate_return_sha256": self.candidate_return_sha256,
        }


@dataclass(frozen=True, slots=True)
class RidgeEconomicGateEvaluation:
    """Strict aggregate evidence for the frozen mechanism-level decision rule."""

    spec_digest: str
    cost_authority_digest: str
    dataset_id: str
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
    final_test_accessed: bool = False
    final_test_authorized: bool = False
    shared_cash_profitability_established: bool = False
    operational_eligibility_established: bool = False
    production_eligible: bool = False
    live_trading_authorized: bool = False
    merge_authorized: bool = False
    schema_version: str = _RESULT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        spec = canonical_ridge_economic_gate_evaluation_spec()
        if self.schema_version != _RESULT_SCHEMA_VERSION:
            raise ValueError("unsupported ridge economic-gate result schema")
        if self.spec_digest != spec.digest:
            raise ValueError("spec_digest differs from frozen evaluation authority")
        _require_hex(self.cost_authority_digest, field="cost authority digest")
        if self.dataset_id != spec.dataset_id:
            raise ValueError("dataset_id differs from frozen evaluation authority")
        if (
            self.symbols != spec.symbols
            or tuple(item.symbol for item in self.by_symbol) != self.symbols
        ):
            raise ValueError("ridge economic-gate result symbol roster mismatch")
        if len(self.by_symbol) != len(spec.symbols):
            raise ValueError("ridge economic-gate result requires all frozen symbols")
        if not math.isfinite(self.median_excess_total_return):
            raise ValueError("median_excess_total_return must be finite")
        for field_name in (
            "positive_effect_symbols",
            "cost_reduction_symbols",
            "turnover_reduction_symbols",
            "drawdown_nonworse_symbols",
            "new_termination_symbols",
            "candidate_positive_total_return_symbols",
        ):
            value = getattr(self, field_name)
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or not 0 <= value <= len(spec.symbols)
            ):
                raise ValueError(f"{field_name} must be an integer within [0, 5]")

        expected_positive = sum(
            item.excess_total_return > 0.0 for item in self.by_symbol
        )
        expected_median = float(
            median(item.excess_total_return for item in self.by_symbol)
        )
        expected_cost = sum(
            item.candidate_total_cost < item.baseline_total_cost
            for item in self.by_symbol
        )
        expected_turnover = sum(
            item.candidate_turnover_total < item.baseline_turnover_total
            for item in self.by_symbol
        )
        expected_drawdown = sum(
            item.candidate_max_drawdown <= item.baseline_max_drawdown
            for item in self.by_symbol
        )
        expected_new_termination = sum(item.new_termination for item in self.by_symbol)
        expected_candidate_positive = sum(
            item.candidate_total_return > 0.0 for item in self.by_symbol
        )
        expected_status = research_status_from_counts(
            positive_effect_symbols=expected_positive,
            median_excess_total_return=expected_median,
            cost_reduction_symbols=expected_cost,
            turnover_reduction_symbols=expected_turnover,
            drawdown_nonworse_symbols=expected_drawdown,
            new_termination_symbols=expected_new_termination,
        )
        expected_fields: dict[str, object] = {
            "positive_effect_symbols": expected_positive,
            "median_excess_total_return": expected_median,
            "cost_reduction_symbols": expected_cost,
            "turnover_reduction_symbols": expected_turnover,
            "drawdown_nonworse_symbols": expected_drawdown,
            "new_termination_symbols": expected_new_termination,
            "candidate_positive_total_return_symbols": expected_candidate_positive,
            "research_status": expected_status,
        }
        for field_name, expected in expected_fields.items():
            if getattr(self, field_name) != expected:
                raise ValueError(
                    f"{field_name} does not match per-symbol evaluation evidence"
                )
        for field_name in (
            "final_test_accessed",
            "final_test_authorized",
            "shared_cash_profitability_established",
            "operational_eligibility_established",
            "production_eligible",
            "live_trading_authorized",
            "merge_authorized",
        ):
            value = getattr(self, field_name)
            if type(value) is not bool or value:
                raise ValueError(
                    "research evaluation cannot authorize production/final/shared-cash use"
                )

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "spec_digest": self.spec_digest,
            "cost_authority_digest": self.cost_authority_digest,
            "dataset_id": self.dataset_id,
            "symbols": list(self.symbols),
            "by_symbol": [item.to_payload() for item in self.by_symbol],
            "positive_effect_symbols": self.positive_effect_symbols,
            "median_excess_total_return": self.median_excess_total_return,
            "cost_reduction_symbols": self.cost_reduction_symbols,
            "turnover_reduction_symbols": self.turnover_reduction_symbols,
            "drawdown_nonworse_symbols": self.drawdown_nonworse_symbols,
            "new_termination_symbols": self.new_termination_symbols,
            "candidate_positive_total_return_symbols": self.candidate_positive_total_return_symbols,
            "research_status": self.research_status,
            "final_test_accessed": self.final_test_accessed,
            "final_test_authorized": self.final_test_authorized,
            "shared_cash_profitability_established": self.shared_cash_profitability_established,
            "operational_eligibility_established": self.operational_eligibility_established,
            "production_eligible": self.production_eligible,
            "live_trading_authorized": self.live_trading_authorized,
            "merge_authorized": self.merge_authorized,
        }

    @property
    def digest(self) -> str:
        return content_digest(self.to_payload())


def canonical_ridge_economic_gate_evaluation_spec() -> RidgeEconomicGateEvaluationSpec:
    protocol = canonical_ridge_economic_gate_protocol()
    if protocol.digest != _SPEC_VALUES["protocol_digest"]:
        raise ValueError(
            "sealed ridge protocol digest differs from evaluation authority"
        )
    return RidgeEconomicGateEvaluationSpec(**_SPEC_VALUES)  # type: ignore[arg-type]


def research_status_from_counts(
    *,
    positive_effect_symbols: int,
    median_excess_total_return: float,
    cost_reduction_symbols: int,
    turnover_reduction_symbols: int,
    drawdown_nonworse_symbols: int,
    new_termination_symbols: int,
) -> str:
    """Replay the frozen mechanism-level PROMOTE/REJECT/INCONCLUSIVE rule."""

    spec = canonical_ridge_economic_gate_evaluation_spec()
    if not math.isfinite(median_excess_total_return):
        raise ValueError("median_excess_total_return must be finite")
    counts = (
        positive_effect_symbols,
        cost_reduction_symbols,
        turnover_reduction_symbols,
        drawdown_nonworse_symbols,
        new_termination_symbols,
    )
    if any(
        isinstance(value, bool)
        or not isinstance(value, int)
        or not 0 <= value <= len(spec.symbols)
        for value in counts
    ):
        raise ValueError("research decision counts must be integers within [0, 5]")
    if (
        positive_effect_symbols <= spec.reject_max_positive_effect_symbols
        or median_excess_total_return <= 0.0
        or cost_reduction_symbols <= spec.reject_max_cost_reduction_symbols
        or turnover_reduction_symbols <= spec.reject_max_turnover_reduction_symbols
        or drawdown_nonworse_symbols <= spec.reject_max_drawdown_nonworse_symbols
        or new_termination_symbols > 0
    ):
        return "REJECT_MECHANISM"
    if (
        positive_effect_symbols >= spec.promote_min_positive_effect_symbols
        and median_excess_total_return > 0.0
        and cost_reduction_symbols >= spec.promote_min_cost_reduction_symbols
        and turnover_reduction_symbols >= spec.promote_min_turnover_reduction_symbols
        and drawdown_nonworse_symbols >= spec.promote_min_drawdown_nonworse_symbols
        and new_termination_symbols == 0
    ):
        return "PROMOTE_RESEARCH_REFERENCE"
    return "INCONCLUSIVE"


def _exact_index(dataset: MarketDataset, timestamp: str, *, field: str) -> int:
    target = np.datetime64(timestamp, "ns")
    values = np.asarray(dataset.timestamps, dtype="datetime64[ns]")
    matches = np.flatnonzero(values == target)
    if matches.size != 1:
        raise ValueError(f"{field} must exactly match one Dataset timestamp")
    return int(matches[0])


def _validate_dataset(
    dataset: MarketDataset,
    spec: RidgeEconomicGateEvaluationSpec,
) -> tuple[int, int]:
    if dataset.dataset_id != spec.dataset_id:
        raise ValueError("dataset identity differs from frozen evaluation")
    if tuple(dataset.symbols) != spec.symbols:
        raise ValueError("dataset symbol roster differs from frozen evaluation")
    for name, index in zip(spec.feature_names, spec.feature_indices, strict=True):
        if index >= len(dataset.feature_names) or dataset.feature_names[index] != name:
            raise ValueError("ridge feature identity differs from frozen evaluation")
    start = _exact_index(dataset, spec.evaluation_start, field="evaluation_start")
    stop = _exact_index(
        dataset, spec.evaluation_stop_exclusive, field="evaluation_stop_exclusive"
    )
    if not 0 <= start < stop < dataset.n_bars:
        raise ValueError("evaluation range differs from frozen evaluation")
    execution_slice = slice(start, stop + 1)
    expected_shape = (stop + 1 - start, len(spec.symbols))
    for field_name, expected in (
        ("fee_rate", spec.fee_rate),
        ("taker_fee_rate", spec.taker_fee_rate),
        ("spread_rate", spec.spread_rate),
    ):
        values = np.asarray(
            dataset.resolved_array(field_name)[execution_slice], dtype=np.float64
        )
        if (
            values.shape != expected_shape
            or not np.isfinite(values).all()
            or np.any(values < 0.0)
            or not np.all(values == expected)
        ):
            raise ValueError(f"evaluation cost drift: {field_name}")
    return start, stop


def evaluation_return_sha256(values: object) -> str:
    """Stable exact-byte digest for one finite one-dimensional float64 return path."""

    array = np.ascontiguousarray(np.asarray(values, dtype=np.float64))
    if array.ndim != 1:
        raise ValueError("evaluation returns must be one-dimensional")
    if not np.isfinite(array).all():
        raise ValueError("evaluation returns must be finite")
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(repr(array.shape).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _has_new_termination(
    baseline_reasons: tuple[str, ...],
    candidate_reasons: tuple[str, ...],
    *,
    baseline_count: int,
    candidate_count: int,
) -> bool:
    if candidate_count > baseline_count:
        return True
    baseline_counter = Counter(baseline_reasons)
    candidate_counter = Counter(candidate_reasons)
    return any(
        candidate_counter[reason] > baseline_counter[reason]
        for reason in candidate_counter
    )


def _validate_cost_authority(
    authority: RidgeEconomicGateCostAuthority | None,
    spec: RidgeEconomicGateEvaluationSpec,
) -> RidgeEconomicGateCostAuthority:
    if not isinstance(authority, RidgeEconomicGateCostAuthority):
        raise ValueError("verified PRE-P&L cost authority is required")
    if (
        authority.dataset_id != spec.dataset_id
        or authority.dataset_artifact_digest != spec.dataset_artifact_digest
        or authority.evaluation_start != spec.evaluation_start
        or authority.evaluation_stop_exclusive != spec.evaluation_stop_exclusive
        or authority.fee_rate != spec.fee_rate
        or authority.taker_fee_rate != spec.taker_fee_rate
        or authority.spread_rate != spec.spread_rate
        or authority.one_way_explicit_cost != spec.one_way_explicit_cost
        or not authority.verified
    ):
        raise ValueError("cost authority differs from frozen evaluation")
    return authority


def evaluate_ridge_economic_gate(
    dataset: MarketDataset,
    spec: RidgeEconomicGateEvaluationSpec,
    cost_authority: RidgeEconomicGateCostAuthority,
) -> RidgeEconomicGateEvaluation:
    """Build one canonical Ridge model and replay baseline/candidate pairwise."""

    canonical = canonical_ridge_economic_gate_evaluation_spec()
    if spec != canonical or spec.digest != canonical.digest:
        raise ValueError("evaluation spec differs from frozen ridge evaluation")
    authority = _validate_cost_authority(cost_authority, spec)
    start, stop = _validate_dataset(dataset, spec)

    fit_symbol_indices = tuple(range(len(spec.symbols)))
    model = fit_ridge_forecast(
        dataset,
        feature_indices=spec.feature_indices,
        fit_symbol_indices=fit_symbol_indices,
        fit_cutoff=np.datetime64(spec.fit_cutoff, "ns"),
        horizon_hours=spec.ridge_horizon_hours,
        alpha=spec.ridge_alpha,
    )
    if (
        model.feature_indices != spec.feature_indices
        or model.horizon_hours != spec.ridge_horizon_hours
        or model.alpha != spec.ridge_alpha
        or model.fit_cutoff != np.datetime64(spec.fit_cutoff, "ns")
    ):
        raise RuntimeError("fitted Ridge model identity drifted from frozen evaluation")

    baseline = RidgeForecastStrategy(
        model,
        entry_threshold=spec.forecast_entry_threshold,
        exit_threshold=spec.forecast_exit_threshold,
    )
    candidate = RidgeEconomicGateStrategy(
        RidgeEconomicGateConfig(
            model=model,
            entry_threshold=spec.forecast_entry_threshold,
            exit_threshold=spec.forecast_exit_threshold,
            one_way_explicit_cost=spec.one_way_explicit_cost,
        )
    )
    if (
        baseline.model is not model
        or candidate.model is not model
        or candidate.config.model is not model
    ):
        raise RuntimeError(
            "baseline and candidate must share the exact fitted Ridge model"
        )
    strategies: dict[str, SingleSymbolStrategy] = {
        "baseline": baseline,
        "candidate": candidate,
    }
    comparison = compare_strategies_by_symbol(
        dataset,
        strategies,
        start_index=start,
        stop_index=stop,
        gross_budget=spec.gross_budget,
        initial_capital=spec.initial_capital,
        execution_cost=execution_cost_for_overlay(spec.execution_overlay),
        risk=None,
    )

    results: list[RidgeEconomicGateSymbolResult] = []
    for symbol_result in comparison.by_symbol:
        entries = {entry.name: entry for entry in symbol_result.comparison.entries}
        if set(entries) != {"baseline", "candidate"}:
            raise RuntimeError("paired ridge evaluation strategy roster drifted")
        baseline_entry = entries["baseline"]
        candidate_entry = entries["candidate"]
        baseline_returns = tuple(
            float(value) for value in baseline_entry.replay.returns.values
        )
        candidate_returns = tuple(
            float(value) for value in candidate_entry.replay.returns.values
        )
        if baseline_entry.metrics.n_periods != len(baseline_returns):
            raise RuntimeError("baseline return count differs from metrics")
        if candidate_entry.metrics.n_periods != len(candidate_returns):
            raise RuntimeError("candidate return count differs from metrics")
        if baseline_entry.metrics.total_return != compound_return(baseline_returns):
            raise RuntimeError("baseline total return differs from raw return path")
        if candidate_entry.metrics.total_return != compound_return(candidate_returns):
            raise RuntimeError("candidate total return differs from raw return path")
        baseline_reasons = tuple(baseline_entry.replay.diagnostics.termination_reasons)
        candidate_reasons = tuple(
            candidate_entry.replay.diagnostics.termination_reasons
        )
        if baseline_entry.metrics.termination_count != len(baseline_reasons):
            raise RuntimeError("baseline termination evidence is inconsistent")
        if candidate_entry.metrics.termination_count != len(candidate_reasons):
            raise RuntimeError("candidate termination evidence is inconsistent")
        results.append(
            RidgeEconomicGateSymbolResult(
                symbol=symbol_result.symbol,
                baseline_total_return=baseline_entry.metrics.total_return,
                candidate_total_return=candidate_entry.metrics.total_return,
                excess_total_return=(
                    candidate_entry.metrics.total_return
                    - baseline_entry.metrics.total_return
                ),
                baseline_total_cost=baseline_entry.metrics.total_cost,
                candidate_total_cost=candidate_entry.metrics.total_cost,
                baseline_turnover_total=baseline_entry.metrics.turnover_total,
                candidate_turnover_total=candidate_entry.metrics.turnover_total,
                baseline_max_drawdown=baseline_entry.metrics.max_drawdown,
                candidate_max_drawdown=candidate_entry.metrics.max_drawdown,
                baseline_termination_count=baseline_entry.metrics.termination_count,
                candidate_termination_count=candidate_entry.metrics.termination_count,
                baseline_termination_reasons=baseline_reasons,
                candidate_termination_reasons=candidate_reasons,
                new_termination=_has_new_termination(
                    baseline_reasons,
                    candidate_reasons,
                    baseline_count=baseline_entry.metrics.termination_count,
                    candidate_count=candidate_entry.metrics.termination_count,
                ),
                baseline_n_periods=baseline_entry.metrics.n_periods,
                candidate_n_periods=candidate_entry.metrics.n_periods,
                baseline_return_sha256=evaluation_return_sha256(baseline_returns),
                candidate_return_sha256=evaluation_return_sha256(candidate_returns),
            )
        )
    if tuple(item.symbol for item in results) != spec.symbols:
        raise RuntimeError("paired ridge evaluation output symbol roster drifted")
    if any(item.baseline_n_periods != item.candidate_n_periods for item in results):
        raise RuntimeError("paired ridge evaluation return horizon drifted")

    positive = sum(item.excess_total_return > 0.0 for item in results)
    excess_median = float(median(item.excess_total_return for item in results))
    cost_reduction = sum(
        item.candidate_total_cost < item.baseline_total_cost for item in results
    )
    turnover_reduction = sum(
        item.candidate_turnover_total < item.baseline_turnover_total for item in results
    )
    drawdown_nonworse = sum(
        item.candidate_max_drawdown <= item.baseline_max_drawdown for item in results
    )
    new_termination = sum(item.new_termination for item in results)
    candidate_positive = sum(item.candidate_total_return > 0.0 for item in results)
    status = research_status_from_counts(
        positive_effect_symbols=positive,
        median_excess_total_return=excess_median,
        cost_reduction_symbols=cost_reduction,
        turnover_reduction_symbols=turnover_reduction,
        drawdown_nonworse_symbols=drawdown_nonworse,
        new_termination_symbols=new_termination,
    )
    return RidgeEconomicGateEvaluation(
        spec_digest=spec.digest,
        cost_authority_digest=authority.digest,
        dataset_id=dataset.dataset_id,
        symbols=spec.symbols,
        by_symbol=tuple(results),
        positive_effect_symbols=positive,
        median_excess_total_return=excess_median,
        cost_reduction_symbols=cost_reduction,
        turnover_reduction_symbols=turnover_reduction,
        drawdown_nonworse_symbols=drawdown_nonworse,
        new_termination_symbols=new_termination,
        candidate_positive_total_return_symbols=candidate_positive,
        research_status=status,
    )


__all__ = [
    "RidgeEconomicGateCostAuthority",
    "RidgeEconomicGateEvaluation",
    "RidgeEconomicGateEvaluationSpec",
    "RidgeEconomicGateSymbolResult",
    "canonical_ridge_economic_gate_evaluation_spec",
    "evaluate_ridge_economic_gate",
    "evaluation_return_sha256",
    "research_status_from_counts",
]
