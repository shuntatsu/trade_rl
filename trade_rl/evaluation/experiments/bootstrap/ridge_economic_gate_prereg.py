"""Result-blind preregistration for the Issue #616 ridge24 economic gate."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, fields
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from trade_rl.artifacts.hashing import content_digest

_SCHEMA_VERSION = "ridge_economic_gate_prereg_v1"
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
_POSITION_VALUES = (("LONG", 1), ("FLAT", 0), ("SHORT", -1))
_UNAFFECTED_STRATEGIES = (
    "cash",
    "constant_long",
    "constant_short",
    "trend",
    "mean_reversion",
    "lightgbm24",
    "ppo",
)

_CANONICAL_FIELD_VALUES: dict[str, object] = {
    "schema_version": _SCHEMA_VERSION,
    "issue_number": 616,
    "parent_roadmap_issue": 604,
    "diagnosis_issue": 543,
    "diagnosis_pr": 544,
    "diagnosis_head_sha": "9f3fcc3c84820b7aca25719777427aa37b90256e",
    "diagnosis_report_digest": "9b300bf8c6d6ac4b8230a6179c4988d8c7bac2f3988e9c0d823a9ed95361fef0",
    "diagnosis_verification_run_id": 34812564402,
    "diagnosis_verification_artifact_id": 10335845094,
    "diagnosis_verification_artifact_digest": "e722c27e2b3903ae91ad81b1d244fdbb85e7deba093e0a0db0dcaf644a05d6e9",
    "successor_bundle_run_id": 34803217815,
    "successor_bundle_artifact_id": 10331899302,
    "successor_bundle_artifact_digest": "89e899427f23fa46929c8be1e71fd49abe0d1d465c7a7f796a0874426b885bce",
    "successor_dataset_id": "6c0b040d317a1bb37b811c3e071f325fa007525a6040927e6793d8cc7c10f538f",
    "successor_dataset_artifact_digest": "af481dd978db7d84cd3aa8ff4f5a35d8608ac44c755dd74f61e934105c02b6b7",
    "successor_study_digest": "bfa2fcb307773f5384d7dcb884444164d6d3b575373d8f7dc1c810a61bf4c820",
    "successor_plan_sha256": "bfa2fcb307773f5384d7dcb884444164d6d3b575373d8f7dc1c810a61bf4c820",
    "successor_execution_overlay": "zero_overlay_dataset_fields_authoritative_previous_completed_bar_capacity",
    "symbols": _SYMBOLS,
    "feature_names": _FEATURE_NAMES,
    "feature_indices": _FEATURE_INDICES,
    "fit_symbol_names": _SYMBOLS,
    "fit_cutoff": datetime(2023, 1, 1, tzinfo=UTC),
    "evaluation_start": datetime(2023, 1, 1, tzinfo=UTC),
    "evaluation_stop_exclusive": datetime(2025, 1, 1, tzinfo=UTC),
    "ridge_horizon_hours": 24,
    "ridge_alpha": 1.0,
    "forecast_entry_threshold": 0.0025,
    "forecast_exit_threshold": 0.0005,
    "gross_budget": 0.5,
    "initial_capital": 100_000.0,
    "target_strategy": "ridge24",
    "proposal_model": "canonical_pooled_ridge24",
    "additional_training_calibration_required": False,
    "position_values": _POSITION_VALUES,
    "edge_formula": "delta_position * ridge_forecast",
    "one_way_cost_formula": "fee_rate + taker_fee_rate + spread_rate",
    "transition_cost_formula": "abs(delta_position) * one_way_cost",
    "gate_operator": ">",
    "equality_action": "HOLD_CURRENT",
    "unavailable_feature_action": "FLAT_BYPASS_GATE",
    "hard_risk_overrides_gate": True,
    "model_unchanged": True,
    "features_unchanged": True,
    "thresholds_unchanged": True,
    "horizon_unchanged": True,
    "symbol_scope_unchanged": True,
    "impact_slippage_invented_by_gate": False,
    "market_order_fee_rate": 0.0005,
    "market_order_taker_fee_rate": 0.0,
    "market_order_spread_rate": 0.0002,
    "require_full_evaluation_cost_constancy": True,
    "future_row_economics_read_by_strategy": False,
    "unaffected_strategies": _UNAFFECTED_STRATEGIES,
    "research_promote_status": "PROMOTE_RESEARCH_REFERENCE",
    "reject_status": "REJECT_MECHANISM",
    "inconclusive_status": "INCONCLUSIVE",
    "invalid_cost_drift_status": "INVALID_EVALUATION_COST_DRIFT",
    "promote_min_positive_effect_symbols": 4,
    "promote_requires_positive_median_excess": True,
    "promote_min_cost_reduction_symbols": 4,
    "promote_min_turnover_reduction_symbols": 4,
    "promote_min_drawdown_nonworse_symbols": 4,
    "promote_requires_no_new_termination": True,
    "reject_max_positive_effect_symbols": 2,
    "reject_on_nonpositive_median_excess": True,
    "reject_max_cost_reduction_symbols": 2,
    "reject_max_turnover_reduction_symbols": 2,
    "reject_max_drawdown_nonworse_symbols": 2,
    "absolute_positive_symbol_count_is_decision_input": False,
    "one_slot_only": True,
    "post_result_variant_allowed": False,
    "development_data_already_used": True,
    "final_test_accessed": False,
    "final_test_authorized": False,
    "shared_cash_profitability_established": False,
    "operational_eligibility_established": False,
    "production_eligible": False,
    "live_trading_authorized": False,
    "merge_authorized": False,
    "evaluation_pnl_inspected": False,
    "evaluation_execution_authorized": False,
}


def _require_finite(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be finite")
    resolved = float(value)
    if not math.isfinite(resolved):
        raise ValueError(f"{field} must be finite")
    return resolved


def _require_aware(value: datetime, *, field: str) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field} must be timezone-aware")


def _datetime_to_text(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _datetime_from_text(value: object, *, field: str) -> datetime:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a timezone-aware datetime")
    try:
        resolved = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError(f"{field} must be a timezone-aware datetime") from error
    _require_aware(resolved, field=field)
    return resolved.astimezone(UTC)


def _require_hex(value: object, *, field: str, length: int) -> str:
    if not isinstance(value, str) or len(value) != length:
        raise ValueError(f"{field} must be a {length}-character lowercase hex digest")
    if value.lower() != value or any(char not in "0123456789abcdef" for char in value):
        raise ValueError(f"{field} must be a {length}-character lowercase hex digest")
    return value


def _json_value(value: object) -> object:
    if isinstance(value, datetime):
        return _datetime_to_text(value)
    if isinstance(value, tuple):
        return [_json_value(item) for item in value]
    return value


@dataclass(frozen=True, slots=True)
class RidgeEconomicGateProtocol:
    """Immutable result-blind protocol for the single Issue #616 experiment slot."""

    schema_version: str
    issue_number: int
    parent_roadmap_issue: int
    diagnosis_issue: int
    diagnosis_pr: int
    diagnosis_head_sha: str
    diagnosis_report_digest: str
    diagnosis_verification_run_id: int
    diagnosis_verification_artifact_id: int
    diagnosis_verification_artifact_digest: str
    successor_bundle_run_id: int
    successor_bundle_artifact_id: int
    successor_bundle_artifact_digest: str
    successor_dataset_id: str
    successor_dataset_artifact_digest: str
    successor_study_digest: str
    successor_plan_sha256: str
    successor_execution_overlay: str
    symbols: tuple[str, ...]
    feature_names: tuple[str, ...]
    feature_indices: tuple[int, ...]
    fit_symbol_names: tuple[str, ...]
    fit_cutoff: datetime
    evaluation_start: datetime
    evaluation_stop_exclusive: datetime
    ridge_horizon_hours: int
    ridge_alpha: float
    forecast_entry_threshold: float
    forecast_exit_threshold: float
    gross_budget: float
    initial_capital: float
    target_strategy: str
    proposal_model: str
    additional_training_calibration_required: bool
    position_values: tuple[tuple[str, int], ...]
    edge_formula: str
    one_way_cost_formula: str
    transition_cost_formula: str
    gate_operator: str
    equality_action: str
    unavailable_feature_action: str
    hard_risk_overrides_gate: bool
    model_unchanged: bool
    features_unchanged: bool
    thresholds_unchanged: bool
    horizon_unchanged: bool
    symbol_scope_unchanged: bool
    impact_slippage_invented_by_gate: bool
    market_order_fee_rate: float
    market_order_taker_fee_rate: float
    market_order_spread_rate: float
    require_full_evaluation_cost_constancy: bool
    future_row_economics_read_by_strategy: bool
    unaffected_strategies: tuple[str, ...]
    research_promote_status: str
    reject_status: str
    inconclusive_status: str
    invalid_cost_drift_status: str
    promote_min_positive_effect_symbols: int
    promote_requires_positive_median_excess: bool
    promote_min_cost_reduction_symbols: int
    promote_min_turnover_reduction_symbols: int
    promote_min_drawdown_nonworse_symbols: int
    promote_requires_no_new_termination: bool
    reject_max_positive_effect_symbols: int
    reject_on_nonpositive_median_excess: bool
    reject_max_cost_reduction_symbols: int
    reject_max_turnover_reduction_symbols: int
    reject_max_drawdown_nonworse_symbols: int
    absolute_positive_symbol_count_is_decision_input: bool
    one_slot_only: bool
    post_result_variant_allowed: bool
    development_data_already_used: bool
    final_test_accessed: bool
    final_test_authorized: bool
    shared_cash_profitability_established: bool
    operational_eligibility_established: bool
    production_eligible: bool
    live_trading_authorized: bool
    merge_authorized: bool
    evaluation_pnl_inspected: bool
    evaluation_execution_authorized: bool

    def __post_init__(self) -> None:
        for field_name in (
            "fit_cutoff",
            "evaluation_start",
            "evaluation_stop_exclusive",
        ):
            _require_aware(getattr(self, field_name), field=field_name)
        if (
            not self.fit_cutoff
            == self.evaluation_start
            < self.evaluation_stop_exclusive
        ):
            raise ValueError("preregistered research clock is invalid")

        if len(self.feature_names) != len(self.feature_indices):
            raise ValueError("preregistered feature roster is invalid")
        if not self.feature_indices or len(set(self.feature_indices)) != len(
            self.feature_indices
        ):
            raise ValueError("preregistered feature indices are invalid")
        if any(
            isinstance(index, bool) or not isinstance(index, int) or index < 0
            for index in self.feature_indices
        ):
            raise ValueError("preregistered feature indices are invalid")
        if self.fit_symbol_names != self.symbols:
            raise ValueError("preregistered fit symbol roster is invalid")

        for field_name in (
            "ridge_alpha",
            "forecast_entry_threshold",
            "forecast_exit_threshold",
            "gross_budget",
            "initial_capital",
            "market_order_fee_rate",
            "market_order_taker_fee_rate",
            "market_order_spread_rate",
        ):
            resolved = _require_finite(getattr(self, field_name), field=field_name)
            if resolved < 0.0:
                raise ValueError(f"{field_name} must be non-negative")
        if self.ridge_alpha <= 0.0:
            raise ValueError("preregistered ridge alpha is invalid")
        if self.forecast_entry_threshold <= self.forecast_exit_threshold:
            raise ValueError("preregistered forecast thresholds are invalid")
        if self.gross_budget <= 0.0 or self.initial_capital <= 0.0:
            raise ValueError("preregistered capital/risk budget is invalid")

        for field_name in (
            "issue_number",
            "parent_roadmap_issue",
            "diagnosis_issue",
            "diagnosis_pr",
            "diagnosis_verification_run_id",
            "diagnosis_verification_artifact_id",
            "successor_bundle_run_id",
            "successor_bundle_artifact_id",
            "ridge_horizon_hours",
            "promote_min_positive_effect_symbols",
            "promote_min_cost_reduction_symbols",
            "promote_min_turnover_reduction_symbols",
            "promote_min_drawdown_nonworse_symbols",
            "reject_max_positive_effect_symbols",
            "reject_max_cost_reduction_symbols",
            "reject_max_turnover_reduction_symbols",
            "reject_max_drawdown_nonworse_symbols",
        ):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{field_name} must be a positive integer")

        for field_name in (
            "additional_training_calibration_required",
            "hard_risk_overrides_gate",
            "model_unchanged",
            "features_unchanged",
            "thresholds_unchanged",
            "horizon_unchanged",
            "symbol_scope_unchanged",
            "impact_slippage_invented_by_gate",
            "require_full_evaluation_cost_constancy",
            "future_row_economics_read_by_strategy",
            "promote_requires_positive_median_excess",
            "promote_requires_no_new_termination",
            "reject_on_nonpositive_median_excess",
            "absolute_positive_symbol_count_is_decision_input",
            "one_slot_only",
            "post_result_variant_allowed",
            "development_data_already_used",
            "final_test_accessed",
            "final_test_authorized",
            "shared_cash_profitability_established",
            "operational_eligibility_established",
            "production_eligible",
            "live_trading_authorized",
            "merge_authorized",
            "evaluation_pnl_inspected",
            "evaluation_execution_authorized",
        ):
            if type(getattr(self, field_name)) is not bool:
                raise ValueError(f"{field_name} must be a boolean")

        _require_hex(self.diagnosis_head_sha, field="diagnosis_head_sha", length=40)
        for field_name in (
            "diagnosis_report_digest",
            "diagnosis_verification_artifact_digest",
            "successor_bundle_artifact_digest",
            "successor_dataset_id",
            "successor_dataset_artifact_digest",
            "successor_study_digest",
            "successor_plan_sha256",
        ):
            _require_hex(getattr(self, field_name), field=field_name, length=64)
        if self.successor_plan_sha256 != self.successor_study_digest:
            raise ValueError("preregistered successor plan/study binding is invalid")

        for item in fields(self):
            expected = _CANONICAL_FIELD_VALUES[item.name]
            if getattr(self, item.name) != expected:
                raise ValueError(f"{item.name} differs from preregistered value")

    @property
    def nominal_one_way_explicit_cost(self) -> float:
        return (
            self.market_order_fee_rate
            + self.market_order_taker_fee_rate
            + self.market_order_spread_rate
        )

    def to_payload(self) -> dict[str, object]:
        return {
            item.name: _json_value(getattr(self, item.name)) for item in fields(self)
        }

    @property
    def digest(self) -> str:
        return content_digest(self.to_payload())


def canonical_ridge_economic_gate_protocol() -> RidgeEconomicGateProtocol:
    """Return the only preregistered protocol authorized by Issue #616."""

    return RidgeEconomicGateProtocol(**_CANONICAL_FIELD_VALUES)  # type: ignore[arg-type]


def _tuple_strings(value: object, *, field: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError(f"{field} must be a string array")
    return tuple(value)


def _tuple_ints(value: object, *, field: str) -> tuple[int, ...]:
    if not isinstance(value, list) or any(
        isinstance(item, bool) or not isinstance(item, int) for item in value
    ):
        raise ValueError(f"{field} must be an integer array")
    return tuple(value)


def _position_values(value: object) -> tuple[tuple[str, int], ...]:
    if not isinstance(value, list):
        raise ValueError("position_values must be an array")
    result: list[tuple[str, int]] = []
    for item in value:
        if (
            not isinstance(item, list)
            or len(item) != 2
            or not isinstance(item[0], str)
            or isinstance(item[1], bool)
            or not isinstance(item[1], int)
        ):
            raise ValueError("position_values must contain [name, integer] pairs")
        result.append((item[0], item[1]))
    return tuple(result)


def load_ridge_economic_gate_protocol(path: str | Path) -> RidgeEconomicGateProtocol:
    """Load a strict Issue #616 preregistration and reject any schema drift."""

    raw: Any = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or any(not isinstance(key, str) for key in raw):
        raise ValueError("preregistration payload must be an object")
    expected_keys = {item.name for item in fields(RidgeEconomicGateProtocol)}
    if set(raw) != expected_keys:
        raise ValueError("preregistration keys differ from preregistered schema")

    converted = dict(raw)
    for field_name in (
        "symbols",
        "feature_names",
        "fit_symbol_names",
        "unaffected_strategies",
    ):
        converted[field_name] = _tuple_strings(raw[field_name], field=field_name)
    converted["feature_indices"] = _tuple_ints(
        raw["feature_indices"], field="feature_indices"
    )
    converted["position_values"] = _position_values(raw["position_values"])
    for field_name in (
        "fit_cutoff",
        "evaluation_start",
        "evaluation_stop_exclusive",
    ):
        converted[field_name] = _datetime_from_text(raw[field_name], field=field_name)
    for field_name in (
        "ridge_alpha",
        "forecast_entry_threshold",
        "forecast_exit_threshold",
        "gross_budget",
        "initial_capital",
        "market_order_fee_rate",
        "market_order_taker_fee_rate",
        "market_order_spread_rate",
    ):
        converted[field_name] = _require_finite(raw[field_name], field=field_name)
    return RidgeEconomicGateProtocol(**converted)


__all__ = [
    "RidgeEconomicGateProtocol",
    "canonical_ridge_economic_gate_protocol",
    "load_ridge_economic_gate_protocol",
]
