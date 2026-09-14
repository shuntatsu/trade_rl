"""Sealed preregistration for a training-only mean-reversion economic action gate."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, fields
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from trade_rl.artifacts.hashing import content_digest

_SCHEMA_VERSION = "mean_reversion_economic_gate_prereg_v1"
_SYMBOLS = ("BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT")
_CAPACITY_CAPS = (
    0.0021629560553901974,
    0.002044685341258238,
    0.002184898995567895,
    0.0020480213652913385,
    0.002346308308284808,
)
_POSITION_VALUES = (("LONG", 1), ("FLAT", 0), ("SHORT", -1))
_UNAFFECTED_STRATEGIES = (
    "cash",
    "constant_long",
    "constant_short",
    "trend",
    "ridge24",
    "lightgbm24",
    "ppo",
)

_CANONICAL_FIELD_VALUES: dict[str, object] = {
    "schema_version": _SCHEMA_VERSION,
    "issue_number": 545,
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
    "successor_dataset_id": "6c0b040d317a1bb73a9273f4135879b31691634aa837f30f0eec005ac7531518",
    "successor_dataset_artifact_digest": "af481dd978db7d84cd3aa8ff4f5a35d8608ac44c755dd74f61e934105c02b6b7",
    "successor_study_digest": "bfa2fcb307773f5384d7dcb884444164d6d3b575373d8f7dc1c810a61bf4c820",
    "successor_execution_overlay": "zero_overlay_dataset_fields_authoritative_previous_completed_bar_capacity",
    "symbols": _SYMBOLS,
    "capacity_caps": _CAPACITY_CAPS,
    "signal_name": "1h__log_return_24bar",
    "signal_index": 2,
    "fit_start": datetime(2021, 1, 1, 1, tzinfo=UTC),
    "fit_cutoff": datetime(2023, 1, 1, tzinfo=UTC),
    "evaluation_start": datetime(2023, 1, 1, tzinfo=UTC),
    "evaluation_stop_exclusive": datetime(2025, 1, 1, tzinfo=UTC),
    "label_execution_offset_bars": 1,
    "label_horizon_bars": 24,
    "label_formula": "log(open[t+25] / open[t+1])",
    "minimum_eligible_observations_per_symbol": 8760,
    "rule_entry_threshold": 0.01,
    "rule_exit_threshold": 0.0025,
    "market_order_fee_rate": 0.0005,
    "market_order_taker_fee_rate": 0.0,
    "market_order_spread_rate": 0.0002,
    "calibration_method": "per_symbol_no_intercept_fixed_order_fsum",
    "calibration_formula": "beta_i = fsum(s_t*y_t) / fsum(s_t*s_t)",
    "required_negative_symbol_slopes": 4,
    "beta_gate_order_statistic": 4,
    "beta_gate_description": "fourth_smallest_weakest_required_negative",
    "invalid_coverage_status": "INVALID_CALIBRATION_COVERAGE",
    "invalid_training_edge_status": "INVALID_NO_TRAINING_EDGE",
    "no_calibration_fallback": True,
    "position_values": _POSITION_VALUES,
    "edge_formula": "delta_position * beta_gate * signal",
    "one_way_cost_formula": "fee_rate + taker_fee_rate + spread_rate",
    "transition_cost_formula": "abs(delta_position) * one_way_cost",
    "gate_operator": ">",
    "equality_action": "HOLD_CURRENT",
    "unavailable_signal_action": "FLAT_BYPASS_GATE",
    "hard_risk_overrides_gate": True,
    "thresholds_unchanged": True,
    "impact_slippage_invented_by_gate": False,
    "target_strategy": "mean_reversion",
    "unaffected_strategies": _UNAFFECTED_STRATEGIES,
    "research_promote_status": "PROMOTE_RESEARCH_REFERENCE",
    "reject_status": "REJECT_MECHANISM",
    "inconclusive_status": "INCONCLUSIVE",
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
    "development_data_already_used": True,
    "production_eligible": False,
    "final_test_authorized": False,
    "shared_cash_profitability_established": False,
    "live_trading_authorized": False,
    "evaluation_pnl_inspected": False,
    "calibration_executed": False,
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
    resolved = value.astimezone(UTC)
    return resolved.isoformat().replace("+00:00", "Z")


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
class MeanReversionEconomicGateProtocol:
    """Immutable preregistration without calibration or evaluation results."""

    schema_version: str
    issue_number: int
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
    successor_execution_overlay: str
    symbols: tuple[str, ...]
    capacity_caps: tuple[float, ...]
    signal_name: str
    signal_index: int
    fit_start: datetime
    fit_cutoff: datetime
    evaluation_start: datetime
    evaluation_stop_exclusive: datetime
    label_execution_offset_bars: int
    label_horizon_bars: int
    label_formula: str
    minimum_eligible_observations_per_symbol: int
    rule_entry_threshold: float
    rule_exit_threshold: float
    market_order_fee_rate: float
    market_order_taker_fee_rate: float
    market_order_spread_rate: float
    calibration_method: str
    calibration_formula: str
    required_negative_symbol_slopes: int
    beta_gate_order_statistic: int
    beta_gate_description: str
    invalid_coverage_status: str
    invalid_training_edge_status: str
    no_calibration_fallback: bool
    position_values: tuple[tuple[str, int], ...]
    edge_formula: str
    one_way_cost_formula: str
    transition_cost_formula: str
    gate_operator: str
    equality_action: str
    unavailable_signal_action: str
    hard_risk_overrides_gate: bool
    thresholds_unchanged: bool
    impact_slippage_invented_by_gate: bool
    target_strategy: str
    unaffected_strategies: tuple[str, ...]
    research_promote_status: str
    reject_status: str
    inconclusive_status: str
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
    development_data_already_used: bool
    production_eligible: bool
    final_test_authorized: bool
    shared_cash_profitability_established: bool
    live_trading_authorized: bool
    evaluation_pnl_inspected: bool
    calibration_executed: bool
    evaluation_execution_authorized: bool

    def __post_init__(self) -> None:
        for field_name in (
            "fit_start",
            "fit_cutoff",
            "evaluation_start",
            "evaluation_stop_exclusive",
        ):
            _require_aware(getattr(self, field_name), field=field_name)
        if not self.fit_start < self.fit_cutoff == self.evaluation_start < self.evaluation_stop_exclusive:
            raise ValueError("preregistered research clock is invalid")
        if self.symbols != _SYMBOLS or len(self.capacity_caps) != len(self.symbols):
            raise ValueError("preregistered symbol/capacity roster is invalid")
        for index, cap in enumerate(self.capacity_caps):
            resolved = _require_finite(cap, field=f"capacity_caps[{index}]")
            if not 0.0 < resolved <= 1.0:
                raise ValueError("capacity caps must be within (0, 1]")
        for field_name in (
            "rule_entry_threshold",
            "rule_exit_threshold",
            "market_order_fee_rate",
            "market_order_taker_fee_rate",
            "market_order_spread_rate",
        ):
            resolved = _require_finite(getattr(self, field_name), field=field_name)
            if resolved < 0.0:
                raise ValueError(f"{field_name} must be non-negative")
        if self.rule_entry_threshold <= self.rule_exit_threshold:
            raise ValueError("preregistered rule thresholds are invalid")
        for field_name in (
            "issue_number",
            "diagnosis_issue",
            "diagnosis_pr",
            "diagnosis_verification_run_id",
            "diagnosis_verification_artifact_id",
            "successor_bundle_run_id",
            "successor_bundle_artifact_id",
            "signal_index",
            "label_execution_offset_bars",
            "label_horizon_bars",
            "minimum_eligible_observations_per_symbol",
            "required_negative_symbol_slopes",
            "beta_gate_order_statistic",
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
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{field_name} must be a non-negative integer")
        _require_hex(self.diagnosis_head_sha, field="diagnosis_head_sha", length=40)
        for field_name in (
            "diagnosis_report_digest",
            "diagnosis_verification_artifact_digest",
            "successor_bundle_artifact_digest",
            "successor_dataset_id",
            "successor_dataset_artifact_digest",
            "successor_study_digest",
        ):
            _require_hex(getattr(self, field_name), field=field_name, length=64)
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
        return {item.name: _json_value(getattr(self, item.name)) for item in fields(self)}

    @property
    def digest(self) -> str:
        return content_digest(self.to_payload())


def canonical_mean_reversion_economic_gate_protocol() -> MeanReversionEconomicGateProtocol:
    """Return the only preregistered protocol authorized by Issue #545."""

    return MeanReversionEconomicGateProtocol(**_CANONICAL_FIELD_VALUES)  # type: ignore[arg-type]


def _tuple_strings(value: object, *, field: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError(f"{field} must be a string array")
    return tuple(value)


def _tuple_floats(value: object, *, field: str) -> tuple[float, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{field} must be a finite numeric array")
    return tuple(_require_finite(item, field=field) for item in value)


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


def load_mean_reversion_economic_gate_protocol(
    path: str | Path,
) -> MeanReversionEconomicGateProtocol:
    """Load a strict preregistration JSON document and reject any schema drift."""

    raw: Any = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or any(not isinstance(key, str) for key in raw):
        raise ValueError("preregistration payload must be an object")
    expected_keys = {item.name for item in fields(MeanReversionEconomicGateProtocol)}
    if set(raw) != expected_keys:
        raise ValueError("preregistration keys differ from preregistered schema")

    converted = dict(raw)
    converted["symbols"] = _tuple_strings(raw["symbols"], field="symbols")
    converted["capacity_caps"] = _tuple_floats(raw["capacity_caps"], field="capacity_caps")
    converted["unaffected_strategies"] = _tuple_strings(
        raw["unaffected_strategies"], field="unaffected_strategies"
    )
    converted["position_values"] = _position_values(raw["position_values"])
    for field_name in (
        "fit_start",
        "fit_cutoff",
        "evaluation_start",
        "evaluation_stop_exclusive",
    ):
        converted[field_name] = _datetime_from_text(raw[field_name], field=field_name)
    for field_name in (
        "rule_entry_threshold",
        "rule_exit_threshold",
        "market_order_fee_rate",
        "market_order_taker_fee_rate",
        "market_order_spread_rate",
    ):
        converted[field_name] = _require_finite(raw[field_name], field=field_name)
    return MeanReversionEconomicGateProtocol(**converted)  # type: ignore[arg-type]


__all__ = [
    "MeanReversionEconomicGateProtocol",
    "canonical_mean_reversion_economic_gate_protocol",
    "load_mean_reversion_economic_gate_protocol",
]
