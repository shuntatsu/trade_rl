"""Result-blind preregistration for the Issue #627 shared-cash robustness gate."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, fields
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from trade_rl.artifacts.hashing import content_digest

_SCHEMA_VERSION = "ridge_shared_cash_prereg_v1"
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

_CANONICAL_FIELD_VALUES: dict[str, object] = {
    "schema_version": _SCHEMA_VERSION,
    "issue_number": 627,
    "trigger_issue": 626,
    "trigger_status": "PROMOTE_RESEARCH_REFERENCE",
    "trigger_run_id": 35201639813,
    "trigger_result_artifact_id": 10487739534,
    "trigger_result_artifact_digest": "787362841f4ff9b68235f157ba82a9e02edc1e995b6bedb54ec05149b6d59148",
    "trigger_fresh_artifact_id": 10488427586,
    "trigger_fresh_artifact_digest": "178bdd2575ab1d22d033a22a15f96b8a613fd786688b6d28647544352afee703",
    "trigger_result_digest": "78a39790c4d11dc903ac48f6044b0ebab46d2d6d26cbd8cf2c380be983b90d4b",
    "ridge_protocol_head": "75999e53c70224c31a62b106e4a8d2caa4b920ac",
    "ridge_protocol_digest": "167af235eeba44b9ade780c95ef7bab11c7b211d81018b82ece5d7044f7c4eb3",
    "ridge_protocol_seal_run_id": 35126253392,
    "ridge_protocol_primary_artifact_id": 10459431804,
    "ridge_protocol_fresh_artifact_id": 10458759061,
    "ridge_implementation_head": "222a082ee28f4f0fd35081912a33649cce27c585",
    "ridge_implementation_seal_run_id": 35200471721,
    "ridge_implementation_document_sha256": "6dd761d28781099f3bd129870b1512e15c2e30e54f91c858bca8d034618bfb59",
    "ridge_implementation_seal_sha256": "1573b8e65c8fc6093adbca5f95e2b5c4fd7b8c6b4e63161021ec4c1c23768fe1",
    "shared_cash_issue": 615,
    "shared_cash_pr": 620,
    "shared_cash_implementation_head": "4b9fc4bc6172b6c2a4e02e1e7cefad79575d6f70",
    "shared_cash_main_head": "d18434799651cfc6c07e0840c40600dcf1dfa763",
    "shared_cash_api": "trade_rl.evaluation.replay.run_shared_cash_replay",
    "require_result_blind_composition": True,
    "successor_bundle_run_id": 34803217815,
    "successor_bundle_artifact_id": 10331899302,
    "successor_bundle_artifact_digest": "89e899427f23fa46929c8be1e71fd49abe0d1d465c7a7f796a0874426b885bce",
    "successor_dataset_id": "6c0b040d317a1bb73a9273f4135879b31691634aa837f30f0eec005ac7531518",
    "successor_dataset_artifact_digest": "af481dd978db7d84cd3aa8ff4f5a35d8608ac44c755dd74f61e934105c02b6b7",
    "successor_study_digest": "bfa2fcb307773f5384d7dcb884444164d6d3b575373d8f7dc1c810a61bf4c820",
    "cost_authority_run_id": 35200841490,
    "cost_authority_artifact_id": 10488425531,
    "cost_authority_digest": "bb33f36edcf69ba91257e85dc68c7d53db396867ce51ea204a6ee14bac5cec04",
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
    "one_way_explicit_cost": 0.0007,
    "fit_ridge_exactly_once": True,
    "baseline_candidate_share_same_model_object": True,
    "distinct_strategy_instance_per_symbol": True,
    "initial_capital": 100_000.0,
    "per_intent_gross_budget": 0.5,
    "portfolio_max_gross": 1.0,
    "portfolio_max_abs_weight": 1.0,
    "portfolio_max_turnover": None,
    "portfolio_drawdown_start": 1.0,
    "portfolio_drawdown_stop": 1.0,
    "one_shared_book": True,
    "one_risk_projection_per_bar": True,
    "one_execution_per_bar": True,
    "qualify_status": "QUALIFY_UNUSED_VALIDATION",
    "stop_status": "STOP_BEFORE_UNUSED_VALIDATION",
    "invalid_status": "INVALID_SHARED_CASH_EVIDENCE",
    "qualify_requires_positive_full_return": True,
    "qualify_requires_full_return_above_baseline": True,
    "qualify_requires_positive_each_calendar_year": True,
    "qualify_requires_each_calendar_year_above_baseline": True,
    "qualify_requires_cost_reduction": True,
    "qualify_requires_turnover_reduction": True,
    "qualify_requires_drawdown_nonworse": True,
    "qualify_requires_equal_period_count": True,
    "qualify_requires_no_new_termination": True,
    "calendar_years": (2023, 2024),
    "calendar_year_account_reset": False,
    "no_new_strategy_degree_of_freedom": True,
    "post_result_retuning_allowed": False,
    "symbol_subset_allowed": False,
    "portfolio_optimization_allowed": False,
    "unused_data_accessed": False,
    "final_test_accessed": False,
    "final_test_authorized": False,
    "operational_eligibility_established": False,
    "production_eligible": False,
    "live_trading_authorized": False,
    "merge_authorized": False,
    "shared_cash_economic_result_inspected": False,
    "shared_cash_execution_authorized": False,
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
class RidgeSharedCashProtocol:
    """Immutable development-only shared-cash qualification protocol."""

    schema_version: str
    issue_number: int
    trigger_issue: int
    trigger_status: str
    trigger_run_id: int
    trigger_result_artifact_id: int
    trigger_result_artifact_digest: str
    trigger_fresh_artifact_id: int
    trigger_fresh_artifact_digest: str
    trigger_result_digest: str
    ridge_protocol_head: str
    ridge_protocol_digest: str
    ridge_protocol_seal_run_id: int
    ridge_protocol_primary_artifact_id: int
    ridge_protocol_fresh_artifact_id: int
    ridge_implementation_head: str
    ridge_implementation_seal_run_id: int
    ridge_implementation_document_sha256: str
    ridge_implementation_seal_sha256: str
    shared_cash_issue: int
    shared_cash_pr: int
    shared_cash_implementation_head: str
    shared_cash_main_head: str
    shared_cash_api: str
    require_result_blind_composition: bool
    successor_bundle_run_id: int
    successor_bundle_artifact_id: int
    successor_bundle_artifact_digest: str
    successor_dataset_id: str
    successor_dataset_artifact_digest: str
    successor_study_digest: str
    cost_authority_run_id: int
    cost_authority_artifact_id: int
    cost_authority_digest: str
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
    one_way_explicit_cost: float
    fit_ridge_exactly_once: bool
    baseline_candidate_share_same_model_object: bool
    distinct_strategy_instance_per_symbol: bool
    initial_capital: float
    per_intent_gross_budget: float
    portfolio_max_gross: float
    portfolio_max_abs_weight: float
    portfolio_max_turnover: float | None
    portfolio_drawdown_start: float
    portfolio_drawdown_stop: float
    one_shared_book: bool
    one_risk_projection_per_bar: bool
    one_execution_per_bar: bool
    qualify_status: str
    stop_status: str
    invalid_status: str
    qualify_requires_positive_full_return: bool
    qualify_requires_full_return_above_baseline: bool
    qualify_requires_positive_each_calendar_year: bool
    qualify_requires_each_calendar_year_above_baseline: bool
    qualify_requires_cost_reduction: bool
    qualify_requires_turnover_reduction: bool
    qualify_requires_drawdown_nonworse: bool
    qualify_requires_equal_period_count: bool
    qualify_requires_no_new_termination: bool
    calendar_years: tuple[int, ...]
    calendar_year_account_reset: bool
    no_new_strategy_degree_of_freedom: bool
    post_result_retuning_allowed: bool
    symbol_subset_allowed: bool
    portfolio_optimization_allowed: bool
    unused_data_accessed: bool
    final_test_accessed: bool
    final_test_authorized: bool
    operational_eligibility_established: bool
    production_eligible: bool
    live_trading_authorized: bool
    merge_authorized: bool
    shared_cash_economic_result_inspected: bool
    shared_cash_execution_authorized: bool

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
        if self.symbols != _SYMBOLS or self.fit_symbol_names != self.symbols:
            raise ValueError("preregistered symbol roster is invalid")
        if (
            self.feature_names != _FEATURE_NAMES
            or self.feature_indices != _FEATURE_INDICES
        ):
            raise ValueError("preregistered feature roster is invalid")
        if self.calendar_years != (2023, 2024):
            raise ValueError("preregistered calendar-year robustness window is invalid")

        for field_name in (
            "ridge_alpha",
            "forecast_entry_threshold",
            "forecast_exit_threshold",
            "one_way_explicit_cost",
            "initial_capital",
            "per_intent_gross_budget",
            "portfolio_max_gross",
            "portfolio_max_abs_weight",
            "portfolio_drawdown_start",
            "portfolio_drawdown_stop",
        ):
            resolved = _require_finite(getattr(self, field_name), field=field_name)
            if resolved < 0.0:
                raise ValueError(f"{field_name} must be non-negative")
        if self.ridge_alpha <= 0.0 or self.initial_capital <= 0.0:
            raise ValueError("preregistered model/capital values are invalid")
        if self.forecast_entry_threshold <= self.forecast_exit_threshold:
            raise ValueError("preregistered forecast thresholds are invalid")
        if self.portfolio_max_turnover is not None:
            _require_finite(self.portfolio_max_turnover, field="portfolio_max_turnover")

        for field_name in (
            "issue_number",
            "trigger_issue",
            "trigger_run_id",
            "trigger_result_artifact_id",
            "trigger_fresh_artifact_id",
            "ridge_protocol_seal_run_id",
            "ridge_protocol_primary_artifact_id",
            "ridge_protocol_fresh_artifact_id",
            "ridge_implementation_seal_run_id",
            "shared_cash_issue",
            "shared_cash_pr",
            "successor_bundle_run_id",
            "successor_bundle_artifact_id",
            "cost_authority_run_id",
            "cost_authority_artifact_id",
            "ridge_horizon_hours",
        ):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{field_name} must be a positive integer")

        boolean_fields = (
            "require_result_blind_composition",
            "fit_ridge_exactly_once",
            "baseline_candidate_share_same_model_object",
            "distinct_strategy_instance_per_symbol",
            "one_shared_book",
            "one_risk_projection_per_bar",
            "one_execution_per_bar",
            "qualify_requires_positive_full_return",
            "qualify_requires_full_return_above_baseline",
            "qualify_requires_positive_each_calendar_year",
            "qualify_requires_each_calendar_year_above_baseline",
            "qualify_requires_cost_reduction",
            "qualify_requires_turnover_reduction",
            "qualify_requires_drawdown_nonworse",
            "qualify_requires_equal_period_count",
            "qualify_requires_no_new_termination",
            "calendar_year_account_reset",
            "no_new_strategy_degree_of_freedom",
            "post_result_retuning_allowed",
            "symbol_subset_allowed",
            "portfolio_optimization_allowed",
            "unused_data_accessed",
            "final_test_accessed",
            "final_test_authorized",
            "operational_eligibility_established",
            "production_eligible",
            "live_trading_authorized",
            "merge_authorized",
            "shared_cash_economic_result_inspected",
            "shared_cash_execution_authorized",
        )
        for field_name in boolean_fields:
            if type(getattr(self, field_name)) is not bool:
                raise ValueError(f"{field_name} must be a boolean")

        for field_name in (
            "ridge_protocol_head",
            "ridge_implementation_head",
            "shared_cash_implementation_head",
            "shared_cash_main_head",
        ):
            _require_hex(getattr(self, field_name), field=field_name, length=40)
        for field_name in (
            "trigger_result_artifact_digest",
            "trigger_fresh_artifact_digest",
            "trigger_result_digest",
            "ridge_protocol_digest",
            "ridge_implementation_document_sha256",
            "ridge_implementation_seal_sha256",
            "successor_bundle_artifact_digest",
            "successor_dataset_id",
            "successor_dataset_artifact_digest",
            "successor_study_digest",
            "cost_authority_digest",
        ):
            _require_hex(getattr(self, field_name), field=field_name, length=64)

        for item in fields(self):
            if getattr(self, item.name) != _CANONICAL_FIELD_VALUES[item.name]:
                raise ValueError(f"{item.name} differs from preregistered value")

    def to_payload(self) -> dict[str, object]:
        return {
            item.name: _json_value(getattr(self, item.name)) for item in fields(self)
        }

    @property
    def digest(self) -> str:
        return content_digest(self.to_payload())


def canonical_ridge_shared_cash_protocol() -> RidgeSharedCashProtocol:
    """Return the only shared-cash protocol authorized by Issue #627."""

    return RidgeSharedCashProtocol(**_CANONICAL_FIELD_VALUES)  # type: ignore[arg-type]


def _tuple_strings(value: object, *, field: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError(f"{field} must be a string array")
    return tuple(value)


def _tuple_ints(value: object, *, field: str) -> tuple[int, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{field} must be an integer array")
    result: list[int] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int):
            raise ValueError(f"{field} must be an integer array")
        result.append(item)
    return tuple(result)


def load_ridge_shared_cash_protocol(path: str | Path) -> RidgeSharedCashProtocol:
    """Load strict canonical JSON and reject any shared-cash protocol drift."""

    raw: Any = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or any(not isinstance(key, str) for key in raw):
        raise ValueError("preregistration payload must be an object")
    expected_keys = {item.name for item in fields(RidgeSharedCashProtocol)}
    if set(raw) != expected_keys:
        raise ValueError("preregistration keys differ from preregistered schema")

    converted = dict(raw)
    for field_name in ("symbols", "feature_names", "fit_symbol_names"):
        converted[field_name] = _tuple_strings(raw[field_name], field=field_name)
    for field_name in ("feature_indices", "calendar_years"):
        converted[field_name] = _tuple_ints(raw[field_name], field=field_name)
    for field_name in ("fit_cutoff", "evaluation_start", "evaluation_stop_exclusive"):
        converted[field_name] = _datetime_from_text(raw[field_name], field=field_name)
    for field_name in (
        "ridge_alpha",
        "forecast_entry_threshold",
        "forecast_exit_threshold",
        "one_way_explicit_cost",
        "initial_capital",
        "per_intent_gross_budget",
        "portfolio_max_gross",
        "portfolio_max_abs_weight",
        "portfolio_drawdown_start",
        "portfolio_drawdown_stop",
    ):
        converted[field_name] = _require_finite(raw[field_name], field=field_name)
    turnover = raw["portfolio_max_turnover"]
    if turnover is not None:
        converted["portfolio_max_turnover"] = _require_finite(
            turnover, field="portfolio_max_turnover"
        )
    return RidgeSharedCashProtocol(**converted)


__all__ = [
    "RidgeSharedCashProtocol",
    "canonical_ridge_shared_cash_protocol",
    "load_ridge_shared_cash_protocol",
]
