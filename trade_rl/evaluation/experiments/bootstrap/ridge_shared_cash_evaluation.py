"""Result-blind shared-cash evaluator for the sealed ridge24 transition gate."""

from __future__ import annotations

import hashlib
import math
from collections import Counter
from dataclasses import dataclass, fields

import numpy as np

from trade_rl._validation import require_sha256
from trade_rl.artifacts.hashing import content_digest
from trade_rl.data.market import MarketDataset
from trade_rl.evaluation.experiments.bootstrap.ridge_shared_cash_prereg import (
    canonical_ridge_shared_cash_protocol,
)
from trade_rl.evaluation.metrics import compound_return, evaluate_performance
from trade_rl.evaluation.replay import run_shared_cash_replay
from trade_rl.evaluation.runs import execution_cost_for_overlay
from trade_rl.strategies.forecasts.ridge import (
    RidgeForecastStrategy,
    fit_ridge_forecast,
)
from trade_rl.strategies.forecasts.ridge_economic_gate import (
    RidgeEconomicGateConfig,
    RidgeEconomicGateStrategy,
)

_SPEC_SCHEMA = "ridge_shared_cash_evaluation_spec_v1"
_ARM_SCHEMA = "ridge_shared_cash_arm_evidence_v1"
_RESULT_SCHEMA = "ridge_shared_cash_evaluation_result_v1"
_QUALIFY = "QUALIFY_UNUSED_VALIDATION"
_STOP = "STOP_BEFORE_UNUSED_VALIDATION"
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
_EXECUTION_OVERLAY = (
    "zero_overlay_dataset_fields_authoritative_previous_completed_bar_capacity"
)
_SPEC_VALUES: dict[str, object] = {
    "schema_version": _SPEC_SCHEMA,
    "issue_number": 630,
    "protocol_issue_number": 627,
    "protocol_head": "8e0c2cc852e26240837614ae6a54180be227ca32",
    "protocol_module_blob": "947de729df3e0937cd09d7340c48da20fd567eea",
    "protocol_digest": "f0c7898d15042caf373af45da5be600462d225eecb3faf0c09f6c55ccbd35a69",
    "protocol_seal_run_id": 35204461338,
    "protocol_primary_artifact_id": 10488664859,
    "protocol_primary_artifact_api_digest": "7521a7bc02478670f5ee3da97f49f5c4f1a632b3369b91998912f926dd7ea925",
    "protocol_fresh_artifact_id": 10489476096,
    "protocol_fresh_artifact_api_digest": "f7ffef600d107fe8667f9e7b5130a1400fdb993151f386f831ac250c096d0366",
    "protocol_seal_sha256": "98be52a584200ed697234a2b07c46c2bc971fbfd2121239f8cdb327eb7206891",
    "trigger_run_id": 35201639813,
    "trigger_result_artifact_id": 10487739534,
    "trigger_result_artifact_api_digest": "787362841f4ff9b68235f157ba82a9e02edc1e995b6bedb54ec05149b6d59148",
    "trigger_fresh_artifact_id": 10488427586,
    "trigger_fresh_artifact_api_digest": "178bdd2575ab1d22d033a22a15f96b8a613fd786688b6d28647544352afee703",
    "trigger_result_digest": "78a39790c4d11dc903ac48f6044b0ebab46d2d6d26cbd8cf2c380be983b90d4b",
    "ridge_implementation_head": "222a082ee28f4f0fd35081912a33649cce27c585",
    "ridge_gate_blob": "db4fac918a421ce5223fa77257adcc234dadee66",
    "ridge_execution_overlay_blob": "24dc24970502a598c6c74c75efe0a348f11cb51d",
    "ridge_runs_facade_blob": "c64142b8e7b2dce82f00f70dc883ceced35123a2",
    "ridge_implementation_seal_run_id": 35200471721,
    "ridge_implementation_document_sha256": "6dd761d28781099f3bd129870b1512e15c2e30e54f91c858bca8d034618bfb59",
    "ridge_implementation_seal_sha256": "1573b8e65c8fc6093adbca5f95e2b5c4fd7b8c6b4e63161021ec4c1c23768fe1",
    "ridge_implementation_primary_artifact_id": 10487711819,
    "ridge_implementation_fresh_artifact_id": 10487836552,
    "ridge_implementation_artifact_api_digest": "3a35036040f28df009daca89e73dc5b050488abfea535e7ee09b67181579c713",
    "shared_cash_implementation_head": "4b9fc4bc6172b6c2a4e02e1e7cefad79575d6f70",
    "shared_cash_main_head": "d18434799651cfc6c07e0840c40600dcf1dfa763",
    "shared_cash_replay_blob": "73c15bc8555bb9d1cbf124bf13df9d8ec6c1f15e",
    "successor_run_id": 34803217815,
    "successor_artifact_id": 10331899302,
    "successor_artifact_api_digest": "89e899427f23fa46929c8be1e71fd49abe0d1d465c7a7f796a0874426b885bce",
    "dataset_id": "6c0b040d317a1bb73a9273f4135879b31691634aa837f30f0eec005ac7531518",
    "dataset_artifact_digest": "af481dd978db7d84cd3aa8ff4f5a35d8608ac44c755dd74f61e934105c02b6b7",
    "study_digest": "bfa2fcb307773f5384d7dcb884444164d6d3b575373d8f7dc1c810a61bf4c820",
    "cost_authority_run_id": 35200841490,
    "cost_authority_artifact_id": 10488425531,
    "cost_authority_artifact_api_digest": "7b4e0749b783fa29e50c3b92f0a21541148c80b8b4f4ed6b0586b0d5b52f64fb",
    "cost_authority_digest": "bb33f36edcf69ba91257e85dc68c7d53db396867ce51ea204a6ee14bac5cec04",
    "execution_overlay": _EXECUTION_OVERLAY,
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
    "one_way_explicit_cost": 0.0007,
    "gross_budget": 0.5,
    "initial_capital": 100_000.0,
    "calendar_years": (2023, 2024),
    "unused_data_accessed": False,
    "final_test_accessed": False,
    "final_test_authorized": False,
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


def _json_value(value: object) -> object:
    if isinstance(value, tuple):
        return [_json_value(item) for item in value]
    return value


@dataclass(frozen=True, slots=True)
class RidgeSharedCashEvaluationSpec:
    """Exact result-blind authorities for one future shared-cash evaluation."""

    schema_version: str
    issue_number: int
    protocol_issue_number: int
    protocol_head: str
    protocol_module_blob: str
    protocol_digest: str
    protocol_seal_run_id: int
    protocol_primary_artifact_id: int
    protocol_primary_artifact_api_digest: str
    protocol_fresh_artifact_id: int
    protocol_fresh_artifact_api_digest: str
    protocol_seal_sha256: str
    trigger_run_id: int
    trigger_result_artifact_id: int
    trigger_result_artifact_api_digest: str
    trigger_fresh_artifact_id: int
    trigger_fresh_artifact_api_digest: str
    trigger_result_digest: str
    ridge_implementation_head: str
    ridge_gate_blob: str
    ridge_execution_overlay_blob: str
    ridge_runs_facade_blob: str
    ridge_implementation_seal_run_id: int
    ridge_implementation_document_sha256: str
    ridge_implementation_seal_sha256: str
    ridge_implementation_primary_artifact_id: int
    ridge_implementation_fresh_artifact_id: int
    ridge_implementation_artifact_api_digest: str
    shared_cash_implementation_head: str
    shared_cash_main_head: str
    shared_cash_replay_blob: str
    successor_run_id: int
    successor_artifact_id: int
    successor_artifact_api_digest: str
    dataset_id: str
    dataset_artifact_digest: str
    study_digest: str
    cost_authority_run_id: int
    cost_authority_artifact_id: int
    cost_authority_artifact_api_digest: str
    cost_authority_digest: str
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
    one_way_explicit_cost: float
    gross_budget: float
    initial_capital: float
    calendar_years: tuple[int, ...]
    unused_data_accessed: bool
    final_test_accessed: bool
    final_test_authorized: bool
    operational_eligibility_established: bool
    production_eligible: bool
    live_trading_authorized: bool
    merge_authorized: bool

    def __post_init__(self) -> None:
        for item in fields(self):
            expected = _SPEC_VALUES[item.name]
            if not _strict_equal(getattr(self, item.name), expected):
                raise ValueError(f"frozen evaluation field {item.name} differs")

    def to_payload(self) -> dict[str, object]:
        return {
            item.name: _json_value(getattr(self, item.name)) for item in fields(self)
        }

    @property
    def digest(self) -> str:
        return content_digest(self.to_payload())


def canonical_ridge_shared_cash_evaluation_spec() -> RidgeSharedCashEvaluationSpec:
    """Return the single result-blind #630 evaluation specification."""

    protocol = canonical_ridge_shared_cash_protocol()
    if protocol.digest != _SPEC_VALUES["protocol_digest"]:
        raise ValueError("sealed shared-cash protocol digest differs from evaluator")
    return RidgeSharedCashEvaluationSpec(**_SPEC_VALUES)  # type: ignore[arg-type]


def shared_cash_return_sha256(values: object) -> str:
    """Stable exact-byte digest for one finite float64 shared-account path."""

    array = np.ascontiguousarray(np.asarray(values, dtype=np.float64))
    if array.ndim != 1:
        raise ValueError("shared-cash returns must be one-dimensional")
    if array.size == 0:
        raise ValueError("shared-cash returns must not be empty")
    if not np.isfinite(array).all() or np.any(array <= -1.0):
        raise ValueError("shared-cash returns must be finite and greater than -1")
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(repr(array.shape).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _validate_year_returns(
    values: tuple[tuple[int, float], ...],
) -> tuple[tuple[int, float], ...]:
    spec = canonical_ridge_shared_cash_evaluation_spec()
    if tuple(year for year, _ in values) != spec.calendar_years:
        raise ValueError("calendar-year evidence differs from frozen year roster")
    for year, value in values:
        if isinstance(year, bool) or not isinstance(year, int):
            raise ValueError("calendar-year evidence year must be an integer")
        if not math.isfinite(value) or value <= -1.0:
            raise ValueError("calendar-year return must be finite and greater than -1")
    return values


@dataclass(frozen=True, slots=True)
class RidgeSharedCashArmEvidence:
    """Raw and recomputable evidence for one shared-account arm."""

    arm: str
    returns: tuple[float, ...]
    return_sha256: str
    total_return: float
    calendar_year_returns: tuple[tuple[int, float], ...]
    total_cost: float
    turnover_total: float
    max_drawdown: float
    termination_count: int
    termination_reasons: tuple[str, ...]
    n_periods: int
    schema_version: str = _ARM_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != _ARM_SCHEMA:
            raise ValueError("unsupported shared-cash arm evidence schema")
        if self.arm not in {"baseline", "candidate"}:
            raise ValueError("arm must be baseline or candidate")
        if (
            isinstance(self.n_periods, bool)
            or not isinstance(self.n_periods, int)
            or self.n_periods <= 0
            or self.n_periods != len(self.returns)
        ):
            raise ValueError("period count does not match raw return evidence")
        expected_sha = shared_cash_return_sha256(self.returns)
        require_sha256(self.return_sha256, field="return_sha256")
        if self.return_sha256 != expected_sha:
            raise ValueError("return_sha256 does not match raw returns")
        expected_total = compound_return(self.returns)
        if not math.isfinite(self.total_return) or self.total_return != expected_total:
            raise ValueError("total_return does not match raw returns")
        _validate_year_returns(self.calendar_year_returns)
        for field_name in ("total_cost", "turnover_total", "max_drawdown"):
            value = getattr(self, field_name)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{field_name} must be finite and non-negative")
        if (
            isinstance(self.termination_count, bool)
            or not isinstance(self.termination_count, int)
            or self.termination_count < 0
            or self.termination_count != len(self.termination_reasons)
        ):
            raise ValueError("termination count/reasons are inconsistent")
        if any(
            not isinstance(reason, str) or not reason
            for reason in self.termination_reasons
        ):
            raise ValueError("termination reasons must contain non-empty strings")

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "arm": self.arm,
            "returns": list(self.returns),
            "return_sha256": self.return_sha256,
            "total_return": self.total_return,
            "calendar_year_returns": [
                {"year": year, "return": value}
                for year, value in self.calendar_year_returns
            ],
            "total_cost": self.total_cost,
            "turnover_total": self.turnover_total,
            "max_drawdown": self.max_drawdown,
            "termination_count": self.termination_count,
            "termination_reasons": list(self.termination_reasons),
            "n_periods": self.n_periods,
        }

    @property
    def digest(self) -> str:
        return content_digest(self.to_payload())


def _has_new_termination(
    baseline: RidgeSharedCashArmEvidence,
    candidate: RidgeSharedCashArmEvidence,
) -> bool:
    if candidate.termination_count > baseline.termination_count:
        return True
    baseline_counter = Counter(baseline.termination_reasons)
    candidate_counter = Counter(candidate.termination_reasons)
    return any(
        candidate_counter[reason] > baseline_counter[reason]
        for reason in candidate_counter
    )


def shared_cash_status(
    baseline: RidgeSharedCashArmEvidence,
    candidate: RidgeSharedCashArmEvidence,
) -> str:
    """Apply the result-blind #627 shared-cash qualification rule."""

    if baseline.arm != "baseline" or candidate.arm != "candidate":
        raise ValueError("shared-cash result arm roster is invalid")
    if baseline.n_periods != candidate.n_periods:
        return _STOP
    baseline_years = dict(baseline.calendar_year_returns)
    candidate_years = dict(candidate.calendar_year_returns)
    spec = canonical_ridge_shared_cash_evaluation_spec()
    qualifies = (
        candidate.total_return > 0.0
        and candidate.total_return > baseline.total_return
        and all(
            candidate_years[year] > 0.0 and candidate_years[year] > baseline_years[year]
            for year in spec.calendar_years
        )
        and candidate.total_cost < baseline.total_cost
        and candidate.turnover_total < baseline.turnover_total
        and candidate.max_drawdown <= baseline.max_drawdown
        and not _has_new_termination(baseline, candidate)
    )
    return _QUALIFY if qualifies else _STOP


@dataclass(frozen=True, slots=True)
class RidgeSharedCashEvaluation:
    """Strict shared-account evidence and frozen qualification status."""

    spec_digest: str
    dataset_id: str
    symbols: tuple[str, ...]
    baseline: RidgeSharedCashArmEvidence
    candidate: RidgeSharedCashArmEvidence
    status: str
    unused_data_accessed: bool = False
    final_test_accessed: bool = False
    final_test_authorized: bool = False
    operational_eligibility_established: bool = False
    production_eligible: bool = False
    live_trading_authorized: bool = False
    merge_authorized: bool = False
    schema_version: str = _RESULT_SCHEMA

    def __post_init__(self) -> None:
        spec = canonical_ridge_shared_cash_evaluation_spec()
        if self.schema_version != _RESULT_SCHEMA:
            raise ValueError("unsupported shared-cash evaluation result schema")
        require_sha256(self.spec_digest, field="spec_digest")
        if self.spec_digest != spec.digest:
            raise ValueError("spec_digest differs from frozen shared-cash evaluation")
        if self.dataset_id != spec.dataset_id or self.symbols != spec.symbols:
            raise ValueError("shared-cash result Dataset/symbol authority mismatch")
        if self.baseline.n_periods != self.candidate.n_periods:
            raise ValueError("period count differs between shared-cash arms")
        expected_status = shared_cash_status(self.baseline, self.candidate)
        if self.status != expected_status:
            raise ValueError("status does not match frozen shared-cash rule")
        for field_name in (
            "unused_data_accessed",
            "final_test_accessed",
            "final_test_authorized",
            "operational_eligibility_established",
            "production_eligible",
            "live_trading_authorized",
            "merge_authorized",
        ):
            value = getattr(self, field_name)
            if type(value) is not bool or value:
                raise ValueError(
                    "research result cannot authorize production/final/unused use"
                )

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "spec_digest": self.spec_digest,
            "dataset_id": self.dataset_id,
            "symbols": list(self.symbols),
            "baseline": self.baseline.to_payload(),
            "candidate": self.candidate.to_payload(),
            "status": self.status,
            "unused_data_accessed": self.unused_data_accessed,
            "final_test_accessed": self.final_test_accessed,
            "final_test_authorized": self.final_test_authorized,
            "operational_eligibility_established": self.operational_eligibility_established,
            "production_eligible": self.production_eligible,
            "live_trading_authorized": self.live_trading_authorized,
            "merge_authorized": self.merge_authorized,
        }

    @property
    def digest(self) -> str:
        return content_digest(self.to_payload())


def _exact_index(dataset: MarketDataset, timestamp: str, *, field: str) -> int:
    target = np.datetime64(timestamp, "ns")
    timestamps = np.asarray(dataset.timestamps, dtype="datetime64[ns]")
    matches = np.flatnonzero(timestamps == target)
    if matches.size != 1:
        raise ValueError(f"{field} must exactly match one Dataset timestamp")
    return int(matches[0])


def _validate_dataset(
    dataset: MarketDataset,
    spec: RidgeSharedCashEvaluationSpec,
) -> tuple[int, int]:
    if dataset.dataset_id != spec.dataset_id:
        raise ValueError("Dataset identity differs from frozen shared-cash evaluation")
    if tuple(dataset.symbols) != spec.symbols:
        raise ValueError(
            "Dataset symbol roster differs from frozen shared-cash evaluation"
        )
    for name, index in zip(spec.feature_names, spec.feature_indices, strict=True):
        if index >= len(dataset.feature_names) or dataset.feature_names[index] != name:
            raise ValueError(
                "Ridge feature identity differs from frozen shared-cash evaluation"
            )
    start = _exact_index(dataset, spec.evaluation_start, field="evaluation_start")
    stop = _exact_index(
        dataset,
        spec.evaluation_stop_exclusive,
        field="evaluation_stop_exclusive",
    )
    if not 0 <= start < stop < dataset.n_bars:
        raise ValueError("shared-cash evaluation range is invalid")
    return start, stop


def _calendar_year_returns(
    dataset: MarketDataset,
    *,
    start_index: int,
    returns: tuple[float, ...],
    years: tuple[int, ...],
) -> tuple[tuple[int, float], ...]:
    timestamps = np.asarray(
        dataset.timestamps[start_index : start_index + len(returns)],
        dtype="datetime64[ns]",
    )
    if len(timestamps) != len(returns):
        raise ValueError("return path extends beyond Dataset evaluation clock")
    result: list[tuple[int, float]] = []
    for year in years:
        lower = np.datetime64(f"{year:04d}-01-01T00:00:00", "ns")
        upper = np.datetime64(f"{year + 1:04d}-01-01T00:00:00", "ns")
        selected = tuple(
            value
            for timestamp, value in zip(timestamps, returns, strict=True)
            if lower <= timestamp < upper
        )
        if not selected:
            raise ValueError(f"shared-cash return path contains no rows for {year}")
        result.append((year, compound_return(selected)))
    return tuple(result)


def _arm_evidence(
    arm: str,
    replay: object,
    *,
    dataset: MarketDataset,
    start_index: int,
    years: tuple[int, ...],
) -> RidgeSharedCashArmEvidence:
    returns_obj = getattr(replay, "returns")
    diagnostics = getattr(replay, "diagnostics")
    returns = tuple(float(value) for value in returns_obj.values)
    termination_reasons = tuple(diagnostics.termination_reasons)
    metrics = evaluate_performance(
        returns_obj,
        turnover_total=diagnostics.turnover_total,
        total_cost=diagnostics.total_cost,
        funding_pnl=diagnostics.funding_pnl,
        borrow_cost=diagnostics.borrow_cost,
        n_trades=diagnostics.n_trades,
        rebalance_events=diagnostics.rebalance_events,
        termination_count=len(termination_reasons),
    )
    return RidgeSharedCashArmEvidence(
        arm=arm,
        returns=returns,
        return_sha256=shared_cash_return_sha256(returns),
        total_return=metrics.total_return,
        calendar_year_returns=_calendar_year_returns(
            dataset,
            start_index=start_index,
            returns=returns,
            years=years,
        ),
        total_cost=metrics.total_cost,
        turnover_total=metrics.turnover_total,
        max_drawdown=metrics.max_drawdown,
        termination_count=metrics.termination_count,
        termination_reasons=termination_reasons,
        n_periods=metrics.n_periods,
    )


def evaluate_ridge_shared_cash(dataset: MarketDataset) -> RidgeSharedCashEvaluation:
    """Evaluate the sealed baseline/candidate pair through one shared account per arm."""

    spec = canonical_ridge_shared_cash_evaluation_spec()
    start, stop = _validate_dataset(dataset, spec)
    model = fit_ridge_forecast(
        dataset,
        feature_indices=spec.feature_indices,
        fit_symbol_indices=tuple(range(len(spec.fit_symbol_names))),
        fit_cutoff=np.datetime64(spec.fit_cutoff, "ns"),
        horizon_hours=spec.ridge_horizon_hours,
        alpha=spec.ridge_alpha,
    )
    baseline = tuple(
        RidgeForecastStrategy(
            model,
            entry_threshold=spec.forecast_entry_threshold,
            exit_threshold=spec.forecast_exit_threshold,
        )
        for _ in spec.symbols
    )
    candidate = tuple(
        RidgeEconomicGateStrategy(
            RidgeEconomicGateConfig(
                model=model,
                entry_threshold=spec.forecast_entry_threshold,
                exit_threshold=spec.forecast_exit_threshold,
                one_way_explicit_cost=spec.one_way_explicit_cost,
            )
        )
        for _ in spec.symbols
    )
    if len({id(item) for item in baseline}) != len(baseline) or len(
        {id(item) for item in candidate}
    ) != len(candidate):
        raise RuntimeError("shared-cash evaluator requires distinct strategy instances")
    if not all(item.model is model for item in (*baseline, *candidate)):
        raise RuntimeError("shared-cash arms must share the exact fitted Ridge model")

    execution_cost = execution_cost_for_overlay(spec.execution_overlay)
    common = {
        "start_index": start,
        "stop_index": stop,
        "gross_budget": spec.gross_budget,
        "initial_capital": spec.initial_capital,
        "execution_cost": execution_cost,
        "risk": None,
    }
    baseline_replay = run_shared_cash_replay(dataset, baseline, **common)
    candidate_replay = run_shared_cash_replay(dataset, candidate, **common)
    baseline_evidence = _arm_evidence(
        "baseline",
        baseline_replay,
        dataset=dataset,
        start_index=start,
        years=spec.calendar_years,
    )
    candidate_evidence = _arm_evidence(
        "candidate",
        candidate_replay,
        dataset=dataset,
        start_index=start,
        years=spec.calendar_years,
    )
    return RidgeSharedCashEvaluation(
        spec_digest=spec.digest,
        dataset_id=spec.dataset_id,
        symbols=spec.symbols,
        baseline=baseline_evidence,
        candidate=candidate_evidence,
        status=shared_cash_status(baseline_evidence, candidate_evidence),
    )


__all__ = [
    "RidgeSharedCashArmEvidence",
    "RidgeSharedCashEvaluation",
    "RidgeSharedCashEvaluationSpec",
    "canonical_ridge_shared_cash_evaluation_spec",
    "evaluate_ridge_shared_cash",
    "shared_cash_return_sha256",
    "shared_cash_status",
]
