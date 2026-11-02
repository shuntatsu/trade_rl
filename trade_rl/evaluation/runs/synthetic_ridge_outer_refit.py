"""Closed synthetic conformance harness for continuous Ridge outer refits.

This module exercises the existing causal Ridge fitter and one native shared
portfolio replay. It is software-path evidence only; it cannot load an external
dataset or authorize a market study.
"""

from __future__ import annotations

import hashlib
import math
import platform
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from enum import StrEnum
from fractions import Fraction
from pathlib import Path
from typing import cast

import numpy as np

from trade_rl._validation import require_sha256
from trade_rl.artifacts.hashing import content_digest
from trade_rl.data.market import MarketDataset
from trade_rl.evaluation.replay import (
    SharedCashPortfolioReplayResult,
    run_shared_portfolio_replay,
)
from trade_rl.risk.pretrade import PreTradeRisk, PreTradeRiskConfig
from trade_rl.simulation.execution import ExecutionCostConfig
from trade_rl.simulation.objectives.binding import BoundObjectiveClock
from trade_rl.simulation.objectives.clock import FinancialClockContract
from trade_rl.simulation.objectives.contract import CapitalContract, ObjectiveContract
from trade_rl.strategies.forecasts.ridge import (
    RidgeForecastModel,
    RidgeForecastStrategy,
    fit_ridge_forecast,
)
from trade_rl.strategies.forecasts.supervised import (
    CausalForecastTrainingSet,
    build_causal_forecast_training_set,
)
from trade_rl.strategies.interface import StrategyObservation
from trade_rl.strategies.portfolio import (
    PortfolioDecisionSnapshot,
    PortfolioTargetProposal,
)
from trade_rl.strategies.position_intent import PositionIntent

_SCHEMA = "synthetic_ridge_outer_refit_v1"
_SYMBOLS = ("SYNTH-A", "SYNTH-B")
_FEATURE_INDICES = (0,)
_HORIZON_HOURS = 1
_RIDGE_ALPHA = 0.1
_ENTRY_THRESHOLD = 0.002
_EXIT_THRESHOLD = 0.001
_PER_SYMBOL_BUDGET = 0.2
_INITIAL_CAPITAL = 1_000.0
_START_INDEX = 9
_STOP_INDEX = 27
_BAR_SECONDS = 3_600
_N_BARS = 30
_BASE_TIMESTAMP = np.datetime64("2026-01-01T00:00:00", "ns")
_EXECUTION_COST = ExecutionCostConfig(
    fee_rate=0.001,
    spread_rate=0.0005,
    impact_rate=0.0,
    max_participation_rate=0.05,
    max_leverage=1.0,
    random_seed=0,
)
_RISK_CONFIG = PreTradeRiskConfig(
    max_gross=0.4,
    max_abs_weight=_PER_SYMBOL_BUDGET,
    max_turnover=None,
    drawdown_start=0.10,
    drawdown_stop=0.20,
)


class SyntheticFixtureVariant(StrEnum):
    """Only internally manufactured variants accepted by the public runner."""

    BASELINE = "baseline"
    FUTURE_TAIL_MUTATED = "future_tail_mutated"


@dataclass(frozen=True, slots=True)
class RefitActivation:
    """One causal fit event and the later decision index where it becomes live."""

    fit_cutoff_index: int
    fit_completed_index: int
    activation_index: int


@dataclass(frozen=True, slots=True)
class SyntheticRidgeOuterRefitSpec:
    """Clock binding and predeclared fit schedule for the closed fixture."""

    bound: BoundObjectiveClock
    schedule: tuple[RefitActivation, ...]


@dataclass(frozen=True, slots=True)
class VintageDecision:
    index: int
    vintage_id: str
    target_weights: tuple[float, ...]
    cash: float
    equity: float
    peak_equity: float
    max_drawdown: float
    exact_quantities: tuple[str, ...]
    active_order_facts: tuple[tuple[int, float, str, int], ...]

    def to_mapping(self) -> dict[str, object]:
        return {
            "index": self.index,
            "cash": self.cash,
            "equity": self.equity,
            "peak_equity": self.peak_equity,
            "max_drawdown": self.max_drawdown,
            "exact_quantities": list(self.exact_quantities),
            "active_order_facts": [list(value) for value in self.active_order_facts],
            "target_weights": list(self.target_weights),
            "vintage_id": self.vintage_id,
        }


@dataclass(frozen=True, slots=True)
class RidgeVintageArtifact:
    """In-memory vintage binding; causal identity is separate from fixture scope."""

    model: RidgeForecastModel
    vintage_id: str
    training_prefix_digest: str
    fixture_dataset_id: str
    coordinator_source_sha256: str
    runtime_digest: str
    artifact_digest: str
    fit_cutoff_index: int
    synthetic_fit_completed_index: int
    activation_index: int
    fit_elapsed_seconds: float
    unit_signal_prediction: float

    def to_mapping(self) -> dict[str, object]:
        return {
            "activation_index": self.activation_index,
            "alpha": self.model.alpha,
            "artifact_digest": self.artifact_digest,
            "coefficients": self.model.coefficients.tolist(),
            "feature_indices": list(self.model.feature_indices),
            "feature_mean": self.model.feature_mean.tolist(),
            "feature_scale": self.model.feature_scale.tolist(),
            "fit_elapsed_seconds": self.fit_elapsed_seconds,
            "fit_cutoff_index": self.fit_cutoff_index,
            "fit_cutoff_ns": int(self.model.fit_cutoff.astype(np.int64)),
            "fixture_dataset_id": self.fixture_dataset_id,
            "horizon_hours": self.model.horizon_hours,
            "coordinator_source_sha256": self.coordinator_source_sha256,
            "intercept": self.model.intercept,
            "n_samples": self.model.n_samples,
            "runtime_digest": self.runtime_digest,
            "schema": "synthetic_ridge_vintage_artifact_v1",
            "synthetic_fit_completed_index": self.synthetic_fit_completed_index,
            "training_prefix_digest": self.training_prefix_digest,
            "unit_signal_prediction": self.unit_signal_prediction,
            "vintage_id": self.vintage_id,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> RidgeVintageArtifact:
        """Reconstruct and verify one in-memory vintage artifact mapping."""

        required = {
            "activation_index",
            "alpha",
            "artifact_digest",
            "coefficients",
            "feature_indices",
            "feature_mean",
            "feature_scale",
            "fit_elapsed_seconds",
            "fit_cutoff_index",
            "fit_cutoff_ns",
            "fixture_dataset_id",
            "horizon_hours",
            "coordinator_source_sha256",
            "intercept",
            "n_samples",
            "runtime_digest",
            "schema",
            "synthetic_fit_completed_index",
            "training_prefix_digest",
            "unit_signal_prediction",
            "vintage_id",
        }
        if set(value) != required or value.get("schema") != (
            "synthetic_ridge_vintage_artifact_v1"
        ):
            raise ValueError("Ridge vintage artifact field closure mismatch")

        def integer(field: str) -> int:
            raw = value[field]
            if type(raw) is not int:
                raise ValueError(f"{field} must be a native integer")
            return raw

        def number(field: str) -> float:
            raw = value[field]
            if isinstance(raw, bool) or not isinstance(raw, (int, float)):
                raise ValueError(f"{field} must be numeric")
            result = float(raw)
            if not math.isfinite(result):
                raise ValueError(f"{field} must be finite")
            return result

        def vector(field: str) -> tuple[float, ...]:
            raw = value[field]
            if type(raw) not in (tuple, list):
                raise ValueError(f"{field} must be a numeric sequence")
            raw_values = cast(list[object] | tuple[object, ...], raw)
            result: list[float] = []
            for index, item in enumerate(raw_values):
                if isinstance(item, bool) or not isinstance(item, (int, float)):
                    raise ValueError(f"{field}[{index}] must be numeric")
                number_value = float(item)
                if not math.isfinite(number_value):
                    raise ValueError(f"{field}[{index}] must be finite")
                result.append(number_value)
            return tuple(result)

        indices_raw = value["feature_indices"]
        if type(indices_raw) not in (tuple, list):
            raise ValueError("feature_indices must be an integer sequence")
        raw_indices = cast(list[object] | tuple[object, ...], indices_raw)
        indices = cast(tuple[int, ...], tuple(raw_indices))
        if any(type(index) is not int for index in indices):
            raise ValueError("feature_indices must contain native integers")

        fixture_dataset_id = value["fixture_dataset_id"]
        coordinator_source_sha256 = value["coordinator_source_sha256"]
        runtime_digest = value["runtime_digest"]
        training_prefix_digest = value["training_prefix_digest"]
        vintage_id = value["vintage_id"]
        artifact_digest = value["artifact_digest"]
        for field, digest in (
            ("fixture_dataset_id", fixture_dataset_id),
            ("coordinator_source_sha256", coordinator_source_sha256),
            ("runtime_digest", runtime_digest),
            ("training_prefix_digest", training_prefix_digest),
            ("vintage_id", vintage_id),
            ("artifact_digest", artifact_digest),
        ):
            if not isinstance(digest, str):
                raise ValueError(f"{field} must be a SHA-256 digest")
            require_sha256(digest, field=field)
        if coordinator_source_sha256 != _coordinator_source_sha256():
            raise ValueError("Ridge artifact coordinator source digest mismatch")
        if runtime_digest != _runtime_digest():
            raise ValueError("Ridge artifact runtime digest mismatch")

        cutoff_index = integer("fit_cutoff_index")
        completed_index = integer("synthetic_fit_completed_index")
        activation_index = integer("activation_index")
        if not (
            1 <= cutoff_index < _N_BARS
            and cutoff_index <= completed_index <= _STOP_INDEX - 2
            and _START_INDEX <= activation_index <= _STOP_INDEX - 2
            and completed_index < activation_index
        ):
            raise ValueError("Ridge artifact fit and activation indexes are invalid")
        cutoff_ns = integer("fit_cutoff_ns")
        if cutoff_ns != _fixture_timestamp_ns(cutoff_index):
            raise ValueError(
                "fit cutoff index and timestamp do not match fixture clock"
            )
        model = RidgeForecastModel(
            feature_indices=indices,
            feature_mean=np.asarray(vector("feature_mean"), dtype=np.float64),
            feature_scale=np.asarray(vector("feature_scale"), dtype=np.float64),
            coefficients=np.asarray(vector("coefficients"), dtype=np.float64),
            intercept=number("intercept"),
            horizon_hours=integer("horizon_hours"),
            alpha=number("alpha"),
            n_samples=integer("n_samples"),
            fit_cutoff=np.datetime64(cutoff_ns, "ns"),
        )
        if int(model.fit_cutoff.astype(np.int64)) != cutoff_ns:
            raise ValueError("Ridge artifact fit cutoff is not exactly representable")
        prefix = str(training_prefix_digest)
        expected_vintage_id = _vintage_identity(
            training_prefix_digest=prefix,
            fit_cutoff_ns=cutoff_ns,
        )
        if vintage_id != expected_vintage_id:
            raise ValueError("Ridge artifact causal vintage identity mismatch")
        expected_prediction = model.predict(np.asarray([1.0], dtype=np.float64))
        unit_signal_prediction = number("unit_signal_prediction")
        if unit_signal_prediction != expected_prediction:
            raise ValueError("Ridge artifact probe prediction mismatch")
        expected_digest = _artifact_identity(
            model=model,
            vintage_id=str(vintage_id),
            training_prefix_digest=prefix,
            fixture_dataset_id=str(fixture_dataset_id),
            coordinator_source_sha256=str(coordinator_source_sha256),
            runtime_digest=str(runtime_digest),
            completed_index=completed_index,
            activation_index=activation_index,
            unit_signal_prediction=unit_signal_prediction,
        )
        if artifact_digest != expected_digest:
            raise ValueError("Ridge artifact content digest mismatch")
        fit_elapsed_seconds = number("fit_elapsed_seconds")
        if fit_elapsed_seconds < 0.0:
            raise ValueError("fit_elapsed_seconds must be non-negative")
        return cls(
            model=model,
            vintage_id=str(vintage_id),
            training_prefix_digest=prefix,
            fixture_dataset_id=str(fixture_dataset_id),
            coordinator_source_sha256=str(coordinator_source_sha256),
            runtime_digest=str(runtime_digest),
            artifact_digest=str(artifact_digest),
            fit_cutoff_index=cutoff_index,
            synthetic_fit_completed_index=completed_index,
            activation_index=activation_index,
            fit_elapsed_seconds=fit_elapsed_seconds,
            unit_signal_prediction=unit_signal_prediction,
        )


@dataclass(frozen=True, slots=True)
class SyntheticRidgeOuterRefitResult:
    """Full native replay plus synthetic fit and per-decision vintage evidence."""

    fixture_variant: SyntheticFixtureVariant
    fixture_dataset_id: str
    bound_digest: str
    fit_events: tuple[RidgeVintageArtifact, ...]
    decisions: tuple[VintageDecision, ...]
    replay: SharedCashPortfolioReplayResult
    evidence_classification: str = "synthetic_software_conformance_only"

    def to_mapping(self) -> dict[str, object]:
        ledger = self.replay.ledger_evidence
        if ledger is None:
            raise RuntimeError("synthetic outer-refit replay omitted its ledger")
        return {
            "bound_digest": self.bound_digest,
            "decisions": [record.to_mapping() for record in self.decisions],
            "evidence_classification": self.evidence_classification,
            "fit_events": [event.to_mapping() for event in self.fit_events],
            "fixture_dataset_id": self.fixture_dataset_id,
            "fixture_variant": self.fixture_variant.value,
            "ledger": ledger.to_mapping(),
            "schema": _SCHEMA,
        }


def _economics_digest() -> str:
    return content_digest(
        {
            "execution_cost": asdict(_EXECUTION_COST),
            "fixture_market_cost_arrays": "zero; explicit executor fee and spread",
            "schema": "synthetic_outer_refit_economics_v1",
        }
    )


def _risk_digest() -> str:
    return content_digest(
        {
            "pretrade_risk": asdict(_RISK_CONFIG),
            "schema": "synthetic_outer_refit_risk_v1",
        }
    )


def _deployment_recipe_digest() -> str:
    return content_digest(
        {
            "alpha": _RIDGE_ALPHA,
            "entry_threshold": _ENTRY_THRESHOLD,
            "exit_threshold": _EXIT_THRESHOLD,
            "feature_indices": _FEATURE_INDICES,
            "horizon_hours": _HORIZON_HOURS,
            "intent_budget_per_symbol": _PER_SYMBOL_BUDGET,
            "schema": "synthetic_ridge_outer_refit_recipe_v1",
            "symbols": _SYMBOLS,
        }
    )


def _datetime_at(index: int) -> datetime:
    raw = np.datetime64(_fixture_timestamp_ns(index), "ns")
    return datetime.fromisoformat(np.datetime_as_string(raw, unit="ns")).replace(
        tzinfo=UTC
    )


def _fixture_timestamp_ns(index: int) -> int:
    return int((_BASE_TIMESTAMP + index * np.timedelta64(1, "h")).astype(np.int64))


def _default_bound() -> BoundObjectiveClock:
    horizon_seconds = (_STOP_INDEX - _START_INDEX) * _BAR_SECONDS
    objective = ObjectiveContract(
        capital=CapitalContract(
            account_mode="shared_portfolio",
            currency="USD",
            initial_equities=(_INITIAL_CAPITAL,),
        ),
        evaluation_start=_datetime_at(_START_INDEX),
        evaluation_stop_exclusive=_datetime_at(_STOP_INDEX),
        terminal_valuation="settled",
        economics_digest=_economics_digest(),
        risk_digest=_risk_digest(),
        deployment_recipe_digest=_deployment_recipe_digest(),
        maximum_drawdown=0.20,
    )
    clock = FinancialClockContract(
        decision_interval_seconds=_BAR_SECONDS,
        execution_interval_seconds=_BAR_SECONDS,
        reward_interval_seconds=_BAR_SECONDS,
        economic_horizon_seconds=horizon_seconds,
        rollout_steps=(_STOP_INDEX - _START_INDEX) - 1,
        gamma=1.0,
        gae_lambda=0.95,
        reward_schema="equity_delta_v1",
    )
    return BoundObjectiveClock(objective=objective, clock=clock)


def default_synthetic_ridge_outer_refit_spec() -> SyntheticRidgeOuterRefitSpec:
    """Return the fixed two-vintage schedule; callers cannot supply market data."""

    return SyntheticRidgeOuterRefitSpec(
        bound=_default_bound(),
        schedule=(
            RefitActivation(
                fit_cutoff_index=8,
                fit_completed_index=8,
                activation_index=_START_INDEX,
            ),
            RefitActivation(
                fit_cutoff_index=16,
                fit_completed_index=16,
                activation_index=17,
            ),
        ),
    )


def _synthetic_dataset(variant: SyntheticFixtureVariant) -> MarketDataset:
    timestamps = _BASE_TIMESTAMP + np.arange(_N_BARS) * np.timedelta64(1, "h")
    signal = np.asarray(
        [-2.0, -1.0, 0.5, 1.5, 2.0, -0.5, 1.5, 1.0, 1.0]
        + [0.0] * 7
        + [1.0] * (_N_BARS - 16),
        dtype=np.float64,
    )
    close = np.full((_N_BARS, len(_SYMBOLS)), 100.0, dtype=np.float64)
    for index in range(_N_BARS - 1):
        coefficient = 0.005 if index <= 6 else 0.015 if index <= 8 else 0.0
        if variant is SyntheticFixtureVariant.FUTURE_TAIL_MUTATED and index >= 20:
            coefficient = 0.035
        step_return = coefficient * signal[index]
        close[index + 1, :] = close[index, :] * math.exp(step_return)
    if variant is SyntheticFixtureVariant.FUTURE_TAIL_MUTATED:
        signal[20:] = -1.5

    features = np.repeat(signal[:, None, None], len(_SYMBOLS), axis=1).astype(
        np.float32
    )
    prices = close.copy()
    high = prices.copy()
    low = prices.copy()
    volume = np.ones((_N_BARS, len(_SYMBOLS)), dtype=np.float64)
    volume[_STOP_INDEX, :] = 10_000.0
    dataset = MarketDataset(
        dataset_id="0" * 64,
        symbols=_SYMBOLS,
        timestamps=timestamps,
        features=features,
        global_features=np.zeros((_N_BARS, 1), dtype=np.float32),
        open=prices,
        high=high,
        low=low,
        close=close,
        volume=volume,
        funding_rate=np.zeros_like(close),
        tradable=np.ones_like(close, dtype=np.bool_),
        feature_available=np.ones(features.shape, dtype=np.bool_),
        feature_names=("causal_signal",),
        global_feature_names=("synthetic_regime",),
        periods_per_year=8_760,
        nominal_bar_hours=1.0,
        calendar_kind="continuous_24_7",
    )
    return dataset.with_content_identity(
        {
            "fixture": "synthetic-ridge-outer-refit",
            "fixture_version": 1,
            "variant": variant.value,
        }
    )


def _validate_spec(
    spec: SyntheticRidgeOuterRefitSpec,
    dataset: MarketDataset,
) -> tuple[int, int, int]:
    if type(spec) is not SyntheticRidgeOuterRefitSpec:
        raise TypeError("spec must be a SyntheticRidgeOuterRefitSpec")
    if spec.bound.digest != _default_bound().digest:
        raise ValueError("bound clock must match the fixed synthetic contract")
    objective = spec.bound.objective
    if (
        objective.capital.account_mode != "shared_portfolio"
        or objective.capital.initial_equities != (_INITIAL_CAPITAL,)
        or objective.terminal_valuation != "settled"
        or objective.maximum_drawdown != 0.20
        or spec.bound.clock.decision_interval_seconds != _BAR_SECONDS
        or spec.bound.clock.execution_interval_seconds != _BAR_SECONDS
        or spec.bound.clock.reward_interval_seconds != _BAR_SECONDS
        or spec.bound.clock.gamma != 1.0
    ):
        raise ValueError("capital, risk, cost, and financial clock are immutable")
    timestamps = dataset.timestamps.astype("datetime64[ns]")
    start_time = objective.evaluation_start.astimezone(UTC).replace(tzinfo=None)
    stop_time = objective.evaluation_stop_exclusive.astimezone(UTC).replace(tzinfo=None)
    start_matches = np.flatnonzero(timestamps == np.datetime64(start_time, "ns"))
    stop_matches = np.flatnonzero(timestamps == np.datetime64(stop_time, "ns"))
    if start_matches.size != 1 or stop_matches.size != 1:
        raise ValueError("bound evaluation period must match fixture timestamps")
    start_index = int(start_matches[0])
    stop_index = int(stop_matches[0])
    if (start_index, stop_index) != (_START_INDEX, _STOP_INDEX):
        raise ValueError("bound evaluation window is fixed to the synthetic fixture")
    schedule = spec.schedule
    if type(schedule) is not tuple or len(schedule) < 2:
        raise ValueError("schedule must declare at least two Ridge vintages")
    if any(type(entry) is not RefitActivation for entry in schedule):
        raise TypeError("schedule entries must be RefitActivation values")
    if schedule[0].activation_index != start_index:
        raise ValueError("initial vintage must activate at the evaluation start")
    if schedule[0].fit_completed_index >= start_index:
        raise ValueError("initial fit must complete explicitly before replay start")

    previous_cutoff = -1
    previous_completion = -1
    previous_activation = start_index - 1
    last_decision = stop_index - 2
    for position, entry in enumerate(schedule):
        fields = (
            entry.fit_cutoff_index,
            entry.fit_completed_index,
            entry.activation_index,
        )
        if any(type(value) is not int for value in fields):
            raise ValueError("refit schedule indexes must be native integers")
        if not 1 <= entry.fit_cutoff_index < dataset.n_bars:
            raise ValueError("fit cutoff must identify a fixture bar")
        if entry.fit_cutoff_index > entry.fit_completed_index:
            raise ValueError("fit cannot use a cutoff later than its readiness event")
        if entry.fit_cutoff_index <= previous_cutoff:
            raise ValueError("fit cutoffs must strictly increase")
        if entry.fit_completed_index <= previous_completion:
            raise ValueError("fit completion events must strictly increase")
        if position > 0 and entry.fit_completed_index <= previous_activation:
            raise ValueError("refit readiness must follow the prior activation")
        if entry.activation_index <= previous_activation:
            raise ValueError("activation indexes must strictly increase")
        if entry.activation_index <= entry.fit_completed_index:
            raise ValueError("fit must complete before its activation index")
        if position > 0 and not (
            start_index <= entry.fit_completed_index < last_decision
        ):
            raise ValueError("scheduled refit readiness must occur during replay")
        if entry.activation_index > last_decision:
            raise ValueError("vintage activation must precede terminal settlement")
        previous_cutoff = entry.fit_cutoff_index
        previous_completion = entry.fit_completed_index
        previous_activation = entry.activation_index
    return start_index, stop_index, last_decision


def _training_prefix_digest(
    training: CausalForecastTrainingSet,
    *,
    cutoff_index: int,
) -> str:
    payload = {
        "feature_indices": list(training.feature_indices),
        "features": np.asarray(training.features, dtype=np.float64).tolist(),
        "fit_cutoff_index": cutoff_index,
        "fit_cutoff_ns": int(training.fit_cutoff.astype(np.int64)),
        "horizon_hours": training.horizon_hours,
        "label_end_times_ns": training.label_end_times.astype(np.int64).tolist(),
        "labels": training.labels.tolist(),
        "sample_weights": training.sample_weights.tolist(),
        "schema": "causal_ridge_training_prefix_v1",
        "symbols": list(_SYMBOLS),
        "alpha": _RIDGE_ALPHA,
    }
    return content_digest(payload)


def _model_payload(model: RidgeForecastModel) -> dict[str, object]:
    return {
        "alpha": model.alpha,
        "coefficients": model.coefficients.tolist(),
        "feature_indices": list(model.feature_indices),
        "feature_mean": model.feature_mean.tolist(),
        "feature_scale": model.feature_scale.tolist(),
        "fit_cutoff_ns": int(model.fit_cutoff.astype(np.int64)),
        "horizon_hours": model.horizon_hours,
        "intercept": model.intercept,
        "n_samples": model.n_samples,
    }


def _coordinator_source_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _runtime_digest() -> str:
    return content_digest(
        {
            "numpy": np.__version__,
            "python": platform.python_version(),
            "schema": "synthetic_ridge_runtime_v1",
        }
    )


def _vintage_identity(*, training_prefix_digest: str, fit_cutoff_ns: int) -> str:
    return content_digest(
        {
            "alpha": _RIDGE_ALPHA,
            "feature_indices": _FEATURE_INDICES,
            "fit_cutoff_ns": fit_cutoff_ns,
            "horizon_hours": _HORIZON_HOURS,
            "schema": "causal_ridge_vintage_identity_v1",
            "training_prefix_digest": training_prefix_digest,
            "training_symbol_indices": list(range(len(_SYMBOLS))),
        }
    )


def _artifact_identity(
    *,
    model: RidgeForecastModel,
    vintage_id: str,
    training_prefix_digest: str,
    fixture_dataset_id: str,
    coordinator_source_sha256: str,
    runtime_digest: str,
    completed_index: int,
    activation_index: int,
    unit_signal_prediction: float,
) -> str:
    return content_digest(
        {
            "activation_index": activation_index,
            "fixture_dataset_id": fixture_dataset_id,
            "coordinator_source_sha256": coordinator_source_sha256,
            "model": _model_payload(model),
            "runtime_digest": runtime_digest,
            "schema": "synthetic_ridge_full_scope_artifact_v1",
            "synthetic_fit_completed_index": completed_index,
            "training_prefix_digest": training_prefix_digest,
            "unit_signal_prediction": unit_signal_prediction,
            "vintage_id": vintage_id,
        }
    )


def _fit_vintage(
    dataset: MarketDataset,
    entry: RefitActivation,
    *,
    fixture_dataset_id: str,
) -> RidgeVintageArtifact:
    fit_started = time.perf_counter()
    fit_cutoff = dataset.timestamps[entry.fit_cutoff_index].astype("datetime64[ns]")
    model = fit_ridge_forecast(
        dataset,
        feature_indices=_FEATURE_INDICES,
        fit_symbol_indices=tuple(range(len(_SYMBOLS))),
        fit_cutoff=fit_cutoff,
        horizon_hours=_HORIZON_HOURS,
        alpha=_RIDGE_ALPHA,
    )
    elapsed_seconds = time.perf_counter() - fit_started
    training = build_causal_forecast_training_set(
        dataset,
        feature_indices=_FEATURE_INDICES,
        fit_symbol_indices=tuple(range(len(_SYMBOLS))),
        fit_cutoff=fit_cutoff,
        horizon_hours=_HORIZON_HOURS,
    )
    prefix_digest = _training_prefix_digest(
        training,
        cutoff_index=entry.fit_cutoff_index,
    )
    fit_cutoff_ns = int(fit_cutoff.astype(np.int64))
    vintage_id = _vintage_identity(
        training_prefix_digest=prefix_digest,
        fit_cutoff_ns=fit_cutoff_ns,
    )
    coordinator_source_sha256 = _coordinator_source_sha256()
    runtime_digest = _runtime_digest()
    unit_signal_prediction = model.predict(np.asarray([1.0], dtype=np.float64))
    artifact_digest = _artifact_identity(
        model=model,
        vintage_id=vintage_id,
        training_prefix_digest=prefix_digest,
        fixture_dataset_id=fixture_dataset_id,
        coordinator_source_sha256=coordinator_source_sha256,
        runtime_digest=runtime_digest,
        completed_index=entry.fit_completed_index,
        activation_index=entry.activation_index,
        unit_signal_prediction=unit_signal_prediction,
    )
    return RidgeVintageArtifact(
        model=model,
        vintage_id=vintage_id,
        training_prefix_digest=prefix_digest,
        fixture_dataset_id=fixture_dataset_id,
        coordinator_source_sha256=coordinator_source_sha256,
        runtime_digest=runtime_digest,
        artifact_digest=artifact_digest,
        fit_cutoff_index=entry.fit_cutoff_index,
        synthetic_fit_completed_index=entry.fit_completed_index,
        activation_index=entry.activation_index,
        fit_elapsed_seconds=elapsed_seconds,
        unit_signal_prediction=unit_signal_prediction,
    )


def _current_intent(exact_quantity: str) -> PositionIntent:
    quantity = Fraction(exact_quantity)
    if str(quantity) != exact_quantity:
        raise ValueError("snapshot quantity must use canonical rational text")
    if quantity > 0:
        return PositionIntent.LONG
    if quantity < 0:
        return PositionIntent.SHORT
    return PositionIntent.FLAT


class _ScheduledRidgePortfolioPolicy:
    """One index/poison guard persists while active Ridge strategies change."""

    __slots__ = (
        "_dataset",
        "_schedule",
        "_start_index",
        "_next_schedule_index",
        "_active_artifact",
        "_staged_artifact",
        "_strategies",
        "_expected_index",
        "_last_timestamp_ns",
        "_agent_stop_index",
        "_poisoned",
        "fit_events",
        "decisions",
    )

    def __init__(
        self,
        dataset: MarketDataset,
        schedule: tuple[RefitActivation, ...],
        *,
        start_index: int,
    ) -> None:
        self._dataset = dataset
        self._schedule = schedule
        self._start_index = start_index
        first = _fit_vintage(
            dataset,
            schedule[0],
            fixture_dataset_id=dataset.dataset_id,
        )
        if first.synthetic_fit_completed_index >= start_index:
            raise ValueError("initial fit is not complete before replay start")
        self.fit_events = [first]
        self._active_artifact = first
        self._staged_artifact: RidgeVintageArtifact | None = None
        self._next_schedule_index = 1
        self._strategies = self._strategies_for(first.model)
        self._expected_index = start_index
        self._last_timestamp_ns: int | None = None
        self._agent_stop_index: int | None = None
        self._poisoned = False
        self.decisions: list[VintageDecision] = []

    @staticmethod
    def _strategies_for(model: RidgeForecastModel) -> tuple[RidgeForecastStrategy, ...]:
        return tuple(
            RidgeForecastStrategy(
                model,
                entry_threshold=_ENTRY_THRESHOLD,
                exit_threshold=_EXIT_THRESHOLD,
            )
            for _ in _SYMBOLS
        )

    def decide_portfolio(
        self,
        snapshot: PortfolioDecisionSnapshot,
    ) -> PortfolioTargetProposal:
        if self._poisoned:
            raise RuntimeError("scheduled Ridge policy is poisoned")
        try:
            self._validate_snapshot(snapshot)
            if self._next_schedule_index < len(self._schedule):
                upcoming = self._schedule[self._next_schedule_index]
                if snapshot.index == upcoming.fit_completed_index:
                    if self._staged_artifact is not None:
                        raise RuntimeError("a Ridge vintage is already staged")
                    self._staged_artifact = _fit_vintage(
                        self._dataset,
                        upcoming,
                        fixture_dataset_id=self._dataset.dataset_id,
                    )
                    self.fit_events.append(self._staged_artifact)
                if snapshot.index == upcoming.activation_index:
                    if self._staged_artifact is None:
                        raise RuntimeError("scheduled Ridge fit was not completed")
                    if (
                        self._staged_artifact.synthetic_fit_completed_index
                        >= snapshot.index
                    ):
                        raise RuntimeError("Ridge vintage activation is backdated")
                    self._active_artifact = self._staged_artifact
                    self._staged_artifact = None
                    self._strategies = self._strategies_for(self._active_artifact.model)
                    self._next_schedule_index += 1

            if snapshot.index >= self._active_artifact.activation_index:
                targets: list[float] = []
                for symbol_index, strategy in enumerate(self._strategies):
                    quantity_text = snapshot.account.exact_quantities[symbol_index]
                    observation = StrategyObservation(
                        index=snapshot.index,
                        timestamp=np.datetime64(snapshot.timestamp_ns, "ns"),
                        symbol=snapshot.symbols[symbol_index],
                        features=np.asarray(
                            snapshot.features[symbol_index], dtype=np.float64
                        ),
                        feature_available=np.asarray(
                            snapshot.feature_available[symbol_index], dtype=np.bool_
                        ),
                        feature_staleness=np.asarray(
                            snapshot.feature_staleness[symbol_index], dtype=np.float32
                        ),
                        global_features=np.asarray(
                            snapshot.global_features, dtype=np.float64
                        ),
                        global_feature_available=np.asarray(
                            snapshot.global_feature_available, dtype=np.bool_
                        ),
                        current_intent=_current_intent(quantity_text),
                        current_weight=float(snapshot.account.weights[symbol_index]),
                        current_position_quantity=float(Fraction(quantity_text)),
                        intent_gross_budget=_PER_SYMBOL_BUDGET,
                    )
                    intent = strategy.decide(observation)
                    targets.append(float(int(intent)) * _PER_SYMBOL_BUDGET)
                proposal = PortfolioTargetProposal(target_weights=tuple(targets))
            else:  # pragma: no cover - schedule validation prevents this branch
                raise RuntimeError("no active Ridge vintage for decision")

            self.decisions.append(
                VintageDecision(
                    index=snapshot.index,
                    vintage_id=self._active_artifact.vintage_id,
                    target_weights=proposal.target_weights,
                    cash=float(snapshot.account.cash),
                    equity=float(snapshot.account.portfolio_value),
                    peak_equity=float(snapshot.account.peak_value),
                    max_drawdown=float(snapshot.account.max_drawdown),
                    exact_quantities=snapshot.account.exact_quantities,
                    active_order_facts=tuple(
                        (
                            order.symbol_index,
                            float(order.remaining_quantity),
                            order.status,
                            order.submit_index,
                        )
                        for order in snapshot.active_orders
                    ),
                )
            )
            self._expected_index += 1
            self._last_timestamp_ns = snapshot.timestamp_ns
            self._agent_stop_index = snapshot.agent_stop_index
            return proposal
        except BaseException:
            self._poisoned = True
            raise

    def _validate_snapshot(self, snapshot: PortfolioDecisionSnapshot) -> None:
        if type(snapshot) is not PortfolioDecisionSnapshot:
            raise TypeError("snapshot must be a PortfolioDecisionSnapshot")
        if snapshot.index != self._expected_index:
            raise ValueError("portfolio decision indexes must advance exactly once")
        if snapshot.symbols != _SYMBOLS:
            raise ValueError("snapshot symbols differ from the fixed synthetic roster")
        if snapshot.index >= snapshot.agent_stop_index:
            raise ValueError("snapshot is outside the declared agent decision range")
        if self._last_timestamp_ns is not None:
            if snapshot.timestamp_ns <= self._last_timestamp_ns:
                raise ValueError("portfolio decision timestamps must increase")
            if snapshot.agent_stop_index != self._agent_stop_index:
                raise ValueError("agent stop index changed during the replay")
        elif snapshot.timestamp_ns != int(
            self._dataset.timestamps[self._start_index].astype(np.int64)
        ):
            raise ValueError("first decision timestamp differs from bound start")


def run_synthetic_ridge_outer_refit(
    *,
    spec: SyntheticRidgeOuterRefitSpec | None = None,
    fixture_variant: SyntheticFixtureVariant = SyntheticFixtureVariant.BASELINE,
) -> SyntheticRidgeOuterRefitResult:
    """Run two or more actual causal Ridge fits on one synthetic native replay."""

    try:
        variant = SyntheticFixtureVariant(fixture_variant)
    except ValueError as error:
        raise ValueError(
            "fixture_variant is not an allowed synthetic variant"
        ) from error
    dataset = _synthetic_dataset(variant)
    resolved_spec = default_synthetic_ridge_outer_refit_spec() if spec is None else spec
    start_index, stop_index, _ = _validate_spec(resolved_spec, dataset)
    policy = _ScheduledRidgePortfolioPolicy(
        dataset,
        resolved_spec.schedule,
        start_index=start_index,
    )
    result = run_shared_portfolio_replay(
        dataset,
        policy,
        start_index=start_index,
        stop_index=stop_index,
        initial_capital=_INITIAL_CAPITAL,
        execution_cost=_EXECUTION_COST,
        risk=PreTradeRisk(_RISK_CONFIG),
        settle_terminal_position=True,
        capture_ledger_evidence=True,
    )
    if len(policy.fit_events) != len(resolved_spec.schedule):
        raise RuntimeError("not every declared Ridge fit completed")
    if policy._staged_artifact is not None or policy._next_schedule_index != len(
        resolved_spec.schedule
    ):
        raise RuntimeError("not every Ridge vintage activated")
    if result.ledger_evidence is None:
        raise RuntimeError("native replay omitted required ledger evidence")
    return SyntheticRidgeOuterRefitResult(
        fixture_variant=variant,
        fixture_dataset_id=dataset.dataset_id,
        bound_digest=resolved_spec.bound.digest,
        fit_events=tuple(policy.fit_events),
        decisions=tuple(policy.decisions),
        replay=result,
    )


__all__ = [
    "RefitActivation",
    "RidgeVintageArtifact",
    "SyntheticFixtureVariant",
    "SyntheticRidgeOuterRefitResult",
    "SyntheticRidgeOuterRefitSpec",
    "VintageDecision",
    "default_synthetic_ridge_outer_refit_spec",
    "run_synthetic_ridge_outer_refit",
]
