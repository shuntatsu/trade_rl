"""Filesystem comparison of shared-cash arms using one frozen Ridge model."""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from typing import Callable, Sequence, cast

import numpy as np

from trade_rl.data import (
    MarketDataset,
    PublishedDatasetArtifact,
    load_published_market_dataset_artifact,
)
from trade_rl.data.market_order_rules import MarketOrderProfile
from trade_rl.evaluation.comparison.strategies import (
    SharedCashStrategyComparisonEntry,
    compare_shared_cash_strategy_factories,
)
from trade_rl.risk import PreTradeRisk, PreTradeRiskConfig
from trade_rl.simulation.execution import ExecutionCostConfig
from trade_rl.strategies.controls import ConstantIntentStrategy
from trade_rl.strategies.forecasts import (
    ForecastIntentConfig,
    RidgeForecastArtifactBinding,
    RidgeForecastModel,
    RidgeForecastStrategy,
    load_ridge_forecast_artifact,
)
from trade_rl.strategies.interface import SingleSymbolStrategy
from trade_rl.strategies.position_intent import PositionIntent

_FORECAST_EXCLUSION_REASONS = (
    "feature_unavailable_or_nonfinite",
    "starting_close_unavailable_at_decision",
    "target_not_mature_by_scoring_as_of",
    "target_close_unavailable_by_scoring_as_of",
)


def _finite_float(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{field} must be a finite number")
    try:
        result = float(value)
    except (OverflowError, TypeError, ValueError) as error:
        raise ValueError(f"{field} must be a finite number") from error
    if not math.isfinite(result):
        raise ValueError(f"{field} must be a finite number")
    return result


def _validate_nested_numeric_booleans(
    execution_cost: ExecutionCostConfig,
    risk: PreTradeRiskConfig,
) -> None:
    execution_numeric_fields = (
        "fee_rate",
        "maker_fee_rate",
        "taker_fee_rate",
        "spread_rate",
        "impact_rate",
        "multiplier",
        "max_participation_rate",
        "slippage_std",
        "tail_slippage_probability",
        "tail_slippage_multiplier",
        "minimum_notional",
        "lot_size",
        "tick_size",
        "borrow_rate_multiplier",
        "max_leverage",
        "maintenance_margin_rate",
        "collateral_haircut",
        "limit_offset_rate",
    )
    risk_numeric_fields = (
        "max_gross",
        "max_abs_weight",
        "max_turnover",
        "drawdown_start",
        "drawdown_stop",
        "fail_closed_tolerance",
    )

    for field_name in execution_numeric_fields:
        if isinstance(getattr(execution_cost, field_name), (bool, np.bool_)):
            raise ValueError(f"execution_cost.{field_name} must not be boolean")
    for index, value in enumerate(execution_cost.trigger_volume_fractions):
        if isinstance(value, (bool, np.bool_)):
            raise ValueError(
                f"execution_cost.trigger_volume_fractions[{index}] must not be boolean"
            )
    for field_name in risk_numeric_fields:
        value = getattr(risk, field_name)
        if value is not None and isinstance(value, (bool, np.bool_)):
            raise ValueError(f"risk.{field_name} must not be boolean")


@dataclass(frozen=True, slots=True)
class RidgeArtifactComparisonConfig:
    """Explicit common replay settings and forecast-controller factors."""

    start_index: int
    stop_index: int
    gross_budget: float
    initial_capital: float
    entry_threshold: float
    exit_threshold: float
    one_way_switch_cost: float
    execution_cost: ExecutionCostConfig
    risk: PreTradeRiskConfig
    market_order_profile: MarketOrderProfile | None = None
    minimum_hold_bars: int = 0
    include_forecast_diagnostics: bool = False

    def __post_init__(self) -> None:
        if (
            type(self.start_index) is not int
            or type(self.stop_index) is not int
            or self.start_index < 0
            or self.stop_index <= self.start_index
        ):
            raise ValueError("replay range requires integer 0 <= start < stop")
        if type(self.minimum_hold_bars) is not int or self.minimum_hold_bars < 0:
            raise ValueError("minimum_hold_bars must be a non-negative integer")
        if type(self.include_forecast_diagnostics) is not bool:
            raise ValueError("include_forecast_diagnostics must be a boolean")

        gross_budget = _finite_float(self.gross_budget, field="gross_budget")
        if not 0.0 < gross_budget <= 1.0:
            raise ValueError("gross_budget must be within (0, 1]")
        initial_capital = _finite_float(self.initial_capital, field="initial_capital")
        if initial_capital <= 0.0:
            raise ValueError("initial_capital must be positive")
        switch_cost = _finite_float(
            self.one_way_switch_cost,
            field="one_way_switch_cost",
        )
        if switch_cost <= 0.0:
            raise ValueError("one_way_switch_cost must be positive")

        entry_threshold = _finite_float(
            self.entry_threshold,
            field="entry_threshold",
        )
        exit_threshold = _finite_float(
            self.exit_threshold,
            field="exit_threshold",
        )
        thresholds = ForecastIntentConfig(
            entry_threshold=entry_threshold,
            exit_threshold=exit_threshold,
        )
        if not isinstance(self.execution_cost, ExecutionCostConfig):
            raise TypeError("execution_cost must be an ExecutionCostConfig")
        if type(self.risk) is not PreTradeRiskConfig:
            raise TypeError("risk must be a PreTradeRiskConfig")
        _validate_nested_numeric_booleans(self.execution_cost, self.risk)
        if self.market_order_profile is not None and not isinstance(
            self.market_order_profile, MarketOrderProfile
        ):
            raise TypeError("market_order_profile must be a MarketOrderProfile or None")

        object.__setattr__(self, "gross_budget", gross_budget)
        object.__setattr__(self, "initial_capital", initial_capital)
        object.__setattr__(self, "entry_threshold", thresholds.entry_threshold)
        object.__setattr__(self, "exit_threshold", thresholds.exit_threshold)
        object.__setattr__(self, "one_way_switch_cost", switch_cost)


@dataclass(frozen=True, slots=True)
class ForecastErrorMetrics:
    """Paired forecast and zero-baseline errors in Ridge log-return units."""

    pair_count: int
    mae: float | None
    rmse: float | None
    zero_mae: float | None
    zero_rmse: float | None
    mae_improvement: float | None
    mse_skill: float | None

    def __post_init__(self) -> None:
        if type(self.pair_count) is not int or self.pair_count < 0:
            raise ValueError("pair_count must be a non-negative integer")
        numeric_fields = (
            "mae",
            "rmse",
            "zero_mae",
            "zero_rmse",
            "mae_improvement",
            "mse_skill",
        )
        values: dict[str, float | None] = {}
        for name in numeric_fields:
            value = getattr(self, name)
            values[name] = None if value is None else _finite_float(value, field=name)
            object.__setattr__(self, name, values[name])
        if self.pair_count == 0:
            if any(value is not None for value in values.values()):
                raise ValueError("empty forecast metrics must be undefined")
            return
        required = ("mae", "rmse", "zero_mae", "zero_rmse", "mae_improvement")
        if any(values[name] is None for name in required):
            raise ValueError("non-empty forecast metrics must be finite")
        for name in ("mae", "rmse", "zero_mae", "zero_rmse"):
            if cast(float, values[name]) < 0.0:
                raise ValueError(f"{name} must be non-negative")
        zero_rmse = cast(float, values["zero_rmse"])
        skill = values["mse_skill"]
        if (zero_rmse == 0.0) != (skill is None):
            raise ValueError("mse_skill is undefined exactly for a zero baseline")
        expected_improvement = cast(float, values["zero_mae"]) - cast(
            float, values["mae"]
        )
        if not math.isclose(
            cast(float, values["mae_improvement"]),
            expected_improvement,
            rel_tol=1e-12,
            abs_tol=1e-15,
        ):
            raise ValueError("mae_improvement is inconsistent")
        if skill is not None:
            ratio = cast(float, values["rmse"]) / zero_rmse
            expected_skill = 1.0 - ratio * ratio
            if not math.isfinite(expected_skill) or not math.isclose(
                cast(float, skill), expected_skill, rel_tol=1e-12, abs_tol=1e-15
            ):
                raise ValueError("mse_skill is inconsistent")


def _validated_exclusion_counts(
    value: object,
) -> tuple[tuple[str, int], ...]:
    if not isinstance(value, tuple) or len(value) != len(_FORECAST_EXCLUSION_REASONS):
        raise ValueError("forecast exclusion counts have an invalid shape")
    result: list[tuple[str, int]] = []
    for expected_reason, item in zip(_FORECAST_EXCLUSION_REASONS, value, strict=True):
        if (
            not isinstance(item, tuple)
            or len(item) != 2
            or item[0] != expected_reason
            or type(item[1]) is not int
            or item[1] < 0
        ):
            raise ValueError("forecast exclusion counts are invalid")
        result.append((expected_reason, item[1]))
    return tuple(result)


@dataclass(frozen=True, slots=True)
class ForecastSymbolDiagnostics:
    """Paired scoring counts and metrics for one ordered Dataset symbol."""

    symbol: str
    candidate_count: int
    eligible_count: int
    excluded_counts: tuple[tuple[str, int], ...]
    metrics: ForecastErrorMetrics

    def __post_init__(self) -> None:
        if not isinstance(self.symbol, str) or not self.symbol:
            raise ValueError("forecast diagnostic symbol must be non-empty")
        if (
            type(self.candidate_count) is not int
            or type(self.eligible_count) is not int
            or self.candidate_count < 0
            or self.eligible_count < 0
            or self.eligible_count > self.candidate_count
        ):
            raise ValueError("forecast symbol counts are invalid")
        counts = _validated_exclusion_counts(self.excluded_counts)
        if (
            sum(count for _, count in counts)
            != self.candidate_count - self.eligible_count
        ):
            raise ValueError("forecast symbol exclusion counts do not reconcile")
        if type(self.metrics) is not ForecastErrorMetrics:
            raise TypeError("forecast symbol metrics are invalid")
        if self.metrics.pair_count != self.eligible_count:
            raise ValueError("forecast symbol metric count does not reconcile")
        object.__setattr__(self, "excluded_counts", counts)


@dataclass(frozen=True, slots=True)
class RidgeForecastDiagnostics:
    """Opt-in, out-of-sample log-return errors against the zero forecast."""

    horizon_hours: int
    fit_cutoff_ns: int
    evaluation_start_index: int
    decision_stop_index: int
    evaluation_stop_index: int
    scoring_as_of_ns: int
    candidate_count: int
    eligible_count: int
    excluded_counts: tuple[tuple[str, int], ...]
    pooled: ForecastErrorMetrics
    by_symbol: tuple[ForecastSymbolDiagnostics, ...]

    def __post_init__(self) -> None:
        if type(self.horizon_hours) is not int or self.horizon_hours <= 0:
            raise ValueError("diagnostic horizon_hours must be positive")
        if any(
            type(value) is not int
            for value in (
                self.fit_cutoff_ns,
                self.evaluation_start_index,
                self.decision_stop_index,
                self.evaluation_stop_index,
                self.scoring_as_of_ns,
                self.candidate_count,
                self.eligible_count,
            )
        ):
            raise ValueError("forecast diagnostic scope values must be integers")
        if not (
            0
            <= self.evaluation_start_index
            < self.decision_stop_index
            < self.evaluation_stop_index
        ):
            raise ValueError("forecast diagnostic index scope is invalid")
        if (
            self.candidate_count < 0
            or self.eligible_count < 0
            or self.eligible_count > self.candidate_count
        ):
            raise ValueError("forecast diagnostic counts are invalid")
        counts = _validated_exclusion_counts(self.excluded_counts)
        if (
            sum(count for _, count in counts)
            != self.candidate_count - self.eligible_count
        ):
            raise ValueError("forecast diagnostic exclusion counts do not reconcile")
        if type(self.pooled) is not ForecastErrorMetrics:
            raise TypeError("pooled forecast metrics are invalid")
        if self.pooled.pair_count != self.eligible_count:
            raise ValueError("pooled forecast metric count does not reconcile")
        symbols = tuple(self.by_symbol)
        if not symbols or any(
            type(item) is not ForecastSymbolDiagnostics for item in symbols
        ):
            raise ValueError("by_symbol must contain forecast diagnostics")
        if len({item.symbol for item in symbols}) != len(symbols):
            raise ValueError("forecast diagnostic symbols must be unique")
        if sum(item.candidate_count for item in symbols) != self.candidate_count:
            raise ValueError("per-symbol candidate counts do not reconcile")
        if sum(item.eligible_count for item in symbols) != self.eligible_count:
            raise ValueError("per-symbol eligible counts do not reconcile")
        for reason_index, (reason, count) in enumerate(counts):
            if sum(item.excluded_counts[reason_index][1] for item in symbols) != count:
                raise ValueError(f"per-symbol {reason} counts do not reconcile")
        object.__setattr__(self, "excluded_counts", counts)
        object.__setattr__(self, "by_symbol", symbols)


def _mean_absolute(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("mean requires at least one value")
    scale = max(abs(value) for value in values)
    if scale == 0.0:
        return 0.0
    result = scale * math.fsum(abs(value) / scale for value in values) / len(values)
    if not math.isfinite(result):
        raise ValueError("forecast diagnostics exceed finite metric range")
    return result


def _root_mean_square(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("root mean square requires at least one value")
    scale = max(abs(value) for value in values)
    if scale == 0.0:
        return 0.0
    normalized = math.fsum((value / scale) ** 2 for value in values) / len(values)
    result = scale * math.sqrt(normalized)
    if not math.isfinite(result):
        raise ValueError("forecast diagnostics exceed finite metric range")
    return result


def _error_metrics(
    forecasts: Sequence[float], targets: Sequence[float]
) -> ForecastErrorMetrics:
    if len(forecasts) != len(targets):
        raise ValueError("forecast and baseline must use the same pairs")
    if not forecasts:
        return ForecastErrorMetrics(0, None, None, None, None, None, None)
    errors: list[float] = []
    zero_errors: list[float] = []
    for forecast, target in zip(forecasts, targets, strict=True):
        error = forecast - target
        zero_error = -target
        if not math.isfinite(error) or not math.isfinite(zero_error):
            raise ValueError("forecast errors must remain finite")
        errors.append(error)
        zero_errors.append(zero_error)
    mae = _mean_absolute(errors)
    rmse = _root_mean_square(errors)
    zero_mae = _mean_absolute(zero_errors)
    zero_rmse = _root_mean_square(zero_errors)
    skill: float | None = None
    if zero_rmse > 0.0:
        ratio = rmse / zero_rmse
        skill = 1.0 - ratio * ratio
        if not math.isfinite(skill):
            raise ValueError("forecast diagnostics exceed finite metric range")
    return ForecastErrorMetrics(
        pair_count=len(forecasts),
        mae=mae,
        rmse=rmse,
        zero_mae=zero_mae,
        zero_rmse=zero_rmse,
        mae_improvement=zero_mae - mae,
        mse_skill=skill,
    )


def _ridge_forecast_diagnostics(
    dataset: MarketDataset,
    model: RidgeForecastModel,
    config: RidgeArtifactComparisonConfig,
) -> RidgeForecastDiagnostics:
    """Score actual model predictions on exact, mature supervised-label pairs."""

    timestamps_ns = dataset.timestamps.astype("datetime64[ns]").astype(np.int64)
    scoring_as_of_ns = int(timestamps_ns[config.stop_index])
    decision_stop_index = (
        config.stop_index - config.execution_cost.order_latency_bars - 1
    )
    if decision_stop_index <= config.start_index:
        raise ValueError("comparison range has no agent decision interval")
    horizon_ns = model.horizon_hours * 3_600_000_000_000
    endpoint_indices = {
        int(timestamp): index
        for index, timestamp in enumerate(timestamps_ns[: config.stop_index])
    }
    price_available_at = (
        dataset.resolved_array("available_at").astype("datetime64[ns]").astype(np.int64)
    )
    selected_indices = list(model.feature_indices)
    forecasts_by_symbol: list[list[float]] = [[] for _ in range(dataset.n_symbols)]
    targets_by_symbol: list[list[float]] = [[] for _ in range(dataset.n_symbols)]
    exclusions_by_symbol = [
        {reason: 0 for reason in _FORECAST_EXCLUSION_REASONS}
        for _ in range(dataset.n_symbols)
    ]
    candidate_count = 0

    for decision_index in range(config.start_index, decision_stop_index):
        decision_ns = int(timestamps_ns[decision_index])
        for symbol_index in range(dataset.n_symbols):
            candidate_count += 1
            symbol_exclusions = exclusions_by_symbol[symbol_index]
            features = np.asarray(dataset.features[decision_index, symbol_index])
            available = np.asarray(
                dataset.feature_available[decision_index, symbol_index], dtype=np.bool_
            )
            selected = features[selected_indices]
            if not bool(np.all(available[selected_indices])) or not bool(
                np.isfinite(selected).all()
            ):
                symbol_exclusions["feature_unavailable_or_nonfinite"] += 1
                continue
            if price_available_at[decision_index, symbol_index] > decision_ns:
                symbol_exclusions["starting_close_unavailable_at_decision"] += 1
                continue

            forecast = _finite_float(model.predict(features), field="Ridge forecast")
            endpoint_ns = decision_ns + horizon_ns
            if endpoint_ns >= scoring_as_of_ns:
                symbol_exclusions["target_not_mature_by_scoring_as_of"] += 1
                continue
            endpoint_index = endpoint_indices.get(endpoint_ns)
            if endpoint_index is None or endpoint_index <= decision_index:
                symbol_exclusions["target_not_mature_by_scoring_as_of"] += 1
                continue
            if price_available_at[endpoint_index, symbol_index] > scoring_as_of_ns:
                symbol_exclusions["target_close_unavailable_by_scoring_as_of"] += 1
                continue
            start_price = _finite_float(
                dataset.close[decision_index, symbol_index],
                field="forecast start close",
            )
            end_price = _finite_float(
                dataset.close[endpoint_index, symbol_index],
                field="forecast target close",
            )
            if start_price <= 0.0 or end_price <= 0.0:
                raise ValueError("forecast target closes must be positive")
            target = math.log(end_price / start_price)
            if not math.isfinite(target):
                raise ValueError("forecast target must be finite")
            forecasts_by_symbol[symbol_index].append(forecast)
            targets_by_symbol[symbol_index].append(target)

    by_symbol = tuple(
        ForecastSymbolDiagnostics(
            symbol=dataset.symbols[symbol_index],
            candidate_count=(decision_stop_index - config.start_index),
            eligible_count=len(forecasts_by_symbol[symbol_index]),
            excluded_counts=tuple(
                (reason, exclusions_by_symbol[symbol_index][reason])
                for reason in _FORECAST_EXCLUSION_REASONS
            ),
            metrics=_error_metrics(
                forecasts_by_symbol[symbol_index], targets_by_symbol[symbol_index]
            ),
        )
        for symbol_index in range(dataset.n_symbols)
    )
    all_forecasts = [value for values in forecasts_by_symbol for value in values]
    all_targets = [value for values in targets_by_symbol for value in values]
    excluded_counts = tuple(
        (
            reason,
            sum(exclusions[reason] for exclusions in exclusions_by_symbol),
        )
        for reason in _FORECAST_EXCLUSION_REASONS
    )
    return RidgeForecastDiagnostics(
        horizon_hours=model.horizon_hours,
        fit_cutoff_ns=int(model.fit_cutoff.astype("datetime64[ns]").astype(np.int64)),
        evaluation_start_index=config.start_index,
        decision_stop_index=decision_stop_index,
        evaluation_stop_index=config.stop_index,
        scoring_as_of_ns=scoring_as_of_ns,
        candidate_count=candidate_count,
        eligible_count=len(all_forecasts),
        excluded_counts=excluded_counts,
        pooled=_error_metrics(all_forecasts, all_targets),
        by_symbol=by_symbol,
    )


@dataclass(frozen=True, slots=True)
class RidgeArtifactComparison:
    """Verified inputs, requested Dataset scope, and ledger-backed entries.

    The time fields identify the requested boundary rows, not proof that a
    replay which terminates early completed the full range or settlement.
    """

    dataset_artifact: PublishedDatasetArtifact
    dataset_id: str
    symbols: tuple[str, ...]
    evaluation_start_ns: int
    evaluation_stop_ns: int
    ridge_artifact_digest: str
    binding: RidgeForecastArtifactBinding
    config: RidgeArtifactComparisonConfig
    entries: tuple[SharedCashStrategyComparisonEntry, ...]
    forecast_diagnostics: RidgeForecastDiagnostics | None = None


def _validate_loaded_inputs(
    dataset_artifact: PublishedDatasetArtifact,
    dataset: MarketDataset,
    model: RidgeForecastModel,
    binding: RidgeForecastArtifactBinding,
    config: RidgeArtifactComparisonConfig,
) -> tuple[int, int]:
    if type(dataset_artifact) is not PublishedDatasetArtifact:
        raise TypeError("paired dataset loader must return a PublishedDatasetArtifact")
    if type(dataset) is not MarketDataset:
        raise TypeError("paired dataset loader must return a MarketDataset")
    if type(model) is not RidgeForecastModel:
        raise TypeError("Ridge artifact loader must return a RidgeForecastModel")
    # The paired loader verifies the manifest, arrays, and Dataset identity in
    # one pass. Recomputing identity_verified here would hash every array twice.
    if dataset.dataset_id != binding.dataset_id:
        raise ValueError("Dataset ID differs from the Ridge artifact binding")
    if dataset_artifact.schema_version != binding.dataset_schema_version:
        raise ValueError("Dataset schema differs from the Ridge artifact binding")
    if dataset_artifact.artifact_digest != binding.dataset_artifact_sha256:
        raise ValueError("Dataset artifact digest differs from the Ridge binding")
    if tuple(dataset.feature_names) != binding.feature_names:
        raise ValueError("ordered Dataset feature names differ from the Ridge binding")
    if model.feature_indices != binding.feature_indices:
        raise ValueError("selected Ridge feature indices differ from the binding")
    if any(index >= dataset.n_features for index in binding.feature_indices):
        raise ValueError("Ridge feature index is outside the Dataset feature roster")
    if any(index >= dataset.n_symbols for index in binding.fit_symbol_indices):
        raise ValueError("Ridge fit symbol index is outside the Dataset symbol roster")
    selected_symbols = tuple(
        dataset.symbols[index] for index in binding.fit_symbol_indices
    )
    if selected_symbols != binding.fit_symbol_names:
        raise ValueError("Ridge fit symbol index/name mapping differs from the Dataset")
    if config.stop_index >= dataset.n_bars or config.start_index >= dataset.n_bars:
        raise ValueError("replay range is outside the loaded Dataset")

    start_ns = int(
        np.asarray(dataset.timestamps[config.start_index])
        .astype("datetime64[ns]")
        .astype(np.int64)
    )
    stop_ns = int(
        np.asarray(dataset.timestamps[config.stop_index])
        .astype("datetime64[ns]")
        .astype(np.int64)
    )
    if binding.fit_cutoff_ns > start_ns:
        raise ValueError("Ridge fit cutoff is after the Dataset decision start")
    return start_ns, stop_ns


def _ridge_factory(
    model: RidgeForecastModel,
    config: RidgeArtifactComparisonConfig,
    *,
    one_way_switch_cost: float | None,
) -> Callable[[], RidgeForecastStrategy]:
    def create() -> RidgeForecastStrategy:
        return RidgeForecastStrategy(
            model,
            entry_threshold=config.entry_threshold,
            exit_threshold=config.exit_threshold,
            one_way_switch_cost=one_way_switch_cost,
        )

    return create


def compare_ridge_artifact_shared_cash(
    *,
    dataset_root: str | Path,
    ridge_artifact_root: str | Path,
    expected_ridge_digest: str,
    expected_binding: RidgeForecastArtifactBinding,
    config: RidgeArtifactComparisonConfig,
) -> RidgeArtifactComparison:
    """Compare four fixed arms using one paired Dataset verification pass.

    The returned entries are descriptive in-memory replay results. This function
    does not fit a model, select a winner, or publish a research result.
    """

    if type(expected_binding) is not RidgeForecastArtifactBinding:
        raise TypeError("expected_binding must be a RidgeForecastArtifactBinding")
    if type(config) is not RidgeArtifactComparisonConfig:
        raise TypeError("config must be a RidgeArtifactComparisonConfig")

    dataset_artifact, dataset = load_published_market_dataset_artifact(dataset_root)
    model = load_ridge_forecast_artifact(
        ridge_artifact_root,
        expected_digest=expected_ridge_digest,
        expected_binding=expected_binding,
    )
    evaluation_start_ns, evaluation_stop_ns = _validate_loaded_inputs(
        dataset_artifact, dataset, model, expected_binding, config
    )
    forecast_diagnostics = (
        _ridge_forecast_diagnostics(dataset, model, config)
        if config.include_forecast_diagnostics
        else None
    )

    factories: dict[str, Callable[[], SingleSymbolStrategy]] = {
        "cash": lambda: ConstantIntentStrategy(PositionIntent.FLAT),
        "legacy_hysteresis": _ridge_factory(
            model,
            config,
            one_way_switch_cost=None,
        ),
        "position_aware_zero_cost": _ridge_factory(
            model,
            config,
            one_way_switch_cost=0.0,
        ),
        "position_aware_cost": _ridge_factory(
            model,
            config,
            one_way_switch_cost=config.one_way_switch_cost,
        ),
    }
    entries = compare_shared_cash_strategy_factories(
        dataset,
        factories,
        start_index=config.start_index,
        stop_index=config.stop_index,
        gross_budget=config.gross_budget,
        initial_capital=config.initial_capital,
        execution_cost=config.execution_cost,
        risk=PreTradeRisk(config.risk),
        market_order_profile=config.market_order_profile,
        minimum_hold_bars=config.minimum_hold_bars,
        settle_terminal_position=True,
        capture_ledger_evidence=True,
    )
    return RidgeArtifactComparison(
        dataset_artifact=dataset_artifact,
        dataset_id=dataset.dataset_id,
        symbols=tuple(dataset.symbols),
        evaluation_start_ns=evaluation_start_ns,
        evaluation_stop_ns=evaluation_stop_ns,
        ridge_artifact_digest=expected_ridge_digest,
        binding=expected_binding,
        config=config,
        entries=entries,
        forecast_diagnostics=forecast_diagnostics,
    )


__all__ = [
    "ForecastErrorMetrics",
    "ForecastSymbolDiagnostics",
    "RidgeArtifactComparison",
    "RidgeArtifactComparisonConfig",
    "RidgeForecastDiagnostics",
    "compare_ridge_artifact_shared_cash",
]
