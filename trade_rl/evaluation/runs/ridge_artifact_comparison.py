"""Filesystem comparison of shared-cash arms using one frozen Ridge model."""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from typing import Callable

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
    )


__all__ = [
    "RidgeArtifactComparison",
    "RidgeArtifactComparisonConfig",
    "compare_ridge_artifact_shared_cash",
]
