"""Fixed four-family comparison over one verified Dataset and native replay owner."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np

from trade_rl._validation import require_sha256
from trade_rl.data.artifacts.publication import (
    PublishedDatasetArtifact,
    load_published_market_dataset_artifact,
)
from trade_rl.data.market import MarketDataset
from trade_rl.data.market_order_rules import MarketOrderProfile
from trade_rl.evaluation.replay import (
    SharedCashPortfolioReplayResult,
    run_shared_portfolio_replay,
)
from trade_rl.risk.pretrade import PreTradeRisk, PreTradeRiskConfig
from trade_rl.simulation.execution import ExecutionCostConfig, MarketExecutor
from trade_rl.simulation.native_transition import (
    validate_risk_execution_compatibility,
)
from trade_rl.simulation.objectives.binding import BoundObjectiveClock
from trade_rl.strategies.forecasts.ridge import (
    RidgeForecastModel,
    RidgeForecastStrategy,
)
from trade_rl.strategies.forecasts.ridge_artifact import (
    RidgeForecastArtifactBinding,
    load_ridge_forecast_artifact,
)
from trade_rl.strategies.portfolio import PortfolioPolicy
from trade_rl.strategies.portfolio_adapter import (
    CashPortfolioPolicy,
    SingleSymbolPortfolioPolicyAdapter,
)
from trade_rl.strategies.rules.trend import TrendIntentConfig, TrendIntentStrategy

if TYPE_CHECKING:
    from trade_rl.strategies.rl.portfolio_env import SharedPortfolioGymEnv
    from trade_rl.strategies.rl.portfolio_inference_bundle import (
        PortfolioInferencePolicy,
    )


FourFamilyName = Literal["cash", "fixed_trend", "ridge_forecast", "joint_ppo"]
_FAMILY_NAMES: tuple[FourFamilyName, ...] = (
    "cash",
    "fixed_trend",
    "ridge_forecast",
    "joint_ppo",
)


@dataclass(frozen=True, slots=True)
class PortfolioFamilyReplayOutcome:
    """One arm's unmodified native result and explicit completion flags."""

    family: FourFamilyName
    replay: SharedCashPortfolioReplayResult
    expected_interval_count: int
    actual_interval_count: int
    termination_reason: str | None
    insolvent: bool
    terminal_flat: bool
    no_active_orders: bool

    @property
    def window_complete(self) -> bool:
        """Whether the native result reached the full non-terminated endpoint."""
        return (
            self.actual_interval_count == self.expected_interval_count
            and self.termination_reason is None
            and not self.insolvent
        )


@dataclass(frozen=True, slots=True)
class FourFamilyPortfolioReplayResult:
    """In-memory, fixed-order outcomes; this is not a persisted study result."""

    dataset_artifact: PublishedDatasetArtifact
    dataset_id: str
    symbols: tuple[str, ...]
    start_index: int
    stop_index: int
    ridge_artifact_digest: str
    ppo_bundle_sha256: str
    families: tuple[PortfolioFamilyReplayOutcome, ...]


def load_portfolio_inference_bundle(
    root: str | Path,
    *,
    expected_bundle_sha256: str,
    expected_manifest_sha256: str,
    expected_completion_sha256: str,
    expected_training_context_sha256: str,
    expected_checkpoint_sha256: str,
    runtime_env: SharedPortfolioGymEnv,
) -> PortfolioInferencePolicy:
    """Load the optional PPO capability lazily, keeping it optional at import time."""
    from trade_rl.strategies.rl.portfolio_inference_bundle import (
        load_portfolio_inference_bundle as load_bundle,
    )

    return load_bundle(
        root,
        expected_bundle_sha256=expected_bundle_sha256,
        expected_manifest_sha256=expected_manifest_sha256,
        expected_completion_sha256=expected_completion_sha256,
        expected_training_context_sha256=expected_training_context_sha256,
        expected_checkpoint_sha256=expected_checkpoint_sha256,
        runtime_env=runtime_env,
    )


def _positive_weights(value: object, *, n_symbols: int) -> tuple[float, ...]:
    if type(value) is not tuple or len(value) != n_symbols:
        raise ValueError("target_abs_weights must match the Dataset symbol roster")
    weights: list[float] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, Real):
            raise TypeError("target_abs_weights must contain real numbers")
        try:
            weight = float(item)
        except OverflowError as error:
            raise ValueError(
                "target_abs_weights must be finite and in (0, 1]"
            ) from error
        if not math.isfinite(weight) or not 0.0 < weight <= 1.0:
            raise ValueError("target_abs_weights must be finite and in (0, 1]")
        weights.append(weight)
    return tuple(weights)


def _validate_ridge_binding(
    artifact: PublishedDatasetArtifact,
    dataset: MarketDataset,
    model: RidgeForecastModel,
    binding: RidgeForecastArtifactBinding,
    *,
    expected_dataset_digest: str,
    expected_symbols: tuple[str, ...],
    start_index: int,
    stop_index: int,
) -> None:
    if type(artifact) is not PublishedDatasetArtifact:
        raise TypeError("paired Dataset loader must return a PublishedDatasetArtifact")
    if type(dataset) is not MarketDataset:
        raise TypeError("paired Dataset loader must return a MarketDataset")
    if type(model) is not RidgeForecastModel:
        raise TypeError("Ridge artifact loader must return a RidgeForecastModel")
    if artifact.artifact_digest != expected_dataset_digest:
        raise ValueError("Dataset artifact digest differs from the external pin")
    if tuple(dataset.symbols) != expected_symbols:
        raise ValueError("Dataset symbols differ from the fixed comparison roster")
    if dataset.dataset_id != binding.dataset_id:
        raise ValueError("Dataset ID differs from the Ridge artifact binding")
    if artifact.schema_version != binding.dataset_schema_version:
        raise ValueError("Dataset schema differs from the Ridge artifact binding")
    if artifact.artifact_digest != binding.dataset_artifact_sha256:
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
    if not 0 <= start_index < stop_index < dataset.n_bars:
        raise ValueError("comparison range is outside the loaded Dataset")

    fit_cutoff_ns = int(
        np.asarray(model.fit_cutoff).astype("datetime64[ns]").astype(np.int64)
    )
    if fit_cutoff_ns != binding.fit_cutoff_ns:
        raise ValueError("Ridge model fit cutoff differs from its binding")
    start_ns = int(
        np.asarray(dataset.timestamps[start_index])
        .astype("datetime64[ns]")
        .astype(np.int64)
    )
    if fit_cutoff_ns > start_ns:
        raise ValueError("Ridge fit cutoff is after the Dataset decision start")


def _adapter(
    dataset: MarketDataset,
    target_abs_weights: tuple[float, ...],
    factory: Callable[[], object],
) -> SingleSymbolPortfolioPolicyAdapter:
    return SingleSymbolPortfolioPolicyAdapter(
        symbols=tuple(dataset.symbols),
        feature_names=tuple(dataset.feature_names),
        global_feature_names=tuple(dataset.global_feature_names),
        intent_gross_budgets=target_abs_weights,
        strategy_factories=tuple(factory for _ in dataset.symbols),
    )


def _validated_initial_capital(value: object) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError("initial_capital must be a real non-boolean number")
    try:
        capital = float(value)
    except OverflowError as error:
        raise ValueError("initial_capital must be finite and positive") from error
    if not math.isfinite(capital) or capital <= 0.0:
        raise ValueError("initial_capital must be finite and positive")
    return capital


def run_four_family_shared_portfolio_replay(
    *,
    dataset_artifact_root: str | Path,
    expected_dataset_artifact_sha256: str,
    expected_symbols: tuple[str, ...],
    ridge_artifact_root: str | Path,
    expected_ridge_artifact_sha256: str,
    expected_ridge_binding: RidgeForecastArtifactBinding,
    bundle_root: str | Path,
    expected_bundle_sha256: str,
    expected_manifest_sha256: str,
    expected_completion_sha256: str,
    expected_training_context_sha256: str,
    expected_checkpoint_sha256: str,
    bound: BoundObjectiveClock,
    start_index: int,
    stop_index: int,
    initial_capital: float,
    target_abs_weights: tuple[float, ...],
    trend_config: TrendIntentConfig,
    ridge_entry_threshold: float,
    ridge_exit_threshold: float,
    ridge_one_way_switch_cost: float | None,
    execution_cost: ExecutionCostConfig,
    risk_config: PreTradeRiskConfig,
    market_order_profile: MarketOrderProfile | None = None,
    on_family_completed: Callable[[PortfolioFamilyReplayOutcome], None] | None = None,
) -> FourFamilyPortfolioReplayResult:
    """Run four fixed policy families through the same native shared-account path.

    All dataset, model, bundle, window, profile, cost, risk, and policy preflight
    completes before the first arm starts. Each arm then receives a fresh policy,
    pre-trade risk controller, native session, and ledger; immutable Dataset and
    execution/risk/profile configuration values are shared. Early native
    termination is retained with its raw replay and endpoint flags. When supplied,
    ``on_family_completed`` receives each outcome immediately before the next arm
    starts, so a later failure cannot hide already completed outcomes. Observer
    failures propagate and stop the comparison; this function owns no journal or
    attempt reservation.
    """
    for name, digest in (
        ("expected_dataset_artifact_sha256", expected_dataset_artifact_sha256),
        ("expected_ridge_artifact_sha256", expected_ridge_artifact_sha256),
        ("expected_bundle_sha256", expected_bundle_sha256),
        ("expected_manifest_sha256", expected_manifest_sha256),
        ("expected_completion_sha256", expected_completion_sha256),
        ("expected_training_context_sha256", expected_training_context_sha256),
        ("expected_checkpoint_sha256", expected_checkpoint_sha256),
    ):
        require_sha256(digest, field=name)
    if type(expected_symbols) is not tuple or not expected_symbols:
        raise TypeError("expected_symbols must be a non-empty tuple")
    if any(not isinstance(symbol, str) or not symbol for symbol in expected_symbols):
        raise ValueError("expected_symbols must contain non-empty names")
    if len(set(expected_symbols)) != len(expected_symbols):
        raise ValueError("expected_symbols must contain unique names")
    if type(expected_ridge_binding) is not RidgeForecastArtifactBinding:
        raise TypeError("expected_ridge_binding must be a RidgeForecastArtifactBinding")
    if type(bound) is not BoundObjectiveClock:
        raise TypeError("bound must be a BoundObjectiveClock")
    if type(trend_config) is not TrendIntentConfig:
        raise TypeError("trend_config must be a TrendIntentConfig")
    if type(execution_cost) is not ExecutionCostConfig:
        raise TypeError("execution_cost must be an ExecutionCostConfig")
    if type(risk_config) is not PreTradeRiskConfig:
        raise TypeError("risk_config must be a PreTradeRiskConfig")
    if (
        market_order_profile is not None
        and type(market_order_profile) is not MarketOrderProfile
    ):
        raise TypeError(
            "market_order_profile must be a verified MarketOrderProfile or None"
        )
    if on_family_completed is not None and not callable(on_family_completed):
        raise TypeError("on_family_completed must be callable or None")
    if (
        isinstance(start_index, bool)
        or type(start_index) is not int
        or isinstance(stop_index, bool)
        or type(stop_index) is not int
        or not 0 <= start_index < stop_index
    ):
        raise ValueError("comparison range must satisfy integer 0 <= start < stop")
    initial_capital = _validated_initial_capital(initial_capital)
    if bound.objective.capital.total_initial_capital != initial_capital:
        raise ValueError("bound capital differs from the shared replay capital")
    weights = _positive_weights(target_abs_weights, n_symbols=len(expected_symbols))

    dataset_artifact, dataset = load_published_market_dataset_artifact(
        dataset_artifact_root
    )
    if type(dataset_artifact) is not PublishedDatasetArtifact:
        raise TypeError("paired Dataset loader must return a PublishedDatasetArtifact")
    if type(dataset) is not MarketDataset:
        raise TypeError("paired Dataset loader must return a MarketDataset")
    if dataset_artifact.artifact_digest != expected_dataset_artifact_sha256:
        raise ValueError("Dataset artifact digest differs from the external pin")
    if tuple(dataset.symbols) != expected_symbols:
        raise ValueError("Dataset symbols differ from the fixed comparison roster")
    if len(weights) != dataset.n_symbols:
        raise ValueError("target budgets differ from the Dataset symbol roster")
    if not 0 <= start_index < stop_index < dataset.n_bars:
        raise ValueError("comparison range is outside the loaded Dataset")
    if trend_config.signal_index >= dataset.n_features:
        raise ValueError("trend signal index is outside the Dataset feature roster")

    model = load_ridge_forecast_artifact(
        ridge_artifact_root,
        expected_digest=expected_ridge_artifact_sha256,
        expected_binding=expected_ridge_binding,
    )
    _validate_ridge_binding(
        dataset_artifact,
        dataset,
        model,
        expected_ridge_binding,
        expected_dataset_digest=expected_dataset_artifact_sha256,
        expected_symbols=expected_symbols,
        start_index=start_index,
        stop_index=stop_index,
    )

    from trade_rl.strategies.rl.portfolio_env import SharedPortfolioGymEnv

    # This environment validates the exact PPO runtime contract but is never
    # reset or stepped. Its constructor leaves the native session uncreated.
    validation_env = SharedPortfolioGymEnv(
        dataset,
        start_index=start_index,
        stop_index=stop_index,
        bound=bound,
        target_abs_weights=weights,
        execution_cost=execution_cost,
        risk_config=risk_config,
        market_order_profile=market_order_profile,
    )
    bundle_policy = load_portfolio_inference_bundle(
        bundle_root,
        expected_bundle_sha256=expected_bundle_sha256,
        expected_manifest_sha256=expected_manifest_sha256,
        expected_completion_sha256=expected_completion_sha256,
        expected_training_context_sha256=expected_training_context_sha256,
        expected_checkpoint_sha256=expected_checkpoint_sha256,
        runtime_env=validation_env,
    )
    if not callable(getattr(bundle_policy, "decide_portfolio", None)):
        raise TypeError("PPO bundle loader must return a portfolio inference policy")

    # Construct all arm-local actor and risk state before any replay begins.
    cash_policy = CashPortfolioPolicy(symbols=expected_symbols)
    trend_policy = _adapter(
        dataset,
        weights,
        lambda: TrendIntentStrategy(trend_config),
    )
    ridge_policy = _adapter(
        dataset,
        weights,
        lambda: RidgeForecastStrategy(
            model,
            entry_threshold=ridge_entry_threshold,
            exit_threshold=ridge_exit_threshold,
            one_way_switch_cost=ridge_one_way_switch_cost,
        ),
    )
    policies: tuple[PortfolioPolicy, ...] = (
        cash_policy,
        trend_policy,
        ridge_policy,
        bundle_policy,
    )
    risk_controllers = tuple(PreTradeRisk(risk_config) for _ in _FAMILY_NAMES)

    # Reuse the simulation owner's compatibility check before any arm session
    # exists; constructing an executor does not advance a book or order ledger.
    preflight_executor = MarketExecutor(
        dataset,
        execution_cost,
        market_order_profile=market_order_profile,
    )
    validate_risk_execution_compatibility(risk_controllers[0], preflight_executor)

    outcomes: list[PortfolioFamilyReplayOutcome] = []
    expected_intervals = stop_index - start_index
    for family, policy, risk in zip(
        _FAMILY_NAMES, policies, risk_controllers, strict=True
    ):
        replay = run_shared_portfolio_replay(
            dataset,
            policy,
            start_index=start_index,
            stop_index=stop_index,
            initial_capital=initial_capital,
            execution_cost=execution_cost,
            risk=risk,
            market_order_profile=market_order_profile,
            settle_terminal_position=True,
            capture_ledger_evidence=True,
        )
        ledger = replay.ledger_evidence
        if ledger is None:
            raise RuntimeError(
                "native portfolio replay omitted required ledger evidence"
            )
        interval_evidence = ledger.intervals
        actual_intervals = len(interval_evidence.intervals)
        termination_reason = interval_evidence.termination_reason
        outcome = PortfolioFamilyReplayOutcome(
            family=family,
            replay=replay,
            expected_interval_count=expected_intervals,
            actual_interval_count=actual_intervals,
            termination_reason=termination_reason,
            insolvent=bool(replay.book.insolvent),
            terminal_flat=all(
                quantity == "0"
                for quantity in interval_evidence.terminal_exact_quantities
            ),
            no_active_orders=not interval_evidence.active_order_remainders,
        )
        outcomes.append(outcome)
        if on_family_completed is not None:
            on_family_completed(outcome)

    return FourFamilyPortfolioReplayResult(
        dataset_artifact=dataset_artifact,
        dataset_id=dataset.dataset_id,
        symbols=expected_symbols,
        start_index=start_index,
        stop_index=stop_index,
        ridge_artifact_digest=expected_ridge_artifact_sha256,
        ppo_bundle_sha256=expected_bundle_sha256,
        families=tuple(outcomes),
    )


__all__ = [
    "FourFamilyPortfolioReplayResult",
    "PortfolioFamilyReplayOutcome",
    "run_four_family_shared_portfolio_replay",
]
