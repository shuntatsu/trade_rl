from __future__ import annotations

from dataclasses import replace
from typing import Callable
from unittest.mock import Mock

import numpy as np
import pytest

from trade_rl.data.market import MarketDataset
from trade_rl.risk.pretrade import PreTradeRiskConfig
from trade_rl.simulation.accounting import BookState
from trade_rl.strategies.controls import ConstantIntentStrategy
from trade_rl.strategies.forecasts.ridge import (
    RidgeForecastModel,
    RidgeForecastStrategy,
)
from trade_rl.strategies.portfolio import (
    PortfolioDecisionSnapshot,
    PortfolioTargetProposal,
)
from trade_rl.strategies.portfolio_observation import (
    build_portfolio_decision_snapshot,
)
from trade_rl.strategies.position_intent import PositionIntent
from trade_rl.strategies.rules.adaptive import (
    AdaptiveProfitConfig,
    RegimeAdaptiveStrategy,
)
from trade_rl.strategies.rules.trend import TrendIntentConfig, TrendIntentStrategy

SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT")
FEATURE_NAMES = ("trend",)
GLOBAL_FEATURE_NAMES = ("regime",)


def _dataset(
    prices: tuple[float, float, float] = (100.0, 100.0, 100.0),
) -> MarketDataset:
    close = np.tile(np.asarray(prices, dtype=np.float64), (3, 1))
    features = np.asarray(
        [
            [[0.0], [0.0], [0.0]],
            [[0.05], [0.05], [-0.05]],
            [[0.0], [0.0], [0.0]],
        ],
        dtype=np.float64,
    )
    return MarketDataset(
        dataset_id="d" * 64,
        symbols=SYMBOLS,
        timestamps=np.datetime64("2026-01-01T00:00:00", "ns")
        + np.arange(3) * np.timedelta64(1, "h"),
        features=features,
        global_features=np.asarray([[0.0], [0.25], [0.0]], dtype=np.float64),
        open=close.copy(),
        high=close + 1.0,
        low=close - 1.0,
        close=close,
        volume=np.full((3, len(SYMBOLS)), 1_000.0, dtype=np.float64),
        funding_rate=np.zeros((3, len(SYMBOLS)), dtype=np.float64),
        tradable=np.ones((3, len(SYMBOLS)), dtype=np.bool_),
        feature_available=np.ones_like(features, dtype=np.bool_),
        feature_names=FEATURE_NAMES,
        global_feature_names=GLOBAL_FEATURE_NAMES,
        periods_per_year=8_760,
    )


def _snapshot(
    *,
    prices: tuple[float, float, float] = (100.0, 100.0, 100.0),
    index: int = 1,
    agent_stop_index: int = 2,
) -> PortfolioDecisionSnapshot:
    dataset = _dataset(prices)
    close = dataset.close
    book = BookState.from_weights(
        weights=np.asarray([0.2, 0.0, -0.15], dtype=np.float64),
        capital=1_000.0,
        prices=close[index],
    )
    return build_portfolio_decision_snapshot(
        dataset,
        index=index,
        agent_stop_index=agent_stop_index,
        book=book,
        active_orders=(),
        risk_config=PreTradeRiskConfig(),
    )


def _trend_factory() -> Callable[[], TrendIntentStrategy]:
    return lambda: TrendIntentStrategy(
        TrendIntentConfig(
            signal_index=0,
            entry_threshold=0.10,
            exit_threshold=0.02,
        )
    )


def _one_symbol_snapshot(*, index: int = 1) -> PortfolioDecisionSnapshot:
    snapshot = _snapshot(index=index)
    account = snapshot.account
    return replace(
        snapshot,
        symbols=(SYMBOLS[0],),
        features=(snapshot.features[0],),
        feature_available=(snapshot.feature_available[0],),
        feature_staleness=(snapshot.feature_staleness[0],),
        account=replace(
            account,
            exact_quantities=(account.exact_quantities[0],),
            quantities=(account.quantities[0],),
            weights=(account.weights[0],),
            mark_prices=(account.mark_prices[0],),
            contract_multipliers=(account.contract_multipliers[0],),
        ),
        active_orders=(),
    )


def test_adapter_maps_real_filled_sides_through_existing_trend_strategies() -> None:
    from trade_rl.strategies.portfolio_adapter import (
        SingleSymbolPortfolioPolicyAdapter,
    )

    snapshot = _snapshot()
    assert tuple(snapshot.account.exact_quantities) == ("2", "0", "-3/2")

    policy = SingleSymbolPortfolioPolicyAdapter(
        symbols=SYMBOLS,
        feature_names=FEATURE_NAMES,
        global_feature_names=GLOBAL_FEATURE_NAMES,
        intent_gross_budgets=(0.4, 0.3, 0.25),
        strategy_factories=(_trend_factory(), _trend_factory(), _trend_factory()),
    )

    proposal = policy.decide_portfolio(snapshot)

    assert isinstance(proposal, PortfolioTargetProposal)
    assert proposal.target_weights == (0.4, 0.0, -0.25)
    assert proposal.hold_current_quantity_symbols == ()


def test_adapter_validates_fixed_roster_before_running_a_strategy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from trade_rl.strategies.portfolio_adapter import (
        SingleSymbolPortfolioPolicyAdapter,
    )

    strategies = tuple(ConstantIntentStrategy(PositionIntent.FLAT) for _ in SYMBOLS)
    calls = tuple(Mock(wraps=strategy.decide) for strategy in strategies)
    for strategy, call in zip(strategies, calls, strict=True):
        monkeypatch.setattr(strategy, "decide", call)
    policy = SingleSymbolPortfolioPolicyAdapter(
        symbols=SYMBOLS,
        feature_names=FEATURE_NAMES,
        global_feature_names=GLOBAL_FEATURE_NAMES,
        intent_gross_budgets=(0.4, 0.3, 0.25),
        strategy_factories=tuple(
            lambda strategy=strategy: strategy for strategy in strategies
        ),
    )
    wrong_context = replace(_snapshot(), symbols=tuple(reversed(SYMBOLS)))

    with pytest.raises(ValueError, match="symbols"):
        policy.decide_portfolio(wrong_context)
    assert tuple(call.call_count for call in calls) == (0, 0, 0)
    assert policy.decide_portfolio(_snapshot()).target_weights == (0.0, 0.0, 0.0)
    assert tuple(call.call_count for call in calls) == (1, 1, 1)


def test_adapter_rejects_repeated_decision_index_before_callback() -> None:
    from trade_rl.strategies.portfolio_adapter import (
        SingleSymbolPortfolioPolicyAdapter,
    )

    strategy = ConstantIntentStrategy(PositionIntent.LONG)
    call = Mock(wraps=strategy.decide)
    policy = SingleSymbolPortfolioPolicyAdapter(
        symbols=(SYMBOLS[0],),
        feature_names=FEATURE_NAMES,
        global_feature_names=GLOBAL_FEATURE_NAMES,
        intent_gross_budgets=(0.4,),
        strategy_factories=(lambda: strategy,),
    )
    # Wrap after construction to keep the same supported built-in instance.
    strategy.decide = call  # type: ignore[method-assign]

    first = _one_symbol_snapshot(index=0)
    assert policy.decide_portfolio(first).target_weights == (0.4,)

    with pytest.raises(ValueError, match="index must advance"):
        policy.decide_portfolio(first)
    assert call.call_count == 1
    assert policy.decide_portfolio(_one_symbol_snapshot(index=1)).target_weights == (
        0.4,
    )
    assert call.call_count == 2


def test_adapter_rejects_exact_quantity_view_mismatch_before_callback() -> None:
    from trade_rl.strategies.portfolio_adapter import (
        SingleSymbolPortfolioPolicyAdapter,
    )

    strategy = ConstantIntentStrategy(PositionIntent.FLAT)
    call = Mock(wraps=strategy.decide)
    policy = SingleSymbolPortfolioPolicyAdapter(
        symbols=(SYMBOLS[0],),
        feature_names=FEATURE_NAMES,
        global_feature_names=GLOBAL_FEATURE_NAMES,
        intent_gross_budgets=(0.4,),
        strategy_factories=(lambda: strategy,),
    )
    strategy.decide = call  # type: ignore[method-assign]
    snapshot = _one_symbol_snapshot()
    inconsistent = replace(
        snapshot,
        account=replace(snapshot.account, quantities=(2.5,)),
    )

    with pytest.raises(ValueError, match="exact and reported"):
        policy.decide_portfolio(inconsistent)
    assert call.call_count == 0
    assert policy.decide_portfolio(snapshot).target_weights == (0.0,)
    assert call.call_count == 1


def test_ridge_adapter_uses_the_configured_per_symbol_intent_budget() -> None:
    from trade_rl.strategies.portfolio_adapter import (
        SingleSymbolPortfolioPolicyAdapter,
    )

    model = RidgeForecastModel(
        feature_indices=(0,),
        feature_mean=np.asarray([0.0]),
        feature_scale=np.asarray([1.0]),
        coefficients=np.asarray([1.0]),
        intercept=0.0,
        horizon_hours=1,
        alpha=1.0,
        n_samples=1,
        fit_cutoff=np.datetime64("2026-01-01T00:00:00", "ns"),
    )
    policy = SingleSymbolPortfolioPolicyAdapter(
        symbols=SYMBOLS,
        feature_names=FEATURE_NAMES,
        global_feature_names=GLOBAL_FEATURE_NAMES,
        intent_gross_budgets=(0.4, 0.3, 0.25),
        strategy_factories=(
            lambda: RidgeForecastStrategy(
                model,
                entry_threshold=0.02,
                exit_threshold=0.01,
            ),
            lambda: RidgeForecastStrategy(
                model,
                entry_threshold=0.02,
                exit_threshold=0.01,
            ),
            lambda: RidgeForecastStrategy(
                model,
                entry_threshold=0.02,
                exit_threshold=0.01,
            ),
        ),
    )

    proposal = policy.decide_portfolio(_snapshot())

    assert proposal.target_weights == (0.4, 0.3, -0.25)


def test_cost_aware_ridge_receives_budget_and_actual_filled_position() -> None:
    from trade_rl.strategies.portfolio_adapter import (
        SingleSymbolPortfolioPolicyAdapter,
    )

    model = RidgeForecastModel(
        feature_indices=(0,),
        feature_mean=np.asarray([0.0]),
        feature_scale=np.asarray([1.0]),
        coefficients=np.asarray([0.0]),
        intercept=-0.05,
        horizon_hours=1,
        alpha=1.0,
        n_samples=1,
        fit_cutoff=np.datetime64("2026-01-01T00:00:00", "ns"),
    )

    def proposal(budget: float) -> PortfolioTargetProposal:
        policy = SingleSymbolPortfolioPolicyAdapter(
            symbols=(SYMBOLS[0],),
            feature_names=FEATURE_NAMES,
            global_feature_names=GLOBAL_FEATURE_NAMES,
            intent_gross_budgets=(budget,),
            strategy_factories=(
                lambda: RidgeForecastStrategy(
                    model,
                    entry_threshold=0.10,
                    exit_threshold=0.01,
                    one_way_switch_cost=0.10,
                ),
            ),
        )
        return policy.decide_portfolio(_one_symbol_snapshot())

    # With the same -5% log-return forecast and a 10% switch-cost proxy, a
    # 10% budget leaves the filled 20% long intact; a 40% budget exits it.
    # This witnesses that the adapter supplies the actual filled weight and
    # configured intent budget to Ridge's cost-aware controller.
    assert proposal(0.1).target_weights == (0.1,)
    assert proposal(0.4).target_weights == (0.0,)


def test_cash_policy_is_an_explicit_fixed_roster_zero_target() -> None:
    from trade_rl.strategies.portfolio_adapter import CashPortfolioPolicy

    policy = CashPortfolioPolicy(symbols=SYMBOLS)

    assert policy.decide_portfolio(_snapshot()) == PortfolioTargetProposal(
        target_weights=(0.0, 0.0, 0.0)
    )


def test_adapter_runs_through_the_existing_shared_portfolio_replay() -> None:
    from trade_rl.evaluation.replay import run_shared_portfolio_replay
    from trade_rl.risk.pretrade import PreTradeRisk
    from trade_rl.simulation.execution import ExecutionCostConfig
    from trade_rl.strategies.portfolio_adapter import (
        SingleSymbolPortfolioPolicyAdapter,
    )

    policy = SingleSymbolPortfolioPolicyAdapter(
        symbols=SYMBOLS,
        feature_names=FEATURE_NAMES,
        global_feature_names=GLOBAL_FEATURE_NAMES,
        intent_gross_budgets=(0.4, 0.3, 0.25),
        strategy_factories=(
            lambda: ConstantIntentStrategy(PositionIntent.LONG),
            lambda: ConstantIntentStrategy(PositionIntent.FLAT),
            lambda: ConstantIntentStrategy(PositionIntent.SHORT),
        ),
    )
    result = run_shared_portfolio_replay(
        _dataset(),
        policy,
        start_index=0,
        stop_index=2,
        initial_capital=1_000.0,
        execution_cost=ExecutionCostConfig.zero(),
        risk=PreTradeRisk(
            PreTradeRiskConfig(
                max_gross=1.0,
                max_abs_weight=0.4,
                max_turnover=None,
                drawdown_start=1.0,
                drawdown_stop=1.0,
            )
        ),
        capture_ledger_evidence=True,
    )

    assert tuple(decision.proposal_weights for decision in result.decisions) == (
        (0.4, 0.0, -0.25),
        (0.4, 0.0, -0.25),
    )
    quantities = result.book.exact_quantities
    assert quantities[0] > 0
    assert quantities[1] == 0
    assert quantities[2] < 0
    assert result.ledger_evidence is not None
    assert len(result.ledger_evidence.intervals.intervals) == 2


@pytest.mark.parametrize(
    "budget",
    (True, 0.0, -0.1, float("nan"), float("inf"), 10**400),
)
def test_adapter_rejects_invalid_symbol_budget_before_strategy_construction(
    budget: object,
) -> None:
    from trade_rl.strategies.portfolio_adapter import (
        SingleSymbolPortfolioPolicyAdapter,
    )

    constructed = 0

    def factory() -> ConstantIntentStrategy:
        nonlocal constructed
        constructed += 1
        return ConstantIntentStrategy(PositionIntent.FLAT)

    with pytest.raises((TypeError, ValueError)):
        SingleSymbolPortfolioPolicyAdapter(
            symbols=SYMBOLS,
            feature_names=FEATURE_NAMES,
            global_feature_names=GLOBAL_FEATURE_NAMES,
            intent_gross_budgets=(budget, 0.2, 0.2),
            strategy_factories=(factory, factory, factory),
        )
    assert constructed == 0


def test_adapter_rejects_strategy_object_reuse_within_an_arm() -> None:
    from trade_rl.strategies.portfolio_adapter import (
        SingleSymbolPortfolioPolicyAdapter,
    )

    shared = ConstantIntentStrategy(PositionIntent.FLAT)

    with pytest.raises(ValueError, match="distinct"):
        SingleSymbolPortfolioPolicyAdapter(
            symbols=SYMBOLS,
            feature_names=FEATURE_NAMES,
            global_feature_names=GLOBAL_FEATURE_NAMES,
            intent_gross_budgets=(0.4, 0.3, 0.25),
            strategy_factories=(lambda: shared, lambda: shared, _trend_factory()),
        )


def test_adapter_rejects_custom_strategy_types_instead_of_claiming_parity() -> None:
    from trade_rl.strategies.portfolio_adapter import (
        SingleSymbolPortfolioPolicyAdapter,
    )

    class CustomConstantStrategy(ConstantIntentStrategy):
        pass

    with pytest.raises(TypeError, match="supported strategy"):
        SingleSymbolPortfolioPolicyAdapter(
            symbols=(SYMBOLS[0],),
            feature_names=FEATURE_NAMES,
            global_feature_names=GLOBAL_FEATURE_NAMES,
            intent_gross_budgets=(0.4,),
            strategy_factories=(lambda: CustomConstantStrategy(PositionIntent.FLAT),),
        )


def test_adapter_rejects_age_dependent_builtin_strategy() -> None:
    from trade_rl.strategies.portfolio_adapter import (
        SingleSymbolPortfolioPolicyAdapter,
    )

    with pytest.raises(TypeError, match="supported strategy"):
        SingleSymbolPortfolioPolicyAdapter(
            symbols=(SYMBOLS[0],),
            feature_names=FEATURE_NAMES,
            global_feature_names=GLOBAL_FEATURE_NAMES,
            intent_gross_budgets=(0.4,),
            strategy_factories=(
                lambda: RegimeAdaptiveStrategy(AdaptiveProfitConfig()),
            ),
        )


def test_callback_exception_after_prior_symbol_poison_prevents_proposal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from trade_rl.strategies.portfolio_adapter import (
        SingleSymbolPortfolioPolicyAdapter,
    )

    first = ConstantIntentStrategy(PositionIntent.LONG)
    bad_second = TrendIntentStrategy(
        TrendIntentConfig(signal_index=3, entry_threshold=0.1, exit_threshold=0.02)
    )
    never_reached = ConstantIntentStrategy(PositionIntent.FLAT)
    first_call = Mock(wraps=first.decide)
    last_call = Mock(wraps=never_reached.decide)
    monkeypatch.setattr(first, "decide", first_call)
    monkeypatch.setattr(never_reached, "decide", last_call)
    policy = SingleSymbolPortfolioPolicyAdapter(
        symbols=SYMBOLS,
        feature_names=FEATURE_NAMES,
        global_feature_names=GLOBAL_FEATURE_NAMES,
        intent_gross_budgets=(0.4, 0.3, 0.25),
        strategy_factories=(
            lambda: first,
            lambda: bad_second,
            lambda: never_reached,
        ),
    )

    with pytest.raises(ValueError, match="signal_index"):
        policy.decide_portfolio(_snapshot())
    assert first_call.call_count == 1
    assert last_call.call_count == 0
    with pytest.raises(RuntimeError, match="poison"):
        policy.decide_portfolio(_snapshot())


@pytest.mark.parametrize("invalid_intent", (None, True, 1, "long"))
def test_invalid_strategy_output_poison_prevents_a_proposal_or_reuse(
    invalid_intent: object,
) -> None:
    from trade_rl.strategies.portfolio_adapter import (
        SingleSymbolPortfolioPolicyAdapter,
    )

    strategy = ConstantIntentStrategy(PositionIntent.FLAT)
    strategy.intent = invalid_intent  # type: ignore[assignment]
    policy = SingleSymbolPortfolioPolicyAdapter(
        symbols=(SYMBOLS[0],),
        feature_names=FEATURE_NAMES,
        global_feature_names=GLOBAL_FEATURE_NAMES,
        intent_gross_budgets=(0.4,),
        strategy_factories=(lambda: strategy,),
    )
    one_symbol = _one_symbol_snapshot()

    with pytest.raises(TypeError, match="PositionIntent"):
        policy.decide_portfolio(one_symbol)
    with pytest.raises(RuntimeError, match="poison"):
        policy.decide_portfolio(one_symbol)
