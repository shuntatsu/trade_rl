from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pytest

from trade_rl.data.market import MarketDataset
from trade_rl.evaluation import replay as replay_module
from trade_rl.risk import PreTradeRisk, PreTradeRiskConfig
from trade_rl.simulation import ExecutionCostConfig, MarketExecutor
from trade_rl.strategies.position_intent import PositionIntent


@dataclass
class FixedIntent:
    intent: PositionIntent
    observations: list[object] = field(default_factory=list)

    def decide(self, observation: object) -> PositionIntent:
        self.observations.append(observation)
        return self.intent


def _market(
    close: np.ndarray, *, symbols: tuple[str, ...] | None = None
) -> MarketDataset:
    close_array = np.asarray(close, dtype=np.float64)
    if close_array.ndim != 2:
        raise ValueError("close must be two-dimensional")
    n_bars, n_symbols = close_array.shape
    resolved_symbols = symbols or tuple(f"SYM{index}" for index in range(n_symbols))
    open_price = np.vstack((close_array[0], close_array[:-1]))
    features = np.zeros((n_bars, n_symbols, 1), dtype=np.float32)
    for symbol_index in range(n_symbols):
        features[:, symbol_index, 0] = np.arange(n_bars) + 10 * symbol_index
    return MarketDataset(
        dataset_id="d" * 64,
        symbols=resolved_symbols,
        timestamps=np.datetime64("2026-01-01T00:00:00", "ns")
        + np.arange(n_bars) * np.timedelta64(1, "h"),
        features=features,
        global_features=np.zeros((n_bars, 1), dtype=np.float32),
        open=open_price,
        high=np.maximum(open_price, close_array),
        low=np.minimum(open_price, close_array),
        close=close_array,
        volume=np.full((n_bars, n_symbols), 1_000_000.0),
        funding_rate=np.zeros((n_bars, n_symbols)),
        tradable=np.ones((n_bars, n_symbols), dtype=np.bool_),
        feature_available=np.ones((n_bars, n_symbols, 1), dtype=np.bool_),
        feature_names=("signal",),
        global_feature_names=("regime",),
        periods_per_year=8_760,
    )


def _two_symbol_market() -> MarketDataset:
    return _market(
        np.asarray(
            [
                [100.0, 200.0],
                [100.0, 200.0],
                [110.0, 190.0],
                [120.0, 180.0],
                [130.0, 170.0],
                [140.0, 160.0],
            ]
        ),
        symbols=("BTCUSDT", "ETHUSDT"),
    )


def _risk(*, max_gross: float, max_turnover: float | None) -> PreTradeRisk:
    return PreTradeRisk(
        PreTradeRiskConfig(
            max_gross=max_gross,
            max_abs_weight=max_gross,
            max_turnover=max_turnover,
            drawdown_start=1.0,
            drawdown_stop=1.0,
        )
    )


def test_shared_cash_replay_api_exists() -> None:
    assert callable(replay_module.run_shared_cash_replay)


def test_one_symbol_shared_cash_is_single_symbol_equivalent() -> None:
    dataset = _market(
        np.asarray([[100.0], [100.0], [110.0], [120.0], [130.0], [140.0]])
    )
    single_strategy = FixedIntent(PositionIntent.LONG)
    shared_strategy = FixedIntent(PositionIntent.LONG)
    execution_cost = ExecutionCostConfig.zero()

    single = replay_module.run_single_symbol_replay(
        dataset,
        single_strategy,
        start_index=0,
        stop_index=5,
        gross_budget=0.5,
        initial_capital=1_000.0,
        execution_cost=execution_cost,
    )
    shared = replay_module.run_shared_cash_replay(
        dataset,
        (shared_strategy,),
        start_index=0,
        stop_index=5,
        gross_budget=0.5,
        initial_capital=1_000.0,
        execution_cost=execution_cost,
    )

    assert shared.returns.values == pytest.approx(single.returns.values)
    np.testing.assert_allclose(shared.book.quantities, single.book.quantities)
    assert shared.book.cash == pytest.approx(single.book.cash)
    assert shared.book.portfolio_value == pytest.approx(single.book.portfolio_value)
    assert shared.diagnostics == single.diagnostics
    assert [item.intents[0] for item in shared.decisions] == [
        item.intent for item in single.decisions
    ]
    assert [item.changed_intents[0] for item in shared.decisions] == [
        item.changed_intent for item in single.decisions
    ]
    assert [item.target_weights[0] for item in shared.decisions] == pytest.approx(
        [item.target_weight for item in single.decisions]
    )


def test_simultaneous_long_proposals_share_one_gross_budget() -> None:
    dataset = _market(np.full((5, 2), [100.0, 200.0]))
    result = replay_module.run_shared_cash_replay(
        dataset,
        (FixedIntent(PositionIntent.LONG), FixedIntent(PositionIntent.LONG)),
        start_index=0,
        stop_index=4,
        gross_budget=1.0,
        initial_capital=1_000.0,
        execution_cost=ExecutionCostConfig.zero(),
    )

    first = result.decisions[0]
    assert first.proposal_weights == pytest.approx((1.0, 1.0))
    assert first.target_weights == pytest.approx((0.5, 0.5))
    assert "max_gross" in first.risk_reasons
    assert result.book.portfolio_value == pytest.approx(1_000.0)
    assert result.book.cash == pytest.approx(0.0)
    np.testing.assert_allclose(result.book.weights, np.asarray([0.5, 0.5]))
    assert 1_000.0 * np.prod(1.0 + np.asarray(result.returns.values)) == pytest.approx(
        result.book.portfolio_value
    )


def test_risk_and_execution_run_once_per_bar_after_all_decisions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = _market(np.full((5, 2), [100.0, 200.0]))
    strategies = (FixedIntent(PositionIntent.LONG), FixedIntent(PositionIntent.LONG))
    execution_calls: list[int] = []
    risk_calls: list[tuple[float, ...]] = []
    original_execute = MarketExecutor.execute_interval
    original_constrain = PreTradeRisk.constrain

    def record_execute(
        executor: MarketExecutor,
        book: object,
        target: np.ndarray,
        *,
        start_index: int,
        bars: int,
    ) -> object:
        execution_calls.append(start_index)
        return original_execute(
            executor,
            book,
            target,
            start_index=start_index,
            bars=bars,  # type: ignore[arg-type]
        )

    def record_constrain(
        controller: PreTradeRisk,
        target: np.ndarray,
        *,
        current: np.ndarray,
        drawdown: float,
        emergency_flatten_mask: np.ndarray | None = None,
        reduce_only_mask: np.ndarray | None = None,
    ) -> object:
        risk_calls.append(tuple(float(value) for value in target))
        return original_constrain(
            controller,
            target,
            current=current,
            drawdown=drawdown,
            emergency_flatten_mask=emergency_flatten_mask,
            reduce_only_mask=reduce_only_mask,
        )

    monkeypatch.setattr(MarketExecutor, "execute_interval", record_execute)
    monkeypatch.setattr(PreTradeRisk, "constrain", record_constrain)

    result = replay_module.run_shared_cash_replay(
        dataset,
        strategies,
        start_index=0,
        stop_index=4,
        gross_budget=1.0,
        initial_capital=1_000.0,
        execution_cost=ExecutionCostConfig.zero(),
    )

    assert execution_calls == [0, 1, 2, 3]
    assert len(risk_calls) == 4
    assert risk_calls[0] == pytest.approx((1.0, 1.0))
    assert [
        getattr(strategy.observations[0], "current_weight") for strategy in strategies
    ] == [0.0, 0.0]
    assert len(result.decisions) == 4


def test_turnover_only_projection_keeps_desired_quantities_for_convergence() -> None:
    dataset = _market(np.full((5, 2), [100.0, 200.0]))
    result = replay_module.run_shared_cash_replay(
        dataset,
        (FixedIntent(PositionIntent.LONG), FixedIntent(PositionIntent.LONG)),
        start_index=0,
        stop_index=4,
        gross_budget=0.5,
        initial_capital=1_000.0,
        execution_cost=ExecutionCostConfig.zero(),
        risk=_risk(max_gross=1.0, max_turnover=0.5),
    )

    assert result.decisions[0].proposal_weights == pytest.approx((0.5, 0.5))
    assert result.decisions[0].target_weights == pytest.approx((0.25, 0.25))
    assert result.decisions[0].risk_reasons == ("max_turnover",)
    assert result.decisions[1].proposal_weights == pytest.approx((0.5, 0.5))
    assert result.decisions[1].target_weights == pytest.approx((0.5, 0.5))


def test_hard_risk_projection_rebinds_desired_quantities() -> None:
    dataset = _market(np.full((5, 2), [100.0, 200.0]))
    result = replay_module.run_shared_cash_replay(
        dataset,
        (FixedIntent(PositionIntent.LONG), FixedIntent(PositionIntent.LONG)),
        start_index=0,
        stop_index=4,
        gross_budget=0.5,
        initial_capital=1_000.0,
        execution_cost=ExecutionCostConfig.zero(),
        risk=_risk(max_gross=0.5, max_turnover=None),
    )

    assert result.decisions[0].proposal_weights == pytest.approx((0.5, 0.5))
    assert result.decisions[0].target_weights == pytest.approx((0.25, 0.25))
    assert "max_gross" in result.decisions[0].risk_reasons
    assert result.decisions[1].proposal_weights == pytest.approx((0.25, 0.25))
    assert result.decisions[1].target_weights == pytest.approx((0.25, 0.25))


def test_shared_cash_fixed_drawdown_does_not_compound_risk_scale() -> None:
    dataset = _market(np.asarray([[100.0], [70.0], [70.0], [70.0], [70.0]]))
    risk = PreTradeRisk(
        PreTradeRiskConfig(
            max_gross=1.0,
            max_abs_weight=1.0,
            max_turnover=None,
            drawdown_start=0.10,
            drawdown_stop=0.20,
        )
    )

    result = replay_module.run_shared_cash_replay(
        dataset,
        (FixedIntent(PositionIntent.LONG),),
        start_index=0,
        stop_index=4,
        gross_budget=0.5,
        initial_capital=1_000.0,
        execution_cost=ExecutionCostConfig.zero(),
        risk=risk,
    )

    assert result.book.max_drawdown == pytest.approx(0.15)
    assert result.decisions[0].target_weights == pytest.approx((0.5,))
    assert result.decisions[1].target_weights == pytest.approx(
        (0.20588235294117646,)
    )
    assert result.decisions[2].target_weights == pytest.approx(
        result.decisions[1].target_weights
    )


def test_symbol_permutation_preserves_shared_portfolio_economics() -> None:
    original = _two_symbol_market()
    swapped = _market(
        np.asarray(original.close)[:, [1, 0]],
        symbols=("ETHUSDT", "BTCUSDT"),
    )
    first = replay_module.run_shared_cash_replay(
        original,
        (FixedIntent(PositionIntent.LONG), FixedIntent(PositionIntent.SHORT)),
        start_index=0,
        stop_index=5,
        gross_budget=0.6,
        initial_capital=1_000.0,
        execution_cost=ExecutionCostConfig.zero(),
    )
    second = replay_module.run_shared_cash_replay(
        swapped,
        (FixedIntent(PositionIntent.SHORT), FixedIntent(PositionIntent.LONG)),
        start_index=0,
        stop_index=5,
        gross_budget=0.6,
        initial_capital=1_000.0,
        execution_cost=ExecutionCostConfig.zero(),
    )

    assert second.returns.values == pytest.approx(first.returns.values)
    assert second.book.portfolio_value == pytest.approx(first.book.portfolio_value)
    assert second.book.turnover_total == pytest.approx(first.book.turnover_total)
    assert second.book.total_cost == pytest.approx(first.book.total_cost)
    np.testing.assert_allclose(second.book.quantities[[1, 0]], first.book.quantities)
    np.testing.assert_allclose(second.book.weights[[1, 0]], first.book.weights)


def test_strategy_roster_fails_closed() -> None:
    dataset = _two_symbol_market()
    shared = FixedIntent(PositionIntent.LONG)
    common = dict(
        start_index=0,
        stop_index=5,
        gross_budget=0.5,
        execution_cost=ExecutionCostConfig.zero(),
    )
    with pytest.raises(ValueError, match="one strategy per dataset symbol"):
        replay_module.run_shared_cash_replay(dataset, (shared,), **common)
    with pytest.raises(ValueError, match="distinct strategy instance"):
        replay_module.run_shared_cash_replay(dataset, (shared, shared), **common)


def test_portfolio_termination_stops_all_symbols() -> None:
    dataset = _market(
        np.asarray(
            [
                [100.0, 100.0],
                [100.0, 100.0],
                [300.0, 100.0],
                [300.0, 100.0],
                [300.0, 100.0],
                [300.0, 100.0],
            ]
        )
    )
    result = replay_module.run_shared_cash_replay(
        dataset,
        (FixedIntent(PositionIntent.SHORT), FixedIntent(PositionIntent.FLAT)),
        start_index=0,
        stop_index=5,
        gross_budget=1.0,
        initial_capital=1_000.0,
        execution_cost=ExecutionCostConfig.zero(),
    )

    assert result.book.termination_reason is not None
    assert len(result.returns.values) < 5
    assert len(result.decisions) == len(result.returns.values)
