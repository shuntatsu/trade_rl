from __future__ import annotations

from dataclasses import FrozenInstanceError, dataclass, field, replace

import numpy as np
import pytest

from trade_rl.data.market import MarketDataset
from trade_rl.evaluation.replay import (
    SharedCashPortfolioReplayResult,
    SharedCashReplayDecision,
    SharedCashReplayResult,
    run_shared_cash_replay,
    run_shared_portfolio_replay,
)
from trade_rl.risk import PreTradeRisk, PreTradeRiskConfig
from trade_rl.simulation import (
    BookState,
    ExecutionCostConfig,
    ExecutionResult,
    MarketExecutor,
)
from trade_rl.strategies.portfolio import (
    PortfolioDecisionSnapshot,
    PortfolioTargetProposal,
)
from trade_rl.strategies.position_intent import PositionIntent


def _market(
    *,
    volume: np.ndarray | None = None,
    close: np.ndarray | None = None,
    features: np.ndarray | None = None,
    funding: np.ndarray | None = None,
) -> MarketDataset:
    resolved_close = np.asarray(
        [[100.0, 200.0], [100.0, 200.0], [100.0, 200.0], [100.0, 200.0]]
        if close is None
        else close,
        dtype=np.float64,
    )
    close = resolved_close
    n_bars, n_symbols = close.shape
    open_price = np.vstack((close[0], close[:-1]))
    resolved_volume = (
        np.full((n_bars, n_symbols), 1_000_000.0)
        if volume is None
        else np.asarray(volume, dtype=np.float64)
    )
    resolved_features = (
        np.arange(n_bars * n_symbols, dtype=np.float32).reshape(n_bars, n_symbols, 1)
        if features is None
        else np.asarray(features, dtype=np.float32)
    )
    resolved_funding = (
        np.zeros((n_bars, n_symbols), dtype=np.float64)
        if funding is None
        else np.asarray(funding, dtype=np.float64)
    )
    return MarketDataset(
        dataset_id="d" * 64,
        symbols=("BTCUSDT", "ETHUSDT"),
        timestamps=np.datetime64("2026-01-01T00:00:00", "ns")
        + np.arange(n_bars) * np.timedelta64(1, "h"),
        features=resolved_features,
        global_features=np.arange(n_bars, dtype=np.float32).reshape(n_bars, 1),
        open=open_price,
        high=np.maximum(open_price, close),
        low=np.minimum(open_price, close),
        close=close,
        volume=resolved_volume,
        funding_rate=resolved_funding,
        tradable=np.ones((n_bars, n_symbols), dtype=np.bool_),
        feature_available=np.ones((n_bars, n_symbols, 1), dtype=np.bool_),
        feature_names=("signal",),
        global_feature_names=("regime",),
        periods_per_year=8_760,
    )


@dataclass
class _HoldPartialFill:
    snapshots: list[PortfolioDecisionSnapshot] = field(default_factory=list)

    def decide_portfolio(
        self, snapshot: PortfolioDecisionSnapshot
    ) -> PortfolioTargetProposal:
        self.snapshots.append(snapshot)
        if snapshot.index == 0:
            return PortfolioTargetProposal(target_weights=(0.5, 0.0))
        return PortfolioTargetProposal(
            target_weights=(snapshot.account.weights[0], 0.0),
            hold_current_quantity_symbols=(0,),
        )


def test_portfolio_policy_gets_detached_shared_state_and_exact_hold_cancels_residual(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    volume = np.full((4, 2), 1_000_000.0)
    volume[1, 0] = 1.0
    dataset = _market(volume=volume)
    policy = _HoldPartialFill()
    risk = PreTradeRisk(
        PreTradeRiskConfig(
            max_gross=0.5,
            max_abs_weight=0.5,
            max_turnover=None,
            drawdown_start=1.0,
            drawdown_stop=1.0,
        )
    )
    calls: list[tuple[MarketExecutor, BookState, int, BookState]] = []
    original_execute_interval = MarketExecutor.execute_interval

    def record_interval(
        executor: MarketExecutor,
        book: BookState,
        target: np.ndarray,
        *,
        start_index: int,
        bars: int,
        hold_current_quantity_symbols: tuple[int, ...] = (),
    ) -> ExecutionResult:
        result = original_execute_interval(
            executor,
            book,
            target,
            start_index=start_index,
            bars=bars,
            hold_current_quantity_symbols=hold_current_quantity_symbols,
        )
        calls.append((executor, book, start_index, result.book))
        return result

    monkeypatch.setattr(MarketExecutor, "execute_interval", record_interval)
    result = run_shared_portfolio_replay(
        dataset,
        policy,
        start_index=0,
        stop_index=2,
        initial_capital=1_000.0,
        execution_cost=replace(
            ExecutionCostConfig.zero(),
            lot_size=0.1,
        ),
        risk=risk,
        capture_ledger_evidence=True,
    )

    assert isinstance(result, SharedCashPortfolioReplayResult)
    assert [snapshot.index for snapshot in policy.snapshots] == [0, 1]
    first, second = policy.snapshots
    assert first.timestamp_ns < second.timestamp_ns
    assert first.symbols == ("BTCUSDT", "ETHUSDT")
    assert first.feature_names == ("signal",)
    assert first.features == ((0.0,), (1.0,))
    assert first.feature_available == ((True,), (True,))
    assert first.global_features == (0.0,)
    assert first.account.exact_quantities == ("0", "0")
    assert first.account.cash == 1_000.0
    assert first.risk.max_abs_weight == 0.5
    assert first.active_orders == ()
    assert isinstance(first.features, tuple)
    assert isinstance(first.features[0], tuple)
    assert first.features is not dataset.features
    assert first.account is not calls[0][1]
    assert not hasattr(first, "dataset")
    assert not hasattr(first, "book")
    assert second.account.exact_quantities == ("1", "0")
    assert len(second.active_orders) == 1
    assert second.active_orders[0].symbol_index == 0
    assert second.active_orders[0].remaining_quantity == pytest.approx(4.0)
    assert second.active_orders[0].last_processed_index == 1
    assert not hasattr(second.active_orders[0], "order_id")

    assert len(calls) == 2
    assert calls[0][0] is calls[1][0]
    assert calls[1][1] is calls[0][3]
    assert [call[2] for call in calls] == [0, 1]
    assert result.decisions[1].effective_hold_symbols == (0,)
    assert tuple(str(value) for value in result.book.exact_quantities) == ("1", "0")
    assert result.ledger_evidence is not None
    assert result.ledger_evidence.intervals.active_order_remainders == ()
    payload = result.ledger_evidence.to_mapping()
    assert payload["schema_version"] == "shared_cash_portfolio_replay_ledger_v1"
    assert "portfolio_decisions" in payload
    assert "intents" not in payload["portfolio_decisions"][0]
    assert first.features == ((0.0,), (1.0,))
    with pytest.raises(FrozenInstanceError):
        first.index = 7  # type: ignore[misc]


@dataclass
class _HoldAfterEntry:
    snapshots: list[PortfolioDecisionSnapshot] = field(default_factory=list)

    def decide_portfolio(
        self, snapshot: PortfolioDecisionSnapshot
    ) -> PortfolioTargetProposal:
        self.snapshots.append(snapshot)
        if snapshot.index == 0:
            return PortfolioTargetProposal((0.5, 0.0))
        return PortfolioTargetProposal(
            (snapshot.account.weights[0], 0.0),
            hold_current_quantity_symbols=(0,),
        )


def test_risk_constraint_cancels_exact_hold_when_weight_drift_exceeds_limit() -> None:
    dataset = _market(
        close=np.asarray(
            [[100.0, 200.0], [200.0, 200.0], [200.0, 200.0], [200.0, 200.0]]
        )
    )
    policy = _HoldAfterEntry()
    risk = PreTradeRisk(
        PreTradeRiskConfig(
            max_gross=0.5,
            max_abs_weight=0.5,
            max_turnover=None,
            drawdown_start=1.0,
            drawdown_stop=1.0,
        )
    )
    result = run_shared_portfolio_replay(
        dataset,
        policy,
        start_index=0,
        stop_index=2,
        initial_capital=1_000.0,
        execution_cost=ExecutionCostConfig.zero(),
        risk=risk,
    )

    assert policy.snapshots[1].account.weights[0] > 0.5
    decision = result.decisions[1]
    assert decision.requested_hold_symbols == (0,)
    assert decision.effective_hold_symbols == ()
    assert decision.target_weights[0] == pytest.approx(0.5)
    assert result.book.quantities[0] < 5.0


@dataclass
class _EnterOnce:
    indexes: list[int] = field(default_factory=list)
    snapshots: list[PortfolioDecisionSnapshot] = field(default_factory=list)

    def decide_portfolio(
        self, snapshot: PortfolioDecisionSnapshot
    ) -> PortfolioTargetProposal:
        self.indexes.append(snapshot.index)
        self.snapshots.append(snapshot)
        return PortfolioTargetProposal((0.5, 0.0))


def test_latency_terminal_settlement_limits_actor_calls_and_flattens() -> None:
    dataset = _market()
    policy = _EnterOnce()
    cost = replace(ExecutionCostConfig.zero(), order_latency_bars=1)
    result = run_shared_portfolio_replay(
        dataset,
        policy,
        start_index=0,
        stop_index=3,
        initial_capital=1_000.0,
        execution_cost=cost,
        risk=PreTradeRisk(
            PreTradeRiskConfig(
                max_gross=0.5,
                max_abs_weight=0.5,
                max_turnover=None,
                drawdown_start=1.0,
                drawdown_stop=1.0,
            )
        ),
        settle_terminal_position=True,
        capture_ledger_evidence=True,
    )

    assert policy.indexes == [0]
    assert tuple(str(value) for value in result.book.exact_quantities) == ("0", "0")
    assert result.ledger_evidence is not None
    payload = result.ledger_evidence.to_mapping()
    assert payload["active_order_remainders"] == ()
    assert len(payload["intervals"]) == 3
    assert payload["schema_version"] == "shared_cash_portfolio_replay_ledger_v1"


def test_invalid_full_vector_fails_before_executor_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _InvalidPolicy:
        def decide_portfolio(
            self, snapshot: PortfolioDecisionSnapshot
        ) -> PortfolioTargetProposal:
            del snapshot
            return PortfolioTargetProposal((0.25,))

    calls: list[int] = []
    original_execute_interval = MarketExecutor.execute_interval

    def record_interval(
        executor: MarketExecutor,
        book: BookState,
        target: np.ndarray,
        *,
        start_index: int,
        bars: int,
        hold_current_quantity_symbols: tuple[int, ...] = (),
    ) -> ExecutionResult:
        calls.append(start_index)
        return original_execute_interval(
            executor,
            book,
            target,
            start_index=start_index,
            bars=bars,
            hold_current_quantity_symbols=hold_current_quantity_symbols,
        )

    monkeypatch.setattr(MarketExecutor, "execute_interval", record_interval)
    with pytest.raises(ValueError, match="must match dataset symbols"):
        run_shared_portfolio_replay(
            _market(),
            _InvalidPolicy(),
            start_index=0,
            stop_index=2,
        )
    assert calls == []


def test_future_tail_changes_do_not_change_current_snapshot_or_interval() -> None:
    dataset = _market()
    changed_features = np.array(dataset.features, copy=True)
    changed_features[2:] += 10_000.0
    changed_close = np.array(dataset.close, copy=True)
    changed_close[2:] *= 3.0
    changed_funding = np.array(dataset.funding_rate, copy=True)
    changed_funding[2:] = 0.25
    changed_tail = _market(
        close=changed_close,
        features=changed_features,
        funding=changed_funding,
    )
    first_policy = _EnterOnce()
    second_policy = _EnterOnce()
    risk = PreTradeRisk(
        PreTradeRiskConfig(
            max_gross=0.5,
            max_abs_weight=0.5,
            max_turnover=None,
            drawdown_start=1.0,
            drawdown_stop=1.0,
        )
    )
    first = run_shared_portfolio_replay(
        dataset,
        first_policy,
        start_index=0,
        stop_index=1,
        initial_capital=1_000.0,
        execution_cost=ExecutionCostConfig.zero(),
        risk=risk,
        capture_ledger_evidence=True,
    )
    second = run_shared_portfolio_replay(
        changed_tail,
        second_policy,
        start_index=0,
        stop_index=1,
        initial_capital=1_000.0,
        execution_cost=ExecutionCostConfig.zero(),
        risk=risk,
        capture_ledger_evidence=True,
    )

    assert first_policy.indexes == second_policy.indexes == [0]
    assert first_policy.snapshots == second_policy.snapshots
    assert first.decisions == second.decisions
    assert first.ledger_evidence is not None
    assert second.ledger_evidence is not None
    first_payload = first.ledger_evidence.to_mapping()
    second_payload = second.ledger_evidence.to_mapping()
    assert first_payload["intervals"] == second_payload["intervals"]


@dataclass(frozen=True)
class _FlatIntent:
    def decide(self, observation: object) -> PositionIntent:
        del observation
        return PositionIntent.FLAT


def test_legacy_replay_result_and_terminal_ledger_schema_remain_intent_shaped() -> None:
    result = run_shared_cash_replay(
        _market(),
        (_FlatIntent(), _FlatIntent()),
        start_index=0,
        stop_index=3,
        gross_budget=0.5,
        execution_cost=ExecutionCostConfig.zero(),
        settle_terminal_position=True,
        capture_ledger_evidence=True,
    )

    assert isinstance(result, SharedCashReplayResult)
    assert len(result.decisions) == 2
    assert all(
        isinstance(value, SharedCashReplayDecision) for value in result.decisions
    )
    assert result.ledger_evidence is not None
    payload = result.ledger_evidence.to_mapping()
    assert payload["schema_version"] == "shared_cash_replay_ledger_v2"
    assert len(payload["decisions"]) == 2
    assert "portfolio_decisions" not in payload
    assert "intents" in payload["decisions"][0]
