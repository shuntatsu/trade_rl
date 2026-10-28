"""Test-only two-symbol synthetic action-value probe for issue #1025.

This is a one-decision native-account learnability smoke, not completed TD1--TD4,
a market model, an empirical economic result, or an authorization to trade.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field, replace

import numpy as np
import pytest

from trade_rl.data.market import MarketDataset
from trade_rl.evaluation.replay import run_shared_portfolio_replay
from trade_rl.risk.pretrade import PreTradeRisk, PreTradeRiskConfig
from trade_rl.simulation.execution import ExecutionCostConfig
from trade_rl.strategies.portfolio import (
    PortfolioDecisionSnapshot,
    PortfolioTargetProposal,
)

CAPITAL = 1_000.0
EXPOSURE = 0.4
COST = replace(ExecutionCostConfig.zero(), fee_rate=0.001, spread_rate=0.0002)
RISK = PreTradeRisk(
    PreTradeRiskConfig(
        max_gross=0.5,
        max_abs_weight=0.5,
        max_turnover=None,
        drawdown_start=0.20,
        drawdown_stop=0.20,
    )
)


def _toy_dataset(*, signal: float, market_move: float, tag: str) -> MarketDataset:
    """Seven rows, two assets: A moves only after an eligible trade, B is flat.

    The observable signal deliberately predicts A in the positive-signal fixture.
    The synthetic ID is a test label, not a trusted provider/content attestation.
    """
    marks = np.array(
        [
            100.0,
            100.0,
            100.0,
            100.0 * (1 + market_move / 3),
            100.0 * (1 + 2 * market_move / 3),
            100.0 * (1 + market_move),
            100.0 * (1 + market_move),
        ],
        dtype=np.float64,
    )
    close = np.column_stack((marks, np.full(marks.shape, 100.0)))
    opening = np.vstack((close[0], close[:-1]))
    features = np.zeros((len(close), 2, 1), dtype=np.float32)
    features[:, 0, 0] = signal
    synthetic_id = hashlib.sha256(
        f"issue1025/artificial/v1/{tag}/{signal}/{market_move}".encode()
    ).hexdigest()
    return MarketDataset(
        dataset_id=synthetic_id,
        symbols=("AAA", "BBB"),
        timestamps=np.datetime64("2026-01-01T00:00:00", "ns")
        + np.arange(len(close)) * np.timedelta64(1, "h"),
        features=features,
        global_features=np.zeros((len(close), 1), dtype=np.float32),
        open=opening,
        high=np.maximum(opening, close),
        low=np.minimum(opening, close),
        close=close,
        volume=np.full_like(close, 1_000_000.0),
        funding_rate=np.zeros_like(close),
        tradable=np.ones_like(close, dtype=np.bool_),
        feature_available=np.ones((len(close), 2, 1), dtype=np.bool_),
        feature_names=("observed_signal",),
        global_feature_names=("constant",),
        periods_per_year=8_760,
    )


@dataclass
class _OneIntervention:
    """Change A only at decision 1, then preserve each branch's own quantity."""

    action: int
    prefix: list[tuple[object, ...]] = field(default_factory=list)

    def decide_portfolio(
        self, snapshot: PortfolioDecisionSnapshot
    ) -> PortfolioTargetProposal:
        if snapshot.index in (0, 1):
            self.prefix.append(
                (
                    snapshot.index,
                    snapshot.account.exact_quantities,
                    snapshot.account.cash,
                    snapshot.active_orders,
                )
            )
        if snapshot.index != 1 or self.action == 0:
            return PortfolioTargetProposal(
                target_weights=snapshot.account.weights,
                hold_current_quantity_symbols=(0, 1),
            )
        return PortfolioTargetProposal(
            target_weights=(EXPOSURE * self.action, snapshot.account.weights[1]),
            hold_current_quantity_symbols=(1,),
        )


def _rollout(
    dataset: MarketDataset, action: int
) -> tuple[float, tuple[tuple[object, ...], ...]]:
    assert action in (-1, 0, 1)
    policy = _OneIntervention(action)
    result = run_shared_portfolio_replay(
        dataset,
        policy,
        start_index=0,
        stop_index=6,
        initial_capital=CAPITAL,
        execution_cost=COST,
        risk=RISK,
        settle_terminal_position=True,
        capture_ledger_evidence=True,
    )
    assert len(policy.prefix) == 2
    assert result.ledger_evidence is not None
    assert len(result.ledger_evidence.intervals) == 6
    assert tuple(str(q) for q in result.book.exact_quantities) == ("0", "0")
    return float(result.book.portfolio_value), tuple(policy.prefix)


def _paired_delta(dataset: MarketDataset, action: int) -> float:
    alternative, alternative_prefix = _rollout(dataset, action)
    hold, hold_prefix = _rollout(dataset, 0)
    assert alternative_prefix == hold_prefix
    return (alternative - hold) / CAPITAL


def _design_row(signal: float, action: int) -> np.ndarray:
    """State-action interaction keeps the action ranking state-dependent."""
    return np.array((abs(action), signal * action), dtype=np.float64)


def _fit_single_fixed_ridge() -> np.ndarray:
    rows: list[np.ndarray] = []
    labels: list[float] = []
    for magnitude in (0.018, 0.020, 0.022):
        for signal in (-1.0, 0.0, 1.0):
            dataset = _toy_dataset(
                signal=signal,
                market_move=signal * magnitude,
                tag=f"train-{magnitude}-{signal}",
            )
            for action in (-1, 1):
                rows.append(_design_row(signal, action))
                labels.append(_paired_delta(dataset, action))
    x = np.vstack(rows)
    y = np.asarray(labels, dtype=np.float64)
    assert x.shape == (18, 2)
    # Exactly one fixed fit; no HPO, checkpoint selection or held-out refitting.
    return np.linalg.solve(x.T @ x + 1e-6 * np.eye(2), x.T @ y)


def _choose(model: np.ndarray, observed_signal: float) -> int:
    values = {
        action: float(_design_row(observed_signal, action) @ model)
        for action in (-1, 0, 1)
    }
    # Prefer HOLD for ties, and whenever no alternative has positive value.
    return max((0, -1, 1), key=lambda action: values[action])


def test_native_paired_labels_include_cost_and_preserve_prefix() -> None:
    market = _toy_dataset(signal=0.0, market_move=0.0, tag="fee-only")
    hold, original_prefix = _rollout(market, 0)
    long, long_prefix = _rollout(market, 1)
    short, short_prefix = _rollout(market, -1)
    expected_cash = CAPITAL - 2 * CAPITAL * EXPOSURE * (
        COST.fee_rate + COST.spread_rate
    )
    assert hold == pytest.approx(CAPITAL)
    assert long == pytest.approx(expected_cash, abs=1e-8)
    assert short == pytest.approx(expected_cash, abs=1e-8)
    assert original_prefix == long_prefix == short_prefix
    assert _paired_delta(market, 0) == pytest.approx(0.0, abs=1e-12)
    assert _paired_delta(market, 1) == pytest.approx((long - hold) / CAPITAL)
    assert _rollout(market, 1)[0] == pytest.approx(long, abs=1e-10)


def test_fixed_small_model_learns_signed_native_value_on_unseen_toy_paths() -> None:
    model = _fit_single_fixed_ridge()
    assert np.isfinite(model).all()
    assert model[0] < 0.0  # unneeded trade loses fee and spread
    assert model[1] > 0.0  # causal toy signal has positive future correlation

    learner_pnl: list[float] = []
    cash_pnl: list[float] = []
    always_long_pnl: list[float] = []
    for i, (signal, market_move) in enumerate(
        ((0.0, 0.0), (0.75, 0.017), (-0.75, -0.017))
    ):
        market = _toy_dataset(
            signal=signal, market_move=market_move, tag=f"holdout-{i}"
        )
        action = _choose(model, float(market.features[1, 0, 0]))
        expected = 0 if signal == 0 else (1 if signal > 0 else -1)
        assert action == expected
        learner, _ = _rollout(market, action)
        cash, _ = _rollout(market, 0)
        always_long, _ = _rollout(market, 1)
        learner_pnl.append(learner - CAPITAL)
        cash_pnl.append(cash - CAPITAL)
        always_long_pnl.append(always_long - CAPITAL)
        if signal == 0:
            assert learner == pytest.approx(cash)
        else:
            assert learner > cash
    assert sum(learner_pnl) > sum(cash_pnl)
    assert sum(learner_pnl) > sum(always_long_pnl)

    # A changed future reverses the realized payoff, not the frozen decision.
    tail = _toy_dataset(signal=0.75, market_move=-0.017, tag="negative-tail")
    assert _choose(model, float(tail.features[1, 0, 0])) == 1
    assert _paired_delta(tail, 1) < 0.0
