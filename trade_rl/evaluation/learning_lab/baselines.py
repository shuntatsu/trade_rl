"""Anchor base stack and the fixed-stride and CASH baselines of the learning lab.

The base stack is the ported agy anchor behind the cash-account scale
(``AnchorCashAccountPolicy``), unchanged, with one addition: the bar on which the anchor may
rebalance is chosen by the overlay (a :class:`RebalanceGate`) instead of a fixed modulus.
With the gate open exactly every ``S`` bars the stack is call-for-call the fixed-stride
baseline of stride ``S``, which is how the overlay's ``f = 1`` rows are proven equal to it.

Baselines run through ``run_shared_portfolio_replay`` (the engine's replay), so they and the
overlay agent are priced by the same oracle.
"""

from __future__ import annotations

from typing import Final

import numpy as np

from trade_rl.data.market import MarketDataset
from trade_rl.evaluation.agy_anchor_oos.cash import AnchorCashAccountPolicy
from trade_rl.evaluation.learning_lab.metrics import RolloutResult
from trade_rl.evaluation.learning_lab.overlay_env import (
    BaseFactory,
    BaseStack,
    RebalanceGate,
)
from trade_rl.evaluation.replay import run_shared_portfolio_replay
from trade_rl.risk.pretrade import PreTradeRisk, PreTradeRiskConfig
from trade_rl.simulation.execution import ExecutionCostConfig
from trade_rl.strategies.agy_anchor.policy import AgyAnchorPolicy
from trade_rl.strategies.agy_anchor.signals import AnchorSignals
from trade_rl.strategies.agy_anchor.targets import DEADBAND
from trade_rl.strategies.portfolio import (
    PortfolioDecisionSnapshot,
    PortfolioPolicy,
    PortfolioTargetProposal,
)

FIXED_STRIDES: Final[tuple[int, ...]] = (6, 18, 42)
CASH_METHOD: Final = "BASE-CASH"
_NEVER: Final = 1 << 62


def stride_method_id(stride: int) -> str:
    return f"BASE-STRIDE-{stride}"


class GateableAnchorPolicy(AgyAnchorPolicy):
    """``AgyAnchorPolicy`` whose rebalance bar is ``gate.rebalance_now``.

    The very first call always rebalances (the anchor has no intended book before it). Every
    other rule (lag, deadband, validity holds, risk-scale handling) is the port's own.
    """

    def __init__(
        self,
        gate: RebalanceGate,
        *,
        signals: AnchorSignals,
        start_index: int,
        signal_lag_bars: int = 1,
        deadband: float = DEADBAND,
    ) -> None:
        super().__init__(
            signals=signals,
            start_index=start_index,
            signal_lag_bars=signal_lag_bars,
            rebalance_every_bars=1,
            deadband=deadband,
        )
        self._gate = gate

    def _decide(self, snapshot: PortfolioDecisionSnapshot) -> PortfolioTargetProposal:
        self._every = (
            1 if (self._gate.rebalance_now or self._intended is None) else _NEVER
        )
        return super()._decide(snapshot)


def anchor_stack_factory(
    signals: AnchorSignals,
    *,
    start_index: int,
    signal_lag_bars: int = 1,
    deadband: float = DEADBAND,
) -> BaseFactory:
    """Factory of fresh gateable anchor stacks (one per episode)."""

    def factory() -> BaseStack:
        gate = RebalanceGate()
        inner = GateableAnchorPolicy(
            gate,
            signals=signals,
            start_index=start_index,
            signal_lag_bars=signal_lag_bars,
            deadband=deadband,
        )
        policy = AnchorCashAccountPolicy(inner, start_index=start_index)
        return BaseStack(policy=policy, gate=gate, shared=(signals,))

    return factory


def fixed_stride_policy(
    signals: AnchorSignals,
    *,
    start_index: int,
    stride: int,
    signal_lag_bars: int = 1,
    deadband: float = DEADBAND,
) -> PortfolioPolicy:
    """The anchor behind the cash-account scale with a plain fixed rebalance stride."""
    return AnchorCashAccountPolicy(
        AgyAnchorPolicy(
            signals=signals,
            start_index=start_index,
            signal_lag_bars=signal_lag_bars,
            rebalance_every_bars=stride,
            deadband=deadband,
        ),
        start_index=start_index,
    )


class CashPolicy:
    """Stay flat: propose zero weights on every bar."""

    def decide_portfolio(
        self, snapshot: PortfolioDecisionSnapshot
    ) -> PortfolioTargetProposal:
        return PortfolioTargetProposal(target_weights=(0.0,) * len(snapshot.symbols))


def run_policy_replay(
    method: str,
    dataset: MarketDataset,
    policy: PortfolioPolicy,
    *,
    start_index: int,
    stop_index: int,
    execution_cost: ExecutionCostConfig,
    risk_config: PreTradeRiskConfig,
    initial_capital: float,
) -> RolloutResult:
    """Replay ``policy`` through the shared-portfolio engine and keep its net returns."""
    result = run_shared_portfolio_replay(
        dataset,
        policy,
        start_index=start_index,
        stop_index=stop_index,
        initial_capital=initial_capital,
        execution_cost=execution_cost,
        risk=PreTradeRisk(risk_config),
        settle_terminal_position=True,
    )
    return RolloutResult(
        method=method,
        bar_returns=tuple(float(v) for v in np.asarray(result.returns.values)),
        final_equity=float(result.book.portfolio_value),
        total_cost=float(result.book.total_cost),
        turnover_total=float(result.book.turnover_total),
        initial_capital=initial_capital,
    )


def run_baseline(
    method: str,
    dataset: MarketDataset,
    signals: AnchorSignals,
    *,
    start_index: int,
    stop_index: int,
    execution_cost: ExecutionCostConfig,
    risk_config: PreTradeRiskConfig,
    initial_capital: float,
    signal_lag_bars: int = 1,
) -> RolloutResult:
    """Run ``BASE-STRIDE-<S>`` (``S`` in :data:`FIXED_STRIDES`) or ``BASE-CASH``."""
    if method == CASH_METHOD:
        policy: PortfolioPolicy = CashPolicy()
    else:
        strides = {stride_method_id(s): s for s in FIXED_STRIDES}
        if method not in strides:
            raise ValueError(f"unknown baseline: {method!r}")
        policy = fixed_stride_policy(
            signals,
            start_index=start_index,
            stride=strides[method],
            signal_lag_bars=signal_lag_bars,
        )
    return run_policy_replay(
        method,
        dataset,
        policy,
        start_index=start_index,
        stop_index=stop_index,
        execution_cost=execution_cost,
        risk_config=risk_config,
        initial_capital=initial_capital,
    )


def baseline_methods() -> tuple[str, ...]:
    return (*(stride_method_id(s) for s in FIXED_STRIDES), CASH_METHOD)


__all__ = [
    "CASH_METHOD",
    "FIXED_STRIDES",
    "CashPolicy",
    "GateableAnchorPolicy",
    "anchor_stack_factory",
    "baseline_methods",
    "fixed_stride_policy",
    "run_baseline",
    "run_policy_replay",
    "stride_method_id",
]
