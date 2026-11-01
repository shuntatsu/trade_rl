"""Gymnasium environment: a discrete trade-throttle residual over the allocator.

One continuous account runs over one pre-generated synthetic market with daily
decisions. At every decision the **real** ``CostAwareAllocationPolicy`` (read-only
reuse) proposes target weights from the noisy forecast ``mu``; the agent then picks
one multiplier ``m`` from a small fixed set and the executed weights are

    w_exec = w_target                          if m == 1   (allocator alone, exact)
    w_exec = w_cur                             if m == 0   (no trade)
    w_exec = project(w_cur + m (w_target - w_cur))  otherwise

so ``m < 1`` throttles the allocator's trade, ``m = 1`` reproduces it bit-for-bit
and ``m = 1.5`` overshoots it by half. ``project`` clips back into the allocator's
own box (``|w_i| <= w_max``, ``sum |w| <= gross_max``), so the residual can never
lever the account beyond what the allocator itself may hold.

The ledger is an independent value ledger (positions in value units plus cash).
Costs are one-way proportional charges on traded notional: ``fee`` plus half the
quoted ``spread`` per side, charged separately and immediately. The reward is
``log(equity_after_day / equity_before_trade)``, so it includes the cost of the
trade and the day's return. There is no financing, funding or borrow charge, no
market impact and no partial fill in this synthetic study.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Any, cast

import gymnasium as gym
import numpy as np

from trade_rl.evaluation.residual_rl.market import (
    NS_PER_DAY,
    MarketSpec,
    Regime,
    SyntheticMarket,
    generate_market,
)
from trade_rl.strategies.allocation.cost_aware import CostAwareAllocationPolicy
from trade_rl.strategies.portfolio import (
    PortfolioAccountSnapshot,
    PortfolioDecisionSnapshot,
    PortfolioRiskLimits,
)

_PROJECTION_TOLERANCE = 1e-12


@dataclass(frozen=True, slots=True)
class AllocatorSettings:
    """Cost-aware allocator constants. ``kappa`` is ``cost_penalty_multiple``."""

    gamma: float = 30.0
    w_max: float = 0.5
    gross_max: float = 1.0
    ewma_halflife_days: float = 30.0
    shrinkage: float = 0.2
    min_trade_fraction: float = 0.005
    min_history_periods: int = 30
    kappa: float = 1.0


@dataclass(frozen=True, slots=True)
class CostSettings:
    """Explicit one-way costs as fractions of traded notional."""

    fee_rate: float = 0.0005  # 0.05% per side
    spread_rate: float = 0.0002  # 0.02% quoted spread, half crossed per side

    @property
    def fee_per_unit(self) -> float:
        return self.fee_rate

    @property
    def spread_per_unit(self) -> float:
        return 0.5 * self.spread_rate

    @property
    def one_way(self) -> float:
        return self.fee_per_unit + self.spread_per_unit


@dataclass(frozen=True, slots=True)
class ObservationScales:
    """Fixed, data-independent scales that bring each feature near unit range."""

    mu: float = 0.0012
    cost_window: int = 10
    cost: float = 0.002
    drawdown: float = 0.1
    return_window: int = 10
    recent_return: float = 0.05
    clip: float = 10.0


@dataclass(frozen=True, slots=True)
class EnvConfig:
    market: MarketSpec
    regime: Regime
    allocator: AllocatorSettings
    costs: CostSettings
    scales: ObservationScales
    multipliers: tuple[float, ...] = (0.0, 0.5, 1.0, 1.5)
    reward_scale: float = 1.0

    def __post_init__(self) -> None:
        if not self.multipliers or any(
            not math.isfinite(m) or m < 0.0 for m in self.multipliers
        ):
            raise ValueError("multipliers must be finite and non-negative")
        if self.allocator.w_max * self.market.n_assets < 1e-9:
            raise ValueError("w_max must be positive")
        if not math.isfinite(self.reward_scale) or self.reward_scale <= 0.0:
            raise ValueError("reward_scale must be finite and positive")

    @property
    def observation_size(self) -> int:
        return 2 * self.market.n_assets + 4


class _MarketForecast:
    """``ExpectedReturnSource`` that reads the forecast known at the decision."""

    def __init__(self, market: SyntheticMarket) -> None:
        self._market = market

    def expected_simple_returns(
        self, snapshot: PortfolioDecisionSnapshot
    ) -> tuple[float, ...]:
        return tuple(float(v) for v in self._market.mu[snapshot.index])


def build_allocator(
    market: SyntheticMarket,
    allocator: AllocatorSettings,
    costs: CostSettings,
    *,
    kappa: float | None = None,
) -> CostAwareAllocationPolicy:
    """The real allocator for one episode, seeded with pre-episode marks only."""

    warm = market.spec.warmup_days
    n = market.spec.n_assets
    return CostAwareAllocationPolicy(
        symbols=market.spec.symbols,
        forecast_source=_MarketForecast(market),
        cost_rates=(costs.one_way,) * n,
        gamma=allocator.gamma,
        ewma_halflife_days=allocator.ewma_halflife_days,
        shrinkage=allocator.shrinkage,
        w_max=allocator.w_max,
        gross_max=allocator.gross_max,
        long_only=False,
        min_trade_fraction=allocator.min_trade_fraction,
        decision_interval_hours=24,
        decision_anchor_hour_utc=0,
        min_history_periods=allocator.min_history_periods,
        warmup_timestamps_ns=tuple(day * NS_PER_DAY for day in range(warm)),
        warmup_mark_prices=tuple(
            tuple(float(v) for v in market.prices[day]) for day in range(warm)
        ),
        cost_penalty_multiple=allocator.kappa if kappa is None else kappa,
    )


def apply_residual(
    current: np.ndarray,
    target: np.ndarray,
    multiplier: float,
    *,
    w_max: float,
    gross_max: float,
) -> np.ndarray:
    """Executed weights for residual ``multiplier``; exact at ``0`` and ``1``."""

    if multiplier == 1.0:
        return target.copy()
    if multiplier == 0.0:
        return current.copy()
    executed = current + multiplier * (target - current)
    executed = np.clip(executed, -w_max, w_max)
    gross = float(np.abs(executed).sum())
    if gross > gross_max + _PROJECTION_TOLERANCE:
        executed = executed * (gross_max / gross)
    return executed


class ResidualAllocatorEnv(gym.Env[np.ndarray, int]):
    """See the module docstring. ``reset(options={"market_seed": s})`` pins a market."""

    metadata: dict[str, Any] = {"render_modes": []}

    def __init__(self, config: EnvConfig, market_seeds: tuple[int, ...]) -> None:
        super().__init__()
        if not market_seeds or any(type(s) is not int or s < 0 for s in market_seeds):
            raise ValueError("market_seeds must be non-negative integers")
        self.config = config
        self._market_seeds = market_seeds
        n = config.market.n_assets
        self.action_space = cast(
            "gym.spaces.Space[int]", gym.spaces.Discrete(len(config.multipliers))
        )
        self.observation_space = gym.spaces.Box(
            low=-config.scales.clip,
            high=config.scales.clip,
            shape=(config.observation_size,),
            dtype=np.float32,
        )
        self._n = n
        self._market: SyntheticMarket | None = None
        self._policy: CostAwareAllocationPolicy | None = None
        self._day = 0
        self._values = np.zeros(n)
        self._cash = 0.0
        self._peak = 0.0
        self._turnover_total = 0.0
        self._cost_total = 0.0
        self._target = np.zeros(n)
        self._recent_costs: deque[float] = deque(maxlen=config.scales.cost_window)
        self._recent_returns: deque[float] = deque(maxlen=config.scales.return_window)
        self.ledger: list[dict[str, float]] = []
        # Audit trail for the oracles: the allocator proposal and the executed
        # weights (both relative to the pre-trade equity) of every step.
        self.targets: list[np.ndarray] = []
        self.executed: list[np.ndarray] = []

    # -- public accessors -------------------------------------------------
    @property
    def market(self) -> SyntheticMarket:
        if self._market is None:
            raise RuntimeError("reset() has not been called")
        return self._market

    @property
    def equity(self) -> float:
        return float(self._cash + self._values.sum())

    @property
    def initial_equity(self) -> float:
        return 1.0

    # -- gym API ------------------------------------------------------------
    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        options = options or {}
        spec = self.config.market
        injected = options.get("market")
        if injected is not None:
            # Test hook: a pre-built market (for example a causality counterfactual).
            if not isinstance(injected, SyntheticMarket) or injected.spec != spec:
                raise ValueError("injected market must match the env market spec")
            self._market = injected
            market_seed = injected.seed
        else:
            pinned = options.get("market_seed")
            if pinned is None:
                index = int(self.np_random.integers(len(self._market_seeds)))
                market_seed = self._market_seeds[index]
            else:
                market_seed = int(pinned)
            self._market = generate_market(spec, self.config.regime, market_seed)
        self._policy = build_allocator(
            self._market, self.config.allocator, self.config.costs
        )
        self._day = spec.warmup_days
        self._values = np.zeros(self._n)
        self._cash = 1.0
        self._peak = 1.0
        self._turnover_total = 0.0
        self._cost_total = 0.0
        self._recent_costs.clear()
        self._recent_returns.clear()
        self.ledger = []
        self.targets = []
        self.executed = []
        self._propose()
        return self._observation(), {"market_seed": market_seed}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        market = self.market
        if self._policy is None:
            raise RuntimeError("reset() has not been called")
        index = int(action)
        if not 0 <= index < len(self.config.multipliers):
            raise ValueError("action outside the multiplier set")
        multiplier = self.config.multipliers[index]
        day = self._day
        equity_before = self.equity
        w_cur = self._values / equity_before
        allocator = self.config.allocator
        w_exec = apply_residual(
            w_cur,
            self._target,
            multiplier,
            w_max=allocator.w_max,
            gross_max=allocator.gross_max,
        )
        self.targets.append(self._target.copy())
        self.executed.append(w_exec.copy())
        trade_value = (w_exec - w_cur) * equity_before
        traded = float(np.abs(trade_value).sum())
        fee = self.config.costs.fee_per_unit * traded
        spread = self.config.costs.spread_per_unit * traded
        self._values = self._values + trade_value
        self._cash -= float(trade_value.sum()) + fee + spread
        equity_traded = self.equity
        self._values = self._values * (1.0 + market.returns[day])
        equity_after = self.equity
        if not math.isfinite(equity_after) or equity_after <= 0.0:
            raise RuntimeError("account equity became non-positive")
        log_return = math.log(equity_after / equity_before)
        self._peak = max(self._peak, equity_after)
        self._turnover_total += traded / equity_before
        self._cost_total += fee + spread
        self._recent_costs.append((fee + spread) / equity_before)
        self._recent_returns.append(log_return)
        self.ledger.append(
            {
                "day": float(day),
                "multiplier": multiplier,
                "equity_before": equity_before,
                "equity_after_trade": equity_traded,
                "equity_after": equity_after,
                "traded_value": traded,
                "fee": fee,
                "spread_cost": spread,
                "log_return": log_return,
            }
        )
        self._day = day + 1
        terminated = False
        truncated = self._day >= market.spec.total_days
        if not truncated:
            self._propose()
        info = {
            "equity": equity_after,
            "multiplier": multiplier,
            "traded_value": traded,
            "fee": fee,
            "spread_cost": spread,
            "log_return": log_return,
        }
        return (
            self._observation(),
            self.config.reward_scale * log_return,
            terminated,
            truncated,
            info,
        )

    # -- internals ----------------------------------------------------------
    def _snapshot(self) -> PortfolioDecisionSnapshot:
        market = self.market
        day = self._day
        n = self._n
        equity = self.equity
        marks = tuple(float(v) for v in market.prices[day])
        weights = tuple(float(v) for v in self._values / equity)
        quantities = tuple(float(v) for v in self._values / market.prices[day])
        drawdown = 1.0 - equity / self._peak
        allocator = self.config.allocator
        account = PortfolioAccountSnapshot(
            cash=float(self._cash),
            portfolio_value=equity,
            peak_value=float(self._peak),
            max_drawdown=float(drawdown),
            exact_quantities=tuple(repr(q) for q in quantities),
            quantities=quantities,
            weights=weights,
            mark_prices=marks,
            contract_multipliers=(1.0,) * n,
            turnover_total=float(self._turnover_total),
            total_cost=float(self._cost_total),
            funding_pnl=0.0,
            borrow_cost=0.0,
            margin_used=0.0,
            maintenance_margin=0.0,
            maintenance_requirement=0.0,
            margin_deficit=0.0,
            insolvent=False,
            termination_reason=None,
        )
        risk = PortfolioRiskLimits(
            max_gross=allocator.gross_max,
            max_abs_weight=allocator.w_max,
            max_turnover=None,
            drawdown_start=1.0,
            drawdown_stop=1.0,
            emergency_turnover_override=False,
            fail_closed_tolerance=1e-9,
        )
        total = market.spec.total_days
        return PortfolioDecisionSnapshot(
            index=day,
            timestamp_ns=day * NS_PER_DAY,
            agent_stop_index=total,
            remaining_decisions=total - day,
            symbols=market.spec.symbols,
            feature_names=(),
            global_feature_names=(),
            features=((),) * n,
            feature_available=((),) * n,
            feature_staleness=((),) * n,
            global_features=(),
            global_feature_available=(),
            account=account,
            active_orders=(),
            risk=risk,
        )

    def _propose(self) -> None:
        assert self._policy is not None
        proposal = self._policy.decide_portfolio(self._snapshot())
        self._target = np.asarray(proposal.target_weights, dtype=np.float64)

    def _observation(self) -> np.ndarray:
        market = self.market
        n = self._n
        day = min(self._day, market.spec.total_days - 1)
        scales = self.config.scales
        equity = self.equity
        w_cur = self._values / equity
        proposed = float(np.abs(self._target - w_cur).sum())
        drawdown = max(0.0, 1.0 - equity / self._peak)
        out = np.empty(self.config.observation_size, dtype=np.float64)
        out[:n] = market.mu[day] / scales.mu
        out[n : 2 * n] = w_cur
        out[2 * n] = proposed
        out[2 * n + 1] = sum(self._recent_costs) / scales.cost
        out[2 * n + 2] = drawdown / scales.drawdown
        out[2 * n + 3] = sum(self._recent_returns) / scales.recent_return
        np.clip(out, -scales.clip, scales.clip, out=out)
        return out.astype(np.float32)


__all__ = [
    "AllocatorSettings",
    "CostSettings",
    "EnvConfig",
    "ObservationScales",
    "ResidualAllocatorEnv",
    "apply_residual",
    "build_allocator",
]
