"""Trading bot execution, interval reporting, development comparison, and tuning."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field, replace
from itertools import product
from pathlib import Path
from typing import Literal

import numpy as np

from trade_rl.data import load_market_dataset_artifact
from trade_rl.data.features.price_channels import CHANNEL_NAMES, with_price_channels
from trade_rl.data.market import MarketDataset
from trade_rl.evaluation.replay import (
    SharedCashReplayResult,
    run_shared_cash_replay,
)
from trade_rl.risk import PreTradeRisk, PreTradeRiskConfig
from trade_rl.simulation import ExecutionCostConfig
from trade_rl.strategies.controls import ConstantIntentStrategy
from trade_rl.strategies.interface import SingleSymbolStrategy
from trade_rl.strategies.position_intent import PositionIntent
from trade_rl.strategies.rules.adaptive import (
    AdaptiveProfitConfig,
    RegimeAdaptiveStrategy,
)
from trade_rl.strategies.rules.channel_breakout import ChannelBreakoutStrategy
from trade_rl.strategies.rules.ensemble import EnsembleIntentStrategy
from trade_rl.strategies.rules.mean_reversion import (
    MeanReversionIntentConfig,
    MeanReversionIntentStrategy,
)
from trade_rl.strategies.rules.trend import TrendIntentConfig, TrendIntentStrategy

_ALLOWED_OBJECTIVES = frozenset({"profit", "sharpe", "balanced"})
_SELECTION_DRAWDOWN_LIMIT_PCT = 20.0
_DEFAULT_HOLDOUT_FRACTION = 0.2


@dataclass(frozen=True, slots=True)
class BotConfig:
    """One shared-cash replay configuration with explicit execution costs."""

    strategy_name: str = "ensemble"
    initial_capital: float = 100_000.0
    gross_budget: float = 0.2
    minimum_hold_bars: int = 4
    signal_index: int = 0
    entry_threshold: float = 0.01
    exit_threshold: float = 0.002
    max_gross: float = 1.0
    max_abs_weight: float = 0.40
    take_profit_threshold: float = 0.0
    stop_loss_threshold: float = 0.0
    trailing_stop_threshold: float = 0.0
    volatility_regime_threshold: float = 0.010
    max_holding_bars: int = 0
    execution_cost: ExecutionCostConfig = field(default_factory=ExecutionCostConfig)

    def __post_init__(self) -> None:
        if not isinstance(self.execution_cost, ExecutionCostConfig):
            raise ValueError("execution_cost must be an ExecutionCostConfig")


@dataclass(frozen=True, slots=True)
class TuningResult:
    """Summary of strategy hyperparameter tuning and profit maximization."""

    strategy_name: str
    objective: str
    baseline_config: BotConfig
    baseline_report: BotReport
    optimized_config: BotConfig
    optimized_report: BotReport
    profit_improvement_pct: float | None
    alpha_dollars: float
    evaluated_combinations: int
    selection_score: float
    selection_drawdown_pct: float
    tuning_start_index: int
    tuning_stop_index: int
    holdout_start_index: int
    holdout_stop_index: int
    dataset_id: str
    dataset_identity_bound: bool
    execution_cost: ExecutionCostConfig
    report_scope: Literal["holdout", "development_family_comparison"] = "holdout"


@dataclass(frozen=True, slots=True)
class BotReport:
    """Portfolio performance and explicitly interval-based return diagnostics."""

    strategy_name: str
    initial_capital: float
    final_equity: float
    net_pnl: float
    total_return_pct: float
    max_drawdown_pct: float
    nonzero_return_intervals: int
    positive_return_rate_pct: float
    interval_profit_factor: float
    sharpe_ratio: float
    is_profitable: bool
    terminal_settled: bool = True
    terminal_position_quantities: tuple[float, ...] = ()
    active_order_remainders: tuple[tuple[str, float], ...] = ()
    termination_reason: str | None = None


def generate_demo_dataset(
    n_bars: int = 500,
    n_symbols: int = 3,
    seed: int = 42,
) -> MarketDataset:
    """Generate synthetic EMA momentum and causal breakout inputs for smoke tests."""
    if isinstance(n_bars, bool) or not isinstance(n_bars, int) or n_bars < 3:
        raise ValueError("n_bars must be an integer >= 3")
    if (
        isinstance(n_symbols, bool)
        or not isinstance(n_symbols, int)
        or not 1 <= n_symbols <= 3
    ):
        raise ValueError("n_symbols must be an integer between 1 and 3")
    rng = np.random.default_rng(seed)
    symbols = tuple(f"CRYPTO_{i}" for i in range(n_symbols))

    drift = np.array([0.0005, -0.0002, 0.0003][:n_symbols])
    vol = np.array([0.015, 0.020, 0.012][:n_symbols])
    # Add GARCH-like volatility clustering: vol spikes persist for ~10 bars
    vol_shocks = rng.normal(0, 1, size=(n_bars, n_symbols))
    vol_path = np.ones((n_bars, n_symbols))
    for t in range(1, n_bars):
        vol_path[t] = 0.85 * vol_path[t - 1] + 0.15 * np.abs(vol_shocks[t])
    vol_path = np.clip(vol_path, 0.5, 3.0)

    log_returns = rng.normal(drift, vol, size=(n_bars, n_symbols)) * vol_path

    close_prices = 100.0 * np.exp(np.cumsum(log_returns, axis=0))
    open_prices = np.vstack([close_prices[0:1], close_prices[:-1]])
    high_prices = np.maximum(open_prices, close_prices) * (
        1.0 + np.abs(rng.normal(0, 0.005, size=(n_bars, n_symbols)))
    )
    low_prices = np.minimum(open_prices, close_prices) * (
        1.0 - np.abs(rng.normal(0, 0.005, size=(n_bars, n_symbols)))
    )
    volume = rng.uniform(10_000, 50_000, size=(n_bars, n_symbols))

    features = np.zeros((n_bars, n_symbols, 1), dtype=np.float32)
    for sym_idx in range(n_symbols):
        prices = close_prices[:, sym_idx]
        # EMA crossover: 8-bar EMA minus 21-bar EMA, normalised by price
        alpha_fast, alpha_slow = 2.0 / (8 + 1), 2.0 / (21 + 1)
        ema_fast = np.empty(n_bars)
        ema_slow = np.empty(n_bars)
        ema_fast[0] = ema_slow[0] = prices[0]
        for t in range(1, n_bars):
            ema_fast[t] = alpha_fast * prices[t] + (1 - alpha_fast) * ema_fast[t - 1]
            ema_slow[t] = alpha_slow * prices[t] + (1 - alpha_slow) * ema_slow[t - 1]
        # Feature 0: normalised EMA crossover momentum signal (positive = uptrend)
        features[:, sym_idx, 0] = ((ema_fast - ema_slow) / prices).astype(np.float32)

    timestamps = np.datetime64("2026-01-01T00:00:00", "ns") + np.arange(
        n_bars
    ) * np.timedelta64(1, "h")

    dataset = MarketDataset(
        dataset_id="d" * 64,
        symbols=symbols,
        timestamps=timestamps,
        features=features,
        global_features=np.zeros((n_bars, 1), dtype=np.float32),
        open=open_prices,
        high=high_prices,
        low=low_prices,
        close=close_prices,
        volume=volume,
        funding_rate=np.zeros((n_bars, n_symbols), dtype=np.float64),
        tradable=np.ones((n_bars, n_symbols), dtype=np.bool_),
        feature_available=np.ones((n_bars, n_symbols, 1), dtype=np.bool_),
        feature_names=("ema_crossover",),
        global_feature_names=("global_index",),
        periods_per_year=8_760,
    )
    entry_bars = min(20, n_bars - 1)
    exit_bars = min(10, entry_bars - 1)
    augmented = with_price_channels(dataset, entry_bars=entry_bars, exit_bars=exit_bars)
    return replace(augmented, identity_payload_json=None)


def create_strategy_instances(
    dataset: MarketDataset,
    config: BotConfig,
) -> list[SingleSymbolStrategy]:
    """Instantiate one independent strategy per symbol in dataset."""
    name = config.strategy_name.lower()
    instances: list[SingleSymbolStrategy] = []

    for _ in range(dataset.n_symbols):
        if name == "trend":
            t_cfg = TrendIntentConfig(
                signal_index=config.signal_index,
                entry_threshold=config.entry_threshold,
                exit_threshold=config.exit_threshold,
            )
            instances.append(TrendIntentStrategy(t_cfg))
        elif name == "mean_reversion":
            m_cfg = MeanReversionIntentConfig(
                signal_index=config.signal_index,
                entry_threshold=config.entry_threshold,
                exit_threshold=config.exit_threshold,
            )
            instances.append(MeanReversionIntentStrategy(m_cfg))
        elif name == "channel_breakout":
            missing = tuple(
                feature
                for feature in CHANNEL_NAMES
                if feature not in dataset.feature_names
            )
            if missing:
                raise ValueError(
                    "channel_breakout requires named price-channel features: "
                    + ", ".join(missing)
                )
            indices = tuple(
                dataset.feature_names.index(feature) for feature in CHANNEL_NAMES
            )
            instances.append(
                ChannelBreakoutStrategy(
                    (indices[0], indices[1], indices[2], indices[3])
                )
            )
        elif name == "ensemble":
            t_strat = TrendIntentStrategy(
                TrendIntentConfig(
                    signal_index=config.signal_index,
                    entry_threshold=config.entry_threshold,
                    exit_threshold=config.exit_threshold,
                )
            )
            m_strat = MeanReversionIntentStrategy(
                MeanReversionIntentConfig(
                    signal_index=config.signal_index,
                    entry_threshold=config.entry_threshold * 1.5,
                    exit_threshold=config.exit_threshold * 1.5,
                )
            )
            instances.append(
                EnsembleIntentStrategy([t_strat, m_strat], min_agreement=1)
            )
        elif name == "adaptive":
            a_cfg = AdaptiveProfitConfig(
                signal_index=config.signal_index,
                # Use the configured signal's magnitude for the momentum regime.
                volatility_index=config.signal_index,
                trend_entry_threshold=config.entry_threshold,
                trend_exit_threshold=config.exit_threshold,
                reversion_entry_threshold=config.entry_threshold * 1.2,
                reversion_exit_threshold=config.exit_threshold * 1.2,
                volatility_regime_threshold=config.volatility_regime_threshold,
                take_profit_threshold=config.take_profit_threshold,
                stop_loss_threshold=config.stop_loss_threshold,
                trailing_stop_threshold=config.trailing_stop_threshold,
                max_holding_bars=config.max_holding_bars,
            )
            instances.append(RegimeAdaptiveStrategy(a_cfg))
        elif name == "constant_long":
            instances.append(ConstantIntentStrategy(PositionIntent.LONG))
        elif name == "constant_short":
            instances.append(ConstantIntentStrategy(PositionIntent.SHORT))
        elif name == "cash":
            instances.append(ConstantIntentStrategy(PositionIntent.FLAT))
        else:
            raise ValueError(f"Unknown strategy: {config.strategy_name}")

    return instances


def calculate_bot_report(
    replay_result: SharedCashReplayResult,
    strategy_name: str,
    initial_capital: float = 100_000.0,
) -> BotReport:
    """Extract performance metrics from replay ledger."""
    raw_returns = np.asarray(replay_result.returns.values, dtype=np.float64)

    initial_cap = initial_capital
    final_cap = float(replay_result.book.portfolio_value)
    net_pnl = final_cap - initial_cap
    total_ret = (net_pnl / initial_cap) * 100.0

    # The shared-cash book tracks the full execution-path drawdown, including
    # adverse movement that interval-end returns cannot represent.
    max_dd = float(replay_result.book.max_drawdown) * 100.0

    # Returns & Sharpe
    mean_ret = float(np.mean(raw_returns)) if len(raw_returns) > 0 else 0.0
    std_ret = float(np.std(raw_returns)) if len(raw_returns) > 0 else 0.0
    annualization_periods = replay_result.returns.annualization_periods_per_year
    sharpe = (
        float((mean_ret / std_ret) * math.sqrt(annualization_periods))
        if std_ret > 1e-8
        else 0.0
    )

    # These are return-interval metrics, not closed-trade diagnostics.
    positive_returns = raw_returns[raw_returns > 1e-6]
    negative_returns = raw_returns[raw_returns < -1e-6]
    n_win = len(positive_returns)
    n_loss = len(negative_returns)
    nonzero_return_intervals = n_win + n_loss
    positive_return_rate = (
        n_win / nonzero_return_intervals * 100.0
        if nonzero_return_intervals > 0
        else 0.0
    )

    gross_profit = float(np.sum(positive_returns)) if n_win > 0 else 0.0
    gross_loss = float(abs(np.sum(negative_returns))) if n_loss > 0 else 0.0
    interval_profit_factor = (
        (gross_profit / gross_loss)
        if gross_loss > 1e-8
        else (99.0 if gross_profit > 0 else 0.0)
    )

    ledger = replay_result.ledger_evidence
    terminal_quantities = tuple(float(value) for value in replay_result.book.quantities)
    terminal_settled = (
        ledger is not None
        and not ledger.active_order_remainders
        and all(value == 0 for value in replay_result.book.exact_quantities)
        and replay_result.book.termination_reason is None
    )
    return BotReport(
        strategy_name=strategy_name,
        initial_capital=initial_cap,
        final_equity=final_cap,
        net_pnl=net_pnl,
        total_return_pct=total_ret,
        max_drawdown_pct=max_dd,
        nonzero_return_intervals=nonzero_return_intervals,
        positive_return_rate_pct=positive_return_rate,
        interval_profit_factor=interval_profit_factor,
        sharpe_ratio=sharpe,
        is_profitable=net_pnl > 0,
        terminal_settled=terminal_settled,
        terminal_position_quantities=terminal_quantities,
        active_order_remainders=()
        if ledger is None
        else ledger.active_order_remainders,
        termination_reason=(
            None
            if replay_result.book.termination_reason is None
            else str(replay_result.book.termination_reason)
        ),
    )


def run_trading_bot(
    dataset: MarketDataset,
    config: BotConfig,
    *,
    start_index: int = 0,
    stop_index: int | None = None,
) -> tuple[SharedCashReplayResult, BotReport]:
    """Execute trading bot simulation on dataset with configured strategy and risk."""
    resolved_stop_index = dataset.n_bars - 1 if stop_index is None else stop_index
    if (
        isinstance(start_index, bool)
        or not isinstance(start_index, int)
        or isinstance(resolved_stop_index, bool)
        or not isinstance(resolved_stop_index, int)
        or not 0 <= start_index < resolved_stop_index < dataset.n_bars
    ):
        raise ValueError("replay range must satisfy 0 <= start < stop < n_bars")

    strategies = create_strategy_instances(dataset, config)
    risk_config = PreTradeRiskConfig(
        max_gross=config.max_gross,
        max_abs_weight=config.max_abs_weight,
    )
    risk = PreTradeRisk(risk_config)

    result = run_shared_cash_replay(
        dataset,
        strategies,
        start_index=start_index,
        stop_index=resolved_stop_index,
        gross_budget=config.gross_budget,
        initial_capital=config.initial_capital,
        risk=risk,
        minimum_hold_bars=config.minimum_hold_bars,
        execution_cost=config.execution_cost,
        settle_terminal_position=True,
        capture_ledger_evidence=True,
    )

    report = calculate_bot_report(
        result,
        config.strategy_name,
        initial_capital=config.initial_capital,
    )
    return result, report


def compare_all_strategies(
    dataset: MarketDataset,
    initial_capital: float = 100_000.0,
    gross_budget: float = 0.2,
    execution_cost: ExecutionCostConfig | None = None,
) -> list[BotReport]:
    """Rank full-range strategy replays as in-sample diagnostics, not selection."""
    strategies_to_test = [
        "adaptive",
        "ensemble",
        "trend",
        "mean_reversion",
        "channel_breakout",
        "constant_long",
        "cash",
    ]
    reports: list[BotReport] = []

    for strat_name in strategies_to_test:
        cfg = BotConfig(
            strategy_name=strat_name,
            initial_capital=initial_capital,
            gross_budget=gross_budget,
            execution_cost=(
                ExecutionCostConfig() if execution_cost is None else execution_cost
            ),
        )
        _, report = run_trading_bot(dataset, cfg)
        reports.append(report)

    # Rank by total net profit descending
    reports.sort(key=lambda r: r.net_pnl, reverse=True)
    return reports


def tune_for_maximum_profit(
    dataset: MarketDataset,
    strategy_name: str = "adaptive",
    initial_capital: float = 100_000.0,
    objective: str = "profit",
    max_combinations: int = 150,
    holdout_fraction: float = _DEFAULT_HOLDOUT_FRACTION,
    execution_cost: ExecutionCostConfig | None = None,
) -> TuningResult:
    """Select parameters on a chronological prefix and report on a later holdout.

    The default holdout is the final 20% of usable bars. Candidate eligibility
    requires tuning-window ledger drawdown at or below 20%; gaps can exceed that
    limit in either window.
    """
    if not isinstance(objective, str) or objective not in _ALLOWED_OBJECTIVES:
        raise ValueError(f"objective must be one of {sorted(_ALLOWED_OBJECTIVES)}")
    if (
        isinstance(max_combinations, bool)
        or not isinstance(max_combinations, int)
        or max_combinations <= 0
    ):
        raise ValueError("max_combinations must be a positive integer")
    if (
        isinstance(holdout_fraction, bool)
        or not isinstance(holdout_fraction, (int, float))
        or not math.isfinite(holdout_fraction)
        or not 0.0 < holdout_fraction < 1.0
    ):
        raise ValueError("holdout_fraction must be finite and within (0, 1)")
    resolved_execution_cost = (
        ExecutionCostConfig() if execution_cost is None else execution_cost
    )
    if not isinstance(resolved_execution_cost, ExecutionCostConfig):
        raise ValueError("execution_cost must be an ExecutionCostConfig")

    usable_stop_index = dataset.n_bars - 1
    minimum_window_span = resolved_execution_cost.order_latency_bars + 2
    tuning_stop_index = math.floor(usable_stop_index * (1.0 - float(holdout_fraction)))
    if (
        tuning_stop_index < minimum_window_span
        or usable_stop_index - tuning_stop_index < minimum_window_span
    ):
        raise ValueError(
            "dataset is too short for separate tuning and holdout windows "
            "with terminal settlement"
        )

    return _tune_with_fixed_windows(
        dataset=dataset,
        strategy_name=strategy_name,
        initial_capital=initial_capital,
        objective=objective,
        max_combinations=max_combinations,
        tune_start_index=0,
        tune_stop_index=tuning_stop_index,
        eval_start_index=tuning_stop_index,
        eval_stop_index=usable_stop_index,
        execution_cost=resolved_execution_cost,
    )


def tune_all_strategies(
    dataset: MarketDataset,
    initial_capital: float = 100_000.0,
    objective: str = "profit",
    max_combinations_per_strategy: int = 60,
    holdout_fraction: float = _DEFAULT_HOLDOUT_FRACTION,
    execution_cost: ExecutionCostConfig | None = None,
) -> list[TuningResult]:
    """Compare strategy families on development data, ranked by tuning-window score.

    Because the later report window is exposed for every family, it is a
    development comparison and must not be treated as a final untouched holdout.
    """
    candidate_strategies = [
        "adaptive",
        "ensemble",
        "trend",
        "mean_reversion",
        "channel_breakout",
    ]
    results: list[TuningResult] = []
    for strat in candidate_strategies:
        res = tune_for_maximum_profit(
            dataset,
            strategy_name=strat,
            initial_capital=initial_capital,
            objective=objective,
            max_combinations=max_combinations_per_strategy,
            holdout_fraction=holdout_fraction,
            execution_cost=execution_cost,
        )
        results.append(res)

    results.sort(key=lambda result: result.selection_score, reverse=True)
    return [
        replace(result, report_scope="development_family_comparison")
        for result in results
    ]


def optimize_bot_parameters(
    dataset: MarketDataset,
    strategy_name: str = "ensemble",
    initial_capital: float = 100_000.0,
    holdout_fraction: float = _DEFAULT_HOLDOUT_FRACTION,
    execution_cost: ExecutionCostConfig | None = None,
) -> tuple[BotConfig, BotReport]:
    """Grid search optimization to maximize net return and profit factor."""
    res = tune_for_maximum_profit(
        dataset,
        strategy_name=strategy_name,
        initial_capital=initial_capital,
        objective="profit",
        holdout_fraction=holdout_fraction,
        execution_cost=execution_cost,
    )
    return res.optimized_config, res.optimized_report


def print_report_table(reports: Sequence[BotReport]) -> None:
    """Print an attractive summary table of performance metrics."""
    header = f"{'Strategy':<18} | {'Final Equity':<14} | {'Net P&L':<12} | {'Return':<9} | {'Max DD':<8} | {'Positive Rate':<14} | {'Interval PF':<12} | {'Sharpe':<6} | {'Settled'}"
    sep = "-" * len(header)
    print("\n" + sep)
    print(header)
    print(sep)
    for r in reports:
        star = " *" if r.is_profitable else ""
        print(
            f"{r.strategy_name:<18} | "
            f"${r.final_equity:>12,.2f} | "
            f"${r.net_pnl:>10,.2f} | "
            f"{r.total_return_pct:>7.2f}% | "
            f"{r.max_drawdown_pct:>6.2f}% | "
            f"{r.positive_return_rate_pct:>12.1f}% | "
            f"{r.interval_profit_factor:>10.2f} | "
            f"{r.sharpe_ratio:>6.2f}{star} | {'yes' if r.terminal_settled else 'NO'}"
        )
    print(sep + "\n")


def print_tuning_comparison(res: TuningResult) -> None:
    """Print detailed comparison between default baseline and tuned bot."""
    b = res.baseline_report
    o = res.optimized_report
    cfg = res.optimized_config

    print("\n" + "=" * 68)
    print(
        f"  PARAMETER TUNING REPORT: {res.strategy_name.upper()} (Objective: {res.objective})"
    )
    print(f"  Combinations evaluated: {res.evaluated_combinations}")
    print(
        f"  Dataset identity: {res.dataset_id} "
        f"(canonical identity bound: {'yes' if res.dataset_identity_bound else 'no'})"
    )
    print(f"  Execution cost config: {asdict(res.execution_cost)}")
    report_label = (
        "development family comparison"
        if res.report_scope == "development_family_comparison"
        else "holdout report"
    )
    print(
        f"  Tuning window: [{res.tuning_start_index}, {res.tuning_stop_index}); "
        f"{report_label}: [{res.holdout_start_index}, {res.holdout_stop_index})"
    )
    print(
        f"  Tuning selection score: {res.selection_score:.4f}; "
        f"selection drawdown: {res.selection_drawdown_pct:.2f}% "
        f"(eligibility limit: {_SELECTION_DRAWDOWN_LIMIT_PCT:.0f}%)"
    )
    print(
        "  Adaptive exits use bar-close gross return from average executed entry price; "
        "fees and carry are excluded, and the exit fills only at a later eligible step."
    )
    print("=" * 68)
    baseline_label = (
        "Baseline development"
        if res.report_scope == "development_family_comparison"
        else "Baseline holdout"
    )
    candidate_label = (
        "Candidate development"
        if res.report_scope == "development_family_comparison"
        else "Candidate holdout"
    )
    print(f"{'Metric':<24} | {baseline_label:<18} | {candidate_label:<18}")
    print("-" * 68)
    print(
        f"{'Final Equity':<24} | ${b.final_equity:>16,.2f} | ${o.final_equity:>16,.2f}"
    )
    print(f"{'Net Profit ($)':<24} | ${b.net_pnl:>16,.2f} | ${o.net_pnl:>16,.2f}")
    print(
        f"{'Terminal settled':<24} | {str(b.terminal_settled):>18} | {str(o.terminal_settled):>18}"
    )
    if not b.terminal_settled or not o.terminal_settled:
        print(
            "  Incomplete settlement: equity includes residual marked inventory; inspect JSON quantities/orders."
        )
    print(
        f"{'Total Return (%)':<24} | {b.total_return_pct:>17.2f}% | {o.total_return_pct:>17.2f}%"
    )
    print(f"{'Sharpe Ratio':<24} | {b.sharpe_ratio:>18.2f} | {o.sharpe_ratio:>18.2f}")
    print(
        f"{'Interval Profit Factor':<24} | {b.interval_profit_factor:>18.2f} | {o.interval_profit_factor:>18.2f}"
    )
    print(
        f"{'Max Drawdown (%)':<24} | {b.max_drawdown_pct:>17.2f}% | {o.max_drawdown_pct:>17.2f}%"
    )
    print(
        f"{'Positive Return Rate (%)':<24} | "
        f"{b.positive_return_rate_pct:>17.1f}% | "
        f"{o.positive_return_rate_pct:>17.1f}%"
    )
    print(
        f"{'Nonzero Return Intervals':<24} | "
        f"{b.nonzero_return_intervals:>18d} | "
        f"{o.nonzero_return_intervals:>18d}"
    )
    print("-" * 68)
    sign = "+" if res.alpha_dollars >= 0 else ""
    relative_improvement = (
        f"{sign}{res.profit_improvement_pct:.2f}%"
        if res.profit_improvement_pct is not None
        else "N/A (baseline P&L is near zero)"
    )
    print(
        f"🚀 Profit Improvement:  {sign}${res.alpha_dollars:,.2f} USD ({relative_improvement})"
    )
    print("🎯 Optimal Parameters:")
    print(f"   - Entry Threshold:      {cfg.entry_threshold}")
    print(f"   - Exit Threshold:       {cfg.exit_threshold}")
    print(f"   - Min Hold Bars:        {cfg.minimum_hold_bars}")
    print(f"   - Gross Budget:         {cfg.gross_budget * 100:.1f}%")
    if cfg.take_profit_threshold > 0:
        print(f"   - Gross TP trigger:     +{cfg.take_profit_threshold * 100:.1f}%")
    if cfg.stop_loss_threshold > 0:
        print(f"   - Gross SL trigger:     -{cfg.stop_loss_threshold * 100:.1f}%")
    if cfg.trailing_stop_threshold > 0:
        print(f"   - Gross trailing exit:  -{cfg.trailing_stop_threshold * 100:.1f}%")
    print("=" * 68 + "\n")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run, compare, and tune shared-cash trading strategies."
    )
    parser.add_argument(
        "--mode",
        choices=["run", "compare", "optimize", "walk-forward"],
        default="run",
        help="Execution mode (default: run)",
    )
    parser.add_argument(
        "--strategy",
        default="adaptive",
        help="Strategy name (adaptive, ensemble, trend, mean_reversion, channel_breakout, constant_long, constant_short, cash, or 'all' for optimize)",
    )
    parser.add_argument(
        "--objective",
        choices=["profit", "sharpe", "balanced"],
        default="profit",
        help="Optimization objective function for --mode optimize (default: profit)",
    )
    parser.add_argument(
        "--windows",
        type=int,
        default=3,
        help="Number of chronological folds for --mode walk-forward (default: 3)",
    )
    parser.add_argument(
        "--max-combinations",
        type=int,
        default=None,
        help="Candidate budget per tuning window (default: 150 single, 60 all/walk-forward)",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="Path to an existing market dataset artifact directory",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Explicitly use the generated synthetic demo dataset",
    )
    parser.add_argument(
        "--capital",
        type=float,
        default=100_000.0,
        help="Initial capital in USDT (default: 100,000)",
    )
    parser.add_argument(
        "--gross-budget",
        type=float,
        default=0.2,
        help="Gross portfolio allocation fraction per symbol (default: 0.2)",
    )
    parser.add_argument(
        "--min-hold",
        type=int,
        default=4,
        help="Minimum position holding bars (default: 4)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output raw JSON instead of text table",
    )

    args = parser.parse_args(argv)

    def announce(message: str) -> None:
        print(message, file=sys.stderr if args.json else sys.stdout)

    if args.dataset is not None and args.demo:
        parser.error("--dataset and --demo cannot be used together")
    if args.dataset is not None:
        if not args.dataset.is_dir():
            parser.error(f"--dataset must be an existing directory: {args.dataset}")
        announce(f"Loading market dataset from {args.dataset}...")
        dataset = load_market_dataset_artifact(args.dataset)
    elif args.demo:
        announce("Using explicitly requested synthetic demo market data...")
        dataset = generate_demo_dataset()
    else:
        parser.error(
            "provide --dataset or --demo; a dataset path must be an existing directory"
        )

    announce(f"Dataset: {dataset.n_symbols} symbols, {dataset.n_bars} bars.")

    if args.mode == "compare":
        announce(
            "Running full-dataset in-sample diagnostic comparisons; "
            "this is not an out-of-sample selection."
        )
        reports = compare_all_strategies(
            dataset,
            initial_capital=args.capital,
            gross_budget=args.gross_budget,
        )
        if args.json:
            print(json.dumps([asdict(r) for r in reports], indent=2))
        else:
            print_report_table(reports)
            winner = reports[0]
            announce(
                f"Highest observed diagnostic P&L: {winner.strategy_name} "
                f"({winner.total_return_pct:+.2f}% / ${winner.net_pnl:,.2f}); "
                "not a holdout selection."
            )
    elif args.mode == "walk-forward":
        if args.strategy.lower() == "all":
            parser.error("walk-forward requires one fixed strategy family")
        announce(
            "Walk-forward development diagnostic; each replay uses independently reset capital."
        )
        result = walk_forward_tune(
            dataset,
            strategy_name=args.strategy,
            initial_capital=args.capital,
            objective=args.objective,
            n_windows=args.windows,
            max_combinations=60
            if args.max_combinations is None
            else args.max_combinations,
        )
        if args.json:
            print(json.dumps(asdict(result), indent=2))
        else:
            print_walk_forward_summary(result)
    elif args.mode == "optimize":
        if args.strategy.lower() == "all":
            announce(f"Tuning all candidate strategies for maximum {args.objective}...")
            announce(
                "This compares strategy families on development data. The later "
                "per-family reports are development comparisons; selecting a "
                "family after viewing them requires a new untouched final window."
            )
            all_tuning = tune_all_strategies(
                dataset,
                initial_capital=args.capital,
                objective=args.objective,
                max_combinations_per_strategy=60
                if args.max_combinations is None
                else args.max_combinations,
            )
            if args.json:
                print(json.dumps([asdict(t) for t in all_tuning], indent=2))
            else:
                for res in all_tuning:
                    print_tuning_comparison(res)
                champion = all_tuning[0]
                print(
                    f"👑 TOP TUNING-WINDOW STRATEGY: {champion.strategy_name.upper()} "
                    f"(selection score {champion.selection_score:.4f}); "
                    f"development comparison: ${champion.optimized_report.net_pnl:,.2f} "
                    f"({champion.optimized_report.total_return_pct:+.2f}%)"
                )
        else:
            announce(
                f"Optimizing parameters for '{args.strategy}' (Objective: {args.objective})..."
            )
            tuning_res = tune_for_maximum_profit(
                dataset,
                strategy_name=args.strategy,
                initial_capital=args.capital,
                objective=args.objective,
                max_combinations=150
                if args.max_combinations is None
                else args.max_combinations,
            )
            if args.json:
                print(json.dumps(asdict(tuning_res), indent=2))
            else:
                print_tuning_comparison(tuning_res)
    else:
        # Run single bot
        cfg = BotConfig(
            strategy_name=args.strategy,
            initial_capital=args.capital,
            gross_budget=args.gross_budget,
            minimum_hold_bars=args.min_hold,
        )
        announce(f"Executing trading bot with strategy '{cfg.strategy_name}'...")
        _, report = run_trading_bot(dataset, cfg)
        if args.json:
            print(json.dumps(asdict(report), indent=2))
        else:
            print_report_table([report])
            if report.is_profitable and report.terminal_settled:
                print(
                    f"✅ Bot execution profitable: +${report.net_pnl:,.2f} (+{report.total_return_pct:.2f}%)"
                )
            else:
                print(f"⚠️ Bot execution resulted in net change: ${report.net_pnl:,.2f}")
            if not report.terminal_settled:
                print(
                    "Terminal settlement incomplete; equity includes residual marked positions."
                )

    return 0


@dataclass(frozen=True, slots=True)
class WalkForwardResult:
    """Aggregated walk-forward validation result across multiple consecutive windows.

    Each window tunes on the preceding fold and reports the next fold with fresh
    capital and strategy state. The return product is a hypothetical normalized
    summary, not a continuously compounded execution ledger. Repeated inspection
    makes these windows development evidence rather than sealed final evidence.
    """

    strategy_name: str
    objective: str
    n_windows: int
    window_results: tuple[TuningResult, ...]
    cumulative_return_pct: float
    mean_window_return_pct: float
    std_window_return_pct: float
    min_window_return_pct: float
    max_window_return_pct: float
    mean_window_sharpe: float
    mean_window_max_drawdown_pct: float
    profitable_windows: int
    profitable_window_rate_pct: float
    dataset_id: str
    execution_cost: ExecutionCostConfig
    dataset_identity_bound: bool
    report_scope: Literal["development_walk_forward"] = "development_walk_forward"
    capital_mode: Literal["reset_each_window"] = "reset_each_window"
    cumulative_return_scope: Literal["hypothetical_normalized_product"] = (
        "hypothetical_normalized_product"
    )


def walk_forward_tune(
    dataset: MarketDataset,
    strategy_name: str = "adaptive",
    initial_capital: float = 100_000.0,
    objective: str = "balanced",
    n_windows: int = 3,
    max_combinations: int = 60,
    execution_cost: ExecutionCostConfig | None = None,
) -> WalkForwardResult:
    """Walk-forward validation: tune on a window, evaluate on the next.

    Splits usable bars into ``n_windows`` equal folds. For each fold i (i >= 1),
    uses fold i-1 as the tuning window and fold i as the evaluation window.
    This gives n_windows - 1 chronological reports. The final fold includes any
    remainder bars. Later folds can reuse previously evaluated data for tuning;
    adjacent reports are not statistically independent.

    Args:
        dataset: Market data to evaluate on.
        strategy_name: Strategy family to tune.
        initial_capital: Starting capital per window.
        objective: Scoring objective for parameter selection within each window.
        n_windows: Number of equal folds. Must be >= 2. Default 3 gives two
            separately reset evaluation windows.
        max_combinations: Candidate combinations per tuning window.
        execution_cost: Execution cost configuration. Defaults to non-zero costs.

    Returns:
        WalkForwardResult summarising performance across all evaluation windows.
    """
    if isinstance(n_windows, bool) or not isinstance(n_windows, int) or n_windows < 2:
        raise ValueError("n_windows must be an integer >= 2")
    if not isinstance(objective, str) or objective not in _ALLOWED_OBJECTIVES:
        raise ValueError(f"objective must be one of {sorted(_ALLOWED_OBJECTIVES)}")

    if (
        isinstance(max_combinations, bool)
        or not isinstance(max_combinations, int)
        or max_combinations <= 0
    ):
        raise ValueError("max_combinations must be a positive integer")

    resolved_cost = ExecutionCostConfig() if execution_cost is None else execution_cost
    if not isinstance(resolved_cost, ExecutionCostConfig):
        raise ValueError("execution_cost must be an ExecutionCostConfig")

    usable_bars = dataset.n_bars - 1  # last bar reserved for terminal settlement
    minimum_window_span = resolved_cost.order_latency_bars + 2
    window_size = usable_bars // n_windows
    if window_size < minimum_window_span:
        raise ValueError(
            f"dataset too short for {n_windows} windows "
            f"(need at least {n_windows * minimum_window_span + 1} bars)"
        )

    window_results: list[TuningResult] = []
    for fold_index in range(1, n_windows):
        tune_start = (fold_index - 1) * window_size
        tune_stop = fold_index * window_size
        eval_stop = (
            usable_bars
            if fold_index == n_windows - 1
            else (fold_index + 1) * window_size
        )

        # Run grid search on tuning window and evaluate on the next window
        tuning_res = _tune_with_fixed_windows(
            dataset=dataset,
            strategy_name=strategy_name,
            initial_capital=initial_capital,
            objective=objective,
            max_combinations=max_combinations,
            tune_start_index=tune_start,
            tune_stop_index=tune_stop,
            eval_start_index=tune_stop,
            eval_stop_index=eval_stop,
            execution_cost=resolved_cost,
        )
        window_results.append(tuning_res)

    returns_pct = [r.optimized_report.total_return_pct for r in window_results]
    sharpes = [r.optimized_report.sharpe_ratio for r in window_results]
    max_dds = [r.optimized_report.max_drawdown_pct for r in window_results]
    profitable_count = sum(
        1 for r in window_results if r.optimized_report.is_profitable
    )

    mean_ret = float(np.mean(returns_pct)) if returns_pct else 0.0
    std_ret = float(np.std(returns_pct)) if len(returns_pct) > 1 else 0.0
    cumulative = float(np.prod([1.0 + r / 100.0 for r in returns_pct]) - 1.0) * 100.0

    return WalkForwardResult(
        strategy_name=strategy_name,
        objective=objective,
        n_windows=n_windows,
        window_results=tuple(window_results),
        cumulative_return_pct=cumulative,
        mean_window_return_pct=mean_ret,
        std_window_return_pct=std_ret,
        min_window_return_pct=float(min(returns_pct)) if returns_pct else 0.0,
        max_window_return_pct=float(max(returns_pct)) if returns_pct else 0.0,
        mean_window_sharpe=float(np.mean(sharpes)) if sharpes else 0.0,
        mean_window_max_drawdown_pct=float(np.mean(max_dds)) if max_dds else 0.0,
        profitable_windows=profitable_count,
        profitable_window_rate_pct=(
            profitable_count / len(window_results) * 100.0 if window_results else 0.0
        ),
        dataset_id=dataset.dataset_id,
        dataset_identity_bound=dataset.identity_payload_json is not None,
        execution_cost=resolved_cost,
    )


def _tune_with_fixed_windows(
    dataset: MarketDataset,
    strategy_name: str,
    initial_capital: float,
    objective: str,
    max_combinations: int,
    tune_start_index: int,
    tune_stop_index: int,
    eval_start_index: int,
    eval_stop_index: int,
    execution_cost: ExecutionCostConfig,
) -> TuningResult:
    """Internal helper: grid search on [tune_start, tune_stop) and evaluate on [eval_start, eval_stop)."""
    baseline_cfg = BotConfig(
        strategy_name=strategy_name,
        initial_capital=initial_capital,
        execution_cost=execution_cost,
    )
    _, baseline_tuning_report = run_trading_bot(
        dataset,
        baseline_cfg,
        start_index=tune_start_index,
        stop_index=tune_stop_index,
    )

    strat_name = strategy_name.lower()

    if strat_name == "adaptive":
        entry_candidates = [0.005, 0.008, 0.012, 0.018]
        exit_multipliers = [0.2, 0.4]
        hold_candidates = [2, 4, 6]
        budget_candidates = [0.15, 0.25, 0.35]
        tp_candidates = [0.0, 0.015, 0.030]
        sl_candidates = [0.0, 0.015, 0.025]
        ts_candidates = [0.0, 0.010]
        vol_regime_candidates = [0.008, 0.012]
    else:
        entry_candidates = [0.005, 0.010, 0.015, 0.020]
        exit_multipliers = [0.2, 0.5]
        hold_candidates = [2, 4, 8]
        budget_candidates = [0.15, 0.25, 0.35]
        tp_candidates = [0.0]
        sl_candidates = [0.0]
        ts_candidates = [0.0]
        vol_regime_candidates = [0.010]

    def score_report(rep: BotReport) -> float:
        if objective == "profit":
            return rep.net_pnl
        if objective == "sharpe":
            return (
                rep.sharpe_ratio if rep.net_pnl > 0 else -100.0 + rep.net_pnl / 1000.0
            )
        if objective == "balanced":
            # Keep return signed: losses must rank below cash, never become rewards.
            dd_denom = max(rep.max_drawdown_pct, 0.1)  # ゼロDD除外
            calmar_proxy = rep.total_return_pct / dd_denom
            pf_bonus = min(rep.interval_profit_factor, 3.0)
            return calmar_proxy * (1.0 + pf_bonus)
        return rep.net_pnl

    best_cfg: BotConfig | None = None
    best_tuning_report: BotReport | None = None
    best_score = float("-inf")
    if (
        baseline_tuning_report.terminal_settled
        and baseline_tuning_report.max_drawdown_pct <= _SELECTION_DRAWDOWN_LIMIT_PCT
    ):
        best_cfg = baseline_cfg
        best_tuning_report = baseline_tuning_report
        best_score = score_report(baseline_tuning_report)

    parameter_dimensions = (
        entry_candidates,
        exit_multipliers,
        hold_candidates,
        budget_candidates,
        tp_candidates,
        sl_candidates,
        ts_candidates,
        vol_regime_candidates,
    )
    parameter_space_size = math.prod(
        len(dimension) for dimension in parameter_dimensions
    )
    sample_count = min(max_combinations, parameter_space_size)
    candidate_combinations: list[
        tuple[float, float, int, float, float, float, float, float]
    ]
    if sample_count == parameter_space_size:
        candidate_combinations = list(product(*parameter_dimensions))
    else:
        sampled_indices: list[list[int]] = []
        dimension_sizes = (
            len(entry_candidates),
            len(exit_multipliers),
            len(hold_candidates),
            len(budget_candidates),
            len(tp_candidates),
            len(sl_candidates),
            len(ts_candidates),
            len(vol_regime_candidates),
        )
        for dimension_index, level_count in enumerate(dimension_sizes):
            if sample_count >= level_count:
                indices = [
                    sample_index * level_count // sample_count
                    for sample_index in range(sample_count)
                ]
            elif sample_count == 1:
                indices = [(level_count - 1) // 2]
            else:
                indices = [
                    math.floor(
                        sample_index * (level_count - 1) / (sample_count - 1) + 0.5
                    )
                    for sample_index in range(sample_count)
                ]
            rotation = (
                dimension_index * max(1, sample_count // len(parameter_dimensions))
            ) % sample_count
            sampled_indices.append(indices[rotation:] + indices[:rotation])

        candidate_combinations = []
        seen_combinations: set[
            tuple[float, float, int, float, float, float, float, float]
        ] = set()
        for sample_index in range(sample_count):
            combination = (
                entry_candidates[sampled_indices[0][sample_index]],
                exit_multipliers[sampled_indices[1][sample_index]],
                hold_candidates[sampled_indices[2][sample_index]],
                budget_candidates[sampled_indices[3][sample_index]],
                tp_candidates[sampled_indices[4][sample_index]],
                sl_candidates[sampled_indices[5][sample_index]],
                ts_candidates[sampled_indices[6][sample_index]],
                vol_regime_candidates[sampled_indices[7][sample_index]],
            )
            if combination not in seen_combinations:
                candidate_combinations.append(combination)
                seen_combinations.add(combination)
        if len(candidate_combinations) < sample_count:
            for combination in product(*parameter_dimensions):
                if combination in seen_combinations:
                    continue
                candidate_combinations.append(combination)
                seen_combinations.add(combination)
                if len(candidate_combinations) == sample_count:
                    break

    count = 0
    for entry, mult, hold, budget, tp, sl, ts, vol_regime in candidate_combinations:
        count += 1
        test_cfg = BotConfig(
            strategy_name=strategy_name,
            initial_capital=initial_capital,
            gross_budget=budget,
            minimum_hold_bars=hold,
            entry_threshold=entry,
            exit_threshold=entry * mult,
            take_profit_threshold=tp,
            stop_loss_threshold=sl,
            trailing_stop_threshold=ts,
            volatility_regime_threshold=vol_regime,
            execution_cost=execution_cost,
        )
        _, tuning_report = run_trading_bot(
            dataset,
            test_cfg,
            start_index=tune_start_index,
            stop_index=tune_stop_index,
        )
        if (
            not tuning_report.terminal_settled
            or tuning_report.max_drawdown_pct > _SELECTION_DRAWDOWN_LIMIT_PCT
        ):
            continue
        score = score_report(tuning_report)
        if best_cfg is None or score > best_score:
            best_cfg = test_cfg
            best_tuning_report = tuning_report
            best_score = score

    if best_cfg is None or best_tuning_report is None:
        raise ValueError(
            "no configuration met the 20% tuning-window drawdown eligibility limit and terminal settlement"
        )

    _, baseline_rep = run_trading_bot(
        dataset,
        baseline_cfg,
        start_index=eval_start_index,
        stop_index=eval_stop_index,
    )
    _, optimized_rep = run_trading_bot(
        dataset,
        best_cfg,
        start_index=eval_start_index,
        stop_index=eval_stop_index,
    )

    profit_diff = optimized_rep.net_pnl - baseline_rep.net_pnl
    if abs(baseline_rep.net_pnl) > 1e-4:
        improvement_pct = (profit_diff / abs(baseline_rep.net_pnl)) * 100.0
    else:
        improvement_pct = None

    return TuningResult(
        strategy_name=strategy_name,
        objective=objective,
        baseline_config=baseline_cfg,
        baseline_report=baseline_rep,
        optimized_config=best_cfg,
        optimized_report=optimized_rep,
        profit_improvement_pct=improvement_pct,
        alpha_dollars=profit_diff,
        evaluated_combinations=count,
        selection_score=best_score,
        selection_drawdown_pct=best_tuning_report.max_drawdown_pct,
        tuning_start_index=tune_start_index,
        tuning_stop_index=tune_stop_index,
        holdout_start_index=eval_start_index,
        holdout_stop_index=eval_stop_index,
        dataset_id=dataset.dataset_id,
        dataset_identity_bound=dataset.identity_payload_json is not None,
        execution_cost=execution_cost,
    )


def print_walk_forward_summary(result: WalkForwardResult) -> None:
    """Print a concise walk-forward validation summary."""
    print("\n" + "=" * 68)
    print(f"  WALK-FORWARD VALIDATION: {result.strategy_name.upper()}")
    print(
        f"  Objective: {result.objective}  |  Windows evaluated: {result.n_windows - 1}"
    )
    print(f"  Dataset: {result.dataset_id}")
    print(f"  Execution cost config: {asdict(result.execution_cost)}")
    print(
        "  Development diagnostic with independently reset capital and strategy state."
    )
    print(
        "  Return product assumes normalized reinvestment; it is not one executed wealth path."
    )
    print("=" * 68)
    print(
        f"  {'Window':<8} | {'Return':>10} | {'Sharpe':>8} | {'Max DD':>8} | {'Terminal settled'}"
    )
    print("  " + "-" * 56)
    for i, wr in enumerate(result.window_results):
        rep = wr.optimized_report
        settled = "yes" if rep.terminal_settled else "NO"
        print(
            f"  {i + 1:<8} | {rep.total_return_pct:>+9.2f}% | {rep.sharpe_ratio:>8.2f} | "
            f"{rep.max_drawdown_pct:>7.2f}% | {settled}"
        )
    print("  " + "-" * 56)
    print(f"  Hypothetical compounded return: {result.cumulative_return_pct:+.2f}%")
    print(
        f"  Mean ± Std return:  {result.mean_window_return_pct:+.2f}% ± {result.std_window_return_pct:.2f}%"
    )
    print(
        f"  Range: [{result.min_window_return_pct:+.2f}%, {result.max_window_return_pct:+.2f}%]"
    )
    print(f"  Mean Sharpe:        {result.mean_window_sharpe:.2f}")
    print(f"  Mean Max Drawdown:  {result.mean_window_max_drawdown_pct:.2f}%")
    print(
        f"  Profitable windows: {result.profitable_windows}/{len(result.window_results)} "
        f"({result.profitable_window_rate_pct:.0f}%)"
    )
    print("=" * 68 + "\n")


__all__ = [
    "BotConfig",
    "BotReport",
    "TuningResult",
    "WalkForwardResult",
    "compare_all_strategies",
    "generate_demo_dataset",
    "main",
    "optimize_bot_parameters",
    "print_report_table",
    "print_tuning_comparison",
    "print_walk_forward_summary",
    "run_trading_bot",
    "tune_all_strategies",
    "tune_for_maximum_profit",
    "walk_forward_tune",
]


if __name__ == "__main__":
    raise SystemExit(main())
