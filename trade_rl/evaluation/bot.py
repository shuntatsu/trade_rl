"""Unified Trading Bot runner, evaluation, parameter optimization, and comparison.

Executes production strategies against shared multi-symbol portfolio cash and strict
risk controls, aiming for maximum profit and risk-adjusted returns.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from trade_rl.data import load_market_dataset_artifact
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
from trade_rl.strategies.rules.channel_breakout import ChannelBreakoutStrategy
from trade_rl.strategies.rules.ensemble import EnsembleIntentStrategy
from trade_rl.strategies.rules.mean_reversion import (
    MeanReversionIntentConfig,
    MeanReversionIntentStrategy,
)
from trade_rl.strategies.rules.trend import TrendIntentConfig, TrendIntentStrategy


@dataclass(frozen=True, slots=True)
class BotConfig:
    """Configuration for one Trading Bot run."""

    strategy_name: str = "ensemble"
    initial_capital: float = 100_000.0
    gross_budget: float = 0.2
    minimum_hold_bars: int = 4
    signal_index: int = 0
    entry_threshold: float = 0.01
    exit_threshold: float = 0.002
    max_gross: float = 1.0
    max_abs_weight: float = 0.40


@dataclass(frozen=True, slots=True)
class BotReport:
    """Summary of trading bot performance and economics."""

    strategy_name: str
    initial_capital: float
    final_equity: float
    net_pnl: float
    total_return_pct: float
    max_drawdown_pct: float
    num_trades: int
    win_rate_pct: float
    profit_factor: float
    sharpe_ratio: float
    is_profitable: bool


def generate_demo_dataset(
    n_bars: int = 500,
    n_symbols: int = 3,
    seed: int = 42,
) -> MarketDataset:
    """Generate deterministic, realistic multi-symbol market data for immediate testing."""
    rng = np.random.default_rng(seed)
    symbols = tuple(f"CRYPTO_{i}" for i in range(n_symbols))

    # Log-normal random walk with drifting trends
    drift = np.array([0.0005, -0.0002, 0.0003][:n_symbols])
    vol = np.array([0.015, 0.020, 0.012][:n_symbols])
    log_returns = rng.normal(drift, vol, size=(n_bars, n_symbols))

    close_prices = 100.0 * np.exp(np.cumsum(log_returns, axis=0))
    open_prices = np.vstack([close_prices[0:1], close_prices[:-1]])
    high_prices = np.maximum(open_prices, close_prices) * (
        1.0 + np.abs(rng.normal(0, 0.005, size=(n_bars, n_symbols)))
    )
    low_prices = np.minimum(open_prices, close_prices) * (
        1.0 - np.abs(rng.normal(0, 0.005, size=(n_bars, n_symbols)))
    )
    volume = rng.uniform(10_000, 50_000, size=(n_bars, n_symbols))

    # Features: return signals and channel bounds
    features = np.zeros((n_bars, n_symbols, 4), dtype=np.float32)
    for sym_idx in range(n_symbols):
        # Feature 0: normalized return signal (momentum)
        returns = (close_prices[:, sym_idx] - open_prices[:, sym_idx]) / open_prices[
            :, sym_idx
        ]
        features[:, sym_idx, 0] = returns
        # Feature 1-3: breakout channel indicators
        features[:, sym_idx, 1] = returns * 1.5
        features[:, sym_idx, 2] = -returns * 1.5
        features[:, sym_idx, 3] = returns * 0.5

    timestamps = np.datetime64("2026-01-01T00:00:00", "ns") + np.arange(
        n_bars
    ) * np.timedelta64(1, "h")

    return MarketDataset(
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
        feature_available=np.ones((n_bars, n_symbols, 4), dtype=np.bool_),
        feature_names=("momentum", "ch_upper", "ch_lower", "ch_mid"),
        global_feature_names=("global_index",),
        periods_per_year=8_760,
    )


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
            instances.append(ChannelBreakoutStrategy((0, 1, 2, 3)))
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
    # Reconstruct cumulative equity curve from actual realized bar returns
    equity_multiplier = np.cumprod(1.0 + raw_returns)
    equity = np.concatenate(([initial_capital], initial_capital * equity_multiplier))

    initial_cap = initial_capital
    final_cap = float(replay_result.book.portfolio_value)
    net_pnl = final_cap - initial_cap
    total_ret = (net_pnl / initial_cap) * 100.0

    # Drawdown
    running_max = np.maximum.accumulate(equity)
    drawdowns = (running_max - equity) / np.maximum(running_max, 1e-8)
    max_dd = float(np.max(drawdowns)) * 100.0

    # Returns & Sharpe
    mean_ret = float(np.mean(raw_returns)) if len(raw_returns) > 0 else 0.0
    std_ret = float(np.std(raw_returns)) if len(raw_returns) > 0 else 0.0
    sharpe = (
        float((mean_ret / std_ret) * math.sqrt(252 * 24)) if std_ret > 1e-8 else 0.0
    )

    # Trades
    positive_returns = raw_returns[raw_returns > 1e-6]
    negative_returns = raw_returns[raw_returns < -1e-6]
    n_win = len(positive_returns)
    n_loss = len(negative_returns)
    total_trades = n_win + n_loss
    win_rate = (n_win / total_trades * 100.0) if total_trades > 0 else 0.0

    gross_profit = float(np.sum(positive_returns)) if n_win > 0 else 0.0
    gross_loss = float(abs(np.sum(negative_returns))) if n_loss > 0 else 0.0
    profit_factor = (
        (gross_profit / gross_loss)
        if gross_loss > 1e-8
        else (99.0 if gross_profit > 0 else 0.0)
    )

    return BotReport(
        strategy_name=strategy_name,
        initial_capital=initial_cap,
        final_equity=final_cap,
        net_pnl=net_pnl,
        total_return_pct=total_ret,
        max_drawdown_pct=max_dd,
        num_trades=total_trades,
        win_rate_pct=win_rate,
        profit_factor=profit_factor,
        sharpe_ratio=sharpe,
        is_profitable=net_pnl > 0,
    )


def run_trading_bot(
    dataset: MarketDataset,
    config: BotConfig,
) -> tuple[SharedCashReplayResult, BotReport]:
    """Execute trading bot simulation on dataset with configured strategy and risk."""
    strategies = create_strategy_instances(dataset, config)
    risk_config = PreTradeRiskConfig(
        max_gross=config.max_gross,
        max_abs_weight=config.max_abs_weight,
    )
    risk = PreTradeRisk(risk_config)

    result = run_shared_cash_replay(
        dataset,
        strategies,
        start_index=0,
        stop_index=dataset.n_bars - 1,
        gross_budget=config.gross_budget,
        initial_capital=config.initial_capital,
        risk=risk,
        minimum_hold_bars=config.minimum_hold_bars,
        execution_cost=ExecutionCostConfig.zero(),
        settle_terminal_position=True,
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
) -> list[BotReport]:
    """Simulate and rank all candidate strategies to find the one with maximum profit."""
    strategies_to_test = [
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
        )
        _, report = run_trading_bot(dataset, cfg)
        reports.append(report)

    # Rank by total net profit descending
    reports.sort(key=lambda r: r.net_pnl, reverse=True)
    return reports


def optimize_bot_parameters(
    dataset: MarketDataset,
    strategy_name: str = "ensemble",
    initial_capital: float = 100_000.0,
) -> tuple[BotConfig, BotReport]:
    """Grid search optimization to maximize net return and profit factor."""
    best_config = BotConfig(
        strategy_name=strategy_name, initial_capital=initial_capital
    )
    best_report: BotReport | None = None

    entry_candidates = [0.005, 0.01, 0.015, 0.02]
    exit_multipliers = [0.2, 0.5]
    hold_bars_candidates = [2, 4, 8]
    budget_candidates = [0.15, 0.25]

    for entry in entry_candidates:
        for mult in exit_multipliers:
            exit_th = entry * mult
            for hold in hold_bars_candidates:
                for budget in budget_candidates:
                    cfg = BotConfig(
                        strategy_name=strategy_name,
                        initial_capital=initial_capital,
                        gross_budget=budget,
                        minimum_hold_bars=hold,
                        entry_threshold=entry,
                        exit_threshold=exit_th,
                    )
                    _, rep = run_trading_bot(dataset, cfg)
                    if best_report is None or rep.net_pnl > best_report.net_pnl:
                        best_report = rep
                        best_config = cfg

    assert best_report is not None
    return best_config, best_report


def print_report_table(reports: Sequence[BotReport]) -> None:
    """Print an attractive summary table of performance metrics."""
    header = f"{'Strategy':<18} | {'Final Equity':<14} | {'Net P&L':<12} | {'Return':<9} | {'Max DD':<8} | {'WinRate':<8} | {'PF':<6} | {'Sharpe':<6}"
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
            f"{r.win_rate_pct:>6.1f}% | "
            f"{r.profit_factor:>6.2f} | "
            f"{r.sharpe_ratio:>6.2f}{star}"
        )
    print(sep + "\n")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Unified Trading Bot: Simulate, compare, and optimize for maximum profit."
    )
    parser.add_argument(
        "--mode",
        choices=["run", "compare", "optimize"],
        default="run",
        help="Execution mode (default: run)",
    )
    parser.add_argument(
        "--strategy",
        default="ensemble",
        help="Strategy name (trend, mean_reversion, channel_breakout, ensemble)",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="Path to market dataset artifact directory (default: synthetic demo)",
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

    if args.dataset is not None and args.dataset.is_dir():
        print(f"Loading market dataset from {args.dataset}...")
        dataset = load_market_dataset_artifact(args.dataset)
    else:
        print("Using built-in multi-symbol demo market data...")
        dataset = generate_demo_dataset()

    print(f"Dataset: {dataset.n_symbols} symbols, {dataset.n_bars} bars.")

    if args.mode == "compare":
        print("Simulating all candidate strategies to determine maximum profit...")
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
            print(
                f"🏆 Top Profit Strategy: {winner.strategy_name} (+{winner.total_return_pct:.2f}% / ${winner.net_pnl:,.2f})"
            )
    elif args.mode == "optimize":
        print(f"Optimizing parameters for '{args.strategy}' to maximize net profit...")
        opt_cfg, opt_report = optimize_bot_parameters(
            dataset,
            strategy_name=args.strategy,
            initial_capital=args.capital,
        )
        if args.json:
            print(
                json.dumps(
                    {"config": asdict(opt_cfg), "report": asdict(opt_report)}, indent=2
                )
            )
        else:
            print("Optimal config found:")
            print(f"  entry_threshold: {opt_cfg.entry_threshold}")
            print(f"  exit_threshold:  {opt_cfg.exit_threshold}")
            print(f"  minimum_hold:    {opt_cfg.minimum_hold_bars} bars")
            print(f"  gross_budget:    {opt_cfg.gross_budget * 100:.1f}%")
            print_report_table([opt_report])
    else:
        # Run single bot
        cfg = BotConfig(
            strategy_name=args.strategy,
            initial_capital=args.capital,
            gross_budget=args.gross_budget,
            minimum_hold_bars=args.min_hold,
        )
        print(f"Executing trading bot with strategy '{cfg.strategy_name}'...")
        _, report = run_trading_bot(dataset, cfg)
        if args.json:
            print(json.dumps(asdict(report), indent=2))
        else:
            print_report_table([report])
            if report.is_profitable:
                print(
                    f"✅ Bot execution profitable: +${report.net_pnl:,.2f} (+{report.total_return_pct:.2f}%)"
                )
            else:
                print(f"⚠️ Bot execution resulted in net change: ${report.net_pnl:,.2f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "BotConfig",
    "BotReport",
    "compare_all_strategies",
    "generate_demo_dataset",
    "main",
    "optimize_bot_parameters",
    "run_trading_bot",
]
