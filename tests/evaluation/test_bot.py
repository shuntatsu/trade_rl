from __future__ import annotations

from trade_rl.evaluation.bot import (
    BotConfig,
    compare_all_strategies,
    generate_demo_dataset,
    optimize_bot_parameters,
    run_trading_bot,
    tune_all_strategies,
    tune_for_maximum_profit,
)


def test_generate_demo_dataset_is_valid() -> None:
    dataset = generate_demo_dataset(n_bars=100, n_symbols=2, seed=123)
    assert dataset.n_bars == 100
    assert dataset.n_symbols == 2
    assert len(dataset.symbols) == 2
    assert dataset.features.shape == (100, 2, 4)


def test_run_trading_bot_executes_successfully() -> None:
    dataset = generate_demo_dataset(n_bars=100, n_symbols=2, seed=123)
    cfg = BotConfig(
        strategy_name="trend",
        initial_capital=50_000.0,
        gross_budget=0.15,
        minimum_hold_bars=2,
    )
    result, report = run_trading_bot(dataset, cfg)
    assert report.initial_capital == 50_000.0
    assert report.final_equity > 0.0
    assert report.max_drawdown_pct >= 0.0
    assert len(result.decisions) == dataset.n_bars - 2


def test_compare_all_strategies_ranks_by_profit() -> None:
    dataset = generate_demo_dataset(n_bars=100, n_symbols=2, seed=123)
    reports = compare_all_strategies(dataset, initial_capital=10_000.0)
    assert len(reports) >= 4
    # Ensure descending order of net_pnl
    for i in range(len(reports) - 1):
        assert reports[i].net_pnl >= reports[i + 1].net_pnl


def test_optimize_bot_parameters_finds_best_config() -> None:
    dataset = generate_demo_dataset(n_bars=80, n_symbols=2, seed=42)
    best_cfg, best_report = optimize_bot_parameters(
        dataset,
        strategy_name="mean_reversion",
        initial_capital=10_000.0,
    )
    assert best_cfg.strategy_name == "mean_reversion"
    assert best_report.initial_capital == 10_000.0
    assert best_report.strategy_name == "mean_reversion"


def test_adaptive_strategy_bot_execution() -> None:
    dataset = generate_demo_dataset(n_bars=100, n_symbols=2, seed=123)
    cfg = BotConfig(
        strategy_name="adaptive",
        initial_capital=20_000.0,
        gross_budget=0.20,
        minimum_hold_bars=4,
    )
    result, report = run_trading_bot(dataset, cfg)
    assert report.strategy_name == "adaptive"
    assert report.initial_capital == 20_000.0
    assert report.final_equity > 0.0
    assert len(result.decisions) == dataset.n_bars - 2


def test_tune_for_maximum_profit_finds_improvements() -> None:
    dataset = generate_demo_dataset(n_bars=80, n_symbols=2, seed=42)
    tuning_res = tune_for_maximum_profit(
        dataset,
        strategy_name="adaptive",
        initial_capital=10_000.0,
        objective="profit",
        max_combinations=20,
    )
    assert tuning_res.strategy_name == "adaptive"
    assert tuning_res.evaluated_combinations > 0
    assert tuning_res.optimized_report.final_equity > 0.0
    assert tuning_res.optimized_report.net_pnl >= tuning_res.baseline_report.net_pnl


def test_tune_all_strategies_executes_and_ranks() -> None:
    dataset = generate_demo_dataset(n_bars=80, n_symbols=2, seed=42)
    ranked = tune_all_strategies(
        dataset,
        initial_capital=10_000.0,
        objective="profit",
        max_combinations_per_strategy=5,
    )
    assert len(ranked) >= 4
    for i in range(len(ranked) - 1):
        assert (
            ranked[i].optimized_report.net_pnl >= ranked[i + 1].optimized_report.net_pnl
        )
