from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from trade_rl.evaluation.bot import (
    BotConfig,
    BotReport,
    TuningResult,
    WalkForwardResult,
    calculate_bot_report,
    compare_all_strategies,
    generate_demo_dataset,
    main,
    optimize_bot_parameters,
    run_trading_bot,
    tune_all_strategies,
    tune_for_maximum_profit,
    walk_forward_tune,
)
from trade_rl.simulation import ExecutionCostConfig


def _with_holdout_shock(dataset, start_index: int):
    close = dataset.close.copy()
    factors = np.linspace(0.95, 0.4, dataset.n_bars - start_index)
    close[start_index:] *= factors[:, np.newaxis]
    high = np.maximum(dataset.open, close) * 1.01
    low = np.minimum(dataset.open, close) * 0.99
    features = dataset.features.copy()
    features[start_index:, :, 0] = -np.abs(features[start_index:, :, 0]) - 0.05
    return replace(dataset, close=close, high=high, low=low, features=features)


def test_generate_demo_dataset_is_valid() -> None:
    dataset = generate_demo_dataset(n_bars=100, n_symbols=2, seed=123)
    assert dataset.n_bars == 100
    assert dataset.n_symbols == 2
    assert len(dataset.symbols) == 2
    assert dataset.features.shape == (100, 2, 5)


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


def test_run_trading_bot_uses_nonzero_execution_costs_by_default() -> None:
    dataset = generate_demo_dataset(n_bars=100, n_symbols=2, seed=123)
    config = BotConfig(strategy_name="constant_long", gross_budget=0.2)
    free_config = replace(config, execution_cost=ExecutionCostConfig.zero())

    costed_result, _ = run_trading_bot(dataset, config)
    free_result, _ = run_trading_bot(dataset, free_config)

    assert config.execution_cost == ExecutionCostConfig()
    assert costed_result.book.total_cost > 0.0
    assert costed_result.book.total_cost > free_result.book.total_cost
    assert costed_result.book.portfolio_value < free_result.book.portfolio_value


def test_tuning_result_binds_dataset_identity_and_execution_costs() -> None:
    dataset = generate_demo_dataset(
        n_bars=50, n_symbols=1, seed=17
    ).with_content_identity({"fixture": "tuning-provenance"})
    execution_cost = replace(ExecutionCostConfig(), fee_rate=0.001)

    result = tune_for_maximum_profit(
        dataset,
        strategy_name="trend",
        max_combinations=1,
        execution_cost=execution_cost,
    )

    assert result.dataset_id == dataset.dataset_id
    assert result.dataset_identity_bound
    assert result.baseline_config.execution_cost == execution_cost
    assert result.optimized_config.execution_cost == execution_cost


def test_run_trading_bot_replays_only_the_requested_window() -> None:
    dataset = generate_demo_dataset(n_bars=40, n_symbols=1, seed=19)
    start_index = 10
    stop_index = 30

    result, report = run_trading_bot(
        dataset,
        BotConfig(strategy_name="constant_long"),
        start_index=start_index,
        stop_index=stop_index,
    )
    _, full_report = run_trading_bot(
        dataset,
        BotConfig(strategy_name="constant_long"),
    )

    assert report.final_equity != full_report.final_equity
    assert len(result.decisions) == stop_index - start_index - 1
    assert result.book.total_cost > 0.0


def test_bot_report_names_interval_metrics_explicitly() -> None:
    dataset = generate_demo_dataset(n_bars=60, n_symbols=1, seed=23)
    replay, report = run_trading_bot(
        dataset,
        BotConfig(strategy_name="constant_long"),
    )
    returns = np.asarray(replay.returns.values, dtype=np.float64)
    positive = returns[returns > 1e-6]
    negative = returns[returns < -1e-6]
    nonzero = len(positive) + len(negative)

    assert report.nonzero_return_intervals == nonzero
    assert report.positive_return_rate_pct == pytest.approx(
        len(positive) / nonzero * 100.0 if nonzero else 0.0
    )
    expected_profit_factor = (
        float(np.sum(positive)) / abs(float(np.sum(negative)))
        if len(negative)
        else (99.0 if len(positive) else 0.0)
    )
    assert report.interval_profit_factor == pytest.approx(expected_profit_factor)


def test_bot_report_sharpe_uses_return_series_periods_per_year() -> None:
    hourly = generate_demo_dataset(n_bars=60, n_symbols=1, seed=43)
    timestamps = hourly.timestamps[0] + np.arange(hourly.n_bars) * np.timedelta64(
        1, "D"
    )
    daily = replace(
        hourly,
        timestamps=timestamps,
        available_at=np.broadcast_to(
            timestamps[:, np.newaxis],
            (hourly.n_bars, hourly.n_symbols),
        ).copy(),
        periods_per_year=365,
    )
    replay, _ = run_trading_bot(daily, BotConfig(strategy_name="constant_long"))
    report = calculate_bot_report(replay, "constant_long")
    returns = np.asarray(replay.returns.values, dtype=np.float64)
    expected = float(np.mean(returns) / np.std(returns) * np.sqrt(365))

    assert report.sharpe_ratio == pytest.approx(expected)


def test_tuning_selection_ignores_the_later_holdout_but_reports_it() -> None:
    dataset = generate_demo_dataset(n_bars=60, n_symbols=1, seed=29)
    prices = (100.0 * np.exp(0.01 * np.arange(dataset.n_bars)))[:, np.newaxis]
    features = dataset.features.copy()
    features[:, :, 0] = 0.1
    dataset = replace(
        dataset,
        open=prices,
        high=prices,
        low=prices,
        close=prices,
        features=features,
        identity_payload_json=None,
    )
    tuning = tune_for_maximum_profit(
        dataset,
        strategy_name="trend",
        max_combinations=8,
        holdout_fraction=0.2,
    )
    shocked = _with_holdout_shock(dataset, tuning.holdout_start_index + 1)
    shocked_tuning = tune_for_maximum_profit(
        shocked,
        strategy_name="trend",
        max_combinations=8,
        holdout_fraction=0.2,
    )

    assert tuning.tuning_start_index == 0
    assert tuning.tuning_stop_index == tuning.holdout_start_index
    assert tuning.holdout_stop_index == dataset.n_bars - 1
    assert tuning.optimized_config == shocked_tuning.optimized_config
    assert tuning.selection_score == shocked_tuning.selection_score
    assert tuning.selection_drawdown_pct == shocked_tuning.selection_drawdown_pct
    assert tuning.optimized_config.strategy_name == "trend"
    assert tuning.optimized_report != shocked_tuning.optimized_report

    _, direct_holdout_report = run_trading_bot(
        dataset,
        tuning.baseline_config,
        start_index=tuning.holdout_start_index,
        stop_index=tuning.holdout_stop_index,
    )
    assert tuning.baseline_report == direct_holdout_report
    _, direct_candidate_report = run_trading_bot(
        dataset,
        tuning.optimized_config,
        start_index=tuning.holdout_start_index,
        stop_index=tuning.holdout_stop_index,
    )
    assert tuning.optimized_report == direct_candidate_report


def test_tuning_rejects_invalid_objective_and_windows() -> None:
    dataset = generate_demo_dataset(n_bars=60, n_symbols=1, seed=31)

    with pytest.raises(ValueError, match="objective"):
        tune_for_maximum_profit(dataset, objective="unknown")
    with pytest.raises(ValueError, match="objective"):
        tune_for_maximum_profit(dataset, objective=[])
    with pytest.raises(ValueError, match="holdout_fraction"):
        tune_for_maximum_profit(dataset, holdout_fraction=1.0)
    with pytest.raises(ValueError, match="max_combinations"):
        tune_for_maximum_profit(dataset, max_combinations=0)
    with pytest.raises(ValueError, match="too short"):
        tune_for_maximum_profit(generate_demo_dataset(n_bars=6, n_symbols=1))
    with pytest.raises(ValueError, match="replay range"):
        run_trading_bot(
            dataset,
            BotConfig(),
            start_index=20,
            stop_index=20,
        )


def test_drawdown_ineligible_candidate_cannot_win(monkeypatch) -> None:
    import trade_rl.evaluation.bot as bot_module

    dataset = generate_demo_dataset(n_bars=40, n_symbols=1, seed=37)
    calls: list[tuple[BotConfig, int, int | None]] = []

    def report_for(
        config: BotConfig,
        *,
        training: bool,
    ) -> BotReport:
        if training:
            values = {
                0.15: (10_000.0, 25.0),
                0.20: (0.0, 5.0),
                0.25: (1_000.0, 15.0),
                0.35: (500.0, 10.0),
            }
            pnl, drawdown = values[config.gross_budget]
        else:
            pnl, drawdown = (11.0, 3.0) if config.gross_budget == 0.20 else (42.0, 7.0)
        return BotReport(
            strategy_name=config.strategy_name,
            initial_capital=config.initial_capital,
            final_equity=config.initial_capital + pnl,
            net_pnl=pnl,
            total_return_pct=pnl / config.initial_capital * 100.0,
            max_drawdown_pct=drawdown,
            nonzero_return_intervals=2,
            positive_return_rate_pct=50.0,
            interval_profit_factor=1.0,
            sharpe_ratio=1.0,
            is_profitable=pnl > 0.0,
        )

    def fake_run(
        _dataset,
        config: BotConfig,
        *,
        start_index: int = 0,
        stop_index: int | None = None,
    ):
        calls.append((config, start_index, stop_index))
        return None, report_for(config, training=start_index == 0)

    monkeypatch.setattr(bot_module, "run_trading_bot", fake_run)
    result = tune_for_maximum_profit(
        dataset,
        strategy_name="trend",
        max_combinations=8,
    )

    assert result.optimized_config.gross_budget == 0.25
    assert result.optimized_report.net_pnl == 42.0
    assert result.baseline_report.net_pnl == 11.0
    assert all(
        stop == result.tuning_stop_index
        for _, start, stop in calls
        if start == result.tuning_start_index
    )
    assert all(
        start == result.holdout_start_index
        for _, start, _ in calls
        if start != result.tuning_start_index
    )


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
    assert tuning_res.selection_drawdown_pct <= 20.0
    assert tuning_res.dataset_id == dataset.dataset_id
    assert not tuning_res.dataset_identity_bound
    assert tuning_res.execution_cost == ExecutionCostConfig()
    assert tuning_res.alpha_dollars == (
        tuning_res.optimized_report.net_pnl - tuning_res.baseline_report.net_pnl
    )


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
        assert ranked[i].selection_score >= ranked[i + 1].selection_score


@pytest.mark.parametrize("objective", ["sharpe", "balanced"])
def test_tune_all_strategies_ranks_by_the_selected_objective(
    monkeypatch,
    objective: str,
) -> None:
    import trade_rl.evaluation.bot as bot_module

    objective_scores = {
        "adaptive": 1.0,
        "ensemble": 2.0,
        "trend": 3.0,
        "mean_reversion": 4.0,
        "channel_breakout": 5.0,
    }
    holdout_pnl = {
        "adaptive": 500.0,
        "ensemble": 400.0,
        "trend": 300.0,
        "mean_reversion": 200.0,
        "channel_breakout": 100.0,
    }
    observed_objectives: list[str] = []

    def fake_tune(_dataset, *, strategy_name: str, objective: str, **_kwargs):
        observed_objectives.append(objective)
        config = BotConfig(strategy_name=strategy_name)
        report = BotReport(
            strategy_name=strategy_name,
            initial_capital=1_000.0,
            final_equity=1_000.0 + holdout_pnl[strategy_name],
            net_pnl=holdout_pnl[strategy_name],
            total_return_pct=holdout_pnl[strategy_name] / 10.0,
            max_drawdown_pct=5.0,
            nonzero_return_intervals=1,
            positive_return_rate_pct=100.0,
            interval_profit_factor=2.0,
            sharpe_ratio=objective_scores[strategy_name],
            is_profitable=True,
        )
        return TuningResult(
            strategy_name=strategy_name,
            objective=objective,
            baseline_config=config,
            baseline_report=report,
            optimized_config=config,
            optimized_report=report,
            profit_improvement_pct=0.0,
            alpha_dollars=0.0,
            evaluated_combinations=1,
            selection_score=objective_scores[strategy_name],
            selection_drawdown_pct=5.0,
            tuning_start_index=0,
            tuning_stop_index=10,
            holdout_start_index=10,
            holdout_stop_index=19,
            dataset_id="0" * 64,
            dataset_identity_bound=False,
            execution_cost=ExecutionCostConfig(),
        )

    monkeypatch.setattr(bot_module, "tune_for_maximum_profit", fake_tune)
    ranked = tune_all_strategies(
        generate_demo_dataset(n_bars=20, n_symbols=1),
        objective=objective,
    )

    assert [result.strategy_name for result in ranked] == [
        "channel_breakout",
        "mean_reversion",
        "trend",
        "ensemble",
        "adaptive",
    ]
    assert observed_objectives == [objective] * 5
    assert {result.report_scope for result in ranked} == {
        "development_family_comparison"
    }


def test_bounded_tuning_grid_samples_every_parameter_axis(monkeypatch) -> None:
    import trade_rl.evaluation.bot as bot_module

    dataset = generate_demo_dataset(n_bars=40, n_symbols=1, seed=47)
    evaluated: list[BotConfig] = []

    def fake_run(
        _dataset,
        config: BotConfig,
        *,
        start_index: int = 0,
        stop_index: int | None = None,
    ):
        if start_index == 0 and config.gross_budget != 0.2:
            evaluated.append(config)
        report = BotReport(
            strategy_name=config.strategy_name,
            initial_capital=config.initial_capital,
            final_equity=config.initial_capital + 1.0,
            net_pnl=1.0,
            total_return_pct=0.001,
            max_drawdown_pct=5.0,
            nonzero_return_intervals=1,
            positive_return_rate_pct=100.0,
            interval_profit_factor=1.0,
            sharpe_ratio=1.0,
            is_profitable=True,
        )
        return None, report

    monkeypatch.setattr(bot_module, "run_trading_bot", fake_run)
    tune_for_maximum_profit(
        dataset,
        strategy_name="adaptive",
        max_combinations=8,
    )

    assert len(evaluated) == 8
    assert {config.entry_threshold for config in evaluated} == {
        0.005,
        0.008,
        0.012,
        0.018,
    }
    assert {config.exit_threshold / config.entry_threshold for config in evaluated} == {
        0.2,
        0.4,
    }
    assert {config.minimum_hold_bars for config in evaluated} == {2, 4, 6}
    assert {config.gross_budget for config in evaluated} == {0.15, 0.25, 0.35}
    assert {config.take_profit_threshold for config in evaluated} == {
        0.0,
        0.015,
        0.030,
    }
    assert {config.stop_loss_threshold for config in evaluated} <= {0.0, 0.015, 0.025}
    assert {config.trailing_stop_threshold for config in evaluated} == {0.0, 0.010}
    assert {config.volatility_regime_threshold for config in evaluated} == {
        0.008,
        0.012,
    }


def test_near_zero_holdout_baseline_has_undefined_relative_improvement(
    monkeypatch,
    capsys,
) -> None:
    import trade_rl.evaluation.bot as bot_module

    dataset = generate_demo_dataset(n_bars=30, n_symbols=1, seed=53)

    def fake_run(
        _dataset,
        config: BotConfig,
        *,
        start_index: int = 0,
        stop_index: int | None = None,
    ):
        is_baseline = config.gross_budget == 0.2
        if start_index == 0:
            pnl = 0.0 if is_baseline else 1.0
        else:
            pnl = 0.00001 if is_baseline else -10.0
        report = BotReport(
            strategy_name=config.strategy_name,
            initial_capital=config.initial_capital,
            final_equity=config.initial_capital + pnl,
            net_pnl=pnl,
            total_return_pct=pnl / config.initial_capital * 100.0,
            max_drawdown_pct=5.0,
            nonzero_return_intervals=1,
            positive_return_rate_pct=50.0,
            interval_profit_factor=1.0,
            sharpe_ratio=1.0,
            is_profitable=pnl > 0.0,
        )
        return None, report

    monkeypatch.setattr(bot_module, "run_trading_bot", fake_run)
    result = tune_for_maximum_profit(
        dataset,
        strategy_name="trend",
        max_combinations=1,
    )

    assert result.alpha_dollars < 0.0
    assert result.profit_improvement_pct is None
    bot_module.print_tuning_comparison(result)
    assert "N/A (baseline P&L is near zero)" in capsys.readouterr().out


def test_compare_cli_marks_full_dataset_ranking_as_in_sample_diagnostic(
    monkeypatch,
    capsys,
) -> None:
    dataset = generate_demo_dataset(n_bars=20, n_symbols=1, seed=59)
    report = BotReport(
        strategy_name="trend",
        initial_capital=100_000.0,
        final_equity=100_001.0,
        net_pnl=1.0,
        total_return_pct=0.001,
        max_drawdown_pct=1.0,
        nonzero_return_intervals=1,
        positive_return_rate_pct=100.0,
        interval_profit_factor=1.0,
        sharpe_ratio=1.0,
        is_profitable=True,
    )
    monkeypatch.setattr(
        "trade_rl.evaluation.bot.generate_demo_dataset",
        lambda: dataset,
    )
    monkeypatch.setattr(
        "trade_rl.evaluation.bot.compare_all_strategies",
        lambda *_args, **_kwargs: [report],
    )

    main(["--mode", "compare", "--demo", "--json"])

    captured = capsys.readouterr()
    assert "in-sample diagnostic" in captured.err.lower()
    assert json.loads(captured.out)[0]["strategy_name"] == "trend"
    assert "not an out-of-sample selection" in captured.err.lower()
    assert "top profit strategy" not in captured.out.lower()


def test_explicit_missing_dataset_path_does_not_use_demo_data(capsys) -> None:
    missing_dataset = Path("nonexistent-market-artifact-for-test")

    with pytest.raises(SystemExit) as error:
        main(["--dataset", str(missing_dataset)])

    assert error.value.code == 2
    assert "existing directory" in capsys.readouterr().err


def test_optimize_cli_requires_explicit_dataset_or_demo_mode(
    monkeypatch, capsys
) -> None:
    monkeypatch.setattr(
        "trade_rl.evaluation.bot.generate_demo_dataset",
        lambda: pytest.fail("optimizer must not silently use demo data"),
    )

    with pytest.raises(SystemExit) as error:
        main(["--mode", "optimize", "--strategy", "all", "--json"])

    assert error.value.code == 2
    assert "--dataset or --demo" in capsys.readouterr().err


def test_optimize_json_is_parseable_and_contains_data_cost_provenance(
    monkeypatch,
    capsys,
    tmp_path: Path,
) -> None:
    dataset = generate_demo_dataset(
        n_bars=60, n_symbols=1, seed=61
    ).with_content_identity({"fixture": "cli-tuning-provenance"})
    dataset_path = tmp_path / "market-artifact"
    dataset_path.mkdir()
    monkeypatch.setattr(
        "trade_rl.evaluation.bot.load_market_dataset_artifact",
        lambda _path: dataset,
    )

    main(
        [
            "--mode",
            "optimize",
            "--strategy",
            "trend",
            "--dataset",
            str(dataset_path),
            "--json",
        ]
    )

    output = capsys.readouterr().out
    result = json.loads(output)
    assert result["dataset_id"] == dataset.dataset_id
    assert result["dataset_identity_bound"] is True
    assert result["execution_cost"]["fee_rate"] == ExecutionCostConfig().fee_rate


def test_cli_help_does_not_print_runtime_adaptive_exit_notes(capsys) -> None:
    with pytest.raises(SystemExit) as error:
        main(["--help"])

    assert error.value.code == 0
    output = capsys.readouterr().out
    assert "--demo" in output
    assert "Adaptive exits use bar-close" not in output


def test_walk_forward_tune_returns_n_minus_one_windows() -> None:
    dataset = generate_demo_dataset(n_bars=200, n_symbols=1, seed=71)
    result = walk_forward_tune(
        dataset,
        strategy_name="trend",
        n_windows=3,
        max_combinations=4,
    )
    assert isinstance(result, WalkForwardResult)
    assert result.n_windows == 3
    assert len(result.window_results) == 2  # n_windows - 1 evaluation windows
    assert result.strategy_name == "trend"
    assert result.dataset_id == dataset.dataset_id


def test_walk_forward_tune_rejects_invalid_n_windows() -> None:
    dataset = generate_demo_dataset(n_bars=200, n_symbols=1, seed=73)
    with pytest.raises(ValueError, match="n_windows"):
        walk_forward_tune(dataset, n_windows=1)
    with pytest.raises(ValueError, match="n_windows"):
        walk_forward_tune(dataset, n_windows=True)


def test_walk_forward_tune_cumulative_return_matches_window_product() -> None:
    dataset = generate_demo_dataset(n_bars=200, n_symbols=1, seed=77)
    result = walk_forward_tune(
        dataset,
        strategy_name="trend",
        n_windows=3,
        max_combinations=4,
    )
    import math

    expected_cumulative = (
        math.prod(
            1.0 + wr.optimized_report.total_return_pct / 100.0
            for wr in result.window_results
        )
        - 1.0
    ) * 100.0
    assert result.cumulative_return_pct == pytest.approx(expected_cumulative, abs=1e-6)
    assert result.profitable_windows == sum(
        1 for wr in result.window_results if wr.optimized_report.is_profitable
    )
