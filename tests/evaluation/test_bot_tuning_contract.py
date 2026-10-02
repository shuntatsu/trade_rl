import json
import subprocess
import sys
from dataclasses import replace

import numpy as np
import pytest

from trade_rl.evaluation import bot


def _report(config, pnl):
    return bot.BotReport(
        strategy_name=config.strategy_name,
        initial_capital=config.initial_capital,
        final_equity=config.initial_capital + pnl,
        net_pnl=pnl,
        total_return_pct=pnl / config.initial_capital * 100,
        max_drawdown_pct=5.0,
        nonzero_return_intervals=2,
        positive_return_rate_pct=50.0,
        interval_profit_factor=1.0,
        sharpe_ratio=1.0,
        is_profitable=pnl > 0,
        terminal_settled=True,
    )


@pytest.mark.parametrize("walk_forward", [False, True])
@pytest.mark.parametrize("baseline_pnl", [0.0, 100.0])
def test_balanced_selection_does_not_reward_losses(
    monkeypatch, walk_forward, baseline_pnl
):
    dataset = bot.generate_demo_dataset(n_bars=41, n_symbols=1)

    def fake_run(dataset, config, **kwargs):
        pnl = baseline_pnl if config.gross_budget == 0.2 else -1000.0
        return None, _report(config, pnl)

    monkeypatch.setattr(bot, "run_trading_bot", fake_run)
    if walk_forward:
        results = bot.walk_forward_tune(
            dataset, strategy_name="trend", objective="balanced", max_combinations=8
        ).window_results
    else:
        results = (
            bot.tune_for_maximum_profit(
                dataset, strategy_name="trend", objective="balanced", max_combinations=8
            ),
        )
    assert all(result.optimized_config.gross_budget == 0.2 for result in results)
    assert all(result.selection_score >= 0.0 for result in results)


@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_walk_forward_validates_budget_before_replay(monkeypatch, value):
    dataset = bot.generate_demo_dataset(n_bars=41, n_symbols=1)
    monkeypatch.setattr(
        bot,
        "run_trading_bot",
        lambda *args, **kwargs: pytest.fail("invalid budget must fail before replay"),
    )
    with pytest.raises(ValueError, match="max_combinations"):
        bot.walk_forward_tune(dataset, max_combinations=value)


def test_walk_forward_reuses_search_and_includes_last_interval(monkeypatch):
    dataset = bot.generate_demo_dataset(n_bars=41, n_symbols=1)
    observed = []

    def fake_run(dataset, config, **kwargs):
        observed.append((config, kwargs["start_index"], kwargs["stop_index"]))
        return None, _report(config, 1.0)

    monkeypatch.setattr(bot, "run_trading_bot", fake_run)
    single = bot.tune_for_maximum_profit(
        dataset, strategy_name="adaptive", max_combinations=8
    )
    single_candidates = [
        config
        for config, start, _ in observed
        if start == 0 and config.gross_budget != 0.2
    ]
    observed.clear()
    walk = bot.walk_forward_tune(
        dataset, strategy_name="adaptive", max_combinations=8, n_windows=3
    )
    walk_candidates = [
        config
        for config, start, _ in observed
        if start == 0 and config.gross_budget != 0.2
    ]
    assert walk_candidates == single_candidates
    assert walk.window_results[-1].holdout_stop_index == dataset.n_bars - 1
    assert (
        walk.window_results[0].holdout_stop_index
        == walk.window_results[1].holdout_start_index
    )
    assert single.evaluated_combinations == 8


def test_bot_reports_unfilled_terminal_close():
    dataset = bot.generate_demo_dataset(n_bars=40, n_symbols=1)
    volume = dataset.volume.copy()
    volume[-2:] = 0.0
    dataset = replace(dataset, volume=volume, identity_payload_json=None)
    result, report = bot.run_trading_bot(
        dataset, bot.BotConfig(strategy_name="constant_long")
    )
    assert result.book.quantities[0] > 0.0
    assert not report.terminal_settled
    assert report.terminal_position_quantities == tuple(result.book.quantities)


def test_walk_forward_does_not_count_unsettled_positive_mark_as_profitable(monkeypatch):
    dataset = bot.generate_demo_dataset(n_bars=41, n_symbols=1)

    def fake_fixed_windows(**kwargs):
        config = bot.BotConfig(
            strategy_name="constant_long",
            initial_capital=kwargs["initial_capital"],
            execution_cost=kwargs["execution_cost"],
        )
        unsettled_profit = replace(_report(config, 100.0), terminal_settled=False)
        return bot.TuningResult(
            strategy_name="constant_long",
            objective=kwargs["objective"],
            baseline_config=config,
            baseline_report=_report(config, 0.0),
            optimized_config=config,
            optimized_report=unsettled_profit,
            profit_improvement_pct=None,
            alpha_dollars=100.0,
            evaluated_combinations=1,
            selection_score=100.0,
            selection_drawdown_pct=5.0,
            tuning_start_index=kwargs["tune_start_index"],
            tuning_stop_index=kwargs["tune_stop_index"],
            holdout_start_index=kwargs["eval_start_index"],
            holdout_stop_index=kwargs["eval_stop_index"],
            dataset_id=dataset.dataset_id,
            dataset_identity_bound=dataset.identity_payload_json is not None,
            execution_cost=kwargs["execution_cost"],
        )

    monkeypatch.setattr(bot, "_tune_with_fixed_windows", fake_fixed_windows)

    result = bot.walk_forward_tune(
        dataset,
        strategy_name="constant_long",
        max_combinations=1,
        n_windows=2,
    )

    report = result.window_results[-1].optimized_report
    assert report.net_pnl > 0.0
    assert not report.terminal_settled
    assert result.profitable_windows == 0


def test_unsettled_tuning_candidate_is_ineligible(monkeypatch):
    dataset = bot.generate_demo_dataset(n_bars=41, n_symbols=1)

    def fake_run(dataset, config, **kwargs):
        report = _report(config, 1.0 if config.gross_budget == 0.2 else 1000.0)
        return None, replace(report, terminal_settled=config.gross_budget == 0.2)

    monkeypatch.setattr(bot, "run_trading_bot", fake_run)
    result = bot.tune_for_maximum_profit(
        dataset, strategy_name="trend", max_combinations=8
    )
    assert result.optimized_config.gross_budget == 0.2


def test_walk_forward_cli_emits_provenance_and_reset_scope(monkeypatch, capsys):
    dataset = bot.generate_demo_dataset(n_bars=41, n_symbols=1)
    monkeypatch.setattr(bot, "generate_demo_dataset", lambda: dataset)
    assert (
        bot.main(
            [
                "--mode",
                "walk-forward",
                "--strategy",
                "cash",
                "--demo",
                "--max-combinations",
                "1",
                "--windows",
                "3",
                "--json",
            ]
        )
        == 0
    )

    result = json.loads(capsys.readouterr().out)
    assert result["report_scope"] == "development_walk_forward"
    assert result["capital_mode"] == "reset_each_window"
    assert result["cumulative_return_scope"] == "hypothetical_normalized_product"
    assert result["dataset_id"] == dataset.dataset_id
    assert result["execution_cost"]["fee_rate"] > 0
    assert len(result["window_results"]) == 2
    assert all(
        window["report_scope"] == "development_walk_forward"
        for window in result["window_results"]
    )
    assert result["window_results"][-1]["holdout_stop_index"] == 40


def test_walk_forward_cli_text_explains_reset_capital(monkeypatch, capsys):
    dataset = bot.generate_demo_dataset(n_bars=41, n_symbols=1)
    monkeypatch.setattr(bot, "generate_demo_dataset", lambda: dataset)
    assert (
        bot.main(
            [
                "--mode",
                "walk-forward",
                "--strategy",
                "cash",
                "--demo",
                "--max-combinations",
                "1",
            ]
        )
        == 0
    )
    output = capsys.readouterr().out
    assert "independently reset" in output
    assert "Terminal settled" in output


def test_walk_forward_module_entry_point():
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "trade_rl.evaluation.bot",
            "--mode",
            "walk-forward",
            "--strategy",
            "cash",
            "--demo",
            "--max-combinations",
            "1",
            "--json",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    result = json.loads(completed.stdout)
    assert result["report_scope"] == "development_walk_forward"
    assert len(result["window_results"]) == 2
    assert all(
        window["optimized_report"]["terminal_settled"]
        for window in result["window_results"]
    )


@pytest.mark.parametrize("objective", ["profit", "sharpe", "balanced"])
@pytest.mark.parametrize("walk_forward", [False, True])
def test_loss_only_tuning_selects_cash_on_the_same_market(objective, walk_forward):
    dataset = bot.generate_demo_dataset(n_bars=41, n_symbols=1)
    prices = np.full_like(dataset.close, 100.0)
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
    if walk_forward:
        results = bot.walk_forward_tune(
            dataset,
            strategy_name="trend",
            objective=objective,
            max_combinations=8,
        ).window_results
    else:
        results = (
            bot.tune_for_maximum_profit(
                dataset,
                strategy_name="trend",
                objective=objective,
                max_combinations=8,
            ),
        )

    for result in results:
        assert result.strategy_name == "trend"
        assert result.baseline_report.net_pnl < 0.0
        assert result.optimized_config.strategy_name == "cash"
        assert result.optimized_report.strategy_name == "cash"
        assert result.optimized_report.net_pnl == 0.0
        assert result.optimized_report.terminal_settled
        assert result.selection_score == 0.0
        assert result.evaluated_combinations == 8
        _, direct_cash = bot.run_trading_bot(
            dataset,
            result.optimized_config,
            start_index=result.holdout_start_index,
            stop_index=result.holdout_stop_index,
        )
        assert result.optimized_report == direct_cash


def test_positive_tuning_candidate_is_not_replaced_after_a_losing_holdout(monkeypatch):
    dataset = bot.generate_demo_dataset(n_bars=41, n_symbols=1)

    def fake_run(dataset, config, **kwargs):
        if config.strategy_name == "cash":
            pnl = 0.0
        elif kwargs["start_index"] == 0:
            pnl = 10.0 if config.gross_budget == 0.2 else 100.0
        else:
            pnl = -1000.0
        return None, _report(config, pnl)

    monkeypatch.setattr(bot, "run_trading_bot", fake_run)
    result = bot.tune_for_maximum_profit(
        dataset, strategy_name="trend", max_combinations=8
    )
    assert result.optimized_config.strategy_name == "trend"
    assert result.selection_score == 100.0
    assert result.optimized_report.net_pnl == -1000.0


def test_cash_is_selected_when_every_trading_configuration_is_ineligible(monkeypatch):
    dataset = bot.generate_demo_dataset(n_bars=41, n_symbols=1)
    observed = []

    def fake_run(dataset, config, **kwargs):
        observed.append((config, kwargs["start_index"], kwargs["stop_index"]))
        is_cash = config.strategy_name == "cash"
        report = _report(config, 0.0 if is_cash else 1000.0)
        return None, replace(report, terminal_settled=is_cash)

    monkeypatch.setattr(bot, "run_trading_bot", fake_run)
    result = bot.tune_for_maximum_profit(
        dataset, strategy_name="trend", initial_capital=12345.0, max_combinations=8
    )
    assert result.optimized_config.strategy_name == "cash"
    assert not result.baseline_report.terminal_settled
    cash_tuning = [
        config
        for config, start, stop in observed
        if config.strategy_name == "cash"
        and start == result.tuning_start_index
        and stop == result.tuning_stop_index
    ]
    assert len(cash_tuning) == 1
    assert cash_tuning[0].initial_capital == 12345.0
    assert cash_tuning[0].execution_cost == result.execution_cost


@pytest.mark.parametrize("cash_pnl", [50.0, -50.0])
def test_cash_control_keeps_actual_economic_return(monkeypatch, cash_pnl):
    dataset = bot.generate_demo_dataset(n_bars=41, n_symbols=1)

    def fake_run(dataset, config, **kwargs):
        return None, _report(
            config, cash_pnl if config.strategy_name == "cash" else 10.0
        )

    monkeypatch.setattr(bot, "run_trading_bot", fake_run)
    result = bot.tune_for_maximum_profit(
        dataset, strategy_name="trend", max_combinations=1
    )
    if cash_pnl > 0.0:
        assert result.optimized_config.strategy_name == "cash"
        assert result.selection_score == cash_pnl
        assert result.optimized_report.net_pnl == cash_pnl
    else:
        assert result.optimized_config.strategy_name == "trend"
        assert result.selection_score == 10.0


def test_cash_control_does_not_fabricate_success_when_its_replay_is_invalid(
    monkeypatch,
):
    dataset = bot.generate_demo_dataset(n_bars=41, n_symbols=1)

    def fake_run(dataset, config, **kwargs):
        return None, replace(_report(config, 0.0), terminal_settled=False)

    monkeypatch.setattr(bot, "run_trading_bot", fake_run)
    with pytest.raises(ValueError, match="no configuration"):
        bot.tune_for_maximum_profit(dataset, strategy_name="trend", max_combinations=1)


def test_tuning_text_names_the_selected_cash_control(monkeypatch, capsys):
    dataset = bot.generate_demo_dataset(n_bars=41, n_symbols=1)

    def fake_run(dataset, config, **kwargs):
        return None, _report(config, 0.0 if config.strategy_name == "cash" else -10.0)

    monkeypatch.setattr(bot, "run_trading_bot", fake_run)
    result = bot.tune_for_maximum_profit(
        dataset, strategy_name="trend", max_combinations=1
    )
    bot.print_tuning_comparison(result)
    assert "Selected strategy: cash" in capsys.readouterr().out
