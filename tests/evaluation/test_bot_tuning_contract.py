import json
import subprocess
import sys
from dataclasses import replace

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
