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


@pytest.mark.parametrize("candidate_settlement", [False, None])
def test_unsettled_tuning_candidate_is_ineligible(monkeypatch, candidate_settlement):
    dataset = bot.generate_demo_dataset(n_bars=41, n_symbols=1)

    def fake_run(dataset, config, **kwargs):
        report = _report(config, 1.0 if config.gross_budget == 0.2 else 1000.0)
        settlement = True if config.gross_budget == 0.2 else candidate_settlement
        return None, replace(report, terminal_settled=settlement)

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
        and stop == result.tuning_stop_index - 1
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


def test_cash_control_losing_more_than_candidate_does_not_win(monkeypatch):
    dataset = bot.generate_demo_dataset(n_bars=41, n_symbols=1)

    def fake_run(dataset, config, **kwargs):
        pnl = -50.0 if config.strategy_name == "cash" else -10.0
        return None, _report(config, pnl)

    monkeypatch.setattr(bot, "run_trading_bot", fake_run)
    result = bot.tune_for_maximum_profit(
        dataset, strategy_name="trend", max_combinations=1
    )

    assert result.optimized_config.strategy_name == "trend"
    assert result.selection_score == -10.0
    assert result.optimized_report.net_pnl == -10.0


@pytest.mark.parametrize("settlement", [False, None])
def test_cash_control_does_not_fabricate_success_when_its_replay_is_invalid(
    monkeypatch, settlement
):
    dataset = bot.generate_demo_dataset(n_bars=41, n_symbols=1)

    def fake_run(dataset, config, **kwargs):
        return None, replace(_report(config, 0.0), terminal_settled=settlement)

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


@pytest.mark.parametrize(
    ("holdout_settlement", "warning"),
    [(False, "Incomplete settlement"), (None, "Settlement status is unknown")],
)
def test_tuning_text_reports_unsettled_or_unknown_status_accurately(
    monkeypatch, capsys, holdout_settlement, warning
):
    dataset = bot.generate_demo_dataset(n_bars=41, n_symbols=1)

    def fake_run(_dataset, config, *, start_index=0, **_kwargs):
        report = replace(
            _report(config, 0.0),
            terminal_settled=(True if start_index == 0 else holdout_settlement),
        )
        return None, report

    monkeypatch.setattr(bot, "run_trading_bot", fake_run)
    result = bot.tune_for_maximum_profit(
        dataset, strategy_name="trend", max_combinations=1
    )

    bot.print_tuning_comparison(result)
    output = capsys.readouterr().out
    assert "Terminal settled" in output
    assert warning in output
    assert (
        "Incomplete settlement: equity includes residual marked inventory" not in output
    )


def test_optimize_all_text_includes_execution_ledger_diagnostics(monkeypatch, capsys):
    dataset = bot.generate_demo_dataset(n_bars=12, n_symbols=1)
    families = (
        "adaptive",
        "ensemble",
        "trend",
        "mean_reversion",
        "channel_breakout",
    )
    results = []
    for index, family in enumerate(families, start=1):
        baseline_config = bot.BotConfig(strategy_name=family, initial_capital=1000.0)
        optimized_config = replace(baseline_config, gross_budget=0.3)
        order_id = f"{index:x}" * 64
        baseline_report = replace(
            _report(baseline_config, -20.0),
            terminal_settled=None if index == 1 else False,
            terminal_position_quantities=()
            if index == 1
            else (index / 100.0, -index / 10.0),
            active_order_remainders=()
            if index == 1
            else (
                (order_id, index / 1000.0),
                (f"{index + 5:x}" * 64, index / 500.0),
            ),
            termination_reason=None if index == 1 else "drawdown_stop",
            total_execution_cost=10.0 + index,
            funding_pnl=-index - 0.25,
            borrow_cost=index + 0.5,
            turnover_total=index + 0.75,
            fill_count=10 + index,
            rebalance_events=None if index == 1 else 5 + index,
        )
        optimized_report = replace(
            _report(optimized_config, 30.0),
            terminal_settled=True,
            terminal_position_quantities=(0.0, 0.0),
            total_execution_cost=20.0 + index,
            funding_pnl=index + 0.25,
            borrow_cost=index + 0.75,
            turnover_total=index + 1.25,
            fill_count=None if index == 1 else 20 + index,
            rebalance_events=15 + index,
        )
        results.append(
            bot.TuningResult(
                strategy_name=family,
                objective="profit",
                baseline_config=baseline_config,
                baseline_report=baseline_report,
                optimized_config=optimized_config,
                optimized_report=optimized_report,
                profit_improvement_pct=250.0,
                alpha_dollars=50.0,
                evaluated_combinations=1,
                selection_score=30.0,
                selection_drawdown_pct=5.0,
                tuning_start_index=0,
                tuning_stop_index=9,
                holdout_start_index=9,
                holdout_stop_index=11,
                dataset_id="0" * 64,
                dataset_identity_bound=True,
                execution_cost=bot.ExecutionCostConfig(),
                report_scope="development_family_comparison",
            )
        )

    monkeypatch.setattr(bot, "generate_demo_dataset", lambda: dataset)
    monkeypatch.setattr(bot, "tune_all_strategies", lambda *_args, **_kwargs: results)

    assert bot.main(["--mode", "optimize", "--strategy", "all", "--demo"]) == 0

    output = capsys.readouterr().out
    blocks = output.split("PARAMETER TUNING REPORT:")[1:]
    assert len(blocks) == len(families)

    def cells(rows, label):
        return [cell.strip() for cell in rows[label]]

    def monetary_pair(rows, label):
        return [
            float(cell.replace("$", "").replace(",", "")) for cell in cells(rows, label)
        ]

    for index, (family, block) in enumerate(
        zip(families, blocks, strict=True), start=1
    ):
        assert block.startswith(f" {family.upper()} ")
        rows = {
            line.split("|", maxsplit=1)[0].strip(): line.split("|")[1:]
            for line in block.splitlines()
            if "|" in line
        }

        assert monetary_pair(rows, "Execution cost (account currency)") == [
            10.0 + index,
            20.0 + index,
        ]
        assert monetary_pair(rows, "Funding P&L (account currency)") == [
            -index - 0.25,
            index + 0.25,
        ]
        assert monetary_pair(rows, "Borrow cost (account currency)") == [
            index + 0.5,
            index + 0.75,
        ]
        assert [float(cell) for cell in cells(rows, "Turnover total")] == [
            index + 0.75,
            index + 1.25,
        ]
        if index == 1:
            assert cells(rows, "Fills / rebalances") == [
                "11 / unavailable",
                "unavailable / 16",
            ]
            assert cells(rows, "Terminal quantities (dataset symbol order)") == [
                "unavailable",
                "[0, 0]",
            ]
            assert cells(rows, "Active order remainders") == ["unavailable", "none"]
            assert cells(rows, "Termination reason") == ["unavailable", "none"]
        else:
            assert [
                tuple(int(value.strip()) for value in cell.split("/"))
                for cell in cells(rows, "Fills / rebalances")
            ] == [(10 + index, 5 + index), (20 + index, 15 + index)]
            assert cells(rows, "Terminal quantities (dataset symbol order)") == [
                f"[{index / 100.0:.6g}, {-index / 10.0:.6g}]",
                "[0, 0]",
            ]
            assert cells(rows, "Active order remainders") == [
                f"{f'{index:x}' * 64}={index / 1000.0:.6g}, "
                f"{f'{index + 5:x}' * 64}={index / 500.0:.6g}",
                "none",
            ]
            assert cells(rows, "Termination reason") == ["drawdown_stop", "none"]
