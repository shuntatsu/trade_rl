import json
from dataclasses import replace

import numpy as np
import pytest

from trade_rl.evaluation import bot


@pytest.mark.parametrize("mode", ["single", "all", "walk-forward"])
def test_tuning_keeps_the_selected_signal_for_every_replay(monkeypatch, mode):
    dataset = bot.generate_demo_dataset(n_bars=41, n_symbols=1)
    observed_indices = []
    real_run = bot.run_trading_bot

    def recording_run(dataset, config, **kwargs):
        observed_indices.append(config.signal_index)
        return real_run(dataset, config, **kwargs)

    monkeypatch.setattr(bot, "run_trading_bot", recording_run)
    if mode == "all":
        results = bot.tune_all_strategies(
            dataset, signal_index=2, max_combinations_per_strategy=1
        )
    elif mode == "walk-forward":
        results = bot.walk_forward_tune(
            dataset, signal_index=2, max_combinations=1
        ).window_results
    else:
        results = (
            bot.tune_for_maximum_profit(dataset, signal_index=2, max_combinations=1),
        )

    assert observed_indices and set(observed_indices) == {2}
    assert all(result.baseline_config.signal_index == 2 for result in results)
    assert all(result.optimized_config.signal_index == 2 for result in results)


@pytest.mark.parametrize("mode", ["run", "compare", "optimize", "walk-forward"])
def test_cli_resolves_a_named_signal_after_feature_reordering(
    monkeypatch, capsys, mode
):
    original = bot.generate_demo_dataset(n_bars=41, n_symbols=1)
    indices = (1, 2, 3, 4, 0)
    dataset = replace(
        original,
        features=original.features[:, :, indices],
        feature_available=original.feature_available[:, :, indices],
        feature_staleness=original.feature_staleness[:, :, indices],
        feature_staleness_hours=original.feature_staleness_hours[:, :, indices],
        feature_missing_reason=original.feature_missing_reason[:, :, indices],
        feature_names=tuple(original.feature_names[index] for index in indices),
    )
    monkeypatch.setattr(bot, "generate_demo_dataset", lambda: dataset)
    observed_indices = []
    real_run = bot.run_trading_bot

    def recording_run(dataset, config, **kwargs):
        observed_indices.append(config.signal_index)
        return real_run(dataset, config, **kwargs)

    monkeypatch.setattr(bot, "run_trading_bot", recording_run)
    assert (
        bot.main(
            [
                "--demo",
                "--mode",
                mode,
                "--strategy",
                "trend",
                "--signal-feature",
                "ema_crossover",
                "--max-combinations",
                "1",
                "--json",
            ]
        )
        == 0
    )

    assert observed_indices and set(observed_indices) == {4}
    payload = json.loads(capsys.readouterr().out)
    if mode == "optimize":
        assert payload["optimized_config"]["signal_index"] == 4
    elif mode == "walk-forward":
        assert payload["window_results"][0]["optimized_config"]["signal_index"] == 4


def test_cli_rejects_a_missing_signal_before_replay(monkeypatch, capsys):
    monkeypatch.setattr(
        bot,
        "run_trading_bot",
        lambda *args, **kwargs: pytest.fail("missing signal must fail before replay"),
    )
    with pytest.raises(SystemExit) as error:
        bot.main(["--demo", "--signal-feature", "missing_signal"])

    assert error.value.code == 2
    assert "signal feature not found" in capsys.readouterr().err


def test_reordered_named_signal_preserves_actual_orders_and_returns():
    original = bot.generate_demo_dataset(n_bars=41, n_symbols=1)
    features = original.features.copy()
    features[:, :, 0] = 0.03
    original = replace(original, features=features)
    indices = (1, 2, 3, 4, 0)
    reordered_features = original.features[:, :, indices].copy()
    reordered_features[:, :, 0] = -0.03
    reordered = replace(
        original,
        features=reordered_features,
        feature_available=original.feature_available[:, :, indices],
        feature_staleness=original.feature_staleness[:, :, indices],
        feature_staleness_hours=original.feature_staleness_hours[:, :, indices],
        feature_missing_reason=original.feature_missing_reason[:, :, indices],
        feature_names=tuple(original.feature_names[index] for index in indices),
    )
    config = bot.BotConfig(strategy_name="trend")
    expected, expected_report = bot.run_trading_bot(original, config)
    actual, actual_report = bot.run_trading_bot(
        reordered,
        replace(config, signal_index=reordered.feature_names.index("ema_crossover")),
    )

    assert actual_report.fill_count > 0
    assert actual_report == expected_report
    np.testing.assert_array_equal(actual.returns.values, expected.returns.values)
    assert actual.decisions == expected.decisions


@pytest.mark.parametrize("signal_index", [-1, True, 1.5, 5])
def test_tuning_rejects_invalid_signal_indices_before_replay(monkeypatch, signal_index):
    dataset = bot.generate_demo_dataset(n_bars=41, n_symbols=1)
    monkeypatch.setattr(
        bot,
        "run_trading_bot",
        lambda *args, **kwargs: pytest.fail("invalid signal must fail before replay"),
    )
    with pytest.raises(ValueError, match="signal_index"):
        bot.tune_for_maximum_profit(
            dataset, signal_index=signal_index, max_combinations=1
        )


@pytest.mark.parametrize("signal_index", [-1, True, 1.5, 5])
def test_run_trading_bot_rejects_invalid_signal_indices_before_replay(
    monkeypatch, signal_index
):
    dataset = bot.generate_demo_dataset(n_bars=41, n_symbols=1)
    monkeypatch.setattr(
        bot,
        "run_shared_cash_replay",
        lambda *args, **kwargs: pytest.fail("invalid signal must fail before replay"),
    )
    config = bot.BotConfig(strategy_name="trend", signal_index=signal_index)

    with pytest.raises(ValueError, match="signal_index"):
        bot.run_trading_bot(dataset, config)
