from dataclasses import replace

import numpy as np
import pytest

from tests.data.test_weekly_context import _weekly_market
from trade_rl.data.features.weekly_context import with_weekly_context
from trade_rl.evaluation.bot import (
    BotConfig,
    create_strategy_instances,
    run_trading_bot,
)


def _source():
    original = _weekly_market()
    shape = (original.n_bars, original.n_symbols, 2)
    return replace(
        original,
        identity_payload_json=None,
        feature_names=(original.feature_names[0], "4h__ichimoku_tenkan_distance_9bar"),
        features=np.broadcast_to((0.03, 0.02), shape).astype(np.float32).copy(),
        feature_available=np.ones(shape, dtype=bool),
        feature_staleness=np.zeros(shape),
        feature_staleness_hours=np.zeros(shape),
        feature_missing_reason=np.zeros(shape, dtype=np.uint8),
    )


def test_weekly_pullback_named_inputs_and_auto_context_preserve_replay():
    source = _source()
    dataset = with_weekly_context(source)
    config = BotConfig(
        strategy_name="weekly_bb_pullback",
        signal_index=0,
        volatility_regime_threshold=0,
    )
    order = [8, 3, 6, 1, 0, 2, 4, 5, 7]
    permuted = replace(
        dataset,
        identity_payload_json=None,
        feature_names=tuple(dataset.feature_names[i] for i in order),
        features=dataset.features[:, :, order],
        feature_available=dataset.feature_available[:, :, order],
        feature_staleness=dataset.feature_staleness[:, :, order],
        feature_staleness_hours=dataset.feature_staleness_hours[:, :, order],
        feature_missing_reason=dataset.feature_missing_reason[:, :, order],
    )
    start, stop = 78 * 168 - 1, 79 * 168 + 12
    original, report = run_trading_bot(
        dataset, config, start_index=start, stop_index=stop
    )
    auto, auto_report = run_trading_bot(
        source, config, start_index=start, stop_index=stop
    )
    other, other_report = run_trading_bot(
        permuted, replace(config, signal_index=4), start_index=start, stop_index=stop
    )
    np.testing.assert_array_equal(original.returns.values, auto.returns.values)
    np.testing.assert_array_equal(original.returns.values, other.returns.values)
    assert report == auto_report == other_report
    assert report.terminal_settled and report.fill_count > 0
    strategy = create_strategy_instances(dataset, config)[0]
    assert strategy.short_term_index == 1


def test_weekly_pullback_rejects_missing_named_four_hour_tenkan():
    dataset = with_weekly_context(_weekly_market())
    with pytest.raises(ValueError, match="4h__ichimoku_tenkan_distance_9bar"):
        create_strategy_instances(
            dataset, BotConfig(strategy_name="weekly_bb_pullback")
        )
    ordinary = create_strategy_instances(
        dataset, BotConfig(strategy_name="weekly_bb_ichimoku")
    )
    assert ordinary[0].short_term_index is None


def test_weekly_pullback_exit_is_voluntary_until_filled_quantity_hold_unlocks():
    dataset = with_weekly_context(_source())
    start, stop = 78 * 168 - 1, 78 * 168 + 40
    values = dataset.features.copy()
    values[:, :, dataset.feature_names.index("weekly_bb_high_position")] = 1.2
    values[start + 4 :, :, 1] = -0.02
    dataset = replace(dataset, features=values, identity_payload_json=None)
    result, report = run_trading_bot(
        dataset,
        BotConfig(
            strategy_name="weekly_bb_pullback",
            signal_index=0,
            volatility_regime_threshold=0,
            minimum_hold_bars=24,
        ),
        start_index=start,
        stop_index=stop,
    )
    assert result.ledger_evidence is not None
    decisions = result.ledger_evidence.decisions
    held = [
        d for d in decisions if d.minimum_hold_suppressed[0] and d.index >= start + 4
    ]
    assert held and all(d.intents[0] == 0 and d.effective_intents[0] == 1 for d in held)
    assert all(
        d.position_quantity_after[0] == d.position_quantity_before[0] for d in held
    )
    unlocked = [d for d in decisions if d.minimum_hold_unlocked[0]]
    assert len(unlocked) == 1 and unlocked[0].position_age_bars_before[0] == 24
    assert unlocked[0].position_quantity_after[0] == 0
    assert report.terminal_settled and report.fill_count == 2
