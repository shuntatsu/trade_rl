from dataclasses import replace

import numpy as np
import pytest

from tests.data.test_weekly_context import _weekly_market
from trade_rl.data.features.weekly_context import WEEKLY_NAMES, with_weekly_context
from trade_rl.evaluation.bot import (
    BotConfig,
    create_strategy_instances,
    run_trading_bot,
)


def test_weekly_bot_named_binding_survives_column_permutation():
    source = _weekly_market()
    signals = source.features.copy()
    signals[:, :, 0] = 0.03
    data = with_weekly_context(replace(source, features=signals))
    config = BotConfig(
        strategy_name="weekly_bb_ichimoku",
        signal_index=0,
        volatility_regime_threshold=0,
        minimum_hold_bars=24,
    )
    order = [7, 4, 1, 0, 5, 2, 6, 3]
    permuted = replace(
        data,
        identity_payload_json=None,
        features=data.features[:, :, order],
        feature_names=tuple(data.feature_names[i] for i in order),
        feature_available=data.feature_available[:, :, order],
        feature_staleness=data.feature_staleness[:, :, order],
        feature_staleness_hours=data.feature_staleness_hours[:, :, order],
        feature_missing_reason=data.feature_missing_reason[:, :, order],
    )
    start, stop = 78 * 168 - 1, 79 * 168 + 12
    original, report = run_trading_bot(data, config, start_index=start, stop_index=stop)
    other, second = run_trading_bot(
        permuted, replace(config, signal_index=3), start_index=start, stop_index=stop
    )
    auto, auto_report = run_trading_bot(
        replace(source, features=signals), config, start_index=start, stop_index=stop
    )
    np.testing.assert_array_equal(original.returns.values, other.returns.values)
    np.testing.assert_array_equal(original.returns.values, auto.returns.values)
    assert report == second == auto_report
    assert report.terminal_settled and report.fill_count > 0


def test_incomplete_weekly_names_reject_before_replay():
    data = with_weekly_context(_weekly_market())
    names = list(data.feature_names)
    names[-1] = "wrong_cloud"
    with pytest.raises(ValueError, match="complete named weekly"):
        create_strategy_instances(
            replace(data, feature_names=tuple(names), identity_payload_json=None),
            BotConfig(strategy_name="weekly_bb_ichimoku"),
        )
    assert tuple(data.feature_names[-7:]) == WEEKLY_NAMES
