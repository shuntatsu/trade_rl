from mars_lite.serving.feature_provider import required_history_bars


def test_history_includes_slow_lookback_and_rebalance_offset() -> None:
    history = required_history_bars(
        rank_window=40,
        vol_lookback=48,
        trend_config={
            "fast_lookback": 24,
            "base_lookback": 48,
            "slow_lookback": 96,
            "rebalance_every": 24,
        },
    )

    assert history == 120


def test_non_residual_history_preserves_existing_requirements() -> None:
    assert required_history_bars(250, 60, {}) == 250
