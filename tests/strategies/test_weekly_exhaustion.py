"""Ordered hourly observations, with hand-specified native four-hour events."""

from dataclasses import replace
from importlib import import_module
from importlib.util import find_spec

import numpy as np
import pytest

from trade_rl.strategies.controls import ConstantIntentStrategy
from trade_rl.strategies.interface import StrategyObservation
from trade_rl.strategies.position_intent import PositionIntent
from trade_rl.strategies.rules.adaptive import (
    AdaptiveProfitConfig,
    RegimeAdaptiveStrategy,
)


def _strategy(base=None, indices=(0, 1), short_term_index=2):
    module_name = "trade_rl.strategies.rules.weekly_exhaustion"
    assert find_spec(module_name) is not None, (
        "Missing planned FormingWeekExhaustionStrategy API: ordered forming-week "
        "band reach followed by a later fresh four-hour Tenkan crossing"
    )
    strategy_type = getattr(
        import_module(module_name), "FormingWeekExhaustionStrategy", None
    )
    assert callable(strategy_type), "Missing planned FormingWeekExhaustionStrategy"
    return strategy_type(
        base if base is not None else ConstantIntentStrategy(PositionIntent.LONG),
        indices,
        short_term_index=short_term_index,
    )


def _observation(hour, *, side=1, reach=False, tenkan=None, **changes):
    high, low = (1.0, 0.0) if reach and side == 1 else (0.0, -1.0) if reach else (0, 0)
    values = np.asarray(
        (high, low, side * 0.1 if tenkan is None else tenkan, side * 0.03),
        dtype=np.float32,
    )
    observation = StrategyObservation(
        index=hour,
        timestamp=np.datetime64("2024-01-01T00:00", "ns") + np.timedelta64(hour, "h"),
        symbol="ANY",
        features=values,
        feature_available=np.ones(4, dtype=bool),
        feature_staleness=np.asarray((0.0, 0.0, (hour % 4) / 4, 0.0)),
        global_features=np.zeros(1),
        global_feature_available=np.ones(1, dtype=bool),
        current_intent=PositionIntent.FLAT,
        current_weight=0.0,
    )
    return replace(observation, **changes)


def _intent(side):
    return PositionIntent.LONG if side == 1 else PositionIntent.SHORT


def _armed_strategy(side=1, *, prior_tenkan=None):
    strategy = _strategy(ConstantIntentStrategy(_intent(side)))
    assert strategy.decide(_observation(4, side=side, tenkan=prior_tenkan)) is _intent(
        side
    )
    for hour in range(5, 8):
        assert strategy.decide(
            _observation(hour, side=side, reach=hour == 5)
        ) is _intent(side)
    return strategy


@pytest.mark.parametrize("side", [-1, 1])
def test_first_later_native_cross_confirms_reach_and_recovery_releases_block(side):
    strategy = _armed_strategy(side)
    assert (
        strategy.decide(_observation(8, side=side, tenkan=-side * 0.1))
        is PositionIntent.FLAT
    )
    for hour in range(9, 12):
        assert (
            strategy.decide(_observation(hour, side=side, tenkan=-side * 0.1))
            is PositionIntent.FLAT
        )
    assert strategy.decide(_observation(12, side=side, tenkan=0.0)) is _intent(side)


@pytest.mark.parametrize("side", [-1, 1])
def test_first_later_native_non_cross_consumes_reach(side):
    strategy = _armed_strategy(side)
    for hour in range(8, 12):
        assert strategy.decide(
            _observation(hour, side=side, tenkan=side * 0.2)
        ) is _intent(side)
    assert strategy.decide(_observation(12, side=side, tenkan=-side * 0.1)) is _intent(
        side
    )


@pytest.mark.parametrize("side", [-1, 1])
def test_same_time_reach_and_cross_cannot_confirm(side):
    strategy = _strategy(ConstantIntentStrategy(_intent(side)))
    for hour in range(4, 8):
        assert strategy.decide(_observation(hour, side=side)) is _intent(side)
    assert strategy.decide(
        _observation(8, side=side, reach=True, tenkan=-side * 0.1)
    ) is _intent(side)
    for hour in range(9, 13):
        assert strategy.decide(
            _observation(hour, side=side, tenkan=-side * 0.2)
        ) is _intent(side)


@pytest.mark.parametrize("side", [-1, 1])
def test_cross_before_reach_cannot_confirm(side):
    strategy = _armed_strategy(side, prior_tenkan=-side * 0.1)
    assert strategy.decide(_observation(8, side=side, tenkan=-side * 0.2)) is _intent(
        side
    )


@pytest.mark.parametrize("side", [-1, 1])
def test_only_new_native_event_can_confirm_not_carried_opposite_distance(side):
    strategy = _strategy(ConstantIntentStrategy(_intent(side)))
    assert strategy.decide(_observation(4, side=side)) is _intent(side)
    assert strategy.decide(_observation(5, side=side, reach=True)) is _intent(side)
    for hour in (6, 7):
        assert strategy.decide(
            _observation(hour, side=side, tenkan=-side * 0.1)
        ) is _intent(side)
    assert (
        strategy.decide(_observation(8, side=side, tenkan=-side * 0.1))
        is PositionIntent.FLAT
    )


@pytest.mark.parametrize("side", [-1, 1])
@pytest.mark.parametrize(
    "prior,current,expected_block", [(0.0, -0.1, True), (0.1, 0.0, False)]
)
def test_native_cross_uses_inclusive_prior_and_strict_current_sign(
    side, prior, current, expected_block
):
    strategy = _armed_strategy(side, prior_tenkan=side * prior)
    actual = strategy.decide(_observation(8, side=side, tenkan=side * current))
    assert actual is (PositionIntent.FLAT if expected_block else _intent(side))


@pytest.mark.parametrize(
    "fault", ["no_staleness", "stale_boundary", "fresh_nonboundary"]
)
def test_native_confirmation_requires_explicit_freshness_and_utc_boundary(fault):
    strategy = _armed_strategy()
    observation = _observation(8, tenkan=-0.1)
    if fault == "no_staleness":
        observation = replace(observation, feature_staleness=None)
    elif fault == "stale_boundary":
        observation = replace(
            observation, feature_staleness=np.asarray((0, 0, 0.25, 0))
        )
    else:
        # Nonboundary freshness cannot manufacture a native four-hour event.
        observation = replace(_observation(8), feature_staleness=np.zeros(4))
        assert strategy.decide(observation) is PositionIntent.LONG
        observation = replace(
            _observation(9, tenkan=-0.1), feature_staleness=np.zeros(4)
        )
    expected = (
        PositionIntent.LONG if fault == "fresh_nonboundary" else PositionIntent.FLAT
    )
    assert strategy.decide(observation) is expected


@pytest.mark.parametrize("side", [-1, 1])
def test_band_reach_is_inclusive_without_a_near_touch_tolerance(side):
    strategy = _strategy(ConstantIntentStrategy(_intent(side)))
    for hour in range(4, 8):
        observation = _observation(hour, side=side)
        values = observation.features.copy()
        if hour == 5:
            values[0 if side == 1 else 1] = np.nextafter(
                np.float32(side), np.float32(0.0)
            )
        assert strategy.decide(replace(observation, features=values)) is _intent(side)
    assert strategy.decide(_observation(8, side=side, tenkan=-side * 0.1)) is _intent(
        side
    )


@pytest.mark.parametrize("side", [-1, 1])
def test_nonzero_native_distance_has_no_magnitude_tolerance(side):
    strategy = _armed_strategy(side, prior_tenkan=0.0)
    tiny_opposite = -side * np.nextafter(np.float32(0.0), np.float32(1.0))
    assert (
        strategy.decide(_observation(8, side=side, tenkan=tiny_opposite))
        is PositionIntent.FLAT
    )


@pytest.mark.parametrize("fault", ["unavailable", "nonfinite", "stale"])
def test_missing_expected_native_event_discards_reach_before_next_event(fault):
    strategy = _armed_strategy()
    observation = _observation(8)
    values = observation.features.copy()
    available = observation.feature_available.copy()
    staleness = observation.feature_staleness.copy()
    if fault == "unavailable":
        available[2] = False
        staleness[2] = 1.0
    elif fault == "nonfinite":
        values[2] = np.nan
    else:
        staleness[2] = np.nextafter(np.float32(0.0), np.float32(1.0))
    # The failed boundary must not leave an arm that can be confirmed four hours late.
    strategy.decide(
        replace(
            observation,
            features=values,
            feature_available=available,
            feature_staleness=staleness,
        )
    )
    for hour in range(9, 12):
        strategy.decide(_observation(hour))
    assert strategy.decide(_observation(12, tenkan=-0.1)) is PositionIntent.LONG


@pytest.mark.parametrize("fault", ["masked_high", "masked_low", "nan_high", "inf_low"])
def test_invalid_forming_context_clears_armed_reach(fault):
    strategy = _armed_strategy()
    observation = _observation(8, tenkan=-0.1)
    values = observation.features.copy()
    available = observation.feature_available.copy()
    staleness = observation.feature_staleness.copy()
    if fault.startswith("masked"):
        index = 0 if fault.endswith("high") else 1
        available[index] = False
        staleness[index] = 1.0
    else:
        values[0 if fault == "nan_high" else 1] = (
            np.nan if fault == "nan_high" else np.inf
        )
    assert (
        strategy.decide(
            replace(
                observation,
                features=values,
                feature_available=available,
                feature_staleness=staleness,
            )
        )
        is PositionIntent.FLAT
    )
    for hour in range(9, 13):
        assert strategy.decide(_observation(hour, tenkan=-0.1)) is PositionIntent.LONG


@pytest.mark.parametrize("index", [0, 1])
def test_carried_forming_context_is_invalid_and_clears_armed_reach(index):
    strategy = _armed_strategy()
    observation = _observation(8, tenkan=-0.1)
    staleness = observation.feature_staleness.copy()
    staleness[index] = np.nextafter(np.float32(0), np.float32(1))
    assert (
        strategy.decide(replace(observation, feature_staleness=staleness))
        is PositionIntent.FLAT
    )
    for hour in range(9, 13):
        assert strategy.decide(_observation(hour, tenkan=-0.1)) is PositionIntent.LONG


def test_flat_base_still_advances_filter_state_before_later_direction_request():
    class DelayedDirection:
        def decide(self, observation):
            return PositionIntent.FLAT if observation.index < 8 else PositionIntent.LONG

    strategy = _strategy(DelayedDirection())
    for hour in range(4, 8):
        assert (
            strategy.decide(_observation(hour, reach=hour == 5)) is PositionIntent.FLAT
        )
    assert strategy.decide(_observation(8, tenkan=-0.1)) is PositionIntent.FLAT


def test_protective_exit_latch_survives_invalid_forming_context_and_recovery():
    strategy = _strategy(
        RegimeAdaptiveStrategy(
            AdaptiveProfitConfig(
                signal_index=3,
                volatility_index=3,
                volatility_regime_threshold=0,
                stop_loss_threshold=0.01,
            )
        )
    )
    missing = np.asarray((False, False, False, True))
    observation = _observation(
        4,
        feature_available=missing,
        feature_staleness=np.asarray((1, 1, 1, 0)),
        current_position_quantity=1.0,
        position_age_bars=1,
        gross_position_return=-0.02,
    )
    assert strategy.decide(observation) is PositionIntent.FLAT
    assert strategy.protective_exit_pending
    assert (
        strategy.decide(
            replace(
                _observation(5),
                current_position_quantity=1.0,
                position_age_bars=2,
                gross_position_return=0.02,
            )
        )
        is PositionIntent.FLAT
    )
    assert strategy.protective_exit_pending


@pytest.mark.parametrize("side", [-1, 1])
def test_new_reach_after_consumed_opportunity_can_arm_again(side):
    strategy = _armed_strategy(side)
    for hour in range(8, 12):
        assert strategy.decide(
            _observation(hour, side=side, reach=hour == 9)
        ) is _intent(side)
    assert (
        strategy.decide(_observation(12, side=side, tenkan=-side * 0.1))
        is PositionIntent.FLAT
    )


@pytest.mark.parametrize("side", [-1, 1])
def test_monday_midnight_finishes_old_week_then_first_hour_resets_state(side):
    strategy = _strategy(ConstantIntentStrategy(_intent(side)))
    for hour in range(164, 168):
        assert strategy.decide(
            _observation(hour, side=side, reach=hour == 165)
        ) is _intent(side)
    # Monday 00:00 is the close of Sunday's final hour, not a new-week reset.
    assert (
        strategy.decide(_observation(168, side=side, tenkan=-side * 0.1))
        is PositionIntent.FLAT
    )
    assert strategy.decide(_observation(169, side=side, tenkan=-side * 0.1)) is _intent(
        side
    )


@pytest.mark.parametrize("gap", ["timestamp", "index"])
def test_hourly_observation_gap_cannot_confirm_across_unseen_history(gap):
    strategy = _armed_strategy()
    if gap == "timestamp":
        next_observation = _observation(9, tenkan=-0.1)
    else:
        next_observation = replace(_observation(8, tenkan=-0.1), index=10)
    assert strategy.decide(next_observation) is PositionIntent.LONG


def test_confirmed_long_block_does_not_create_short_block_or_reverse_base_intent():
    class LaterShort:
        def decide(self, observation):
            return (
                PositionIntent.SHORT if observation.index >= 9 else PositionIntent.LONG
            )

    strategy = _strategy(LaterShort())
    for hour in range(4, 8):
        assert (
            strategy.decide(_observation(hour, reach=hour == 5)) is PositionIntent.LONG
        )
    assert strategy.decide(_observation(8, tenkan=-0.1)) is PositionIntent.FLAT
    assert strategy.decide(_observation(9, tenkan=-0.1)) is PositionIntent.SHORT


def test_prefix_replay_reconstructs_armed_and_blocked_state_without_checkpoint_api():
    observations = [
        _observation(
            hour, reach=hour in (5, 13), tenkan=-0.1 if 8 <= hour < 12 else 0.1
        )
        for hour in range(4, 17)
    ]
    continuous = _strategy()
    expected = [continuous.decide(observation) for observation in observations]
    for stop in (3, 5, 9):
        restarted = _strategy()
        for observation in observations[:stop]:
            restarted.decide(observation)
        actual = [restarted.decide(observation) for observation in observations[stop:]]
        assert actual == expected[stop:]
    assert expected[4] is PositionIntent.FLAT
    assert expected[8] is PositionIntent.LONG


def test_named_feature_permutation_preserves_ordered_filter():
    order = np.asarray((2, 1, 3, 0))
    ordinary = _strategy()
    permuted = _strategy(indices=(3, 1), short_term_index=0)
    for hour in range(4, 13):
        observation = _observation(
            hour, reach=hour == 5, tenkan=-0.1 if hour >= 8 else 0.1
        )
        shuffled = replace(
            observation,
            features=observation.features[order],
            feature_available=observation.feature_available[order],
            feature_staleness=observation.feature_staleness[order],
        )
        assert permuted.decide(shuffled) is ordinary.decide(observation)


@pytest.mark.parametrize(
    "indices", [(0,), (0, 1, 2), (0, 0), (-1, 1), (True, 1), (0, 1.0), ("0", 1)]
)
def test_forming_indices_require_two_distinct_nonnegative_integers(indices):
    with pytest.raises(ValueError):
        _strategy(indices=indices)


@pytest.mark.parametrize("index", [True, -1, 1.0, "2", 0, 1, None])
def test_short_term_index_requires_distinct_nonnegative_integer(index):
    with pytest.raises(ValueError):
        _strategy(short_term_index=index)


@pytest.mark.parametrize("indices,index", [((4, 1), 2), ((0, 4), 2), ((0, 1), 4)])
def test_out_of_range_indices_rejected_even_while_base_is_flat(indices, index):
    strategy = _strategy(ConstantIntentStrategy(PositionIntent.FLAT), indices, index)
    with pytest.raises(ValueError, match="out of range"):
        strategy.decide(_observation(4))


@pytest.mark.parametrize("side", [-1, 1])
@pytest.mark.parametrize(
    "fault",
    [
        "missing_forming",
        "nonfinite_forming",
        "missing_native",
        "nan_native",
        "inf_native",
        "no_freshness",
        "stale_boundary",
    ],
)
def test_invalid_required_context_clears_existing_block_and_reseeds_native_memory(
    side, fault
):
    strategy = _armed_strategy(side)
    for hour in range(8, 12):
        assert (
            strategy.decide(_observation(hour, side=side, tenkan=-side * 0.1))
            is PositionIntent.FLAT
        )
    observation = _observation(12, side=side, tenkan=-side * 0.1)
    values = observation.features.copy()
    available = observation.feature_available.copy()
    staleness = observation.feature_staleness.copy()
    if fault in ("missing_forming", "missing_native"):
        index = 0 if fault == "missing_forming" else 2
        available[index] = False
        staleness[index] = 1.0
    elif fault == "nonfinite_forming":
        values[1] = np.nan
    elif fault in ("nan_native", "inf_native"):
        values[2] = np.nan if fault == "nan_native" else np.inf
    elif fault == "stale_boundary":
        staleness[2] = 0.25
    assert (
        strategy.decide(
            replace(
                observation,
                features=values,
                feature_available=available,
                feature_staleness=None if fault == "no_freshness" else staleness,
            )
        )
        is PositionIntent.FLAT
    )
    for hour in range(13, 16):
        # Clearing a block is observable before the next fresh native event.
        assert strategy.decide(
            _observation(hour, side=side, tenkan=-side * 0.1)
        ) is _intent(side)
    assert strategy.decide(_observation(16, side=side, tenkan=side * 0.1)) is _intent(
        side
    )
    for hour in range(17, 20):
        assert strategy.decide(
            _observation(hour, side=side, reach=hour == 17)
        ) is _intent(side)
    assert (
        strategy.decide(_observation(20, side=side, tenkan=-side * 0.1))
        is PositionIntent.FLAT
    )


@pytest.mark.parametrize("fault", ["missing", "nonfinite", "no_freshness"])
def test_invalid_native_channel_between_boundaries_fails_closed_and_clears_block(fault):
    strategy = _armed_strategy()
    assert strategy.decide(_observation(8, tenkan=-0.1)) is PositionIntent.FLAT
    observation = _observation(9, tenkan=-0.1)
    values = observation.features.copy()
    available = observation.feature_available.copy()
    staleness = observation.feature_staleness.copy()
    if fault == "missing":
        available[2] = False
        staleness[2] = 1.0
    elif fault == "nonfinite":
        values[2] = np.nan
    assert (
        strategy.decide(
            replace(
                observation,
                features=values,
                feature_available=available,
                feature_staleness=None if fault == "no_freshness" else staleness,
            )
        )
        is PositionIntent.FLAT
    )
    assert strategy.decide(_observation(10, tenkan=-0.1)) is PositionIntent.LONG


@pytest.mark.parametrize(
    "fault",
    [
        "duplicate_index",
        "backward_index",
        "duplicate_timestamp",
        "backward_timestamp",
        "nat",
        "symbol",
    ],
)
def test_rejected_observation_does_not_call_base_or_mutate_armed_state(fault):
    class CountingBase:
        def __init__(self):
            self.calls = []

        def decide(self, observation):
            self.calls.append(observation)
            return PositionIntent.LONG

    base = CountingBase()
    strategy = _strategy(base)
    for hour in range(4, 8):
        assert (
            strategy.decide(_observation(hour, reach=hour == 5)) is PositionIntent.LONG
        )
    changes = {
        "duplicate_index": {"index": 7},
        "backward_index": {"index": 6},
        "duplicate_timestamp": {"timestamp": _observation(7).timestamp},
        "backward_timestamp": {"timestamp": _observation(6).timestamp},
        "nat": {"timestamp": np.datetime64("NaT", "ns")},
        "symbol": {"symbol": "OTHER"},
    }[fault]
    with pytest.raises(ValueError):
        strategy.decide(replace(_observation(8, tenkan=-0.1), **changes))
    assert len(base.calls) == 4
    assert strategy.decide(_observation(8, tenkan=-0.1)) is PositionIntent.FLAT
    assert len(base.calls) == 5


@pytest.mark.parametrize("gap", ["timestamp", "index"])
def test_forward_gap_clears_existing_block(gap):
    strategy = _armed_strategy()
    assert strategy.decide(_observation(8, tenkan=-0.1)) is PositionIntent.FLAT
    observation = (
        _observation(10, tenkan=-0.1)
        if gap == "timestamp"
        else replace(_observation(9, tenkan=-0.1), index=11)
    )
    assert strategy.decide(observation) is PositionIntent.LONG


def test_new_week_discards_previous_native_value_before_first_fresh_event():
    strategy = _strategy()
    for hour in range(168, 172):
        assert (
            strategy.decide(_observation(hour, reach=hour == 170))
            is PositionIntent.LONG
        )
    # Monday 00:00's positive native distance belongs to the prior week. The
    # first event after Monday 01:00 can seed a value but cannot form a crossing.
    assert strategy.decide(_observation(172, tenkan=-0.1)) is PositionIntent.LONG


def test_every_accepted_observation_calls_base_even_when_context_is_invalid():
    class CountingBase:
        def __init__(self):
            self.observations = []

        def decide(self, observation):
            self.observations.append(observation)
            return PositionIntent.LONG

    base = CountingBase()
    strategy = _strategy(base)
    observations = [
        _observation(4),
        _observation(5, feature_staleness=None),
        _observation(6),
    ]
    expected = [PositionIntent.LONG, PositionIntent.FLAT, PositionIntent.LONG]
    assert [strategy.decide(observation) for observation in observations] == expected
    assert all(
        actual is expected
        for actual, expected in zip(base.observations, observations, strict=True)
    )
