"""Synthetic bot wiring and native-clock contracts, without market economics."""

import json
from dataclasses import replace

import numpy as np
import pytest

from tests.data.test_forming_week_context import _market
from trade_rl.data.contracts import FeatureAlignment, FeatureKind, FeatureSpec
from trade_rl.data.features.forming_week_context import with_forming_week_context
from trade_rl.evaluation.bot import (
    BotConfig,
    create_strategy_instances,
    run_trading_bot,
)
from trade_rl.strategies.rules.weekly_exhaustion import FormingWeekExhaustionStrategy

_NATIVE = "4h__ichimoku_tenkan_distance_9bar"


def _source():
    original = _market(symbols=2)
    shape = (original.n_bars, original.n_symbols, 2)
    hours = original.timestamps.astype(np.int64) // 3_600_000_000_000
    age = np.broadcast_to((hours % 4)[:, None], shape[:2])
    ages = np.stack((np.zeros_like(age), age), axis=2).astype(float)
    spec = FeatureSpec(
        _NATIVE,
        FeatureKind.ICHIMOKU_TENKAN_DISTANCE,
        lookback=9,
        timeframe="4h",
        alignment=FeatureAlignment.UNSHIFTED_DECISION_TIME,
        max_staleness_hours=8,
    )
    source = replace(
        original,
        identity_payload_json=None,
        feature_names=("signal", _NATIVE),
        features=np.broadcast_to((0.03, 0.02), shape).astype(np.float32).copy(),
        feature_available=np.ones(shape, dtype=bool),
        feature_staleness_hours=ages,
        feature_staleness=(ages / 8).astype(np.float32),
        feature_missing_reason=np.zeros(shape, dtype=np.int8),
        volume=np.full_like(original.volume, 1_000_000_000.0),
    )
    return source.with_content_identity(
        {"config": {"base_timeframe": "1h", "features": [spec.canonical_payload()]}}
    )


def _config():
    return BotConfig(
        strategy_name="forming_week_bb_ichimoku",
        signal_index=0,
        minimum_hold_bars=24,
        volatility_regime_threshold=0,
    )


def _rebind(source, **changes):
    payload = json.loads(source.identity_payload_json)
    return replace(source, identity_payload_json=None, **changes).with_content_identity(
        payload
    )


def test_opt_in_builds_one_independent_filter_per_symbol_and_preserves_base_config():
    source = with_forming_week_context(_source())
    strategies = create_strategy_instances(source, _config())
    assert len(strategies) == 2
    assert all(
        isinstance(strategy, FormingWeekExhaustionStrategy) for strategy in strategies
    )
    assert strategies[0] is not strategies[1]
    assert strategies[0].strategy is not strategies[1].strategy
    assert strategies[0].short_term_index == 1
    assert strategies[0].feature_indices == (2, 3)
    assert strategies[0].strategy.config.signal_index == 0
    assert strategies[0].strategy.config.volatility_regime_threshold == 0


def test_auto_and_named_permuted_context_replay_have_identical_evidence():
    source = _source()
    augmented = with_forming_week_context(source)
    order = [3, 1, 0, 2]
    permuted = _rebind(
        augmented,
        feature_names=tuple(augmented.feature_names[index] for index in order),
        features=augmented.features[:, :, order],
        feature_available=augmented.feature_available[:, :, order],
        feature_staleness=augmented.feature_staleness[:, :, order],
        feature_staleness_hours=augmented.feature_staleness_hours[:, :, order],
        feature_missing_reason=augmented.feature_missing_reason[:, :, order],
    )
    start, stop = 19 * 168, 19 * 168 + 40
    ordinary, report = run_trading_bot(
        augmented, _config(), start_index=start, stop_index=stop
    )
    automatic, automatic_report = run_trading_bot(
        source, _config(), start_index=start, stop_index=stop
    )
    shuffled, shuffled_report = run_trading_bot(
        permuted, replace(_config(), signal_index=2), start_index=start, stop_index=stop
    )
    np.testing.assert_array_equal(ordinary.returns.values, automatic.returns.values)
    np.testing.assert_array_equal(ordinary.returns.values, shuffled.returns.values)
    assert report == automatic_report == shuffled_report
    assert report.terminal_settled
    assert ordinary.ledger_evidence.decisions == automatic.ledger_evidence.decisions


@pytest.mark.parametrize("unit", ["s", "ms", "us"])
def test_datetime_storage_unit_preserves_native_validation_and_replay(unit):
    source = _source()
    changed = _rebind(
        source, timestamps=source.timestamps.astype(f"datetime64[{unit}]")
    )
    start, stop = 19 * 168, 19 * 168 + 40
    original, report = run_trading_bot(
        source, _config(), start_index=start, stop_index=stop
    )
    converted, converted_report = run_trading_bot(
        changed, _config(), start_index=start, stop_index=stop
    )
    np.testing.assert_array_equal(original.returns.values, converted.returns.values)
    assert converted_report == report
    assert original.ledger_evidence.decisions == converted.ledger_evidence.decisions


@pytest.mark.parametrize(
    "field,value",
    [
        ("kind", "log_return"),
        ("lookback", 8),
        ("lookback", True),
        ("timeframe", "1h"),
        ("normalization", "rolling_zscore"),
        ("alignment", "plotted_cloud"),
        ("max_staleness_hours", 0),
    ],
)
def test_named_native_channel_requires_identity_bound_raw_tenkan_spec(field, value):
    source = _source()
    payload = json.loads(source.identity_payload_json)
    payload["config"]["features"][0][field] = value
    source = replace(source, identity_payload_json=None).with_content_identity(payload)
    with pytest.raises(ValueError, match="native"):
        run_trading_bot(source, _config())


@pytest.mark.parametrize(
    "fault",
    [
        "missing_identity",
        "name_only",
        "default_age",
        "wrong_age",
        "wrong_normalized_age",
        "changed_carry",
    ],
)
def test_native_metadata_failures_are_rejected_before_replay(fault):
    source = _source()
    if fault == "missing_identity":
        source = replace(source, identity_payload_json=None)
    elif fault == "name_only":
        source = replace(source, identity_payload_json=None).with_content_identity(
            {"name": _NATIVE}
        )
    elif fault == "default_age":
        source = _rebind(source, feature_staleness_hours=None, feature_staleness=None)
    elif fault == "wrong_age":
        source = _rebind(
            source, feature_staleness_hours=source.feature_staleness_hours + 0.5
        )
    elif fault == "wrong_normalized_age":
        source = _rebind(
            source, feature_staleness=np.zeros_like(source.feature_staleness)
        )
    else:
        values = source.features.copy()
        values[1, :, 1] = -0.02
        source = _rebind(source, features=values)
    with pytest.raises(ValueError, match="native"):
        run_trading_bot(source, _config())


def test_partial_forming_feature_set_is_rejected_without_augmenting():
    source = with_forming_week_context(_source())
    partial = _rebind(source, feature_names=(*source.feature_names[:3], "unrelated"))
    with pytest.raises(ValueError, match="forming"):
        run_trading_bot(partial, _config())


def test_positive_raw_age_cannot_underflow_into_fresh_native_metadata():
    source = _source()
    payload = json.loads(source.identity_payload_json)
    payload["config"]["features"][0]["max_staleness_hours"] = 1e50
    source = replace(
        source,
        identity_payload_json=None,
        feature_staleness=np.zeros_like(source.feature_staleness),
    ).with_content_identity(payload)
    with pytest.raises(ValueError, match="native"):
        run_trading_bot(source, _config())


@pytest.mark.parametrize("maximum", [7.3, 8.1])
def test_native_normalized_age_matches_builders_float64_division(maximum):
    source = _source()
    payload = json.loads(source.identity_payload_json)
    payload["config"]["features"][0]["max_staleness_hours"] = maximum
    source = replace(
        source,
        identity_payload_json=None,
        feature_staleness=(
            source.feature_staleness_hours.astype(np.float64) / maximum
        ).astype(np.float32),
    ).with_content_identity(payload)
    assert (
        len(create_strategy_instances(with_forming_week_context(source), _config()))
        == 2
    )


def test_ordinary_forming_veto_preserves_actual_quantity_until_hold24_unlocks():
    source = _source()
    start, stop = 19 * 168, 19 * 168 + 36
    prices = np.broadcast_to(
        (100 + np.arange(source.n_bars) // 168)[:, None], source.close.shape
    ).astype(float)
    values = source.features.copy()
    values[start + 7 :, :, 1] = -0.02  # A new Monday08 native event.
    source = _rebind(
        source,
        open=prices,
        close=prices,
        high=2 * prices,
        low=prices,
        mark_price=prices,
        index_price=prices,
        features=values,
    )
    result, report = run_trading_bot(
        source, _config(), start_index=start, stop_index=stop
    )
    assert result.ledger_evidence is not None
    held = [
        decision
        for decision in result.ledger_evidence.decisions
        if decision.index >= start + 7 and any(decision.minimum_hold_suppressed)
    ]
    assert held and held[0].index == start + 7
    assert all(decision.intents == (0, 0) for decision in held)
    assert all(decision.effective_intents == (1, 1) for decision in held)
    assert all(
        decision.position_quantity_after == decision.position_quantity_before
        for decision in held
    )
    unlocked = [
        decision
        for decision in result.ledger_evidence.decisions
        if any(decision.minimum_hold_unlocked)
    ]
    assert len(unlocked) == 1
    assert unlocked[0].position_age_bars_before == (24, 24)
    assert unlocked[0].position_quantity_after == (0, 0)
    assert report.terminal_settled and report.fill_count == 4
