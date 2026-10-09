"""Native MTF Signature: causal alignment, independent math and failure oracles."""

from __future__ import annotations

from dataclasses import replace
from math import log
from pathlib import Path

import numpy as np
import pytest

from tests.evaluation.test_shared_cash_replay import _market
from trade_rl.data.artifacts.publication import (
    load_market_dataset_artifact,
    publish_market_dataset_artifact,
)
from trade_rl.data.contracts import InstrumentContract
from trade_rl.data.features.signature import (
    _segment_signature,
    with_path_signatures,
)
from trade_rl.data.features.signature_mtf import (
    SignatureClock,
    _native_signature_events,
    with_native_multitimeframe_signatures,
)
from trade_rl.data.source import RawMarketSeries


class _MultiSource:
    def __init__(self, values: dict[tuple[str, str], RawMarketSeries]) -> None:
        self.values = values
        self.calls: list[tuple[str, str]] = []

    def load(self, symbol: str) -> RawMarketSeries:
        return self.load_timeframe(symbol, "1h")

    def load_timeframe(self, symbol: str, timeframe: str) -> RawMarketSeries:
        self.calls.append((symbol, timeframe))
        return self.values[(symbol, timeframe)]


def _raw(
    close: np.ndarray,
    *,
    minutes: int = 15,
    times: np.ndarray | None = None,
    available_at: np.ndarray | None = None,
    volume: np.ndarray | None = None,
    tradable: np.ndarray | None = None,
) -> RawMarketSeries:
    prices = np.asarray(close, dtype=np.float64)
    n = len(prices)
    timestamps = (
        np.datetime64("2026-01-01T00:00:00", "ns")
        + np.arange(n) * np.timedelta64(minutes, "m")
        if times is None
        else times
    )
    return RawMarketSeries(
        timestamps=timestamps,
        open=prices,
        high=prices,
        low=prices,
        close=prices,
        volume=(
            np.full(n, 100.0) if volume is None else np.asarray(volume, dtype=np.float64)
        ),
        funding_rate=np.zeros(n),
        funding_available=np.zeros(n, dtype=np.bool_),
        tradable=(
            np.ones(n, dtype=np.bool_)
            if tradable is None
            else np.asarray(tradable, dtype=np.bool_)
        ),
        available_at=available_at,
    )


def _base(n: int = 6) -> object:
    return _market(np.full((n, 1), 100.0)).with_content_identity({"fixture": "base"})


def _get(data: object, word: str) -> np.ndarray:
    return data.features[:, 0, data.feature_names.index(word)]


def test_native_15m_path_keeps_excursion_lost_in_hourly_samples() -> None:
    base = _base()
    prices = np.full(21, 100.0)
    prices[3] = 110.0
    prices[7] = 110.0
    src = _MultiSource({("SYM0", "15m"): _raw(prices)})
    clock = SignatureClock("15m", window_bars=3, depth=2)
    result = with_native_multitimeframe_signatures(
        base, src, (InstrumentContract("SYM0"),), clocks=(clock,)
    )
    hour_only = with_path_signatures(base, window_bars=3, depth=2)
    tp = _get(result, "15m__path_sig_v1_w3_d2_tp_tp")
    assert result.feature_available[1, 0, -6:].all()
    np.testing.assert_allclose(tp[1], -log(1.1) / 2, atol=1e-7)
    np.testing.assert_allclose(tp[2], -log(1.1) / 2, atol=1e-7)
    assert not hour_only.feature_available[1, 0, -6:].any()
    assert hour_only.features[2, 0, -6:].sum() == 1.5
    assert result.identity_verified
    assert result.dataset_id != base.dataset_id
    assert src.calls == [("SYM0", "15m")]


def test_same_native_clock_agrees_with_existing_base_clock_signature() -> None:
    prices = np.array([[100.0], [107.0], [99.0], [102.0], [104.0]])
    base = _market(prices).with_content_identity({"fixture": "same-clock"})
    raw = _raw(prices[:, 0], minutes=60)
    clock = SignatureClock("1h", window_bars=3, depth=3)
    candidate = with_native_multitimeframe_signatures(
        base,
        _MultiSource({("SYM0", "1h"): raw}),
        (InstrumentContract("SYM0"),),
        clocks=(clock,),
    )
    expected = with_path_signatures(base, window_bars=3, depth=3)
    np.testing.assert_allclose(
        candidate.features[:, 0, -14:], expected.features[:, 0, -14:],
        rtol=1e-6,
        atol=1e-7,
    )
    np.testing.assert_array_equal(
        candidate.feature_available[:, 0, -14:],
        expected.feature_available[:, 0, -14:],
    )


def test_two_stack_chen_queue_matches_naive_rolling_reference() -> None:
    prices = 100.0 + np.cumsum(np.sin(np.arange(65) / 4.0))
    raw = _raw(prices)
    contract = InstrumentContract("SYM0")
    for window in (3, 7, 16):
        for depth in (1, 2, 3):
            clock = SignatureClock("15m", window_bars=window, depth=depth)
            rows, valid = _native_signature_events(raw, contract, clock)
            assert valid.sum() == 65 - window + 1
            for end in range(window - 1, len(prices)):
                increments = np.column_stack(
                    (
                        np.full(window - 1, 1.0 / (window - 1)),
                        np.diff(np.log(prices[end - window + 1 : end + 1])),
                    )
                )
                expected = _segment_signature(increments, depth)
                np.testing.assert_allclose(
                    rows[end], expected, rtol=2e-5, atol=2e-6
                )


def test_4h_signature_appears_only_after_third_completed_native_bar() -> None:
    base = _base(n=10)
    native = _raw(np.array([100.0, 110.0, 100.0]), minutes=240)
    c = SignatureClock("4h", window_bars=3, max_staleness_hours=2.0)
    result = with_native_multitimeframe_signatures(
        base,
        _MultiSource({("SYM0", "4h"): native}),
        (InstrumentContract("SYM0"),),
        clocks=(c,),
    )
    assert not result.feature_available[:8, 0, -6:].any()
    assert result.feature_available[8:10, 0, -6:].all()
    assert result.feature_staleness_hours[8, 0, -1] == 0
    assert result.feature_staleness_hours[9, 0, -1] == 1


def test_missing_native_interval_restarts_window_and_recovers() -> None:
    base = _base()
    values = np.arange(100.0, 121.0)
    complete = _raw(values)
    missing = np.arange(len(values)) != 3
    broken = _raw(values[missing], times=complete.timestamps[missing])
    clock = SignatureClock("15m", window_bars=3, max_staleness_hours=0.25)
    result = with_native_multitimeframe_signatures(
        base,
        _MultiSource({("SYM0", "15m"): broken}),
        (InstrumentContract("SYM0"),),
        clocks=(clock,),
    )
    assert not result.feature_available[1, 0, -6:].any()
    assert result.feature_available[2, 0, -6:].all()


def test_delayed_native_row_is_not_backfilled_into_earlier_decision() -> None:
    base = _base()
    original = _raw(np.arange(100.0, 121.0))
    avail = original.timestamps.copy()
    avail[3] = original.timestamps[4]
    late = replace(original, available_at=avail)
    clock = SignatureClock("15m", window_bars=3, max_staleness_hours=0.25)
    result = with_native_multitimeframe_signatures(
        base,
        _MultiSource({("SYM0", "15m"): late}),
        (InstrumentContract("SYM0"),),
        clocks=(clock,),
    )
    assert not result.feature_available[1, 0, -6:].any()
    assert result.feature_available[2, 0, -6:].all()
    assert (result.features[1, 0, -6:] == 0).all()


def test_future_native_changes_leave_past_features_and_masks_unchanged() -> None:
    base = _base()
    raw = _raw(np.arange(100.0, 121.0))
    future = raw.close.copy()
    future[14:] *= 1.25
    changed = _raw(future)
    clock = SignatureClock("15m", window_bars=3)
    first = with_native_multitimeframe_signatures(
        base,
        _MultiSource({("SYM0", "15m"): raw}),
        (InstrumentContract("SYM0"),),
        clocks=(clock,),
    )
    second = with_native_multitimeframe_signatures(
        base,
        _MultiSource({("SYM0", "15m"): changed}),
        (InstrumentContract("SYM0"),),
        clocks=(clock,),
    )
    np.testing.assert_array_equal(first.features[:4], second.features[:4])
    np.testing.assert_array_equal(
        first.feature_available[:4], second.feature_available[:4]
    )
    assert first.dataset_id != second.dataset_id


def test_volume_channel_with_zero_volume_recovers_only_after_full_window() -> None:
    base = _base()
    vol = np.full(21, 100.0)
    vol[3] = 0.0
    raw = _raw(np.arange(100.0, 121.0), volume=vol)
    with_vol = with_native_multitimeframe_signatures(
        base,
        _MultiSource({("SYM0", "15m"): raw}),
        (InstrumentContract("SYM0"),),
        clocks=(
            SignatureClock(
                "15m",
                window_bars=3,
                include_volume=True,
                max_staleness_hours=0.25,
            ),
        ),
    )
    plain = with_native_multitimeframe_signatures(
        base,
        _MultiSource({("SYM0", "15m"): raw}),
        (InstrumentContract("SYM0"),),
        clocks=(SignatureClock("15m", window_bars=3),),
    )
    assert not with_vol.feature_available[1, 0, -12:].all()
    assert plain.feature_available[1, 0, -6:].all()
    assert with_vol.feature_available[2, 0, -12:].all()


def test_out_of_scope_native_future_is_rejected() -> None:
    base = _base(n=6)
    raw = _raw(np.arange(100.0, 123.0))
    with pytest.raises(ValueError, match="extends beyond Dataset time scope"):
        with_native_multitimeframe_signatures(
            base,
            _MultiSource({("SYM0", "15m"): raw}),
            (InstrumentContract("SYM0"),),
            clocks=(SignatureClock("15m", window_bars=3),),
        )


def test_multiple_native_clocks_are_independent_and_preserve_economics(
    tmp_path: Path,
) -> None:
    base = _base(n=10)
    raw_15 = _raw(np.arange(100.0, 137.0))
    raw_4h = _raw(np.array([100.0, 105.0, 99.0]), minutes=240)
    source = _MultiSource(
        {("SYM0", "15m"): raw_15, ("SYM0", "4h"): raw_4h}
    )
    clocks = (
        SignatureClock("15m", window_bars=3),
        SignatureClock("4h", window_bars=3),
    )
    a = with_native_multitimeframe_signatures(
        base, source, (InstrumentContract("SYM0"),), clocks=clocks
    )
    b = with_native_multitimeframe_signatures(
        base, source, (InstrumentContract("SYM0"),), clocks=clocks
    )
    assert a.dataset_id == b.dataset_id
    assert a.feature_names[-12] == "15m__path_sig_v1_w3_d2_tp_t"
    assert a.feature_names[-6] == "4h__path_sig_v1_w3_d2_tp_t"
    np.testing.assert_array_equal(a.close, base.close)
    np.testing.assert_array_equal(a.funding_rate, base.funding_rate)
    np.testing.assert_array_equal(a.fee_rate, base.fee_rate)
    np.testing.assert_array_equal(a.features[:, :, : base.n_features], base.features)
    restored_path = publish_market_dataset_artifact(tmp_path / "native", a).root
    restored = load_market_dataset_artifact(restored_path)
    assert restored.dataset_id == a.dataset_id
    np.testing.assert_array_equal(restored.features, a.features)


def test_native_source_cannot_be_mismatched_or_unverified() -> None:
    raw = _raw(np.arange(100.0, 121.0))
    source = _MultiSource({("SYM0", "15m"): raw})
    clock = SignatureClock("15m", window_bars=3)
    with pytest.raises(ValueError, match="identity-verified"):
        with_native_multitimeframe_signatures(
            _market(np.full((6, 1), 100.0)),
            source,
            (InstrumentContract("SYM0"),),
            clocks=(clock,),
        )
    with pytest.raises(ValueError, match="symbol order"):
        with_native_multitimeframe_signatures(
            _base(), source, (InstrumentContract("WRONG"),), clocks=(clock,)
        )
    with pytest.raises(ValueError, match="feature names"):
        with_native_multitimeframe_signatures(
            _base(), source, (InstrumentContract("SYM0"),), clocks=(clock, clock)
        )


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"timeframe": "5m"}, "unsupported timeframe"),
        ({"timeframe": "15m", "window_bars": 2}, "window_bars"),
        ({"timeframe": "15m", "window_bars": True}, "window_bars"),
        ({"timeframe": "15m", "depth": 4}, "depth"),
        ({"timeframe": "15m", "depth": True}, "depth"),
        ({"timeframe": "15m", "include_volume": 1}, "include_volume"),
        ({"timeframe": "15m", "max_staleness_hours": 0}, "max_staleness_hours"),
        ({"timeframe": "15m", "max_staleness_hours": True}, "max_staleness_hours"),
    ],
)
def test_native_clock_contract_rejects_bad_spec(
    kwargs: dict[str, object], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        SignatureClock(**kwargs)


def test_native_signed_area_matches_hand_computed_two_segment_oracle() -> None:
    raw = _raw(np.array([100.0, 110.0, 100.0]))
    result, available = _native_signature_events(
        raw, InstrumentContract("SYM0"), SignatureClock("15m", window_bars=3)
    )
    assert available.tolist() == [False, False, True]
    first = np.array([0.5, log(1.1)])
    second = np.array([0.5, -log(1.1)])
    expected = (
        0.5 * np.outer(first, first)
        + np.outer(first, second)
        + 0.5 * np.outer(second, second)
    )
    np.testing.assert_allclose(result[2, 2:6], expected.reshape(-1), atol=1e-7)
