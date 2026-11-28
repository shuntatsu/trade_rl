"""Independent admission counterexamples for a declared native clock (#1357)."""

from fractions import Fraction

import numpy as np
import pytest

from tests.data.test_native_execution_dataset import native_values
from tests.data.test_native_timeframe_contract import _raw
from tests.integrations.test_binance_native_kline_clock import _row
from tests.integrations.test_binance_native_spot_capture import _encode, _message
from trade_rl.data.contracts import FeatureKind, FeatureSpec, InstrumentContract
from trade_rl.data.features.multitimeframe import align_clocked_native_feature
from trade_rl.data.market import MarketDataset
from trade_rl.data.native_clock import NativeBarClock, NativeClockKind
from trade_rl.data.native_resampling import resample_complete_native_bars
from trade_rl.evaluation.native_annualization import evaluate_native_bar_annualization
from trade_rl.integrations.binance.native_kline_clock import validate_native_kline_times
from trade_rl.integrations.binance.native_spot_capture import (
    resample_captured_spot_klines,
)
from trade_rl.risk.inputs import RollingPortfolioRiskInputsProvider

ANCHOR = np.datetime64("2026-01-05", "ns")


class DailySeconds(int):
    def __mul__(self, other):
        # The serialized integer remains 259200, but the bar lasts one day.
        return 86400 * other


class DailyClock(NativeBarClock):
    def interval_bounds(self, closes):
        return NativeBarClock(
            NativeClockKind.FIXED_ELAPSED, 86400, ANCHOR
        ).interval_bounds(closes)


def test_dataset_rejects_duration_that_disagrees_with_declared_frequency():
    values = native_values(3, weekly=False)
    closes = ANCHOR + np.arange(1, 7) * np.timedelta64(1, "D")
    with pytest.raises(ValueError, match="fixed_seconds"):
        clock = NativeBarClock(
            NativeClockKind.FIXED_ELAPSED, DailySeconds(259200), ANCHOR
        )
        values.update(
            native_clock=clock,
            timestamps=closes,
            available_at=closes[:, None].copy(),
        )
        MarketDataset(**values).with_content_identity()
    # Ordinary daily timestamps cannot be relabeled 3d either.
    values.update(
        native_clock=NativeBarClock(NativeClockKind.FIXED_ELAPSED, 259200, ANCHOR),
        timestamps=closes,
        available_at=closes[:, None].copy(),
    )
    with pytest.raises(ValueError):
        MarketDataset(**values)
    # Positive control uses the complete genuine 3d grid and exact frequency.
    valid = MarketDataset(**native_values(3, weekly=False)).with_content_identity()
    assert valid.identity_verified
    assert valid.bar_hours == 72
    assert valid.periods_per_year == Fraction(365, 3)


@pytest.mark.parametrize(
    "entrypoint",
    [
        "feature",
        "annualization",
        "risk",
        "resample_source",
        "resample_target",
        "kline",
        "capture_source",
        "capture_target",
    ],
)
def test_native_clock_consumers_refuse_overridden_interval_contract(entrypoint):
    foreign = DailyClock(NativeClockKind.FIXED_ELAPSED, 259200, ANCHOR)
    daily = NativeBarClock(NativeClockKind.FIXED_ELAPSED, 86400, ANCHOR)
    raw = _raw(1440, np.arange(1, 3))
    closes = raw.timestamps
    with pytest.raises(ValueError, match="NativeBarClock"):
        if entrypoint == "feature":
            align_clocked_native_feature(
                FeatureSpec("return", FeatureKind.LOG_RETURN),
                raw,
                InstrumentContract("SYNTHETIC"),
                closes,
                np.ones(len(closes), dtype=bool),
                clock=foreign,
            )
        elif entrypoint == "annualization":
            evaluate_native_bar_annualization(
                (0.01, -0.02), clock=foreign, closes=closes
            )
        elif entrypoint == "risk":
            RollingPortfolioRiskInputsProvider(native_clock=foreign)
        elif entrypoint.startswith("resample"):
            resample_complete_native_bars(
                source_clock=foreign if entrypoint == "resample_source" else daily,
                source_closes=closes,
                source_available_at=closes,
                source_ohlcv=np.tile([100, 103, 98, 101, 2.5, 250], (2, 1)),
                target_clock=foreign if entrypoint == "resample_target" else daily,
                target_closes=closes,
                as_of=closes[-1],
            )
        elif entrypoint == "kline":
            validate_native_kline_times(
                [_row("2026-01-05", "2026-01-06"), _row("2026-01-06", "2026-01-07")],
                clock=foreign,
                expected_closes=closes,
                timestamp_unit="ms",
                available_at=closes,
                as_of=closes[-1],
            )
        else:
            interval = "3d" if entrypoint == "capture_source" else "1d"
            captures = [
                (
                    _encode(
                        _message(
                            close - np.timedelta64(1, "D"), close, interval=interval
                        )
                    ),
                    close,
                )
                for close in closes
            ]
            resample_captured_spot_klines(
                captures=captures,
                symbol="TESTUSDT",
                source_interval=interval,
                source_clock=foreign if entrypoint == "capture_source" else daily,
                source_closes=closes,
                timestamp_unit="ms",
                target_clock=foreign if entrypoint == "capture_target" else daily,
                target_closes=closes,
                as_of=closes[-1],
            )
