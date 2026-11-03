"""The prequential adapter must bind every forecast packet to its requested asset and time."""

from __future__ import annotations

import math
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from trade_rl.evaluation.dev_realdata_study.runner import PrequentialSimpleReturnSource
from trade_rl.strategies.forecasts.prequential import ForecastPacket

WHEN = np.datetime64("2022-03-01T00:00:00", "ns")
SYMBOLS = ("S0", "S1", "S2")


def _packet(
    symbol_index: int = 1,
    *,
    horizon_hours: int = 24,
    point_forecast: float = 0.03,
) -> ForecastPacket:
    fit = WHEN - np.timedelta64(1, "h")
    return ForecastPacket(
        symbol=SYMBOLS[symbol_index],
        symbol_index=symbol_index,
        horizon_hours=horizon_hours,
        decision_time=WHEN,
        forecast_available_at=WHEN,
        return_unit="log",
        point_forecast=point_forecast,
        residual_variance=0.01,
        vintage_id="0" * 64,
        fit_time=fit,
        fit_cutoff=fit,
        max_training_label_available_at=fit - np.timedelta64(1, "h"),
        n_training_samples=1,
        fitter_identity="synthetic-adapter-test",
        training_digest="1" * 64,
        model_digest="2" * 64,
        feature_digest="3" * 64,
    )


class FakeSource:
    def __init__(self, packets: dict[int, object], horizon_hours: int = 24) -> None:
        self.symbols = SYMBOLS
        self.horizon_hours = horizon_hours
        self.packets = packets
        self.lookups: list[int] = []

    def packet(self, symbol_index: int, decision_time: np.datetime64) -> object:
        self.lookups.append(symbol_index)
        return self.packets.get(symbol_index)


def _snapshot() -> object:
    return SimpleNamespace(symbols=SYMBOLS, timestamp_ns=int(WHEN.astype(np.int64)))


@pytest.mark.parametrize(
    "bad_kind",
    ["wrong_symbol", "wrong_index", "stale_time", "wrong_horizon", "untyped"],
)
def test_mismatched_packet_fails_without_recording_partial_diagnostics(
    bad_kind: str,
) -> None:
    good = _packet()
    bad: object = {
        "wrong_symbol": replace(good, symbol="S0"),
        "wrong_index": replace(good, symbol_index=0),
        "stale_time": replace(good, decision_time=WHEN + np.timedelta64(1, "h")),
        "wrong_horizon": replace(good, horizon_hours=72),
        "untyped": SimpleNamespace(point_forecast=0.03, residual_variance=0.01),
    }[bad_kind]
    source = FakeSource({0: None, 1: bad, 2: _packet(2)})
    adapter = PrequentialSimpleReturnSource(source, decision_interval_hours=24)  # type: ignore[arg-type]

    with pytest.raises((TypeError, ValueError), match="packet"):
        adapter.expected_simple_returns(_snapshot())  # type: ignore[arg-type]
    assert source.lookups == [0, 1]
    assert adapter.calls == 0
    assert adapter.null_packets == 0
    assert adapter.history == []

    # An unsuccessful decision must not poison the subsequent valid decision.
    source.packets[1] = good
    values = adapter.expected_simple_returns(_snapshot())  # type: ignore[arg-type]
    expected = math.expm1(0.03 + 0.005)
    assert values == (0.0, pytest.approx(expected), pytest.approx(expected))
    assert adapter.calls == 1
    assert adapter.null_packets == 1
    assert adapter.history == [values]


def test_forecast_conversion_overflow_is_an_atomic_rejection() -> None:
    source = FakeSource({0: None, 1: _packet(point_forecast=1000.0)})
    adapter = PrequentialSimpleReturnSource(source, decision_interval_hours=24)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="forecast"):
        adapter.expected_simple_returns(_snapshot())  # type: ignore[arg-type]
    assert source.lookups == [0, 1]
    assert adapter.calls == 0
    assert adapter.null_packets == 0
    assert adapter.history == []


@pytest.mark.parametrize("interval", [0, -24, True, 24.0])
def test_decision_interval_must_be_a_positive_integer(interval: object) -> None:
    source = FakeSource({})
    with pytest.raises(ValueError, match="decision interval"):
        PrequentialSimpleReturnSource(source, decision_interval_hours=interval)  # type: ignore[arg-type]


@pytest.mark.parametrize("horizon, divisor", [(24, 1.0), (72, 3.0)])
def test_valid_packet_binding_keeps_forecast_conversion(
    horizon: int, divisor: float
) -> None:
    source = FakeSource({0: _packet(0, horizon_hours=horizon)}, horizon_hours=horizon)
    adapter = PrequentialSimpleReturnSource(source, decision_interval_hours=24)  # type: ignore[arg-type]
    values = adapter.expected_simple_returns(_snapshot())  # type: ignore[arg-type]
    assert values == (pytest.approx(math.expm1(0.035) / divisor), 0.0, 0.0)
    assert adapter.calls == 1
    assert adapter.null_packets == 2
