"""Phase 2's Ridge daily cache must reject a foreign forecast packet, not alias symbols."""

from __future__ import annotations

import math
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from trade_rl.evaluation.dev_realdata_study.trial_sources import RidgeDailyComponent
from trade_rl.strategies.forecasts.prequential import ForecastPacket

WHEN = np.datetime64("2022-03-01T00:00:00", "ns")
SYMBOLS = ("S0", "S1")


def _packet(symbol_index: int = 1, horizon_hours: int = 24) -> ForecastPacket:
    fit = WHEN - np.timedelta64(1, "h")
    return ForecastPacket(
        symbol=SYMBOLS[symbol_index],
        symbol_index=symbol_index,
        horizon_hours=horizon_hours,
        decision_time=WHEN,
        forecast_available_at=WHEN,
        return_unit="log",
        point_forecast=0.025,
        residual_variance=0.0,
        vintage_id="0" * 64,
        fit_time=fit,
        fit_cutoff=fit,
        max_training_label_available_at=fit - np.timedelta64(1, "h"),
        n_training_samples=1,
        fitter_identity="synthetic-phase2-packet-test",
        training_digest="1" * 64,
        model_digest="2" * 64,
        feature_digest="3" * 64,
    )


class FakeSource:
    def __init__(
        self,
        packets: dict[int, object],
        *,
        symbols: tuple[str, ...] = SYMBOLS,
        horizon_hours: int = 24,
    ) -> None:
        self.packets = packets
        self.symbols = symbols
        self.horizon_hours = horizon_hours
        self.schedule = SimpleNamespace(first_fit_time=WHEN, activation_delay_hours=0)
        self.lookups: list[int] = []

    def packet(self, index: int, when: np.datetime64) -> object:
        self.lookups.append(index)
        assert when == WHEN
        return self.packets.get(index)


def _panel() -> object:
    return SimpleNamespace(
        symbols=SYMBOLS, timestamps_ns=np.array([int(WHEN.astype(np.int64))])
    )


def test_phase2_ridge_source_roster_must_match_panel_before_packet_lookup() -> None:
    source = FakeSource({0: _packet(0)}, symbols=("S1", "S0"))
    with pytest.raises(ValueError, match="symbol roster"):
        RidgeDailyComponent(source, _panel(), divisor=1.0)  # type: ignore[arg-type]
    assert source.lookups == []


@pytest.mark.parametrize(
    "bad",
    ["wrong_symbol", "wrong_index", "stale_time", "wrong_horizon", "untyped"],
)
def test_phase2_ridge_cache_rejects_misbound_packets_atomically(bad: str) -> None:
    good = _packet(1)
    changed: object = {
        "wrong_symbol": replace(good, symbol="S0"),
        "wrong_index": replace(good, symbol_index=0),
        "stale_time": replace(good, decision_time=WHEN + np.timedelta64(1, "h")),
        "wrong_horizon": replace(good, horizon_hours=72),
        "untyped": SimpleNamespace(point_forecast=0.025, residual_variance=0.0),
    }[bad]
    source = FakeSource({0: None, 1: changed})
    component = RidgeDailyComponent(source, _panel(), divisor=1.0)  # type: ignore[arg-type]
    with pytest.raises((TypeError, ValueError), match="packet"):
        component.value_at(0)
    assert source.lookups == [0, 1]
    assert component._cache == {}

    source.packets[1] = good
    values = component.value_at(0)
    assert values is not None
    assert np.isnan(values[0])
    assert values[1] == pytest.approx(math.expm1(0.025))
    assert source.lookups == [0, 1, 0, 1]
    assert component.value_at(0) is values  # valid day is cached only after success
    assert source.lookups == [0, 1, 0, 1]


def test_phase2_ridge_cache_rejects_overflow_without_caching() -> None:
    source = FakeSource({0: replace(_packet(0), point_forecast=1000.0)})
    component = RidgeDailyComponent(source, _panel(), divisor=1.0)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="forecast"):
        component.value_at(0)
    assert component._cache == {}


@pytest.mark.parametrize("horizon,divisor", [(24, 1.0), (72, 3.0)])
def test_phase2_ridge_keeps_valid_horizon_conversion(
    horizon: int, divisor: float
) -> None:
    source = FakeSource({0: _packet(0, horizon)}, horizon_hours=horizon)
    component = RidgeDailyComponent(source, _panel(), divisor=divisor)  # type: ignore[arg-type]
    values = component.value_at(0)
    assert values is not None
    assert values[0] == pytest.approx(math.expm1(0.025) / divisor)
    assert np.isnan(values[1])
