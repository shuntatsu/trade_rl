"""``AnchorInputs.from_market_dataset`` over the synthetic Tier D fake-source dataset."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from tests.evaluation.test_core7_mtf_tierd import END, START, FakeRows, _rules
from trade_rl.evaluation.core7_mtf_study import protocol, tierd
from trade_rl.integrations.market_store.carry_source import load_carry_pair
from trade_rl.strategies.agy_anchor.inputs import (
    from_market_dataset,
    funding_observations,
)
from trade_rl.strategies.agy_anchor.signals import build_anchor_signals

HOUR_MS = 3_600_000


def _build() -> tuple[object, np.ndarray]:
    guarded = tierd.GuardedRowSource(FakeRows())
    pairs = {
        symbol: load_carry_pair(
            guarded, symbol, start_time=START, end_time=END, halt_intervals=()
        )
        for symbol in protocol.ROSTER
    }
    dataset = tierd.assemble_tierd_dataset(pairs, rules=_rules())
    # the dataset carries no taker-buy column; the fake rows publish half of the quote
    # turnover as taker-buy, standing in for the externally supplied (T, legs) array
    return dataset, 0.5 * np.asarray(dataset.volume)


def test_the_tierd_dataset_maps_onto_the_anchor_bundle() -> None:
    dataset, taker = _build()
    bundle = from_market_dataset(dataset, taker_buy_quote_volume=taker)  # type: ignore[arg-type]

    assert bundle.symbols == protocol.ROSTER
    assert bundle.n_rows == 36
    np.testing.assert_array_equal(bundle.perp_close, dataset.close[:, 1::2])  # type: ignore[attr-defined]
    np.testing.assert_array_equal(bundle.spot_close, dataset.close[:, 0::2])  # type: ignore[attr-defined]
    np.testing.assert_array_equal(bundle.perp_taker_buy_quote_volume, taker[:, 1::2])
    np.testing.assert_array_equal(bundle.spot_taker_buy_quote_volume, taker[:, 0::2])
    assert bundle.perp_valid.all() and bundle.spot_valid.all()


def test_row_derived_funding_events_equal_the_exact_events_on_the_bar_grid() -> None:
    dataset, taker = _build()
    derived = from_market_dataset(dataset, taker_buy_quote_volume=taker)  # type: ignore[arg-type]
    # FakeRows publishes one 0.0001 event every 8h, exactly on the 4h grid
    first = int(derived.close_ns[0] // 1_000_000)
    first_event = first - (first % (8 * HOUR_MS)) + 8 * HOUR_MS
    stamps = np.arange(
        first_event, int(derived.close_ns[-1] // 1_000_000) + 1, 8 * HOUR_MS
    )
    exact = tuple(stamps.astype(np.int64) * 1_000_000 for _ in protocol.ROSTER)
    rates = tuple(np.full(stamps.shape, 0.0001) for _ in protocol.ROSTER)
    explicit = from_market_dataset(
        dataset,  # type: ignore[arg-type]
        taker_buy_quote_volume=taker,
        funding_events=tuple(zip(exact, rates, strict=True)),
    )
    for a, b in zip(
        funding_observations(derived), funding_observations(explicit), strict=True
    ):
        np.testing.assert_array_equal(a, b)


class _TwoEventsInOneBar:
    """Delegates to a real dataset but reports two funding events in one 4h bar."""

    def __init__(self, inner: Any) -> None:
        self._inner = inner

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def resolved_array(self, name: str) -> np.ndarray:
        array = np.asarray(self._inner.resolved_array(name)).copy()
        if name == "funding_event_count":
            array[3, 1] = 2
        return array


def test_a_bar_with_two_funding_events_requires_the_exact_events() -> None:
    dataset, taker = _build()
    with pytest.raises(ValueError, match="two or more funding events"):
        from_market_dataset(
            _TwoEventsInOneBar(dataset),  # type: ignore[arg-type]
            taker_buy_quote_volume=taker,
        )
    with pytest.raises(ValueError, match="taker_buy_quote_volume must match"):
        from_market_dataset(dataset, taker_buy_quote_volume=taker[:, :4])  # type: ignore[arg-type]


def test_the_anchor_builds_finite_causal_outputs_from_the_tierd_dataset() -> None:
    dataset, taker = _build()
    bundle = from_market_dataset(dataset, taker_buy_quote_volume=taker)  # type: ignore[arg-type]
    signals = build_anchor_signals(bundle)
    for name in ("w_aw_bidir_75_25", "c_weekly_iv", "vol_cushion_tail"):
        assert np.isfinite(getattr(signals, name)).all(), name
    short = build_anchor_signals(bundle.truncate(20))
    assert np.array_equal(short.w_aw_bidir_75_25, signals.w_aw_bidir_75_25[:20])
