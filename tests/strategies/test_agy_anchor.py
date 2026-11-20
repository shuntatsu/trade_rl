"""Parity, causality and input-contract tests of the agy anchor port.

The parity fixture (``tests/fixtures/agy_anchor``) holds klines + funding inputs for
2021 and the outputs of the read-only agy lab. It contains known-used pre-2026-08 data
only. Nothing here runs a backtest or reads an economic result.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from trade_rl.strategies.agy_anchor.inputs import (
    AnchorInputs,
    funding_observations,
    pair_names,
)
from trade_rl.strategies.agy_anchor.signals import AnchorSignals, build_anchor_signals
from trade_rl.strategies.agy_anchor.targets import interleave_legs, synthesize_row

FIXTURE = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "agy_anchor"
    / "agy_anchor_klines_funding_2021.npz"
)
PARITY_TOLERANCE = 1e-12


@pytest.fixture(scope="module")
def fixture() -> dict[str, np.ndarray]:
    with np.load(FIXTURE, allow_pickle=False) as handle:
        return {name: handle[name] for name in handle.files}


def _inputs(fx: dict[str, np.ndarray]) -> AnchorInputs:
    symbols = tuple(str(s) for s in fx["symbols"])
    owner = fx["funding_event_symbol_index"]
    return AnchorInputs(
        symbols=symbols,
        close_ns=fx["close_ns"],
        perp_close=fx["perp_close"],
        spot_close=fx["spot_close"],
        perp_quote_volume=fx["perp_quote_volume"],
        spot_quote_volume=fx["spot_quote_volume"],
        perp_taker_buy_quote_volume=fx["perp_taker_buy_quote_volume"],
        spot_taker_buy_quote_volume=fx["spot_taker_buy_quote_volume"],
        perp_valid=fx["perp_valid"],
        spot_valid=fx["spot_valid"],
        funding_event_ns=tuple(
            fx["funding_event_ns"][owner == i] for i in range(len(symbols))
        ),
        funding_event_rate=tuple(
            fx["funding_event_rate"][owner == i] for i in range(len(symbols))
        ),
    )


@pytest.fixture(scope="module")
def inputs(fixture: dict[str, np.ndarray]) -> AnchorInputs:
    return _inputs(fixture)


@pytest.fixture(scope="module")
def signals(inputs: AnchorInputs) -> AnchorSignals:
    return build_anchor_signals(inputs)


def _max_diff(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(np.asarray(a, dtype=np.float64) - b)))


# --- parity against the (corrected) lab reference ----------------------------------


def test_funding_observations_match_the_lab(
    inputs: AnchorInputs, fixture: dict[str, np.ndarray]
) -> None:
    last, ma3, ma7 = funding_observations(inputs)
    assert _max_diff(last, fixture["out_funding_obs_last"]) <= PARITY_TOLERANCE
    assert _max_diff(ma3, fixture["out_funding_obs_3d"]) <= PARITY_TOLERANCE
    assert _max_diff(ma7, fixture["out_funding_obs_7d"]) <= PARITY_TOLERANCE


def test_anchor_signal_chain_matches_the_lab_stage_by_stage(
    signals: AnchorSignals, fixture: dict[str, np.ndarray]
) -> None:
    assert _max_diff(signals.c_weekly_iv, fixture["out_c_weekly_iv"]) <= (
        PARITY_TOLERANCE
    )
    assert _max_diff(signals.w_aw_regime, fixture["out_w_aw_regime"]) <= (
        PARITY_TOLERANCE
    )
    assert np.array_equal(signals.carry_mask, fixture["out_carry_mask"])
    assert np.array_equal(signals.perp_mask, fixture["out_perp_mask"])
    assert _max_diff(signals.w_perp_bidir_carry, fixture["fix_w_perp_bidir_carry"]) <= (
        PARITY_TOLERANCE
    )
    assert _max_diff(signals.w_aw_bidir_75_25, fixture["fix_w_aw_bidir_75_25"]) <= (
        PARITY_TOLERANCE
    )
    assert _max_diff(signals.vol_cushion_tail, fixture["fix_vol_cushion_tail"]) <= (
        PARITY_TOLERANCE
    )


def test_final_4h_targets_match_the_lab_synthesis_on_every_row(
    signals: AnchorSignals, fixture: dict[str, np.ndarray]
) -> None:
    worst = 0.0
    for row in range(signals.n_rows):
        spot, perp = synthesize_row(signals, row)
        worst = max(
            worst,
            _max_diff(spot, fixture["fix_target_spot"][row]),
            _max_diff(perp, fixture["fix_target_perp"][row]),
        )
    assert worst <= PARITY_TOLERANCE


def test_the_known_lab_defect_is_the_only_difference_to_the_as_is_lab(
    signals: AnchorSignals, fixture: dict[str, np.ndarray]
) -> None:
    """Lab round 7 indexes ``factor_scores[:, :, 9]`` (a symbol) instead of ``[:, 9, :]``.

    Everything upstream of the bidirectional carry (sleeves, regime blend, weekly carry)
    is identical to the unmodified lab; the bidirectional sleeve, the blended anchor
    weights, the cushion and the final targets differ. If this ever fails the fixture or
    the documented decision in ``tests/fixtures/agy_anchor/README.md`` changed.
    """
    assert _max_diff(signals.w_aw_regime, fixture["out_w_aw_regime"]) <= (
        PARITY_TOLERANCE
    )
    assert (
        _max_diff(signals.w_perp_bidir_carry, fixture["out_w_perp_bidir_carry"]) > 0.1
    )
    assert _max_diff(signals.w_aw_bidir_75_25, fixture["out_w_aw_bidir_75_25"]) > 0.1


# --- causality ----------------------------------------------------------------------


@pytest.mark.parametrize("rows", [300, 1000, 1999])
def test_truncating_inputs_never_changes_earlier_rows(
    inputs: AnchorInputs, signals: AnchorSignals, rows: int
) -> None:
    short = build_anchor_signals(inputs.truncate(rows))
    assert short.n_rows == rows
    for name in (
        "w_aw_bidir_75_25",
        "c_weekly_iv",
        "vol_cushion_tail",
        "w_aw_regime",
        "w_perp_bidir_carry",
        "carry_mask",
        "perp_mask",
    ):
        assert np.array_equal(getattr(short, name), getattr(signals, name)[:rows]), name


def test_the_last_row_changes_nothing_that_precedes_it(
    inputs: AnchorInputs, signals: AnchorSignals
) -> None:
    corrupted = replace(
        inputs,
        perp_close=np.vstack([inputs.perp_close[:-1], inputs.perp_close[-1] * 3.0]),
        spot_close=np.vstack([inputs.spot_close[:-1], inputs.spot_close[-1] * 0.4]),
    )
    other = build_anchor_signals(corrupted)
    assert np.array_equal(other.w_aw_bidir_75_25[:-1], signals.w_aw_bidir_75_25[:-1])
    assert np.array_equal(other.vol_cushion_tail[:-1], signals.vol_cushion_tail[:-1])


def test_symbol_order_does_not_change_the_anchor(
    inputs: AnchorInputs, signals: AnchorSignals
) -> None:
    order = np.asarray([0, 1, 3, 2, 4, 6, 5, 9, 7, 10, 8])  # BTC first, rest shuffled

    def take(array: np.ndarray) -> np.ndarray:
        return array[:, order]

    permuted = AnchorInputs(
        symbols=tuple(inputs.symbols[i] for i in order),
        close_ns=inputs.close_ns,
        perp_close=take(inputs.perp_close),
        spot_close=take(inputs.spot_close),
        perp_quote_volume=take(inputs.perp_quote_volume),
        spot_quote_volume=take(inputs.spot_quote_volume),
        perp_taker_buy_quote_volume=take(inputs.perp_taker_buy_quote_volume),
        spot_taker_buy_quote_volume=take(inputs.spot_taker_buy_quote_volume),
        perp_valid=take(inputs.perp_valid),
        spot_valid=take(inputs.spot_valid),
        funding_event_ns=tuple(inputs.funding_event_ns[i] for i in order),
        funding_event_rate=tuple(inputs.funding_event_rate[i] for i in order),
    )
    result = build_anchor_signals(permuted)
    assert _max_diff(result.w_aw_bidir_75_25, take(signals.w_aw_bidir_75_25)) <= (
        PARITY_TOLERANCE
    )
    assert _max_diff(result.c_weekly_iv, take(signals.c_weekly_iv)) <= PARITY_TOLERANCE
    assert _max_diff(result.vol_cushion_tail, signals.vol_cushion_tail) <= (
        PARITY_TOLERANCE
    )


# --- structural invariants ----------------------------------------------------------


def test_anchor_weights_are_dollar_neutral_and_gross_bounded(
    signals: AnchorSignals,
) -> None:
    w = signals.w_aw_bidir_75_25
    active = np.abs(w).sum(axis=1) > 0.0
    assert active.any()
    assert np.max(np.abs(w.sum(axis=1))) < 1e-12
    assert np.max(np.abs(w).sum(axis=1)) <= 1.0 + 1e-12
    assert np.max(np.abs(w)) <= 0.25 + 1e-12
    assert np.all(signals.vol_cushion_tail >= 0.76)
    assert np.all(signals.vol_cushion_tail <= 1.0)


def test_targets_keep_spot_long_only_and_respect_the_caps(
    signals: AnchorSignals,
) -> None:
    for row in range(0, signals.n_rows, 7):
        spot, perp = synthesize_row(signals, row)
        legs = interleave_legs(spot, perp)
        assert np.all(spot >= 0.0)
        assert np.max(np.abs(legs)) <= 0.55 + 1e-12
        assert np.abs(legs).sum() <= 2.5 + 1e-9
        unusable = ~signals.spot_valid[row]
        # a leg without a carry mask never carries a spot target
        assert np.all(spot[~signals.carry_mask[row]] == 0.0)
        assert unusable.shape == spot.shape


def test_synthesize_row_rejects_rows_outside_the_range(signals: AnchorSignals) -> None:
    with pytest.raises(ValueError, match="row outside"):
        synthesize_row(signals, signals.n_rows)
    with pytest.raises(ValueError, match="row outside"):
        synthesize_row(signals, -1)


# --- input contract -----------------------------------------------------------------


def test_funding_events_become_known_at_the_first_close_at_or_after_them() -> None:
    t = 4
    close_ns = np.asarray([10, 20, 30, 40], dtype=np.int64)
    ones = np.ones((t, 4))
    base = {
        "close_ns": close_ns,
        "perp_close": ones,
        "spot_close": ones,
        "perp_quote_volume": ones,
        "spot_quote_volume": ones,
        "perp_taker_buy_quote_volume": ones * 0.5,
        "spot_taker_buy_quote_volume": ones * 0.5,
        "perp_valid": np.ones((t, 4), dtype=bool),
        "spot_valid": np.ones((t, 4), dtype=bool),
    }
    events = (
        np.asarray([20, 31], dtype=np.int64),
        np.asarray([], dtype=np.int64),
        np.asarray([5], dtype=np.int64),
        np.asarray([10, 40], dtype=np.int64),
    )
    rates = (
        np.asarray([0.01, 0.03]),
        np.asarray([]),
        np.asarray([0.02]),
        np.asarray([0.04, 0.08]),
    )
    bundle = AnchorInputs(
        symbols=("A", "B", "C", "D"),
        funding_event_ns=events,
        funding_event_rate=rates,
        **base,
    )
    last, ma3, ma7 = funding_observations(bundle)
    # A: the event at 20 is known at close 20 (<=), the event at 31 only at close 40
    assert last[:, 0].tolist() == [0.0, 0.01, 0.01, 0.03]
    assert ma3[:, 0].tolist() == pytest.approx([0.0, 0.01, 0.01, 0.02])
    assert last[:, 1].tolist() == [0.0] * 4  # no events: stays zero
    assert last[:, 2].tolist() == [0.02] * 4
    assert last[:, 3].tolist() == [0.04, 0.04, 0.04, 0.08]
    assert ma7[:, 3].tolist() == pytest.approx([0.04, 0.04, 0.04, 0.06])


def test_funding_means_use_the_last_9_and_21_events() -> None:
    n_events = 30
    close_ns = np.arange(1, n_events + 1, dtype=np.int64) * 10
    rate = np.arange(1, n_events + 1, dtype=np.float64)
    ones = np.ones((n_events, 4))
    bundle = AnchorInputs(
        symbols=("A", "B", "C", "D"),
        close_ns=close_ns,
        perp_close=ones,
        spot_close=ones,
        perp_quote_volume=ones,
        spot_quote_volume=ones,
        perp_taker_buy_quote_volume=ones,
        spot_taker_buy_quote_volume=ones,
        perp_valid=np.ones((n_events, 4), dtype=bool),
        spot_valid=np.ones((n_events, 4), dtype=bool),
        funding_event_ns=(close_ns, close_ns, close_ns, close_ns),
        funding_event_rate=(rate, rate, rate, rate),
    )
    _, ma3, ma7 = funding_observations(bundle)
    assert ma3[-1, 0] == pytest.approx(np.mean(rate[-9:]))
    assert ma7[-1, 0] == pytest.approx(np.mean(rate[-21:]))
    assert ma3[4, 0] == pytest.approx(np.mean(rate[:5]))  # short history: all events


def _kwargs(inputs: AnchorInputs) -> dict[str, object]:
    return {
        "symbols": inputs.symbols,
        "close_ns": inputs.close_ns,
        "perp_close": inputs.perp_close,
        "spot_close": inputs.spot_close,
        "perp_quote_volume": inputs.perp_quote_volume,
        "spot_quote_volume": inputs.spot_quote_volume,
        "perp_taker_buy_quote_volume": inputs.perp_taker_buy_quote_volume,
        "spot_taker_buy_quote_volume": inputs.spot_taker_buy_quote_volume,
        "perp_valid": inputs.perp_valid,
        "spot_valid": inputs.spot_valid,
        "funding_event_ns": inputs.funding_event_ns,
        "funding_event_rate": inputs.funding_event_rate,
    }


def test_inputs_refuse_inconsistent_or_look_ahead_data(inputs: AnchorInputs) -> None:
    good = _kwargs(inputs)

    def refused(match: str, **changes: object) -> None:
        with pytest.raises(ValueError, match=match):
            AnchorInputs(**{**good, **changes})  # type: ignore[arg-type]

    refused(
        "taker-buy volume exceeds",
        perp_taker_buy_quote_volume=(inputs.perp_quote_volume * 2.0 + 1.0),
    )
    refused("must be strictly positive", perp_close=inputs.perp_close * 0.0)
    refused("must be finite", spot_close=np.where(inputs.spot_close > 0, np.nan, 1.0))
    refused("strictly increasing", close_ns=inputs.close_ns[::-1])
    refused(
        "later than the last bar close",
        funding_event_ns=tuple(
            np.append(ns, int(inputs.close_ns[-1]) + 1)
            for ns in inputs.funding_event_ns
        ),
        funding_event_rate=tuple(
            np.append(rate, 0.0) for rate in inputs.funding_event_rate
        ),
    )
    refused(
        "strictly increasing",
        funding_event_ns=tuple(ns[::-1] for ns in inputs.funding_event_ns),
    )
    refused("at least four symbols", symbols=inputs.symbols[:3])
    refused("boolean", perp_valid=inputs.perp_valid.astype(np.int8))


def test_truncate_keeps_only_events_known_at_the_last_row(inputs: AnchorInputs) -> None:
    short = inputs.truncate(500)
    last_close = int(short.close_ns[-1])
    assert short.n_rows == 500
    assert all(int(ns[-1]) <= last_close for ns in short.funding_event_ns if ns.size)
    with pytest.raises(ValueError, match="rows must lie"):
        inputs.truncate(1)
    with pytest.raises(ValueError, match="rows must lie"):
        inputs.truncate(inputs.n_rows + 1)


def test_pair_names_accepts_only_the_alternating_layout() -> None:
    assert pair_names(("AUSDT:spot", "AUSDT:perp", "BUSDT:spot", "BUSDT:perp")) == (
        "AUSDT",
        "BUSDT",
    )
    for legs in (
        ("AUSDT:perp", "AUSDT:spot"),
        ("AUSDT:spot", "BUSDT:perp"),
        ("AUSDT:spot",),
        (),
    ):
        with pytest.raises(ValueError):
            pair_names(legs)
