"""Synthetic tests of the out-of-sample test ``trend_long_spot_oos_v1``.

Nothing here reads market data, the network or an economic result: every dataset is a
seeded synthetic fixture and every fixture value is hand-computed or derived from a slow
reference.
"""

from __future__ import annotations

import hashlib
import io
import json
from collections.abc import Sequence
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from tests.evaluation import test_core7_mtf_tierd as tierd_tests
from tools import core7_g4_comment as g4
from tools import trend_oos as driver
from trade_rl.data.source import RawMarketSeries
from trade_rl.evaluation.core7_mtf_study import tierd
from trade_rl.evaluation.lowturn_screen import policies as lowturn_policies
from trade_rl.evaluation.lowturn_screen import protocol as lowturn
from trade_rl.evaluation.lowturn_screen.policies import LowturnPolicy, family_targets
from trade_rl.evaluation.replay import run_shared_portfolio_replay
from trade_rl.evaluation.trend_oos import data, protocol, runner
from trade_rl.evaluation.trend_oos.decision import (
    OosCell,
    bootstrap_excess,
    combined_claim,
    conditions,
    decide,
    interim_report,
    label_of,
    stationary_bootstrap,
)
from trade_rl.evaluation.trend_oos.signals import build_oos_signals
from trade_rl.integrations.binance.carry import CarryCoverageError
from trade_rl.integrations.market_store import embargo as store_embargo
from trade_rl.integrations.market_store.contracts import KlineRow, StoreMarket
from trade_rl.integrations.market_store.fetch import VisionFetcher
from trade_rl.risk.pretrade import PreTradeRisk

N_LEGS = 22
SPOT = np.arange(11) * 2
PERP = SPOT + 1
HOUR_MS = 3_600_000
AUTH = "https://github.com/shuntatsu/trade_rl/issues/810#issuecomment-1"
DATA_AUTH = protocol.DATA_AUTHORIZATION_URL
LOWTURN_DIGEST_PIN = "9de83d14ec756924cd9d5bc36da8a95fe7d3e3daf258c59a01584fd23eb2c433"

# small synthetic stage windows: the first decision keeps the frozen 1085-row warm-up
SMALL_BACKCAST = {
    "load_start": "2018-01-01T00:00:00Z",
    "first_decision": "2018-07-01T00:00:00Z",
    "last_decision_bar": "2018-07-22T20:00:00Z",
    "settles": "2018-07-23T00:00:00Z",
}
SMALL_FORWARD = {
    "load_start": "2026-02-01T00:00:00Z",
    "first_decision": "2026-08-01T00:00:00Z",
    "last_decision_bar": "2026-08-22T20:00:00Z",
    "settles": "2026-08-23T00:00:00Z",
}
RULES = {
    symbol: (tierd.LegRules(1e-5, 5.0), tierd.LegRules(1e-3, 10.0))
    for symbol in protocol.ROSTER
}


def _small_window(monkeypatch: pytest.MonkeyPatch, stage: str) -> data.StageWindow:
    table = SMALL_BACKCAST if stage == "BACKCAST" else SMALL_FORWARD
    monkeypatch.setitem(protocol.STAGE_WINDOWS, stage, table)
    return data.frozen_window(stage)


def _series(
    window: data.StageWindow,
    *,
    first_row: int,
    seed: int,
    limit: datetime | None = None,
    drift: float = 0.0004,
    volume: float = 1.5e8,
) -> RawMarketSeries:
    """Complete hourly spot rows (close-stamped) from the 4h bar of ``first_row``."""
    last_row = window.row_index(limit or window.last_row_close)
    n_hours = (last_row + 1 - first_row) * 4
    first_open = window.load_start + timedelta(hours=4 * (first_row + 1) - 4)
    stamps = np.datetime64(first_open.replace(tzinfo=None), "ns") + (
        np.arange(n_hours) + 1
    ) * np.timedelta64(1, "h")
    rng = np.random.default_rng(seed)
    close = 100.0 * np.exp(np.cumsum(rng.normal(drift, 0.01, n_hours)))
    open_ = np.concatenate([[close[0]], close[:-1]])
    return RawMarketSeries(
        timestamps=stamps,
        open=open_,
        high=np.maximum(open_, close) * 1.002,
        low=np.minimum(open_, close) * 0.998,
        close=close,
        volume=rng.uniform(volume, 2 * volume, n_hours),
        funding_rate=np.zeros(n_hours),
        funding_available=np.zeros(n_hours, dtype=bool),
        funding_event_count=np.zeros(n_hours, dtype=np.int32),
        tradable=np.ones(n_hours, dtype=bool),
    )


def _spot_map(
    window: data.StageWindow,
) -> dict[str, RawMarketSeries | None]:
    """Five pairs listed from row 0, two from row 700, one after the first decision
    (row 1200) and three with no spot data at all."""
    out: dict[str, RawMarketSeries | None] = {}
    for index, symbol in enumerate(protocol.ROSTER):
        if index < 5:
            out[symbol] = _series(window, first_row=0, seed=index + 1)
        elif index < 7:
            out[symbol] = _series(window, first_row=700, seed=index + 1)
        elif index == 7:
            out[symbol] = _series(window, first_row=1200, seed=index + 1)
        else:
            out[symbol] = None
    return out


def _assemble(
    window: data.StageWindow,
    *,
    limit: datetime | None = None,
    spot: dict[str, RawMarketSeries | None] | None = None,
    authorization_url: str | None = None,
) -> Any:
    end = limit or window.last_row_close
    spot = spot or {
        symbol: None if series is None else _truncate(series, window, end)
        for symbol, series in _spot_map(window).items()
    }
    return data.assemble_stage_dataset(
        window,
        spot,
        rules=RULES,
        limit_close=end,
        data_authorization_url=authorization_url,
    )


def _without_identity(dataset: Any) -> Any:
    """The same arrays with a fresh id and no recorded identity payload."""
    return replace(
        dataset,
        dataset_id=hashlib.sha256(b"no-identity").hexdigest(),
        identity_payload_json=None,
    )


def _truncate(
    series: RawMarketSeries, window: data.StageWindow, limit: datetime
) -> RawMarketSeries | None:
    keep = int(
        np.searchsorted(
            np.asarray(series.timestamps).astype("datetime64[ns]"),
            np.datetime64(limit.replace(tzinfo=None), "ns"),
            side="right",
        )
    )
    if keep < 8:
        return None
    return RawMarketSeries(
        timestamps=np.asarray(series.timestamps)[:keep],
        open=series.open[:keep],
        high=series.high[:keep],
        low=series.low[:keep],
        close=series.close[:keep],
        volume=series.volume[:keep],
        funding_rate=series.funding_rate[:keep],
        funding_available=series.funding_available[:keep],
        funding_event_count=series.funding_event_count[:keep],
        tradable=series.tradable[:keep],
    )


# ============================================================ digest and constants


def test_protocol_values_are_imported_from_lowturn_not_retyped() -> None:
    assert protocol.ROSTER is lowturn.ROSTER
    assert protocol.LEG_ROSTER is lowturn.LEG_ROSTER
    assert protocol.LOOKBACK_BARS == lowturn.LOOKBACK_BARS
    for name in (
        "WARMUP_BARS",
        "REBALANCE_EVERY_BARS",
        "DEADBAND",
        "VOL_TARGET",
        "PER_NAME_CAP",
        "GROSS_CAP",
        "PERIODS_PER_YEAR",
        "INITIAL_CAPITAL",
    ):
        assert getattr(protocol, name) == getattr(lowturn, name), name
    assert protocol.SCENARIOS == lowturn.SCENARIOS
    assert protocol.risk_config() == lowturn.risk_config()
    for scenario in protocol.SCENARIOS:
        assert protocol.scenario_cost(scenario) == lowturn.scenario_cost(scenario)
        assert protocol.scenario_signal_lag_bars(
            scenario
        ) == lowturn.scenario_signal_lag_bars(scenario)
    assert protocol.ARMS == ("TS_TREND_LONG_SPOT", "CASH", "EW_HOLD_SPOT")
    assert protocol.CANDIDATE in lowturn.CANDIDATE_FAMILIES
    assert set(protocol.CONTROLS) <= set(lowturn.CONTROL_FAMILIES)


def test_frozen_constants_are_the_spec_values() -> None:
    assert protocol.PROTOCOL_ID == "trend_long_spot_oos_v1"
    assert protocol.MAX_DRAWDOWN_LIMIT == 0.20
    assert protocol.MIN_BASE_SHARPE == 0.30
    assert protocol.BOOTSTRAP_BLOCK_BARS == 42
    assert protocol.BOOTSTRAP_RESAMPLES == 10_000
    assert protocol.BOOTSTRAP_SEED == 20261120
    assert protocol.BOOTSTRAP_CONFIDENCE == 0.95
    assert protocol.FINAL_NOT_BEFORE == "2027-05-02T00:00:00Z"
    assert protocol.DATA_AUTHORIZATION_URL.endswith("#issuecomment-7028416072")
    assert protocol.STAGE_WINDOWS == {
        "BACKCAST": {
            "load_start": "2018-01-01T00:00:00Z",
            "first_decision": "2018-07-01T00:00:00Z",
            "last_decision_bar": "2020-12-31T20:00:00Z",
            "settles": "2021-01-01T00:00:00Z",
        },
        "FORWARD": {
            "load_start": "2026-02-01T00:00:00Z",
            "first_decision": "2026-08-01T00:00:00Z",
            "last_decision_bar": "2027-04-30T20:00:00Z",
            "settles": "2027-05-01T00:00:00Z",
        },
    }
    for stage in protocol.STAGES:
        window = data.frozen_window(stage)
        assert window.first_decision_index == 1085  # >= the 1080-bar warm-up
        assert window.settles == window.last_decision_bar + timedelta(hours=4)
        assert window.last_row_close == window.last_decision_bar + timedelta(hours=8)


def test_the_digest_is_canonical_stable_and_pinned() -> None:
    digest = protocol.protocol_digest()
    assert digest == protocol.protocol_digest()
    assert len(digest) == 64 and digest == digest.lower()
    assert json.loads(json.dumps(protocol.protocol_payload())) == json.loads(
        json.dumps(protocol.protocol_payload(), sort_keys=True)
    )
    assert digest == "535a25162edd843687a845e920d178b0ea7af184eb8fd6968202e5222bc3a1bb"


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("MIN_BASE_SHARPE", 0.31),
        ("MAX_DRAWDOWN_LIMIT", 0.21),
        ("BOOTSTRAP_SEED", 1),
        ("BOOTSTRAP_RESAMPLES", 9_999),
        ("BOOTSTRAP_BLOCK_BARS", 41),
        ("FINAL_NOT_BEFORE", "2027-05-03T00:00:00Z"),
        (
            "DATA_AUTHORIZATION_URL",
            "https://github.com/shuntatsu/trade_rl/issues/810#issuecomment-1",
        ),
    ],
)
def test_changing_any_frozen_constant_changes_the_digest(
    monkeypatch: pytest.MonkeyPatch, name: str, value: object
) -> None:
    before = protocol.protocol_digest()
    monkeypatch.setattr(protocol, name, value)
    assert protocol.protocol_digest() != before


def test_changing_a_stage_window_or_the_halts_changes_the_digest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    before = protocol.protocol_digest()
    windows = {k: dict(v) for k, v in protocol.STAGE_WINDOWS.items()}
    windows["BACKCAST"]["first_decision"] = "2018-07-02T00:00:00Z"
    monkeypatch.setattr(protocol, "STAGE_WINDOWS", windows)
    assert protocol.protocol_digest() != before
    monkeypatch.undo()
    halts = (*protocol.SPOT_HALTS, (datetime(2018, 1, 1, tzinfo=UTC),) * 2)
    monkeypatch.setattr(protocol, "SPOT_HALTS", halts)
    assert protocol.protocol_digest() != before


def test_the_lowturn_digest_is_pinned_and_a_change_there_is_detected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert protocol.LOWTURN_PROTOCOL_DIGEST == LOWTURN_DIGEST_PIN
    assert lowturn.protocol_digest() == protocol.LOWTURN_PROTOCOL_DIGEST
    protocol.verify_lowturn_binding()
    payload = protocol.protocol_payload()
    assert payload["reuses"]["protocol_digest"] == LOWTURN_DIGEST_PIN  # type: ignore[index]
    for name, value in (
        ("PER_NAME_CAP", 0.41),
        ("DEADBAND", 0.03),
        ("VOL_TARGET", 0.6),
    ):
        with monkeypatch.context() as ctx:
            ctx.setattr(lowturn, name, value)
            with pytest.raises(protocol.ProtocolBindingError):
                protocol.protocol_digest()
    # the run refuses too, before it touches the dataset
    with monkeypatch.context() as ctx:
        ctx.setattr(lowturn, "REBALANCE_EVERY_BARS", 41)
        with pytest.raises(protocol.ProtocolBindingError):
            runner.run_stage(
                None,  # type: ignore[arg-type]
                window=data.frozen_window("BACKCAST"),
                output_dir=Path("unused"),
                expected_protocol_digest="0" * 64,
            )


# ===================================================== unavailable legs (no listing)


def test_unlisted_spot_rows_and_all_perp_legs_are_unavailable_never_invented() -> None:
    window = data.frozen_window("BACKCAST").__class__(
        stage="BACKCAST",
        load_start=datetime(2018, 1, 1, tzinfo=UTC),
        first_decision=datetime(2018, 7, 1, tzinfo=UTC),
        last_decision_bar=datetime(2018, 7, 22, 20, tzinfo=UTC),
        settles=datetime(2018, 7, 23, tzinfo=UTC),
    )
    dataset = _assemble(window)
    last_index = data.validate_stage_dataset(dataset, window)
    assert last_index == window.last_row_index
    active = dataset.resolved_array("asset_active")
    tradable = np.asarray(dataset.tradable)
    info = dataset.resolved_array("information_available")
    close = np.asarray(dataset.close)
    # every perp leg is unavailable in every row
    assert not active[:, PERP].any() and not tradable[:, PERP].any()
    assert not info[:, PERP].any()
    # pair 0 is listed in every row; pair 5 from row 700; pair 7 from row 1200
    assert active[:, 0].all()
    assert not active[:700, SPOT[5]].any() and active[700:, SPOT[5]].all()
    assert not active[:1200, SPOT[7]].any() and active[1200:, SPOT[7]].all()
    # a symbol with no spot data is a valid all-unavailable leg
    for pair in (8, 9, 10):
        assert not active[:, SPOT[pair]].any()
        assert not tradable[:, SPOT[pair]].any()
    # unavailable rows hold only the constant sentinel; listed rows hold market prices
    unavailable = ~active
    assert np.all(close[unavailable] == data.UNAVAILABLE_PRICE)
    assert np.all(np.asarray(dataset.volume)[unavailable] == 0.0)
    assert not np.any(close[active] == data.UNAVAILABLE_PRICE)
    # the identity records the listing facts and the assumptions
    payload = json.loads(dataset.identity_payload_json or "{}")
    assert payload["listing"]["BTCUSDT"]["first_listed_row"] == 0
    assert payload["listing"]["ADAUSDT"]["first_listed_row"] == 700
    assert payload["listing"]["DOTUSDT"]["first_listed_row"] is None
    assert (
        payload["rules_assumption"] == "static_exchange_rule_snapshot_not_point_in_time"
    )
    assert payload["perp_legs"] == "not_loaded_inactive_in_every_row"


def test_signals_before_listing_are_missing_and_the_policy_puts_zero_weight_there(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    window = _small_window(monkeypatch, "BACKCAST")
    dataset = _assemble(window)
    signals = build_oos_signals(dataset)
    ada, sol, dot = SPOT[5], SPOT[7], SPOT[8]
    # the sentinel never leaks into a lookback: a listed leg has no signal until every
    # close its lookback needs is real (r180 needs 1080 rows after the first listed row)
    assert not signals.usable[:700, ada].any()
    assert np.isnan(signals.r180[: 700 + 1080, ada]).all()
    assert np.isfinite(signals.r180[700 + 1080 :, ada]).all()
    assert np.isnan(signals.vol60[: 700 + 360, ada]).all()
    assert not signals.usable[:, dot].any()
    assert np.isnan(signals.r30[:, dot]).all()
    assert np.isnan(signals.r30[:, PERP]).all() and not signals.usable[:, PERP].any()

    row = window.first_decision_index
    for arm in protocol.ARMS:
        target = family_targets(arm, signals, row)
        assert np.all(target[PERP] == 0.0), arm
        for unavailable in (sol, dot, SPOT[9], SPOT[10]):
            assert target[unavailable] == 0.0, (arm, unavailable)
        assert np.all(target >= 0.0)
    trend = family_targets("TS_TREND_LONG_SPOT", signals, row)
    assert trend[ada] == 0.0  # listed but its 180-day history does not exist yet
    assert trend[SPOT[:5]].sum() > 0.0  # the five long-listed pairs can trade
    ew = family_targets("EW_HOLD_SPOT", signals, row)
    assert ew[ada] == pytest.approx(1 / 11) and ew[sol] == 0.0 and ew[dot] == 0.0
    assert np.count_nonzero(ew) == 7
    assert family_targets("CASH", signals, row).sum() == 0.0


def test_the_targets_never_weight_an_unavailable_leg_in_any_row_or_lag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    window = _small_window(monkeypatch, "BACKCAST")
    dataset = _assemble(window)
    signals = build_oos_signals(dataset)
    active = dataset.resolved_array("asset_active")
    for arm in protocol.ARMS:
        for row in range(window.first_decision_index, window.last_row_index - 1):
            for lag in (0, 1):
                target = family_targets(arm, signals, max(0, row - lag))
                assert np.all(target[~active[row - lag]] == 0.0), (arm, row, lag)
                assert np.all(target[~active[row]] == 0.0), (arm, row, lag)


def test_a_gap_a_late_end_or_an_off_clock_start_is_refused() -> None:
    window = data.StageWindow(
        stage="BACKCAST",
        load_start=datetime(2018, 1, 1, tzinfo=UTC),
        first_decision=datetime(2018, 7, 1, tzinfo=UTC),
        last_decision_bar=datetime(2018, 7, 22, 20, tzinfo=UTC),
        settles=datetime(2018, 7, 23, tzinfo=UTC),
    )
    spot = _spot_map(window)
    btc = spot["BTCUSDT"]
    assert btc is not None
    # a missing hour in the middle: the hourly clock is not complete
    holed = RawMarketSeries(
        timestamps=np.delete(np.asarray(btc.timestamps), 50),
        open=np.delete(btc.open, 50),
        high=np.delete(btc.high, 50),
        low=np.delete(btc.low, 50),
        close=np.delete(btc.close, 50),
        volume=np.delete(btc.volume, 50),
        funding_rate=np.delete(btc.funding_rate, 50),
        funding_available=np.delete(btc.funding_available, 50),
        funding_event_count=np.delete(btc.funding_event_count, 50),
        tradable=np.delete(btc.tradable, 50),
    )
    with pytest.raises(data.StageDataError, match="complete and regular|whole 4h"):
        _assemble(window, spot={**spot, "BTCUSDT": holed})
    # a series that stops four hours early
    short = _truncate(btc, window, window.last_row_close - timedelta(hours=4))
    with pytest.raises(data.StageDataError, match="without a gap to the read limit"):
        _assemble(window, spot={**spot, "BTCUSDT": short})
    # a series that starts on the clock but a different roster
    with pytest.raises(data.StageDataError, match="exactly the roster"):
        _assemble(window, spot={"BTCUSDT": btc})
    # a perpetual-like funding payload is refused on spot
    funded = RawMarketSeries(
        timestamps=btc.timestamps,
        open=btc.open,
        high=btc.high,
        low=btc.low,
        close=btc.close,
        volume=btc.volume,
        funding_rate=np.full(btc.close.shape, 0.0001),
        funding_available=btc.funding_available,
        funding_event_count=btc.funding_event_count,
        tradable=btc.tradable,
    )
    with pytest.raises(data.StageDataError, match="must not carry funding"):
        _assemble(window, spot={**spot, "BTCUSDT": funded})


def test_a_dataset_without_the_first_roster_pair_from_the_start_is_refused() -> None:
    window = data.StageWindow(
        stage="BACKCAST",
        load_start=datetime(2018, 1, 1, tzinfo=UTC),
        first_decision=datetime(2018, 7, 1, tzinfo=UTC),
        last_decision_bar=datetime(2018, 7, 22, 20, tzinfo=UTC),
        settles=datetime(2018, 7, 23, tzinfo=UTC),
    )
    spot = _spot_map(window)
    for bad in (None, _series(window, first_row=700, seed=1)):
        with pytest.raises(
            data.StageDataError, match="listed from the stage load start"
        ):
            _assemble(window, spot={**spot, "BTCUSDT": bad})


def test_a_declared_spot_halt_is_a_non_tradable_bar_and_an_undeclared_one_is_refused() -> (
    None
):
    window = data.StageWindow(
        stage="BACKCAST",
        load_start=datetime(2018, 1, 1, tzinfo=UTC),
        first_decision=datetime(2018, 7, 1, tzinfo=UTC),
        last_decision_bar=datetime(2018, 7, 22, 20, tzinfo=UTC),
        settles=datetime(2018, 7, 23, tzinfo=UTC),
    )
    spot = _spot_map(window)
    btc = spot["BTCUSDT"]
    assert btc is not None
    lo = window.load_start + timedelta(hours=100)
    halt = ((lo, lo + timedelta(hours=4)),)

    def halted_copy(series: RawMarketSeries) -> RawMarketSeries:
        tradable = series.tradable.copy()
        tradable[100:104] = False  # hours 100..103 form bar 25
        return RawMarketSeries(
            timestamps=series.timestamps,
            open=series.open,
            high=series.high,
            low=series.low,
            close=series.close,
            volume=np.where(tradable, series.volume, 0.0),
            funding_rate=series.funding_rate,
            funding_available=series.funding_available,
            funding_event_count=series.funding_event_count,
            tradable=tradable,
        )

    # the halt calendar is global: every pair listed then is non-tradable in that bar
    halted = {
        symbol: halted_copy(series) if index < 5 and series is not None else series
        for index, (symbol, series) in enumerate(spot.items())
    }
    dataset = data.assemble_stage_dataset(
        window,
        halted,
        rules=RULES,
        limit_close=window.last_row_close,
        halt_intervals=halt,
    )
    assert not np.asarray(dataset.tradable)[25, 0]
    assert dataset.resolved_array("asset_active")[25, 0]  # priced, just halted
    assert np.asarray(dataset.tradable)[26, 0]
    # the same bar without a declared halt is tradable inside a "halt": refused
    with pytest.raises(ValueError, match="tradable bar inside the declared halt"):
        data.assemble_stage_dataset(
            window,
            {**spot, "BTCUSDT": btc},
            rules=RULES,
            limit_close=window.last_row_close,
            halt_intervals=halt,
        )


# ================================================== embargo and stage-window refusals


def _forward() -> data.StageWindow:
    return data.frozen_window("FORWARD")


def test_the_forward_read_limit_needs_the_exact_authorization_and_never_exceeds_closed_bars() -> (
    None
):
    window = _forward()
    now = datetime(2026, 11, 20, 6, 30, tzinfo=UTC)
    with pytest.raises(data.StageEmbargoError, match="owner authorization"):
        data.read_limit(window, now=now)
    for wrong in (
        "https://github.com/shuntatsu/trade_rl/issues/810#issuecomment-1",
        protocol.DATA_AUTHORIZATION_URL + "0",
        "",
    ):
        with pytest.raises(data.StageEmbargoError):
            data.read_limit(window, now=now, authorization_url=wrong)
    limit = data.read_limit(window, now=now, authorization_url=DATA_AUTH)
    assert limit == datetime(2026, 11, 20, 4, tzinfo=UTC)  # the latest fully closed bar
    assert data.latest_closed_bar(datetime(2026, 11, 20, 3, 59, 59, tzinfo=UTC)) == (
        datetime(2026, 11, 20, 0, tzinfo=UTC)
    )
    assert data.latest_closed_bar(datetime(2026, 11, 20, 4, tzinfo=UTC)) == limit
    # after the stage ends, the limit is the stage's last row, never later
    late = datetime(2028, 1, 1, tzinfo=UTC)
    assert data.read_limit(window, now=late, authorization_url=DATA_AUTH) == (
        window.last_row_close
    )
    assert window.last_row_close == datetime(2027, 5, 1, 4, tzinfo=UTC)
    # `through` can only lower the limit, on the 4h grid, after the first decision
    assert data.read_limit(
        window,
        now=now,
        authorization_url=DATA_AUTH,
        through=datetime(2026, 10, 1, 8, tzinfo=UTC),
    ) == datetime(2026, 10, 1, 8, tzinfo=UTC)
    for bad in (
        datetime(2026, 11, 20, 8, tzinfo=UTC),  # not yet closed
        datetime(2026, 10, 1, 9, tzinfo=UTC),  # off the grid
        datetime(2026, 8, 1, 4, tzinfo=UTC),  # no decision room
    ):
        with pytest.raises(data.StageEmbargoError):
            data.read_limit(window, now=now, authorization_url=DATA_AUTH, through=bad)
    # before the first decision nothing is readable
    with pytest.raises(data.StageEmbargoError, match="no closed bar"):
        data.read_limit(
            window,
            now=datetime(2026, 8, 1, 4, tzinfo=UTC),
            authorization_url=DATA_AUTH,
        )


def test_the_backcast_limit_is_the_whole_window_only_and_never_past_the_embargo() -> (
    None
):
    window = data.frozen_window("BACKCAST")
    now = datetime(2026, 11, 20, tzinfo=UTC)
    assert data.read_limit(window, now=now) == datetime(2021, 1, 1, 4, tzinfo=UTC)
    assert data.read_limit(window, now=now) < data.DEFAULT_EMBARGO
    with pytest.raises(data.StageEmbargoError, match="not over yet"):
        data.read_limit(window, now=datetime(2019, 1, 1, tzinfo=UTC))


class _CountingRows:
    def __init__(self, extra_open_ms: int | None = None) -> None:
        self.calls = 0
        self.extra = extra_open_ms

    def read_klines(
        self, market: Any, symbol: str, interval: str, start_ms: int, end_ms: int
    ) -> list[KlineRow]:
        self.calls += 1
        opens = list(range(start_ms, min(end_ms, start_ms + 3 * HOUR_MS), HOUR_MS))
        if self.extra is not None:
            opens.append(self.extra)
        return [
            KlineRow(
                o,
                *(Decimal(1),) * 4,
                Decimal(1),
                o + 3_599_999,
                Decimal(1),
                1,
                Decimal(1),
                Decimal(1),
            )  # type: ignore[arg-type]
            for o in opens
        ]


def test_the_stage_row_source_refuses_before_reading_and_on_every_returned_row() -> (
    None
):
    window = _forward()
    limit = datetime(2026, 11, 1, tzinfo=UTC)
    lo = int(window.load_start.timestamp() * 1_000)
    hi = int(limit.timestamp() * 1_000)
    inner = _CountingRows()
    guard = data.StageRowSource(inner, window=window, limit_close=limit)
    spot = StoreMarket.SPOT
    rows = guard.read_klines(spot, "BTCUSDT", "1h", lo, lo + 5 * HOUR_MS)
    assert len(rows) == 3 and guard.rows_read == 3 and inner.calls == 1
    # beyond the limit: refused before the inner source is read
    with pytest.raises(data.StageEmbargoError, match="past the read limit"):
        guard.read_klines(spot, "BTCUSDT", "1h", lo, hi + HOUR_MS)
    # before the stage load start
    with pytest.raises(data.StageEmbargoError, match="before the stage load start"):
        guard.read_klines(spot, "BTCUSDT", "1h", lo - HOUR_MS, lo + HOUR_MS)
    # perpetuals, funding, other intervals, other symbols and empty ranges
    with pytest.raises(data.StageDataError, match="only spot"):
        guard.read_klines(StoreMarket.USDS_M, "BTCUSDT", "1h", lo, lo + HOUR_MS)
    with pytest.raises(data.StageDataError, match="funding is never read"):
        guard.read_funding(StoreMarket.USDS_M, "BTCUSDT", lo, hi)
    with pytest.raises(ValueError, match="only 1h"):
        guard.read_klines(spot, "BTCUSDT", "4h", lo, lo + HOUR_MS)
    with pytest.raises(data.StageDataError, match="not in the roster"):
        guard.read_klines(spot, "XYZUSDT", "1h", lo, lo + HOUR_MS)
    with pytest.raises(ValueError, match="empty"):
        guard.read_klines(spot, "BTCUSDT", "1h", lo, lo)
    assert inner.calls == 1
    # a source that returns a row past the requested range is refused
    leaky = data.StageRowSource(
        _CountingRows(extra_open_ms=hi + 5 * HOUR_MS), window=window, limit_close=limit
    )
    with pytest.raises(data.StageEmbargoError, match="outside the requested range"):
        leaky.read_klines(spot, "BTCUSDT", "1h", lo, lo + 5 * HOUR_MS)
    # the guard cannot be built past the stage's last row
    with pytest.raises(data.StageEmbargoError):
        data.StageRowSource(
            inner, window=window, limit_close=window.last_row_close + timedelta(hours=4)
        )


def test_rows_past_the_default_embargo_need_the_forward_stage_and_the_exact_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    window = _small_window(monkeypatch, "FORWARD")
    spot_series = {
        symbol: None if s is None else s for symbol, s in _spot_map(window).items()
    }
    with pytest.raises(data.StageEmbargoError, match="owner authorization"):
        data.assemble_stage_dataset(
            window, spot_series, rules=RULES, limit_close=window.last_row_close
        )
    with pytest.raises(data.StageEmbargoError):
        data.assemble_stage_dataset(
            window,
            spot_series,
            rules=RULES,
            limit_close=window.last_row_close,
            data_authorization_url="https://github.com/shuntatsu/trade_rl/issues/810#issuecomment-2",
        )
    # the BACKCAST stage may not read past it even with the URL
    back = data.StageWindow(
        stage="BACKCAST",
        load_start=window.load_start,
        first_decision=window.first_decision,
        last_decision_bar=window.last_decision_bar,
        settles=window.settles,
    )
    with pytest.raises(data.StageEmbargoError):
        data.assemble_stage_dataset(
            back,
            spot_series,
            rules=RULES,
            limit_close=back.last_row_close,
            data_authorization_url=DATA_AUTH,
        )
    dataset = data.assemble_stage_dataset(
        window,
        spot_series,
        rules=RULES,
        limit_close=window.last_row_close,
        data_authorization_url=DATA_AUTH,
    )
    assert data.validate_stage_dataset(dataset, window) == window.last_row_index
    assert json.loads(dataset.identity_payload_json or "{}")[
        "data_authorization_url"
    ] == (DATA_AUTH)
    # a dataset past the embargo whose identity lacks the URL is refused by the runner
    # side validation
    bare = _without_identity(dataset)
    with pytest.raises(data.StageEmbargoError, match="no FORWARD authorization"):
        data.validate_stage_dataset(bare, window)


def test_the_validation_refuses_a_dataset_that_is_not_the_stage_clock_or_has_a_perp() -> (
    None
):
    window = data.StageWindow(
        stage="BACKCAST",
        load_start=datetime(2018, 1, 1, tzinfo=UTC),
        first_decision=datetime(2018, 7, 1, tzinfo=UTC),
        last_decision_bar=datetime(2018, 7, 22, 20, tzinfo=UTC),
        settles=datetime(2018, 7, 23, tzinfo=UTC),
    )
    dataset = _assemble(window)
    shifted = data.StageWindow(
        stage="BACKCAST",
        load_start=datetime(2018, 1, 1, 4, tzinfo=UTC),
        first_decision=window.first_decision,
        last_decision_bar=window.last_decision_bar,
        settles=window.settles,
    )
    with pytest.raises(data.StageDataError, match="stage clock"):
        data.validate_stage_dataset(dataset, shifted)
    shorter = data.StageWindow(
        stage="BACKCAST",
        load_start=window.load_start,
        first_decision=window.first_decision,
        last_decision_bar=window.last_decision_bar - timedelta(hours=40),
        settles=window.settles - timedelta(hours=40),
    )
    with pytest.raises(data.StageEmbargoError, match="past the stage's last row"):
        data.validate_stage_dataset(dataset, shorter)


def test_the_default_core7_embargo_is_untouched_by_the_forward_path() -> None:
    forward = driver.ForwardEmbargoPolicy(
        warmup_authorization_url=DATA_AUTH,
        admit_before=datetime(2026, 11, 20, tzinfo=UTC),
    )
    default = store_embargo.EmbargoPolicy.enforced()
    assert default.sealed_start == store_embargo.SEALED_WINDOW_START
    assert default.start == store_embargo.FS1_WARMUP_START
    assert store_embargo.SEALED_WINDOW_START == datetime(2026, 11, 1, tzinfo=UTC)
    with pytest.raises(store_embargo.Fs1WarmupSpanError):
        default.check_ms(
            int(datetime(2026, 9, 1, tzinfo=UTC).timestamp() * 1_000), context="x"
        )
    with pytest.raises(store_embargo.SealedWindowEmbargoError):
        default.check_ms(
            int(datetime(2026, 11, 5, tzinfo=UTC).timestamp() * 1_000), context="x"
        )
    # tierd still refuses the embargo
    with pytest.raises(tierd.TierDEmbargoError):
        tierd.validate_load_window(
            datetime(2021, 10, 1, tzinfo=UTC), datetime(2026, 9, 1, tzinfo=UTC)
        )
    assert tierd.DATA_END_EXCLUSIVE == datetime(2026, 8, 1, tzinfo=UTC)
    # the forward policy admits strictly before its bound and refuses at or after it
    inside = int(datetime(2026, 11, 19, 23, tzinfo=UTC).timestamp() * 1_000)
    forward.check_ms(inside, context="x")
    with pytest.raises(store_embargo.SealedWindowEmbargoError):
        forward.check_ms(
            int(datetime(2026, 11, 20, tzinfo=UTC).timestamp() * 1_000), context="x"
        )
    forward.check_archive_end(datetime(2026, 11, 20, tzinfo=UTC), url="u")
    with pytest.raises(store_embargo.SealedWindowEmbargoError):
        forward.check_archive_end(datetime(2026, 11, 21, tzinfo=UTC), url="u")
    # constructing it did not move the default one
    assert store_embargo.EmbargoPolicy.enforced().sealed_start == (
        store_embargo.SEALED_WINDOW_START
    )
    assert store_embargo.EmbargoPolicy.enforced().warmup_start == (
        store_embargo.FS1_WARMUP_START
    )


def test_the_forward_policy_exists_only_for_the_exact_authorization_and_a_midnight() -> (
    None
):
    admit = datetime(2026, 11, 20, tzinfo=UTC)
    for url in (
        None,
        "https://github.com/shuntatsu/trade_rl/issues/810#issuecomment-1",
    ):
        with pytest.raises(ValueError):
            driver.ForwardEmbargoPolicy(
                warmup_authorization_url=url, admit_before=admit
            )
    with pytest.raises(ValueError, match="midnight"):
        driver.ForwardEmbargoPolicy(
            warmup_authorization_url=DATA_AUTH,
            admit_before=datetime(2026, 11, 20, 4, tzinfo=UTC),
        )
    # BACKCAST uses the enforced default policy itself
    assert type(
        driver.embargo_for(datetime(2021, 1, 1, 4, tzinfo=UTC), stage="BACKCAST")
    ) is (store_embargo.EmbargoPolicy)
    policy = driver.embargo_for(datetime(2027, 5, 1, 4, tzinfo=UTC), stage="FORWARD")
    assert isinstance(policy, driver.ForwardEmbargoPolicy)
    assert policy.admit_before == datetime(2027, 5, 2, tzinfo=UTC)
    assert policy.warmup_authorization_url == DATA_AUTH
    with pytest.raises(data.StageEmbargoError):
        driver.embargo_for(datetime(2026, 9, 1, tzinfo=UTC), stage="BACKCAST")


def test_the_forward_fetcher_serves_only_archives_before_its_bound() -> None:
    admit = datetime(2026, 11, 20, tzinfo=UTC)
    policy = driver.ForwardEmbargoPolicy(
        warmup_authorization_url=DATA_AUTH, admit_before=admit
    )
    requested: list[str] = []

    class _Response:
        status = 200

        def __init__(self, url: str) -> None:
            self._url = url

        def __enter__(self) -> _Response:
            return self

        def __exit__(self, *_: object) -> None:
            return None

        def geturl(self) -> str:
            return self._url

        def read(self, _: int) -> bytes:
            return b"x"

    def opener(request: Any, timeout: float) -> _Response:
        requested.append(request.full_url)
        return _Response(request.full_url)

    base = "https://data.binance.vision/data/spot"
    fetcher = driver._StageVisionFetcher(policy, _opener=opener)
    ok = f"{base}/monthly/klines/BTCUSDT/1h/BTCUSDT-1h-2026-09.zip"
    assert fetcher.fetch(ok).status == 200
    assert fetcher.fetch(ok + ".CHECKSUM").status == 200
    daily_ok = f"{base}/daily/klines/BTCUSDT/1h/BTCUSDT-1h-2026-11-19.zip"
    assert fetcher.fetch(daily_ok).status == 200
    n = len(requested)
    for bad in (
        f"{base}/daily/klines/BTCUSDT/1h/BTCUSDT-1h-2026-11-20.zip",
        f"{base}/monthly/klines/BTCUSDT/1h/BTCUSDT-1h-2026-11.zip",
        f"{base}/monthly/klines/BTCUSDT/1h/BTCUSDT-1h-2027-01.zip",
    ):
        with pytest.raises(store_embargo.SealedWindowEmbargoError):
            fetcher.fetch(bad)
    assert len(requested) == n  # refused before any network call
    # the default fetcher still refuses the same warm-up archive
    default = VisionFetcher(_opener=opener)
    with pytest.raises(store_embargo.Fs1WarmupSpanError):
        default.fetch(ok)
    assert len(requested) == n


# ============================================================== decision rule


def _cell(arm: str, scenario: str, **over: Any) -> OosCell:
    values: dict[str, Any] = {
        "arm": arm,
        "scenario": scenario,
        "total_return": 0.10,
        "annual_sharpe": 1.0,
        "max_drawdown": 0.05,
        "total_cost": 10.0,
        "funding_pnl": 0.0,
        "turnover_total": 3.0,
        "complete": True,
        "terminal_flat": True,
        "terminal_dust_only": False,
        "execution_valid": True,
        "causality_passed": True,
        "ledger_reconciled": True,
        "stopped": False,
        "stop_reason": None,
        "invalid_reasons": (),
        "n_returns": 4,
        "calendar_year_returns": {"2019": 0.1},
        "returns": (0.01, -0.01, 0.02, 0.0),
    }
    values.update(over)
    return OosCell(**values)


def _grid(
    cells: dict[tuple[str, str], dict[str, Any]] | None = None,
) -> dict[Any, OosCell]:
    base: dict[tuple[str, str], dict[str, Any]] = {
        ("TS_TREND_LONG_SPOT", "BASE"): {
            "total_return": 0.2,
            "annual_sharpe": 0.8,
            "max_drawdown": 0.1,
        },
        ("TS_TREND_LONG_SPOT", "HIGH_COST"): {
            "total_return": 0.18,
            "annual_sharpe": 0.7,
            "max_drawdown": 0.1,
        },
        ("TS_TREND_LONG_SPOT", "DELAY_1BAR"): {
            "total_return": 0.19,
            "annual_sharpe": 0.7,
            "max_drawdown": 0.1,
        },
    }
    for scenario in protocol.SCENARIOS:
        base[("EW_HOLD_SPOT", scenario)] = {
            "total_return": 0.3,
            "annual_sharpe": 0.25,
            "max_drawdown": 0.5,
        }
        base[("CASH", scenario)] = {
            "total_return": 0.0,
            "annual_sharpe": 0.0,
            "max_drawdown": 0.0,
            "total_cost": 0.0,
            "turnover_total": 0.0,
        }
    for key, extra in (cells or {}).items():
        base[key] = {**base[key], **extra}
    return {k: _cell(*k, **v) for k, v in base.items()}


def _key(name: str) -> tuple[str, str]:
    arm, _, scenario = name.partition("__")
    return arm, scenario


CAND = "TS_TREND_LONG_SPOT"


def test_the_baseline_grid_is_supported_and_every_condition_holds() -> None:
    grid = _grid()
    assert all(conditions(grid).values())
    result = decide(grid)
    assert result["label"] == "OOS_SUPPORTED" and result["supported"] is True
    assert result["failed_conditions"] == [] and result["invalid_cells"] == []
    assert result["claim_ceiling"] == "OOS_SUPPORTED"
    assert set(result["cells"]) == {
        f"{a}__{s}" for a in protocol.ARMS for s in protocol.SCENARIOS
    }


@pytest.mark.parametrize(
    ("cell", "override", "failed"),
    [
        (f"{CAND}__BASE", {"max_drawdown": 0.20}, "2_"),  # exactly the limit fails
        (f"{CAND}__HIGH_COST", {"max_drawdown": 0.25}, "2_"),
        (f"{CAND}__DELAY_1BAR", {"max_drawdown": 0.2000001}, "2_"),
        (f"{CAND}__BASE", {"total_return": 0.0}, "3_"),  # exactly zero fails
        (f"{CAND}__BASE", {"total_return": -0.01}, "3_"),
        (f"{CAND}__BASE", {"annual_sharpe": 0.2999}, "4_"),
        (f"{CAND}__HIGH_COST", {"total_return": 0.0}, "5_"),
        (f"{CAND}__DELAY_1BAR", {"total_return": -0.001}, "5_"),
        ("EW_HOLD_SPOT__BASE", {"annual_sharpe": 0.8}, "6_"),  # equal to EW fails
        ("EW_HOLD_SPOT__BASE", {"annual_sharpe": 0.9}, "6_"),
    ],
)
def test_each_condition_is_load_bearing_and_fails_to_not_supported(
    cell: str, override: dict[str, Any], failed: str
) -> None:
    grid = _grid(cells={_key(cell): override})
    result = decide(grid)
    assert result["label"] == "OOS_NOT_SUPPORTED"
    assert [name[:2] for name in result["failed_conditions"]] == [failed]
    assert result["invalid_cells"] == []


def test_a_sharpe_of_exactly_the_minimum_passes_and_a_dd_stop_is_not_invalid() -> None:
    grid = _grid(cells={_key(f"{CAND}__BASE"): {"annual_sharpe": 0.30}})
    assert label_of(grid) == "OOS_SUPPORTED"
    # a 20% stop: flat, reconciled, valid; it fails condition 2 only
    stopped = _grid(
        cells={
            _key(f"{CAND}__HIGH_COST"): {
                "max_drawdown": 0.2,
                "stopped": True,
                "stop_reason": "max_drawdown_hard_stop",
            }
        }
    )
    result = decide(stopped)
    assert result["label"] == "OOS_NOT_SUPPORTED"
    assert result["failed_conditions"] == [
        "2_no_candidate_cell_drawdown_at_least_limit"
    ]


@pytest.mark.parametrize("arm", protocol.ARMS)
@pytest.mark.parametrize("scenario", protocol.SCENARIOS)
@pytest.mark.parametrize(
    "broken",
    [
        {"ledger_reconciled": False},
        {"causality_passed": False},
        {"execution_valid": False},  # insolvent
        {"complete": False},
        {"terminal_flat": False},  # residual that is not dust
        {"invalid_reasons": ("terminal_position_not_flat",)},
        {"total_return": float("nan")},
    ],
)
def test_any_invalid_cell_makes_the_label_invalid(
    arm: str, scenario: str, broken: dict[str, Any]
) -> None:
    grid = _grid(cells={(arm, scenario): broken})
    assert label_of(grid) == "INVALID"
    assert decide(grid)["label"] == "INVALID"
    assert decide(grid)["invalid_cells"] == [f"{arm}__{scenario}"]
    assert not conditions(grid)["1_all_nine_cells_valid"]


def test_invalid_outranks_a_failed_condition_and_never_supports() -> None:
    grid = _grid(
        cells={
            _key(f"{CAND}__BASE"): {"total_return": -0.5},
            _key("CASH__BASE"): {"ledger_reconciled": False},
        }
    )
    assert label_of(grid) == "INVALID"


def test_terminal_dust_is_accepted_but_other_residuals_are_not() -> None:
    dust = {"terminal_flat": False, "terminal_dust_only": True}
    grid = _grid(cells={_key(f"{CAND}__BASE"): dust, _key("EW_HOLD_SPOT__BASE"): dust})
    assert grid[(CAND, "BASE")].valid
    assert label_of(grid) == "OOS_SUPPORTED"
    # a residual that is neither flat nor dust (the summary adds the reason) is invalid
    residual = {"terminal_flat": False, "terminal_dust_only": False}
    assert label_of(_grid(cells={_key(f"{CAND}__BASE"): residual})) == "INVALID"
    reasons = {**residual, "invalid_reasons": ("terminal_position_not_flat",)}
    assert label_of(_grid(cells={_key("CASH__DELAY_1BAR"): reasons})) == "INVALID"


def test_the_decision_refuses_an_incomplete_or_mislabelled_grid() -> None:
    grid = _grid()
    short = dict(grid)
    del short[("CASH", "BASE")]
    with pytest.raises(ValueError, match="frozen grid"):
        decide(short)
    extra = {**grid, ("X", "BASE"): _cell("X", "BASE")}
    with pytest.raises(ValueError, match="frozen grid"):
        decide(extra)
    swapped = dict(grid)
    swapped[("CASH", "BASE")] = _cell("CASH", "HIGH_COST")
    with pytest.raises(ValueError, match="does not match"):
        decide(swapped)


def test_the_interim_report_has_metrics_but_no_label_and_no_verdict() -> None:
    report = interim_report(_grid())
    assert report["label"] is None
    assert report["status"] == "INTERIM_NOT_A_DECISION"
    assert "conditions" not in report and "supported" not in report
    assert "bootstrap" in report and "calendar_year_returns" in report
    assert set(report["cells"]) == {
        f"{a}__{s}" for a in protocol.ARMS for s in protocol.SCENARIOS
    }


def test_the_combined_claim_needs_both_stages_supported() -> None:
    assert (
        combined_claim("OOS_SUPPORTED", "OOS_SUPPORTED") == "OOS_AND_FORWARD_SUPPORTED"
    )
    for pair in (
        ("OOS_SUPPORTED", "OOS_NOT_SUPPORTED"),
        ("OOS_NOT_SUPPORTED", "OOS_SUPPORTED"),
        ("OOS_SUPPORTED", "INVALID"),
    ):
        assert combined_claim(*pair) is None


def test_per_year_returns_and_the_bootstrap_are_reported_but_never_gate() -> None:
    grid = _grid(
        cells={
            _key(f"{CAND}__BASE"): {
                "calendar_year_returns": {"2018": 0.1, "2019": -0.02, "2020": 0.12},
            },
        }
    )
    result = decide(grid)
    assert result["calendar_year_returns"][f"{CAND}__BASE"] == {
        "2018": 0.1,
        "2019": -0.02,
        "2020": 0.12,
    }
    assert result["label"] == "OOS_SUPPORTED"  # unchanged by what is reported
    for key in ("excess_over_CASH", "excess_over_EW_HOLD_SPOT"):
        assert result["bootstrap"][key]["gates"] is False
    # misaligned returns are reported as unavailable, not an error and not a verdict
    grid = _grid(cells={_key(f"{CAND}__BASE"): {"returns": (0.1,) * 3}})
    assert bootstrap_excess(grid)["excess_over_CASH"]["available"] is False
    assert bootstrap_excess(_grid())["excess_over_CASH"]["available"] is True
    grid = _grid(cells={_key("CASH__BASE"): {"returns": (0.0,) * 7}})
    assert bootstrap_excess(grid)["excess_over_CASH"]["available"] is False
    assert label_of(grid) == "OOS_SUPPORTED"


# ================================================================== bootstrap


def _slow_bootstrap(
    x: np.ndarray,
    *,
    block_mean: int,
    resamples: int,
    seed: int,
    chunk: int,
    confidence: float,
) -> float:
    """The stationary bootstrap written from its definition with Python loops."""
    rng = np.random.default_rng(seed)
    n = len(x)
    means: list[float] = []
    while len(means) < resamples:
        m = min(chunk, resamples - len(means))
        new = rng.random((m, n)) < 1.0 / block_mean
        new[:, 0] = True
        starts = rng.integers(0, n, size=(m, n))
        for r in range(m):
            index: list[int] = []
            current = 0
            for t in range(n):
                current = int(starts[r, t]) if new[r, t] else (current + 1) % n
                index.append(current)
            means.append(float(np.mean(x[index])))
    return float(np.quantile(means, 1.0 - confidence))


def test_the_vectorised_bootstrap_equals_the_definition_with_the_same_stream() -> None:
    x = np.sin(np.arange(37) * 1.3) * 0.01 + np.arange(37) * 0.0002
    for block_mean in (1, 5, 42):
        got = stationary_bootstrap(
            x, block_mean=block_mean, resamples=45, seed=7, chunk=16
        )
        want = _slow_bootstrap(
            x, block_mean=block_mean, resamples=45, seed=7, chunk=16, confidence=0.95
        )
        assert got["lower_bound"] == pytest.approx(want, abs=1e-15), block_mean
        assert got["mean"] == pytest.approx(float(x.mean()))


def test_the_bootstrap_known_cases() -> None:
    # a constant series: every resample mean is that constant
    flat = stationary_bootstrap(np.full(50, 0.003), resamples=200)
    assert flat["lower_bound"] == pytest.approx(0.003, abs=1e-15)
    assert flat["mean"] == pytest.approx(0.003)
    assert flat["block_mean_bars"] == 42 and flat["seed"] == 20261120
    assert flat["resamples"] == 200 and flat["gates"] is False
    # frozen parameters, fixed seed: a pinned regression value
    series = np.array(
        [0.01, -0.02, 0.03, 0.0, 0.015, -0.005, 0.02, 0.01, -0.01, 0.025, 0.0, 0.005]
    )
    got = stationary_bootstrap(series)
    assert got["n_bars"] == 12 and got["resamples"] == 10_000
    assert got["lower_bound"] == pytest.approx(0.005416666666666667, abs=1e-12)
    slow = _slow_bootstrap(
        series,
        block_mean=42,
        resamples=10_000,
        seed=20261120,
        chunk=protocol.BOOTSTRAP_CHUNK,
        confidence=0.95,
    )
    assert got["lower_bound"] == pytest.approx(slow, abs=1e-15)
    assert stationary_bootstrap(series) == got  # deterministic
    # iid normal data with block 1: the bound is the textbook one-sided bound
    rng = np.random.default_rng(1)
    noise = rng.normal(0.0, 1.0, 400)
    iid = stationary_bootstrap(noise, block_mean=1, resamples=4000, seed=3)
    textbook = float(noise.mean()) - 1.645 * float(noise.std(ddof=1)) / np.sqrt(400)
    assert iid["lower_bound"] == pytest.approx(textbook, abs=0.015)
    # a clearly positive series has a positive bound, a clearly negative one a negative
    drift = stationary_bootstrap(np.full(200, 0.001) + rng.normal(0, 1e-4, 200))
    assert drift["lower_bound"] > 0.0
    assert stationary_bootstrap(-np.abs(rng.normal(1, 0.1, 100)))["lower_bound"] < 0.0


def test_the_bootstrap_refuses_bad_input() -> None:
    with pytest.raises(ValueError, match="at least two finite"):
        stationary_bootstrap(np.array([0.1]))
    with pytest.raises(ValueError, match="at least two finite"):
        stationary_bootstrap(np.array([0.1, float("nan")]))
    with pytest.raises(ValueError, match="invalid bootstrap"):
        stationary_bootstrap(np.ones(5), block_mean=0)
    with pytest.raises(ValueError, match="invalid bootstrap"):
        stationary_bootstrap(np.ones(5), confidence=1.0)


# ========================================================= end-to-end synthetic runs


def _run(
    tmp_path: Path,
    dataset: Any,
    window: data.StageWindow,
    *,
    mode: str = "decision",
    name: str = "out",
    run_date: Any = None,
) -> dict[str, Any]:
    return runner.run_stage(
        dataset,
        window=window,
        output_dir=tmp_path / name,
        expected_protocol_digest=protocol.protocol_digest(),
        mode=mode,
        code_commit="a" * 40,
        run_date=run_date,
    )


def test_a_small_backcast_run_covers_every_cell_through_the_shared_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = _small_window(monkeypatch, "BACKCAST")
    dataset = _assemble(window)
    seen: list[OosCell] = []
    result = runner.run_stage(
        dataset,
        window=window,
        output_dir=tmp_path / "out",
        expected_protocol_digest=protocol.protocol_digest(),
        code_commit="b" * 40,
        on_cell=seen.append,
    )
    out = tmp_path / "out"
    assert len(seen) == 9
    cell_files = sorted((out / "cells").glob("*.json"))
    assert len(cell_files) == 9
    cell = json.loads(
        (out / "cells" / "TS_TREND_LONG_SPOT__BASE.json").read_text("utf-8")
    )
    assert cell["causality_passed"] and cell["ledger_reconciled"] and cell["cell_valid"]
    assert len(cell["returns"]) == cell["n_returns"] > 100
    decision_file = json.loads((out / "decision.json").read_text("utf-8"))
    assert decision_file == json.loads(json.dumps(result, default=str))
    assert result["label"] in protocol.LABELS and result["label"] != "INVALID"
    assert result["stage"] == "BACKCAST" and result["mode"] == "decision"
    assert result["protocol_id"] == protocol.PROTOCOL_ID
    assert result["protocol_digest"] == protocol.protocol_digest()
    assert result["lowturn_protocol_digest"] == LOWTURN_DIGEST_PIN
    assert result["code_commit"] == "b" * 40
    assert result["dataset_id"] == dataset.dataset_id
    assert result["replay_start_index"] == 1085
    assert result["replay_stop_index"] == window.last_row_index
    assert any("not point-in-time" in text for text in result["disclosures"])
    manifest = json.loads((out / runner.MANIFEST_NAME).read_text("utf-8"))
    assert manifest["label"] == result["label"]
    assert set(manifest["causality_probes"]) == {
        f"{a}__lag{lag}" for a in protocol.ARMS for lag in (0, 1)
    }
    assert all(p["passed"] for p in manifest["causality_probes"].values())
    # CASH never trades; the candidate trades; the controls hold the same BASE start
    assert result["cells"]["CASH__BASE"]["total_return"] == 0.0
    assert result["cells"]["TS_TREND_LONG_SPOT__BASE"]["turnover_total"] > 0.0
    # DELAY_1BAR lags the signal inputs, so the candidate differs from BASE
    assert (
        result["cells"]["TS_TREND_LONG_SPOT__DELAY_1BAR"]["total_return"]
        != result["cells"]["TS_TREND_LONG_SPOT__BASE"]["total_return"]
    )
    assert result["bootstrap"]["excess_over_CASH"]["available"] is True
    assert result["calendar_year_returns"]["TS_TREND_LONG_SPOT__BASE"].keys() == {
        "2018"
    }


def test_the_replay_never_holds_an_unavailable_leg(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    window = _small_window(monkeypatch, "BACKCAST")
    dataset = _assemble(window)
    signals = build_oos_signals(dataset)
    active = dataset.resolved_array("asset_active")
    for arm in ("EW_HOLD_SPOT", "TS_TREND_LONG_SPOT"):
        result = run_shared_portfolio_replay(
            dataset,
            LowturnPolicy(
                family=arm, signals=signals, start_index=window.first_decision_index
            ),
            start_index=window.first_decision_index,
            stop_index=window.last_row_index,
            initial_capital=protocol.INITIAL_CAPITAL,
            execution_cost=protocol.scenario_cost("BASE"),
            risk=PreTradeRisk(protocol.risk_config()),
            settle_terminal_position=True,
            capture_ledger_evidence=True,
        )
        assert result.ledger_evidence is not None
        for interval in result.ledger_evidence.intervals.intervals:
            qty = np.asarray(
                [float(Fraction(q)) for q in interval.exact_quantities_after]
            )
            assert np.all(qty[~active[interval.next_index]] == 0.0), (
                arm,
                interval.next_index,
            )
            assert np.all(qty[PERP] == 0.0)


def test_the_run_refuses_a_wrong_digest_an_existing_run_a_short_window_and_a_decision_gap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = _small_window(monkeypatch, "BACKCAST")
    dataset = _assemble(window)
    with pytest.raises(ValueError, match="frozen protocol digest"):
        runner.run_stage(
            dataset,
            window=window,
            output_dir=tmp_path / "a",
            expected_protocol_digest="0" * 64,
        )
    assert not (tmp_path / "a").exists()
    (tmp_path / "b").mkdir()
    (tmp_path / "b" / runner.DECISION_NAME).write_text("{}", encoding="utf-8")
    with pytest.raises(FileExistsError):
        _run(tmp_path, dataset, window, name="b")
    # a decision needs the whole stage window
    cut = window.last_row_close - timedelta(hours=40)
    partial = _assemble(window, limit=cut)
    with pytest.raises(data.StageDataError, match="whole stage window"):
        _run(tmp_path, partial, window, name="c")
    # too short even for an interim
    tiny = _assemble(window, limit=window.first_decision + timedelta(hours=40))
    with pytest.raises(data.StageDataError, match="two rebalance periods"):
        _run(tmp_path, tiny, window, mode="interim", name="d")


def test_a_small_forward_interim_writes_a_dated_report_with_no_label(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = _small_window(monkeypatch, "FORWARD")
    through = window.first_decision + timedelta(days=15)
    dataset = _assemble(window, limit=through, authorization_url=DATA_AUTH)
    result = _run(
        tmp_path,
        dataset,
        window,
        mode="interim",
        run_date=datetime(2026, 8, 17, tzinfo=UTC).date(),
    )
    out = tmp_path / "out"
    assert (out / "INTERIM_NOT_A_DECISION-20260817.json").is_file()
    assert not (out / "decision.json").exists()
    assert result["label"] is None and result["status"] == "INTERIM_NOT_A_DECISION"
    assert result["through_close"].startswith("2026-08-16T00:00")
    assert result["stage"] == "FORWARD" and result["mode"] == "interim"
    assert "conditions" not in result and "failed_conditions" not in result
    manifest = json.loads((out / runner.MANIFEST_NAME).read_text("utf-8"))
    assert manifest["label"] is None
    assert manifest["result_file"] == "INTERIM_NOT_A_DECISION-20260817.json"
    assert len(list((out / "cells").glob("*.json"))) == 9


def test_a_small_forward_decision_runs_on_the_authorized_dataset_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = _small_window(monkeypatch, "FORWARD")
    dataset = _assemble(window, authorization_url=DATA_AUTH)
    result = _run(tmp_path, dataset, window)
    assert result["stage"] == "FORWARD" and result["label"] in protocol.LABELS
    assert (tmp_path / "out" / "decision.json").is_file()
    assert result["replay_stop_index"] == window.last_row_index
    # the same arrays without the recorded authorization are refused by the runner
    bare = _without_identity(dataset)
    with pytest.raises(data.StageEmbargoError, match="no FORWARD authorization"):
        _run(tmp_path, bare, window, name="bare")


# ====================================================================== the tool


class _Rows:
    """Deterministic fake spot source with listing dates (no network, no files)."""

    def __init__(self, listed: dict[str, datetime | None]) -> None:
        self.listed = listed
        self.requests: list[tuple[str, int, int]] = []
        self._cache: dict[str, np.ndarray] = {}
        self._epoch = int(datetime(2018, 1, 1, tzinfo=UTC).timestamp() * 1_000)

    def _path(self, symbol: str) -> np.ndarray:
        if symbol not in self._cache:
            rng = np.random.default_rng(sum(map(ord, symbol)))
            steps = rng.normal(0.0004, 0.01, 100_000)
            self._cache[symbol] = 100.0 * np.exp(np.cumsum(steps))
        return self._cache[symbol]

    def read_klines(
        self, market: Any, symbol: str, interval: str, start_ms: int, end_ms: int
    ) -> list[KlineRow]:
        assert market is StoreMarket.SPOT and interval == "1h"
        self.requests.append((symbol, start_ms, end_ms))
        listed = self.listed[symbol]
        first = None if listed is None else int(listed.timestamp() * 1_000)
        if first is None or end_ms <= first:
            raise CarryCoverageError(
                f"archive is not published: {symbol}",
                {"kind": "archive_not_published", "status": 404},
            )
        path = self._path(symbol)
        rows: list[KlineRow] = []
        for open_ms in range(max(start_ms, first), end_ms, HOUR_MS):
            i = (open_ms - self._epoch) // HOUR_MS
            o, c = float(path[i - 1]), float(path[i])
            rows.append(
                KlineRow(
                    open_time_ms=open_ms,
                    open=Decimal(f"{o:.8f}"),
                    high=Decimal(f"{max(o, c) * 1.001:.8f}"),
                    low=Decimal(f"{min(o, c) * 0.999:.8f}"),
                    close=Decimal(f"{c:.8f}"),
                    volume=Decimal("1000000.0"),
                    close_time_ms=open_ms + HOUR_MS - 1,
                    quote_volume=Decimal("150000000.0"),
                    trades=100,
                    taker_buy_base_volume=Decimal("1.0"),
                    taker_buy_quote_volume=Decimal("1.0"),
                )
            )
        return rows


def _listings() -> dict[str, datetime | None]:
    """Mixed listings: three at the start, one mid-window (not on the 4h grid), three
    after the first decision or never."""
    out: dict[str, datetime | None] = {}
    for index, symbol in enumerate(protocol.ROSTER):
        if index < 3:
            out[symbol] = datetime(2018, 1, 1, tzinfo=UTC)
        elif index == 3:
            out[symbol] = datetime(2018, 2, 3, 5, tzinfo=UTC)  # mid 4h bar
        elif index == 4:
            out[symbol] = datetime(
                2018, 7, 5, 8, tzinfo=UTC
            )  # after the first decision
        else:
            out[symbol] = None
    return out


def _body(digest: str | None = None, *, pid: bool = True) -> str:
    lines = ["G4 authorization: trend long spot OOS"]
    if pid:
        lines.append(f"protocol_id: {protocol.PROTOCOL_ID}")
    lines.append(f"protocol_digest: {digest or protocol.protocol_digest()}")
    return "\n".join(lines) + "\n"


class _Gh:
    """Serves the G4 comment and the data-authorization comment by comment id."""

    def __init__(self, *, body: str | None = None, **g4_kwargs: Any) -> None:
        self.g4 = tierd_tests.FakeGh(url=AUTH, body=body or _body(), **g4_kwargs)
        self.data = tierd_tests.FakeGh(url=DATA_AUTH, body="owner relay comment\n")
        self.calls: list[list[str]] = []

    def __call__(self, args: Any) -> str:
        self.calls.append(list(args))
        target = self.data if str(args[1]).endswith("/7028416072") else self.g4
        return target(args)


def _tool(
    tmp_path: Path,
    argv: Sequence[str],
    *,
    gh: Any,
    now: datetime,
    clean: bool = True,
    rows: Any = None,
) -> tuple[int, str, str]:
    spot, perp = tmp_path / "spot.json", tmp_path / "perp.json"
    spot.write_bytes(tierd_tests._payload("spot"))
    perp.write_bytes(tierd_tests._payload("usds_m"))
    out, err = io.StringIO(), io.StringIO()
    code = driver.main(
        [
            *argv,
            "--archive-dir",
            str(tmp_path / "archives"),
            "--exchange-info-spot",
            str(spot),
            "--exchange-info-perp",
            str(perp),
        ],
        stdout=out,
        stderr=err,
        state=lambda: ("c" * 40, clean),
        rows_factory=lambda plan: rows if rows is not None else _Rows(_listings()),
        now=lambda: now,
        gh=gh,
    )
    return code, out.getvalue(), err.getvalue()


NOW = datetime(2026, 11, 20, 6, 30, tzinfo=UTC)


def _backcast_argv(tmp_path: Path, name: str = "out", url: str = AUTH) -> list[str]:
    return [
        "run",
        "--stage",
        "BACKCAST",
        "--output-dir",
        str(tmp_path / name),
        "--authorization-url",
        url,
    ]


def test_plan_prints_the_protocol_digest_and_both_windows() -> None:
    for stage in protocol.STAGES:
        out, err = io.StringIO(), io.StringIO()
        code = driver.main(
            ["plan", "--stage", stage],
            stdout=out,
            stderr=err,
            now=lambda: NOW,
        )
        assert code == driver.EXIT_OK, err.getvalue()
        document = json.loads(out.getvalue())
        assert document["protocol_id"] == "trend_long_spot_oos_v1"
        assert document["protocol_digest"] == protocol.protocol_digest()
        assert document["lowturn_protocol_digest"] == LOWTURN_DIGEST_PIN
        assert document["stage"] == stage
        assert document["window"] == document["windows"][stage]
        assert document["windows"]["BACKCAST"]["first_decision_index"] == 1085
        assert document["windows"]["BACKCAST"]["last_row_close"].startswith(
            "2021-01-01T04:00"
        )
        assert document["windows"]["FORWARD"]["last_row_close"].startswith(
            "2027-05-01T04:00"
        )
        assert document["final_not_before"] == "2027-05-02T00:00:00Z"
        assert document["latest_closed_bar"].startswith("2026-11-20T04:00")
        assert document["protocol"] == json.loads(
            json.dumps(protocol.protocol_payload(), sort_keys=True)
        )


def test_run_needs_an_authorization_url_a_clean_tree_and_a_new_output_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _small_window(monkeypatch, "BACKCAST")
    monkeypatch.setattr(
        driver.runner, "run_stage", lambda *a, **k: pytest.fail("must not run")
    )
    with pytest.raises(SystemExit):
        driver.main(["run", "--stage", "BACKCAST", "--output-dir", "x"])
    gh = _Gh()
    code, _, err = _tool(
        tmp_path, _backcast_argv(tmp_path, url="https://example.com/x"), gh=gh, now=NOW
    )
    assert code == driver.EXIT_REFUSED and "authorization_url" in err
    code, _, err = _tool(
        tmp_path, _backcast_argv(tmp_path), gh=gh, now=NOW, clean=False
    )
    assert code == driver.EXIT_REFUSED and "clean" in err
    (tmp_path / "out").mkdir()
    code, _, err = _tool(tmp_path, _backcast_argv(tmp_path), gh=gh, now=NOW)
    assert code == driver.EXIT_REFUSED and "write-once" in err


@pytest.mark.parametrize(
    "gh_factory",
    [
        lambda: _Gh(body=_body(pid=False)),
        lambda: _Gh(body=_body(digest=lowturn.protocol_digest())),
        lambda: _Gh(body=_body(digest="e" * 64)),
        lambda: _Gh(
            body=_body().replace(protocol.PROTOCOL_ID, protocol.PROTOCOL_ID + "x")
        ),
        lambda: _Gh(login="mallory", user_id=1),
    ],
)
def test_the_comment_must_be_the_owners_and_quote_this_protocol_id_and_digest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, gh_factory: Any
) -> None:
    _small_window(monkeypatch, "BACKCAST")
    monkeypatch.setattr(
        driver.runner, "run_stage", lambda *a, **k: pytest.fail("must not run")
    )
    code, _, err = _tool(tmp_path, _backcast_argv(tmp_path), gh=gh_factory(), now=NOW)
    assert code == driver.EXIT_UNAUTHENTICATED, err
    assert not (tmp_path / "out").exists()


def test_an_edited_comment_authorizes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _small_window(monkeypatch, "BACKCAST")
    spy_calls: list[Any] = []
    monkeypatch.setattr(driver.runner, "run_stage", lambda *a, **k: spy_calls.append(1))
    gh = _Gh()
    gh.g4.variants = {2: {"updated_at": "2026-11-21T00:00:00Z"}}
    code, _, err = _tool(tmp_path, _backcast_argv(tmp_path), gh=gh, now=NOW)
    assert code == driver.EXIT_UNAUTHENTICATED and "edited" in err
    assert not spy_calls and not (tmp_path / "out").exists()


def test_an_authenticated_backcast_run_writes_receipts_then_prints_the_label(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = _small_window(monkeypatch, "BACKCAST")
    rows = _Rows(_listings())
    gh = _Gh()
    code, out, err = _tool(
        tmp_path, _backcast_argv(tmp_path), gh=gh, now=NOW, rows=rows
    )
    assert code in (driver.EXIT_OK, driver.EXIT_INVALID), err
    printed = json.loads(out)
    out_dir = tmp_path / "out"
    decision = json.loads((out_dir / "decision.json").read_text("utf-8"))
    assert printed["label"] == decision["label"] and printed["stage"] == "BACKCAST"
    assert len(list((out_dir / "cells").glob("*.json"))) == 9
    data_receipt = json.loads((out_dir / driver.DATA_RECEIPT_NAME).read_text("utf-8"))
    assert data_receipt["dataset_id"] == decision["dataset_id"] == printed["dataset_id"]
    assert data_receipt["code_commit"] == decision["code_commit"] == "c" * 40
    assert data_receipt["protocol_id"] == protocol.PROTOCOL_ID
    assert data_receipt["exchange_info_sha256"]["spot"]
    assert "not point-in-time" in data_receipt["exchange_rules_note"]
    assert data_receipt["forward_data_authorization"] is None
    assert data_receipt["read_limit_close"].startswith("2018-07-23T04:00")
    coverage = data_receipt["listing_coverage"]
    assert coverage["BTCUSDT"]["first_bar_open"].startswith("2018-01-01T00:00")
    # the listing at 05:00 leaves a partial first bar: dropped, the 08:00 bar starts it
    assert coverage["SOLUSDT"]["first_listed_open"].startswith("2018-02-03T05:00")
    assert coverage["SOLUSDT"]["dropped_leading_partial_hours"] == 3
    assert coverage["SOLUSDT"]["first_bar_open"].startswith("2018-02-03T08:00")
    assert coverage["XRPUSDT"]["first_listed_open"].startswith("2018-07-05T08:00")
    assert coverage["XRPUSDT"]["unpublished_leading_months"] >= 6
    assert coverage["DOGEUSDT"]["first_listed_open"] is None
    assert coverage["DOTUSDT"]["first_listed_open"] is None
    run_receipt = json.loads((out_dir / driver.RUN_RECEIPT_NAME).read_text("utf-8"))
    assert run_receipt["label"] == decision["label"]
    assert run_receipt["authorization_receipt"]["comment_id"] == 1
    assert run_receipt["authorization_receipt"]["author_login"] == "shuntatsu"
    assert run_receipt["authorization_receipt"]["quoted_protocol_digest"] == (
        protocol.protocol_digest()
    )
    assert run_receipt["combined_claim"] is None
    # nothing past the stage's last row was requested, and no perpetual or funding row
    last = int(window.last_row_close.timestamp() * 1_000)
    assert max(end for _, _, end in rows.requests) <= last
    assert [c[1] for c in gh.calls] == [
        "repos/shuntatsu/trade_rl/issues/comments/1"
    ] * 2


def test_without_network_a_cache_miss_halts_and_no_fetcher_is_built(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _small_window(monkeypatch, "BACKCAST")
    built: list[Any] = []
    monkeypatch.setattr(
        driver,
        "_StageVisionFetcher",
        lambda *a, **k: built.append(1) or pytest.fail("net"),
    )
    spot, perp = tmp_path / "spot.json", tmp_path / "perp.json"
    spot.write_bytes(tierd_tests._payload("spot"))
    perp.write_bytes(tierd_tests._payload("usds_m"))
    out, err = io.StringIO(), io.StringIO()
    code = driver.main(
        [
            *_backcast_argv(tmp_path),
            "--archive-dir",
            str(tmp_path / "archives"),
            "--exchange-info-spot",
            str(spot),
            "--exchange-info-perp",
            str(perp),
            "--daily-fallback",
        ],
        stdout=out,
        stderr=err,
        state=lambda: ("c" * 40, True),
        now=lambda: NOW,
        gh=_Gh(),
    )
    assert code == driver.EXIT_HALTED, err.getvalue()
    assert "not in the offline archive cache" in err.getvalue()
    assert not built and not (tmp_path / "out" / "decision.json").exists()


def test_an_invalid_run_is_not_reported_as_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _small_window(monkeypatch, "BACKCAST")

    def fake(dataset: Any, **kwargs: Any) -> dict[str, Any]:
        (kwargs["output_dir"] / "decision.json").write_text("{}", encoding="utf-8")
        return {"label": "INVALID", "status": None, "failed_conditions": []}

    monkeypatch.setattr(driver.runner, "run_stage", fake)
    code, out, err = _tool(tmp_path, _backcast_argv(tmp_path), gh=_Gh(), now=NOW)
    assert code == driver.EXIT_INVALID and "INVALID" in err
    assert json.loads(out)["label"] == "INVALID"


def test_a_symbol_gap_after_listing_halts_instead_of_filling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _small_window(monkeypatch, "BACKCAST")

    class _Gappy(_Rows):
        def read_klines(
            self, market: Any, symbol: str, interval: str, s: int, e: int
        ) -> list[KlineRow]:
            rows = super().read_klines(market, symbol, interval, s, e)
            if symbol == "BTCUSDT" and e - s > 24 * HOUR_MS:
                return [r for r in rows if r.open_time_ms % (97 * HOUR_MS) != 0]
            return rows

    code, _, err = _tool(
        tmp_path, _backcast_argv(tmp_path), gh=_Gh(), now=NOW, rows=_Gappy(_listings())
    )
    assert code == driver.EXIT_REFUSED, err  # a documented gap: refused, never filled
    assert "unexplained missing hourly" in err
    assert "first_missing_open" in err  # the gap runs are reported
    assert not (tmp_path / "out" / "decision.json").exists()


def test_an_unpublished_archive_after_listing_is_not_a_not_yet_listed_market(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _small_window(monkeypatch, "BACKCAST")

    class _Hole(_Rows):
        def read_klines(
            self, market: Any, symbol: str, interval: str, s: int, e: int
        ) -> list[KlineRow]:
            if symbol == "ETHUSDT" and e > int(
                datetime(2018, 2, 15, tzinfo=UTC).timestamp() * 1_000
            ):
                raise CarryCoverageError("gone", {"kind": "archive_not_published"})
            return super().read_klines(market, symbol, interval, s, e)

    code, _, err = _tool(
        tmp_path, _backcast_argv(tmp_path), gh=_Gh(), now=NOW, rows=_Hole(_listings())
    )
    # detection stops at the first row; the reader then fails on the missing archive
    assert code == driver.EXIT_REFUSED and "gone" in err, err
    assert not (tmp_path / "out" / "decision.json").exists()


# ---- FORWARD through the tool: the final gate, the stop rule and the bound path


def _backcast_decision(
    tmp_path: Path, label: str = "OOS_SUPPORTED", **over: Any
) -> Path:
    document = {
        "protocol_id": protocol.PROTOCOL_ID,
        "protocol_digest": protocol.protocol_digest(),
        "stage": "BACKCAST",
        "mode": "decision",
        "label": label,
        **over,
    }
    path = tmp_path / "backcast-decision.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def _forward_argv(
    tmp_path: Path,
    *flags: str,
    decision_path: Path | None = None,
    with_decision: bool = True,
    name: str = "fwd",
) -> list[str]:
    argv = [
        "run",
        "--stage",
        "FORWARD",
        *flags,
        "--output-dir",
        str(tmp_path / name),
        "--authorization-url",
        AUTH,
    ]
    if with_decision:
        argv += [
            "--backcast-decision",
            str(decision_path or _backcast_decision(tmp_path)),
        ]
    return argv


def _no_run(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    calls: list[Any] = []
    monkeypatch.setattr(
        driver.runner,
        "run_stage",
        lambda *a, **k: calls.append(1) or pytest.fail("ran"),
    )
    return calls


def test_final_is_refused_before_the_scoring_instant_even_when_everything_else_is_valid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _no_run(monkeypatch)
    for when in (
        NOW,
        datetime(2027, 5, 1, 23, 59, 59, tzinfo=UTC),
        datetime(2027, 5, 2, 0, 0, 0, tzinfo=UTC) - timedelta(microseconds=1),
    ):
        gh = _Gh()
        code, _, err = _tool(
            tmp_path, _forward_argv(tmp_path, "--final"), gh=gh, now=when
        )
        assert code == driver.EXIT_REFUSED and "--final is refused before" in err
        assert gh.calls == []  # refused before any gh call, file or row
        assert not (tmp_path / "fwd").exists()


def test_final_is_allowed_from_the_scoring_instant_on_and_reads_the_whole_stage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = _small_window(monkeypatch, "FORWARD")
    # the small window is long over at a late `now`; the gate is the frozen instant
    monkeypatch.setattr(protocol, "FINAL_NOT_BEFORE", "2026-09-01T00:00:00Z")
    rows = _Rows({s: datetime(2018, 1, 1, tzinfo=UTC) for s in protocol.ROSTER})
    gh = _Gh()
    code, out, err = _tool(
        tmp_path,
        _forward_argv(tmp_path, "--final"),
        gh=gh,
        now=datetime(2026, 9, 1, tzinfo=UTC),
        rows=rows,
    )
    assert code in (driver.EXIT_OK, driver.EXIT_INVALID), err
    printed = json.loads(out)
    assert printed["mode"] == "decision" and printed["label"] in protocol.LABELS
    receipt = json.loads(
        (tmp_path / "fwd" / driver.DATA_RECEIPT_NAME).read_text("utf-8")
    )
    assert receipt["read_limit_close"] == window.last_row_close.isoformat()
    bound = receipt["forward_data_authorization"]
    assert bound["url"] == DATA_AUTH and bound["author_login"] == "shuntatsu"
    assert bound["embargo_policy"] == "ForwardEmbargoPolicy"
    assert receipt["backcast_decision"]["label"] == "OOS_SUPPORTED"
    run_receipt = json.loads(
        (tmp_path / "fwd" / driver.RUN_RECEIPT_NAME).read_text("utf-8")
    )
    both = run_receipt["label"] == "OOS_SUPPORTED"
    assert (run_receipt["combined_claim"] == "OOS_AND_FORWARD_SUPPORTED") is both
    # both comments were fetched: the run comment twice, the data authorization once
    ids = [c[1].rsplit("/", 1)[1] for c in gh.calls]
    assert sorted(ids) == ["1", "1", "7028416072"]
    last = int(window.last_row_close.timestamp() * 1_000)
    assert max(end for _, _, end in rows.requests) <= last


def test_interim_runs_before_the_scoring_instant_to_the_latest_closed_day_without_a_label(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _small_window(monkeypatch, "FORWARD")
    rows = _Rows({s: datetime(2018, 1, 1, tzinfo=UTC) for s in protocol.ROSTER})
    now = datetime(2026, 8, 20, 6, 30, tzinfo=UTC)
    code, out, err = _tool(
        tmp_path, _forward_argv(tmp_path, "--interim"), gh=_Gh(), now=now, rows=rows
    )
    assert code == driver.EXIT_OK, err
    printed = json.loads(out)
    assert printed["label"] is None and printed["status"] == "INTERIM_NOT_A_DECISION"
    assert printed["mode"] == "interim" and printed["combined_claim"] is None
    out_dir = tmp_path / "fwd"
    assert (out_dir / "INTERIM_NOT_A_DECISION-20260820.json").is_file()
    assert not (out_dir / "decision.json").exists()
    receipt = json.loads((out_dir / driver.DATA_RECEIPT_NAME).read_text("utf-8"))
    # up to the latest fully closed *day* (a day archive exists only after it closed)
    assert receipt["read_limit_close"] == "2026-08-20T00:00:00+00:00"
    last = int(datetime(2026, 8, 20, tzinfo=UTC).timestamp() * 1_000)
    assert max(end for _, _, end in rows.requests) <= last
    # an earlier explicit end is allowed, a later one is not
    code, out, err = _tool(
        tmp_path,
        _forward_argv(
            tmp_path, "--interim", "--through", "2026-08-18T00:00", name="f2"
        ),
        gh=_Gh(),
        now=now,
        rows=rows,
    )
    assert code == driver.EXIT_OK, err
    assert (
        json.loads((tmp_path / "f2" / driver.DATA_RECEIPT_NAME).read_text("utf-8"))[
            "read_limit_close"
        ]
        == "2026-08-18T00:00:00+00:00"
    )
    code, _, err = _tool(
        tmp_path,
        _forward_argv(
            tmp_path, "--interim", "--through", "2026-08-21T00:00", name="f3"
        ),
        gh=_Gh(),
        now=now,
        rows=rows,
    )
    assert code == driver.EXIT_REFUSED and "through" in err


def test_forward_mode_flags_and_the_stop_rule(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _small_window(monkeypatch, "FORWARD")
    _no_run(monkeypatch)
    gh = _Gh()

    def refused(argv: list[str], needle: str, *, now: datetime = NOW) -> None:
        code, _, err = _tool(tmp_path, argv, gh=gh, now=now)
        assert code == driver.EXIT_REFUSED and needle in err, err
        assert gh.calls == [] and not (tmp_path / "fwd").exists()

    # one of --final/--interim is required, never both
    refused(_forward_argv(tmp_path), "exactly one")
    with pytest.raises(SystemExit):
        _tool(tmp_path, _forward_argv(tmp_path, "--final", "--interim"), gh=gh, now=NOW)
    # FORWARD without the BACKCAST decision
    refused(
        _forward_argv(tmp_path, "--interim", with_decision=False), "--backcast-decision"
    )
    # BACKCAST was not supported: the line ended, nothing is scored (not even interim)
    for label in ("OOS_NOT_SUPPORTED", "INVALID"):
        decision_path = _backcast_decision(tmp_path, label)
        code, _, err = _tool(
            tmp_path,
            _forward_argv(tmp_path, "--interim", decision_path=decision_path),
            gh=gh,
            now=NOW,
        )
        assert code == driver.EXIT_REFUSED and "line ended" in err, err
        assert not (tmp_path / "fwd").exists()
    # a decision of another digest, stage or mode is not a BACKCAST decision
    for over in (
        {"protocol_digest": "0" * 64},
        {"stage": "FORWARD"},
        {"mode": "interim"},
        {"protocol_id": "other"},
    ):
        code, _, err = _tool(
            tmp_path,
            _forward_argv(
                tmp_path,
                "--interim",
                decision_path=_backcast_decision(tmp_path, **over),
            ),
            gh=gh,
            now=NOW,
        )
        assert code == driver.EXIT_REFUSED and "not a BACKCAST decision" in err, err
    # --through is interim-only; BACKCAST takes no mode flags
    refused(
        _forward_argv(tmp_path, "--final", "--through", "2026-08-18T00:00"),
        "--through is for --interim only",
        now=NOW,
    )
    for extra in ("--final", "--interim"):
        code, _, err = _tool(
            tmp_path, [*_backcast_argv(tmp_path), extra], gh=gh, now=NOW
        )
        assert code == driver.EXIT_REFUSED and "BACKCAST takes no" in err
    code, _, err = _tool(
        tmp_path,
        [
            *_backcast_argv(tmp_path),
            "--backcast-decision",
            str(_backcast_decision(tmp_path)),
        ],
        gh=gh,
        now=NOW,
    )
    assert code == driver.EXIT_REFUSED and "belongs to FORWARD" in err
    assert gh.calls == []


def test_the_forward_path_is_bound_to_the_owner_authorization_comment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _small_window(monkeypatch, "FORWARD")
    _no_run(monkeypatch)
    # the data-authorization comment must be the owner's
    gh = _Gh()
    gh.data = tierd_tests.FakeGh(url=DATA_AUTH, body="x\n", login="mallory", user_id=1)
    code, _, err = _tool(tmp_path, _forward_argv(tmp_path, "--interim"), gh=gh, now=NOW)
    assert code == driver.EXIT_UNAUTHENTICATED and "owner" in err
    assert not (tmp_path / "fwd").exists()
    # BACKCAST never fetches it
    gh = _Gh()
    _small_window(monkeypatch, "BACKCAST")
    monkeypatch.setattr(
        driver.runner,
        "run_stage",
        lambda *a, **k: {"label": "OOS_NOT_SUPPORTED", "status": None},
    )
    _tool(tmp_path, _backcast_argv(tmp_path, name="b1"), gh=gh, now=NOW)
    assert all(not c[1].endswith("7028416072") for c in gh.calls)


def test_the_g4_helpers_default_to_core7_and_accept_this_protocol_by_parameter() -> (
    None
):
    snap = g4.CommentSnapshot(
        comment_id=1,
        html_url=AUTH,
        login="shuntatsu",
        user_id=g4.OWNER_ID,
        body=_body(),
        body_sha256="0" * 64,
        updated_at="t",
        fetched_at="t",
    )
    g4.require_protocol_digest(snap, protocol.protocol_digest())
    g4.require_protocol_id(snap, protocol.PROTOCOL_ID)
    with pytest.raises(g4.G4AuthError):
        g4.require_protocol_digest(snap)  # the Core7 digest is not quoted
    with pytest.raises(g4.G4AuthError):
        g4.require_protocol_id(snap, "trend_long_spot_oos")  # not a whole token


def test_the_package_reuses_the_unchanged_lowturn_policy_module() -> None:
    assert LowturnPolicy is lowturn_policies.LowturnPolicy
    assert runner.LowturnPolicy is lowturn_policies.LowturnPolicy
    assert callable(family_targets)


# ============================================================ the coverage step


def _ms(*parts: int) -> int:
    return int(datetime(*parts, tzinfo=UTC).timestamp() * 1_000)


class _CovRows(_Rows):
    """Spot rows with an exchange-wide gap, own gaps, a zero-volume row and fills."""

    def __init__(self, listed: dict[str, datetime | None]) -> None:
        super().__init__(listed)
        shared = {_ms(2018, 3, 10, h) for h in (2, 3, 4)}
        self.skip: dict[str, set[int]] = {symbol: set(shared) for symbol in listed}
        self.skip["BTCUSDT"].add(_ms(2018, 4, 1, 10))
        self.skip["BNBUSDT"] |= {_ms(2018, 5, 2, 0), _ms(2018, 5, 2, 1)}
        self.zero = {("BNBUSDT", _ms(2018, 6, 1, 6))}
        self.fail: dict[str, Exception] = {}
        self.placeholders = [
            ("ETHUSDT", _ms(2018, 3, 3, 14), _ms(2018, 3, 3, 14) + 3_599_999, "u/eth")
        ]

    def read_klines(
        self, market: Any, symbol: str, interval: str, s: int, e: int
    ) -> list[KlineRow]:
        if symbol in self.fail and s > _ms(2018, 3, 1):
            raise self.fail[symbol]
        rows = super().read_klines(market, symbol, interval, s, e)
        kept = [r for r in rows if r.open_time_ms not in self.skip[symbol]]
        return [
            replace(r, volume=Decimal(0))
            if (symbol, r.open_time_ms) in self.zero
            else r
            for r in kept
        ]

    def evidence(self) -> list[dict[str, Any]]:
        return [
            {
                "symbol": symbol,
                "url": f"https://x/{symbol}-{i}.zip",
                "archive_sha256": "a" * 64,
                "checksum_file_sha256": "b" * 64,
            }
            for symbol in ("BTCUSDT", "ETHUSDT")
            for i in range(2)
        ]

    def fills(self) -> list[dict[str, object]]:
        return [
            {
                "market": "spot",
                "symbol": "SOLUSDT",
                "open_time": datetime(2018, 4, 5, h, tzinfo=UTC).isoformat(),
                "source": "vision_daily",
                "archive_url": "https://x/SOLUSDT-2018-04-05.zip",
                "archive_sha256": "c" * 64,
            }
            for h in (7, 8)
        ]


def _cov_argv(tmp_path: Path, stage: str = "BACKCAST", name: str = "cov") -> list[str]:
    return ["coverage", "--stage", stage, "--output-dir", str(tmp_path / name)]


def _cov_tool(
    tmp_path: Path,
    argv: Sequence[str],
    *,
    rows: Any,
    gh: Any = None,
    now: datetime = NOW,
) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    code = driver.main(
        [*argv, "--archive-dir", str(tmp_path / "archives")],
        stdout=out,
        stderr=err,
        state=lambda: ("d" * 40, False),
        rows_factory=lambda plan: rows,
        now=lambda: now,
        gh=gh or _Gh(),
    )
    return code, out.getvalue(), err.getvalue()


def _no_economics(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*a: Any, **k: Any) -> Any:
        pytest.fail("coverage must not build signals, replay or return anything")

    monkeypatch.setattr(driver.runner, "run_stage", boom)
    monkeypatch.setattr(runner, "build_oos_signals", boom)
    monkeypatch.setattr(runner, "run_shared_portfolio_replay", boom)
    monkeypatch.setattr(runner, "decide", boom)


def test_coverage_reports_gaps_ranges_listing_fills_and_placeholders_without_refusing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _small_window(monkeypatch, "BACKCAST")
    _no_economics(monkeypatch)
    gh = _Gh()
    code, out, err = _cov_tool(
        tmp_path, _cov_argv(tmp_path), rows=_CovRows(_listings()), gh=gh
    )
    assert code == driver.EXIT_OK, err  # gaps are reported, never a refusal
    assert gh.calls == []  # BACKCAST needs no comment
    doc = json.loads((tmp_path / "cov" / "coverage.json").read_text("utf-8"))
    assert json.loads(out)["exchange_wide_gap_hours"] == 3
    assert doc["schema_version"] == "trend_oos_coverage_v1"
    assert doc["protocol_digest"] == protocol.protocol_digest()
    assert doc["code_commit"] == "d" * 40 and doc["working_tree_clean"] is False
    assert "no signal, replay, return" in doc["not_computed"]
    assert "label" not in doc and "cells" not in doc
    syms = doc["symbols"]
    # listing start
    assert syms["BTCUSDT"]["first_listed_open"].startswith("2018-01-01T00:00")
    assert syms["SOLUSDT"]["first_listed_open"].startswith("2018-02-03T05:00")
    assert syms["XRPUSDT"]["first_listed_open"].startswith("2018-07-05T08:00")
    assert syms["XRPUSDT"]["unpublished_leading_months"] >= 6
    assert syms["DOGEUSDT"]["first_listed_open"] is None
    # missing hours as contiguous ranges
    assert syms["BTCUSDT"]["missing_hours"] == 4
    assert syms["BTCUSDT"]["missing_ranges"] == [
        {"first_missing_open": "2018-03-10T02:00:00+00:00", "hours": 3},
        {"first_missing_open": "2018-04-01T10:00:00+00:00", "hours": 1},
    ]
    assert syms["BNBUSDT"]["missing_ranges"][1] == {
        "first_missing_open": "2018-05-02T00:00:00+00:00",
        "hours": 2,
    }
    assert syms["XRPUSDT"]["missing_hours"] == 0
    # exchange-wide: only the hours every listed symbol misses
    assert doc["exchange_wide_gaps"]["ranges"] == [
        {"first_missing_open": "2018-03-10T02:00:00+00:00", "hours": 3}
    ]
    # zero-volume row, placeholder rows, fallback fills and checksum results
    assert syms["BNBUSDT"]["zero_volume_regular_rows"] == [
        {"first_missing_open": "2018-06-01T06:00:00+00:00", "hours": 1}
    ]
    placeholder = syms["ETHUSDT"]["placeholder_rows"][0]
    assert placeholder["open_time"].startswith("2018-03-03T14:00")
    assert syms["BTCUSDT"]["placeholder_rows"] == []
    fill = syms["SOLUSDT"]["filled_by_daily_fallback"]
    assert fill["hours"] == 2
    assert fill["ranges"] == [
        {"first_missing_open": "2018-04-05T07:00:00+00:00", "hours": 2}
    ]
    assert fill["archives"] == [["https://x/SOLUSDT-2018-04-05.zip", "c" * 64]]
    assert syms["BTCUSDT"]["filled_by_daily_fallback"]["hours"] == 0
    assert syms["BTCUSDT"]["checksums"]["archives_verified"] == 2
    assert syms["BTCUSDT"]["checksums"]["failures"] == []
    assert syms["BNBUSDT"]["checksums"]["archives_verified"] == 0
    assert doc["declared_halts"][0][0].startswith("2023-03-24")


def test_coverage_never_reads_past_the_stage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = _small_window(monkeypatch, "BACKCAST")
    rows = _CovRows(_listings())
    _cov_tool(tmp_path, _cov_argv(tmp_path), rows=rows)
    assert max(end for _, _, end in rows.requests) <= int(
        window.last_row_close.timestamp() * 1_000
    )


def test_coverage_records_a_checksum_failure_and_an_unpublished_month_and_exits_3(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from trade_rl.integrations.binance.carry import CarryIntegrityHalt

    _small_window(monkeypatch, "BACKCAST")
    rows = _CovRows(_listings())
    rows.fail["ETHUSDT"] = CarryIntegrityHalt(
        "checksum mismatch", {"kind": "checksum_mismatch", "url": "u"}
    )
    rows.fail["BNBUSDT"] = CarryCoverageError(
        "archive is not published: u", {"kind": "archive_not_published"}
    )
    code, out, _ = _cov_tool(tmp_path, _cov_argv(tmp_path), rows=rows)
    assert code == driver.EXIT_HALTED  # an integrity fault is not a documented gap
    doc = json.loads((tmp_path / "cov" / "coverage.json").read_text("utf-8"))
    assert json.loads(out)["symbols_with_read_errors"] == ["ETHUSDT"]
    eth = doc["symbols"]["ETHUSDT"]
    assert eth["checksums"]["failures"]
    assert eth["unread_chunks"][0]["detail"] == "checksum_mismatch"
    bnb = doc["symbols"]["BNBUSDT"]
    assert bnb["unread_chunks"][0]["detail"] == "archive_not_published"
    assert bnb["missing_hours"] > 0  # reported, counted missing, not an error
    assert doc["symbols"]["BTCUSDT"]["missing_hours"] == 4


def test_coverage_refusals_and_forward_binding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _small_window(monkeypatch, "FORWARD")
    _no_economics(monkeypatch)
    old = {s: datetime(2018, 1, 1, tzinfo=UTC) for s in protocol.ROSTER}
    # FORWARD fetches the owner's data-authorization comment, which must be the owner's
    bad = _Gh()
    bad.data = tierd_tests.FakeGh(url=DATA_AUTH, body="x\n", login="mallory", user_id=1)
    rows = _Rows(old)
    code, _, err = _cov_tool(
        tmp_path, _cov_argv(tmp_path, "FORWARD"), rows=rows, gh=bad
    )
    assert code == driver.EXIT_UNAUTHENTICATED, err
    assert not (tmp_path / "cov").exists() and rows.requests == []
    gh = _Gh()
    now = datetime(2026, 8, 20, 6, 30, tzinfo=UTC)
    rows = _Rows(old)
    argv = _cov_argv(tmp_path, "FORWARD")
    code, _, err = _cov_tool(tmp_path, argv, rows=rows, gh=gh, now=now)
    assert code == driver.EXIT_OK, err
    assert [c[1].rsplit("/", 1)[1] for c in gh.calls] == ["7028416072"]
    doc = json.loads((tmp_path / "cov" / "coverage.json").read_text("utf-8"))
    assert doc["read_limit_close"] == "2026-08-20T00:00:00+00:00"
    bound = doc["forward_data_authorization"]
    assert (
        bound["embargo_policy"] == "ForwardEmbargoPolicy" and bound["url"] == DATA_AUTH
    )
    assert max(e for _, _, e in rows.requests) <= _ms(2026, 8, 20)
    # write-once, and BACKCAST takes no --through
    code, _, err = _cov_tool(tmp_path, argv, rows=rows, gh=gh, now=now)
    assert code == driver.EXIT_REFUSED and "write-once" in err
    code, _, err = _cov_tool(
        tmp_path,
        [*_cov_argv(tmp_path, name="c2"), "--through", "2018-07-10T00:00"],
        rows=_Rows(old),
    )
    assert code == driver.EXIT_REFUSED and "FORWARD only" in err


def test_the_real_coverage_source_logs_the_skipped_placeholder_row() -> None:
    from tests.integrations.market_store import support
    from trade_rl.integrations.market_store.contracts import Universe
    from trade_rl.integrations.market_store.plan import plan_ingestion

    start, end = datetime(2020, 12, 1, tzinfo=UTC), datetime(2021, 1, 1, tzinfo=UTC)
    plan = plan_ingestion(
        Universe(symbols=("BTCUSDT",), interval="1h", markets=(StoreMarket.SPOT,)),
        start=start,
        end=end,
        now=datetime(2026, 11, 20, tzinfo=UTC),
    )
    (spec,) = plan.specs
    lo, hi = _ms(2020, 12, 1), _ms(2021, 1, 1)
    placeholder_ms = support.PLACEHOLDER_OPEN_MS
    rows = support.kline_csv_rows("BTCUSDT", lo, hi, skip=frozenset({placeholder_ms}))
    at = next(i for i, r in enumerate(rows) if int(r[0]) > placeholder_ms)
    rows.insert(at, support.REAL_BTC_PLACEHOLDER)
    fetcher = support.FakeFetcher()
    fetcher.add(spec.url, support.zip_csv(rows))
    log: list[tuple[str, int, int, str]] = []
    source = driver._CoverageVisionSource(fetcher, plan, log)
    got = source.read_klines(StoreMarket.SPOT, "BTCUSDT", "1h", lo, hi)
    assert len(got) == (hi - lo) // HOUR_MS - 1  # the placeholder is not a row
    assert placeholder_ms not in {r.open_time_ms for r in got}
    assert log == [("BTCUSDT", placeholder_ms, 1_608_558_440_521, spec.url)]
    (evidence,) = source.evidence()
    assert evidence["symbol"] == "BTCUSDT" and len(evidence["archive_sha256"]) == 64


# ======================================== monthly fallback, declared halts, evidence


class _DailySource:
    """Fake daily-archive source: per day OK, unpublished, malformed or checksum fault."""

    def __init__(self, plan: Any, behaviour: dict[str, str], rows: _Rows) -> None:
        self.plan = plan
        self.behaviour = behaviour
        self.rows = rows
        self._evidence: list[dict[str, Any]] = []

    def read_klines(
        self, market: Any, symbol: str, interval: str, s: int, e: int
    ) -> list[KlineRow]:
        from trade_rl.integrations.binance import BinanceMarket
        from trade_rl.integrations.binance.carry import CarryIntegrityHalt
        from trade_rl.integrations.binance.vision import vision_kline_url

        day = datetime.fromtimestamp(s / 1_000, UTC)
        url = vision_kline_url(BinanceMarket.SPOT, symbol, interval, day)
        mode = self.behaviour.get(day.date().isoformat(), "ok")
        if mode == "404":
            raise CarryCoverageError("gone", {"kind": "archive_not_published"})
        if mode == "checksum":
            raise CarryIntegrityHalt("bad", {"kind": "checksum_mismatch", "url": url})
        self._evidence.append(
            {
                "url": url,
                "symbol": symbol,
                "archive_sha256": f"{day.day:064x}",
                "checksum_file_sha256": "b" * 64,
            }
        )
        if mode == "malformed":
            raise CarryIntegrityHalt(
                "bad row", {"kind": "malformed_archive", "url": url}
            )
        return self.rows.read_klines(market, symbol, interval, s, e)

    def evidence(self) -> list[dict[str, Any]]:
        return list(self._evidence)


class _BrokenMonth:
    """Primary source whose Feb 2018 monthly archive is unusable."""

    def __init__(self, error: Exception, rows: _Rows) -> None:
        self.error = error
        self.rows = rows

    def read_klines(
        self, market: Any, symbol: str, interval: str, s: int, e: int
    ) -> list[KlineRow]:
        if s >= _ms(2018, 2, 1) and e <= _ms(2018, 3, 1):
            raise self.error
        return self.rows.read_klines(market, symbol, interval, s, e)


def _month_source(
    error: Exception, behaviour: dict[str, str]
) -> tuple[Any, list[_DailySource]]:
    rows = _Rows({s: datetime(2018, 1, 1, tzinfo=UTC) for s in protocol.ROSTER})
    made: list[_DailySource] = []

    def factory(plan: Any) -> _DailySource:
        made.append(_DailySource(plan, behaviour, rows))
        return made[-1]

    source = driver._MonthFallbackSource(
        _BrokenMonth(error, rows), factory, halts=(), embargo=None
    )
    return source, made


@pytest.mark.parametrize(
    "error",
    [
        CarryCoverageError("404", {"kind": "archive_not_published"}),
        pytest.param(None, id="malformed"),
    ],
)
def test_an_unusable_monthly_archive_is_rebuilt_from_verified_daily_archives(
    error: Exception | None,
) -> None:
    from trade_rl.integrations.binance.carry import CarryIntegrityHalt

    error = error or CarryIntegrityHalt(
        "row not aligned", {"kind": "malformed_archive"}
    )
    source, _ = _month_source(error, {"2018-02-10": "404", "2018-02-11": "malformed"})
    rows = source.read_klines(
        StoreMarket.SPOT, "BTCUSDT", "1h", _ms(2018, 1, 15), _ms(2018, 3, 5)
    )
    opens = {r.open_time_ms for r in rows}
    days = {datetime.fromtimestamp(o / 1_000, UTC).date().isoformat() for o in opens}
    feb = {d for d in days if d.startswith("2018-02")}
    # every February day except the unpublished and the malformed one is present
    assert feb == {f"2018-02-{d:02d}" for d in range(1, 29)} - {
        "2018-02-10",
        "2018-02-11",
    }
    assert (
        _ms(2018, 1, 20) in opens and _ms(2018, 3, 2) in opens
    )  # other months untouched
    assert [r.open_time_ms for r in rows] == sorted(opens)
    # provenance per day: the day archive and its SHA-256, labelled as such
    fills = source.fills()
    assert len(fills) == 26 * 24
    sample = next(f for f in fills if str(f["open_time"]).startswith("2018-02-03T05"))
    assert sample["source"] == "vision_daily_unusable_month"
    assert sample["archive_url"].endswith("BTCUSDT-1h-2018-02-03.zip")
    assert sample["archive_sha256"] == f"{3:064x}"
    # the days that could not be used stay missing and are recorded with the reason
    unusable = {(u["day"], u["reason"]) for u in source.unusable_days()}
    assert unusable == {
        ("2018-02-10", "archive_not_published"),
        ("2018-02-11", "malformed_archive"),
    }


def test_a_checksum_fault_is_never_rescued_by_the_daily_fallback() -> None:
    from trade_rl.integrations.binance.carry import CarryIntegrityHalt

    bad = CarryIntegrityHalt("checksum", {"kind": "checksum_mismatch"})
    source, _ = _month_source(bad, {})
    with pytest.raises(CarryIntegrityHalt, match="checksum"):
        source.read_klines(
            StoreMarket.SPOT, "BTCUSDT", "1h", _ms(2018, 1, 15), _ms(2018, 3, 5)
        )
    # and a checksum fault on a day archive of a rebuilt month still halts
    source, _ = _month_source(
        CarryCoverageError("404", {"kind": "archive_not_published"}),
        {"2018-02-04": "checksum"},
    )
    with pytest.raises(CarryIntegrityHalt, match="bad"):
        source.read_klines(
            StoreMarket.SPOT, "BTCUSDT", "1h", _ms(2018, 2, 1), _ms(2018, 3, 1)
        )


HALT = ("2018-03-10T02:00:00Z", "2018-03-10T05:00:00Z")
HALT_EVIDENCE = "e" * 64


class _OutageRows(_Rows):
    """Every symbol misses the same hours (an exchange-wide outage)."""

    def __init__(
        self,
        listed: dict[str, datetime | None],
        missing: set[int],
        only: str | None = None,
    ) -> None:
        super().__init__(listed)
        self.missing = missing
        self.only = only

    def read_klines(
        self, market: Any, symbol: str, interval: str, s: int, e: int
    ) -> list[KlineRow]:
        rows = super().read_klines(market, symbol, interval, s, e)
        if self.only is not None and symbol == self.only:
            return rows
        return [r for r in rows if r.open_time_ms not in self.missing]


def _outage(only: str | None = None) -> _OutageRows:
    gone = {_ms(2018, 3, 10, h) for h in (2, 3, 4)}
    return _OutageRows(_listings(), gone, only)


def _load(monkeypatch: pytest.MonkeyPatch, rows: Any) -> Any:
    window = _small_window(monkeypatch, "BACKCAST")
    limit = window.last_row_close
    dataset, coverage = driver.load_stage_dataset(
        rows, rows, window=window, limit_close=limit, rules=RULES
    )
    return window, dataset, coverage


def test_an_undeclared_gap_refuses_and_a_declared_exchange_wide_halt_does_not(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(CarryCoverageError, match="unexplained missing hourly"):
        _load(monkeypatch, _outage())
    monkeypatch.undo()
    monkeypatch.setattr(protocol, "DECLARED_HALTS", (HALT,))
    monkeypatch.setattr(protocol, "COVERAGE_EVIDENCE_SHA256", HALT_EVIDENCE)
    _, dataset, _ = _load(monkeypatch, _outage())
    assert dataset.n_bars > 1000


def test_a_declared_halt_is_masked_like_spot_halts_and_signals_stay_finite_after_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(protocol, "DECLARED_HALTS", (HALT,))
    monkeypatch.setattr(protocol, "COVERAGE_EVIDENCE_SHA256", HALT_EVIDENCE)
    rows = _outage()
    window, dataset, _ = _load(monkeypatch, rows)
    # bars [00:00, 04:00) (hours 2 and 3 missing) and [04:00, 08:00) (hour 4 missing)
    first = window.row_index(datetime(2018, 3, 10, 4, tzinfo=UTC))
    tradable = np.asarray(dataset.tradable)
    active = dataset.resolved_array("asset_active")
    btc = 0
    assert not tradable[first, btc] and not tradable[first + 1, btc]
    assert tradable[first - 1, btc] and tradable[first + 2, btc]
    assert active[first, btc] and active[first + 1, btc]  # priced, only halted
    # the mark is the preceding published close (hour 01:00's close), nothing invented
    path = rows._path("BTCUSDT")
    epoch = rows._epoch
    hour = lambda h: (_ms(2018, 3, 10, h) - epoch) // HOUR_MS  # noqa: E731
    assert np.asarray(dataset.close)[first, btc] == pytest.approx(
        float(path[hour(1)]), rel=1e-7
    )
    # the same mask path as SPOT_HALTS: identity records the halts, and the check that
    # every bar touching a halt is non-tradable is the Tier D one
    payload = json.loads(dataset.identity_payload_json or "{}")
    assert [HALT[0], HALT[1]] in [
        [a.replace("+00:00", "Z"), b.replace("+00:00", "Z")]
        for a, b in payload["spot_halts"]
    ]
    assert payload["halted_bar_marks"] == (
        "preceding_published_close_non_tradable_zero_volume"
    )
    signals = build_oos_signals(dataset)
    # unusable (weight 0) only on the halted rows ...
    assert not signals.usable[first, btc] and np.isnan(signals.r30[first, btc])
    # ... and every window that spans the halt keeps a finite value afterwards
    after = first + 2
    assert signals.usable[after, btc]
    assert np.isfinite(signals.vol60[after, btc]) and np.isfinite(
        signals.r30[after, btc]
    )
    assert np.isfinite(signals.vol60[after + 300, btc])
    target = family_targets("TS_TREND_LONG_SPOT", signals, after + 40)
    assert np.all(np.isfinite(target))


def test_a_declared_halt_that_is_not_exchange_wide_refuses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(protocol, "DECLARED_HALTS", (HALT,))
    monkeypatch.setattr(protocol, "COVERAGE_EVIDENCE_SHA256", HALT_EVIDENCE)
    with pytest.raises(data.StageDataError, match="not exchange-wide: ETHUSDT"):
        _load(monkeypatch, _outage(only="ETHUSDT"))


def test_the_declared_halts_and_their_evidence_are_bound_to_the_digest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = protocol.protocol_digest()
    assert protocol.stage_halt_intervals() == tuple(protocol.SPOT_HALTS)
    monkeypatch.setattr(protocol, "DECLARED_HALTS", (HALT,))
    monkeypatch.setattr(protocol, "COVERAGE_EVIDENCE_SHA256", HALT_EVIDENCE)
    with_halt = protocol.protocol_digest()
    assert with_halt != base
    merged = protocol.stage_halt_intervals()
    assert merged[0][0] == datetime(2018, 3, 10, 2, tzinfo=UTC) and len(merged) == 2
    assert protocol.protocol_payload()["data"]["declared_halts"]["intervals"] == [
        list(HALT)
    ]  # type: ignore[index]
    monkeypatch.setattr(protocol, "COVERAGE_EVIDENCE_SHA256", "f" * 64)
    assert protocol.protocol_digest() not in (base, with_halt)
    monkeypatch.setattr(
        protocol,
        "DECLARED_HALTS",
        (HALT, ("2018-03-10T04:00:00Z", "2018-03-10T06:00:00Z")),
    )
    with pytest.raises(protocol.ProtocolBindingError, match="disjoint"):
        protocol.protocol_digest()
    monkeypatch.setattr(protocol, "DECLARED_HALTS", (HALT,))
    for bad in (None, "xyz"):
        monkeypatch.setattr(protocol, "COVERAGE_EVIDENCE_SHA256", bad)
        with pytest.raises(protocol.ProtocolBindingError, match="SHA-256"):
            protocol.protocol_digest()
    monkeypatch.setattr(protocol, "COVERAGE_EVIDENCE_SHA256", HALT_EVIDENCE)
    monkeypatch.setattr(
        protocol, "DECLARED_HALTS", (("2018-03-10T02:30:00Z", "2018-03-10T05:00:00Z"),)
    )
    with pytest.raises(protocol.ProtocolBindingError, match="invalid"):
        protocol.protocol_digest()


def test_halts_from_coverage_emits_only_the_exchange_wide_ranges_with_the_file_sha(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _small_window(monkeypatch, "BACKCAST")
    code, _, err = _cov_tool(tmp_path, _cov_argv(tmp_path), rows=_CovRows(_listings()))
    assert code == driver.EXIT_OK, err
    path = tmp_path / "cov" / "coverage.json"
    out, err_io = io.StringIO(), io.StringIO()
    assert (
        driver.main(
            ["halts-from-coverage", "--coverage", str(path)], stdout=out, stderr=err_io
        )
        == 0
    )
    result = json.loads(out.getvalue())
    assert (
        result["coverage_json_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    )
    assert result["declared_halts"] == [
        ["2018-03-10T02:00:00Z", "2018-03-10T05:00:00Z"]
    ]
    assert result["declared_hours"] == 3
    # what is not exchange-wide stays a gap and would still refuse the run
    left = result["gaps_that_would_remain_and_refuse_the_run"]
    assert left["BTCUSDT"] == [
        {"first_missing_open": "2018-04-01T10:00:00+00:00", "hours": 1}
    ]
    assert left["BNBUSDT"][0]["hours"] == 2 and "ETHUSDT" not in left
    # a report with read errors is refused, and a non-coverage file too
    bad = tmp_path / "bad.json"
    document = json.loads(path.read_text("utf-8"))
    document["symbols"]["ETHUSDT"]["errors"] = [{"kind": "integrity"}]
    bad.write_text(json.dumps(document), encoding="utf-8")
    err2 = io.StringIO()
    assert (
        driver.main(
            ["halts-from-coverage", "--coverage", str(bad)],
            stdout=io.StringIO(),
            stderr=err2,
        )
        == driver.EXIT_REFUSED
    )
    assert "read errors" in err2.getvalue()
    other = tmp_path / "other.json"
    other.write_text("{}", encoding="utf-8")
    assert (
        driver.main(
            ["halts-from-coverage", "--coverage", str(other)],
            stdout=io.StringIO(),
            stderr=io.StringIO(),
        )
        == driver.EXIT_REFUSED
    )


def test_the_known_placeholder_covers_the_symbols_seen_in_the_coverage_run() -> None:
    from trade_rl.integrations.market_store import parse

    for symbol in ("LTCUSDT", "AVAXUSDT", "DOGEUSDT", "DOTUSDT", "LINKUSDT", "BTCUSDT"):
        assert symbol in parse.KNOWN_PLACEHOLDER_SYMBOLS
    assert "XYZUSDT" not in parse.KNOWN_PLACEHOLDER_SYMBOLS
