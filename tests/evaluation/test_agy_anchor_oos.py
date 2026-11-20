"""Synthetic tests of the out-of-sample test ``agy_anchor_realistic_v1``.

Nothing here reads market data, the network or an economic result: every dataset is a seeded
synthetic fixture and every expected value is hand-computed or derived from a slow reference.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from trade_rl.artifacts import canonical_json_bytes
from trade_rl.data.source import RawMarketSeries
from trade_rl.evaluation.agy_anchor_oos import data, protocol, runner
from trade_rl.evaluation.core7_mtf_study import tierd

HOUR_MS = 3_600_000
BAR = timedelta(hours=4)
RULES = {
    symbol: (tierd.LegRules(1e-5, 5.0), tierd.LegRules(1e-3, 10.0))
    for symbol in protocol.ROSTER
}
REPLAY_BARS = 120


def _iso(moment: datetime) -> str:
    return moment.strftime("%Y-%m-%dT%H:%M:%SZ")


def _small_table(load_start: datetime) -> dict[str, str]:
    """A window whose first decision is the frozen 540-row warm-up, then 120 bars."""
    first = load_start + 541 * BAR
    last = first + REPLAY_BARS * BAR
    return {
        "load_start": _iso(load_start),
        "first_decision": _iso(first),
        "last_decision_bar": _iso(last),
        "settles": _iso(last + BAR),
    }


SMALL_LOADS = {
    protocol.STAGE_DEV: datetime(2021, 10, 1, tzinfo=UTC),
    protocol.STAGE_BACKCAST: datetime(2020, 1, 1, tzinfo=UTC),
    protocol.STAGE_FORWARD: datetime(2026, 5, 15, tzinfo=UTC),
}


def small_window(monkeypatch: pytest.MonkeyPatch, stage: str) -> data.StageWindow:
    monkeypatch.setitem(protocol.STAGE_WINDOWS, stage, _small_table(SMALL_LOADS[stage]))
    return data.frozen_window(stage)


def _leg(
    window: data.StageWindow,
    *,
    first_row: int,
    seed: int,
    perp: bool,
    limit: datetime | None = None,
    drift: float = 0.0004,
    sigma: float = 0.01,
    volume: float = 1.5e8,
) -> data.HourlyLeg:
    """A complete hourly series (close-stamped) from the 4h bar of ``first_row``."""
    last_row = window.row_index(limit or window.last_row_close)
    n_hours = (last_row + 1 - first_row) * 4
    first_open = window.load_start + timedelta(hours=4 * first_row)
    stamps = np.datetime64(first_open.replace(tzinfo=None), "ns") + (
        np.arange(n_hours) + 1
    ) * np.timedelta64(1, "h")
    rng = np.random.default_rng(seed)
    close = 100.0 * np.exp(np.cumsum(rng.normal(drift, sigma, n_hours)))
    open_ = np.concatenate([[close[0]], close[:-1]])
    quote = rng.uniform(volume, 2 * volume, n_hours)
    event = (np.arange(n_hours) % 8) == 7
    series = RawMarketSeries(
        timestamps=stamps,
        open=open_,
        high=np.maximum(open_, close) * 1.002,
        low=np.minimum(open_, close) * 0.998,
        close=close,
        volume=quote,
        funding_rate=np.where(event, rng.normal(1e-4, 3e-5, n_hours), 0.0)
        if perp
        else np.zeros(n_hours),
        funding_available=event if perp else np.zeros(n_hours, dtype=bool),
        funding_event_count=event.astype(np.int32)
        if perp
        else np.zeros(n_hours, dtype=np.int32),
        tradable=np.ones(n_hours, dtype=bool),
    )
    return data.HourlyLeg(
        series=series, taker_buy_quote=quote * rng.uniform(0.3, 0.7, n_hours)
    )


#: ``(spot first row, perp first row)`` per roster index; ``None`` is no market at all.
LISTINGS: dict[int, tuple[int | None, int | None]] = {
    0: (0, 0),
    1: (0, 0),
    2: (0, 0),
    3: (0, 0),
    4: (0, 0),
    5: (100, 150),  # perp lists later: activation 150, available from row 330
    6: (300, 600),  # both listed, but available only from row 780 (after the replay)
    7: (0, None),  # spot only
    8: (None, 0),  # perp only
    9: (None, None),
    10: (None, None),
}


def leg_maps(
    window: data.StageWindow, *, limit: datetime | None = None
) -> tuple[dict[str, Any], dict[str, Any]]:
    spot: dict[str, Any] = {}
    perp: dict[str, Any] = {}
    for index, symbol in enumerate(protocol.ROSTER):
        spot_row, perp_row = LISTINGS[index]
        spot[symbol] = (
            None
            if spot_row is None
            else _leg(
                window, first_row=spot_row, seed=10 + index, perp=False, limit=limit
            )
        )
        perp[symbol] = (
            None
            if perp_row is None
            else _leg(
                window, first_row=perp_row, seed=50 + index, perp=True, limit=limit
            )
        )
    return spot, perp


def truncate_leg(leg: Any, limit: datetime) -> Any:
    """The same leg cut at a read limit (a leg not yet listed there disappears)."""
    if leg is None:
        return None
    series = leg.series
    keep = int(
        np.searchsorted(
            np.asarray(series.timestamps).astype("datetime64[ns]"),
            np.datetime64(limit.replace(tzinfo=None), "ns"),
            side="right",
        )
    )
    if keep < 8:
        return None
    return data.HourlyLeg(
        RawMarketSeries(
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
        ),
        leg.taker_buy_quote[:keep],
    )


def assemble(
    window: data.StageWindow,
    *,
    limit: datetime | None = None,
    spot: dict[str, Any] | None = None,
    perp: dict[str, Any] | None = None,
    authorization_url: str | None = None,
) -> data.StageData:
    end = limit or window.last_row_close
    if spot is None or perp is None:
        spot, perp = leg_maps(window, limit=end)
    return data.assemble_stage_dataset(
        window,
        spot,
        perp,
        rules=RULES,
        limit_close=end,
        halt_intervals=(),
        data_authorization_url=authorization_url,
    )


@pytest.fixture(autouse=True)
def _no_real_halts(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(protocol, "DECLARED_HALTS", ())
    monkeypatch.setattr(protocol, "COVERAGE_EVIDENCE_SHA256", None)


# ============================================================ protocol and bindings


def _payload_digest() -> str:
    return hashlib.sha256(canonical_json_bytes(protocol.protocol_payload())).hexdigest()


def test_the_digest_is_stable_and_binds_the_fixture_the_port_and_the_margin_rate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    digest = protocol.protocol_digest()
    assert len(digest) == 64 and digest == protocol.protocol_digest()
    assert digest == _payload_digest()
    payload = protocol.protocol_payload()
    assert payload["protocol_id"] == "agy_anchor_realistic_v1"
    signal = payload["signal"]
    assert isinstance(signal, dict)
    assert signal["parity_fixture_sha256"] == protocol.FIXTURE_SHA256
    assert signal["port_source_digest"] == protocol.PORT_SOURCE_DIGEST
    cash = payload["cash_account"]
    assert isinstance(cash, dict) and cash["initial_margin_rate"] == 0.4
    for name, value in (
        ("FIXTURE_SHA256", "1" * 64),
        ("PORT_SOURCE_DIGEST", "2" * 64),
        ("INITIAL_MARGIN_RATE", 0.5),
        ("REBALANCE_EVERY_BARS", 6),
        ("MIN_BASE_SHARPE", 0.3),
        ("BOOTSTRAP_SEED", 1),
    ):
        with monkeypatch.context() as patch:
            patch.setattr(protocol, name, value)
            assert _payload_digest() != digest, name


def test_a_changed_port_is_detected_by_the_digest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protocol.verify_port_binding()
    monkeypatch.setattr(protocol, "PORT_SOURCE_DIGEST", "3" * 64)
    with pytest.raises(protocol.ProtocolBindingError, match="signal changed"):
        protocol.protocol_digest()


def test_the_port_digest_hashes_every_file_and_ignores_line_endings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from trade_rl.strategies.agy_anchor import targets

    source = Path(targets.__file__).resolve().parent
    copy = tmp_path / "agy_anchor"
    copy.mkdir()
    for path in source.glob("*.py"):
        (copy / path.name).write_bytes(path.read_bytes().replace(b"\n", b"\r\n"))
    monkeypatch.setattr(targets, "__file__", str(copy / "targets.py"))
    assert protocol.port_source_digest() == protocol.PORT_SOURCE_DIGEST
    (copy / "signals.py").write_bytes((copy / "signals.py").read_bytes() + b"# x\n")
    assert protocol.port_source_digest() != protocol.PORT_SOURCE_DIGEST
    (copy / "signals.py").unlink()
    assert protocol.port_source_digest() != protocol.PORT_SOURCE_DIGEST


def test_the_fixture_binding_detects_a_changed_or_missing_fixture(
    tmp_path: Path,
) -> None:
    protocol.verify_fixture_binding()
    real = protocol.fixture_path()
    assert hashlib.sha256(real.read_bytes()).hexdigest() == protocol.FIXTURE_SHA256
    changed = tmp_path / "fixture.npz"
    changed.write_bytes(real.read_bytes() + b"\0")
    with pytest.raises(protocol.ProtocolBindingError, match="differs"):
        protocol.verify_fixture_binding(changed)
    with pytest.raises(protocol.ProtocolBindingError, match="missing"):
        protocol.verify_fixture_binding(tmp_path / "absent.npz")


def test_the_initial_margin_rate_is_the_engines_one_over_max_leverage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for scenario in protocol.SCENARIOS:
        cost = protocol.scenario_cost(scenario)
        assert cost.max_leverage == 2.5 and cost.max_participation_rate == 0.01
        assert protocol.INITIAL_MARGIN_RATE == pytest.approx(1.0 / cost.max_leverage)
    protocol.verify_margin_binding()
    monkeypatch.setattr(protocol, "INITIAL_MARGIN_RATE", 0.25)
    with pytest.raises(protocol.ProtocolBindingError, match="1 / max_leverage"):
        protocol.verify_margin_binding()
    execution = Path(protocol.core7.__file__).resolve().parents[2] / "simulation"
    line = (execution / "execution.py").read_text("utf-8").splitlines()[755]
    assert "margin_used = gross_notional / self.cost.max_leverage" in line


def test_costs_risk_and_roster_are_the_core7_values_and_the_spec_text() -> None:
    risk = protocol.risk_config()
    assert (
        risk.max_gross,
        risk.max_abs_weight,
        risk.max_turnover,
        risk.drawdown_start,
        risk.drawdown_stop,
    ) == (2.5, 0.55, 2.5, 0.125, 0.25)  # Amendment 2: the owner's 25% budget
    assert risk.emergency_turnover_override is True
    assert protocol.RISK_DRAWDOWN_START * 2 == protocol.RISK_DRAWDOWN_STOP
    assert (
        protocol.core7.risk_config().drawdown_stop == 0.20
    )  # shared core7 stays at 0.20
    assert protocol.ROSTER == protocol.core7.ROSTER and len(protocol.LEG_ROSTER) == 22
    assert protocol.REBALANCE_EVERY_BARS == 18
    assert protocol.DEADBAND == 0.04
    assert protocol.MIN_BASE_SHARPE == 0.50 and protocol.MAX_DRAWDOWN_LIMIT == 0.25
    assert protocol.BOOTSTRAP_SEED == 20261121
    assert (protocol.BOOTSTRAP_BLOCK_BARS, protocol.BOOTSTRAP_RESAMPLES) == (42, 10_000)
    assert protocol.FINAL_NOT_BEFORE == "2027-05-02T00:00:00Z"
    assert protocol.DATA_AUTHORIZATION_URL.endswith("issuecomment-7028416072")
    assert protocol.PROTOCOL_ID == "agy_anchor_realistic_v1"
    protocol.verify_lowturn_binding()
    assert protocol.LOWTURN_PROTOCOL_DIGEST.startswith("9de83d14")


def test_signal_lags_follow_the_ports_l1_and_the_delay_scenario() -> None:
    assert [
        protocol.arm_signal_lag_bars("AGY_ANCHOR", s) for s in protocol.SCENARIOS
    ] == [
        1,
        1,
        2,
    ]
    assert [protocol.arm_signal_lag_bars("CASH", s) for s in protocol.SCENARIOS] == [
        0,
        0,
        1,
    ]
    with pytest.raises(ValueError):
        protocol.arm_signal_lag_bars("TS_TREND_LS", "BASE")
    with pytest.raises(ValueError):
        protocol.scenario_signal_lag_bars("LOW_COST")


def test_the_frozen_windows_match_the_spec_table() -> None:
    dev = data.frozen_window(protocol.STAGE_DEV)
    assert dev.load_start == datetime(2021, 10, 1, tzinfo=UTC)
    assert dev.first_decision_index == protocol.STAGE_WARMUP_BARS == 540
    assert dev.last_row_close == datetime(
        2026, 7, 31, 20, tzinfo=UTC
    )  # Tier D last row
    assert dev.last_row_close < data.DEFAULT_EMBARGO
    assert dev.last_row_index == tierd_row_count() - 1
    back = data.frozen_window(protocol.STAGE_BACKCAST)
    assert back.load_start == datetime(2020, 1, 1, tzinfo=UTC)
    assert back.first_decision == datetime(2020, 5, 1, tzinfo=UTC)
    assert back.last_decision_bar == datetime(2020, 12, 31, 20, tzinfo=UTC)
    assert back.settles == datetime(2021, 1, 1, tzinfo=UTC)
    fwd = data.frozen_window(protocol.STAGE_FORWARD)
    assert fwd.load_start == datetime(2026, 2, 1, tzinfo=UTC)
    assert fwd.first_decision == datetime(2026, 8, 1, tzinfo=UTC)
    assert fwd.last_decision_bar == datetime(2027, 4, 30, 20, tzinfo=UTC)
    assert fwd.settles == datetime(2027, 5, 1, tzinfo=UTC)
    for window in (dev, back, fwd):
        assert window.first_decision_index >= protocol.STAGE_WARMUP_BARS
        assert window.last_row_close == window.last_decision_bar + 2 * BAR


def tierd_row_count() -> int:
    """Rows of the Tier D clock from 2021-10-01 to its last bar (closing 2026-07-31T20:00)."""
    return int((tierd.LOAD_END - datetime(2021, 10, 1, tzinfo=UTC)) / BAR)


def test_a_window_whose_first_decision_precedes_the_warmup_is_refused() -> None:
    load = datetime(2020, 1, 1, tzinfo=UTC)
    with pytest.raises(ValueError, match="warm-up"):
        data.StageWindow(
            stage=protocol.STAGE_BACKCAST,
            load_start=load,
            first_decision=load + 100 * BAR,
            last_decision_bar=load + 300 * BAR,
            settles=load + 301 * BAR,
        )
    with pytest.raises(ValueError, match="unknown stage"):
        data.StageWindow(
            stage="BACKCAST",
            load_start=load,
            first_decision=load + 541 * BAR,
            last_decision_bar=load + 700 * BAR,
            settles=load + 701 * BAR,
        )


def test_declared_halts_need_their_coverage_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    halt = (("2020-02-09T02:00:00Z", "2020-02-09T03:00:00Z"),)
    monkeypatch.setattr(protocol, "DECLARED_HALTS", halt)
    with pytest.raises(protocol.ProtocolBindingError, match="coverage"):
        protocol.declared_halt_intervals()
    monkeypatch.setattr(protocol, "COVERAGE_EVIDENCE_SHA256", "a" * 64)
    assert len(protocol.declared_halt_intervals()) == 1
    assert len(protocol.stage_halt_intervals()) == len(protocol.SPOT_HALTS) + 1
    monkeypatch.setattr(protocol, "DECLARED_HALTS", (halt[0], halt[0]))
    with pytest.raises(protocol.ProtocolBindingError, match="ordered"):
        protocol.declared_halt_intervals()


# =========================================================== unavailable before listing


def test_a_market_is_unavailable_before_its_listing_spot_and_perp_separately(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    window = small_window(monkeypatch, protocol.STAGE_BACKCAST)
    stage = assemble(window)
    dataset = stage.dataset
    active = dataset.resolved_array("asset_active")
    tradable = np.asarray(dataset.tradable)
    info = dataset.resolved_array("information_available")
    # pair 5: spot lists at row 100, perp at row 150
    spot5, perp5 = 2 * 5, 2 * 5 + 1
    assert not active[:100, spot5].any() and active[100:, spot5].all()
    assert not active[:150, perp5].any() and active[150:, perp5].all()
    assert not tradable[:100, spot5].any() and not tradable[:150, perp5].any()
    assert not info[:100, spot5].any() and not info[:150, perp5].any()
    # spot-only pair 7 and perp-only pair 8
    assert active[:, 14].all() and not active[:, 15].any()
    assert not active[:, 16].any() and active[:, 17].all()
    # pairs with no market at all
    assert not active[:, 18:].any() and not tradable[:, 18:].any()
    # unavailable rows carry the constant sentinel and no volume or funding
    inactive = ~active
    for field in ("open", "high", "low", "close"):
        assert np.all(
            np.asarray(getattr(dataset, field))[inactive] == data.UNAVAILABLE_PRICE
        )
    assert not np.asarray(dataset.volume)[inactive].any()
    assert not np.asarray(dataset.funding_rate)[inactive].any()
    assert not np.asarray(stage.taker_buy_quote_volume)[inactive].any()
    # activation = the later of the two markets; a missing market means never
    assert stage.activation_rows[:7] == (0, 0, 0, 0, 0, 150, 600)
    assert all(row is None for row in stage.activation_rows[7:])
    # spot legs never carry funding; perp legs of listed pairs do
    assert not np.asarray(dataset.funding_event_count)[:, 0::2].any()
    assert np.asarray(dataset.funding_event_count)[:, 1].sum() > 0
    assert stage.dataset.identity_payload_json
    listing = json.loads(stage.dataset.identity_payload_json)["listing"]
    assert listing[protocol.ROSTER[5]]["perp"]["first_listed_row"] == 150
    assert listing[protocol.ROSTER[9]]["spot"]["first_listed_row"] is None
    assert data.validate_stage_dataset(stage, window) == window.last_row_index


def test_the_first_roster_pair_must_list_both_markets_from_the_load_start(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    window = small_window(monkeypatch, protocol.STAGE_BACKCAST)
    spot, perp = leg_maps(window)
    perp[protocol.ROSTER[0]] = _leg(window, first_row=40, seed=1, perp=True)
    with pytest.raises(data.StageDataError, match="listed from the stage load start"):
        assemble(window, spot=spot, perp=perp)
    spot, perp = leg_maps(window)
    spot[protocol.ROSTER[0]] = None
    with pytest.raises(data.StageDataError, match="listed from the stage load start"):
        assemble(window, spot=spot, perp=perp)


def test_a_series_with_a_gap_a_late_end_or_a_wrong_clock_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    window = small_window(monkeypatch, protocol.STAGE_BACKCAST)
    spot, perp = leg_maps(window)
    btc = spot[protocol.ROSTER[0]]
    cut = np.r_[0:200, 204 : len(btc.series.timestamps)]  # one missing 4h bar
    series = btc.series
    holed = RawMarketSeries(
        timestamps=np.asarray(series.timestamps)[cut],
        open=series.open[cut],
        high=series.high[cut],
        low=series.low[cut],
        close=series.close[cut],
        volume=series.volume[cut],
        funding_rate=series.funding_rate[cut],
        funding_available=series.funding_available[cut],
        funding_event_count=series.funding_event_count[cut],
        tradable=series.tradable[cut],
    )
    spot[protocol.ROSTER[0]] = data.HourlyLeg(holed, btc.taker_buy_quote[cut])
    with pytest.raises(data.StageDataError, match="complete and regular"):
        assemble(window, spot=spot, perp=perp)
    spot, perp = leg_maps(window)
    short = spot[protocol.ROSTER[1]]
    keep = slice(0, len(short.series.timestamps) - 4)  # ends one bar early
    series = short.series
    spot[protocol.ROSTER[1]] = data.HourlyLeg(
        RawMarketSeries(
            timestamps=np.asarray(series.timestamps)[keep],
            open=series.open[keep],
            high=series.high[keep],
            low=series.low[keep],
            close=series.close[keep],
            volume=series.volume[keep],
            funding_rate=series.funding_rate[keep],
            funding_available=series.funding_available[keep],
            funding_event_count=series.funding_event_count[keep],
            tradable=series.tradable[keep],
        ),
        short.taker_buy_quote[keep],
    )
    with pytest.raises(data.StageDataError, match="without a gap to the read limit"):
        assemble(window, spot=spot, perp=perp)


def test_the_assembler_demands_exactly_the_roster_and_aligned_taker_volume(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    window = small_window(monkeypatch, protocol.STAGE_BACKCAST)
    spot, perp = leg_maps(window)
    with pytest.raises(data.StageDataError, match="spot must cover exactly"):
        assemble(window, spot={k: v for k, v in list(spot.items())[:-1]}, perp=perp)
    with pytest.raises(data.StageDataError, match="perp must cover exactly"):
        assemble(window, spot=spot, perp={**perp, "XYZUSDT": None})
    leg = spot[protocol.ROSTER[0]]
    with pytest.raises(data.StageDataError, match="align"):
        data.HourlyLeg(leg.series, leg.taker_buy_quote[:-1])
    with pytest.raises(data.StageDataError, match="non-negative"):
        data.HourlyLeg(leg.series, -np.abs(leg.taker_buy_quote))


# ============================================================== taker-buy: carried


def test_the_taker_buy_array_is_the_4h_sum_of_the_hourly_values_and_aligned(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    window = small_window(monkeypatch, protocol.STAGE_BACKCAST)
    spot, perp = leg_maps(window)
    stage = assemble(window, spot=spot, perp=perp)
    taker = stage.taker_buy_quote_volume
    dataset = stage.dataset
    assert taker.shape == (dataset.n_bars, dataset.n_symbols)
    for pair, leg_name in ((0, "spot"), (0, "perp"), (5, "perp"), (7, "spot")):
        column = 2 * pair + (leg_name == "perp")
        source = (spot if leg_name == "spot" else perp)[protocol.ROSTER[pair]]
        first_row = LISTINGS[pair][leg_name == "perp"]
        hourly = source.taker_buy_quote.reshape(-1, 4).sum(axis=1)
        rows = slice(first_row, first_row + len(hourly))
        np.testing.assert_allclose(taker[rows, column], hourly, rtol=1e-12)
        assert not taker[:first_row, column].any()
        # aligned with the same row's quote volume: never above it
        assert np.all(taker[rows, column] <= np.asarray(dataset.volume)[rows, column])
    # the dataset identity binds the array
    identity = json.loads(dataset.identity_payload_json or "{}")
    assert (
        identity["taker_buy_quote_volume_sha256"]
        == hashlib.sha256(np.ascontiguousarray(taker).tobytes()).hexdigest()
    )
    # taker volume is not a field of the dataset (carried beside it)
    assert not hasattr(dataset, "taker_buy_quote_volume")


def _halted(leg: data.HourlyLeg, hours: slice) -> data.HourlyLeg:
    series = leg.series
    tradable = series.tradable.copy()
    tradable[hours] = False
    volume = series.volume.copy()
    volume[hours] = 0.0
    return data.HourlyLeg(
        RawMarketSeries(
            timestamps=series.timestamps,
            open=series.open,
            high=series.high,
            low=series.low,
            close=series.close,
            volume=volume,
            funding_rate=series.funding_rate,
            funding_available=series.funding_available,
            funding_event_count=series.funding_event_count,
            tradable=tradable,
        ),
        leg.taker_buy_quote,  # the taker volume of the halted hours stays above zero
    )


def test_taker_volume_of_a_halted_spot_hour_is_dropped_with_the_bar(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    window = small_window(monkeypatch, protocol.STAGE_BACKCAST)
    spot, perp = leg_maps(window)
    # an exchange-wide outage of row 10: every spot market listed then is halted
    for index in (0, 1, 2, 3, 4, 7):
        symbol = protocol.ROSTER[index]
        assert np.all(spot[symbol].taker_buy_quote[40:44] > 0.0)
        spot[symbol] = _halted(spot[symbol], slice(40, 44))
    lo = window.load_start + 10 * BAR
    stage = data.assemble_stage_dataset(
        window,
        spot,
        perp,
        rules=RULES,
        limit_close=window.last_row_close,
        halt_intervals=[(lo, lo + BAR)],
    )
    assert not np.asarray(stage.dataset.tradable)[10, 0]
    assert stage.taker_buy_quote_volume[10, 0] == 0.0
    assert stage.taker_buy_quote_volume[11, 0] > 0.0
    assert stage.taker_buy_quote_volume[10, 1] > 0.0  # the perp keeps trading
    data.validate_stage_dataset(stage, window)


def test_a_prefix_of_the_stage_gives_the_same_arrays_inputs_and_signals(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Truncating the future never changes an earlier row: dataset, taker array, anchor
    inputs and every signal row (the taker array is causal)."""
    window = small_window(monkeypatch, protocol.STAGE_BACKCAST)
    full = assemble(window)
    short_limit = window.last_row_close - 30 * BAR
    spot, perp = leg_maps(window)
    short = assemble(
        window,
        limit=short_limit,
        spot={k: truncate_leg(v, short_limit) for k, v in spot.items()},
        perp={k: truncate_leg(v, short_limit) for k, v in perp.items()},
    )
    n = short.dataset.n_bars
    for field in ("open", "high", "low", "close", "volume", "funding_rate"):
        np.testing.assert_array_equal(
            np.asarray(getattr(short.dataset, field)),
            np.asarray(getattr(full.dataset, field))[:n],
        )
    np.testing.assert_array_equal(
        short.taker_buy_quote_volume, full.taker_buy_quote_volume[:n]
    )
    assert short.activation_rows == full.activation_rows
    full_inputs = data.build_anchor_inputs(full)
    short_inputs = data.build_anchor_inputs(short)
    for name in ("spot_close", "perp_close", "spot_valid", "perp_valid"):
        np.testing.assert_array_equal(
            getattr(short_inputs, name), getattr(full_inputs, name)[:n]
        )
    full_signals = runner.anchor_signals_of(full)
    short_signals = runner.anchor_signals_of(short)
    for name in runner.ANCHOR_ROW_FIELDS:
        np.testing.assert_array_equal(
            getattr(short_signals, name), getattr(full_signals, name)[:n]
        )


def test_perturbing_only_the_future_taker_volume_changes_no_earlier_signal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    window = small_window(monkeypatch, protocol.STAGE_BACKCAST)
    stage = assemble(window)
    base_signals = runner.anchor_signals_of(stage)
    cut = window.first_decision_index + 40
    taker = np.array(stage.taker_buy_quote_volume, copy=True)
    volume = np.asarray(stage.dataset.volume)
    taker[cut + 1 :] = (
        volume[cut + 1 :] * 0.97
    )  # only the taker leg of the future moves
    other = anchor_with_taker(stage, taker)
    assert runner._anchor_prefix_diff(base_signals, other, cut_index=cut) == 0.0
    # the perturbation is real: the signal differs after the cut
    assert (
        runner._anchor_prefix_diff(base_signals, other, cut_index=len(taker) - 1) > 0.0
    )


def anchor_with_taker(stage: data.StageData, taker: np.ndarray) -> Any:
    return runner.anchor_signals_of(replace(stage, taker_buy_quote_volume=taker))


def test_the_prefix_diff_detects_a_peek_at_the_next_row_and_a_nan_pattern_change(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    window = small_window(monkeypatch, protocol.STAGE_BACKCAST)
    signals = runner.anchor_signals_of(assemble(window))
    assert runner._anchor_prefix_diff(signals, signals, cut_index=100) == 0.0
    ahead = np.roll(np.asarray(signals.w_aw_bidir_75_25), -1, axis=0)
    leaky = replace(signals, w_aw_bidir_75_25=ahead)
    assert runner._anchor_prefix_diff(signals, leaky, cut_index=300) > 0.0
    nanned = np.array(signals.c_weekly_iv, copy=True)
    nanned[50, 0] = np.nan
    assert runner._anchor_prefix_diff(
        signals, replace(signals, c_weekly_iv=nanned), cut_index=100
    ) == float("inf")


def test_perturb_future_stage_keeps_the_prefix_and_keeps_taker_below_volume(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    window = small_window(monkeypatch, protocol.STAGE_BACKCAST)
    stage = assemble(window)
    cut = 600
    moved = runner.perturb_future_stage(stage, cut_index=cut, seed=17)
    np.testing.assert_array_equal(
        moved.taker_buy_quote_volume[: cut + 1], stage.taker_buy_quote_volume[: cut + 1]
    )
    assert not np.array_equal(
        moved.taker_buy_quote_volume[cut + 1 :], stage.taker_buy_quote_volume[cut + 1 :]
    )
    assert np.all(
        moved.taker_buy_quote_volume <= np.asarray(moved.dataset.volume) * (1 + 1e-12)
    )
    np.testing.assert_array_equal(
        np.asarray(moved.dataset.close)[: cut + 1],
        np.asarray(stage.dataset.close)[: cut + 1],
    )


# =================================================================== anchor inputs


def test_anchor_inputs_gate_late_pairs_and_leave_early_pairs_untouched(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    window = small_window(monkeypatch, protocol.STAGE_BACKCAST)
    stage = assemble(window)
    dataset = stage.dataset
    inputs = data.build_anchor_inputs(stage)
    close = np.asarray(dataset.close)
    # pairs listed at the first row: exactly the dataset values, valid from row 0
    for pair in range(5):
        np.testing.assert_array_equal(inputs.spot_close[:, pair], close[:, 2 * pair])
        np.testing.assert_array_equal(
            inputs.perp_close[:, pair], close[:, 2 * pair + 1]
        )
    assert inputs.spot_valid[:, :5].all() and inputs.perp_valid[:, :5].all()
    # pair 5 (activation 150): constant 1.0 before, rescaled by the spot close after, so
    # there is no price step; valid only from activation + 180 bars
    assert data.available_from_rows(stage.activation_rows, n_rows=dataset.n_bars)[
        :7
    ] == (
        0,
        0,
        0,
        0,
        0,
        330,
        dataset.n_bars,
    )
    assert np.all(inputs.spot_close[:150, 5] == 1.0) and np.all(
        inputs.perp_close[:150, 5] == 1.0
    )
    assert inputs.spot_close[150, 5] == pytest.approx(1.0)
    factor = close[150, 10]
    np.testing.assert_allclose(inputs.spot_close[150:, 5], close[150:, 10] / factor)
    np.testing.assert_allclose(inputs.perp_close[150:, 5], close[150:, 11] / factor)
    basis = np.log(inputs.perp_close[150:, 5] / inputs.spot_close[150:, 5])
    np.testing.assert_allclose(basis, np.log(close[150:, 11] / close[150:, 10]))
    assert not inputs.spot_valid[:330, 5].any() and not inputs.perp_valid[:330, 5].any()
    assert inputs.spot_valid[330:, 5].all() and inputs.perp_valid[330:, 5].all()
    # a pair available only after the window and the pairs missing a market are never valid
    for pair in range(6, 11):
        assert (
            not inputs.spot_valid[:, pair].any()
            and not inputs.perp_valid[:, pair].any()
        )
    for pair in range(7, 11):
        assert np.all(inputs.spot_close[:, pair] == 1.0)
        assert np.all(inputs.perp_close[:, pair] == 1.0)
    # the rescale reads only the activation row: perturbing later rows cannot move earlier ones
    other = runner.perturb_future_stage(stage, cut_index=400, seed=3)
    other_inputs = data.build_anchor_inputs(other)
    np.testing.assert_array_equal(
        other_inputs.spot_close[:401, 5], inputs.spot_close[:401, 5]
    )
    np.testing.assert_array_equal(
        other_inputs.perp_close[:401, 5], inputs.perp_close[:401, 5]
    )


def test_a_halted_spot_bar_is_not_valid_for_the_anchor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    window = small_window(monkeypatch, protocol.STAGE_BACKCAST)
    spot, perp = leg_maps(window)
    symbol = protocol.ROSTER[2]
    spot[symbol] = _halted(spot[symbol], slice(400, 404))  # row 100 only
    stage = assemble(window, spot=spot, perp=perp)
    inputs = data.build_anchor_inputs(stage)
    assert not inputs.spot_valid[100, 2] and inputs.perp_valid[100, 2]
    assert inputs.spot_valid[99, 2]
    assert inputs.spot_quote_volume[100, 2] == 0.0


# ============================================================ windows and embargo


NOW = datetime(2026, 11, 20, 6, 30, tzinfo=UTC)


def test_read_limits_and_the_embargo(monkeypatch: pytest.MonkeyPatch) -> None:
    dev = small_window(monkeypatch, protocol.STAGE_DEV)
    assert data.read_limit(dev, now=NOW) == dev.last_row_close
    early = dev.load_start + 500 * BAR
    with pytest.raises(data.StageEmbargoError, match="no closed bar"):
        data.read_limit(dev, now=early)
    mid = dev.first_decision + 60 * BAR
    with pytest.raises(data.StageEmbargoError, match="not over yet"):
        data.read_limit(dev, now=mid)
    forward = small_window(monkeypatch, protocol.STAGE_FORWARD)
    with pytest.raises(data.StageEmbargoError, match="owner authorization"):
        data.read_limit(forward, now=datetime(2026, 12, 1, tzinfo=UTC))
    limit = data.read_limit(
        forward,
        now=datetime(2026, 12, 1, tzinfo=UTC),
        authorization_url=protocol.DATA_AUTHORIZATION_URL,
    )
    assert limit == forward.last_row_close > data.DEFAULT_EMBARGO
    mid = forward.first_decision + 10 * BAR
    assert (
        data.read_limit(
            forward,
            now=datetime(2026, 12, 1, tzinfo=UTC),
            authorization_url=protocol.DATA_AUTHORIZATION_URL,
            through=mid,
        )
        == mid
    )
    with pytest.raises(data.StageEmbargoError, match="through"):
        data.read_limit(
            forward,
            now=datetime(2026, 12, 1, tzinfo=UTC),
            authorization_url=protocol.DATA_AUTHORIZATION_URL,
            through=forward.last_row_close + BAR,
        )
    with pytest.raises(data.StageEmbargoError, match="owner authorization"):
        data.read_limit(
            forward,
            now=datetime(2026, 12, 1, tzinfo=UTC),
            authorization_url="https://github.com/shuntatsu/trade_rl/issues/810#issuecomment-1",
        )


def test_the_real_dev_window_never_reaches_the_embargo() -> None:
    dev = data.frozen_window(protocol.STAGE_DEV)
    limit = data.read_limit(dev, now=NOW)
    assert limit == datetime(2026, 7, 31, 20, tzinfo=UTC) < data.DEFAULT_EMBARGO
    stamp = int(limit.timestamp() * 1_000)

    class Rows:
        def read_klines(self, *a: Any) -> list[Any]:
            return []

    guard = data.StageRowSource(Rows(), window=dev, limit_close=limit)
    assert (
        guard.read_klines("spot", protocol.ROSTER[0], "1h", stamp - HOUR_MS, stamp)
        == []
    )
    with pytest.raises(data.StageEmbargoError, match="past the read limit"):
        guard.read_klines(
            "spot", protocol.ROSTER[0], "1h", stamp - HOUR_MS, stamp + HOUR_MS
        )


def test_the_row_guard_serves_both_markets_and_funding_and_refuses_the_rest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    window = small_window(monkeypatch, protocol.STAGE_BACKCAST)
    limit = window.last_row_close
    start = int(window.load_start.timestamp() * 1_000)
    end = int(limit.timestamp() * 1_000)

    class Row:
        def __init__(self, open_time_ms: int) -> None:
            self.open_time_ms = open_time_ms
            self.funding_time_ms = open_time_ms

    class Rows:
        def __init__(self, stamps: list[int]) -> None:
            self.stamps = stamps

        def read_klines(self, *a: Any) -> list[Row]:
            return [Row(s) for s in self.stamps]

        def read_funding(self, *a: Any) -> list[Row]:
            return [Row(s) for s in self.stamps]

    symbol = protocol.ROSTER[0]
    ok = data.StageRowSource(
        Rows([start, start + HOUR_MS]), window=window, limit_close=limit
    )
    assert len(ok.read_klines("spot", symbol, "1h", start, end)) == 2
    assert len(ok.read_klines("usds_m", symbol, "1h", start, end)) == 2
    assert len(ok.read_funding("usds_m", symbol, start, end)) == 2
    assert ok.rows_read == 6
    with pytest.raises(data.StageDataError, match="spot carries no funding"):
        ok.read_funding("spot", symbol, start, end)
    with pytest.raises(data.StageDataError, match="not admitted"):
        ok.read_klines("coin_m", symbol, "1h", start, end)
    with pytest.raises(data.StageDataError, match="not in the roster"):
        ok.read_klines("spot", "XYZUSDT", "1h", start, end)
    with pytest.raises(ValueError, match="1h"):
        ok.read_klines("spot", symbol, "4h", start, end)
    with pytest.raises(data.StageEmbargoError, match="before the stage load start"):
        ok.read_klines("spot", symbol, "1h", start - HOUR_MS, end)
    with pytest.raises(data.StageEmbargoError, match="past the read limit"):
        ok.read_funding("usds_m", symbol, start, end + HOUR_MS)
    late = data.StageRowSource(Rows([end + HOUR_MS]), window=window, limit_close=limit)
    with pytest.raises(data.StageEmbargoError, match="outside the requested"):
        late.read_klines("spot", symbol, "1h", start, end)
    with pytest.raises(data.StageEmbargoError, match="outside the requested"):
        late.read_funding("usds_m", symbol, start, end)
    with pytest.raises(data.StageEmbargoError, match="past the stage's last row"):
        data.StageRowSource(Rows([]), window=window, limit_close=limit + BAR)
    assert ok.probe_published_day("spot", symbol, start) is None


def test_rows_past_the_default_embargo_need_the_forward_stage_and_the_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    forward = small_window(monkeypatch, protocol.STAGE_FORWARD)
    with pytest.raises(data.StageEmbargoError, match="need the FORWARD stage"):
        assemble(forward)
    stage = assemble(forward, authorization_url=protocol.DATA_AUTHORIZATION_URL)
    assert protocol.DATA_AUTHORIZATION_URL in (
        stage.dataset.identity_payload_json or ""
    )
    assert data.validate_stage_dataset(stage, forward) == forward.last_row_index
    bare = replace(
        stage.dataset,
        dataset_id=hashlib.sha256(b"x").hexdigest(),
        identity_payload_json=None,
    )
    with pytest.raises(data.StageEmbargoError, match="no FORWARD authorization"):
        data.validate_stage_dataset(replace(stage, dataset=bare), forward)


def test_validate_stage_dataset_refuses_a_bad_taker_array_and_an_inactive_leg_with_volume(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    window = small_window(monkeypatch, protocol.STAGE_BACKCAST)
    stage = assemble(window)
    with pytest.raises(data.StageDataError, match="does not match"):
        data.validate_stage_dataset(
            replace(stage, taker_buy_quote_volume=stage.taker_buy_quote_volume[:-1]),
            window,
        )
    big = stage.taker_buy_quote_volume * 10.0
    with pytest.raises(data.StageDataError, match="exceeds"):
        data.validate_stage_dataset(replace(stage, taker_buy_quote_volume=big), window)
    volume = np.array(stage.dataset.volume, copy=True)
    volume[0, 19] = 5.0  # a leg with no market at all carries volume
    bad = replace(
        stage.dataset,
        dataset_id=hashlib.sha256(b"y").hexdigest(),
        identity_payload_json=None,
        volume=volume,
    )
    with pytest.raises(data.StageDataError, match="inactive leg"):
        data.validate_stage_dataset(replace(stage, dataset=bad), window)
    summary = data.stage_summary(stage, window)
    assert summary["shape"] == [stage.dataset.n_bars, 22]
    assert summary["activation_rows"][:7] == [0, 0, 0, 0, 0, 150, 600]
