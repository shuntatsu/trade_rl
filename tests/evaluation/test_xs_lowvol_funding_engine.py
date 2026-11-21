"""Engine-level oracles (G2) of ``xs_lowvol_funding_v1`` on synthetic fixtures.

Forced close, funding sign, causality, embargo and INVALID are checked through the shared
replay engine and the stage guards; the fixture's returns and funding are closed-form (see
``xs_fixtures``) and every expected number below is derived by hand from them.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from tests.evaluation import xs_fixtures as fx
from trade_rl.evaluation.replay import (
    SharedCashPortfolioReplayResult,
    run_shared_portfolio_replay,
)
from trade_rl.evaluation.trend_oos.data import StageDataError, StageEmbargoError
from trade_rl.evaluation.xs_lowvol_funding import (
    data,
    decision,
    policy,
    protocol,
    runner,
)
from trade_rl.risk.pretrade import PreTradeRisk
from trade_rl.strategies.portfolio import (
    PortfolioDecisionSnapshot,
    PortfolioPolicy,
    PortfolioTargetProposal,
)

LAST_ROW = 398  # AAA's last available bar when it is delisted in the fixtures


def _delisted(**extra: object) -> data.StageData:
    return fx.stage(
        overrides={
            "AAAUSDT": {"last_hour": fx.last_hour_of_row(fx.WINDOW, LAST_ROW), **extra}
        }
    )


def _replay(
    stage_data: data.StageData,
    arm: str = protocol.CANDIDATE,
    scenario: str = protocol.BASE_SCENARIO,
    *,
    policy_override: PortfolioPolicy | None = None,
    risk: bool = True,
) -> SharedCashPortfolioReplayResult:
    inputs = runner.inputs_of(stage_data)
    start = fx.WINDOW.first_decision_index
    pol = policy_override or runner.make_policy(
        arm,
        inputs=inputs,
        start_index=start,
        lag=protocol.scenario_signal_lag_bars(scenario),
    )
    stop = stage_data.dataset.n_bars - 1
    if risk:
        return runner.replay(
            stage_data.dataset,
            pol,
            scenario=scenario,
            start_index=start,
            stop_index=stop,
            risk=PreTradeRisk(protocol.risk_config()),
        )
    return run_shared_portfolio_replay(
        stage_data.dataset,
        pol,
        start_index=start,
        stop_index=stop,
        initial_capital=protocol.INITIAL_CAPITAL,
        execution_cost=protocol.scenario_cost(scenario),
        risk=None,
        settle_terminal_position=True,
        capture_ledger_evidence=True,
    )


def _cell(
    stage_data: data.StageData, result: SharedCashPortfolioReplayResult, arm: str
) -> Any:
    return runner._build_cell(
        arm=arm,
        scenario=protocol.BASE_SCENARIO,
        result=result,
        dataset=stage_data.dataset,
        causality_passed=True,
        probe_key="probe",
        policy_statistics={},
    )


def _interval(result: SharedCashPortfolioReplayResult, next_index: int) -> Any:
    (found,) = [
        iv
        for iv in result.ledger_evidence.intervals.intervals  # type: ignore[union-attr]
        if iv.next_index == next_index
    ]
    return found


# ========================================================================= forced close


def test_the_forced_close_fills_one_bar_before_the_last_bar_and_charges_five_percent() -> (
    None
):
    stage_data = _delisted()
    result = _replay(stage_data)
    dataset = stage_data.dataset
    leg = dataset.symbols.index("AAAUSDT")
    # the name is held long when the notice arrives, then the decision one bar before its
    # last available bar targets zero and the order fills on the last available bar
    held = _interval(result, LAST_ROW - 1)
    assert Fraction(held.exact_quantities_after[leg]) > 0
    by_index = {d.index: d for d in result.decisions}
    assert by_index[LAST_ROW - 2].proposal_weights[leg] > 0.0
    assert by_index[LAST_ROW - 1].proposal_weights[leg] == 0.0
    closing = _interval(result, LAST_ROW)
    fills = [
        event
        for event in closing.order_events
        if event.symbol_index == leg
        and event.processing_index == LAST_ROW
        and event.event_type in ("filled", "partial_fill")
    ]
    assert len(fills) == 1
    closed_quantity = abs(Fraction(held.exact_quantities_after[leg]))
    assert float(closed_quantity) == pytest.approx(abs(fills[0].filled_quantity))
    notional = fills[0].filled_notional
    # no position remains, on the last bar or at the end
    assert Fraction(closing.exact_quantities_after[leg]) == 0
    assert result.book.quantities[leg] == 0.0
    # the engine already charged the scenario cost on that fill; the harness adds exactly 5%
    assert closing.interval_cost > 0.0001 * notional
    events = runner.forced_closes(result, dataset)
    assert [(e.leg, e.last_bar) for e in events] == [("AAAUSDT", LAST_ROW)]
    assert events[0].fill_notional == pytest.approx(notional)
    assert events[0].residual_notional == 0.0
    assert events[0].fill_charge == pytest.approx(0.05 * notional)


def test_the_forced_close_charge_is_a_cash_debit_on_the_equity_path_from_the_fill_bar() -> (
    None
):
    stage_data = _delisted()
    result = _replay(stage_data)
    cell = _cell(stage_data, result, protocol.CANDIDATE)
    forced = cell.cap_statistics["forced_close"]
    assert forced["count"] == 1
    charge = forced["total_charge_usdt"]
    assert charge > 1_000.0  # about 5% of a 25% weight of 100,000 USDT
    # the whole path is lower by the debit from the fill bar on, so the final equity is
    # exactly the engine's final equity (less terminal dust) minus the charge
    assert forced["without_charge"][
        "total_return"
    ] - cell.total_return == pytest.approx(charge / protocol.INITIAL_CAPITAL, abs=1e-12)
    returns = np.asarray(cell.returns)
    path = protocol.INITIAL_CAPITAL * np.cumprod(1.0 + returns)
    raw = protocol.INITIAL_CAPITAL * np.cumprod(
        1.0 + np.asarray(result.returns.values, dtype=np.float64)
    )
    fill_position = LAST_ROW - fx.WINDOW.first_decision_index - 1
    ledger_ends = [iv.next_index for iv in result.ledger_evidence.intervals.intervals]  # type: ignore[union-attr]
    assert ledger_ends[fill_position] == LAST_ROW
    assert np.allclose(path[:fill_position], raw[:fill_position])
    assert path[fill_position] == pytest.approx(raw[fill_position] - charge, rel=1e-9)
    assert path[-1] == pytest.approx(raw[-1] - charge, rel=1e-6)
    # reported and gating: the cell is still valid and cash is not borrowed
    assert cell.valid and cell.max_borrowing is not None and cell.max_borrowing < 0.1


def test_an_unfillable_close_fails_closed_instead_of_inventing_a_write_off() -> None:
    # the last two bars of the name are too thin to fill anything (1% of ~4 USDT), so the
    # position is still held when the asset becomes unavailable. The spec writes it off at
    # the last close less 5%, but the shared replay cannot continue past an engine
    # settlement (its fill tracker diverges), so the harness refuses rather than improvise.
    thin_from = (LAST_ROW - 1) * protocol.BAR_HOURS
    stage_data = fx.stage(
        overrides={
            "AAAUSDT": {
                "last_hour": fx.last_hour_of_row(fx.WINDOW, LAST_ROW),
                "volume_from": (thin_from, 1.0),
            }
        }
    )
    with pytest.raises(runner.UnsupportedWriteOffError, match="cannot write it off"):
        _replay(stage_data)
    assert issubclass(runner.UnsupportedWriteOffError, StageDataError)
    # the dataset still carries the settlement price for the day the branch is supported
    dataset = stage_data.dataset
    leg = dataset.symbols.index("AAAUSDT")
    assert dataset.open[LAST_ROW + 1, leg] == dataset.close[LAST_ROW, leg]


def test_the_unavailable_rows_carry_a_sentinel_not_an_invented_price() -> None:
    dataset = _delisted().dataset
    leg = dataset.symbols.index("AAAUSDT")
    active = dataset.resolved_array("asset_active")[:, leg]
    assert active[LAST_ROW] and not active[LAST_ROW + 1]
    assert np.all(dataset.close[LAST_ROW + 2 :, leg] == 1.0)
    assert not np.asarray(dataset.tradable)[~active, leg].any()
    assert not np.asarray(dataset.volume)[~active, leg].any()
    # only the first inactive bar opens at the last published close (the settlement price)
    assert dataset.open[LAST_ROW + 1, leg] == dataset.close[LAST_ROW, leg]
    assert dataset.open[LAST_ROW + 2, leg] == 1.0


# =========================================================================== funding sign


class _Constant(PortfolioPolicy):
    """Targets fixed weights on every bar (a scripted book)."""

    def __init__(self, weights: np.ndarray) -> None:
        self._weights = tuple(float(w) for w in weights)

    def decide_portfolio(
        self, snapshot: PortfolioDecisionSnapshot
    ) -> PortfolioTargetProposal:
        return PortfolioTargetProposal(target_weights=self._weights)


def _funding_pnl(weight: float) -> tuple[float, float]:
    stage_data = fx.stage()
    leg = stage_data.dataset.symbols.index("GGGUSDT")  # funding +3e-4 per event
    weights = np.zeros(stage_data.dataset.n_symbols)
    weights[leg] = weight
    result = _replay(stage_data, policy_override=_Constant(weights), risk=False)
    start = fx.WINDOW.first_decision_index
    events = int(stage_data.dataset.funding_event_count[start + 1 :, leg].sum())
    return float(result.diagnostics.funding_pnl), events


def test_a_short_perp_leg_with_positive_funding_receives_it_and_a_long_pays() -> None:
    short_pnl, events = _funding_pnl(-0.1)
    long_pnl, _ = _funding_pnl(0.1)
    assert events > 40
    # notional 0.1 x 100,000 = 10,000 USDT, rate 3e-4 per event: 3 USDT per event
    expected = events * 3e-4 * 10_000.0
    assert short_pnl > 0.0 and long_pnl < 0.0
    assert short_pnl == pytest.approx(expected, rel=0.1)
    assert long_pnl == pytest.approx(-expected, rel=0.1)


def test_funding_is_booked_only_on_available_tradable_bars() -> None:
    dataset = _delisted().dataset
    leg = dataset.symbols.index("AAAUSDT")
    counts = dataset.funding_event_count[:, leg]
    assert counts[: LAST_ROW + 1].sum() > 0 and counts[LAST_ROW + 1 :].sum() == 0
    assert np.all(dataset.funding_rate[LAST_ROW + 1 :, leg] == 0.0)


# ================================================================================ causality


def test_appending_future_rows_never_changes_an_earlier_decision() -> None:
    short = fx.stage()
    longer_window = data.StageWindow(
        stage=protocol.STAGE_DEV,
        load_start=fx.LOAD_START,
        first_decision=fx.WINDOW.first_decision,
        last_decision_bar=datetime(2022, 4, 4, tzinfo=UTC),
        settles=datetime(2022, 4, 8, tzinfo=UTC),
    )
    longer = fx.stage(longer_window)
    a, b = runner.inputs_of(short), runner.inputs_of(longer)
    assert b.n_rows > a.n_rows
    compared = 0
    for arm in protocol.ARMS:
        for lag in (0, 1):
            for row in range(
                fx.WINDOW.first_decision_index,
                a.n_rows - 2,
                protocol.REBALANCE_EVERY_BARS,
            ):
                one = policy.book_target(
                    a, arm=arm, decision_row=row, signal_row=max(0, row - lag)
                )
                two = policy.book_target(
                    b, arm=arm, decision_row=row, signal_row=max(0, row - lag)
                )
                assert np.array_equal(one[0], two[0]) and one[1:] == two[1:]
                compared += 1
    assert compared == 2 * 5 * 3


def test_a_decision_reads_no_row_after_its_signal_row() -> None:
    inputs = runner.inputs_of(fx.stage())
    row = fx.WINDOW.first_decision_index
    reference = policy.book_target(
        inputs, arm=protocol.CANDIDATE, decision_row=row, signal_row=row
    )[0]
    assert reference.any()
    mutated = inputs.copy()
    mutated.close[row + 1 :] *= 7.0
    mutated.funding_rate[row + 1 :] += 0.5
    assert np.array_equal(
        reference,
        policy.book_target(
            mutated, arm=protocol.CANDIDATE, decision_row=row, signal_row=row
        )[0],
    )
    # ... but the signal row itself is read: changing it changes the book
    touched = inputs.copy()
    touched.close[row] *= 1.5
    assert not np.array_equal(
        reference,
        policy.book_target(
            touched, arm=protocol.CANDIDATE, decision_row=row, signal_row=row
        )[0],
    )


def test_the_delay_scenario_reads_the_signal_one_bar_earlier() -> None:
    inputs = runner.inputs_of(fx.stage())
    row = fx.WINDOW.first_decision_index
    mutated = inputs.copy()
    mutated.funding_rate[row] += 0.5  # only the decision row differs
    base = policy.book_target(
        inputs, arm=protocol.CANDIDATE, decision_row=row, signal_row=row - 1
    )[0]
    delayed = policy.book_target(
        mutated, arm=protocol.CANDIDATE, decision_row=row, signal_row=row - 1
    )[0]
    assert np.array_equal(base, delayed)


# ================================================================================= run_stage


def test_run_stage_writes_every_cell_a_manifest_and_a_dev_report_once(
    tmp_path: Path,
) -> None:
    stage_data = _delisted()
    digest = protocol.protocol_digest()
    out = tmp_path / "dev"
    report = runner.run_stage(
        stage_data,
        window=fx.WINDOW,
        output_dir=out,
        expected_protocol_digest=digest,
        code_commit="c" * 40,
        run_date=datetime(2026, 11, 21, tzinfo=UTC).date(),
    )
    assert report["label"] is None and report["status"] == decision.DEV_STATUS
    assert report["invalid_cells"] == [] and report["protocol_digest"] == digest
    assert report["candidates_sha256"] == protocol.CANDIDATES_SHA256
    cells = sorted(p.name for p in (out / "cells").iterdir())
    assert cells == sorted(
        f"{a}__{s}.json" for a in protocol.ARMS for s in protocol.SCENARIOS
    )
    manifest = json.loads((out / runner.MANIFEST_NAME).read_text("utf-8"))
    assert manifest["result_file"] == runner.DEV_NAME and manifest["label"] is None
    assert set(manifest["causality_probes"]) == {
        f"{a}__lag{lag}" for a in protocol.ARMS for lag in (0, 1)
    }
    for probe in manifest["causality_probes"].values():
        assert probe["passed"] is True
    candidate = json.loads((out / "cells" / "CANDIDATE__BASE.json").read_text("utf-8"))
    assert (
        candidate["cell_valid"] and len(candidate["returns"]) == candidate["n_returns"]
    )
    # the cash control never trades; the delisting charged the candidate in every scenario
    cash = json.loads((out / "cells" / "CASH__BASE.json").read_text("utf-8"))
    assert cash["total_return"] == 0.0 and cash["turnover_total"] == 0.0
    forced = report["forced_close"]
    for scenario in protocol.SCENARIOS:
        assert forced[f"CANDIDATE__{scenario}"]["count"] == 1
        assert forced[f"CASH__{scenario}"]["count"] == 0
    # outputs are write-once; a wrong digest or an interim DEV run refuses before anything runs
    with pytest.raises(FileExistsError):
        runner.run_stage(
            stage_data,
            window=fx.WINDOW,
            output_dir=out,
            expected_protocol_digest=digest,
        )
    with pytest.raises(ValueError, match="digest"):
        runner.run_stage(
            stage_data,
            window=fx.WINDOW,
            output_dir=tmp_path / "x",
            expected_protocol_digest="0" * 64,
        )
    with pytest.raises(ValueError, match="interim"):
        runner.run_stage(
            stage_data,
            window=fx.WINDOW,
            output_dir=tmp_path / "y",
            expected_protocol_digest=digest,
            mode=runner.MODE_INTERIM,
        )
    assert not (tmp_path / "x").exists() and not (tmp_path / "y").exists()


def test_run_stage_refuses_a_decision_that_does_not_span_the_window(
    tmp_path: Path,
) -> None:
    short = fx.stage()
    clipped = data.StageWindow(
        stage=protocol.STAGE_DEV,
        load_start=fx.LOAD_START,
        first_decision=fx.WINDOW.first_decision,
        last_decision_bar=fx.WINDOW.last_decision_bar,
        settles=datetime(2022, 3, 31, tzinfo=UTC),
    )
    with pytest.raises(StageDataError, match="whole stage window"):
        runner.run_stage(
            short,
            window=clipped,
            output_dir=tmp_path / "o",
            expected_protocol_digest=protocol.protocol_digest(),
        )


def test_a_forward_decision_needs_the_disjoint_label(tmp_path: Path) -> None:
    window = data.StageWindow(
        stage=protocol.STAGE_FORWARD,
        load_start=fx.LOAD_START,
        first_decision=fx.WINDOW.first_decision,
        last_decision_bar=fx.WINDOW.last_decision_bar,
        settles=fx.WINDOW.settles,
    )
    stage_data = fx.stage(window, data_authorization_url=AUTH)
    with pytest.raises(ValueError, match="needs the DISJOINT label"):
        runner.run_stage(
            stage_data,
            window=window,
            output_dir=tmp_path / "o",
            expected_protocol_digest=protocol.protocol_digest(),
        )


# ================================================================================== embargo


NOW = datetime(2026, 11, 21, 6, 30, tzinfo=UTC)
AUTH = protocol.DATA_AUTHORIZATION_URL


def test_disjoint_and_forward_reads_need_the_owner_authorization_url() -> None:
    for stage_name, now in (
        (protocol.STAGE_DISJOINT, NOW),
        (protocol.STAGE_FORWARD, datetime(2026, 9, 10, 5, 30, tzinfo=UTC)),
    ):
        window = data.frozen_window(stage_name)
        for url in (
            None,
            "https://github.com/shuntatsu/trade_rl/issues/810#issuecomment-1",
        ):
            with pytest.raises(StageEmbargoError, match="owner authorization"):
                data.read_limit(window, now=now, authorization_url=url)
        assert data.read_limit(window, now=now, authorization_url=AUTH) is not None
    # DEV reads known-used data and needs none
    assert (
        data.read_limit(data.frozen_window("DEV"), now=NOW)
        == data.frozen_window("DEV").last_row_close
    )


def test_disjoint_stops_strictly_before_the_pre_fs1_embargo() -> None:
    window = data.frozen_window(protocol.STAGE_DISJOINT)
    assert (
        data.read_limit(window, now=NOW, authorization_url=AUTH) == data.DEFAULT_EMBARGO
    )
    # the window must be over: a clock before 2026-08-01 refuses
    with pytest.raises(StageEmbargoError, match="not over yet"):
        data.read_limit(
            window, now=datetime(2026, 6, 1, tzinfo=UTC), authorization_url=AUTH
        )
    # a DISJOINT window that reached past the embargo is refused
    past = data.StageWindow(
        stage=protocol.STAGE_DISJOINT,
        load_start=datetime(2026, 3, 30, tzinfo=UTC),
        first_decision=datetime(2026, 7, 6, tzinfo=UTC),
        last_decision_bar=datetime(2026, 8, 3, tzinfo=UTC),
        settles=datetime(2026, 8, 8, tzinfo=UTC),
    )
    with pytest.raises(StageEmbargoError, match="only the FORWARD stage"):
        data.read_limit(past, now=NOW, authorization_url=AUTH)


class _Source:
    def __init__(self, rows: list[Any] | None = None) -> None:
        self.calls: list[tuple[str, int, int]] = []
        self.rows = rows or []

    def read_klines(
        self, market: Any, symbol: str, interval: str, start_ms: int, end_ms: int
    ) -> list[Any]:
        self.calls.append((symbol, start_ms, end_ms))
        return self.rows

    def read_funding(
        self, market: Any, symbol: str, start_ms: int, end_ms: int
    ) -> list[Any]:
        self.calls.append((symbol, start_ms, end_ms))
        return self.rows


class _Row:
    def __init__(self, open_time_ms: int) -> None:
        self.open_time_ms = open_time_ms
        self.funding_time_ms = open_time_ms


def test_the_row_guard_refuses_a_read_at_or_after_2026_08_01_before_and_after_reading() -> (
    None
):
    window = data.frozen_window(protocol.STAGE_DISJOINT)
    limit = window.last_row_close
    embargo_ms = int(limit.timestamp() * 1_000)
    inner = _Source()
    guard = data.StageRowSource(
        inner, window=window, limit_close=limit, symbols=["AAAUSDT"]
    )
    with pytest.raises(StageEmbargoError, match="refused before reading"):
        guard.read_klines(
            "usds_m", "AAAUSDT", "1h", embargo_ms - 3_600_000, embargo_ms + 1
        )
    with pytest.raises(StageEmbargoError):
        guard.read_funding("usds_m", "AAAUSDT", embargo_ms - 3_600_000, embargo_ms + 4)
    assert inner.calls == []  # nothing was read
    # a source that returns a row at the embargo anyway is refused on the row
    leaking = data.StageRowSource(
        _Source([_Row(embargo_ms)]),
        window=window,
        limit_close=limit,
        symbols=["AAAUSDT"],
    )
    with pytest.raises(StageEmbargoError, match="outside the requested range"):
        leaking.read_klines(
            "usds_m", "AAAUSDT", "1h", embargo_ms - 7_200_000, embargo_ms
        )
    # the last admissible hourly open is the hour before the embargo
    ok = data.StageRowSource(
        _Source([_Row(embargo_ms - 3_600_000)]),
        window=window,
        limit_close=limit,
        symbols=["AAAUSDT"],
    )
    assert ok.read_klines("usds_m", "AAAUSDT", "1h", embargo_ms - 7_200_000, embargo_ms)
    # the guard cannot be constructed past the stage's last row
    with pytest.raises(StageEmbargoError, match="past the stage's last row"):
        data.StageRowSource(
            _Source(),
            window=window,
            limit_close=datetime(2026, 8, 1, 4, tzinfo=UTC),
            symbols=["AAAUSDT"],
        )


def test_the_row_guard_admits_only_declared_perp_symbols_inside_the_stage() -> None:
    window = data.frozen_window(protocol.STAGE_DEV)
    start_ms = int(window.load_start.timestamp() * 1_000)
    guard = data.StageRowSource(
        _Source(), window=window, limit_close=window.last_row_close, symbols=["AAAUSDT"]
    )
    with pytest.raises(StageDataError, match="not a symbol"):
        guard.read_klines("usds_m", "BTCUSDT", "1h", start_ms, start_ms + 3_600_000)
    with pytest.raises(StageDataError, match="perps only"):
        guard.read_klines("spot", "AAAUSDT", "1h", start_ms, start_ms + 3_600_000)
    with pytest.raises(StageEmbargoError, match="before the stage load start"):
        guard.read_klines("usds_m", "AAAUSDT", "1h", start_ms - 3_600_000, start_ms)
    with pytest.raises(ValueError, match="only 1h"):
        guard.read_klines("usds_m", "AAAUSDT", "4h", start_ms, start_ms + 3_600_000)


def test_forward_never_reads_past_the_latest_closed_bar() -> None:
    window = data.frozen_window(protocol.STAGE_FORWARD)
    now = datetime(2026, 9, 10, 5, 30, tzinfo=UTC)
    limit = data.read_limit(window, now=now, authorization_url=AUTH)
    assert limit == datetime(2026, 9, 10, 4, tzinfo=UTC)
    with pytest.raises(StageEmbargoError, match="through"):
        data.read_limit(
            window,
            now=now,
            authorization_url=AUTH,
            through=datetime(2026, 9, 10, 8, tzinfo=UTC),
        )
    guard = data.StageRowSource(
        _Source(), window=window, limit_close=limit, symbols=["AAAUSDT"]
    )
    end_ms = int(limit.timestamp() * 1_000)
    with pytest.raises(StageEmbargoError, match="refused before reading"):
        guard.read_klines(
            "usds_m", "AAAUSDT", "1h", end_ms - 3_600_000, end_ms + 3_600_000
        )
    # nothing before the first decision's two bars can be replayed
    with pytest.raises(StageEmbargoError, match="no closed bar"):
        data.read_limit(
            window,
            now=datetime(2026, 8, 3, 1, tzinfo=UTC),
            authorization_url=AUTH,
        )


def test_an_authorized_stage_dataset_carries_the_authorization_in_its_identity() -> (
    None
):
    window = data.StageWindow(
        stage=protocol.STAGE_DISJOINT,
        load_start=fx.LOAD_START,
        first_decision=fx.WINDOW.first_decision,
        last_decision_bar=fx.WINDOW.last_decision_bar,
        settles=fx.WINDOW.settles,
    )
    hourly = fx.legs(window)
    clock = data.stage_clock(window, window.last_row_close)
    kwargs: dict[str, Any] = {
        "member": np.ones((clock.size, len(hourly)), dtype=bool),
        "universe": {},
        "static_universe": False,
        "rules": {name: fx.LegRules(0.001, 5.0) for name in hourly},
        "limit_close": window.last_row_close,
    }
    with pytest.raises(StageEmbargoError, match="owner authorization"):
        data.assemble_stage_dataset(window, hourly, **kwargs)
    built = data.assemble_stage_dataset(
        window, hourly, data_authorization_url=AUTH, **kwargs
    )
    assert data.validate_stage_dataset(built, window) == clock.size - 1
    assert AUTH in (built.dataset.identity_payload_json or "")


def test_the_dataset_refuses_a_hole_a_wrong_grid_and_unsorted_legs() -> None:
    hourly = fx.legs()
    clock = data.stage_clock(fx.WINDOW, fx.WINDOW.last_row_close)
    base: dict[str, Any] = {
        "member": np.ones((clock.size, len(hourly)), dtype=bool),
        "universe": {},
        "static_universe": True,
        "rules": {name: fx.LegRules(0.001, 5.0) for name in hourly},
        "limit_close": fx.WINDOW.last_row_close,
    }
    short = dict(hourly)
    short["AAAUSDT"] = fx.hourly_leg(fx.n_hours() - 4, step=0.01, rate=0.0)
    with pytest.raises(StageDataError, match="hourly grid"):
        data.assemble_stage_dataset(fx.WINDOW, short, **base)
    with pytest.raises(StageDataError, match="rules"):
        data.assemble_stage_dataset(
            fx.WINDOW, hourly, **{**base, "rules": {"AAAUSDT": fx.LegRules(0.1, 5.0)}}
        )
    with pytest.raises(StageDataError, match="membership"):
        data.assemble_stage_dataset(
            fx.WINDOW, hourly, **{**base, "member": np.ones((3, 3), dtype=bool)}
        )


def test_an_unexplained_hourly_gap_makes_the_bar_unavailable_and_a_declared_halt_does_not() -> (
    None
):
    total = fx.n_hours()
    gap_hour = 100 * protocol.BAR_HOURS + 1
    leg = fx.hourly_leg(total, step=0.01, rate=0.0)
    present = leg.present.copy()
    present[gap_hour] = False
    gapped = data.PerpHourly(
        present=present,
        halted=leg.halted,
        open=leg.open,
        high=leg.high,
        low=leg.low,
        close=leg.close,
        quote_volume=leg.quote_volume,
        funding_ms=leg.funding_ms,
        funding_rate=leg.funding_rate,
    )
    halted_flags = leg.halted.copy()
    halted_flags[gap_hour] = True
    halted = data.PerpHourly(
        present=present,
        halted=halted_flags,
        open=leg.open,
        high=leg.high,
        low=leg.low,
        close=leg.close,
        quote_volume=leg.quote_volume,
        funding_ms=leg.funding_ms,
        funding_rate=leg.funding_rate,
    )
    clock = data.stage_clock(fx.WINDOW, fx.WINDOW.last_row_close)
    for variant, expect_active, expect_tradable in (
        (gapped, False, False),
        (halted, True, False),
    ):
        built = data.assemble_stage_dataset(
            fx.WINDOW,
            {"AAAUSDT": variant},
            member=np.ones((clock.size, 1), dtype=bool),
            universe={},
            static_universe=True,
            rules={"AAAUSDT": fx.LegRules(0.001, 5.0)},
            limit_close=fx.WINDOW.last_row_close,
        )
        active = built.dataset.resolved_array("asset_active")[100, 0]
        tradable = np.asarray(built.dataset.tradable)[100, 0]
        assert bool(active) is expect_active and bool(tradable) is expect_tradable
        if expect_active:  # the halted bar carries the preceding close, zero volume
            assert built.dataset.close[100, 0] == built.dataset.close[99, 0]
            assert built.dataset.volume[100, 0] == 0.0
    # a halted hour outside the listed span is refused
    late = present.copy()
    late[:8] = False  # the series is published from hour 8 on
    edge = np.zeros_like(halted_flags)
    edge[0] = True
    with pytest.raises(StageDataError, match="between the first and the last"):
        data.PerpHourly(
            present=late,
            halted=edge,
            open=leg.open,
            high=leg.high,
            low=leg.low,
            close=leg.close,
            quote_volume=leg.quote_volume,
            funding_ms=leg.funding_ms,
            funding_rate=leg.funding_rate,
        )


# ===================================================================================== INVALID


def test_a_borrowing_cell_is_invalid_and_makes_the_stage_label_invalid() -> None:
    stage_data = fx.stage()
    weights = np.zeros(stage_data.dataset.n_symbols)
    weights[:6] = 0.25  # six names at the per-name cap: 150% long, cash about minus 50%
    borrowing = _replay(stage_data, policy_override=_Constant(weights))
    cell = _cell(stage_data, borrowing, protocol.CANDIDATE)
    assert protocol.REASON_NEGATIVE_CASH in cell.invalid_reasons and not cell.valid
    assert cell.max_borrowing is not None and cell.max_borrowing > 40_000.0
    # the same arm on the cash account of the protocol (EW_LONG, net capped at 0.97) is valid
    honest = _replay(stage_data, protocol.ARM_EW_LONG)
    ew = _cell(stage_data, honest, protocol.ARM_EW_LONG)
    assert ew.valid and (ew.max_borrowing or 0.0) <= 1e-6 * protocol.INITIAL_CAPITAL
    from tests.evaluation.test_xs_lowvol_funding import _grid

    cells = _grid()
    cells[(protocol.CANDIDATE, protocol.BASE_SCENARIO)] = cell
    outcome = decision.decide_disjoint(cells)
    assert outcome["label"] == protocol.LABEL_INVALID
    assert "CANDIDATE__BASE" in outcome["invalid_cells"]


def test_the_cash_account_scales_an_equal_weight_book_to_a_net_exposure_of_0_97() -> (
    None
):
    stage_data = fx.stage()
    result = _replay(stage_data, protocol.ARM_EW_LONG)
    by_index = {d.index: d for d in result.decisions}
    proposal = by_index[fx.WINDOW.first_decision_index].proposal_weights
    # 7 names at 1 / 7 each, net 1.0 -> scaled by 0.97 / 1.0
    assert np.allclose(proposal, 0.97 / 7.0, atol=1e-12)


def test_a_non_dust_residual_after_settlement_is_invalid() -> None:
    stage_data = fx.stage()
    result = _replay(stage_data, protocol.ARM_EW_LONG)
    cell = _cell(stage_data, result, protocol.ARM_EW_LONG)
    assert cell.terminal_flat and cell.valid
    # simulate an unsettled residual: the shared summary reports it as invalid
    broken = runner._summarize_replay_cell(
        arm="X",
        scenario="BASE",
        result=_with_residual(result),
        dataset=stage_data.dataset,
        initial_capital=protocol.INITIAL_CAPITAL,
        causality_passed=True,
        execution_cost=protocol.scenario_cost("BASE"),
    )
    assert "terminal_position_not_flat" in broken.invalid_reasons


def _with_residual(result: SharedCashPortfolioReplayResult) -> Any:
    from dataclasses import replace

    ledger = result.ledger_evidence
    assert ledger is not None
    quantities = list(ledger.intervals.terminal_exact_quantities)
    quantities[0] = "1000"
    intervals = replace(ledger.intervals, terminal_exact_quantities=tuple(quantities))
    return replace(result, ledger_evidence=replace(ledger, intervals=intervals))
