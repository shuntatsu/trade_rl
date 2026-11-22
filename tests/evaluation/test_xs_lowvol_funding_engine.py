"""Engine-level oracles (G2) of ``xs_lowvol_funding_v1`` on synthetic fixtures.

Forced close, funding sign, causality, embargo and INVALID are checked through the shared
replay engine and the stage guards; the fixture's returns and funding are closed-form (see
``xs_fixtures``) and every expected number below is derived by hand from them.
"""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import UTC, datetime
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
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


def _cash_path(cash: tuple[float, ...], costs: tuple[float, ...]) -> Any:
    intervals = []
    previous = 0.0
    for balance, total in zip(cash, costs, strict=True):
        intervals.append(
            SimpleNamespace(
                cash_after=balance, total_cost_before=previous, total_cost_after=total
            )
        )
        previous = total
    return SimpleNamespace(
        ledger_evidence=SimpleNamespace(
            intervals=SimpleNamespace(intervals=tuple(intervals))
        )
    )


@pytest.mark.parametrize(
    "cash,costs,debits,want",
    [
        ((-3000.0, 500.0, -2500.0), (3000.0, 3000.0, 3000.0), (0.0, 0.0, 0.0), 2500.0),
        ((-3000.0, -100.0, -2500.0), (3000.0, 3000.0, 3000.0), (0.0, 0.0, 0.0), 2400.0),
        ((-3000.0, 500.0, -2500.0), (3000.0, 3000.0, 3050.0), (0.0, 0.0, 0.0), 2450.0),
        ((-3000.0, 500.0, -2500.0), (3000.0, 3000.0, 3000.0), (0.0, 0.0, 50.0), 2550.0),
        ((-3000.0, 0.0), (3000.0, 3000.0), (0.0, 0.0), 0.0),
    ],
)
def test_xs_cash_fee_allowance_expires_on_recovery(
    cash: tuple[float, ...],
    costs: tuple[float, ...],
    debits: tuple[float, ...],
    want: float,
) -> None:
    low, borrowed = runner._min_cash_and_borrowing(
        _cash_path(cash, costs), np.asarray(debits)
    )
    assert low == pytest.approx(min(c - d for c, d in zip(cash, debits, strict=True)))
    assert borrowed == pytest.approx(want)


@pytest.mark.parametrize(
    "cash,costs,debits",
    [
        ((float("nan"),), (1.0,), (0.0,)),
        ((-1.0,), (float("inf"),), (0.0,)),
        ((-1.0, -2.0), (10.0, 9.0), (0.0, 0.0)),
        ((-1.0, -2.0), (10.0, 10.0), (0.0, float("nan"))),
        ((-1.0, -2.0), (10.0, 10.0), (0.0, -1.0)),
        ((-1.0, -2.0), (10.0, 10.0), (0.0,)),
    ],
)
def test_xs_cash_fee_accounting_fails_closed_on_malformed_evidence(
    cash: tuple[float, ...], costs: tuple[float, ...], debits: tuple[float, ...]
) -> None:
    with pytest.raises(ValueError):
        runner._min_cash_and_borrowing(_cash_path(cash, costs), np.asarray(debits))


def test_xs_cash_fee_accounting_checks_interval_cost_continuity() -> None:
    probe = _cash_path((-5.0, -7.0), (10.0, 11.0))
    probe.ledger_evidence.intervals.intervals[1].total_cost_before = 9.0
    with pytest.raises(ValueError, match="discontinuous"):
        runner._min_cash_and_borrowing(probe, np.zeros(2))


LAST_ROW = 470  # AAA's last available bar when it is delisted in the fixtures
NOTICE_LEAD = 42  # bars between the notice's first forced row and the last bar
NOTICE_ROW = LAST_ROW - NOTICE_LEAD  # 428: the first forced decision row


def _notice_book(row: int = NOTICE_ROW) -> Any:
    """A notice released exactly at the open of ``row`` (that row is the first forced)."""
    return fx.book(("AAAUSDT", fx.close_of(row - 1)))


def _delisted(**extra: object) -> data.StageData:
    return fx.stage(
        overrides={
            "AAAUSDT": {"last_hour": fx.last_hour_of_row(fx.WINDOW, LAST_ROW), **extra}
        },
        notice_book=_notice_book(),
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
        forced=stage_data.forced,
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


def test_the_forced_close_starts_at_the_notice_bar_and_charges_five_percent() -> None:
    stage_data = _delisted()
    result = _replay(stage_data)
    dataset = stage_data.dataset
    leg = dataset.symbols.index("AAAUSDT")
    start = NOTICE_ROW  # 428
    # held long (bought at the weekly decision of row 419) until the announcement window
    held = _interval(result, start)
    assert Fraction(held.exact_quantities_after[leg]) > 0
    by_index = {d.index: d for d in result.decisions}
    assert by_index[start - 1].proposal_weights[leg] > 0.0
    assert by_index[start].proposal_weights[leg] == 0.0
    # the next weekly decision (row 461) lies in the window: the name is not bought again
    assert by_index[461].proposal_weights[leg] == 0.0
    closing = _interval(result, start + 1)
    fills = [
        event
        for event in closing.order_events
        if event.symbol_index == leg
        and event.processing_index == start + 1
        and event.event_type in ("filled", "partial_fill")
    ]
    assert len(fills) == 1
    closed_quantity = abs(Fraction(held.exact_quantities_after[leg]))
    assert float(closed_quantity) == pytest.approx(abs(fills[0].filled_quantity))
    notional = fills[0].filled_notional
    # no position remains afterwards, on the last bar or at the end
    assert Fraction(_interval(result, LAST_ROW).exact_quantities_after[leg]) == 0
    assert result.book.quantities[leg] == 0.0
    # the engine already charged the scenario cost on that fill; the harness adds exactly 5%
    assert closing.interval_cost > 0.0001 * notional
    events = runner.forced_closes(result, dataset, stage_data.forced)
    assert [(e.leg, e.last_bar) for e in events] == [("AAAUSDT", LAST_ROW)]
    assert events[0].fills == ((start + 1, pytest.approx(notional)),)
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
    fill_bar = NOTICE_ROW + 1
    fill_position = fill_bar - fx.WINDOW.first_decision_index - 1
    ledger_ends = [iv.next_index for iv in result.ledger_evidence.intervals.intervals]  # type: ignore[union-attr]
    assert ledger_ends[fill_position] == fill_bar
    assert np.allclose(path[:fill_position], raw[:fill_position])
    assert path[fill_position] == pytest.approx(raw[fill_position] - charge, rel=1e-9)
    assert path[-1] == pytest.approx(raw[-1] - charge, rel=1e-6)
    # reported and gating: the cell is still valid and cash is not borrowed
    assert cell.valid and cell.max_borrowing is not None and cell.max_borrowing < 0.1


def _thin_delisted() -> data.StageData:
    # from the first forced-close fill bar on the name is too thin to fill anything (1% of
    # ~4 USDT), so the position is still held at its last available bar
    thin_from = NOTICE_ROW * protocol.BAR_HOURS
    return fx.stage(
        overrides={
            "AAAUSDT": {
                "last_hour": fx.last_hour_of_row(fx.WINDOW, LAST_ROW),
                "volume_from": (thin_from, 1.0),
            }
        },
        notice_book=_notice_book(),
    )


def test_a_position_held_at_the_last_bar_is_written_off_at_that_close_with_the_five_percent() -> (
    None
):
    stage_data = _thin_delisted()
    dataset = stage_data.dataset
    leg = dataset.symbols.index("AAAUSDT")
    # only the delisted leg is held, so the settlement interval has no other cash flow
    weights = np.zeros(len(dataset.symbols))
    weights[leg] = 0.25
    result = _replay(stage_data, policy_override=_Constant(weights), risk=False)
    held = _interval(result, LAST_ROW)
    quantity = Fraction(held.exact_quantities_after[leg])
    assert quantity > 0  # still held at the last available bar
    settle = _interval(result, LAST_ROW + 1)
    assert Fraction(settle.exact_quantities_after[leg]) == 0
    assert result.book.quantities[leg] == 0.0
    price = float(dataset.close[LAST_ROW, leg])
    assert float(dataset.open[LAST_ROW + 1, leg]) == price
    proceeds = float(quantity) * price
    # cash conservation: the settlement adds exactly the quantity times the last close (the
    # engine books no fill, fee or funding on an inactive bar) and equity is unchanged by it
    assert settle.cash_after - settle.cash_before == pytest.approx(proceeds, rel=1e-9)
    assert settle.portfolio_value_after == pytest.approx(
        settle.portfolio_value_before, rel=1e-9
    )
    events = runner.write_offs(result, dataset)
    assert [(e.leg, e.last_bar, e.settlement_bar) for e in events] == [
        ("AAAUSDT", LAST_ROW, LAST_ROW + 1)
    ]
    assert events[0].notional == pytest.approx(proceeds, rel=1e-9)
    assert events[0].charge == pytest.approx(0.05 * proceeds, rel=1e-9)
    cell = _cell(stage_data, result, protocol.CANDIDATE)
    stats = cell.cap_statistics
    assert stats["write_off_count"] == 1
    assert stats["write_off_events"] == [
        {
            "symbol": "AAAUSDT",
            "last_bar": LAST_ROW,
            "bar": LAST_ROW + 1,
            "settled_notional": pytest.approx(proceeds, rel=1e-9),
            "charge": pytest.approx(0.05 * proceeds, rel=1e-9),
        }
    ]
    charge = stats["write_off_total_charge_usdt"]
    assert charge == pytest.approx(0.05 * proceeds, rel=1e-9)
    # the charge is a cash debit on the equity path from the settlement bar on
    raw = protocol.INITIAL_CAPITAL * np.cumprod(
        1.0 + np.asarray(result.returns.values, dtype=np.float64)
    )
    path = protocol.INITIAL_CAPITAL * np.cumprod(1.0 + np.asarray(cell.returns))
    position = LAST_ROW + 1 - fx.WINDOW.first_decision_index - 1
    forced_charge = stats["forced_close"]["total_charge_usdt"]
    assert path[position] == pytest.approx(
        raw[position] - forced_charge - charge, rel=1e-6
    )
    assert cell.cap_statistics["forced_close"]["without_charge"][
        "total_return"
    ] - cell.total_return == pytest.approx(
        (forced_charge + charge) / protocol.INITIAL_CAPITAL,
        abs=1e-9,
    )
    assert cell.valid


def test_a_long_and_a_short_leg_are_both_written_off_at_their_last_close() -> None:
    thin_from = NOTICE_ROW * protocol.BAR_HOURS
    overrides = {
        name: {
            "last_hour": fx.last_hour_of_row(fx.WINDOW, LAST_ROW),
            "volume_from": (thin_from, 1.0),
        }
        for name in ("AAAUSDT", "BBBUSDT")
    }
    stage_data = fx.stage(
        overrides=overrides,
        notice_book=fx.book(
            ("AAAUSDT", fx.close_of(NOTICE_ROW - 1)),
            ("BBBUSDT", fx.close_of(NOTICE_ROW - 1)),
        ),
    )
    dataset = stage_data.dataset
    names = list(dataset.symbols)
    weights = np.zeros(len(names))
    weights[names.index("AAAUSDT")] = 0.2
    weights[names.index("BBBUSDT")] = -0.2
    result = _replay(stage_data, policy_override=_Constant(weights), risk=False)
    events = {e.leg: e for e in runner.write_offs(result, dataset)}
    assert set(events) == {"AAAUSDT", "BBBUSDT"}
    before = _interval(result, LAST_ROW + 1)
    after_cash = before.cash_after - before.cash_before
    signed = 0.0
    for name, event in events.items():
        leg = names.index(name)
        quantity = float(Fraction(before.exact_quantities_before[leg]))
        assert (quantity > 0) == (name == "AAAUSDT")  # long and short
        assert Fraction(before.exact_quantities_after[leg]) == 0
        price = float(dataset.close[LAST_ROW, leg])
        assert event.notional == pytest.approx(abs(quantity) * price, rel=1e-9)
        assert event.charge == pytest.approx(0.05 * abs(quantity) * price, rel=1e-9)
        signed += quantity * price
    # the cash identity of the settlement interval: short proceeds are negative
    assert after_cash == pytest.approx(signed, rel=1e-9, abs=1e-6)
    assert before.portfolio_value_after == pytest.approx(
        before.portfolio_value_before, rel=1e-9
    )


def test_a_settlement_the_ledger_cannot_show_exactly_still_refuses() -> None:
    stage_data = _thin_delisted()
    result = _replay(stage_data)
    leg = stage_data.dataset.symbols.index("AAAUSDT")
    ledger = result.ledger_evidence
    assert ledger is not None
    intervals = list(ledger.intervals.intervals)
    position = [iv.next_index for iv in intervals].index(LAST_ROW + 1)
    quantities = list(intervals[position].exact_quantities_after)
    quantities[leg] = intervals[position].exact_quantities_before[leg]  # not flat
    intervals[position] = replace(
        intervals[position], exact_quantities_after=tuple(quantities)
    )
    broken = replace(
        result,
        ledger_evidence=replace(
            ledger, intervals=replace(ledger.intervals, intervals=tuple(intervals))
        ),
    )
    with pytest.raises(runner.UnsupportedWriteOffError, match="AAAUSDT"):
        runner.write_offs(broken, stage_data.dataset)
    assert issubclass(runner.UnsupportedWriteOffError, StageDataError)


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
            # the gap that no halt explains is an event (a member leg): it needs a notice
            notices=fx.book(("AAAUSDT", fx.close_of(98))),
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
    ledger = result.ledger_evidence
    assert ledger is not None
    quantities = list(ledger.intervals.terminal_exact_quantities)
    quantities[0] = "1000"
    intervals = replace(ledger.intervals, terminal_exact_quantities=tuple(quantities))
    return replace(result, ledger_evidence=replace(ledger, intervals=intervals))


# ===================================================================== terminal unwind


def _thin_tail_stage() -> data.StageData:
    """Every leg is thin from the last weekly decision on (1% of ~1M per 4h bar)."""
    start = fx.WINDOW.first_decision_index
    unwind_row = runner.last_weekly_row(start, fx.WINDOW.last_row_index)
    thin_from = (unwind_row - 1) * protocol.BAR_HOURS
    return fx.stage(
        overrides={s: {"volume_from": (thin_from, 250_000.0)} for s in fx.SYMBOLS}
    )


def _terminal_cell(
    stage_data: data.StageData, arm: str, *, unwind: bool
) -> tuple[Any, SharedCashPortfolioReplayResult]:
    start = fx.WINDOW.first_decision_index
    stop = stage_data.dataset.n_bars - 1
    pol = runner.make_policy(
        arm,
        inputs=runner.inputs_of(stage_data),
        start_index=start,
        lag=0,
        last_weekly_row=runner.last_weekly_row(start, stop) if unwind else None,
    )
    result = runner.replay(
        stage_data.dataset,
        pol,
        scenario=protocol.BASE_SCENARIO,
        start_index=start,
        stop_index=stop,
        risk=PreTradeRisk(protocol.risk_config()),
    )
    return _cell(stage_data, result, arm), result


def test_the_last_weekly_decision_is_the_last_monday_before_the_last_engine_decision() -> (
    None
):
    start = fx.WINDOW.first_decision_index
    assert runner.last_weekly_row(start, fx.WINDOW.last_row_index) == 461
    # the frozen stages: the last Monday decision bar is the last weekly row
    for name in protocol.STAGES:
        window = data.frozen_window(name)
        row = runner.last_weekly_row(window.first_decision_index, window.last_row_index)
        assert row == window.row_index(window.last_decision_bar)


def test_a_one_bar_terminal_close_of_a_thin_book_leaves_a_residual_and_the_unwind_clears_it() -> (
    None
):
    stage_data = _thin_tail_stage()
    for arm in (protocol.CANDIDATE, protocol.ARM_EW_LONG):
        without, _ = _terminal_cell(stage_data, arm, unwind=False)
        assert not without.terminal_flat
        assert "terminal_position_not_flat" in without.invalid_reasons
        assert not without.valid
        with_unwind, result = _terminal_cell(stage_data, arm, unwind=True)
        assert with_unwind.terminal_flat and with_unwind.valid, (
            with_unwind.invalid_reasons
        )
        # the book is flat before the end: the unwind completed over the last week
        assert all(float(Fraction(q)) == 0.0 for q in result.book.exact_quantities)
        by_index = {d.index: d for d in result.decisions}
        assert not any(by_index[461].proposal_weights)
        assert not any(by_index[470].proposal_weights)


def test_every_stage_policy_holds_no_book_after_the_last_weekly_decision() -> None:
    stage_data = fx.stage()
    start = fx.WINDOW.first_decision_index
    stop = stage_data.dataset.n_bars - 1
    for arm in protocol.ARMS:
        pol = runner.make_policy(
            arm,
            inputs=runner.inputs_of(stage_data),
            start_index=start,
            lag=0,
            last_weekly_row=runner.last_weekly_row(start, stop),
        )
        result = runner.replay(
            stage_data.dataset,
            pol,
            scenario=protocol.BASE_SCENARIO,
            start_index=start,
            stop_index=stop,
            risk=PreTradeRisk(protocol.risk_config()),
        )
        late = [d for d in result.decisions if d.index >= 461]
        assert late and all(not any(d.proposal_weights) for d in late)


def test_the_settlement_reconciles_only_when_the_package_opts_in() -> None:
    from trade_rl.evaluation.core7_mtf_study.runner import _check_ledger_reconciled

    stage_data = _thin_delisted()
    result = _replay(stage_data)
    ledger = result.ledger_evidence
    dataset = stage_data.dataset
    capital = protocol.INITIAL_CAPITAL
    assert _check_ledger_reconciled(ledger, dataset, capital, engine_settlement=True)
    # the shared default still rebuilds quantities from fills only
    assert not _check_ledger_reconciled(ledger, dataset, capital)
