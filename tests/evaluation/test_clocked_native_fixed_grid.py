"""Monthly publication on independent complete native spot grids, synthetic only."""

from dataclasses import replace
from fractions import Fraction

import numpy as np
import pytest

from trade_rl.data.artifacts.codec import load_dataset_files, write_market_dataset_files
from trade_rl.data.build.native_spot import build_native_fixed_spot_dataset
from trade_rl.data.contracts import FeatureKind, FeatureSpec, InstrumentContract
from trade_rl.data.features.clocked_overlay import (
    ClockedNativeFeatureGroup,
    ClockedNativeSource,
    clocked_native_source_digest,
)
from trade_rl.data.identity import parse_identity_json
from trade_rl.data.native_clock import NativeBarClock, NativeClockKind
from trade_rl.data.source import NativeRawMarketSeries
from trade_rl.evaluation.clocked_native_replay import (
    clocked_native_decision_schedule,
    run_clocked_native_portfolio_replay,
)
from trade_rl.evaluation.native_fixed_returns import (
    load_native_fixed_return_payload,
    native_fixed_return_payload,
)
from trade_rl.simulation import ExecutionCostConfig
from trade_rl.strategies.portfolio import PortfolioTargetProposal


def _source(contract, clock, closes, *, prices, received=None):
    n = len(closes)
    raw = NativeRawMarketSeries(
        timestamps=closes,
        available_at=closes if received is None else received,
        open=prices.copy(),
        high=prices + 2.0,
        low=prices - 2.0,
        close=prices + 1.0,
        volume=np.full(n, 10000.0),
        funding_rate=np.zeros(n),
        funding_available=np.zeros(n, dtype=bool),
        funding_event_count=np.zeros(n, dtype=np.int32),
        tradable=np.ones(n, dtype=bool),
    )
    return ClockedNativeSource(
        contract, raw, clocked_native_source_digest(raw, contract=contract, clock=clock)
    )


def _fixture(days, *, late_ns=1, future_bump=0.0):
    anchor = np.datetime64("2024-01-01", "ns")
    fine_clock = (
        NativeBarClock(NativeClockKind.UTC_WEEK)
        if days == 7
        else NativeBarClock(NativeClockKind.FIXED_ELAPSED, days * 86400, anchor)
    )
    fine = np.arange(
        anchor + np.timedelta64(days, "D"),
        np.datetime64("2024-04-23", "ns"),
        np.timedelta64(days, "D"),
    )
    coarse_clock = NativeBarClock(NativeClockKind.UTC_MONTH)
    coarse = np.arange("2024-02", "2024-05", dtype="datetime64[M]").astype(
        "datetime64[ns]"
    )
    contracts = (InstrumentContract("A"), InstrumentContract("B"))
    execution = tuple(
        _source(c, fine_clock, fine, prices=100.0 + 100 * i + np.arange(len(fine)))
        for i, c in enumerate(contracts)
    )
    sources = []
    for i, c in enumerate(contracts):
        receipts = coarse.copy()
        receipts[0] = np.datetime64("2024-02-05", "ns") + np.timedelta64(
            i * late_ns, "ns"
        )
        # The March publication is stale by the April boundary.
        receipts[1] = coarse[2]
        prices = np.array([300.0, 310.0, 320.0 + future_bump]) + 100 * i
        sources.append(
            _source(c, coarse_clock, coarse, prices=prices, received=receipts)
        )
    group = ClockedNativeFeatureGroup(
        coarse_clock,
        tuple(sources),
        (
            FeatureSpec(
                "month_body", FeatureKind.BODY_RETURN, max_staleness_hours=31 * 24
            ),
        ),
    )
    base = ClockedNativeFeatureGroup(
        fine_clock,
        execution,
        (FeatureSpec("native_body", FeatureKind.BODY_RETURN),),
    )
    dataset = build_native_fixed_spot_dataset(
        clock=fine_clock,
        closes=fine,
        sources=execution,
        feature_groups=(base, group),
        as_of=fine[-1],
    )
    return dataset, group, base


def _cost(**changes):
    return replace(
        ExecutionCostConfig.zero(),
        **(
            dict(
                allow_short=False,
                partial_fill_carry=False,
                processing_bar_volume_capacity=False,
            )
            | changes
        ),
    )


@pytest.mark.parametrize("days,expected", [(3, "2024-02-06"), (7, "2024-02-12")])
def test_non_nested_month_publication_on_actual_fixed_grid(days, expected):
    d, group, _ = _fixture(days)
    slots = clocked_native_decision_schedule(
        d, group, start_index=0, stop_index=d.n_bars - 1
    )
    assert [s.status for s in slots] == ["scheduled", "expired", "scheduled"]
    assert np.datetime64(slots[0].decision_time_ns, "ns") == np.datetime64(
        expected, "ns"
    )
    assert (
        slots[0].available_at_ns
        == int(np.datetime64("2024-02-05", "ns").astype(np.int64)) + 1
    )
    assert np.datetime64(slots[2].decision_time_ns, "ns") == np.datetime64(
        "2024-04-01" if days == 7 else "2024-04-03", "ns"
    )
    assert d.periods_per_year == Fraction(365, days)
    later = clocked_native_decision_schedule(
        d, group, start_index=slots[0].decision_index + 1, stop_index=d.n_bars - 1
    )
    assert later[0].status == "outside_window"


def test_weekly_exact_tie_vs_one_ns_and_future_mutation():
    a, g, _ = _fixture(7, late_ns=0)
    b, h, _ = _fixture(7, late_ns=1)
    c, k, _ = _fixture(7, late_ns=1, future_bump=100)

    def slots(d, group):
        return clocked_native_decision_schedule(
            d, group, start_index=0, stop_index=d.n_bars - 1
        )

    assert np.datetime64(slots(a, g)[0].decision_time_ns, "ns") == np.datetime64(
        "2024-02-05", "ns"
    )
    assert slots(b, h)[0].decision_index == slots(a, g)[0].decision_index + 1
    assert slots(b, h) == slots(c, k)
    assert b.dataset_id != c.dataset_id
    idx = slots(b, h)[0].decision_index
    np.testing.assert_array_equal(b.features[: idx + 1], c.features[: idx + 1])


@pytest.mark.parametrize("days", [3, 7])
def test_fixed_grid_accounting_and_native_return_identity(days, tmp_path):
    dataset, group, _ = _fixture(days)
    written = write_market_dataset_files(tmp_path, dataset)
    d = load_dataset_files(tmp_path)
    assert written.schema_version == "market_dataset_artifact_v5"
    assert d.dataset_id == dataset.dataset_id
    first = clocked_native_decision_schedule(
        d, group, start_index=0, stop_index=d.n_bars - 1
    )[0].decision_index
    assert first is not None
    calls = []

    class Policy:
        def decide_portfolio(self, snapshot):
            calls.append(snapshot.index)
            assert snapshot.feature_available[0][1]
            return PortfolioTargetProposal((0.25, 0.0))

    stop = first + 4
    result = run_clocked_native_portfolio_replay(
        d,
        Policy(),
        decision_group=group,
        start_index=first - 1,
        stop_index=stop,
        initial_capital=1000.0,
        execution_cost=_cost(fee_rate=0.001),
        capture_ledger_evidence=True,
    )
    assert calls == [first]
    assert result.policy_decision_indices == (first,)
    ledger = result.replay.ledger_evidence
    assert ledger is not None
    cash = previous = 1000.0
    quantity = 0.0
    target = 250.0 / d.close[first, 0]
    for row in ledger.intervals.intervals:
        idx = row.next_index
        delta = target if idx == first + 1 else -quantity if idx == stop else 0.0
        fee = abs(delta) * d.open[idx, 0] * 0.001
        cash -= delta * d.open[idx, 0] + fee
        quantity += delta
        equity = cash + quantity * d.close[idx, 0]
        assert float(Fraction(row.exact_quantities_after[0])) == pytest.approx(
            quantity, abs=1e-12
        )
        assert row.cash_after == pytest.approx(cash)
        assert row.portfolio_value_after == pytest.approx(equity)
        assert row.interval_cost == pytest.approx(fee)
        assert row.interval_net_return == pytest.approx(
            equity / previous - 1, abs=1e-12
        )
        previous = equity
    assert result.replay.book.exact_quantities == (0, 0)
    assert ledger.intervals.active_order_remainders == ()
    payload = native_fixed_return_payload(
        d, result.replay.returns, start_index=first - 1
    )
    elapsed = Fraction(5 * days, 365)
    assert payload["elapsed_years"] == {
        "numerator": elapsed.numerator,
        "denominator": elapsed.denominator,
    }
    assert load_native_fixed_return_payload(d, payload) == result.replay.returns


def test_native_fixed_schedule_rejects_same_clock_and_unbound_sources():
    d, g, base = _fixture(7)
    src = g.sources[0]
    raw = replace(src.raw, available_at=src.raw.available_at + np.timedelta64(1, "ns"))
    altered = replace(src, raw=raw)

    def schedule(group, dataset=d):
        return clocked_native_decision_schedule(
            dataset, group, start_index=0, stop_index=d.n_bars - 1
        )

    with pytest.raises(ValueError, match="larger"):
        schedule(base)
    with pytest.raises(ValueError, match="source digest"):
        schedule(replace(g, sources=(altered, g.sources[1])))
    with pytest.raises(ValueError, match="bound"):
        schedule(replace(g, specs=(FeatureSpec("foreign", FeatureKind.BODY_RETURN),)))
    with pytest.raises(ValueError, match="bound"):
        schedule(
            replace(
                g, sources=(replace(src, expected_source_digest="f" * 64), g.sources[1])
            )
        )
    with pytest.raises(ValueError, match="contract"):
        schedule(
            replace(
                g,
                sources=(
                    replace(src, contract=replace(src.contract, lot_size=0.5)),
                    g.sources[1],
                ),
            )
        )
    identity = parse_identity_json(d.identity_payload_json)
    identity["feature_source_digests"] = identity["feature_source_digests"][:-1]
    bad = replace(d, identity_payload_json=None).with_content_identity(identity)
    with pytest.raises(ValueError, match="binding"):
        schedule(g, bad)


def test_native_fixed_clocked_replay_refuses_nonspot_before_policy():
    d, g, _ = _fixture(7)

    class Never:
        def decide_portfolio(self, snapshot):
            pytest.fail("unsupported linear carry reached policy")

    with pytest.raises(ValueError, match="native fixed.*strict spot"):
        run_clocked_native_portfolio_replay(
            d,
            Never(),
            decision_group=g,
            start_index=0,
            stop_index=d.n_bars - 1,
            initial_capital=1000.0,
            execution_cost=_cost(),
            execution_mode="linear_carry",
        )


def test_weekly_pending_order_uses_native_latency_and_no_coarse_resubmission():
    d, group, _ = _fixture(7)
    first = clocked_native_decision_schedule(
        d, group, start_index=0, stop_index=d.n_bars - 1
    )[0].decision_index
    assert first is not None
    calls = []

    class Policy:
        def decide_portfolio(self, snapshot):
            calls.append(snapshot.index)
            return PortfolioTargetProposal((0.25, 0.0))

    stop = first + 7
    result = run_clocked_native_portfolio_replay(
        d,
        Policy(),
        decision_group=group,
        start_index=first - 1,
        stop_index=stop,
        initial_capital=1000.0,
        execution_cost=_cost(order_latency_bars=2, fee_rate=0.001),
        continue_pending_orders=True,
        capture_ledger_evidence=True,
    )
    assert calls == [first]
    assert result.policy_decision_indices == (first,)
    ledger = result.replay.ledger_evidence
    assert ledger is not None
    assert (
        ledger.to_mapping()["schema_version"]
        == "shared_cash_portfolio_replay_ledger_v3"
    )
    assert ledger.to_mapping()["actor_decision_indices"] == (first,)
    events = [e for row in ledger.intervals.intervals for e in row.order_events]
    submitted = [e for e in events if e.event_type == "submitted"]
    filled = [e for e in events if e.event_type == "filled"]
    assert len(submitted) == len(filled) == 2
    assert [e.processing_index for e in submitted] == [first, stop - 3]
    assert [e.processing_index for e in filled] == [first + 2, stop - 1]
    assert [e.order_id for e in submitted] == [e.order_id for e in filled]
    assert result.replay.book.exact_quantities == (0, 0)
    assert ledger.intervals.active_order_remainders == ()
    quantity = 250.0 / d.close[first, 0]
    buy = d.open[first + 2, 0]
    sell = d.open[stop - 1, 0]
    assert result.replay.book.cash == pytest.approx(
        1000 + quantity * (sell - buy) - quantity * (buy + sell) * 0.001
    )


def test_native_fixed_requires_bound_builder_and_feature_roster():
    d, group, _ = _fixture(7)
    identity = parse_identity_json(d.identity_payload_json)
    unbound = replace(d, identity_payload_json=None).with_content_identity()
    with pytest.raises(ValueError, match="bound spot build"):
        clocked_native_decision_schedule(
            unbound, group, start_index=0, stop_index=d.n_bars - 1
        )
    bad = replace(
        d, identity_payload_json=None, feature_names=("native_body", "foreign")
    ).with_content_identity(identity)
    with pytest.raises(ValueError, match="feature roster"):
        clocked_native_decision_schedule(
            bad, group, start_index=0, stop_index=d.n_bars - 1
        )
