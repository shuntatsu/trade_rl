"""Literal publication-clock and next-open synthetic replay oracles."""

import math
from dataclasses import fields, replace
from fractions import Fraction

import numpy as np
import pytest

from tests.data.test_clocked_feature_overlay import _fixture
from trade_rl.data.contracts import FeatureKind, FeatureSpec
from trade_rl.data.features.clocked_overlay import (
    ClockedNativeFeatureGroup,
    ClockedNativeSource,
    append_clocked_native_feature_groups,
    clocked_native_source_digest,
)
from trade_rl.data.source import NativeRawMarketSeries, RawMarketSeries
from trade_rl.evaluation import clocked_native_replay as scheduling
from trade_rl.risk import PreTradeRisk, PreTradeRiskConfig
from trade_rl.simulation import ExecutionCostConfig
from trade_rl.strategies.portfolio import (
    PortfolioDecisionSnapshot,
    PortfolioTargetProposal,
)


def _group(kind="3d", *, expire=True, first_delay_ns=1):
    base, clock, sources, _ = _fixture(kind)
    pinned = []
    for i, source in enumerate(sources):
        raw = source.raw
        receipt = raw.timestamps.copy()
        receipt[0] += np.timedelta64(2, "h")
        if i:
            receipt[0] += np.timedelta64(first_delay_ns, "ns")
            if expire:
                receipt[1] = raw.timestamps[2]
        strict = NativeRawMarketSeries(
            **(
                {f.name: getattr(raw, f.name) for f in fields(RawMarketSeries)}
                | {"available_at": receipt}
            )
        )
        pinned.append(
            ClockedNativeSource(
                source.contract,
                strict,
                clocked_native_source_digest(
                    strict, contract=source.contract, clock=clock
                ),
            )
        )
    group = ClockedNativeFeatureGroup(
        clock,
        tuple(pinned),
        (FeatureSpec("native_body", FeatureKind.BODY_RETURN, max_staleness_hours=48),),
    )
    dataset = append_clocked_native_feature_groups(base, groups=(group,))
    return dataset, group


def _index(dataset, time):
    return int(np.flatnonzero(dataset.timestamps == time)[0])


@pytest.mark.parametrize("kind", ["3d", "week", "month"])
def test_literal_group_receipt_expiry_and_terminal_window(kind):
    dataset, group = _group(kind)
    slots = scheduling.clocked_native_decision_schedule(
        dataset, group, start_index=0, stop_index=dataset.n_bars - 1
    )
    closes = group.sources[0].raw.timestamps
    first = _index(dataset, closes[0])
    third = _index(dataset, closes[2])
    assert [slot.status for slot in slots] == [
        "scheduled",
        "expired",
        "scheduled",
        "outside_window",
    ]
    assert [slot.decision_index for slot in slots] == [first + 3, None, third, None]
    assert (
        slots[0].available_at_ns == int(closes[0].astype(np.int64)) + 7_200_000_000_001
    )
    assert slots[0].decision_time_ns == int(
        dataset.timestamps[first + 3].astype(np.int64)
    )


def test_equal_boundary_and_no_catchup_after_window_start():
    dataset, group = _group(first_delay_ns=0)
    close = group.sources[0].raw.timestamps[0]
    first = _index(dataset, close)
    slots = scheduling.clocked_native_decision_schedule(
        dataset, group, start_index=first + 2, stop_index=first + 8
    )
    assert slots[0].decision_index == first + 2
    later = scheduling.clocked_native_decision_schedule(
        dataset, group, start_index=first + 3, stop_index=first + 8
    )
    assert later[0].status == "outside_window"


def test_refuses_changed_pin_spec_roster_and_unverified_dataset():
    dataset, group = _group()
    bad_source = replace(group.sources[0], expected_source_digest="f" * 64)
    for altered in (
        replace(group, sources=(bad_source, group.sources[1])),
        replace(group, sources=group.sources[::-1]),
        replace(group, specs=(FeatureSpec("wrong", FeatureKind.BODY_RETURN),)),
    ):
        with pytest.raises(ValueError):
            scheduling.clocked_native_decision_schedule(
                dataset, altered, start_index=0, stop_index=dataset.n_bars - 1
            )
    with pytest.raises(ValueError, match="identity"):
        scheduling.clocked_native_decision_schedule(
            replace(dataset, identity_payload_json=None),
            group,
            start_index=0,
            stop_index=dataset.n_bars - 1,
        )


@pytest.mark.parametrize(
    "start,stop", [(True, 5), (0, True), (-1, 5), (5, 5), (0, 999999)]
)
def test_window_validation(start, stop):
    dataset, group = _group()
    with pytest.raises(ValueError, match="window"):
        scheduling.clocked_native_decision_schedule(
            dataset, group, start_index=start, stop_index=stop
        )


def test_actual_next_open_replay_one_call_with_holds_and_terminal_flat():
    dataset, group = _group()
    first = _index(dataset, group.sources[0].raw.timestamps[0])
    seen = []

    class Policy:
        def decide_portfolio(
            self, snapshot: PortfolioDecisionSnapshot
        ) -> PortfolioTargetProposal:
            seen.append(snapshot.index)
            assert all(row[1] for row in snapshot.feature_available)
            return PortfolioTargetProposal((0.3, 0.0))

    result = scheduling.run_clocked_native_portfolio_replay(
        dataset,
        Policy(),
        decision_group=group,
        start_index=first,
        stop_index=first + 9,
        initial_capital=1000,
        execution_cost=replace(
            ExecutionCostConfig.zero(),
            allow_short=False,
            processing_bar_volume_capacity=False,
            partial_fill_carry=False,
            fee_rate=0.001,
        ),
        risk=PreTradeRisk(PreTradeRiskConfig(max_gross=1.0, max_abs_weight=1.0)),
        capture_ledger_evidence=True,
    )
    assert seen == [first + 3]
    assert result.policy_decision_indices == tuple(seen)
    replay = result.replay
    assert len(replay.returns.values) == 9
    assert replay.returns.periods_per_year == 8760
    assert replay.ledger_evidence is not None
    fills = [
        e
        for row in replay.ledger_evidence.intervals.intervals
        for e in row.order_events
        if e.event_type in {"filled", "partial_fill"}
    ]
    assert [e.processing_index for e in fills] == [first + 4, first + 9]
    assert [e.execution_price for e in fills] == pytest.approx(
        [dataset.open[first + 4, 0], dataset.open[first + 9, 0]]
    )
    assert replay.book.total_cost == pytest.approx(
        sum(e.filled_notional for e in fills) * 0.001
    )
    assert replay.book.exact_quantities == (0, 0)
    assert replay.book.total_cost > 0
    assert all(
        d.requested_hold_symbols == (0, 1)
        for d in replay.decisions
        if d.index not in seen
    )


@pytest.mark.parametrize(
    "change",
    [
        {"order_latency_bars": 1},
        {"partial_fill_carry": True},
        {"processing_bar_volume_capacity": True},
        {"allow_short": True},
    ],
)
def test_refuses_ambiguous_pending_or_lookahead_execution(change):
    dataset, group = _group()

    class Never:
        def decide_portfolio(self, snapshot):
            pytest.fail("unsupported execution called policy")

    cost = replace(
        ExecutionCostConfig.zero(),
        allow_short=False,
        partial_fill_carry=False,
        processing_bar_volume_capacity=False,
    )
    with pytest.raises(ValueError, match="scheduled replay requires"):
        scheduling.run_clocked_native_portfolio_replay(
            dataset,
            Never(),
            decision_group=group,
            start_index=0,
            stop_index=10,
            initial_capital=1000,
            execution_cost=replace(cost, **change),
        )


def test_schedule_refuses_changed_source_bytes_even_with_matching_declared_pin():
    dataset, group = _group()
    source = group.sources[0]
    receipt = source.raw.available_at.copy()
    receipt[0] += np.timedelta64(1, "ns")
    altered = replace(source, raw=replace(source.raw, available_at=receipt))
    with pytest.raises(ValueError, match="source digest"):
        scheduling.clocked_native_decision_schedule(
            dataset,
            replace(group, sources=(altered, group.sources[1])),
            start_index=0,
            stop_index=dataset.n_bars - 1,
        )


def test_legacy_normalized_receipts_are_not_explicit_native_sources():
    dataset, group = _group()
    source = group.sources[0]
    raw = RawMarketSeries(
        **{f.name: getattr(source.raw, f.name) for f in fields(RawMarketSeries)}
    )
    altered = replace(source, raw=raw)
    # Same normalized digest is not a proof of explicit constructor admission.
    assert (
        clocked_native_source_digest(raw, contract=source.contract, clock=group.clock)
        == source.expected_source_digest
    )
    with pytest.raises(ValueError, match="strict explicit"):
        scheduling.clocked_native_decision_schedule(
            dataset,
            replace(group, sources=(altered, group.sources[1])),
            start_index=0,
            stop_index=dataset.n_bars - 1,
        )


def test_future_receipt_change_does_not_retime_earlier_decision():
    first, original = _group(expire=True)
    second, changed = _group(expire=False)
    a = scheduling.clocked_native_decision_schedule(
        first, original, start_index=0, stop_index=first.n_bars - 1
    )
    b = scheduling.clocked_native_decision_schedule(
        second, changed, start_index=0, stop_index=second.n_bars - 1
    )
    assert first.dataset_id != second.dataset_id
    assert a[0] == b[0]
    assert a[1].status == "expired" and b[1].status == "scheduled"


@pytest.mark.parametrize(
    "hours_before_boundary, expected", [(0, "expired"), (1, "scheduled")]
)
def test_month_expiry_on_exact_calendar_boundary_not_30_days(
    hours_before_boundary, expected
):
    # The first event closes February 1, and the next closes March 1.
    # February 2024 has 29 days. The latest group receipt is at the
    # next calendar boundary or one complete fine interval earlier.
    delay = (29 * 24 - hours_before_boundary - 2) * 3_600_000_000_000
    dataset, group = _group("month", first_delay_ns=delay, expire=False)
    slots = scheduling.clocked_native_decision_schedule(
        dataset, group, start_index=0, stop_index=dataset.n_bars - 1
    )
    assert slots[0].status == expected
    if expected == "scheduled":
        assert slots[0].decision_time_ns == int(
            np.datetime64("2024-02-29T23", "ns").astype(np.int64)
        )
    else:
        assert slots[0].decision_index is None


def test_unavailable_fine_mark_refused_before_policy_even_with_later_receipt():
    from trade_rl.data.identity import parse_identity_json

    dataset, group = _group()
    first = _index(dataset, group.sources[0].raw.timestamps[0])
    info = dataset.resolved_array("information_available").copy()
    info[first + 1, 0] = False
    # An identity-verified synthetic alteration still cannot pass fine admission.
    altered = replace(
        dataset, identity_payload_json=None, information_available=info
    ).with_content_identity(parse_identity_json(dataset.identity_payload_json))

    class Never:
        def decide_portfolio(self, snapshot):
            pytest.fail("unavailable fine bar called policy")

    with pytest.raises(ValueError, match="complete available fine"):
        scheduling.run_clocked_native_portfolio_replay(
            altered,
            Never(),
            decision_group=group,
            start_index=first,
            stop_index=first + 9,
            initial_capital=1000,
            execution_cost=replace(
                ExecutionCostConfig.zero(),
                allow_short=False,
                processing_bar_volume_capacity=False,
                partial_fill_carry=False,
            ),
        )


def test_declared_receipt_in_terminal_reserve_does_not_call_policy():
    dataset, group = _group()
    first = _index(dataset, group.sources[0].raw.timestamps[0])

    class Never:
        def decide_portfolio(self, snapshot):
            pytest.fail("reserved terminal row called policy")

    # Arrival at first+3; that row is already reserved for flattening.
    result = scheduling.run_clocked_native_portfolio_replay(
        dataset,
        Never(),
        decision_group=group,
        start_index=first,
        stop_index=first + 4,
        initial_capital=1000,
        execution_cost=replace(
            ExecutionCostConfig.zero(),
            allow_short=False,
            processing_bar_volume_capacity=False,
            partial_fill_carry=False,
        ),
    )
    assert result.slots[0].status == "outside_window"
    assert result.policy_decision_indices == ()
    assert result.replay.book.exact_quantities == (0, 0)
    assert result.replay.returns.values == (0, 0, 0, 0)


def test_multi_slot_ledger_and_future_receipt_oracle():
    """Use literal fine marks and two decisions, not another replay as oracle."""
    dataset, group = _group()
    closes = group.sources[0].raw.timestamps
    first = _index(dataset, closes[0])
    middle = _index(dataset, closes[1])
    third = _index(dataset, closes[2])
    enter = first + 3
    stop = third + 4
    cost = replace(
        ExecutionCostConfig.zero(),
        allow_short=False,
        processing_bar_volume_capacity=False,
        partial_fill_carry=False,
    )

    def execute(ds, selected_group):
        seen = []

        class Policy:
            def decide_portfolio(self, snapshot):
                seen.append(snapshot)
                return PortfolioTargetProposal(
                    (0.25, 0.0) if snapshot.index == enter else (0.0, 0.0)
                )

        result = scheduling.run_clocked_native_portfolio_replay(
            ds,
            Policy(),
            decision_group=selected_group,
            start_index=first,
            stop_index=stop,
            initial_capital=1000.0,
            execution_cost=cost,
            risk=PreTradeRisk(PreTradeRiskConfig(max_gross=1.0, max_abs_weight=1.0)),
            capture_ledger_evidence=True,
        )
        return seen, result

    seen, result = execute(dataset, group)
    assert result.policy_decision_indices == (enter, third)
    assert [slot.status for slot in result.slots] == [
        "scheduled",
        "expired",
        "scheduled",
        "outside_window",
    ]
    assert [s.index for s in seen] == [enter, third]
    for snap, coarse_row in zip(seen, (0, 2), strict=True):
        for i, source in enumerate(group.sources):
            assert snap.feature_available[i][1]
            expected = math.log(
                float(source.raw.close[coarse_row]) / float(source.raw.open[coarse_row])
            )
            assert snap.features[i][1] == pytest.approx(expected, rel=1e-7)
    # Order target uses the decision close, fill uses the next fine open.
    # Independently reconstruct the entire zero-fee cash/equity path.
    q = Fraction(250) / Fraction.from_float(float(dataset.close[enter, 0]))
    cash_after_buy = Fraction(1000) - q * Fraction.from_float(
        float(dataset.open[enter + 1, 0])
    )
    cash_after_sell = cash_after_buy + q * Fraction.from_float(
        float(dataset.open[third + 1, 0])
    )
    assert seen[0].account.quantities == (0.0, 0.0)
    assert seen[1].account.quantities == pytest.approx((float(q), 0.0))
    assert seen[1].account.cash == pytest.approx(float(cash_after_buy))
    ledger = result.replay.ledger_evidence
    assert ledger is not None
    rows = ledger.intervals.intervals
    assert len(rows) == stop - first
    previous = Fraction(1000)
    for row, index in zip(rows, range(first, stop), strict=True):
        assert (row.start_index, row.next_index) == (index, index + 1)
        if index < enter:
            quantity, cash = Fraction(0), Fraction(1000)
        elif index < third:
            quantity, cash = q, cash_after_buy
        else:
            quantity, cash = Fraction(0), cash_after_sell
        equity = cash + quantity * Fraction.from_float(
            float(dataset.close[index + 1, 0])
        )
        assert float(Fraction(row.exact_quantities_after[0])) == pytest.approx(
            float(quantity), abs=1e-12
        )
        assert row.exact_quantities_after[1] == "0"
        assert row.cash_after == pytest.approx(float(cash), abs=1e-10)
        assert row.portfolio_value_after == pytest.approx(float(equity), abs=1e-10)
        expected_return = float(equity / previous - 1)
        assert row.interval_net_return == pytest.approx(expected_return, abs=1e-12)
        assert result.replay.returns.values[index - first] == pytest.approx(
            expected_return, abs=1e-12
        )
        previous = equity
    assert result.replay.book.cash == pytest.approx(float(cash_after_sell))
    assert result.replay.book.exact_quantities == (0, 0)
    # Change only a later native receipt. Earlier actor snapshot and the
    # realized prefix before that receipt must be unchanged, not just its slot.
    changed, changed_group = _group(expire=False)
    changed_seen, changed_result = execute(changed, changed_group)
    assert changed_result.policy_decision_indices == (enter, middle, third)
    assert changed_seen[0] == seen[0]
    assert (
        changed_result.replay.returns.values[: middle - first]
        == result.replay.returns.values[: middle - first]
    )


@pytest.mark.parametrize("bad", ["unverified", "changed_pin", "terminal_symbol_b"])
def test_replay_refuses_identity_and_terminal_mark_before_policy(bad):
    from trade_rl.data.identity import parse_identity_json

    dataset, group = _group()
    first = _index(dataset, group.sources[0].raw.timestamps[0])
    stop = first + 9
    expected = "identity"
    if bad == "unverified":
        dataset = replace(dataset, identity_payload_json=None)
    elif bad == "changed_pin":
        group = replace(
            group,
            sources=(
                replace(group.sources[0], expected_source_digest="f" * 64),
                group.sources[1],
            ),
        )
        expected = "bound overlay group"
    else:
        info = dataset.resolved_array("information_available").copy()
        info[stop, 1] = False
        dataset = replace(
            dataset, identity_payload_json=None, information_available=info
        ).with_content_identity(parse_identity_json(dataset.identity_payload_json))
        expected = "complete available fine"

    class Never:
        def decide_portfolio(self, snapshot):
            pytest.fail("invalid input reached policy")

    with pytest.raises(ValueError, match=expected):
        scheduling.run_clocked_native_portfolio_replay(
            dataset,
            Never(),
            decision_group=group,
            start_index=first,
            stop_index=stop,
            initial_capital=1000,
            execution_cost=replace(
                ExecutionCostConfig.zero(),
                allow_short=False,
                processing_bar_volume_capacity=False,
                partial_fill_carry=False,
            ),
        )


@pytest.mark.parametrize("unit", ["h", "s", "ms", "us"])
def test_fine_timestamp_units_preserve_publication_schedule(unit):
    from trade_rl.data.identity import parse_identity_json

    dataset, group = _group()
    expected = scheduling.clocked_native_decision_schedule(
        dataset, group, start_index=0, stop_index=dataset.n_bars - 1
    )
    alternate = replace(
        dataset,
        identity_payload_json=None,
        timestamps=dataset.timestamps.astype(f"datetime64[{unit}]"),
    ).with_content_identity(parse_identity_json(dataset.identity_payload_json))
    assert alternate.identity_verified
    assert alternate.timestamps.dtype == np.dtype(f"datetime64[{unit}]")
    actual = scheduling.clocked_native_decision_schedule(
        alternate, group, start_index=0, stop_index=alternate.n_bars - 1
    )
    assert actual == expected


@pytest.mark.parametrize("unit", ["s", "ms"])
def test_fine_timestamp_units_preserve_actual_actor_and_ledger(unit):
    from trade_rl.data.identity import parse_identity_json

    dataset, group = _group()
    first = _index(dataset, group.sources[0].raw.timestamps[0])
    alternate = replace(
        dataset,
        identity_payload_json=None,
        timestamps=dataset.timestamps.astype(f"datetime64[{unit}]"),
    ).with_content_identity(parse_identity_json(dataset.identity_payload_json))

    class Policy:
        def decide_portfolio(self, snapshot):
            assert snapshot.timestamp_ns == int(
                np.datetime64("2024-01-04T03:00:00", "ns").astype(np.int64)
            )
            return PortfolioTargetProposal((0.3, 0.0))

    def run(data):
        return scheduling.run_clocked_native_portfolio_replay(
            data,
            Policy(),
            decision_group=group,
            start_index=first,
            stop_index=first + 9,
            initial_capital=1000,
            execution_cost=replace(
                ExecutionCostConfig.zero(),
                allow_short=False,
                processing_bar_volume_capacity=False,
                partial_fill_carry=False,
                fee_rate=0.001,
            ),
            capture_ledger_evidence=True,
        )

    expected, actual = run(dataset), run(alternate)
    assert actual.slots == expected.slots
    assert actual.policy_decision_indices == expected.policy_decision_indices
    assert actual.replay.returns.values == expected.replay.returns.values
    assert actual.replay.book.cash == expected.replay.book.cash
    assert actual.replay.book.total_cost == expected.replay.book.total_cost
    assert actual.replay.book.exact_quantities == expected.replay.book.exact_quantities
