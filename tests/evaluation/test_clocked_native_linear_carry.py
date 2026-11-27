"""Independent cash and carry oracles for sparse decisions on a fine grid."""

from dataclasses import replace
from fractions import Fraction

import numpy as np
import pytest

from trade_rl.data import load_market_dataset_artifact, write_market_dataset_files
from trade_rl.data.build import ExecutionEconomicsProfile, MarketDatasetBuilder
from trade_rl.data.contracts import (
    FeatureKind,
    FeatureSpec,
    InstrumentContract,
    MarketBuildConfig,
    VolumeUnit,
)
from trade_rl.data.features.clocked_overlay import (
    ClockedNativeFeatureGroup,
    ClockedNativeSource,
    append_clocked_native_feature_groups,
    clocked_native_source_digest,
)
from trade_rl.data.identity import parse_identity_json
from trade_rl.data.native_clock import NativeBarClock, NativeClockKind
from trade_rl.data.native_resampling import resample_complete_native_bars
from trade_rl.data.source import InMemoryMarketDataSource, NativeRawMarketSeries
from trade_rl.evaluation.clocked_native_replay import (
    run_clocked_native_portfolio_replay,
)
from trade_rl.simulation import ExecutionCostConfig
from trade_rl.strategies.portfolio import PortfolioTargetProposal


def _rebind(dataset, **changes):
    assert dataset.identity_payload_json is not None
    return replace(
        dataset, identity_payload_json=None, **changes
    ).with_content_identity(parse_identity_json(dataset.identity_payload_json))


def _fixture(kind="3d", *, borrow_open_price=200.0):
    anchor = np.datetime64("2024-01-01", "ns")
    if kind == "month":
        clock = NativeBarClock(NativeClockKind.UTC_MONTH)
        closes = np.arange("2024-02", "2024-05", dtype="datetime64[M]").astype(
            "datetime64[ns]"
        )
    else:
        days = 3 if kind == "3d" else 7
        clock = (
            NativeBarClock(NativeClockKind.FIXED_ELAPSED, days * 86400, anchor)
            if days == 3
            else NativeBarClock(NativeClockKind.UTC_WEEK)
        )
        closes = anchor + np.arange(1, 4) * np.timedelta64(days, "D")
    fine_clock = NativeBarClock(NativeClockKind.FIXED_ELAPSED, 3600, anchor)
    fine = np.arange(
        anchor + np.timedelta64(1, "h"),
        closes[-1] + np.timedelta64(1, "h"),
        np.timedelta64(1, "h"),
    ).astype("datetime64[ns]")
    decision = int(np.flatnonzero(fine == closes[0])[0]) + 3
    stop = decision + 6
    contracts = (
        InstrumentContract(
            "SYNTH_A", volume_unit=VolumeUnit.CONTRACTS, contract_multiplier=2.0
        ),
        InstrumentContract(
            "SYNTH_B", volume_unit=VolumeUnit.CONTRACTS, contract_multiplier=4.0
        ),
    )
    raws = {}
    sources = []
    price_rates = np.zeros((len(fine), 2))
    for i, contract in enumerate(contracts):
        prices = np.full(len(fine), 100.0 * (i + 1))
        opens = prices.copy()
        if i == 1:
            opens[decision + 3] = borrow_open_price
        highs = np.maximum(prices, opens)
        lows = np.minimum(prices, opens)
        volumes = np.full(len(fine), 10000.0)
        rates = np.zeros(len(fine))
        count = np.zeros(len(fine), dtype=np.int32)
        # Single close-assigned events after the first open fill, at the
        # next HOLD interval, and after the terminal flatten.
        events = (decision + 1, decision + 2, stop)
        rates[list(events)] = (0.01, -0.005, 0.02) if i == 0 else (0.02, 0.01, 0.03)
        count[list(events)] = 1
        event_prices = (110.0, 90.0, 120.0) if i == 0 else (210.0, 190.0, 220.0)
        price_rates[list(events), i] = rates[list(events)] * np.asarray(event_prices)
        raws[contract.symbol] = NativeRawMarketSeries(
            timestamps=fine,
            available_at=fine,
            open=opens,
            high=highs,
            low=lows,
            close=prices,
            volume=volumes,
            funding_rate=rates,
            funding_available=count > 0,
            funding_event_count=count,
            tradable=np.ones(len(fine), dtype=bool),
        )
        receipt = fine.copy()
        receipt[fine == closes[0]] += np.timedelta64(2, "h") + np.timedelta64(i, "ns")
        coarse = resample_complete_native_bars(
            source_clock=fine_clock,
            source_closes=fine,
            source_available_at=receipt,
            source_ohlcv=np.column_stack(
                (
                    opens,
                    highs,
                    lows,
                    prices,
                    volumes,
                    prices * volumes * contract.contract_multiplier,
                )
            ),
            target_clock=clock,
            target_closes=closes,
            as_of=closes[-1],
        )
        raw = NativeRawMarketSeries(
            timestamps=coarse.closes,
            available_at=coarse.available_at,
            open=coarse.ohlcv[:, 0],
            high=coarse.ohlcv[:, 1],
            low=coarse.ohlcv[:, 2],
            close=coarse.ohlcv[:, 3],
            volume=coarse.ohlcv[:, 4],
            funding_rate=np.zeros(len(closes)),
            funding_available=np.zeros(len(closes), dtype=bool),
            funding_event_count=np.zeros(len(closes), dtype=np.int32),
            tradable=np.ones(len(closes), dtype=bool),
        )
        sources.append(
            ClockedNativeSource(
                contract,
                raw,
                clocked_native_source_digest(raw, contract=contract, clock=clock),
            )
        )
    base = MarketDatasetBuilder(
        MarketBuildConfig("1h", (FeatureSpec("base_return", FeatureKind.LOG_RETURN),))
    ).build(
        InMemoryMarketDataSource(raws),
        contracts,
        execution_economics=ExecutionEconomicsProfile(
            "generated_carry", borrow_rate=0.0876
        ),
    )
    base = _rebind(base, funding_price_rate=price_rates)
    group = ClockedNativeFeatureGroup(
        clock, tuple(sources), (FeatureSpec("body", FeatureKind.BODY_RETURN),)
    )
    return (
        append_clocked_native_feature_groups(base, groups=(group,)),
        group,
        decision,
        stop,
    )


def _cost(**changes):
    return replace(
        ExecutionCostConfig.zero(),
        **(
            {
                "allow_short": True,
                "borrow_rate_multiplier": 1.0,
                "processing_bar_volume_capacity": False,
                "partial_fill_carry": False,
                "fee_rate": 0.001,
            }
            | changes
        ),
    )


@pytest.mark.parametrize("kind", ["3d", "week", "month"])
def test_linear_carry_is_native_accounting_on_every_fine_interval(tmp_path, kind):
    dataset, group, decision, stop = _fixture(kind)
    write_market_dataset_files(tmp_path, dataset)
    restored = load_market_dataset_artifact(tmp_path)
    assert restored.identity_verified and restored.dataset_id == dataset.dataset_id
    seen = []

    class Policy:
        def decide_portfolio(self, snapshot):
            seen.append(snapshot.index)
            return PortfolioTargetProposal((0.25, -0.25))

    result = run_clocked_native_portfolio_replay(
        restored,
        Policy(),
        decision_group=group,
        start_index=decision - 3,
        stop_index=stop,
        initial_capital=1000.0,
        execution_cost=_cost(),
        execution_mode="linear_carry",
        capture_ledger_evidence=True,
    )
    assert result.policy_decision_indices == tuple(seen) == (decision,)
    ledger = result.replay.ledger_evidence
    assert ledger is not None
    rows = ledger.intervals.intervals
    assert len(rows) == 9
    # Positive rate charges the long and credits the short at the explicit
    # event price, not the bar mark. The second event occurs during HOLD.
    funding = {decision + 1: 2.5, decision + 2: 3.5, stop: 0.0}
    cash = 1000.0
    previous = 1000.0
    for row in rows:
        idx = row.next_index
        held = decision + 1 <= idx < stop
        if idx == decision + 1 or idx == stop:
            cash -= 0.5
        borrow = 0.0025 if held else 0.0
        cash += funding.get(idx, 0.0) - borrow
        assert tuple(Fraction(q) for q in row.exact_quantities_after) == (
            (Fraction(5, 4), Fraction(-5, 16)) if held else (0, 0)
        )
        assert row.interval_funding == pytest.approx(funding.get(idx, 0.0))
        assert row.interval_borrow_cost == pytest.approx(borrow)
        assert row.cash_after == pytest.approx(cash)
        assert row.portfolio_value_after == pytest.approx(cash)
        assert row.interval_net_return == pytest.approx(cash / previous - 1, abs=1e-12)
        if idx in funding:
            (boundary,) = row.funding_events
            assert boundary.processing_index == idx
            assert boundary.timestamp_ns == int(
                restored.timestamps[idx].astype("datetime64[ns]").astype(np.int64)
            )
            assert boundary.funding_amount == pytest.approx(funding[idx])
            assert boundary.funding_price_rates == tuple(
                restored.funding_price_rate[idx]
            )
            assert boundary.contract_multipliers == (2.0, 4.0)
            assert boundary.signed_quantities == (
                (1.25, -0.3125) if held else (0.0, 0.0)
            )
        previous = cash
    fills = [e for row in rows for e in row.order_events if e.event_type == "filled"]
    assert [(e.processing_index, e.symbol_index) for e in fills] == [
        (decision + 1, 0),
        (decision + 1, 1),
        (stop, 0),
        (stop, 1),
    ]
    assert result.replay.book.exact_quantities == (0, 0)
    assert result.replay.book.total_cost == pytest.approx(1.0)
    assert result.replay.book.funding_pnl == pytest.approx(6.0)
    assert result.replay.book.borrow_cost == pytest.approx(0.0125)
    assert result.replay.book.portfolio_value == pytest.approx(1004.9875)
    assert result.replay.returns.periods_per_year == 8760
    assert result.replay.returns.values == pytest.approx(
        [row.interval_net_return for row in rows], abs=1e-12
    )
    assert ledger.intervals.active_order_remainders == ()


@pytest.mark.parametrize(
    "bad",
    [
        "missing_price_rate",
        "multiple_events",
        "due_mismatch",
        "off_event_rate",
        "price_sign",
        "price_zero",
        "price_overflow",
        "contract_multiplier",
        "volume_unit",
    ],
)
def test_linear_carry_refuses_ambiguous_economics_before_policy(bad):
    ds, group, decision, stop = _fixture()
    if bad == "missing_price_rate":
        ds = _rebind(ds, funding_price_rate=None)
    elif bad in ("multiple_events", "due_mismatch"):
        name = "funding_event_count" if bad == "multiple_events" else "funding_due"
        array = ds.resolved_array(name).copy()
        array[decision + 1, 0] = 2 if bad == "multiple_events" else False
        ds = _rebind(ds, **{name: array})
    elif bad == "off_event_rate":
        rate = ds.funding_rate.copy()
        rate[decision, 0] = 0.01
        ds = _rebind(ds, funding_rate=rate)
    elif bad in ("price_sign", "price_zero", "price_overflow"):
        price_rate = ds.resolved_array("funding_price_rate").copy()
        price_rate[decision + 1, 0] = {
            "price_sign": -1.0,
            "price_zero": 0.0,
            "price_overflow": 1e308,
        }[bad]
        ds = _rebind(ds, funding_price_rate=price_rate)
    elif bad == "contract_multiplier":
        ds = _rebind(ds, contract_multipliers=np.array([3.0, 4.0]))
    else:
        ds = _rebind(ds, volume_units=(VolumeUnit.BASE_ASSET, VolumeUnit.CONTRACTS))

    class Never:
        def decide_portfolio(self, snapshot):
            pytest.fail("ambiguous linear carry reached actor")

    with pytest.raises(ValueError, match="linear carry"):
        run_clocked_native_portfolio_replay(
            ds,
            Never(),
            decision_group=group,
            start_index=decision - 3,
            stop_index=stop,
            initial_capital=1000.0,
            execution_cost=_cost(),
            execution_mode="linear_carry",
        )


@pytest.mark.parametrize("mode", ["wrong", True, object()])
def test_execution_mode_requires_explicit_supported_literal(mode):
    ds, group, decision, stop = _fixture()

    class Never:
        def decide_portfolio(self, snapshot):
            pytest.fail("invalid mode reached actor")

    with pytest.raises(ValueError, match="execution_mode"):
        run_clocked_native_portfolio_replay(
            ds,
            Never(),
            decision_group=group,
            start_index=decision - 3,
            stop_index=stop,
            initial_capital=1000.0,
            execution_cost=_cost(),
            execution_mode=mode,
        )


def test_linear_carry_respects_borrow_admission_without_retry_or_new_actor():
    ds, group, decision, stop = _fixture()
    available = ds.resolved_array("borrow_available").copy()
    available[:, 1] = False
    ds = _rebind(ds, borrow_available=available)
    seen = []

    class Policy:
        def decide_portfolio(self, snapshot):
            seen.append(snapshot.index)
            return PortfolioTargetProposal((0.0, -0.25))

    result = run_clocked_native_portfolio_replay(
        ds,
        Policy(),
        decision_group=group,
        start_index=decision - 3,
        stop_index=stop,
        initial_capital=1000.0,
        execution_cost=_cost(),
        execution_mode="linear_carry",
        capture_ledger_evidence=True,
    )
    assert tuple(seen) == result.policy_decision_indices == (decision,)
    assert result.replay.book.fill_count == 0
    assert result.replay.book.exact_quantities == (0, 0)
    assert result.replay.book.borrow_cost == result.replay.book.funding_pnl == 0.0
    assert result.replay.book.portfolio_value == 1000.0
    assert result.replay.ledger_evidence is not None
    assert all(
        row.exact_quantities_after == ("0", "0")
        for row in result.replay.ledger_evidence.intervals.intervals
    )


def test_zero_rate_event_and_later_event_change_do_not_rewrite_earlier_path():
    ds, group, decision, stop = _fixture()
    rates = ds.funding_rate.copy()
    products = ds.resolved_array("funding_price_rate").copy()
    rates[decision + 2, :] = 0.0
    products[decision + 2, :] = 0.0
    changed = _rebind(ds, funding_rate=rates, funding_price_rate=products)

    def run(data):
        seen = []

        class Policy:
            def decide_portfolio(self, snapshot):
                seen.append(snapshot)
                return PortfolioTargetProposal((0.25, -0.25))

        return seen, run_clocked_native_portfolio_replay(
            data,
            Policy(),
            decision_group=group,
            start_index=decision - 3,
            stop_index=stop,
            initial_capital=1000.0,
            execution_cost=_cost(),
            execution_mode="linear_carry",
            capture_ledger_evidence=True,
        )

    seen, first = run(ds)
    second_seen, second = run(changed)
    assert seen == second_seen
    assert first.slots == second.slots
    assert first.replay.returns.values[:4] == second.replay.returns.values[:4]
    assert (
        first.replay.ledger_evidence is not None
        and second.replay.ledger_evidence is not None
    )
    zero_event = second.replay.ledger_evidence.intervals.intervals[4]
    assert zero_event.next_index == decision + 2
    assert zero_event.interval_funding == 0.0
    (boundary,) = zero_event.funding_events
    assert boundary.funding_due == (True, True)
    assert boundary.funding_amount == 0.0
    assert (
        first.replay.book.portfolio_value - second.replay.book.portfolio_value
        == pytest.approx(3.5)
    )


@pytest.mark.parametrize(
    "change",
    [
        {"order_latency_bars": 1},
        {"partial_fill_carry": True},
        {"processing_bar_volume_capacity": True},
        {"order_type": "limit"},
        {"max_leverage": 2.0},
        {"margin_mode": "isolated"},
        {"collateral_haircut": 0.9},
        {"slippage_std": 0.1},
    ],
)
def test_linear_carry_keeps_pending_intrabar_and_leverage_out_of_scope(change):
    ds, group, decision, stop = _fixture()

    class Never:
        def decide_portfolio(self, snapshot):
            pytest.fail("unsupported order semantics reached actor")

    with pytest.raises(ValueError, match="scheduled replay requires"):
        run_clocked_native_portfolio_replay(
            ds,
            Never(),
            decision_group=group,
            start_index=decision - 3,
            stop_index=stop,
            initial_capital=1000.0,
            execution_cost=_cost(**change),
            execution_mode="linear_carry",
        )


def test_default_spot_mode_does_not_admit_linear_carry_data():
    ds, group, decision, stop = _fixture()

    class Never:
        def decide_portfolio(self, snapshot):
            pytest.fail("linear data reached default spot actor")

    with pytest.raises(ValueError, match="spot-style"):
        run_clocked_native_portfolio_replay(
            ds,
            Never(),
            decision_group=group,
            start_index=decision - 3,
            stop_index=stop,
            initial_capital=1000.0,
            execution_cost=_cost(allow_short=False),
        )


@pytest.mark.parametrize("field, value", [("dividend", 1.0), ("split_factor", 2.0)])
def test_linear_carry_refuses_corporate_actions(field, value):
    ds, group, decision, stop = _fixture()
    modified = ds.resolved_array(field).copy()
    modified[decision + 1, 0] = value
    ds = _rebind(ds, **{field: modified})

    class Never:
        def decide_portfolio(self, snapshot):
            pytest.fail("corporate action reached actor")

    with pytest.raises(ValueError, match="corporate actions"):
        run_clocked_native_portfolio_replay(
            ds,
            Never(),
            decision_group=group,
            start_index=decision - 3,
            stop_index=stop,
            initial_capital=1000.0,
            execution_cost=_cost(),
            execution_mode="linear_carry",
        )


def test_off_schedule_hard_risk_can_close_short_after_funding_loss():
    from trade_rl.risk import PreTradeRisk, PreTradeRiskConfig

    ds, group, decision, stop = _fixture()
    rates = ds.funding_rate.copy()
    price_rates = ds.resolved_array("funding_price_rate").copy()
    rates[decision + 1, 1] = -0.02
    price_rates[decision + 1, 1] = -4.2
    ds = _rebind(ds, funding_rate=rates, funding_price_rate=price_rates)
    seen = []

    class Policy:
        def decide_portfolio(self, snapshot):
            seen.append(snapshot.index)
            return PortfolioTargetProposal((0.0, -0.25))

    result = run_clocked_native_portfolio_replay(
        ds,
        Policy(),
        decision_group=group,
        start_index=decision - 3,
        stop_index=stop,
        initial_capital=1000.0,
        execution_cost=_cost(),
        execution_mode="linear_carry",
        risk=PreTradeRisk(
            PreTradeRiskConfig(
                max_gross=1.0,
                max_abs_weight=1.0,
                max_turnover=None,
                drawdown_start=0.0,
                drawdown_stop=0.001,
            )
        ),
        capture_ledger_evidence=True,
    )
    assert tuple(seen) == result.policy_decision_indices == (decision,)
    assert result.replay.ledger_evidence is not None
    rows = result.replay.ledger_evidence.intervals.intervals
    fills = [
        event
        for row in rows
        for event in row.order_events
        if event.event_type == "filled"
    ]
    assert [(e.processing_index, e.symbol_index) for e in fills] == [
        (decision + 1, 1),
        (decision + 2, 1),
    ]
    assert result.replay.book.exact_quantities == (0, 0)
    assert result.replay.book.funding_pnl == pytest.approx(-5.25)
    assert result.replay.book.borrow_cost == pytest.approx(0.0025)
    assert result.replay.book.total_cost == pytest.approx(0.5)
    assert result.replay.book.portfolio_value == pytest.approx(994.2475)


def test_borrow_uses_native_post_open_book_not_close_or_funding_mark():
    ds, group, decision, stop = _fixture(borrow_open_price=300.0)

    class Policy:
        def decide_portfolio(self, snapshot):
            return PortfolioTargetProposal((0.25, -0.25))

    result = run_clocked_native_portfolio_replay(
        ds,
        Policy(),
        decision_group=group,
        start_index=decision - 3,
        stop_index=stop,
        initial_capital=1000.0,
        execution_cost=_cost(),
        execution_mode="linear_carry",
        capture_ledger_evidence=True,
    )
    assert result.replay.ledger_evidence is not None
    rows = result.replay.ledger_evidence.intervals.intervals
    held = next(row for row in rows if row.next_index == decision + 3)
    # 5/16 short contracts, multiplier 4, actual processing open 300,
    # 0.0876 annual borrow rate and one hour. The close is 200.
    assert held.exact_quantities_before[1] == held.exact_quantities_after[1] == "-5/16"
    assert held.interval_borrow_cost == pytest.approx(0.00375)
    assert held.interval_funding == 0.0
    assert result.replay.book.borrow_cost == pytest.approx(0.01375)
    assert result.replay.book.portfolio_value == pytest.approx(1004.98625)
