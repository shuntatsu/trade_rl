"""Source-bound compatibility baseline for the native-clock expansion (#1357).

The nine existing clocks, fixed bar-index Ichimoku arithmetic, exact as-of
publication boundary, and common ledger are exercised on artificial prices.
This is not 3d/weekly/monthly implementation or an economic result.
"""

from dataclasses import replace
from fractions import Fraction

import numpy as np
import pytest

from trade_rl.data.build.builder import MarketDatasetBuilder
from trade_rl.data.contracts import (
    FeatureKind,
    FeatureSpec,
    InstrumentContract,
    MarketBuildConfig,
    timeframe_hours,
)
from trade_rl.data.features.multitimeframe import align_native_feature
from trade_rl.data.market import MarketDataset
from trade_rl.data.source import InMemoryMarketDataSource, RawMarketSeries
from trade_rl.evaluation.metrics import evaluate_performance
from trade_rl.evaluation.replay import run_shared_portfolio_replay
from trade_rl.evaluation.series import ReturnKind
from trade_rl.risk import PreTradeRisk, PreTradeRiskConfig
from trade_rl.simulation import ExecutionCostConfig
from trade_rl.strategies.portfolio import (
    PortfolioDecisionSnapshot,
    PortfolioTargetProposal,
)

# Literal compatibility roster; do not derive the oracle from timeframe_hours.
CLOCKS = (
    ("15m", 15, 35040),
    ("30m", 30, 17520),
    ("1h", 60, 8760),
    ("2h", 120, 4380),
    ("4h", 240, 2190),
    ("6h", 360, 1460),
    ("8h", 480, 1095),
    ("12h", 720, 730),
    ("1d", 1440, 365),
)
SYMBOL = "SYNTHETIC"
EPOCH = np.datetime64("2026-01-05T00:00:00", "ns")
NS_HOUR = 3_600_000_000_000


def _raw(
    minutes: int,
    indexes: np.ndarray,
    *,
    delay: bool = False,
    mutate: bool = False,
    flat: bool = False,
) -> RawMarketSeries:
    times = EPOCH + indexes * np.timedelta64(minutes, "m")
    close = (
        np.full(len(indexes), 100.0)
        if flat
        else 100.0 + indexes * 0.3 + (indexes % 7) * 0.7
    )
    if mutate:
        close[indexes >= 70] *= 2
    available_at = times.copy()
    if delay:
        available_at[indexes == 51] += np.timedelta64(2 * minutes, "m")
    counts = (
        (indexes == 2).astype(np.int32) if flat else np.zeros(len(indexes), np.int32)
    )
    return RawMarketSeries(
        timestamps=times,
        available_at=available_at,
        open=close,
        high=close if flat else close + 2,
        low=close if flat else close - 1,
        close=close,
        volume=np.full(len(indexes), 1_000_000.0),
        funding_rate=0.01 * counts,
        funding_available=counts > 0,
        funding_event_count=counts,
        tradable=np.ones(len(indexes), dtype=np.bool_),
    )


def _cloud(raw: RawMarketSeries, j: int) -> float:
    # Independent literal 9/26/52 unshifted computation, not the visible
    # chart cloud displaced forward by 26 native bars.
    def midpoint(n: int) -> float:
        high = max(float(raw.high[k]) for k in range(j - n + 1, j + 1))
        low = min(float(raw.low[k]) for k in range(j - n + 1, j + 1))
        return (high + low) / 2

    return float(
        (raw.close[j] - ((midpoint(9) + midpoint(26)) / 2 + midpoint(52)) / 2)
        / raw.close[j]
    )


def _cloud_spec(*, staleness: float) -> FeatureSpec:
    return FeatureSpec(
        name="cloud",
        kind=FeatureKind.ICHIMOKU_CLOUD_POSITION,
        max_staleness_hours=staleness,
    )


def _build(raw: RawMarketSeries, timeframe: str, spec: FeatureSpec) -> MarketDataset:
    return MarketDatasetBuilder(
        MarketBuildConfig(base_timeframe=timeframe, features=(spec,))
    ).build(InMemoryMarketDataSource({SYMBOL: raw}), (InstrumentContract(SYMBOL),))


@pytest.mark.parametrize(("timeframe", "minutes", "annual"), CLOCKS)
def test_existing_native_cloud_values_and_annualization(
    timeframe: str, minutes: int, annual: int
) -> None:
    raw = _raw(minutes, np.arange(120))
    dataset = _build(raw, timeframe, _cloud_spec(staleness=2 * minutes / 60))
    assert timeframe_hours(timeframe) == minutes / 60
    assert Fraction(365 * 24 * 60, minutes) == annual
    assert dataset.periods_per_year == annual
    assert dataset.nominal_bar_hours == minutes / 60
    assert dataset.calendar_kind.value == "continuous_24_7"
    np.testing.assert_array_equal(dataset.timestamps, raw.timestamps)
    np.testing.assert_array_equal(
        dataset.feature_available[:, 0, 0], np.arange(120) >= 51
    )
    np.testing.assert_allclose(
        dataset.features[51:, 0, 0],
        [_cloud(raw, j) for j in range(51, 120)],
        rtol=1e-6,
        atol=1e-8,
    )


@pytest.mark.parametrize(("timeframe", "minutes", "annual"), CLOCKS)
def test_native_publication_and_prefix_oracle_on_quarter_hour_base(
    timeframe: str, minutes: int, annual: int
) -> None:
    del annual
    raw = _raw(minutes, np.arange(80), delay=True)
    assert raw.available_at is not None
    spec = _cloud_spec(staleness=1.5 * minutes / 60)
    # A finer base clock includes timestamps between native observations.
    base = EPOCH + np.arange(81 * minutes // 15 + 1) * np.timedelta64(15, "m")
    active = np.ones(len(base), dtype=np.bool_)
    actual = align_native_feature(
        spec, raw, InstrumentContract(SYMBOL), base, active, timeframe=timeframe
    )
    values, available, ages, stale = actual
    expected_values = np.zeros(len(base))
    expected_available = np.zeros(len(base), dtype=np.bool_)
    expected_ages = np.full(len(base), spec.max_staleness_hours)
    expected_stale = np.ones(len(base))
    for i, t in enumerate(base):
        # Literal chronological selection, checking ALL 52 dependencies;
        # cannot use an event simply because its own close is known.
        admitted = [
            j
            for j in range(51, len(raw.timestamps))
            if raw.timestamps[j] <= t
            and all(raw.available_at[k] <= t for k in range(j - 51, j + 1))
        ]
        if admitted:
            j = admitted[-1]
            age = (
                int((t - raw.timestamps[j]).astype("timedelta64[ns]").astype(np.int64))
                / NS_HOUR
            )
            expected_ages[i] = age
            expected_stale[i] = min(age / spec.max_staleness_hours, 1.0)
            if age <= spec.max_staleness_hours:
                expected_values[i] = _cloud(raw, j)
                expected_available[i] = True
    np.testing.assert_array_equal(available, expected_available)
    np.testing.assert_allclose(values, expected_values, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(ages, expected_ages)
    np.testing.assert_allclose(stale, expected_stale)
    assert not available[51 * minutes // 15]
    assert not available[52 * minutes // 15]
    assert available[53 * minutes // 15]
    assert not available[-1]

    # Tail mutation and physical truncation must not affect a past as-of view.
    prefix = base < raw.timestamps[70]
    mutated = _raw(minutes, np.arange(80), delay=True, mutate=True)
    truncated = _raw(minutes, np.arange(70), delay=True)
    for candidate in (mutated, truncated):
        asof = align_native_feature(
            spec,
            candidate,
            InstrumentContract(SYMBOL),
            base[prefix],
            active[prefix],
            timeframe=timeframe,
        )
        for original, earlier in zip(actual, asof, strict=True):
            np.testing.assert_array_equal(original[prefix], earlier)


@pytest.mark.parametrize(("timeframe", "minutes", "annual"), CLOCKS)
def test_incomplete_native_clock_refuses_without_interpolation(
    timeframe: str, minutes: int, annual: int
) -> None:
    del annual
    raw = _raw(minutes, np.delete(np.arange(80), 25))
    with pytest.raises(ValueError, match="complete and exactly regular"):
        align_native_feature(
            _cloud_spec(staleness=24),
            raw,
            InstrumentContract(SYMBOL),
            raw.timestamps,
            np.ones(len(raw.timestamps), dtype=np.bool_),
            timeframe=timeframe,
        )


class _EnterThenHold:
    def __init__(self) -> None:
        self.indexes: list[int] = []

    def decide_portfolio(
        self, snapshot: PortfolioDecisionSnapshot
    ) -> PortfolioTargetProposal:
        self.indexes.append(snapshot.index)
        if snapshot.index == 0:
            return PortfolioTargetProposal((0.5,))
        return PortfolioTargetProposal(
            tuple(snapshot.account.weights), hold_current_quantity_symbols=(0,)
        )


@pytest.mark.parametrize(("timeframe", "minutes", "annual"), CLOCKS)
@pytest.mark.parametrize("cap", (0.5, 0.6))
def test_native_replay_fee_funding_and_terminal_settlement(
    timeframe: str, minutes: int, annual: int, cap: float
) -> None:
    raw = _raw(minutes, np.arange(5), flat=True)
    dataset = _build(
        raw, timeframe, FeatureSpec(name="ret", kind=FeatureKind.LOG_RETURN)
    )
    policy = _EnterThenHold()
    result = run_shared_portfolio_replay(
        dataset,
        policy,
        start_index=0,
        stop_index=4,
        initial_capital=1000.0,
        execution_cost=replace(ExecutionCostConfig.zero(), fee_rate=0.001),
        risk=PreTradeRisk(
            PreTradeRiskConfig(
                max_gross=cap,
                max_abs_weight=cap,
                max_turnover=None,
                drawdown_start=1.0,
                drawdown_stop=1.0,
            )
        ),
        settle_terminal_position=True,
        capture_ledger_evidence=True,
    )
    # Derive the common ledger independently with rational arithmetic.
    # A roomy cap permits exact quantity HOLD. At a binding 50% cap, fees
    # and funding reduce NAV and require de-risking despite the HOLD hint.
    # Order fills precede funding in the processing interval.
    price = Fraction(100)
    fee = Fraction(1, 1000)
    funding_rate = Fraction(1, 100)
    cap_fraction = Fraction(str(cap))
    equity = [Fraction(1000)]
    quantity = Fraction(5)
    quantities = [Fraction(0), quantity]
    fees = [quantity * price * fee]
    equity.append(equity[-1] - fees[-1])
    funding = Fraction(0)
    for processing_index in (2, 3):
        target = min(quantity, equity[-1] * cap_fraction / price)
        fees.append((quantity - target) * price * fee)
        quantity = target
        quantities.append(quantity)
        cashflow = (
            -quantity * price * funding_rate if processing_index == 2 else Fraction(0)
        )
        funding += cashflow
        equity.append(equity[-1] - fees[-1] + cashflow)
    fees.append(quantity * price * fee)
    equity.append(equity[-1] - fees[-1])
    quantities.append(Fraction(0))
    assert sum(fees) == 1
    assert funding == (Fraction(-1999, 400) if cap == 0.5 else -5)
    assert result.book.total_cost == pytest.approx(float(sum(fees)))
    assert result.book.funding_pnl == pytest.approx(float(funding))
    assert result.book.cash == pytest.approx(float(equity[-1]))
    assert result.book.portfolio_value == pytest.approx(float(equity[-1]))
    assert tuple(result.book.exact_quantities) == (0,)
    assert result.returns.kind is ReturnKind.BASE_BAR
    assert result.returns.periods_per_year == annual
    assert result.returns.annualization_periods_per_year == pytest.approx(annual)
    assert policy.indexes == [0, 1, 2]
    assert result.ledger_evidence is not None
    intervals = result.ledger_evidence.to_mapping()["intervals"]
    assert len(intervals) == 4
    for i, interval in enumerate(intervals):
        assert interval["interval_cost"] == pytest.approx(float(fees[i]))
        assert float(Fraction(interval["exact_quantities_before"][0])) == pytest.approx(
            float(quantities[i])
        )
        assert float(Fraction(interval["exact_quantities_after"][0])) == pytest.approx(
            float(quantities[i + 1])
        )
    assert len(intervals[1]["funding_events"]) == 1
    event = intervals[1]["funding_events"][0]
    assert event["timestamp_ns"] == int(raw.timestamps[2].astype(np.int64))
    assert event["signed_quantities"] == pytest.approx((float(quantities[2]),))
    assert event["funding_amount"] == pytest.approx(float(funding))
    for decision in result.decisions[1:]:
        assert decision.requested_hold_symbols == (0,)
        assert decision.effective_hold_symbols == (() if cap == 0.5 else (0,))
    oracle_returns = np.asarray(
        [float(b / a - 1) for a, b in zip(equity[:-1], equity[1:], strict=True)]
    )
    np.testing.assert_allclose(result.returns.values, oracle_returns, atol=1e-12)
    metrics = evaluate_performance(result.returns)
    assert metrics.total_return == pytest.approx(float(equity[-1] / equity[0] - 1))
    assert metrics.sharpe == pytest.approx(
        float(oracle_returns.mean() / oracle_returns.std() * np.sqrt(annual))
    )


@pytest.mark.parametrize("name", ("3d", "1w", "1M"))
def test_longer_timeframes_are_not_silently_admitted(name: str) -> None:
    with pytest.raises(ValueError, match="unsupported timeframe"):
        timeframe_hours(name)
    with pytest.raises(ValueError, match="unsupported timeframe"):
        MarketBuildConfig(
            base_timeframe=name,
            features=(FeatureSpec(name="ret", kind=FeatureKind.LOG_RETURN),),
        )
    with pytest.raises(ValueError, match="unsupported timeframe"):
        FeatureSpec(name="ret", kind=FeatureKind.LOG_RETURN, timeframe=name)


@pytest.mark.parametrize("days,rounded", ((3, 122), (7, 52)))
def test_continuous_long_clock_cannot_be_rounded_to_integer_annualization(
    days: int, rounded: int
) -> None:
    # Construct a real current-schema Dataset, then challenge its calendar
    # with an exact long interval. Rounding frequency must not make it valid.
    dataset = _build(
        _raw(1440, np.arange(5)),
        "1d",
        FeatureSpec(name="ret", kind=FeatureKind.LOG_RETURN),
    )
    assert Fraction(365, days).denominator != 1
    timestamps = EPOCH + np.arange(5) * np.timedelta64(days, "D")
    with pytest.raises(
        ValueError, match="timestamp cadence does not resolve to annual periods"
    ):
        replace(
            dataset,
            timestamps=timestamps,
            periods_per_year=rounded,
            nominal_bar_hours=float(days * 24),
        )
