"""Run the preregistered development real-data study on one verified Dataset.

Per arm and scenario this makes exactly ONE continuous
``run_shared_portfolio_replay`` over the evaluation window (no resets at refit
boundaries), settles the book flat at the end, captures the ledger, reconciles it
against raw fills, and reduces it to equity-path metrics. Nothing here can emit
a ``WINNER``; the strongest label is ``DEVELOPMENT_PROMISING``.
"""

from __future__ import annotations

import math
import subprocess
from collections import Counter
from collections.abc import Callable, Mapping
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
from fractions import Fraction
from typing import Any, cast

import numpy as np

from trade_rl.data.market import MarketDataset
from trade_rl.evaluation.dev_realdata_study.metrics import (
    bootstrap_excess,
    daily_equity,
    daily_returns,
    moving_block_indices,
    performance_summary,
    select_development_label,
)
from trade_rl.evaluation.dev_realdata_study.protocol import (
    ARM_ALLOCATOR_24H,
    ARM_ALLOCATOR_24H_LONG_ONLY,
    ARM_ALLOCATOR_72H,
    ARM_BUY_AND_HOLD,
    ARM_CASH,
    SCENARIO_BASE,
    SCENARIO_COST_X2,
    SCENARIO_LATENCY,
    DevStudyProtocol,
    protocol_digest,
)
from trade_rl.evaluation.replay import (
    SharedCashPortfolioReplayResult,
    run_shared_portfolio_replay,
)
from trade_rl.simulation import ExecutionCostConfig
from trade_rl.strategies.allocation.buy_and_hold import BuyAndHoldPortfolioPolicy
from trade_rl.strategies.allocation.cost_aware import CostAwareAllocationPolicy
from trade_rl.strategies.allocation.costs import one_way_cost_rates
from trade_rl.strategies.forecasts.prequential import (
    PrequentialForecastSource,
    PrequentialSchedule,
    RidgeForecastFitter,
)
from trade_rl.strategies.portfolio import (
    PortfolioDecisionSnapshot,
    PortfolioTargetProposal,
)

_NS_PER_HOUR = 3_600 * 1_000_000_000
_NS_PER_DAY = 24 * _NS_PER_HOUR

Progress = Callable[[str], None]


def _ns(text: str) -> int:
    return int(np.datetime64(text, "ns").astype(np.int64))


def _timestamps_ns(dataset: MarketDataset) -> np.ndarray:
    return dataset.timestamps.astype("datetime64[ns]").astype(np.int64)


def _row_of(dataset: MarketDataset, text: str) -> int:
    target = _ns(text)
    stamps = _timestamps_ns(dataset)
    row = int(np.searchsorted(stamps, target))
    if row >= stamps.size or int(stamps[row]) != target:
        raise ValueError(f"{text} is not a dataset timestamp")
    return row


class ZeroWeightPolicy:
    """CASH control: always the all-zero target; the shared session owns the book."""

    def __init__(self, symbols: tuple[str, ...]) -> None:
        self.symbols = symbols

    def decide_portfolio(
        self, snapshot: PortfolioDecisionSnapshot
    ) -> PortfolioTargetProposal:
        return PortfolioTargetProposal(target_weights=(0.0,) * len(snapshot.symbols))


class PrequentialSimpleReturnSource:
    """Adapt log-return forecast packets into per-decision-interval simple returns.

    ``mu_simple = expm1(mu_log + 0.5 * residual_variance) / (horizon_hours / interval)``.
    A symbol without a usable packet (unavailable features, e.g. an exchange gap)
    contributes ``0.0``. Only the packet for ``snapshot.timestamp_ns`` is read.
    """

    def __init__(
        self, source: PrequentialForecastSource, *, decision_interval_hours: int
    ) -> None:
        horizon = source.horizon_hours
        if horizon % decision_interval_hours != 0:
            raise ValueError("horizon must be a multiple of the decision interval")
        self._source = source
        self._divisor = float(horizon // decision_interval_hours)
        self.calls = 0
        self.null_packets = 0
        self.history: list[tuple[float, ...]] = []

    def expected_simple_returns(
        self, snapshot: PortfolioDecisionSnapshot
    ) -> tuple[float, ...]:
        when = np.datetime64(snapshot.timestamp_ns, "ns")
        values: list[float] = []
        for symbol_index in range(len(snapshot.symbols)):
            packet = self._source.packet(symbol_index, when)
            if packet is None:
                self.null_packets += 1
                values.append(0.0)
                continue
            simple = math.expm1(packet.point_forecast + 0.5 * packet.residual_variance)
            values.append(simple / self._divisor)
        self.calls += 1
        result = tuple(values)
        self.history.append(result)
        return result


def feature_indices(
    dataset: MarketDataset, protocol: DevStudyProtocol
) -> tuple[int, ...]:
    indices = tuple(
        index
        for index, name in enumerate(dataset.feature_names)
        if name.startswith(protocol.feature_prefix)
        and protocol.feature_exclude_substring not in name
    )
    if not indices:
        raise ValueError("no features match the protocol roster rule")
    return indices


def build_forecast_sources(
    dataset: MarketDataset, protocol: DevStudyProtocol
) -> dict[int, PrequentialForecastSource]:
    """One prequential source per horizon; vintages are fitted lazily and cached."""

    schedule = PrequentialSchedule(
        first_fit_time=np.datetime64(protocol.evaluation_start, "ns"),
        refit_interval_days=protocol.refit_interval_days,
        activation_delay_hours=protocol.activation_delay_hours,
    )
    indices = feature_indices(dataset, protocol)
    return {
        horizon: PrequentialForecastSource(
            dataset,
            schedule=schedule,
            fitter=RidgeForecastFitter(alpha=protocol.ridge_alpha),
            feature_indices=indices,
            horizon_hours=horizon,
        )
        for horizon in protocol.horizons_hours
    }


def warmup_marks(
    dataset: MarketDataset,
    protocol: DevStudyProtocol,
    *,
    decision_interval_hours: int | None = None,
    decision_anchor_hour_utc: int | None = None,
) -> tuple[tuple[int, ...], tuple[tuple[float, ...], ...]]:
    """Scheduled-instant marks strictly before the evaluation start (fit-period data).

    The schedule defaults to the protocol's daily 00:00 UTC cadence; a trial with its
    own cadence passes the interval and anchor so the marks lie on its schedule.
    """

    stamps = _timestamps_ns(dataset)
    start = _ns(protocol.covariance_warmup_start)
    end = _ns(protocol.evaluation_start)
    interval_hours = (
        protocol.decision_interval_hours
        if decision_interval_hours is None
        else decision_interval_hours
    )
    anchor_hour = (
        protocol.decision_anchor_hour_utc
        if decision_anchor_hour_utc is None
        else decision_anchor_hour_utc
    )
    interval = interval_hours * _NS_PER_HOUR
    anchor = anchor_hour * _NS_PER_HOUR
    marks = dataset.resolved_array("mark_price")
    chosen = [
        row
        for row in range(stamps.size)
        if start <= int(stamps[row]) < end and int(stamps[row]) % interval == anchor
    ]
    return (
        tuple(int(stamps[row]) for row in chosen),
        tuple(tuple(float(value) for value in marks[row]) for row in chosen),
    )


def scenario_cost(protocol: DevStudyProtocol, scenario: str) -> ExecutionCostConfig:
    multiplier = (
        protocol.cost_stress_multiplier if scenario == SCENARIO_COST_X2 else 1.0
    )
    return ExecutionCostConfig(
        fee_rate=protocol.fee_rate * multiplier,
        spread_rate=protocol.spread_rate * multiplier,
        impact_rate=protocol.impact_rate,
        max_participation_rate=protocol.max_participation_rate,
        order_latency_bars=(
            protocol.latency_stress_bars if scenario == SCENARIO_LATENCY else 0
        ),
    )


@dataclass
class BuiltArm:
    """A fresh policy for one replay plus its forecast diagnostics (if any).

    ``adapter`` is any object exposing ``calls``, ``null_packets`` and ``history``
    (per-decision expected-return vectors); it feeds the shared forecast summary.
    """

    policy: Any
    adapter: Any | None


ArmBuilder = Callable[[str, MarketDataset, DevStudyProtocol, Any], BuiltArm]


def build_allocator(
    dataset: MarketDataset,
    protocol: DevStudyProtocol,
    forecast_source: Any,
    *,
    long_only: bool,
    decision_interval_hours: int | None = None,
    decision_anchor_hour_utc: int | None = None,
) -> CostAwareAllocationPolicy:
    """The protocol's cost-aware allocator around any expected-return source."""

    interval_hours = (
        protocol.decision_interval_hours
        if decision_interval_hours is None
        else decision_interval_hours
    )
    anchor_hour = (
        protocol.decision_anchor_hour_utc
        if decision_anchor_hour_utc is None
        else decision_anchor_hour_utc
    )
    symbols = dataset.symbols
    stamps, prices = warmup_marks(
        dataset,
        protocol,
        decision_interval_hours=interval_hours,
        decision_anchor_hour_utc=anchor_hour,
    )
    base_cost = scenario_cost(protocol, SCENARIO_BASE)
    return CostAwareAllocationPolicy(
        symbols=symbols,
        forecast_source=forecast_source,
        cost_rates=one_way_cost_rates(base_cost, n_symbols=len(symbols)),
        gamma=protocol.gamma,
        ewma_halflife_days=protocol.ewma_halflife_days,
        shrinkage=protocol.shrinkage,
        w_max=protocol.w_max,
        gross_max=protocol.gross_max,
        long_only=long_only,
        min_trade_fraction=protocol.min_trade_fraction,
        decision_interval_hours=interval_hours,
        decision_anchor_hour_utc=anchor_hour,
        min_history_periods=protocol.min_history_periods,
        warmup_timestamps_ns=stamps,
        warmup_mark_prices=prices,
    )


def build_arm(
    arm: str,
    dataset: MarketDataset,
    protocol: DevStudyProtocol,
    sources: Mapping[int, PrequentialForecastSource],
) -> BuiltArm:
    """Fresh policy instance for one replay (policies hold replay state)."""

    symbols = dataset.symbols
    if arm == ARM_CASH:
        return BuiltArm(ZeroWeightPolicy(symbols), None)
    if arm == ARM_BUY_AND_HOLD:
        return BuiltArm(
            BuyAndHoldPortfolioPolicy(
                symbols=symbols, gross_budget=protocol.buy_and_hold_gross_budget
            ),
            None,
        )
    horizon, long_only = {
        ARM_ALLOCATOR_24H: (24, False),
        ARM_ALLOCATOR_72H: (72, False),
        ARM_ALLOCATOR_24H_LONG_ONLY: (24, True),
    }[arm]
    adapter = PrequentialSimpleReturnSource(
        sources[horizon], decision_interval_hours=protocol.decision_interval_hours
    )
    policy = build_allocator(dataset, protocol, adapter, long_only=long_only)
    return BuiltArm(policy, adapter)


def reconcile_ledger(
    result: SharedCashPortfolioReplayResult,
    dataset: MarketDataset,
    protocol: DevStudyProtocol,
) -> dict[str, object]:
    """Independently rebuild terminal cash/equity from raw fills and ledger flows.

    ``terminal = initial + sum(-qty * price * multiplier over fills) - costs
    + funding - borrow + dividends + cash interest`` where fills come from the
    raw order events, never from the book's own cash. A flat terminal book makes
    cash and equity coincide.
    """

    wrapper = result.ledger_evidence
    if wrapper is None:
        raise ValueError("replay result carries no ledger evidence")
    ledger = wrapper.intervals
    intervals = ledger.intervals
    multipliers = dataset.resolved_array("contract_multipliers")
    initial = Fraction(str(protocol.initial_capital))
    signed_fill_cash = Fraction(0)
    filled_notional = Fraction(0)
    n_fills = 0
    cost = funding = borrow = dividend = interest = Fraction(0)
    chained = True
    previous_next: int | None = None
    previous_after: float | None = None
    tolerance = protocol.reconciliation_tolerance
    continuity_gap = 0.0
    for interval in intervals:
        if previous_next is not None and interval.start_index != previous_next:
            chained = False
        if previous_after is not None:
            continuity_gap = max(
                continuity_gap, abs(interval.portfolio_value_before - previous_after)
            )
        previous_next = interval.next_index
        previous_after = interval.portfolio_value_after
        for event in interval.order_events:
            if event.filled_quantity == 0.0:
                continue
            if event.execution_price is None:
                raise ValueError("fill without an execution price")
            quantity = Fraction(str(event.filled_quantity))
            price = Fraction(str(event.execution_price))
            multiplier = Fraction(str(float(multipliers[event.symbol_index])))
            signed_fill_cash -= quantity * price * multiplier
            filled_notional += abs(quantity) * price * multiplier
            n_fills += 1
        cost += Fraction(interval.interval_cost)
        funding += Fraction(interval.interval_funding)
        borrow += Fraction(interval.interval_borrow_cost)
        dividend += Fraction(interval.interval_dividend)
        interest += Fraction(interval.interval_cash_interest)
    expected = (
        initial + signed_fill_cash - cost + funding - borrow + dividend + interest
    )
    terminal_flat = all(
        Fraction(value) == 0 for value in ledger.terminal_exact_quantities
    )
    final_equity = ledger.final_portfolio_value
    cash_residual = float(Fraction(ledger.final_cash) - expected)
    equity_residual = float(Fraction(final_equity) - expected)
    totals_residual = max(
        abs(float(cost) - ledger.final_total_cost),
        abs(float(funding) - ledger.final_funding_pnl),
        abs(float(borrow) - ledger.final_borrow_cost),
    )
    passed = bool(
        terminal_flat
        and chained
        and ledger.termination_reason is None
        and abs(cash_residual) <= tolerance
        and abs(equity_residual) <= tolerance
        and continuity_gap <= tolerance
        and totals_residual <= tolerance
    )
    return {
        "passed": passed,
        "terminal_flat": terminal_flat,
        "intervals_chained": chained,
        "termination_reason": ledger.termination_reason,
        "n_intervals": len(intervals),
        "n_fills": n_fills,
        "signed_fill_cash": float(signed_fill_cash),
        "filled_notional": float(filled_notional),
        "execution_costs": float(cost),
        "funding_pnl": float(funding),
        "borrow_cost": float(borrow),
        "independent_terminal_equity": float(expected),
        "ledger_final_cash": ledger.final_cash,
        "ledger_final_equity": final_equity,
        "cash_residual": cash_residual,
        "equity_residual": equity_residual,
        "continuity_gap": continuity_gap,
        "ledger_total_residual": totals_residual,
    }


def equity_path(
    result: SharedCashPortfolioReplayResult,
    dataset: MarketDataset,
) -> tuple[np.ndarray, np.ndarray]:
    """Hourly mark-to-market equity (initial value first) and its timestamps."""

    ledger = result.ledger_evidence
    assert ledger is not None
    intervals = ledger.intervals.intervals
    stamps = _timestamps_ns(dataset)
    equity = np.empty(len(intervals) + 1, dtype=np.float64)
    times = np.empty(len(intervals) + 1, dtype=np.int64)
    equity[0] = intervals[0].portfolio_value_before
    times[0] = stamps[intervals[0].start_index]
    for position, interval in enumerate(intervals, start=1):
        equity[position] = interval.portfolio_value_after
        times[position] = stamps[interval.next_index]
    return equity, times


def exposure_summary(
    result: SharedCashPortfolioReplayResult, dataset: MarketDataset
) -> dict[str, float]:
    """Time-average gross and net exposure from ledger quantities and marks."""

    ledger = result.ledger_evidence
    assert ledger is not None
    intervals = ledger.intervals.intervals
    marks = dataset.resolved_array("mark_price")
    multipliers = dataset.resolved_array("contract_multipliers")
    gross: list[float] = []
    net: list[float] = []
    for interval in intervals:
        quantities = np.asarray(
            [float(Fraction(value)) for value in interval.exact_quantities_after]
        )
        notional = quantities * marks[interval.next_index] * multipliers
        equity = interval.portfolio_value_after
        gross.append(float(np.abs(notional).sum() / equity))
        net.append(float(notional.sum() / equity))
    return {
        "mean_gross_exposure": float(np.mean(gross)),
        "mean_net_exposure": float(np.mean(net)),
        "max_gross_exposure": float(np.max(gross)),
    }


@dataclass
class ReplayOutcome:
    arm: str
    scenario: str
    equity: np.ndarray
    timestamps_ns: np.ndarray
    summary: dict[str, object]


def run_replay(
    dataset: MarketDataset,
    protocol: DevStudyProtocol,
    arm: str,
    scenario: str,
    sources: Any,
    *,
    start_index: int,
    stop_index: int,
    arm_builder: ArmBuilder | None = None,
) -> ReplayOutcome:
    built = (arm_builder or build_arm)(arm, dataset, protocol, sources)
    result = run_shared_portfolio_replay(
        dataset,
        built.policy,
        start_index=start_index,
        stop_index=stop_index,
        initial_capital=protocol.initial_capital,
        execution_cost=scenario_cost(protocol, scenario),
        settle_terminal_position=True,
        capture_ledger_evidence=True,
    )
    equity, times = equity_path(result, dataset)
    performance = performance_summary(times, equity, protocol)
    reconciliation = reconcile_ledger(result, dataset, protocol)
    diagnostics = result.diagnostics
    net_pnl = float(equity[-1] - equity[0])
    costs = float(diagnostics.total_cost)
    funding = float(diagnostics.funding_pnl)
    years = float(performance["years"])
    summary: dict[str, object] = {
        "arm": arm,
        "scenario": scenario,
        **performance,
        "net_pnl_usdt": net_pnl,
        "execution_costs_usdt": costs,
        "funding_pnl_usdt": funding,
        "borrow_cost_usdt": float(diagnostics.borrow_cost),
        "gross_price_pnl_usdt": net_pnl + costs - funding,
        "fees_plus_funding_paid_usdt": costs - funding,
        "turnover_total": float(diagnostics.turnover_total),
        "turnover_per_year": float(diagnostics.turnover_total) / years,
        "n_trades": int(diagnostics.n_trades),
        "terminated": list(diagnostics.termination_reasons),
        "reconciliation": reconciliation,
        "exposure": exposure_summary(result, dataset),
    }
    if built.adapter is not None:
        history = np.asarray(built.adapter.history, dtype=np.float64)
        cost_rate = protocol.fee_rate + protocol.spread_rate
        summary["forecast"] = {
            "decision_calls": built.adapter.calls,
            "null_packets": built.adapter.null_packets,
            "mean_abs_mu_per_day": float(np.mean(np.abs(history))),
            "median_abs_mu_per_day": float(np.median(np.abs(history))),
            "share_abs_mu_above_one_way_cost": float(
                np.mean(np.abs(history) > cost_rate)
            ),
            "one_way_cost": cost_rate,
        }
    overlay_summary = getattr(built.policy, "overlay_summary", None)
    if callable(overlay_summary):
        summary["overlay"] = overlay_summary()
    records = getattr(built.policy, "records", None)
    if records is not None:
        outcomes = Counter(record.outcome for record in records)
        reasons = Counter(record.reason for record in records if record.reason)
        summary["allocator_decisions"] = {
            "n_decisions": len(records),
            "fail_closed_hold_count": int(outcomes.get("fail_closed", 0)),
            "outcomes": dict(sorted(outcomes.items())),
            "fail_closed_reasons": dict(sorted(reasons.items())),
        }
    return ReplayOutcome(arm, scenario, equity, times, summary)


def perturb_future(
    dataset: MarketDataset, *, after_index: int, seed: int = 12345
) -> MarketDataset:
    """Copy of ``dataset`` whose rows after ``after_index`` carry unrelated values."""

    rng = np.random.default_rng(seed)
    rows = dataset.n_bars - after_index - 1
    if rows <= 0:
        raise ValueError("nothing to perturb after the cut")
    shape = (rows, dataset.n_symbols)
    factor = np.exp(rng.normal(0.0, 0.5, size=shape))
    changes: dict[str, Any] = {}
    for name in ("open", "high", "low", "close", "mark_price", "index_price"):
        value = getattr(dataset, name, None)
        if isinstance(value, np.ndarray) and value.shape[:2] == (
            dataset.n_bars,
            dataset.n_symbols,
        ):
            copy = value.copy()
            copy[after_index + 1 :] = copy[after_index + 1 :] * factor
            changes[name] = copy
    volume = dataset.volume.copy()
    volume[after_index + 1 :] = volume[after_index + 1 :] * rng.uniform(
        0.2, 5.0, size=shape
    )
    changes["volume"] = volume
    features = dataset.features.copy()
    features[after_index + 1 :] = rng.normal(
        0.0, 1.0, size=features[after_index + 1 :].shape
    ).astype(features.dtype)
    changes["features"] = features
    funding = dataset.funding_rate.copy()
    funding[after_index + 1 :] = rng.normal(0.0, 0.001, size=shape)
    changes["funding_rate"] = funding
    # The dataset_id stays unchanged on purpose: same-bar order ids (and therefore
    # their admission order) are digests that include it, so a different id would
    # change pre-cut behaviour without any look-ahead.
    return replace(dataset, identity_payload_json=None, **changes)


def _trace(result: SharedCashPortfolioReplayResult) -> list[tuple[object, ...]]:
    ledger = result.ledger_evidence
    assert ledger is not None
    return [
        (
            interval.start_index,
            interval.exact_quantities_after,
            interval.cash_after,
            interval.portfolio_value_after,
        )
        for interval in ledger.intervals.intervals
    ] + [(d.index, d.proposal_weights, d.target_weights) for d in result.decisions]


def causality_probe(
    dataset: MarketDataset,
    protocol: DevStudyProtocol,
    sources: Any,
    *,
    start_index: int,
    arm: str = ARM_ALLOCATOR_24H,
    arm_builder: ArmBuilder | None = None,
    sources_builder: Callable[[MarketDataset, DevStudyProtocol], Any] | None = None,
) -> dict[str, object]:
    """Replay one arm to the cut on real and future-perturbed data.

    Every decision, quantity, cash and equity value up to the cut must be
    bit-identical: nothing after the cut may influence what happened before it.
    ``arm_builder`` / ``sources_builder`` let another arm roster (the Phase 2
    trials) reuse the probe; ``sources_builder`` builds the per-dataset sources
    for the perturbed copy and defaults to :func:`build_forecast_sources`.
    """

    build = arm_builder or build_arm
    make_sources = sources_builder or build_forecast_sources
    cut = _row_of(dataset, protocol.causality_cut)
    if cut <= start_index + 48:
        raise ValueError("causality cut must leave a meaningful pre-cut window")
    perturbed = perturb_future(dataset, after_index=cut)
    traces = []
    for data, source_map in (
        (dataset, sources),
        (perturbed, make_sources(perturbed, protocol)),
    ):
        built = build(arm, data, protocol, source_map)
        result = run_shared_portfolio_replay(
            data,
            built.policy,
            start_index=start_index,
            stop_index=cut,
            initial_capital=protocol.initial_capital,
            execution_cost=scenario_cost(protocol, SCENARIO_BASE),
            capture_ledger_evidence=True,
        )
        traces.append((_trace(result), built.adapter))
    real_trace, real_adapter = traces[0]
    perturbed_trace, perturbed_adapter = traces[1]
    history_real = [] if real_adapter is None else real_adapter.history
    history_perturbed = [] if perturbed_adapter is None else perturbed_adapter.history
    identical = real_trace == perturbed_trace and history_real == history_perturbed
    traded = any(
        any(Fraction(q) != 0 for q in cast(tuple[str, ...], row[1]))
        for row in real_trace[: len(real_trace) // 2]
    )
    return {
        "passed": bool(identical and traded),
        "identical_pre_cut_trace": bool(identical),
        "pre_cut_trace_has_trades": bool(traded),
        "cut": protocol.causality_cut,
        "n_trace_rows": len(real_trace),
        "n_decisions": len(history_real),
    }


def _git_head() -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip() or None


def load_dataset(path: str) -> MarketDataset:
    """Picklable loader so worker processes can read the verified Dataset themselves."""

    from trade_rl.data.artifacts.publication import (
        load_published_market_dataset_artifact,
    )

    return load_published_market_dataset_artifact(path)[1]


def _replay_scenario(
    dataset: MarketDataset,
    protocol: DevStudyProtocol,
    scenario: str,
    sources: Mapping[int, PrequentialForecastSource],
    *,
    start_index: int,
    stop_index: int,
    say: Progress,
) -> dict[str, ReplayOutcome]:
    outcomes: dict[str, ReplayOutcome] = {}
    for arm in protocol.arms:
        say(f"replay {arm} / {scenario}")
        outcomes[arm] = run_replay(
            dataset,
            protocol,
            arm,
            scenario,
            sources,
            start_index=start_index,
            stop_index=stop_index,
        )
    return outcomes


def _scenario_worker(
    loader: Callable[[], MarketDataset],
    protocol: DevStudyProtocol,
    scenario: str,
    start_index: int,
    stop_index: int,
) -> dict[str, ReplayOutcome]:
    dataset = loader()
    sources = build_forecast_sources(dataset, protocol)
    return _replay_scenario(
        dataset,
        protocol,
        scenario,
        sources,
        start_index=start_index,
        stop_index=stop_index,
        say=lambda message: print(message, flush=True),
    )


def run_study(
    dataset: MarketDataset,
    protocol: DevStudyProtocol,
    *,
    progress: Progress | None = None,
    parallel_loader: Callable[[], MarketDataset] | None = None,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Run every arm in every scenario; return the result mapping and equity arrays.

    With ``parallel_loader`` (a picklable zero-argument callable returning the same
    Dataset) each scenario runs in its own process; results are identical to the
    sequential path because every replay is deterministic and independent.
    """

    say = progress or (lambda message: None)
    if dataset.dataset_id != protocol.dataset_id:
        raise ValueError("dataset_id does not match the protocol")
    if tuple(protocol.arms) != (
        ARM_CASH,
        ARM_BUY_AND_HOLD,
        ARM_ALLOCATOR_24H,
        ARM_ALLOCATOR_72H,
        ARM_ALLOCATOR_24H_LONG_ONLY,
    ):
        raise ValueError("arm roster differs from the preregistered five arms")
    start_index = _row_of(dataset, protocol.evaluation_start)
    stop_index = _row_of(dataset, protocol.evaluation_end)
    sources = build_forecast_sources(dataset, protocol)

    by_scenario: dict[str, dict[str, ReplayOutcome]] = {}
    if parallel_loader is not None:
        with ProcessPoolExecutor(max_workers=len(protocol.scenarios)) as pool:
            futures = {
                scenario: pool.submit(
                    _scenario_worker,
                    parallel_loader,
                    protocol,
                    scenario,
                    start_index,
                    stop_index,
                )
                for scenario in protocol.scenarios
            }
            say("causality probe")
            causality = causality_probe(
                dataset, protocol, sources, start_index=start_index
            )
            say(f"causality probe passed={causality['passed']}")
            for scenario, future in futures.items():
                by_scenario[scenario] = future.result()
                say(f"scenario {scenario} complete")
    else:
        say("causality probe")
        causality = causality_probe(dataset, protocol, sources, start_index=start_index)
        say(f"causality probe passed={causality['passed']}")
        for scenario in protocol.scenarios:
            by_scenario[scenario] = _replay_scenario(
                dataset,
                protocol,
                scenario,
                sources,
                start_index=start_index,
                stop_index=stop_index,
                say=say,
            )
    outcomes: dict[tuple[str, str], ReplayOutcome] = {
        (arm, scenario): outcome
        for scenario, per_arm in by_scenario.items()
        for arm, outcome in per_arm.items()
    }
    arms_table: dict[str, dict[str, dict[str, Any]]] = {
        arm: {
            scenario: outcomes[(arm, scenario)].summary
            for scenario in protocol.scenarios
        }
        for arm in protocol.arms
    }

    validity = {
        arm: bool(
            causality["passed"]
            and all(
                bool(arms_table[arm][scenario]["reconciliation"]["passed"])
                for scenario in protocol.scenarios
            )
            and all(
                not arms_table[arm][scenario]["terminated"]
                for scenario in protocol.scenarios
            )
        )
        for arm in protocol.arms
    }

    bootstrap: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
    for scenario in protocol.scenarios:
        daily = {}
        for arm in protocol.arms:
            outcome = outcomes[(arm, scenario)]
            _, values = daily_equity(outcome.timestamps_ns, outcome.equity)
            daily[arm] = daily_returns(values)
        length = len(daily[ARM_CASH])
        indices = moving_block_indices(
            length,
            block=protocol.bootstrap_block_days,
            resamples=protocol.bootstrap_resamples,
            seed=protocol.bootstrap_seed,
        )
        table: dict[str, dict[str, dict[str, float]]] = {}
        for arm in protocol.arms:
            if arm == ARM_CASH:
                continue
            versus = {
                "vs_cash": bootstrap_excess(
                    daily[arm] - daily[ARM_CASH], indices, protocol
                ).to_mapping()
            }
            if arm in protocol.candidate_arms:
                versus["vs_equal_weight_buy_and_hold"] = bootstrap_excess(
                    daily[arm] - daily[ARM_BUY_AND_HOLD], indices, protocol
                ).to_mapping()
            table[arm] = versus
        bootstrap[scenario] = table

    selection = select_development_label(
        {
            arm: {
                scenario: {
                    "terminal_net_profit_pct": float(
                        arms_table[arm][scenario]["terminal_net_profit_pct"]
                    ),
                    "max_drawdown_pct": float(
                        arms_table[arm][scenario]["max_drawdown_pct"]
                    ),
                }
                for scenario in protocol.scenarios
            }
            for arm in protocol.candidate_arms
        },
        validity,
        protocol,
    )
    results: dict[str, Any] = {
        "schema_version": "dev_realdata_study_result_v1",
        "status": "development_evidence_only",
        "protocol_digest": protocol_digest(protocol),
        "protocol": protocol.canonical_payload(),
        "dataset_id": dataset.dataset_id,
        "symbols": list(dataset.symbols),
        "git_head": _git_head(),
        "causality_probe": causality,
        "validity": validity,
        "arms": arms_table,
        "bootstrap": bootstrap,
        "selection": selection.to_mapping(),
        "label": selection.label,
        "notes": [
            "development evidence only; never a WINNER",
            "allocator cost c_i = fee + spread is a lower bound: participation impact "
            "(impact_rate) is charged by the executor but not modelled by the allocator",
            "fail_closed_hold_count per arm/scenario counts allocator decisions that "
            "ended as a fail-closed HOLD instead of an optimal solution",
        ],
    }
    equity_arrays: dict[str, np.ndarray] = {}
    for (arm, scenario), outcome in outcomes.items():
        equity_arrays[f"{arm}__{scenario}__equity"] = outcome.equity
        equity_arrays[f"{arm}__{scenario}__timestamp_ns"] = outcome.timestamps_ns
    return results, equity_arrays


__all__ = [
    "ArmBuilder",
    "BuiltArm",
    "PrequentialSimpleReturnSource",
    "ReplayOutcome",
    "ZeroWeightPolicy",
    "build_allocator",
    "build_arm",
    "build_forecast_sources",
    "causality_probe",
    "equity_path",
    "feature_indices",
    "perturb_future",
    "reconcile_ledger",
    "run_replay",
    "run_study",
    "scenario_cost",
    "warmup_marks",
]
