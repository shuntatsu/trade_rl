"""Runner of ``xs_lowvol_funding_v1`` through ``run_shared_portfolio_replay`` only.

Arms: the weekly low-volatility / funding-contrarian book (``CANDIDATE``), its two ablations,
an equal-weight long control and ``CASH`` (all :class:`XsPolicy`), every one on the same cash
account. Scenarios: BASE, HIGH_COST and DELAY. The ledger reconciliation, the cell summary and
the future-perturbation helpers are the Core7 MTF study's and the trend harness's (imported,
not copied); the summary is called with ``execution_cost=`` so lean-core terminal dust is
recognised.

Forced closes. A held name whose archive ends (or whose gap starts) is closed by the policy
from 42 bars before its last available bar on. The engine charges the scenario cost on each
fill; this runner adds the extra adverse 5% of the closed notional as a cash debit at each
fill bar, applied to the equity path from that bar on. A position still held at the last
available bar refuses the run (no write-off branch). The debits are
reported (count, total charge, the result without them) and enter the cash check, the
returns, the drawdown and every metric of the cell.

The causality probe runs per ``(arm, signal lag)``: the market rows after a cut are perturbed
and every input row up to the cut and every decision must be identical.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, date, datetime
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np

from trade_rl.artifacts import canonical_json_bytes
from trade_rl.data.market import MarketDataset
from trade_rl.evaluation.core7_mtf_study import runner as core7_runner
from trade_rl.evaluation.replay import (
    SharedCashPortfolioReplayResult,
    run_shared_portfolio_replay,
)
from trade_rl.evaluation.trend_oos import runner as trend_runner
from trade_rl.evaluation.trend_oos.data import StageDataError
from trade_rl.evaluation.xs_lowvol_funding import decision, protocol
from trade_rl.evaluation.xs_lowvol_funding.data import (
    StageData,
    StageWindow,
    validate_stage_dataset,
)
from trade_rl.evaluation.xs_lowvol_funding.decision import CellKey, OosCell
from trade_rl.evaluation.xs_lowvol_funding.policy import (
    UnsupportedWriteOffError,
    XsInputs,
    XsPolicy,
)
from trade_rl.risk.pretrade import PreTradeRisk
from trade_rl.strategies.portfolio import PortfolioPolicy

_summarize_replay_cell = trend_runner._summarize_replay_cell
_perturb_future_market_rows = trend_runner._perturb_future_market_rows
_build_synthetic_prefix_snapshots = trend_runner._build_synthetic_prefix_snapshots
_decision_diff = trend_runner._decision_diff
_run_decisions = trend_runner._run_decisions
_calendar_year_returns = trend_runner._calendar_year_returns
_dust_adjusted = trend_runner._dust_adjusted
_annual_sharpe = core7_runner._compute_annual_sharpe
ProbeResult = trend_runner.ProbeResult

_PROBE_SEED_CUT = 17
_PROBE_SEED_FUTURE = 18
_QUANTITY_ATOL = 1e-12
CELL_DIR = "cells"
DECISION_NAME = "decision.json"
DEV_NAME = "dev-result.json"
MANIFEST_NAME = "xs-lowvol-funding-manifest.json"
INTERIM_PREFIX = "INTERIM_NOT_A_DECISION"
MODE_DECISION = "decision"
MODE_INTERIM = "interim"
#: A replay needs at least two weekly decision periods.
MIN_REPLAY_BARS = 2 * protocol.REBALANCE_EVERY_BARS

DISCLOSURES: tuple[str, ...] = (
    "every arm trades a cash account: after the per-name and gross caps the target is scaled "
    "by the tighter of 0.4 x perp gross <= 1.0 and net signed exposure <= 0.97; a cell whose "
    "cash net of cumulative execution costs and forced-close debits falls below -1e-6 x "
    "capital is INVALID (negative_cash_borrowing)",
    "the forced-close rule closes a held name from 42 bars before its last available bar "
    "(a modelled one-week delisting announcement that uses knowledge of the archive end); "
    "the extra adverse 5% is a post-replay cash debit at each fill bar, so the engine's "
    "risk layer does not see it",
    "a position still held at a name's last available bar refuses the run (no write-off "
    "branch; a disclosed limitation that would need an amendment)",
    "exchange rules (lot size, minimum notional) are one static snapshot, not point-in-time; "
    "a symbol absent from the snapshot uses the fallback rules",
    "an unavailable row carries a constant sentinel price, never a market price",
    "the funding mean treats each event of a bar as the bar's mean rate (exact when a bar "
    "holds at most one event)",
    "the causality probe is per (arm, signal lag), shared by the cost scenarios; the "
    "forced-close schedule is the declared exception to 'future rows never change a "
    "decision' (an availability notice, not a price)",
)


class _Hybrid:
    """Inputs that are *base* up to the current decision row and *perturbed* after."""

    def __init__(self, base: XsInputs, future: XsInputs) -> None:
        self._base = base
        self.inputs = future
        self._restored = -1

    def restore_through(self, index: int) -> None:
        lo = self._restored + 1
        if index < lo:
            return
        self.inputs.close[lo : index + 1] = self._base.close[lo : index + 1]
        self.inputs.funding_rate[lo : index + 1] = self._base.funding_rate[
            lo : index + 1
        ]
        self._restored = index


#: The shared replay's fill tracker raises this when the engine changes a position without a
#: fill (it settles a position held on an unavailable asset).
_TRACKER_DIVERGENCE = "execution fill events diverged from book quantities"


def inputs_of(stage: StageData) -> XsInputs:
    return XsInputs.from_dataset(stage.dataset, stage.member)


def replay(
    dataset: MarketDataset,
    policy: PortfolioPolicy,
    *,
    scenario: str,
    start_index: int,
    stop_index: int,
    risk: PreTradeRisk,
) -> SharedCashPortfolioReplayResult:
    """The only replay path: ``run_shared_portfolio_replay`` with settlement and ledger."""
    try:
        return run_shared_portfolio_replay(
            dataset,
            policy,
            start_index=start_index,
            stop_index=stop_index,
            initial_capital=protocol.INITIAL_CAPITAL,
            execution_cost=protocol.scenario_cost(scenario),
            risk=risk,
            settle_terminal_position=True,
            capture_ledger_evidence=True,
        )
    except RuntimeError as error:
        if _TRACKER_DIVERGENCE not in str(error):
            raise
        raise UnsupportedWriteOffError(
            "a forced close left a position on an asset that became unavailable; the "
            "shared replay cannot write it off (fill tracker diverged from the book)"
        ) from error


def make_policy(
    arm: str,
    *,
    inputs: XsInputs,
    start_index: int,
    lag: int,
    refuse_unclosed: bool = False,
) -> PortfolioPolicy:
    """The policy of an arm; ``lag`` is the scenario's extra signal lag in bars.

    ``refuse_unclosed`` (the real replays) refuses a position still held at a name's last
    available bar; the synthetic causality probe leaves it off.
    """
    return XsPolicy(
        arm=arm,
        inputs=inputs,
        start_index=start_index,
        signal_lag_bars=lag,
        refuse_unclosed=refuse_unclosed,
    )


def _prefix_diff(a: XsInputs, b: XsInputs, *, cut_index: int) -> float:
    """Largest absolute input difference over rows ``[0, cut_index]``."""
    sl = slice(0, cut_index + 1)
    worst = 0.0
    for name in ("close", "funding_rate"):
        diff = np.abs(getattr(a, name)[sl] - getattr(b, name)[sl])
        if not np.isfinite(diff).all():
            return math.inf
        worst = max(worst, float(diff.max(initial=0.0)))
    for name in ("active", "tradable", "member", "funding_count"):
        if not np.array_equal(getattr(a, name)[sl], getattr(b, name)[sl]):
            return math.inf
    return worst


def _arm_lags() -> list[int]:
    return sorted({protocol.scenario_signal_lag_bars(s) for s in protocol.SCENARIOS})


def _probe_stage(
    stage: StageData, *, stage_name: str, start_index: int, stop_index: int
) -> dict[tuple[str, int], Any]:
    """Probe every ``(arm, lag)`` over the decisions of the replay window."""
    dataset = stage.dataset
    last_decision = stop_index - 2
    cut = start_index + (last_decision - start_index) // 2
    base = inputs_of(stage)
    cut_inputs = XsInputs.from_dataset(
        _perturb_future_market_rows(dataset, cut_index=cut, seed=_PROBE_SEED_CUT),
        stage.member,
    )
    future_inputs = XsInputs.from_dataset(
        _perturb_future_market_rows(
            dataset, cut_index=start_index, seed=_PROBE_SEED_FUTURE
        ),
        stage.member,
    )
    input_diff = max(
        _prefix_diff(base, cut_inputs, cut_index=cut),
        _prefix_diff(base, future_inputs, cut_index=start_index),
    )
    snapshots = _build_synthetic_prefix_snapshots(
        dataset, start_index=start_index, stop_inclusive=last_decision
    )
    n_cut = sum(1 for snap in snapshots if snap.index <= cut)
    out: dict[tuple[str, int], Any] = {}
    for arm in protocol.ARMS:
        for lag in _arm_lags():
            reference = _run_decisions(
                make_policy(arm, inputs=base, start_index=start_index, lag=lag),
                snapshots,
            )
            at_cut = _run_decisions(
                make_policy(arm, inputs=cut_inputs, start_index=start_index, lag=lag),
                snapshots[:n_cut],
            )
            hybrid = _Hybrid(base, future_inputs.copy())
            every = _run_decisions(
                make_policy(
                    arm, inputs=hybrid.inputs, start_index=start_index, lag=lag
                ),
                snapshots,
                before_step=hybrid.restore_through,
            )
            prop_diff = max(
                _decision_diff(reference[:n_cut], at_cut),
                _decision_diff(reference, every),
            )
            out[(arm, lag)] = ProbeResult(
                family=arm,
                segment=stage_name,
                signal_lag_bars=lag,
                cut_index=cut,
                signal_max_abs_diff=input_diff,
                proposal_max_abs_diff=prop_diff,
                decisions_probed=len(snapshots),
            )
    return out


@dataclass(frozen=True, slots=True)
class ForcedClose:
    """One forced close: the leg, its last available bar and every fill bar with its notional."""

    leg: str
    last_bar: int
    fills: tuple[tuple[int, float], ...]

    @property
    def fill_notional(self) -> float:
        return float(sum(notional for _, notional in self.fills))

    @property
    def fill_charge(self) -> float:
        return protocol.FORCED_CLOSE_ADVERSE * self.fill_notional

    def to_dict(self) -> dict[str, object]:
        return {
            "leg": self.leg,
            "last_bar": self.last_bar,
            "fills": [[bar, notional] for bar, notional in self.fills],
            "fill_notional": self.fill_notional,
            "charge": self.fill_charge,
        }


def forced_closes(
    result: SharedCashPortfolioReplayResult, dataset: MarketDataset
) -> list[ForcedClose]:
    """The forced closes of a replay: legs held when the 42-bar notice window began.

    A leg's last available bar is ``L`` when it is active at ``L`` and not at ``L + 1``. Every
    fill of the leg on the bars ``L - 41 .. L`` is a forced-close fill and carries the extra
    adverse charge. A position still held after ``L`` refuses the run (the policy raises
    first; this is the same check on the ledger).
    """
    ledger = result.ledger_evidence
    if ledger is None or not ledger.intervals.intervals:
        return []
    intervals = ledger.intervals.intervals
    by_next = {iv.next_index: iv for iv in intervals}
    active = np.asarray(dataset.resolved_array("asset_active"), dtype=np.bool_)
    lead = protocol.FORCED_CLOSE_LEAD_BARS
    events: list[ForcedClose] = []
    for leg in range(active.shape[1]):
        for last in np.flatnonzero(active[:-1, leg] & ~active[1:, leg]):
            row = int(last)
            tail = by_next.get(row)
            if (
                tail is not None
                and abs(float(Fraction(tail.exact_quantities_after[leg])))
                > _QUANTITY_ATOL
            ):
                raise UnsupportedWriteOffError(
                    f"{dataset.symbols[leg]} still held at its last available bar {row}; "
                    "the protocol has no write-off branch (disclosed limitation)"
                )
            fills: list[tuple[int, float]] = []
            for bar in range(row - lead + 1, row + 1):
                interval = by_next.get(bar)
                if interval is None:
                    continue
                filled = sum(
                    float(event.filled_notional)
                    for event in interval.order_events
                    if event.symbol_index == leg
                    and event.processing_index == bar
                    and event.event_type in ("filled", "partial_fill")
                )
                if filled > 0.0:
                    fills.append((bar, filled))
            if fills:
                events.append(
                    ForcedClose(
                        leg=str(dataset.symbols[leg]), last_bar=row, fills=tuple(fills)
                    )
                )
    return events


def _equity_path(returns: np.ndarray) -> np.ndarray:
    return protocol.INITIAL_CAPITAL * np.cumprod(1.0 + returns)


def _charged_returns(
    returns: np.ndarray,
    intervals: Any,
    closes: list[ForcedClose],
) -> tuple[np.ndarray, np.ndarray]:
    """``(returns after the debits, cumulative debit per interval)``.

    The charge of each forced-close fill is debited at its fill bar (an interval is identified
    by the row it ends on); each debit lowers the equity path from that interval on by its
    full amount.
    """
    debit = np.zeros(returns.size, dtype=np.float64)
    if len(intervals) != returns.size:
        return returns, debit
    position = {iv.next_index: k for k, iv in enumerate(intervals)}
    for event in closes:
        for bar, notional in event.fills:
            if bar in position:
                debit[position[bar]] += protocol.FORCED_CLOSE_ADVERSE * notional
    cumulative = np.cumsum(debit)
    if not cumulative.any():
        return returns, cumulative
    path = _equity_path(returns) - cumulative
    previous = np.concatenate([[protocol.INITIAL_CAPITAL], path[:-1]])
    safe = previous > 0.0
    adjusted = np.where(safe, path / np.where(safe, previous, 1.0) - 1.0, -1.0)
    return np.maximum(adjusted, -1.0), cumulative


def _half_sharpes(
    dataset: MarketDataset, intervals: Any, returns: np.ndarray
) -> dict[str, Any]:
    """Annualised Sharpe of the two DISJOINT halves, split by the date of each bar close."""
    if len(intervals) != returns.size:
        return {"H1": None, "H2": None}
    stamps = dataset.timestamps.astype("datetime64[D]")
    days = np.asarray([stamps[iv.next_index] for iv in intervals])
    out: dict[str, Any] = {}
    for name, (lo, hi) in (("H1", protocol.HALF_1), ("H2", protocol.HALF_2)):
        mask = (days >= np.datetime64(lo)) & (days <= np.datetime64(hi))
        out[name] = _annual_sharpe(returns[mask])
    return out


def _leg_attribution(
    result: SharedCashPortfolioReplayResult, dataset: MarketDataset
) -> dict[str, object] | None:
    """Close-to-close PnL of the held long and short quantities (reported, never gating)."""
    ledger = result.ledger_evidence
    if ledger is None or len(ledger.intervals.intervals) < 2:
        return None
    intervals = ledger.intervals.intervals
    close = np.asarray(dataset.close, dtype=np.float64)
    long_pnl = short_pnl = 0.0
    for current, following in zip(intervals[:-1], intervals[1:], strict=True):
        quantity = np.asarray(
            [float(Fraction(q)) for q in current.exact_quantities_after],
            dtype=np.float64,
        )
        move = close[following.next_index] - close[current.next_index]
        pnl = quantity * move
        long_pnl += float(pnl[quantity > 0.0].sum())
        short_pnl += float(pnl[quantity < 0.0].sum())
    return {
        "long_pnl_usdt": long_pnl,
        "short_pnl_usdt": short_pnl,
        "basis": "close_to_close_of_held_quantity_excluding_costs_funding_and_debits",
    }


def _min_cash_and_borrowing(
    result: SharedCashPortfolioReplayResult, cumulative_debit: np.ndarray
) -> tuple[float | None, float | None]:
    """``(min cash, max borrowing)`` with the forced-close debits taken out of cash."""
    ledger = result.ledger_evidence
    if ledger is None or not ledger.intervals.intervals:
        return None, None
    intervals = ledger.intervals.intervals
    cash = np.asarray([float(iv.cash_after) for iv in intervals]) - cumulative_debit
    cost = np.asarray([float(iv.total_cost_after) for iv in intervals])
    return float(cash.min()), float(np.maximum(0.0, -cash - cost).max())


def _build_cell(
    *,
    arm: str,
    scenario: str,
    result: SharedCashPortfolioReplayResult,
    dataset: MarketDataset,
    causality_passed: bool,
    probe_key: str,
    policy_statistics: Mapping[str, Any],
) -> OosCell:
    summary = _summarize_replay_cell(
        arm=arm,
        scenario=scenario,
        result=result,
        dataset=dataset,
        initial_capital=protocol.INITIAL_CAPITAL,
        causality_passed=causality_passed,
        execution_cost=protocol.scenario_cost(scenario),
    )
    stop_reason = summary.stop_reason
    if (
        stop_reason == "max_drawdown_hard_stop"
        and summary.max_drawdown < protocol.RISK_DRAWDOWN_STOP
    ):
        # the shared core7 summary reports a hard stop at its own 20%; this protocol's
        # stop is 25% (reporting only, a stop never invalidates a cell)
        stop_reason = None
    raw_returns = np.asarray(result.returns.values, dtype=np.float64)
    returns = _dust_adjusted(raw_returns, result, summary.terminal_dust)
    reasons = list(summary.invalid_reasons)
    ledger = result.ledger_evidence
    intervals = ledger.intervals.intervals if ledger is not None else ()
    closes = forced_closes(result, dataset)
    charged, cumulative = _charged_returns(returns, intervals, closes)
    years = _calendar_year_returns(dataset, result, charged)
    if years is None or len(intervals) != returns.size:
        reasons.append("returns_not_aligned_with_ledger_intervals")
        years = {}
    total_charge = float(sum(e.fill_charge for e in closes))
    total_return = summary.total_return
    sharpe = summary.annual_sharpe
    drawdown = summary.max_drawdown
    if total_charge > 0.0 and charged.size:
        path = _equity_path(charged)
        total_return = float(path[-1] / protocol.INITIAL_CAPITAL - 1.0)
        sharpe = float(_annual_sharpe(charged))
        peak = np.maximum(protocol.INITIAL_CAPITAL, np.maximum.accumulate(path))
        drawdown = max(drawdown, float(np.max(1.0 - path / peak)))
    min_cash, max_borrowing = _min_cash_and_borrowing(result, cumulative)
    tolerance = protocol.CASH_TOLERANCE_FRACTION * protocol.INITIAL_CAPITAL
    if max_borrowing is not None and max_borrowing > tolerance:
        reasons.append(protocol.REASON_NEGATIVE_CASH)
    extras: dict[str, Any] = {
        "policy": dict(policy_statistics),
        "half_sharpes": _half_sharpes(dataset, intervals, charged),
        "forced_close": {
            "count": len(closes),
            "total_charge_usdt": total_charge,
            "events": [event.to_dict() for event in closes],
            "without_charge": {
                "total_return": summary.total_return,
                "annual_sharpe": summary.annual_sharpe,
                "max_drawdown": summary.max_drawdown,
            },
        },
        "leg_attribution": (
            None if arm == protocol.CONTROL_CASH else _leg_attribution(result, dataset)
        ),
    }
    return OosCell(
        arm=arm,
        scenario=scenario,
        total_return=total_return,
        annual_sharpe=sharpe,
        max_drawdown=drawdown,
        total_cost=summary.total_cost,
        funding_pnl=summary.funding_pnl,
        turnover_total=summary.turnover_total,
        complete=summary.complete,
        terminal_flat=summary.terminal_flat,
        terminal_dust_only=summary.terminal_dust_only,
        terminal_dust=tuple(summary.terminal_dust),
        execution_valid=summary.execution_valid,
        causality_passed=summary.causality_passed,
        ledger_reconciled=summary.ledger_reconciled,
        stopped=stop_reason is not None,
        stop_reason=stop_reason,
        invalid_reasons=tuple(reasons),
        n_returns=int(charged.size),
        calendar_year_returns=years,
        returns=tuple(float(r) for r in charged),
        min_cash=min_cash,
        max_borrowing=max_borrowing,
        cap_statistics=extras,
        causality_probe_key=probe_key,
    )


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(dict(payload)))


def _stats_of(policy: PortfolioPolicy) -> dict[str, object]:
    stats = getattr(policy, "stats", None)
    return stats.to_dict() if stats is not None else {}


def run_stage(
    stage: StageData,
    *,
    window: StageWindow,
    output_dir: Path,
    expected_protocol_digest: str,
    mode: str = MODE_DECISION,
    disjoint_label: str | None = None,
    code_commit: str | None = None,
    authorization: Mapping[str, Any] | None = None,
    run_date: date | None = None,
    on_cell: Callable[[OosCell], None] | None = None,
) -> dict[str, Any]:
    """Run every arm x scenario cell of one stage and report, decide or write an interim.

    Fails closed unless the protocol digest is exactly the frozen one, the dataset is the
    stage's layout with a clean clock, and (decision mode) it spans the whole window.
    ``output_dir`` must not hold a previous run. Every cell and the manifest are written
    before the result is written and returned. DEV writes a diagnostic report and never a
    label; an interim report carries no label; a FORWARD decision needs the DISJOINT label.
    """
    if mode not in (MODE_DECISION, MODE_INTERIM):
        raise ValueError(f"unknown mode: {mode!r}")
    if mode == MODE_INTERIM and window.stage != protocol.STAGE_FORWARD:
        raise ValueError("only FORWARD has an interim mode")
    digest = protocol.protocol_digest()
    if expected_protocol_digest != digest:
        raise ValueError(
            "expected protocol digest does not match the frozen protocol digest "
            f"{digest}"
        )
    if (
        window.stage == protocol.STAGE_FORWARD
        and mode == MODE_DECISION
        and disjoint_label not in protocol.LABELS
    ):
        raise ValueError("a FORWARD decision needs the DISJOINT label")
    last_index = validate_stage_dataset(stage, window)
    dataset = stage.dataset
    start = window.first_decision_index
    stop = last_index
    if mode == MODE_DECISION and stop != window.last_row_index:
        raise StageDataError(
            "a decision needs the whole stage window: the dataset ends at row "
            f"{stop}, the stage at {window.last_row_index}"
        )
    if stop - start < MIN_REPLAY_BARS:
        raise StageDataError("the replay window is too short for two decision periods")
    when = run_date or datetime.now(UTC).date()
    report_name = f"{INTERIM_PREFIX}-{when.strftime('%Y%m%d')}.json"
    if mode == MODE_INTERIM:
        final_name = report_name
    elif window.stage == protocol.STAGE_DEV:
        final_name = DEV_NAME
    else:
        final_name = DECISION_NAME
    if any(
        (output_dir / name).exists()
        for name in (DECISION_NAME, DEV_NAME, MANIFEST_NAME, report_name)
    ):
        raise FileExistsError(f"refusing to overwrite an existing run in {output_dir}")

    inputs = inputs_of(stage)
    risk = PreTradeRisk(protocol.risk_config())
    probes = _probe_stage(
        stage, stage_name=window.stage, start_index=start, stop_index=stop
    )
    cells: dict[CellKey, OosCell] = {}
    for arm in protocol.ARMS:
        for scenario in protocol.SCENARIOS:
            lag = protocol.scenario_signal_lag_bars(scenario)
            policy = make_policy(
                arm, inputs=inputs, start_index=start, lag=lag, refuse_unclosed=True
            )
            result = replay(
                dataset,
                policy,
                scenario=scenario,
                start_index=start,
                stop_index=stop,
                risk=risk,
            )
            cell = _build_cell(
                arm=arm,
                scenario=scenario,
                result=result,
                dataset=dataset,
                causality_passed=probes[(arm, lag)].passed,
                probe_key=f"{arm}__lag{lag}",
                policy_statistics=_stats_of(policy),
            )
            cells[cell.key] = cell
            _write_json(
                output_dir / CELL_DIR / f"{arm}__{scenario}.json",
                cell.to_dict(include_returns=True),
            )
            if on_cell is not None:
                on_cell(cell)

    if mode == MODE_INTERIM:
        outcome = decision.interim_report(cells)
    elif window.stage == protocol.STAGE_DEV:
        outcome = decision.dev_report(cells)
    elif window.stage == protocol.STAGE_DISJOINT:
        outcome = decision.decide_disjoint(cells)
    else:
        assert disjoint_label is not None
        outcome = decision.decide_forward(cells, disjoint_label=disjoint_label)
    header: dict[str, Any] = {
        "protocol_id": protocol.PROTOCOL_ID,
        "protocol_digest": digest,
        "candidates_sha256": protocol.CANDIDATES_SHA256,
        "stage": window.stage,
        "mode": mode,
        "window": window.to_dict(),
        "through_close": str(dataset.timestamps[-1]) + "Z",
        "replay_start_index": start,
        "replay_stop_index": stop,
        "dataset_id": dataset.dataset_id,
        "membership_sha256": stage.member_sha256(),
        "harness_source_digest": protocol.harness_source_digest(),
        "code_commit": code_commit,
        "run_date": when.isoformat(),
        "output_dir_name": output_dir.name,
        "single_use_stage": window.stage == protocol.STAGE_DISJOINT,
        "disclosures": list(DISCLOSURES),
    }
    manifest: dict[str, Any] = {
        **header,
        "authorization": None if authorization is None else dict(authorization),
        "arms": list(protocol.ARMS),
        "scenarios": list(protocol.SCENARIOS),
        "cell_files": sorted(
            f"{CELL_DIR}/{a}__{s}.json"
            for a in protocol.ARMS
            for s in protocol.SCENARIOS
        ),
        "causality_probes": {
            f"{arm}__lag{lag}": probe.to_dict() for (arm, lag), probe in probes.items()
        },
        "label": outcome.get("label"),
        "result_file": final_name,
    }
    _write_json(output_dir / MANIFEST_NAME, manifest)
    full: dict[str, Any] = {**header, **outcome}
    _write_json(output_dir / final_name, full)
    return full


__all__ = [
    "CELL_DIR",
    "DECISION_NAME",
    "DEV_NAME",
    "DISCLOSURES",
    "INTERIM_PREFIX",
    "MANIFEST_NAME",
    "MIN_REPLAY_BARS",
    "MODE_DECISION",
    "MODE_INTERIM",
    "ForcedClose",
    "ProbeResult",
    "UnsupportedWriteOffError",
    "forced_closes",
    "inputs_of",
    "make_policy",
    "replay",
    "run_stage",
]
