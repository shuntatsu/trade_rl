"""Runner of the out-of-sample test through ``run_shared_portfolio_replay`` only.

The policy and its targets are lowturn's, unchanged (:class:`LowturnPolicy`,
``family_targets``); the ledger reconciliation, the cell summary and the
future-perturbation helpers are the Core7 MTF study's and lowturn's (imported, not
copied). The cell summary is called with ``execution_cost=`` so that lean-core terminal
dust is recognised (its default ``None`` fails closed). The causality probe is the same
kind of test as lowturn's: every decision of every ``(arm, signal lag)`` policy must be
identical when all market rows after the decision are perturbed. Only the signal builder
differs (:func:`~trade_rl.evaluation.trend_oos.signals.build_oos_signals` masks unlisted
rows).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import numpy as np

from trade_rl.artifacts import canonical_json_bytes
from trade_rl.data.market import MarketDataset
from trade_rl.evaluation.core7_mtf_study import runner as core7_runner
from trade_rl.evaluation.lowturn_screen import runner as lowturn_runner
from trade_rl.evaluation.lowturn_screen.policies import LowturnPolicy
from trade_rl.evaluation.lowturn_screen.signals import LowturnSignals
from trade_rl.evaluation.replay import (
    SharedCashPortfolioReplayResult,
    run_shared_portfolio_replay,
)
from trade_rl.evaluation.trend_oos import protocol
from trade_rl.evaluation.trend_oos.cash_policy import CashAccountPolicy
from trade_rl.evaluation.trend_oos.data import (
    StageDataError,
    StageWindow,
    validate_stage_dataset,
)
from trade_rl.evaluation.trend_oos.decision import (
    CellKey,
    OosCell,
    decide,
    interim_report,
)
from trade_rl.evaluation.trend_oos.signals import build_oos_signals
from trade_rl.risk.pretrade import PreTradeRisk

# Shared with the Core7 MTF and lowturn runners (single implementation of each check).
_summarize_replay_cell = core7_runner._summarize_replay_cell
_perturb_future_market_rows = core7_runner._perturb_future_market_rows
_build_synthetic_prefix_snapshots = core7_runner._build_synthetic_prefix_snapshots
_decision_diff = core7_runner._decision_diff
_run_decisions = core7_runner._run_decisions
_HybridSignals = lowturn_runner._HybridSignals
_prefix_signal_diff = lowturn_runner._prefix_signal_diff
_calendar_year_returns = lowturn_runner._calendar_year_returns
ProbeResult = lowturn_runner.ProbeResult

_PROBE_SEED_CUT = 17
_PROBE_SEED_FUTURE = 18
CELL_DIR = "cells"
DECISION_NAME = "decision.json"
MANIFEST_NAME = "trend-oos-manifest.json"
INTERIM_PREFIX = "INTERIM_NOT_A_DECISION"
MODE_DECISION = "decision"
MODE_INTERIM = "interim"
#: An interim replay needs at least two weekly rebalance periods.
MIN_REPLAY_BARS = 2 * protocol.REBALANCE_EVERY_BARS

DISCLOSURES: tuple[str, ...] = (
    "every arm trades a spot cash account: after the lowturn caps the spot gross is "
    "capped at 1.0 (lowturn v1 allowed 2.0 with zero-cost financing); a cell that "
    "borrows at any ordered spot fill or ledger close is INVALID "
    "(negative_cash_borrowing); later sells cannot prefinance earlier buys; "
    "the fee-inclusive conservative lower bound can invalidate safe controls "
    "when native per-fill fee timing is not recorded",
    "the causality probe is per (arm, signal lag) and shared by the cost scenarios",
    "exchange rules (lot size, minimum notional) are one static snapshot applied to "
    "every row, not point-in-time; for BACKCAST this approximates 2018-2020 by the "
    "retrieval-time rules",
    "no perpetual leg is loaded: every perp leg is unavailable in every row and neither "
    "arm reads it",
    "an unlisted spot row carries a constant sentinel price, never a market price; "
    "its signals are missing and the leg has weight 0",
    "the last dataset row is two bars after the last decision bar (the replay engine "
    "decides through stop - 2, flattens on stop - 1 and marks on stop)",
)


def _probe_stage(
    dataset: MarketDataset,
    signals: LowturnSignals,
    *,
    stage: str,
    start_index: int,
    stop_index: int,
) -> dict[tuple[str, int], Any]:
    """Probe every ``(arm, lag)`` over the decisions of the replay window."""
    last_decision = stop_index - 2  # agent_stop_index = stop_index - 1 with settlement
    cut = start_index + (last_decision - start_index) // 2
    pack_cut = build_oos_signals(
        _perturb_future_market_rows(dataset, cut_index=cut, seed=_PROBE_SEED_CUT)
    )
    pack_future = build_oos_signals(
        _perturb_future_market_rows(
            dataset, cut_index=start_index, seed=_PROBE_SEED_FUTURE
        )
    )
    signal_diff = max(
        _prefix_signal_diff(signals, pack_cut, cut_index=cut),
        _prefix_signal_diff(signals, pack_future, cut_index=start_index),
    )
    snapshots = _build_synthetic_prefix_snapshots(
        dataset, start_index=start_index, stop_inclusive=last_decision
    )
    n_cut = sum(1 for snap in snapshots if snap.index <= cut)
    out: dict[tuple[str, int], Any] = {}
    lags = sorted({protocol.scenario_signal_lag_bars(s) for s in protocol.SCENARIOS})
    for arm in protocol.ARMS:
        for lag in lags:

            def make(
                pack: LowturnSignals, arm: str = arm, lag: int = lag
            ) -> CashAccountPolicy:
                return CashAccountPolicy(
                    LowturnPolicy(
                        family=arm,
                        signals=pack,
                        start_index=start_index,
                        signal_lag_bars=lag,
                    ),
                    start_index=start_index,
                )

            base = _run_decisions(make(signals), snapshots)
            at_cut = _run_decisions(make(pack_cut), snapshots[:n_cut])
            hybrid = _HybridSignals(signals, pack_future)
            every = _run_decisions(
                make(hybrid.signals), snapshots, before_step=hybrid.restore_through
            )
            prop_diff = max(
                _decision_diff(base[:n_cut], at_cut), _decision_diff(base, every)
            )
            out[(arm, lag)] = ProbeResult(
                family=arm,
                segment=stage,
                signal_lag_bars=lag,
                cut_index=cut,
                signal_max_abs_diff=signal_diff,
                proposal_max_abs_diff=prop_diff,
                decisions_probed=len(snapshots),
            )
    return out


def _dust_adjusted(
    returns: np.ndarray,
    result: SharedCashPortfolioReplayResult,
    terminal_dust: Any,
) -> np.ndarray:
    """The return series with the terminal dust exit cost deducted from the last bar.

    The cell's total return is ``(portfolio value - dust exit cost) / capital - 1``; the
    per-bar series used for the per-year returns and the bootstrap must agree with it.
    """
    dust_cost = sum(float(item["exit_cost_usdt"]) for item in terminal_dust)
    value = float(result.book.portfolio_value)
    if dust_cost <= 0.0 or returns.size == 0 or value <= dust_cost:
        return returns
    adjusted = returns.copy()
    adjusted[-1] = (1.0 + adjusted[-1]) * (value - dust_cost) / value - 1.0
    return adjusted


def _cash_extremes(
    result: SharedCashPortfolioReplayResult,
) -> tuple[float | None, float | None]:
    """``(min cash, max borrowing)`` over the ledger intervals.

    The fee-only overdraft diagnostic is tracked only while that overdraft remains
    outstanding. A prior fee is not new purchasing power after cash recovers. Raw
    negative cash (even if entirely explained by fees) separately invalidates the
    v2 spot-cash cell in ``_build_cell``.
    """
    ledger = result.ledger_evidence
    if ledger is None or not ledger.intervals.intervals:
        return None, None
    lowest = math.inf
    greatest_borrow = 0.0
    unpaid_fee_overdraft = 0.0
    previous_cost: float | None = None
    for iv in ledger.intervals.intervals:
        cash = float(iv.cash_after)
        before = float(iv.total_cost_before)
        after = float(iv.total_cost_after)
        if not all(math.isfinite(value) for value in (cash, before, after)):
            raise ValueError("nonfinite cash or execution cost in ledger evidence")
        if before < 0.0 or after < before:
            raise ValueError("execution cost must be cumulative and nonnegative")
        if previous_cost is not None and not math.isclose(
            previous_cost, before, rel_tol=1e-12, abs_tol=1e-9
        ):
            raise ValueError("execution cost ledger intervals are discontinuous")
        interval_cost = after - before
        unpaid_fee_overdraft = min(
            max(0.0, -cash), unpaid_fee_overdraft + interval_cost
        )
        greatest_borrow = max(greatest_borrow, max(0.0, -cash - unpaid_fee_overdraft))
        lowest = min(lowest, cash)
        previous_cost = after
    return lowest, greatest_borrow


def _spot_fill_cash_floor(result: SharedCashPortfolioReplayResult) -> float | None:
    """Conservative cost-inclusive cash floor over native ordered spot fills.

    OrderEvents carry no per-fill fee. Reserve all negative external flows
    before any order but deduct the complete verified interval execution cost
    immediately *after the first native fill's signed cash transfer*. This is
    still a lower bound at each fill: actual fee costs never exceed their
    nonnegative whole-interval sum, but sale proceeds cannot arrive before the
    sale itself. Positive external credits are not anticipated. A first sale
    with sufficient proceeds must not be invalidated by precharging its fee.
    The lack of exact per-fill fee timing can still cause false positives.
    """

    ledger = result.ledger_evidence
    if ledger is None or not ledger.intervals.intervals:
        return None
    lowest = math.inf
    for iv in ledger.intervals.intervals:
        credits = tuple(
            float(value)
            for value in (
                iv.interval_funding,
                iv.interval_dividend,
                iv.interval_cash_interest,
            )
        )
        if not all(math.isfinite(value) for value in credits):
            raise ValueError("nonfinite external credit in fill evidence")
        cost_before = float(iv.total_cost_before)
        cost_after = float(iv.total_cost_after)
        interval_cost = float(iv.interval_cost)
        if (
            not all(math.isfinite(v) for v in (cost_before, cost_after, interval_cost))
            or cost_before < 0.0
            or interval_cost < 0.0
            or cost_after < cost_before
            or not math.isclose(
                cost_after - cost_before, interval_cost, rel_tol=1e-12, abs_tol=1e-9
            )
        ):
            raise ValueError("invalid interval execution cost in fill evidence")
        cash = float(iv.cash_before) + sum(min(0.0, value) for value in credits)
        cash_after = float(iv.cash_after)
        if not math.isfinite(cash) or not math.isfinite(cash_after):
            raise ValueError("nonfinite cash in fill evidence")
        lowest = min(lowest, cash, cash_after)
        fees_charged = False
        for event in sorted(iv.order_events, key=lambda event: event.sequence):
            if event.event_type not in ("filled", "partial_fill"):
                continue
            # The v2 research policy is spot-only; a perp fill would break
            # the cash-principal reconstruction and must not be accepted.
            if event.symbol_index % 2:
                raise ValueError("unexpected perpetual fill in spot cash account")
            quantity = float(event.filled_quantity)
            notional = float(event.filled_notional)
            if (
                not math.isfinite(quantity)
                or quantity == 0.0
                or not math.isfinite(notional)
                or notional < 0.0
            ):
                raise ValueError("invalid native spot fill in cash evidence")
            cash += (-1 if quantity > 0 else 1) * notional
            if not fees_charged:
                # No native per-fill fee is available. Charging all interval
                # fees after the first fill underestimates every subsequent
                # actual cash balance without inventing an earlier fee debit.
                cash -= interval_cost
                fees_charged = True
            if not math.isfinite(cash):
                raise ValueError("nonfinite running cash in fill evidence")
            lowest = min(lowest, cash)
        if not fees_charged:
            # No fill can be shown to finance a reported interval fee.
            # Refuse the unallocated cost conservatively at interval start.
            lowest = min(lowest, cash - interval_cost)
    return lowest


def _build_cell(
    *,
    arm: str,
    scenario: str,
    result: SharedCashPortfolioReplayResult,
    dataset: MarketDataset,
    causality_passed: bool,
    probe_key: str,
    cap_statistics: Mapping[str, Any],
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
    raw_returns = np.asarray(result.returns.values, dtype=np.float64)
    returns = _dust_adjusted(raw_returns, result, summary.terminal_dust)
    reasons = list(summary.invalid_reasons)
    min_cash, max_borrowing = _cash_extremes(result)
    fill_floor = _spot_fill_cash_floor(result)
    if min_cash is not None and fill_floor is not None:
        min_cash = min(min_cash, fill_floor)
    tolerance = protocol.CASH_TOLERANCE_FRACTION * protocol.INITIAL_CAPITAL
    # A real spot cash account cannot finance even execution fees by overdraft.
    # The net-of-unpaid-fees measure is a separate diagnostic, never credit.
    if (min_cash is not None and min_cash < -tolerance) or (
        max_borrowing is not None and max_borrowing > tolerance
    ):
        reasons.append(protocol.REASON_NEGATIVE_CASH)
    years = _calendar_year_returns(dataset, result, returns)
    if years is None:
        reasons.append("returns_not_aligned_with_ledger_intervals")
        years = {}
    return OosCell(
        arm=arm,
        scenario=scenario,
        total_return=summary.total_return,
        annual_sharpe=summary.annual_sharpe,
        max_drawdown=summary.max_drawdown,
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
        stopped=summary.stop_reason is not None,
        stop_reason=summary.stop_reason,
        invalid_reasons=tuple(reasons),
        n_returns=int(returns.size),
        calendar_year_returns=years,
        returns=tuple(float(r) for r in returns),
        min_cash=min_cash,
        max_borrowing=max_borrowing,
        cap_statistics=dict(cap_statistics),
        causality_probe_key=probe_key,
    )


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(dict(payload)))


def run_stage(
    dataset: MarketDataset,
    *,
    window: StageWindow,
    output_dir: Path,
    expected_protocol_digest: str,
    mode: str = MODE_DECISION,
    code_commit: str | None = None,
    authorization: Mapping[str, Any] | None = None,
    run_date: date | None = None,
    on_cell: Callable[[OosCell], None] | None = None,
) -> dict[str, Any]:
    """Run every arm x scenario cell of one stage and decide (or report an interim).

    Fails closed unless the protocol digest is exactly the frozen one, the dataset is the
    stage's 22-leg layout with unavailable perpetuals and a clean clock, and (decision
    mode) it spans the whole stage window. ``output_dir`` must not hold a previous run.
    Every cell and the manifest are written before the decision (or interim report) is
    written and returned. An interim report carries no label and never decides.
    """
    if mode not in (MODE_DECISION, MODE_INTERIM):
        raise ValueError(f"unknown mode: {mode!r}")
    if expected_protocol_digest != protocol.protocol_digest():
        raise ValueError(
            "expected protocol digest does not match the frozen protocol digest "
            f"{protocol.protocol_digest()}"
        )
    last_index = validate_stage_dataset(dataset, window)
    start = window.first_decision_index
    stop = last_index
    if mode == MODE_DECISION and stop != window.last_row_index:
        raise StageDataError(
            "a decision needs the whole stage window: the dataset ends at row "
            f"{stop}, the stage at {window.last_row_index}"
        )
    if stop - start < MIN_REPLAY_BARS:
        raise StageDataError("the replay window is too short for two rebalance periods")
    when = run_date or datetime.now(UTC).date()
    report_name = f"{INTERIM_PREFIX}-{when.strftime('%Y%m%d')}.json"
    final_name = DECISION_NAME if mode == MODE_DECISION else report_name
    if any(
        (output_dir / name).exists()
        for name in (DECISION_NAME, MANIFEST_NAME, report_name)
    ):
        raise FileExistsError(f"refusing to overwrite an existing run in {output_dir}")

    signals = build_oos_signals(dataset)
    risk = PreTradeRisk(protocol.risk_config())
    probes = _probe_stage(
        dataset, signals, stage=window.stage, start_index=start, stop_index=stop
    )
    cells: dict[CellKey, OosCell] = {}
    for arm in protocol.ARMS:
        for scenario in protocol.SCENARIOS:
            lag = protocol.scenario_signal_lag_bars(scenario)
            policy = CashAccountPolicy(
                LowturnPolicy(
                    family=arm,
                    signals=signals,
                    start_index=start,
                    signal_lag_bars=lag,
                ),
                start_index=start,
            )
            result = run_shared_portfolio_replay(
                dataset,
                policy,
                start_index=start,
                stop_index=stop,
                initial_capital=protocol.INITIAL_CAPITAL,
                execution_cost=protocol.scenario_cost(scenario),
                risk=risk,
                settle_terminal_position=True,
                capture_ledger_evidence=True,
            )
            cell = _build_cell(
                arm=arm,
                scenario=scenario,
                result=result,
                dataset=dataset,
                causality_passed=probes[(arm, lag)].passed,
                probe_key=f"{arm}__lag{lag}",
                cap_statistics=policy.stats.to_dict(),
            )
            cells[cell.key] = cell
            _write_json(
                output_dir / CELL_DIR / f"{arm}__{scenario}.json",
                cell.to_dict(include_returns=True),
            )
            if on_cell is not None:
                on_cell(cell)

    outcome = decide(cells) if mode == MODE_DECISION else interim_report(cells)
    header: dict[str, Any] = {
        "protocol_id": protocol.PROTOCOL_ID,
        "protocol_digest": protocol.protocol_digest(),
        "lowturn_protocol_digest": protocol.LOWTURN_PROTOCOL_DIGEST,
        "stage": window.stage,
        "mode": mode,
        "window": window.to_dict(),
        "through_close": str(dataset.timestamps[-1]) + "Z",
        "replay_start_index": start,
        "replay_stop_index": stop,
        "dataset_id": dataset.dataset_id,
        "code_commit": code_commit,
        "run_date": when.isoformat(),
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
        "label": outcome["label"],
        "result_file": final_name,
    }
    _write_json(output_dir / MANIFEST_NAME, manifest)
    full: dict[str, Any] = {**header, **outcome}
    _write_json(output_dir / final_name, full)
    return full


__all__ = [
    "CELL_DIR",
    "DECISION_NAME",
    "DISCLOSURES",
    "INTERIM_PREFIX",
    "MANIFEST_NAME",
    "MIN_REPLAY_BARS",
    "MODE_DECISION",
    "MODE_INTERIM",
    "ProbeResult",
    "run_stage",
]
