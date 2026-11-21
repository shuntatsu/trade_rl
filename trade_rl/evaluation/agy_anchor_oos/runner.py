"""Runner of ``agy_anchor_realistic_v1`` through ``run_shared_portfolio_replay`` only.

Arms: the ported anchor (``AGY_ANCHOR``, stride 18, lab total-turnover deadband) and the two
controls ``CASH`` and ``EW_HOLD_SPOT`` (lowturn policies, unchanged), every one on a cash
account. Scenarios: BASE, HIGH_COST and DELAY_1BAR with the Core7 MTF costs. The ledger
reconciliation, the cell summary and the future-perturbation helpers are the Core7 MTF
study's and the trend harness's (imported, not copied); the summary is called with
``execution_cost=`` so lean-core terminal dust is recognised.

The causality probe runs per ``(arm, signal lag)``: the future market rows *and* the
taker-buy volume after a cut are perturbed; every signal row up to the cut and every decision
must be identical.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable, Mapping
from dataclasses import replace
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import numpy as np

from trade_rl.artifacts import canonical_json_bytes
from trade_rl.data.market import MarketDataset
from trade_rl.evaluation.agy_anchor_oos import decision, protocol
from trade_rl.evaluation.agy_anchor_oos.cash import AnchorCashAccountPolicy
from trade_rl.evaluation.agy_anchor_oos.data import (
    StageData,
    StageWindow,
    build_anchor_inputs,
    validate_stage_dataset,
)
from trade_rl.evaluation.agy_anchor_oos.decision import CellKey, OosCell
from trade_rl.evaluation.lowturn_screen import protocol as lowturn_protocol
from trade_rl.evaluation.lowturn_screen.policies import LowturnPolicy
from trade_rl.evaluation.lowturn_screen.signals import LowturnSignals
from trade_rl.evaluation.replay import (
    SharedCashPortfolioReplayResult,
    run_shared_portfolio_replay,
)
from trade_rl.evaluation.trend_oos import runner as trend_runner
from trade_rl.evaluation.trend_oos.data import StageDataError
from trade_rl.evaluation.trend_oos.signals import build_oos_signals
from trade_rl.risk.pretrade import PreTradeRisk
from trade_rl.strategies.agy_anchor.policy import AgyAnchorPolicy
from trade_rl.strategies.agy_anchor.signals import AnchorSignals, build_anchor_signals
from trade_rl.strategies.portfolio import PortfolioPolicy

_summarize_replay_cell = trend_runner._summarize_replay_cell
_perturb_future_market_rows = trend_runner._perturb_future_market_rows
_build_synthetic_prefix_snapshots = trend_runner._build_synthetic_prefix_snapshots
_decision_diff = trend_runner._decision_diff
_run_decisions = trend_runner._run_decisions
_HybridSignals = trend_runner._HybridSignals
_prefix_signal_diff = trend_runner._prefix_signal_diff
_calendar_year_returns = trend_runner._calendar_year_returns
_dust_adjusted = trend_runner._dust_adjusted
_cash_extremes = trend_runner._cash_extremes
ProbeResult = trend_runner.ProbeResult

_PROBE_SEED_CUT = 17
_PROBE_SEED_FUTURE = 18
CELL_DIR = "cells"
DECISION_NAME = "decision.json"
DEV_NAME = "dev-result.json"
MANIFEST_NAME = "agy-anchor-oos-manifest.json"
INTERIM_PREFIX = "INTERIM_NOT_A_DECISION"
MODE_DECISION = "decision"
MODE_INTERIM = "interim"
#: An interim replay needs at least two anchor rebalance periods.
MIN_REPLAY_BARS = 2 * protocol.REBALANCE_EVERY_BARS
#: The controls keep lowturn's weekly rebalance (only the statistics use it here).
CONTROL_REBALANCE_EVERY_BARS = lowturn_protocol.REBALANCE_EVERY_BARS

ANCHOR_ROW_FIELDS: tuple[str, ...] = (
    "w_aw_bidir_75_25",
    "c_weekly_iv",
    "vol_cushion_tail",
    "w_aw_regime",
    "w_perp_bidir_carry",
    "carry_mask",
    "perp_mask",
    "spot_valid",
    "perp_valid",
)

DISCLOSURES: tuple[str, ...] = (
    "every arm trades a cash account: after the signal's own caps the target is scaled by "
    "the tighter of (a) spot long gross plus 0.4 (the engine's 1 / max_leverage) times perp "
    "gross <= 1.0 and (b) net signed exposure <= 0.97 (Amendment 4; the engine books "
    "perps through cash like spot); a cell whose cash net of costs falls below -1e-6 x capital is INVALID "
    "(negative_cash_borrowing)",
    "the causality probe is per (arm, signal lag) and perturbs market rows and taker-buy "
    "volume; it is shared by the cost scenarios",
    "exchange rules (lot size, minimum notional) are one static snapshot, not point-in-time",
    "an unlisted market carries a constant sentinel price, never a market price; a pair is "
    "available to the signal 180 bars after both markets are listed",
    "the lab's all-ones cross-sectional demeans include unavailable pairs as zero columns",
    "funding events are stamped at the close of their 4h bar and the count-based funding "
    "means start cold at the first row",
    "the last dataset row is two bars after the last decision bar; DEV ends at the Tier D "
    "last row 2026-07-31T20:00Z",
)


def _anchor_prefix_diff(a: AnchorSignals, b: AnchorSignals, *, cut_index: int) -> float:
    """Largest absolute difference over rows ``[0, cut_index]``; NaN patterns must match."""
    sl = slice(0, cut_index + 1)
    worst = 0.0
    for name in ANCHOR_ROW_FIELDS:
        arr_a = np.asarray(getattr(a, name)[sl], dtype=np.float64)
        arr_b = np.asarray(getattr(b, name)[sl], dtype=np.float64)
        if not np.array_equal(np.isnan(arr_a), np.isnan(arr_b)):
            return math.inf
        both = ~np.isnan(arr_a)
        if both.any():
            worst = max(worst, float(np.max(np.abs(arr_a[both] - arr_b[both]))))
    return worst


class _AnchorHybrid:
    """Anchor signals that are *base* up to the current decision row, *perturbed* after."""

    def __init__(self, base: AnchorSignals, future: AnchorSignals) -> None:
        self._base = base
        self._arrays = {
            name: np.array(getattr(future, name), copy=True)
            for name in ANCHOR_ROW_FIELDS
        }
        self._restored = -1
        arrays: dict[str, Any] = dict(self._arrays)
        self.signals = replace(base, **arrays)

    def restore_through(self, index: int) -> None:
        lo = self._restored + 1
        if index < lo:
            return
        for name, arr in self._arrays.items():
            arr[lo : index + 1] = getattr(self._base, name)[lo : index + 1]
        self._restored = index


def perturb_future_stage(stage: StageData, *, cut_index: int, seed: int) -> StageData:
    """The stage with every market row *and taker-buy value* after ``cut_index`` perturbed."""
    dataset = _perturb_future_market_rows(stage.dataset, cut_index=cut_index, seed=seed)
    rng = np.random.default_rng(seed + 1_000)
    taker = np.array(stage.taker_buy_quote_volume, copy=True)
    sl = slice(cut_index + 1, dataset.n_bars)
    volume = np.asarray(dataset.volume, dtype=np.float64)[sl]
    taker[sl] = volume * rng.uniform(0.0, 1.0, size=volume.shape)
    return stage.with_dataset(dataset, taker)


def anchor_signals_of(stage: StageData) -> AnchorSignals:
    return build_anchor_signals(build_anchor_inputs(stage))


def make_policy(
    arm: str,
    *,
    anchor: AnchorSignals,
    control: LowturnSignals,
    start_index: int,
    lag: int,
) -> PortfolioPolicy:
    """The cash-account policy of an arm; ``lag`` is the arm's own signal lag."""
    if arm == protocol.CANDIDATE:
        return AnchorCashAccountPolicy(
            AgyAnchorPolicy(
                signals=anchor,
                start_index=start_index,
                signal_lag_bars=lag,
                rebalance_every_bars=protocol.REBALANCE_EVERY_BARS,
                deadband=protocol.DEADBAND,
            ),
            start_index=start_index,
        )
    # the controls are spot-only lowturn policies (weekly rebalance), on the same cash account
    return AnchorCashAccountPolicy(
        LowturnPolicy(
            family=arm, signals=control, start_index=start_index, signal_lag_bars=lag
        ),
        start_index=start_index,
        rebalance_every_bars=CONTROL_REBALANCE_EVERY_BARS,
    )


def _arm_lags(arm: str) -> list[int]:
    return sorted({protocol.arm_signal_lag_bars(arm, s) for s in protocol.SCENARIOS})


def _probe_stage(
    stage: StageData,
    anchor: AnchorSignals,
    control: LowturnSignals,
    *,
    stage_name: str,
    start_index: int,
    stop_index: int,
) -> dict[tuple[str, int], Any]:
    """Probe every ``(arm, lag)`` over the decisions of the replay window."""
    dataset = stage.dataset
    last_decision = stop_index - 2
    cut = start_index + (last_decision - start_index) // 2
    cut_stage = perturb_future_stage(stage, cut_index=cut, seed=_PROBE_SEED_CUT)
    future_stage = perturb_future_stage(
        stage, cut_index=start_index, seed=_PROBE_SEED_FUTURE
    )
    anchor_cut = anchor_signals_of(cut_stage)
    anchor_future = anchor_signals_of(future_stage)
    control_cut = build_oos_signals(cut_stage.dataset)
    control_future = build_oos_signals(future_stage.dataset)
    anchor_diff = max(
        _anchor_prefix_diff(anchor, anchor_cut, cut_index=cut),
        _anchor_prefix_diff(anchor, anchor_future, cut_index=start_index),
    )
    control_diff = max(
        _prefix_signal_diff(control, control_cut, cut_index=cut),
        _prefix_signal_diff(control, control_future, cut_index=start_index),
    )
    snapshots = _build_synthetic_prefix_snapshots(
        dataset, start_index=start_index, stop_inclusive=last_decision
    )
    n_cut = sum(1 for snap in snapshots if snap.index <= cut)
    out: dict[tuple[str, int], Any] = {}
    for arm in protocol.ARMS:
        signal_diff = anchor_diff if arm == protocol.CANDIDATE else control_diff
        for lag in _arm_lags(arm):

            def make(
                a: AnchorSignals, c: LowturnSignals, arm: str = arm, lag: int = lag
            ) -> PortfolioPolicy:
                return make_policy(
                    arm, anchor=a, control=c, start_index=start_index, lag=lag
                )

            base = _run_decisions(make(anchor, control), snapshots)
            at_cut = _run_decisions(make(anchor_cut, control_cut), snapshots[:n_cut])
            ahybrid = _AnchorHybrid(anchor, anchor_future)
            chybrid = _HybridSignals(control, control_future)

            def restore(
                index: int, ah: _AnchorHybrid = ahybrid, ch: Any = chybrid
            ) -> None:
                ah.restore_through(index)
                ch.restore_through(index)

            every = _run_decisions(
                make(ahybrid.signals, chybrid.signals), snapshots, before_step=restore
            )
            prop_diff = max(
                _decision_diff(base[:n_cut], at_cut), _decision_diff(base, every)
            )
            out[(arm, lag)] = ProbeResult(
                family=arm,
                segment=stage_name,
                signal_lag_bars=lag,
                cut_index=cut,
                signal_max_abs_diff=signal_diff,
                proposal_max_abs_diff=prop_diff,
                decisions_probed=len(snapshots),
            )
    return out


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
    stop_reason = summary.stop_reason
    if (
        stop_reason == "max_drawdown_hard_stop"
        and summary.max_drawdown < protocol.RISK_DRAWDOWN_STOP
    ):
        # the shared core7 summary reports a hard stop at its own 20%; this protocol's
        # stop is the amended 25% (reporting only, a stop never invalidates a cell)
        stop_reason = None
    raw_returns = np.asarray(result.returns.values, dtype=np.float64)
    returns = _dust_adjusted(raw_returns, result, summary.terminal_dust)
    reasons = list(summary.invalid_reasons)
    min_cash, max_borrowing = _cash_extremes(result)
    tolerance = protocol.CASH_TOLERANCE_FRACTION * protocol.INITIAL_CAPITAL
    if max_borrowing is not None and max_borrowing > tolerance:
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
        stopped=stop_reason is not None,
        stop_reason=stop_reason,
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
    backcast_verdict: str | None = None,
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
    label; an interim report carries no label; a FORWARD decision needs the BACKCAST_2020
    stage verdict to combine the label.
    """
    if mode not in (MODE_DECISION, MODE_INTERIM):
        raise ValueError(f"unknown mode: {mode!r}")
    if mode == MODE_INTERIM and window.stage != protocol.STAGE_FORWARD:
        raise ValueError("only FORWARD has an interim mode")
    if expected_protocol_digest != protocol.protocol_digest():
        raise ValueError(
            "expected protocol digest does not match the frozen protocol digest "
            f"{protocol.protocol_digest()}"
        )
    if (
        window.stage == protocol.STAGE_FORWARD
        and mode == MODE_DECISION
        and backcast_verdict not in decision.VERDICTS
    ):
        raise ValueError("a FORWARD decision needs the BACKCAST_2020 stage verdict")
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
        raise StageDataError("the replay window is too short for two rebalance periods")
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

    anchor = anchor_signals_of(stage)
    control = build_oos_signals(dataset)
    risk = PreTradeRisk(protocol.risk_config())
    probes = _probe_stage(
        stage,
        anchor,
        control,
        stage_name=window.stage,
        start_index=start,
        stop_index=stop,
    )
    cells: dict[CellKey, OosCell] = {}
    for arm in protocol.ARMS:
        for scenario in protocol.SCENARIOS:
            lag = protocol.arm_signal_lag_bars(arm, scenario)
            policy = make_policy(
                arm, anchor=anchor, control=control, start_index=start, lag=lag
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
                cap_statistics=_stats_of(policy),
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
    else:
        outcome = decision.decide_stage(cells, stage=window.stage)
        verdict = outcome["stage_verdict"]
        if window.stage == protocol.STAGE_FORWARD:
            assert backcast_verdict is not None
            outcome["label"] = decision.combine_label(backcast_verdict, verdict)
            outcome["backcast_stage_verdict"] = backcast_verdict
        else:
            # BACKCAST_2020 is decided when it runs; the combined label needs FORWARD
            outcome["label"] = (
                protocol.LABEL_INVALID if verdict == decision.VERDICT_INVALID else None
            )
    header: dict[str, Any] = {
        "protocol_id": protocol.PROTOCOL_ID,
        "protocol_digest": protocol.protocol_digest(),
        "port_source_digest": protocol.PORT_SOURCE_DIGEST,
        "parity_fixture_sha256": protocol.FIXTURE_SHA256,
        "stage": window.stage,
        "mode": mode,
        "window": window.to_dict(),
        "through_close": str(dataset.timestamps[-1]) + "Z",
        "replay_start_index": start,
        "replay_stop_index": stop,
        "dataset_id": dataset.dataset_id,
        "harness_source_digest": protocol.harness_source_digest(),
        "taker_buy_quote_volume_sha256": hashlib.sha256(
            np.ascontiguousarray(stage.taker_buy_quote_volume).tobytes()
        ).hexdigest(),
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
        "label": outcome.get("label"),
        "stage_verdict": outcome.get("stage_verdict"),
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
    "ProbeResult",
    "anchor_signals_of",
    "make_policy",
    "perturb_future_stage",
    "run_stage",
]
