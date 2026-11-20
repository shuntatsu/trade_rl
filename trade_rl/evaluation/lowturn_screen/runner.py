"""Runner of the low-turnover family screen through ``run_shared_portfolio_replay`` only.

The ledger reconciliation, the cell summary and the future-perturbation helpers are the
Core7 MTF study's (imported, not copied). The causality probe is the same kind of test:
every decision of every ``(family, signal lag)`` policy must be identical when all market
rows after the decision are perturbed.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from trade_rl.artifacts import canonical_json_bytes
from trade_rl.data.features.mtf6_signals import (
    enforce_pre_fs1_boundary,
    validate_alternating_spot_perp_dataset,
)
from trade_rl.data.market import MarketDataset
from trade_rl.evaluation.core7_mtf_study import runner as core7_runner
from trade_rl.evaluation.lowturn_screen import protocol
from trade_rl.evaluation.lowturn_screen.decision import CellKey, ScreenCell, decide
from trade_rl.evaluation.lowturn_screen.policies import LowturnPolicy
from trade_rl.evaluation.lowturn_screen.signals import (
    ROW_FIELDS,
    LowturnSignals,
    build_signals,
)
from trade_rl.evaluation.replay import (
    SharedCashPortfolioReplayResult,
    run_shared_portfolio_replay,
)
from trade_rl.risk.pretrade import PreTradeRisk

# Shared with the Core7 MTF runner (single implementation of each check).
_summarize_replay_cell = core7_runner._summarize_replay_cell
_perturb_future_market_rows = core7_runner._perturb_future_market_rows
_build_synthetic_prefix_snapshots = core7_runner._build_synthetic_prefix_snapshots
_decision_diff = core7_runner._decision_diff
_run_decisions = core7_runner._run_decisions

_PROBE_SEED_CUT = 17
_PROBE_SEED_FUTURE = 18
CELL_DIR = "cells"
DECISION_NAME = "decision.json"
MANIFEST_NAME = "lowturn-manifest.json"


@dataclass(frozen=True, slots=True)
class ProbeResult:
    """Future-perturbation causality probe of one ``(family, segment, lag)``."""

    family: str
    segment: str
    signal_lag_bars: int
    cut_index: int
    signal_max_abs_diff: float
    proposal_max_abs_diff: float
    decisions_probed: int

    @property
    def passed(self) -> bool:
        return self.signal_max_abs_diff == 0.0 and self.proposal_max_abs_diff == 0.0

    def to_dict(self) -> dict[str, object]:
        return {
            "family": self.family,
            "segment": self.segment,
            "signal_lag_bars": self.signal_lag_bars,
            "cut_index": self.cut_index,
            "signal_max_abs_diff": _json_safe(self.signal_max_abs_diff),
            "proposal_max_abs_diff": _json_safe(self.proposal_max_abs_diff),
            "decisions_probed": self.decisions_probed,
            "passed": self.passed,
        }


def _json_safe(value: float) -> float:
    return value if math.isfinite(value) else 1.0e308


def _prefix_signal_diff(
    a: LowturnSignals, b: LowturnSignals, *, cut_index: int
) -> float:
    """Largest absolute difference over rows ``[0, cut_index]``; NaN patterns must match."""
    sl = slice(0, cut_index + 1)
    worst = 0.0
    for name in ROW_FIELDS:
        arr_a = np.asarray(getattr(a, name)[sl], dtype=np.float64)
        arr_b = np.asarray(getattr(b, name)[sl], dtype=np.float64)
        if not np.array_equal(np.isnan(arr_a), np.isnan(arr_b)):
            return math.inf
        both = ~np.isnan(arr_a)
        if both.any():
            worst = max(worst, float(np.max(np.abs(arr_a[both] - arr_b[both]))))
    return worst


class _HybridSignals:
    """Signals that are *base* up to the current decision row and *perturbed* after."""

    def __init__(self, base: LowturnSignals, future: LowturnSignals) -> None:
        self._base = base
        self._arrays = {
            name: np.array(getattr(future, name), copy=True) for name in ROW_FIELDS
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


def _probe_segment(
    dataset: MarketDataset,
    signals: LowturnSignals,
    *,
    segment: str,
    start_index: int,
    stop_index: int,
) -> dict[tuple[str, int], ProbeResult]:
    """Probe every ``(family, lag)`` over the decisions of one segment."""
    last_decision = stop_index - 2  # agent_stop_index = stop_index - 1 with settlement
    cut = start_index + (last_decision - start_index) // 2
    pack_cut = build_signals(
        _perturb_future_market_rows(dataset, cut_index=cut, seed=_PROBE_SEED_CUT)
    )
    pack_future = build_signals(
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
    out: dict[tuple[str, int], ProbeResult] = {}
    lags = sorted({protocol.scenario_signal_lag_bars(s) for s in protocol.SCENARIOS})
    for family in protocol.FAMILIES:
        for lag in lags:

            def make(
                pack: LowturnSignals, family: str = family, lag: int = lag
            ) -> LowturnPolicy:
                return LowturnPolicy(
                    family=family,
                    signals=pack,
                    start_index=start_index,
                    signal_lag_bars=lag,
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
            out[(family, lag)] = ProbeResult(
                family=family,
                segment=segment,
                signal_lag_bars=lag,
                cut_index=cut,
                signal_max_abs_diff=signal_diff,
                proposal_max_abs_diff=prop_diff,
                decisions_probed=len(snapshots),
            )
    return out


def _moments(returns: np.ndarray) -> tuple[float, float]:
    """Population skewness and raw (non-excess) kurtosis; ``(0, 3)`` when degenerate."""
    if returns.size < 3:
        return 0.0, 3.0
    centered = returns - float(np.mean(returns))
    variance = float(np.mean(centered * centered))
    if variance <= 1e-24 or not math.isfinite(variance):
        return 0.0, 3.0
    sd = math.sqrt(variance)
    return (
        float(np.mean(centered**3) / sd**3),
        float(np.mean(centered**4) / variance**2),
    )


def _calendar_year_returns(
    dataset: MarketDataset,
    result: SharedCashPortfolioReplayResult,
    returns: np.ndarray,
) -> dict[str, float] | None:
    """Compounded per-interval returns by the calendar year of each interval's end row."""
    ledger = result.ledger_evidence
    if ledger is None:
        return None
    intervals = ledger.intervals.intervals
    if len(intervals) != returns.size:
        return None
    years = dataset.timestamps.astype("datetime64[Y]").astype(int) + 1970
    growth: dict[int, float] = {}
    for interval, ret in zip(intervals, returns, strict=True):
        year = int(years[interval.next_index])
        growth[year] = growth.get(year, 1.0) * (1.0 + float(ret))
    return {str(year): value - 1.0 for year, value in sorted(growth.items())}


def _build_cell(
    *,
    family: str,
    segment: str,
    scenario: str,
    result: SharedCashPortfolioReplayResult,
    dataset: MarketDataset,
    causality_passed: bool,
) -> ScreenCell:
    summary = _summarize_replay_cell(
        arm=family,
        scenario=scenario,
        result=result,
        dataset=dataset,
        initial_capital=protocol.INITIAL_CAPITAL,
        causality_passed=causality_passed,
    )
    returns = np.asarray(result.returns.values, dtype=np.float64)
    reasons = list(summary.invalid_reasons)
    years = _calendar_year_returns(dataset, result, returns)
    if years is None:
        reasons.append("returns_not_aligned_with_ledger_intervals")
        years = {}
    skewness, kurtosis = _moments(returns)
    return ScreenCell(
        family=family,
        segment=segment,
        scenario=scenario,
        total_return=summary.total_return,
        annual_sharpe=summary.annual_sharpe,
        max_drawdown=summary.max_drawdown,
        total_cost=summary.total_cost,
        funding_pnl=summary.funding_pnl,
        turnover_total=summary.turnover_total,
        complete=summary.complete,
        terminal_flat=summary.terminal_flat,
        execution_valid=summary.execution_valid,
        causality_passed=summary.causality_passed,
        ledger_reconciled=summary.ledger_reconciled,
        stopped=summary.stop_reason is not None,
        stop_reason=summary.stop_reason,
        invalid_reasons=tuple(reasons),
        n_returns=int(returns.size),
        return_skewness=skewness,
        return_kurtosis=kurtosis,
        calendar_year_returns=years,
    )


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(dict(payload)))


def run_screen(
    dataset: MarketDataset,
    *,
    output_dir: Path,
    expected_protocol_digest: str,
    expected_dataset_id: str,
    code_commit: str | None = None,
    authorization: Mapping[str, Any] | None = None,
    on_cell: Callable[[ScreenCell], None] | None = None,
) -> dict[str, Any]:
    """Run every family x {DEV, HOLDOUT} x {BASE, HIGH_COST, DELAY_1BAR} cell and decide.

    Fails closed unless the protocol digest and the dataset id are exactly the expected
    ones, the dataset is the complete 11-pair alternating roster and nothing reaches the
    Pre-FS1 embargo. Every cell, the manifest and ``decision.json`` are written to
    ``output_dir`` (which must not hold a previous run) before the decision is returned.
    """
    if expected_protocol_digest != protocol.protocol_digest():
        raise ValueError(
            "expected protocol digest does not match the frozen protocol digest "
            f"{protocol.protocol_digest()}"
        )
    if dataset.dataset_id != expected_dataset_id:
        raise ValueError(
            "dataset id does not match the expected dataset id: "
            f"{dataset.dataset_id!r} != {expected_dataset_id!r}"
        )
    if (output_dir / DECISION_NAME).exists() or (output_dir / MANIFEST_NAME).exists():
        raise FileExistsError(f"refusing to overwrite an existing run in {output_dir}")
    pair_names = validate_alternating_spot_perp_dataset(dataset)
    if pair_names != protocol.ROSTER:
        raise ValueError(
            f"dataset pair roster {pair_names!r} does not match the 11-pair ROSTER"
        )
    enforce_pre_fs1_boundary(dataset)
    bounds = protocol.segment_bounds(dataset.timestamps)

    signals = build_signals(dataset)
    risk = PreTradeRisk(protocol.risk_config())
    cells: dict[CellKey, ScreenCell] = {}
    probes: dict[str, dict[str, object]] = {}
    for segment in protocol.SEGMENTS:
        start, stop = bounds[segment]
        segment_probes = _probe_segment(
            dataset, signals, segment=segment, start_index=start, stop_index=stop
        )
        for (family, lag), probe in segment_probes.items():
            probes[f"{family}__{segment}__lag{lag}"] = probe.to_dict()
        for family in protocol.FAMILIES:
            for scenario in protocol.SCENARIOS:
                lag = protocol.scenario_signal_lag_bars(scenario)
                policy = LowturnPolicy(
                    family=family,
                    signals=signals,
                    start_index=start,
                    signal_lag_bars=lag,
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
                    family=family,
                    segment=segment,
                    scenario=scenario,
                    result=result,
                    dataset=dataset,
                    causality_passed=segment_probes[(family, lag)].passed,
                )
                cells[cell.key] = cell
                _write_json(
                    output_dir / CELL_DIR / f"{family}__{segment}__{scenario}.json",
                    cell.to_dict(),
                )
                if on_cell is not None:
                    on_cell(cell)

    decision = decide(cells)
    header: dict[str, Any] = {
        "protocol_id": protocol.PROTOCOL_ID,
        "protocol_digest": protocol.protocol_digest(),
        "dataset_id": dataset.dataset_id,
        "code_commit": code_commit,
        "segments": {name: list(span) for name, span in bounds.items()},
    }
    manifest: dict[str, Any] = {
        **header,
        "expected_protocol_digest": expected_protocol_digest,
        "expected_dataset_id": expected_dataset_id,
        "pair_names": list(pair_names),
        "authorization": None if authorization is None else dict(authorization),
        "families": list(protocol.FAMILIES),
        "scenarios": list(protocol.SCENARIOS),
        "cell_files": sorted(
            f"{CELL_DIR}/{f}__{s}__{sc}.json"
            for f in protocol.FAMILIES
            for s in protocol.SEGMENTS
            for sc in protocol.SCENARIOS
        ),
        "causality_probes": probes,
        "label": decision["label"],
    }
    full_decision: dict[str, Any] = {**header, **decision}
    _write_json(output_dir / MANIFEST_NAME, manifest)
    _write_json(output_dir / DECISION_NAME, full_decision)
    return full_decision


def causality_probe_segment(
    dataset: MarketDataset, *, start_index: int, stop_index: int, segment: str = "DEV"
) -> Sequence[ProbeResult]:
    """Probe every ``(family, lag)`` over one replay window (used by tests)."""
    signals = build_signals(dataset)
    return list(
        _probe_segment(
            dataset,
            signals,
            segment=segment,
            start_index=start_index,
            stop_index=stop_index,
        ).values()
    )


__all__ = [
    "CELL_DIR",
    "DECISION_NAME",
    "MANIFEST_NAME",
    "ProbeResult",
    "causality_probe_segment",
    "run_screen",
]
