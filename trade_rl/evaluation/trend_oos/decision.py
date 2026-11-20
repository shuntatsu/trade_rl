"""Pure, data-free decision rule and reported statistics of ``trend_long_spot_oos_v1``.

``SUPPORTED(S)`` of a stage holds iff all nine cells (3 arms x 3 scenarios) are valid,
no candidate cell reached a 20% drawdown, the candidate's BASE return is above 0 and its
BASE annual Sharpe is at least 0.30, its HIGH_COST and DELAY_1BAR returns are above 0,
and its BASE Sharpe is above ``EW_HOLD_SPOT``'s. The labels are ``OOS_SUPPORTED``,
``OOS_NOT_SUPPORTED`` (all cells valid, a condition fails) and ``INVALID`` (any cell
invalid). Per-year returns and the one-sided stationary block bootstrap lower bounds of
the mean BASE excess return are reported and never gate.

A cell is valid when its ledger reconciles, its causality probe passed, it is not
insolvent, and its terminal book is flat **or** only lean-core terminal dust (a residual
the engine rejected for ``below_minimum_notional``, valued at mark minus the modelled
exit cost). A 20% drawdown stop is not an integrity failure; it fails condition 2 only.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from trade_rl.evaluation.trend_oos import protocol

CellKey = tuple[str, str]  # (arm, scenario)


def _safe(value: float) -> float | None:
    return value if math.isfinite(value) else None


@dataclass(frozen=True, slots=True)
class OosCell:
    """Summary of one ``(arm, scenario)`` replay cell of a stage."""

    arm: str
    scenario: str
    total_return: float
    annual_sharpe: float
    max_drawdown: float
    total_cost: float
    funding_pnl: float
    turnover_total: float
    complete: bool
    terminal_flat: bool
    terminal_dust_only: bool
    execution_valid: bool
    causality_passed: bool
    ledger_reconciled: bool
    stopped: bool
    stop_reason: str | None
    invalid_reasons: tuple[str, ...] = ()
    terminal_dust: tuple[Mapping[str, Any], ...] = ()
    n_returns: int = 0
    calendar_year_returns: Mapping[str, float] = field(default_factory=dict)
    returns: tuple[float, ...] = ()

    @property
    def valid(self) -> bool:
        numeric = (
            self.total_return,
            self.annual_sharpe,
            self.max_drawdown,
            self.total_cost,
            self.funding_pnl,
            self.turnover_total,
        )
        return bool(
            all(math.isfinite(v) for v in numeric)
            and self.max_drawdown >= 0.0
            and self.total_cost >= 0.0
            and self.turnover_total >= 0.0
            and self.complete
            and (self.terminal_flat or self.terminal_dust_only)
            and self.execution_valid
            and self.causality_passed
            and self.ledger_reconciled
            and not self.invalid_reasons
        )

    @property
    def key(self) -> CellKey:
        return (self.arm, self.scenario)

    def to_dict(self, *, include_returns: bool = False) -> dict[str, object]:
        out: dict[str, object] = {
            "arm": self.arm,
            "scenario": self.scenario,
            "total_return": _safe(self.total_return),
            "annual_sharpe": _safe(self.annual_sharpe),
            "max_drawdown": _safe(self.max_drawdown),
            "total_cost": _safe(self.total_cost),
            "funding_pnl": _safe(self.funding_pnl),
            "turnover_total": _safe(self.turnover_total),
            "complete": self.complete,
            "terminal_flat": self.terminal_flat,
            "terminal_dust_only": self.terminal_dust_only,
            "terminal_dust": [dict(item) for item in self.terminal_dust],
            "execution_valid": self.execution_valid,
            "causality_passed": self.causality_passed,
            "ledger_reconciled": self.ledger_reconciled,
            "stopped": self.stopped,
            "stop_reason": self.stop_reason,
            "invalid_reasons": list(self.invalid_reasons),
            "n_returns": self.n_returns,
            "calendar_year_returns": dict(self.calendar_year_returns),
            "cell_valid": self.valid,
        }
        if include_returns:
            out["returns"] = list(self.returns)
        return out


def expected_grid() -> set[CellKey]:
    return {(a, s) for a in protocol.ARMS for s in protocol.SCENARIOS}


def _require_grid(cells: Mapping[CellKey, OosCell]) -> None:
    expected = expected_grid()
    if set(cells) != expected:
        missing = sorted(expected - set(cells))
        extra = sorted(set(cells) - expected)
        raise ValueError(
            f"cells must be exactly the frozen grid (missing={missing}, extra={extra})"
        )
    for key, cell in cells.items():
        if cell.key != key:
            raise ValueError(f"cell key {key!r} does not match its content")


def conditions(cells: Mapping[CellKey, OosCell]) -> dict[str, bool]:
    """The six conditions of ``SUPPORTED(S)``, in the spec's order."""
    _require_grid(cells)
    cand = {s: cells[(protocol.CANDIDATE, s)] for s in protocol.SCENARIOS}
    base = cand[protocol.BASE_SCENARIO]
    ew_base = cells[(protocol.CONTROL_EW_SPOT, protocol.BASE_SCENARIO)]
    return {
        "1_all_nine_cells_valid": all(c.valid for c in cells.values()),
        "2_no_candidate_cell_drawdown_at_least_limit": all(
            c.max_drawdown < protocol.MAX_DRAWDOWN_LIMIT for c in cand.values()
        ),
        "3_candidate_base_return_above_zero": base.total_return > 0.0,
        "4_candidate_base_sharpe_at_least_min": (
            base.annual_sharpe >= protocol.MIN_BASE_SHARPE
        ),
        "5_candidate_high_cost_and_delay_returns_above_zero": (
            cand[protocol.HIGH_COST_SCENARIO].total_return > 0.0
            and cand[protocol.DELAY_SCENARIO].total_return > 0.0
        ),
        "6_candidate_base_sharpe_above_ew_hold_spot": (
            base.annual_sharpe > ew_base.annual_sharpe
        ),
    }


def label_of(cells: Mapping[CellKey, OosCell]) -> str:
    """The stage label; ``INVALID`` whenever any cell is invalid."""
    conds = conditions(cells)
    if not conds["1_all_nine_cells_valid"]:
        return protocol.LABEL_INVALID
    return (
        protocol.LABEL_SUPPORTED
        if all(conds.values())
        else protocol.LABEL_NOT_SUPPORTED
    )


def stationary_bootstrap(
    series: np.ndarray,
    *,
    block_mean: int = protocol.BOOTSTRAP_BLOCK_BARS,
    resamples: int = protocol.BOOTSTRAP_RESAMPLES,
    seed: int = protocol.BOOTSTRAP_SEED,
    confidence: float = protocol.BOOTSTRAP_CONFIDENCE,
    chunk: int = protocol.BOOTSTRAP_CHUNK,
) -> dict[str, object]:
    """One-sided lower bound of the mean of ``series`` by the stationary bootstrap.

    Politis-Romano: every resample step starts a new block (a uniformly random start)
    with probability ``1 / block_mean`` and otherwise continues the previous block one
    step, wrapping circularly. The bound is the ``1 - confidence`` quantile of the
    resampled means. Resamples are drawn in fixed chunks from one generator seeded with
    ``seed``, so the result is deterministic.
    """
    x = np.asarray(series, dtype=np.float64)
    n = int(x.size)
    if n < 2 or not np.all(np.isfinite(x)):
        raise ValueError("the bootstrap needs at least two finite observations")
    if block_mean < 1 or resamples < 1 or chunk < 1 or not 0.0 < confidence < 1.0:
        raise ValueError("invalid bootstrap parameters")
    rng = np.random.default_rng(seed)
    p_new = 1.0 / block_mean
    positions = np.arange(n)
    means = np.empty(resamples, dtype=np.float64)
    done = 0
    while done < resamples:
        m = min(chunk, resamples - done)
        new = rng.random((m, n)) < p_new
        new[:, 0] = True
        starts = rng.integers(0, n, size=(m, n))
        last_new = np.maximum.accumulate(np.where(new, positions, 0), axis=1)
        base = np.take_along_axis(starts, last_new, axis=1)
        index = (base + (positions - last_new)) % n
        means[done : done + m] = x[index].mean(axis=1)
        done += m
    return {
        "n_bars": n,
        "mean": float(x.mean()),
        "lower_bound": float(np.quantile(means, 1.0 - confidence)),
        "block_mean_bars": block_mean,
        "resamples": resamples,
        "seed": seed,
        "one_sided_confidence": confidence,
        "gates": False,
    }


def bootstrap_excess(cells: Mapping[CellKey, OosCell]) -> dict[str, object]:
    """Lower bounds of the mean BASE excess return over CASH and over EW_HOLD_SPOT."""
    _require_grid(cells)
    cand = np.asarray(
        cells[(protocol.CANDIDATE, protocol.BASE_SCENARIO)].returns, dtype=np.float64
    )
    out: dict[str, object] = {}
    for control in protocol.CONTROLS:
        ctrl = np.asarray(
            cells[(control, protocol.BASE_SCENARIO)].returns, dtype=np.float64
        )
        key = f"excess_over_{control}"
        if cand.size != ctrl.size or cand.size < 2:
            out[key] = {"available": False, "reason": "return series are not aligned"}
            continue
        try:
            out[key] = {"available": True, **stationary_bootstrap(cand - ctrl)}
        except ValueError as error:
            out[key] = {"available": False, "reason": str(error)}
    return out


def _reported(cells: Mapping[CellKey, OosCell]) -> dict[str, Any]:
    ordered = sorted(cells.items())
    return {
        "calendar_year_returns": {
            "__".join(key): dict(cell.calendar_year_returns) for key, cell in ordered
        },
        "bootstrap": bootstrap_excess(cells),
        "cells": {"__".join(key): cell.to_dict() for key, cell in ordered},
    }


def decide(cells: Mapping[CellKey, OosCell]) -> dict[str, Any]:
    """Apply the frozen decision rule to the complete set of cells of a stage."""
    conds = conditions(cells)
    label = label_of(cells)
    return {
        "label": label,
        "claim_ceiling": protocol.CLAIM_BACKCAST,
        "supported": label == protocol.LABEL_SUPPORTED,
        "conditions": conds,
        "failed_conditions": [name for name, ok in conds.items() if not ok],
        "invalid_cells": sorted(
            "__".join(key) for key, cell in cells.items() if not cell.valid
        ),
        **_reported(cells),
    }


def interim_report(cells: Mapping[CellKey, OosCell]) -> dict[str, Any]:
    """The same metrics with no label and no condition verdict (never a decision)."""
    _require_grid(cells)
    return {
        "label": None,
        "status": protocol.LABEL_INTERIM,
        "invalid_cells": sorted(
            "__".join(key) for key, cell in cells.items() if not cell.valid
        ),
        **_reported(cells),
    }


def combined_claim(backcast_label: str, forward_label: str) -> str | None:
    """``OOS_AND_FORWARD_SUPPORTED`` only when both stages are ``OOS_SUPPORTED``."""
    if backcast_label == forward_label == protocol.LABEL_SUPPORTED:
        return protocol.CLAIM_BOTH
    return None


__all__ = [
    "CellKey",
    "OosCell",
    "bootstrap_excess",
    "combined_claim",
    "conditions",
    "decide",
    "expected_grid",
    "interim_report",
    "label_of",
    "stationary_bootstrap",
]
