"""Pure, data-free decision rule and reported statistics of ``xs_lowvol_funding_v1``.

A cell is INVALID when its ledger does not reconcile, its causality probe failed, it is
insolvent, its cash net of cumulative execution costs fell below -1e-6 x capital, a non-dust
residual remains after terminal settlement, or a row outside the stage window was read (the
last is enforced by the row guard before any cell exists). The DISJOINT label of the
CANDIDATE is ``XS_SUPPORTED`` iff C1-C6 hold, ``INVALID`` when C1 fails and
``XS_NOT_SUPPORTED`` otherwise; the FORWARD label needs a supported DISJOINT label. DEV is a
diagnostic and never gates. Everything else (per-year returns, turnover and cost drag,
funding PnL, beta and correlation to ``EW_LONG``, leg attribution, forced-close counts, the
deflated Sharpe with 40 prior trials) is reported and never gates.

The cell record is the out-of-sample harness's :class:`OosCell` (imported, not copied). The
runner stores the per-cell extras this rule reads under ``cap_statistics``: ``half_sharpes``
(``{"H1": ..., "H2": ...}``), ``policy`` and ``forced_close``.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import numpy as np

from trade_rl.evaluation.lowturn_screen.decision import deflated_sharpe_ratio
from trade_rl.evaluation.trend_oos.decision import OosCell, stationary_bootstrap
from trade_rl.evaluation.xs_lowvol_funding import protocol

CellKey = tuple[str, str]  # (arm, scenario)
DEV_STATUS = "DEV_DIAGNOSTIC_NOT_A_DECISION"
_EXTRA_HALVES = "half_sharpes"


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


def bootstrap_lower_bound(cells: Mapping[CellKey, OosCell]) -> dict[str, object]:
    """One-sided 90% lower bound of the mean BASE 4h net return of the candidate (C6)."""
    _require_grid(cells)
    returns = np.asarray(
        cells[(protocol.CANDIDATE, protocol.BASE_SCENARIO)].returns, dtype=np.float64
    )
    try:
        stats = stationary_bootstrap(
            returns,
            block_mean=protocol.BOOTSTRAP_BLOCK_BARS,
            resamples=protocol.BOOTSTRAP_RESAMPLES,
            seed=protocol.BOOTSTRAP_SEED,
            confidence=protocol.BOOTSTRAP_CONFIDENCE,
            chunk=protocol.BOOTSTRAP_CHUNK,
        )
    except ValueError as error:
        return {"available": False, "reason": str(error)}
    stats["gates"] = True
    return {"available": True, **stats}


def _half(cell: OosCell, name: str) -> float | None:
    halves = cell.cap_statistics.get(_EXTRA_HALVES)
    if not isinstance(halves, Mapping):
        return None
    value = halves.get(name)
    return float(value) if isinstance(value, int | float) else None


def conditions(
    cells: Mapping[CellKey, OosCell], *, bootstrap: Mapping[str, Any] | None = None
) -> dict[str, bool]:
    """C1-C6 of the DISJOINT label, in the spec's order."""
    _require_grid(cells)
    base = cells[(protocol.CANDIDATE, protocol.BASE_SCENARIO)]
    high = cells[(protocol.CANDIDATE, protocol.HIGH_COST_SCENARIO)]
    delay = cells[(protocol.CANDIDATE, protocol.DELAY_SCENARIO)]
    boot = bootstrap_lower_bound(cells) if bootstrap is None else dict(bootstrap)
    h1, h2 = _half(base, "H1"), _half(base, "H2")
    lower = boot.get("lower_bound")
    return {
        "C1_no_invalid_cell": all(c.valid for c in cells.values()),
        "C2_base_sharpe_at_least_min": base.annual_sharpe
        >= protocol.C2_MIN_BASE_SHARPE,
        "C3_high_cost_and_delay_sharpe": (
            high.annual_sharpe >= protocol.C3_MIN_HIGH_COST_SHARPE
            and delay.annual_sharpe >= protocol.C3_MIN_DELAY_SHARPE
        ),
        "C4_base_max_drawdown_below_limit": (
            base.max_drawdown < protocol.MAX_DRAWDOWN_LIMIT
        ),
        "C5_base_sharpe_positive_in_both_halves": (
            h1 is not None and h2 is not None and h1 > 0.0 and h2 > 0.0
        ),
        "C6_bootstrap_lower_bound_positive": (
            bool(boot.get("available")) and isinstance(lower, float) and lower > 0.0
        ),
    }


def label_of(conds: Mapping[str, bool]) -> str:
    """``INVALID`` whenever C1 fails, else supported iff every condition holds."""
    if not conds["C1_no_invalid_cell"]:
        return protocol.LABEL_INVALID
    return (
        protocol.LABEL_SUPPORTED
        if all(conds.values())
        else protocol.LABEL_NOT_SUPPORTED
    )


def _invalid_cells(cells: Mapping[CellKey, OosCell]) -> list[str]:
    return sorted("__".join(key) for key, cell in cells.items() if not cell.valid)


def _beta_and_correlation(cells: Mapping[CellKey, OosCell]) -> dict[str, object]:
    cand = np.asarray(
        cells[(protocol.CANDIDATE, protocol.BASE_SCENARIO)].returns, dtype=np.float64
    )
    ew = np.asarray(
        cells[(protocol.ARM_EW_LONG, protocol.BASE_SCENARIO)].returns, dtype=np.float64
    )
    if cand.size != ew.size or cand.size < 2:
        return {"available": False}
    var = float(np.var(ew, ddof=1))
    sd_c, sd_e = float(np.std(cand, ddof=1)), float(np.std(ew, ddof=1))
    if var <= 0.0 or sd_c <= 0.0 or not math.isfinite(var):
        return {"available": False}
    cov = float(np.cov(cand, ew, ddof=1)[0, 1])
    return {"available": True, "beta": cov / var, "correlation": cov / (sd_c * sd_e)}


def _skew_kurtosis(x: np.ndarray) -> tuple[float, float]:
    centred = x - x.mean()
    sd = float(np.sqrt(np.mean(centred**2)))
    if sd <= 0.0:
        return 0.0, 3.0
    return (
        float(np.mean(centred**3) / sd**3),
        float(np.mean(centred**4) / sd**4),
    )


def deflated_sharpe(cells: Mapping[CellKey, OosCell]) -> dict[str, object]:
    """Reported deflated Sharpe of the candidate's BASE cell with 40 declared prior trials.

    The per-period Sharpe variance across trials is the variance of the BASE per-period
    Sharpe ratios of this protocol's four non-cash arms (an interpretation: the prior trials'
    Sharpe ratios are not available).
    """
    base = {arm: cells[(arm, protocol.BASE_SCENARIO)] for arm in protocol.ARMS}
    per_period = np.asarray(
        [
            base[arm].annual_sharpe / math.sqrt(protocol.PERIODS_PER_YEAR)
            for arm in protocol.ARMS
            if arm != protocol.CONTROL_CASH
        ],
        dtype=np.float64,
    )
    variance = (
        float(np.var(per_period, ddof=1))
        if np.all(np.isfinite(per_period))
        else math.nan
    )
    cand = base[protocol.CANDIDATE]
    returns = np.asarray(cand.returns, dtype=np.float64)
    skew, kurt = _skew_kurtosis(returns) if returns.size >= 2 else (math.nan, math.nan)
    value = deflated_sharpe_ratio(
        sharpe=cand.annual_sharpe / math.sqrt(protocol.PERIODS_PER_YEAR),
        n_obs=cand.n_returns,
        skewness=skew,
        kurtosis=kurt,
        sharpe_variance=variance,
        n_trials=protocol.PRIOR_TRIALS,
    )
    return {
        "gates": False,
        "n_trials": protocol.PRIOR_TRIALS,
        "trial_sharpe_variance_per_period": variance,
        "candidate_base": value,
    }


def _reported(cells: Mapping[CellKey, OosCell]) -> dict[str, Any]:
    ordered = sorted(cells.items())
    capital = protocol.INITIAL_CAPITAL
    return {
        "calendar_year_returns": {
            "__".join(key): dict(cell.calendar_year_returns) for key, cell in ordered
        },
        "turnover_and_cost_drag": {
            "__".join(key): {
                "turnover_total": cell.turnover_total,
                "total_cost": cell.total_cost,
                "cost_drag_fraction_of_capital": cell.total_cost / capital,
            }
            for key, cell in ordered
        },
        "funding_pnl": {"__".join(key): cell.funding_pnl for key, cell in ordered},
        "beta_to_ew_long": _beta_and_correlation(cells),
        "leg_attribution": {
            "__".join(key): cell.cap_statistics.get("leg_attribution")
            for key, cell in ordered
        },
        "forced_close": {
            "__".join(key): cell.cap_statistics.get("forced_close")
            for key, cell in ordered
        },
        "deflated_sharpe": deflated_sharpe(cells),
        "policy_statistics": {
            "__".join(key): cell.cap_statistics.get("policy") for key, cell in ordered
        },
        "cells": {"__".join(key): cell.to_dict() for key, cell in ordered},
    }


def decide_disjoint(cells: Mapping[CellKey, OosCell]) -> dict[str, Any]:
    """Apply the frozen rule to the complete cells of DISJOINT."""
    boot = bootstrap_lower_bound(cells)
    conds = conditions(cells, bootstrap=boot)
    label = label_of(conds)
    return {
        "label": label,
        "stage_verdict": label,
        "supported": label == protocol.LABEL_SUPPORTED,
        "conditions": conds,
        "failed_conditions": [name for name, ok in conds.items() if not ok],
        "invalid_cells": _invalid_cells(cells),
        "bootstrap": boot,
        **_reported(cells),
    }


def forward_conditions(cells: Mapping[CellKey, OosCell]) -> dict[str, bool]:
    """The FORWARD conditions on the candidate's cells (BASE Sharpe, HIGH_COST, drawdown)."""
    _require_grid(cells)
    base = cells[(protocol.CANDIDATE, protocol.BASE_SCENARIO)]
    high = cells[(protocol.CANDIDATE, protocol.HIGH_COST_SCENARIO)]
    return {
        "F1_no_invalid_cell": all(c.valid for c in cells.values()),
        "F2_base_sharpe_positive": base.annual_sharpe > 0.0,
        "F3_high_cost_return_positive": high.total_return > 0.0,
        "F4_base_max_drawdown_below_limit": (
            base.max_drawdown < protocol.MAX_DRAWDOWN_LIMIT
        ),
    }


def decide_forward(
    cells: Mapping[CellKey, OosCell], *, disjoint_label: str
) -> dict[str, Any]:
    """The FORWARD label at the final run (needs the DISJOINT label)."""
    if disjoint_label not in protocol.LABELS:
        raise ValueError(f"unknown DISJOINT label: {disjoint_label!r}")
    conds = forward_conditions(cells)
    supported = disjoint_label == protocol.LABEL_SUPPORTED and all(conds.values())
    invalid = not conds["F1_no_invalid_cell"]
    return {
        "label": (
            protocol.LABEL_INVALID
            if invalid
            else (
                protocol.LABEL_FORWARD_SUPPORTED
                if supported
                else protocol.LABEL_FORWARD_NOT_SUPPORTED
            )
        ),
        "disjoint_label": disjoint_label,
        "supported": supported and not invalid,
        "conditions": conds,
        "failed_conditions": [name for name, ok in conds.items() if not ok],
        "invalid_cells": _invalid_cells(cells),
        **_reported(cells),
    }


def dev_report(cells: Mapping[CellKey, OosCell]) -> dict[str, Any]:
    """DEV numbers: reported, never a label and never a gate."""
    _require_grid(cells)
    return {
        "label": None,
        "status": DEV_STATUS,
        "gates": False,
        "invalid_cells": _invalid_cells(cells),
        **_reported(cells),
    }


def interim_report(cells: Mapping[CellKey, OosCell]) -> dict[str, Any]:
    """The same metrics with no label and no condition verdict (never a decision)."""
    _require_grid(cells)
    return {
        "label": None,
        "status": protocol.LABEL_INTERIM,
        "invalid_cells": _invalid_cells(cells),
        **_reported(cells),
    }


__all__ = [
    "DEV_STATUS",
    "CellKey",
    "OosCell",
    "bootstrap_lower_bound",
    "conditions",
    "decide_disjoint",
    "decide_forward",
    "deflated_sharpe",
    "dev_report",
    "expected_grid",
    "forward_conditions",
    "interim_report",
    "label_of",
]
