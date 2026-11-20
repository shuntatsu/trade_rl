"""Pure, data-free decision rule of the low-turnover family screen.

An economic STOP (the 20% drawdown hard stop) is not an integrity failure: a stopped cell
is *valid* when its ledger reconciles, its causality probe passed, the book is flat after
settlement and it is not insolvent. It still fails the ``no stop`` gate of ``DEV_PASS``
and ``HOLDOUT_CONFIRMED``.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from statistics import NormalDist
from typing import Any

import numpy as np

from trade_rl.evaluation.lowturn_screen import protocol

CellKey = tuple[str, str, str]  # (family, segment, scenario)

_EULER_MASCHERONI = 0.5772156649015329
_NORMAL = NormalDist()


@dataclass(frozen=True, slots=True)
class ScreenCell:
    """Summary of one ``(family, segment, scenario)`` replay cell."""

    family: str
    segment: str
    scenario: str
    total_return: float
    annual_sharpe: float
    max_drawdown: float
    total_cost: float
    funding_pnl: float
    turnover_total: float
    complete: bool
    terminal_flat: bool
    execution_valid: bool
    causality_passed: bool
    ledger_reconciled: bool
    stopped: bool
    stop_reason: str | None
    invalid_reasons: tuple[str, ...] = ()
    n_returns: int = 0
    return_skewness: float = 0.0
    return_kurtosis: float = 3.0
    calendar_year_returns: Mapping[str, float] = field(default_factory=dict)

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
            and self.terminal_flat
            and self.execution_valid
            and self.causality_passed
            and self.ledger_reconciled
            and not self.invalid_reasons
        )

    @property
    def key(self) -> CellKey:
        return (self.family, self.segment, self.scenario)

    def to_dict(self) -> dict[str, object]:
        return {
            "family": self.family,
            "segment": self.segment,
            "scenario": self.scenario,
            "total_return": self.total_return,
            "annual_sharpe": self.annual_sharpe,
            "max_drawdown": self.max_drawdown,
            "total_cost": self.total_cost,
            "funding_pnl": self.funding_pnl,
            "turnover_total": self.turnover_total,
            "complete": self.complete,
            "terminal_flat": self.terminal_flat,
            "execution_valid": self.execution_valid,
            "causality_passed": self.causality_passed,
            "ledger_reconciled": self.ledger_reconciled,
            "stopped": self.stopped,
            "stop_reason": self.stop_reason,
            "invalid_reasons": list(self.invalid_reasons),
            "n_returns": self.n_returns,
            "return_skewness": self.return_skewness,
            "return_kurtosis": self.return_kurtosis,
            "calendar_year_returns": dict(self.calendar_year_returns),
            "cell_valid": self.valid,
        }


def deflated_sharpe_ratio(
    *,
    sharpe: float,
    n_obs: int,
    skewness: float,
    kurtosis: float,
    sharpe_variance: float,
    n_trials: int,
) -> float | None:
    """Deflated Sharpe ratio of Bailey and Lopez de Prado (2014).

    ``sharpe`` is the *per-period* (non-annualised) Sharpe ratio, ``kurtosis`` the raw
    (non-excess) kurtosis and ``sharpe_variance`` the variance of the per-period Sharpe
    ratios across the ``n_trials`` trials. Returns ``None`` when it is undefined (too few
    observations or a non-positive variance term).
    """
    if n_trials < 2 or n_obs < 2:
        return None
    inputs = (sharpe, skewness, kurtosis, sharpe_variance)
    if not all(math.isfinite(v) for v in inputs) or sharpe_variance < 0.0:
        return None
    benchmark = math.sqrt(sharpe_variance) * (
        (1.0 - _EULER_MASCHERONI) * _NORMAL.inv_cdf(1.0 - 1.0 / n_trials)
        + _EULER_MASCHERONI * _NORMAL.inv_cdf(1.0 - 1.0 / (n_trials * math.e))
    )
    radicand = 1.0 - skewness * sharpe + (kurtosis - 1.0) / 4.0 * sharpe * sharpe
    if radicand <= 0.0:
        return None
    statistic = (sharpe - benchmark) * math.sqrt(n_obs - 1.0) / math.sqrt(radicand)
    return float(_NORMAL.cdf(statistic))


def _per_period_sharpe(cell: ScreenCell) -> float:
    return cell.annual_sharpe / math.sqrt(protocol.PERIODS_PER_YEAR)


def _deflated_sharpes(cells: Mapping[CellKey, ScreenCell]) -> dict[str, Any]:
    trials = [cells[(f, "DEV", "BASE")] for f in protocol.CANDIDATE_FAMILIES]
    sharpes = np.asarray([_per_period_sharpe(c) for c in trials], dtype=np.float64)
    variance = (
        float(np.var(sharpes, ddof=1)) if np.all(np.isfinite(sharpes)) else math.nan
    )
    per_family: dict[str, float | None] = {}
    for cell, sr in zip(trials, sharpes, strict=True):
        per_family[cell.family] = deflated_sharpe_ratio(
            sharpe=float(sr),
            n_obs=cell.n_returns,
            skewness=cell.return_skewness,
            kurtosis=cell.return_kurtosis,
            sharpe_variance=variance,
            n_trials=protocol.DSR_TRIALS,
        )
    return {
        "gates": False,
        "n_trials": protocol.DSR_TRIALS,
        "trial_sharpe_variance_per_period": variance,
        "dev_base": per_family,
    }


def _dev_pass(
    cells: Mapping[CellKey, ScreenCell], family: str
) -> tuple[bool, dict[str, bool]]:
    dev = {s: cells[(family, "DEV", s)] for s in protocol.SCENARIOS}
    base = dev["BASE"]
    conditions = {
        "dev_cells_valid": all(c.valid for c in dev.values()),
        "dev_no_stop": not any(c.stopped for c in dev.values()),
        "dev_base_sharpe_at_least_min": base.annual_sharpe >= protocol.DEV_MIN_SHARPE,
        "dev_base_sharpe_above_controls": all(
            base.annual_sharpe > cells[(c, "DEV", "BASE")].annual_sharpe
            for c in protocol.SHARPE_BENCHMARK_CONTROLS
        ),
        "dev_high_cost_return_positive": dev["HIGH_COST"].total_return > 0.0,
        "dev_delay_return_positive": dev["DELAY_1BAR"].total_return > 0.0,
    }
    return all(conditions.values()), conditions


def _holdout_confirmed(
    cells: Mapping[CellKey, ScreenCell], family: str
) -> tuple[bool, dict[str, bool]]:
    hold = {s: cells[(family, "HOLDOUT", s)] for s in protocol.SCENARIOS}
    conditions = {
        "holdout_cells_valid": all(c.valid for c in hold.values()),
        "holdout_no_stop": not any(c.stopped for c in hold.values()),
        "holdout_base_sharpe_at_least_min": hold["BASE"].annual_sharpe
        >= protocol.HOLDOUT_MIN_SHARPE,
        "holdout_high_cost_return_positive": hold["HIGH_COST"].total_return > 0.0,
    }
    return all(conditions.values()), conditions


def decide(cells: Mapping[CellKey, ScreenCell]) -> dict[str, Any]:
    """Apply the frozen decision rule to the complete set of cells."""
    expected = {
        (f, seg, sc)
        for f in protocol.FAMILIES
        for seg in protocol.SEGMENTS
        for sc in protocol.SCENARIOS
    }
    if set(cells) != expected:
        missing = sorted(expected - set(cells))
        extra = sorted(set(cells) - expected)
        raise ValueError(
            f"cells must be exactly the frozen grid (missing={missing}, extra={extra})"
        )
    for key, cell in cells.items():
        if cell.key != key:
            raise ValueError(f"cell key {key!r} does not match its content")

    invalid_controls = sorted(
        "__".join(key)
        for key, cell in cells.items()
        if key[0] in protocol.CONTROL_FAMILIES and not cell.valid
    )
    families: dict[str, dict[str, Any]] = {}
    confirmed: list[str] = []
    dev_passed: list[str] = []
    for family in protocol.CANDIDATE_FAMILIES:
        dev_ok, dev_conditions = _dev_pass(cells, family)
        entry: dict[str, Any] = {
            "dev_pass": dev_ok,
            "dev_conditions": dev_conditions,
            "holdout_confirmed": None,
            "holdout_conditions": None,
        }
        if dev_ok:
            dev_passed.append(family)
            hold_ok, hold_conditions = _holdout_confirmed(cells, family)
            entry["holdout_confirmed"] = hold_ok
            entry["holdout_conditions"] = hold_conditions
            if hold_ok:
                confirmed.append(family)
        families[family] = entry

    if invalid_controls:
        label = protocol.LABEL_INVALID
    elif confirmed:
        label = protocol.LABEL_EXPLORATORY_CANDIDATE
    else:
        label = protocol.LABEL_NO_CANDIDATE

    ordered = sorted(cells.items())
    return {
        "label": label,
        "claim_ceiling": protocol.CLAIM_CEILING,
        "invalid_control_cells": invalid_controls,
        "dev_pass_families": dev_passed,
        "confirmed_families": confirmed,
        "families": families,
        "deflated_sharpe": _deflated_sharpes(cells),
        "calendar_year_returns": {
            "__".join(key): dict(cell.calendar_year_returns) for key, cell in ordered
        },
        "cells": {"__".join(key): cell.to_dict() for key, cell in ordered},
    }


__all__ = [
    "CellKey",
    "ScreenCell",
    "decide",
    "deflated_sharpe_ratio",
]
