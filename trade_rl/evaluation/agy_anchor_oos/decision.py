"""Pure, data-free decision rule and reported statistics of ``agy_anchor_realistic_v1``.

``SUPPORTED(S)`` of a stage holds iff all nine cells (3 arms x 3 scenarios) are valid, no
candidate cell reached a 25% drawdown, the candidate's BASE return is above 0 and its BASE
annual Sharpe is at least 0.50, and its HIGH_COST and DELAY_1BAR returns are above 0. A
stage verdict is ``STAGE_SUPPORTED``, ``STAGE_NOT_SUPPORTED`` (all cells valid, a condition
fails) or ``INVALID`` (any cell invalid). The label is combined over BACKCAST_2020 and
FORWARD by :func:`combine_label`: ``ANCHOR_SUPPORTED``, ``ANCHOR_FORWARD_ONLY``,
``ANCHOR_NOT_SUPPORTED`` or ``INVALID`` (any invalid cell of either stage takes precedence).
The DEV stage is a diagnostic and never gates. The stationary block bootstrap lower bound of
the mean BASE excess return over CASH is reported and never gates.

The cell record is the out-of-sample harness's :class:`OosCell` (imported, not copied).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from trade_rl.evaluation.agy_anchor_oos import protocol
from trade_rl.evaluation.trend_oos.decision import OosCell, stationary_bootstrap

CellKey = tuple[str, str]  # (arm, scenario)
VERDICT_SUPPORTED = protocol.STAGE_SUPPORTED
VERDICT_NOT_SUPPORTED = protocol.STAGE_NOT_SUPPORTED
VERDICT_INVALID = protocol.LABEL_INVALID
VERDICTS = (VERDICT_SUPPORTED, VERDICT_NOT_SUPPORTED, VERDICT_INVALID)
DEV_STATUS = "DEV_DIAGNOSTIC_NOT_A_DECISION"


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
    """The five conditions of ``SUPPORTED(S)``, in the spec's order."""
    _require_grid(cells)
    cand = {s: cells[(protocol.CANDIDATE, s)] for s in protocol.SCENARIOS}
    base = cand[protocol.BASE_SCENARIO]
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
    }


def stage_verdict(cells: Mapping[CellKey, OosCell]) -> str:
    """``INVALID`` whenever any cell is invalid, else supported iff every condition holds."""
    conds = conditions(cells)
    if not conds["1_all_nine_cells_valid"]:
        return VERDICT_INVALID
    return VERDICT_SUPPORTED if all(conds.values()) else VERDICT_NOT_SUPPORTED


def combine_label(backcast: str, forward: str) -> str:
    """The label of the two stage verdicts (the spec's table; INVALID takes precedence)."""
    for name, verdict in (("backcast", backcast), ("forward", forward)):
        if verdict not in VERDICTS:
            raise ValueError(f"unknown {name} verdict: {verdict!r}")
    if VERDICT_INVALID in (backcast, forward):
        return protocol.LABEL_INVALID
    if forward == VERDICT_SUPPORTED:
        return (
            protocol.LABEL_SUPPORTED
            if backcast == VERDICT_SUPPORTED
            else protocol.LABEL_FORWARD_ONLY
        )
    return protocol.LABEL_NOT_SUPPORTED


def bootstrap_excess(cells: Mapping[CellKey, OosCell]) -> dict[str, object]:
    """Lower bound of the mean BASE excess return of the candidate over CASH (reported)."""
    _require_grid(cells)
    cand = np.asarray(
        cells[(protocol.CANDIDATE, protocol.BASE_SCENARIO)].returns, dtype=np.float64
    )
    ctrl = np.asarray(
        cells[(protocol.CONTROL_CASH, protocol.BASE_SCENARIO)].returns,
        dtype=np.float64,
    )
    key = f"excess_over_{protocol.CONTROL_CASH}"
    if cand.size != ctrl.size or cand.size < 2:
        return {key: {"available": False, "reason": "return series are not aligned"}}
    try:
        stats = stationary_bootstrap(
            cand - ctrl,
            block_mean=protocol.BOOTSTRAP_BLOCK_BARS,
            resamples=protocol.BOOTSTRAP_RESAMPLES,
            seed=protocol.BOOTSTRAP_SEED,
            confidence=protocol.BOOTSTRAP_CONFIDENCE,
            chunk=protocol.BOOTSTRAP_CHUNK,
        )
    except ValueError as error:
        return {key: {"available": False, "reason": str(error)}}
    return {key: {"available": True, **stats}}


def _reported(cells: Mapping[CellKey, OosCell]) -> dict[str, Any]:
    ordered = sorted(cells.items())
    return {
        "calendar_year_returns": {
            "__".join(key): dict(cell.calendar_year_returns) for key, cell in ordered
        },
        "bootstrap": bootstrap_excess(cells),
        "cap_statistics": {
            "__".join(key): dict(cell.cap_statistics) for key, cell in ordered
        },
        "cells": {"__".join(key): cell.to_dict() for key, cell in ordered},
    }


def _invalid_cells(cells: Mapping[CellKey, OosCell]) -> list[str]:
    return sorted("__".join(key) for key, cell in cells.items() if not cell.valid)


def decide_stage(cells: Mapping[CellKey, OosCell], *, stage: str) -> dict[str, Any]:
    """Apply the frozen rule to the complete cells of BACKCAST_2020 or FORWARD."""
    if stage not in (protocol.STAGE_BACKCAST, protocol.STAGE_FORWARD):
        raise ValueError(
            "only BACKCAST_2020 and FORWARD are decided; DEV is diagnostic"
        )
    conds = conditions(cells)
    verdict = stage_verdict(cells)
    return {
        "stage_verdict": verdict,
        "supported": verdict == VERDICT_SUPPORTED,
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
    "VERDICTS",
    "VERDICT_INVALID",
    "VERDICT_NOT_SUPPORTED",
    "VERDICT_SUPPORTED",
    "CellKey",
    "OosCell",
    "bootstrap_excess",
    "combine_label",
    "conditions",
    "decide_stage",
    "dev_report",
    "expected_grid",
    "interim_report",
    "stage_verdict",
]
