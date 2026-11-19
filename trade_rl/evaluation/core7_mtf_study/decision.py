"""Pure, data-free decision rules and Non-RL control diagnostics for Core7 MTF study."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from trade_rl.evaluation.core7_mtf_study import protocol

PASS_PARITY_ONLY = "PASS_PARITY_ONLY"
STOP_FAIL_NO_RESCUE = "STOP_FAIL_NO_RESCUE"
INVALID = "INVALID"
SYNTHETIC_ONLY_NOT_EVIDENCE = "SYNTHETIC_ONLY_NOT_EVIDENCE"

_CONTROL_BENCHMARKS: tuple[str, ...] = ("ANC_BIDIR_75_25", "ANC_REGIME_GATE_NONRL")


@dataclass(frozen=True, slots=True)
class CellSummary:
    """Summary metrics and validity checks for one ``(arm, scenario)`` replay cell."""

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
    execution_valid: bool
    causality_passed: bool
    ledger_reconciled: bool
    stop_reason: str | None
    invalid_reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "arm": self.arm,
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
            "stop_reason": self.stop_reason,
            "invalid_reasons": list(self.invalid_reasons),
            "cell_valid": cell_valid(self),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> CellSummary:
        stop_raw = value.get("stop_reason")
        return cls(
            arm=str(value["arm"]),
            scenario=str(value["scenario"]),
            total_return=float(value["total_return"]),
            annual_sharpe=float(value["annual_sharpe"]),
            max_drawdown=float(value["max_drawdown"]),
            total_cost=float(value["total_cost"]),
            funding_pnl=float(value["funding_pnl"]),
            turnover_total=float(value["turnover_total"]),
            complete=bool(value["complete"]),
            terminal_flat=bool(value["terminal_flat"]),
            execution_valid=bool(value["execution_valid"]),
            causality_passed=bool(value["causality_passed"]),
            ledger_reconciled=bool(value["ledger_reconciled"]),
            stop_reason=None if stop_raw is None else str(stop_raw),
            invalid_reasons=tuple(
                str(item) for item in value.get("invalid_reasons", ())
            ),
        )


def _cell_structurally_sound(cell: CellSummary) -> bool:
    numeric = (
        cell.total_return,
        cell.annual_sharpe,
        cell.max_drawdown,
        cell.total_cost,
        cell.funding_pnl,
        cell.turnover_total,
    )
    return (
        all(math.isfinite(v) for v in numeric)
        and cell.max_drawdown >= 0.0
        and cell.total_cost >= 0.0
        and cell.turnover_total >= 0.0
        and cell.complete
        and cell.terminal_flat
        and cell.execution_valid
        and cell.causality_passed
        and cell.ledger_reconciled
        and not cell.invalid_reasons
    )


def drawdown_breached(cell: CellSummary) -> bool:
    """True iff the cell reached the frozen hard drawdown stop (``drawdown_stop``).

    The engine flattens at ``risk_scale == 0``, i.e. at ``max_drawdown >= drawdown_stop``,
    so reaching it is a stop regardless of whether a termination reason was emitted.
    """
    return cell.max_drawdown >= protocol.risk_config().drawdown_stop


def _cell_stopped(cell: CellSummary) -> bool:
    return cell.stop_reason is not None or drawdown_breached(cell)


def cell_valid(cell: CellSummary) -> bool:
    """Return True iff the cell completed without errors or risk/economic stops."""
    return _cell_structurally_sound(cell) and not _cell_stopped(cell)


def arm_passes_parity(
    cells: Mapping[tuple[str, str], CellSummary],
    arm: str,
    *,
    scenarios: Sequence[str] = protocol.SCENARIOS,
) -> bool:
    """Return True iff every scenario cell for ``arm`` is present and valid."""
    return all(
        (arm, scenario) in cells and cell_valid(cells[(arm, scenario)])
        for scenario in scenarios
    )


def _rl_vs_control_diagnostics(
    cells: Mapping[tuple[str, str], CellSummary],
    *,
    scenarios: Sequence[str],
) -> dict[str, object]:
    """Compare RL arms against ``ANC_BIDIR_75_25`` and ``ANC_REGIME_GATE_NONRL``."""
    diagnostics: dict[str, object] = {}
    for rl_arm in protocol.RL_ARMS:
        by_ctrl: dict[str, object] = {}
        for ctrl_arm in _CONTROL_BENCHMARKS:
            per_scenario: dict[str, object] = {}
            for sc in scenarios:
                rl_cell = cells.get((rl_arm, sc))
                ctrl_cell = cells.get((ctrl_arm, sc))
                if rl_cell is None or ctrl_cell is None:
                    continue
                if not (
                    _cell_structurally_sound(rl_cell)
                    and _cell_structurally_sound(ctrl_cell)
                ):
                    continue
                entry: dict[str, object] = {
                    "return_delta": rl_cell.total_return - ctrl_cell.total_return,
                    "sharpe_delta": rl_cell.annual_sharpe - ctrl_cell.annual_sharpe,
                    "drawdown_delta": rl_cell.max_drawdown - ctrl_cell.max_drawdown,
                }
                if ctrl_arm == "ANC_REGIME_GATE_NONRL":
                    entry["nonrl_regime_gate_dominates_dd"] = (
                        ctrl_cell.max_drawdown < rl_cell.max_drawdown
                    )
                    entry["nonrl_regime_gate_dominates_sharpe"] = (
                        ctrl_cell.annual_sharpe >= rl_cell.annual_sharpe
                    )
                per_scenario[sc] = entry
            by_ctrl[ctrl_arm] = per_scenario
        diagnostics[rl_arm] = by_ctrl
    return diagnostics


def decide(
    cells: Mapping[tuple[str, str], CellSummary],
    *,
    scenarios: Sequence[str] = protocol.SCENARIOS,
    evidence_eligible: bool = True,
) -> dict[str, object]:
    """Evaluate all ``(arm, scenario)`` cells into a strict fail-closed study outcome.

    ``evidence_eligible=False`` (a URL-less synthetic run) can never report a parity
    pass or a stop as evidence: those outcomes become ``SYNTHETIC_ONLY_NOT_EVIDENCE``
    and the computed label is kept as ``underlying_outcome``. ``INVALID`` is kept.
    """
    scenario_tuple = tuple(scenarios)
    expected_pairs = {(arm, sc) for arm in protocol.ARMS for sc in scenario_tuple}

    missing_cells = sorted(
        f"{arm}__{sc}" for arm, sc in expected_pairs if (arm, sc) not in cells
    )
    unexpected_cells = sorted(
        f"{arm}__{sc}" for arm, sc in cells if (arm, sc) not in expected_pairs
    )
    invalid_cells = sorted(
        f"{arm}__{sc}"
        for (arm, sc), cell in cells.items()
        if not _cell_structurally_sound(cell)
    )
    stopped_cells = sorted(
        f"{arm}__{sc}"
        for (arm, sc), cell in cells.items()
        if _cell_structurally_sound(cell) and _cell_stopped(cell)
    )

    # PASS_PARITY_ONLY is only defined over the complete frozen scenario set.
    scenarios_complete = scenario_tuple == protocol.SCENARIOS

    if missing_cells or unexpected_cells or invalid_cells or not scenarios_complete:
        outcome = INVALID
    elif stopped_cells:
        outcome = STOP_FAIL_NO_RESCUE
    else:
        outcome = PASS_PARITY_ONLY

    underlying_outcome = outcome
    if not evidence_eligible and outcome != INVALID:
        outcome = SYNTHETIC_ONLY_NOT_EVIDENCE

    return {
        "outcome": outcome,
        "underlying_outcome": underlying_outcome,
        "evidence_eligible": evidence_eligible,
        "edge_claim": False,
        "production_eligible": False,
        "live_trading_authorized": False,
        "scenarios": list(scenario_tuple),
        "scenarios_complete": scenarios_complete,
        "arm_parity_pass": {
            arm: arm_passes_parity(cells, arm, scenarios=scenario_tuple)
            for arm in protocol.ARMS
        },
        "missing_cells": missing_cells,
        "unexpected_cells": unexpected_cells,
        "invalid_cells": invalid_cells,
        "stopped_cells": stopped_cells,
        "rl_vs_control_diagnostics": _rl_vs_control_diagnostics(
            cells, scenarios=scenario_tuple
        ),
    }


__all__ = [
    "INVALID",
    "PASS_PARITY_ONLY",
    "STOP_FAIL_NO_RESCUE",
    "SYNTHETIC_ONLY_NOT_EVIDENCE",
    "CellSummary",
    "arm_passes_parity",
    "cell_valid",
    "decide",
    "drawdown_breached",
]
