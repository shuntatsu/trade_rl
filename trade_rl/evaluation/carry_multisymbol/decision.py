"""Pure, data-free decision rule of the multi-symbol carry protocol.

The functions here read only ``CellSummary`` values, so the rule is exercised on
hand-written summaries and cannot be tuned against a replay. They never raise on a
failing economic cell: a failure is a result, not an error.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from trade_rl.evaluation.carry_multisymbol import protocol

GENERALIZES = "CARRY_GENERALIZES_DEVELOPMENT"
DOES_NOT_GENERALIZE = "CARRY_DOES_NOT_GENERALIZE"
INCONCLUSIVE = "INCONCLUSIVE_UNIVERSE"


@dataclass(frozen=True, slots=True)
class CellSummary:
    arm: str
    treatment: str
    dataset_id: str
    total_return: float
    year_returns: Mapping[str, float]
    ledger_max_drawdown: float
    complete: bool
    terminal_flat: bool
    execution_valid: bool
    stop_reason: str | None
    invalid_reasons: tuple[str, ...]
    total_cost: float
    funding_pnl: float
    average_realized_gross: float

    @classmethod
    def from_replay(
        cls, arm: str, treatment: str, replay: Mapping[str, Any]
    ) -> CellSummary:
        gross = [float(value) for value in replay["realized_gross"]]
        return cls(
            arm=arm,
            treatment=treatment,
            dataset_id=str(replay["dataset_id"]),
            total_return=float(replay["total_return"]),
            year_returns={
                str(key): float(value) for key, value in replay["year_returns"].items()
            },
            ledger_max_drawdown=float(replay["ledger_max_drawdown"]),
            complete=bool(replay["complete"]),
            terminal_flat=bool(replay["terminal_flat"]),
            execution_valid=bool(replay["execution_valid"]),
            stop_reason=replay["stop_reason"],
            invalid_reasons=tuple(str(item) for item in replay["invalid_reasons"]),
            total_cost=float(replay["total_cost"]),
            funding_pnl=float(replay["funding_pnl"]),
            average_realized_gross=sum(gross) / len(gross) if gross else 0.0,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "arm": self.arm,
            "treatment": self.treatment,
            "dataset_id": self.dataset_id,
            "total_return": self.total_return,
            "year_returns": dict(self.year_returns),
            "ledger_max_drawdown": self.ledger_max_drawdown,
            "complete": self.complete,
            "terminal_flat": self.terminal_flat,
            "execution_valid": self.execution_valid,
            "stop_reason": self.stop_reason,
            "invalid_reasons": list(self.invalid_reasons),
            "total_cost": self.total_cost,
            "funding_pnl": self.funding_pnl,
            "average_realized_gross": self.average_realized_gross,
            "cell_pass": cell_passes(self),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> CellSummary:
        return cls(
            arm=str(value["arm"]),
            treatment=str(value["treatment"]),
            dataset_id=str(value["dataset_id"]),
            total_return=float(value["total_return"]),
            year_returns={str(k): float(v) for k, v in value["year_returns"].items()},
            ledger_max_drawdown=float(value["ledger_max_drawdown"]),
            complete=bool(value["complete"]),
            terminal_flat=bool(value["terminal_flat"]),
            execution_valid=bool(value["execution_valid"]),
            stop_reason=value["stop_reason"],
            invalid_reasons=tuple(str(item) for item in value["invalid_reasons"]),
            total_cost=float(value["total_cost"]),
            funding_pnl=float(value["funding_pnl"]),
            average_realized_gross=float(value["average_realized_gross"]),
        )


def cell_passes(cell: CellSummary) -> bool:
    """The per-cell rule, recomputed from the summary (never from a stored flag)."""
    finite = all(
        math.isfinite(value)
        for value in (
            cell.total_return,
            cell.ledger_max_drawdown,
            *cell.year_returns.values(),
        )
    )
    common = (
        finite
        and cell.complete
        and cell.terminal_flat
        and cell.execution_valid
        and cell.stop_reason is None
        and not cell.invalid_reasons
        and cell.ledger_max_drawdown < protocol.MAXIMUM_DRAWDOWN
        and cell.total_return > 0
    )
    if not common:
        return False
    if cell.treatment != "base":
        return True
    return set(cell.year_returns) == set(protocol.SUB_PERIODS) and all(
        cell.year_returns[period] > 0 for period in protocol.SUB_PERIODS
    )


def arm_passes(cells: Mapping[tuple[str, str], CellSummary], arm: str) -> bool:
    """An arm passes only when its base and all three stress cells pass."""
    return all(
        (arm, treatment.name) in cells and cell_passes(cells[(arm, treatment.name)])
        for treatment in protocol.TREATMENTS
    )


def binomial_tail(n: int, k: int, p: float) -> float:
    """P(X >= k) for X ~ Binomial(n, p); a diagnostic that assumes independence."""
    return sum(math.comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(k, n + 1))


def decide(
    admitted: Sequence[str],
    excluded: Mapping[str, str],
    cells: Mapping[tuple[str, str], CellSummary],
    *,
    reference_arms: Sequence[str] = (),
) -> dict[str, object]:
    """The family decision over exactly the admitted symbols; nothing is dropped."""
    admitted = tuple(admitted)
    count = len(admitted)
    base = {
        "admitted": list(admitted),
        "excluded": dict(sorted(excluded.items())),
        "min_admitted": protocol.MIN_ADMITTED,
        "edge_claim": False,
        "economic_scope": "exploratory known-used Tier D development evidence only",
        "production_eligible": False,
        "live_trading_authorized": False,
    }
    if count < protocol.MIN_ADMITTED:
        return base | {
            "outcome": INCONCLUSIVE,
            "reason": f"{count} admitted symbols is below the minimum {protocol.MIN_ADMITTED}",
        }
    required = protocol.required_symbol_passes(count)
    symbol_pass = {symbol: arm_passes(cells, symbol) for symbol in admitted}
    portfolio = arm_passes(cells, protocol.PORTFOLIO_ARM)
    passes = sum(symbol_pass.values())
    generalizes = portfolio and passes >= required
    return base | {
        "outcome": GENERALIZES if generalizes else DOES_NOT_GENERALIZE,
        "portfolio_pass": portfolio,
        "symbol_pass": symbol_pass,
        "symbol_passes": passes,
        "required_symbol_passes": required,
        "null_pass_probability": protocol.null_pass_probability(),
        "markov_bound_if_subperiods_independent": count
        * protocol.null_pass_probability()
        / required,
        "error_rate_controlled": False,
        "independent_binomial_tail_diagnostic": binomial_tail(
            count, required, protocol.null_pass_probability()
        ),
        "failed_cells": sorted(
            f"{arm}__{treatment}"
            for (arm, treatment), cell in cells.items()
            if (arm == protocol.PORTFOLIO_ARM or arm in admitted)
            and not cell_passes(cell)
        ),
        "reference_arm_pass": {arm: arm_passes(cells, arm) for arm in reference_arms},
        "decision_cell_count": (count + 1) * len(protocol.TREATMENTS),
    }


__all__ = [
    "DOES_NOT_GENERALIZE",
    "GENERALIZES",
    "INCONCLUSIVE",
    "CellSummary",
    "arm_passes",
    "binomial_tail",
    "cell_passes",
    "decide",
]
