"""Validate the complete fixed family/seed/symbol/scenario evidence matrix."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from trade_rl.evaluation.rl_family_comparison.contract import (
    fixed_comparison_contract,
)


def _finite(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be a finite number")
    return result


def _required_text(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value


def validate_family_rows(
    cells: object,
    *,
    symbols: tuple[str, ...],
) -> dict[tuple[str, int, str, str], dict[str, Any]]:
    """Check every preregistered cell and return normalized evidence rows."""

    if not isinstance(cells, Sequence) or isinstance(cells, (str, bytes)):
        raise ValueError("family replay cells must be a sequence of evidence rows")
    contract = fixed_comparison_contract()
    families = tuple(contract["families"])
    seeds = tuple(contract["seeds"])
    scenarios = tuple(contract["evaluation"]["scenarios"])
    years = tuple(contract["evaluation"]["years"])
    expected = {
        (family, seed, symbol, scenario)
        for family in families
        for seed in seeds
        for symbol in symbols
        for scenario in scenarios
    }
    rows: dict[tuple[str, int, str, str], dict[str, Any]] = {}
    for cell in cells:
        if not isinstance(cell, Mapping):
            raise ValueError("family replay cell must be an evidence mapping")
        family = _required_text(cell.get("family"), field="family")
        seed = cell.get("seed")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("seed must be a preregistered integer")
        symbol = _required_text(cell.get("symbol"), field="symbol")
        scenario = _required_text(cell.get("scenario"), field="scenario")
        key = (family, seed, symbol, scenario)
        if key not in expected:
            raise ValueError("family replay cell is outside the preregistered roster")
        if key in rows:
            raise ValueError("duplicate family seed-symbol-scenario cell")

        interval_count = cell.get("interval_count")
        if (
            type(interval_count) is not int
            or interval_count != contract["evaluation"]["interval_count"]
        ):
            raise ValueError("family replay cell has incomplete clock coverage")
        total_return = _finite(cell.get("total_return"), field="total return")
        if total_return <= -1.0:
            raise ValueError("total return must be greater than -1")
        raw_years = cell.get("year_returns")
        if not isinstance(raw_years, Mapping) or set(raw_years) != set(years):
            raise ValueError("family replay cell has incomplete year coverage")
        year_returns = {
            year: _finite(raw_years[year], field=f"{year} return") for year in years
        }
        if any(value <= -1.0 for value in year_returns.values()):
            raise ValueError("year return must be greater than -1")
        drawdown = _finite(cell.get("ledger_max_drawdown"), field="ledger drawdown")
        if not 0.0 <= drawdown <= 1.0:
            raise ValueError("ledger drawdown must be within [0, 1]")
        termination_reasons = cell.get("termination_reasons")
        if not isinstance(termination_reasons, (list, tuple)) or any(
            not isinstance(reason, str) for reason in termination_reasons
        ):
            raise ValueError("termination evidence is malformed")
        terminal_flat = cell.get("terminal_flat")
        if not isinstance(terminal_flat, bool):
            raise ValueError("terminal-flat evidence is malformed")
        active_remainders = cell.get("active_order_remainders")
        if not isinstance(active_remainders, (list, tuple)):
            raise ValueError("active-order remainder evidence is malformed")
        rows[key] = {
            "total_return": total_return,
            "year_returns": year_returns,
            "ledger_max_drawdown": drawdown,
            "termination_reasons": tuple(termination_reasons),
            "terminal_flat": terminal_flat,
            "active_order_remainders": tuple(active_remainders),
        }
    if set(rows) != expected:
        raise ValueError("family replay matrix is incomplete")
    return rows
