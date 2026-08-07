"""Two-layer canonical parity for execution runtimes.

Fill structure and economic closure are compared independently. This prevents a
runtime from receiving an exact-parity PASS merely because its order timeline
matches while fee, funding, PnL, or terminal-account semantics differ.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Iterable


@dataclass(frozen=True, slots=True)
class CanonicalFillSignature:
    sequence: int
    timestamp_ns: int
    price_ticks: int
    quantity_lots: int
    position_lots: int


@dataclass(frozen=True, slots=True)
class CanonicalEconomicClosure:
    fee_minor: int
    funding_minor: int
    realized_pnl_minor: int
    final_equity_minor: int
    terminal_position_lots: int
    terminal_open_orders: int


@dataclass(frozen=True, slots=True)
class DualShadowParityReport:
    fill_parity: bool
    economic_parity: bool
    exact_parity: bool
    mismatches: tuple[str, ...]


def compare_dual_shadow_execution(
    *,
    legacy_fills: Iterable[CanonicalFillSignature],
    candidate_fills: Iterable[CanonicalFillSignature],
    legacy_economics: CanonicalEconomicClosure,
    candidate_economics: CanonicalEconomicClosure,
) -> DualShadowParityReport:
    legacy_fill_values = tuple(legacy_fills)
    candidate_fill_values = tuple(candidate_fills)
    mismatches: list[str] = []

    comparable = min(len(legacy_fill_values), len(candidate_fill_values))
    fill_fields = tuple(field.name for field in fields(CanonicalFillSignature))
    for index in range(comparable):
        legacy_fill = legacy_fill_values[index]
        candidate_fill = candidate_fill_values[index]
        for name in fill_fields:
            if getattr(legacy_fill, name) != getattr(candidate_fill, name):
                mismatches.append(f"fills[{index}].{name}")
    if len(legacy_fill_values) != len(candidate_fill_values):
        mismatches.append("fills.length")

    fill_mismatch_count = len(mismatches)
    economic_fields = tuple(field.name for field in fields(CanonicalEconomicClosure))
    for name in economic_fields:
        if getattr(legacy_economics, name) != getattr(candidate_economics, name):
            mismatches.append(f"economics.{name}")

    fill_parity = fill_mismatch_count == 0
    economic_parity = len(mismatches) == fill_mismatch_count
    return DualShadowParityReport(
        fill_parity=fill_parity,
        economic_parity=economic_parity,
        exact_parity=fill_parity and economic_parity,
        mismatches=tuple(mismatches),
    )


__all__ = [
    "CanonicalEconomicClosure",
    "CanonicalFillSignature",
    "DualShadowParityReport",
    "compare_dual_shadow_execution",
]
