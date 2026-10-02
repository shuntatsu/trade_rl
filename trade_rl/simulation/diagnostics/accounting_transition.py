"""Ordered state evidence for shared-cash accounting transitions."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from trade_rl.simulation.accounting import BookState


@dataclass(frozen=True, slots=True)
class AccountingStateSnapshot:
    """Cash, exact inventory, marks, and multipliers at one mutation boundary."""

    cash: float
    exact_quantities: tuple[str, ...]
    mark_prices: tuple[float, ...]
    contract_multipliers: tuple[float, ...]

    @classmethod
    def capture(cls, book: BookState) -> AccountingStateSnapshot:
        multipliers = book.contract_multipliers
        if multipliers is None:
            raise RuntimeError("accounting evidence requires contract multipliers")
        return cls(
            cash=float(book.cash),
            exact_quantities=tuple(str(value) for value in book.exact_quantities),
            mark_prices=tuple(float(value) for value in book.mark_prices),
            contract_multipliers=tuple(float(value) for value in multipliers),
        )

    def to_mapping(self) -> dict[str, object]:
        return {
            "cash": self.cash,
            "contract_multipliers": self.contract_multipliers,
            "exact_quantities": self.exact_quantities,
            "mark_prices": self.mark_prices,
        }


@dataclass(frozen=True, slots=True)
class AccountingTransitionEvidence:
    """A typed mutation with detached before/after account snapshots."""

    sequence: int
    processing_index: int
    transition_type: str
    state_before: AccountingStateSnapshot
    state_after: AccountingStateSnapshot
    evidence: Mapping[str, object]
    order_event_sequence: int | None = None

    def to_mapping(self) -> dict[str, object]:
        return {
            "evidence": dict(self.evidence),
            "order_event_sequence": self.order_event_sequence,
            "processing_index": self.processing_index,
            "sequence": self.sequence,
            "state_after": self.state_after.to_mapping(),
            "state_before": self.state_before.to_mapping(),
            "transition_type": self.transition_type,
        }


__all__ = ["AccountingStateSnapshot", "AccountingTransitionEvidence"]
