"""Separate, opt-in observations of actual native order-fill cash transitions."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Final

from trade_rl._validation import require_sha256
from trade_rl.simulation.orders.model import OrderEvent
from trade_rl.simulation.quantities import parse_quantity, project_quantity

NATIVE_FILL_CASH_SCHEMA: Final = "native_fill_cash_observation_v1"


@dataclass(frozen=True, slots=True)
class NativeFillCashRecord:
    """Observed BookState.execute_fill transition, before margin/settlement/carry.

    The cash debit uses the projection of the actual accepted exact quantity
    delta, not an estimate from the surrounding interval's fees. This is native
    accounting evidence, not independent verification of the execution model.
    """

    event: OrderEvent
    quantity_before: str
    quantity_after: str
    contract_multiplier: float
    cash_before: float
    cash_after: float
    cost_amount: float
    total_cost_before: float
    total_cost_after: float

    def __post_init__(self) -> None:
        if type(self.event) is not OrderEvent or self.event.event_type not in (
            "filled",
            "partial_fill",
        ):
            raise ValueError("cash evidence requires a native fill event")
        price = self.event.execution_price
        if price is None or not math.isfinite(price) or price <= 0.0:
            raise ValueError("cash evidence requires a positive execution price")
        if not all(
            math.isfinite(value)
            for value in (
                self.contract_multiplier,
                self.cash_before,
                self.cash_after,
                self.cost_amount,
                self.total_cost_before,
                self.total_cost_after,
            )
        ):
            raise ValueError("fill cash evidence must be finite")
        if (
            self.contract_multiplier <= 0.0
            or self.cost_amount < 0.0
            or self.total_cost_before < 0.0
        ):
            raise ValueError("invalid fill cash multiplier or cost")
        delta = parse_quantity(self.quantity_after) - parse_quantity(
            self.quantity_before
        )
        projected = project_quantity(delta)
        if projected == 0.0 or projected != self.event.filled_quantity:
            raise ValueError("fill event does not match accepted quantity delta")
        notional = self.signed_notional
        if not math.isfinite(notional) or notional == 0.0:
            raise ValueError("fill cash notional is not representable")
        if abs(notional) != self.event.filled_notional:
            raise ValueError("fill cash notional differs from native fill event")
        if self.cash_after != self.cash_before - (notional + self.cost_amount):
            raise ValueError("observed fill cash transition does not reconcile")
        if self.total_cost_after != self.total_cost_before + self.cost_amount:
            raise ValueError("observed fill cost transition does not reconcile")

    @property
    def signed_notional(self) -> float:
        """The accepted quantity's signed principal using native float arithmetic."""
        price = self.event.execution_price
        assert price is not None
        return (
            project_quantity(
                parse_quantity(self.quantity_after)
                - parse_quantity(self.quantity_before)
            )
            * price
            * self.contract_multiplier
        )


@dataclass(frozen=True, slots=True)
class NativeFillCashObservation:
    """One successful invocation's order fills in native event sequence.

    Sequence numbers are invocation-local. Empty fills mean no native order
    fills occurred in this invocation, not that no other cash flow occurred.
    Non-order delisting, carry, margin termination/flattening and other cash
    movements are outside this boundary. No legacy event or ledger is modified.
    """

    dataset_id: str
    execution_policy_digest: str
    start_index: int
    next_index: int
    fills: tuple[NativeFillCashRecord, ...]
    interval_cost: float
    schema_version: str = field(default=NATIVE_FILL_CASH_SCHEMA, init=False)

    def __post_init__(self) -> None:
        require_sha256(self.dataset_id, field="dataset_id")
        require_sha256(self.execution_policy_digest, field="execution_policy_digest")
        if (
            type(self.start_index) is not int
            or type(self.next_index) is not int
            or not 0 <= self.start_index < self.next_index
            or type(self.fills) is not tuple
        ):
            raise ValueError("invalid native fill observation interval")
        last_sequence = -1
        last_processing = self.start_index
        cost = 0.0
        for row in self.fills:
            if type(row) is not NativeFillCashRecord:
                raise ValueError("invalid native fill observation row")
            event = row.event
            if (
                event.dataset_id != self.dataset_id
                or event.execution_policy_digest != self.execution_policy_digest
                or event.sequence <= last_sequence
                or not self.start_index < event.processing_index <= self.next_index
                or event.processing_index < last_processing
            ):
                raise ValueError("native fill observation identity or order mismatch")
            last_sequence = event.sequence
            last_processing = event.processing_index
            # Match the actual runtime accumulation order, not a compensated sum.
            cost += row.cost_amount
        if not math.isfinite(cost) or cost != self.interval_cost:
            raise ValueError("native fill costs do not match observed interval cost")


__all__ = [
    "NATIVE_FILL_CASH_SCHEMA",
    "NativeFillCashRecord",
    "NativeFillCashObservation",
]
