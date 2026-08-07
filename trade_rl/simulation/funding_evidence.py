"""Immutable funding-boundary evidence emitted by stateful execution."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class FundingBoundaryEvidence:
    """Economic inputs and equity closure for one actual funding boundary."""

    processing_index: int
    timestamp_ns: int
    funding_due: tuple[bool, ...]
    signed_quantities: tuple[float, ...]
    mark_prices: tuple[float, ...]
    contract_multipliers: tuple[float, ...]
    funding_rates: tuple[float, ...]
    funding_amount: float
    equity_before_funding: float
    equity_after_funding: float

    def __post_init__(self) -> None:
        if (
            isinstance(self.processing_index, bool)
            or not isinstance(self.processing_index, int)
            or self.processing_index < 0
        ):
            raise ValueError("processing_index must be a non-negative integer")
        if isinstance(self.timestamp_ns, bool) or not isinstance(
            self.timestamp_ns, int
        ):
            raise ValueError("timestamp_ns must be an integer")

        size = len(self.funding_due)
        if size == 0:
            raise ValueError(
                "funding boundary evidence must contain at least one symbol"
            )
        if not all(isinstance(value, bool) for value in self.funding_due):
            raise ValueError("funding_due values must be booleans")
        if not any(self.funding_due):
            raise ValueError(
                "funding boundary evidence requires at least one due symbol"
            )
        if any(
            len(values) != size
            for values in (
                self.signed_quantities,
                self.mark_prices,
                self.contract_multipliers,
                self.funding_rates,
            )
        ):
            raise ValueError("funding boundary evidence vectors must have equal length")
        if any(not math.isfinite(value) for value in self.signed_quantities):
            raise ValueError("signed_quantities must be finite")
        if any(not math.isfinite(value) or value <= 0.0 for value in self.mark_prices):
            raise ValueError("mark_prices must be finite and positive")
        if any(
            not math.isfinite(value) or value <= 0.0
            for value in self.contract_multipliers
        ):
            raise ValueError("contract_multipliers must be finite and positive")
        if any(not math.isfinite(value) for value in self.funding_rates):
            raise ValueError("funding_rates must be finite")
        for name, value in (
            ("funding_amount", self.funding_amount),
            ("equity_before_funding", self.equity_before_funding),
            ("equity_after_funding", self.equity_after_funding),
        ):
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")

        expected_funding = -math.fsum(
            quantity * mark * multiplier * rate
            for due, quantity, mark, multiplier, rate in zip(
                self.funding_due,
                self.signed_quantities,
                self.mark_prices,
                self.contract_multipliers,
                self.funding_rates,
                strict=True,
            )
            if due
        )
        if not math.isclose(
            self.funding_amount,
            expected_funding,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError("funding_amount does not match boundary mark notional")
        if not math.isclose(
            self.equity_after_funding,
            self.equity_before_funding + self.funding_amount,
            rel_tol=1e-12,
            abs_tol=1e-9,
        ):
            raise ValueError("funding boundary equity closure is inconsistent")


__all__ = ["FundingBoundaryEvidence"]
