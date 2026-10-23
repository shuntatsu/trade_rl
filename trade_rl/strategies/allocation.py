"""Pure single-account optimization of a declared after-cost return surrogate.

Inputs are expected SIMPLE returns from the current decision valuation, with
variance and every cost/carry estimate on that same horizon. They are supplied
assumptions, not verified forecasts or calibrated costs. This module fits no
model, reads no account, and grants no execution or research authorization.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, fields
from fractions import Fraction

import numpy as np

from trade_rl.artifacts.hashing import content_digest


def _finite(name: str, value: float, *, nonnegative: bool = False) -> None:
    if isinstance(value, bool) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    if nonnegative and value < 0.0:
        raise ValueError(f"{name} must be nonnegative")


def _identity(name: str, value: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")


def _time(value: np.datetime64) -> np.datetime64:
    if not isinstance(value, np.datetime64) or np.isnat(value):
        raise ValueError("time must be a finite numpy datetime64")
    normalized = value.astype("datetime64[ns]")
    if normalized.astype(value.dtype) != value:
        raise ValueError("time must be representable in nanoseconds")
    return normalized


@dataclass(frozen=True, slots=True)
class AllocationInputs:
    symbol: str
    decision_time: np.datetime64
    available_at: np.datetime64
    horizon_end: np.datetime64
    source_identity: str
    expected_simple_return: float
    return_variance: float = 0.0
    buy_cost: float = 0.0
    sell_cost: float = 0.0
    exit_cost: float = 0.0
    funding_return: float = 0.0
    borrow_return: float = 0.0
    cash_return: float = 0.0
    return_unit: str = "expected_simple_return"

    def __post_init__(self) -> None:
        _identity("symbol", self.symbol)
        _identity("source_identity", self.source_identity)
        for name in ("decision_time", "available_at", "horizon_end"):
            object.__setattr__(self, name, _time(getattr(self, name)))
        if self.available_at > self.decision_time:
            raise ValueError("inputs are not available at the decision time")
        if self.horizon_end <= self.decision_time:
            raise ValueError("return horizon must end after the decision time")
        if self.return_unit != "expected_simple_return":
            raise ValueError("return unit must be expected_simple_return")
        _finite("expected_simple_return", self.expected_simple_return)
        if self.expected_simple_return < -1.0:
            raise ValueError("expected simple return cannot be below -1")
        for name in (
            "return_variance",
            "buy_cost",
            "sell_cost",
            "exit_cost",
            "borrow_return",
        ):
            _finite(name, getattr(self, name), nonnegative=True)
        for name in ("funding_return", "cash_return"):
            _finite(name, getattr(self, name))


@dataclass(frozen=True, slots=True)
class AllocationContext:
    """Detached account facts; evaluation owns their canonical provenance."""

    account_id: str
    symbol: str
    decision_time: np.datetime64
    state_digest: str
    current_weight: float
    quantity: str
    cash: float
    equity: float
    pending_remaining: str

    def __post_init__(self) -> None:
        _identity("account_id", self.account_id)
        _identity("symbol", self.symbol)
        object.__setattr__(self, "decision_time", _time(self.decision_time))
        if not re.fullmatch(r"[0-9a-f]{64}", self.state_digest):
            raise ValueError("state_digest must be a lowercase SHA256 digest")
        for name in ("current_weight", "cash", "equity"):
            _finite(name, getattr(self, name))
        if self.equity <= 0.0:
            raise ValueError("equity must be positive")
        _identity("quantity", self.quantity)
        _identity("pending_remaining", self.pending_remaining)


@dataclass(frozen=True, slots=True)
class AllocationProposal:
    inputs: AllocationInputs
    context: AllocationContext
    allocator: AfterCostTargetAllocator
    target_weight: float
    objective_value: float
    decision_digest: str

    @property
    def is_hold(self) -> bool:
        return self.target_weight == self.context.current_weight


@dataclass(frozen=True, slots=True)
class AfterCostTargetAllocator:
    """Exact maximum of a concave piecewise quadratic on a fixed interval.

    Hard risk is a separate, final projection. Equal utility chooses actual HOLD,
    then minimum turnover, then smaller absolute exposure, then signed weight.
    """

    lower_weight: float = -1.0
    upper_weight: float = 1.0
    max_turnover: float | None = None
    risk_aversion: float = 0.0

    def __post_init__(self) -> None:
        for name in ("lower_weight", "upper_weight"):
            _finite(name, getattr(self, name))
        if not -10.0 <= self.lower_weight <= self.upper_weight <= 10.0:
            raise ValueError("weight bounds must be ordered within [-10, 10]")
        _finite("risk_aversion", self.risk_aversion, nonnegative=True)
        if self.max_turnover is not None:
            _finite("max_turnover", self.max_turnover, nonnegative=True)

    def propose(
        self, inputs: AllocationInputs, context: AllocationContext
    ) -> AllocationProposal:
        if inputs.symbol != context.symbol:
            raise ValueError("input symbol does not match the account context")
        if inputs.decision_time != context.decision_time:
            raise ValueError("return interval must start at the current decision time")
        current = context.current_weight
        lower, upper = self.lower_weight, self.upper_weight
        if self.max_turnover is not None:
            lower = max(lower, current - self.max_turnover)
            upper = min(upper, current + self.max_turnover)
        if lower > upper:
            raise ValueError("allocation feasible interval is empty")
        quadratic = self.risk_aversion * inputs.return_variance
        _finite("risk penalty coefficient", quadratic, nonnegative=True)
        breakpoints = sorted(
            {lower, upper, *(w for w in (0.0, current) if lower <= w <= upper)}
        )
        candidates = set(breakpoints)
        coefficients = {
            name: Fraction(float(getattr(inputs, name)))
            for name in (
                "expected_simple_return",
                "return_variance",
                "buy_cost",
                "sell_cost",
                "exit_cost",
                "funding_return",
                "borrow_return",
                "cash_return",
            )
        }
        exact_quadratic = (
            Fraction(float(self.risk_aversion)) * coefficients["return_variance"]
        )
        if exact_quadratic > 0:
            for left, right in zip(breakpoints, breakpoints[1:]):
                midpoint = left + (right - left) / 2.0
                trade_slope = (
                    -coefficients["buy_cost"]
                    if midpoint > current
                    else coefficients["sell_cost"]
                )
                holding_slope = (
                    -coefficients["exit_cost"]
                    if midpoint > 0.0
                    else coefficients["exit_cost"] + coefficients["borrow_return"]
                )
                slope = (
                    coefficients["expected_simple_return"]
                    - coefficients["cash_return"]
                    - coefficients["funding_return"]
                    + trade_slope
                    + holding_slope
                )
                stationary = slope / (2 * exact_quadratic)
                if Fraction(left) < stationary < Fraction(right):
                    candidates.add(float(stationary))

        def utility(weight: float) -> Fraction:
            # Compare the mathematical values of the supplied IEEE coefficients.
            # Rounding each endpoint separately can invent a gain on a flat
            # segment. Exact comparison preserves ties and nextafter-small gains.
            w, w0 = Fraction(weight), Fraction(current)
            return (
                coefficients["cash_return"] * (1 - w)
                + coefficients["expected_simple_return"] * w
                - exact_quadratic * w * w
                - coefficients["buy_cost"] * max(w - w0, Fraction(0))
                - coefficients["sell_cost"] * max(w0 - w, Fraction(0))
                - coefficients["exit_cost"] * abs(w)
                - coefficients["funding_return"] * w
                - coefficients["borrow_return"] * max(-w, Fraction(0))
            )

        target = max(
            candidates,
            key=lambda w: (
                utility(w),
                w == current,
                -abs(Fraction(w) - Fraction(current)),
                -abs(w),
                -w,
            ),
        )
        try:
            score = float(utility(target))
        except OverflowError as error:
            raise ValueError("allocation objective must be finite") from error
        _finite("allocation objective", score)
        payload: dict[str, object] = {}
        for name, record in (
            ("inputs", inputs),
            ("context", context),
            ("allocator", self),
        ):
            payload[name] = {
                field.name: str(value)
                if isinstance(value := getattr(record, field.name), np.datetime64)
                else value
                for field in fields(record)
            }
        payload.update(target=target, objective_value=score)
        return AllocationProposal(
            inputs, context, self, target, score, content_digest(payload)
        )


__all__ = [
    "AllocationInputs",
    "AllocationContext",
    "AllocationProposal",
    "AfterCostTargetAllocator",
]
