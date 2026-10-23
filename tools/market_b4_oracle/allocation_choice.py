"""Independent scalar allocation arithmetic for the frozen forecast driver.

Inputs are supplied horizon assumptions, not authenticated forecast packets or
observed profits. This calculator fits no model, imports no production optimizer
and grants no risk, intent, execution or research authority.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from fractions import Fraction
from typing import cast


def _finite(value: object, *, nonnegative: bool = False) -> float:
    if type(value) not in (int, float):
        raise ValueError("allocation arithmetic requires finite ordinary numbers")
    try:
        result = float(cast(int | float, value))
    except OverflowError as error:
        raise ValueError("allocation number exceeds its finite range") from error
    if not math.isfinite(result) or (nonnegative and result < 0):
        raise ValueError("invalid finite allocation assumption or parameter")
    return result


@dataclass(frozen=True, slots=True)
class AllocationAssumptions:
    expected_simple_return: float
    return_variance: float = 0.0
    buy_cost: float = 0.0
    sell_cost: float = 0.0
    exit_cost: float = 0.0
    funding_return: float = 0.0
    borrow_return: float = 0.0
    cash_return: float = 0.0

    def __post_init__(self) -> None:
        nonnegative = {
            "return_variance",
            "buy_cost",
            "sell_cost",
            "exit_cost",
            "borrow_return",
        }
        for item in fields(self):
            _finite(getattr(self, item.name), nonnegative=item.name in nonnegative)
        if self.expected_simple_return < -1:
            raise ValueError("expected simple return cannot be below -1")


@dataclass(frozen=True, slots=True)
class AllocationChoice:
    target_weight: float
    objective_value: float

    @property
    def scope(self) -> str:
        return "AFTER_COST_ALLOCATION_ARITHMETIC_ONLY"


def choose_after_cost_weight(
    inputs: AllocationAssumptions,
    *,
    current_weight: float,
    lower_weight: float = -1.0,
    upper_weight: float = 1.0,
    risk_aversion: float = 0.0,
    max_turnover: float | None = None,
) -> AllocationChoice:
    """Maximize the declared piecewise quadratic, retaining binary64 tie rules.

    Candidate weights are feasible endpoints, kinks and rounded interior roots
    of the utility derivative. Utilities compare exact rational representations
    of supplied binary64 coefficients; rounded scores cannot erase a small gain
    or manufacture a gain on a flat segment. Exact ties prefer current holdings,
    least turnover, least absolute exposure, then the smaller signed weight.
    """
    if type(inputs) is not AllocationAssumptions:
        raise ValueError("requires exact allocation assumptions")
    inputs.__post_init__()
    current = _finite(current_weight)
    lower, upper = _finite(lower_weight), _finite(upper_weight)
    aversion = _finite(risk_aversion, nonnegative=True)
    if not -10 <= lower <= upper <= 10:
        raise ValueError("allocation bounds must be ordered within [-10, 10]")
    if max_turnover is not None:
        turnover = _finite(max_turnover, nonnegative=True)
        lower, upper = max(lower, current - turnover), min(upper, current + turnover)
    if lower > upper:
        raise ValueError("allocation turnover interval is empty")
    _finite(aversion * inputs.return_variance, nonnegative=True)

    coefficients = {
        item.name: Fraction(float(getattr(inputs, item.name)))
        for item in fields(inputs)
    }
    mu = coefficients["expected_simple_return"]
    cash = coefficients["cash_return"]
    funding = coefficients["funding_return"]
    buy, sell = coefficients["buy_cost"], coefficients["sell_cost"]
    exit_cost, borrow = coefficients["exit_cost"], coefficients["borrow_return"]
    penalty = Fraction(aversion) * coefficients["return_variance"]
    held = Fraction(current)
    knots = sorted({lower, upper} | {x for x in (0.0, current) if lower <= x <= upper})
    candidates = set(knots)

    if penalty:
        for left, right in zip(knots, knots[1:]):
            midpoint = (Fraction(left) + Fraction(right)) / 2
            slope = mu - cash - funding
            slope += -buy if midpoint > held else sell
            slope += -exit_cost if midpoint > 0 else exit_cost + borrow
            root = slope / (2 * penalty)
            if Fraction(left) < root < Fraction(right):
                candidates.add(float(root))

    def utility(candidate: float) -> Fraction:
        weight = Fraction(candidate)
        return (
            cash * (1 - weight)
            + (mu - funding) * weight
            - penalty * weight**2
            - buy * max(weight - held, Fraction(0))
            - sell * max(held - weight, Fraction(0))
            - exit_cost * abs(weight)
            - borrow * max(-weight, Fraction(0))
        )

    # Rank only mathematical candidates; this is not a numerical grid search.
    target = max(
        candidates,
        key=lambda weight: (
            utility(weight),
            weight == current,
            -abs(Fraction(weight) - held),
            -abs(weight),
            -weight,
        ),
    )
    try:
        score = float(utility(target))
    except OverflowError as error:
        raise ValueError("allocation objective exceeds its finite range") from error
    return AllocationChoice(target, _finite(score))
