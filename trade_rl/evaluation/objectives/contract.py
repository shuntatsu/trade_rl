"""Finite-horizon business objective, independent of training surrogates."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal

from trade_rl._validation import (
    require_aware_datetime,
    require_non_empty,
    require_sha256,
)
from trade_rl.artifacts.hashing import content_digest


def _finite(value: float, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a finite number")
    try:
        result = float(value)
    except OverflowError as error:
        raise ValueError(f"{field} must be finite") from error
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


def _equities(
    values: tuple[float, ...], *, count: int, field: str
) -> tuple[float, ...]:
    if not isinstance(values, tuple) or len(values) != count:
        raise ValueError(f"{field} must contain exactly {count} account values")
    return tuple(_finite(value, field=field) for value in values)


def _sum(values: tuple[float, ...], *, field: str) -> float:
    try:
        return _finite(math.fsum(values), field=field)
    except OverflowError as error:
        raise ValueError(f"{field} must be finite") from error


@dataclass(frozen=True, slots=True)
class CapitalContract:
    """Initial capital for actual independent books or one shared book."""

    account_mode: Literal["independent_symbol", "shared_portfolio"]
    currency: str
    initial_equities: tuple[float, ...]

    def __post_init__(self) -> None:
        if self.account_mode not in ("independent_symbol", "shared_portfolio"):
            raise ValueError("unknown account_mode")
        currency = require_non_empty(self.currency, field="currency")
        if not isinstance(self.initial_equities, tuple) or not self.initial_equities:
            raise ValueError("initial_equities must be a non-empty tuple")
        equities = _equities(
            self.initial_equities,
            count=len(self.initial_equities),
            field="initial_equities",
        )
        if any(value <= 0.0 for value in equities):
            raise ValueError("initial_equities must be positive")
        if self.account_mode == "shared_portfolio" and len(equities) != 1:
            raise ValueError("shared_portfolio requires exactly one capital book")
        _sum(equities, field="total_initial_capital")
        object.__setattr__(self, "currency", currency)
        object.__setattr__(self, "initial_equities", equities)

    @property
    def total_initial_capital(self) -> float:
        return _sum(self.initial_equities, field="total_initial_capital")


def net_equity_increment(
    before: float,
    after: float,
    *,
    initial_capital: float,
    net_external_cash_flow: float = 0.0,
) -> float:
    """After-cost equity increment on a fixed denominator; keep debt losses."""
    capital = _finite(initial_capital, field="initial_capital")
    if capital <= 0.0:
        raise ValueError("initial_capital must be positive")
    delta = _sum(
        (
            _finite(after, field="after"),
            -_finite(before, field="before"),
            -_finite(net_external_cash_flow, field="net_external_cash_flow"),
        ),
        field="net_equity_delta",
    )
    return _finite(delta / capital, field="net_equity_increment")


@dataclass(frozen=True, slots=True)
class ObjectiveContract:
    """Declare a new study's endpoint, capital and immutable economic references.

    Equity inputs must come from the canonical after-cost ledger. This declaration
    neither verifies referenced profile bytes nor authorizes a runner or study.
    """

    capital: CapitalContract
    evaluation_start: datetime
    evaluation_stop_exclusive: datetime
    terminal_valuation: Literal["settled", "marked_continuation"]
    economics_digest: str
    risk_digest: str
    deployment_recipe_digest: str
    maximum_drawdown: float = 0.20

    def __post_init__(self) -> None:
        if not isinstance(self.capital, CapitalContract):
            raise ValueError("capital must be a CapitalContract")
        start = require_aware_datetime(
            self.evaluation_start, field="evaluation_start"
        ).astimezone(UTC)
        stop = require_aware_datetime(
            self.evaluation_stop_exclusive, field="evaluation_stop_exclusive"
        ).astimezone(UTC)
        if stop <= start:
            raise ValueError("evaluation_stop_exclusive must follow evaluation_start")
        if self.terminal_valuation not in ("settled", "marked_continuation"):
            raise ValueError("unknown terminal_valuation")
        for field in ("economics_digest", "risk_digest", "deployment_recipe_digest"):
            require_sha256(getattr(self, field), field=field)
        drawdown = _finite(self.maximum_drawdown, field="maximum_drawdown")
        if not 0.0 < drawdown <= 0.20:
            raise ValueError("maximum_drawdown must be within (0, 0.20]")
        object.__setattr__(self, "evaluation_start", start)
        object.__setattr__(self, "evaluation_stop_exclusive", stop)
        object.__setattr__(self, "maximum_drawdown", drawdown)

    def net_profit_rate(
        self,
        terminal_equities: tuple[float, ...],
        *,
        net_external_cash_flows: tuple[float, ...] | None = None,
    ) -> float:
        """One realized capital-weighted endpoint, not expected future profit."""
        count = len(self.capital.initial_equities)
        terminal = _equities(terminal_equities, count=count, field="terminal_equities")
        flows = _equities(
            (0.0,) * count
            if net_external_cash_flows is None
            else net_external_cash_flows,
            count=count,
            field="net_external_cash_flows",
        )
        delta = _sum(
            terminal
            + tuple(-value for value in self.capital.initial_equities)
            + tuple(-value for value in flows),
            field="net_equity_delta",
        )
        return _finite(
            delta / self.capital.total_initial_capital,
            field="net_profit_rate",
        )

    def payload(self) -> dict[str, object]:
        return {
            "schema": "net_profit_objective_v1",
            "objective_id": "expected_terminal_net_profit_v1",
            "tax_treatment": "pretax",
            "infrastructure_cost_treatment": "reported_separately",
            "external_cash_flow_convention": "net_deposits_positive_withdrawals_negative",
            "capital": {
                "mode": self.capital.account_mode,
                "currency": self.capital.currency,
                "initial_equities": list(self.capital.initial_equities),
            },
            "evaluation_start": self.evaluation_start.isoformat(),
            "evaluation_stop_exclusive": self.evaluation_stop_exclusive.isoformat(),
            "terminal_valuation": self.terminal_valuation,
            "economics_digest": self.economics_digest,
            "risk_digest": self.risk_digest,
            "deployment_recipe_digest": self.deployment_recipe_digest,
            "maximum_drawdown": self.maximum_drawdown,
        }

    @property
    def digest(self) -> str:
        return content_digest(self.payload())
