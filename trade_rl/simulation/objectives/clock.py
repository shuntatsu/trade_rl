"""Explicit financial time for fixed and calendar-month learning adapters."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from dataclasses import field as dataclass_field
from datetime import UTC, datetime
from typing import ClassVar, Literal

from trade_rl._validation import _ordered_ns_durations, _require_nominal_session_seconds
from trade_rl.artifacts.hashing import content_digest
from trade_rl.simulation.objectives.contract import _finite


@dataclass(frozen=True, slots=True)
class FinancialClockContract:
    """Declare clock/reward choices without changing historical PPO defaults."""

    decision_interval_seconds: int
    execution_interval_seconds: int
    reward_interval_seconds: int
    economic_horizon_seconds: int
    rollout_steps: int
    gamma: float
    gae_lambda: float
    reward_schema: Literal["net_log_return_v1", "equity_delta_v1"]

    def __post_init__(self) -> None:
        for field in (
            "decision_interval_seconds",
            "execution_interval_seconds",
            "reward_interval_seconds",
            "economic_horizon_seconds",
            "rollout_steps",
        ):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{field} must be a positive integer")
        if self.decision_interval_seconds % self.execution_interval_seconds:
            raise ValueError("decisions must align with execution intervals")
        if self.reward_interval_seconds != self.decision_interval_seconds:
            raise ValueError("one declared reward interval is required per decision")
        if self.economic_horizon_seconds % self.decision_interval_seconds:
            raise ValueError("finite horizon must align with decision intervals")
        gamma = _finite(self.gamma, field="gamma")
        gae = _finite(self.gae_lambda, field="gae_lambda")
        if not 0.0 < gamma <= 1.0:
            raise ValueError("gamma must be within (0, 1]")
        if not 0.0 <= gae <= 1.0:
            raise ValueError("gae_lambda must be within [0, 1]")
        if self.reward_schema not in ("net_log_return_v1", "equity_delta_v1"):
            raise ValueError("unknown reward_schema")
        object.__setattr__(self, "gamma", gamma)
        object.__setattr__(self, "gae_lambda", gae)

    def discount_after(self, seconds: int) -> float:
        if isinstance(seconds, bool) or not isinstance(seconds, int) or seconds < 0:
            raise ValueError("seconds must be a non-negative integer")
        if self.gamma == 1.0:
            return 1.0
        try:
            elapsed_decisions = seconds / self.decision_interval_seconds
        except OverflowError:
            return 0.0
        return self.gamma**elapsed_decisions

    @property
    def terminal_profit_aligned(self) -> bool:
        """Algebraic alignment only; does not prove actual environment parity."""
        return self.reward_schema == "equity_delta_v1" and self.gamma == 1.0

    def payload(self) -> dict[str, object]:
        return {"schema": "financial_clock_v1", **asdict(self)}

    @property
    def digest(self) -> str:
        return content_digest(self.payload())


def _calendar_month_endpoint(start: datetime, months: int) -> datetime:
    """Move along UTC first-day midnight boundaries without a nominal duration."""
    if start.utcoffset() is None:
        raise ValueError("calendar month boundary requires an aware datetime")
    start = start.astimezone(UTC)
    if (start.day, start.hour, start.minute, start.second, start.microsecond) != (
        1,
        0,
        0,
        0,
        0,
    ):
        raise ValueError("calendar month boundary must be UTC first-day midnight")
    if type(months) is not int or months < 0:
        raise ValueError("calendar month count must be a non-negative integer")
    year, month = divmod(start.year * 12 + start.month - 1 + months, 12)
    try:
        return datetime(year, month + 1, 1, tzinfo=UTC)
    except (ValueError, OverflowError) as error:
        raise ValueError(
            "calendar month boundary is outside the datetime range"
        ) from error


@dataclass(frozen=True, slots=True)
class CalendarMonthFinancialClockContract:
    """One action/reward per UTC month, undiscounted terminal-equity objective.

    Horizon seconds are the actual elapsed duration of the bound window.
    GAE lambda and rollout_steps count decision steps, not equal durations.
    No nominal seconds are supplied for a calendar month.
    """

    economic_horizon_months: int
    economic_horizon_seconds: int
    rollout_steps: int
    gamma: float
    gae_lambda: float
    reward_schema: Literal["equity_delta_v1"]

    decision_interval_seconds: ClassVar[None] = None
    execution_interval_seconds: ClassVar[None] = None
    reward_interval_seconds: ClassVar[None] = None

    def __post_init__(self) -> None:
        for field in (
            "economic_horizon_months",
            "economic_horizon_seconds",
            "rollout_steps",
        ):
            value = getattr(self, field)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{field} must be a positive native integer")
        gamma = _finite(self.gamma, field="gamma")
        gae = _finite(self.gae_lambda, field="gae_lambda")
        if (
            gamma != 1.0
            or type(self.reward_schema) is not str
            or self.reward_schema != "equity_delta_v1"
        ):
            raise ValueError("calendar month clock requires gamma-one equity deltas")
        if not 0.0 <= gae <= 1.0:
            raise ValueError("gae_lambda must be within [0, 1]")
        object.__setattr__(self, "gamma", gamma)
        object.__setattr__(self, "gae_lambda", gae)

    def discount_after(self, seconds: int) -> float:
        if isinstance(seconds, bool) or not isinstance(seconds, int) or seconds < 0:
            raise ValueError("seconds must be a non-negative integer")
        return 1.0

    @property
    def terminal_profit_aligned(self) -> bool:
        return self.reward_schema == "equity_delta_v1" and self.gamma == 1.0

    def payload(self) -> dict[str, object]:
        return {"schema": "calendar_month_financial_clock_v1", **asdict(self)}

    @property
    def digest(self) -> str:
        return content_digest(self.payload())


@dataclass(frozen=True, slots=True)
class SessionFinancialClockContract:
    """Caller-established session closes, one undiscounted reward per decision.

    Boundaries include the initial mark and all processing closes. They are a
    declaration, not proof of venue calendar completeness. Nominal seconds
    describe the processing bar, not the elapsed time between closes.
    """

    boundaries_ns: tuple[int, ...]
    nominal_bar_seconds: int
    economic_horizon_seconds: int
    rollout_steps: int
    gamma: float
    gae_lambda: float
    reward_schema: Literal["equity_delta_v1"]
    _digest: str = dataclass_field(init=False, repr=False, compare=False)

    decision_interval_seconds: ClassVar[None] = None
    execution_interval_seconds: ClassVar[None] = None
    reward_interval_seconds: ClassVar[None] = None

    def __post_init__(self) -> None:
        durations = _ordered_ns_durations(self.boundaries_ns)
        _require_nominal_session_seconds(self.nominal_bar_seconds, durations)
        if (
            len(durations) < 2
            or any(x % 1_000_000_000 for x in self.boundaries_ns)
            or type(self.economic_horizon_seconds) is not int
            or self.economic_horizon_seconds * 1_000_000_000 != sum(durations)
            or type(self.rollout_steps) is not int
            or self.rollout_steps <= 0
        ):
            raise ValueError(
                "session clock requires a whole-second settled horizon matching its boundaries"
            )
        gamma = _finite(self.gamma, field="gamma")
        gae = _finite(self.gae_lambda, field="gae_lambda")
        if (
            gamma != 1.0
            or type(self.reward_schema) is not str
            or self.reward_schema != "equity_delta_v1"
        ):
            raise ValueError("session clock requires gamma-one equity deltas")
        if not 0.0 <= gae <= 1.0:
            raise ValueError("gae_lambda must be within [0, 1]")
        object.__setattr__(self, "gamma", gamma)
        object.__setattr__(self, "gae_lambda", gae)
        object.__setattr__(self, "_digest", content_digest(self.payload()))

    def discount_after(self, seconds: int) -> float:
        if isinstance(seconds, bool) or not isinstance(seconds, int) or seconds < 0:
            raise ValueError("seconds must be a non-negative integer")
        return 1.0

    @property
    def terminal_profit_aligned(self) -> bool:
        return self.reward_schema == "equity_delta_v1" and self.gamma == 1.0

    def payload(self) -> dict[str, object]:
        return {
            "schema": "session_financial_clock_v1",
            "boundaries_ns": list(self.boundaries_ns),
            "nominal_bar_seconds": self.nominal_bar_seconds,
            "economic_horizon_seconds": self.economic_horizon_seconds,
            "rollout_steps": self.rollout_steps,
            "gamma": self.gamma,
            "gae_lambda": self.gae_lambda,
            "reward_schema": self.reward_schema,
        }

    @property
    def digest(self) -> str:
        return self._digest
