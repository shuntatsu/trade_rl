"""Explicit regular financial time for a future learning adapter."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

from trade_rl.artifacts.hashing import content_digest
from trade_rl.evaluation.objectives.contract import _finite


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
        return self.gamma ** (seconds / self.decision_interval_seconds)

    @property
    def terminal_profit_aligned(self) -> bool:
        """Algebraic alignment only; does not prove actual environment parity."""
        return self.reward_schema == "equity_delta_v1" and self.gamma == 1.0

    def payload(self) -> dict[str, object]:
        return {"schema": "financial_clock_v1", **asdict(self)}

    @property
    def digest(self) -> str:
        return content_digest(self.payload())
