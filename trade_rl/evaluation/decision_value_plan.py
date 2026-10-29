"""Typed minimal plan contract for #1025 direct decision-value studies (TD0).

The contract validates that a plan names every economic and data-use field before
anything runs. It never launches work. Real-development scope (TD5) is rejected
unless the caller supplies explicit prior authorization evidence, so a software
or synthetic plan cannot silently become an economic study.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from trade_rl.strategies.portfolio_policy_codec import (
    ACTION_HOLD,
    decode_portfolio_action_codes,
)

DataScope = Literal["synthetic_only", "real_development"]
_ALLOWED_SCOPES = ("synthetic_only", "real_development")


@dataclass(frozen=True, slots=True)
class DecisionValueStudyPlan:
    """Every field is required; nothing is defaulted from a result."""

    data_scope: str
    account_mode: str
    symbols: tuple[str, ...]
    initial_capital: float
    horizon_bars: int
    terminal_reserve_bars: int
    candidate_actions: tuple[tuple[int, ...], ...]
    fee_rate: float
    spread_rate: float
    max_policy_improvements: int
    learner_roster: tuple[str, ...]
    real_scope_authorization_digest: str | None = None

    def __post_init__(self) -> None:
        if self.data_scope not in _ALLOWED_SCOPES:
            raise ValueError("data_scope must be synthetic_only or real_development")
        if self.account_mode != "shared_portfolio":
            raise ValueError("only the opt-in shared_portfolio account is supported")
        if type(self.symbols) is not tuple or not self.symbols:
            raise ValueError("symbols must be a non-empty tuple")
        if len(set(self.symbols)) != len(self.symbols) or any(
            type(s) is not str or not s for s in self.symbols
        ):
            raise ValueError("symbols must be unique non-empty strings")
        for name in ("initial_capital", "fee_rate", "spread_rate"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be a finite real")
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if self.initial_capital <= 0:
            raise ValueError("initial_capital must be positive")
        if self.fee_rate < 0 or self.spread_rate < 0:
            raise ValueError("cost rates must be non-negative")
        if self.fee_rate + self.spread_rate <= 0:
            # fee-only learnability must not be tested at zero cost
            raise ValueError("execution cost must be non-zero")
        for name in ("horizon_bars", "terminal_reserve_bars"):
            if type(getattr(self, name)) is not int:
                raise ValueError(f"{name} must be an integer")
        if self.horizon_bars < 2 or self.terminal_reserve_bars < 1:
            raise ValueError("horizon and terminal reserve are too small")
        if self.terminal_reserve_bars >= self.horizon_bars:
            raise ValueError("terminal reserve must leave decision bars")
        if self.max_policy_improvements != 1:
            raise ValueError("initial plan allows exactly one policy improvement")
        if type(self.candidate_actions) is not tuple or len(self.candidate_actions) < 2:
            raise ValueError("candidates require joint HOLD and another action")
        n = len(self.symbols)
        decoded = tuple(
            decode_portfolio_action_codes(a, n_symbols=n)
            for a in self.candidate_actions
        )
        if decoded[0] != (ACTION_HOLD,) * n:
            raise ValueError("first candidate must be joint HOLD")
        if len(set(decoded)) != len(decoded):
            raise ValueError("candidate actions must be unique")
        if (
            type(self.learner_roster) is not tuple
            or not 1 <= len(self.learner_roster) <= 2
            or any(name not in ("linear", "shallow") for name in self.learner_roster)
            or len(set(self.learner_roster)) != len(self.learner_roster)
        ):
            raise ValueError("learner roster is at most linear and shallow once each")
        digest = self.real_scope_authorization_digest
        if self.data_scope == "real_development":
            if (
                digest is None
                or len(digest) != 64
                or any(c not in "0123456789abcdef" for c in digest)
            ):
                raise ValueError(
                    "real_development requires a separate authorization digest"
                )
        elif digest is not None:
            raise ValueError("synthetic plans must not carry a real authorization")


__all__ = ["DecisionValueStudyPlan"]
