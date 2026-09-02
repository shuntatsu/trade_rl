"""Pure realized-wealth reward oracle for Universal Trade RL U1."""

from __future__ import annotations

import math
from collections.abc import Iterable

_UNIVERSAL_TRADE_REWARD_SCALE = 100.0
_DEFAULT_RECONCILIATION_ATOL = 1e-10


def _require_positive_finite_wealth(value: float, *, name: str) -> float:
    resolved = float(value)
    if not math.isfinite(resolved) or resolved <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return resolved


def universal_net_log_growth_reward(
    *,
    before_value: float,
    after_value: float,
) -> float:
    """Return U1 V1 reward: 100 times realized net log wealth growth."""

    before = _require_positive_finite_wealth(before_value, name="before_value")
    after = _require_positive_finite_wealth(after_value, name="after_value")
    return _UNIVERSAL_TRADE_REWARD_SCALE * math.log(after / before)


def reconcile_universal_trade_reward(
    *,
    rewards: Iterable[float],
    initial_value: float,
    final_value: float,
    atol: float = _DEFAULT_RECONCILIATION_ATOL,
) -> None:
    """Fail closed unless cumulative rewards telescope to realized wealth growth."""

    initial = _require_positive_finite_wealth(initial_value, name="initial_value")
    final = _require_positive_finite_wealth(final_value, name="final_value")
    tolerance = float(atol)
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("atol must be finite and non-negative")

    reward_total = 0.0
    for reward in rewards:
        resolved = float(reward)
        if not math.isfinite(resolved):
            raise ValueError("rewards must contain only finite values")
        reward_total += resolved

    observed = reward_total / _UNIVERSAL_TRADE_REWARD_SCALE
    expected = math.log(final / initial)
    if not math.isclose(observed, expected, rel_tol=0.0, abs_tol=tolerance):
        raise ValueError(
            "Universal Trade U1 reward reconciliation mismatch: "
            f"observed={observed!r}, expected={expected!r}, atol={tolerance!r}"
        )


__all__ = [
    "reconcile_universal_trade_reward",
    "universal_net_log_growth_reward",
]
