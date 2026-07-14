"""Pure classification of economic environment transition outcomes."""

from __future__ import annotations

from dataclasses import dataclass

from trade_rl.simulation.accounting import BookState, EconomicTerminationReason


@dataclass(frozen=True, slots=True)
class EconomicTransition:
    terminated: bool
    truncated: bool
    reason: str | None


def classify_economic_transition(
    *,
    hybrid: BookState,
    shadow: BookState,
    time_limit_reached: bool,
    liquidation_terminal: bool,
    liquidation_complete: bool,
) -> EconomicTransition:
    """Resolve Gymnasium flags and a stable terminal reason from book state."""

    terminated = hybrid.insolvent or shadow.insolvent or liquidation_terminal
    truncated = time_limit_reached and not terminated
    if not terminated:
        reason = None
    elif hybrid.termination_reason is not None:
        reason = EconomicTerminationReason(hybrid.termination_reason).value
    elif shadow.termination_reason is not None:
        reason = EconomicTerminationReason(shadow.termination_reason).value
    elif liquidation_complete:
        reason = "forced_close"
    else:
        reason = "liquidation_incomplete"
    return EconomicTransition(
        terminated=terminated,
        truncated=truncated,
        reason=reason,
    )


__all__ = ["EconomicTransition", "classify_economic_transition"]
