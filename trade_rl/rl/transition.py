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
    """Resolve Gymnasium flags and a stable terminal or truncation reason."""

    terminated = hybrid.insolvent or liquidation_terminal
    shadow_truncated = shadow.insolvent and not terminated
    truncated = (time_limit_reached or shadow_truncated) and not terminated
    if hybrid.termination_reason is not None:
        reason = EconomicTerminationReason(hybrid.termination_reason).value
    elif liquidation_terminal:
        reason = "forced_close" if liquidation_complete else "liquidation_incomplete"
    elif shadow.termination_reason is not None:
        shadow_reason = EconomicTerminationReason(shadow.termination_reason).value
        reason = f"shadow_{shadow_reason}"
    else:
        reason = None
    return EconomicTransition(
        terminated=terminated,
        truncated=truncated,
        reason=reason,
    )


__all__ = ["EconomicTransition", "classify_economic_transition"]
