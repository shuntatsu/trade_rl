from __future__ import annotations

from trade_rl.simulation.runtime_promotion import (
    ExecutionPromotionEvidence,
    RuntimeMode,
    assess_runtime_promotion,
)


def _evidence(**overrides: bool) -> ExecutionPromotionEvidence:
    values = {
        "capability_passed": False,
        "causal_bridge_passed": False,
        "funding_passed": False,
        "terminal_flat_passed": False,
        "exact_parity_passed": False,
        "determinism_passed": False,
        "performance_approved": False,
    }
    values.update(overrides)
    return ExecutionPromotionEvidence(**values)


def test_legacy_authoritative_is_always_available_as_fail_closed_default() -> None:
    decision = assess_runtime_promotion(
        requested=RuntimeMode.LEGACY_AUTHORITATIVE,
        evidence=_evidence(),
    )

    assert decision.allowed is True
    assert decision.missing == ()


def test_dual_shadow_requires_runtime_and_causal_execution_contracts() -> None:
    denied = assess_runtime_promotion(
        requested=RuntimeMode.DUAL_SHADOW,
        evidence=_evidence(capability_passed=True),
    )
    assert denied.allowed is False
    assert denied.missing == (
        "causal_bridge_passed",
        "funding_passed",
        "terminal_flat_passed",
    )

    allowed = assess_runtime_promotion(
        requested=RuntimeMode.DUAL_SHADOW,
        evidence=_evidence(
            capability_passed=True,
            causal_bridge_passed=True,
            funding_passed=True,
            terminal_flat_passed=True,
        ),
    )
    assert allowed.allowed is True


def test_nautilus_authoritative_requires_all_promotion_evidence() -> None:
    denied = assess_runtime_promotion(
        requested=RuntimeMode.NAUTILUS_AUTHORITATIVE,
        evidence=_evidence(
            capability_passed=True,
            causal_bridge_passed=True,
            funding_passed=True,
            terminal_flat_passed=True,
            exact_parity_passed=True,
            determinism_passed=True,
        ),
    )

    assert denied.allowed is False
    assert denied.missing == ("performance_approved",)

    allowed = assess_runtime_promotion(
        requested=RuntimeMode.NAUTILUS_AUTHORITATIVE,
        evidence=_evidence(
            capability_passed=True,
            causal_bridge_passed=True,
            funding_passed=True,
            terminal_flat_passed=True,
            exact_parity_passed=True,
            determinism_passed=True,
            performance_approved=True,
        ),
    )
    assert allowed.allowed is True
