"""Pure FS1 label rule: FORWARD_PASS / FORWARD_FAIL / FORWARD_INVALID per candidate arm.

The rule consumes only numbers that the scoring step computed; it reads no dataset,
clock or file. It never returns a ``WINNER`` and never a profitability claim.

Order of precedence for one candidate arm:

1. ``FORWARD_INVALID`` when the arm, or any control it is compared against, is not a
   valid measurement, or when a needed number is missing / non-finite. An invalid
   control (CASH, FS1-C) makes every candidate ``FORWARD_INVALID``: a broken baseline
   is never silently skipped.
2. ``FORWARD_FAIL`` when any falsifier holds: the maximum drawdown exceeds the limit in
   the base or cost x2 scenario, or the 95% lower bound of annualised daily excess
   over CASH at cost x2 is not above zero.
3. ``FORWARD_PASS`` otherwise.

``passive_status`` separately records whether the 95% lower bound of the excess over
the vol-targeted passive control FS1-C (cost x2) is above zero. ``FORWARD_PASS`` is only
the minimum bar. The HEADLINE flag is ``edge_claim_supported``: true only for
``FORWARD_PASS`` together with ``BETTER_THAN_PASSIVE`` and ``survives_bonferroni_k2``
(the Bonferroni k=2 lower bound over CASH at cost x2 above zero).
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

from trade_rl.evaluation.forward_study.registry import (
    LABEL_FAIL,
    LABEL_INVALID,
    LABEL_PASS,
    PASSIVE_BETTER,
    PASSIVE_NOT_BETTER,
    Fs1Registry,
)


@dataclass(frozen=True)
class CandidateEvidence:
    """Numbers for one candidate arm (``None`` marks a missing measurement)."""

    valid: bool
    invalid_reasons: tuple[str, ...]
    max_drawdown_pct: Mapping[str, float | None]
    excess_vs_cash_ci_low: float | None
    excess_vs_cash_bonferroni_low: float | None
    excess_vs_passive_ci_low: float | None


@dataclass(frozen=True)
class CandidateLabel:
    label: str
    failed_falsifiers: tuple[str, ...]
    invalid_reasons: tuple[str, ...]
    passive_status: str | None
    edge_claim_supported: bool
    survives_bonferroni_k2: bool | None

    def to_mapping(self) -> dict[str, object]:
        return {
            "label": self.label,
            "failed_falsifiers": list(self.failed_falsifiers),
            "invalid_reasons": list(self.invalid_reasons),
            "passive_status": self.passive_status,
            "edge_claim_supported": self.edge_claim_supported,
            "survives_bonferroni_k2": self.survives_bonferroni_k2,
        }


def _finite(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def classify_candidate(
    evidence: CandidateEvidence,
    *,
    controls_valid: bool,
    registry: Fs1Registry,
) -> CandidateLabel:
    """Apply the frozen falsifiers to one candidate."""

    reasons = list(evidence.invalid_reasons)
    if not evidence.valid and not reasons:
        reasons.append("arm_invalid")
    if not controls_valid:
        reasons.append("control_invalid")
    for scenario in registry.drawdown_scenarios:
        if not _finite(evidence.max_drawdown_pct.get(scenario)):
            reasons.append(f"missing_max_drawdown_{scenario}")
    if not _finite(evidence.excess_vs_cash_ci_low):
        reasons.append("missing_excess_vs_cash_ci")
    if not _finite(evidence.excess_vs_passive_ci_low):
        reasons.append("missing_excess_vs_passive_ci")
    if reasons:
        return CandidateLabel(
            LABEL_INVALID, (), tuple(dict.fromkeys(reasons)), None, False, None
        )
    failed: list[str] = []
    for scenario in registry.drawdown_scenarios:
        drawdown = evidence.max_drawdown_pct[scenario]
        assert drawdown is not None
        if drawdown > registry.max_drawdown_limit_pct:
            failed.append(f"max_drawdown_above_limit_{scenario}")
    cash_low = evidence.excess_vs_cash_ci_low
    assert cash_low is not None
    if not cash_low > 0.0:
        failed.append(f"excess_vs_cash_ci_low_not_positive_{registry.excess_scenario}")
    passive_low = evidence.excess_vs_passive_ci_low
    assert passive_low is not None
    passive = PASSIVE_BETTER if passive_low > 0.0 else PASSIVE_NOT_BETTER
    bonferroni = evidence.excess_vs_cash_bonferroni_low
    survives = None if not _finite(bonferroni) else bool(bonferroni > 0.0)  # type: ignore[operator]
    label = LABEL_FAIL if failed else LABEL_PASS
    return CandidateLabel(
        label,
        tuple(failed),
        (),
        passive,
        bool(label == LABEL_PASS and passive == PASSIVE_BETTER and survives is True),
        survives,
    )


__all__ = ["CandidateEvidence", "CandidateLabel", "classify_candidate"]
