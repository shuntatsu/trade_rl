from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

GateState = Literal["passed", "failed", "skipped", "not_required"]


@dataclass(frozen=True)
class ReleaseEligibility:
    """Immutable release classification derived from resolved pipeline state."""

    eligible: bool
    forced: bool
    skipped_gates: tuple[str, ...]
    optimization_steps_skipped: tuple[str, ...]
    sealed_holdout_used: bool
    required_gates: dict[str, GateState]

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation for bundle metadata."""

        return asdict(self)


def derive_release_eligibility(
    *,
    forced: bool,
    skip_p0: bool,
    skip_pbt: bool,
    skip_wf: bool,
    skip_gate: bool,
    sealed_holdout_used: bool,
    p0_passed: bool,
    walk_forward_passed: bool,
    gate2_passed: bool,
    significance_passed: bool | None,
) -> ReleaseEligibility:
    """Classify whether a resolved pipeline run may create a release candidate."""

    skipped_gates = tuple(
        name
        for name, skipped in (
            ("p0", skip_p0),
            ("walk_forward", skip_wf),
            ("gate2", skip_gate),
        )
        if skipped
    )
    required_gates: dict[str, GateState] = {
        "p0": "skipped" if skip_p0 else "passed" if p0_passed else "failed",
        "walk_forward": (
            "skipped" if skip_wf else "passed" if walk_forward_passed else "failed"
        ),
        "gate2": "skipped" if skip_gate else "passed" if gate2_passed else "failed",
        "significance": (
            "not_required"
            if significance_passed is None
            else "passed"
            if significance_passed
            else "failed"
        ),
    }
    eligible = (
        not forced
        and not skipped_gates
        and sealed_holdout_used
        and all(
            state in {"passed", "not_required"} for state in required_gates.values()
        )
    )
    return ReleaseEligibility(
        eligible=eligible,
        forced=forced,
        skipped_gates=skipped_gates,
        optimization_steps_skipped=("pbt",) if skip_pbt else (),
        sealed_holdout_used=sealed_holdout_used,
        required_gates=required_gates,
    )
