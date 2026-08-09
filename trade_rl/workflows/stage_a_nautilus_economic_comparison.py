"""Exact economic normalization for persisted Stage A Nautilus differentials."""

from __future__ import annotations

from dataclasses import dataclass

from trade_rl.workflows.stage_a_nautilus_historical_differential import (
    StageANautilusHistoricalDifferentialEvidence,
)


@dataclass(frozen=True, slots=True)
class StageANautilusHistoricalEconomicClosure:
    """One runtime's final equity and execution-cost burden in minor units."""

    final_equity_minor: int
    execution_cost_minor: int

    def __post_init__(self) -> None:
        _non_negative_int(self.final_equity_minor, field="final_equity_minor")
        _non_negative_int(self.execution_cost_minor, field="execution_cost_minor")


@dataclass(frozen=True, slots=True)
class StageANautilusHistoricalEconomicEvidence:
    """Cost-normalized economics gated by strict structural and funding parity."""

    replay_digest: str
    funding_matches: bool
    structural_passed: bool
    legacy_cost_neutral_equity_minor: int
    candidate_cost_neutral_equity_minor: int
    execution_cost_representation_delta_minor: int
    normalized_equity_delta_minor: int
    economic_passed: bool


def compare_stage_a_nautilus_historical_economics(
    *,
    structural: StageANautilusHistoricalDifferentialEvidence,
    legacy: StageANautilusHistoricalEconomicClosure,
    candidate: StageANautilusHistoricalEconomicClosure,
) -> StageANautilusHistoricalEconomicEvidence:
    """Normalize execution-cost representation without relaxing factual parity gates."""

    if not isinstance(structural, StageANautilusHistoricalDifferentialEvidence):
        raise TypeError("structural evidence has an invalid type")
    if not isinstance(legacy, StageANautilusHistoricalEconomicClosure):
        raise TypeError("legacy economic closure has an invalid type")
    if not isinstance(candidate, StageANautilusHistoricalEconomicClosure):
        raise TypeError("candidate economic closure has an invalid type")

    legacy_cost_neutral_equity_minor = (
        legacy.final_equity_minor + legacy.execution_cost_minor
    )
    candidate_cost_neutral_equity_minor = (
        candidate.final_equity_minor + candidate.execution_cost_minor
    )
    normalized_equity_delta_minor = (
        candidate_cost_neutral_equity_minor - legacy_cost_neutral_equity_minor
    )
    execution_cost_representation_delta_minor = (
        candidate.execution_cost_minor - legacy.execution_cost_minor
    )
    economic_passed = (
        structural.structural_passed
        and structural.funding_matches
        and normalized_equity_delta_minor == 0
    )

    return StageANautilusHistoricalEconomicEvidence(
        replay_digest=structural.replay_digest,
        funding_matches=structural.funding_matches,
        structural_passed=structural.structural_passed,
        legacy_cost_neutral_equity_minor=legacy_cost_neutral_equity_minor,
        candidate_cost_neutral_equity_minor=candidate_cost_neutral_equity_minor,
        execution_cost_representation_delta_minor=(
            execution_cost_representation_delta_minor
        ),
        normalized_equity_delta_minor=normalized_equity_delta_minor,
        economic_passed=economic_passed,
    )


def _non_negative_int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return value


__all__ = [
    "StageANautilusHistoricalEconomicClosure",
    "StageANautilusHistoricalEconomicEvidence",
    "compare_stage_a_nautilus_historical_economics",
]
