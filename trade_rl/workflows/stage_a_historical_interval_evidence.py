"""Step-bound historical equity evidence projected from Stage A replay v4."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from trade_rl.simulation.funding_evidence import FundingBoundaryEvidence
from trade_rl.workflows.stage_a_execution_replay import (
    STAGE_A_EXECUTION_REPLAY_SCHEMA_V4,
    StageAExecutionReplayArtifact,
)


@dataclass(frozen=True, slots=True)
class StageAHistoricalIntervalEvidence:
    """One factual environment-step interval and its observed equity endpoints."""

    sequence: int
    start_index: int
    end_index: int
    action: tuple[float, ...]
    equity_before: float
    equity_after: float
    funding_boundaries: tuple[FundingBoundaryEvidence, ...] = ()


def build_stage_a_historical_interval_evidence(
    artifact: StageAExecutionReplayArtifact,
    *,
    funding_evidence: Sequence[FundingBoundaryEvidence] = (),
) -> tuple[StageAHistoricalIntervalEvidence, ...]:
    """Project validated replay-v4 transition boundaries into step intervals.

    Equity is attached only to environment-step boundaries. This deliberately does
    not infer or synthesize fill-level equity between those factual observations.
    Funding at the evaluation start belongs to the first interval. Later shared
    transition boundaries belong to the interval which ends at that boundary, so a
    funding settlement is never counted in two adjacent intervals.
    """

    if artifact.schema_version != STAGE_A_EXECUTION_REPLAY_SCHEMA_V4:
        raise ValueError("Stage A historical interval evidence requires replay v4")

    evaluation_range = artifact.cell_identity.evaluation_range
    funding = tuple(funding_evidence)
    previous_funding_index: int | None = None
    for boundary in funding:
        if not isinstance(boundary, FundingBoundaryEvidence):
            raise ValueError("Stage A historical funding evidence is invalid")
        if (
            boundary.processing_index < evaluation_range.start
            or boundary.processing_index > evaluation_range.stop
        ):
            raise ValueError("Stage A historical funding evidence outside replay range")
        if (
            previous_funding_index is not None
            and boundary.processing_index <= previous_funding_index
        ):
            raise ValueError("Stage A historical funding evidence must increase")
        previous_funding_index = boundary.processing_index

    start_index = evaluation_range.start
    funding_offset = 0
    intervals: list[StageAHistoricalIntervalEvidence] = []
    for offset, end_index in enumerate(artifact.transition_end_indices):
        interval_funding: list[FundingBoundaryEvidence] = []
        while funding_offset < len(funding):
            boundary = funding[funding_offset]
            if boundary.processing_index > end_index:
                break
            if offset > 0 and boundary.processing_index <= start_index:
                raise ValueError(
                    "Stage A historical funding interval assignment overlaps"
                )
            interval_funding.append(boundary)
            funding_offset += 1
        intervals.append(
            StageAHistoricalIntervalEvidence(
                sequence=offset + 1,
                start_index=start_index,
                end_index=end_index,
                action=artifact.actions[offset],
                equity_before=artifact.equity_curve[offset],
                equity_after=artifact.equity_curve[offset + 1],
                funding_boundaries=tuple(interval_funding),
            )
        )
        start_index = end_index

    if funding_offset != len(funding):
        raise ValueError("Stage A historical funding evidence was not assigned")
    return tuple(intervals)


__all__ = [
    "StageAHistoricalIntervalEvidence",
    "build_stage_a_historical_interval_evidence",
]
