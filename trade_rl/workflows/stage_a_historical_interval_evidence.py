"""Step-bound historical equity evidence projected from Stage A replay v4."""

from __future__ import annotations

from dataclasses import dataclass

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
    equity_before: float
    equity_after: float


def build_stage_a_historical_interval_evidence(
    artifact: StageAExecutionReplayArtifact,
) -> tuple[StageAHistoricalIntervalEvidence, ...]:
    """Project validated replay-v4 transition boundaries into step intervals.

    Equity is attached only to environment-step boundaries. This deliberately does
    not infer or synthesize fill-level equity between those factual observations.
    """

    if artifact.schema_version != STAGE_A_EXECUTION_REPLAY_SCHEMA_V4:
        raise ValueError("Stage A historical interval evidence requires replay v4")

    start_index = artifact.cell_identity.evaluation_range.start
    intervals: list[StageAHistoricalIntervalEvidence] = []
    for offset, end_index in enumerate(artifact.transition_end_indices):
        intervals.append(
            StageAHistoricalIntervalEvidence(
                sequence=offset + 1,
                start_index=start_index,
                end_index=end_index,
                equity_before=artifact.equity_curve[offset],
                equity_after=artifact.equity_curve[offset + 1],
            )
        )
        start_index = end_index
    return tuple(intervals)


__all__ = [
    "StageAHistoricalIntervalEvidence",
    "build_stage_a_historical_interval_evidence",
]
