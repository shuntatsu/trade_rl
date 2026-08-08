"""Bind Stage A historical interval evidence to Nautilus source bars."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from trade_rl.data.market import MarketDataset
from trade_rl.integrations.nautilus.event_projection import (
    ProjectedMarketEvent,
    SourceBar,
    project_bar_events,
)
from trade_rl.integrations.nautilus.historical_projection import (
    project_historical_interval_source_bars,
)
from trade_rl.simulation.funding_evidence import FundingBoundaryEvidence
from trade_rl.workflows.stage_a_execution_replay import StageAExecutionReplayArtifact
from trade_rl.workflows.stage_a_historical_interval_evidence import (
    StageAHistoricalIntervalEvidence,
    build_stage_a_historical_interval_evidence,
)


@dataclass(frozen=True, slots=True)
class StageANautilusHistoricalReplayInterval:
    """One factual Stage A interval paired with exactly its consumed source bars."""

    evidence: StageAHistoricalIntervalEvidence
    source_bars: tuple[SourceBar, ...]


def build_stage_a_nautilus_historical_replay_intervals(
    artifact: StageAExecutionReplayArtifact,
    market: MarketDataset,
    *,
    funding_evidence: Sequence[FundingBoundaryEvidence] = (),
) -> tuple[StageANautilusHistoricalReplayInterval, ...]:
    """Build replay inputs without inventing fill-level equity observations."""

    if artifact.cell_identity.dataset_id != market.dataset_id:
        raise ValueError("Stage A Nautilus historical replay dataset identity mismatch")

    evidence = build_stage_a_historical_interval_evidence(
        artifact,
        funding_evidence=funding_evidence,
    )
    return tuple(
        StageANautilusHistoricalReplayInterval(
            evidence=interval,
            source_bars=project_historical_interval_source_bars(
                market,
                start_index=interval.start_index,
                end_index=interval.end_index,
            ),
        )
        for interval in evidence
    )


def project_stage_a_nautilus_historical_interval_events(
    interval: StageANautilusHistoricalReplayInterval,
) -> tuple[ProjectedMarketEvent, ...]:
    """Project one interval with its queued target activating after the first open."""

    if not interval.source_bars:
        raise ValueError("historical replay interval must contain source bars")

    events = [
        event
        for index, bar in enumerate(interval.source_bars)
        for event in project_bar_events(bar, activate_queued_target=index == 0)
    ]
    return tuple(sorted(events, key=lambda event: event.timestamp_ns))


__all__ = [
    "StageANautilusHistoricalReplayInterval",
    "build_stage_a_nautilus_historical_replay_intervals",
    "project_stage_a_nautilus_historical_interval_events",
]
