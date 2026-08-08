"""Bind Stage A historical interval evidence to Nautilus source bars."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from decimal import ROUND_HALF_EVEN, Decimal

from trade_rl.data.market import MarketDataset
from trade_rl.integrations.nautilus.event_projection import (
    ProjectedMarketEvent,
    SourceBar,
    project_bar_events,
)
from trade_rl.integrations.nautilus.funding_adapter import (
    CanonicalFundingLedger,
    FundingSettlementInput,
    canonicalize_funding_settlement_record,
)
from trade_rl.integrations.nautilus.historical_execution import (
    NautilusHistoricalExecutionResult,
    NautilusHistoricalTargetInterval,
    run_historical_target_intervals,
)
from trade_rl.integrations.nautilus.historical_projection import (
    project_historical_interval_source_bars,
)
from trade_rl.integrations.nautilus.instrument import MAINTAINED_BTCUSDT_PERPETUAL
from trade_rl.simulation.execution_parity import CanonicalExecutionRecord
from trade_rl.simulation.funding_evidence import FundingBoundaryEvidence
from trade_rl.workflows.stage_a_execution_replay import StageAExecutionReplayArtifact
from trade_rl.workflows.stage_a_historical_interval_evidence import (
    StageAHistoricalIntervalEvidence,
    build_stage_a_historical_interval_evidence,
)

_SETTLEMENT_CURRENCY_PRECISION = 8


@dataclass(frozen=True, slots=True)
class StageANautilusHistoricalReplayInterval:
    """One factual Stage A interval paired with exactly its consumed source bars."""

    evidence: StageAHistoricalIntervalEvidence
    source_bars: tuple[SourceBar, ...]


@dataclass(frozen=True, slots=True)
class StageANautilusHistoricalExecutionResult:
    """Actual Nautilus execution plus separately canonicalized funding evidence."""

    execution: NautilusHistoricalExecutionResult
    funding_records: tuple[CanonicalExecutionRecord, ...]


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


def execute_stage_a_nautilus_historical_replay(
    artifact: StageAExecutionReplayArtifact,
    market: MarketDataset,
    *,
    funding_evidence: Sequence[FundingBoundaryEvidence] = (),
    no_trade_band: float = 0.05,
) -> StageANautilusHistoricalExecutionResult:
    """Execute factual Stage A targets and settle funding from actual positions."""

    funding = tuple(funding_evidence)
    replay_intervals = build_stage_a_nautilus_historical_replay_intervals(
        artifact,
        market,
        funding_evidence=funding,
    )
    target_intervals: list[NautilusHistoricalTargetInterval] = []
    for interval in replay_intervals:
        if len(interval.evidence.action) != 1:
            raise ValueError(
                "Stage A Nautilus historical execution requires one action value"
            )
        target_intervals.append(
            NautilusHistoricalTargetInterval(
                sequence=interval.evidence.sequence,
                target_exposure=interval.evidence.action[0],
                allocated_equity=interval.evidence.equity_before,
                source_bars=interval.source_bars,
            )
        )

    evaluation_start = artifact.cell_identity.evaluation_range.start
    snapshot_timestamps = tuple(
        boundary.timestamp_ns
        for boundary in funding
        if boundary.processing_index != evaluation_start
    )
    execution = run_historical_target_intervals(
        tuple(target_intervals),
        snapshot_timestamps_ns=snapshot_timestamps,
        starting_balance=Decimal(str(replay_intervals[0].evidence.equity_before)),
        no_trade_band=no_trade_band,
    )
    snapshot_by_timestamp = {
        snapshot.timestamp_ns: snapshot.signed_quantity
        for snapshot in execution.position_snapshots
    }

    ledger = CanonicalFundingLedger()
    funding_records: list[CanonicalExecutionRecord] = []
    spec = MAINTAINED_BTCUSDT_PERPETUAL
    for sequence, boundary in enumerate(funding, start=1):
        _require_single_instrument_funding_boundary(boundary)
        if boundary.processing_index == evaluation_start:
            candidate_quantity = Decimal("0")
        else:
            try:
                candidate_quantity = snapshot_by_timestamp[boundary.timestamp_ns]
            except KeyError as error:
                raise RuntimeError(
                    "Stage A Nautilus funding boundary lacks candidate position snapshot"
                ) from error
        factual_quantity = Decimal(str(boundary.signed_quantities[0]))
        if candidate_quantity != factual_quantity:
            raise ValueError(
                "Stage A Nautilus candidate funding position mismatch: "
                f"candidate={candidate_quantity} factual={factual_quantity}"
            )

        settlement = ledger.settle(
            FundingSettlementInput(
                instrument_id=spec.instrument_id,
                settlement_currency=spec.settlement_currency,
                currency_precision=_SETTLEMENT_CURRENCY_PRECISION,
                signed_quantity=candidate_quantity,
                settlement_price=Decimal(str(boundary.mark_prices[0])),
                contract_multiplier=Decimal(str(boundary.contract_multipliers[0])),
                funding_rate=Decimal(str(boundary.funding_rates[0])),
                boundary_ns=boundary.timestamp_ns,
            )
        )
        if not math.isclose(
            float(settlement.amount),
            boundary.funding_amount,
            rel_tol=1e-12,
            abs_tol=10 ** (-_SETTLEMENT_CURRENCY_PRECISION),
        ):
            raise ValueError("Stage A Nautilus candidate funding amount mismatch")
        record = canonicalize_funding_settlement_record(
            settlement,
            sequence=sequence,
            price_tick=Decimal(spec.price_increment),
            lot_size=Decimal(spec.size_increment),
            equity_before_minor=_currency_minor_units(
                boundary.equity_before_funding,
                field="equity_before_funding",
            ),
        )
        expected_equity_after_minor = _currency_minor_units(
            boundary.equity_after_funding,
            field="equity_after_funding",
        )
        if record.equity_minor != expected_equity_after_minor:
            raise ValueError("Stage A Nautilus candidate funding equity mismatch")
        funding_records.append(record)

    return StageANautilusHistoricalExecutionResult(
        execution=execution,
        funding_records=tuple(funding_records),
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


def _require_single_instrument_funding_boundary(
    boundary: FundingBoundaryEvidence,
) -> None:
    lengths = (
        len(boundary.funding_due),
        len(boundary.signed_quantities),
        len(boundary.mark_prices),
        len(boundary.contract_multipliers),
        len(boundary.funding_rates),
    )
    if any(length != 1 for length in lengths):
        raise ValueError(
            "Stage A Nautilus historical execution requires single-instrument funding"
        )


def _currency_minor_units(value: float, *, field: str) -> int:
    decimal_value = Decimal(str(value))
    if not decimal_value.is_finite():
        raise ValueError(f"Stage A Nautilus {field} must be finite")
    scaled = decimal_value * (Decimal(10) ** _SETTLEMENT_CURRENCY_PRECISION)
    return int(scaled.quantize(Decimal("1"), rounding=ROUND_HALF_EVEN))


__all__ = [
    "StageANautilusHistoricalExecutionResult",
    "StageANautilusHistoricalReplayInterval",
    "build_stage_a_nautilus_historical_replay_intervals",
    "execute_stage_a_nautilus_historical_replay",
    "project_stage_a_nautilus_historical_interval_events",
]
