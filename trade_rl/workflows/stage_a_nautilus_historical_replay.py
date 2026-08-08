"""Replay persisted Stage A evidence through the isolated Nautilus runtime."""

from __future__ import annotations

import math
from dataclasses import dataclass
from decimal import ROUND_HALF_EVEN, Decimal

from trade_rl.integrations.nautilus.event_projection import ProjectedMarketEvent, project_bar_events
from trade_rl.integrations.nautilus.funding_adapter import (
    CanonicalFundingLedger,
    FundingSettlementInput,
    canonicalize_funding_settlement_record,
)
from trade_rl.integrations.nautilus.historical_execution import (
    NautilusHistoricalExecutionResult,
    NautilusHistoricalTargetInterval,
)
from trade_rl.integrations.nautilus.historical_projection import (
    project_historical_interval_source_bars,
)
from trade_rl.integrations.nautilus.historical_subprocess import (
    run_historical_target_intervals_subprocess,
)
from trade_rl.integrations.nautilus.instrument import MAINTAINED_BTCUSDT_PERPETUAL
from trade_rl.simulation.execution_parity import CanonicalExecutionRecord
from trade_rl.simulation.funding_evidence import FundingBoundaryEvidence
from trade_rl.workflows.stage_a_execution_store import StageAStoredExecutionEvidence

_SETTLEMENT_CURRENCY_PRECISION = 8


@dataclass(frozen=True, slots=True)
class StageANautilusHistoricalReplayInterval:
    sequence: int
    target_exposure: float
    allocated_equity: float
    start_index: int
    end_index: int
    source_bars: tuple


@dataclass(frozen=True, slots=True)
class StageANautilusHistoricalExecutionResult:
    execution: NautilusHistoricalExecutionResult
    funding_records: tuple[CanonicalExecutionRecord, ...]


def build_stage_a_nautilus_historical_replay_intervals(
    stored: StageAStoredExecutionEvidence,
    market,
) -> tuple[StageANautilusHistoricalReplayInterval, ...]:
    artifact = stored.artifact
    evaluation_start = artifact.evaluation_range.start
    previous_end = evaluation_start
    intervals: list[StageANautilusHistoricalReplayInterval] = []
    for sequence, (action, end_index) in enumerate(
        zip(artifact.actions, artifact.transition_end_indices, strict=True), start=1
    ):
        if len(action) != 1:
            raise ValueError("Stage A Nautilus replay requires single-symbol actions")
        intervals.append(
            StageANautilusHistoricalReplayInterval(
                sequence=sequence,
                target_exposure=float(action[0]),
                allocated_equity=float(artifact.equity_curve[sequence - 1]),
                start_index=previous_end,
                end_index=end_index,
                source_bars=project_historical_interval_source_bars(
                    market,
                    start_index=previous_end,
                    end_index=end_index,
                ),
            )
        )
        previous_end = end_index
    return tuple(intervals)


def execute_stage_a_nautilus_historical_replay(
    artifact,
    market,
    *,
    funding_evidence: tuple[FundingBoundaryEvidence, ...] = (),
    no_trade_band: float = 0.05,
) -> StageANautilusHistoricalExecutionResult:
    intervals = build_stage_a_nautilus_historical_replay_intervals(artifact, market)
    historical_intervals = tuple(
        NautilusHistoricalTargetInterval(
            sequence=interval.sequence,
            target_exposure=interval.target_exposure,
            allocated_equity=interval.allocated_equity,
            source_bars=interval.source_bars,
        )
        for interval in intervals
    )
    evaluation_start = artifact.artifact.evaluation_range.start
    funding = tuple(funding_evidence)
    snapshot_timestamps_ns = tuple(
        boundary.timestamp_ns
        for boundary in funding
        if boundary.processing_index != evaluation_start
    )
    execution = run_historical_target_intervals_subprocess(
        historical_intervals,
        snapshot_timestamps_ns=snapshot_timestamps_ns,
        starting_balance=Decimal(str(artifact.artifact.equity_curve[0])),
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
