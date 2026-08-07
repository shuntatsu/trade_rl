"""Stage A collection and dataset revalidation for funding-boundary evidence."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from trade_rl.data.market import MarketDataset
from trade_rl.simulation.funding_evidence import FundingBoundaryEvidence
from trade_rl.workflows.stage_a_zero_shot_runner_contracts import (
    StageAEvaluationCellRequest,
)


def collect_stage_a_funding_evidence(
    info: Mapping[str, object],
) -> tuple[FundingBoundaryEvidence, ...]:
    """Collect funding evidence emitted by hybrid execution objects for one step."""

    collected: list[FundingBoundaryEvidence] = []
    for key in ("hybrid_execution", "hybrid_liquidation"):
        source = info.get(key)
        raw_evidence = () if source is None else getattr(source, "funding_evidence", ())
        values = tuple(raw_evidence)
        if any(not isinstance(value, FundingBoundaryEvidence) for value in values):
            raise ValueError("Stage A environment returned invalid funding evidence")
        collected.extend(values)
    return tuple(collected)


def validate_stage_a_funding_evidence(
    evidence: Sequence[FundingBoundaryEvidence],
    *,
    request: StageAEvaluationCellRequest,
    dataset: MarketDataset,
) -> tuple[FundingBoundaryEvidence, ...]:
    """Revalidate funding evidence against the exact authorized market rows."""

    if dataset.dataset_id != request.dataset_id:
        raise ValueError("Stage A funding evidence dataset identity mismatch")

    normalized = tuple(evidence)
    previous_index: int | None = None
    previous_timestamp: int | None = None
    for position, boundary in enumerate(normalized):
        if not isinstance(boundary, FundingBoundaryEvidence):
            raise ValueError(
                f"Stage A funding evidence[{position}] is not funding evidence"
            )
        index = boundary.processing_index
        if index < request.evaluation_range.start or index > request.evaluation_range.stop:
            raise ValueError("Stage A funding evidence outside authorized range")
        if index >= dataset.n_bars:
            raise ValueError("Stage A funding evidence processing index outside dataset")
        expected_timestamp = int(
            dataset.timestamps[index].astype("datetime64[ns]").astype(np.int64)
        )
        if boundary.timestamp_ns != expected_timestamp:
            raise ValueError("Stage A funding evidence timestamp mismatch")
        if len(boundary.funding_due) != dataset.n_symbols:
            raise ValueError("Stage A funding evidence symbol count mismatch")

        expected_due = tuple(
            bool(value) for value in dataset.resolved_array("funding_due")[index]
        )
        if boundary.funding_due != expected_due:
            raise ValueError("Stage A funding evidence funding due mismatch")
        expected_marks = tuple(
            float(value) for value in dataset.resolved_array("mark_price")[index]
        )
        if boundary.mark_prices != expected_marks:
            raise ValueError("Stage A funding evidence mark price mismatch")
        expected_multipliers = tuple(
            float(value) for value in dataset.resolved_array("contract_multipliers")
        )
        if boundary.contract_multipliers != expected_multipliers:
            raise ValueError("Stage A funding evidence contract multiplier mismatch")
        expected_rates = tuple(float(value) for value in dataset.funding_rate[index])
        if boundary.funding_rates != expected_rates:
            raise ValueError("Stage A funding evidence funding rate mismatch")

        if previous_index is not None and (
            index <= previous_index
            or previous_timestamp is None
            or boundary.timestamp_ns <= previous_timestamp
        ):
            raise ValueError("Stage A funding evidence must be strictly increasing")
        previous_index = index
        previous_timestamp = boundary.timestamp_ns
    return normalized


__all__ = [
    "collect_stage_a_funding_evidence",
    "validate_stage_a_funding_evidence",
]
