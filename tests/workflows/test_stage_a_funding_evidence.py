from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from trade_rl.data.market import MarketDataset
from trade_rl.evaluation.walk_forward.folds import IndexRange
from trade_rl.simulation.funding_evidence import FundingBoundaryEvidence
from trade_rl.workflows.stage_a_funding_evidence import (
    collect_stage_a_funding_evidence,
    validate_stage_a_funding_evidence,
)
from trade_rl.workflows.stage_a_zero_shot_runner_contracts import (
    StageAEvaluationCellRequest,
)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _request(dataset_id: str) -> StageAEvaluationCellRequest:
    return StageAEvaluationCellRequest(
        plan_digest=_digest("plan"),
        evaluation_dataset_manifest_digest=_digest("manifest"),
        split="validation",
        triplet_id=_digest("triplet"),
        fold=0,
        seed=0,
        candidate_id=None,
        checkpoint_digest=None,
        dataset_id=dataset_id,
        evaluation_range=IndexRange(start=1, stop=4),
        feature_identity=_digest("features"),
        execution_identity=_digest("execution"),
        evaluation_identity=_digest("evaluation"),
    )


def _dataset() -> MarketDataset:
    n_bars = 6
    shape = (n_bars, 1)
    close = np.full(shape, 100.0, dtype=np.float64)
    mark_price = close.copy()
    mark_price[2, 0] = 120.0
    funding_rate = np.zeros(shape, dtype=np.float64)
    funding_rate[2, 0] = 0.001
    funding_due = np.zeros(shape, dtype=np.bool_)
    funding_due[2, 0] = True
    return MarketDataset(
        dataset_id=_digest("dataset"),
        symbols=("BTCUSDT",),
        timestamps=np.datetime64("2026-01-01", "ns")
        + np.arange(n_bars) * np.timedelta64(1, "h"),
        features=np.zeros((n_bars, 1, 1), dtype=np.float32),
        global_features=np.zeros((n_bars, 1), dtype=np.float32),
        open=close.copy(),
        high=np.maximum(close, mark_price),
        low=np.minimum(close, mark_price),
        close=close,
        volume=np.full(shape, 1_000.0, dtype=np.float64),
        funding_rate=funding_rate,
        tradable=np.ones(shape, dtype=np.bool_),
        feature_available=np.ones((n_bars, 1, 1), dtype=np.bool_),
        feature_names=("probe",),
        global_feature_names=("probe",),
        periods_per_year=8_760,
        feature_config_digest=_digest("features"),
        funding_due=funding_due,
        mark_price=mark_price,
        contract_multipliers=np.array([1.0], dtype=np.float64),
    )


def _boundary(dataset: MarketDataset, *, index: int = 2) -> FundingBoundaryEvidence:
    mark = float(dataset.resolved_array("mark_price")[index, 0])
    rate = 0.001
    quantity = 10.0
    funding = -(quantity * mark * rate)
    return FundingBoundaryEvidence(
        processing_index=index,
        timestamp_ns=int(dataset.timestamps[index].astype(np.int64)),
        funding_due=(True,),
        signed_quantities=(quantity,),
        mark_prices=(mark,),
        contract_multipliers=(1.0,),
        funding_rates=(rate,),
        funding_amount=funding,
        equity_before_funding=1_200.0,
        equity_after_funding=1_200.0 + funding,
    )


def test_stage_a_collects_and_revalidates_funding_boundaries() -> None:
    dataset = _dataset()
    request = _request(dataset.dataset_id)
    boundary = _boundary(dataset)
    info = {"hybrid_execution": SimpleNamespace(funding_evidence=(boundary,))}

    collected = collect_stage_a_funding_evidence(info)
    normalized = validate_stage_a_funding_evidence(
        collected,
        request=request,
        dataset=dataset,
    )

    assert normalized == (boundary,)


def test_stage_a_rejects_non_funding_values_from_environment() -> None:
    with pytest.raises(ValueError, match="invalid funding evidence"):
        collect_stage_a_funding_evidence(
            {"hybrid_execution": SimpleNamespace(funding_evidence=(object(),))}
        )


@pytest.mark.parametrize(
    ("boundary", "message"),
    [
        (
            lambda dataset: replace(
                _boundary(dataset),
                timestamp_ns=int(dataset.timestamps[2].astype(np.int64)) + 1,
            ),
            "timestamp mismatch",
        ),
        (
            lambda dataset: FundingBoundaryEvidence(
                processing_index=3,
                timestamp_ns=int(dataset.timestamps[3].astype(np.int64)),
                funding_due=(True,),
                signed_quantities=(10.0,),
                mark_prices=(100.0,),
                contract_multipliers=(1.0,),
                funding_rates=(0.001,),
                funding_amount=-1.0,
                equity_before_funding=1_000.0,
                equity_after_funding=999.0,
            ),
            "funding due mismatch",
        ),
        (
            lambda dataset: FundingBoundaryEvidence(
                processing_index=2,
                timestamp_ns=int(dataset.timestamps[2].astype(np.int64)),
                funding_due=(True,),
                signed_quantities=(10.0,),
                mark_prices=(121.0,),
                contract_multipliers=(1.0,),
                funding_rates=(0.001,),
                funding_amount=-1.21,
                equity_before_funding=1_210.0,
                equity_after_funding=1_208.79,
            ),
            "mark price mismatch",
        ),
    ],
)
def test_stage_a_rejects_funding_evidence_not_bound_to_dataset(
    boundary: Callable[[MarketDataset], FundingBoundaryEvidence],
    message: str,
) -> None:
    dataset = _dataset()
    request = _request(dataset.dataset_id)
    forged = boundary(dataset)

    with pytest.raises(ValueError, match=message):
        validate_stage_a_funding_evidence(
            (forged,),
            request=request,
            dataset=dataset,
        )


def test_stage_a_rejects_funding_evidence_at_half_open_stop() -> None:
    dataset = _dataset()
    request = _request(dataset.dataset_id)
    stop = request.evaluation_range.stop
    forged = FundingBoundaryEvidence(
        processing_index=stop,
        timestamp_ns=int(dataset.timestamps[stop].astype(np.int64)),
        funding_due=(False,),
        signed_quantities=(0.0,),
        mark_prices=(100.0,),
        contract_multipliers=(1.0,),
        funding_rates=(0.0,),
        funding_amount=0.0,
        equity_before_funding=1_000.0,
        equity_after_funding=1_000.0,
    )

    with pytest.raises(ValueError, match="outside authorized range"):
        validate_stage_a_funding_evidence(
            (forged,),
            request=request,
            dataset=dataset,
        )
