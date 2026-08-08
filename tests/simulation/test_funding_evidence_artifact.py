from __future__ import annotations

from dataclasses import replace

import pytest

import trade_rl.simulation.funding_evidence as funding_evidence
from trade_rl.simulation.funding_evidence import FundingBoundaryEvidence

_DATASET_ID = "a" * 64
_EXECUTION_POLICY_DIGEST = "b" * 64


def _boundary(
    *,
    processing_index: int = 10,
    timestamp_ns: int = 1_000,
    mark_price: float = 120.0,
    funding_rate: float = 0.001,
    funding_amount: float = -1.2,
    equity_before: float = 1_200.0,
) -> FundingBoundaryEvidence:
    return FundingBoundaryEvidence(
        processing_index=processing_index,
        timestamp_ns=timestamp_ns,
        funding_due=(True,),
        signed_quantities=(10.0,),
        mark_prices=(mark_price,),
        contract_multipliers=(1.0,),
        funding_rates=(funding_rate,),
        funding_amount=funding_amount,
        equity_before_funding=equity_before,
        equity_after_funding=equity_before + funding_amount,
    )


def test_funding_evidence_artifact_round_trips_canonically() -> None:
    artifact = funding_evidence.build_funding_evidence_artifact(
        dataset_id=_DATASET_ID,
        execution_policy_digest=_EXECUTION_POLICY_DIGEST,
        symbol_count=1,
        boundaries=(
            _boundary(),
            _boundary(
                processing_index=20,
                timestamp_ns=2_000,
                mark_price=110.0,
                funding_rate=0.002,
                funding_amount=-2.2,
                equity_before=1_198.8,
            ),
        ),
    )

    loaded = funding_evidence.load_funding_evidence_artifact_bytes(artifact.raw_bytes)

    assert loaded == artifact
    assert loaded.digest == artifact.digest
    assert loaded.boundary_count == 2
    assert loaded.to_mapping()["schema_version"] == (
        "execution_funding_boundary_artifact_v1"
    )


def test_empty_funding_evidence_is_explicit_and_identity_bound() -> None:
    artifact = funding_evidence.build_funding_evidence_artifact(
        dataset_id=_DATASET_ID,
        execution_policy_digest=_EXECUTION_POLICY_DIGEST,
        symbol_count=1,
        boundaries=(),
    )

    assert artifact.boundary_count == 0
    assert artifact.to_mapping()["boundaries"] == ()
    assert len(artifact.digest) == 64


def test_funding_evidence_artifact_digest_changes_with_economics() -> None:
    original = funding_evidence.build_funding_evidence_artifact(
        dataset_id=_DATASET_ID,
        execution_policy_digest=_EXECUTION_POLICY_DIGEST,
        symbol_count=1,
        boundaries=(_boundary(),),
    )
    changed_boundary = _boundary(
        mark_price=130.0,
        funding_amount=-1.3,
        equity_before=1_300.0,
    )
    changed = funding_evidence.build_funding_evidence_artifact(
        dataset_id=_DATASET_ID,
        execution_policy_digest=_EXECUTION_POLICY_DIGEST,
        symbol_count=1,
        boundaries=(changed_boundary,),
    )

    assert changed.digest != original.digest


def test_funding_evidence_artifact_rejects_boundary_regression() -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        funding_evidence.build_funding_evidence_artifact(
            dataset_id=_DATASET_ID,
            execution_policy_digest=_EXECUTION_POLICY_DIGEST,
            symbol_count=1,
            boundaries=(
                _boundary(processing_index=20, timestamp_ns=2_000),
                _boundary(processing_index=10, timestamp_ns=1_000),
            ),
        )


def test_funding_evidence_artifact_rejects_symbol_vector_mismatch() -> None:
    mismatched = replace(
        _boundary(),
        funding_due=(True, False),
        signed_quantities=(10.0, 0.0),
        mark_prices=(120.0, 50.0),
        contract_multipliers=(1.0, 1.0),
        funding_rates=(0.001, 0.0),
    )

    with pytest.raises(ValueError, match="symbol_count"):
        funding_evidence.build_funding_evidence_artifact(
            dataset_id=_DATASET_ID,
            execution_policy_digest=_EXECUTION_POLICY_DIGEST,
            symbol_count=1,
            boundaries=(mismatched,),
        )
