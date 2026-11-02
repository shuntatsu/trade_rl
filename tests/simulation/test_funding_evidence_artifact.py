from __future__ import annotations

from dataclasses import replace

import pytest

import trade_rl.simulation.diagnostics.funding as funding_evidence
from trade_rl.simulation.diagnostics.funding import (
    FUNDING_BOUNDARY_ARTIFACT_SCHEMA_V2,
    FundingBoundaryEvidence,
)

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


def test_v2_funding_evidence_preserves_event_price_rates_and_amount_oracle() -> None:
    boundary = FundingBoundaryEvidence(
        processing_index=10,
        timestamp_ns=1_000,
        funding_due=(True, True),
        signed_quantities=(10.0, -4.0),
        mark_prices=(120.0, 80.0),
        contract_multipliers=(2.0, 3.0),
        funding_rates=(0.001, -0.002),
        funding_amount=-4.3,
        equity_before_funding=1_200.0,
        equity_after_funding=1_195.7,
        funding_price_rates=(0.125, -0.15),
    )
    artifact = funding_evidence.build_funding_evidence_artifact(
        dataset_id=_DATASET_ID,
        execution_policy_digest=_EXECUTION_POLICY_DIGEST,
        symbol_count=2,
        boundaries=(boundary,),
    )

    restored = funding_evidence.load_funding_evidence_artifact_bytes(artifact.raw_bytes)

    assert artifact.schema_version == FUNDING_BOUNDARY_ARTIFACT_SCHEMA_V2
    assert restored == artifact
    assert restored.boundaries[0].funding_price_rates == pytest.approx((0.125, -0.15))
    assert restored.boundaries[0].funding_rates == pytest.approx((0.001, -0.002))


def test_funding_evidence_schema_rejects_missing_or_mixed_event_price_rates() -> None:
    boundary = FundingBoundaryEvidence(
        processing_index=10,
        timestamp_ns=1_000,
        funding_due=(True,),
        signed_quantities=(10.0,),
        mark_prices=(120.0,),
        contract_multipliers=(1.0,),
        funding_rates=(0.001,),
        funding_amount=-1.25,
        equity_before_funding=1_200.0,
        equity_after_funding=1_198.75,
        funding_price_rates=(0.125,),
    )
    v1_mapping = boundary.to_mapping()
    del v1_mapping["funding_price_rates"]
    with pytest.raises(ValueError, match="mark notional"):
        FundingBoundaryEvidence.from_mapping(v1_mapping)

    legacy = FundingBoundaryEvidence(
        processing_index=11,
        timestamp_ns=2_000,
        funding_due=(True,),
        signed_quantities=(10.0,),
        mark_prices=(120.0,),
        contract_multipliers=(1.0,),
        funding_rates=(0.001,),
        funding_amount=-1.2,
        equity_before_funding=1_198.75,
        equity_after_funding=1_197.55,
    )
    with pytest.raises(ValueError, match="requires event-price rates"):
        funding_evidence.FundingEvidenceArtifact(
            dataset_id=_DATASET_ID,
            execution_policy_digest=_EXECUTION_POLICY_DIGEST,
            symbol_count=1,
            boundaries=(boundary, legacy),
            schema_version=FUNDING_BOUNDARY_ARTIFACT_SCHEMA_V2,
        )


def test_v2_funding_evidence_rejects_missing_event_price_rates() -> None:
    boundary = FundingBoundaryEvidence(
        processing_index=10,
        timestamp_ns=1_000,
        funding_due=(True,),
        signed_quantities=(10.0,),
        mark_prices=(120.0,),
        contract_multipliers=(1.0,),
        funding_rates=(0.001,),
        funding_amount=-1.2,
        equity_before_funding=1_200.0,
        equity_after_funding=1_198.8,
    )
    v2_mapping: dict[str, object] = {
        "boundaries": (boundary.to_mapping(),),
        "dataset_id": _DATASET_ID,
        "execution_policy_digest": _EXECUTION_POLICY_DIGEST,
        "schema_version": FUNDING_BOUNDARY_ARTIFACT_SCHEMA_V2,
        "symbol_count": 1,
    }

    with pytest.raises(ValueError, match="v2 requires event-price rates"):
        funding_evidence.FundingEvidenceArtifact.from_mapping(v2_mapping)


def test_v2_funding_evidence_rejects_tampered_event_price_rate() -> None:
    boundary = FundingBoundaryEvidence(
        processing_index=10,
        timestamp_ns=1_000,
        funding_due=(True,),
        signed_quantities=(10.0,),
        mark_prices=(120.0,),
        contract_multipliers=(1.0,),
        funding_rates=(0.001,),
        funding_amount=-1.25,
        equity_before_funding=1_200.0,
        equity_after_funding=1_198.75,
        funding_price_rates=(0.125,),
    )
    boundary_mapping = boundary.to_mapping()
    boundary_mapping["funding_price_rates"] = (0.126,)
    v2_mapping: dict[str, object] = {
        "boundaries": (boundary_mapping,),
        "dataset_id": _DATASET_ID,
        "execution_policy_digest": _EXECUTION_POLICY_DIGEST,
        "schema_version": FUNDING_BOUNDARY_ARTIFACT_SCHEMA_V2,
        "symbol_count": 1,
    }

    with pytest.raises(ValueError, match="event-price rates"):
        funding_evidence.FundingEvidenceArtifact.from_mapping(v2_mapping)


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


def test_empty_event_price_artifact_can_retain_v2_schema_explicitly() -> None:
    # With no due bars, the boundary vector cannot reveal which funding
    # semantics the executor bound into the source policy digest.
    artifact = funding_evidence.build_funding_evidence_artifact(
        dataset_id=_DATASET_ID,
        execution_policy_digest=_EXECUTION_POLICY_DIGEST,
        symbol_count=1,
        boundaries=(),
        schema_version=FUNDING_BOUNDARY_ARTIFACT_SCHEMA_V2,
    )
    assert artifact.schema_version == FUNDING_BOUNDARY_ARTIFACT_SCHEMA_V2
    assert artifact.boundary_count == 0
    assert (
        funding_evidence.load_funding_evidence_artifact_bytes(artifact.raw_bytes)
        == artifact
    )
