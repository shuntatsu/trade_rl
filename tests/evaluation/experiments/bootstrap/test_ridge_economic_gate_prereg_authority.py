from __future__ import annotations

from trade_rl.evaluation.experiments.bootstrap.ridge_economic_gate_prereg import (
    canonical_ridge_economic_gate_protocol,
)


def test_protocol_binds_pre_result_diagnosis_and_exact_successor_plan_bytes() -> None:
    protocol = canonical_ridge_economic_gate_protocol()

    assert protocol.diagnosis_report_digest == (
        "9b300bf8c6d6ac4b8230a6179c4988d8c7bac2f3988e9c0d823a9ed95361fef0"
    )
    assert protocol.diagnosis_verification_artifact_id == 10335845094
    assert protocol.diagnosis_verification_artifact_digest == (
        "e722c27e2b3903ae91ad81b1d244fdbb85e7deba093e0a0db0dcaf644a05d6e9"
    )
    assert protocol.successor_plan_sha256 == (
        "bfa2fcb307773f5384d7dcb884444164d6d3b575373d8f7dc1c810a61bf4c820"
    )
    assert protocol.successor_plan_sha256 == protocol.successor_study_digest
