from __future__ import annotations

import pytest

from trade_rl.workflows.universal_causal_alpha_v3_admission import (
    CausalAlphaV3AdmissionRecordV2,
    evaluate_causal_alpha_v3_admission_gate,
)


def _sha(token: str) -> str:
    assert token in "0123456789abcdef"
    return token * 64


def _record(
    symbol: str,
    *,
    gross_return: float = 0.02,
    net_return: float = 0.01,
    trade_count: int = 2,
) -> CausalAlphaV3AdmissionRecordV2:
    return CausalAlphaV3AdmissionRecordV2(
        run_manifest_digest=_sha("1"),
        freeze_digest=_sha("2"),
        selection_digest=_sha("3"),
        selected_candidate_digest=_sha("4"),
        symbol=symbol,
        contract_digest=_sha("5"),
        gross_return=gross_return,
        net_return=net_return,
        turnover_per_day=0.2,
        total_execution_cost=1.0,
        trade_count=trade_count,
        maximum_drawdown=0.01,
        execution_rejection_reason_counts=(),
        risk_projection_reason_counts=(),
        hard_risk_violation=False,
    )


def test_v3_admission_inherits_common_zero_trade_rejection() -> None:
    evidence = evaluate_causal_alpha_v3_admission_gate(
        (_record("BTCUSDT", trade_count=0),)
    )

    assert evidence.passed is False
    assert evidence.total_trade_count == 0
    assert "no_meaningful_trades" in evidence.rejection_reasons
    assert len(evidence.base_admission_digest) == 64


def test_v3_admission_inherits_common_symbol_net_floor() -> None:
    evidence = evaluate_causal_alpha_v3_admission_gate(
        (
            _record("BTCUSDT", gross_return=0.02, net_return=-0.051),
            _record("ETHUSDT", gross_return=0.08, net_return=0.08),
        )
    )

    assert evidence.aggregate_net_return == pytest.approx(0.029)
    assert evidence.worst_symbol_net_return == pytest.approx(-0.051)
    assert evidence.passed is False
    assert "symbol_net_return_below_floor" in evidence.rejection_reasons
    assert len(evidence.base_admission_digest) == 64


def test_v3_admission_evidence_uses_hardened_schema() -> None:
    evidence = evaluate_causal_alpha_v3_admission_gate((_record("BTCUSDT"),))

    assert evidence.schema_version == "causal_alpha_v3_admission_evidence_v3"
    assert evidence.to_payload()["schema_version"] == (
        "causal_alpha_v3_admission_evidence_v3"
    )
