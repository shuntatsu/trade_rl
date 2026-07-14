from __future__ import annotations

from datetime import UTC, datetime

from trade_rl.domain.evaluation import GateCheck
from trade_rl.evaluation.gates import resolve_gate

DATASET_ID = "a" * 64
EVALUATION_DIGEST = "b" * 64


def test_optional_failure_does_not_block_gate() -> None:
    checks = (
        GateCheck(name="mandatory", passed=True, mandatory=True),
        GateCheck(name="advisory", passed=False, mandatory=False),
    )

    result = resolve_gate(
        checks,
        dataset_id=DATASET_ID,
        selected_policy_digest=None,
        evaluation_digest=EVALUATION_DIGEST,
        decided_at=datetime(2026, 7, 13, tzinfo=UTC),
    )

    assert result.passed is True
    assert result.failed_mandatory_checks == ()
    assert result.dataset_id == DATASET_ID
    assert result.evaluation_digest == EVALUATION_DIGEST


def test_mandatory_failure_blocks_gate() -> None:
    checks = (
        GateCheck(name="return_positive", passed=True, mandatory=True),
        GateCheck(name="significant", passed=False, mandatory=True),
    )

    result = resolve_gate(
        checks,
        dataset_id=DATASET_ID,
        selected_policy_digest="c" * 64,
        evaluation_digest=EVALUATION_DIGEST,
        decided_at=datetime(2026, 7, 13, tzinfo=UTC),
    )

    assert result.passed is False
    assert tuple(check.name for check in result.failed_mandatory_checks) == (
        "significant",
    )
    assert result.selected_policy_digest == "c" * 64


def test_release_eligible_gate_requires_metric_evidence() -> None:
    import pytest

    check = GateCheck(name="return_positive", passed=True)
    with pytest.raises(ValueError, match="evidence"):
        resolve_gate(
            (check,),
            dataset_id="a" * 64,
            selected_policy_digest=None,
            evaluation_digest="b" * 64,
            decided_at=datetime(2026, 7, 13, tzinfo=UTC),
            require_evidence=True,
        )


def test_metric_gate_derives_passed_from_comparator() -> None:
    check = GateCheck.from_metric(
        name="return_positive",
        metric_name="total_return",
        observed_value=0.1,
        comparator=">",
        threshold=0.0,
        evidence_digest="c" * 64,
        implementation_digest="d" * 64,
    )
    assert check.passed
