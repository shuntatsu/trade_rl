from __future__ import annotations

from datetime import UTC, datetime

from trade_rl.domain.evaluation import GateCheck
from trade_rl.evaluation.gates import resolve_gate


def test_optional_failure_does_not_block_gate() -> None:
    checks = (
        GateCheck(name="mandatory", passed=True, mandatory=True),
        GateCheck(name="advisory", passed=False, mandatory=False),
    )

    result = resolve_gate(
        checks,
        decided_at=datetime(2026, 7, 13, tzinfo=UTC),
    )

    assert result.passed is True
    assert result.failed_mandatory_checks == ()


def test_mandatory_failure_blocks_gate() -> None:
    checks = (
        GateCheck(name="return_positive", passed=True, mandatory=True),
        GateCheck(name="significant", passed=False, mandatory=True),
    )

    result = resolve_gate(
        checks,
        decided_at=datetime(2026, 7, 13, tzinfo=UTC),
    )

    assert result.passed is False
    assert tuple(check.name for check in result.failed_mandatory_checks) == (
        "significant",
    )
