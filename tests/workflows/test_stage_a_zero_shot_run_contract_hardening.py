from __future__ import annotations

import pytest

from tests.workflows.test_stage_a_zero_shot_runner import _orchestrator
from trade_rl.workflows.stage_a_zero_shot_runner_contracts import StageASealedTestRun


def test_sealed_test_run_requires_access_for_every_evidence_fold() -> None:
    orchestrator, _, _, _ = _orchestrator()
    validation_run = orchestrator.evaluate_validation()
    sealed_run = orchestrator.evaluate_sealed_test(validation_run)

    with pytest.raises(ValueError, match="access cell closure mismatch"):
        StageASealedTestRun(
            validation_run=sealed_run.validation_run,
            access_records=sealed_run.access_records[:-1],
            evidence=sealed_run.evidence,
            decision=sealed_run.decision,
        )
