from __future__ import annotations

import inspect

from trade_rl.evaluation.experiments.bootstrap.spot_aggtrades_structural_validator import (
    build_structural_report,
)


def test_structural_report_builder_requires_exact_validator_authority() -> None:
    parameters = inspect.signature(build_structural_report).parameters
    assert "validator_head" in parameters
    assert "validator_verification_run_id" in parameters
    assert parameters["validator_head"].default is inspect.Parameter.empty
    assert parameters["validator_verification_run_id"].default is inspect.Parameter.empty
