from __future__ import annotations

import inspect
from dataclasses import fields

from trade_rl.evaluation.causal_scenario_c3_report import (
    CausalScenarioFoldReport,
    build_c3_fold_report,
)


def test_fold_report_contract_requires_adverse_evidence_digest() -> None:
    field_names = {item.name for item in fields(CausalScenarioFoldReport)}
    assert "required_adverse_evidence_digest" in field_names
    parameters = inspect.signature(build_c3_fold_report).parameters
    assert "required_adverse_evidence_digest" in parameters
