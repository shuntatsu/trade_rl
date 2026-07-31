"""Deterministic Stage A zero-shot aggregation, selection, and sealed-test gates."""

from trade_rl.evaluation._stage_a_zero_shot_gate_compute import (
    evaluate_stage_a_sealed_test,
    select_stage_a_validation_candidate,
    summarize_stage_a_candidate,
)
from trade_rl.evaluation._stage_a_zero_shot_gate_decisions import (
    StageASealedTestDecision,
    StageAValidationSelection,
)
from trade_rl.evaluation._stage_a_zero_shot_gate_io import (
    load_stage_a_sealed_test_decision,
    load_stage_a_validation_selection,
    write_stage_a_sealed_test_decision,
    write_stage_a_validation_selection,
)
from trade_rl.evaluation._stage_a_zero_shot_gate_values import (
    STAGE_A_CANDIDATE_SUMMARY_SCHEMA,
    STAGE_A_SEALED_TEST_DECISION_SCHEMA,
    STAGE_A_VALIDATION_SELECTION_SCHEMA,
    StageACandidateSummary,
)

__all__ = [
    "STAGE_A_CANDIDATE_SUMMARY_SCHEMA",
    "STAGE_A_SEALED_TEST_DECISION_SCHEMA",
    "STAGE_A_VALIDATION_SELECTION_SCHEMA",
    "StageACandidateSummary",
    "StageASealedTestDecision",
    "StageAValidationSelection",
    "evaluate_stage_a_sealed_test",
    "load_stage_a_sealed_test_decision",
    "load_stage_a_validation_selection",
    "select_stage_a_validation_candidate",
    "summarize_stage_a_candidate",
    "write_stage_a_sealed_test_decision",
    "write_stage_a_validation_selection",
]
