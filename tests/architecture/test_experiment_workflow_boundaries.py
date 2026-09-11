from __future__ import annotations

import ast
import importlib
from pathlib import Path

import trade_rl.evaluation.experiments as experiments
from tests.architecture.imports import ImportCollector, within_module
from trade_rl.evaluation.experiments import workflow as workflow_module

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "trade_rl"
EXPERIMENTS = PACKAGE / "evaluation" / "experiments"

EXPECTED_PUBLIC_API = {
    "ArtifactIntegrityError",
    "CANDIDATE_STRATEGY_NAMES",
    "CONTROL_STRATEGY_NAMES",
    "CanonicalM2BootstrapConfig",
    "CanonicalM2BootstrapResult",
    "ContractViolationError",
    "ControlledExperimentError",
    "ControlledFactor",
    "ControlledVerification",
    "ControlledVerificationStatus",
    "EvidenceSet",
    "ExperimentBudgetExceededError",
    "ExperimentComparison",
    "ExperimentDecision",
    "ExperimentDecisionKind",
    "ExperimentDefinition",
    "ExperimentFailure",
    "FACTOR_RULES",
    "FactorRule",
    "InvalidExperimentStateError",
    "LoadedEvidenceSet",
    "ResolvedRunConfig",
    "StudyFreeze",
    "StudyFrozenError",
    "StudyOutcome",
    "StudyPlan",
    "StudySnapshot",
    "UncontrolledDeltaError",
    "bootstrap_canonical_m2_study",
    "compare_experiment",
    "create_study",
    "decide_experiment",
    "define_experiment",
    "execute_evidence_set",
    "freeze_study",
    "inspect_canonical_m2_bootstrap",
    "inspect_study",
    "load_evidence_set",
    "record_experiment_failure",
    "run_baseline",
    "run_experiment",
    "verify_controlled_delta",
    "verify_experiment",
}

CODEC_OWNED = {
    "_AnalysisBinding",
    "_expect_keys",
    "_as_dict",
    "_as_list",
    "_as_string",
    "_as_optional_string",
    "_as_int",
    "_as_float",
    "_as_string_tuple",
    "_as_int_tuple",
    "_as_datetime",
    "_resolved_from_payload",
    "_study_plan_from_payload",
    "_candidate_config_payload",
    "_candidate_config_from_resolved",
    "_definition_from_payload",
    "_verification_from_payload",
    "_comparison_from_payload",
    "_decision_from_payload",
    "_failure_from_payload",
    "_freeze_from_payload",
    "_resolved_contract",
    "_semantic_without_seed",
    "_semantic_payload_without_seed",
    "_baseline_context_digest",
    "_analysis_binding",
    "_analysis_binding_from_payload",
}

INSPECTION_OWNED = {
    "StudySnapshot",
    "_ExperimentState",
    "_StudyState",
    "_load_evidence_node",
    "_experiment_dir",
    "_find_evidence",
    "_read_plan",
    "_reconstruct",
    "_experiment_state",
    "inspect_study",
}

WORKFLOW_COMMANDS = {
    "create_study",
    "run_baseline",
    "define_experiment",
    "run_experiment",
    "verify_experiment",
    "compare_experiment",
    "decide_experiment",
    "record_experiment_failure",
    "freeze_study",
}


def _defined_names(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }


def test_experiment_boundary_modules_exist() -> None:
    assert (EXPERIMENTS / "codec.py").is_file()
    assert (EXPERIMENTS / "inspection.py").is_file()


def test_workflow_owns_commands_not_codec_or_reconstruction() -> None:
    names = _defined_names(EXPERIMENTS / "workflow.py")
    assert WORKFLOW_COMMANDS <= names
    assert names.isdisjoint(CODEC_OWNED | INSPECTION_OWNED)


def test_codec_and_inspection_own_their_responsibilities() -> None:
    codec_names = _defined_names(EXPERIMENTS / "codec.py")
    inspection_names = _defined_names(EXPERIMENTS / "inspection.py")
    assert CODEC_OWNED <= codec_names
    assert INSPECTION_OWNED <= inspection_names


def test_read_side_does_not_depend_back_on_mutation_workflow() -> None:
    collector = ImportCollector(PACKAGE)
    for name in ("codec.py", "inspection.py"):
        imports = collector.collect(EXPERIMENTS / name)
        assert not any(
            within_module(value, "trade_rl.evaluation.experiments.workflow")
            for value in imports
        ), name


def test_workflow_keeps_failure_injection_seams() -> None:
    for name in (
        "resolve_candidate_run_spec",
        "verify_controlled_delta",
        "analyze_evidence_set",
    ):
        assert callable(getattr(workflow_module, name, None)), name


def test_inspection_exports_preserve_existing_module_identity() -> None:
    inspection = importlib.import_module("trade_rl.evaluation.experiments.inspection")
    assert workflow_module.StudySnapshot is inspection.StudySnapshot
    assert workflow_module.inspect_study is inspection.inspect_study
    assert experiments.StudySnapshot is inspection.StudySnapshot
    assert experiments.inspect_study is inspection.inspect_study


def test_experiment_package_public_api_is_unchanged() -> None:
    assert set(experiments.__all__) == EXPECTED_PUBLIC_API
