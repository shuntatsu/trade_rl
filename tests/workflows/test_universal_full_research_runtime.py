from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from trade_rl.artifacts.hashing import content_digest
from trade_rl.rl.universal_architecture import UniversalArchitectureName
from trade_rl.workflows.full_research_state import (
    FullResearchStatus,
    ResearchPhase,
    run_research_phase,
)
from trade_rl.workflows.universal_research import (
    FullResearchAlgorithm,
    UniversalFullResearchPlan,
    UniversalResearchManifest,
    build_full_research_pair_closure,
    validate_full_research_completion,
    validate_full_research_start_inputs,
)


def _digest(label: str) -> str:
    return content_digest({"label": label})


def _pairs() -> tuple[str, ...]:
    return build_full_research_pair_closure(
        algorithms=tuple(FullResearchAlgorithm),
        baseline_names=("supervised_allocator",),
        folds=(0, 1),
        seeds=(7, 11),
    )


def _manifest(*, completed_pairs: tuple[str, ...] = ()) -> UniversalResearchManifest:
    return UniversalResearchManifest(
        catalog_digest=_digest("catalog"),
        split_manifest_digest=_digest("split"),
        normalizer_digest=_digest("normalizer"),
        feature_schema_digest=_digest("features"),
        seed_manifest_digest=_digest("seeds"),
        architecture_name=UniversalArchitectureName.U_MEDIUM_DIRECT,
        checkpoint_digest=_digest("checkpoint"),
        cost_model_digest=_digest("cost"),
        required_pairs=_pairs(),
        completed_pairs=completed_pairs,
        bc_teacher_digest=_digest("bc-teacher"),
        software_identity=_digest("software"),
        universe_manifest_digest=_digest("universe"),
        normalization_fit_scope_digest=_digest("normalizer-scope"),
        observation_contract_digest=_digest("observation"),
        architecture_evidence_digest=_digest("u5-evidence"),
        zero_shot_gate_digest=_digest("zero-shot-gate"),
        paired_baseline_digest=_digest("baseline"),
    )


def _plan() -> UniversalFullResearchPlan:
    return UniversalFullResearchPlan.create(
        selected_architecture=UniversalArchitectureName.U_MEDIUM_DIRECT,
        zero_shot_gate_passed=True,
        algorithms=tuple(FullResearchAlgorithm),
    )


def test_u6_start_validation_allows_incomplete_pair_closure() -> None:
    manifest = _manifest(completed_pairs=(_pairs()[0],))

    validate_full_research_start_inputs(manifest)

    with pytest.raises(ValueError, match="missing paired deliverables"):
        validate_full_research_completion(manifest)


def test_u6_start_validation_rejects_out_of_closure_completion() -> None:
    manifest = _manifest(completed_pairs=("unexpected",))

    with pytest.raises(ValueError, match="outside the manifest closure"):
        validate_full_research_start_inputs(manifest)


def test_u6_requires_all_immutable_research_identity() -> None:
    manifest = _manifest()

    with pytest.raises(ValueError, match="normalization_fit_scope_digest"):
        validate_full_research_start_inputs(
            UniversalResearchManifest(
                **{
                    **manifest.__dict__,
                    "normalization_fit_scope_digest": "",
                }
            )
        )


def test_u6_stage_machine_binds_selected_architecture_and_authorizations(
    tmp_path: Path,
) -> None:
    from trade_rl.workflows.universal_research_runtime import UniversalResearchStages

    manifest = _manifest(completed_pairs=_pairs())
    stages = UniversalResearchStages(
        manifest=manifest,
        plan=_plan(),
        selection_authorization_digest=None,
        fresh_confirmation_digest=None,
    )
    develop = run_research_phase(
        phase=ResearchPhase.DEVELOP,
        work_root=tmp_path / "develop",
        stages=stages,
    )
    assert develop.status is FullResearchStatus.AWAITING_SELECTION_AUTHORIZATION
    assert develop.summary["manifest_digest"] == manifest.manifest_digest
    assert develop.summary["selected_architecture"] == "u_medium_direct"

    blocked_training = run_research_phase(
        phase=ResearchPhase.TRAIN_SELECTED,
        work_root=tmp_path / "blocked-training",
        stages=stages,
    )
    assert blocked_training.status is FullResearchStatus.INFRASTRUCTURE_ERROR
    assert "selection authorization" in str(blocked_training.summary["error"])

    authorized = UniversalResearchStages(
        manifest=manifest,
        plan=_plan(),
        selection_authorization_digest=_digest("selection-auth"),
        fresh_confirmation_digest=None,
    )
    trained = run_research_phase(
        phase=ResearchPhase.TRAIN_SELECTED,
        work_root=tmp_path / "trained",
        stages=authorized,
    )
    assert trained.status is FullResearchStatus.AWAITING_FRESH_CONFIRMATION

    confirmed = UniversalResearchStages(
        manifest=manifest,
        plan=_plan(),
        selection_authorization_digest=_digest("selection-auth"),
        fresh_confirmation_digest=_digest("fresh-confirmation"),
    )
    finalized = run_research_phase(
        phase=ResearchPhase.FINALIZE,
        work_root=tmp_path / "finalized",
        stages=confirmed,
    )
    assert finalized.status is FullResearchStatus.AWAITING_RELEASE_APPROVAL


def test_u6_stage_machine_rejects_architecture_identity_drift() -> None:
    from trade_rl.workflows.universal_research_runtime import UniversalResearchStages

    plan = UniversalFullResearchPlan.create(
        selected_architecture=UniversalArchitectureName.U_LARGE_DIRECT,
        zero_shot_gate_passed=True,
        algorithms=tuple(FullResearchAlgorithm),
    )
    with pytest.raises(ValueError, match="selected architecture"):
        UniversalResearchStages(
            manifest=_manifest(),
            plan=plan,
            selection_authorization_digest=None,
            fresh_confirmation_digest=None,
        )


def test_u6_cli_enters_maintained_full_research_state_machine() -> None:
    from scripts import run_full_research_experiment

    source = inspect.getsource(run_full_research_experiment.main)
    assert "run_research_phase" in source
    assert "ResearchPhase" in source
