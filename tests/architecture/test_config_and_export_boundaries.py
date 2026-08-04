from __future__ import annotations

import ast
from pathlib import Path

from tests.architecture.import_references import scan_import_references

ROOT = Path(__file__).resolve().parents[2]


def _defined_names(path: Path) -> frozenset[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return frozenset(
        node.name
        for node in tree.body
        if isinstance(node, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef)
    )


def _import_targets(path: Path, *, module_name: str) -> frozenset[str]:
    return frozenset(
        reference.target
        for reference in scan_import_references(path, module_name=module_name)
        if reference.target is not None
    )


def _imports_prefix(targets: frozenset[str], prefix: str) -> bool:
    return any(
        target == prefix or target.startswith(f"{prefix}.") for target in targets
    )


def test_training_run_config_contract_lives_below_workflows() -> None:
    contract = ROOT / "trade_rl/rl/training_run_config.py"
    workflow = ROOT / "trade_rl/workflows/training_run.py"
    studio = ROOT / "trade_rl/studio/config_catalog.py"

    assert contract.is_file()
    assert "TrainingRunConfig" in _defined_names(contract)

    workflow_targets = _import_targets(
        workflow,
        module_name="trade_rl.workflows.training_run",
    )
    studio_targets = _import_targets(
        studio,
        module_name="trade_rl.studio.config_catalog",
    )

    assert "trade_rl.rl.training_run_config.TrainingRunConfig" in workflow_targets
    assert (
        "trade_rl.rl.training_run_config._signal_artifact_digest"
        not in workflow_targets
    )
    assert "trade_rl.rl.training_run_config.TrainingRunConfig" in studio_targets
    assert not _imports_prefix(studio_targets, "trade_rl.workflows")


def test_generic_config_field_validation_lives_in_domain() -> None:
    domain_helper = ROOT / "trade_rl/domain/config_fields.py"
    compatibility = ROOT / "trade_rl/workflows/config_fields.py"

    assert domain_helper.is_file()
    assert {
        "require_dataclass_fields",
        "require_exact_fields",
    } <= _defined_names(domain_helper)

    compatibility_targets = _import_targets(
        compatibility,
        module_name="trade_rl.workflows.config_fields",
    )
    assert {
        "trade_rl.domain.config_fields.require_dataclass_fields",
        "trade_rl.domain.config_fields.require_exact_fields",
    } <= compatibility_targets


def test_selection_authorization_lives_in_release_with_workflow_facade() -> None:
    release_contract = ROOT / "trade_rl/release/selection_authorization.py"
    compatibility = ROOT / "trade_rl/workflows/selection_authorization.py"
    studio = ROOT / "trade_rl/studio/evidence.py"

    assert release_contract.is_file()
    assert {
        "SelectionAuthorization",
        "SelectionProposal",
        "load_selection_authorization",
        "load_selection_proposal",
        "write_selection_authorization",
        "write_selection_proposal",
    } <= _defined_names(release_contract)

    compatibility_targets = _import_targets(
        compatibility,
        module_name="trade_rl.workflows.selection_authorization",
    )
    assert _imports_prefix(
        compatibility_targets,
        "trade_rl.release.selection_authorization",
    )

    studio_targets = _import_targets(
        studio,
        module_name="trade_rl.studio.evidence",
    )
    assert _imports_prefix(
        studio_targets,
        "trade_rl.release.selection_authorization",
    )
    assert not _imports_prefix(studio_targets, "trade_rl.workflows")


def test_structured_policy_contract_is_neutral_and_serving_owned() -> None:
    contract = ROOT / "trade_rl/artifacts/structured_policy_contract.py"
    exporter = ROOT / "trade_rl/rl/structured_export.py"
    serving_paths = (
        (
            ROOT / "trade_rl/serving/policy_loader.py",
            "trade_rl.serving.policy_loader",
        ),
        (
            ROOT / "trade_rl/serving/structured_policy.py",
            "trade_rl.serving.structured_policy",
        ),
    )

    assert contract.is_file()
    assert {
        "StructuredExportManifest",
        "StructuredInputSpec",
    } <= _defined_names(contract)

    contract_targets = _import_targets(
        contract,
        module_name="trade_rl.artifacts.structured_policy_contract",
    )
    assert not _imports_prefix(contract_targets, "torch")
    assert not _imports_prefix(contract_targets, "trade_rl.rl")

    exporter_targets = _import_targets(
        exporter,
        module_name="trade_rl.rl.structured_export",
    )
    assert _imports_prefix(
        exporter_targets,
        "trade_rl.artifacts.structured_policy_contract",
    )

    for path, module_name in serving_paths:
        targets = _import_targets(path, module_name=module_name)
        assert _imports_prefix(
            targets,
            "trade_rl.artifacts.structured_policy_contract",
        )
        assert not _imports_prefix(targets, "trade_rl.rl.structured_export")


def test_future_stage_b_market_roles_are_explicit_but_not_claimed_complete() -> None:
    architecture = (ROOT / "docs/ARCHITECTURE.md").read_text(encoding="utf-8")
    status = (ROOT / "docs/RESEARCH_STATUS.md").read_text(encoding="utf-8")

    for text in (architecture, status):
        assert "SpotLongBook: FUTURE_LONG_ONLY_ROLE" in text
        assert "USDSMShortBook: FUTURE_SHORT_ONLY_ROLE" in text
        assert "StageBSpotFuturesGeneralization: NOT_IMPLEMENTED" in text
