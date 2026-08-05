from __future__ import annotations

import ast
from pathlib import Path

from tests.architecture.import_references import (
    cross_package_private_usage_violations,
    module_name_from_path,
    scan_import_references,
)

ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = ROOT / "trade_rl"


def _tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _defined_names(path: Path) -> frozenset[str]:
    names: set[str] = set()
    for node in _tree(path).body:
        if isinstance(node, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            names.update(
                target.id for target in node.targets if isinstance(target, ast.Name)
            )
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
    return frozenset(names)


def _class_methods(path: Path, class_name: str) -> frozenset[str]:
    for node in _tree(path).body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return frozenset(
                item.name
                for item in node.body
                if isinstance(item, ast.FunctionDef | ast.AsyncFunctionDef)
            )
    raise AssertionError(f"class not found: {class_name}")


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


def _cross_package_private_imports() -> tuple[str, ...]:
    violations: list[str] = []
    for source_path in sorted(PACKAGE_ROOT.rglob("*.py")):
        module_name = module_name_from_path(
            source_path,
            package_root=PACKAGE_ROOT,
            root_package="trade_rl",
        )
        violations.extend(
            cross_package_private_usage_violations(
                source_path,
                module_name=module_name,
                root_package="trade_rl",
            )
        )
    return tuple(sorted(violations))


def test_training_environment_contract_is_public_and_framework_neutral() -> None:
    path = PACKAGE_ROOT / "rl/training_environment_contract.py"
    assert {
        "training_environment_identity",
        "validate_training_environment",
    } <= _defined_names(path)
    targets = _import_targets(
        path,
        module_name="trade_rl.rl.training_environment_contract",
    )
    for forbidden in (
        "gymnasium",
        "stable_baselines3",
        "sb3_contrib",
        "torch",
        "trade_rl.integrations",
        "trade_rl.workflows",
    ):
        assert not _imports_prefix(targets, forbidden)


def test_sb3_training_uses_only_the_public_environment_contract() -> None:
    path = PACKAGE_ROOT / "integrations/sb3_training.py"
    targets = _import_targets(path, module_name="trade_rl.integrations.sb3_training")
    assert {
        "trade_rl.rl.training_environment_contract.training_environment_identity",
        "trade_rl.rl.training_environment_contract.validate_training_environment",
    } <= targets
    training_source = (PACKAGE_ROOT / "rl/training.py").read_text(encoding="utf-8")
    assert "_environment_identity =" not in training_source
    assert "_validate_training_environment =" not in training_source


def test_production_cross_package_imports_never_target_private_symbols() -> None:
    assert _cross_package_private_imports() == ()


def test_sealed_test_persistence_has_an_explicit_dedicated_adapter() -> None:
    postgres = PACKAGE_ROOT / "catalog/postgres.py"
    postgres_methods = _class_methods(postgres, "PostgresArtifactCatalog")
    assert "reserve_sealed_test_access" not in postgres_methods
    assert "database_url" not in postgres_methods
    assert "connection_factory" not in postgres_methods

    sealed_store = PACKAGE_ROOT / "catalog/postgres_sealed_test.py"
    sealed_methods = _class_methods(
        sealed_store,
        "PostgresSealedTestReservationStore",
    )
    assert "reserve_sealed_test_access" in sealed_methods
    assert "migrate" not in sealed_methods
    ledger_targets = _import_targets(
        PACKAGE_ROOT / "catalog/sealed_test.py",
        module_name="trade_rl.catalog.sealed_test",
    )
    assert not _imports_prefix(ledger_targets, "trade_rl.catalog.postgres")
    for workflow_path in (
        PACKAGE_ROOT / "workflows/market_walk_forward.py",
        PACKAGE_ROOT / "workflows/_market_walk_forward_core.py",
    ):
        workflow_source = workflow_path.read_text(encoding="utf-8")
        assert "PostgresSealedTestReservationStore" in workflow_source
        assert "store.migrate()" not in workflow_source
        assert "PostgresSealedTestLedger(store)" in workflow_source
        assert "PostgresArtifactCatalog" not in workflow_source


def test_rl_terminal_info_owns_runtime_facts_not_evaluation_metrics() -> None:
    path = PACKAGE_ROOT / "rl/environment_info.py"
    source = path.read_text(encoding="utf-8")
    targets = _import_targets(path, module_name="trade_rl.rl.environment_info")
    assert not _imports_prefix(targets, "trade_rl.evaluation")
    assert not _imports_prefix(targets, "trade_rl.domain.performance")
    assert "book_metrics" not in _class_methods(path, "EnvironmentInfoBuilder")
    for forbidden in ("PerformanceMetrics", "ReturnSeries", "evaluate_performance"):
        assert forbidden not in source


def test_evaluation_is_the_single_owner_of_performance_metrics() -> None:
    assert not (PACKAGE_ROOT / "domain/performance.py").exists()
    assert not (PACKAGE_ROOT / "simulation/performance.py").exists()
    assert {
        "PerformanceMetrics",
        "compound_return",
        "evaluate_performance",
    } <= _defined_names(PACKAGE_ROOT / "evaluation/metrics.py")
    assert {"ReturnKind", "ReturnSeries"} <= _defined_names(
        PACKAGE_ROOT / "evaluation/series.py"
    )


def test_artifacts_own_the_serialized_policy_identity_vocabulary() -> None:
    contract = PACKAGE_ROOT / "artifacts/policy_identity_contract.py"
    source = contract.read_text(encoding="utf-8")
    for value in (
        'SB3_POLICY_IDENTITY_SCHEMA: Final = "sb3_policy_identity_v4"',
        'HIERARCHICAL_SEQUENCE_ENCODER: Final = "hierarchical_sequence_v2"',
        'STRUCTURED_TIMEFRAMES: Final = ("15m", "1h", "4h", "1d")',
    ):
        assert value in source
    assert not (PACKAGE_ROOT / "domain/policy_contracts.py").exists()
    expected_prefix = "trade_rl.artifacts.policy_identity_contract"
    consumers = (
        (
            PACKAGE_ROOT / "artifacts/structured_policy_contract.py",
            "trade_rl.artifacts.structured_policy_contract",
        ),
        (PACKAGE_ROOT / "rl/policy_identity.py", "trade_rl.rl.policy_identity"),
        (
            PACKAGE_ROOT / "rl/sequence_observations.py",
            "trade_rl.rl.sequence_observations",
        ),
    )
    for path, module_name in consumers:
        targets = _import_targets(path, module_name=module_name)
        assert _imports_prefix(targets, expected_prefix)


def test_market_walk_forward_has_no_dynamic_core_facade() -> None:
    path = PACKAGE_ROOT / "workflows/market_walk_forward.py"
    tree = _tree(path)
    assert not any(
        isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
        and node.name == "__getattr__"
        for node in tree.body
    )
    core_aliases: list[str] = []
    for node in tree.body:
        if not isinstance(node, ast.Assign) or not isinstance(
            node.value, ast.Attribute
        ):
            continue
        if not isinstance(node.value.value, ast.Name) or node.value.value.id != "_core":
            continue
        core_aliases.extend(
            target.id for target in node.targets if isinstance(target, ast.Name)
        )
    assert core_aliases == []


def test_documentation_and_ci_match_the_maintained_contracts() -> None:
    architecture = (ROOT / "docs/ARCHITECTURE.md").read_text(encoding="utf-8")
    configuration = (ROOT / "docs/CONFIGURATION.md").read_text(encoding="utf-8")
    workflow = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    for text in (architecture, configuration):
        assert "sb3_policy_identity_v4" in text
        assert "sb3_policy_identity_v1" not in text
    assert "agent/causal-training-hardening" not in workflow
    assert "agent/causal-sequence-feature-encoder" not in workflow
    assert "tests/architecture" in workflow
    assert not (ROOT / "docs/superpowers").exists()


def test_oracle_accelerator_backend_has_no_process_global_registry() -> None:
    learning_source = (PACKAGE_ROOT / "learning/oracle_solver.py").read_text(
        encoding="utf-8"
    )
    integrations_source = (PACKAGE_ROOT / "integrations/__init__.py").read_text(
        encoding="utf-8"
    )
    sb3_source = (PACKAGE_ROOT / "integrations/sb3_training.py").read_text(
        encoding="utf-8"
    )
    benchmark_source = (
        PACKAGE_ROOT / "operations/oracle_teacher_benchmark.py"
    ).read_text(encoding="utf-8")
    smoke_source = (PACKAGE_ROOT / "operations/oracle_cuda_smoke.py").read_text(
        encoding="utf-8"
    )

    assert "_ACCELERATOR_BACKENDS" not in learning_source
    assert "register_oracle_accelerator_backend" not in learning_source
    assert "register_default_oracle_accelerators" not in integrations_source
    assert "register_oracle_accelerator_backend" not in integrations_source
    assert "accelerator_backend=_oracle_accelerator_backend(" in sb3_source
    assert "accelerator_backend=accelerator_backend" in benchmark_source
    assert "accelerator_backend=solve_torch_cuda_oracle_batch" in smoke_source
