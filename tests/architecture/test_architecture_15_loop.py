from __future__ import annotations

import ast
from pathlib import Path

from tests.architecture.import_references import scan_import_references

ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = ROOT / "trade_rl"


def _tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _defined_names(path: Path) -> frozenset[str]:
    return frozenset(
        node.name
        for node in _tree(path).body
        if isinstance(node, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef)
    )


def _import_targets(path: Path, *, module_name: str) -> frozenset[str]:
    return frozenset(
        reference.target
        for reference in scan_import_references(path, module_name=module_name)
        if reference.target is not None
    )


def _imports_prefix(targets: frozenset[str], prefix: str) -> bool:
    return any(target == prefix or target.startswith(f"{prefix}.") for target in targets)


def _literal_assignments(path: Path) -> dict[str, object]:
    assignments: dict[str, object] = {}
    for node in _tree(path).body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        try:
            assignments[target.id] = ast.literal_eval(node.value)
        except (ValueError, TypeError):
            continue
    return assignments


def _class_methods(path: Path, class_name: str) -> frozenset[str]:
    for node in _tree(path).body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return frozenset(
                item.name
                for item in node.body
                if isinstance(item, ast.FunctionDef | ast.AsyncFunctionDef)
            )
    raise AssertionError(f"class not found: {class_name}")


def _function_source(path: Path, function_name: str) -> str:
    source = path.read_text(encoding="utf-8")
    for node in _tree(path).body:
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and node.name == function_name:
            segment = ast.get_source_segment(source, node)
            assert segment is not None
            return segment
        if isinstance(node, ast.ClassDef):
            for item in node.body:
                if (
                    isinstance(item, ast.FunctionDef | ast.AsyncFunctionDef)
                    and item.name == function_name
                ):
                    segment = ast.get_source_segment(source, item)
                    assert segment is not None
                    return segment
    raise AssertionError(f"function not found: {function_name}")


def _module_body_calls(path: Path) -> tuple[str, ...]:
    calls: list[str] = []
    for node in _tree(path).body:
        if not isinstance(node, ast.Expr) or not isinstance(node.value, ast.Call):
            continue
        function = node.value.func
        if isinstance(function, ast.Name):
            calls.append(function.id)
        elif isinstance(function, ast.Attribute):
            calls.append(function.attr)
    return tuple(calls)


def _cross_package_private_imports() -> tuple[str, ...]:
    violations: list[str] = []
    for path in sorted(PACKAGE_ROOT.rglob("*.py")):
        relative = path.relative_to(PACKAGE_ROOT)
        if len(relative.parts) < 2:
            continue
        source_package = relative.parts[0]
        tree = _tree(path)
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom) or node.level != 0:
                continue
            module = node.module or ""
            if not module.startswith("trade_rl."):
                continue
            parts = module.split(".")
            if len(parts) < 2 or parts[1] == source_package:
                continue
            for alias in node.names:
                if alias.name.startswith("_"):
                    violations.append(
                        f"{relative.as_posix()} imports {module}.{alias.name}"
                    )
    return tuple(violations)


def test_loop_01_training_environment_contract_is_public_and_framework_neutral() -> None:
    path = PACKAGE_ROOT / "rl/training_environment_contract.py"
    assert path.is_file()
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


def test_loop_02_sb3_training_uses_public_training_environment_contract() -> None:
    path = PACKAGE_ROOT / "integrations/sb3_training.py"
    targets = _import_targets(path, module_name="trade_rl.integrations.sb3_training")
    assert {
        "trade_rl.rl.training_environment_contract.training_environment_identity",
        "trade_rl.rl.training_environment_contract.validate_training_environment",
    } <= targets
    assert "trade_rl.rl.training._environment_identity" not in targets
    assert "trade_rl.rl.training._validate_training_environment" not in targets


def test_loop_03_production_cross_package_imports_never_target_private_symbols() -> None:
    assert _cross_package_private_imports() == ()


def test_loop_04_generic_postgres_catalog_excludes_sealed_evaluation() -> None:
    path = PACKAGE_ROOT / "catalog/postgres.py"
    targets = _import_targets(path, module_name="trade_rl.catalog.postgres")
    assert not _imports_prefix(targets, "trade_rl.evaluation")
    assert "reserve_sealed_test_access" not in _class_methods(
        path,
        "PostgresArtifactCatalog",
    )


def test_loop_05_postgres_adapters_share_public_connection_factory() -> None:
    contract = PACKAGE_ROOT / "catalog/postgres_connection.py"
    assert contract.is_file()
    assert {"default_connection_factory", "import_psycopg"} <= _defined_names(contract)
    postgres_targets = _import_targets(
        PACKAGE_ROOT / "catalog/postgres.py",
        module_name="trade_rl.catalog.postgres",
    )
    sealed_targets = _import_targets(
        PACKAGE_ROOT / "catalog/postgres_sealed_test.py",
        module_name="trade_rl.catalog.postgres_sealed_test",
    )
    expected = "trade_rl.catalog.postgres_connection.default_connection_factory"
    assert expected in postgres_targets
    assert expected in sealed_targets
    assert "trade_rl.catalog.postgres._default_connection_factory" not in sealed_targets


def test_loop_06_rl_environment_info_does_not_import_evaluation() -> None:
    path = PACKAGE_ROOT / "rl/environment_info.py"
    targets = _import_targets(path, module_name="trade_rl.rl.environment_info")
    assert not _imports_prefix(targets, "trade_rl.evaluation")


def test_loop_07_performance_contract_lives_below_rl_and_evaluation() -> None:
    contract = PACKAGE_ROOT / "simulation/performance.py"
    assert contract.is_file()
    assert {
        "PerformanceMetrics",
        "ReturnKind",
        "ReturnSeries",
        "compound_return",
        "evaluate_performance",
    } <= _defined_names(contract)
    metrics_targets = _import_targets(
        PACKAGE_ROOT / "evaluation/metrics.py",
        module_name="trade_rl.evaluation.metrics",
    )
    series_targets = _import_targets(
        PACKAGE_ROOT / "evaluation/series.py",
        module_name="trade_rl.evaluation.series",
    )
    assert {
        "trade_rl.simulation.performance.PerformanceMetrics",
        "trade_rl.simulation.performance.compound_return",
        "trade_rl.simulation.performance.evaluate_performance",
    } <= metrics_targets
    assert {
        "trade_rl.simulation.performance.ReturnKind",
        "trade_rl.simulation.performance.ReturnSeries",
    } <= series_targets


def test_loop_08_neutral_policy_contract_owns_shared_identifiers() -> None:
    path = PACKAGE_ROOT / "domain/policy_contracts.py"
    assert path.is_file()
    assignments = _literal_assignments(path)
    assert assignments["SB3_POLICY_IDENTITY_SCHEMA"] == "sb3_policy_identity_v4"
    assert assignments["HIERARCHICAL_SEQUENCE_ENCODER"] == "hierarchical_sequence_v2"
    assert assignments["STRUCTURED_TIMEFRAMES"] == ("15m", "1h", "4h", "1d")


def test_loop_09_structured_export_consumes_neutral_policy_identifiers() -> None:
    path = PACKAGE_ROOT / "artifacts/structured_policy_contract.py"
    targets = _import_targets(
        path,
        module_name="trade_rl.artifacts.structured_policy_contract",
    )
    assert {
        "trade_rl.domain.policy_contracts.HIERARCHICAL_SEQUENCE_ENCODER",
        "trade_rl.domain.policy_contracts.SB3_POLICY_IDENTITY_SCHEMA",
        "trade_rl.domain.policy_contracts.STRUCTURED_TIMEFRAMES",
    } <= targets
    assignments = _literal_assignments(path)
    assert "_POLICY_IDENTITY_SCHEMA" not in assignments
    assert "STRUCTURED_TIMEFRAMES" not in assignments


def test_loop_10_rl_policy_modules_consume_neutral_policy_identifiers() -> None:
    policy_targets = _import_targets(
        PACKAGE_ROOT / "rl/policy_identity.py",
        module_name="trade_rl.rl.policy_identity",
    )
    sequence_targets = _import_targets(
        PACKAGE_ROOT / "rl/sequence_observations.py",
        module_name="trade_rl.rl.sequence_observations",
    )
    assert {
        "trade_rl.domain.policy_contracts.HIERARCHICAL_SEQUENCE_ENCODER",
        "trade_rl.domain.policy_contracts.SB3_POLICY_IDENTITY_SCHEMA",
    } <= policy_targets
    assert "trade_rl.domain.policy_contracts.STRUCTURED_TIMEFRAMES" in sequence_targets


def test_loop_11_candidate_recipe_identity_excludes_export_transport() -> None:
    path = PACKAGE_ROOT / "rl/training_run_config.py"
    candidate = _function_source(path, "candidate_digest_payload")
    recipe = _function_source(path, "_recipe_identity_payload")
    assert "_recipe_identity_payload" in candidate
    assert "export_" not in recipe


def test_loop_12_candidate_recipe_identity_excludes_source_provenance() -> None:
    path = PACKAGE_ROOT / "rl/training_run_config.py"
    recipe = _function_source(path, "_recipe_identity_payload")
    assert "git_commit" not in recipe
    assert "git_dirty" not in recipe


def test_loop_13_full_run_identity_retains_transport_and_provenance() -> None:
    path = PACKAGE_ROOT / "rl/training_run_config.py"
    run_identity = _function_source(path, "_run_identity_payload")
    digest_payload = _function_source(path, "digest_payload")
    for required in (
        "export_onnx",
        "export_structured_torchscript",
        "export_tolerance",
        "export_torchscript",
        "git_commit",
        "git_dirty",
        "resume_checkpoint_digests",
        "transfer_checkpoint_digests",
    ):
        assert required in run_identity
    assert "_run_identity_payload" in digest_payload


def test_loop_14_oracle_accelerator_registration_is_explicit() -> None:
    integrations = PACKAGE_ROOT / "integrations/__init__.py"
    sb3_training = PACKAGE_ROOT / "integrations/sb3_training.py"
    assert "register_oracle_accelerator_backend" not in _module_body_calls(integrations)
    assert "register_default_oracle_accelerators" in _defined_names(integrations)
    solver_config_source = _function_source(sb3_training, "_oracle_solver_config")
    assert "register_default_oracle_accelerators" in solver_config_source


def test_loop_15_documentation_and_ci_match_maintained_contracts() -> None:
    architecture = (ROOT / "docs/ARCHITECTURE.md").read_text(encoding="utf-8")
    workflow = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    assert "sb3_policy_identity_v4" in architecture
    assert "sb3_policy_identity_v1" not in architecture
    assert "agent/causal-training-hardening" not in workflow
    assert "agent/causal-sequence-feature-encoder" not in workflow
    assert "tests/architecture" in workflow
