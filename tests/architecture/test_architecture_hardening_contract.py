from __future__ import annotations

import ast

import yaml

from tests.architecture.import_linter_config import configured_layers
from tests.architecture.repository_paths import PYTHON_SOURCE_ROOT, REPOSITORY_ROOT


def _top_level_packages() -> set[str]:
    return {
        f"trade_rl.{path.name}"
        for path in PYTHON_SOURCE_ROOT.iterdir()
        if path.is_dir() and (path / "__init__.py").is_file()
    }


def test_every_top_level_production_package_has_one_declared_layer() -> None:
    assert _top_level_packages() == set(configured_layers())


def test_runtime_factory_implementation_is_owned_by_integrations() -> None:
    implementation = PYTHON_SOURCE_ROOT / "integrations/runtime_factory.py"
    facade = PYTHON_SOURCE_ROOT / "runtime_factory.py"

    assert implementation.is_file()
    source = facade.read_text(encoding="utf-8")
    assert "importlib" not in source
    assert "from trade_rl.integrations.runtime_factory import" in source


def test_causal_alpha_generation_script_is_a_thin_operations_adapter() -> None:
    path = REPOSITORY_ROOT / "scripts/control_causal_alpha_v3_research_generation.py"
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    declarations = [
        node.name
        for node in tree.body
        if isinstance(node, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef)
    ]

    assert declarations == []
    assert "subprocess" not in source
    assert "trade_rl.operations.causal_alpha_v3_generation" in source
    assert len(source.splitlines()) <= 12


def test_top_level_modules_are_only_explicit_bootstrap_facades() -> None:
    modules = {path.stem for path in PYTHON_SOURCE_ROOT.glob("*.py") if path.is_file()}
    assert modules == {
        "__init__",
        "_source_checkout",
        "_version",
        "runtime_factory",
    }


def test_ci_exposes_one_conditional_full_training_capability_gate() -> None:
    payload = yaml.safe_load(
        (REPOSITORY_ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    )
    job = payload["jobs"]["full-training-capability"]
    assert job["name"] == "Full training capability gate"
    assert job["runs-on"] == "ubuntu-latest"
    assert job["timeout-minutes"] == 75
    steps = job["steps"]
    names = tuple(step.get("name") for step in steps)
    assert names.count("Detect training-sensitive changes") == 1
    assert names.count("Run full training capability audit") == 1
    scripts = "\n".join(str(step.get("run", "")) for step in steps if step.get("run"))
    assert "scripts/run_training_capability_audit.py" in scripts
    assert "full_training_capability_audit_v1" in scripts
    assert "steps.changes.outputs.required == 'true'" in str(steps)
