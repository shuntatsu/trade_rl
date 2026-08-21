from __future__ import annotations

import ast

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
