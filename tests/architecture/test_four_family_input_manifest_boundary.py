from __future__ import annotations

import ast
from pathlib import Path

import trade_rl.evaluation.experiments as experiments
from tools.agent_repo.source_index import ImportCollector, within_module

ROOT = Path(__file__).resolve().parents[2]
MODULE = (
    ROOT / "trade_rl" / "evaluation" / "experiments" / "four_family_input_manifest.py"
)
IDENTITY_FACADE = "trade_rl.evaluation.runs"
PRIVATE_IDENTITY_MODULE = "trade_rl.evaluation.runs.portfolio_family_result_artifact"
FORBIDDEN_IMPORTS = {
    "trade_rl.data.artifacts.publication",
    "trade_rl.evaluation.experiments.workflow",
    "trade_rl.evaluation.runs.portfolio_family_attempt",
    PRIVATE_IDENTITY_MODULE,
    "trade_rl.strategies.rl.portfolio_inference_bundle",
}


def _forbidden_direct_imports(module: Path, package: Path) -> set[str]:
    imported_modules = ImportCollector(package).collect_direct(module)
    return {
        imported
        for imported in imported_modules
        for forbidden in FORBIDDEN_IMPORTS
        if within_module(imported, forbidden)
    }


def test_guard_detects_ordinary_imports_inside_functions(tmp_path: Path) -> None:
    package = tmp_path / "trade_rl"
    physical_child = package / "evaluation" / "runs" / "portfolio_family_attempt.py"
    physical_child.parent.mkdir(parents=True)
    physical_child.write_text("", encoding="utf-8")
    module = package / "evaluation" / "experiments" / "fixture.py"
    module.parent.mkdir(parents=True)
    cases = (
        (
            "import trade_rl.evaluation.runs.portfolio_family_attempt as attempt",
            "trade_rl.evaluation.runs.portfolio_family_attempt",
        ),
        (
            "from trade_rl.evaluation.runs import portfolio_family_attempt as attempt",
            "trade_rl.evaluation.runs.portfolio_family_attempt",
        ),
        (
            "import trade_rl.evaluation.runs.portfolio_family_attempt.detail as attempt",
            "trade_rl.evaluation.runs.portfolio_family_attempt.detail",
        ),
    )
    for statement, expected_import in cases:
        module.write_text(f"def deferred():\n    {statement}\n", encoding="utf-8")
        assert _forbidden_direct_imports(module, package) == {expected_import}


def test_guard_does_not_follow_the_public_identity_facade(tmp_path: Path) -> None:
    package = tmp_path / "trade_rl"
    runs = package / "evaluation" / "runs"
    runs.mkdir(parents=True)
    (runs / "__init__.py").write_text(
        "from .portfolio_family_result_artifact import "
        "FourFamilyPortfolioAttemptIdentity\n",
        encoding="utf-8",
    )
    (runs / "portfolio_family_result_artifact.py").write_text(
        "class FourFamilyPortfolioAttemptIdentity: ...\n", encoding="utf-8"
    )
    module = package / "evaluation" / "experiments" / "fixture.py"
    module.parent.mkdir(parents=True)
    module.write_text(
        "from trade_rl.evaluation.runs import FourFamilyPortfolioAttemptIdentity\n",
        encoding="utf-8",
    )

    collector = ImportCollector(package)
    assert PRIVATE_IDENTITY_MODULE in collector.collect(module)
    assert _forbidden_direct_imports(module, package) == set()


def test_post_freeze_manifest_stays_private_and_metadata_only() -> None:
    assert "FourFamilyPostFreezeInputManifest" not in experiments.__all__

    assert _forbidden_direct_imports(MODULE, ROOT / "trade_rl") == set()
    tree = ast.parse(MODULE.read_text(encoding="utf-8"), filename=str(MODULE))
    imported_identity_names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module == IDENTITY_FACADE:
                imported_identity_names.update(alias.name for alias in node.names)

    assert imported_identity_names == {"FourFamilyPortfolioAttemptIdentity"}
