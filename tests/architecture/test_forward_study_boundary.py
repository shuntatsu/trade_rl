from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

import trade_rl.evaluation as evaluation
from tools.agent_repo.source_index import ImportCollector, within_module

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "trade_rl"
FORWARD = PACKAGE / "evaluation" / "forward_study"
MODULES = (
    "__init__.py",
    "__main__.py",
    "cli.py",
    "fs1.py",
    "labels.py",
    "registry.py",
    "seal.py",
)
NETWORK_AND_PROCESS = {"socket", "http", "urllib", "requests", "ssl", "ftplib"}


def _imports(name: str) -> set[str]:
    return ImportCollector(PACKAGE).collect(FORWARD / name)


def _trade_rl_imports(name: str) -> set[str]:
    return {item for item in _imports(name) if within_module(item, "trade_rl")}


def test_forward_study_layout_and_public_api() -> None:
    assert {path.name for path in FORWARD.glob("*.py")} == set(MODULES)
    assert "forward_study" not in evaluation.__all__
    assert not any(name.startswith("Fs1") for name in evaluation.__all__)


def test_forward_study_has_no_network_or_unused_window_dependency() -> None:
    for name in MODULES:
        top_level = {item.split(".", 1)[0] for item in _imports(name)}
        assert not top_level & NETWORK_AND_PROCESS, name
        for imported in _trade_rl_imports(name):
            assert not within_module(imported, "trade_rl.integrations"), (
                name,
                imported,
            )
            assert not within_module(imported, "trade_rl.evaluation.final_test"), (
                name,
                imported,
            )
            assert not within_module(imported, "trade_rl.evaluation.paper"), (
                name,
                imported,
            )
            assert not within_module(imported, "trade_rl.strategies.rl"), (
                name,
                imported,
            )


def test_registry_and_labels_are_pure_and_import_only_frozen_definitions() -> None:
    assert _trade_rl_imports("registry.py") == {
        "trade_rl.evaluation.dev_realdata_study.phase3_registry",
        "trade_rl.evaluation.dev_realdata_study.protocol",
        "trade_rl.evaluation.dev_realdata_study.trials",
    }
    assert _trade_rl_imports("labels.py") == {
        "trade_rl.evaluation.forward_study.registry"
    }
    for name in ("registry.py", "labels.py"):
        top_level = {item.split(".", 1)[0] for item in _imports(name)}
        assert top_level - set(sys.stdlib_module_names) <= {"trade_rl"}, name


def test_only_the_cli_reads_the_wall_clock_and_only_the_seal_runs_git() -> None:
    for name in MODULES:
        source = (FORWARD / name).read_text(encoding="utf-8")
        if name != "cli.py":
            assert "datetime.now" not in source, name
        if name != "seal.py":
            assert "subprocess" not in source, name
    assert "datetime.now(UTC)" in (FORWARD / "cli.py").read_text(encoding="utf-8")


def test_scoring_is_reachable_only_through_the_permit() -> None:
    tree = ast.parse((FORWARD / "fs1.py").read_text(encoding="utf-8"))
    functions = {
        node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)
    }
    for name in ("run_roster", "score_fs1"):
        calls = {
            getattr(node.func, "id", None)
            for node in ast.walk(functions[name])
            if isinstance(node, ast.Call)
        }
        assert "require_permit" in calls or "run_roster" in calls, name
    assert "verify_dataset" in {
        getattr(node.func, "id", None)
        for node in ast.walk(functions["run_roster"])
        if isinstance(node, ast.Call)
    }


def test_nothing_in_the_dev_study_or_allocator_depends_on_forward_study() -> None:
    for root in (
        PACKAGE / "evaluation" / "dev_realdata_study",
        PACKAGE / "strategies",
        PACKAGE / "simulation",
        PACKAGE / "data",
    ):
        for path in root.rglob("*.py"):
            text = path.read_text(encoding="utf-8")
            assert "forward_study" not in text, path.relative_to(ROOT)


def test_the_design_record_and_registry_export_are_in_the_current_docs() -> None:
    assurance = (ROOT / "docs" / "architecture" / "research-assurance.md").read_text(
        encoding="utf-8"
    )
    boundaries = (ROOT / "docs" / "architecture" / "package-boundaries.md").read_text(
        encoding="utf-8"
    )
    status = (ROOT / "docs" / "research" / "current-status.md").read_text(
        encoding="utf-8"
    )
    assert "## Research-specific contract: forward study FS1" in assurance
    for heading in ("### G0 — FS1", "### G1 — FS1", "### G2 — FS1"):
        assert heading in assurance
    from trade_rl.evaluation.forward_study.registry import FROZEN_PARAMS_DIGEST

    assert FROZEN_PARAMS_DIGEST in assurance
    assert "evaluation/forward_study/" in boundaries
    assert "report/issue810-fs1-registry-20261030.json" in boundaries
    section = status[status.index("## Issue #810 FS1") :]
    for required in (
        "G0-G2 review of the design and code is pending",
        "no economic result exists",
        "fetching",
        "single G4 scoring run",
    ):
        assert required in section
    assert re.search(r"authoriz", section)
