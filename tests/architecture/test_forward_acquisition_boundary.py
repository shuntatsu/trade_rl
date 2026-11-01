from __future__ import annotations

import ast
from pathlib import Path

import trade_rl.evaluation as evaluation
from tools.agent_repo.source_index import ImportCollector, within_module

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "trade_rl"
ACQUISITION = PACKAGE / "evaluation" / "forward_acquisition"
MODULES = (
    "__init__.py",
    "__main__.py",
    "acquire.py",
    "cli.py",
    "plan.py",
    "seal_record.py",
)
NETWORK = {"socket", "http", "urllib", "requests", "ssl", "ftplib", "httpx", "aiohttp"}


def _tree(name: str) -> ast.Module:
    return ast.parse((ACQUISITION / name).read_text(encoding="utf-8"))


def _imports(name: str) -> set[str]:
    return ImportCollector(PACKAGE).collect(ACQUISITION / name)


def _module_scope_imports(name: str) -> set[str]:
    found: set[str] = set()
    for node in _tree(name).body:
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            found.add(node.module)
    return found


def _function_scope_imports(name: str) -> set[str]:
    found: set[str] = set()
    for function in ast.walk(_tree(name)):
        if isinstance(function, ast.FunctionDef):
            for node in ast.walk(function):
                if isinstance(node, ast.ImportFrom) and node.module:
                    found.add(node.module)
                elif isinstance(node, ast.Import):
                    found.update(alias.name for alias in node.names)
    return found


def test_acquisition_layout_and_public_api() -> None:
    assert {path.name for path in ACQUISITION.glob("*.py")} == set(MODULES)
    assert "forward_acquisition" not in evaluation.__all__


def test_no_module_imports_a_network_client_at_module_scope() -> None:
    for name in MODULES:
        top = {item.split(".", 1)[0] for item in _module_scope_imports(name)}
        assert not top & NETWORK, name
        for imported in _module_scope_imports(name):
            assert not within_module(imported, "trade_rl.integrations"), (
                name,
                imported,
            )
        function_top = {item.split(".", 1)[0] for item in _function_scope_imports(name)}
        assert not function_top & NETWORK, name


def test_only_the_authorized_path_of_acquire_imports_the_binance_integration() -> None:
    for name in MODULES:
        lazy = {
            item
            for item in _function_scope_imports(name)
            if within_module(item, "trade_rl.integrations")
        }
        if name != "acquire.py":
            assert not lazy, (name, lazy)
    tree = _tree("acquire.py")
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "run_acquire":
            assert not any(
                isinstance(inner, (ast.Import, ast.ImportFrom))
                for inner in ast.walk(node)
            )
    # the planning module has no integration dependency at all, lazy or not
    assert not {
        i for i in _imports("plan.py") if within_module(i, "trade_rl.integrations")
    }


def test_the_tooling_scores_nothing_and_owns_no_replay_or_strategy() -> None:
    allowed_forward_study = {
        "trade_rl.evaluation.forward_study.registry",
        "trade_rl.evaluation.forward_study.seal",
    }
    for name in MODULES:
        for imported in _imports(name):
            if not within_module(imported, "trade_rl"):
                continue
            for forbidden in (
                "trade_rl.strategies",
                "trade_rl.simulation",
                "trade_rl.risk",
                "trade_rl.evaluation.final_test",
                "trade_rl.evaluation.paper",
                "trade_rl.evaluation.dev_realdata_study",
                "trade_rl.evaluation.forward_study.fs1",
                "trade_rl.evaluation.forward_study.labels",
                "trade_rl.evaluation.forward_study.cli",
            ):
                assert not within_module(imported, forbidden), (name, imported)
            if within_module(imported, "trade_rl.evaluation.forward_study"):
                assert imported in allowed_forward_study | {
                    "trade_rl.evaluation.forward_study"
                }, (name, imported)


def test_nothing_else_depends_on_the_acquisition_tooling() -> None:
    for root in (
        PACKAGE / "evaluation" / "forward_study",
        PACKAGE / "evaluation" / "dev_realdata_study",
        PACKAGE / "strategies",
        PACKAGE / "simulation",
        PACKAGE / "data",
        PACKAGE / "integrations",
    ):
        for path in root.rglob("*.py"):
            text = path.read_text(encoding="utf-8")
            assert "forward_acquisition" not in text, path.relative_to(ROOT)


def test_only_the_private_environment_reads_the_clock_and_nothing_posts_or_spawns() -> (
    None
):
    for name in MODULES:
        source = (ACQUISITION / name).read_text(encoding="utf-8")
        assert "subprocess" not in source, name
        assert "api.github.com" not in source and "gh issue" not in source, name
        assert "urlopen" not in source and "requests." not in source, name
        if name != "acquire.py":
            assert "datetime.now" not in source, name
    acquire_source = (ACQUISITION / "acquire.py").read_text(encoding="utf-8")
    assert acquire_source.count("datetime.now(UTC)") == 1
    environment = next(
        node
        for node in _tree("acquire.py").body
        if isinstance(node, ast.FunctionDef) and node.name == "_environment"
    )
    assert "datetime.now(UTC)" in ast.unparse(environment)


def test_public_entrypoints_expose_no_registry_digest_identity_clock_or_transport() -> (
    None
):
    forbidden = {
        "registry",
        "frozen_params_digest",
        "implemented_policy_digest",
        "dev_reference",
        "code_identity",
        "clock",
        "now",
        "network_transport",
        "transport",
        "env",
    }
    seen = set()
    for name in ("acquire.py", "cli.py", "seal_record.py"):
        for node in _tree(name).body:
            if isinstance(node, ast.FunctionDef) and node.name in {
                "run_acquire",
                "main",
                "render_seal_record",
            }:
                arguments = {a.arg for a in node.args.kwonlyargs + node.args.args}
                assert not arguments & forbidden, (name, node.name)
                seen.add(node.name)
    assert seen == {"run_acquire", "main", "render_seal_record"}


def test_network_is_gated_behind_allow_network_in_the_orchestrator() -> None:
    tree = _tree("acquire.py")
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_run_acquire"
    )
    gate = next(
        node
        for node in function.body
        if isinstance(node, ast.If) and "allow_network" in ast.unparse(node.test)
    )
    assert any(isinstance(node, ast.Return) for node in gate.body)
    # the first use of the transport factory comes after the plan-only return
    positions = {
        ast.unparse(node): node.lineno
        for node in ast.walk(function)
        if isinstance(node, ast.Call) and "network_transport" in ast.unparse(node.func)
    }
    assert positions and min(positions.values()) > gate.lineno


def test_the_design_record_and_current_docs_describe_the_acquisition() -> None:
    assurance = (ROOT / "docs" / "architecture" / "research-assurance.md").read_text(
        encoding="utf-8"
    )
    boundaries = (ROOT / "docs" / "architecture" / "package-boundaries.md").read_text(
        encoding="utf-8"
    )
    status = (ROOT / "docs" / "research" / "current-status.md").read_text(
        encoding="utf-8"
    )
    assert "#### FS1 acquisition procedure" in assurance
    flat = " ".join(assurance.split())
    for required in (
        "python -m trade_rl.evaluation.forward_acquisition acquire",
        "--allow-network",
        "seal-record",
        "LOW-B",
        "zero-byte",
        "#1049",
    ):
        assert required in flat, required
    assert "evaluation/forward_acquisition/" in boundaries
    section = status[status.index("## Issue #810 FS1") :]
    assert "acquisition" in section and "No data was fetched" in section
