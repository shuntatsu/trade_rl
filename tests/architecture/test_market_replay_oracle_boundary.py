"""Keep the independent market content verifier outside production ownership."""

from __future__ import annotations

import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OWNER = "tools.market_replay_oracle"


def _absolute_imports(path: Path) -> set[str]:
    module_path = path.relative_to(ROOT).with_suffix("")
    package = list(module_path.parts[:-1])
    imports: set[str] = set()
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0:
                if node.module:
                    imports.add(node.module)
                continue
            base = package[: len(package) - node.level + 1]
            if node.module:
                base.extend(node.module.split("."))
            if base:
                imports.add(".".join(base))
                imports.update(
                    f"{'.'.join(base)}.{alias.name}"
                    for alias in node.names
                    if alias.name != "*"
                )
    return imports


def _is_allowed(module: str) -> bool:
    return (
        module.split(".", 1)[0] in sys.stdlib_module_names
        or module == "numpy"
        or module.startswith("numpy.")
        or module == OWNER
        or module.startswith(OWNER + ".")
    )


def test_verifier_imports_only_stdlib_numpy_and_its_own_package() -> None:
    sources = tuple(sorted((ROOT / OWNER.replace(".", "/")).rglob("*.py")))
    assert sources
    for path in sources:
        for module in _absolute_imports(path):
            assert _is_allowed(module), (path.relative_to(ROOT), module)
            assert not module.startswith("tools.market_account_oracle"), module
            assert not module.startswith("trade_rl"), module


def test_production_modules_do_not_import_the_content_verifier() -> None:
    for path in (ROOT / "trade_rl").rglob("*.py"):
        assert not any(
            module == OWNER or module.startswith(OWNER + ".")
            for module in _absolute_imports(path)
        ), path.relative_to(ROOT)


def test_both_research_typing_routes_include_verifier() -> None:
    workflow = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    routes = [
        line
        for line in workflow.splitlines()
        if "uv run mypy" in line and "tools/market_account_oracle" in line
    ]
    assert len(routes) == 2
    assert all("tools/market_replay_oracle" in line for line in routes)
    assert workflow.count("tools/market_replay_oracle") == 2
    attributes = (ROOT / ".gitattributes").read_text(encoding="utf-8")
    assert "tools/market_replay_oracle/*.py text eol=lf" in attributes


def test_public_facade_is_the_closed_pinned_attempt_binding_surface() -> None:
    facade = ROOT / OWNER.replace(".", "/") / "__init__.py"
    tree = ast.parse(facade.read_text(encoding="utf-8"))
    expected = [
        "DatasetContentPin",
        "VerifiedDatasetContent",
        "load_pinned_v4",
        "ProtocolPin",
        "NativeSourcePin",
        "FrozenAttemptBinding",
        "load_pinned_attempt",
        "EventBatchPin",
        "OrderEventContext",
        "OrderEventEvidence",
        "read_pinned_order_events",
        "ExactFillAttemptBinding",
        "load_pinned_exact_fill_attempt",
        "BoundEventAdvance",
        "BoundEventRowCoverage",
        "walk_bound_event_rows",
        "DeclaredProtocolPair",
        "load_declared_protocol_pair",
        "ARRAY_FIELDS",
        "canonical",
        "content_id",
    ]

    exports = None
    imported_names: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.ImportFrom):
            imported_names.update(alias.asname or alias.name for alias in node.names)
        elif isinstance(node, ast.Assign):
            if any(
                isinstance(target, ast.Name) and target.id == "__all__"
                for target in node.targets
            ):
                exports = ast.literal_eval(node.value)
        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name) and node.target.id == "__all__":
                exports = ast.literal_eval(node.value)

    assert exports == expected
    assert len(imported_names.intersection(expected)) == len(expected)


def test_native_static_readback_composes_pair_and_v4_readers_through_facade() -> None:
    consumer = ROOT / "tools/market_b4_oracle/offline_replay_consumer.py"
    tree = ast.parse(consumer.read_text(encoding="utf-8"))
    facade_names: set[str] = set()
    private_imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.level == 0:
            if node.module == OWNER:
                facade_names.update(alias.name for alias in node.names)
            elif node.module and node.module.startswith(OWNER + "."):
                private_imports.add(node.module)

    assert {
        "load_declared_protocol_pair",
        "load_pinned_exact_fill_attempt",
        "walk_bound_event_rows",
    } <= facade_names
    assert "load_pinned_attempt" not in facade_names
    assert not private_imports
