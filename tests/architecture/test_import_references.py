from __future__ import annotations

from pathlib import Path

from tests.architecture.import_references import (
    ImportReference,
    module_name_from_path,
    scan_import_references,
)


def _scan(
    tmp_path: Path, source: str, *, module_name: str = "trade_rl.sample"
) -> tuple[ImportReference, ...]:
    path = tmp_path / "sample.py"
    path.write_text(source, encoding="utf-8")
    return scan_import_references(path, module_name=module_name)


def _resolved(references: tuple[ImportReference, ...]) -> tuple[tuple[str, str], ...]:
    return tuple(
        (reference.kind, reference.target)
        for reference in references
        if not reference.unresolved and reference.target is not None
    )


def test_module_name_from_path_handles_modules_and_packages() -> None:
    package_root = Path("trade_rl")

    assert (
        module_name_from_path(
            Path("trade_rl/workflows/example.py"),
            package_root=package_root,
            root_package="trade_rl",
        )
        == "trade_rl.workflows.example"
    )
    assert (
        module_name_from_path(
            Path("trade_rl/workflows/example/__init__.py"),
            package_root=package_root,
            root_package="trade_rl",
        )
        == "trade_rl.workflows.example"
    )


def test_scanner_extracts_static_and_function_local_imports(tmp_path: Path) -> None:
    references = _scan(
        tmp_path,
        """
import trade_rl.data
import trade_rl.serving as serving
from trade_rl.workflows import causal_scenario
from trade_rl.workflows.causal_scenario import replay as replay_module


def load_lazily() -> None:
    from trade_rl.integrations import sb3_serving
""",
    )

    assert _resolved(references) == (
        ("import", "trade_rl.data"),
        ("import", "trade_rl.serving"),
        ("from", "trade_rl.workflows.causal_scenario"),
        ("from", "trade_rl.workflows.causal_scenario.replay"),
        ("from", "trade_rl.integrations.sb3_serving"),
    )
    assert [reference.line for reference in references] == [2, 3, 4, 5, 9]


def test_scanner_resolves_relative_imports(tmp_path: Path) -> None:
    references = _scan(
        tmp_path,
        """
from . import sibling
from ..causal_scenario import replay
from ..causal_scenario.library import build
""",
        module_name="trade_rl.workflows.runtime.worker",
    )

    assert _resolved(references) == (
        ("from", "trade_rl.workflows.runtime.sibling"),
        ("from", "trade_rl.workflows.causal_scenario.replay"),
        ("from", "trade_rl.workflows.causal_scenario.library.build"),
    )


def test_scanner_extracts_literal_dynamic_import_aliases(tmp_path: Path) -> None:
    references = _scan(
        tmp_path,
        """
import importlib as imports
import builtins as python_builtins
from importlib import import_module as load_module
from builtins import __import__ as load_builtin

imports.import_module("trade_rl.workflows.causal_scenario")
load_module("trade_rl.workflows.causal_scenario.replay")
__import__("trade_rl.workflows.causal_scenario.library")
python_builtins.__import__("trade_rl.workflows.causal_scenario.library_artifact")
load_builtin("trade_rl.workflows.causal_scenario.conditions")
""",
    )

    assert _resolved(references) == (
        ("import", "importlib"),
        ("import", "builtins"),
        ("from", "importlib.import_module"),
        ("from", "builtins.__import__"),
        ("dynamic", "trade_rl.workflows.causal_scenario"),
        ("dynamic", "trade_rl.workflows.causal_scenario.replay"),
        ("dynamic", "trade_rl.workflows.causal_scenario.library"),
        ("dynamic", "trade_rl.workflows.causal_scenario.library_artifact"),
        ("dynamic", "trade_rl.workflows.causal_scenario.conditions"),
    )


def test_scanner_resolves_relative_importlib_calls(tmp_path: Path) -> None:
    references = _scan(
        tmp_path,
        """
import importlib
importlib.import_module(
    ".replay",
    package="trade_rl.workflows.causal_scenario",
)
""",
    )

    assert _resolved(references) == (
        ("import", "importlib"),
        ("dynamic", "trade_rl.workflows.causal_scenario.replay"),
    )


def test_scanner_fails_closed_for_non_literal_dynamic_targets(tmp_path: Path) -> None:
    references = _scan(
        tmp_path,
        """
import importlib
from importlib import import_module

module_name = "trade_rl.workflows.causal_scenario"
importlib.import_module(module_name)
import_module(".replay", package=module_name)
__import__(module_name)
""",
    )

    unresolved = tuple(reference for reference in references if reference.unresolved)
    assert [
        (reference.kind, reference.target, reference.line) for reference in unresolved
    ] == [
        ("dynamic", None, 6),
        ("dynamic", None, 7),
        ("dynamic", None, 8),
    ]


def test_comments_docstrings_and_ordinary_strings_are_not_dependencies(
    tmp_path: Path,
) -> None:
    references = _scan(
        tmp_path,
        '''
"""trade_rl.workflows.causal_scenario.replay"""
# import trade_rl.workflows.causal_scenario.library
MESSAGE = "from trade_rl.workflows.causal_scenario import replay"
''',
    )

    assert references == ()
