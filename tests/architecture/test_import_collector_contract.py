from __future__ import annotations

from pathlib import Path

import pytest
from tools.agent_repo.source_index import ImportCollector

from tests.architecture import test_lean_dependency_boundaries as boundaries

SEALED = "trade_rl.evaluation.robustness.walk_forward.sealed_test"
CLIENT = "trade_rl/evaluation/experiments/client.py"


def _write(root: Path, relative: str, source: str) -> Path:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")
    return path


@pytest.fixture
def source_tree(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    files = {
        "trade_rl/__init__.py": "",
        "trade_rl/evaluation/__init__.py": "",
        "trade_rl/evaluation/experiments/__init__.py": "",
        "trade_rl/evaluation/robustness/__init__.py": "",
        "trade_rl/evaluation/robustness/walk_forward/__init__.py": (
            "from .sealed_test import SealedTestLedger as FinalLedger\n"
            "from .safe import Safe\n"
        ),
        "trade_rl/evaluation/robustness/walk_forward/sealed_test.py": (
            "raise RuntimeError('source inspection must not execute imports')\n"
            "class SealedTestLedger: pass\n"
        ),
        "trade_rl/evaluation/robustness/walk_forward/sealed_test_helpers.py": (
            "harmless = 1\n"
        ),
        "trade_rl/evaluation/robustness/walk_forward/safe.py": "class Safe: pass\n",
        "trade_rl/evaluation/facade.py": (
            "from .robustness.walk_forward import FinalLedger as Ledger\n"
            "from .robustness.walk_forward import Safe\n"
        ),
    }
    for name, text in files.items():
        _write(tmp_path, name, text)
    monkeypatch.setattr(boundaries, "ROOT", tmp_path)
    monkeypatch.setattr(boundaries, "PACKAGE", tmp_path / "trade_rl")
    return tmp_path


@pytest.mark.parametrize(
    "source",
    [
        f"import {SEALED}\n",
        f"import {SEALED} as other\n",
        f"from {SEALED} import SealedTestLedger\n",
        "from ..robustness.walk_forward.sealed_test import SealedTestLedger\n",
        "from ..robustness.walk_forward.sealed_test import SealedTestLedger as Other\n",
        "from trade_rl.evaluation.robustness.walk_forward import sealed_test\n",
        "from ..robustness.walk_forward import sealed_test as other\n",
        "from ..robustness.walk_forward import FinalLedger\n",
        "from trade_rl.evaluation.facade import Ledger as Other\n",
        "from ..facade import Ledger\n",
        "def deferred():\n    from ..facade import Ledger\n",
        "if False:\n    from ..robustness.walk_forward import sealed_test\n",
    ],
)
def test_forbidden_import_spellings_are_detected(
    source_tree: Path, source: str
) -> None:
    path = _write(source_tree, CLIENT, source)
    assert SEALED in boundaries.collect_trade_rl_imports(path)
    assert boundaries._offenders(path.parent, (SEALED,)) == [CLIENT]


@pytest.mark.parametrize(
    "relative,source",
    [
        (
            "trade_rl/evaluation/experiments/__init__.py",
            "from ..robustness.walk_forward import sealed_test\n",
        ),
        (
            "trade_rl/evaluation/robustness/walk_forward/client.py",
            "from . import sealed_test\n",
        ),
        (
            "trade_rl/evaluation/robustness/walk_forward/__init__.py",
            "from . import sealed_test\n",
        ),
    ],
)
def test_relative_imports_use_file_or_package_context(
    source_tree: Path, relative: str, source: str
) -> None:
    path = _write(source_tree, relative, source)
    assert SEALED in boundaries.collect_trade_rl_imports(path)


@pytest.mark.parametrize(
    "source",
    [
        "from ..facade import Safe\n",
        "from ..robustness.walk_forward import Safe\n",
        f"from {SEALED}_helpers import harmless\n",
        "from trade_rl_aux import evaluation\n",
        "import pathlib\nfrom datetime import datetime\n",
    ],
)
def test_allowed_imports_do_not_inherit_unrelated_facade_dependencies(
    source_tree: Path, source: str
) -> None:
    path = _write(source_tree, CLIENT, source)
    assert boundaries._offenders(path.parent, (SEALED,)) == []
    assert all(
        name == "trade_rl" or name.startswith("trade_rl.")
        for name in boundaries.collect_trade_rl_imports(path)
    )


def test_reexport_cycles_terminate_and_do_not_hide_a_forbidden_owner(
    source_tree: Path,
) -> None:
    _write(source_tree, "trade_rl/evaluation/a.py", "from .b import Ledger\n")
    _write(
        source_tree,
        "trade_rl/evaluation/b.py",
        "from .a import Ledger\nfrom .facade import Ledger\n",
    )
    path = _write(source_tree, CLIENT, "from ..a import Ledger\n")
    assert SEALED in boundaries.collect_trade_rl_imports(path)


def test_import_collection_is_fresh_after_source_changes(source_tree: Path) -> None:
    path = _write(source_tree, CLIENT, "from ..facade import Safe\n")
    assert boundaries._offenders(path.parent, (SEALED,)) == []
    path.write_text("from ..facade import Ledger\n", encoding="utf-8")
    assert boundaries._offenders(path.parent, (SEALED,)) == [CLIENT]


def test_star_import_respects_literal_all(source_tree: Path) -> None:
    facade = source_tree / "trade_rl/evaluation/facade.py"
    facade.write_text(facade.read_text() + "__all__ = ['Safe']\n", encoding="utf-8")
    path = _write(source_tree, CLIENT, "from ..facade import *\n")
    assert boundaries._offenders(path.parent, (SEALED,)) == []
    facade.write_text(
        facade.read_text().replace("['Safe']", "['Ledger']"), encoding="utf-8"
    )
    assert boundaries._offenders(path.parent, (SEALED,)) == [CLIENT]


def test_star_import_without_all_resolves_public_reexports(source_tree: Path) -> None:
    path = _write(source_tree, CLIENT, "from ..facade import *\n")
    assert SEALED in boundaries.collect_trade_rl_imports(path)


def test_dynamic_star_exports_fail_closed_without_execution(source_tree: Path) -> None:
    facade = source_tree / "trade_rl/evaluation/facade.py"
    facade.write_text(
        facade.read_text() + "__all__ = discover_exports()\n", encoding="utf-8"
    )
    path = _write(source_tree, CLIENT, "from ..facade import *\n")
    with pytest.raises(ValueError, match="__all__"):
        boundaries.collect_imports(path)


def test_invalid_relative_depth_is_not_silently_ignored(source_tree: Path) -> None:
    path = _write(source_tree, CLIENT, "from .....outside import value\n")
    with pytest.raises((ValueError, ImportError)):
        boundaries.collect_imports(path)


def test_malformed_source_fails_closed(source_tree: Path) -> None:
    path = _write(source_tree, CLIENT, "from . import (\n")
    with pytest.raises(SyntaxError):
        boundaries.collect_imports(path)


def test_chained_star_reexports_preserve_named_and_star_ownership(
    source_tree: Path,
) -> None:
    _write(source_tree, "trade_rl/evaluation/bridge.py", "from .facade import *\n")
    for spelling in ("Ledger", "*"):
        path = _write(source_tree, CLIENT, f"from ..bridge import {spelling}\n")
        assert SEALED in boundaries.collect_trade_rl_imports(path)


def test_module_alias_reexport_is_traced(source_tree: Path) -> None:
    _write(
        source_tree,
        "trade_rl/evaluation/bridge.py",
        f"import {SEALED} as Ledger\n",
    )
    path = _write(source_tree, CLIENT, "from ..bridge import Ledger\n")
    assert SEALED in boundaries.collect_trade_rl_imports(path)


def test_local_import_is_not_a_public_reexport(source_tree: Path) -> None:
    _write(
        source_tree,
        "trade_rl/evaluation/bridge.py",
        "class Scope:\n    from .facade import Ledger as Safe\nclass Safe: pass\n",
    )
    path = _write(source_tree, CLIENT, "from ..bridge import Safe\n")
    assert boundaries._offenders(path.parent, (SEALED,)) == []


def test_conditional_reexport_is_not_hidden(source_tree: Path) -> None:
    _write(
        source_tree,
        "trade_rl/evaluation/bridge.py",
        "if FLAG:\n    from .facade import Ledger\n",
    )
    path = _write(source_tree, CLIENT, "from ..bridge import Ledger\n")
    assert SEALED in boundaries.collect_trade_rl_imports(path)


def test_star_cycle_terminates_and_preserves_export(source_tree: Path) -> None:
    _write(source_tree, "trade_rl/evaluation/a.py", "from .b import *\n")
    _write(
        source_tree,
        "trade_rl/evaluation/b.py",
        "from .a import *\nfrom .facade import Ledger\n",
    )
    path = _write(source_tree, CLIENT, "from ..a import *\n")
    assert SEALED in boundaries.collect_trade_rl_imports(path)


@pytest.mark.parametrize(
    "suffix",
    ["__all__ += ['Ledger']\n", "__all__.append('Ledger')\n"],
)
def test_mutated_star_exports_fail_closed(source_tree: Path, suffix: str) -> None:
    facade = source_tree / "trade_rl/evaluation/facade.py"
    facade.write_text(
        facade.read_text() + "__all__ = ['Safe']\n" + suffix, encoding="utf-8"
    )
    path = _write(source_tree, CLIENT, "from ..facade import *\n")
    with pytest.raises(ValueError, match="__all__"):
        boundaries.collect_imports(path)


@pytest.mark.parametrize("exports", ["[]", "()"])
def test_empty_literal_all_does_not_reexport_hidden_names(
    source_tree: Path, exports: str
) -> None:
    facade = source_tree / "trade_rl/evaluation/facade.py"
    facade.write_text(facade.read_text() + f"__all__ = {exports}\n", encoding="utf-8")
    path = _write(source_tree, CLIENT, "from ..facade import *\n")
    assert boundaries._offenders(path.parent, (SEALED,)) == []


@pytest.mark.parametrize("exports", ["[1]", "['Safe', 2]", "None", "{'Safe'}"])
def test_non_string_or_non_sequence_all_is_rejected(
    source_tree: Path, exports: str
) -> None:
    facade = source_tree / "trade_rl/evaluation/facade.py"
    facade.write_text(facade.read_text() + f"__all__ = {exports}\n", encoding="utf-8")
    path = _write(source_tree, CLIENT, "from ..facade import *\n")
    with pytest.raises(ValueError, match="__all__"):
        boundaries.collect_imports(path)


def test_explicit_named_import_is_not_hidden_by_all(source_tree: Path) -> None:
    facade = source_tree / "trade_rl/evaluation/facade.py"
    facade.write_text(facade.read_text() + "__all__ = ['Safe']\n", encoding="utf-8")
    path = _write(source_tree, CLIENT, "from ..facade import Ledger\n")
    assert SEALED in boundaries.collect_trade_rl_imports(path)


def test_direct_collection_does_not_follow_symbol_reexports(source_tree: Path) -> None:
    path = _write(source_tree, CLIENT, "from ..facade import Ledger\n")
    direct = ImportCollector(source_tree / "trade_rl").collect_direct(path)
    assert direct == {"trade_rl.evaluation.facade"}
    assert SEALED not in direct


def test_direct_collection_detects_physical_child_module_import(
    source_tree: Path,
) -> None:
    path = _write(
        source_tree,
        CLIENT,
        "from ..robustness.walk_forward import sealed_test\n",
    )
    direct = ImportCollector(source_tree / "trade_rl").collect_direct(path)
    assert "trade_rl.evaluation.robustness.walk_forward" in direct
    assert SEALED in direct


def test_direct_collection_resolves_relative_child_import(source_tree: Path) -> None:
    path = _write(
        source_tree,
        "trade_rl/evaluation/robustness/walk_forward/client.py",
        "from . import sealed_test\n",
    )
    direct = ImportCollector(source_tree / "trade_rl").collect_direct(path)
    assert SEALED in direct


def test_direct_collection_includes_local_scope_imports(source_tree: Path) -> None:
    path = _write(
        source_tree,
        CLIENT,
        "def deferred():\n    from ..robustness.walk_forward import sealed_test\n",
    )
    direct = ImportCollector(source_tree / "trade_rl").collect_direct(path)
    assert SEALED in direct
