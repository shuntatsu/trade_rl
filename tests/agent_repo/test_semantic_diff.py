from __future__ import annotations

import subprocess
from pathlib import Path

from tools.agent_repo.semantic_diff import SemanticSignal, semantic_diff


def _git(repository: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _write(root: Path, relative: str, source: str) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")


def _baseline(root: Path) -> None:
    _git(root, "init", "-b", "main")
    _git(root, "config", "user.name", "Test User")
    _git(root, "config", "user.email", "test@example.com")
    _write(root, "trade_rl/__init__.py", "Existing = object()\n")
    _write(root, "trade_rl/evaluation/__init__.py", "def evaluate_performance(): ...\n")
    _write(
        root,
        "trade_rl/example.py",
        "from trade_rl import Existing\n\n"
        "OLD_SCHEMA = 'old_schema_v1'\n"
        "__all__ = ['Existing']\n",
    )
    _write(
        root,
        "trade_rl/private.py",
        '"""Original documentation."""\n\nclass ExistingPrivate:\n    pass\n',
    )
    _git(root, "add", ".")
    _git(root, "commit", "-m", "baseline")


def test_semantic_diff_reports_narrow_review_signals(tmp_path: Path) -> None:
    _baseline(tmp_path)
    _write(
        tmp_path,
        "trade_rl/example.py",
        "from dataclasses import dataclass\n"
        "from pathlib import Path\n"
        "from trade_rl import Existing\n"
        "from trade_rl.evaluation import evaluate_performance\n\n"
        "OLD_SCHEMA = 'old_schema_v2'\n"
        "NEW_SCHEMA = 'new_schema_v1'\n\n"
        "@dataclass(frozen=True)\n"
        "class NewConfig:\n"
        "    value: int\n\n"
        "class PrivateHelper:\n"
        "    pass\n\n"
        "def persist(path: Path) -> None:\n"
        "    path.write_text('x', encoding='utf-8')\n\n"
        "__all__ = ['Existing', 'NewConfig']\n",
    )

    signals = semantic_diff(tmp_path, base_ref="main")

    assert all(isinstance(signal, SemanticSignal) for signal in signals)
    assert tuple(signals) == tuple(sorted(signals))
    kinds = {signal.kind for signal in signals}
    assert {
        "DATA_SHAPE",
        "SCHEMA",
        "PUBLIC_EXPORT",
        "DEPENDENCY",
        "FILESYSTEM_EFFECT",
    } <= kinds
    assert (
        SemanticSignal(
            kind="DATA_SHAPE",
            change="added",
            path="trade_rl/example.py",
            name="NewConfig",
            detail="dataclass",
        )
        in signals
    )
    assert any(
        signal.kind == "PUBLIC_EXPORT"
        and signal.change == "added"
        and signal.name == "NewConfig"
        for signal in signals
    )
    assert any(
        signal.kind == "SCHEMA"
        and signal.name == "OLD_SCHEMA"
        and signal.change == "changed"
        for signal in signals
    )
    assert any(
        signal.kind == "FILESYSTEM_EFFECT"
        and signal.name == "write_text"
        and signal.change == "added"
        for signal in signals
    )


def test_semantic_diff_ignores_docstrings_comments_and_private_plain_classes(
    tmp_path: Path,
) -> None:
    _baseline(tmp_path)
    _write(
        tmp_path,
        "trade_rl/private.py",
        '"""Changed documentation only."""\n\n'
        "# changed comment\n"
        "class ExistingPrivate:\n"
        "    pass\n\n"
        "class AnotherPrivate:\n"
        "    pass\n",
    )

    signals = semantic_diff(tmp_path, base_ref="main")

    private_signals = [
        signal for signal in signals if signal.path == "trade_rl/private.py"
    ]
    assert private_signals == []
