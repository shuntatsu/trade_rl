from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from tools.agent_repo.git_state import read_git_state
from tools.agent_repo.source_index import SourceIndex


def _write(root: Path, relative: str, source: str) -> Path:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")
    return path


def _symlink(link: Path, target: Path, *, target_is_directory: bool = False) -> None:
    link.parent.mkdir(parents=True, exist_ok=True)
    try:
        link.symlink_to(target, target_is_directory=target_is_directory)
    except (NotImplementedError, OSError) as error:
        pytest.skip(f"symlinks unavailable: {error}")


def _git(repository: Path, *args: str) -> None:
    subprocess.run(
        ["git", *args],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )


def test_source_index_rejects_symlinked_production_file(tmp_path: Path) -> None:
    _write(tmp_path, "trade_rl/__init__.py", "")
    external = tmp_path.parent / f"{tmp_path.name}-external.py"
    external.write_text("EXTERNAL = True\n", encoding="utf-8")
    _symlink(tmp_path / "trade_rl" / "escape.py", external)

    with pytest.raises(ValueError, match="symlink|repository"):
        SourceIndex.build(tmp_path)


def test_context_rejects_symlinked_parent_directory(tmp_path: Path) -> None:
    _write(tmp_path, "trade_rl/__init__.py", "")
    external = tmp_path.parent / f"{tmp_path.name}-external-package"
    external.mkdir()
    _write(external, "module.py", "VALUE = 1\n")
    _symlink(
        tmp_path / "trade_rl" / "linked",
        external,
        target_is_directory=True,
    )

    index = SourceIndex.build(tmp_path)
    with pytest.raises(ValueError, match="symlink|repository"):
        index.context("trade_rl/linked/module.py")


def test_context_rejects_symlinked_documentation(tmp_path: Path) -> None:
    _write(tmp_path, "trade_rl/__init__.py", "")
    _write(tmp_path, "trade_rl/example.py", "VALUE = 1\n")
    external = tmp_path.parent / f"{tmp_path.name}-external.md"
    external.write_text("trade_rl.example\n", encoding="utf-8")
    _symlink(tmp_path / "docs" / "architecture" / "escape.md", external)

    index = SourceIndex.build(tmp_path)
    with pytest.raises(ValueError, match="symlink|repository"):
        index.context("trade_rl/example.py")


def test_preflight_rejects_symlinked_active_doc(tmp_path: Path) -> None:
    _git(tmp_path, "init", "-b", "main")
    _git(tmp_path, "config", "user.name", "Test User")
    _git(tmp_path, "config", "user.email", "test@example.com")
    _write(tmp_path, "README.md", "baseline\n")
    _git(tmp_path, "add", "README.md")
    _git(tmp_path, "commit", "-m", "baseline")

    external = tmp_path.parent / f"{tmp_path.name}-active.md"
    external.write_text("Status: Active\n", encoding="utf-8")
    _symlink(tmp_path / "docs" / "specs" / "escape.md", external)

    with pytest.raises(ValueError, match="symlink|repository"):
        read_git_state(tmp_path)
