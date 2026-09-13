from __future__ import annotations

from importlib import import_module
from importlib.util import find_spec
from pathlib import Path

import pytest

MODULE = "research.issue522_exp003_fresh_reverify_recovery"


def _module():
    spec = find_spec(MODULE)
    assert spec is not None, "Experiment 0003 fresh-reverify recovery helper is not implemented"
    return import_module(MODULE)


def _write_tree(root: Path) -> None:
    (root / "nested").mkdir(parents=True, exist_ok=True)
    (root / "a.txt").write_bytes(b"alpha\n")
    (root / "nested/b.bin").write_bytes(b"\x00\x01\x02")
    (root / ".mutation.lock").write_text("ephemeral", encoding="utf-8")


def test_canonical_tree_digest_is_root_independent(tmp_path: Path) -> None:
    module = _module()
    first = tmp_path / "first-root"
    second = tmp_path / "different-root-name"
    _write_tree(first)
    _write_tree(second)
    assert module.canonical_tree_digest(first) == module.canonical_tree_digest(second)


def test_canonical_tree_digest_tracks_relative_path_and_bytes(tmp_path: Path) -> None:
    module = _module()
    first = tmp_path / "first"
    second = tmp_path / "second"
    _write_tree(first)
    _write_tree(second)
    original = module.canonical_tree_digest(first)

    (second / "nested/b.bin").write_bytes(b"changed")
    assert module.canonical_tree_digest(second) != original

    _write_tree(second)
    (second / "nested/b.bin").rename(second / "nested/c.bin")
    assert module.canonical_tree_digest(second) != original


def test_canonical_tree_digest_ignores_mutation_lock_only(tmp_path: Path) -> None:
    module = _module()
    root = tmp_path / "root"
    _write_tree(root)
    before = module.canonical_tree_digest(root)
    (root / ".mutation.lock").write_text("different", encoding="utf-8")
    assert module.canonical_tree_digest(root) == before


def test_canonical_tree_digest_rejects_missing_or_empty_tree(tmp_path: Path) -> None:
    module = _module()
    with pytest.raises(RuntimeError, match="missing"):
        module.canonical_tree_digest(tmp_path / "missing")
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(RuntimeError, match="no immutable files"):
        module.canonical_tree_digest(empty)
