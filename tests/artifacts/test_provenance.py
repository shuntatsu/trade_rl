from __future__ import annotations

from pathlib import Path

import pytest

from trade_rl.artifacts.provenance import (
    capture_runtime_provenance,
    source_tree_digest,
)


def _source_tree(root: Path, *, marker: str = "same") -> None:
    (root / "trade_rl").mkdir(parents=True)
    (root / "examples").mkdir()
    (root / "pyproject.toml").write_text(
        "[project]\nname='trade-rl'\n", encoding="utf-8"
    )
    (root / "uv.lock").write_text("same-lock", encoding="utf-8")
    (root / "trade_rl" / "module.py").write_text(marker, encoding="utf-8")
    (root / "examples" / "runner.py").write_text("runner", encoding="utf-8")


def _capture(root: Path, **overrides: object):
    values: dict[str, object] = {
        "git_commit": "a" * 40,
        "git_dirty": False,
        "deterministic_seed_config": {"seeds": (1, 2)},
        "package_versions": {"numpy": "2.0"},
        "python_version": "3.12.0",
        "platform_name": "test-platform",
        "hardware_name": "test-hardware",
    }
    values.update(overrides)
    return capture_runtime_provenance(root, **values)


def test_runtime_provenance_is_content_addressed_and_path_independent(
    tmp_path: Path,
) -> None:
    left = tmp_path / "left"
    right = tmp_path / "right"
    _source_tree(left)
    _source_tree(right)

    first = _capture(left)
    second = _capture(right)

    assert first == second
    assert first.digest == second.digest
    assert first.lockfile_digest is not None
    assert first.source_tree_digest == source_tree_digest(left)
    assert "/left" not in repr(first)


def test_source_tree_digest_changes_when_packaged_source_changes(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    _source_tree(root)
    before = source_tree_digest(root)

    (root / "trade_rl" / "module.py").write_text("changed", encoding="utf-8")

    assert source_tree_digest(root) != before


def test_runtime_provenance_rejects_packaged_source_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "repo"
    _source_tree(root)
    monkeypatch.setenv("TRADE_RL_SOURCE_TREE_DIGEST", "f" * 64)

    with pytest.raises(ValueError, match="source tree digest"):
        _capture(root)


def test_runtime_provenance_rejects_packaged_lockfile_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "repo"
    _source_tree(root)
    monkeypatch.setenv("TRADE_RL_LOCKFILE_DIGEST", "f" * 64)

    with pytest.raises(ValueError, match="lockfile digest"):
        _capture(root)


def test_runtime_provenance_binds_container_image_digest(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    _source_tree(root)

    first = _capture(root, image_digest="1" * 64)
    second = _capture(root, image_digest="2" * 64)

    assert first.image_digest == "1" * 64
    assert first.digest != second.digest
