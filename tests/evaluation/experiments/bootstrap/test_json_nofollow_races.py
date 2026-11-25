"""Deterministic bootstrap JSON path-swap and no-follow regressions (synthetic)."""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from functools import partial
from pathlib import Path
from typing import BinaryIO

import pytest

from tests.evaluation.experiments.bootstrap.test_config import _valid_payload
from trade_rl.artifacts import verified_file
from trade_rl.evaluation.experiments.bootstrap.binance import (
    _read_json_object as read_vision_json,
)
from trade_rl.evaluation.experiments.bootstrap.config import (
    load_canonical_m2_bootstrap_config,
)
from trade_rl.evaluation.experiments.bootstrap.workflow import (
    _read_json_object as read_bootstrap_manifest,
)

_READERS = (
    pytest.param(
        "bootstrap config",
        load_canonical_m2_bootstrap_config,
        id="bootstrap-config",
    ),
    pytest.param(
        "Vision plan",
        partial(read_vision_json, label="Vision plan"),
        id="vision-plan",
    ),
    pytest.param(
        "bootstrap manifest",
        partial(read_bootstrap_manifest, label="bootstrap manifest"),
        id="bootstrap-manifest",
    ),
)


@pytest.mark.skipif(os.name != "posix", reason="POSIX descriptor no-follow boundary")
@pytest.mark.parametrize(("label", "reader"), _READERS)
def test_bootstrap_read_refuses_symlink_swapped_after_file_check(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    label: str,
    reader: Callable[[Path], object],
) -> None:
    """A verified name must never be reopened through a replacement symlink."""
    source = tmp_path / "published.json"
    source.write_text('{"trusted": true}\n', encoding="utf-8")
    outside = tmp_path / "outside.json"
    outside.write_text('{"trusted": false}\n', encoding="utf-8")
    original_is_file = Path.is_file
    swapped = False

    def swap_after_file_check(path: Path) -> bool:
        nonlocal swapped
        checked = original_is_file(path)
        if path == source and not swapped:
            assert checked
            source.unlink()
            source.symlink_to(outside)
            swapped = True
        return checked

    monkeypatch.setattr(Path, "is_file", swap_after_file_check)
    with pytest.raises(ValueError, match=f"{label} must be a regular file"):
        reader(source)

    assert swapped
    assert outside.read_bytes() == b'{"trusted": false}\n'


@pytest.mark.skipif(os.name != "posix", reason="POSIX descriptor no-follow boundary")
@pytest.mark.parametrize(("label", "reader"), _READERS)
def test_bootstrap_read_refuses_symlinked_parent(
    tmp_path: Path, label: str, reader: Callable[[Path], object]
) -> None:
    """Reject a linked ancestor even when the leaf is a regular JSON file."""
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "payload.json").write_text('{"trusted": false}\n', encoding="utf-8")
    alias = tmp_path / "alias"
    alias.symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match=f"{label} must be a regular file"):
        reader(alias / "payload.json")


@pytest.mark.parametrize(("label", "reader"), _READERS)
def test_bootstrap_read_preserves_invalid_utf8_boundary(
    tmp_path: Path, label: str, reader: Callable[[Path], object]
) -> None:
    """The descriptor fix must preserve malformed-JSON failure semantics."""
    source = tmp_path / "invalid.json"
    source.write_bytes(b"\xff{")

    with pytest.raises(ValueError, match=r"(invalid JSON|not valid JSON)"):
        reader(source)


@pytest.mark.parametrize(
    "reader",
    (
        pytest.param(
            partial(read_vision_json, label="Vision plan"),
            id="vision-plan",
        ),
        pytest.param(
            partial(read_bootstrap_manifest, label="bootstrap manifest"),
            id="bootstrap-manifest",
        ),
    ),
)
def test_bootstrap_json_object_read_preserves_valid_payload(
    tmp_path: Path, reader: Callable[[Path], object]
) -> None:
    source = tmp_path / "valid.json"
    source.write_text('{"trusted": true}\n', encoding="utf-8")

    assert reader(source) == {"trusted": True}


@pytest.mark.skipif(os.name != "posix", reason="POSIX regular-file replacement race")
@pytest.mark.parametrize(("label", "reader"), _READERS)
def test_bootstrap_read_refuses_regular_inode_swapped_after_file_check(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    label: str,
    reader: Callable[[Path], object],
) -> None:
    source = tmp_path / "published.json"
    original = json.dumps(_valid_payload()).encode()
    replacement = original.replace(b'"bootstrap_seed": 17', b'"bootstrap_seed": 18')
    assert original != replacement and len(original) == len(replacement)
    source.write_bytes(original)
    foreign = tmp_path / "foreign.json"
    foreign.write_bytes(replacement)
    detached = tmp_path / "detached.json"
    real_is_file = Path.is_file
    swapped = False

    def swap_after_check(path: Path) -> bool:
        nonlocal swapped
        checked = real_is_file(path)
        if path == source and checked and not swapped:
            source.rename(detached)
            foreign.rename(source)
            swapped = True
        return checked

    monkeypatch.setattr(Path, "is_file", swap_after_check)
    with pytest.raises(ValueError, match=f"{label} changed during read"):
        reader(source)

    assert swapped
    assert detached.read_bytes() == original
    assert source.read_bytes() == replacement


@pytest.mark.skipif(os.name != "posix", reason="POSIX descriptor mutation race")
@pytest.mark.parametrize(("label", "reader"), _READERS)
@pytest.mark.parametrize("mutation", ["overwrite", "unlink", "replace"])
def test_bootstrap_read_refuses_source_changed_after_bytes_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    label: str,
    reader: Callable[[Path], object],
    mutation: str,
) -> None:
    source = tmp_path / "published.json"
    original = json.dumps(_valid_payload()).encode()
    replacement = original.replace(b'"bootstrap_seed": 17', b'"bootstrap_seed": 18')
    assert original != replacement and len(original) == len(replacement)
    source.write_bytes(original)
    initial = source.stat()
    real_fdopen = verified_file.os.fdopen
    raced = False

    class MutatingRead:
        def __init__(self, handle: BinaryIO) -> None:
            self.handle = handle

        def fileno(self) -> int:
            return self.handle.fileno()

        def read(self, size: int = -1) -> bytes:
            nonlocal raced
            raw = self.handle.read(size)
            if not raced:
                if mutation == "overwrite":
                    source.write_bytes(replacement)
                    # Even restoring mtime must not hide an in-place write.
                    os.utime(source, ns=(initial.st_atime_ns, initial.st_mtime_ns))
                elif mutation == "unlink":
                    source.unlink()
                else:
                    source.rename(tmp_path / "detached.json")
                    source.write_bytes(replacement)
                raced = True
            return raw

    @contextmanager
    def changing_fdopen(fd: int, *args: object, **kwargs: object) -> Iterator[object]:
        with real_fdopen(fd, *args, **kwargs) as handle:
            current = os.fstat(fd)
            if (current.st_dev, current.st_ino) == (initial.st_dev, initial.st_ino):
                yield MutatingRead(handle)
            else:
                yield handle

    monkeypatch.setattr(verified_file.os, "fdopen", changing_fdopen)
    with pytest.raises(ValueError, match=f"{label} changed during read"):
        reader(source)

    assert raced
    if mutation == "unlink":
        assert not source.exists()
    else:
        assert source.read_bytes() == replacement
