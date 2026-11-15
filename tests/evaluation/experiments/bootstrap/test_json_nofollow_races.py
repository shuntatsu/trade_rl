"""Deterministic bootstrap JSON path-swap and no-follow regressions (synthetic)."""

from __future__ import annotations

import os
from functools import partial
from pathlib import Path

import pytest

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
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, label: str, reader: object
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
        reader(source)  # type: ignore[operator]

    assert swapped
    assert outside.read_bytes() == b'{"trusted": false}\n'


@pytest.mark.skipif(os.name != "posix", reason="POSIX descriptor no-follow boundary")
@pytest.mark.parametrize(("label", "reader"), _READERS)
def test_bootstrap_read_refuses_symlinked_parent(
    tmp_path: Path, label: str, reader: object
) -> None:
    """Reject a linked ancestor even when the leaf is a regular JSON file."""
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "payload.json").write_text('{"trusted": false}\n', encoding="utf-8")
    alias = tmp_path / "alias"
    alias.symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match=f"{label} must be a regular file"):
        reader(alias / "payload.json")  # type: ignore[operator]


@pytest.mark.parametrize(("label", "reader"), _READERS)
def test_bootstrap_read_preserves_invalid_utf8_boundary(
    tmp_path: Path, label: str, reader: object
) -> None:
    """The descriptor fix must preserve malformed-JSON failure semantics."""
    source = tmp_path / "invalid.json"
    source.write_bytes(b"\xff{")

    with pytest.raises(ValueError, match=r"(invalid JSON|not valid JSON)"):
        reader(source)  # type: ignore[operator]


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
    tmp_path: Path, reader: object
) -> None:
    source = tmp_path / "valid.json"
    source.write_text('{"trusted": true}\n', encoding="utf-8")

    assert reader(source) == {"trusted": True}  # type: ignore[operator]
