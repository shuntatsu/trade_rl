"""Manufactured native-format file indexes; no process or financial evidence."""

from __future__ import annotations

import gzip
import hashlib
import json
import os
from pathlib import Path

import pytest

from tools.market_b4_oracle.output_index import (
    ArtifactPin,
    capture_output_manifest_pin,
    load_pinned_output_index,
)


def _pin(raw: bytes) -> ArtifactPin:
    return ArtifactPin(len(raw), hashlib.sha256(raw).hexdigest())


def _write(root: Path, files: dict[str, bytes]) -> ArtifactPin:
    manifest = {}
    for name, raw in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
        manifest[name] = {"bytes": len(raw), "sha256": _pin(raw).sha256}
    raw = json.dumps(manifest, sort_keys=True).encode()
    (root / "output-manifest.json").write_bytes(raw)
    return _pin(raw)


def test_indexes_all_original_files_including_empty_streams(tmp_path: Path) -> None:
    pin = _write(tmp_path, {"attempt.json": b"{}", "logs/empty.jsonl": b""})
    index = load_pinned_output_index(tmp_path, pin)
    assert index.files == ("attempt.json", "logs/empty.jsonl")
    assert index.pin("logs/empty.jsonl") == _pin(b"")
    assert index.path("attempt.json") == tmp_path / "attempt.json"
    assert index.manifest_pin == pin
    assert index.scope == "PINNED_NATIVE_OUTPUT_FILES_ONLY"
    assert index.recheck() is None
    with pytest.raises(ValueError):
        index.pin("absent.json")
    with pytest.raises(ValueError):
        index.path("absent.json")


def test_outer_owner_can_capture_a_bounded_manifest_pin(tmp_path: Path) -> None:
    _write(tmp_path, {"attempt.json": b"{}"})

    observed_pin = capture_output_manifest_pin(tmp_path)

    assert load_pinned_output_index(tmp_path, observed_pin).manifest_pin == observed_pin


def test_bounded_read_enforces_pin_limit_replacement_and_growth(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "attempt"
    root.mkdir()
    pin = _write(root, {"attempt.json": b"{}"})
    index = load_pinned_output_index(root, pin)
    assert index.read_pinned_bytes("attempt.json", maximum=2) == b"{}"
    with pytest.raises(ValueError, match="caller read limit"):
        index.read_pinned_bytes("attempt.json", maximum=1)

    path = root / "attempt.json"
    path.unlink()
    path.write_bytes(b"[]")
    with pytest.raises(ValueError, match="replaced|identity|byte length|pin"):
        index.read_pinned_bytes("attempt.json", maximum=2)

    pin = _write(root, {"attempt.json": b"{}"})
    index = load_pinned_output_index(root, pin)
    original_open = Path.open

    class GrowAfterFirstRead:
        def __init__(self, inner) -> None:
            self._inner = inner
            self._grown = False

        def __enter__(self):
            self._inner.__enter__()
            return self

        def __exit__(self, *args):
            return self._inner.__exit__(*args)

        def fileno(self):
            return self._inner.fileno()

        def read(self, size: int = -1):
            data = self._inner.read(size)
            if not self._grown:
                self._grown = True
                with original_open(path, "ab") as append:
                    append.write(b"more than the pinned bytes")
            return data

    def growing_open(target: Path, mode: str = "r", *args, **kwargs):
        inner = original_open(target, mode, *args, **kwargs)
        if target == path and mode == "rb":
            return GrowAfterFirstRead(inner)
        return inner

    monkeypatch.setattr(Path, "open", growing_open)
    with pytest.raises(ValueError, match="grew beyond its pin"):
        index.read_pinned_bytes("attempt.json", maximum=2)


@pytest.mark.parametrize("size", [True, -1, 1.0, "1"])
def test_pin_requires_exact_nonnegative_integer(size: object) -> None:
    with pytest.raises(ValueError):
        ArtifactPin(size, "a" * 64)


@pytest.mark.parametrize("digest", ["A" * 64, "a" * 63, "g" * 64, None])
def test_pin_requires_lowercase_sha256(digest: object) -> None:
    with pytest.raises(ValueError):
        ArtifactPin(0, digest)


@pytest.mark.parametrize(
    "name",
    [
        "../outside",
        "/absolute",
        "C:/drive",
        "\\\\host\\share",
        "a\\b",
        "./attempt.json",
        "a/../attempt.json",
        "a//b",
        "a/./b",
        "a/",
        "a:stream",
        "CON",
        "logs/NUL.json",
        "logs/file.",
        "logs/file ",
        "logs/line\nfeed",
        "output-manifest.json",
        "a/output-manifest.json",
    ],
)
def test_rejects_unsafe_or_self_referential_manifest_paths(
    tmp_path: Path, name: str
) -> None:
    raw = json.dumps({name: {"bytes": 0, "sha256": _pin(b"").sha256}}).encode()
    (tmp_path / "output-manifest.json").write_bytes(raw)
    with pytest.raises(ValueError):
        load_pinned_output_index(tmp_path, _pin(raw))


def test_rejects_unicode_control_character_in_manifest_path(tmp_path: Path) -> None:
    name = "logs/receipt\x7f.json"
    try:
        pin = _write(tmp_path, {name: b"{}"})
    except OSError as error:
        pytest.skip(f"host cannot create a control-character path: {error}")

    with pytest.raises(ValueError, match="unsafe|control"):
        load_pinned_output_index(tmp_path, pin)


@pytest.mark.parametrize(
    "raw",
    [
        b"[]",
        b"{}",
        b'{"a":{"bytes":0,"sha256":"a"},"a":{"bytes":0,"sha256":"b"}}',
        b'{"a":{"bytes":0,"bytes":1,"sha256":"a"}}',
        b'{"a":{"bytes":NaN,"sha256":"a"}}',
        b'{"a":{"bytes":true,"sha256":"a"}}',
        b'{"a":{"bytes":0,"sha256":"a","foreign":1}}',
        b'{"a":{"sha256":"a"}}',
        b'{"a":null}',
        b"{} trailing",
        b"\xff",
    ],
)
def test_rejects_invalid_or_open_manifest_shapes(tmp_path: Path, raw: bytes) -> None:
    (tmp_path / "output-manifest.json").write_bytes(raw)
    with pytest.raises(ValueError):
        load_pinned_output_index(tmp_path, _pin(raw))


def test_rejects_case_aliases_before_reading_missing_artifacts(tmp_path: Path) -> None:
    row = {"bytes": 0, "sha256": _pin(b"").sha256}
    raw = json.dumps({"A.json": row, "a.json": row}).encode()
    (tmp_path / "output-manifest.json").write_bytes(raw)
    with pytest.raises(ValueError, match="alias"):
        load_pinned_output_index(tmp_path, _pin(raw))


@pytest.mark.parametrize("change", ["missing", "extra", "changed", "manifest"])
def test_recheck_detects_observed_output_changes(tmp_path: Path, change: str) -> None:
    pin = _write(tmp_path, {"attempt.json": b"{}"})
    index = load_pinned_output_index(tmp_path, pin)
    if change == "missing":
        (tmp_path / "attempt.json").unlink()
    elif change == "extra":
        (tmp_path / "extra.bin").write_bytes(b"x")
    elif change == "changed":
        (tmp_path / "attempt.json").write_bytes(b"[]")
    else:
        (tmp_path / "output-manifest.json").write_bytes(b"{}")
    with pytest.raises(ValueError):
        index.recheck()


def test_rejects_wrong_manifest_pin_and_changed_artifact(tmp_path: Path) -> None:
    pin = _write(tmp_path, {"attempt.json": b"{}"})
    with pytest.raises(ValueError):
        load_pinned_output_index(tmp_path, ArtifactPin(pin.bytes, "0" * 64))
    (tmp_path / "attempt.json").write_bytes(b"[]")
    with pytest.raises(ValueError):
        load_pinned_output_index(tmp_path, pin)


def test_rejects_hardlink_alias_even_outside_index(tmp_path: Path) -> None:
    root = tmp_path / "attempt"
    root.mkdir()
    pin = _write(root, {"attempt.json": b"{}"})
    os.link(root / "attempt.json", tmp_path / "alias.json")
    with pytest.raises(ValueError, match="link"):
        load_pinned_output_index(root, pin)


def test_rejects_symlink_directory_without_following_it(tmp_path: Path) -> None:
    root, outside = tmp_path / "attempt", tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    pin = _write(root, {"attempt.json": b"{}"})
    try:
        (root / "foreign").symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("host lacks symlink privilege")
    with pytest.raises(ValueError, match="link|reparse"):
        load_pinned_output_index(root, pin)


def test_rejects_relative_attempt_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pin = _write(tmp_path, {"attempt.json": b"{}"})
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError):
        load_pinned_output_index(Path("."), pin)


def test_actor_and_reserve_scans_use_the_same_manifest_bound_original(
    tmp_path: Path,
) -> None:
    raw = gzip.compress(
        b'{"native_receipt":{"rows":[{"actor":1}]},"closing_rows":[{"reserve":1}]}',
        mtime=0,
    )
    pin = _write(tmp_path, {"receipts/arm.json.gz": raw})
    index = load_pinned_output_index(tmp_path, pin)
    for phase, expected in (("actor", {"actor": 1}), ("reserve", {"reserve": 1})):
        reader = index.receipt_rows("receipts/arm.json.gz", phase=phase)
        assert list(reader) == [expected]
        assert reader.proof.compressed_sha256 == _pin(raw).sha256
    (tmp_path / "receipts/arm.json.gz").write_bytes(raw[:-1] + b"x")
    with pytest.raises(ValueError):
        reader = index.receipt_rows("receipts/arm.json.gz", phase="reserve")
        list(reader)
