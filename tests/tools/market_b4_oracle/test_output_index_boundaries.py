"""Adversarial boundaries for the structural native-output file index."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from tools.market_b4_oracle.output_index import ArtifactPin, load_pinned_output_index

_SHA = "a" * 64
_MANIFEST = "output-manifest.json"
_MANIFEST_LIMIT = 4 * 1024**2
_FILE_LIMIT = 8 * 1024**3
_FILE_COUNT = 16_384


def _pin(raw: bytes) -> ArtifactPin:
    return ArtifactPin(len(raw), hashlib.sha256(raw).hexdigest())


def _write(root: Path, files: dict[str, bytes]) -> ArtifactPin:
    manifest: dict[str, dict[str, object]] = {}
    for name, raw in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
        manifest[name] = {"bytes": len(raw), "sha256": _pin(raw).sha256}
    raw_manifest = json.dumps(manifest, sort_keys=True).encode("utf-8")
    (root / _MANIFEST).write_bytes(raw_manifest)
    return _pin(raw_manifest)


def _symlink_or_skip(link: Path, target: Path, *, directory: bool = False) -> None:
    try:
        link.symlink_to(target, target_is_directory=directory)
    except (OSError, NotImplementedError) as error:
        pytest.skip(f"host cannot create the required symlink: {error}")


def test_recheck_detects_same_inode_same_size_mutation_at_read_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "attempt"
    root.mkdir()
    artifact = root / "attempt.json"
    pin = _write(root, {"attempt.json": b"{}"})
    index = load_pinned_output_index(root, pin)
    original_open = Path.open
    mutated = False

    class MutateOnClose:
        def __init__(self, stream: object) -> None:
            self._stream = stream

        def __enter__(self) -> "MutateOnClose":
            self._stream.__enter__()  # type: ignore[attr-defined]
            return self

        def __exit__(self, *exc_info: object) -> object:
            nonlocal mutated
            result = self._stream.__exit__(*exc_info)  # type: ignore[attr-defined]
            if not mutated:
                mutated = True
                with original_open(artifact, "r+b") as stream:
                    stream.seek(0)
                    stream.write(b"[]")
                    stream.flush()
            return result

        def __getattr__(self, name: str) -> object:
            return getattr(self._stream, name)

    def open_with_same_size_mutation(
        path: Path, mode: str = "r", *args: object, **kwargs: object
    ) -> object:
        stream = original_open(path, mode, *args, **kwargs)
        if path == artifact and mode == "rb" and not mutated:
            return MutateOnClose(stream)
        return stream

    monkeypatch.setattr(Path, "open", open_with_same_size_mutation)
    with pytest.raises(ValueError, match="changed|differs"):
        index.recheck()
    assert mutated


def test_rejects_symlink_attempt_root(tmp_path: Path) -> None:
    actual = tmp_path / "actual"
    actual.mkdir()
    pin = _write(actual, {"attempt.json": b"{}"})
    linked = tmp_path / "linked"
    _symlink_or_skip(linked, actual, directory=True)

    with pytest.raises(ValueError, match="link|reparse"):
        load_pinned_output_index(linked, pin)


def test_rejects_symlink_manifest(tmp_path: Path) -> None:
    root = tmp_path / "attempt"
    root.mkdir()
    pin = _write(root, {"attempt.json": b"{}"})
    outside = tmp_path / "manifest.json"
    outside.write_bytes((root / _MANIFEST).read_bytes())
    (root / _MANIFEST).unlink()
    _symlink_or_skip(root / _MANIFEST, outside)

    with pytest.raises(ValueError, match="link|reparse"):
        load_pinned_output_index(root, pin)


def test_rejects_symlink_artifact(tmp_path: Path) -> None:
    root = tmp_path / "attempt"
    root.mkdir()
    pin = _write(root, {"attempt.json": b"{}"})
    outside = tmp_path / "outside.json"
    outside.write_bytes(b"{}")
    (root / "attempt.json").unlink()
    _symlink_or_skip(root / "attempt.json", outside)

    with pytest.raises(ValueError, match="link|reparse"):
        load_pinned_output_index(root, pin)


def test_rejects_fifo_instead_of_blocking_on_it(tmp_path: Path) -> None:
    mkfifo = getattr(os, "mkfifo", None)
    if mkfifo is None:
        pytest.skip("host does not provide FIFO creation")

    root = tmp_path / "attempt"
    root.mkdir()
    pin = _write(root, {"attempt.json": b"{}"})
    try:
        mkfifo(root / "pipe")
    except (OSError, NotImplementedError) as error:
        pytest.skip(f"host cannot create a FIFO: {error}")

    with pytest.raises(ValueError, match="unsupported file type"):
        load_pinned_output_index(root, pin)


def test_rejects_manifest_pin_above_four_mib_without_large_fixture(
    tmp_path: Path,
) -> None:
    oversized_pin = ArtifactPin(_MANIFEST_LIMIT + 1, _SHA)
    with pytest.raises(ValueError, match="metadata limit"):
        load_pinned_output_index(tmp_path, oversized_pin)


def test_rejects_more_than_16384_manifest_entries_with_valid_small_pins(
    tmp_path: Path,
) -> None:
    row = {"bytes": 0, "sha256": _SHA}
    manifest = {f"artifact-{index:05d}.bin": row for index in range(_FILE_COUNT + 1)}
    raw = json.dumps(manifest, separators=(",", ":")).encode("utf-8")
    assert len(raw) < _MANIFEST_LIMIT
    (tmp_path / _MANIFEST).write_bytes(raw)

    with pytest.raises(ValueError, match="bounded nonempty file mapping"):
        load_pinned_output_index(tmp_path, _pin(raw))


def test_rejects_aggregate_eight_gib_limit_with_valid_pins_no_payloads(
    tmp_path: Path,
) -> None:
    four_gib = 4 * 1024**3
    manifest = {
        "first.bin": {"bytes": four_gib, "sha256": _SHA},
        "second.bin": {"bytes": four_gib + 1, "sha256": _SHA},
    }
    raw = json.dumps(manifest, separators=(",", ":")).encode("utf-8")
    (tmp_path / _MANIFEST).write_bytes(raw)

    with pytest.raises(ValueError, match="aggregate read budget"):
        load_pinned_output_index(tmp_path, _pin(raw))


def test_rejects_single_file_pin_over_eight_gib() -> None:
    with pytest.raises(ValueError, match="bounded nonnegative integer"):
        ArtifactPin(_FILE_LIMIT + 1, _SHA)


def test_case_alias_is_rejected_before_any_artifact_is_opened(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    row = {"bytes": 0, "sha256": _SHA}
    raw = json.dumps(
        {"Actor.json": row, "actor.json": row}, separators=(",", ":")
    ).encode("utf-8")
    manifest_path = tmp_path / _MANIFEST
    manifest_path.write_bytes(raw)
    opened: list[Path] = []
    original_open = Path.open

    def record_open(path: Path, mode: str = "r", *args: object, **kwargs: object):
        opened.append(path)
        return original_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", record_open)
    with pytest.raises(ValueError, match="case aliases"):
        load_pinned_output_index(tmp_path, _pin(raw))
    assert opened == [manifest_path]


@pytest.mark.parametrize("alias", ["OUTPUT-MANIFEST.JSON", "Output-Manifest.Json"])
def test_rejects_case_variant_of_manifest_before_roster_comparison(
    tmp_path: Path, alias: str
) -> None:
    """A case alias of the self-excluding manifest is never an artifact entry."""
    row = {"bytes": 0, "sha256": _SHA}
    raw = json.dumps({alias: row}, separators=(",", ":")).encode("utf-8")
    (tmp_path / _MANIFEST).write_bytes(raw)

    with pytest.raises(ValueError, match="self-excluding"):
        load_pinned_output_index(tmp_path, _pin(raw))


def test_rejects_real_manifest_case_alias_file_on_case_sensitive_volume(
    tmp_path: Path,
) -> None:
    """A complete, pinned shadow file must not pass as an extra artifact."""
    root = tmp_path / "attempt"
    root.mkdir()
    alias = "OUTPUT-MANIFEST.JSON"
    (root / alias).write_bytes(b"shadow")
    if (root / _MANIFEST).exists():
        pytest.skip("volume aliases filenames by case; cannot create both files")
    pin = _write(root, {alias: b"shadow"})

    with pytest.raises(ValueError, match="self-excluding"):
        load_pinned_output_index(root, pin)
