"""Negative oracle for total size limits on untrusted TD8 failure receipts.

The reader must request a finite number of bytes before JSON parsing, rather
than asking the OS for all bytes of an attacker-sized regular file.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import BinaryIO, cast

import pytest

from tests.evaluation.test_decision_value_failure_receipt_robustness import (
    _persist,
    _row,
)
from trade_rl.evaluation import decision_value_dev_study as study


def test_strict_receipt_reader_never_uses_unbounded_file_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "attempt"
    root.mkdir()
    (root / "incomplete-pairs.json").write_bytes(b"{}")
    requested_sizes: list[int] = []

    class ReadProbe:
        def __init__(self, handle: BinaryIO) -> None:
            self.handle = handle

        def fileno(self) -> int:
            return self.handle.fileno()

        def read(self, size: int = -1) -> bytes:
            if size < 0:
                raise AssertionError(
                    "strict receipt reader requested an unbounded read"
                )
            requested_sizes.append(size)
            return self.handle.read(size)

    @contextmanager
    def checked_open(path: Path, *, field: str) -> Iterator[ReadProbe]:
        assert path == root / "incomplete-pairs.json"
        assert field == "failure receipt"
        with path.open("rb") as handle:
            yield ReadProbe(handle)

    monkeypatch.setattr(study, "open_regular_binary", checked_open)
    # Once a bounded read succeeds, a malformed receipt must still fail closed
    # at schema validation rather than being admitted as financial evidence.
    with pytest.raises(ValueError, match="failure receipt fields are malformed"):
        study.load_incomplete_pair_attempt(root)

    assert requested_sizes
    assert all(0 < size <= 64 * 1024 * 1024 + 1 for size in requested_sizes)


def test_oversized_sparse_receipt_is_rejected_before_json_decode(
    tmp_path: Path,
) -> None:
    root = tmp_path / "attempt"
    root.mkdir()
    receipt = root / "incomplete-pairs.json"
    with receipt.open("wb") as handle:
        handle.seek(64 * 1024 * 1024 + 4096)
        handle.write(b"x")
    with pytest.raises(ValueError, match="byte budget"):
        study.load_incomplete_pair_attempt(root)


@pytest.mark.parametrize("budget", (0, -1, True, 0.5, 512 * 1024 * 1024 + 1))
def test_invalid_receipt_byte_budget_fails_closed(
    tmp_path: Path, budget: object
) -> None:
    root = tmp_path / "attempt"
    root.mkdir()
    (root / "incomplete-pairs.json").write_bytes(b"{}")
    with pytest.raises(ValueError, match="max_receipt_bytes"):
        study.load_incomplete_pair_attempt(root, max_receipt_bytes=cast(int, budget))


def test_large_budget_requires_a_trusted_receipt_pin(tmp_path: Path) -> None:
    root = tmp_path / "attempt"
    root.mkdir()
    (root / "incomplete-pairs.json").write_bytes(b"{}")
    with pytest.raises(ValueError, match="pinned digest"):
        study.load_incomplete_pair_attempt(root, max_receipt_bytes=64 * 1024 * 1024 + 1)


def test_reader_rejects_file_growth_while_loading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "attempt"
    root.mkdir()
    receipt = root / "incomplete-pairs.json"
    receipt.write_bytes(b"{}")
    original_open = study.open_regular_binary

    class GrowingRead:
        def __init__(self, handle: BinaryIO) -> None:
            self.handle = handle
            self.reads = 0

        def fileno(self) -> int:
            return self.handle.fileno()

        def read(self, size: int = -1) -> bytes:
            assert size > 0
            raw = self.handle.read(size)
            self.reads += 1
            if self.reads == 1:
                with receipt.open("ab") as writer:
                    writer.write(b"x")
            return raw

    @contextmanager
    def changing_open(path: Path, *, field: str) -> Iterator[GrowingRead]:
        with original_open(path, field=field) as handle:
            yield GrowingRead(handle)

    monkeypatch.setattr(study, "open_regular_binary", changing_open)
    with pytest.raises(ValueError, match="changed during read"):
        study.load_incomplete_pair_attempt(root)


def test_pinned_reader_accepts_real_receipt_at_declared_budget(
    tmp_path: Path,
) -> None:
    root = tmp_path / "attempt"
    digest = _persist(root, _row())
    size = (root / "incomplete-pairs.json").stat().st_size
    assert size > 1
    with pytest.raises(ValueError, match="byte budget"):
        study.load_incomplete_pair_attempt(
            root, expected_receipt_digest=digest, max_receipt_bytes=size - 1
        )
    loaded = study.load_incomplete_pair_attempt(
        root, expected_receipt_digest=digest, max_receipt_bytes=size
    )
    assert loaded["schema_version"] == "decision_value_incomplete_pair_attempt_v3"
    assert loaded["receipt_digest"] == digest


def test_oversized_regular_json_is_rejected_without_mutation(
    tmp_path: Path,
) -> None:
    root = tmp_path / "attempt"
    root.mkdir()
    receipt = root / "incomplete-pairs.json"
    raw = b'{"pairs":[' + b"0," * 2048 + b"0]}"
    receipt.write_bytes(raw)
    with pytest.raises(ValueError, match="byte budget"):
        study.load_incomplete_pair_attempt(root, max_receipt_bytes=1024)
    assert receipt.read_bytes() == raw


def test_reader_rejects_file_truncation_while_loading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "attempt"
    root.mkdir()
    receipt = root / "incomplete-pairs.json"
    receipt.write_bytes(b"{}")
    real_open = study.open_regular_binary

    class TruncatingRead:
        def __init__(self, handle: BinaryIO) -> None:
            self.handle = handle
            self.done = False

        def fileno(self) -> int:
            return self.handle.fileno()

        def read(self, size: int = -1) -> bytes:
            assert size > 0
            result = self.handle.read(size)
            if not self.done:
                self.done = True
                receipt.write_bytes(b"")
            return result

    @contextmanager
    def changing_open(path: Path, *, field: str) -> Iterator[TruncatingRead]:
        with real_open(path, field=field) as handle:
            yield TruncatingRead(handle)

    monkeypatch.setattr(study, "open_regular_binary", changing_open)
    with pytest.raises(ValueError, match="changed during read"):
        study.load_incomplete_pair_attempt(root)


def test_root_enumeration_is_bounded_before_receipt_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A malicious root with many children must not be fully enumerated."""
    root = tmp_path / "attempt"
    root.mkdir()
    (root / "incomplete-pairs.json").write_bytes(b"{}")
    real_scandir = study.os.scandir

    class ScanItem:
        def __init__(self, name: str) -> None:
            self.name = name
            self.path = str(root / name)

    class FloodedScandir:
        def __init__(self) -> None:
            self.count = 0

        def __enter__(self) -> FloodedScandir:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def __iter__(self) -> FloodedScandir:
            return self

        def __next__(self) -> ScanItem:
            self.count += 1
            if self.count == 1:
                return ScanItem("incomplete-pairs.json")
            if self.count == 2:
                return ScanItem("second-file")
            raise AssertionError("unbounded root directory enumeration")

    def checked_scandir(path: object):  # type: ignore[no-untyped-def]
        if Path(path) == root:  # type: ignore[arg-type]
            return FloodedScandir()
        return real_scandir(path)  # type: ignore[arg-type]

    monkeypatch.setattr(study.os, "scandir", checked_scandir)
    with pytest.raises(ValueError, match="root contents are malformed"):
        study.load_incomplete_pair_attempt(root)


def test_deeply_nested_untrusted_receipt_json_fails_as_value_error(
    tmp_path: Path,
) -> None:
    """A very deep JSON tree must not leak a raw parser RecursionError."""
    root = tmp_path / "attempt"
    root.mkdir()
    receipt = root / "incomplete-pairs.json"
    raw = b"[" * 10_000 + b"0" + b"]" * 10_000
    receipt.write_bytes(raw)

    with pytest.raises(ValueError, match="failure receipt JSON"):
        study.load_incomplete_pair_attempt(root)

    assert receipt.read_bytes() == raw


def test_json_canonicalization_recursion_fails_as_value_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A recursion failure after parsing must use the same malformed-JSON contract."""
    root = tmp_path / "attempt"
    root.mkdir()
    receipt = root / "incomplete-pairs.json"
    raw = b'{"x":0}'
    receipt.write_bytes(raw)

    def recursive_canonical_json_bytes(_value: object) -> bytes:
        raise RecursionError("synthetic canonicalization recursion")

    monkeypatch.setattr(study, "canonical_json_bytes", recursive_canonical_json_bytes)
    with pytest.raises(ValueError, match="failure receipt JSON"):
        study.load_incomplete_pair_attempt(root)

    assert receipt.read_bytes() == raw


def test_shallow_non_object_receipt_is_refused_before_json_decode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A broad root array must not allocate decoded objects before type rejection."""
    raw = b"[" + b"0," * 100_000 + b"0]"

    def forbidden_loads(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("non-object receipt reached the JSON decoder")

    monkeypatch.setattr(study.json, "loads", forbidden_loads)
    with pytest.raises(ValueError, match="failure receipt must contain a JSON object"):
        study._decode_incomplete_receipt(raw)


@pytest.mark.parametrize(
    "raw",
    (
        b'{"pairs":[' + b"0," * 100_000 + b"0]}",
        b'{"pairs":[{"note":"quoted } ] { [ delimiters"},0]}',
        b'{"pairs" : [[],0]}',
    ),
)
def test_shallow_object_receipt_rejects_nonobject_pair_members_before_json_decode(
    monkeypatch: pytest.MonkeyPatch, raw: bytes
) -> None:
    """A broad pairs array of non-records must not reach json.loads."""

    def forbidden_loads(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("non-object pair members reached the JSON decoder")

    monkeypatch.setattr(study.json, "loads", forbidden_loads)
    with pytest.raises(
        ValueError, match="failure receipt pairs must contain JSON objects"
    ):
        study._decode_incomplete_receipt(raw)


def test_overdeep_json_is_refused_before_decoder_allocates_nested_objects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only source-structural receipt nesting is admitted to the JSON decoder."""
    raw = b"[" * 33 + b"0" + b"]" * 33
    decoded: list[bool] = []
    original_loads = study.json.loads

    def recording_loads(*args: object, **kwargs: object) -> object:
        decoded.append(True)
        return original_loads(*args, **kwargs)

    monkeypatch.setattr(study.json, "loads", recording_loads)
    with pytest.raises(ValueError, match="failure receipt JSON nesting"):
        study._decode_incomplete_receipt(raw)
    assert not decoded, "an overdeep receipt reached the JSON decoder"


def test_preparse_nesting_ignores_delimiters_inside_escaped_json_strings() -> None:
    """Quoted brackets and escaped quotes must not consume the nesting budget."""
    noise = '[{"\\}]' * 100 + '"}{' * 100
    raw = study.canonical_json_bytes({"text": noise})
    with pytest.raises(ValueError, match="failure receipt fields are malformed"):
        study._decode_incomplete_receipt(raw)


def test_32_nested_arrays_reach_existing_non_object_rejection() -> None:
    """The boundary itself does not alter the established JSON root validation."""
    raw = b"[" * 32 + b"0" + b"]" * 32
    with pytest.raises(ValueError, match="failure receipt must contain a JSON object"):
        study._decode_incomplete_receipt(raw)


def test_deeply_nested_json_objects_are_rejected_before_decode() -> None:
    raw = b'{"child":' * 33 + b"0" + b"}" * 33
    with pytest.raises(ValueError, match="failure receipt JSON nesting"):
        study._decode_incomplete_receipt(raw)


def test_escaped_unicode_control_cannot_corrupt_json_quote_state() -> None:
    """Valid JSON escapes cannot alter the structural preflight quote state."""
    noise = '[{"\\}]' * 3 + chr(14) + '"}{'
    raw = study.canonical_json_bytes(noise)
    with pytest.raises(ValueError, match="failure receipt must contain a JSON object"):
        study._decode_incomplete_receipt(raw)


@pytest.mark.parametrize("escaped", (b"\\n", b"\\t", b"\\u0022", b"\\/"))
def test_escaped_ordinary_byte_cannot_hide_actual_json_nesting(
    monkeypatch: pytest.MonkeyPatch, escaped: bytes
) -> None:
    """An escaped ordinary byte must not hide real 33-level JSON arrays."""
    nested = b"[" * 33 + b"0" + b"]" * 33
    raw = b'{"escape":"' + escaped + b'","nested":' + nested + b"}"
    assert isinstance(study.json.loads(raw), dict), "test input must be valid JSON"

    decoder_calls: list[bool] = []
    original_loads = study.json.loads

    def recording_loads(*args: object, **kwargs: object) -> object:
        decoder_calls.append(True)
        return original_loads(*args, **kwargs)

    monkeypatch.setattr(study.json, "loads", recording_loads)
    with pytest.raises(ValueError, match="failure receipt JSON nesting"):
        study._decode_incomplete_receipt(raw)
    assert decoder_calls == [], "actual nested arrays reached the JSON decoder"
