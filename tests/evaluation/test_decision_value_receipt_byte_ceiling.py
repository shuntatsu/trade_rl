"""Negative oracle for total size limits on untrusted TD8 failure receipts.

The reader must request a finite number of bytes before JSON parsing, rather
than asking the OS for all bytes of an attacker-sized regular file.
"""

from __future__ import annotations

import os
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


@pytest.mark.parametrize("budget_mib", (64, 512))
def test_strict_reader_sizes_buffer_to_real_file_not_the_large_allowed_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, budget_mib: int
) -> None:
    """A tiny signed receipt must not allocate 64/512 MiB up front at read()."""
    root = tmp_path / "attempt"
    digest = _persist(root, _row())
    size = (root / "incomplete-pairs.json").stat().st_size
    assert 1 < size < 1024 * 1024
    actual_open = study.open_regular_binary
    reads: list[int] = []

    class ReadProbe:
        def __init__(self, handle: BinaryIO) -> None:
            self.handle = handle

        def fileno(self) -> int:
            return self.handle.fileno()

        def read(self, request: int = -1) -> bytes:
            reads.append(request)
            assert 0 < request <= size + 1, (
                "read request must be bounded by stat size, not budget"
            )
            return self.handle.read(request)

    @contextmanager
    def checked_open(path: Path, *, field: str) -> Iterator[ReadProbe]:
        with actual_open(path, field=field) as handle:
            yield ReadProbe(handle)

    monkeypatch.setattr(study, "open_regular_binary", checked_open)
    result = study.load_incomplete_pair_attempt(
        root,
        expected_receipt_digest=digest,
        max_receipt_bytes=budget_mib * 1024 * 1024,
    )
    assert result["ineligible_pairs"] == 1
    assert len(reads) >= 2 and all(request > 0 for request in reads)


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


@pytest.mark.skipif(os.name != "posix", reason="directory-fd source identity is POSIX")
def test_root_swap_after_inventory_never_admits_foreign_valid_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A valid, self-signed foreign root cannot replace the inspected directory."""
    original_root = tmp_path / "attempt"
    foreign_root = tmp_path / "foreign"
    detached = tmp_path / "detached"
    original_digest = _persist(original_root, _row())
    foreign_digest = _persist(foreign_root, _row(action=(-1, 0)))
    assert original_digest != foreign_digest
    original_bytes = (original_root / "incomplete-pairs.json").read_bytes()
    foreign_bytes = (foreign_root / "incomplete-pairs.json").read_bytes()

    real_scandir = study.os.scandir
    real_loads = study.json.loads
    replaced = False
    decoded = 0

    @contextmanager
    def switch_after_inventory(path: object) -> Iterator[object]:
        nonlocal replaced
        with real_scandir(path) as entries:  # type: ignore[arg-type]
            yield entries
        if not replaced and (path == original_root or isinstance(path, int)):
            original_root.rename(detached)
            foreign_root.rename(original_root)
            replaced = True

    def counted_decode(*args: object, **kwargs: object) -> object:
        nonlocal decoded
        decoded += 1
        return real_loads(*args, **kwargs)

    monkeypatch.setattr(study.os, "scandir", switch_after_inventory)
    monkeypatch.setattr(study.json, "loads", counted_decode)
    with pytest.raises(ValueError, match="failure receipt root changed"):
        study.load_incomplete_pair_attempt(original_root)

    assert replaced
    assert decoded == 0
    assert (detached / "incomplete-pairs.json").read_bytes() == original_bytes
    assert (original_root / "incomplete-pairs.json").read_bytes() == foreign_bytes


@pytest.mark.skipif(os.name != "posix", reason="inode swap is POSIX specific")
def test_receipt_leaf_swap_after_read_is_rejected_before_json_decode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Even a pinned digest cannot authorize a pathname redirected after open."""
    root = tmp_path / "attempt"
    other = tmp_path / "other"
    original_digest = _persist(root, _row())
    _persist(other, _row(action=(-1, 0)))
    receipt = root / "incomplete-pairs.json"
    replacement = other / "incomplete-pairs.json"
    detached = tmp_path / "detached-receipt.json"
    old_bytes = receipt.read_bytes()
    new_bytes = replacement.read_bytes()
    real_open = study.open_regular_binary
    real_loads = study.json.loads
    swapped = False
    decoded = 0

    class SwappingRead:
        def __init__(self, handle: BinaryIO) -> None:
            self.handle = handle

        def fileno(self) -> int:
            return self.handle.fileno()

        def read(self, size: int = -1) -> bytes:
            nonlocal swapped
            data = self.handle.read(size)
            if not swapped:
                receipt.rename(detached)
                replacement.rename(receipt)
                swapped = True
            return data

    @contextmanager
    def changing_open(path: Path, *, field: str) -> Iterator[SwappingRead]:
        with real_open(path, field=field) as handle:
            yield SwappingRead(handle)

    def counted_decode(*args: object, **kwargs: object) -> object:
        nonlocal decoded
        decoded += 1
        return real_loads(*args, **kwargs)

    monkeypatch.setattr(study, "open_regular_binary", changing_open)
    monkeypatch.setattr(study.json, "loads", counted_decode)

    # The existing ctime fence may reject the unlink before the new held-root
    # identity check; both must reject before JSON can be decoded.
    with pytest.raises(ValueError, match="failure receipt (root )?changed"):
        study.load_incomplete_pair_attempt(
            root, expected_receipt_digest=original_digest
        )

    assert swapped
    assert decoded == 0
    assert detached.read_bytes() == old_bytes
    assert receipt.read_bytes() == new_bytes


@pytest.mark.skipif(os.name != "posix", reason="directory-fd inventory is POSIX")
def test_ab_a_directory_swap_during_root_inventory_must_refuse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Inspect the pinned inode, never a temporary clean pathname alias."""
    root = tmp_path / "attempt"
    clean_foreign = tmp_path / "foreign"
    detached = tmp_path / "detached"
    original_digest = _persist(root, _row())
    foreign_digest = _persist(clean_foreign, _row(action=(-1, 0)))
    assert original_digest != foreign_digest
    (root / "extra.json").write_bytes(b"should-refuse")
    original_bytes = (root / "incomplete-pairs.json").read_bytes()
    foreign_bytes = (clean_foreign / "incomplete-pairs.json").read_bytes()
    real_scandir = study.os.scandir
    real_loads = study.json.loads
    swapped = False
    decoder_calls = 0

    @contextmanager
    def swap_only_while_inventory_open(path: object) -> Iterator[object]:
        nonlocal swapped
        pinned_inode = (
            isinstance(path, int) and os.fstat(path).st_ino == root.stat().st_ino
        )
        if path == root or pinned_inode:
            assert not swapped
            root.rename(detached)
            clean_foreign.rename(root)
            swapped = True
            try:
                # Old code sees B's single entry; corrected descriptor scan
                # still sees both entries of pinned A.
                with real_scandir(path) as entries:  # type: ignore[arg-type]
                    yield entries
            finally:
                root.rename(clean_foreign)
                detached.rename(root)
        else:
            with real_scandir(path) as entries:  # type: ignore[arg-type]
                yield entries

    def counted_decode(*args: object, **kwargs: object) -> object:
        nonlocal decoder_calls
        decoder_calls += 1
        return real_loads(*args, **kwargs)

    monkeypatch.setattr(study.os, "scandir", swap_only_while_inventory_open)
    monkeypatch.setattr(study.json, "loads", counted_decode)
    with pytest.raises(ValueError, match="failure receipt root contents are malformed"):
        study.load_incomplete_pair_attempt(
            root, expected_receipt_digest=original_digest
        )

    assert swapped
    assert decoder_calls == 0
    assert (root / "extra.json").read_bytes() == b"should-refuse"
    assert (root / "incomplete-pairs.json").read_bytes() == original_bytes
    assert (clean_foreign / "incomplete-pairs.json").read_bytes() == foreign_bytes


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
        same_root_fd = isinstance(path, int) and (
            os.fstat(path).st_dev,
            os.fstat(path).st_ino,
        ) == (root.stat().st_dev, root.stat().st_ino)
        if path == root or same_root_fd:
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
    raw = b'{"fit_admitted":false}'
    receipt.write_bytes(raw)

    def recursive_canonical_json_bytes(_value: object) -> bytes:
        raise RecursionError("synthetic canonicalization recursion")

    monkeypatch.setattr(study, "canonical_json_bytes", recursive_canonical_json_bytes)
    with pytest.raises(ValueError, match="failure receipt JSON"):
        study.load_incomplete_pair_attempt(root)

    assert receipt.read_bytes() == raw


@pytest.mark.parametrize(
    "field", ("block_id", "fit_admitted", "ineligible_pairs", "observed_view_digest")
)
def test_scalar_root_field_rejects_broad_array_before_json_decode(
    monkeypatch: pytest.MonkeyPatch, field: str
) -> None:
    """Schema-scalar root members cannot expand a wide JSON container."""
    raw = b'{"' + field.encode("ascii") + b'":[' + b"0," * 100_000 + b"0]}"

    def forbidden_loads(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("scalar root array reached the JSON decoder")

    monkeypatch.setattr(study.json, "loads", forbidden_loads)
    with pytest.raises(ValueError, match="failure receipt root scalar field"):
        study._decode_incomplete_receipt(raw)


@pytest.mark.parametrize(
    "field", ("schema", "intervention_index", "admissible_for_fit")
)
def test_scalar_pair_field_rejects_broad_array_before_json_decode(
    monkeypatch: pytest.MonkeyPatch, field: str
) -> None:
    """Direct pair scalars must be checked before allocating parsed arrays."""
    pair = study.canonical_json_bytes(dict.fromkeys(study._PAIR_KEYS))
    key = b'"' + field.encode("ascii") + b'":null'
    assert pair.count(key) == 1
    pair = pair.replace(key, key[:-4] + b"[" + b"0," * 100_000 + b"0]")
    raw = b'{"pairs":[' + pair + b"]}"

    def forbidden_loads(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("scalar pair array reached the JSON decoder")

    monkeypatch.setattr(study.json, "loads", forbidden_loads)
    with pytest.raises(ValueError, match="failure receipt pair scalar field"):
        study._decode_incomplete_receipt(raw)


@pytest.mark.parametrize(
    ("raw", "message"),
    (
        (b'{"block_id":{}}', "root scalar field"),
        (b'{"pairs":{}}', "pairs must be a JSON array"),
        (
            b'{"pairs":['
            + study.canonical_json_bytes(dict.fromkeys(study._PAIR_KEYS)).replace(
                b'"schema":null', b'"schema":{}'
            )
            + b"]}",
            "pair scalar field",
        ),
    ),
)
def test_invalid_root_and_pair_object_types_refuse_before_json_decode(
    monkeypatch: pytest.MonkeyPatch, raw: bytes, message: str
) -> None:
    """Only schema-allowed containers may cross the untrusted JSON boundary."""

    def forbidden_loads(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("malformed container reached JSON decoder")

    monkeypatch.setattr(study.json, "loads", forbidden_loads)
    with pytest.raises(ValueError, match=message):
        study._decode_incomplete_receipt(raw)


@pytest.mark.parametrize(
    ("field", "wrong_container", "message"),
    (
        ("baseline", "array", "pair object field"),
        ("candidate", "array", "pair object field"),
        ("reference_vintage", "array", "pair object field"),
        ("unvalidated_source_record", "array", "pair object field"),
        ("candidate_action", "object", "pair array field"),
        ("reason_codes", "object", "pair array field"),
    ),
)
def test_wrong_pair_container_kind_is_refused_before_json_decode(
    monkeypatch: pytest.MonkeyPatch, field: str, wrong_container: str, message: str
) -> None:
    """An impossible direct-pair container must not be expanded by json.loads."""
    pair = dict.fromkeys(study._PAIR_KEYS)
    pair.update(baseline={}, candidate={}, candidate_action=[0], reason_codes=["x"])
    pair[field] = {} if wrong_container == "array" else []
    original = study.canonical_json_bytes(pair)
    key = b'"' + field.encode("ascii") + b'":'
    original_container = b"{}" if wrong_container == "array" else b"[]"
    assert original.count(key + original_container) == 1
    if wrong_container == "array":
        hostile = b"[" + b"0," * 100_000 + b"0]"
    else:
        hostile = (
            b"{"
            + b",".join(b'"k' + str(i).encode("ascii") + b'":0' for i in range(20_000))
            + b"}"
        )
    pair_raw = original.replace(key + original_container, key + hostile, 1)
    raw = b'{"pairs":[' + pair_raw + b"]}"

    def forbidden_loads(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("impossible pair container reached JSON decoder")

    monkeypatch.setattr(study.json, "loads", forbidden_loads)
    with pytest.raises(ValueError, match="failure receipt " + message):
        study._decode_incomplete_receipt(raw)


@pytest.mark.parametrize(
    ("field", "container", "message"),
    (
        ("baseline", b"{}", "pair object field"),
        ("candidate", b"{}", "pair object field"),
        ("reference_vintage", b"{}", "pair object field"),
        ("unvalidated_source_record", b"{}", "pair object field"),
        ("candidate_action", b"[]", "pair array field"),
        ("reason_codes", b"[]", "pair array field"),
    ),
)
@pytest.mark.parametrize("wrong_value", (b"1" * 100_000, b'"' + b"x" * 100_000 + b'"'))
def test_pair_container_field_rejects_wrong_primitive_before_json_decode(
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    container: bytes,
    message: str,
    wrong_value: bytes,
) -> None:
    """A schema-impossible primitive may not allocate a huge decoded string."""
    pair = dict.fromkeys(study._PAIR_KEYS)
    pair.update(baseline={}, candidate={}, candidate_action=[], reason_codes=[])
    if field in {"reference_vintage", "unvalidated_source_record"}:
        pair[field] = {}
    raw_pair = study.canonical_json_bytes(pair)
    target = b'"' + field.encode("ascii") + b'":' + container
    assert raw_pair.count(target) == 1
    raw = (
        b'{"pairs":['
        + raw_pair.replace(target, target[: -len(container)] + wrong_value, 1)
        + b"]}"
    )

    def forbidden_loads(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("incorrect primitive reached JSON decoding")

    monkeypatch.setattr(study.json, "loads", forbidden_loads)
    with pytest.raises(ValueError, match="failure receipt " + message):
        study._decode_incomplete_receipt(raw)


@pytest.mark.parametrize(
    ("field", "bad_value", "message"),
    (
        ("baseline", b"[0]", "pair object field"),
        ("reason_codes", b"{}", "pair array field"),
    ),
)
def test_wrong_pair_container_in_second_pair_with_whitespace_fails_predecode(
    monkeypatch: pytest.MonkeyPatch, field: str, bad_value: bytes, message: str
) -> None:
    """Whitespace and a preceding nested object do not reset pair-level checking."""
    first = dict.fromkeys(study._PAIR_KEYS)
    first.update(
        baseline={"candidate_action": {"reason_codes": [1]}},
        candidate={},
        candidate_action=[0],
        reason_codes=["x"],
    )
    second = dict.fromkeys(study._PAIR_KEYS)
    second.update(baseline={}, candidate={}, candidate_action=[], reason_codes=[])
    first_raw = study.canonical_json_bytes(first)
    second_raw = study.canonical_json_bytes(second)
    target = (
        b'"' + field.encode("ascii") + b'":' + (b"{}" if field == "baseline" else b"[]")
    )
    assert second_raw.count(target) == 1
    second_raw = second_raw.replace(target, target[:-2] + b" \n " + bad_value, 1)
    raw = b'{"pairs":[' + first_raw + b"," + second_raw + b"]}"

    def forbidden_loads(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("wrong container in second pair reached JSON decoder")

    monkeypatch.setattr(study.json, "loads", forbidden_loads)
    with pytest.raises(ValueError, match="failure receipt " + message):
        study._decode_incomplete_receipt(raw)


def test_nested_pair_key_decoys_do_not_change_direct_pair_type_detection() -> None:
    """Only direct field names are subject to the preflight container rule."""
    pair = dict.fromkeys(study._PAIR_KEYS)
    pair.update(
        baseline={"reason_codes": {"candidate_action": ["nested"]}},
        candidate={},
        candidate_action=[0],
        reason_codes=["reason"],
        reference_vintage={"baseline": [0]},
    )
    study._preflight_receipt_pair_objects(study.canonical_json_bytes({"pairs": [pair]}))


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
        # A complete first pair must not mask the invalid second member.
        (
            b'{"pairs":[{"admissible_for_fit":false,"baseline":{},'
            b'"candidate":{},"candidate_action":[],"intervention_index":0,'
            b'"prefix_start_index":0,"terminal_index":0,'
            b'"schema":"decision_value_ineligible_pair_v1",'
            b'"reason_codes":["quoted } ] { [ delimiters"]},0]}'
        ),
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


@pytest.mark.parametrize(
    ("raw", "expected_message"),
    (
        pytest.param(
            b'{"pairs":[],"pairs":[' + b"0," * 100_000 + b"0]}",
            "failure receipt pairs key is duplicated",
            id="duplicate-top-level-pairs",
        ),
        pytest.param(
            b'{"observed_view_digest":{"pairs":[]},"pairs":['
            + b"0," * 100_000
            + b"0]}",
            "failure receipt root scalar field",
            id="nested-decoy",
        ),
        pytest.param(
            b'{"pa\\u0069rs":[' + b"0," * 100_000 + b"0]}",
            "failure receipt root keys must be unescaped",
            id="escaped-top-level-key-i",
        ),
        pytest.param(
            b'{"pair\\u0073":[' + b"0," * 100_000 + b"0]}",
            "failure receipt root keys must be unescaped",
            id="escaped-top-level-key-s",
        ),
    ),
)
def test_pairs_key_spoofing_is_rejected_before_json_decode(
    monkeypatch: pytest.MonkeyPatch, raw: bytes, expected_message: str
) -> None:
    """Duplicate, decoy and escaped root keys must not allocate broad arrays."""

    def forbidden_loads(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("ambiguous pairs key reached the JSON decoder")

    monkeypatch.setattr(study.json, "loads", forbidden_loads)
    with pytest.raises(ValueError, match=expected_message):
        study._decode_incomplete_receipt(raw)


@pytest.mark.parametrize(
    "key", ("fit_admitted", "schema_version", "observed_view_digest")
)
def test_duplicate_known_root_fields_fail_before_json_decode(
    monkeypatch: pytest.MonkeyPatch, key: str
) -> None:
    """A second allowed root name cannot hide a wide decoder allocation."""
    raw_key = key.encode("ascii")
    raw = b'{"' + raw_key + b'":null,"' + raw_key + b'":['
    raw += b"0," * 16_384 + b"0]}"

    def forbidden_loads(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("duplicate root name reached JSON decoding")

    monkeypatch.setattr(study.json, "loads", forbidden_loads)
    with pytest.raises(ValueError, match="failure receipt root key is duplicated"):
        study._decode_incomplete_receipt(raw)


def test_root_duplicate_guard_ignores_nested_key_lookalikes() -> None:
    """Nested and quoted names cannot forge duplicate direct root fields."""
    pair = dict.fromkeys(study._PAIR_KEYS)
    pair["reference_vintage"] = {"fit_admitted": False}
    raw = study.canonical_json_bytes(
        {"pairs": [pair], "fit_admitted": False, "reason": "quoted fit_admitted pairs"}
    )
    study._preflight_receipt_pair_objects(raw)


@pytest.mark.parametrize(
    ("raw", "message"),
    (
        pytest.param(
            b'{"pairs":[{"intruder":[' + b"0," * 100_000 + b"0]}]}",
            "failure receipt pair field is not recognised",
            id="unrecognised-pair-key-before-broad-array",
        ),
        pytest.param(
            b'{"pairs":[{"reason\\u005fcodes":[' + b"0," * 100_000 + b"0]}]}",
            "failure receipt pair keys must be unescaped",
            id="escaped-pair-key-before-broad-array",
        ),
        pytest.param(
            b'{"pairs":[{"' + b"x" * 100_000 + b'":0}]}',
            "failure receipt pair key exceeds schema length",
            id="oversized-pair-key",
        ),
    ),
)
def test_invalid_pair_member_names_fail_before_decode(
    monkeypatch: pytest.MonkeyPatch, raw: bytes, message: str
) -> None:
    """Refuse invalid direct pair keys before decoding their broad values."""

    def forbidden_loads(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("invalid pair member reached the JSON decoder")

    monkeypatch.setattr(study.json, "loads", forbidden_loads)
    with pytest.raises(ValueError, match=message):
        study._decode_incomplete_receipt(raw)


@pytest.mark.parametrize(
    "pair",
    (
        b"{}",
        b'{"schema":"decision_value_ineligible_pair_v1"}',
        b'{"candidate_action":[1,0],"reason_codes":["blocked"]}',
    ),
)
def test_wide_incomplete_pair_objects_rejected_before_json_decode(
    monkeypatch: pytest.MonkeyPatch, pair: bytes
) -> None:
    """Thousands of shallow but schema-incomplete pair objects must not allocate."""
    raw = b'{"pairs":[' + b",".join((pair,) * 16_384) + b"]}"

    def forbidden_loads(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("incomplete pair records reached JSON decoding")

    monkeypatch.setattr(study.json, "loads", forbidden_loads)
    with pytest.raises(ValueError, match="failure receipt pair fields are malformed"):
        study._decode_incomplete_receipt(raw)


@pytest.mark.parametrize("decoy", (False, True))
def test_missing_direct_fields_cannot_borrow_from_other_pairs_or_nested_objects(
    monkeypatch: pytest.MonkeyPatch, decoy: bool
) -> None:
    """Each direct pair must carry its own required fields, not nested lookalikes."""
    complete = dict.fromkeys(study._PAIR_KEYS)
    incomplete = (
        {"reference_vintage": dict.fromkeys(study._PAIR_KEYS)}
        if decoy
        else {"schema": "decision_value_ineligible_pair_v1"}
    )
    raw = study.canonical_json_bytes({"pairs": [complete, incomplete]})

    def forbidden_loads(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("incomplete direct pair reached JSON decoding")

    monkeypatch.setattr(study.json, "loads", forbidden_loads)
    with pytest.raises(ValueError, match="failure receipt pair fields are malformed"):
        study._decode_incomplete_receipt(raw)


@pytest.mark.parametrize("repeated_key", ("candidate_action", "reason_codes"))
def test_duplicate_direct_pair_keys_rejected_before_json_decode(
    monkeypatch: pytest.MonkeyPatch, repeated_key: str
) -> None:
    """Even a structurally complete pair cannot conceal a broad duplicate value."""
    complete = study.canonical_json_bytes(dict.fromkeys(study._PAIR_KEYS))
    duplicate = b',"' + repeated_key.encode("ascii") + b'":['
    raw = b'{"pairs":[' + complete[:-1] + duplicate + b"0," * 16_384 + b"0]}]}"

    def forbidden_loads(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("duplicate direct pair key reached JSON decoding")

    monkeypatch.setattr(study.json, "loads", forbidden_loads)
    with pytest.raises(ValueError, match="failure receipt pair key is duplicated"):
        study._decode_incomplete_receipt(raw)


def test_pairs_preflight_preserves_nested_and_quoted_delimiters() -> None:
    """Nested arrays and quoted delimiters cannot forge direct pair fields."""
    required = dict.fromkeys(study._PAIR_KEYS)
    raw = study.canonical_json_bytes(
        {
            "pairs": [
                {
                    **required,
                    "reason_codes": ["quoted } ] { ["],
                    "reference_vintage": {"pairs": [0]},
                },
                {
                    **required,
                    "candidate_action": [1, 2],
                    "unvalidated_source_record": {"text": "{ nested ]"},
                },
            ]
        }
    )
    study._preflight_receipt_pair_objects(raw)


@pytest.mark.parametrize(
    "raw",
    (
        b'{"unexpected":[' + b"0," * 100_000 + b"0]}",
        b'{"pairs":[],"alien":[' + b"0," * 100_000 + b"0]}",
    ),
)
def test_unknown_root_member_rejected_before_json_decode(
    monkeypatch: pytest.MonkeyPatch, raw: bytes
) -> None:
    """Unknown direct root names must not allocate broad decoded values."""

    def forbidden_loads(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("unknown root member reached the JSON decoder")

    monkeypatch.setattr(study.json, "loads", forbidden_loads)
    with pytest.raises(ValueError, match="failure receipt fields are malformed"):
        study._decode_incomplete_receipt(raw)


def test_large_root_member_key_rejected_before_decode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Root keys cannot allocate a second unbounded copy before JSON parsing."""
    raw = b'{"' + b"x" * 100_000 + b'":0}'

    def forbidden_loads(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("oversized root key reached the JSON decoder")

    monkeypatch.setattr(study.json, "loads", forbidden_loads)
    with pytest.raises(
        ValueError, match="failure receipt root key exceeds schema length"
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


@pytest.mark.parametrize("declared", (b"0", b"1", b"9" * 16_384))
def test_mismatched_receipt_pair_count_rejected_before_json_decode(
    monkeypatch: pytest.MonkeyPatch, declared: bytes
) -> None:
    """Do not allocate a wide pair list when its declared count is impossible."""
    pair = (
        b'{"admissible_for_fit":false,"baseline":{},"candidate":{},'
        b'"candidate_action":[0,0],"intervention_index":0,'
        b'"prefix_start_index":0,"reason_codes":["blocked"],'
        b'"schema":"decision_value_ineligible_pair_v1","terminal_index":1}'
    )
    raw = b'{"ineligible_pairs":' + declared + b',"pairs":['
    raw += b",".join((pair,) * 1_024) + b"]}"

    def forbidden_loads(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("inconsistent pair count reached JSON decoding")

    monkeypatch.setattr(study.json, "loads", forbidden_loads)
    with pytest.raises(ValueError, match="failure receipt pair count mismatch"):
        study._decode_incomplete_receipt(raw)


def test_matching_receipt_pair_count_remains_reloadable(tmp_path: Path) -> None:
    """Both v3 writer and strict reader accept the unmodified correct count."""
    root = tmp_path / "attempt"
    digest = _persist(root, _row(), _row(action=(0, 1)))
    receipt = study.load_incomplete_pair_attempt(root, expected_receipt_digest=digest)
    assert receipt["ineligible_pairs"] == 2
    assert len(receipt["pairs"]) == 2


def test_broad_consistent_pair_count_reaches_existing_json_validator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The count preflight is not an artificial cap on valid-width arrays."""
    pair = (
        b'{"admissible_for_fit":false,"baseline":{},"candidate":{},'
        b'"candidate_action":[0,0],"intervention_index":0,'
        b'"prefix_start_index":0,"reason_codes":["blocked"],'
        b'"schema":"decision_value_ineligible_pair_v1","terminal_index":1}'
    )
    raw = b'{"ineligible_pairs":1024,"pairs":['
    raw += b",".join((pair,) * 1_024) + b"]}"

    class DecoderReached(Exception):
        pass

    def stop_at_decoder(*_args: object, **_kwargs: object) -> object:
        raise DecoderReached()

    monkeypatch.setattr(study.json, "loads", stop_at_decoder)
    with pytest.raises(DecoderReached):
        study._decode_incomplete_receipt(raw)


@pytest.mark.parametrize("field", ("candidate_action", "reason_codes"))
@pytest.mark.parametrize("kind", ("array", "object"))
@pytest.mark.parametrize("second_pair", (False, True))
def test_flat_pair_arrays_reject_nested_containers_before_decode(
    monkeypatch: pytest.MonkeyPatch, field: str, kind: str, second_pair: bool
) -> None:
    """Scalar-only arrays cannot carry a broad nested allocation."""
    pair = dict.fromkeys(study._PAIR_KEYS)
    pair.update(
        baseline={},
        candidate={},
        candidate_action=[0],
        reason_codes=["blocked"],
        reference_vintage={"candidate_action": [[1]], "reason_codes": [{"x": 0}]},
    )
    original = study.canonical_json_bytes(pair)
    value = b"[0]" if field == "candidate_action" else b'["blocked"]'
    target = b'"' + field.encode("ascii") + b'":' + value
    assert original.count(target) == 1
    nested = (
        b"[" + b"0," * 100_000 + b"0]"
        if kind == "array"
        else b"{"
        + b",".join(b'"k' + str(i).encode("ascii") + b'":0' for i in range(20_000))
        + b"}"
    )
    malformed = original.replace(target, target[:-1] + b", " + nested + b"]", 1)
    raw = (
        b'{"pairs":[' + (original + b",\n " if second_pair else b"") + malformed + b"]}"
    )

    def forbidden_loads(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("nested pair array reached JSON decoding")

    monkeypatch.setattr(study.json, "loads", forbidden_loads)
    with pytest.raises(ValueError, match="failure receipt pair array elements"):
        study._decode_incomplete_receipt(raw)


@pytest.mark.parametrize("case", ["v2", "v3", "quarantine"])
def test_receipt_canonical_identity_without_aggregate_encoding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, case: str
) -> None:
    from tests.evaluation.test_decision_value_failure_receipt_robustness import _as_v2

    root = tmp_path / "attempt"
    rows = (
        [_row(baseline={"terminal_cash": "invalid-" + "\U0001f600" * 70_000})] * 3
        if case == "quarantine"
        else [_row(), _row(action=(0, 1))]
    )
    digest = _persist(root, *rows)
    if case == "v2":
        _as_v2(root)
        digest = str(study.load_incomplete_pair_attempt(root)["receipt_digest"])
    original = (root / "incomplete-pairs.json").read_bytes()
    expected = study.load_incomplete_pair_attempt(root, expected_receipt_digest=digest)
    encode = study.canonical_json_bytes
    hash_value = study.content_digest
    # The unchanged canonical encoder is the independent oracle for both
    # writer output and each digest, even if writer and reader agree wrongly.
    assert original == encode(expected)
    assert expected["pairs_digest"] == hash_value(expected["pairs"])
    assert digest == hash_value(
        {key: value for key, value in expected.items() if key != "receipt_digest"}
    )

    def reject_aggregate(value: object) -> None:
        if isinstance(value, list) or (
            isinstance(value, dict) and "schema_version" in value and "pairs" in value
        ):
            raise AssertionError(
                "aggregate canonical encoder received a complete receipt or pair array"
            )

    def checked_encode(value: object) -> bytes:
        reject_aggregate(value)
        return encode(value)

    def checked_digest(value: object) -> str:
        reject_aggregate(value)
        return hash_value(value)

    monkeypatch.setattr(study, "canonical_json_bytes", checked_encode)
    monkeypatch.setattr(study, "content_digest", checked_digest)
    assert (
        study.load_incomplete_pair_attempt(root, expected_receipt_digest=digest)
        == expected
    )
    assert (root / "incomplete-pairs.json").read_bytes() == original
    if case != "v2":
        # The current writer must emit the same v3 bytes, not a new schema.
        replay = tmp_path / "repeat"
        assert _persist(replay, *rows) == digest
        assert (replay / "incomplete-pairs.json").read_bytes() == original


@pytest.mark.parametrize("legacy_v2", [False, True])
def test_externally_bounded_receipt_does_not_decode_whole_unicode_document(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, legacy_v2: bool
) -> None:
    from dataclasses import replace

    from tests.evaluation.test_decision_value_failure_receipt_robustness import _as_v2

    root = tmp_path / "attempt"
    reason = 'synthetic "\\" delimiters [{,}] ' + "\U0001f600" * 20_000
    rows = (
        replace(_row(), reason_codes=(reason,)),
        replace(_row(action=(0, 1)), reason_codes=(reason + " second",)),
    )
    digest = _persist(root, *rows)
    if legacy_v2:
        _as_v2(root)
        digest = str(study.load_incomplete_pair_attempt(root)["receipt_digest"])
    raw = (root / "incomplete-pairs.json").read_bytes()
    expected = study.load_incomplete_pair_attempt(root, expected_receipt_digest=digest)
    real_loads = study.json.loads
    calls: list[int] = []

    def bounded_loads(value, *args, **kwargs):
        size = len(value.encode("utf-8") if isinstance(value, str) else value)
        calls.append(size)
        assert size < len(raw), "aggregate Unicode document reached JSON decoder"
        return real_loads(value, *args, **kwargs)

    monkeypatch.setattr(study.json, "loads", bounded_loads)
    result = study.load_incomplete_pair_attempt(
        root, expected_receipt_digest=digest, expected_max_ineligible_pairs=2
    )
    assert result == expected and len(calls) >= 3
    assert (root / "incomplete-pairs.json").read_bytes() == raw
    if not legacy_v2:
        repeat = tmp_path / "repeat"
        assert _persist(repeat, *rows) == digest
        assert (repeat / "incomplete-pairs.json").read_bytes() == raw


@pytest.mark.parametrize("separator", [b"", b",,", b",null,", b", "])
def test_partitioned_receipt_refuses_malformed_or_noncanonical_pair_separators(
    tmp_path: Path, separator: bytes
) -> None:
    root = tmp_path / "attempt"
    digest = _persist(root, _row(), _row(action=(0, 1)))
    path = root / "incomplete-pairs.json"
    original = path.read_bytes()
    value = study.json.loads(original)
    first, second = [study.canonical_json_bytes(pair) for pair in value["pairs"]]
    needle = first + b"," + second
    assert original.count(needle) == 1
    malformed = original.replace(needle, first + separator + second, 1)
    path.write_bytes(malformed)
    with pytest.raises(ValueError):
        study.load_incomplete_pair_attempt(
            root, expected_receipt_digest=digest, expected_max_ineligible_pairs=2
        )
    assert path.read_bytes() == malformed


def test_large_bounded_receipt_with_no_complete_pair_layout_never_decodes_whole_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A deliberately lowered default exercises the real large-document path
    # without allocating a large fixture or changing the production limit.
    raw = b'{"block_id":"' + b"x" * 1000 + b'"}'
    monkeypatch.setattr(study, "_DEFAULT_INCOMPLETE_RECEIPT_BYTES", 100)

    def forbidden_loads(*args, **kwargs):
        raise AssertionError("missing large pair layout reached aggregate decoding")

    monkeypatch.setattr(study.json, "loads", forbidden_loads)
    with pytest.raises(ValueError, match="failure receipt JSON is malformed"):
        study._decode_incomplete_receipt(raw, expected_max_ineligible_pairs=2)


@pytest.mark.parametrize(
    "mutation", ["leading", "trailing", "utf8", "duplicate", "root_order"]
)
def test_partitioned_receipt_keeps_full_document_canonical_and_utf8_checks(
    tmp_path: Path,
    mutation: str,
) -> None:
    root = tmp_path / "attempt"
    digest = _persist(root, _row(), _row(action=(0, 1)))
    raw = (root / "incomplete-pairs.json").read_bytes()
    value = study.json.loads(raw)
    if mutation == "leading":
        malformed = raw.replace(b'"pairs":[', b'"pairs":[,', 1)
    elif mutation == "trailing":
        pair = study.canonical_json_bytes(value["pairs"][-1])
        malformed = raw.replace(pair + b"]", pair + b",]", 1)
    elif mutation == "utf8":
        malformed = raw.replace(
            b'"candidate_terminated"', b'"candidate_terminated\xff"', 1
        )
    elif mutation == "duplicate":
        malformed = raw.replace(
            b'"terminal_cash":"99000.5"',
            b'"terminal_cash":"99000.5","terminal_cash":"99000.5"',
            1,
        )
    else:
        malformed = raw.replace(
            b'"actual_artifact_digest":', b' "actual_artifact_digest":', 1
        )
    assert malformed != raw
    with pytest.raises(ValueError):
        study._decode_incomplete_receipt(
            malformed, expected_receipt_digest=digest, expected_max_ineligible_pairs=2
        )


@pytest.mark.parametrize("legacy_v2", [False, True])
@pytest.mark.parametrize("tamper", ["body", "resealed"])
def test_trusted_receipt_pin_refuses_forged_bytes_before_json_scanning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, legacy_v2: bool, tamper: str
) -> None:
    import json
    from dataclasses import replace

    from tests.evaluation.test_decision_value_failure_receipt_robustness import _as_v2
    from trade_rl.artifacts.canonical import canonical_json_bytes
    from trade_rl.artifacts.hashing import content_digest

    root = tmp_path / "attempt"
    _persist(root, replace(_row(), reason_codes=("original" + chr(0x1F600) * 40000,)))
    if legacy_v2:
        _as_v2(root)
    path = root / "incomplete-pairs.json"
    original = path.read_bytes()
    value = json.loads(original)
    pin = value["receipt_digest"]
    value["pairs"][0]["reason_codes"][0] += " altered"
    if tamper == "resealed":
        value["pairs_digest"] = content_digest(value["pairs"])
        value["receipt_digest"] = content_digest(
            {k: v for k, v in value.items() if k != "receipt_digest"}
        )
    forged = canonical_json_bytes(value)
    assert forged != original
    path.write_bytes(forged)

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("bytes inconsistent with the pin reached JSON scanning")

    monkeypatch.setattr(study, "_preflight_receipt_json_depth", forbidden)
    monkeypatch.setattr(study.json, "loads", forbidden)
    with pytest.raises(ValueError, match="pinned"):
        study.load_incomplete_pair_attempt(
            root, expected_receipt_digest=pin, expected_max_ineligible_pairs=1
        )
    assert path.read_bytes() == forged


@pytest.mark.parametrize("case", ["v2", "v3", "quarantine"])
def test_raw_pin_check_preserves_nested_and_quoted_digest_lookalikes(
    tmp_path: Path, case: str
) -> None:
    import json
    from dataclasses import replace

    from tests.evaluation.test_decision_value_failure_receipt_robustness import _as_v2
    from trade_rl.artifacts.canonical import canonical_json_bytes
    from trade_rl.artifacts.hashing import content_digest

    root = tmp_path / "attempt"
    quoted = (
        'quoted ,"receipt_digest":"' + "f" * 64 + '" escaped ' + chr(92) + chr(0x1F600)
    )
    first = replace(_row(), reason_codes=(quoted,))
    if case == "quarantine":
        first = replace(first, baseline={"terminal_cash": "invalid" + quoted})
    _persist(root, first, _row(action=(0, 1)))
    if case == "v2":
        _as_v2(root)
    path = root / "incomplete-pairs.json"
    value = json.loads(path.read_bytes())
    value["pairs"][-1]["reference_vintage"] = {
        "decoy": {"a": 1, "receipt_digest": "e" * 64},
        "quoted": quoted,
    }
    value["pairs_digest"] = content_digest(value["pairs"])
    pin = content_digest({k: v for k, v in value.items() if k != "receipt_digest"})
    value["receipt_digest"] = pin
    raw = canonical_json_bytes(value)
    path.write_bytes(raw)
    assert raw.count(b',"receipt_digest":"') == 2
    assert (
        study.load_incomplete_pair_attempt(root, expected_receipt_digest=pin) == value
    )
    assert (
        study.load_incomplete_pair_attempt(
            root, expected_receipt_digest=pin, expected_max_ineligible_pairs=2
        )
        == value
    )
    assert path.read_bytes() == raw


def test_raw_pin_match_is_not_a_substitute_for_strict_validation(
    tmp_path: Path,
) -> None:
    import hashlib
    import json

    root = tmp_path / "attempt"
    _persist(root, _row())
    path = root / "incomplete-pairs.json"
    raw = path.read_bytes()
    old = json.loads(raw)["receipt_digest"]
    field = b',"receipt_digest":"' + old.encode("ascii") + b'"'
    assert raw.count(field) == 1
    body = raw.replace(field, b"", 1)
    # Noncanonical but JSON-parseable body with a duplicated direct key.
    body = body.replace(
        b'"fit_admitted":false', b'"fit_admitted":false,"fit_admitted":false', 1
    )
    forged_pin = hashlib.sha256(body).hexdigest()
    forged = raw.replace(
        b'"fit_admitted":false', b'"fit_admitted":false,"fit_admitted":false', 1
    )
    new_field = b',"receipt_digest":"' + forged_pin.encode("ascii") + b'"'
    forged = forged.replace(field, new_field, 1)
    assert hashlib.sha256(forged.replace(new_field, b"", 1)).hexdigest() == forged_pin
    path.write_bytes(forged)
    with pytest.raises(ValueError, match="duplicated"):
        study.load_incomplete_pair_attempt(root, expected_receipt_digest=forged_pin)
    assert path.read_bytes() == forged
