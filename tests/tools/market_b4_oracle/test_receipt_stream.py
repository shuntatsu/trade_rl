"""Original-receipt parser mechanics, separate from native/B4 integration.

These parser-mechanics fixtures are synthetic, not native-compatible receipts.
"""

from __future__ import annotations

import dataclasses
import gzip
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from tools.market_b4_oracle.receipt_stream import (
    PinnedReceiptRows,
    ReceiptLimits,
    ReceiptPin,
)


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _gzip(raw: bytes) -> bytes:
    return gzip.compress(raw, mtime=0)


def _pin(data: bytes) -> ReceiptPin:
    return ReceiptPin(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())


def _document(
    *,
    actor_rows: list[dict[str, Any]] | None = None,
    reserve_rows: list[dict[str, Any]] | None = None,
    extra: Any = None,
) -> dict[str, Any]:
    return {
        "header": {"kind": "synthetic-parser-fixture", "revision": 1},
        "closing_rows": list(
            reserve_rows if reserve_rows is not None else [{"reserve": 1}]
        ),
        "native_receipt": {
            "rows": list(actor_rows if actor_rows is not None else [{"actor": 1}])
        },
        "other": {"preserved": True} if extra is None else extra,
    }


def _reader_from_compressed(
    tmp_path: Path,
    compressed: bytes,
    *,
    phase: str = "actor",
    limits: ReceiptLimits | None = None,
    pin: ReceiptPin | None = None,
) -> PinnedReceiptRows:
    path = tmp_path / "synthetic-receipt.json.gz"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(compressed)
    return PinnedReceiptRows(
        path=path,
        pin=_pin(compressed) if pin is None else pin,
        phase=phase,
        limits=ReceiptLimits() if limits is None else limits,
    )


def _reader_from_raw(
    tmp_path: Path,
    raw: bytes,
    *,
    phase: str = "actor",
    limits: ReceiptLimits | None = None,
    pin: ReceiptPin | None = None,
) -> PinnedReceiptRows:
    return _reader_from_compressed(
        tmp_path, _gzip(raw), phase=phase, limits=limits, pin=pin
    )


def _expected_metadata(document: dict[str, Any]) -> dict[str, Any]:
    result = json.loads(json.dumps(document))
    result["closing_rows"] = []
    result["native_receipt"]["rows"] = []
    return result


@pytest.mark.parametrize(
    ("phase", "expected"),
    [("actor", [{"actor": 1}]), ("reserve", [{"reserve": 1}])],
)
def test_selects_only_requested_rows_and_finalizes_full_document_proof(
    tmp_path: Path, phase: str, expected: list[dict[str, int]]
) -> None:
    document = _document()
    compressed = _gzip(_json_bytes(document))
    reader = _reader_from_compressed(tmp_path, compressed, phase=phase)

    rows = list(reader)
    assert rows == expected
    proof = reader.proof
    assert proof.actor_rows == 1
    assert proof.reserve_rows == 1
    assert proof.selected_rows == 1
    assert proof.compressed_bytes == len(compressed)
    assert proof.decompressed_bytes == len(_json_bytes(document))
    assert proof.compressed_sha256 == hashlib.sha256(compressed).hexdigest()
    assert reader.metadata == _expected_metadata(document)


def test_full_6599_manufactured_count_is_phase_selection_only(tmp_path: Path) -> None:
    document = _document(
        actor_rows=[{"i": i} for i in range(6_527)],
        reserve_rows=[{"i": i} for i in range(72)],
    )
    compressed = _gzip(_json_bytes(document))

    for phase, selected in (("actor", 6_527), ("reserve", 72)):
        reader = _reader_from_compressed(tmp_path / phase, compressed, phase=phase)
        assert sum(1 for _ in reader) == selected
        proof = reader.proof
        assert (proof.actor_rows, proof.reserve_rows, proof.selected_rows) == (
            6_527,
            72,
            selected,
        )
    # This synthetic count fixture does not assert a native roster or B4 result.


def test_metadata_and_yielded_rows_are_isolated_copies(tmp_path: Path) -> None:
    document = _document(
        actor_rows=[{"nested": {"value": "actor"}}],
        reserve_rows=[{"nested": {"value": "reserve"}}],
    )
    reader = _reader_from_raw(tmp_path, _json_bytes(document))
    rows = list(reader)
    expected = _expected_metadata(document)
    assert reader.metadata == expected

    rows[0]["nested"]["value"] = "mutated row"
    first_metadata = reader.metadata
    first_metadata["header"]["kind"] = "mutated metadata"
    assert reader.metadata == expected
    assert reader.metadata is not first_metadata


def test_rejects_unknown_phase(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        reader = _reader_from_raw(
            tmp_path,
            _json_bytes(_document()),
            phase="unknown",
        )
        list(reader)


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("bytes", -1),
        ("bytes", True),
        ("bytes", 1),  # valid syntax, mismatched exact byte count
        ("sha256", "not-a-sha256"),
        ("sha256", "0" * 64),  # valid shape, wrong content digest
    ],
)
def test_rejects_invalid_or_mismatched_content_pin(
    tmp_path: Path, field: str, bad_value: object
) -> None:
    compressed = _gzip(_json_bytes(_document()))
    fields: dict[str, object] = {
        "bytes": len(compressed),
        "sha256": hashlib.sha256(compressed).hexdigest(),
    }
    fields[field] = bad_value
    with pytest.raises(ValueError):
        reader = _reader_from_compressed(
            tmp_path,
            compressed,
            pin=ReceiptPin(**fields),  # type: ignore[arg-type]
        )
        list(reader)


@pytest.mark.parametrize(
    "mutation",
    [
        "not-gzip",
        "crc",
        "truncated",
        "trailing-compressed-bytes",
        "concatenated-member",
        "trailing-json",
        "malformed-utf8",
    ],
)
def test_rejects_gzip_and_json_closure_failures(tmp_path: Path, mutation: str) -> None:
    valid_raw = _json_bytes(_document())
    valid_gzip = _gzip(valid_raw)
    if mutation == "not-gzip":
        compressed = b"not a gzip stream"
    elif mutation == "crc":
        compressed = valid_gzip[:-8] + bytes([valid_gzip[-8] ^ 1]) + valid_gzip[-7:]
    elif mutation == "truncated":
        compressed = valid_gzip[:-3]
    elif mutation == "trailing-compressed-bytes":
        compressed = valid_gzip + b"x"
    elif mutation == "concatenated-member":
        compressed = valid_gzip + _gzip(b"  ")
    elif mutation == "trailing-json":
        compressed = _gzip(valid_raw + b" {}")
    else:
        malformed = b'{"closing_rows":[],"native_receipt":{"rows":[]},"other":"\xff"}'
        compressed = _gzip(malformed)

    reader: PinnedReceiptRows | None = None
    with pytest.raises(ValueError):
        reader = _reader_from_compressed(tmp_path, compressed)
        list(reader)
    if reader is not None:
        with pytest.raises(ValueError):
            _ = reader.proof
        with pytest.raises(ValueError):
            _ = reader.metadata


def test_late_gzip_crc_failure_never_finalizes_provisional_rows(tmp_path: Path) -> None:
    raw = _json_bytes(
        _document(actor_rows=[{"i": i} for i in range(200)], reserve_rows=[])
    )
    valid = _gzip(raw)
    corrupt = valid[:-8] + bytes([valid[-8] ^ 1]) + valid[-7:]
    reader = _reader_from_compressed(tmp_path, corrupt)

    with pytest.raises(ValueError):
        list(reader)
    with pytest.raises(ValueError):
        _ = reader.proof


@pytest.mark.parametrize(
    "raw",
    [
        b'{"closing_rows":[],"closing_rows":[],"native_receipt":{"rows":[]}}',
        b'{"closing_rows":[],"native_receipt":{"rows":[]},"other":{"k":1,"k":2}}',
        b'{"closing_rows":[],"native_receipt":{"rows":[{"k":1,"k":2}]}}',
        b'{"closing_rows":[{"k":1,"k":2}],"native_receipt":{"rows":[]}}',
        b'{"closing_rows":[],"native_receipt":{"rows":[]},"other":NaN}',
        b'{"closing_rows":[],"native_receipt":{"rows":[Infinity]}}',
        b'{"closing_rows":[],"native_receipt":{"rows":[]},"other":-Infinity}',
        b'{"closing_rows":[],"native_receipt":{"rows":[]},"other":1e309}',
    ],
)
def test_strict_json_checks_duplicates_and_nonfinite_values_everywhere(
    tmp_path: Path, raw: bytes
) -> None:
    # Selection is actor; malformed values in closing_rows/other must still fail.
    reader: PinnedReceiptRows | None = None
    with pytest.raises(ValueError):
        reader = _reader_from_raw(tmp_path, raw, phase="actor")
        list(reader)
    if reader is not None:
        with pytest.raises(ValueError):
            _ = reader.proof


@pytest.mark.parametrize(
    ("bad_section", "bad_value", "phase"),
    [
        ("actor-missing", None, "reserve"),
        ("actor-null", None, "reserve"),
        ("actor-nonarray", {"row": 1}, "actor"),
        ("reserve-missing", None, "actor"),
        ("reserve-null", None, "actor"),
        ("reserve-nonarray", {"row": 1}, "reserve"),
    ],
)
def test_both_row_targets_must_be_arrays_even_when_unselected(
    tmp_path: Path, bad_section: str, bad_value: object, phase: str
) -> None:
    document = _document()
    if bad_section.startswith("actor-"):
        if bad_section == "actor-missing":
            document["native_receipt"].pop("rows")
        else:
            document["native_receipt"]["rows"] = bad_value
    elif bad_section == "reserve-missing":
        document.pop("closing_rows")
    else:
        document["closing_rows"] = bad_value
    reader: PinnedReceiptRows | None = None
    with pytest.raises(ValueError):
        reader = _reader_from_raw(tmp_path, _json_bytes(document), phase=phase)
        list(reader)
    if reader is not None:
        with pytest.raises(ValueError):
            _ = reader.proof


@pytest.mark.parametrize(
    "budget",
    [
        "compressed_bytes",
        "decompressed_bytes",
        "row_bytes",
        "scalar_bytes",
        "metadata_bytes",
        "depth",
        "object_members",
        "array_items",
        "total_values",
    ],
)
def test_all_limits_apply_to_whole_document_with_small_test_caps(
    tmp_path: Path, budget: str
) -> None:
    document = _document()
    overrides: dict[str, int] = {}
    if budget == "row_bytes":
        document["native_receipt"]["rows"] = [{"payload": "r" * 80}]
        overrides[budget] = 32
    elif budget == "scalar_bytes":
        document["other"] = {"payload": "s" * 80}
        overrides[budget] = 64
    elif budget == "metadata_bytes":
        document["header"] = {"payload": "m" * 80}
        overrides[budget] = len(_json_bytes(_expected_metadata(_document()))) + 1
    elif budget == "depth":
        document["native_receipt"]["rows"] = []
        document["closing_rows"] = []
        document["other"] = {"a": {"b": {"c": {"d": {"e": 1}}}}}
        overrides[budget] = 4
    elif budget == "object_members":
        document["other"] = {str(i): i for i in range(5)}
        overrides[budget] = 4
    elif budget == "array_items":
        document["other"] = [1, 2, 3]
        overrides[budget] = 2
    elif budget == "total_values":
        document["other"] = list(range(40))
        overrides[budget] = 32

    raw = _json_bytes(document)
    compressed = _gzip(raw)
    if budget == "compressed_bytes":
        overrides[budget] = len(compressed) - 1
    elif budget == "decompressed_bytes":
        overrides[budget] = len(raw) - 1
    limits = dataclasses.replace(ReceiptLimits(), **overrides)

    reader: PinnedReceiptRows | None = None
    with pytest.raises(ValueError):
        reader = _reader_from_compressed(
            tmp_path,
            compressed,
            phase="reserve" if budget == "row_bytes" else "actor",
            limits=limits,
        )
        list(reader)
    if reader is not None:
        with pytest.raises(ValueError):
            _ = reader.proof


def test_proof_is_gated_on_complete_exhaustion_and_reader_is_one_use(
    tmp_path: Path,
) -> None:
    reader = _reader_from_raw(
        tmp_path,
        _json_bytes(_document(actor_rows=[{"i": 1}, {"i": 2}], reserve_rows=[])),
    )
    iterator = iter(reader)
    assert next(iterator) == {"i": 1}
    with pytest.raises(ValueError):
        _ = reader.proof

    assert list(iterator) == [{"i": 2}]
    assert reader.proof.actor_rows == 2
    with pytest.raises(ValueError):
        iter(reader)
