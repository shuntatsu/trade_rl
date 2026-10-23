"""High-value stream-boundary checks for the generic receipt reader.

Fixtures here test parser mechanics only; they are not native receipts.
"""

from __future__ import annotations

import base64
import gzip
import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any

import pytest

from tools.market_b4_oracle.receipt_stream import (
    PinnedReceiptRows,
    ReceiptPin,
)


def _raw_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def _pinned_reader(
    tmp_path: Path,
    compressed: bytes,
    *,
    phase: str = "actor",
) -> PinnedReceiptRows:
    path = tmp_path / "receipt.gz"
    path.write_bytes(compressed)
    pin = ReceiptPin(
        bytes=len(compressed), sha256=hashlib.sha256(compressed).hexdigest()
    )
    return PinnedReceiptRows(path=path, pin=pin, phase=phase)


def _large_fixture() -> tuple[bytes, list[dict[str, object]]]:
    rng = random.Random(17)
    actors = [
        {"index": i, "payload": base64.b64encode(rng.randbytes(1024)).decode("ascii")}
        for i in range(200)
    ]
    document = {
        "closing_rows": [{"reserve": 1}],
        "header": {"fixture": "deterministic-large-stream"},
        "native_receipt": {"rows": actors},
        "other": {"preserved": True},
    }
    compressed = gzip.compress(_raw_json(document), mtime=0)
    assert len(compressed) > 65_536
    return compressed, actors


def test_utf8_codepoint_can_cross_inflater_chunk_boundary_and_metadata_survives(
    tmp_path: Path,
) -> None:
    emoji = "😀".encode("utf-8")
    document = {
        "closing_rows": [],
        "header": {
            "escaped": 'quote " and slash \\ plus café',
            "empty": [],
        },
        "native_receipt": {"rows": []},
        "other": "😀",
    }
    first = _raw_json(document)
    prefix_length = first.index(emoji)
    document["other"] = "x" * (65_535 - prefix_length) + "😀"
    raw = _raw_json(document)
    assert raw.index(emoji) == 65_535

    compressed = gzip.compress(raw, mtime=0)
    reader = _pinned_reader(tmp_path, compressed)
    assert list(reader) == []
    assert reader.proof.decompressed_bytes == len(raw)
    metadata = reader.metadata
    assert metadata["header"] == {
        "escaped": 'quote " and slash \\ plus café',
        "empty": [],
    }
    assert metadata["other"] == document["other"]
    assert metadata["closing_rows"] == []
    assert metadata["native_receipt"] == {"rows": []}


def test_large_deterministic_gzip_over_64k_scans_all_rows_and_verifies_pin(
    tmp_path: Path,
) -> None:
    compressed, actors = _large_fixture()
    reader = _pinned_reader(tmp_path, compressed)

    rows = list(reader)
    assert len(rows) == 200
    assert rows[0] == actors[0]
    assert rows[-1] == actors[-1]
    assert reader.proof.actor_rows == 200
    assert reader.proof.reserve_rows == 1
    assert reader.proof.selected_rows == 200
    assert reader.proof.compressed_bytes == len(compressed)
    assert reader.proof.compressed_sha256 == hashlib.sha256(compressed).hexdigest()


def test_late_crc_failure_after_provisional_large_rows_never_exposes_proof_or_metadata(
    tmp_path: Path,
) -> None:
    compressed, _ = _large_fixture()
    corrupt = compressed[:-8] + bytes([compressed[-8] ^ 1]) + compressed[-7:]
    reader = _pinned_reader(tmp_path, corrupt)
    iterator = iter(reader)

    provisional = []
    with pytest.raises(ValueError):
        while True:
            provisional.append(next(iterator))
    assert provisional  # yielded rows remain provisional until the trailer passes
    with pytest.raises(ValueError):
        _ = reader.proof
    with pytest.raises(ValueError):
        _ = reader.metadata


def test_duplicate_key_after_json_escape_is_rejected_in_unselected_metadata(
    tmp_path: Path,
) -> None:
    escaped_key = b"\\u006b"
    raw = (
        b'{"closing_rows":[],"header":{"k":1,"'
        + escaped_key
        + b'":2},"native_receipt":{"rows":[{"actor":1}]},"other":{}}'
    )
    reader = _pinned_reader(tmp_path, gzip.compress(raw, mtime=0), phase="actor")

    with pytest.raises(ValueError):
        list(reader)
    with pytest.raises(ValueError):
        _ = reader.proof


@pytest.mark.parametrize(
    ("phase", "actor_rows", "reserve_rows"),
    [
        ("actor", [{"actor": 1}], [None]),
        ("reserve", [7], [{"reserve": 1}]),
    ],
)
def test_unselected_row_arrays_still_reject_non_object_rows(
    tmp_path: Path,
    phase: str,
    actor_rows: list[object],
    reserve_rows: list[object],
) -> None:
    document = {
        "closing_rows": reserve_rows,
        "header": {},
        "native_receipt": {"rows": actor_rows},
        "other": {},
    }
    reader = _pinned_reader(
        tmp_path,
        gzip.compress(_raw_json(document), mtime=0),
        phase=phase,
    )

    with pytest.raises(ValueError):
        list(reader)
    with pytest.raises(ValueError):
        _ = reader.proof


@pytest.mark.parametrize("number", [b"01", b"1e", b"+1", b"1."])
def test_invalid_json_number_forms_are_rejected(tmp_path: Path, number: bytes) -> None:
    raw = (
        b'{"closing_rows":[],"header":{},"native_receipt":{"rows":[]},"other":'
        + number
        + b"}"
    )
    reader = _pinned_reader(tmp_path, gzip.compress(raw, mtime=0))

    with pytest.raises(ValueError):
        list(reader)


def test_valid_negative_zero_integer_and_exponent_metadata_are_preserved(
    tmp_path: Path,
) -> None:
    raw = (
        b'{"closing_rows":[],"header":{"negative_zero":-0.0,"integer":17,'
        b'"exponent":1.25e+2},"native_receipt":{"rows":[]},"other":{}}'
    )
    reader = _pinned_reader(tmp_path, gzip.compress(raw, mtime=0))
    assert list(reader) == []

    header = reader.metadata["header"]
    assert header["integer"] == 17 and type(header["integer"]) is int
    assert header["exponent"] == 125.0
    assert math.copysign(1.0, header["negative_zero"]) == -1.0


def test_unpaired_escaped_unicode_surrogate_is_rejected(tmp_path: Path) -> None:
    raw = (
        b'{"closing_rows":[],"header":{},"native_receipt":{"rows":[]},"other":"'
        + b"\\ud800"
        + b'"}'
    )
    reader = _pinned_reader(tmp_path, gzip.compress(raw, mtime=0))

    with pytest.raises(ValueError):
        list(reader)
    with pytest.raises(ValueError):
        _ = reader.proof
