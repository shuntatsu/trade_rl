"""File bindings and ZIP limits must precede untrusted parsing or allocation."""

import hashlib
import json
import struct
from pathlib import Path

import numpy as np
import pytest

import tools.market_replay_oracle.artifact as artifact_module
from tests.tools.market_replay_oracle.test_artifact import (
    _canonical,
    _pin,
    _write_fixture,
)
from tools.market_replay_oracle import load_pinned_v4


@pytest.mark.parametrize("filename", ["manifest.json", "arrays.npz"])
@pytest.mark.parametrize("resize", [False, True])
def test_raw_pin_mismatch_stops_before_manifest_parsing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, filename: str, resize: bool
) -> None:
    original = _write_fixture(tmp_path)
    pin = _pin(tmp_path, original.dataset_id)
    path = tmp_path / filename
    raw = path.read_bytes()
    path.write_bytes(raw + b"x" if resize else bytes([raw[0] ^ 1]) + raw[1:])

    def forbidden_parser(*args: object) -> None:
        raise AssertionError("parsing reached before both raw pins passed")

    monkeypatch.setattr(artifact_module, "json_object", forbidden_parser)
    with pytest.raises(ValueError, match="raw source"):
        load_pinned_v4(tmp_path, pin)


def test_oversized_central_directory_stops_before_zip_entry_allocation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = _write_fixture(tmp_path)
    path = tmp_path / "arrays.npz"
    raw = bytearray(path.read_bytes())
    assert raw[-22:-18] == b"PK\x05\x06"
    struct.pack_into("<L", raw, len(raw) - 10, 65537)
    path.write_bytes(raw)
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    manifest["arrays_digest"] = hashlib.sha256(raw).hexdigest()
    manifest["artifact_digest"] = hashlib.sha256(
        _canonical({k: v for k, v in manifest.items() if k != "artifact_digest"})
    ).hexdigest()
    manifest_path.write_bytes(_canonical(manifest))
    pin = _pin(tmp_path, original.dataset_id)

    def forbidden_reader(*args: object) -> None:
        raise AssertionError("ZipFile allocated before central-directory bound")

    monkeypatch.setattr(artifact_module.zipfile, "ZipFile", forbidden_reader)
    with pytest.raises(ValueError, match="central directory"):
        load_pinned_v4(tmp_path, pin)


def test_returned_arrays_do_not_alias_files_modified_after_verification(
    tmp_path: Path,
) -> None:
    original = _write_fixture(tmp_path)
    loaded = load_pinned_v4(tmp_path, _pin(tmp_path, original.dataset_id))
    before = loaded.arrays["close"].copy()
    (tmp_path / "arrays.npz").write_bytes(b"replacement outside the verified view")
    np.testing.assert_array_equal(loaded.arrays["close"], before)
