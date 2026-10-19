from __future__ import annotations

import copy
import hashlib
import importlib
import json
import os
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Any

import pytest

from tests.tools.market_replay_oracle.test_protocol import (
    SOURCE_HEAD as G_SOURCE_HEAD,
)
from tests.tools.market_replay_oracle.test_protocol import (
    _document as _v3_document,
)
from tests.tools.market_replay_oracle.test_protocol import (
    _wire,
)
from tests.tools.market_replay_oracle.test_protocol_v4 import (
    H_SOURCE_HEAD,
    MAX_PROTOCOL_BYTES,
)
from tests.tools.market_replay_oracle.test_protocol_v4 import (
    _document as _v4_document,
)
from tools.market_replay_oracle.protocol import ProtocolPin

_PAIR_MODULE = "tools.market_replay_oracle.protocol_pair"
_H3_CANONICAL_SHA256 = (
    "742e883b405a0d7f600998b820a190f872673f1a439d56dd3b8001a6cf017393"
)


def _pair_api() -> Any:
    try:
        module = importlib.import_module(_PAIR_MODULE)
    except ModuleNotFoundError as error:
        if error.name != _PAIR_MODULE:
            raise
        pytest.fail(f"{_PAIR_MODULE} is not implemented", pytrace=False)
    assert callable(getattr(module, "load_declared_protocol_pair", None))
    assert getattr(module, "DeclaredProtocolPair", None) is not None
    return module


def _h3_document() -> dict[str, Any]:
    document = copy.deepcopy(_v3_document())
    document["source_head"] = H_SOURCE_HEAD
    return document


def _canonical(document: object) -> bytes:
    return json.dumps(
        document,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _pin(raw: bytes, source_head: str) -> ProtocolPin:
    return ProtocolPin(
        sha256=hashlib.sha256(raw).hexdigest(),
        raw_bytes=len(raw),
        source_head=source_head,
    )


def _write(path: Path, raw: bytes, source_head: str) -> tuple[Path, ProtocolPin]:
    path.write_bytes(raw)
    return path, _pin(raw, source_head)


def _write_pair(
    root: Path,
    *,
    native_document: dict[str, Any] | None = None,
    comparison_document: dict[str, Any] | None = None,
    native_raw: bytes | None = None,
    comparison_raw: bytes | None = None,
    native_path: Path | None = None,
    comparison_path: Path | None = None,
) -> tuple[Path, ProtocolPin, Path, ProtocolPin]:
    if native_raw is None:
        native_raw = _wire(native_document or _h3_document())
    if comparison_raw is None:
        comparison_raw = _wire(comparison_document or _v4_document())
    v3_path = native_path or root / "native-h-v3.json"
    v4_path = comparison_path or root / "comparison-h-v4.json"
    return (
        *_write(v3_path, native_raw, H_SOURCE_HEAD),
        *_write(v4_path, comparison_raw, H_SOURCE_HEAD),
    )


def _load_pair(
    root: Path,
    *,
    native_document: dict[str, Any] | None = None,
    comparison_document: dict[str, Any] | None = None,
) -> tuple[Any, bytes, bytes, ProtocolPin, ProtocolPin]:
    v3_path, v3_pin, v4_path, v4_pin = _write_pair(
        root,
        native_document=native_document,
        comparison_document=comparison_document,
    )
    result = _pair_api().load_declared_protocol_pair(v3_path, v3_pin, v4_path, v4_pin)
    return result, v3_path.read_bytes(), v4_path.read_bytes(), v3_pin, v4_pin


def test_protocol_pair_module_exports_loader_and_result_type() -> None:
    module = _pair_api()

    assert callable(module.load_declared_protocol_pair)
    assert module.DeclaredProtocolPair.__name__ == "DeclaredProtocolPair"


def test_pair_accepts_frozen_h_v3_and_v4_and_returns_only_canonical_documents(
    tmp_path: Path,
) -> None:
    result, v3_raw, v4_raw, v3_pin, v4_pin = _load_pair(tmp_path)
    native = _h3_document()
    comparison = _v4_document()
    projected = copy.deepcopy(comparison)
    projected.pop("exact_fill_evidence")
    projected["schema"] = "issue810_tier_d_nonrl48_result_blind_v3"

    assert result.scope == "STATIC_DECLARED_PROTOCOL_PAIR_ONLY"
    assert result.native_pin == v3_pin
    assert result.comparison_pin == v4_pin
    assert isinstance(result.native_canonical, bytes)
    assert isinstance(result.comparison_canonical, bytes)
    assert result.native_canonical == _canonical(native)
    assert result.comparison_canonical == _canonical(comparison)
    assert result.native_canonical == _canonical(projected)
    assert hashlib.sha256(result.native_canonical).hexdigest() == _H3_CANONICAL_SHA256
    assert v3_raw == result.native_canonical
    assert v4_raw == result.comparison_canonical
    assert not hasattr(result, "schedule")
    assert not hasattr(result, "admission")


def test_pair_allows_whitespace_variants_but_binds_each_raw_file_independently(
    tmp_path: Path,
) -> None:
    v3_raw = json.dumps(_h3_document(), indent=2, ensure_ascii=False).encode()
    v4_raw = json.dumps(_v4_document(), indent=4, ensure_ascii=False).encode()
    v3_path, v3_pin, v4_path, v4_pin = _write_pair(
        tmp_path, native_raw=v3_raw, comparison_raw=v4_raw
    )

    result = _pair_api().load_declared_protocol_pair(v3_path, v3_pin, v4_path, v4_pin)

    assert v3_pin.sha256 == hashlib.sha256(v3_raw).hexdigest()
    assert v4_pin.sha256 == hashlib.sha256(v4_raw).hexdigest()
    assert v3_pin.sha256 != hashlib.sha256(result.native_canonical).hexdigest()
    assert v4_pin.sha256 != hashlib.sha256(result.comparison_canonical).hexdigest()
    assert result.native_canonical == _canonical(_h3_document())
    assert result.comparison_canonical == _canonical(_v4_document())


def test_pair_result_is_frozen(tmp_path: Path) -> None:
    result, *_ = _load_pair(tmp_path)

    with pytest.raises((FrozenInstanceError, AttributeError)):
        result.scope = "OTHER"


@pytest.mark.parametrize("side", ["native", "comparison"])
@pytest.mark.parametrize("wrong_pin", ["sha", "length"])
def test_pair_checks_both_raw_pins(tmp_path: Path, side: str, wrong_pin: str) -> None:
    v3_path, v3_pin, v4_path, v4_pin = _write_pair(tmp_path)
    path, pin = (v3_path, v3_pin) if side == "native" else (v4_path, v4_pin)
    bad_pin = ProtocolPin(
        sha256=("0" * 64 if wrong_pin == "sha" else pin.sha256),
        raw_bytes=(pin.raw_bytes + 1 if wrong_pin == "length" else pin.raw_bytes),
        source_head=H_SOURCE_HEAD,
    )
    args = (
        (path, bad_pin, v4_path, v4_pin)
        if side == "native"
        else (
            v3_path,
            v3_pin,
            path,
            bad_pin,
        )
    )

    with pytest.raises(ValueError, match="SHA-256|byte count"):
        _pair_api().load_declared_protocol_pair(*args)


@pytest.mark.parametrize("side", ["native", "comparison"])
def test_pair_rejects_h_and_g_source_pin_crossing(tmp_path: Path, side: str) -> None:
    v3_path, v3_pin, v4_path, v4_pin = _write_pair(tmp_path)
    if side == "native":
        bad_pin = ProtocolPin(v3_pin.sha256, v3_pin.raw_bytes, G_SOURCE_HEAD)
        args = (v3_path, bad_pin, v4_path, v4_pin)
    else:
        bad_pin = ProtocolPin(v4_pin.sha256, v4_pin.raw_bytes, G_SOURCE_HEAD)
        args = (v3_path, v3_pin, v4_path, bad_pin)

    with pytest.raises(ValueError):
        _pair_api().load_declared_protocol_pair(*args)


def test_pair_rejects_g_v3_document_as_the_h_native_declaration(
    tmp_path: Path,
) -> None:
    v3_path, v3_pin, v4_path, v4_pin = _write_pair(
        tmp_path, native_document=_v3_document()
    )
    # Re-pin the G bytes against the H head to verify the document itself is rejected.
    raw = v3_path.read_bytes()
    forged_h_pin = _pin(raw, H_SOURCE_HEAD)

    with pytest.raises(ValueError):
        _pair_api().load_declared_protocol_pair(v3_path, forged_h_pin, v4_path, v4_pin)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("concurrency", {"max_workers": 3}),
        ("economics", {"dataset_fee": 0.0004}),
        ("actor", {"start": 2904}),
        ("policy", {"nonrl_action": "different residual behavior"}),
        ("budget", {"total_native_advances": 316751}),
    ],
)
def test_pair_rejects_coherently_changed_shared_contract(
    tmp_path: Path, field: str, value: dict[str, Any]
) -> None:
    native = _h3_document()
    comparison = _v4_document()
    native[field].update(value)
    comparison[field].update(value)

    with pytest.raises(ValueError):
        _load_pair(
            tmp_path,
            native_document=native,
            comparison_document=comparison,
        )


def test_pair_rejects_unknown_shared_and_comparison_only_fields(
    tmp_path: Path,
) -> None:
    native = _h3_document()
    comparison = _v4_document()
    native["unreviewed_extension"] = True
    comparison["unreviewed_extension"] = True
    comparison["unreviewed_comparison_field"] = True

    with pytest.raises(ValueError):
        _load_pair(
            tmp_path,
            native_document=native,
            comparison_document=comparison,
        )


@pytest.mark.parametrize("side", ["native", "comparison"])
@pytest.mark.parametrize(
    "raw",
    [
        b'{"schema":"first","schema":"second"}',
        b'{"schema":NaN}',
        b'{"schema":1e999}',
    ],
)
def test_pair_strict_json_rejects_duplicate_and_nonfinite_values(
    tmp_path: Path, side: str, raw: bytes
) -> None:
    v3_raw = raw if side == "native" else _wire(_h3_document())
    v4_raw = raw if side == "comparison" else _wire(_v4_document())
    v3_path, v3_pin, v4_path, v4_pin = _write_pair(
        tmp_path, native_raw=v3_raw, comparison_raw=v4_raw
    )

    with pytest.raises(ValueError):
        _pair_api().load_declared_protocol_pair(v3_path, v3_pin, v4_path, v4_pin)


def test_pair_bounds_h3_protocol_size_before_parsing(tmp_path: Path) -> None:
    oversized = b" " * (MAX_PROTOCOL_BYTES + 1)
    v3_path = tmp_path / "oversized-native-h3.json"
    v3_path.write_bytes(oversized)
    # ProtocolPin itself is bounded; the loader must reject file size before pin length.
    v3_pin = ProtocolPin(
        sha256=hashlib.sha256(oversized).hexdigest(),
        raw_bytes=MAX_PROTOCOL_BYTES,
        source_head=H_SOURCE_HEAD,
    )
    v4_path, v4_pin = _write(
        tmp_path / "comparison-h-v4.json", _wire(_v4_document()), H_SOURCE_HEAD
    )

    with pytest.raises(ValueError, match="1 MiB|byte count"):
        _pair_api().load_declared_protocol_pair(v3_path, v3_pin, v4_path, v4_pin)


def test_pair_rejects_nonregular_protocol_path(tmp_path: Path) -> None:
    v3_path, v3_pin, v4_path, v4_pin = _write_pair(tmp_path)

    with pytest.raises(ValueError):
        _pair_api().load_declared_protocol_pair(tmp_path, v3_pin, v4_path, v4_pin)


def test_pair_rejects_same_path_for_both_protocol_versions(tmp_path: Path) -> None:
    v3_path, v3_pin, _, _ = _write_pair(tmp_path)

    with pytest.raises(ValueError, match="distinct|different|alias|same"):
        _pair_api().load_declared_protocol_pair(v3_path, v3_pin, v3_path, v3_pin)


def test_pair_rejects_hardlinked_protocol_aliases(tmp_path: Path) -> None:
    v3_path, v3_pin, _, _ = _write_pair(tmp_path)
    hardlink = tmp_path / "comparison-hardlink.json"
    try:
        os.link(v3_path, hardlink)
    except OSError as error:
        pytest.skip(f"hardlink creation is unavailable: {error}")
    hardlink_pin = _pin(hardlink.read_bytes(), H_SOURCE_HEAD)

    with pytest.raises(ValueError, match="distinct|different|alias|same"):
        _pair_api().load_declared_protocol_pair(v3_path, v3_pin, hardlink, hardlink_pin)


def test_pair_rejects_symlinked_protocol_alias(tmp_path: Path) -> None:
    v3_path, v3_pin, _, _ = _write_pair(tmp_path)
    symlink = tmp_path / "comparison-symlink.json"
    try:
        symlink.symlink_to(v3_path)
    except (OSError, NotImplementedError) as error:
        pytest.skip(f"symlink creation is unavailable: {error}")
    symlink_pin = _pin(symlink.read_bytes(), H_SOURCE_HEAD)

    with pytest.raises(ValueError):
        _pair_api().load_declared_protocol_pair(v3_path, v3_pin, symlink, symlink_pin)


@pytest.mark.parametrize("changed_side", ["native", "comparison"])
def test_pair_rechecks_both_raw_files_after_v4_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, changed_side: str
) -> None:
    from tools.market_replay_oracle import protocol_v4

    v3_path, v3_pin, v4_path, v4_pin = _write_pair(tmp_path)
    changed_path = v3_path if changed_side == "native" else v4_path
    original = protocol_v4.load_protocol_v4

    def load_then_mutate(path: Path, pin: ProtocolPin) -> dict[str, Any]:
        document = original(path, pin)
        if path == v4_path:
            changed_path.write_bytes(changed_path.read_bytes() + b"\n")
        return document

    monkeypatch.setattr(protocol_v4, "load_protocol_v4", load_then_mutate)

    with pytest.raises(ValueError, match="changed|mutat|SHA-256|byte count|pin"):
        _pair_api().load_declared_protocol_pair(v3_path, v3_pin, v4_path, v4_pin)
