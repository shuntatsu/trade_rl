from __future__ import annotations

import copy
import hashlib
import importlib
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from tests.tools.market_replay_oracle.test_protocol import (
    SOURCE_HEAD as G_SOURCE_HEAD,
)
from tests.tools.market_replay_oracle.test_protocol import (
    _content as _v3_content,
)
from tests.tools.market_replay_oracle.test_protocol import (
    _document as _v3_document,
)
from tests.tools.market_replay_oracle.test_protocol import (
    _wire,
)
from tools.market_replay_oracle.protocol import (
    MAX_PROTOCOL_BYTES,
    ProtocolPin,
    bind_schedule,
    load_protocol,
)

H_SOURCE_HEAD = "d361b6e3c3cdc83a8f5f05da9f82e1f72b66e3e5"
V4_SCHEMA = "issue810_tier_d_nonrl48_result_blind_v4"
_V4_MODULE = "tools.market_replay_oracle.protocol_v4"
_EXACT_FILL_EVIDENCE = {
    "positive_effective_lot": "order_event_v2_signed_integer_count_and_step",
    "zero_effective_lot": "order_event_v1_exact_reported_float",
    "nonfills": "order_event_v1_zero_fill",
    "source_transition": "fresh_attempt_required_no_v3_relabeling",
}


def _document() -> dict[str, Any]:
    """Build the v4 document by extending the unchanged frozen v3 fixture."""
    document = copy.deepcopy(_v3_document())
    document["schema"] = V4_SCHEMA
    document["source_head"] = H_SOURCE_HEAD
    document["exact_fill_evidence"] = copy.deepcopy(_EXACT_FILL_EVIDENCE)
    return document


def _v4_api() -> Any:
    try:
        return importlib.import_module(_V4_MODULE)
    except ModuleNotFoundError as error:
        if error.name != _V4_MODULE:
            raise
        pytest.fail(f"{_V4_MODULE} is not implemented", pytrace=False)


def _pin(raw: bytes, source_head: str = H_SOURCE_HEAD) -> ProtocolPin:
    return ProtocolPin(
        sha256=hashlib.sha256(raw).hexdigest(),
        raw_bytes=len(raw),
        source_head=source_head,
    )


def _write_document(
    path: Path, document: dict[str, Any], source_head: str
) -> tuple[Path, ProtocolPin]:
    raw = _wire(document)
    path.write_bytes(raw)
    return path, _pin(raw, source_head)


def _load_v4(
    tmp_path: Path,
    document: dict[str, Any],
    *,
    pin_source_head: str = H_SOURCE_HEAD,
) -> dict[str, Any]:
    path, pin = _write_document(
        tmp_path / "protocol-v4.json", document, pin_source_head
    )
    return _v4_api().load_protocol_v4(path, pin)


def test_protocol_v4_module_is_available_and_exports_entrypoints() -> None:
    try:
        module = importlib.import_module(_V4_MODULE)
    except ModuleNotFoundError as error:
        assert error.name == _V4_MODULE, f"unexpected missing dependency: {error.name}"
        module = None

    assert module is not None, f"missing required feature module: {_V4_MODULE}"
    assert callable(getattr(module, "load_protocol_v4", None))
    assert callable(getattr(module, "bind_schedule_v4", None))


def test_v4_fixture_only_adds_the_declared_schema_source_and_fill_evidence() -> None:
    v3 = _v3_document()
    v4 = _document()
    v3_allowed = {"schema", "source_head"}
    v4_allowed = v3_allowed | {"exact_fill_evidence"}

    assert {key: value for key, value in v4.items() if key not in v4_allowed} == {
        key: value for key, value in v3.items() if key not in v3_allowed
    }
    assert v4["schema"] == V4_SCHEMA
    assert v4["source_head"] == H_SOURCE_HEAD
    assert v4["exact_fill_evidence"] == _EXACT_FILL_EVIDENCE


def test_load_protocol_v4_accepts_pinned_v4_without_changing_frozen_finance(
    tmp_path: Path,
) -> None:
    protocol = _load_v4(tmp_path, _document())

    assert protocol == _document()
    assert protocol["scope"] == "CONSUMED_2025_DEVELOPMENT_NONRL48_ONLY"
    assert protocol["budget"]["native_arms"] == 48
    assert protocol["budget"]["advances_per_arm"] == 6599
    assert protocol["source_head"] == H_SOURCE_HEAD
    assert protocol["exact_fill_evidence"] == _EXACT_FILL_EVIDENCE


@pytest.mark.parametrize(
    ("mutation", "pin_source_head"),
    [
        pytest.param(
            lambda doc: doc["economics"].__setitem__("dataset_fee", 0.0004),
            H_SOURCE_HEAD,
            id="financial-contract",
        ),
        pytest.param(
            lambda doc: doc["budget"].__setitem__("total_native_advances", 316751),
            H_SOURCE_HEAD,
            id="48-arm-budget",
        ),
        pytest.param(
            lambda doc: doc["policy"].__setitem__(
                "nonrl_action", "direct HOLD (code0), scale1"
            ),
            H_SOURCE_HEAD,
            id="action-policy",
        ),
        pytest.param(
            lambda doc: doc.__setitem__("source_head", G_SOURCE_HEAD),
            G_SOURCE_HEAD,
            id="coherently-repinned-g-source",
        ),
        pytest.param(
            lambda doc: doc.__setitem__(
                "schema", "issue810_tier_d_nonrl48_result_blind_v3"
            ),
            H_SOURCE_HEAD,
            id="v3-schema",
        ),
        pytest.param(
            lambda doc: doc.pop("exact_fill_evidence"),
            H_SOURCE_HEAD,
            id="missing-v4-fill-contract",
        ),
        pytest.param(
            lambda doc: doc["exact_fill_evidence"].__setitem__(
                "positive_effective_lot", "order_event_v1_exact_reported_float"
            ),
            H_SOURCE_HEAD,
            id="changed-effective-lot-contract",
        ),
    ],
)
def test_v4_closed_contract_rejects_coherently_rehashed_mutations(
    tmp_path: Path,
    mutation: Callable[[dict[str, Any]], object],
    pin_source_head: str,
) -> None:
    document = _document()
    mutation(document)

    with pytest.raises(ValueError):
        _load_v4(tmp_path, document, pin_source_head=pin_source_head)


def test_v3_and_v4_loaders_reject_each_others_protocol_documents(
    tmp_path: Path,
) -> None:
    v3_path, v3_pin = _write_document(
        tmp_path / "protocol-v3.json", _v3_document(), G_SOURCE_HEAD
    )
    v4_path, v4_pin = _write_document(
        tmp_path / "protocol-v4.json", _document(), H_SOURCE_HEAD
    )

    with pytest.raises(ValueError):
        _v4_api().load_protocol_v4(v3_path, v3_pin)
    with pytest.raises(ValueError):
        load_protocol(v4_path, v4_pin)


def test_source_head_pins_do_not_cross_accept_h_and_g(
    tmp_path: Path,
) -> None:
    v4_path, v4_pin = _write_document(
        tmp_path / "protocol-v4.json", _document(), G_SOURCE_HEAD
    )
    v3_path, v3_pin = _write_document(
        tmp_path / "protocol-v3.json", _v3_document(), H_SOURCE_HEAD
    )

    with pytest.raises(ValueError):
        _v4_api().load_protocol_v4(v4_path, v4_pin)
    with pytest.raises(ValueError):
        load_protocol(v3_path, v3_pin)


def test_v4_raw_pin_rejects_hash_and_length_mismatch_before_parse(
    tmp_path: Path,
) -> None:
    raw = b'{"invalid":NaN'
    path = tmp_path / "invalid.json"
    path.write_bytes(raw)
    bad_hash = ProtocolPin(
        sha256="0" * 64,
        raw_bytes=len(raw),
        source_head=H_SOURCE_HEAD,
    )

    with pytest.raises(ValueError, match="SHA-256"):
        _v4_api().load_protocol_v4(path, bad_hash)

    valid_raw = _wire(_document())
    valid_path = tmp_path / "valid.json"
    valid_path.write_bytes(valid_raw)
    bad_length = ProtocolPin(
        sha256=hashlib.sha256(valid_raw).hexdigest(),
        raw_bytes=len(valid_raw) + 1,
        source_head=H_SOURCE_HEAD,
    )
    with pytest.raises(ValueError, match="byte count"):
        _v4_api().load_protocol_v4(valid_path, bad_length)


@pytest.mark.parametrize(
    "raw",
    [
        b'{"schema":"first","schema":"second"}',
        b'{"schema":NaN}',
        b'{"schema":1e999}',
    ],
)
def test_v4_strict_json_rejects_duplicate_and_nonfinite_values(
    tmp_path: Path, raw: bytes
) -> None:
    path = tmp_path / "strict.json"
    path.write_bytes(raw)

    with pytest.raises(ValueError):
        _v4_api().load_protocol_v4(path, _pin(raw))


def test_v4_protocol_limit_rejects_oversized_input_before_parse(
    tmp_path: Path,
) -> None:
    raw = b" " * (MAX_PROTOCOL_BYTES + 1)
    path = tmp_path / "oversized.json"
    path.write_bytes(raw)
    pin = ProtocolPin(
        sha256=hashlib.sha256(raw).hexdigest(),
        raw_bytes=MAX_PROTOCOL_BYTES,
        source_head=H_SOURCE_HEAD,
    )

    with pytest.raises(ValueError, match="1 MiB"):
        _v4_api().load_protocol_v4(path, pin)


def test_v4_schedule_equals_v3_schedule_from_the_same_verified_clock(
    tmp_path: Path,
) -> None:
    v3 = load_protocol(
        *_write_document(tmp_path / "protocol-v3.json", _v3_document(), G_SOURCE_HEAD)
    )
    v4 = _load_v4(tmp_path, _document())
    content = _v3_content(v3)

    assert _v4_api().bind_schedule_v4(v4, content) == bind_schedule(v3, content)


def test_v4_schedule_rejects_a_clock_that_differs_from_the_frozen_grid(
    tmp_path: Path,
) -> None:
    protocol = _load_v4(tmp_path, _document())
    content = _v3_content(protocol, timestamp_shift_hours=1)

    with pytest.raises(ValueError):
        _v4_api().bind_schedule_v4(protocol, content)
