from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import UTC, datetime

import pytest

from trade_rl.artifacts.canonical import canonical_json_bytes
from trade_rl.artifacts.hashing import content_digest


@dataclass(frozen=True)
class ExampleManifest:
    dataset_id: str
    symbols: tuple[str, ...]
    created_at: datetime


def test_canonical_json_is_stable_for_mapping_order() -> None:
    left = {"z": 1, "a": {"y": 2, "x": 3}}
    right = {"a": {"x": 3, "y": 2}, "z": 1}

    assert canonical_json_bytes(left) == canonical_json_bytes(right)
    assert content_digest(left) == content_digest(right)


def test_canonical_json_serializes_dataclass_and_utc_timestamp() -> None:
    manifest = ExampleManifest(
        dataset_id="a" * 64,
        symbols=("BTCUSDT", "ETHUSDT"),
        created_at=datetime(2026, 7, 13, 6, 0, tzinfo=UTC),
    )

    payload = canonical_json_bytes(manifest).decode("utf-8")

    assert '"created_at":"2026-07-13T06:00:00Z"' in payload
    assert '"symbols":["BTCUSDT","ETHUSDT"]' in payload


def test_content_digest_changes_when_content_changes() -> None:
    assert content_digest({"value": 1}) != content_digest({"value": 2})


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_canonical_json_rejects_non_finite_floats(value: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        canonical_json_bytes({"value": value})
