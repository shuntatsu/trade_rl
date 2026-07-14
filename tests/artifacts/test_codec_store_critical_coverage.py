from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path

import pytest

import trade_rl.artifacts.codec as codec
from trade_rl.artifacts.store import ArtifactStore


class ExampleEnum(Enum):
    VALUE = "value"


@dataclass(frozen=True)
class ExampleDataclass:
    value: int


def test_canonical_codec_covers_supported_and_rejected_values() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        codec.to_json_value(datetime(2026, 1, 1))
    with pytest.raises(TypeError, match="keys"):
        codec.to_json_value({1: "value"})
    with pytest.raises(ValueError, match="finite"):
        codec.to_json_value(float("nan"))
    with pytest.raises(TypeError, match="unsupported"):
        codec.to_json_value(b"bytes")

    aware = datetime(2026, 7, 14, tzinfo=UTC)
    assert codec.to_json_value(aware) == "2026-07-14T00:00:00Z"
    assert codec.to_json_value(ExampleEnum.VALUE) == "value"
    assert codec.to_json_value(Path("a/b")) == "a/b"
    assert codec.to_json_value(ExampleDataclass(3)) == {"value": 3}
    assert codec.to_json_value((1, 2)) == [1, 2]


def _write(root: Path, name: str, payload: bytes = b"x") -> Path:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


def test_artifact_store_rejects_invalid_publish_and_failure_transitions(
    tmp_path: Path,
) -> None:
    store = ArtifactStore(tmp_path / "store")
    with pytest.raises(ValueError, match="run_id"):
        store.stage_run("../bad")
    with pytest.raises(FileNotFoundError, match="does not exist"):
        store.publish_run("missing", validate=lambda _: True)

    rejected = store.stage_run("rejected")
    _write(rejected, "artifact")
    with pytest.raises(ValueError, match="validation failed"):
        store.publish_run("rejected", validate=lambda _: False)

    first = store.stage_run("duplicate")
    _write(first, "artifact")
    store.publish_run("duplicate", validate=lambda _: True)
    second = store.stage_run("duplicate")
    _write(second, "artifact")
    with pytest.raises(FileExistsError, match="already exists"):
        store.publish_run("duplicate", validate=lambda _: True)

    with pytest.raises(FileNotFoundError, match="does not exist"):
        store.mark_failed("missing")
    failed = store.stage_run("failed")
    _write(failed, "artifact")
    (store.failed_root / "failed").mkdir()
    with pytest.raises(FileExistsError, match="already exists"):
        store.mark_failed("failed")
