from __future__ import annotations

import json
import runpy
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

_SCRIPT = Path("examples/binance-multitimeframe/run_symbol_triplet_training_stage.py")


def _namespace() -> dict[str, Any]:
    return runpy.run_path(str(_SCRIPT))


def test_parse_utc_requires_an_explicit_timezone() -> None:
    namespace = _namespace()

    with pytest.raises(ValueError, match="timezone"):
        namespace["_parse_utc"]("2026-01-01T00:00:00")

    assert namespace["_parse_utc"]("2026-01-01T09:00:00+09:00").isoformat() == (
        "2026-01-01T00:00:00+00:00"
    )


def test_result_payload_distinguishes_complete_and_published_stage() -> None:
    namespace = _namespace()

    assert namespace["_result_payload"](None) == {
        "schema_version": "binance_symbol_triplet_stage_command_result_v1",
        "status": "complete",
    }

    result = SimpleNamespace(
        request=SimpleNamespace(stage_id="a" * 64, stage_index=7),
        completion=SimpleNamespace(digest="b" * 64),
        cursor=SimpleNamespace(next_stage_index=8),
        training=SimpleNamespace(run_id="run-7", status="published"),
    )
    assert namespace["_result_payload"](result) == {
        "completion_digest": "b" * 64,
        "next_stage_index": 8,
        "run_id": "run-7",
        "schema_version": "binance_symbol_triplet_stage_command_result_v1",
        "stage_id": "a" * 64,
        "stage_index": 7,
        "status": "published",
    }


def test_main_passes_paths_and_prints_one_json_object(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    namespace = _namespace()
    observed: dict[str, Any] = {}

    def execute(**kwargs: Any) -> None:
        observed.update(kwargs)
        return None

    namespace["execute_binance_symbol_triplet_stage_command"] = execute
    exit_code = namespace["main"](
        [
            "--manifest-path",
            str(tmp_path / "manifest.json"),
            "--plan-path",
            str(tmp_path / "plan.json"),
            "--cursor-path",
            str(tmp_path / "cursor.json"),
            "--base-config-path",
            str(tmp_path / "training.json"),
            "--work-root",
            str(tmp_path / "work"),
            "--cache-root",
            str(tmp_path / "cache"),
            "--metadata-mode",
            "frozen_snapshot",
            "--start-time",
            "2024-12-01T00:00:00Z",
            "--end-time",
            "2026-07-01T00:00:00Z",
        ]
    )

    assert exit_code == 0
    assert observed["metadata_mode"] == "frozen_snapshot"
    assert observed["manifest_path"] == tmp_path / "manifest.json"
    assert observed["start_time"].isoformat() == "2024-12-01T00:00:00+00:00"
    assert json.loads(capsys.readouterr().out) == {
        "schema_version": "binance_symbol_triplet_stage_command_result_v1",
        "status": "complete",
    }
