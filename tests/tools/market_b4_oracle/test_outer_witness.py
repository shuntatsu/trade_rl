"""Exercise the bounded, externally pinned root-witness readback contract."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from tools.market_b4_oracle import outer_witness


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _witness_pair(
    tmp_path: Path,
) -> tuple[
    outer_witness.RootObservedAttempt,
    outer_witness.WitnessFilePin,
    outer_witness.WitnessFilePin,
    dict[str, Any],
    dict[str, Any],
]:
    attempt_uuid = "a" * 32
    repository_head = "b" * 40
    source = {
        "closure_sha256": "c" * 64,
        "repository_head": repository_head,
        "working_tree_clean": True,
        "git_status_sha256": "d" * 64,
    }
    argv = ["python", "-m", "tools.market_b4_oracle.offline_replay_consumer"]
    output_root = str((tmp_path / "output").absolute())
    start = {
        "schema": "issue810_root_owned_attempt_start_v1",
        "attempt_uuid": attempt_uuid,
        "clock": {"wall_utc_ns": 10, "monotonic_ns": 20},
        "argv": argv,
        "output_root": output_root,
        "implementation_source": source,
    }
    finish = {
        "schema": "issue810_root_owned_attempt_finish_v1",
        "attempt_uuid": attempt_uuid,
        "clock": {"wall_utc_ns": 11, "monotonic_ns": 21},
        "argv_sha256": hashlib.sha256(_canonical(argv)).hexdigest(),
        "output_root": output_root,
        "implementation_source": source,
        "child": {
            "pid": 123,
            "birth_ticks": 456,
            "exit_code": 0,
            "owned_tree_empty": True,
            "observed_process_identities": [[123, 456]],
        },
        "output_manifest": {"bytes": 1, "sha256": "e" * 64},
    }
    start_raw = _canonical(start)
    finish_raw = _canonical(finish)
    start_path = tmp_path / "attempt-start.json"
    finish_path = tmp_path / "attempt-finish.json"
    start_path.write_bytes(start_raw)
    finish_path.write_bytes(finish_raw)
    root_attempt = outer_witness._make_root_observed_attempt(
        start_path,
        finish_path,
        repository_head=repository_head,
        implementation_source_sha256="f" * 64,
    )
    return (
        root_attempt,
        outer_witness.WitnessFilePin(
            len(start_raw), hashlib.sha256(start_raw).hexdigest()
        ),
        outer_witness.WitnessFilePin(
            len(finish_raw), hashlib.sha256(finish_raw).hexdigest()
        ),
        start,
        finish,
    )


def test_root_observed_attempt_reads_exact_pinned_canonical_pair(
    tmp_path: Path,
) -> None:
    attempt, start_pin, finish_pin, expected_start, expected_finish = _witness_pair(
        tmp_path
    )

    start, finish = attempt.readback(
        expected_start_pin=start_pin,
        expected_finish_pin=finish_pin,
    )

    assert start == expected_start
    assert finish == expected_finish


def test_root_observed_attempt_rejects_pinned_finish_record_tampering(
    tmp_path: Path,
) -> None:
    attempt, start_pin, finish_pin, _, _ = _witness_pair(tmp_path)
    finish_path = attempt.finish_path
    original = finish_path.read_bytes()
    tampered = original.replace(b'"exit_code":0', b'"exit_code":1')
    assert len(tampered) == len(original) and tampered != original
    finish_path.write_bytes(tampered)

    with pytest.raises(ValueError, match="differs from its external pin"):
        attempt.readback(
            expected_start_pin=start_pin,
            expected_finish_pin=finish_pin,
        )
