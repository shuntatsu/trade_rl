"""Exact claim timestamp identity; synthetic markers only."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from tools import stage_claims_status as status


def _claim(stamp: str, *, legacy: bool = False) -> bytes:
    identity = {
        "stage": "STAGE_ONE",
        "protocol_digest": "a1" * 32,
        "commit": "d4" * 20,
        "claimed_at": stamp,
    }
    if legacy:
        body = {**identity, "authorization_comment_id": 22}
    else:
        body = {
            **identity,
            "schema_version": "single_use_stage_claim_v1",
            "repository_id": 5,
            "comment_id": 22,
        }
    return json.dumps(body).encode("utf-8")


@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.parametrize(
    "stamp",
    [
        "2027-01-02T03:04:05.000000001+00:00",
        "2027-01-02T03:04:05.1234567Z",
        "2027-01-02T03:04:05,0000002+00:00",
        "2027-01-02T03:04:05+00:00:00.000001",
        "2027-01-02T03:04:05+08:60",
        "2027-01-02T03:04+00:00",
        "20270102T030405+00:00",
        "2027-01-02X03:04:05+00:00",
        "2027-01-02T03:04:05",
        "2027-02-29T03:04:05Z",
    ],
)
def test_parser_refuses_unrepresentable_or_ambiguous_claim_time(
    legacy: bool, stamp: str
) -> None:
    with pytest.raises(ValueError, match="^invalid claim field$"):
        status._parse_claim(_claim(stamp, legacy=legacy), "STAGE_ONE")


@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.parametrize(
    ("stamp", "expected"),
    [
        ("2027-01-02T03:04:05+00:00", "2027-01-02T03:04:05+00:00"),
        ("2027-01-02T03:04:05Z", "2027-01-02T03:04:05+00:00"),
        ("2027-01-02T12:04:05.123456+09:00", "2027-01-02T03:04:05.123456+00:00"),
        ("2027-01-01T22:04:05-05:00", "2027-01-02T03:04:05+00:00"),
        ("2027-01-02 03:04:05,123456+00:00", "2027-01-02T03:04:05.123456+00:00"),
        ("2027-01-02T03:04:05.123456000Z", "2027-01-02T03:04:05.123456+00:00"),
    ],
)
def test_parser_preserves_exact_claim_time(
    legacy: bool, stamp: str, expected: str
) -> None:
    claim = status._parse_claim(_claim(stamp, legacy=legacy), "STAGE_ONE")
    assert claim.claimed_at == expected


@pytest.mark.skipif(os.name != "posix", reason="strict dir-fd POSIX contract")
@pytest.mark.parametrize(
    ("stage_time", "comment_time", "expected"),
    [
        (
            "2027-01-02T03:04:05.000000001Z",
            "2027-01-02T03:04:05.000000002Z",
            status.INVALID,
        ),
        (
            "2027-01-02T03:04:05.123456Z",
            "2027-01-02T12:04:05.123456+09:00",
            status.CLAIMED,
        ),
        (
            "2027-01-02T03:04:05.123456Z",
            "2027-01-02T03:04:05.123457Z",
            status.INVALID,
        ),
    ],
)
def test_report_does_not_collapse_distinct_marker_times(
    tmp_path: Path, stage_time: str, comment_time: str, expected: str
) -> None:
    (tmp_path / "stage-STAGE_ONE.claim").write_bytes(_claim(stage_time))
    (tmp_path / "comment-5-22.claim").write_bytes(_claim(comment_time))
    report = status.build_report([tmp_path])
    entry = next(
        item for item in report["ledgers"][0]["entries"] if item["stage"] == "STAGE_ONE"
    )
    assert entry["state"] == expected
