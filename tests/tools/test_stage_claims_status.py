"""Synthetic oracles for the read-only stage claim status report."""

from __future__ import annotations

import json
import os
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pytest

from tools import stage_claims_status as status

DIGEST_A = "a1" * 32
DIGEST_B = "b2" * 32
RESULT_SHA = "c3" * 32
COMMIT = "d4" * 20
TODAY = date(2027, 1, 10)

# Distinctive sentinels: none may ever reach any output.
METRIC_VALUES = ("0.987654321", "31415.9265", "-0.4242424", "7777.123", "55.5551")
METRIC_KEYS = (
    "pnl",
    "sharpe",
    "max_drawdown",
    "equity",
    "returns",
    "net_return",
    "positions",
    "metrics",
    "summary",
)


def _write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
    path.write_bytes(data)


def _legacy_claim(
    stage: str = "INTERIM_A", digest: str = DIGEST_A, comment: int = 11
) -> dict[str, Any]:
    return {
        "authorization_comment_id": comment,
        "claimed_at": "2026-11-22T08:29:51.137423+00:00",
        "commit": COMMIT,
        "protocol_digest": digest,
        "stage": stage,
    }


def _v1_claim(stage: str = "STAGE_ONE", comment: int = 22) -> dict[str, Any]:
    return {
        "schema_version": "single_use_stage_claim_v1",
        "stage": stage,
        "protocol_digest": DIGEST_A,
        "commit": COMMIT,
        "repository_id": 5,
        "comment_id": comment,
        "claimed_at": "2027-01-02T03:04:05+00:00",
    }


def _result_with_metrics(
    stage: str = "INTERIM_A", digest: str = DIGEST_A, label: str = "NOT_SUPPORTED"
) -> dict[str, Any]:
    return {
        "stage": stage,
        "label": label,
        "protocol_digest": digest,
        "result_sha256": RESULT_SHA,
        "supported": True,
        "pnl": 0.987654321,
        "sharpe": 31415.9265,
        "max_drawdown": -0.4242424,
        "equity": [1.0, 7777.123],
        "returns": {"daily": [55.5551], "total": 0.987654321},
        "metrics": {"nested": {"net_return": -0.4242424, "positions": [7777.123]}},
        "summary": {"sharpe": 31415.9265, "label": "SHADOW_LABEL"},
    }


def _outputs(root: Path, tmp_path: Path, **kwargs: Any) -> tuple[str, str, str]:
    report = status.build_report([root], **kwargs)
    return (
        status.render_text(report),
        status.render_json(report),
        status.render_html(report),
    )


def _entries(root: Path, **kwargs: Any) -> list[dict[str, Any]]:
    return status.build_report([root], **kwargs)["ledgers"][0]["entries"]


def test_claimed_and_recorded_legacy_stages(tmp_path: Path) -> None:
    root = tmp_path / "ledger"
    _write(root / "INTERIM_A.claim", _legacy_claim())
    _write(root / "INTERIM_A.result.json", _result_with_metrics())
    _write(root / "FINAL.claim", _legacy_claim("FINAL", comment=12))
    entries = {e["stage"]: e for e in _entries(root)}
    recorded = entries["INTERIM_A"]
    assert recorded["state"] == status.RESULT_RECORDED
    assert recorded["result_label"] == "NOT_SUPPORTED"
    assert recorded["protocol_digest"] == DIGEST_A
    assert recorded["result_sha256"] == RESULT_SHA
    assert recorded["claimed_at"] == "2026-11-22T08:29:51.137423+00:00"
    assert recorded["result_recorded_at"].endswith("+00:00")
    assert recorded["comment_id"] == 11
    assert entries["FINAL"]["state"] == status.CLAIMED
    assert entries["FINAL"]["result_label"] is None


def test_no_metric_data_in_any_output(tmp_path: Path) -> None:
    root = tmp_path / "ledger"
    _write(root / "INTERIM_A.claim", _legacy_claim())
    _write(root / "INTERIM_A.result.json", _result_with_metrics())
    html_path = tmp_path / "page.html"
    assert status.main([str(root), "--html", str(html_path)]) == 0
    outputs = [*_outputs(root, tmp_path), html_path.read_text(encoding="utf-8")]
    for text in outputs:
        lowered = text.lower()
        for key in METRIC_KEYS:
            assert key not in lowered
        for value in METRIC_VALUES:
            assert value not in text
        assert "SHADOW_LABEL" not in text
        assert '"supported"' not in lowered
    for text in outputs:
        assert "no performance data" in text.lower()


def test_result_view_uses_only_the_allowlist(tmp_path: Path) -> None:
    root = tmp_path / "ledger"
    _write(root / "INTERIM_A.claim", _legacy_claim())
    _write(root / "INTERIM_A.result.json", _result_with_metrics())
    (entry,) = _entries(root)
    assert status.RESULT_ALLOWLIST == (
        "stage",
        "label",
        "protocol_digest",
        "result_sha256",
    )
    assert set(entry) == {
        "location",
        "stage",
        "state",
        "protocol_digest",
        "commit",
        "comment_id",
        "claimed_at",
        "result_recorded_at",
        "result_label",
        "result_sha256",
        "files",
        "reason",
        "earliest_allowed",
        "days_until_due",
        "schedule_status",
        "file",
    }


def test_v1_claim_with_comment_marker(tmp_path: Path) -> None:
    root = tmp_path / "ledger"
    _write(root / "stage-STAGE_ONE.claim", _v1_claim())
    _write(root / "comment-5-22.claim", _v1_claim())
    (entry,) = _entries(root)
    assert entry["state"] == status.CLAIMED
    assert entry["comment_id"] == 22
    assert entry["files"] == ["comment-5-22.claim", "stage-STAGE_ONE.claim"]


def test_v1_comment_marker_without_stage_is_invalid(tmp_path: Path) -> None:
    root = tmp_path / "ledger"
    _write(root / "comment-5-22.claim", _v1_claim())
    (entry,) = _entries(root)
    assert entry["state"] == status.INVALID
    assert "without a stage claim" in entry["reason"]


def test_namespaced_digest_directory(tmp_path: Path) -> None:
    root = tmp_path / "ledger"
    good = root / DIGEST_A
    _write(good / "BACKCAST.claim", _legacy_claim("BACKCAST"))
    _write(good / "BACKCAST.result.json", _result_with_metrics("BACKCAST", DIGEST_A))
    bad = root / DIGEST_B
    _write(bad / "BACKCAST.claim", _legacy_claim("BACKCAST"))
    states = {e["location"]: e for e in _entries(root)}
    assert states[DIGEST_A]["state"] == status.RESULT_RECORDED
    assert states[DIGEST_B]["state"] == status.INVALID
    assert "digest-named directory" in states[DIGEST_B]["reason"]


@pytest.mark.parametrize(
    ("claim_payload", "reason"),
    [
        (b"{not json", "not valid UTF-8 JSON"),
        (b"[1, 2]", "not an object"),
        (b"\xff\xfe", "not valid UTF-8 JSON"),
        ({"stage": "INTERIM_A"}, "fields differ"),
        ({**_legacy_claim(), "extra": 1}, "fields differ"),
        ({**_legacy_claim(), "protocol_digest": "0" * 64}, "invalid claim field"),
        ({**_legacy_claim(), "commit": "xyz"}, "invalid claim field"),
        ({**_legacy_claim(), "authorization_comment_id": True}, "invalid claim"),
        ({**_legacy_claim(), "claimed_at": "2026-11-22T08:29:51"}, "invalid claim"),
        ({**_legacy_claim(), "claimed_at": "yesterday"}, "invalid claim"),
        ({**_legacy_claim(), "claimed_at": 5}, "not a string"),
        (_legacy_claim("OTHER"), "differs from its file name"),
        ({**_v1_claim(), "schema_version": "v9"}, "unknown claim schema"),
        (_v1_claim("INTERIM_A"), "claim schema differs from its file name"),
    ],
)
def test_invalid_legacy_claims_report_a_reason(
    tmp_path: Path, claim_payload: Any, reason: str
) -> None:
    root = tmp_path / "ledger"
    _write(root / "INTERIM_A.claim", claim_payload)
    (entry,) = _entries(root)
    assert entry["state"] == status.INVALID
    assert reason in entry["reason"]


def test_foreign_files_and_orphan_results_are_invalid(tmp_path: Path) -> None:
    root = tmp_path / "ledger"
    _write(root / "notes.txt", b"hello")
    _write(root / "lowercase.claim", _legacy_claim())
    _write(root / "ORPHAN.result.json", _result_with_metrics("ORPHAN"))
    _write(root / "BADRES.claim", _legacy_claim("BADRES"))
    _write(root / "BADRES.result.json", {"stage": "BADRES", "label": "x y"})
    entries = _entries(root)
    assert {e["state"] for e in entries} == {status.INVALID}
    reasons = " | ".join(e["reason"] for e in entries)
    assert "foreign file" in reasons
    assert "without a stage claim" in reasons
    assert "result label is missing or malformed" in reasons
    text = "".join(_outputs(root, tmp_path))
    for value in METRIC_VALUES:
        assert value not in text


def test_result_digest_mismatch_and_stage_mismatch(tmp_path: Path) -> None:
    root = tmp_path / "ledger"
    _write(root / "INTERIM_A.claim", _legacy_claim())
    _write(root / "INTERIM_A.result.json", _result_with_metrics(digest=DIGEST_B))
    _write(root / "OTHER.claim", _legacy_claim("OTHER"))
    _write(root / "OTHER.result.json", _result_with_metrics("INTERIM_A"))
    by_stage = {e["stage"]: e for e in _entries(root)}
    assert "result protocol digest differs" in by_stage["INTERIM_A"]["reason"]
    assert "differs from its file name" in by_stage["OTHER"]["reason"]


def test_oversize_and_symlink_and_unreadable(tmp_path: Path) -> None:
    root = tmp_path / "ledger"
    _write(root / "BIG.claim", b"x" * (70 * 1024))
    target = tmp_path / "target.json"
    _write(target, _legacy_claim("LINK"))
    try:
        os.symlink(target, root / "LINK.claim")
    except (OSError, NotImplementedError):
        pass
    entries = _entries(root)
    big = next(e for e in entries if e["stage"] == "BIG")
    assert big["state"] == status.INVALID
    assert "size limit" in big["reason"]
    assert all(e["state"] == status.INVALID for e in entries)
    if (root / "LINK.claim").is_symlink():
        assert any("symlink" in (e["reason"] or "") for e in entries)


def test_missing_ledger_is_reported_not_raised(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert status.main([str(tmp_path / "absent")]) == 1
    assert "not a readable real directory" in capsys.readouterr().out


def test_json_is_deterministic_and_order_independent(tmp_path: Path) -> None:
    one, two = tmp_path / "one", tmp_path / "two"
    names = ["B_STAGE", "A_STAGE", "C_STAGE"]
    for root, order in ((one, names), (two, list(reversed(names)))):
        for name in order:
            _write(root / f"{name}.claim", _legacy_claim(name, comment=1))
    first = status.render_json(status.build_report([one]))
    again = status.render_json(status.build_report([one]))
    assert first == again
    second = json.loads(status.render_json(status.build_report([two])))
    parsed = json.loads(first)["ledgers"][0]["entries"]
    assert [e["stage"] for e in parsed] == sorted(names)
    assert [e["stage"] for e in second["ledgers"][0]["entries"]] == [
        "A_STAGE",
        "B_STAGE",
        "C_STAGE",
    ]
    assert first.endswith("\n") and "\r" not in first
    assert json.loads(first)["as_of"] is None


def test_html_is_self_contained_and_exclusive(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = tmp_path / "ledger"
    _write(root / "INTERIM_A.claim", _legacy_claim())
    _write(root / "a&b'x.txt", b"x")
    page = tmp_path / "out.html"
    assert status.main([str(root), "--html", str(page)]) == 0
    first = page.read_bytes()
    lowered = first.decode("utf-8").lower()
    assert "<script" not in lowered
    assert "a&amp;b&#x27;x.txt: foreign file" in lowered
    for banned in ("http://", "https://", "src=", "<link", "<img", "@import", "url("):
        assert banned not in lowered
    assert "no performance data" in lowered
    assert b"\r" not in first
    page.write_bytes(b"SENTINEL")
    assert status.main([str(root), "--html", str(page)]) == 2
    assert page.read_bytes() == b"SENTINEL"
    assert "cannot write HTML" in capsys.readouterr().err
    assert status.main([str(root), "--html", str(page), "--replace"]) == 0
    assert page.read_bytes() == first
    assert not [p for p in tmp_path.iterdir() if p.name.startswith(".stage-status-")]


def test_schedule_due_and_overdue_with_injected_clock(tmp_path: Path) -> None:
    root = tmp_path / "ledger"
    _write(root / "INTERIM_A.claim", _legacy_claim())
    schedule = {
        DIGEST_A: {
            "INTERIM_A": date(2027, 1, 1),
            "INTERIM_B": date(2027, 1, 15),
            "INTERIM_C": date(2027, 1, 10),
            "INTERIM_D": date(2027, 1, 7),
        },
        DIGEST_B: {"OTHER": date(2027, 2, 1)},
    }
    report = status.build_report([root], schedule, TODAY)
    entries = {e["stage"]: e for e in report["ledgers"][0]["entries"]}
    assert entries["INTERIM_A"]["state"] == status.CLAIMED
    assert entries["INTERIM_A"]["earliest_allowed"] == "2027-01-01"
    assert entries["INTERIM_A"]["days_until_due"] is None
    assert entries["INTERIM_B"]["state"] == status.NOT_CLAIMED
    assert entries["INTERIM_B"]["days_until_due"] == 5
    assert entries["INTERIM_B"]["schedule_status"] == "NOT_YET_DUE"
    assert entries["INTERIM_C"]["days_until_due"] == 0
    assert entries["INTERIM_C"]["schedule_status"] == "DUE_TODAY"
    assert entries["INTERIM_D"]["days_until_due"] == -3
    assert entries["INTERIM_D"]["schedule_status"] == "OVERDUE"
    assert report["schedule_unmatched_protocols"] == [DIGEST_B]
    assert report["as_of"] == "2027-01-10"
    text = status.render_text(report)
    assert "2027-01-15 (in 5d)" in text
    assert "2027-01-07 (overdue 3d)" in text


def test_no_schedule_shows_no_due_information(tmp_path: Path) -> None:
    root = tmp_path / "ledger"
    _write(root / "INTERIM_A.claim", _legacy_claim())
    (entry,) = _entries(root)
    assert entry["earliest_allowed"] is None
    assert entry["schedule_status"] is None


def test_cli_schedule_uses_injected_clock_and_rejects_bad_schedule(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = tmp_path / "ledger"
    _write(root / "INTERIM_A.claim", _legacy_claim())
    good = tmp_path / "schedule.json"
    _write(good, {DIGEST_A: {"INTERIM_B": "2027-01-12"}})
    code = status.main(
        [str(root), "--schedule", str(good), "--json"],
        today=lambda: datetime(2027, 1, 10, 23, 59, tzinfo=UTC).date(),
    )
    assert code == 0
    parsed = json.loads(capsys.readouterr().out)
    due = [e for e in parsed["ledgers"][0]["entries"] if e["stage"] == "INTERIM_B"]
    assert due[0]["days_until_due"] == 2
    for bad in (
        {"nothex": {"INTERIM_B": "2027-01-12"}},
        {DIGEST_A: {"lower": "2027-01-12"}},
        {DIGEST_A: {"INTERIM_B": "12/01/2027"}},
        {DIGEST_A: {"INTERIM_B": "2027-13-45"}},
        {DIGEST_A: "oops"},
    ):
        broken = tmp_path / "bad.json"
        _write(broken, bad)
        assert status.main([str(root), "--schedule", str(broken)]) == 2
        capsys.readouterr()
    assert status.main([str(root), "--schedule", str(tmp_path / "none.json")]) == 2


def test_ledger_directory_is_never_modified(tmp_path: Path) -> None:
    root = tmp_path / "ledger"
    _write(root / "INTERIM_A.claim", _legacy_claim())
    _write(root / "INTERIM_A.result.json", _result_with_metrics())
    before = sorted(
        (p.name, p.read_bytes(), p.stat().st_mtime_ns) for p in root.iterdir()
    )
    status.main([str(root), "--json"])
    after = sorted(
        (p.name, p.read_bytes(), p.stat().st_mtime_ns) for p in root.iterdir()
    )
    assert before == after
