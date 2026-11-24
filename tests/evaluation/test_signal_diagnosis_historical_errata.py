"""Immutable historical SD2 packet errata are explicit and non-authorizing.

Historical research_assurance_v1 packet bytes are never corrected in place.
This tests a separate versioned lineage ledger, not economic research results.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
ERRATA = ROOT / "report/sd2-signal-diagnosis-historical-errata-20261121.json"
CURRENT = (
    ROOT
    / "report/sd2-signal-diagnosis-research-assurance-20261121-after-core7-m2-main.json"
)
DOC = ROOT / "docs/architecture/research-assurance.md"
PROTOCOL = ROOT / "report/issue1146-sd0-signal-diagnosis-protocol-v4-20261111.json"
SUBJECTS = {
    "report/sd2-signal-diagnosis-research-assurance-20261120-after-trackd-main.json": (
        "f67e84011eec0950f142171887f0847c81396d18",
        "3e206e9620a4eeb0970eecbcceb550faa8275c15",
        {
            "stale_falsifier_source_head",
            "trackd_overlay_oracle_not_pinned",
            "fs1_bar_boundary_wording",
        },
    ),
    "report/sd2-signal-diagnosis-research-assurance-20261120-safe-unified-runner.json": (
        "2f444c043bf4dccc4e4647986774ab6d24022a4a",
        "2f725160b4f585deb74eb431d846814676de996d",
        {"later_d2_ic_cli_claims_not_pinned", "fs1_bar_boundary_wording"},
    ),
    "report/sd2-signal-diagnosis-research-assurance-20261120-after-main1288-consistent.json": (
        "f6ab2464aebf4efc687b8cbb3ee1a6642a5c425e",
        "04d7f22701b3dd0067e5d72cebe6db6d11a60c69",
        {"later_ic_cli_claims_not_pinned", "fs1_bar_boundary_wording"},
    ),
}


def _ledger() -> dict[str, Any]:
    return json.loads(ERRATA.read_text(encoding="utf-8"))


def _git_blob_sha(raw: bytes) -> str:
    prefix = b"blob " + str(len(raw)).encode("ascii") + b"\0"
    return hashlib.sha1(prefix + raw).hexdigest()  # noqa: S324 - Git blob identity is SHA-1


def test_historical_errata_is_distinct_and_never_authorizes_studies() -> None:
    entry = _ledger()
    assert entry["schema_version"] == "sd2_historical_assurance_errata_v1"
    assert entry["status"] == "HISTORICAL_FINDINGS_OPEN"
    assert entry["source_scope"] == "software_only"
    assert entry["immutable_originals_untouched"] is True
    assert entry["authorizes_economic_execution"] is False
    assert entry["authorizes_live_trading"] is False
    assert entry["independent_historical_packet_review_pass"] is False
    assert entry["current_author_packet"] == CURRENT.relative_to(ROOT).as_posix()
    current = json.loads(CURRENT.read_text(encoding="utf-8"))
    assert current["claims"]["production_eligible"] is False
    assert current["claims"]["live_trading_authorized"] is False


def test_historical_errata_pins_exact_original_blobs_and_records_all_findings() -> None:
    entry = _ledger()
    subjects = entry["historical_packets"]
    assert {subject["path"] for subject in subjects} == set(SUBJECTS)
    assert len(subjects) == len(SUBJECTS)
    for subject in subjects:
        path = subject["path"]
        sha, head, required_ids = SUBJECTS[path]
        raw = (ROOT / path).read_bytes()
        packet = json.loads(raw)
        assert _git_blob_sha(raw) == subject["git_blob_sha"] == sha
        assert packet["identity"]["implementation_head"] == head
        assert subject["historical_source_head"] == head
        assert packet["claims"]["live_trading_authorized"] is False
        assert packet["claims"]["production_eligible"] is False
        assert subject["status"] == "historical_unreviewed_not_authoritative"
        assert required_ids <= {finding["id"] for finding in subject["findings"]}
        assert all(finding["correction"] for finding in subject["findings"])


def test_historical_errata_uses_frozen_half_open_bar_period_boundary() -> None:
    ledger = _ledger()
    data = json.loads(PROTOCOL.read_text(encoding="utf-8"))["data"]
    cutoff = ledger["window_boundary_interpretation"]
    assert data["window_semantics"] == "half_open_bar_period_interval"
    assert data["base_interval"] == "1h"
    assert cutoff["period_semantics"] == data["window_semantics"]
    assert cutoff["bar_close_timestamp_inclusive_end"] == data["evaluation_end"]
    assert (
        cutoff["first_forbidden_fs1_period_open"] == data["forbidden_spans"][0]["start"]
    )
    assert cutoff["bar_closing_at_end_is_admitted"] is True
    assert cutoff["bar_closing_after_end_is_refused"] is True
    assert cutoff["forbids_any_fs1_period_access"] is True


def test_current_owner_docs_link_the_non_authorizing_errata() -> None:
    contents = DOC.read_text(encoding="utf-8")
    assert (
        "../../report/sd2-signal-diagnosis-historical-errata-20261121.json" in contents
    )
    assert "HISTORICAL_FINDINGS_OPEN" in contents
