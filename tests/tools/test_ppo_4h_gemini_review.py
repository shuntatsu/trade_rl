from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tools import ppo_4h_gemini_review as review
from trade_rl.artifacts import canonical_json_bytes


REVIEWED_SHA = "a" * 40
TAG_OBJECT_SHA = "b" * 40
WORKFLOW_SHA = "c" * 40
PACKET_SHA = "d" * 64


def _target(root: Path) -> None:
    for relative in review.PACKET_FILES:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"content for {relative}\n", encoding="utf-8")


def _gemini_response(**updates: object) -> dict[str, object]:
    result: dict[str, object] = {
        "g0": "PASS",
        "g1": "PASS",
        "g2": "EVIDENCE_BOUND",
        "disposition": "G0_G1_CLEAR_G2_EVIDENCE_BOUND",
        "blocking_findings": [],
        "strongest_counterexample": "future-only information changes a past action",
        "missing_evidence": [],
        "claim_downgrade": "development smoke authorization only",
        "machine_oracles": ["prefix invariance"],
        "what_this_cannot_prove": ["profitability"],
    }
    result.update(updates)
    return {
        "modelVersion": "gemini-model-version",
        "responseId": "response-123",
        "candidates": [
            {"content": {"parts": [{"text": json.dumps(result)}]}}
        ],
    }


def test_review_tag_name_is_strictly_versioned() -> None:
    assert review.require_review_tag("review/ppo-4h-indicator-smoke-v1") == 1
    assert review.require_review_tag("review/ppo-4h-indicator-smoke-v12") == 12
    for invalid in (
        "review/ppo-4h-indicator-smoke-v0",
        "review/ppo-4h-indicator-smoke-v01",
        "seal/ppo-4h-indicator-smoke-v1",
        "review/ppo-4h-indicator-smoke-v1/extra",
    ):
        with pytest.raises(ValueError, match="review tag"):
            review.require_review_tag(invalid)


def test_result_blind_packet_has_fixed_whitelist_and_content_hashes(
    tmp_path: Path,
) -> None:
    _target(tmp_path)
    packet = review.build_result_blind_packet(
        tmp_path,
        repository="owner/repo",
        reviewed_code_sha=REVIEWED_SHA,
        review_tag="review/ppo-4h-indicator-smoke-v1",
        review_tag_object_sha=TAG_OBJECT_SHA,
    )

    assert packet["schema"] == "ppo_4h_gemini_review_packet_v1"
    assert packet["reviewed_code_sha"] == REVIEWED_SHA
    assert packet["review_tag_object_sha"] == TAG_OBJECT_SHA
    files = packet["files"]
    assert [item["path"] for item in files] == list(review.PACKET_FILES)
    assert all(not item["path"].startswith("report/") for item in files)
    for item in files:
        raw = (tmp_path / item["path"]).read_bytes()
        assert item["sha256"] == hashlib.sha256(raw).hexdigest()
        assert item["text"].encode("utf-8") == raw


def test_result_blind_packet_rejects_missing_or_symlinked_source(
    tmp_path: Path,
) -> None:
    _target(tmp_path)
    missing = tmp_path / review.PACKET_FILES[0]
    missing.unlink()
    with pytest.raises(ValueError, match="packet source"):
        review.build_result_blind_packet(
            tmp_path,
            repository="owner/repo",
            reviewed_code_sha=REVIEWED_SHA,
            review_tag="review/ppo-4h-indicator-smoke-v1",
            review_tag_object_sha=TAG_OBJECT_SHA,
        )

    _target(tmp_path)
    target = tmp_path / review.PACKET_FILES[0]
    target.unlink()
    target.symlink_to(tmp_path / review.PACKET_FILES[1])
    with pytest.raises(ValueError, match="packet source"):
        review.build_result_blind_packet(
            tmp_path,
            repository="owner/repo",
            reviewed_code_sha=REVIEWED_SHA,
            review_tag="review/ppo-4h-indicator-smoke-v1",
            review_tag_object_sha=TAG_OBJECT_SHA,
        )


def test_software_ci_requires_exact_head_and_all_three_green_jobs() -> None:
    run = {
        "id": 123,
        "run_attempt": 2,
        "name": "CI",
        "event": "pull_request",
        "head_sha": REVIEWED_SHA,
        "status": "completed",
        "conclusion": "failure",
    }
    jobs = [
        {"name": "Lean Core", "status": "completed", "conclusion": "success"},
        {"name": "PPO Runtime", "status": "completed", "conclusion": "success"},
        {"name": "Human Guide", "status": "completed", "conclusion": "success"},
        {
            "name": "Independent Research Review",
            "status": "completed",
            "conclusion": "failure",
        },
    ]

    assert review.validate_software_ci(run, jobs, REVIEWED_SHA) == (123, 2)

    bad = [dict(item) for item in jobs]
    bad[1]["conclusion"] = "failure"
    with pytest.raises(ValueError, match="software verification"):
        review.validate_software_ci(run, bad, REVIEWED_SHA)


def test_parse_gemini_response_captures_provider_model_and_response_identity() -> None:
    parsed = review.parse_gemini_response(_gemini_response())

    assert parsed["reviewer_provider"] == "google_gemini"
    assert parsed["reviewer_model"] == "gemini-model-version"
    assert parsed["reviewer_response_id"] == "response-123"
    assert parsed["g0"] == "PASS"
    assert parsed["g1"] == "PASS"
    assert parsed["g2"] == "EVIDENCE_BOUND"
    assert parsed["blocking_findings"] == []


@pytest.mark.parametrize(
    ("updates", "match"),
    (
        ({"g0": "UNKNOWN"}, "G0"),
        ({"disposition": "APPROVE"}, "disposition"),
        ({"blocking_findings": "none"}, "blocking"),
    ),
)
def test_parse_gemini_response_rejects_noncanonical_semantics(
    updates: dict[str, object], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        review.parse_gemini_response(_gemini_response(**updates))


def test_attestation_binds_trusted_workflow_ci_packet_and_gemini_identity() -> None:
    parsed = review.parse_gemini_response(_gemini_response())
    record = review.build_attestation(
        repository="owner/repo",
        repository_id=99,
        reviewed_code_sha=REVIEWED_SHA,
        review_tag="review/ppo-4h-indicator-smoke-v1",
        review_tag_object_sha=TAG_OBJECT_SHA,
        trusted_workflow_sha=WORKFLOW_SHA,
        trusted_workflow_ref=(
            "owner/repo/.github/workflows/ppo-4h-gemini-review.yml@refs/heads/main"
        ),
        reviewer_run_id=456,
        reviewer_run_attempt=1,
        ci_run_id=123,
        ci_run_attempt=2,
        packet_sha256=PACKET_SHA,
        parsed_review=parsed,
    )

    assert record["schema"] == "ppo_4h_gemini_reviewer_run_v1"
    assert record["reviewer_provider"] == "google_gemini"
    assert record["reviewed_code_sha"] == REVIEWED_SHA
    assert record["trusted_workflow_sha"] == WORKFLOW_SHA
    assert record["ci_run_id"] == 123
    assert record["ci_run_attempt"] == 2
    assert record["packet_sha256"] == PACKET_SHA
    assert canonical_json_bytes(record) == canonical_json_bytes(
        json.loads(canonical_json_bytes(record))
    )
