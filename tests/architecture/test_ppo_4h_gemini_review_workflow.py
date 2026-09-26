from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github" / "workflows" / "ppo-4h-gemini-review.yml"
REQUEST_MARKER = "<!-- ppo-4h-gemini-review-request-v1 -->"
UPLOAD = "actions/upload-artifact@043fb46d1a93c77aae656e7c1c64a875d1fc6a0a"
DOWNLOAD = "actions/download-artifact@484a0b528fb4d7bd804637ccb632e47a0e638317"


def _job(text: str, name: str, next_name: str | None) -> str:
    start = text.index(f"  {name}:\n")
    if next_name is None:
        return text[start:]
    end = text.index(f"  {next_name}:\n", start + 1)
    return text[start:end]


def test_gemini_review_workflow_is_default_branch_comment_triggered_and_read_only() -> (
    None
):
    text = WORKFLOW.read_text(encoding="utf-8")

    assert re.search(r"(?m)^on:\s*\n\s+issue_comment:\s*\n\s+types: \[created\]", text)
    assert "github.event.issue.pull_request" in text
    assert REQUEST_MARKER in text
    assert not re.search(r"(?m)^\s+create:", text)
    assert (
        "permissions:\n  actions: read\n  contents: read\n  issues: read\n  "
        "pull-requests: read" in text
    )
    assert "contents: write" not in text
    assert "pull-requests: write" not in text
    assert "actions: write" not in text
    assert "issues: write" not in text
    assert "runs-on: ubuntu-24.04" in text
    assert "timeout-minutes: 20" in text


def test_gemini_review_workflow_uses_request_packet_as_only_cross_job_evidence() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    for job in ("request", "core", "ppo-runtime", "guide", "review"):
        assert f"  {job}:\n" in text

    request = _job(text, "request", "core")
    core = _job(text, "core", "ppo-runtime")
    ppo = _job(text, "ppo-runtime", "guide")
    guide = _job(text, "guide", "review")
    review = _job(text, "review", None)

    assert "id: request" in request
    assert "id: packet" in request
    assert "ref: ${{ steps.request.outputs.reviewed_sha }}" in request
    assert "path: evidence" in request
    assert "python3 trusted/tools/ppo_4h_gemini_review.py packet" in request
    assert "output/review-packet.json" in request
    assert "packet_sha256: ${{ steps.packet.outputs.packet_sha256 }}" in request
    assert UPLOAD in request

    for verification in (core, ppo, guide):
        assert "ref: ${{ needs.request.outputs.reviewed_sha }}" in verification
        assert "GEMINI_API_KEY:" not in verification
        assert UPLOAD not in verification

    assert "needs: [request, core, ppo-runtime, guide]" in review
    assert "GEMINI_API_KEY: ${{ secrets.GEMINI_API_KEY }}" in review
    assert "ref: ${{ github.workflow_sha }}" in review
    assert "path: trusted" in review
    assert "path: target" not in review
    assert "path: evidence" not in review
    assert "actions/checkout@" in review
    assert DOWNLOAD in review
    assert "name: ppo-4h-review-packet-${{ github.run_id }}-${{ github.run_attempt }}" in review
    assert "PACKET_PATH:" in review
    assert "EXPECTED_PACKET_SHA256: ${{ needs.request.outputs.packet_sha256 }}" in review
    assert "python target/" not in review
    assert "uv run" not in review
    assert "pip install" not in review

    before_review = text[: text.index("  review:\n")]
    assert "GEMINI_API_KEY:" not in before_review
    assert "name: Trusted Lean Core verification" in core
    assert "name: Trusted PPO Runtime verification" in ppo
    assert "name: Trusted Human Guide verification" in guide


def test_gemini_review_workflow_binds_request_runtime_and_uploads_attestation() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    request = _job(text, "request", "core")
    review = _job(text, "review", None)

    assert "reviewed_sha: ${{ steps.request.outputs.reviewed_sha }}" in request
    assert "review_tag: ${{ steps.request.outputs.review_tag }}" in request
    assert (
        "review_tag_object_sha: ${{ steps.request.outputs.review_tag_object_sha }}"
        in request
    )
    assert "packet_sha256: ${{ steps.packet.outputs.packet_sha256 }}" in request

    assert "GEMINI_MODEL: ${{ vars.PPO_GEMINI_REVIEW_MODEL }}" in review
    assert "TRUSTED_WORKFLOW_SHA: ${{ github.workflow_sha }}" in review
    assert "REVIEWED_SHA: ${{ needs.request.outputs.reviewed_sha }}" in review
    assert "REVIEW_TAG: ${{ needs.request.outputs.review_tag }}" in review
    assert (
        "REVIEW_TAG_OBJECT_SHA: ${{ needs.request.outputs.review_tag_object_sha }}"
        in review
    )
    assert "REQUEST_PULL_NUMBER: ${{ github.event.issue.number }}" in review
    assert "REQUEST_COMMENT_ID: ${{ github.event.comment.id }}" in review
    assert "REQUESTER_LOGIN: ${{ github.event.comment.user.login }}" in review
    assert "GITHUB_RUN_ID: ${{ github.run_id }}" in review
    assert "GITHUB_RUN_ATTEMPT: ${{ github.run_attempt }}" in review
    assert "output/reviewer-attestation.json" in review
    assert UPLOAD in review

    action_refs = re.findall(r"(?m)^\s+uses:\s+([^\s]+)$", text)
    assert action_refs
    assert all(re.search(r"@[0-9a-f]{40}$", action) for action in action_refs)
