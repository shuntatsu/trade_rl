from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github" / "workflows" / "ppo-4h-gemini-review.yml"
REQUEST_MARKER = "<!-- ppo-4h-gemini-review-request-v1 -->"


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


def test_gemini_review_workflow_separates_verification_from_secret_reviewer() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    for job in ("request", "core", "ppo-runtime", "guide", "review"):
        assert f"  {job}:\n" in text

    review_start = text.index("  review:\n")
    review = text[review_start:]
    assert "needs: [request, core, ppo-runtime, guide]" in review
    assert "GEMINI_API_KEY: ${{ secrets.GEMINI_API_KEY }}" in review
    assert "ref: ${{ github.workflow_sha }}" in review
    assert "path: trusted" in review
    assert "path: target" not in review
    assert "steps.request.outputs.reviewed_sha" in review
    assert "python target/" not in review
    assert "uv run" not in review
    assert "pip install" not in review

    before_review = text[:review_start]
    assert "GEMINI_API_KEY:" not in before_review
    assert before_review.count("ref: ${{ needs.request.outputs.reviewed_sha }}") >= 3
    assert "name: Trusted Lean Core verification" in before_review
    assert "name: Trusted PPO Runtime verification" in before_review
    assert "name: Trusted Human Guide verification" in before_review


def test_gemini_review_workflow_binds_comment_runtime_and_uploads_attestation() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    assert "GEMINI_MODEL: ${{ vars.PPO_GEMINI_REVIEW_MODEL }}" in text
    assert "TRUSTED_WORKFLOW_SHA: ${{ github.workflow_sha }}" in text
    assert "REVIEWED_SHA: ${{ needs.request.outputs.reviewed_sha }}" in text
    assert "REVIEW_TAG: ${{ needs.request.outputs.review_tag }}" in text
    assert "REQUEST_PULL_NUMBER: ${{ github.event.issue.number }}" in text
    assert "REQUEST_COMMENT_ID: ${{ github.event.comment.id }}" in text
    assert "REQUESTER_LOGIN: ${{ github.event.comment.user.login }}" in text
    assert "GITHUB_RUN_ID: ${{ github.run_id }}" in text
    assert "GITHUB_RUN_ATTEMPT: ${{ github.run_attempt }}" in text
    assert "output/reviewer-attestation.json" in text
    assert "actions/upload-artifact@043fb46d1a93c77aae656e7c1c64a875d1fc6a0a" in text

    action_refs = re.findall(r"(?m)^\s+uses:\s+([^\s]+)$", text)
    assert action_refs
    assert all(re.search(r"@[0-9a-f]{40}$", action) for action in action_refs)
