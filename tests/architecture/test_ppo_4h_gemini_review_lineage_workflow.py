from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github" / "workflows" / "ppo-4h-gemini-review.yml"


def test_review_packet_artifact_is_identity_and_attempt_bound() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    assert "review_identity_digest:" in text
    assert "packet_sha256:" in text
    assert (
        "ppo-4h-review-packet-${{ steps.request.outputs.review_identity_digest }}"
        in text
    )
    assert "${{ github.run_id }}-${{ github.run_attempt }}" in text


def test_provider_step_marks_terminal_response_before_disposition_propagation() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    review_start = text.index("  review:\n")
    review_job = text[review_start:]
    assert "name: Trusted Gemini semantic review" in review_job
    assert "name: Call Gemini reviewer" in review_job
    assert "id: provider" in review_job
    assert "continue-on-error: true" in review_job
    assert (
        "REVIEW_IDENTITY_DIGEST: ${{ needs.request.outputs.review_identity_digest }}"
        in review_job
    )
    assert "PROVIDER_OUTCOME: ${{ steps.provider.outcome }}" in review_job
    assert "output/reviewer-disposition.txt" in review_job
    assert review_job.index("name: Call Gemini reviewer") < review_job.index(
        "name: Upload immutable Gemini review evidence"
    )
    assert review_job.index("name: Upload immutable Gemini review evidence") < (
        review_job.index("name: Propagate reviewer disposition")
    )


def test_secret_bearing_review_job_never_checks_out_or_executes_target() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    review_start = text.index("  review:\n")
    review_job = text[review_start:]
    assert "GEMINI_API_KEY:" in review_job
    assert "actions/download-artifact@" in review_job
    assert "path: target" not in review_job
    assert "path: evidence" not in review_job
    assert "python target/" not in review_job
    assert "uv run" not in review_job
    assert "npm " not in review_job
