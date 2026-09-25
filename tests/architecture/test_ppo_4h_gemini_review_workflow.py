from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github" / "workflows" / "ppo-4h-gemini-review.yml"


def test_gemini_review_workflow_is_default_branch_tag_triggered_and_read_only() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    assert re.search(r"(?m)^on:\s*\n\s+create:", text)
    assert "github.ref_type == 'tag'" in text
    assert "review/ppo-4h-indicator-smoke-v" in text
    assert (
        "permissions:\n  actions: read\n  contents: read\n  pull-requests: read" in text
    )
    assert "contents: write" not in text
    assert "pull-requests: write" not in text
    assert "actions: write" not in text
    assert "issues: write" not in text
    assert "runs-on: ubuntu-24.04" in text
    assert "timeout-minutes: 20" in text


def test_gemini_review_workflow_separates_trusted_runner_from_untrusted_target() -> (
    None
):
    text = WORKFLOW.read_text(encoding="utf-8")

    assert "ref: ${{ github.workflow_sha }}" in text
    assert "path: trusted" in text
    assert "ref: ${{ github.sha }}" in text
    assert "path: target" in text
    assert text.count("persist-credentials: false") >= 2
    assert "trusted/tools/ppo_4h_gemini_review.py" in text
    assert "python target/" not in text
    assert "uv run" not in text
    assert "pip install" not in text


def test_gemini_review_workflow_binds_runtime_identity_and_uploads_attestation() -> (
    None
):
    text = WORKFLOW.read_text(encoding="utf-8")

    assert "GEMINI_API_KEY: ${{ secrets.GEMINI_API_KEY }}" in text
    assert "GEMINI_MODEL: ${{ vars.PPO_GEMINI_REVIEW_MODEL }}" in text
    assert "TRUSTED_WORKFLOW_SHA: ${{ github.workflow_sha }}" in text
    assert "REVIEWED_SHA: ${{ github.sha }}" in text
    assert "REVIEW_TAG: ${{ github.ref_name }}" in text
    assert "GITHUB_RUN_ID: ${{ github.run_id }}" in text
    assert "GITHUB_RUN_ATTEMPT: ${{ github.run_attempt }}" in text
    assert "output/reviewer-attestation.json" in text
    assert "actions/upload-artifact@043fb46d1a93c77aae656e7c1c64a875d1fc6a0a" in text

    action_refs = re.findall(r"(?m)^\s+uses:\s+([^\s]+)$", text)
    assert action_refs
    assert all(re.search(r"@[0-9a-f]{40}$", action) for action in action_refs)
