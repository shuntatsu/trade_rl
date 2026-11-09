from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github" / "workflows" / "ppo-feature-checkpoint.yml"


def test_checkpoint_workflow_is_manual_pinned_and_read_only() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    assert re.search(r"(?m)^on:\s*\n\s+workflow_dispatch:", text)
    assert not re.search(r"(?m)^\s+(push|pull_request|schedule|workflow_run):", text)
    assert "permissions:\n  actions: read\n  contents: read" in text
    assert "runs-on: ubuntu-24.04" in text
    assert "timeout-minutes: 340" in text
    assert 'version: "0.10.0"' in text
    assert 'python-version: "3.12"' in text
    assert "uv sync --locked --extra train-sb3" in text
    assert "persist-credentials: false" in text
    assert "contents: write" not in text
    assert "actions: write" not in text
    assert "matrix:" not in text

    action_refs = re.findall(r"(?m)^\s+uses:\s+([^\s]+)$", text)
    assert action_refs
    assert all(re.search(r"@[0-9a-f]{40}$", action) for action in action_refs)


def test_checkpoint_workflow_always_uploads_only_checkpoint_and_receipt() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    upload_at = text.index("name: Upload checkpoint and receipt")
    upload_block = text[upload_at : text.find("\n  - name:", upload_at + 1)]
    assert "if: always()" in upload_block
    assert "include-hidden-files: true" in upload_block
    assert "output/checkpoint" in upload_block
    assert "output/receipt.json" in upload_block
    assert "output/source" not in upload_block
    assert "output/logs" not in upload_block
    assert "fetch-depth: 1" in text
    assert "actions/upload-artifact@043fb46d1a93c77aae656e7c1c64a875d1fc6a0a" in text

    transport_step = text.index("name: Run bounded checkpoint transport")
    upload_step = text.index("name: Upload checkpoint and receipt")
    failure_step = text.index("name: Propagate stage failure after artifact upload")
    assert transport_step < upload_step < failure_step
    assert "continue-on-error: true" in text[transport_step:upload_at]
    assert "CHECKPOINT_GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}" in text
    assert "TRANSPORT_OUTCOME: ${{ steps.transport.outcome }}" in text[failure_step:]
    assert "exit 1" in text[failure_step:]


def test_checkpoint_workflow_requires_review_record_digest_and_comment_read_access() -> (
    None
):
    workflow = WORKFLOW.read_text(encoding="utf-8")

    assert "review_record_sha256:" in workflow
    assert "CHECKPOINT_REVIEW_RECORD_SHA256" in workflow
    assert "issues: read" in workflow
    assert (
        "Operator-supplied review reference; not independently authenticated"
        not in workflow
    )


def test_checkpoint_workflow_uses_runner_python_without_managed_downloads() -> None:
    """Manual checkpoint must not fetch the unavailable standalone Python."""
    text = WORKFLOW.read_text(encoding="utf-8")

    assert (
        "  checkpoint:\n"
        "    runs-on: ubuntu-24.04\n"
        "    timeout-minutes: 340\n"
        "    env:\n"
        "      UV_PYTHON_PREFERENCE: only-system\n"
        "      UV_PYTHON_DOWNLOADS: never\n"
        in text
    )

    checkout = text.index("name: Checkout immutable code SHA")
    runner = text.index("name: Set up trusted runner Python 3.12")
    uv = text.index("name: Install pinned uv")
    sync = text.index("name: Install frozen PPO runtime")
    assert checkout < runner < uv < sync

    runner_step = text[runner:uv]
    assert (
        "uses: actions/setup-python@6d4fd22b20465ccc0f7354476588e8df7c9ca0e0"
        in runner_step
    )
    assert 'python-version: "3.12"' in runner_step
    assert "check-latest: false" in runner_step

    uv_step = text[uv:sync]
    assert (
        "uses: astral-sh/setup-uv@68559e99cc807c1eb7e9238fd43a15bba460eec2"
        in uv_step
    )
    assert 'version: "0.10.0"' in uv_step
    assert "enable-cache: false" in uv_step
    assert "python-version:" not in uv_step
