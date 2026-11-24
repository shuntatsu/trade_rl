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
    assert 'version: "0.13.9"' in text
    assert 'python-version: "3.12.16"' in text
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

    expected_env = (
        "    env:\n"
        "      UV_PYTHON_PREFERENCE: only-system\n"
        "      UV_PYTHON_DOWNLOADS: never\n"
    )
    assert expected_env in text
    assert text.count("UV_PYTHON_DOWNLOADS: never") == 1

    checkpoint = text.index("  checkpoint:\n")
    checkout = text.index("name: Checkout immutable code SHA")
    runner = text.index("name: Set up trusted runner Python 3.12.16")
    uv = text.index("name: Install pinned uv")
    sync = text.index("name: Install frozen PPO runtime")
    assert checkpoint < text.index(expected_env) < checkout < runner < uv < sync

    runner_step = text[runner:uv]
    python_sha = "6d4fd22b20465ccc0f7354476588e8df7c9ca0e0"
    assert "uses: actions/setup-python@" + python_sha in runner_step
    assert 'python-version: "3.12.16"' in runner_step
    assert "check-latest: false" in runner_step

    uv_step = text[uv:sync]
    uv_sha = "1db35dd8eee11c5894872e0b383ae6c406cf738d"
    assert "uses: astral-sh/setup-uv@" + uv_sha in uv_step
    assert 'version: "0.13.9"' in uv_step
    assert "enable-cache: false" in uv_step
    assert "python-version:" not in uv_step


RUNTIME_SMOKE = ROOT / ".github" / "workflows" / "ppo-checkpoint-runtime-preflight.yml"


def test_manual_checkpoint_is_new_lineage_only_on_exact_cached_python() -> None:
    """No silent 3.12 patch changes, managed downloads, or old-code resume."""
    workflow = WORKFLOW.read_text(encoding="utf-8")
    checkout = workflow.index("name: Checkout immutable code SHA")
    cached = workflow.index("name: Require Python 3.12.16 in runner toolcache")
    setup = workflow.index("name: Set up trusted runner Python 3.12.16")
    uv_setup = workflow.index("name: Install pinned uv")
    runtime = workflow.index("name: Verify exact checkpoint Python/uv environment")
    sync = workflow.index("name: Install frozen PPO runtime")
    installed = workflow.index("name: Verify frozen environment Python identity")
    transport = workflow.index("name: Run bounded checkpoint transport")
    assert (
        checkout
        < workflow.index("name: Refuse previous checkpoint workflow lineage")
        < cached
        < setup
        < uv_setup
        < runtime
        < sync
        < installed
        < transport
    )

    assert 'python-version: "3.12.16"' in workflow[setup:uv_setup]
    assert "check-latest: false" in workflow[setup:uv_setup]
    assert "RUNNER_TOOL_CACHE" in workflow[cached:setup]
    assert "Python/3.12.16/x64/bin/python" in workflow[cached:setup]
    assert "3.12.16" in workflow[cached:setup]
    assert "sys.version_info[:3] == (3, 12, 16)" in workflow[runtime:sync]
    assert r"grep -Eq '^uv 0\.13\.9([[:space:]]|$)'" in workflow[runtime:sync]
    assert "sys.version_info[:3] == (3, 12, 16)" in workflow[installed:transport]
    assert "uv run --no-sync python" in workflow[installed:transport]
    assert "UV_PYTHON_PREFERENCE: only-system" in workflow
    assert "UV_PYTHON_DOWNLOADS: never" in workflow
    assert "UV_PYTHON_DOWNLOADS: always" not in workflow
    assert 'python-version: "3.12"' not in workflow
    # First use of a new identity must not accept an old source checkout.
    assert "Refuse previous checkpoint workflow lineage" in workflow
    assert "ppo-feature-checkpoint.yml" in workflow[checkout:cached]
    assert "checkout workflow does not contain 3.12.16" in workflow

    assert "CHECKPOINT_CODE_SHA: " + chr(36) + "{{ inputs.code_sha }}" in workflow
    assert "CHECKPOINT_CODE_SHA: }" not in workflow


def test_non_economic_hosted_checkpoint_python_smoke_is_bounded() -> None:
    """PR-only smoke may exercise runtime setup, never actual PPO stages."""
    text = RUNTIME_SMOKE.read_text(encoding="utf-8")
    assert "on:\n  pull_request:" in text
    assert "workflow_dispatch" not in text
    assert "schedule:" not in text and "push:" not in text
    assert "permissions:\n  contents: read" in text
    assert "runs-on: ubuntu-24.04" in text
    assert "RUNNER_TOOL_CACHE" in text
    assert "Python/3.12.16/x64/bin/python" in text
    assert 'python-version: "3.12.16"' in text
    assert "UV_PYTHON_PREFERENCE: only-system" in text
    assert "UV_PYTHON_DOWNLOADS: never" in text
    assert 'version: "0.13.9"' in text
    assert "uv sync --locked --dry-run --extra train-sb3" in text
    assert ("ref: " + chr(36) + "{{ github.event.pull_request.head.sha }}") in text
    assert "ref: }" not in text
    for forbidden in (
        "tools.ppo_checkpoint_actions",
        "CHECKPOINT_STAGE",
        "checkpoint_artifacts_json",
        "CHECKPOINT_GITHUB_TOKEN",
        "upload-artifact",
        "uv run --no-sync python -m",
        "train --",
        "fit --",
    ):
        assert forbidden not in text
