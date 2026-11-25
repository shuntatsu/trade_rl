from __future__ import annotations

import os
import re
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

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


ORIGINAL_RUNTIME_PROBE = (
    ROOT / ".github" / "workflows" / "ci-original-runtime-probe.yml"
)


def test_original_standalone_probe_is_exact_and_separate_from_execution() -> None:
    current = RUNTIME_SMOKE.read_text(encoding="utf-8")
    text = ORIGINAL_RUNTIME_PROBE.read_text(encoding="utf-8")
    header, original = text.split("\n  original-standalone:", 1)
    original = original.split("\n  immutable-normalization-runtime:", 1)[0]
    assert "original-standalone" not in current
    assert "on:\n  pull_request:" in header
    assert "paths:\n      - .github/workflows/ci-original-runtime-probe.yml\n" in header
    assert "permissions: {}" in header
    for forbidden in (
        "workflow_dispatch:",
        "schedule:",
        "push:",
        "issue_comment:",
        "pull_request_target:",
    ):
        assert forbidden not in header
    assert "UV_PYTHON_DOWNLOADS: never" in current
    assert 'python-version: "3.12.16"' in current
    assert "runs-on: ubuntu-24.04" in original
    assert "timeout-minutes: 5" in original
    assert (
        "20260203/cpython-3.12.12%2B20260203-x86_64-unknown-linux-gnu-install_only_stripped.tar.gz"
        in original
    )
    assert (
        "5164e8ddfca4b0b04a160e11bf411bc453e6c2f8f111f6a9bbd152a1b1a6d7b4" in original
    )
    assert "--max-time 90" in original
    assert "--max-filesize 33892322" in original
    assert '[[ "$(stat -c %s "$archive")" == 33892322 ]]' in original
    assert original.index("sha256sum --check --strict") < original.index("tar -xzf")
    assert original.index("tar -xzf") < original.index("/python/bin/python3.12")
    assert "sys.version_info[:3] == (3, 12, 12)" in original
    assert "sys.implementation.name == 'cpython'" in original
    assert "uses:" not in original
    assert "uv sync" not in original
    assert "github.event" not in original
    assert "secrets." not in original
    assert "|| true" not in original


@pytest.mark.skipif(sys.platform != "linux", reason="Ubuntu runner uses GNU coreutils")
@pytest.mark.parametrize("failure", ["http", "size", "digest"])
def test_original_standalone_probe_fails_before_extracting_unverified_bytes(
    tmp_path: Path, failure: str
) -> None:
    text = ORIGINAL_RUNTIME_PROBE.read_text(encoding="utf-8")
    job = text.split("\n  original-standalone:", 1)[1]
    job = job.split("\n  immutable-normalization-runtime:", 1)[0]
    script = textwrap.dedent(job.split("        run: |\n", 1)[1])
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    marker = tmp_path / "extracted"
    curl = bin_dir / "curl"
    curl.write_text(
        "#!/bin/bash\nset -eu\n"
        'while [[ "$1" != --output ]]; do shift; done\n'
        'case "$FAILURE" in\n'
        "  http) exit 22 ;;\n"
        '  size) printf x > "$2" ;;\n'
        '  digest) truncate -s 33892322 "$2" ;;\n'
        "esac\n",
        encoding="utf-8",
    )
    tar = bin_dir / "tar"
    tar.write_text('#!/bin/bash\ntouch "$EXTRACTED"\nexit 99\n', encoding="utf-8")
    for command in (curl, tar):
        command.chmod(0o700)
    env = {
        **os.environ,
        "PATH": str(bin_dir) + os.pathsep + os.environ["PATH"],
        "TMPDIR": str(tmp_path),
        "LC_ALL": "C",
        "FAILURE": failure,
        "EXTRACTED": str(marker),
    }
    result = subprocess.run(
        ["bash", "-c", script], env=env, capture_output=True, timeout=10, check=False
    )
    assert result.returncode != 0
    assert not marker.exists()
    assert set(tmp_path.iterdir()) == {bin_dir}
    if failure == "digest":
        assert b"FAILED" in result.stdout


def test_original_runtime_probe_checks_managed_frozen_source_without_execution() -> (
    None
):
    text = ORIGINAL_RUNTIME_PROBE.read_text(encoding="utf-8")
    job = text.split("\n  immutable-normalization-runtime:", 1)[1]
    assert "needs: original-standalone" in job
    assert "permissions:\n      contents: read" in job
    assert "runs-on: ubuntu-24.04" in job
    assert "timeout-minutes: 15" in job
    assert "ref: 72ec5a082a1e298ccfe904f06c16d423e12a76b2" in job
    assert "path: target" in job
    assert "persist-credentials: false" in job
    assert "73ee39ebd561913ed8fe7ac0b8017f039bf3af3e" in job
    assert "0044c6bd0e95868e092fc52eaad5b9e0e821cb2db87a5b6f4fbaefbe1b981cf8" in job
    assert "6440b63c463035d0c523869a2bbc62fac6c96f9ac0585a32b77f16bf5ea176f6" in job
    assert "uses: astral-sh/setup-uv@1db35dd8eee11c5894872e0b383ae6c406cf738d" in job
    assert 'python-version: "3.12.12"' in job
    assert "enable-cache: false" in job
    assert "working-directory: target" in job
    assert 'version: "0.10.0"' not in job  # must resolve from frozen source
    assert "uv sync --locked --extra dev --extra train-sb3" in job
    assert "20341168c41f91f4328c68d1f1a8d563f5734d515e617ff80a8dcfce323218b7" in job
    assert "2ccb92133add99f4981f0f9e6b38fb58b205bd97da6f8c0f7646c47ccf0b1461" in job
    assert "sys._base_executable" in job and "sys.base_prefix" in job
    assert "uv python find --system --managed-python 3.12.12" in job
    assert "uv run --no-sync python -I -m pytest" in job
    assert '--junitxml="$report"' in job
    assert 'suite.attrib["tests"] == "10"' in job
    assert '("failures", "errors", "skipped")' in job
    assert "test_prior_receipt_binds_checkout_and_workflow_identity" in job
    assert "test_activation_rejects_runtime_drift" in job
    assert "test_execution_is_fail_closed_without_sealed_activation" in job
    assert job.index("Verify frozen source and locks") < job.index(
        "Set up uv from frozen source"
    )
    assert job.index("sha256sum --check --strict") < job.index("uv sync --locked")
    assert job.index(
        "20341168c41f91f4328c68d1f1a8d563f5734d515e617ff80a8dcfce323218b7"
    ) < job.index("uv run --no-sync python -I -m pytest")
    for forbidden in (
        "secrets.",
        "contents: write",
        "actions: write",
        "workflow_dispatch:",
        "upload-artifact",
        "continue-on-error:",
        "|| true",
        "tools.ppo_normalization_actions",
        "tools.ppo_checkpoint_actions",
    ):
        assert forbidden not in job
    assert job.count("uses:") == 2
    special = (
        ROOT / ".github" / "workflows" / "ppo-normalization-execution.yml"
    ).read_text(encoding="utf-8")
    assert (
        special.count(
            "uses: astral-sh/setup-uv@1db35dd8eee11c5894872e0b383ae6c406cf738d"
        )
        == 3
    )
    assert special.count('python-version: "3.12.12"') == 3
    assert (
        'IMPLEMENTATION_CODE_SHA: Final = "72ec5a082a1e298ccfe904f06c16d423e12a76b2"'
        in (ROOT / "tools" / "ppo_normalization_actions.py").read_text(encoding="utf-8")
    )
