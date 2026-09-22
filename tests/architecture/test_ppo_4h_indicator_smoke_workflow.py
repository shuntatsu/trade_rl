from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github" / "workflows" / "ppo-4h-indicator-smoke.yml"


def test_smoke_workflow_is_branch_scoped_and_exact_trigger_only() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    assert "push:" in text
    assert "research/ppo-4h-indicator-smoke" in text
    assert "workflow_dispatch:" not in text
    assert "schedule:" not in text
    assert "pull_request_target:" not in text
    assert (
        "github.event.head_commit.message == 'run: execute 4h PPO indicator smoke'"
        in text
    )
    assert "actions: read" in text
    assert "contents: read" in text
    assert "issues: read" in text
    assert "pull-requests: read" in text
    assert "contents: write" not in text
    assert "actions: write" not in text


def test_smoke_workflow_uses_pinned_runtime_and_uploads_evidence() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    assert "runs-on: ubuntu-24.04" in text
    assert "timeout-minutes: 340" in text
    assert "actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1" in text
    assert "astral-sh/setup-uv@bec219d24cd3e171d82865faccec33120bb574f4" in text
    assert "actions/upload-artifact@043fb46d1a93c77aae656e7c1c64a875d1fc6a0a" in text
    assert 'version: "0.10.0"' in text
    assert 'python-version: "3.12"' in text
    assert "uv sync --locked --extra train-sb3" in text
    assert "tools.ppo_4h_indicator_smoke_actions" in text
    assert "if: always()" in text
    assert "output/smoke" in text
