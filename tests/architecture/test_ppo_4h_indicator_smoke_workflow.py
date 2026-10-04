from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"
TRIGGER = "run: execute 4h PPO indicator smoke"
BRANCH = "refs/heads/research/ppo-4h-indicator-smoke"


def test_smoke_trigger_is_scoped_inside_permanent_fast_push_workflow() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    assert "PPO 4h Indicator Smoke" not in text
    assert BRANCH in text
    assert TRIGGER in text
    assert "pull_request_target:" not in text
    assert "actions: read" in text
    assert "contents: read" in text
    assert "issues: read" in text
    assert "pull-requests: read" in text
    assert "contents: write" not in text
    assert "actions: write" not in text


def test_smoke_trigger_uses_pinned_runtime_and_uploads_evidence() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    assert "uv sync --locked --extra train-sb3" in text
    assert "tools.ppo_4h_indicator_smoke_actions" in text
    assert "actions/upload-artifact@043fb46d1a93c77aae656e7c1c64a875d1fc6a0a" in text
    assert "output/smoke" in text
    assert "always()" in text
    assert "fetch-depth: 2" in text


def test_independent_review_status_runs_only_after_full_verification() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    generic_start = text.index("  independent-review:\n")
    smoke_start = text.index("  ppo-4h-independent-review:\n")
    generic = text[generic_start:smoke_start]
    smoke = text[smoke_start:]

    assert "pull_request_review:" in text
    assert "submitted" in text
    assert "edited" in text
    assert "dismissed" in text
    assert "name: Independent Research Review" in smoke
    assert "needs: [core, ppo-runtime, guide]" in smoke
    assert "CORE_RESULT: ${{ needs.core.result }}" in smoke
    assert "PPO_RESULT: ${{ needs.ppo-runtime.result }}" in smoke
    assert "GUIDE_RESULT: ${{ needs.guide.result }}" in smoke
    assert 'test "$CORE_RESULT" = "success"' in smoke
    assert 'test "$PPO_RESULT" = "success"' in smoke
    assert 'test "$GUIDE_RESULT" = "success"' in smoke
    assert (
        "uv run python -m tools.ppo_4h_indicator_smoke_actions review-status" in smoke
    )
    assert "gh api --paginate" not in smoke
    assert "independent-research-review-audit" not in smoke
    assert (
        "github.event.pull_request.head.ref == 'research/ppo-4h-indicator-smoke'"
        in smoke
    )
    assert (
        "github.event.pull_request.head.ref != 'research/ppo-4h-indicator-smoke'"
        in generic
    )
    assert "name: Generic Independent Research Review" in generic
    assert "name: Independent Research Review" in smoke
    assert "gh api --paginate" in generic
    assert "permissions:\n      pull-requests: read" in generic
    assert "tools/independent_research_review.py" in generic
    assert '--expected-head "$HEAD_SHA"' in generic
    assert "env -u GITHUB_TOKEN -u GH_TOKEN uv run" in generic
    assert 'contains("### Disposition: APPROVED")' not in generic
    assert 'contains("### Disposition: BLOCKED")' not in generic

    review_evaluator = generic.split(
        "      - name: Evaluate exact-head independent research review", 1
    )[1].split("  ppo-4h-independent-review:", 1)[0]
    assert "GITHUB_TOKEN: ${{" not in review_evaluator
    assert "GH_TOKEN: ${{" not in review_evaluator


def test_review_events_repeat_full_software_verification_before_status() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    review_event_guard = "github.event_name == 'pull_request_review'"

    assert text.count(review_event_guard) >= 4


def test_generic_review_guard_is_type_checked_by_repository_ci() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    assert (
        text.count(
            "uv run mypy tools/agent_repo tests/architecture/distribution.py "
            "tools/independent_research_review.py"
        )
        == 2
    )
