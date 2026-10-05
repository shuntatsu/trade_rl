from __future__ import annotations

import ast
import fnmatch
import os
import re
import shutil
import subprocess
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"
TRIGGER = "run: execute 4h PPO indicator smoke"
BRANCH = "refs/heads/research/ppo-4h-indicator-smoke"


def _job(name: str) -> str:
    block = WORKFLOW.read_text(encoding="utf-8").split(f"\n  {name}:\n", 1)[1]
    return re.split(r"\n  [\w-]+:\n", block, maxsplit=1)[0]


def _routes(job: str, event: str, action: str, base: str, changed: bool, head: str):
    text = WORKFLOW.read_text(encoding="utf-8")
    if event == "pull_request":
        trigger = text.split("  pull_request:\n", 1)[1].split(
            "  pull_request_review:", 1
        )[0]
        types = re.search(r"types: \[([^]]+)\]", trigger)
        actions = (
            types[1].replace(" ", "").split(",")
            if types
            else ["opened", "synchronize", "reopened"]
        )
        patterns = re.findall(r"^      - ['\"]?([^'\"\n]+)", trigger, re.M)
        if action not in actions or not any(
            fnmatch.fnmatchcase(base, p) for p in patterns
        ):
            return False
    expression = _job(job).split("    if:", 1)[1].split("    runs-on:", 1)[0]
    return _evaluate(expression, event, action, base, changed, head)


def _evaluate(expression, event, action, base, changed, head):
    expression = expression.strip().removeprefix(">-").strip()
    expression = " ".join(expression.splitlines())
    for field, value in (
        ("github.event_name", event),
        ("github.event.action", action),
        ("github.event.pull_request.base.ref", base),
        ("github.event.pull_request.head.ref", head),
        ("github.event.changes.base", changed),
        ("github.workflow", "CI"),
        ("github.ref", "refs/heads/main" if event == "push" else "refs/pull/42/merge"),
    ):
        expression = expression.replace(field, repr(value))
    expression = re.sub(r"!(?!=)", "not ", expression)
    tree = ast.parse(expression.replace("&&", "and").replace("||", "or"), mode="eval")
    allowed = (
        ast.Expression,
        ast.BoolOp,
        ast.And,
        ast.Or,
        ast.UnaryOp,
        ast.Not,
        ast.Compare,
        ast.Eq,
        ast.NotEq,
        ast.Call,
        ast.Name,
        ast.Load,
        ast.Constant,
    )
    assert all(isinstance(n, allowed) for n in ast.walk(tree))
    assert all(
        n.id in {"always", "startsWith"}
        for n in ast.walk(tree)
        if isinstance(n, ast.Name)
    )
    return eval(
        compile(tree, "<actual CI if>", "eval"),
        {"__builtins__": {}},
        {"always": lambda: True, "startsWith": str.startswith},
    )


@pytest.mark.parametrize(
    "event,action,base,changed,expected",
    [
        ("pull_request", "opened", "main", False, True),
        ("pull_request", "opened", "codex/topic", False, True),
        ("pull_request", "synchronize", "codex/topic", False, True),
        ("pull_request", "reopened", "codex/topic", False, True),
        ("pull_request", "edited", "codex/parent", True, True),
        ("pull_request", "edited", "main", True, True),
        ("pull_request", "edited", "main", False, False),
        ("pull_request", "edited", "codex/parent", False, False),
        ("pull_request_review", "edited", "codex/topic", False, True),
        ("pull_request_review", "submitted", "main", False, True),
        ("pull_request_review", "dismissed", "codex/topic", False, True),
        ("pull_request_review", "submitted", "codex-lookalike", False, False),
        ("pull_request_review", "submitted", "release/topic", False, False),
        ("pull_request", "opened", "release/topic", False, False),
        ("push", "", "main", False, False),
    ],
)
@pytest.mark.parametrize("smoke", [False, True])
def test_actual_ci_event_conditions_route_the_same_full_jobs(
    event, action, base, changed, expected, smoke
):
    head = "research/ppo-4h-indicator-smoke" if smoke else "codex/child"
    for job in ("core", "ppo-runtime", "guide"):
        assert bool(_routes(job, event, action, base, changed, head)) is expected
    gate_expected = expected or event == "pull_request_review"
    assert bool(_routes("independent-review", event, action, base, changed, head)) is (
        gate_expected and not smoke
    )
    assert bool(
        _routes("ppo-4h-independent-review", event, action, base, changed, head)
    ) is (gate_expected and smoke)

    def render(template):
        return re.sub(
            r"\$\{\{(.*?)\}\}",
            lambda match: str(_evaluate(match[1], event, action, base, changed, head)),
            template,
        )

    metadata = event == "pull_request" and action == "edited" and not changed
    group = re.search(r"^  group: (.+)$", WORKFLOW.read_text(), re.M)[1]
    ref = "refs/heads/main" if event == "push" else "refs/pull/42/merge"
    names = {
        "core": "Lean Core",
        "ppo-runtime": "PPO Runtime",
        "guide": "Human Guide",
        "independent-review": "Generic Independent Research Review",
        "ppo-4h-independent-review": "Independent Research Review",
    }
    actual = [render(group)] + [
        render(re.search(r"^    name: (.+)$", _job(job), re.M)[1]) for job in names
    ]
    expected_values = [f"ci-CI-{ref}" + ("-metadata" if metadata else "")] + [
        name + (" (metadata-only)" if metadata else "") for name in names.values()
    ]
    assert actual == expected_values


@pytest.mark.parametrize("gate", ["independent-review", "ppo-4h-independent-review"])
@pytest.mark.parametrize("job", ["CORE_RESULT", "PPO_RESULT", "GUIDE_RESULT"])
@pytest.mark.parametrize(
    "state", ["", "skipped", "pending", "failure", "cancelled", "success"]
)
def test_actual_comprehensive_scripts_require_three_successes(gate, job, state):
    block = _job(gate).split("      - name: Require comprehensive verification", 1)[1]
    script = textwrap.dedent(
        block.split("        run: |\n", 1)[1].split("      - name:", 1)[0]
    )
    bash = (
        str(
            Path(os.environ.get("ProgramFiles", "C:/Program Files"))
            / "Git/bin/bash.exe"
        )
        if os.name == "nt"
        else shutil.which("bash")
    )
    assert bash is not None
    results = dict.fromkeys(("CORE_RESULT", "PPO_RESULT", "GUIDE_RESULT"), "success")
    results[job] = state
    completed = subprocess.run(
        [bash, "-c", script],
        env={**os.environ, **results},
        capture_output=True,
        text=True,
        check=False,
    )
    assert (completed.returncode == 0) is (state == "success")


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
