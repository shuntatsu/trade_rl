from pathlib import Path

from tools.branch_hygiene import (
    BranchInfo,
    OpenPullRequestHead,
    anchor_branch_names,
    anchor_shas,
    plan_cleanup,
)


def test_anchor_selection_preserves_default_protected_open_pr_and_research_refs() -> (
    None
):
    branches = (
        BranchInfo("main", "a" * 40, False),
        BranchInfo("protected/release", "b" * 40, True),
        BranchInfo("feature/open", "c" * 40, False),
        BranchInfo("research/study", "d" * 40, False),
        BranchInfo("seal/protocol", "e" * 40, False),
        BranchInfo("freeze/source", "f" * 40, False),
        BranchInfo("run/evaluation", "1" * 40, False),
        BranchInfo("verify/redundant", "2" * 40, False),
    )
    open_heads = (OpenPullRequestHead("feature/open", "c" * 40),)

    names = anchor_branch_names(
        branches,
        default_branch="main",
        open_pull_request_heads=open_heads,
    )
    shas = anchor_shas(
        branches,
        default_branch="main",
        open_pull_request_heads=open_heads,
    )

    assert names == {
        "main",
        "protected/release",
        "feature/open",
        "research/study",
        "seal/protocol",
        "freeze/source",
        "run/evaluation",
    }
    assert shas == {
        "a" * 40,
        "b" * 40,
        "c" * 40,
        "d" * 40,
        "e" * 40,
        "f" * 40,
        "1" * 40,
    }


def test_cleanup_deletes_only_non_anchor_tips_reachable_from_durable_anchor() -> None:
    branches = (
        BranchInfo("main", "a" * 40, False),
        BranchInfo("feature/open", "b" * 40, False),
        BranchInfo("verify/absorbed", "c" * 40, False),
        BranchInfo("tmp/unique-red", "d" * 40, False),
        BranchInfo("feature/merged", "e" * 40, False),
        BranchInfo("research/evidence", "f" * 40, False),
    )
    decisions = plan_cleanup(
        branches,
        default_branch="main",
        open_pull_request_heads=(OpenPullRequestHead("feature/open", "b" * 40),),
        reachable_from_anchors=frozenset({"a" * 40, "b" * 40, "c" * 40, "e" * 40}),
    )

    by_name = {decision.branch.name: decision for decision in decisions}
    assert by_name["main"].reason == "default-branch"
    assert by_name["feature/open"].reason == "open-pr-head"
    assert by_name["research/evidence"].reason == "research-provenance"
    assert by_name["verify/absorbed"].action == "delete"
    assert by_name["feature/merged"].action == "delete"
    assert by_name["tmp/unique-red"].action == "keep"
    assert by_name["tmp/unique-red"].reason == "unique-unmerged-tip"


def test_cleanup_is_fail_closed_when_reachability_is_not_proven() -> None:
    branch = BranchInfo("automation/orphan", "9" * 40, False)
    decisions = plan_cleanup(
        (BranchInfo("main", "a" * 40, False), branch),
        default_branch="main",
        open_pull_request_heads=(),
        reachable_from_anchors=frozenset({"a" * 40}),
    )

    decision = next(item for item in decisions if item.branch.name == branch.name)
    assert decision.action == "keep"
    assert decision.reason == "unique-unmerged-tip"


def test_branch_hygiene_workflow_never_checks_out_pull_request_head() -> None:
    repository_root = Path(__file__).resolve().parents[2]
    workflow = (repository_root / ".github/workflows/branch-hygiene.yml").read_text(
        encoding="utf-8"
    )

    assert "pull_request_target:" in workflow
    assert "schedule:" in workflow
    assert "workflow_dispatch:" in workflow
    assert "contents: write" in workflow
    assert "pull-requests: read" in workflow
    assert "ref: ${{ github.event.repository.default_branch }}" in workflow
    assert "persist-credentials: false" in workflow
    assert "github.event.pull_request.head.sha" not in workflow
    assert "python3 -m tools.branch_hygiene" in workflow