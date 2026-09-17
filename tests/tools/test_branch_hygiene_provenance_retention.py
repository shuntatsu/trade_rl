from tools.branch_hygiene import (
    RETENTION_BRANCH,
    BranchInfo,
    OpenPullRequestRefs,
    anchor_branch_names,
    plan_cleanup,
)


def test_completed_research_refs_are_not_permanent_branch_anchors() -> None:
    main = BranchInfo("main", "a" * 40, False)
    retention = BranchInfo(RETENTION_BRANCH, "b" * 40, False)
    research = BranchInfo("research/completed-study", "c" * 40, False)
    seal = BranchInfo("seal/completed-protocol", "d" * 40, False)
    freeze = BranchInfo("freeze/completed-source", "e" * 40, False)
    run = BranchInfo("run/completed-evaluation", "f" * 40, False)

    anchors = anchor_branch_names(
        (main, retention, research, seal, freeze, run),
        default_branch="main",
        open_pull_request_refs=(),
        active_workflow_branches=frozenset(),
    )

    assert anchors == {"main", RETENTION_BRANCH}


def test_completed_research_refs_enter_retention_then_delete_after_grace() -> None:
    main = BranchInfo("main", "a" * 40, False)
    branches = (
        main,
        BranchInfo("research/completed-study", "b" * 40, False),
        BranchInfo("seal/completed-protocol", "c" * 40, False),
        BranchInfo("freeze/completed-source", "d" * 40, False),
        BranchInfo("run/completed-evaluation", "e" * 40, False),
    )

    first = plan_cleanup(
        branches,
        default_branch="main",
        open_pull_request_refs=(),
        active_workflow_branches=frozenset(),
        reachable_from_anchors=frozenset({main.sha}),
        retention_observed_at={},
        now_timestamp=10_000,
    )
    first_by_name = {decision.branch.name: decision for decision in first}
    for branch in branches[1:]:
        assert first_by_name[branch.name].action == "archive-keep"
        assert first_by_name[branch.name].reason == "first-seen-non-anchor-tip"

    observed = {(branch.name, branch.sha): 10_000 for branch in branches[1:]}
    after_grace = plan_cleanup(
        branches,
        default_branch="main",
        open_pull_request_refs=(),
        active_workflow_branches=frozenset(),
        reachable_from_anchors=frozenset(branch.sha for branch in branches),
        retention_observed_at=observed,
        now_timestamp=13_601,
    )
    after_by_name = {decision.branch.name: decision for decision in after_grace}
    for branch in branches[1:]:
        assert after_by_name[branch.name].action == "delete"
        assert after_by_name[branch.name].reason == "observed-tip-reachable-from-anchor"


def test_open_research_pr_ref_remains_an_anchor() -> None:
    main = BranchInfo("main", "a" * 40, False)
    research = BranchInfo("research/active-study", "b" * 40, False)
    refs = (
        OpenPullRequestRefs(
            head_name=research.name,
            head_sha=research.sha,
            base_name="main",
            base_sha=main.sha,
        ),
    )

    decisions = plan_cleanup(
        (main, research),
        default_branch="main",
        open_pull_request_refs=refs,
        active_workflow_branches=frozenset(),
        reachable_from_anchors=frozenset({main.sha, research.sha}),
        retention_observed_at={(research.name, research.sha): 0},
        now_timestamp=10_000,
    )

    by_name = {decision.branch.name: decision for decision in decisions}
    assert by_name[research.name].action == "keep"
    assert by_name[research.name].reason == "open-pr-ref"
