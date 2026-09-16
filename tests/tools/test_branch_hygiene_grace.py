from tools.branch_hygiene import BranchInfo, plan_cleanup


def test_first_seen_branch_at_main_tip_is_observed_before_deletion() -> None:
    shared_sha = "a" * 40
    main = BranchInfo("main", shared_sha, False)
    newly_created = BranchInfo("feature/new", shared_sha, False)

    decisions = plan_cleanup(
        (main, newly_created),
        default_branch="main",
        open_pull_request_refs=(),
        active_workflow_branches=frozenset(),
        reachable_from_anchors=frozenset({shared_sha}),
        retention_observed_at={},
        now_timestamp=10_000,
    )

    by_name = {decision.branch.name: decision for decision in decisions}
    assert by_name[newly_created.name].action == "archive-keep"
    assert by_name[newly_created.name].reason == "first-seen-non-anchor-tip"


def test_tip_change_restarts_observation_grace() -> None:
    main = BranchInfo("main", "a" * 40, False)
    changed = BranchInfo("automation/work", "c" * 40, False)

    decisions = plan_cleanup(
        (main, changed),
        default_branch="main",
        open_pull_request_refs=(),
        active_workflow_branches=frozenset(),
        reachable_from_anchors=frozenset({main.sha, changed.sha}),
        retention_observed_at={(changed.name, "b" * 40): 1},
        now_timestamp=10_000,
    )

    decision = next(item for item in decisions if item.branch.name == changed.name)
    assert decision.action == "archive-keep"
    assert decision.reason == "first-seen-non-anchor-tip"


def test_exact_grace_boundary_allows_cleanup() -> None:
    main = BranchInfo("main", "a" * 40, False)
    observed = BranchInfo("tmp/old", "b" * 40, False)
    now = 10_000

    decisions = plan_cleanup(
        (main, observed),
        default_branch="main",
        open_pull_request_refs=(),
        active_workflow_branches=frozenset(),
        reachable_from_anchors=frozenset({main.sha, observed.sha}),
        retention_observed_at={(observed.name, observed.sha): now - 3_600},
        now_timestamp=now,
    )

    decision = next(item for item in decisions if item.branch.name == observed.name)
    assert decision.action == "delete"
    assert decision.reason == "observed-tip-reachable-from-anchor"
