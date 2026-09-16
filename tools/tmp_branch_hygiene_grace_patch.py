from pathlib import Path


source_path = Path("tools/branch_hygiene.py")
source = source_path.read_text(encoding="utf-8")
source = source.replace("import sys\n", "import sys\nimport time\n", 1)
source = source.replace(
    "from typing import Any, Iterable, Sequence\n",
    "from typing import Any, Iterable, Mapping, Sequence\n",
    1,
)
source = source.replace(
    'DELETE_BATCH_SIZE = 25\nAPI_VERSION = "2022-11-28"\n',
    'DELETE_BATCH_SIZE = 25\nRETENTION_GRACE_SECONDS = 60 * 60\nRETENTION_MESSAGE_PREFIXES = (\n'
    '    "Archive unique non-anchor branch tips before ref cleanup",\n'
    '    "Observe non-anchor branch tips before cleanup grace period",\n'
    ')\nAPI_VERSION = "2022-11-28"\n',
    1,
)

insert = r'''

def parse_retention_log(log_output: str) -> dict[tuple[str, str], int]:
    observed: dict[tuple[str, str], int] = {}
    for raw_record in log_output.split("\x1e"):
        record = raw_record.strip("\n")
        if not record.strip():
            continue
        timestamp_text, separator, message = record.partition("\x1f")
        if not separator:
            raise RuntimeError("retention log record is missing its separator")
        try:
            timestamp = int(timestamp_text.strip())
        except ValueError as exc:
            raise RuntimeError("retention log timestamp is invalid") from exc
        if not message.lstrip().startswith(RETENTION_MESSAGE_PREFIXES):
            continue
        for line in message.splitlines():
            branch_name, mapping_separator, sha = line.partition("\t")
            if not mapping_separator:
                continue
            if (
                not branch_name
                or len(sha) != 40
                or any(character not in "0123456789abcdef" for character in sha)
            ):
                continue
            key = (branch_name, sha)
            previous = observed.get(key)
            observed[key] = timestamp if previous is None else min(previous, timestamp)
    return observed


def retention_tip_observations() -> dict[tuple[str, str], int]:
    ref = f"refs/remotes/origin/{RETENTION_BRANCH}"
    probe = subprocess.run(
        ["git", "show-ref", "--verify", "--quiet", ref],
        text=True,
        capture_output=True,
        check=False,
    )
    if probe.returncode == 1:
        return {}
    if probe.returncode != 0:
        raise RuntimeError(f"git show-ref failed: {probe.stderr.strip()}")
    process = subprocess.run(
        ["git", "log", "--first-parent", "--format=%ct%x1f%B%x1e", ref],
        text=True,
        capture_output=True,
        check=False,
    )
    if process.returncode != 0:
        raise RuntimeError(f"git log failed: {process.stderr.strip()}")
    return parse_retention_log(process.stdout)
'''
marker = "\ndef plan_cleanup(\n"
if source.count(marker) != 1:
    raise SystemExit("plan_cleanup insertion marker mismatch")
source = source.replace(marker, insert + marker, 1)

old_signature = '''def plan_cleanup(
    branches: Sequence[BranchInfo],
    *,
    default_branch: str,
    open_pull_request_refs: Sequence[OpenPullRequestRefs],
    active_workflow_branches: frozenset[str],
    reachable_from_anchors: frozenset[str],
) -> tuple[BranchDecision, ...]:
'''
new_signature = '''def plan_cleanup(
    branches: Sequence[BranchInfo],
    *,
    default_branch: str,
    open_pull_request_refs: Sequence[OpenPullRequestRefs],
    active_workflow_branches: frozenset[str],
    reachable_from_anchors: frozenset[str],
    retention_observed_at: Mapping[tuple[str, str], int],
    now_timestamp: int,
    grace_seconds: int = RETENTION_GRACE_SECONDS,
) -> tuple[BranchDecision, ...]:
'''
if source.count(old_signature) != 1:
    raise SystemExit("plan_cleanup signature target mismatch")
source = source.replace(old_signature, new_signature, 1)

old_decision = '''        elif is_preserved_branch(branch.name):
            decision = BranchDecision(branch, "keep", "provenance")
        elif branch.sha in reachable_from_anchors:
            decision = BranchDecision(branch, "delete", "tip-reachable-from-anchor")
        else:
            decision = BranchDecision(branch, "archive-delete", "unique-non-anchor-tip")
'''
new_decision = '''        elif is_preserved_branch(branch.name):
            decision = BranchDecision(branch, "keep", "provenance")
        else:
            observed_at = retention_observed_at.get((branch.name, branch.sha))
            if observed_at is None:
                decision = BranchDecision(
                    branch, "archive-keep", "first-seen-non-anchor-tip"
                )
            elif now_timestamp - observed_at < grace_seconds:
                decision = BranchDecision(branch, "keep", "observation-grace-period")
            elif branch.sha in reachable_from_anchors:
                decision = BranchDecision(
                    branch, "delete", "observed-tip-reachable-from-anchor"
                )
            else:
                decision = BranchDecision(branch, "keep", "retained-tip-not-reachable")
'''
if source.count(old_decision) != 1:
    raise SystemExit("plan_cleanup decision target mismatch")
source = source.replace(old_decision, new_decision, 1)

old_message = '''        "Archive unique non-anchor branch tips before ref cleanup",
        "",
        "Each mapping below preserves the deleted remote branch name and exact tip SHA.",
        "Restore with: git branch <name> <sha>",
'''
new_message = '''        "Observe non-anchor branch tips before cleanup grace period",
        "",
        "Each mapping below records the remote branch name and exact observed tip SHA.",
        "Cleanup requires the same observed tip to survive the grace period.",
        "Restore with: git branch <name> <sha>",
'''
if source.count(old_message) != 1:
    raise SystemExit("retention message target mismatch")
source = source.replace(old_message, new_message, 1)
source = source.replace("def archive_unique_tips(\n", "def archive_observed_tips(\n", 1)
source = source.replace(
    "    retention_sha = archive_unique_tips(\n",
    "    retention_sha = archive_observed_tips(\n",
    1,
)
source = source.replace(
    '        if decision.action != "archive-delete":\n',
    '        if decision.action != "archive-keep":\n',
    1,
)

old_delete_block = '''    archived_names = {branch.name for branch in archive_candidates}
    delete_candidates: list[BranchInfo] = []
    for decision in decisions:
        if decision.action not in {"delete", "archive-delete"}:
            continue
        expected = decision.branch
        if decision.action == "archive-delete" and expected.name not in archived_names:
            continue
        current = current_by_name.get(expected.name)
'''
new_delete_block = '''    delete_candidates: list[BranchInfo] = []
    for decision in decisions:
        if decision.action != "delete":
            continue
        expected = decision.branch
        current = current_by_name.get(expected.name)
'''
if source.count(old_delete_block) != 1:
    raise SystemExit("delete candidate block target mismatch")
source = source.replace(old_delete_block, new_delete_block, 1)
source = source.replace(
    '        item.branch.name for item in decisions if item.action == "archive-delete"\n',
    '        item.branch.name for item in decisions if item.action == "archive-keep"\n',
    1,
)
source = source.replace(
    '        f"- archive-then-delete candidates: {len(archive_candidates)}",\n',
    '        f"- first-observation branches archived and kept: {len(archive_candidates)}",\n'
    '        f"- grace period: {RETENTION_GRACE_SECONDS} seconds",\n',
    1,
)
source = source.replace(
    '            "Delete redundant remote branches and archive unique non-anchor tips "\n'
    '            "before deleting their refs."\n',
    '            "Observe non-anchor branch tips first, preserve them in retention history, "\n'
    '            "and delete only unchanged tips that survive the grace period."\n',
    1,
)
old_main = '''    reachable = reachable_commits(anchors)
    decisions = plan_cleanup(
        branches,
        default_branch=default_branch,
        open_pull_request_refs=open_refs,
        active_workflow_branches=active_branches,
        reachable_from_anchors=reachable,
    )
'''
new_main = '''    reachable = reachable_commits(anchors)
    retention_observed_at = retention_tip_observations()
    decisions = plan_cleanup(
        branches,
        default_branch=default_branch,
        open_pull_request_refs=open_refs,
        active_workflow_branches=active_branches,
        reachable_from_anchors=reachable,
        retention_observed_at=retention_observed_at,
        now_timestamp=int(time.time()),
    )
'''
if source.count(old_main) != 1:
    raise SystemExit("main planner target mismatch")
source = source.replace(old_main, new_main, 1)
source_path.write_text(source, encoding="utf-8")


test_path = Path("tests/tools/test_branch_hygiene.py")
tests = test_path.read_text(encoding="utf-8")
tests = tests.replace(
    "    apply_cleanup,\n    plan_cleanup,\n",
    "    apply_cleanup,\n    parse_retention_log,\n    plan_cleanup,\n",
    1,
)
tests = tests.replace(
    '''        reachable_from_anchors=frozenset(
            {"a" * 40, "b" * 40, "c" * 40, "d" * 40, "2" * 40}
        ),
    )
''',
    '''        reachable_from_anchors=frozenset(
            {"a" * 40, "b" * 40, "c" * 40, "d" * 40, "2" * 40}
        ),
        retention_observed_at={
            ("verify/absorbed", "d" * 40): 0,
            ("feature/merged", "2" * 40): 0,
        },
        now_timestamp=10_000,
    )
''',
    1,
)
tests = tests.replace(
    '''    assert by_name["tmp/unique-red"].action == "archive-delete"
    assert by_name["automation/unique"].action == "archive-delete"
    assert by_name["tmp/unique-red"].reason == "unique-non-anchor-tip"
    assert by_name["automation/unique"].reason == "unique-non-anchor-tip"
    assert by_name["feature/unique"].action == "archive-delete"
    assert by_name["feature/unique"].reason == "unique-non-anchor-tip"
''',
    '''    assert by_name["tmp/unique-red"].action == "archive-keep"
    assert by_name["automation/unique"].action == "archive-keep"
    assert by_name["tmp/unique-red"].reason == "first-seen-non-anchor-tip"
    assert by_name["automation/unique"].reason == "first-seen-non-anchor-tip"
    assert by_name["feature/unique"].action == "archive-keep"
    assert by_name["feature/unique"].reason == "first-seen-non-anchor-tip"
''',
    1,
)
tests = tests.replace(
    '''        reachable_from_anchors=frozenset({"a" * 40, "b" * 40, "c" * 40}),
    )
''',
    '''        reachable_from_anchors=frozenset({"a" * 40, "b" * 40, "c" * 40}),
        retention_observed_at={},
        now_timestamp=10_000,
    )
''',
    1,
)
tests = tests.replace(
    '        BranchDecision(archived, "archive-delete", "unique-non-anchor-tip"),\n',
    '        BranchDecision(archived, "archive-keep", "first-seen-non-anchor-tip"),\n',
    1,
)
tests = tests.replace(
    "    assert set(deleted) == {archived.name, direct.name}\n",
    "    assert deleted == (direct.name,)\n",
    1,
)
tests = tests.replace(
    '''    delete_event = next(
        index
        for index, event in enumerate(api.events)
        if event.startswith("delete:") and archived.name in event
    )
    assert archive_event < delete_event
''',
    '''    delete_event = next(
        index for index, event in enumerate(api.events) if event.startswith("delete:")
    )
    assert archive_event < delete_event
    assert archived.name in {branch.name for branch in api.branches()}
''',
    1,
)

extra = r'''


def test_cleanup_requires_observation_and_grace_before_delete() -> None:
    main = BranchInfo("main", "a" * 40, False)
    first_seen = BranchInfo("feature/first", "b" * 40, False)
    recent = BranchInfo("feature/recent", "c" * 40, False)
    stale = BranchInfo("feature/stale", "d" * 40, False)
    unreachable = BranchInfo("feature/unreachable", "e" * 40, False)
    now = 10_000
    decisions = plan_cleanup(
        (main, first_seen, recent, stale, unreachable),
        default_branch="main",
        open_pull_request_refs=(),
        active_workflow_branches=frozenset(),
        reachable_from_anchors=frozenset({main.sha, recent.sha, stale.sha}),
        retention_observed_at={
            (recent.name, recent.sha): now - 30,
            (stale.name, stale.sha): now - 3_601,
            (unreachable.name, unreachable.sha): now - 3_601,
        },
        now_timestamp=now,
    )

    by_name = {decision.branch.name: decision for decision in decisions}
    assert by_name[first_seen.name].action == "archive-keep"
    assert by_name[first_seen.name].reason == "first-seen-non-anchor-tip"
    assert by_name[recent.name].action == "keep"
    assert by_name[recent.name].reason == "observation-grace-period"
    assert by_name[stale.name].action == "delete"
    assert by_name[stale.name].reason == "observed-tip-reachable-from-anchor"
    assert by_name[unreachable.name].action == "keep"
    assert by_name[unreachable.name].reason == "retained-tip-not-reachable"


def test_retention_log_parser_uses_earliest_observation() -> None:
    branch = "automation/recent"
    sha = "a" * 40
    old_message = (
        "Archive unique non-anchor branch tips before ref cleanup\n\n"
        f"{branch}\t{sha}\n"
    )
    new_message = (
        "Observe non-anchor branch tips before cleanup grace period\n\n"
        f"{branch}\t{sha}\n"
    )
    payload = (
        f"200\x1f{new_message}\x1e"
        f"100\x1f{old_message}\x1e"
        "50\x1funrelated commit\x1e"
    )

    assert parse_retention_log(payload) == {(branch, sha): 100}
'''
if "def test_cleanup_requires_observation_and_grace_before_delete" not in tests:
    tests += extra
test_path.write_text(tests, encoding="utf-8")


agents_path = Path("AGENTS.md")
agents = agents_path.read_text(encoding="utf-8")
old = "`.github/workflows/branch-hygiene.yml` は、anchorからtip commitへ到達できる非anchor branchを直接削除してよい。anchorに含まれない固有tipはprefixに依存せず、そのexact tip SHAと元branch名を `provenance/branch-retention` のmerge-historyとcommit messageへ先に保存し、retention refの更新をread-backしてから元refだけを削除する。"
new = "`.github/workflows/branch-hygiene.yml` は、非anchor branchを初めて観測したrunでは、そのexact tip SHAと元branch名を `provenance/branch-retention` のmerge-historyとcommit messageへ保存するだけで元refを残す。同じbranch名・同じtip SHAが少なくとも1時間retentionで観測済みで、その間にopen PR / active workflow / protected / research provenance anchorにならなかった場合に限り元refを削除してよい。tipが変われば新しい観測として猶予をやり直す。"
if agents.count(old) != 1:
    raise SystemExit("AGENTS hygiene policy target mismatch")
agents_path.write_text(agents.replace(old, new), encoding="utf-8")
