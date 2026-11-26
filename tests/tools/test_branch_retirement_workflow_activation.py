"""Keep exactly one current exact-SHA retirement workflow."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = ROOT / ".github" / "workflows"


def test_only_active_retirement_workflow_is_shipped_with_exact_sha_gate() -> None:
    legacy = WORKFLOWS / "branch-hygiene.yml"
    replacement = WORKFLOWS / "retire-merged-pr-branches.yml"

    assert not legacy.exists()
    assert replacement.is_file()

    content = replacement.read_text(encoding="utf-8")
    assert "push:" in content
    assert "- main" in content
    assert "schedule:" in content
    assert "workflow_dispatch:" in content
    assert "contents: write" in content
    assert "group: branch-hygiene" in content
    assert "if: github.repository == 'shuntatsu/trade_rl'" in content
    assert "--retire-merged --apply --max-deletes 100" in content


def test_retired_workflow_docs_match_checked_in_tree() -> None:
    """Both obsolete workflows are absent and the active source is documented."""
    assert not (WORKFLOWS / "prune-merged-branch.yml").exists()
    assert not (WORKFLOWS / "branch-hygiene.yml").exists()
    policy = (ROOT / "AGENTS.md").read_text(encoding="utf-8")
    assert "旧手動`prune-merged-branch.yml`はソースから廃止" in policy
    assert "`branch-hygiene.yml`もtracked treeから除去" in policy
    assert "`.github/workflows/retire-merged-pr-branches.yml`" in policy


def test_retirement_has_no_expired_owner_command_surface() -> None:
    """Historical one-shot exceptions must not be executable in current source."""
    content = (WORKFLOWS / "retire-merged-pr-branches.yml").read_text(encoding="utf-8")
    events, jobs = content.split("\njobs:\n", maxsplit=1)
    assert re.findall(
        r"^  ([a-z_]+):",
        events.split("\non:\n", 1)[1].split("\npermissions:", 1)[0],
        re.M,
    ) == [
        "push",
        "schedule",
        "workflow_dispatch",
    ]
    assert re.findall(r"^  ([a-z][a-z0-9-]*):", jobs, re.M) == ["cleanup"]
    assert "issue_comment" not in content
    assert "github.event.comment" not in content
    assert "--retire-merged --apply --max-deletes 100" in jobs
    assert not list((ROOT / "tools").glob("archived_*retirement*.py"))
    assert not (ROOT / "tools/archived_obsolete_20261121.py").exists()


def test_automatic_retirement_cannot_apply_consolidated_sources() -> None:
    content = (WORKFLOWS / "retire-merged-pr-branches.yml").read_text(encoding="utf-8")
    preview, apply = content.split("- name: Prune main ancestors", maxsplit=1)
    assert "--include-consolidated" in preview
    assert "--include-consolidated" not in apply
    assert "--consolidated-ref" not in apply
