"""Keep exactly one current exact-SHA retirement workflow."""

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
