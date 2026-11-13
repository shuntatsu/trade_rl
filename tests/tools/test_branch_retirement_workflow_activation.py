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
