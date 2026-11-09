"""Check the newly registered retirement workflow without mutating GitHub."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = ROOT / ".github" / "workflows"


def test_replacement_retirement_workflow_has_fresh_path_and_same_gate() -> None:
    legacy = WORKFLOWS / "branch-hygiene.yml"
    replacement = WORKFLOWS / "retire-merged-pr-branches.yml"

    assert legacy.is_file()
    assert replacement.is_file()
    assert legacy.read_bytes() == replacement.read_bytes()

    content = replacement.read_text(encoding="utf-8")
    assert "push:" in content
    assert "- main" in content
    assert "schedule:" in content
    assert "workflow_dispatch:" in content
    assert "contents: write" in content
    assert "group: branch-hygiene" in content
    assert "if: github.repository == 'shuntatsu/trade_rl'" in content
    assert "--retire-merged --apply --max-deletes 100" in content
