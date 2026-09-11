from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TEMPLATE = ROOT / ".github" / "pull_request_template.md"
REQUIRED = (
    "## Objective",
    "## Non-goals",
    "## Acceptance Criteria",
    "## Invariants",
    "## Failure Modes",
    "## Test Oracle",
    "## Changed authorities / public surfaces",
    "## Tests / verification",
    "## Falsification",
    "## Unverified items",
    "## Residual risk",
)


def test_pr_template_contains_quality_contract() -> None:
    text = TEMPLATE.read_text(encoding="utf-8")
    missing = [heading for heading in REQUIRED if heading not in text]
    assert missing == []
