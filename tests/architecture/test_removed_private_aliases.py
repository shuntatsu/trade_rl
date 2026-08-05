from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_removed_oracle_portfolio_state_alias_does_not_return() -> None:
    obsolete = "_" + "portfolio_states"
    violations: list[str] = []
    for root in (ROOT / "trade_rl", ROOT / "tests"):
        for path in sorted(root.rglob("*.py")):
            if obsolete in path.read_text(encoding="utf-8"):
                violations.append(path.relative_to(ROOT).as_posix())

    assert violations == []
