from __future__ import annotations

from pathlib import Path

PROHIBITED_PREFIX = "trade_rl.evaluation.causal_scenario_c3_"
ALLOWED_WORKFLOW_ROOT = Path("trade_rl/workflows/causal_scenario")
PROTECTED_ROOTS = (
    Path("trade_rl/rl"),
    Path("trade_rl/serving"),
    Path("trade_rl/release"),
    Path("trade_rl/promotion"),
    Path("trade_rl/workflows"),
)


def test_causal_scenario_c3_remains_evaluation_only() -> None:
    violations: list[str] = []
    for root in PROTECTED_ROOTS:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*.py")):
            if path.is_relative_to(ALLOWED_WORKFLOW_ROOT):
                continue
            source = path.read_text(encoding="utf-8")
            if PROHIBITED_PREFIX in source:
                violations.append(f"{path}: {PROHIBITED_PREFIX}")
    assert violations == []
