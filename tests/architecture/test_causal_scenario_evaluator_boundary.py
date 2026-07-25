from __future__ import annotations

from pathlib import Path

PROHIBITED_IMPORTS = (
    "trade_rl.evaluation.causal_scenario_values",
    "trade_rl.evaluation.causal_scenario_artifact",
)
PROTECTED_ROOTS = (
    Path("trade_rl/rl"),
    Path("trade_rl/serving"),
    Path("trade_rl/release"),
    Path("trade_rl/workflows"),
)


def test_causal_scenario_evaluator_remains_outside_maintained_runtime_paths() -> None:
    violations: list[str] = []
    for root in PROTECTED_ROOTS:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*.py")):
            source = path.read_text(encoding="utf-8")
            for prohibited in PROHIBITED_IMPORTS:
                if prohibited in source:
                    violations.append(f"{path}: {prohibited}")

    config = Path("examples/binance-multitimeframe/walk-forward-full.json")
    if config.exists():
        source = config.read_text(encoding="utf-8")
        if "causal_scenario" in source:
            violations.append(f"{config}: causal_scenario")

    assert violations == []
