from __future__ import annotations

from pathlib import Path

from trade_rl.workflows.causal_scenario import (
    CAUSAL_SCENARIO_LIBRARY_ARTIFACT_SCHEMA,
    CausalConditionConfig,
    CausalScenarioLibraryConfig,
    CausalScenarioReplayIdentity,
    build_causal_scenario_library,
    load_causal_scenario_library_artifact,
    materialize_causal_scenario_dataset,
    select_causal_scenarios,
    write_causal_scenario_library_artifact,
)

PROHIBITED_IMPORTS = (
    "trade_rl.workflows.causal_scenario.conditions",
    "trade_rl.workflows.causal_scenario.library",
    "trade_rl.workflows.causal_scenario.library_artifact",
    "trade_rl.workflows.causal_scenario.replay",
)
CAUSAL_SCENARIO_ROOT = Path("trade_rl/workflows/causal_scenario")

PROTECTED_ROOTS = (
    Path("trade_rl/rl"),
    Path("trade_rl/serving"),
    Path("trade_rl/release"),
    Path("trade_rl/workflows"),
    Path("trade_rl/integrations"),
)


def test_c2_public_api_is_available() -> None:
    assert CAUSAL_SCENARIO_LIBRARY_ARTIFACT_SCHEMA.endswith("_v1")
    assert CausalConditionConfig is not None
    assert CausalScenarioLibraryConfig is not None
    assert CausalScenarioReplayIdentity is not None
    assert build_causal_scenario_library is not None
    assert select_causal_scenarios is not None
    assert materialize_causal_scenario_dataset is not None
    assert write_causal_scenario_library_artifact is not None
    assert load_causal_scenario_library_artifact is not None


def test_causal_scenario_library_remains_outside_runtime_paths() -> None:
    violations: list[str] = []
    for root in PROTECTED_ROOTS:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*.py")):
            if path.is_relative_to(CAUSAL_SCENARIO_ROOT):
                continue
            source = path.read_text(encoding="utf-8")
            for prohibited in PROHIBITED_IMPORTS:
                if prohibited in source:
                    violations.append(f"{path}: {prohibited}")
    config = Path("examples/binance-multitimeframe/walk-forward-full.json")
    if config.exists() and "causal_scenario_library" in config.read_text(
        encoding="utf-8"
    ):
        violations.append(f"{config}: causal_scenario_library")
    assert violations == []
