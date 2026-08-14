from __future__ import annotations

import json
from pathlib import Path

from tests.support.training_config import complete_execution_config


def test_docs_match_maintained_schemas_and_boundaries() -> None:
    root = Path(__file__).resolve().parents[1]
    readme = (root / "README.md").read_text()
    architecture = (root / "docs/ARCHITECTURE.md").read_text()
    configuration = (root / "docs/CONFIGURATION.md").read_text()
    binance = (root / "docs/BINANCE.md").read_text()
    assert "training_run_config_v4" in readme
    assert "training_run_config_v3" not in readme
    assert "structured_policy_export_v1" not in readme
    assert "training_run_config_v4" in architecture
    assert "structured_policy_export_v2" in architecture
    assert "# Training Configuration v4" in configuration
    assert '"schema_version": "training_run_config_v4"' in configuration
    assert "structured_policy_export_v2" in configuration
    assert "change intensity" in architecture.lower()
    assert "constraint cost" in architecture.lower()
    assert "runner classification" in architecture.lower()
    assert "offline_signing" in architecture
    assert "7種類のconstraint cost" in architecture
    assert "binance_vision_raw_cache_v1" in binance


def test_configuration_example_is_valid_explicit_v4_json() -> None:
    root = Path(__file__).resolve().parents[1]
    configuration = (root / "docs/CONFIGURATION.md").read_text()
    start = configuration.index("```json\n") + len("```json\n")
    end = configuration.index("\n```", start)

    payload = json.loads(configuration[start:end])

    assert payload["schema_version"] == "training_run_config_v4"
    assert payload["environment"]["require_full_reward_preroll"] is True
    assert set(payload["execution"]) == set(complete_execution_config())


def test_quickstart_pins_hybrid_reward() -> None:
    root = Path(__file__).resolve().parents[1]
    reward = json.loads((root / "examples/quickstart/training.json").read_text())[
        "reward"
    ]
    assert reward["absolute_growth_weight"] == 1.0
    assert reward["incremental_drawdown_weight"] == 0.05
    assert reward["baseline_underperformance_weight"] == 0.10
    assert reward["terminal_equity_weight"] == 1.0
    assert reward["margin_deficit_weight"] == 1.0
