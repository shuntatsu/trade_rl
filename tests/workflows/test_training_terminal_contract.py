from __future__ import annotations

import json
import runpy
from dataclasses import replace
from pathlib import Path

import pytest

from trade_rl.workflows._market_walk_forward_core import (
    _maintained_training_environment,
)
from trade_rl.workflows.training_run import (
    TrainingRunConfig,
    normalize_training_run_config,
)

ROOT = Path(__file__).resolve().parents[2]
PROFILE = (
    ROOT
    / "examples/binance-multitimeframe/training-target-weight-constrained-growth.json"
)
FULL_RESEARCH_SCRIPT = (
    ROOT / "examples/binance-multitimeframe/run_full_research_state.py"
)


def test_standalone_training_preserves_mark_to_market_truncation() -> None:
    config = TrainingRunConfig.from_json(PROFILE)

    normalized = normalize_training_run_config(config)

    assert normalized is config
    assert normalized.environment.liquidate_on_end is False
    assert normalized.environment.terminal_accounting_mode == "mark_to_market"
    assert normalized.environment.finite_horizon_observation is False


def test_training_contract_rejects_forced_close_configuration() -> None:
    config = TrainingRunConfig.from_json(PROFILE)
    forced_close = replace(
        config,
        environment=replace(config.environment, liquidate_on_end=True),
    )

    with pytest.raises(ValueError, match="mark-to-market truncation"):
        normalize_training_run_config(forced_close)


def test_walk_forward_training_preserves_mark_to_market_truncation() -> None:
    config = TrainingRunConfig.from_json(PROFILE)

    environment = _maintained_training_environment(
        config.environment,
        episode_bars=1_024,
    )

    assert environment.episode_bars == 1_024
    assert environment.episode_hour_choices == ()
    assert environment.liquidate_on_end is False
    assert environment.terminal_accounting_mode == "mark_to_market"
    assert environment.require_full_reward_preroll is True


def test_selected_final_config_does_not_rewrite_terminal_accounting(
    tmp_path: Path,
) -> None:
    path = tmp_path / "selected.json"
    path.write_text(PROFILE.read_text(encoding="utf-8"), encoding="utf-8")
    namespace = runpy.run_path(
        str(FULL_RESEARCH_SCRIPT),
        run_name="test_run_full_research_state",
    )
    normalize_selected = namespace["_normalize_selected_config"]
    assert callable(normalize_selected)

    config = normalize_selected(path)
    persisted = json.loads(path.read_text(encoding="utf-8"))

    assert config.environment.liquidate_on_end is False
    assert persisted["environment"]["liquidate_on_end"] is False
