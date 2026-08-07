from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pytest

from trade_rl.workflows.market_walk_forward_config import MarketWalkForwardConfig
from trade_rl.workflows.training_run import TrainingRunConfig

ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_ROOT = ROOT / "examples" / "binance-multitimeframe"


def _pipeline() -> object:
    if str(EXAMPLE_ROOT) not in sys.path:
        sys.path.insert(0, str(EXAMPLE_ROOT))
    return importlib.import_module("full_research_pipeline")


def test_all_maintained_profiles_are_one_action() -> None:
    training_names = (
        "training-full.json",
        "training-target-weight-growth-ppo.json",
        "training-target-weight-constrained-growth.json",
        "training-target-weight-constrained-growth-discounted.json",
    )
    for name in training_names:
        config = TrainingRunConfig.from_json(EXAMPLE_ROOT / name)
        assert config.action.mode.value == "target_weight"
        assert config.action.target_weight_count == 1
        assert config.action.names_for_symbols(("BTCUSDT",)) == (
            "target_weight:BTCUSDT",
        )

    for name in (
        "walk-forward-full.json",
        "walk-forward-target-weight-constrained-growth.json",
    ):
        workflow = MarketWalkForwardConfig.from_json(
            EXAMPLE_ROOT / name,
            n_bars=55_392,
        )
        assert workflow.candidates
        assert all(
            candidate.run.action.target_weight_count == 1
            for candidate in workflow.candidates
        )


def test_maintained_config_writer_rejects_three_actions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pipeline = _pipeline()
    monkeypatch.setenv("TRADE_RL_GIT_COMMIT", "a" * 40)
    monkeypatch.setenv("TRADE_RL_GIT_DIRTY", "false")
    template = tmp_path / "three-action.json"
    template.write_text(
        json.dumps(
            {
                "schema_version": "training_run_config_v4",
                "action": {
                    "mode": "target_weight",
                    "target_weight_count": 3,
                },
                "training": {},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="maintained training requires exactly one target-weight action",
    ):
        pipeline.write_run_config(
            template_path=template,
            output_path=tmp_path / "output.json",
        )


def test_legacy_multi_asset_pipeline_is_not_the_maintained_entrypoint() -> None:
    maintained = (EXAMPLE_ROOT / "full_research_pipeline.py").read_text(
        encoding="utf-8"
    )
    legacy = (EXAMPLE_ROOT / "full_research_pipeline_legacy.py").read_text(
        encoding="utf-8"
    )

    assert '_SYMBOLS = ("BTCUSDT",)' in maintained
    assert "_activate_symbol_triplet" not in maintained
    assert "build_symbol_disjoint_manifest" not in maintained
    assert "full_research_pipeline_legacy" in maintained
    assert "_activate_symbol_triplet" in legacy
    assert '"BTCUSDT", "ETHUSDT", "BNBUSDT"' in legacy
