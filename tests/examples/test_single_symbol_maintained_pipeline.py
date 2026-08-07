from __future__ import annotations

import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_ROOT = ROOT / "examples" / "binance-multitimeframe"


def _pipeline() -> object:
    sys.path.insert(0, str(EXAMPLE_ROOT))
    return importlib.import_module("full_research_pipeline")


def test_maintained_pipeline_uses_exactly_one_btc_symbol() -> None:
    pipeline = _pipeline()

    assert pipeline._SYMBOLS == ("BTCUSDT",)
    assert not hasattr(pipeline, "_SLOT_SYMBOLS")
    assert not hasattr(pipeline, "_SYMBOL_POOL")
    assert not hasattr(pipeline, "_ACTIVE_SYMBOL_TRIPLET")
    assert not hasattr(pipeline, "activate_symbol_triplet")


def test_maintained_pipeline_has_no_triplet_or_literal_three_action_path() -> None:
    source = (EXAMPLE_ROOT / "full_research_pipeline.py").read_text(encoding="utf-8")

    for forbidden in (
        "symbol_disjoint_manifest",
        "symbol_disjoint_triplet_manifest",
        "symbol_triplet_manifest",
        "action_size=3",
    ):
        assert forbidden not in source


def test_state_runner_derives_action_size_and_has_no_triplet_controls() -> None:
    source = (EXAMPLE_ROOT / "run_full_research_state.py").read_text(encoding="utf-8")

    assert "action_size=dataset.n_symbols" in source
    for forbidden in (
        "dynamic_symbol_triplets",
        "activate_symbol_triplet",
        "selected-symbol-triplet.json",
        "--dynamic-symbol-triplets",
        "--symbol-triplet-seed",
        "--symbol-triplet-train-slot",
    ):
        assert forbidden not in source


def test_maintained_pipeline_keeps_four_context_timeframes() -> None:
    pipeline = _pipeline()

    assert pipeline._NATIVE_TIMEFRAMES == ("15m", "1h", "4h", "1d")
    assert pipeline._FEATURE_TIMEFRAMES == ("1h", "4h", "1d")
