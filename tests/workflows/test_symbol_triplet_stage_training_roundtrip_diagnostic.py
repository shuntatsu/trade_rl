from __future__ import annotations

import runpy
from pathlib import Path
from typing import Any

from trade_rl.workflows.symbol_triplet_stage_orchestrator import (
    build_symbol_triplet_stage_request,
    training_config_for_symbol_triplet_stage,
)
from trade_rl.workflows.symbol_triplet_stage_training import _training_config_mapping
from trade_rl.workflows.symbol_triplet_training_cursor import (
    initial_symbol_triplet_training_cursor,
)
from trade_rl.workflows.training_run import TrainingRunConfig


def _first_difference(left: object, right: object, path: str = "root") -> str | None:
    if type(left) is not type(right):
        return f"{path}: type {type(left).__name__} != {type(right).__name__}; {left!r} != {right!r}"
    if isinstance(left, dict):
        left_dict = left
        right_dict = right
        assert isinstance(right_dict, dict)
        if set(left_dict) != set(right_dict):
            return f"{path}: keys {sorted(left_dict)} != {sorted(right_dict)}"
        for key in sorted(left_dict):
            difference = _first_difference(left_dict[key], right_dict[key], f"{path}.{key}")
            if difference is not None:
                return difference
        return None
    if isinstance(left, (tuple, list)):
        right_items = right
        assert isinstance(right_items, (tuple, list))
        if len(left) != len(right_items):
            return f"{path}: length {len(left)} != {len(right_items)}"
        for index, (left_item, right_item) in enumerate(zip(left, right_items, strict=True)):
            difference = _first_difference(left_item, right_item, f"{path}[{index}]")
            if difference is not None:
                return difference
        return None
    if left != right:
        return f"{path}: {left!r} != {right!r}"
    return None


def test_diagnose_stage_config_roundtrip(tmp_path: Path) -> None:
    namespace: dict[str, Any] = runpy.run_path(
        "tests/workflows/test_symbol_triplet_stage_training.py"
    )
    plan = namespace["_plan"]()
    cursor = initial_symbol_triplet_training_cursor(plan)
    config_path = namespace["_write_config"](tmp_path / "base-config.json")
    base_config = TrainingRunConfig.from_json(config_path)
    request = build_symbol_triplet_stage_request(
        plan,
        cursor,
        training_seeds=base_config.training.seeds,
        previous_completion=None,
    )
    assert request is not None
    stage_config = training_config_for_symbol_triplet_stage(base_config, request)
    mapping = _training_config_mapping(stage_config)
    round_tripped = TrainingRunConfig.from_mapping(mapping).resolve_artifact_paths(tmp_path)
    left = stage_config.digest_payload()
    right = round_tripped.digest_payload()
    difference = _first_difference(left, right)
    assert difference is None, difference
