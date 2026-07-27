from __future__ import annotations

import ast
from pathlib import Path


def test_sb3_backend_routes_model_lifecycle_through_typed_assembly() -> None:
    source = Path("trade_rl/integrations/sb3_training.py").read_text(encoding="utf-8")
    tree = ast.parse(source)

    imported: dict[str, str] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom) or node.module is None:
            continue
        for alias in node.names:
            imported[alias.asname or alias.name] = f"{node.module}.{alias.name}"

    assert imported["resolve_sb3_policy_assembly"] == (
        "trade_rl.integrations.sb3_model_assembly.resolve_sb3_policy_assembly"
    )
    assert imported["build_sb3_model"] == (
        "trade_rl.integrations.sb3_model_assembly.build_sb3_model"
    )
    assert imported["load_sb3_checkpoint_model"] == (
        "trade_rl.integrations.sb3_checkpoint_assembly.load_sb3_checkpoint_model"
    )

    called_names = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert {
        "resolve_sb3_policy_assembly",
        "build_sb3_model",
        "load_sb3_checkpoint_model",
    } <= called_names

    forbidden_backend_names = {
        "estimate_ppo_rollout_buffer_bytes",
        "estimate_index_backed_ppo_rollout_buffer_bytes",
        "build_learning_rate_schedule",
        "SequenceRolloutReconstructor",
        "IndexBackedDictRolloutBuffer",
    }
    referenced_names = {
        node.id for node in ast.walk(tree) if isinstance(node, ast.Name)
    }
    assert forbidden_backend_names.isdisjoint(referenced_names)
