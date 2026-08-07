from __future__ import annotations

import ast

from tests.architecture.repository_paths import PYTHON_SOURCE_ROOT


def _parsed(path: str) -> ast.Module:
    return ast.parse((PYTHON_SOURCE_ROOT / path).read_text(encoding="utf-8"))


def test_sb3_backend_routes_model_lifecycle_through_typed_assembly() -> None:
    tree = _parsed("integrations/sb3_training.py")

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


def test_external_sb3_runtime_values_remain_dynamically_typed() -> None:
    model_tree = _parsed("integrations/sb3_model_assembly.py")
    checkpoint_tree = _parsed("integrations/sb3_checkpoint_assembly.py")

    build_function = next(
        node
        for node in model_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "build_sb3_model"
    )
    assert isinstance(build_function.returns, ast.Name)
    assert build_function.returns.id == "Any"

    policy_class = next(
        node
        for node in model_tree.body
        if isinstance(node, ast.ClassDef) and node.name == "SB3PolicyAssembly"
    )
    sequence_metadata = next(
        node
        for node in policy_class.body
        if isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name)
        and node.target.id == "sequence_metadata"
    )
    assert "Any" in ast.unparse(sequence_metadata.annotation)

    checkpoint_class = next(
        node
        for node in checkpoint_tree.body
        if isinstance(node, ast.ClassDef) and node.name == "LoadedSB3Checkpoint"
    )
    model_field = next(
        node
        for node in checkpoint_class.body
        if isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name)
        and node.target.id == "model"
    )
    assert isinstance(model_field.annotation, ast.Name)
    assert model_field.annotation.id == "Any"
