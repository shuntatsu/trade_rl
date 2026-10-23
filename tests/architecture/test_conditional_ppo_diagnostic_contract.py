from __future__ import annotations

import ast
import inspect
from pathlib import Path

from trade_rl.strategies.rl.ppo import PPOTradingEnv, fit_ppo_strategy

ROOT = Path(__file__).resolve().parents[2]


def test_conditional_ppo_evidence_is_opt_in_and_published_after_validation() -> None:
    assert (
        inspect.signature(PPOTradingEnv).parameters["execution_observer"].default
        is None
    )
    assert "execution_observer" not in inspect.signature(fit_ppo_strategy).parameters

    tool_path = ROOT / "tools" / "ppo_conditional_diagnostic.py"
    tool_source = tool_path.read_text(encoding="utf-8")
    tree = ast.parse(tool_source, filename=str(tool_path))
    run_function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "run_diagnostic"
    )
    run_source = ast.get_source_segment(tool_source, run_function)
    assert run_source is not None
    assert (
        run_source.index("save_ppo_inference_bundle(")
        < run_source.index("load_ppo_inference_bundle(")
        < run_source.index("_heldout_clauses(policy, heldout_dataset)")
    )
    assert "feature_names=feed_feature_names" in run_source
    assert "loaded_strategy.observation_schema != strategy.observation_schema" in (
        run_source
    )
    assert (
        run_source.index("evidence_manifest_metadata = _write_verified_json_artifact(")
        < run_source.index("_final_verified_policy_bundle(")
        < run_source.index("_final_verified_json(")
        < run_source.index("_publish_completed_result(output, result)")
    )

    architecture = (ROOT / "docs" / "architecture" / "lean-core.md").read_text(
        encoding="utf-8"
    )
    assert "ppo_conditional_diagnostic_result_v2" in architecture
    assert "evidence_contract_version=2" in architecture
    assert "既存loaderで再読込し" in architecture
    assert "publish直前" in architecture
    assert "learnability、G0-G3" in architecture
    assert "通常のfitへ自動接続しない" in architecture
