from __future__ import annotations

import ast
from pathlib import Path

import trade_rl.evaluation.experiments as experiments

ROOT = Path(__file__).resolve().parents[2]
MODULE = (
    ROOT / "trade_rl" / "evaluation" / "experiments" / "four_family_input_manifest.py"
)
IDENTITY_FACADE = "trade_rl.evaluation.runs"
PRIVATE_IDENTITY_MODULE = "trade_rl.evaluation.runs.portfolio_family_result_artifact"
FORBIDDEN_IMPORTS = {
    "trade_rl.data.artifacts.publication",
    "trade_rl.evaluation.experiments.workflow",
    "trade_rl.evaluation.runs.portfolio_family_attempt",
    PRIVATE_IDENTITY_MODULE,
    "trade_rl.strategies.rl.portfolio_inference_bundle",
}


def test_post_freeze_manifest_stays_private_and_metadata_only() -> None:
    assert "FourFamilyPostFreezeInputManifest" not in experiments.__all__

    tree = ast.parse(MODULE.read_text(encoding="utf-8"), filename=str(MODULE))
    imported_identity_names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            assert module not in FORBIDDEN_IMPORTS
            if module == IDENTITY_FACADE:
                imported_identity_names.update(alias.name for alias in node.names)

    assert imported_identity_names == {"FourFamilyPortfolioAttemptIdentity"}
