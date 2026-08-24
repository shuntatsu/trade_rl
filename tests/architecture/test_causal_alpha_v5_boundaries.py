from __future__ import annotations

import ast
import hashlib
from pathlib import Path

from trade_rl.learning.causal_alpha_v5 import CAUSAL_ALPHA_V5_CALIBRATION_FEATURE_NAMES

_ROOT = Path(__file__).resolve().parents[2]


def _imports(path: Path) -> tuple[tuple[str, tuple[str, ...]], ...]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    result: list[tuple[str, tuple[str, ...]]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            result.extend((alias.name, ()) for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            result.append(
                (node.module or "", tuple(alias.name for alias in node.names))
            )
    return tuple(result)


def test_v4_modules_never_import_v5() -> None:
    paths = (
        _ROOT / "trade_rl/learning/causal_alpha_v4.py",
        *(_ROOT / "trade_rl/workflows").glob("universal_causal_alpha_v4*.py"),
    )
    for path in paths:
        assert all("causal_alpha_v5" not in module for module, _ in _imports(path)), (
            path
        )


def test_v5_learning_uses_only_public_declared_v4_target_contracts() -> None:
    path = _ROOT / "trade_rl/learning/causal_alpha_v5.py"
    v4_imports = [
        names
        for module, names in _imports(path)
        if module == "trade_rl.learning.causal_alpha_v4"
    ]
    assert v4_imports
    assert all(not name.startswith("_") for names in v4_imports for name in names)


def test_v5_calibrator_features_have_no_symbol_identity() -> None:
    assert len(CAUSAL_ALPHA_V5_CALIBRATION_FEATURE_NAMES) == 12
    assert all(
        "symbol" not in name.lower() and "identity" not in name.lower()
        for name in CAUSAL_ALPHA_V5_CALIBRATION_FEATURE_NAMES
    )


def test_v5_runner_surface_imports_no_bc_rl_or_serving_stack() -> None:
    paths = tuple(
        (_ROOT / "trade_rl/workflows").glob("universal_causal_alpha_v5_*runner.py")
    ) + (
        _ROOT / "trade_rl/workflows/universal_causal_alpha_v5_stage_execution.py",
        _ROOT / "trade_rl/workflows/universal_causal_alpha_v5_stage_entry.py",
    )
    banned = (
        "episode_oracle_bc",
        "stable_baselines3",
        "sb3_contrib",
        "trade_rl.rl",
        "trade_rl.serving",
    )
    for path in paths:
        modules = tuple(module for module, _ in _imports(path))
        assert not any(
            module.startswith(prefix) for module in modules for prefix in banned
        ), path


def test_v5_paths_are_already_training_sensitive_in_current_classifier() -> None:
    workflow = (_ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    assert "trade_rl/" in workflow
    assert "tests/" in workflow
    assert "scripts/" in workflow
    assert "examples/" in workflow


def test_v4_schema_constants_and_authored_example_are_unchanged() -> None:
    source = (_ROOT / "trade_rl/learning/causal_alpha_v4.py").read_text(
        encoding="utf-8"
    )
    for value in (
        "causal_alpha_v4_symbol_samples_v1",
        "causal_alpha_v4_residual_labels_v1",
        "causal_alpha_v4_fit_config_v1",
        "causal_alpha_v4_forecast_v1",
        "causal_alpha_v4_uncertainty_v1",
        "causal_alpha_v4_target_v1",
    ):
        assert value in source
    payload = (
        _ROOT / "examples/binance/universal-causal-alpha-v4-research.json"
    ).read_bytes()
    assert (
        hashlib.sha256(payload).hexdigest()
        == "560e08e00b44b6cf559c5451eb7dd368f6006ed118979f80b44d2fe982c495dc"
    )
