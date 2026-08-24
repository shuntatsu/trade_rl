from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]


def _imports(path: Path) -> tuple[str, ...]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    result: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            result.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            result.append(node.module or "")
    return tuple(result)


def test_v4_and_v5_never_import_v6() -> None:
    paths = (
        _ROOT / "trade_rl/learning/causal_alpha_v4.py",
        _ROOT / "trade_rl/learning/causal_alpha_v5.py",
        *(_ROOT / "trade_rl/workflows").glob("universal_causal_alpha_v4*.py"),
        *(_ROOT / "trade_rl/workflows").glob("universal_causal_alpha_v5*.py"),
    )
    for path in paths:
        assert all("causal_alpha_v6" not in module for module in _imports(path)), path


def test_v6_learning_remains_framework_and_workflow_independent() -> None:
    paths = (
        _ROOT / "trade_rl/learning/causal_alpha_v6.py",
        _ROOT / "trade_rl/learning/causal_alpha_v6_target.py",
    )
    banned = (
        "trade_rl.workflows",
        "trade_rl.integrations",
        "trade_rl.rl",
        "trade_rl.serving",
        "stable_baselines3",
        "sb3_contrib",
        "torch",
    )
    for path in paths:
        assert not any(
            module.startswith(prefix)
            for module in _imports(path)
            for prefix in banned
        ), path


def test_v6_runner_surface_imports_no_bc_rl_serving_or_v5() -> None:
    paths = tuple(
        (_ROOT / "trade_rl/workflows").glob("universal_causal_alpha_v6*.py")
    )
    banned = (
        "episode_oracle_bc",
        "stable_baselines3",
        "sb3_contrib",
        "trade_rl.rl",
        "trade_rl.serving",
        "universal_causal_alpha_v5",
        "causal_alpha_v5",
    )
    for path in paths:
        assert not any(
            module.startswith(prefix) or prefix in module
            for module in _imports(path)
            for prefix in banned
        ), path


def test_serving_never_imports_v6_research() -> None:
    for path in (_ROOT / "trade_rl/serving").rglob("*.py"):
        assert all("causal_alpha_v6" not in module for module in _imports(path)), path


def test_v6_script_bootstraps_from_outside_repository(tmp_path: Path) -> None:
    script = _ROOT / "scripts/run_universal_causal_alpha_v6_research.py"
    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "--v4-context-manifest" in result.stdout
