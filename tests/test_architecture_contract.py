from __future__ import annotations

import tomllib
from pathlib import Path

ROOT = Path(__file__).parents[1]


def test_legacy_execution_trees_are_absent() -> None:
    for name in ("mars_lite", "legacy_tests", "scripts"):
        assert not (ROOT / name).exists(), f"legacy path still exists: {name}"


def test_only_trade_rl_is_packaged() -> None:
    config = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    assert config["project"]["name"] == "trade-rl"
    assert config["tool"]["setuptools"]["packages"]["find"]["include"] == ["trade_rl*"]
    assert config["project"]["scripts"] == {"trade-rl": "trade_rl.cli.app:main"}


def test_source_contains_no_maintained_direct_action_mode() -> None:
    source = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted((ROOT / "trade_rl").rglob("*.py"))
    )

    assert "action_mode" not in source
    assert "direct-action" not in source
    assert "MarsLiteEnv" not in source


def test_workflows_do_not_import_model_frameworks() -> None:
    workflow_root = Path("trade_rl/workflows")
    for path in workflow_root.glob("*.py"):
        source = path.read_text(encoding="utf-8")
        assert "stable_baselines3" not in source, path
        assert "sb3_contrib" not in source, path


def test_walk_forward_evaluation_helpers_live_in_focused_module() -> None:
    workflow = Path("trade_rl/workflows/market_walk_forward.py").read_text(
        encoding="utf-8"
    )
    focused = Path("trade_rl/workflows/walk_forward_evaluation.py")
    assert "def _evaluate_range(" not in workflow
    assert focused.is_file()


def test_environment_terminal_helpers_live_in_transition_module() -> None:
    transition = Path("trade_rl/rl/transition.py")
    assert transition.is_file()
    assert "class EconomicTransition" in transition.read_text(encoding="utf-8")


def test_maintained_docs_reference_reward_schema_v4() -> None:
    for path in (
        Path("README.md"),
        Path("README.ja.md"),
        Path("docs/ARCHITECTURE.md"),
    ):
        text = path.read_text(encoding="utf-8").lower()
        assert "reward schema v3" not in text
        assert "reward schema v4" in text


def test_critical_modules_do_not_disable_index_typing_file_wide() -> None:
    for path in (
        Path("trade_rl/rl/environment.py"),
        Path("trade_rl/rl/observations.py"),
        Path("trade_rl/simulation/execution.py"),
        Path("trade_rl/strategies/trend.py"),
    ):
        assert 'mypy: disable-error-code="index"' not in path.read_text(
            encoding="utf-8"
        )


def test_large_facades_delegate_configuration_to_focused_modules() -> None:
    environment = Path("trade_rl/rl/environment.py").read_text(encoding="utf-8")
    walk_forward = Path("trade_rl/workflows/market_walk_forward.py").read_text(
        encoding="utf-8"
    )

    assert "class ResidualMarketEnvConfig" not in environment
    assert Path("trade_rl/rl/environment_config.py").is_file()
    assert "class MarketWalkForwardConfig" not in walk_forward
    assert Path("trade_rl/workflows/market_walk_forward_config.py").is_file()
