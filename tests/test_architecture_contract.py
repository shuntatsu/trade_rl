from __future__ import annotations

import tomllib

from tests.architecture.import_linter_config import (
    configured_layers,
    import_linter_contract,
)
from tests.architecture.repository_paths import PYTHON_SOURCE_ROOT, REPOSITORY_ROOT

ROOT = REPOSITORY_ROOT
PYTHON_ROOT = PYTHON_SOURCE_ROOT


def test_only_trade_rl_is_packaged() -> None:
    config = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert config["project"]["name"] == "trade-rl"
    package_find = config["tool"]["setuptools"]["packages"]["find"]
    assert package_find["where"] == ["."]
    assert package_find["include"] == ["trade_rl*"]
    assert config["project"]["scripts"] == {"trade-rl": "trade_rl.cli:main"}


def test_source_contains_maintained_direct_target_mode_without_legacy_env() -> None:
    source = "\n".join(
        path.read_text(encoding="utf-8") for path in sorted(PYTHON_ROOT.rglob("*.py"))
    )
    actions = (PYTHON_ROOT / "rl" / "actions.py").read_text(encoding="utf-8")
    assert 'TARGET_WEIGHT = "target_weight"' in actions
    assert "class TargetWeightAction" in actions
    assert "MarsLiteEnv" not in source


def test_walk_forward_evaluation_helpers_live_in_focused_module() -> None:
    workflow = (PYTHON_ROOT / "workflows" / "market_walk_forward.py").read_text(
        encoding="utf-8"
    )
    assert "def _evaluate_range(" not in workflow
    assert (PYTHON_ROOT / "workflows" / "walk_forward_evaluation.py").is_file()


def test_environment_terminal_helpers_live_in_transition_module() -> None:
    transition = PYTHON_ROOT / "rl" / "transition.py"
    assert transition.is_file()
    assert "class EconomicTransition" in transition.read_text(encoding="utf-8")


def test_maintained_docs_reference_reward_schema_v4() -> None:
    for path in (
        ROOT / "README.md",
        ROOT / "docs/ARCHITECTURE.md",
    ):
        text = path.read_text(encoding="utf-8").lower()
        assert "reward schema v3" not in text
        assert "reward schema v4" in text


def test_quickstart_installs_training_dependencies_before_training() -> None:
    text = (ROOT / "START.md").read_text(encoding="utf-8")
    assert "uv sync --extra dev --extra train-sb3" in text
    assert "uv run trade-rl train run" in text


def test_telemetry_has_an_enforced_low_level_dependency_boundary() -> None:
    layers = configured_layers()
    assert "trade_rl.telemetry" in layers
    assert layers.index("trade_rl.artifacts") < layers.index("trade_rl.telemetry")
    assert layers.index("trade_rl.telemetry") < layers.index("trade_rl.domain")
    forbidden_modules = {
        str(value) for value in import_linter_contract("telemetry")["forbidden_modules"]
    }
    for forbidden in (
        "numpy",
        "gymnasium",
        "stable_baselines3",
        "torch",
        "psycopg",
        "trade_rl.studio",
        "trade_rl.workflows",
        "trade_rl.integrations",
    ):
        assert forbidden in forbidden_modules


def test_critical_modules_do_not_disable_index_typing_file_wide() -> None:
    for path in (
        PYTHON_ROOT / "rl" / "environment.py",
        PYTHON_ROOT / "rl" / "observations.py",
        PYTHON_ROOT / "simulation" / "execution.py",
        PYTHON_ROOT / "strategies" / "trend.py",
    ):
        assert 'mypy: disable-error-code="index"' not in path.read_text(
            encoding="utf-8"
        )


def test_large_facades_delegate_configuration_to_focused_modules() -> None:
    environment = (PYTHON_ROOT / "rl" / "environment.py").read_text(encoding="utf-8")
    walk_forward = (PYTHON_ROOT / "workflows" / "market_walk_forward.py").read_text(
        encoding="utf-8"
    )
    assert "class ResidualMarketEnvConfig" not in environment
    assert (PYTHON_ROOT / "rl" / "environment_config.py").is_file()
    assert "class MarketWalkForwardConfig" not in walk_forward
    assert (PYTHON_ROOT / "workflows" / "market_walk_forward_config.py").is_file()


def test_sb3_and_torch_are_optional_training_dependencies() -> None:
    config = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    core = set(config["project"]["dependencies"])
    training = set(config["project"]["optional-dependencies"]["train-sb3"])
    assert not any(item.startswith("stable-baselines3") for item in core)
    assert not any(item.startswith("sb3-contrib") for item in core)
    assert not any(item.startswith("torch") for item in core)
    assert any(item.startswith("stable-baselines3") for item in training)
    assert any(item.startswith("torch") for item in training)
