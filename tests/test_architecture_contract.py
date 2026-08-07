from __future__ import annotations

import tomllib

from tests.architecture.repository_paths import PYTHON_SOURCE_ROOT, REPOSITORY_ROOT

ROOT = REPOSITORY_ROOT
PYTHON_ROOT = PYTHON_SOURCE_ROOT


def test_legacy_execution_trees_are_absent() -> None:
    for name in ("mars_lite", "legacy_tests"):
        assert not (ROOT / name).exists(), f"legacy path still exists: {name}"


def test_only_trade_rl_is_packaged() -> None:
    config = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert config["project"]["name"] == "trade-rl"
    package_find = config["tool"]["setuptools"]["packages"]["find"]
    assert package_find["where"] == ["src"]
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


def test_workflows_do_not_import_model_frameworks() -> None:
    workflow_root = PYTHON_ROOT / "workflows"
    for path in workflow_root.glob("*.py"):
        source = path.read_text(encoding="utf-8")
        assert "stable_baselines3" not in source, path
        assert "sb3_contrib" not in source, path


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


def test_maintained_docs_reference_serving_bundle_v6() -> None:
    for path in (
        ROOT / "README.md",
        ROOT / "docs/ARCHITECTURE.md",
        ROOT / "docs/RESEARCH_STATUS.md",
    ):
        text = path.read_text(encoding="utf-8").lower()
        assert "serving_bundle_v5" not in text, path
        assert "serving_bundle_v6" in text, path


def test_quickstart_installs_training_dependencies_before_training() -> None:
    text = (ROOT / "START.md").read_text(encoding="utf-8")
    assert "uv sync --extra dev --extra train-sb3" in text
    assert "uv run trade-rl train run" in text


def test_architecture_doc_matches_enforced_layer_order() -> None:
    architecture = (ROOT / "docs/ARCHITECTURE.md").read_text(encoding="utf-8")
    import_linter = (ROOT / ".importlinter").read_text(encoding="utf-8")
    layer_block = import_linter.split("layers =", maxsplit=1)[1].split(
        "[importlinter:contract:domain]", maxsplit=1
    )[0]
    enforced_layers = tuple(
        line.strip().removeprefix("trade_rl.")
        for line in layer_block.splitlines()
        if line.strip().startswith("trade_rl.")
    )

    marker = "Import Linterの強制順序は次のとおりです:"
    documented_section = architecture.split(marker, maxsplit=1)[1]
    documented_block = documented_section.split("```", maxsplit=2)[1]
    documented_layers = tuple(
        line.strip()
        for line in documented_block.splitlines()
        if line.strip() and line.strip() != "text"
    )

    assert enforced_layers
    assert documented_layers == enforced_layers


def test_telemetry_has_an_enforced_low_level_dependency_boundary() -> None:
    import_linter = (ROOT / ".importlinter").read_text(encoding="utf-8")
    layer_block = import_linter.split("layers =", maxsplit=1)[1].split(
        "[importlinter:contract:domain]", maxsplit=1
    )[0]
    layers = tuple(
        line.strip()
        for line in layer_block.splitlines()
        if line.strip().startswith("trade_rl.")
    )

    assert "trade_rl.telemetry" in layers
    assert layers.index("trade_rl.artifacts") < layers.index("trade_rl.telemetry")
    assert layers.index("trade_rl.telemetry") < layers.index("trade_rl.domain")
    assert "[importlinter:contract:telemetry]" in import_linter
    telemetry_contract = import_linter.split(
        "[importlinter:contract:telemetry]", maxsplit=1
    )[1]
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
        assert forbidden in telemetry_contract


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


def test_core_training_contract_does_not_import_gym_or_model_frameworks() -> None:
    source = (PYTHON_ROOT / "rl" / "training.py").read_text(encoding="utf-8")
    assert "import gymnasium" not in source
    assert "stable_baselines3" not in source
    assert "sb3_contrib" not in source
    assert "import torch" not in source
