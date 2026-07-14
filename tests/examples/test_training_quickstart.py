from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import numpy as np

from trade_rl.data import load_market_dataset_artifact, write_market_dataset_files
from trade_rl.workflows.training_run import TrainingRunConfig, _environment_factory

ROOT = Path(__file__).resolve().parents[2]
QUICKSTART = ROOT / "examples" / "quickstart"


def _load_dataset_builder() -> ModuleType:
    path = QUICKSTART / "create_demo_dataset.py"
    spec = importlib.util.spec_from_file_location("quickstart_dataset_builder", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load quickstart dataset builder")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_demo_dataset_is_deterministic_and_round_trips(tmp_path: Path) -> None:
    module = _load_dataset_builder()
    first = module.build_demo_dataset(512)
    second = module.build_demo_dataset(512)

    assert first.dataset_id == second.dataset_id
    assert first.symbols == ("BTCUSDT",)
    assert first.feature_names == ("log_return_1h", "volatility_24h")
    assert first.global_feature_names == ("market_trend_24h",)
    np.testing.assert_array_equal(first.close, second.close)
    np.testing.assert_array_equal(first.features, second.features)

    result = write_market_dataset_files(tmp_path, first)
    restored = load_market_dataset_artifact(tmp_path)

    assert result.manifest_path == tmp_path / "manifest.json"
    assert result.arrays_path == tmp_path / "arrays.npz"
    assert restored.dataset_id == first.dataset_id
    np.testing.assert_array_equal(restored.close, first.close)


def test_quickstart_config_builds_a_compatible_environment() -> None:
    module = _load_dataset_builder()
    dataset = module.build_demo_dataset(1_024)
    config = TrainingRunConfig.from_json(QUICKSTART / "training.json")

    assert config.training.timesteps == 512
    assert config.training.device == "cpu"
    assert config.training.seeds == (0,)
    assert config.action.names == ("fast_tilt", "slow_tilt", "risk_tilt")
    assert config.environment.initial_capital == 100_000.0

    environment = _environment_factory(dataset, config)()
    try:
        observation, info = environment.reset(seed=0)
        assert observation.shape == environment.observation_space.shape
        assert info["dataset_id"] == dataset.dataset_id
        action = np.zeros(environment.action_space.shape, dtype=np.float32)
        next_observation, reward, terminated, truncated, step_info = environment.step(
            action
        )
        assert next_observation.shape == environment.observation_space.shape
        assert np.isfinite(reward)
        assert not (terminated and truncated)
        assert step_info["dataset_id"] == dataset.dataset_id
    finally:
        environment.close()


def test_start_document_references_existing_quickstart_assets() -> None:
    start = (ROOT / "START.md").read_text(encoding="utf-8")

    assert "examples/quickstart/create_demo_dataset.py" in start
    assert "examples/quickstart/training.json" in start
    assert "trade-rl train run" in start
    assert "production_status" in start
    assert (QUICKSTART / "create_demo_dataset.py").is_file()
    assert (QUICKSTART / "training.json").is_file()
