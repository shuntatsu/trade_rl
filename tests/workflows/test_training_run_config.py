from __future__ import annotations

import pytest

from trade_rl.workflows.training_run import TrainingRunConfig


def _mapping() -> dict[str, object]:
    return {
        "training": {
            "timesteps": 8,
            "gamma": 0.99,
            "seeds": [0],
            "n_steps": 8,
            "batch_size": 8,
        },
        "environment": {
            "episode_bars": 4,
            "decision_every": 1,
            "initial_capital": 1_000.0,
        },
        "risk": {},
        "reward": {},
        "trend": {"fast_lookback": 1, "base_lookback": 2, "slow_lookback": 3},
        "action": {"alpha_enabled": True, "n_factors": 0},
    }


def test_training_config_requires_alpha_artifact_when_action_enables_alpha() -> None:
    with pytest.raises(ValueError, match="alpha artifact"):
        TrainingRunConfig.from_mapping(_mapping())


def test_training_config_accepts_alpha_artifact_path() -> None:
    raw = _mapping()
    raw["alpha_artifact"] = "artifacts/alpha"
    config = TrainingRunConfig.from_mapping(raw)
    assert str(config.alpha_artifact) == "artifacts/alpha"


def test_training_config_from_json_resolves_signal_artifact_paths(
    tmp_path: object,
) -> None:
    import json
    from pathlib import Path

    root = Path(str(tmp_path))
    raw = _mapping()
    raw["alpha_artifact"] = "signals/alpha"
    path = root / "config.json"
    path.write_text(json.dumps(raw), encoding="utf-8")

    config = TrainingRunConfig.from_json(path)

    assert config.alpha_artifact == root / "signals" / "alpha"


def test_training_config_identity_uses_signal_content_not_filesystem_path(
    tmp_path: object,
) -> None:
    from pathlib import Path

    import numpy as np

    from trade_rl.artifacts.hashing import content_digest
    from trade_rl.artifacts.signals import write_signal_artifact

    root = Path(str(tmp_path))
    paths = (root / "first", root / "second")
    for path in paths:
        write_signal_artifact(
            path,
            kind="alpha",
            dataset_id="d" * 64,
            fit_start=0,
            fit_stop=2,
            names=("BTC",),
            values=np.zeros((4, 1)),
        )
    raw = _mapping()
    raw["alpha_artifact"] = str(paths[0])
    first = TrainingRunConfig.from_mapping(raw)
    raw["alpha_artifact"] = str(paths[1])
    second = TrainingRunConfig.from_mapping(raw)

    assert content_digest(first.digest_payload()) == content_digest(
        second.digest_payload()
    )
    assert first.digest_payload()["alpha_artifact_digest"] is not None


def _workflow_dataset():
    import numpy as np

    from trade_rl.data.market import MarketDataset

    n_bars = 32
    timestamps = np.datetime64("2026-01-01T00:00:00", "ns") + np.arange(
        n_bars
    ) * np.timedelta64(1, "h")
    close = np.ones((n_bars, 1), dtype=np.float64)
    return MarketDataset(
        dataset_id="f" * 64,
        symbols=("BTC",),
        timestamps=timestamps,
        features=np.zeros((n_bars, 1, 1), dtype=np.float32),
        global_features=np.zeros((n_bars, 1), dtype=np.float32),
        open=close,
        high=close,
        low=close,
        close=close,
        volume=np.ones((n_bars, 1), dtype=np.float64),
        funding_rate=np.zeros((n_bars, 1), dtype=np.float64),
        tradable=np.ones((n_bars, 1), dtype=np.bool_),
        feature_available=np.ones((n_bars, 1, 1), dtype=np.bool_),
        feature_names=("feature",),
        global_feature_names=("regime",),
        periods_per_year=8_760,
    )


def test_authoritative_training_factory_requires_complete_reward_preroll() -> None:
    from trade_rl.workflows.training_run import _environment_factory

    raw = _mapping()
    raw["action"] = {"alpha_enabled": False, "n_factors": 0}
    raw["reward"] = {
        "baseline_window_hours": 4.0,
        "baseline_minimum_history_hours": 4.0,
    }
    config = TrainingRunConfig.from_mapping(raw)

    env = _environment_factory(_workflow_dataset(), config)()

    assert config.environment.require_full_reward_preroll is True
    assert env.config.require_full_reward_preroll is True


def test_walk_forward_environment_requires_complete_reward_preroll() -> None:
    from trade_rl.workflows.walk_forward_evaluation import build_market_environment

    raw = _mapping()
    raw["action"] = {"alpha_enabled": False, "n_factors": 0}
    raw["reward"] = {
        "baseline_window_hours": 4.0,
        "baseline_minimum_history_hours": 4.0,
    }
    config = TrainingRunConfig.from_mapping(raw)

    env = build_market_environment(
        _workflow_dataset(),
        config,
        normalizer=None,
        episode_bars=4,
        liquidate_on_end=False,
    )

    assert env.config.require_full_reward_preroll is True


def test_normalizer_fit_begins_after_complete_reward_preroll() -> None:
    from trade_rl.evaluation.walk_forward.folds import IndexRange
    from trade_rl.workflows.market_walk_forward import _fit_normalizer

    raw = _mapping()
    raw["action"] = {"alpha_enabled": False, "n_factors": 0}
    raw["reward"] = {
        "baseline_window_hours": 4.0,
        "baseline_minimum_history_hours": 4.0,
    }
    config = TrainingRunConfig.from_mapping(raw)

    normalizer = _fit_normalizer(
        _workflow_dataset(), IndexRange(start=0, stop=24), config
    )

    assert normalizer.train_end > normalizer.train_start
