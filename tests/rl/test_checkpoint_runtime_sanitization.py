from __future__ import annotations

from pathlib import Path

from trade_rl.rl.checkpointing import save_policy_without_runtime_state
from trade_rl.rl.training_performance import (
    TRAINING_RUNTIME_PATCHES_ATTRIBUTE,
    TrainingPerformanceRecorder,
)


class _Policy:
    def extract_features(self) -> None:
        return None


class _Environment:
    def step(self) -> None:
        return None


class _Model:
    def __init__(self) -> None:
        self.policy = _Policy()
        self.env = _Environment()
        self.rollout_buffer_kwargs: dict[str, object] = {}
        self.saved_runtime_state: dict[str, bool] | None = None

    def collect_rollouts(self) -> None:
        return None

    def train(self) -> None:
        return None

    def save(self, target: str) -> None:
        self.saved_runtime_state = {
            "collect_rollouts": "collect_rollouts" in self.__dict__,
            "train": "train" in self.__dict__,
            "policy_extract_features": "extract_features" in self.policy.__dict__,
            "environment_step": "step" in self.env.__dict__,
            "registry": hasattr(self, TRAINING_RUNTIME_PATCHES_ATTRIBUTE),
        }
        Path(target).with_suffix(".zip").write_bytes(b"checkpoint")


def test_checkpoint_save_suspends_and_restores_training_runtime_patches(
    tmp_path: Path,
) -> None:
    model = _Model()
    recorder = TrainingPerformanceRecorder()

    with recorder.instrument_model(model):
        assert "collect_rollouts" in model.__dict__
        assert "train" in model.__dict__
        assert "extract_features" in model.policy.__dict__
        assert "step" in model.env.__dict__
        assert hasattr(model, TRAINING_RUNTIME_PATCHES_ATTRIBUTE)

        save_policy_without_runtime_state(model, str(tmp_path / "policy"))

        assert "collect_rollouts" in model.__dict__
        assert "train" in model.__dict__
        assert "extract_features" in model.policy.__dict__
        assert "step" in model.env.__dict__
        assert hasattr(model, TRAINING_RUNTIME_PATCHES_ATTRIBUTE)

    assert model.saved_runtime_state == {
        "collect_rollouts": False,
        "train": False,
        "policy_extract_features": False,
        "environment_step": False,
        "registry": False,
    }
    assert "collect_rollouts" not in model.__dict__
    assert "train" not in model.__dict__
    assert "extract_features" not in model.policy.__dict__
    assert "step" not in model.env.__dict__
    assert not hasattr(model, TRAINING_RUNTIME_PATCHES_ATTRIBUTE)
