from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def replace_once(path: str, old: str, new: str) -> None:
    target = ROOT / path
    text = target.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{path}: expected one anchor, found {count}")
    updated = text.replace(old, new, 1)
    ast.parse(updated, filename=path)
    target.write_text(updated, encoding="utf-8")


def update_training_performance() -> None:
    replace_once(
        "trade_rl/rl/training_performance.py",
        '''            def timed(*args: Any, **kwargs: Any) -> Any:
                with self._measure(metric_name):
                    return original(*args, **kwargs)

            setattr(owner, name, timed)
''',
        '''            def timed(*args: Any, **kwargs: Any) -> Any:
                with self._measure(metric_name):
                    return original(*args, **kwargs)

            setattr(timed, "_trade_rl_transient_instrumentation", True)
            setattr(owner, name, timed)
''',
    )


def update_checkpointing() -> None:
    replace_once(
        "trade_rl/rl/checkpointing.py",
        '''def save_policy_without_runtime_state(model: SavablePolicy, target: str) -> None:
    """Save without serializing dataset-bound rollout reconstruction objects."""

    missing = object()
    original = getattr(model, "rollout_buffer_kwargs", missing)
    if isinstance(original, dict) and "sequence_reconstructor" in original:
        sanitized = {
            key: value
            for key, value in original.items()
            if key != "sequence_reconstructor"
        }
        setattr(model, "rollout_buffer_kwargs", sanitized)
    try:
        model.save(target)
    finally:
        if original is not missing:
            setattr(model, "rollout_buffer_kwargs", original)
''',
        '''def save_policy_without_runtime_state(model: SavablePolicy, target: str) -> None:
    """Save without dataset-bound or transient performance instrumentation state."""

    missing = object()
    original = getattr(model, "rollout_buffer_kwargs", missing)
    if isinstance(original, dict) and "sequence_reconstructor" in original:
        sanitized = {
            key: value
            for key, value in original.items()
            if key != "sequence_reconstructor"
        }
        setattr(model, "rollout_buffer_kwargs", sanitized)

    removed: list[tuple[object, str, object]] = []

    def remove_transient(owner: object | None, name: str) -> None:
        if owner is None:
            return
        namespace = getattr(owner, "__dict__", None)
        if not isinstance(namespace, dict) or name not in namespace:
            return
        value = namespace[name]
        if getattr(value, "_trade_rl_transient_instrumentation", False) is not True:
            return
        removed.append((owner, name, value))
        delattr(owner, name)

    remove_transient(model, "collect_rollouts")
    remove_transient(model, "train")
    remove_transient(getattr(model, "policy", None), "extract_features")
    remove_transient(getattr(model, "env", None), "step")
    try:
        model.save(target)
    finally:
        for owner, name, value in reversed(removed):
            setattr(owner, name, value)
        if original is not missing:
            setattr(model, "rollout_buffer_kwargs", original)
''',
    )


def write_tests() -> None:
    path = ROOT / "tests/rl/test_checkpoint_runtime_state.py"
    path.write_text(
        '''from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from trade_rl.rl.checkpointing import save_policy_without_runtime_state
from trade_rl.rl.training_performance import TrainingPerformanceRecorder


class _Policy:
    def extract_features(self, value: object) -> object:
        return value


class _Environment:
    def step(self, action: object) -> object:
        return action


class _InstrumentedModel:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.policy = _Policy()
        self.env = _Environment()
        self.reconstructor = object()
        self.rollout_buffer_kwargs = {
            "sequence_reconstructor": self.reconstructor,
            "sequence_transfer_mode": "synchronous",
        }
        self.observed_during_save: dict[str, object] = {}

    def collect_rollouts(self) -> None:
        return None

    def train(self) -> None:
        return None

    def save(self, target: str) -> None:
        self.observed_during_save = {
            "collect_rollouts_local": "collect_rollouts" in self.__dict__,
            "train_local": "train" in self.__dict__,
            "extract_features_local": "extract_features" in self.policy.__dict__,
            "step_local": "step" in self.env.__dict__,
            "rollout_buffer_kwargs": dict(self.rollout_buffer_kwargs),
        }
        if self.fail:
            raise RuntimeError("save failed")
        Path(target).write_bytes(b"policy")


def _assert_sanitized(model: _InstrumentedModel) -> None:
    assert model.observed_during_save == {
        "collect_rollouts_local": False,
        "train_local": False,
        "extract_features_local": False,
        "step_local": False,
        "rollout_buffer_kwargs": {"sequence_transfer_mode": "synchronous"},
    }


def test_checkpoint_save_strips_and_restores_transient_instrumentation(
    tmp_path: Path,
) -> None:
    model = _InstrumentedModel()
    recorder = TrainingPerformanceRecorder()

    with recorder.instrument_model(model):
        wrapped: dict[str, Any] = {
            "collect_rollouts": model.__dict__["collect_rollouts"],
            "train": model.__dict__["train"],
            "extract_features": model.policy.__dict__["extract_features"],
            "step": model.env.__dict__["step"],
        }
        save_policy_without_runtime_state(model, str(tmp_path / "policy.zip"))
        _assert_sanitized(model)
        assert model.__dict__["collect_rollouts"] is wrapped["collect_rollouts"]
        assert model.__dict__["train"] is wrapped["train"]
        assert model.policy.__dict__["extract_features"] is wrapped["extract_features"]
        assert model.env.__dict__["step"] is wrapped["step"]
        assert model.rollout_buffer_kwargs["sequence_reconstructor"] is model.reconstructor

    assert "collect_rollouts" not in model.__dict__
    assert "train" not in model.__dict__
    assert "extract_features" not in model.policy.__dict__
    assert "step" not in model.env.__dict__


def test_checkpoint_save_restores_runtime_state_after_failure(tmp_path: Path) -> None:
    model = _InstrumentedModel(fail=True)
    recorder = TrainingPerformanceRecorder()

    with recorder.instrument_model(model):
        wrapped_train = model.__dict__["train"]
        with pytest.raises(RuntimeError, match="save failed"):
            save_policy_without_runtime_state(model, str(tmp_path / "policy.zip"))
        _assert_sanitized(model)
        assert model.__dict__["train"] is wrapped_train
        assert model.rollout_buffer_kwargs["sequence_reconstructor"] is model.reconstructor
''',
        encoding="utf-8",
    )


def main() -> None:
    update_training_performance()
    update_checkpointing()
    write_tests()


if __name__ == "__main__":
    main()
