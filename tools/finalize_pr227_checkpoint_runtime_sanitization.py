from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _replace_once(path: str, old: str, new: str) -> None:
    target = ROOT / path
    text = target.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{path}: expected one anchor, found {count}")
    updated = text.replace(old, new, 1)
    ast.parse(updated, filename=path)
    target.write_text(updated, encoding="utf-8")


def _update_training_performance_registry() -> None:
    path = "trade_rl/rl/training_performance.py"
    _replace_once(
        path,
        '''_METRIC_NAMES = (
    "collect_rollouts",
    "optimization",
    "environment_step",
    "feature_extraction",
    "sequence_reconstruction",
    "sequence_tensor_conversion",
)
''',
        '''_METRIC_NAMES = (
    "collect_rollouts",
    "optimization",
    "environment_step",
    "feature_extraction",
    "sequence_reconstruction",
    "sequence_tensor_conversion",
)
TRAINING_RUNTIME_PATCHES_ATTRIBUTE = "_trade_rl_training_runtime_patches"
''',
    )
    _replace_once(
        path,
        '''        patches: list[tuple[object, str, bool, object | None]] = []

        def patch(owner: object | None, name: str, metric_name: str) -> None:
''',
        '''        patches: list[tuple[object, str, bool, object | None]] = []
        missing = object()
        previous_registry = getattr(
            model,
            TRAINING_RUNTIME_PATCHES_ATTRIBUTE,
            missing,
        )
        if previous_registry is not missing:
            raise RuntimeError("training performance instrumentation is already active")

        def patch(owner: object | None, name: str, metric_name: str) -> None:
''',
    )
    _replace_once(
        path,
        '''        patch(policy, "extract_features", "feature_extraction")
        environment = getattr(model, "env", None)
        patch(environment, "step", "environment_step")
        try:
            yield
        finally:
            for owner, name, had_local, local_value in reversed(patches):
                if had_local:
                    setattr(owner, name, local_value)
                else:
                    delattr(owner, name)
''',
        '''        patch(policy, "extract_features", "feature_extraction")
        environment = getattr(model, "env", None)
        patch(environment, "step", "environment_step")
        setattr(model, TRAINING_RUNTIME_PATCHES_ATTRIBUTE, tuple(patches))
        try:
            yield
        finally:
            for owner, name, had_local, local_value in reversed(patches):
                if had_local:
                    setattr(owner, name, local_value)
                else:
                    delattr(owner, name)
            if previous_registry is missing:
                delattr(model, TRAINING_RUNTIME_PATCHES_ATTRIBUTE)
            else:
                setattr(
                    model,
                    TRAINING_RUNTIME_PATCHES_ATTRIBUTE,
                    previous_registry,
                )
''',
    )
    _replace_once(
        path,
        '''__all__ = [
    "TrainingPerformanceEvidence",
''',
        '''__all__ = [
    "TRAINING_RUNTIME_PATCHES_ATTRIBUTE",
    "TrainingPerformanceEvidence",
''',
    )


def _update_checkpoint_save_boundary() -> None:
    path = "trade_rl/rl/checkpointing.py"
    _replace_once(
        path,
        '''from trade_rl.rl.training_telemetry import build_training_telemetry_callback
''',
        '''from trade_rl.rl.training_performance import TRAINING_RUNTIME_PATCHES_ATTRIBUTE
from trade_rl.rl.training_telemetry import build_training_telemetry_callback
''',
    )
    _replace_once(
        path,
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
    """Save without serializing dataset-bound or temporary training runtime state."""

    missing = object()
    original_rollout_kwargs = getattr(model, "rollout_buffer_kwargs", missing)
    if (
        isinstance(original_rollout_kwargs, dict)
        and "sequence_reconstructor" in original_rollout_kwargs
    ):
        sanitized = {
            key: value
            for key, value in original_rollout_kwargs.items()
            if key != "sequence_reconstructor"
        }
        setattr(model, "rollout_buffer_kwargs", sanitized)

    raw_patches = getattr(model, TRAINING_RUNTIME_PATCHES_ATTRIBUTE, missing)
    suspended: list[tuple[object, str, object]] = []
    if raw_patches is not missing:
        if not isinstance(raw_patches, tuple):
            raise TypeError("training runtime patch registry must be a tuple")
        try:
            for entry in reversed(raw_patches):
                if not isinstance(entry, tuple) or len(entry) != 4:
                    raise TypeError("training runtime patch entry is invalid")
                owner, name, had_local, local_value = entry
                if not isinstance(name, str) or not name:
                    raise TypeError("training runtime patch name is invalid")
                if not isinstance(had_local, bool):
                    raise TypeError("training runtime patch locality is invalid")
                namespace = getattr(owner, "__dict__", None)
                if not isinstance(namespace, dict) or name not in namespace:
                    raise RuntimeError("training runtime patch registry is stale")
                suspended.append((owner, name, namespace[name]))
                if had_local:
                    setattr(owner, name, local_value)
                else:
                    delattr(owner, name)
            delattr(model, TRAINING_RUNTIME_PATCHES_ATTRIBUTE)
        except BaseException:
            for owner, name, wrapper in reversed(suspended):
                setattr(owner, name, wrapper)
            raise

    try:
        model.save(target)
    finally:
        if raw_patches is not missing:
            for owner, name, wrapper in reversed(suspended):
                setattr(owner, name, wrapper)
            setattr(model, TRAINING_RUNTIME_PATCHES_ATTRIBUTE, raw_patches)
        if original_rollout_kwargs is not missing:
            setattr(model, "rollout_buffer_kwargs", original_rollout_kwargs)
''',
    )


def _write_regression_test() -> None:
    path = ROOT / "tests" / "rl" / "test_checkpoint_runtime_sanitization.py"
    if path.exists():
        raise FileExistsError(path)
    path.write_text(
        '''from __future__ import annotations

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
''',
        encoding="utf-8",
    )
    ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def main() -> None:
    _update_training_performance_registry()
    _update_checkpoint_save_boundary()
    _write_regression_test()


if __name__ == "__main__":
    main()
