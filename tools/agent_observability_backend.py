from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _write(relative: str, content: str) -> None:
    path = ROOT / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.rstrip() + "\n", encoding="utf-8")


def _replace_once(relative: str, old: str, new: str) -> None:
    path = ROOT / relative
    text = path.read_text(encoding="utf-8")
    if new in text:
        return
    if old not in text:
        raise RuntimeError(f"guarded replacement failed for {relative}: {old[:80]!r}")
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


def _append_once(relative: str, marker: str, content: str) -> None:
    path = ROOT / relative
    text = path.read_text(encoding="utf-8")
    if marker in text:
        return
    path.write_text(text.rstrip() + "\n\n" + content.rstrip() + "\n", encoding="utf-8")


def apply() -> None:
    _write(
        "trade_rl/rl/schedules.py",
        '''"""Pure, validated learning-rate schedules for RL algorithms."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Literal

LearningRateScheduleKind = Literal["constant", "linear", "cosine"]


def build_learning_rate_schedule(
    *,
    initial_rate: float,
    final_ratio: float,
    kind: str,
) -> float | Callable[[float], float]:
    """Build an SB3-compatible schedule from progress remaining in ``[0, 1]``."""

    if not math.isfinite(initial_rate) or initial_rate <= 0.0:
        raise ValueError("initial_rate must be finite and positive")
    if not math.isfinite(final_ratio) or not 0.0 < final_ratio <= 1.0:
        raise ValueError("final_ratio must be within (0, 1]")
    if kind not in {"constant", "linear", "cosine"}:
        raise ValueError("kind must be constant, linear, or cosine")
    if kind == "constant":
        return float(initial_rate)

    def schedule(progress_remaining: float) -> float:
        if (
            not math.isfinite(progress_remaining)
            or not 0.0 <= progress_remaining <= 1.0
        ):
            raise ValueError("progress_remaining must be finite and within [0, 1]")
        if kind == "linear":
            multiplier = final_ratio + (1.0 - final_ratio) * progress_remaining
        else:
            completed = 1.0 - progress_remaining
            multiplier = final_ratio + (1.0 - final_ratio) * (
                0.5 * (1.0 + math.cos(math.pi * completed))
            )
        rate = initial_rate * multiplier
        if not math.isfinite(rate) or rate <= 0.0:
            raise ValueError("resolved learning rate must be finite and positive")
        return float(rate)

    return schedule


__all__ = ["LearningRateScheduleKind", "build_learning_rate_schedule"]
''',
    )

    _write(
        "trade_rl/rl/tensorboard_logging.py",
        '''"""Project-specific finite scalar aggregation for SB3 TensorBoard logs."""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from stable_baselines3.common.callbacks import BaseCallback

_TAGS = (
    "trade_rl/reward_mean",
    "trade_rl/portfolio_value_mean",
    "trade_rl/drawdown_mean",
    "trade_rl/interval_cost_mean",
    "trade_rl/action_abs_mean",
    "trade_rl/action_abs_max",
)


def _finite_values(value: object) -> tuple[float, ...]:
    try:
        values = np.asarray(value, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        return ()
    return tuple(float(item) for item in values[np.isfinite(values)])


def build_tensorboard_metrics_callback(
    *,
    enabled: bool,
    log_interval: int = 1,
) -> BaseCallback | None:
    """Return a callback that records only the project's explicit scalar allowlist."""

    if not isinstance(enabled, bool):
        raise ValueError("enabled must be a boolean")
    if isinstance(log_interval, bool) or not isinstance(log_interval, int) or log_interval <= 0:
        raise ValueError("log_interval must be a positive integer")
    if not enabled:
        return None

    from stable_baselines3.common.callbacks import BaseCallback

    class TensorBoardMetricsCallback(BaseCallback):
        def __init__(self) -> None:
            super().__init__(verbose=0)
            self._rollouts = 0
            self._values: dict[str, list[float]] = defaultdict(list)

        def _extend(self, tag: str, value: object) -> None:
            self._values[tag].extend(_finite_values(value))

        def _on_step(self) -> bool:
            self._extend("trade_rl/reward_mean", self.locals.get("rewards", ()))
            actions = _finite_values(self.locals.get("actions", ()))
            if actions:
                absolute = tuple(abs(item) for item in actions)
                self._values["trade_rl/action_abs_mean"].extend(absolute)
                self._values["trade_rl/action_abs_max"].append(max(absolute))
            infos = self.locals.get("infos", ())
            if isinstance(infos, (list, tuple)):
                for info in infos:
                    if not isinstance(info, dict):
                        continue
                    self._extend(
                        "trade_rl/portfolio_value_mean",
                        info.get("portfolio_value", ()),
                    )
                    self._extend(
                        "trade_rl/drawdown_mean",
                        info.get("drawdown", ()),
                    )
                    self._extend(
                        "trade_rl/interval_cost_mean",
                        info.get("interval_cost", ()),
                    )
            return True

        def _on_rollout_end(self) -> None:
            self._rollouts += 1
            if self._rollouts % log_interval == 0:
                for tag in _TAGS:
                    values = self._values.get(tag, ())
                    if not values:
                        continue
                    aggregate = max(values) if tag.endswith("action_abs_max") else float(np.mean(values))
                    if np.isfinite(aggregate):
                        self.logger.record(tag, float(aggregate))
            self._values.clear()

    return TensorBoardMetricsCallback()


__all__ = ["build_tensorboard_metrics_callback"]
''',
    )

    _replace_once(
        "trade_rl/rl/training.py",
        "    learning_rate: float = 3e-4\n",
        "    learning_rate: float = 3e-4\n"
        "    learning_rate_schedule: str = \"constant\"\n"
        "    learning_rate_final_ratio: float = 0.1\n"
        "    tensorboard_enabled: bool = False\n"
        "    tensorboard_log_interval: int = 1\n",
    )
    _replace_once(
        "trade_rl/rl/training.py",
        '            ("n_epochs", self.n_epochs),\n',
        '            ("n_epochs", self.n_epochs),\n'
        '            ("tensorboard_log_interval", self.tensorboard_log_interval),\n',
    )
    _replace_once(
        "trade_rl/rl/training.py",
        '        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0.0:\n'
        '            raise ValueError("learning_rate must be finite and positive")\n',
        '        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0.0:\n'
        '            raise ValueError("learning_rate must be finite and positive")\n'
        '        if self.learning_rate_schedule not in {"constant", "linear", "cosine"}:\n'
        '            raise ValueError(\n'
        '                "learning_rate_schedule must be constant, linear, or cosine"\n'
        '            )\n'
        '        if (\n'
        '            not math.isfinite(self.learning_rate_final_ratio)\n'
        '            or not 0.0 < self.learning_rate_final_ratio <= 1.0\n'
        '        ):\n'
        '            raise ValueError("learning_rate_final_ratio must be within (0, 1]")\n'
        '        if not isinstance(self.tensorboard_enabled, bool):\n'
        '            raise ValueError("tensorboard_enabled must be a boolean")\n',
    )
    _replace_once(
        "trade_rl/rl/training.py",
        '            "learning_rate": self.learning_rate,\n',
        '            "learning_rate": self.learning_rate,\n'
        '            "learning_rate_final_ratio": self.learning_rate_final_ratio,\n'
        '            "learning_rate_schedule": self.learning_rate_schedule,\n',
    )
    _replace_once(
        "trade_rl/rl/training.py",
        '            "target_kl": self.target_kl,\n',
        '            "target_kl": self.target_kl,\n'
        '            "tensorboard_enabled": self.tensorboard_enabled,\n'
        '            "tensorboard_log_interval": self.tensorboard_log_interval,\n',
    )

    _replace_once(
        "trade_rl/rl/algorithm_configs.py",
        "    learning_rate: float\n    batch_size: int\n",
        "    learning_rate: float\n"
        "    learning_rate_schedule: str\n"
        "    learning_rate_final_ratio: float\n"
        "    batch_size: int\n",
    )
    for anchor in (
        "            learning_rate=source.learning_rate,\n            batch_size=source.batch_size,\n",
        "        learning_rate=source.learning_rate,\n        batch_size=source.batch_size,\n",
    ):
        _replace_once(
            "trade_rl/rl/algorithm_configs.py",
            anchor,
            anchor.replace(
                "batch_size=source.batch_size,",
                "learning_rate_schedule=source.learning_rate_schedule,\n"
                + ("            " if anchor.startswith("            ") else "        ")
                + "learning_rate_final_ratio=source.learning_rate_final_ratio,\n"
                + ("            " if anchor.startswith("            ") else "        ")
                + "batch_size=source.batch_size,",
            ),
        )
    # SAC and TQC explicitly repeat the common fields.
    text = (ROOT / "trade_rl/rl/algorithm_configs.py").read_text(encoding="utf-8")
    explicit = "            learning_rate=source.learning_rate,\n            batch_size=source.batch_size,\n"
    replacement = (
        "            learning_rate=source.learning_rate,\n"
        "            learning_rate_schedule=source.learning_rate_schedule,\n"
        "            learning_rate_final_ratio=source.learning_rate_final_ratio,\n"
        "            batch_size=source.batch_size,\n"
    )
    text = text.replace(explicit, replacement)
    (ROOT / "trade_rl/rl/algorithm_configs.py").write_text(text, encoding="utf-8")

    _replace_once(
        "pyproject.toml",
        '    "torch==2.3.1",\n',
        '    "torch==2.3.1",\n    "tensorboard>=2.17,<3",\n',
    )
    _replace_once(
        "pyproject.toml",
        '    "torch.*",\n',
        '    "torch.*",\n    "tensorboard.*",\n',
    )

    _replace_once(
        "trade_rl/integrations/sb3_training.py",
        "from trade_rl.rl.rollout_memory import (\n",
        "from trade_rl.rl.schedules import build_learning_rate_schedule\n"
        "from trade_rl.rl.tensorboard_logging import (\n"
        "    build_tensorboard_metrics_callback,\n"
        ")\n"
        "from trade_rl.rl.rollout_memory import (\n",
    )
    _replace_once(
        "trade_rl/integrations/sb3_training.py",
        '                "learning_rate": algorithm_config.learning_rate,\n',
        '                "learning_rate": build_learning_rate_schedule(\n'
        '                    initial_rate=algorithm_config.learning_rate,\n'
        '                    final_ratio=algorithm_config.learning_rate_final_ratio,\n'
        '                    kind=algorithm_config.learning_rate_schedule,\n'
        '                ),\n',
    )
    _replace_once(
        "trade_rl/integrations/sb3_training.py",
        '                "verbose": self.verbose,\n            }\n',
        '                "verbose": self.verbose,\n            }\n'
        '            if config.tensorboard_enabled:\n'
        '                common["tensorboard_log"] = str(output_path.parent / "tensorboard")\n',
    )
    old_callback = '''            callback = build_checkpoint_callback(
                checkpoint_root=output_path.parent / "checkpoints",
                algorithm=config.algorithm,
                seed=seed,
                interval_steps=config.resolved_checkpoint_interval,
                max_checkpoints=config.max_checkpoints,
                environment_digest=str(identity["environment_digest"]),
                training_config_digest=content_digest(config.digest_payload()),
            )
            remaining_timesteps = config.timesteps
            if resume_manifest is not None:
                remaining_timesteps = max(
                    0, config.timesteps - resume_manifest.observed_timestep
                )
            if remaining_timesteps > 0:
                learn_kwargs: dict[str, object] = {
                    "total_timesteps": remaining_timesteps,
                    "callback": callback,
                }
                if resume_manifest is not None:
                    learn_kwargs["reset_num_timesteps"] = False
                model.learn(**learn_kwargs)
'''
    new_callback = '''            remaining_timesteps = config.timesteps
            starting_timestep = 0
            if resume_manifest is not None:
                starting_timestep = resume_manifest.observed_timestep
                remaining_timesteps = max(0, config.timesteps - starting_timestep)
            checkpoint_callback = build_checkpoint_callback(
                checkpoint_root=output_path.parent / "checkpoints",
                algorithm=config.algorithm,
                seed=seed,
                interval_steps=config.resolved_checkpoint_interval,
                max_checkpoints=config.max_checkpoints,
                total_timesteps=config.timesteps,
                starting_timestep=starting_timestep,
                environment_digest=str(identity["environment_digest"]),
                training_config_digest=content_digest(config.digest_payload()),
            )
            metrics_callback = build_tensorboard_metrics_callback(
                enabled=config.tensorboard_enabled,
                log_interval=config.tensorboard_log_interval,
            )
            callback: object = checkpoint_callback
            if metrics_callback is not None:
                from stable_baselines3.common.callbacks import CallbackList

                callback = CallbackList([checkpoint_callback, metrics_callback])
            if config.tensorboard_enabled:
                model.tensorboard_log = str(output_path.parent / "tensorboard")
            if remaining_timesteps > 0:
                learn_kwargs: dict[str, object] = {
                    "total_timesteps": remaining_timesteps,
                    "callback": callback,
                }
                if config.tensorboard_enabled:
                    learn_kwargs["tb_log_name"] = f"seed-{seed}-{config.algorithm}"
                if resume_manifest is not None:
                    learn_kwargs["reset_num_timesteps"] = False
                model.learn(**learn_kwargs)
'''
    _replace_once("trade_rl/integrations/sb3_training.py", old_callback, new_callback)

    checkpoint_path = ROOT / "trade_rl/rl/checkpointing.py"
    checkpoint_text = checkpoint_path.read_text(encoding="utf-8")
    marker = "\ndef build_checkpoint_callback(\n"
    if "def planned_checkpoint_steps(" not in checkpoint_text:
        planned = '''
def planned_checkpoint_steps(
    *,
    total_timesteps: int,
    interval_steps: int,
    max_checkpoints: int,
) -> tuple[int, ...]:
    """Select deterministic requested steps across the complete training horizon."""

    if isinstance(total_timesteps, bool) or not isinstance(total_timesteps, int) or total_timesteps <= 0:
        raise ValueError("total_timesteps must be a positive integer")
    if isinstance(interval_steps, bool) or not isinstance(interval_steps, int) or interval_steps < 0:
        raise ValueError("interval_steps must be a non-negative integer")
    if isinstance(max_checkpoints, bool) or not isinstance(max_checkpoints, int) or max_checkpoints <= 0:
        raise ValueError("max_checkpoints must be a positive integer")
    if interval_steps == 0:
        return ()
    candidates = tuple(range(interval_steps, total_timesteps, interval_steps))
    if len(candidates) <= max_checkpoints:
        return candidates
    if max_checkpoints == 1:
        return (candidates[-1],)
    positions = tuple(
        round(index * (len(candidates) - 1) / (max_checkpoints - 1))
        for index in range(max_checkpoints)
    )
    return tuple(candidates[position] for position in positions)

'''
        checkpoint_text = checkpoint_text.replace(marker, planned + marker, 1)
    start = checkpoint_text.index("def build_checkpoint_callback(\n")
    end = checkpoint_text.index("\n\n__all__ = [", start)
    replacement = '''def build_checkpoint_callback(
    *,
    checkpoint_root: Path,
    algorithm: str,
    seed: int,
    interval_steps: int,
    max_checkpoints: int,
    total_timesteps: int,
    starting_timestep: int = 0,
    environment_digest: str,
    training_config_digest: str,
) -> Any:
    """Build full-horizon checkpoint and sampled Studio telemetry callbacks lazily."""

    if (
        isinstance(starting_timestep, bool)
        or not isinstance(starting_timestep, int)
        or starting_timestep < 0
        or starting_timestep > total_timesteps
    ):
        raise ValueError("starting_timestep must be within the training horizon")
    from stable_baselines3.common.callbacks import BaseCallback, CallbackList

    checkpoint_root = Path(checkpoint_root)
    telemetry_callback = build_training_telemetry_callback(
        path=checkpoint_root.parent / "telemetry" / "training-telemetry.jsonl",
        seed=seed,
    )
    planned = tuple(
        step
        for step in planned_checkpoint_steps(
            total_timesteps=total_timesteps,
            interval_steps=interval_steps,
            max_checkpoints=max_checkpoints,
        )
        if step > starting_timestep
    )
    if not planned:
        return telemetry_callback

    class AtomicCheckpointCallback(BaseCallback):
        def __init__(self) -> None:
            super().__init__(verbose=0)
            self.cursor = 0

        def _on_step(self) -> bool:
            if self.cursor >= len(planned):
                return True
            observed = int(self.model.num_timesteps)
            requested = planned[self.cursor]
            if observed < requested:
                return True
            publish_checkpoint(
                model=self.model,
                checkpoint_root=checkpoint_root,
                algorithm=algorithm,
                seed=seed,
                requested_timestep=requested,
                observed_timestep=observed,
                environment_digest=environment_digest,
                training_config_digest=training_config_digest,
            )
            self.cursor += 1
            return True

    return CallbackList([AtomicCheckpointCallback(), telemetry_callback])
'''
    checkpoint_text = checkpoint_text[:start] + replacement + checkpoint_text[end:]
    checkpoint_text = checkpoint_text.replace(
        '    "load_checkpoint_manifest",\n',
        '    "load_checkpoint_manifest",\n    "planned_checkpoint_steps",\n',
    )
    checkpoint_path.write_text(checkpoint_text, encoding="utf-8")

    _write(
        "tests/rl/test_schedules.py",
        '''from __future__ import annotations

import pytest

from trade_rl.rl.schedules import build_learning_rate_schedule


def test_linear_schedule_uses_progress_remaining() -> None:
    schedule = build_learning_rate_schedule(
        initial_rate=1.2e-4,
        final_ratio=0.1,
        kind="linear",
    )
    assert callable(schedule)
    assert schedule(1.0) == pytest.approx(1.2e-4)
    assert schedule(0.0) == pytest.approx(1.2e-5)


def test_cosine_schedule_uses_exact_endpoints() -> None:
    schedule = build_learning_rate_schedule(
        initial_rate=1.2e-4,
        final_ratio=0.1,
        kind="cosine",
    )
    assert callable(schedule)
    assert schedule(1.0) == pytest.approx(1.2e-4)
    assert schedule(0.0) == pytest.approx(1.2e-5)


def test_constant_schedule_returns_float() -> None:
    assert build_learning_rate_schedule(
        initial_rate=1.2e-4,
        final_ratio=0.1,
        kind="constant",
    ) == pytest.approx(1.2e-4)


@pytest.mark.parametrize("kind", ["linear", "cosine"])
def test_schedule_rejects_progress_outside_unit_interval(kind: str) -> None:
    schedule = build_learning_rate_schedule(
        initial_rate=1.2e-4,
        final_ratio=0.1,
        kind=kind,
    )
    assert callable(schedule)
    with pytest.raises(ValueError, match="progress_remaining"):
        schedule(-0.01)
    with pytest.raises(ValueError, match="progress_remaining"):
        schedule(1.01)


@pytest.mark.parametrize(
    ("initial_rate", "final_ratio", "kind"),
    [(0.0, 0.1, "linear"), (1e-4, 0.0, "linear"), (1e-4, 1.1, "linear"), (1e-4, 0.1, "step")],
)
def test_schedule_rejects_invalid_configuration(
    initial_rate: float,
    final_ratio: float,
    kind: str,
) -> None:
    with pytest.raises(ValueError):
        build_learning_rate_schedule(
            initial_rate=initial_rate,
            final_ratio=final_ratio,
            kind=kind,
        )
''',
    )

    _write(
        "tests/rl/test_tensorboard_logging.py",
        '''from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from trade_rl.rl.tensorboard_logging import build_tensorboard_metrics_callback


class FakeLogger:
    def __init__(self) -> None:
        self.values: dict[str, float] = {}

    def record(self, key: str, value: float, *args: object, **kwargs: object) -> None:
        del args, kwargs
        self.values[key] = value


def test_tensorboard_callback_aggregates_finite_rollout_metrics() -> None:
    callback = build_tensorboard_metrics_callback(enabled=True)
    assert callback is not None
    logger = FakeLogger()
    callback.model = SimpleNamespace(logger=logger)
    callback.locals = {
        "rewards": np.array([1.0, 3.0]),
        "actions": np.array([[-0.25, 0.75]]),
        "infos": [
            {"portfolio_value": 101.0, "drawdown": 0.1, "interval_cost": 0.5},
            {"portfolio_value": 103.0, "drawdown": 0.2, "interval_cost": 0.7},
        ],
    }
    assert callback._on_step()
    callback._on_rollout_end()
    assert logger.values["trade_rl/reward_mean"] == pytest.approx(2.0)
    assert logger.values["trade_rl/portfolio_value_mean"] == pytest.approx(102.0)
    assert logger.values["trade_rl/drawdown_mean"] == pytest.approx(0.15)
    assert logger.values["trade_rl/interval_cost_mean"] == pytest.approx(0.6)
    assert logger.values["trade_rl/action_abs_mean"] == pytest.approx(0.5)
    assert logger.values["trade_rl/action_abs_max"] == pytest.approx(0.75)


def test_tensorboard_callback_skips_missing_malformed_and_non_finite_values() -> None:
    callback = build_tensorboard_metrics_callback(enabled=True)
    assert callback is not None
    logger = FakeLogger()
    callback.model = SimpleNamespace(logger=logger)
    callback.locals = {
        "rewards": [float("nan"), "bad"],
        "actions": None,
        "infos": [{"portfolio_value": float("inf")}, object()],
    }
    assert callback._on_step()
    callback._on_rollout_end()
    assert logger.values == {}


def test_tensorboard_callback_is_optional_and_interval_validated() -> None:
    assert build_tensorboard_metrics_callback(enabled=False) is None
    with pytest.raises(ValueError, match="log_interval"):
        build_tensorboard_metrics_callback(enabled=True, log_interval=0)
''',
    )

    _append_once(
        "tests/rl/test_training_config_active_fields.py",
        "test_learning_rate_schedule_and_tensorboard_fields_are_identity_bound",
        '''def test_learning_rate_schedule_and_tensorboard_fields_are_identity_bound() -> None:
    baseline = _config()
    configured = _config(
        learning_rate_schedule="linear",
        learning_rate_final_ratio=0.2,
        tensorboard_enabled=True,
        tensorboard_log_interval=2,
    )
    typed = build_algorithm_config(configured)

    assert typed.learning_rate_schedule == "linear"
    assert typed.learning_rate_final_ratio == pytest.approx(0.2)
    assert configured.digest_payload() != baseline.digest_payload()
    with pytest.raises(ValueError, match="learning_rate_schedule"):
        _config(learning_rate_schedule="step")
    with pytest.raises(ValueError, match="learning_rate_final_ratio"):
        _config(learning_rate_final_ratio=0.0)
    with pytest.raises(ValueError, match="tensorboard_log_interval"):
        _config(tensorboard_log_interval=0)
''',
    )

    _replace_once(
        "tests/rl/test_checkpointing.py",
        "    load_checkpoint_manifest,\n    publish_checkpoint,\n",
        "    load_checkpoint_manifest,\n    planned_checkpoint_steps,\n    publish_checkpoint,\n",
    )
    _append_once(
        "tests/rl/test_checkpointing.py",
        "test_capped_checkpoints_cover_late_training",
        '''def test_capped_checkpoints_cover_late_training() -> None:
    steps = planned_checkpoint_steps(
        total_timesteps=524_288,
        interval_steps=32_768,
        max_checkpoints=8,
    )

    assert len(steps) == 8
    assert steps[0] > 0
    assert steps[-1] >= 450_000
    assert tuple(sorted(steps)) == steps
    assert 524_288 not in steps


def test_checkpoint_plan_skips_completed_steps_on_resume() -> None:
    steps = planned_checkpoint_steps(
        total_timesteps=100,
        interval_steps=20,
        max_checkpoints=10,
    )
    assert tuple(step for step in steps if step > 40) == (60, 80)
''',
    )

    _append_once(
        "tests/integrations/test_sb3_training.py",
        "test_backend_wires_learning_rate_schedule_and_tensorboard",
        '''def test_backend_wires_learning_rate_schedule_and_tensorboard(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}

    class CapturingPPO(FakePPO):
        def __init__(self, *args: object, **kwargs: object) -> None:
            captured["constructor"] = kwargs
            super().__init__(*args, **kwargs)

        def learn(self, **kwargs: object) -> "CapturingPPO":
            captured["learn"] = kwargs
            return super().learn(**kwargs)

    monkeypatch.setattr("stable_baselines3.PPO", CapturingPPO)
    backend = StableBaselines3PPOBackend(lambda: FakeEnvironment())
    config = ResidualTrainingConfig(
        timesteps=8,
        gamma=0.99,
        seeds=(5,),
        n_steps=8,
        batch_size=8,
        device="cpu",
        asset_set_encoder=False,
        learning_rate_schedule="linear",
        tensorboard_enabled=True,
    )
    backend.train(seed=5, config=config, output_path=tmp_path / "member" / "policy.zip")

    constructor = captured["constructor"]
    assert isinstance(constructor, dict)
    assert callable(constructor["learning_rate"])
    assert constructor["tensorboard_log"] == str(tmp_path / "member" / "tensorboard")
    learn = captured["learn"]
    assert isinstance(learn, dict)
    assert learn["tb_log_name"] == "seed-5-ppo"
''',
    )


if __name__ == "__main__":
    apply()
