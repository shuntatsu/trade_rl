from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from trade_rl.rl.tensorboard_logging import build_tensorboard_metrics_callback


class FakeLogger:
    def __init__(self) -> None:
        self.values: dict[str, float] = {}

    def record(self, key: str, value: float, *args: object, **kwargs: object) -> None:
        del args, kwargs
        self.values[key] = value


def test_shared_target_head_records_common_action_stages_without_gate_metric() -> None:
    callback = build_tensorboard_metrics_callback(enabled=True)
    assert callback is not None
    logger = FakeLogger()
    outputs = SimpleNamespace(
        change_intensity=None,
        current_weights=torch.tensor([[0.1, -0.1]]),
        deterministic_actions=torch.tensor([[0.1, -0.1]]),
        active_mask=torch.tensor([[True, True]]),
    )
    policy = SimpleNamespace(action_stage_outputs=lambda observations: outputs)
    callback.model = SimpleNamespace(logger=logger, policy=policy)
    callback.locals = {
        "rewards": (),
        "obs_tensor": {"current_weights": torch.tensor([[0.1, -0.1]])},
        "actions": np.array([[0.4, -0.2]]),
        "infos": [
            {
                "action_path": SimpleNamespace(
                    submitted_order_target=np.array([0.3, -0.1])
                ),
                "effective_filled_weights": np.array([0.25, -0.05]),
            }
        ],
    }

    assert callback._on_step()
    callback._on_rollout_end()

    assert "trade_rl/change_intensity_mean" not in logger.values
    assert logger.values["trade_rl/deterministic_change_l1_mean"] == pytest.approx(0.0)
    assert logger.values["trade_rl/exploration_l1_mean"] == pytest.approx(0.4)
    assert logger.values["trade_rl/sampled_change_l1_mean"] == pytest.approx(0.4)
    assert logger.values["trade_rl/submission_l1_mean"] == pytest.approx(0.2)
    assert logger.values["trade_rl/effective_action_l1_mean"] == pytest.approx(0.3)


def test_tensorboard_uses_soft_rollout_mean_for_exploration_telemetry() -> None:
    callback = build_tensorboard_metrics_callback(enabled=True)
    assert callback is not None
    logger = FakeLogger()
    calls: list[str] = []
    hard_outputs = SimpleNamespace(
        change_intensity=torch.tensor([[0.0, 0.0]]),
        current_weights=torch.tensor([[0.1, -0.1]]),
        deterministic_actions=torch.tensor([[0.1, -0.1]]),
        active_mask=torch.tensor([[True, True]]),
    )
    smooth_outputs = SimpleNamespace(
        change_intensity=torch.tensor([[0.25, 0.25]]),
        current_weights=torch.tensor([[0.1, -0.1]]),
        deterministic_actions=torch.tensor([[0.2, -0.1]]),
        active_mask=torch.tensor([[True, True]]),
    )

    def rollout_outputs(observations: object) -> object:
        del observations
        calls.append("rollout")
        return smooth_outputs

    policy = SimpleNamespace(
        action_stage_outputs=lambda observations: hard_outputs,
        rollout_action_stage_outputs=rollout_outputs,
    )
    callback.model = SimpleNamespace(logger=logger, policy=policy)
    callback.locals = {
        "rewards": (),
        "obs_tensor": {"current_weights": torch.tensor([[0.1, -0.1]])},
        "actions": np.array([[0.4, -0.2]]),
        "infos": [{}],
    }

    assert callback._on_step()
    callback._on_rollout_end()

    assert calls == ["rollout"]
    assert logger.values["trade_rl/change_intensity_mean"] == pytest.approx(0.25)
    assert logger.values["trade_rl/deterministic_change_l1_mean"] == pytest.approx(0.1)
    assert logger.values["trade_rl/exploration_l1_mean"] == pytest.approx(0.3)
    assert logger.values["trade_rl/sampled_change_l1_mean"] == pytest.approx(0.4)
