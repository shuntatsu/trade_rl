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


def test_tensorboard_callback_aggregates_finite_rollout_metrics() -> None:
    callback = build_tensorboard_metrics_callback(enabled=True)
    assert callback is not None
    logger = FakeLogger()
    callback.model = SimpleNamespace(logger=logger)
    callback.locals = {
        "rewards": np.array([1.0, 3.0]),
        "actions": np.array([[-0.25, 0.75]]),
        "infos": [
            {
                "portfolio_value_after": 101.0,
                "drawdown_after": 0.1,
                "interval_cost": 0.5,
            },
            {
                "portfolio_value_after": 103.0,
                "drawdown_after": 0.2,
                "interval_cost": 0.7,
            },
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


def test_tensorboard_callback_records_low_gate_exploration_and_action_stages() -> None:
    callback = build_tensorboard_metrics_callback(enabled=True)
    assert callback is not None
    logger = FakeLogger()
    outputs = SimpleNamespace(
        change_intensity=torch.tensor([[0.0, 0.0]]),
        current_weights=torch.tensor([[0.1, -0.1]]),
        composed_actions=torch.tensor([[0.1, -0.1]]),
    )
    policy = SimpleNamespace(hierarchical_actor_outputs=lambda observations: outputs)
    callback.model = SimpleNamespace(logger=logger, policy=policy)
    callback.locals = {
        "rewards": (),
        "obs_tensor": {"current_weights": torch.tensor([[0.1, -0.1]])},
        "actions": np.array([[0.4, -0.2]]),
        "infos": [
            {
                "sampled_policy_action": np.array([0.4, -0.2]),
                "submitted_order_target": np.array([0.3, -0.1]),
                "effective_filled_weights": np.array([0.25, -0.05]),
            }
        ],
    }

    assert callback._on_step()
    callback._on_rollout_end()

    assert logger.values["trade_rl/change_intensity_mean"] == pytest.approx(0.0)
    assert logger.values["trade_rl/deterministic_change_l1_mean"] == pytest.approx(0.0)
    assert logger.values["trade_rl/exploration_l1_mean"] == pytest.approx(0.4)
    assert logger.values["trade_rl/sampled_change_l1_mean"] == pytest.approx(0.4)
    assert logger.values["trade_rl/submission_l1_mean"] == pytest.approx(0.2)
    assert logger.values["trade_rl/effective_action_l1_mean"] == pytest.approx(0.3)


def test_tensorboard_callback_ignores_noncanonical_info_aliases() -> None:
    callback = build_tensorboard_metrics_callback(enabled=True)
    assert callback is not None
    logger = FakeLogger()
    callback.model = SimpleNamespace(logger=logger)
    callback.locals = {
        "rewards": (),
        "actions": (),
        "infos": [{"portfolio_value": 101.0, "drawdown": 0.1}],
    }
    assert callback._on_step()
    callback._on_rollout_end()
    assert logger.values == {}


def test_tensorboard_callback_skips_missing_malformed_and_non_finite_values() -> None:
    callback = build_tensorboard_metrics_callback(enabled=True)
    assert callback is not None
    logger = FakeLogger()
    callback.model = SimpleNamespace(logger=logger)
    callback.locals = {
        "rewards": [float("nan"), "bad"],
        "actions": None,
        "infos": [{"portfolio_value_after": float("inf")}, object()],
    }
    assert callback._on_step()
    callback._on_rollout_end()
    assert logger.values == {}


def test_tensorboard_callback_is_optional_and_interval_validated() -> None:
    assert build_tensorboard_metrics_callback(enabled=False) is None
    with pytest.raises(ValueError, match="log_interval"):
        build_tensorboard_metrics_callback(enabled=True, log_interval=0)
