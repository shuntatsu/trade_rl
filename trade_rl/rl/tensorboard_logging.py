"""Project-specific finite scalar aggregation for SB3 TensorBoard logs."""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

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
    "trade_rl/change_intensity_mean",
    "trade_rl/exploration_l1_mean",
    "trade_rl/effective_action_l1_mean",
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
    if (
        isinstance(log_interval, bool)
        or not isinstance(log_interval, int)
        or log_interval <= 0
    ):
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
            if self.n_calls % log_interval == 0:
                observations = self.locals.get("obs_tensor")
                output_factory = getattr(
                    getattr(self.model, "policy", None),
                    "hierarchical_actor_outputs",
                    None,
                )
                if observations is not None and callable(output_factory):
                    import torch

                    with torch.no_grad():
                        outputs = output_factory(observations)
                    intensity = outputs.change_intensity.detach().cpu().numpy()
                    deterministic = outputs.composed_actions.detach().cpu().numpy()
                    sampled_matrix = np.asarray(
                        self.locals.get("actions", ()), dtype=np.float64
                    )
                    if sampled_matrix.shape == deterministic.shape:
                        self._extend("trade_rl/change_intensity_mean", intensity)
                        self._extend(
                            "trade_rl/exploration_l1_mean",
                            np.sum(
                                np.abs(sampled_matrix - deterministic), axis=1
                            ),
                        )
            infos = self.locals.get("infos", ())
            if isinstance(infos, (list, tuple)):
                for info in infos:
                    if not isinstance(info, dict):
                        continue
                    self._extend(
                        "trade_rl/portfolio_value_mean",
                        info.get("portfolio_value_after", ()),
                    )
                    self._extend(
                        "trade_rl/drawdown_mean",
                        info.get("drawdown_after", ()),
                    )
                    self._extend(
                        "trade_rl/interval_cost_mean",
                        info.get("interval_cost", ()),
                    )
                    self._extend(
                        "trade_rl/effective_action_l1_mean",
                        info.get("sampled_policy_to_filled_l1", ()),
                    )
            return True

        def _on_rollout_end(self) -> None:
            self._rollouts += 1
            if self._rollouts % log_interval == 0:
                for tag in _TAGS:
                    values = self._values.get(tag, ())
                    if not values:
                        continue
                    aggregate = (
                        max(values)
                        if tag.endswith("action_abs_max")
                        else float(np.mean(values))
                    )
                    if np.isfinite(aggregate):
                        self.logger.record(tag, float(aggregate))
            self._values.clear()

    return TensorBoardMetricsCallback()


__all__ = ["build_tensorboard_metrics_callback"]
