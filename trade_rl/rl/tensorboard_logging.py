"""Project-specific finite scalar aggregation for SB3 TensorBoard logs."""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

import numpy as np

from trade_rl.rl.action_telemetry import hierarchical_action_stage_metrics

if TYPE_CHECKING:
    from stable_baselines3.common.callbacks import BaseCallback

_TAGS = (
    "trade_rl/reward_mean",
    "trade_rl/portfolio_value_mean",
    "trade_rl/baseline_portfolio_value_mean",
    "trade_rl/drawdown_mean",
    "trade_rl/interval_cost_mean",
    "trade_rl/reward_growth_raw_mean",
    "trade_rl/reward_absolute_component_mean",
    "trade_rl/reward_excess_component_mean",
    "trade_rl/reward_baseline_penalty_weighted_mean",
    "trade_rl/reward_drawdown_penalty_weighted_mean",
    "trade_rl/reward_projection_penalty_weighted_mean",
    "trade_rl/reward_terminal_penalty_weighted_mean",
    "trade_rl/reward_margin_penalty_weighted_mean",
    "trade_rl/reward_total_raw_mean",
    "trade_rl/rolling_growth_gap_mean",
    "trade_rl/action_abs_mean",
    "trade_rl/action_abs_max",
    "trade_rl/change_intensity_mean",
    "trade_rl/deterministic_change_l1_mean",
    "trade_rl/exploration_l1_mean",
    "trade_rl/sampled_change_l1_mean",
    "trade_rl/submission_l1_mean",
    "trade_rl/effective_action_l1_mean",
)
_ACTION_STAGE_TAGS = {
    "deterministic_change_l1": "trade_rl/deterministic_change_l1_mean",
    "exploration_l1": "trade_rl/exploration_l1_mean",
    "sampled_change_l1": "trade_rl/sampled_change_l1_mean",
    "submission_l1": "trade_rl/submission_l1_mean",
    "effective_action_l1": "trade_rl/effective_action_l1_mean",
}
_INFO_MEAN_TAGS = {
    "trade_rl/portfolio_value_mean": "portfolio_value_after",
    "trade_rl/baseline_portfolio_value_mean": "baseline_portfolio_value_after",
    "trade_rl/drawdown_mean": "drawdown_after",
    "trade_rl/interval_cost_mean": "interval_cost",
    "trade_rl/reward_growth_raw_mean": "reward_growth_raw",
    "trade_rl/reward_absolute_component_mean": "reward_absolute_component",
    "trade_rl/reward_excess_component_mean": "reward_excess_component",
    "trade_rl/reward_baseline_penalty_weighted_mean": (
        "reward_baseline_penalty_weighted"
    ),
    "trade_rl/reward_drawdown_penalty_weighted_mean": (
        "reward_drawdown_penalty_weighted"
    ),
    "trade_rl/reward_projection_penalty_weighted_mean": (
        "reward_projection_penalty_weighted"
    ),
    "trade_rl/reward_terminal_penalty_weighted_mean": (
        "reward_terminal_penalty_weighted"
    ),
    "trade_rl/reward_margin_penalty_weighted_mean": (
        "reward_margin_penalty_weighted"
    ),
    "trade_rl/reward_total_raw_mean": "reward_total_raw",
    "trade_rl/rolling_growth_gap_mean": "rolling_growth_gap",
}


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

            infos = self.locals.get("infos", ())
            info_items = tuple(infos) if isinstance(infos, (list, tuple)) else ()
            hierarchical_effective_indices: set[int] = set()
            if self.n_calls % log_interval == 0:
                observations = self.locals.get("obs_tensor")
                policy = getattr(self.model, "policy", None)
                output_factory = getattr(policy, "action_stage_outputs", None)
                if not callable(output_factory):
                    output_factory = getattr(
                        policy,
                        "hierarchical_actor_outputs",
                        None,
                    )
                if observations is not None and callable(output_factory):
                    import torch

                    with torch.no_grad():
                        outputs = output_factory(observations)
                    raw_intensity = getattr(outputs, "change_intensity", None)
                    intensity = (
                        None
                        if raw_intensity is None
                        else raw_intensity.detach().cpu().numpy()
                    )
                    current = outputs.current_weights.detach().cpu().numpy()
                    raw_deterministic = getattr(
                        outputs,
                        "deterministic_actions",
                        getattr(outputs, "composed_actions", None),
                    )
                    if raw_deterministic is None:
                        return True
                    deterministic = raw_deterministic.detach().cpu().numpy()
                    try:
                        sampled_matrix = np.asarray(
                            self.locals.get("actions", ()),
                            dtype=np.float64,
                        )
                    except (TypeError, ValueError):
                        sampled_matrix = np.empty((0,), dtype=np.float64)
                    if (
                        sampled_matrix.shape == deterministic.shape
                        and current.shape == deterministic.shape
                    ):
                        if intensity is not None:
                            self._extend("trade_rl/change_intensity_mean", intensity)
                        for index, (
                            current_row,
                            deterministic_row,
                            sampled_row,
                        ) in enumerate(
                            zip(current, deterministic, sampled_matrix, strict=True)
                        ):
                            info = (
                                info_items[index] if index < len(info_items) else None
                            )
                            action_path = (
                                info.get("action_path")
                                if isinstance(info, dict)
                                else None
                            )
                            submitted_order = getattr(
                                action_path,
                                "submitted_order_target",
                                None,
                            )
                            effective = (
                                info.get("effective_filled_weights")
                                if isinstance(info, dict)
                                else None
                            )
                            try:
                                metrics = hierarchical_action_stage_metrics(
                                    current_weights=current_row,
                                    deterministic_composed=deterministic_row,
                                    sampled_policy_action=sampled_row,
                                    submitted_order_target=submitted_order,
                                    effective_filled_weights=effective,
                                )
                            except ValueError:
                                continue
                            for metric_name, metric_value in metrics.items():
                                self._extend(
                                    _ACTION_STAGE_TAGS[metric_name],
                                    metric_value,
                                )
                            if "effective_action_l1" in metrics:
                                hierarchical_effective_indices.add(index)

            for index, info in enumerate(info_items):
                if not isinstance(info, dict):
                    continue
                for tag, info_key in _INFO_MEAN_TAGS.items():
                    self._extend(tag, info.get(info_key, ()))
                if index not in hierarchical_effective_indices:
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
