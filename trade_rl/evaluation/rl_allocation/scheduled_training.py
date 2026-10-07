"""Actual PPO fitting over a declared chronological allocation schedule.

This capability fits only declared train windows. Inference handoff requires a
distinct schedule manifest; reset/window counts are not market experience.
"""

from __future__ import annotations

import importlib
import json
from typing import Any, cast

from trade_rl.artifacts import canonical_json_bytes, content_digest
from trade_rl.evaluation.rl_allocation.env import AllocationTradingEnv
from trade_rl.evaluation.rl_allocation.preprocessing import (
    validate_preprocessing_application,
    validate_training_preprocessing,
)
from trade_rl.evaluation.rl_allocation.training_protocol import (
    AllocationUpdateRecorder,
    construct_protocol_ppo,
    validate_protocol_clock,
)
from trade_rl.evaluation.rl_allocation.training_schedule import (
    AllocationTrainingScheduleEnv,
)
from trade_rl.evaluation.rl_allocation.training_source import allocation_training_source
from trade_rl.strategies.rl.allocation_model import (
    AllocationPPOPolicy,
    validate_allocation_protocol_model,
)
from trade_rl.strategies.rl.allocation_training_protocol import (
    AllocationPPOTrainingProtocol,
)
from trade_rl.strategies.rl.allocation_training_schedule import _native_json


class ScheduledAllocationPPOFit:
    """Detached fit facts and optional inference handoff; no published artifact."""

    def __init__(
        self,
        model: Any,
        receipt: dict[str, object],
        *,
        manifest: dict[str, Any] | None = None,
    ) -> None:
        self.model = model
        self._receipt = canonical_json_bytes(receipt)
        if manifest is not None:
            _native_json(manifest)
        self._manifest = None if manifest is None else canonical_json_bytes(manifest)

    @property
    def receipt(self) -> dict[str, Any]:
        return json.loads(self._receipt)

    @property
    def receipt_digest(self) -> str:
        return content_digest(self.receipt)

    def inference_policy(self) -> AllocationPPOPolicy:
        """Validate frozen scheduled facts/model before existing write-once save."""
        if self._manifest is None:
            raise ValueError("scheduled fit has no frozen inference manifest")
        return AllocationPPOPolicy(self.model, json.loads(self._manifest))


def _preprocessing_fit(
    env: AllocationTrainingScheduleEnv,
    protocol: AllocationPPOTrainingProtocol,
) -> dict[str, object] | None:
    children = env.training_environments
    declarations = [child.feature_preprocessing for child in children]
    if all(value is None for value in declarations):
        return None
    if any(value is None for value in declarations):
        raise ValueError("scheduled training requires one common preprocessing lane")
    declaration = declarations[0]
    assert declaration is not None
    if any(value != declaration for value in declarations[1:]):
        raise ValueError("scheduled training preprocessing declarations differ")

    owners = [
        child
        for child in children
        if child.dataset.dataset_id == declaration.normalizer.source_dataset_id
        and declaration.policy_start_index < child.dataset.n_bars
    ]
    if not owners:
        raise ValueError("scheduled training cannot reconstruct preprocessing source")
    owner = owners[0]
    fit = validate_training_preprocessing(
        owner.dataset,
        declaration,
        feature_indices=owner.feature_indices,
        symbol_index=owner.symbol_index,
        first_decision_index=declaration.policy_start_index,
    )
    for child in children:
        validate_preprocessing_application(
            child.dataset,
            declaration,
            feature_indices=child.feature_indices,
            decision_time=child.dataset.timestamps[child.start_index],
        )
    # The protocol is required for this entire capability; keep the dependency
    # explicit rather than allowing a preprocessing-only implicit learner lane.
    if not isinstance(protocol, AllocationPPOTrainingProtocol):
        raise ValueError("scheduled preprocessing requires an explicit protocol")
    return fit


def fit_allocation_ppo_schedule(
    env: AllocationTrainingScheduleEnv,
    *,
    total_timesteps: int,
    seed: int = 0,
    training_protocol: AllocationPPOTrainingProtocol | None = None,
) -> ScheduledAllocationPPOFit:
    """Fit one PPO across declared independent-account windows in causal order."""
    if type(env) is not AllocationTrainingScheduleEnv:
        raise ValueError("scheduled fit requires AllocationTrainingScheduleEnv")
    if training_protocol is None:
        raise ValueError("scheduled fit requires an explicit training protocol")
    if type(seed) is not int or seed < 0:
        raise ValueError("policy seed must be a nonnegative integer")
    if (
        type(total_timesteps) is not int
        or total_timesteps <= 0
        or total_timesteps % training_protocol.n_steps
    ):
        raise ValueError(
            "scheduled training budget must be a positive rollout multiple"
        )

    usage_before = env.usage_payload()
    before_windows = cast(list[dict[str, Any]], usage_before["windows"])
    if any(row["reset_count"] or row["decision_count"] for row in before_windows):
        raise ValueError("scheduled fit requires a fresh unused runtime")

    env.validate_sources()
    template = env.template_env
    recipe = json.loads(canonical_json_bytes(template.recipe))
    objectives = [
        json.loads(canonical_json_bytes(child.bound.objective.payload()))
        for child in env.training_environments
    ]
    validate_protocol_clock(template, training_protocol)
    cycle_steps = sum(
        window.stop_index - window.start_index
        for window in env.schedule.training_windows
    )
    if total_timesteps < cycle_steps:
        raise ValueError("scheduled fit budget must consume every train window")
    preprocessing_fit = _preprocessing_fit(env, training_protocol)

    # construct_protocol_ppo only requires the concrete Gym spaces/runtime; the
    # cast keeps its historical single-env annotation unchanged.
    model = construct_protocol_ppo(
        cast(AllocationTradingEnv, env), training_protocol, seed=seed
    )
    updates = AllocationUpdateRecorder(model)
    callbacks = importlib.import_module("stable_baselines3.common.callbacks")
    decision_counts: dict[str, dict[int, int]] = {
        identity: {} for identity in env.schedule.train_window_ids
    }
    observation_indices: dict[str, set[int]] = {
        identity: set() for identity in env.schedule.train_window_ids
    }
    declared = {window.window_id: window for window in env.schedule.training_windows}

    def observe_step(callback: Any) -> bool:
        infos = callback.locals["infos"]
        dones = callback.locals["dones"]
        if len(infos) != 1 or len(dones) != 1:
            raise ValueError("scheduled receipt requires one actual account")
        info = infos[0]
        identity = info.get("training_window_id")
        if identity not in declared:
            raise ValueError("scheduled step differs from declared training window")
        execution = info["execution"]
        if execution.bars_advanced != 1:
            raise ValueError("scheduled receipt requires one processing bar")
        index = execution.next_index - 1
        window = declared[identity]
        if not window.start_index <= index < window.stop_index:
            raise ValueError("scheduled decision lies outside its declared window")
        counts = decision_counts[identity]
        counts[index] = counts.get(index, 0) + 1
        observation_indices[identity].add(index)
        if not bool(dones[0]):
            successor = execution.next_index
            if not window.start_index <= successor < window.stop_index:
                raise ValueError("scheduled bootstrap lies outside its declared window")
            observation_indices[identity].add(successor)
        return True

    consumption_callback = type(
        "AllocationScheduleConsumptionCallback",
        (callbacks.BaseCallback,),
        {"_on_step": observe_step},
    )()
    try:
        callback = callbacks.CallbackList(
            [consumption_callback, updates.callback(callbacks.BaseCallback)]
        )
        model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            reset_num_timesteps=True,
            progress_bar=False,
            log_interval=1,
            tb_log_name="PPO",
        )
    finally:
        updates.close()

    validate_allocation_protocol_model(model, training_protocol)
    env.validate_sources()
    usage = env.usage_payload()
    windows = cast(list[dict[str, Any]], usage["windows"])
    if (
        model.num_timesteps != total_timesteps
        or sum(row["decision_count"] for row in windows) != total_timesteps
        or any(row["decision_count"] <= 0 for row in windows)
    ):
        raise ValueError("scheduled fit usage differs from realized training budget")

    consumption_rows: list[dict[str, object]] = []
    for identity in env.schedule.train_window_ids:
        counts = decision_counts[identity]
        indices = sorted(counts)
        if (
            sum(counts.values())
            != next(
                row["decision_count"] for row in windows if row["window_id"] == identity
            )
            or not indices
        ):
            raise ValueError("scheduled row consumption differs from runtime usage")
        consumption_rows.append(
            {
                "window_id": identity,
                "decision_indices": indices,
                "decision_counts": [counts[index] for index in indices],
                "observation_indices": sorted(observation_indices[identity]),
            }
        )
    consumption = {
        "schema": "allocation_training_schedule_consumption_v1",
        "windows": consumption_rows,
    }

    receipt: dict[str, object] = {
        "schema": "allocation_ppo_schedule_fit_receipt_v1",
        "schedule": env.schedule.payload(),
        "schedule_digest": env.schedule.digest,
        "recipe_digest": template.recipe_digest,
        "seed": seed,
        "requested_timesteps": total_timesteps,
        "actual_timesteps": model.num_timesteps,
        "clock": template.bound.clock.payload(),
        "protocol": training_protocol.payload(),
        "protocol_digest": training_protocol.digest,
        "optimization": updates.payload(),
        "usage": usage,
        "consumption": consumption,
    }
    if preprocessing_fit is not None:
        receipt["preprocessing_fit"] = preprocessing_fit
    sources = []
    for child, objective, row in zip(
        env.training_environments, objectives, consumption_rows, strict=True
    ):
        sources.append(
            {
                "window_id": row["window_id"],
                "objective": objective,
                "source": allocation_training_source(
                    child,
                    decision_counts=decision_counts[cast(str, row["window_id"])],
                    observation_indices=tuple(
                        cast(list[int], row["observation_indices"])
                    ),
                ),
            }
        )
    manifest = {
        "schema": "allocation_ppo_inference_bundle_v5",
        "recipe": recipe,
        "recipe_digest": template.recipe_digest,
        "training": receipt
        | {
            "schema": "allocation_ppo_schedule_training_receipt_v1",
            "sources": sources,
            "ppo": {
                "gamma": model.gamma,
                "gae_lambda": model.gae_lambda,
                "n_steps": model.n_steps,
                "batch_size": model.batch_size,
                "n_epochs": model.n_epochs,
                "learning_rate": model.learning_rate,
                "net_arch": model.policy.net_arch,
                "device": str(model.device),
            },
        },
    }
    return ScheduledAllocationPPOFit(model, receipt, manifest=manifest)


__all__ = ["ScheduledAllocationPPOFit", "fit_allocation_ppo_schedule"]
