"""Explicit, opt-in single-account PPO with a bound undiscounted profit clock."""

from __future__ import annotations

import importlib
from typing import Any

from trade_rl.evaluation.rl_allocation.env import AllocationTradingEnv
from trade_rl.evaluation.rl_allocation.input_receipt import AllocationInputRecorder
from trade_rl.evaluation.rl_allocation.preprocessing import (
    validate_training_preprocessing,
)
from trade_rl.evaluation.rl_allocation.training_protocol import (
    AllocationUpdateRecorder,
    construct_protocol_ppo,
    validate_protocol_clock,
)
from trade_rl.evaluation.rl_allocation.training_source import allocation_training_source
from trade_rl.strategies.rl.allocation_manifest import allocation_bundle_schema
from trade_rl.strategies.rl.allocation_model import (
    AllocationPPOPolicy,
    validate_allocation_protocol_model,
)
from trade_rl.strategies.rl.allocation_training_protocol import (
    AllocationPPOTrainingProtocol,
)


def _preprocessing_fit(
    env: AllocationTradingEnv, protocol: AllocationPPOTrainingProtocol | None
) -> dict[str, object] | None:
    if env.feature_preprocessing is None:
        return None
    if protocol is None:
        raise ValueError("frozen preprocessing requires an explicit training protocol")
    return validate_training_preprocessing(
        env.dataset,
        env.feature_preprocessing,
        feature_indices=env.feature_indices,
        symbol_index=env.symbol_index,
        first_decision_index=env.start_index,
    )


def build_allocation_ppo(
    env: AllocationTradingEnv,
    *,
    seed: int = 0,
    training_protocol: AllocationPPOTrainingProtocol | None = None,
) -> Any:
    """Fixed software learning protocol; no search or economic authorization."""
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("policy seed must be a nonnegative integer")
    env.validate_binding()
    _preprocessing_fit(env, training_protocol)
    clock = env.bound.clock
    if training_protocol is not None:
        validate_protocol_clock(env, training_protocol)
        return construct_protocol_ppo(env, training_protocol, seed=seed)
    if clock.rollout_steps < 2:
        raise ValueError("PPO rollout must contain at least two transitions")
    batch_size = min(clock.rollout_steps, 64)
    if clock.rollout_steps % batch_size:
        raise ValueError("fixed PPO rollout must contain complete training batches")
    torch = importlib.import_module("torch")
    sb3 = importlib.import_module("stable_baselines3")
    torch.set_num_threads(1)
    return sb3.PPO(
        "MlpPolicy",
        env,
        device="cpu",
        verbose=0,
        seed=seed,
        learning_rate=0.002,
        n_steps=clock.rollout_steps,
        batch_size=batch_size,
        n_epochs=10,
        gamma=clock.gamma,
        gae_lambda=clock.gae_lambda,
        policy_kwargs={"net_arch": {"pi": [32, 32], "vf": [32, 32]}},
    )


def fit_allocation_ppo(
    env: AllocationTradingEnv,
    *,
    total_timesteps: int,
    seed: int = 0,
    training_protocol: AllocationPPOTrainingProtocol | None = None,
) -> AllocationPPOPolicy:
    """Fit only the supplied episode; publish actual budget and consumed sources.

    This callable does not open data, create a Study or grant research approval.
    The resulting bundle is inference-only, not an exact account/RNG restart.
    """
    if (
        isinstance(total_timesteps, bool)
        or not isinstance(total_timesteps, int)
        or total_timesteps <= 0
        or total_timesteps % env.bound.clock.rollout_steps
    ):
        raise ValueError("training budget must be a positive exact rollout multiple")
    env.validate_binding()
    envelope = allocation_training_source(env)
    preprocessing_fit = _preprocessing_fit(env, training_protocol)
    recipe, recipe_digest = env.recipe, env.recipe_digest
    if training_protocol is None:
        model = build_allocation_ppo(env, seed=seed)
    else:
        validate_protocol_clock(env, training_protocol)
        model = build_allocation_ppo(
            env, seed=seed, training_protocol=training_protocol
        )
    counts: dict[int, int] = {}
    observed = {env.start_index}

    def observe_steps(
        local_values: dict[str, Any], global_values: dict[str, Any]
    ) -> bool:
        del global_values
        infos = local_values["infos"]
        if len(infos) != 1:
            raise ValueError("allocation receipt requires one actual account")
        execution = infos[0]["execution"]
        if execution.bars_advanced != 1:
            raise ValueError("allocation receipt requires one processing bar")
        index = execution.next_index - 1
        counts[index] = counts.get(index, 0) + 1
        observed.add(index)
        if not local_values["dones"][0]:
            observed.add(execution.next_index)
        return True

    recorder = None
    callback: Any = observe_steps
    if env.observation_schema is not None:
        recorder = AllocationInputRecorder(env)
        callbacks = importlib.import_module("stable_baselines3.common.callbacks")
        callback = recorder.callback(callbacks.BaseCallback, observe_steps)
    if training_protocol is None:
        model.learn(total_timesteps=total_timesteps, callback=callback)
    else:
        updates = AllocationUpdateRecorder(model)
        try:
            callback = callbacks.CallbackList(
                [callback, updates.callback(callbacks.BaseCallback)]
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
    env.validate_binding()
    if (
        model.num_timesteps != total_timesteps
        or sum(counts.values()) != total_timesteps
        or allocation_training_source(env) != envelope
        or env.recipe_digest != recipe_digest
        or _preprocessing_fit(env, training_protocol) != preprocessing_fit
    ):
        raise ValueError("realized training budget, recipe or consumed sources changed")
    source = allocation_training_source(
        env, decision_counts=counts, observation_indices=tuple(sorted(observed))
    )
    manifest: dict[str, Any] = {
        "schema": allocation_bundle_schema(recipe),
        "recipe": recipe,
        "recipe_digest": recipe_digest,
        "training": {
            "seed": seed,
            "requested_timesteps": total_timesteps,
            "actual_timesteps": model.num_timesteps,
            "clock": env.bound.clock.payload(),
            "objective": env.bound.objective.payload(),
            "source": source,
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
    if recorder is not None:
        manifest["training"]["observation_consumption"] = recorder.payload()
    if training_protocol is not None:
        manifest["schema"] = "allocation_ppo_inference_bundle_v3"
        manifest["training"].update(
            schema="allocation_ppo_training_receipt_v3",
            protocol=training_protocol.payload(),
            protocol_digest=training_protocol.digest,
            optimization=updates.payload(),
        )
        if preprocessing_fit is not None:
            manifest["schema"] = "allocation_ppo_inference_bundle_v4"
            manifest["training"].update(
                schema="allocation_ppo_training_receipt_v4",
                preprocessing_fit=preprocessing_fit,
            )
    return AllocationPPOPolicy(model, manifest)
