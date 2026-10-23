"""Fixed explicit single-account CPU fit with observed settled admission."""

from __future__ import annotations

import importlib
from typing import Any

from trade_rl.evaluation.rl_allocation.settled_env import SettledAllocationTradingEnv
from trade_rl.evaluation.rl_allocation.settled_training_source import (
    settled_training_source_receipt,
)
from trade_rl.evaluation.rl_allocation.settled_transition_trace import (
    SettledTransitionRecorder,
    _best_effort_note,
)
from trade_rl.evaluation.rl_allocation.settled_transition_validation import (
    validate_settled_transition_capture,
)
from trade_rl.evaluation.rl_allocation.training_protocol import (
    AllocationUpdateRecorder,
    construct_protocol_ppo,
)
from trade_rl.strategies.rl.allocation_model import (
    AllocationPPOPolicy,
    validate_allocation_protocol_model,
)
from trade_rl.strategies.rl.allocation_training_protocol import (
    AllocationPPOTrainingProtocol,
)


def _declaration(
    env: SettledAllocationTradingEnv, protocol: AllocationPPOTrainingProtocol, seed: int
) -> None:
    if not isinstance(env, SettledAllocationTradingEnv) or env._episode_sequence:
        raise ValueError("settled fit requires a fresh distinct settled environment")
    if not isinstance(protocol, AllocationPPOTrainingProtocol):
        raise ValueError("settled fit requires an explicit immutable CPU protocol")
    if type(seed) is not int or seed < 0:
        raise ValueError("settled policy seed must be a nonnegative native integer")
    env.validate_binding()
    clock = env.bound.clock
    if (protocol.n_steps, protocol.gamma, protocol.gae_lambda) != (
        clock.rollout_steps,
        clock.gamma,
        clock.gae_lambda,
    ):
        raise ValueError("settled protocol differs from the declared actor clock")


def build_settled_allocation_ppo(
    env: SettledAllocationTradingEnv,
    *,
    seed: int = 0,
    training_protocol: AllocationPPOTrainingProtocol,
) -> Any:
    """Construct a fresh pinned learner; this does not establish fit or admission."""
    _declaration(env, training_protocol, seed)
    return construct_protocol_ppo(env, training_protocol, seed=seed)


def fit_settled_allocation_ppo(
    env: SettledAllocationTradingEnv,
    *,
    total_timesteps: int,
    seed: int = 0,
    training_protocol: AllocationPPOTrainingProtocol,
    transition_trace: SettledTransitionRecorder | None = None,
) -> AllocationPPOPolicy:
    """Fit only supplied synthetic/authorized data; no data discovery or Study."""
    _declaration(env, training_protocol, seed)
    if (
        type(total_timesteps) is not int
        or total_timesteps <= 0
        or total_timesteps % training_protocol.n_steps
    ):
        raise ValueError(
            "settled training budget must be a positive exact rollout multiple"
        )
    trace = (
        transition_trace
        if transition_trace is not None
        else SettledTransitionRecorder(env)
    )
    if not isinstance(trace, SettledTransitionRecorder):
        raise ValueError("settled fit requires its distinct observed recorder")
    trace.validate_fit(env)
    source = settled_training_source_receipt(env)
    recipe, recipe_digest = env.recipe, env.recipe_digest
    model = build_settled_allocation_ppo(
        env, seed=seed, training_protocol=training_protocol
    )
    callbacks = importlib.import_module("stable_baselines3.common.callbacks")
    updates = AllocationUpdateRecorder(model, completed_update=trace.observe_update)
    primary_error: BaseException | None = None
    try:
        trace.attach(model)
        callback = callbacks.CallbackList(
            [
                updates.callback(callbacks.BaseCallback),
                trace.callback(callbacks.BaseCallback),
            ]
        )
        model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            reset_num_timesteps=True,
            progress_bar=False,
            log_interval=1,
            tb_log_name="PPO",
        )
    except BaseException as error:
        primary_error = error
        trace.record_failure("fit", error)
        raise
    finally:
        cleanup_errors = []
        for cleanup in (
            lambda: trace.detach(primary_error=primary_error),
            updates.close,
        ):
            try:
                cleanup()
            except BaseException as cleanup_error:
                cleanup_errors.append(cleanup_error)
                if primary_error is not None:
                    _best_effort_note(
                        primary_error,
                        "Settled cleanup: " + type(cleanup_error).__name__,
                    )
        if primary_error is None and cleanup_errors:
            trace.record_failure("cleanup", cleanup_errors[0])
            raise cleanup_errors[0]
    try:
        validate_allocation_protocol_model(model, training_protocol)
        trace.validate_model_frontier(model)
        env.validate_binding()
        if (
            model.num_timesteps != total_timesteps
            or settled_training_source_receipt(env) != source
            or env.recipe_digest != recipe_digest
        ):
            raise ValueError(
                "settled realized budget, recipe or declared source changed"
            )
        optimization = updates.payload()
        projection = validate_settled_transition_capture(
            env,
            trace.payload(),
            requested_timesteps=total_timesteps,
            optimization=optimization,
            training_protocol=training_protocol,
        )
        training = {
            "schema": "allocation_settled_training_receipt_v1",
            "seed": seed,
            "requested_timesteps": total_timesteps,
            "actual_timesteps": model.num_timesteps,
            "clock": env.bound.clock.payload(),
            "objective": env.bound.objective.payload(),
            "protocol": training_protocol.payload(),
            "protocol_digest": training_protocol.digest,
            "optimization": optimization,
            "ppo": {
                key: getattr(model, key)
                for key in (
                    "gamma",
                    "gae_lambda",
                    "n_steps",
                    "batch_size",
                    "n_epochs",
                    "learning_rate",
                )
            }
            | {"net_arch": model.policy.net_arch, "device": str(model.device)},
            **projection,
        }
        policy = AllocationPPOPolicy(
            model,
            {
                "schema": "allocation_ppo_inference_bundle_v6",
                "recipe": recipe,
                "recipe_digest": recipe_digest,
                "training": training,
            },
        )
        trace.mark_complete(policy.manifest)
        return policy
    except BaseException as error:
        trace.record_failure("post_fit_admission", error)
        raise
