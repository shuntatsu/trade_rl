"""Strict explicit-protocol receipt; mechanical consistency, not authenticity."""

from __future__ import annotations

from hashlib import sha256
from typing import Any

from trade_rl._validation import require_sha256
from trade_rl.strategies.rl.allocation_training_protocol import (
    AllocationPPOTrainingProtocol,
)
from trade_rl.strategies.rl.allocation_training_receipt import (
    validate_allocation_training,
)


def validate_allocation_training_v3(
    training: object, recipe: dict[str, Any], recipe_digest: str
) -> None:
    extras = {"schema", "protocol", "protocol_digest", "optimization"}
    if not isinstance(training, dict) or not extras <= training.keys():
        raise ValueError("v3 training requires its explicit protocol fields")
    if training["schema"] != "allocation_ppo_training_receipt_v3":
        raise ValueError("unsupported explicit protocol training receipt")
    common = {key: value for key, value in training.items() if key not in extras}
    validate_allocation_training(common, recipe, recipe_digest)
    protocol = AllocationPPOTrainingProtocol.from_payload(training["protocol"])
    declaration: dict[str, Any] = protocol.payload()
    require_sha256(training["protocol_digest"], field="protocol_digest")
    if protocol.digest != training["protocol_digest"]:
        raise ValueError("protocol digest differs from its declaration")
    ppo, clock = training["ppo"], training["clock"]
    for name in (
        "n_steps",
        "batch_size",
        "n_epochs",
        "gamma",
        "gae_lambda",
        "learning_rate",
    ):
        if ppo[name] != getattr(protocol, name):
            raise ValueError("actual PPO summary differs from explicit protocol")
    if ppo["net_arch"] != declaration["policy"]["net_arch"] or any(
        getattr(protocol, key) != clock[clock_key]
        for key, clock_key in (
            ("n_steps", "rollout_steps"),
            ("gamma", "gamma"),
            ("gae_lambda", "gae_lambda"),
        )
    ):
        raise ValueError("protocol differs from the architecture or financial clock")
    update = training["optimization"]
    keys = {
        "schema",
        "rollout_update_count",
        "successful_optimizer_step_calls",
        "epoch_iterations",
        "optimizer_step_events_digest",
        "final_capture",
    }
    if not isinstance(update, dict) or set(update) != keys:
        raise ValueError("optimization receipt requires exactly its declared fields")
    for key in (
        "rollout_update_count",
        "successful_optimizer_step_calls",
        "epoch_iterations",
    ):
        if type(update[key]) is not int or update[key] < 0:
            raise ValueError(
                "optimization counters must be nonnegative native integers"
            )
    require_sha256(
        update["optimizer_step_events_digest"], field="optimizer_step_events_digest"
    )
    rollouts = training["actual_timesteps"] // protocol.n_steps
    batches = protocol.n_steps // protocol.batch_size
    steps, epochs = (
        update["successful_optimizer_step_calls"],
        update["epoch_iterations"],
    )
    if (
        update["schema"] != "allocation_ppo_optimization_receipt_v1"
        or update["final_capture"] != "after_final_train_v1"
        or update["rollout_update_count"] != rollouts
        or not rollouts <= epochs <= rollouts * protocol.n_epochs
        or not batches * (epochs - rollouts) <= steps <= batches * epochs
        or protocol.target_kl is None
        and (epochs != rollouts * protocol.n_epochs or steps != epochs * batches)
        or steps == 0
        and update["optimizer_step_events_digest"] != sha256(b"").hexdigest()
    ):
        raise ValueError(
            "optimization counters differ from the realized protocol budget"
        )
