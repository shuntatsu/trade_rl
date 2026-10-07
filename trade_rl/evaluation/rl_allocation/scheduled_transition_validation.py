"""Closed scheduled child/actor/buffer/GAE joins without importing a learner."""

from __future__ import annotations

from hashlib import sha256
from typing import Any

import numpy as np

from trade_rl._validation import require_sha256
from trade_rl.artifacts import canonical_json_bytes, content_digest
from trade_rl.evaluation.rl_allocation.transition_arrays import (
    buffer_payload,
    read_array,
)
from trade_rl.evaluation.rl_allocation.transition_facts import (
    _closed,
    validate_allocation_execution_facts,
)
from trade_rl.strategies.rl.allocation_manifest import validate_allocation_manifest
from trade_rl.strategies.rl.allocation_preprocessing import (
    AllocationFeaturePreprocessing,
    _native_json,
)
from trade_rl.strategies.rl.allocation_training_schedule import (
    AllocationTrainingSchedule,
)


def scheduled_gae(
    rows: list[dict[str, Any]], bootstrap: str, *, gamma: float, gae_lambda: float
) -> tuple[str, str]:
    """Pinned SB3 mixed NumPy dtype/order, including unrounded carried advantage."""
    rewards = np.stack(
        [read_array(row["actor"]["reward"], "<f4", (1,)) for row in rows]
    )
    values = np.stack(
        [read_array(row["actor"]["value"], "<f4", (1, 1)).reshape(1) for row in rows]
    )
    starts = np.asarray([[row["episode_start"]] for row in rows], dtype="<f4")
    last_values = read_array(bootstrap, "<f4", (1, 1)).flatten()
    advantages = np.zeros_like(values)
    running: Any = 0
    for step in reversed(range(len(rows))):
        if step == len(rows) - 1:
            nonterminal = 1.0 - np.array([rows[-1]["done"]], dtype=bool)
            successor = last_values
        else:
            nonterminal = 1.0 - starts[step + 1]
            successor = values[step + 1]
        delta = rewards[step] + gamma * successor * nonterminal - values[step]
        running = delta + gamma * gae_lambda * nonterminal * running
        advantages[step] = running
    if not np.isfinite(advantages).all():
        raise ValueError("scheduled raw GAE target is nonfinite")
    return advantages.tobytes().hex(), (advantages + values).tobytes().hex()


def validate_scheduled_transition_events(
    events: object, bundle: dict[str, Any], *, learner_diagnostics: str = "none"
) -> dict[str, Any]:
    """Consistency of supplied evidence, not source or historical fit authentication."""
    try:
        return _validate(events, bundle, learner_diagnostics)
    except (KeyError, TypeError, OverflowError, RecursionError, IndexError) as error:
        raise ValueError("invalid closed scheduled transition trace") from error


def _validate(events: Any, bundle: dict[str, Any], mode: str) -> dict[str, Any]:
    _native_json(events)
    if type(mode) is not str or mode not in ("none", "native_ppo_update_v1"):
        raise ValueError("scheduled diagnostics mode is undeclared")
    bundle = validate_allocation_manifest(
        bundle, require_policy="policy_sha256" in bundle
    )
    if bundle["schema"] != "allocation_ppo_inference_bundle_v5":
        raise ValueError("scheduled transition trace requires bundle v5")
    training, recipe = bundle["training"], bundle["recipe"]
    schedule = AllocationTrainingSchedule.from_payload(training["schedule"])
    windows = schedule.training_windows
    sources = {row["window_id"]: row["source"] for row in training["sources"]}
    width = len(recipe["observation"]["fields"])
    steps, count = training["ppo"]["n_steps"], training["actual_timesteps"]
    expected_count = 1 + count + count // steps * (2 if mode != "none" else 1)
    if type(events) is not list or len(events) != expected_count:
        raise ValueError("scheduled trace must contain every admitted rollout/update")
    digests = {
        name: sha256() for name in ("actor", "bootstrap_critic", "terminal_sentinel")
    }
    phase_counts = dict.fromkeys(digests, 0)
    coverage = {
        identity: {
            "actor_count": 0,
            "terminal_count": 0,
            "reset_count": 0,
            "critic_count": 0,
            "prepared_reset_without_actor_count": 0,
        }
        for identity in schedule.train_window_ids
    }
    counts: dict[str, dict[int, int]] = {identity: {} for identity in sources}
    clocks: dict[str, dict[int, list[int]]] = {identity: {} for identity in sources}
    observations: dict[str, set[int]] = {identity: set() for identity in sources}
    coordinates: dict[str, tuple[int, int]] = {}
    feature_rows: dict[tuple[str, str, int], tuple[bytes, bytes]] = {}
    optimizer_events = sha256()

    def record(phase: str, identity: dict[str, Any], raw: str) -> None:
        header = canonical_json_bytes({"phase": phase, "identity": identity})
        data = bytes.fromhex(raw)
        digests[phase].update(
            len(header).to_bytes(8, "big")
            + header
            + len(data).to_bytes(8, "big")
            + data
        )
        phase_counts[phase] += 1

    def identity(
        raw: Any, ordinal: int, episode: int, index: int, time_ns: int
    ) -> dict[str, Any]:
        window = windows[ordinal]
        expected = {
            "window_id": window.window_id,
            "dataset_id": window.dataset_id,
            "source_scope_digest": window.source_digest,
            "window_ordinal": ordinal,
            "episode": episode,
            "index": index,
            "time_ns": time_ns,
        }
        if canonical_json_bytes(raw) != canonical_json_bytes(expected):
            raise ValueError("scheduled child identity/source/clock/episode differs")
        return expected

    initial = _closed(events[0], {"kind", "identity", "observation"})
    if initial["kind"] != "start":
        raise ValueError("scheduled trace requires one initial actual input")
    current = identity(
        initial["identity"], 0, 0, windows[0].start_index, windows[0].decision_start_ns
    )
    read_array(initial["observation"], "<f4", (1, width))
    next_observation = initial["observation"]
    coverage[current["window_id"]]["reset_count"] += 1
    previous: dict[str, Any] | None = None
    order_states: dict[str, str] = {}
    cursor, optimizer_steps, epochs = 1, 0, 0
    parameter_after = None
    for rollout in range(count // steps):
        batch = []
        for local in range(steps):
            row = _closed(
                events[cursor],
                {
                    "kind",
                    "sequence",
                    "rollout",
                    "local",
                    "old",
                    "episode_start",
                    "done",
                    "actor",
                    "next",
                    "next_role",
                    "next_observation",
                    "terminal",
                    "facts",
                },
            )
            cursor += 1
            if (
                row["kind"] != "transition"
                or type(row["episode_start"]) is not bool
                or type(row["done"]) is not bool
                or any(
                    type(row[key]) is not int or row[key] != expected
                    for key, expected in (
                        ("sequence", rollout * steps + local),
                        ("rollout", rollout),
                        ("local", local),
                    )
                )
            ):
                raise ValueError("scheduled transition ordinal/flags differ")
            identity(
                row["old"],
                current["window_ordinal"],
                current["episode"],
                current["index"],
                current["time_ns"],
            )
            old, facts = current, row["facts"]
            start = previous is None or previous["done"]
            if row["episode_start"] != start:
                raise ValueError(
                    "scheduled episode-start mask differs from true resets"
                )
            if start:
                order_states = {}
            actor = _closed(
                row["actor"],
                {"observation", "action", "action_code", "value", "log_prob", "reward"},
            )
            if actor["observation"] != next_observation:
                raise ValueError(
                    "scheduled actor did not consume the actual successor/reset"
                )
            obs = read_array(actor["observation"], "<f4", (1, width))
            code = read_array(actor["action"], "<i8", (1,))[0]
            if (
                type(actor["action_code"]) is not int
                or not 0 <= code <= 3
                or code != actor["action_code"]
            ):
                raise ValueError("scheduled sampled action differs from native mapper")
            read_array(actor["value"], "<f4", (1, 1))
            if read_array(actor["log_prob"], "<f4", (1,))[0] > 0:
                raise ValueError(
                    "scheduled categorical log probability must be nonpositive"
                )
            reward = validate_allocation_execution_facts(
                facts,
                recipe,
                dataset_id=old["dataset_id"],
                action_code=actor["action_code"],
            )
            if (
                read_array(actor["reward"], "<f4", (1,)).tobytes()
                != np.array([reward], "<f4").tobytes()
            ):
                raise ValueError(
                    "scheduled collector reward differs from native equity increment"
                )
            if (
                facts["decision_index"] != old["index"]
                or facts["decision_time_ns"] != old["time_ns"]
            ):
                raise ValueError(
                    "scheduled native decision differs from old child clock"
                )
            window, source = windows[old["window_ordinal"]], sources[old["window_id"]]
            decision = facts["proposal"]["decision"]
            context = decision["baseline"]["context"]
            if (
                context["symbol"] != source["symbol"]
                or decision["remaining_steps"] != window.stop_index - old["index"]
            ):
                raise ValueError("scheduled decision symbol/full horizon differs")
            coordinate = (facts["symbol_index"], len(facts["book"]["quantities"]))
            if (
                old["window_id"] in coordinates
                and coordinates[old["window_id"]] != coordinate
            ):
                raise ValueError(
                    "scheduled full symbol coordinates changed in a source"
                )
            coordinates[old["window_id"]] = coordinate
            if previous is not None and not start:
                book = previous["facts"]["book"]
                if (
                    context["cash"] != book["cash"]
                    or context["equity"] != book["equity"]
                    or context["quantity"]
                    != book["exact_quantities"][facts["symbol_index"]]
                ):
                    raise ValueError(
                        "scheduled account context differs from prior native book"
                    )
            for event in facts["order_events"]:
                prior = order_states.get(event["order_id"])
                if (
                    prior is None
                    and (
                        event["event_type"] != "submitted"
                        or event["previous_status"] != "submitted"
                    )
                    or prior is not None
                    and prior != event["previous_status"]
                ):
                    raise ValueError("scheduled native order-status chain differs")
                order_states[event["order_id"]] = event["new_status"]
            for terminal in facts["terminal_order_reasons"]:
                order_states.pop(terminal["order_id"], None)
            if order_states != {
                order["intent"]["order_id"]: order["status"]
                for order in facts["active_orders"]
            }:
                raise ValueError(
                    "scheduled active order state differs from native events"
                )
            features = np.asarray(decision["feature_values"], dtype="<f4")
            if recipe["schema"] == "allocation_ppo_recipe_v3":
                frozen = AllocationFeaturePreprocessing.from_payload(
                    recipe["observation"]["feature_preprocessing"]
                )
                features = frozen.transform(
                    tuple(decision["feature_values"]),
                    feature_names=tuple(recipe["feature_names"]),
                    feature_config_digest=frozen.feature_config_digest,
                    source_normalization_digest=frozen.source_normalization_digest,
                    decision_time_ns=facts["decision_time_ns"],
                )
            if obs[0, : len(features)].tobytes() != features.tobytes():
                raise ValueError(
                    "scheduled actor feature bits differ from old source transform"
                )
            feature_key = (old["window_id"], old["dataset_id"], old["index"])
            feature_bits = (
                canonical_json_bytes(decision["feature_values"]),
                features.tobytes(),
            )
            if (
                feature_key in feature_rows
                and feature_rows[feature_key] != feature_bits
            ):
                raise ValueError("scheduled source feature bits changed across reuse")
            feature_rows[feature_key] = feature_bits
            done = (
                facts["processing_index"] == window.stop_index
                or facts["book"]["termination_reason"] is not None
            )
            if row["done"] != done:
                raise ValueError("scheduled terminal differs from native horizon/book")
            coverage[old["window_id"]]["actor_count"] += 1
            record("actor", old, actor["observation"])
            count_map = counts[old["window_id"]]
            count_map[old["index"]] = count_map.get(old["index"], 0) + 1
            pair = [facts["decision_time_ns"], facts["processing_time_ns"]]
            prior_clock = clocks[old["window_id"]].get(old["index"])
            if prior_clock is not None and prior_clock != pair:
                raise ValueError("scheduled source clock changed across resets")
            clocks[old["window_id"]][old["index"]] = pair
            observations[old["window_id"]].add(old["index"])
            if done:
                if (
                    np.any(read_array(row["terminal"], "<f4", (width,)))
                    or row["next_role"] != "prepared_reset"
                ):
                    raise ValueError("scheduled true terminal/reset sentinel differs")
                terminal_id = dict(
                    old,
                    index=facts["processing_index"],
                    time_ns=facts["processing_time_ns"],
                )
                record("terminal_sentinel", terminal_id, row["terminal"])
                coverage[old["window_id"]]["terminal_count"] += 1
                ordinal = (old["window_ordinal"] + 1) % len(windows)
                next_window = windows[ordinal]
                current = identity(
                    row["next"],
                    ordinal,
                    old["episode"] + 1,
                    next_window.start_index,
                    next_window.decision_start_ns,
                )
                coverage[current["window_id"]]["reset_count"] += 1
            else:
                if row["terminal"] is not None or row["next_role"] != "live_successor":
                    raise ValueError(
                        "scheduled live transition cannot claim reset or sentinel"
                    )
                current = identity(
                    row["next"],
                    old["window_ordinal"],
                    old["episode"],
                    facts["processing_index"],
                    facts["processing_time_ns"],
                )
                observations[old["window_id"]].add(current["index"])
            read_array(row["next_observation"], "<f4", (1, width))
            next_observation = row["next_observation"]
            previous = row
            batch.append(row)
        boundary = _closed(
            events[cursor],
            {
                "kind",
                "rollout",
                "first",
                "last",
                "buffer_digest",
                "last_old",
                "current",
                "done",
                "bootstrap",
                "advantages",
                "returns",
            },
        )
        cursor += 1
        bootstrap = _closed(boundary["bootstrap"], {"observation", "value"})
        read_array(bootstrap["value"], "<f4", (1, 1))
        gae, returns = scheduled_gae(
            batch,
            bootstrap["value"],
            gamma=training["ppo"]["gamma"],
            gae_lambda=training["ppo"]["gae_lambda"],
        )
        expected = {
            "kind": "rollout",
            "rollout": rollout,
            "first": batch[0]["sequence"],
            "last": batch[-1]["sequence"],
            "buffer_digest": content_digest(buffer_payload(batch, width)),
            "last_old": batch[-1]["old"],
            "current": current,
            "done": batch[-1]["done"],
            "bootstrap": {"observation": next_observation, "value": bootstrap["value"]},
            "advantages": gae,
            "returns": returns,
        }
        if canonical_json_bytes(boundary) != canonical_json_bytes(expected):
            raise ValueError(
                "scheduled admitted buffer/bootstrap/raw GAE boundary differs"
            )
        record("bootstrap_critic", current, next_observation)
        coverage[current["window_id"]]["critic_count"] += 1
        if mode != "none":
            update = _closed(
                events[cursor],
                {
                    "kind",
                    "rollout",
                    "first",
                    "last",
                    "buffer_digest",
                    "optimizer_steps",
                    "epochs",
                    "n_updates",
                    "diagnostics",
                    "explained_variance_reason",
                    "parameters_before",
                    "parameters_after",
                },
            )
            cursor += 1
            for key, expected_value in (
                ("kind", "update"),
                ("rollout", rollout),
                ("first", boundary["first"]),
                ("last", boundary["last"]),
                ("buffer_digest", boundary["buffer_digest"]),
            ):
                if canonical_json_bytes(update[key]) != canonical_json_bytes(
                    expected_value
                ):
                    raise ValueError(
                        "scheduled update is not bound to its admitted rollout"
                    )
            if any(
                type(update[key]) is not int or update[key] < 0
                for key in ("optimizer_steps", "epochs", "n_updates")
            ):
                raise ValueError(
                    "scheduled update counters must be native nonnegative integers"
                )
            if not 1 <= update["epochs"] <= training["ppo"]["n_epochs"]:
                raise ValueError("scheduled update entered-epoch count differs")
            if (
                update["optimizer_steps"]
                > update["epochs"] * steps // training["ppo"]["batch_size"]
            ):
                raise ValueError(
                    "scheduled update optimizer calls exceed entered minibatches"
                )
            for _ in range(update["optimizer_steps"]):
                optimizer_steps += 1
                event = canonical_json_bytes(
                    {
                        "step": optimizer_steps,
                        "rollout": rollout + 1,
                        "timesteps": (rollout + 1) * steps,
                    }
                )
                optimizer_events.update(len(event).to_bytes(8, "big") + event)
            epochs += update["epochs"]
            if update["n_updates"] != epochs:
                raise ValueError("scheduled native logger update ordinal is stale")
            diagnostics = _closed(
                update["diagnostics"],
                {
                    "value_loss",
                    "policy_gradient_loss",
                    "entropy_loss",
                    "approx_kl",
                    "clip_fraction",
                    "explained_variance",
                },
            )
            for key, value in diagnostics.items():
                if key == "explained_variance" and value is None:
                    continue
                if type(value) is not float or not np.isfinite(value):
                    raise ValueError(
                        "scheduled native diagnostic must be a finite float"
                    )
            if (
                diagnostics["value_loss"] < 0
                or not 0 <= diagnostics["clip_fraction"] <= 1
            ):
                raise ValueError("scheduled native loss/clip diagnostic exceeds bounds")
            values = np.stack(
                [
                    read_array(row["actor"]["value"], "<f4", (1, 1)).reshape(1)
                    for row in batch
                ]
            )
            targets = read_array(returns, "<f4", (steps, 1))
            variance = np.var(targets.flatten())
            ev = (
                None
                if variance == 0
                else float(1 - np.var(targets.flatten() - values.flatten()) / variance)
            )
            if diagnostics["explained_variance"] != ev or update[
                "explained_variance_reason"
            ] != ("zero_target_variance" if ev is None else None):
                raise ValueError(
                    "scheduled explained variance differs from collection targets"
                )
            for name in ("parameters_before", "parameters_after"):
                params = _closed(update[name], {"actor", "critic"})
                for pin in params.values():
                    require_sha256(pin, field="scheduled parameter digest")
            if (
                parameter_after is not None
                and update["parameters_before"] != parameter_after
            ):
                raise ValueError(
                    "scheduled actual parameter chain differs between updates"
                )
            if (
                update["optimizer_steps"] == 0
                and update["parameters_before"] != update["parameters_after"]
            ):
                raise ValueError(
                    "scheduled parameters changed without returned optimizer call"
                )
            parameter_after = update["parameters_after"]
    if previous is not None and previous["done"]:
        coverage[current["window_id"]]["prepared_reset_without_actor_count"] += 1
    for identity_ in schedule.train_window_ids:
        indices = sorted(counts[identity_])
        source = sources[identity_]
        if (
            indices != source["decision_indices"]
            or [counts[identity_][i] for i in indices] != source["decision_counts"]
            or sorted(observations[identity_]) != source["observation_indices"]
            or content_digest([clocks[identity_][i] for i in indices])
            != source["clock_consumption_digest"]
        ):
            raise ValueError("scheduled per-window source/clock/reuse coverage differs")
        usage = next(
            row for row in training["usage"]["windows"] if row["window_id"] == identity_
        )
        if (
            usage["reset_count"] != coverage[identity_]["reset_count"]
            or usage["decision_count"] != coverage[identity_]["actor_count"]
        ):
            raise ValueError(
                "scheduled actual actor/reset counts differ from fit usage"
            )
    if mode != "none":
        optimization = training["optimization"]
        if (
            optimizer_steps != optimization["successful_optimizer_step_calls"]
            or epochs != optimization["epoch_iterations"]
        ):
            raise ValueError("scheduled native update totals differ from finalized fit")
        if optimizer_events.hexdigest() != optimization["optimizer_step_events_digest"]:
            raise ValueError("scheduled optimizer event digest differs from updates")
    summary = {
        "schema": "allocation_ppo_scheduled_observation_consumption_v1",
        "width": width,
        "phases": {
            name: {"count": phase_counts[name], "digest": digest.hexdigest()}
            for name, digest in digests.items()
        },
        "windows": [
            {"window_id": identity_, **coverage[identity_]}
            for identity_ in schedule.train_window_ids
        ],
    }
    return {
        "transition_count": count,
        "rollout_count": count // steps,
        "width": width,
        "symbol_counts": [
            coordinates[identity_][1] for identity_ in schedule.train_window_ids
        ],
        "learner_diagnostics": mode,
        "input_summary": summary,
        "input_digest": content_digest(summary),
    }
