"""Closed chronological actor/buffer/economic links, without a PPO backend."""

from __future__ import annotations

from hashlib import sha256
from typing import Any

import numpy as np

from trade_rl.artifacts import canonical_json_bytes, content_digest
from trade_rl.evaluation.rl_allocation.transition_arrays import (
    buffer_digest,
    read_array,
)
from trade_rl.evaluation.rl_allocation.transition_facts import (
    _closed,
    validate_allocation_execution_facts,
)
from trade_rl.strategies.rl.allocation_preprocessing import (
    AllocationFeaturePreprocessing,
    _native_json,
)


def validate_transition_events(
    events: object, bundle: dict[str, Any]
) -> dict[str, int]:
    """Reject inconsistent declared evidence; content pins are not authenticity."""
    try:
        return _validate(events, bundle)
    except (KeyError, TypeError, OverflowError, RecursionError, IndexError) as error:
        raise ValueError("invalid closed transition trace") from error


def _validate(events: Any, bundle: dict[str, Any]) -> dict[str, int]:
    _native_json(events)
    training, recipe = bundle["training"], bundle["recipe"]
    if bundle["schema"] not in (
        "allocation_ppo_inference_bundle_v3",
        "allocation_ppo_inference_bundle_v4",
    ):
        raise ValueError("transition trace requires explicit bundle v3 or v4")
    source, receipt = training["source"], training["observation_consumption"]
    width, steps, count = (
        receipt["width"],
        training["ppo"]["n_steps"],
        training["actual_timesteps"],
    )
    if type(events) is not list or len(events) != count + count // steps:
        raise ValueError("transition trace must contain complete admitted rollouts")
    digests = {
        name: sha256() for name in ("actor", "rollout_boundary", "terminal_sentinel")
    }
    phase_counts = dict.fromkeys(digests, 0)

    def record(phase: str, raw: str, **facts: object) -> None:
        header = canonical_json_bytes({"phase": phase, **facts})
        data = bytes.fromhex(raw)
        digests[phase].update(
            len(header).to_bytes(8, "big")
            + header
            + len(data).to_bytes(8, "big")
            + data
        )
        phase_counts[phase] += 1

    rows: list[dict[str, Any]] = []
    counts: dict[int, int] = {}
    clocks: dict[int, list[int]] = {}
    observations = {source["start_index"]}
    previous: dict[str, Any] | None = None
    order_states: dict[str, str] = {}
    episode, cursor = -1, 0
    size = 0
    selected_index: int | None = None
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
                    "episode",
                    "episode_start",
                    "done",
                    "actor",
                    "next_observation",
                    "terminal",
                    "facts",
                },
            )
            cursor += 1
            sequence = len(rows)
            for name, expected in (
                ("sequence", sequence),
                ("rollout", rollout),
                ("local", local),
            ):
                if type(row[name]) is not int or row[name] != expected:
                    raise ValueError(
                        "transition ordinal differs from chronological admission"
                    )
            if (
                row["kind"] != "transition"
                or type(row["episode_start"]) is not bool
                or type(row["done"]) is not bool
            ):
                raise ValueError("transition kind/termination flags differ")
            start = previous is None or previous["done"]
            if start:
                episode += 1
                order_states = {}
            if (
                row["episode_start"] != start
                or type(row["episode"]) is not int
                or row["episode"] != episode
            ):
                raise ValueError("transition episode/reset order differs")
            facts = row["facts"]
            index = facts["decision_index"]
            expected_index = source["start_index"]
            if not start:
                assert previous is not None
                expected_index = previous["facts"]["processing_index"]
            if (
                index != expected_index
                or not source["start_index"] <= index < source["stop_index"]
            ):
                raise ValueError("transition decision clock differs from reset prefix")
            actor = _closed(
                row["actor"],
                {"observation", "action", "action_code", "value", "log_prob", "reward"},
            )
            action = read_array(actor["action"], "<i8", (1,))
            if (
                type(actor["action_code"]) is not int
                or not 0 <= actor["action_code"] <= 3
                or action[0] != actor["action_code"]
            ):
                raise ValueError("transition sampled action differs from mapper action")
            actor_values = read_array(actor["observation"], "<f4", (1, width))
            read_array(actor["value"], "<f4", (1, 1))
            log_prob = read_array(actor["log_prob"], "<f4", (1,))
            if log_prob[0] > 0:
                raise ValueError(
                    "categorical transition log probability must be nonpositive"
                )
            reward = validate_allocation_execution_facts(
                facts,
                recipe,
                dataset_id=source["dataset_id"],
                action_code=actor["action_code"],
            )
            if selected_index is None:
                selected_index = facts["symbol_index"]
            elif facts["symbol_index"] != selected_index:
                raise ValueError("transition selected symbol coordinate changed")
            for event in facts["order_events"]:
                identity = event["order_id"]
                prior = order_states.get(identity)
                if prior is None:
                    if (
                        event["event_type"] != "submitted"
                        or event["previous_status"] != "submitted"
                    ):
                        raise ValueError(
                            "transition order begins without a submitted event"
                        )
                elif prior != event["previous_status"]:
                    raise ValueError("transition order status chain differs")
                order_states[identity] = event["new_status"]
            terminals = {r["order_id"] for r in facts["terminal_order_reasons"]}
            active = {
                o["intent"]["order_id"]: o["status"] for o in facts["active_orders"]
            }
            for identity in terminals:
                order_states.pop(identity, None)
            if order_states != active:
                raise ValueError(
                    "transition final order states differ from native events"
                )
            if (
                read_array(actor["reward"], "<f4", (1,)).tobytes()
                != np.array([reward], "<f4").tobytes()
            ):
                raise ValueError(
                    "transition collector reward differs from signed equity increment"
                )
            decision = facts["proposal"]["decision"]
            context = decision["baseline"]["context"]
            if (
                context["symbol"] != source["symbol"]
                or decision["remaining_steps"] != source["stop_index"] - index
            ):
                raise ValueError("transition decision symbol/horizon differs")
            if previous is not None:
                if row["actor"]["observation"] != previous["next_observation"]:
                    raise ValueError(
                        "transition actor did not consume collector successor"
                    )
                if not start:
                    book = previous["facts"]["book"]
                    if (
                        context["cash"] != book["cash"]
                        or context["equity"] != book["equity"]
                        or context["quantity"]
                        != book["exact_quantities"][facts["symbol_index"]]
                    ):
                        raise ValueError(
                            "transition raw account context differs from prior book"
                        )
            raw_features = tuple(decision["feature_values"])
            features = np.array(raw_features, "<f4")
            if bundle["schema"] == "allocation_ppo_inference_bundle_v4":
                frozen = AllocationFeaturePreprocessing.from_payload(
                    recipe["observation"]["feature_preprocessing"]
                )
                if facts["decision_time_ns"] < frozen.policy_start_time_ns:
                    raise ValueError("transition precedes declared policy clock")
                features = frozen.transform(
                    raw_features,
                    feature_names=tuple(recipe["feature_names"]),
                    feature_config_digest=frozen.feature_config_digest,
                    source_normalization_digest=frozen.source_normalization_digest,
                    decision_time_ns=facts["decision_time_ns"],
                )
            if actor_values[0, : len(features)].tobytes() != features.tobytes():
                raise ValueError(
                    "transition actor feature prefix differs from frozen transform"
                )
            read_array(row["next_observation"], "<f4", (1, width))
            expected_done = (
                facts["processing_index"] == source["stop_index"]
                or facts["book"]["termination_reason"] is not None
            )
            if row["done"] != expected_done:
                raise ValueError(
                    "transition termination differs from native horizon/book"
                )
            if row["done"]:
                if np.any(read_array(row["terminal"], "<f4", (width,))):
                    raise ValueError(
                        "transition true terminal must preserve zero sentinel"
                    )
                record(
                    "terminal_sentinel",
                    row["terminal"],
                    index=facts["processing_index"],
                )
            elif row["terminal"] is not None:
                raise ValueError("live transition cannot declare a terminal sentinel")
            else:
                observations.add(facts["processing_index"])
            record("actor", actor["observation"], index=index, episode_start=start)
            counts[index] = counts.get(index, 0) + 1
            pair = [facts["decision_time_ns"], facts["processing_time_ns"]]
            if index in clocks and clocks[index] != pair:
                raise ValueError("transition reset changed source timestamps")
            clocks[index] = pair
            current_size = len(facts["book"]["quantities"])
            if size and size != current_size:
                raise ValueError("transition full account shape changed")
            size = current_size
            rows.append(row)
            batch.append(row)
            previous = row
        boundary = _closed(
            events[cursor],
            {
                "kind",
                "rollout",
                "first",
                "last",
                "buffer_digest",
                "new_observation",
                "done",
                "index",
                "last_transition_index",
            },
        )
        cursor += 1
        last = batch[-1]
        expected_boundary = {
            "kind": "rollout",
            "rollout": rollout,
            "first": batch[0]["sequence"],
            "last": last["sequence"],
            "buffer_digest": buffer_digest(batch, width),
            "new_observation": last["next_observation"],
            "done": last["done"],
            "index": source["start_index"]
            if last["done"]
            else last["facts"]["processing_index"],
            "last_transition_index": last["facts"]["processing_index"],
        }
        if canonical_json_bytes(boundary) != canonical_json_bytes(expected_boundary):
            raise ValueError("transition admitted buffer/boundary differs")
        record(
            "rollout_boundary",
            boundary["new_observation"],
            index=boundary["index"],
            last_transition_index=boundary["last_transition_index"],
            done=boundary["done"],
        )
    if (
        sorted(counts) != source["decision_indices"]
        or [counts[i] for i in sorted(counts)] != source["decision_counts"]
        or sorted(observations | set(counts)) != source["observation_indices"]
        or content_digest([clocks[i] for i in sorted(clocks)])
        != source["clock_consumption_digest"]
    ):
        raise ValueError("transition actual source coverage/clock digest differs")
    for phase, stem in (
        ("actor", "actor"),
        ("rollout_boundary", "rollout_boundary"),
        ("terminal_sentinel", "terminal"),
    ):
        if (
            phase_counts[phase] != receipt[f"{stem}_count"]
            or digests[phase].hexdigest() != receipt[f"{stem}_digest"]
        ):
            raise ValueError("transition actual tensor input receipt differs")
    return {
        "transition_count": count,
        "rollout_count": count // steps,
        "width": width,
        "symbol_count": size,
    }
