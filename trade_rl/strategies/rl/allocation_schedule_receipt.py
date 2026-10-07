"""Scheduled inference consistency; declarations do not authenticate fitting."""

from __future__ import annotations

from typing import Any

from trade_rl._validation import require_sha256
from trade_rl.artifacts import canonical_json_bytes, content_digest
from trade_rl.strategies.rl.allocation_clock_receipt import (
    validate_allocation_clock,
    validate_ppo_training_budget,
)
from trade_rl.strategies.rl.allocation_objective_receipt import (
    validate_allocation_objective,
)
from trade_rl.strategies.rl.allocation_preprocessing import (
    AllocationFeaturePreprocessing,
)
from trade_rl.strategies.rl.allocation_preprocessing_receipt import (
    preprocessing_fit_payload,
)
from trade_rl.strategies.rl.allocation_protocol_receipt import (
    _validate_protocol_optimization,
)
from trade_rl.strategies.rl.allocation_receipt_time import (
    parse_source_timestamp,
    receipt_datetime_ns,
)
from trade_rl.strategies.rl.allocation_recipe_validation import (
    validate_allocation_recipe,
)
from trade_rl.strategies.rl.allocation_training_schedule import (
    AllocationTrainingSchedule,
    _mapping,
    _native_json,
)

_SOURCE_KEYS = {
    "dataset_id",
    "start_index",
    "stop_index",
    "decision_indices",
    "decision_counts",
    "observation_indices",
    "symbol",
    "feature_names",
    "decision_start",
    "terminal_time",
    "feature_consumption_digest",
    "clock_consumption_digest",
    "execution_consumption_digest",
    "forecast_consumption_digest",
    "cost_consumption_digest",
}


def validate_allocation_schedule_training(
    training: object,
    recipe: dict[str, Any],
    recipe_digest: str,
) -> None:
    """Close clocks, complete-window source scope and actual cyclic reuse counts."""
    if (
        type(recipe) is not dict
        or type(recipe.get("schema")) is not str
        or recipe["schema"]
        not in ("allocation_ppo_recipe_v2", "allocation_ppo_recipe_v3")
    ):
        raise ValueError("scheduled reader requires a native recipe v2/v3 mapping/tag")
    try:
        _native_json(recipe)
    except RecursionError as error:
        raise ValueError("scheduled recipe must contain native finite JSON") from error
    validate_allocation_recipe(recipe)
    if type(recipe_digest) is not str:
        raise ValueError("scheduled recipe pin must be a native SHA256 string")
    require_sha256(recipe_digest, field="recipe_digest")
    if content_digest(recipe) != recipe_digest:
        raise ValueError("scheduled recipe pin differs from its payload")
    _native_json(training)
    frozen = recipe["schema"] == "allocation_ppo_recipe_v3"
    values = _mapping(
        training,
        {
            "schema",
            "schedule",
            "schedule_digest",
            "recipe_digest",
            "seed",
            "requested_timesteps",
            "actual_timesteps",
            "clock",
            "protocol",
            "protocol_digest",
            "optimization",
            "usage",
            "consumption",
            "sources",
            "ppo",
        }
        | ({"preprocessing_fit"} if frozen else set()),
        "scheduled training",
    )
    if (
        values["schema"] != "allocation_ppo_schedule_training_receipt_v1"
        or values["recipe_digest"] != recipe_digest
    ):
        raise ValueError("scheduled training schema or recipe differs")
    schedule = AllocationTrainingSchedule.from_payload(values["schedule"])
    if schedule.digest != values["schedule_digest"]:
        raise ValueError("scheduled training digest differs from its declaration")
    ppo = validate_ppo_training_budget(values)
    clock = validate_allocation_clock(values["clock"], ppo, recipe["runtime_profile"])
    _validate_protocol_optimization(values)
    lengths = [w.stop_index - w.start_index for w in schedule.training_windows]
    steps = values["actual_timesteps"]
    cycles, remainder = divmod(steps, sum(lengths))
    if not cycles:
        raise ValueError(
            "scheduled inference requires complete training-window coverage"
        )
    usage = _mapping(
        values["usage"],
        {
            "schema",
            "schedule_digest",
            "sampler",
            "reset_semantics",
            "rollout_boundary_semantics",
            "windows",
        },
        "schedule usage",
    )
    if (
        usage["schema"] != "allocation_training_schedule_usage_v1"
        or usage["schedule_digest"] != schedule.digest
        or any(
            usage[k] != schedule.payload()[k]
            for k in ("sampler", "reset_semantics", "rollout_boundary_semantics")
        )
    ):
        raise ValueError("schedule usage differs from its declared semantics")
    consumption = _mapping(
        values["consumption"], {"schema", "windows"}, "schedule consumption"
    )
    if consumption["schema"] != "allocation_training_schedule_consumption_v1":
        raise ValueError("unsupported schedule consumption schema")
    arrays = (usage["windows"], consumption["windows"], values["sources"])
    if any(type(v) is not list or len(v) != len(lengths) for v in arrays):
        raise ValueError("scheduled records must cover exactly the train roster")
    offset = 0
    for window, length, used, consumed, record in zip(
        schedule.training_windows,
        lengths,
        *arrays,
        strict=True,
    ):
        used = _mapping(
            used, {"window_id", "reset_count", "decision_count"}, "window usage"
        )
        consumed = _mapping(
            consumed,
            {
                "window_id",
                "decision_indices",
                "decision_counts",
                "observation_indices",
            },
            "window consumption",
        )
        record = _mapping(record, {"window_id", "source", "objective"}, "window source")
        if any(
            row["window_id"] != window.window_id for row in (used, consumed, record)
        ):
            raise ValueError("scheduled records differ from chronological train order")
        rows = list(range(window.start_index, window.stop_index))
        counts = [cycles + int(remainder > offset + i) for i in range(length)]
        if (
            type(used["reset_count"]) is not int
            or type(used["decision_count"]) is not int
            or used["reset_count"] != cycles + int(remainder >= offset)
            or used["decision_count"] != sum(counts)
            or consumed["decision_indices"] != rows
            or consumed["observation_indices"] != rows
            or consumed["decision_counts"] != counts
            or any(
                type(n) is not int
                for k in ("decision_indices", "observation_indices", "decision_counts")
                for n in consumed[k]
            )
        ):
            raise ValueError(
                "scheduled rows or usage differ from realized cyclic budget"
            )
        source = _mapping(record["source"], _SOURCE_KEYS, "scheduled source")
        for key in (
            "dataset_id",
            "feature_consumption_digest",
            "clock_consumption_digest",
            "execution_consumption_digest",
            "forecast_consumption_digest",
            "cost_consumption_digest",
        ):
            require_sha256(source[key], field=key)
        if any(
            type(source[k]) is not int for k in ("start_index", "stop_index")
        ) or any(
            type(source[k]) is not list or any(type(n) is not int for n in source[k])
            for k in ("decision_indices", "decision_counts", "observation_indices")
        ):
            raise ValueError(
                "scheduled source rows require native integer indices/counts"
            )
        if (
            source["dataset_id"] != window.dataset_id
            or source["symbol"] != window.symbol
            or source["start_index"] != window.start_index
            or source["stop_index"] != window.stop_index
            or source["feature_names"] != recipe["feature_names"]
            or any(
                source[k] != consumed[k]
                for k in ("decision_indices", "decision_counts", "observation_indices")
            )
        ):
            raise ValueError("scheduled consumed source differs from its window")
        envelope = source | {"decision_counts": [1] * length}
        if (
            content_digest({k: v for k, v in envelope.items() if k != "dataset_id"})
            != window.source_digest
        ):
            raise ValueError("scheduled source scope differs from its declared digest")
        start, stop = validate_allocation_objective(
            record["objective"], recipe, recipe_digest, clock
        )
        if (
            length * clock["decision_interval_seconds"]
            != clock["economic_horizon_seconds"]
            or int(receipt_datetime_ns(start).astype("int64"))
            != window.decision_start_ns
            or int(receipt_datetime_ns(stop).astype("int64")) != window.terminal_time_ns
            or parse_source_timestamp(source["decision_start"])
            != receipt_datetime_ns(start)
            or parse_source_timestamp(source["terminal_time"])
            != receipt_datetime_ns(stop)
        ):
            raise ValueError("scheduled source/objective clock differs from its window")
        offset += length
    if frozen:
        declaration = AllocationFeaturePreprocessing.from_payload(
            recipe["observation"]["feature_preprocessing"]
        )
        if (
            canonical_json_bytes(values["preprocessing_fit"])
            != canonical_json_bytes(preprocessing_fit_payload(declaration))
            or not any(
                w.dataset_id == declaration.normalizer.source_dataset_id
                for w in schedule.training_windows
            )
            or any(
                w.decision_start_ns < declaration.policy_start_time_ns
                for w in schedule.training_windows
            )
        ):
            raise ValueError(
                "scheduled preprocessing differs from its fit source or clock"
            )
