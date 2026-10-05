"""Training-only source consumption receipt and linkage; no source authentication."""

from __future__ import annotations

from typing import Any

from trade_rl._validation import require_sha256
from trade_rl.strategies.rl.allocation_clock_receipt import (
    validate_allocation_clock,
    validate_ppo_training_budget,
)
from trade_rl.strategies.rl.allocation_objective_receipt import (
    validate_allocation_objective,
)
from trade_rl.strategies.rl.allocation_receipt_time import (
    parse_source_timestamp,
    receipt_datetime_ns,
)


def _mapping(value: object, keys: set[str], field: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError(f"{field} must contain exactly its declared fields")
    return value


def _positive_integer(value: object, field: str, *, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{field} must be an integer >= {minimum}")
    return value


def validate_allocation_training(
    training: object, recipe: dict[str, Any], recipe_digest: str
) -> None:
    values = _mapping(
        training,
        {
            "seed",
            "requested_timesteps",
            "actual_timesteps",
            "clock",
            "objective",
            "source",
            "ppo",
        },
        "training",
    )
    ppo = validate_ppo_training_budget(values)
    clock = validate_allocation_clock(values["clock"], ppo, recipe["runtime_profile"])
    start, stop = validate_allocation_objective(
        values["objective"], recipe, recipe_digest, clock
    )
    source = _mapping(
        values["source"],
        {
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
        },
        "training source",
    )
    for name in (
        "dataset_id",
        "feature_consumption_digest",
        "clock_consumption_digest",
        "execution_consumption_digest",
        "forecast_consumption_digest",
        "cost_consumption_digest",
    ):
        require_sha256(source[name], field=name)
    beginning = _positive_integer(source["start_index"], "start_index", minimum=0)
    end = _positive_integer(source["stop_index"], "stop_index")
    indices, counts = source["decision_indices"], source["decision_counts"]
    if (
        not isinstance(indices, list)
        or not indices
        or any(
            isinstance(index, bool)
            or not isinstance(index, int)
            or not beginning <= index < end
            for index in indices
        )
        or indices != sorted(set(indices))
        or not isinstance(counts, list)
        or len(counts) != len(indices)
    ):
        raise ValueError(
            "sampled decision rows must be sorted unique indices within the episode"
        )
    for count in counts:
        _positive_integer(count, "decision_count")
    if indices != list(range(beginning, beginning + len(indices))) or any(
        left < right for left, right in zip(counts, counts[1:])
    ):
        raise ValueError(
            "sampled decisions must form a reset prefix with nonincreasing counts"
        )
    if sum(counts) != values["actual_timesteps"]:
        raise ValueError(
            "sampled decision counts differ from the realized training budget"
        )
    observations = source["observation_indices"]
    eligible_observations = {beginning, *indices, *(index + 1 for index in indices)}
    if (
        not isinstance(observations, list)
        or not observations
        or any(
            isinstance(index, bool)
            or not isinstance(index, int)
            or not beginning <= index < end
            for index in observations
        )
        or observations != sorted(set(observations))
        or not set(indices).issubset(observations)
        or not set(observations).issubset(eligible_observations)
    ):
        raise ValueError(
            "consumed observation rows must include every decision and only eligible episode successors"
        )
    if (end - beginning) * clock["decision_interval_seconds"] != clock[
        "economic_horizon_seconds"
    ]:
        raise ValueError("training source indices differ from the declared horizon")
    if (
        not isinstance(source["symbol"], str)
        or not source["symbol"].strip()
        or source["feature_names"] != recipe["feature_names"]
    ):
        raise ValueError("training source schema differs from the recipe")
    for name, time in (("decision_start", start), ("terminal_time", stop)):
        stamp = parse_source_timestamp(source[name])
        if stamp != receipt_datetime_ns(time):
            raise ValueError("training source timestamps differ from the objective")
