"""Closed settled witnesses and local admission, never runtime authentication.

This reader retains failed and DD-ineligible episodes. It derives settlement
from exact native frontier facts; upper owners prove source, append, buffer and
optimizer observation completeness. A matching digest proves neither a fit nor
economic readiness. Legacy receipt readers are deliberately not dispatched here.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from fractions import Fraction
from hashlib import sha256
from typing import Any, cast

from trade_rl._validation import require_sha256
from trade_rl.artifacts.canonical import canonical_json_bytes
from trade_rl.artifacts.hashing import content_digest
from trade_rl.strategies.rl.allocation_protocol_receipt import (
    _validate_protocol_optimization,
)
from trade_rl.strategies.rl.allocation_recipe_v4 import (
    validate_settled_allocation_recipe,
)
from trade_rl.strategies.rl.allocation_settled_clock import (
    SettledAllocationClockContract,
)
from trade_rl.strategies.rl.allocation_training_protocol import (
    AllocationPPOTrainingProtocol,
    _native_json,
)

_TRAINING_KEYS = {
    "schema",
    "seed",
    "requested_timesteps",
    "actual_timesteps",
    "ppo",
    "protocol",
    "protocol_digest",
    "clock",
    "objective",
    "source",
    "episodes",
    "bindings",
    "optimization",
    "failure",
}
_EPISODE_KEYS = {
    "schema",
    "episode_index",
    "actor_transition_indices",
    "processing_rows",
    "final_book",
    "final_active_orders",
    "failure",
}
_BOOK_KEYS = {
    "as_of_index",
    "as_of_dataset_id",
    "cash",
    "equity",
    "peak_value",
    "exact_quantities",
    "max_drawdown",
    "insolvent",
    "termination_reason",
}
_FAILURE_KEYS = {
    "stage",
    "attempted_start_index",
    "error_type",
    "error_message",
}
_NATIVE_REASONS = {
    "minimum_equity",
    "execution_cost_exhaustion",
    "margin_call",
    "liquidation",
    "drawdown_stop",
    "insolvency",
}
_ACTIVE_STATUSES = {
    "submitted",
    "latency_wait",
    "eligible",
    "triggered",
    "partially_filled",
}
_DATETIME = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?"
    r"(?:Z|[+-](?:[01]\d|2[0-3]):[0-5]\d)$"
)
_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)


def _mapping(value: object, keys: set[str], field: str) -> dict[str, Any]:
    if type(value) is not dict or set(value) != keys:
        raise ValueError(f"{field} requires exactly its declared native JSON fields")
    return value


def _integer(value: object, field: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{field} must be a native integer >= {minimum}")
    return cast(int, value)


def _number(value: object, field: str) -> float:
    if type(value) not in (int, float):
        raise ValueError(f"{field} must be a finite native non-boolean number")
    try:
        result = float(cast(int | float, value))
    except OverflowError as error:
        raise ValueError(f"{field} is outside finite reporting") from error
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


def _text(value: object, field: str, *, nonempty: bool = True) -> str:
    if type(value) is not str or nonempty and not value.strip():
        raise ValueError(f"{field} must be a native string")
    return cast(str, value)


def _digest(value: object, field: str) -> str:
    return require_sha256(_text(value, field), field=field)


def _fraction(value: object, field: str) -> Fraction:
    text = _text(value, field)
    try:
        result = Fraction(text)
        if str(result) != text or not math.isfinite(float(result)):
            raise ValueError(f"{field} must be a canonical finite exact quantity")
        return result
    except (ValueError, ZeroDivisionError, OverflowError) as error:
        raise ValueError(
            f"{field} must be a canonical finite exact quantity"
        ) from error


def _failure(value: object) -> dict[str, Any] | None:
    if value is None:
        return None
    failure = _mapping(value, _FAILURE_KEYS, "failure")
    _text(failure["stage"], "failure.stage")
    _text(failure["error_type"], "failure.error_type")
    _text(failure["error_message"], "failure.error_message", nonempty=False)
    if failure["attempted_start_index"] is not None:
        _integer(failure["attempted_start_index"], "failure.attempted_start_index")
    return failure


def _timestamp_ns(value: object) -> int:
    text = _text(value, "objective timestamp")
    if not _DATETIME.fullmatch(text):
        raise ValueError("objective timestamps require complete aware ISO datetimes")
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        elapsed = parsed.astimezone(UTC) - _EPOCH
    except (ValueError, OverflowError) as error:
        raise ValueError("invalid objective timestamp") from error
    result = (
        (elapsed.days * 86400 + elapsed.seconds) * 1_000_000 + elapsed.microseconds
    ) * 1000
    if not -(2**63) < result < 2**63:
        raise ValueError("objective timestamp is outside native non-NaT int64 ns")
    return result


def _objective(
    value: object, recipe: dict[str, Any], recipe_digest: str
) -> tuple[float, float, int, int]:
    fixed = {
        "schema": "net_profit_objective_v1",
        "objective_id": "expected_terminal_net_profit_v1",
        "tax_treatment": "pretax",
        "infrastructure_cost_treatment": "reported_separately",
        "external_cash_flow_convention": "net_deposits_positive_withdrawals_negative",
        "terminal_valuation": "settled",
    }
    objective = _mapping(
        value,
        set(fixed)
        | {
            "capital",
            "evaluation_start",
            "evaluation_stop_exclusive",
            "economics_digest",
            "risk_digest",
            "deployment_recipe_digest",
            "maximum_drawdown",
        },
        "objective",
    )
    if any(objective[key] != expected for key, expected in fixed.items()):
        raise ValueError("settled training requires its fixed net-profit objective")
    profile = recipe["runtime_profile"]
    capital = _mapping(
        objective["capital"], {"mode", "currency", "initial_equities"}, "capital"
    )
    expected = {
        "mode": "independent_symbol",
        "currency": profile["currency"],
        "initial_equities": [profile["initial_capital"]],
    }
    if canonical_json_bytes(capital) != canonical_json_bytes(expected):
        raise ValueError("objective capital differs from the settled recipe")
    for field in ("economics_digest", "risk_digest", "deployment_recipe_digest"):
        _digest(objective[field], field)
    if objective["deployment_recipe_digest"] != recipe_digest or any(
        objective[key] != profile[key] for key in ("economics_digest", "risk_digest")
    ):
        raise ValueError("objective profiles differ from the settled recipe")
    limit = _number(objective["maximum_drawdown"], "maximum_drawdown")
    if not 0.0 < limit <= 0.20:
        raise ValueError("maximum_drawdown must be within (0, 0.20]")
    return (
        _number(profile["initial_capital"], "initial_capital"),
        limit,
        _timestamp_ns(objective["evaluation_start"]),
        _timestamp_ns(objective["evaluation_stop_exclusive"]),
    )


def _source(
    value: object, clock: SettledAllocationClockContract, start_ns: int, stop_ns: int
) -> dict[str, Any]:
    source = _mapping(
        value,
        {
            "dataset_id",
            "symbol",
            "symbol_count",
            "symbol_index",
            "start_index",
            "stop_index",
            "start_time_ns",
            "terminal_time_ns",
        },
        "source",
    )
    _digest(source["dataset_id"], "source.dataset_id")
    _text(source["symbol"], "source.symbol")
    count = _integer(source["symbol_count"], "source.symbol_count", 1)
    if _integer(source["symbol_index"], "source.symbol_index") >= count:
        raise ValueError("source symbol_index must belong to the full native vector")
    begin = _integer(source["start_index"], "source.start_index")
    end = _integer(source["stop_index"], "source.stop_index")
    for field in ("start_time_ns", "terminal_time_ns"):
        item = source[field]
        if type(item) is not int or not -(2**63) < item < 2**63:
            raise ValueError(f"source.{field} must be native non-NaT int64 ns")
    if (
        end - begin != clock.physical_horizon_steps
        or source["start_time_ns"] != start_ns
        or source["terminal_time_ns"] != stop_ns
        or stop_ns - start_ns
        != clock.physical_horizon_steps * clock.execution_interval_seconds * 10**9
    ):
        raise ValueError("source and objective must bind the same complete H interval")
    return source


def _book(
    value: object, source: dict[str, Any], initial_capital: float
) -> tuple[dict[str, Any] | None, bool, bool]:
    if value is None:
        return None, False, False
    book = _mapping(value, _BOOK_KEYS, "final_book")
    if book["as_of_index"] is not None:
        _integer(book["as_of_index"], "final_book.as_of_index")
    if book["as_of_dataset_id"] is not None:
        _digest(book["as_of_dataset_id"], "final_book.as_of_dataset_id")
    for key in ("cash", "equity", "peak_value", "max_drawdown"):
        _number(book[key], f"final_book.{key}")
    equity, peak, dd = (book[key] for key in ("equity", "peak_value", "max_drawdown"))
    if peak <= 0.0 or dd < 0.0:
        raise ValueError("native peak must be positive and native DD nonnegative")
    if type(book["insolvent"]) is not bool:
        raise ValueError("final_book.insolvent must be a native boolean")
    reason = book["termination_reason"]
    if reason is not None and (
        type(reason) is not str or reason not in _NATIVE_REASONS
    ):
        raise ValueError("unknown native economic termination reason")
    quantities = book["exact_quantities"]
    if type(quantities) is not list or len(quantities) != source["symbol_count"]:
        raise ValueError("final_book requires its full native exact-quantity vector")
    exact = tuple(_fraction(item, "final_book.exact_quantities") for item in quantities)
    flat = all(item == 0 for item in exact)
    sane = (
        equity > 0.0
        and not book["insolvent"]
        and reason is None
        and peak >= initial_capital
        and peak >= equity
        and dd >= 1.0 - max(equity, 0.0) / peak
        and (not flat or book["cash"] == book["equity"])
    )
    return book, flat, sane


def _orders(value: object) -> list[dict[str, Any]]:
    if type(value) is not list:
        raise ValueError("final_active_orders must be a native JSON array")
    identities: set[str] = set()
    for item in value:
        order = _mapping(
            item,
            {
                "order_id",
                "status",
                "requested_quantity",
                "remaining_quantity",
                "exact_cumulative_filled_quantity",
            },
            "active order",
        )
        identity = _digest(order["order_id"], "active order.order_id")
        if identity in identities:
            raise ValueError("active order identities must be unique")
        identities.add(identity)
        if type(order["status"]) is not str or order["status"] not in _ACTIVE_STATUSES:
            raise ValueError("final_active_orders requires native active statuses")
        _number(order["requested_quantity"], "active order.requested_quantity")
        _number(order["remaining_quantity"], "active order.remaining_quantity")
        _fraction(
            order["exact_cumulative_filled_quantity"], "active order.filled_quantity"
        )
    return value


def _rows(
    value: object,
    source: dict[str, Any],
    clock: SettledAllocationClockContract,
    *,
    failure_present: bool,
) -> tuple[int, bool]:
    if type(value) is not list or len(value) > clock.physical_horizon_steps:
        raise ValueError("processing_rows must be a bounded native JSON array")
    valid = True
    actors = 0
    for offset, item in enumerate(value):
        row = _mapping(
            item,
            {
                "kind",
                "start_index",
                "processing_index",
                "bars_advanced",
                "returned_clock_types",
            },
            "processing row",
        )
        kind = "actor" if offset < clock.actor_steps else "reserve"
        if row["kind"] != kind:
            raise ValueError("native rows must retain actor-then-reserve chronology")
        actors += kind == "actor"
        _integer(row["start_index"], "processing row.start_index")
        types = _mapping(
            row["returned_clock_types"],
            {"next_index", "bars_advanced"},
            "returned_clock_types",
        )
        for field in types:
            _text(types[field], f"returned_clock_types.{field}")
        for field in ("processing_index", "bars_advanced"):
            if row[field] is not None and type(row[field]) not in (
                str,
                bool,
                int,
                float,
            ):
                raise ValueError("returned clocks require native diagnostic scalars")
        expected = source["start_index"] + offset
        current_valid = (
            row["start_index"] == expected
            and type(row["processing_index"]) is int
            and row["processing_index"] == expected + 1
            and type(row["bars_advanced"]) is int
            and row["bars_advanced"] == 1
            and types == {"next_index": "int", "bars_advanced": "int"}
        )
        if not current_valid and (not failure_present or offset != len(value) - 1):
            raise ValueError("invalid returned clock requires a final failure witness")
        valid = valid and current_valid
    return actors, valid


def _episode_state(
    value: object,
    position: int,
    total: int,
    source: dict[str, Any],
    clock: SettledAllocationClockContract,
    initial_capital: float,
    limit: float,
    *,
    global_failure: bool,
) -> tuple[str, int]:
    episode = _mapping(value, _EPISODE_KEYS, "episode")
    if episode["schema"] != "allocation_settled_episode_witness_v1":
        raise ValueError("unsupported settled episode witness")
    if _integer(episode["episode_index"], "episode_index") != position:
        raise ValueError("episode witnesses must retain their exact observed order")
    indices = episode["actor_transition_indices"]
    if (
        type(indices) is not list
        or len(indices) > clock.actor_steps
        or any(
            type(item) is not int or item != index for index, item in enumerate(indices)
        )
    ):
        raise ValueError("actor_transition_indices must be an exact native prefix")
    count = len(indices)
    failure = _failure(episode["failure"])
    failed = failure is not None or global_failure
    actors, valid_rows = _rows(
        episode["processing_rows"], source, clock, failure_present=failed
    )
    if actors not in (count, count + int(failed)):
        raise ValueError("native actor rows differ from the actual append prefix")
    book, flat, sane = _book(episode["final_book"], source, initial_capital)
    orders = _orders(episode["final_active_orders"])
    if book is None and not failed:
        raise ValueError("an unavailable native frontier requires an explicit failure")
    if count == 0 and not episode["processing_rows"] and not failed:
        raise ValueError("a prepared reset is not an observed episode witness")
    economic_end = book is not None and (
        book["termination_reason"] is not None
        or book["insolvent"]
        or book["equity"] <= 0
    )
    ended = (
        failure is not None
        or economic_end
        or count == clock.actor_steps
        or (actors == clock.actor_steps)
    )
    complete = (
        ended
        and count == clock.actor_steps
        and failure is None
        and valid_rows
        and len(episode["processing_rows"]) == clock.physical_horizon_steps
        and book is not None
        and sane
        and flat
        and not orders
        and book["as_of_index"] == source["stop_index"]
        and book["as_of_dataset_id"] == source["dataset_id"]
    )
    if complete:
        assert book is not None
        return ("eligible" if book["max_drawdown"] <= limit else "settled"), count
    if ended:
        return "unsettled", count
    live = (
        failure is None
        and 0 < count < clock.actor_steps
        and actors == count
        and valid_rows
        and len(episode["processing_rows"]) == count
        and book is not None
        and sane
        and book["as_of_index"] == source["start_index"] + count
        and book["as_of_dataset_id"] == source["dataset_id"]
    )
    if live:
        if position != total - 1:
            raise ValueError("a live budget cut may only be the last episode witness")
        return "live", count
    if not failed:
        raise ValueError("an incomplete ordinary frontier requires an explicit failure")
    return "incomplete", count


def _validated(
    value: object, recipe: dict[str, Any], recipe_digest: str
) -> tuple[bytes, tuple[str, ...], bool]:
    _native_json(value)
    training = _mapping(value, _TRAINING_KEYS, "settled training receipt")
    if training["schema"] != "allocation_settled_training_receipt_v1":
        raise ValueError("unsupported settled training receipt")
    _native_json(recipe)
    validate_settled_allocation_recipe(recipe)
    if _digest(recipe_digest, "recipe_digest") != content_digest(recipe):
        raise ValueError("recipe_digest differs from the actual settled recipe")
    clock = SettledAllocationClockContract.from_payload(training["clock"])
    if canonical_json_bytes(clock.payload()) != canonical_json_bytes(
        recipe["settled_clock"]
    ):
        raise ValueError("training clock differs from the settled recipe")
    protocol = AllocationPPOTrainingProtocol.from_payload(training["protocol"])
    if _digest(training["protocol_digest"], "protocol_digest") != protocol.digest:
        raise ValueError("protocol_digest differs from its resolved protocol")
    declaration: dict[str, Any] = protocol.payload()
    summary = _mapping(
        training["ppo"],
        {
            "n_steps",
            "batch_size",
            "n_epochs",
            "gamma",
            "gae_lambda",
            "learning_rate",
            "net_arch",
            "device",
        },
        "ppo",
    )
    expected = {
        **{
            key: declaration["ppo"][key]
            for key in (
                "n_steps",
                "batch_size",
                "n_epochs",
                "gamma",
                "gae_lambda",
                "learning_rate",
            )
        },
        "net_arch": declaration["policy"]["net_arch"],
        "device": "cpu",
    }
    if canonical_json_bytes(summary) != canonical_json_bytes(expected) or any(
        getattr(protocol, field) != getattr(clock, clock_field)
        for field, clock_field in (
            ("n_steps", "rollout_steps"),
            ("gamma", "gamma"),
            ("gae_lambda", "gae_lambda"),
        )
    ):
        raise ValueError("PPO summary, protocol and actor rollout clock must agree")
    _integer(training["seed"], "seed")
    requested = _integer(training["requested_timesteps"], "requested_timesteps", 1)
    actual = _integer(training["actual_timesteps"], "actual_timesteps")
    if actual > requested:
        raise ValueError("actual append budget exceeds the fixed requested budget")
    initial, limit, start_ns, stop_ns = _objective(
        training["objective"], recipe, recipe_digest
    )
    source = _source(training["source"], clock, start_ns, stop_ns)
    bindings = _mapping(
        training["bindings"],
        {
            "schema",
            "source_consumption_digest",
            "observation_consumption_digest",
            "transition_trace_digest",
            "buffer_digest",
            "append_ledger_digest",
        },
        "bindings",
    )
    if bindings["schema"] != "allocation_settled_upper_proof_bindings_v1":
        raise ValueError("unsupported settled upper proof bindings")
    for field in bindings.keys() - {"schema"}:
        _digest(bindings[field], field)
    failure = _failure(training["failure"])
    episodes = training["episodes"]
    if type(episodes) is not list:
        raise ValueError("episodes must be a native ordered witness array")
    derived = tuple(
        _episode_state(
            episode,
            index,
            len(episodes),
            source,
            clock,
            initial,
            limit,
            global_failure=failure is not None,
        )
        for index, episode in enumerate(episodes)
    )
    if sum(count for _, count in derived) != actual:
        raise ValueError(
            "actual actor timesteps differ from the observed append prefixes"
        )
    if training["optimization"] is not None:
        _validate_protocol_optimization(training)
    complete_fit = (
        actual == requested
        and requested % protocol.n_steps == 0
        and failure is None
        and training["optimization"] is not None
    )
    return (
        canonical_json_bytes(training),
        tuple(state for state, _ in derived),
        complete_fit,
    )


@dataclass(frozen=True, slots=True, init=False)
class SettledAllocationTrainingReceipt:
    """Immutable diagnostic receipt with independently derived local admission."""

    _raw: bytes
    _states: tuple[str, ...]
    _complete_fit: bool

    def __init__(
        self, value: object, *, recipe: dict[str, Any], recipe_digest: str
    ) -> None:
        try:
            raw, states, complete_fit = _validated(value, recipe, recipe_digest)
        except (KeyError, TypeError, OverflowError, RecursionError) as error:
            raise ValueError("invalid settled allocation training receipt") from error
        object.__setattr__(self, "_raw", raw)
        object.__setattr__(self, "_states", states)
        object.__setattr__(self, "_complete_fit", complete_fit)

    @classmethod
    def from_payload(
        cls, value: object, *, recipe: dict[str, Any], recipe_digest: str
    ) -> SettledAllocationTrainingReceipt:
        return cls(value, recipe=recipe, recipe_digest=recipe_digest)

    def payload(self) -> dict[str, Any]:
        return cast(dict[str, Any], json.loads(self._raw))

    @property
    def digest(self) -> str:
        return sha256(self._raw).hexdigest()

    @property
    def completed_episode_count(self) -> int:
        return sum(
            state in {"unsettled", "settled", "eligible"} for state in self._states
        )

    @property
    def structurally_settled_episode_count(self) -> int:
        return sum(state in {"settled", "eligible"} for state in self._states)

    @property
    def eligible_episode_count(self) -> int:
        return self._states.count("eligible")

    @property
    def live_episode_count(self) -> int:
        return self._states.count("live")

    def require_admission(self) -> None:
        if (
            not self._complete_fit
            or "incomplete" in self._states
            or self.completed_episode_count == 0
            or self.structurally_settled_episode_count != self.completed_episode_count
            or self.eligible_episode_count == 0
        ):
            raise ValueError(
                "settled admission requires a completed fit and eligible settled episodes"
            )


def validate_allocation_settled_training(
    training: object, recipe: dict[str, Any], recipe_digest: str
) -> None:
    """Necessary local bundle-v6 gate; upper proof verification remains mandatory."""
    SettledAllocationTrainingReceipt.from_payload(
        training, recipe=recipe, recipe_digest=recipe_digest
    ).require_admission()


__all__ = [
    "SettledAllocationTrainingReceipt",
    "validate_allocation_settled_training",
]
