"""Validate an observed settled fit; never execute an account or learner.

Numeric/account/source projections are validated from detached native returns.
Opaque context IDs are observed identities, not
authentication of omitted order archives, return history, or executor RNG.
"""

from __future__ import annotations

import json
import math
from dataclasses import fields, replace
from fractions import Fraction
from hashlib import sha256
from typing import TYPE_CHECKING, Any

import numpy as np

from trade_rl.artifacts import canonical_json_bytes, content_digest
from trade_rl.evaluation.objectives import net_equity_increment
from trade_rl.evaluation.rl_allocation.settled_training_source import (
    settled_training_source_receipt,
)
from trade_rl.evaluation.rl_allocation.transition_facts import (
    _construct,
    _json,
    _pending,
    _returned_book,
    validate_allocation_execution_facts_v2,
    validate_terminal_allocation_facts,
)
from trade_rl.risk.pretrade import PreTradeRisk
from trade_rl.simulation.accounting import BookState
from trade_rl.simulation.orders.model import PendingOrder
from trade_rl.simulation.quantities import (
    accepted_fill_quantity,
    exact_quantity,
    project_quantity,
)
from trade_rl.strategies.allocation import AllocationContext, AllocationInputs
from trade_rl.strategies.allocation_action import AllocationDecision
from trade_rl.strategies.allocation_snapshot import AllocationAccountSnapshot
from trade_rl.strategies.rl.allocation_observation_encoder_v2 import (
    encode_allocation_observation_v2,
)
from trade_rl.strategies.rl.allocation_training_protocol import (
    AllocationPPOTrainingProtocol,
)

if TYPE_CHECKING:
    from trade_rl.evaluation.rl_allocation.settled_env import (
        SettledAllocationTradingEnv,
    )

_FIELDS = (
    "observations",
    "actions",
    "rewards",
    "episode_starts",
    "values",
    "log_probs",
    "advantages",
    "returns",
)
_ROLES = (
    "actor",
    "live_boundary_critic",
    "prepared_reset_boundary_critic",
    "terminal_sentinel",
)
_ID_KEYS = set(
    "episode_id actor_step actor_index time_ns account_id recipe_digest source_envelope_digest reset_ordinal".split()
)
_EVENT_KEYS = {
    "reset": set(
        "reset_ordinal prepared identity observation native_account observation_witness".split()
    ),
    "training_start": {"actual_timesteps"},
    "rollout_start": {"rollout", "actual_timesteps"},
    "actor_forward": set("identity actor parameters_digest input_ordinal".split()),
    "env_transition": set(
        "identity settled_transition_hex supplied_action returned_observation reward terminated truncated observation_witness".split()
    ),
    "stage": set(
        "sequence rollout local actor_forward_ordinal env_transition_ordinal identity episode_start done actor terminal next_observation next_identity next_role".split()
    ),
    "append": set(
        "sequence rollout local pos_before pos_after stage_ordinal arguments stored_row actual_timesteps".split()
    ),
    "critic": set("identity role input result parameters_digest input_ordinal".split()),
    "rollout": set(
        "rollout first last critic_ordinal last_values dones same_critic_result_object chronological_buffer parameters_before actual_timesteps".split()
    ),
    "rollout_end": {"rollout", "actual_timesteps"},
    "update": set(
        "rollout actual_timesteps optimizer_calls epochs n_updates diagnostics parameters_after".split()
    ),
    "training_end": {"actual_timesteps"},
}
_ROW_KEYS = set(
    "kind start_index processing_index clock_valid returned_clock_types opening_equity before_state_digest after_state_digest risk native accepted_orders".split()
)
_PAYLOAD_KEYS = set(
    "schema episode_id account_id actor_step actor_index bound_digest recipe_digest source_envelope_digest initial_capital rows reward terminated truncated settlement_witness".split()
)
_WITNESS_KEYS = set(
    "actor_steps_processed physical_steps_processed reserve_steps_processed full_coverage settled eligible maximum_drawdown frontier_index native_frontier".split()
)
_LOWER_BOOK_KEYS = set(
    "as_of_index as_of_dataset_id cash equity peak_value exact_quantities max_drawdown insolvent termination_reason".split()
)


def _closed(value: Any, keys: set[str]) -> dict[str, Any]:
    if type(value) is not dict or set(value) != keys:
        raise ValueError("settled capture has missing or undeclared fields")
    return value


def _native(value: Any) -> None:
    if value is None or type(value) in (str, int, bool):
        return
    if type(value) is float and math.isfinite(value):
        return
    if type(value) is list:
        for item in value:
            _native(item)
        return
    if type(value) is dict and all(type(k) is str for k in value):
        for item in value.values():
            _native(item)
        return
    raise ValueError("settled capture must preserve finite native JSON")


def _same(left: Any, right: Any) -> None:
    if canonical_json_bytes(left) != canonical_json_bytes(right):
        raise ValueError(
            "settled observed facts differ from independently derived facts"
        )


def _integer(value: Any, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError("settled capture requires a native bounded integer")
    return value


def _digest(value: Any) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(c not in "0123456789abcdef" for c in value)
    ):
        raise ValueError("settled capture requires a lowercase native SHA256")
    return value


def _tensor(value: Any, dtype: str, shape: tuple[int, ...]) -> np.ndarray:
    raw = _closed(value, {"dtype", "shape", "order", "hex"})
    _same(
        {k: raw[k] for k in ("dtype", "shape", "order")},
        {"dtype": dtype, "shape": list(shape), "order": "C"},
    )
    if type(raw["hex"]) is not str:
        raise ValueError("tensor bytes must be a native lowercase hexadecimal string")
    try:
        data = bytes.fromhex(raw["hex"])
    except ValueError as error:
        raise ValueError("invalid observed tensor bytes") from error
    if (
        data.hex() != raw["hex"]
        or len(data) != math.prod(shape) * np.dtype(dtype).itemsize
    ):
        raise ValueError("tensor bytes differ from the exact declared dtype/shape")
    array = np.frombuffer(data, dtype=np.dtype(dtype)).copy().reshape(shape)
    if not np.isfinite(array).all():
        raise ValueError("observed tensor is nonfinite")
    return array


def _record(value: np.ndarray) -> dict[str, Any]:
    return {
        "dtype": value.dtype.str,
        "shape": list(value.shape),
        "order": "C",
        "hex": value.tobytes(order="C").hex(),
    }


def _time(env: Any, index: int) -> int:
    return int(env.dataset.timestamps[index].astype("datetime64[ns]").astype(np.int64))


def _identity(
    env: Any, value: Any, *, episode: int, step: int, index: int, reset: int
) -> None:
    _closed(value, _ID_KEYS)
    _same(
        value,
        {
            "episode_id": episode,
            "actor_step": step,
            "actor_index": index,
            "time_ns": _time(env, index),
            "account_id": env.account_id,
            "recipe_digest": env.recipe_digest,
            "source_envelope_digest": env._source_digest,
            "reset_ordinal": reset,
        },
    )


def _state(env: Any, episode: int, index: int, account: dict[str, Any]) -> str:
    return content_digest(
        {
            "schema": "allocation_settled_visible_state_v1",
            "bound_digest": env.bound.digest,
            "recipe_digest": env.recipe_digest,
            "source_envelope_digest": env._source_digest,
            "episode_id": episode,
            "index": index,
            **account,
        }
    )


def _account(env: Any, raw: Any, index: int) -> BookState:
    raw = _closed(raw, {"book", "active_orders"})
    book = _returned_book(
        {
            "book": raw["book"],
            "processing_index": index,
            "symbol_index": env.symbol_index,
        },
        dataset_id=env.dataset.dataset_id,
    )
    if len(book.quantities) != env.dataset.n_symbols:
        raise ValueError("native account lost its full symbol vector")
    if book.peak_value <= 0 or book.max_drawdown < max(
        0.0, 1.0 - max(book.portfolio_value, 0.0) / book.peak_value
    ):
        raise ValueError("native maximum drawdown hides its current drawdown")
    _same(
        raw["active_orders"],
        [_pending(PendingOrder.from_mapping(v)) for v in raw["active_orders"]],
    )
    _same(
        raw["book"]["mark_prices"],
        env.dataset.resolved_array("mark_price")[index].tolist(),
    )
    _same(
        raw["book"]["contract_multipliers"],
        env.dataset.resolved_array("contract_multipliers").tolist(),
    )
    for symbol, quantity in enumerate(book.exact_quantities):
        if symbol != env.symbol_index and quantity != 0:
            raise ValueError("independent account acquired another symbol")
    for order in raw["active_orders"]:
        if (
            order["intent"]["symbol_index"] != env.symbol_index
            or order["intent"]["dataset_id"] != env.dataset.dataset_id
            or order["intent"]["execution_policy_digest"]
            != env.recipe["runtime_profile"]["economics_digest"]
        ):
            raise ValueError("native active order changed source/account policy")
    return book


def _reset_account(env: Any, raw: Any) -> None:
    _account(env, raw, env.start_index)
    expected = BookState(
        np.zeros(env.dataset.n_symbols),
        env.initial_capital,
        env.dataset.resolved_array("mark_price")[env.start_index],
        env.initial_capital,
        env.dataset.resolved_array("contract_multipliers"),
        as_of_index=env.start_index,
        as_of_dataset_id=env.dataset.dataset_id,
    )
    # The only margin declaration of the zero-position reset is the native rate.
    expected.maintenance_margin = env.execution_cost.maintenance_margin_rate
    facts = {
        field.name: _json(getattr(expected, field.name))
        for field in fields(expected)
        if field.name not in ("_exact_quantities", "returns_history")
    }
    facts.update(
        exact_quantities=[str(q) for q in expected.exact_quantities],
        equity=expected.portfolio_value,
    )
    _same(raw, {"book": facts, "active_orders": []})


def _decision(env: Any, raw: Any) -> AllocationDecision:
    value = _closed(
        raw,
        set(
            "baseline action_contract feature_names feature_values max_drawdown initial_capital remaining_steps pending_gross pending_count".split()
        ),
    )
    baseline = _closed(
        value["baseline"],
        {
            "inputs",
            "context",
            "allocator",
            "target_weight",
            "objective_value",
            "decision_digest",
        },
    )
    inputs = _construct(
        AllocationInputs,
        baseline["inputs"],
        clocks=("decision_time", "available_at", "horizon_end"),
    )
    context = _construct(
        AllocationContext, baseline["context"], clocks=("decision_time",)
    )
    allocator = _construct(type(env.allocator), baseline["allocator"])
    decision = AllocationDecision(
        baseline=allocator.propose(inputs, context),
        action_contract=env.action_contract,
        feature_names=tuple(value["feature_names"]),
        feature_values=tuple(value["feature_values"]),
        **{
            key: value[key]
            for key in (
                "max_drawdown",
                "initial_capital",
                "remaining_steps",
                "pending_gross",
                "pending_count",
            )
        },
    )
    _same(raw, _json(decision))
    return decision


def _source_decision(
    env: Any, index: int, account: dict[str, Any], decision: AllocationDecision
) -> None:
    """Check the actually captured proposal inputs; never ask its model to forecast."""
    dataset, symbol, selected = env.dataset, env.symbol_index, env.feature_indices
    timestamp, book = dataset.timestamps[index], account["book"]
    packets = [
        p
        for p in env.stream.packets
        if p.symbol == dataset.symbols[symbol] and p.as_of == timestamp
    ]
    if len(packets) != 1:
        raise ValueError("actual actor clock has no unique causal packet")
    packet = packets[0]
    owners = [v for v in env.stream.vintages if v.digest == packet.vintage_digest]
    if len(owners) != 1:
        raise ValueError("actual actor packet has no unique frozen vintage")
    owner, estimate = owners[0], env._estimates[_time(env, index)]
    forecast_indices = owner.model.feature_indices
    if (
        not dataset.resolved_array("information_available")[index, symbol]
        or dataset.resolved_array("available_at")[index, symbol] > timestamp
        or not np.all(dataset.feature_available[index, symbol, list(selected)])
        or not np.all(dataset.feature_available[index, symbol, list(forecast_indices)])
        or tuple(dataset.feature_names[i] for i in forecast_indices)
        != owner.training.selected_feature_names
        or packet.source_available_at
        != dataset.resolved_array("available_at")[index, symbol]
        or packet.forecast_available_at != timestamp
        or owner.block.inference_delay_seconds != 0
        or packet.horizon_seconds != env.expected_horizon_seconds
        or estimate.horizon_end != packet.horizon_end
        or packet.decision_close != dataset.close[index, symbol]
        or packet.decision_close != book["mark_prices"][symbol]
        or packet.feature_values
        != tuple(
            float(v) for v in dataset.features[index, symbol, list(forecast_indices)]
        )
    ):
        raise ValueError(
            "actual actor feature/forecast source selection is unavailable or changed"
        )
    arguments = estimate._arguments()
    arguments["available_at"] = max(packet.forecast_available_at, estimate.available_at)
    arguments["source_identity"] = content_digest(
        {
            "schema": "forecast_allocation_source_v1",
            "dataset_id": dataset.dataset_id,
            "packet": packet.digest,
            "vintage": owner.digest,
            "estimates": estimate.payload(),
        }
    )
    expected_inputs = AllocationInputs(
        **arguments,
        expected_simple_return=packet.expected_simple_return,
        return_variance=packet.fit_prefix_marginal_variance,
    )
    _same(_json(decision.baseline.inputs), _json(expected_inputs))
    context = decision.baseline.context
    _digest(
        context.state_digest
    )  # observed full-context identity; omitted archive is not recreated
    pending = sum(
        (Fraction(str(o["remaining_quantity"])) for o in account["active_orders"]),
        Fraction(0),
    )
    current = (
        float(book["quantities"][symbol])
        * float(book["mark_prices"][symbol])
        * float(book["contract_multipliers"][symbol])
        / float(book["equity"])
    )
    _same(
        {
            k: _json(getattr(context, k))
            for k in (
                "account_id",
                "symbol",
                "decision_time",
                "current_weight",
                "quantity",
                "cash",
                "equity",
                "pending_remaining",
            )
        },
        {
            "account_id": env.account_id,
            "symbol": dataset.symbols[symbol],
            "decision_time": _time(env, index),
            "current_weight": current,
            "quantity": book["exact_quantities"][symbol],
            "cash": book["cash"],
            "equity": book["equity"],
            "pending_remaining": str(pending),
        },
    )
    cap = min(env.risk_config.max_abs_weight, env.execution_cost.max_leverage)
    resolved = replace(
        env.allocator,
        lower_weight=max(env.allocator.lower_weight, -cap),
        upper_weight=min(env.allocator.upper_weight, cap),
    )
    _same(_json(decision.baseline.allocator), _json(resolved))
    gross = float(
        sum(
            (
                abs(Fraction(str(o["remaining_quantity"])))
                for o in account["active_orders"]
            ),
            Fraction(0),
        )
    )
    gross *= float(book["mark_prices"][symbol]) * float(
        book["contract_multipliers"][symbol]
    )
    _same(
        {
            k: _json(getattr(decision, k))
            for k in (
                "feature_names",
                "feature_values",
                "max_drawdown",
                "initial_capital",
                "remaining_steps",
                "pending_gross",
                "pending_count",
                "action_contract",
            )
        },
        {
            "feature_names": [dataset.feature_names[i] for i in selected],
            "feature_values": [
                float(v) for v in dataset.features[index, symbol, list(selected)]
            ],
            "max_drawdown": book["max_drawdown"],
            "initial_capital": env.initial_capital,
            "remaining_steps": env.settlement_stop_index - index,
            "pending_gross": gross / env.initial_capital,
            "pending_count": len(account["active_orders"]),
            "action_contract": _json(env.action_contract),
        },
    )


def _observation_witness(
    env: Any, index: int, account: dict[str, Any], raw: Any
) -> np.ndarray:
    raw = _closed(raw, {"decision", "snapshot"})
    decision = _decision(env, raw["decision"])
    _source_decision(env, index, account, decision)
    value = _closed(
        raw["snapshot"],
        {"schema"} | {field.name for field in fields(AllocationAccountSnapshot)},
    )
    if value["schema"] != "independent_allocation_account_snapshot_v1":
        raise ValueError("actual encoded snapshot schema changed")
    values = {k: v for k, v in value.items() if k != "schema"}
    values["active_orders"] = tuple(values["active_orders"])
    snapshot = AllocationAccountSnapshot(**values)
    _same(value, snapshot.payload())
    symbol, book = env.symbol_index, account["book"]
    expected_book = dict(book)
    for name in (
        "quantities",
        "mark_prices",
        "contract_multipliers",
        "exact_quantities",
    ):
        expected_book[name] = [book[name][symbol]]
    expected_book["current_drawdown"] = max(
        0.0, 1 - book["equity"] / book["peak_value"]
    )
    # History is present in this actual witness but not consumed by this encoder.
    # Native visible facts omitted it; do not invent/replay it or authenticate it.
    expected_book["returns_history"] = value["book_facts"]["returns_history"]
    history = expected_book["returns_history"]
    if (
        type(history) is not list
        or len(history) > env.bound.clock.physical_horizon_steps
        or any(
            type(value) not in (int, float) or not math.isfinite(value)
            for value in history
        )
    ):
        raise ValueError("actual snapshot contains invalid finite return-history shape")
    _same(value["book_facts"], expected_book)
    expected_orders = []
    for order in account["active_orders"]:
        value_order = _json(PendingOrder.from_mapping(order))
        requested = Fraction(str(order["intent"]["requested_quantity"]))
        cumulative = Fraction(order["exact_cumulative_filled_quantity"])
        value_order["exact_requested_quantity"] = str(requested)
        value_order["exact_remaining_quantity"] = str(requested - cumulative)
        expected_orders.append(value_order)
    _same(value["active_orders"], expected_orders)

    def stamp(item: Any) -> str:
        return str(np.datetime_as_string(item.astype("datetime64[ns]"), unit="ns"))

    _same(
        {
            k: value[k]
            for k in (
                "account_id",
                "dataset_id",
                "symbol",
                "symbol_index",
                "decision_index",
                "decision_time",
                "available_at",
                "execution_policy_digest",
                "risk_digest",
                "source_state_digest",
                "observable_tradable",
            )
        },
        {
            "account_id": env.account_id,
            "dataset_id": env.dataset.dataset_id,
            "symbol": env.dataset.symbols[symbol],
            "symbol_index": symbol,
            "decision_index": index,
            "decision_time": stamp(env.dataset.timestamps[index]),
            "available_at": stamp(
                env.dataset.resolved_array("available_at")[index, symbol]
            ),
            "execution_policy_digest": env.recipe["runtime_profile"][
                "economics_digest"
            ],
            "risk_digest": env.recipe["runtime_profile"]["risk_digest"],
            "source_state_digest": decision.baseline.context.state_digest,
            "observable_tradable": bool(env.dataset.observable_tradable(index)[symbol]),
        },
    )
    return encode_allocation_observation_v2(
        snapshot,
        decision,
        schema=env.observation_schema,
        episode_steps=env.bound.clock.physical_horizon_steps,
    )


def _amount(actual: Any, expected: float, label: str) -> None:
    # Native JSON type/finite validation remains the caller's responsibility.
    # Exact numeric equality allows native accumulated +0 / evidence -0.
    if actual != expected:
        raise ValueError(f"actual native {label} differs from its source arithmetic")


def _fractions(env: Any, previous: int, index: int) -> tuple[float, float]:
    elapsed = env.dataset.elapsed_hours(previous, index)
    elapsed_years = env.dataset.elapsed_year_fraction(previous, index)
    if elapsed <= env.dataset.bar_hours + 1e-12:
        return elapsed_years, 0.0
    processing = elapsed_years * env.dataset.bar_hours / elapsed
    return processing, elapsed_years - processing


def _borrow(
    quantities: np.ndarray,
    marks: np.ndarray,
    multipliers: np.ndarray,
    rates: np.ndarray,
    years: float,
    multiplier: float,
) -> float:
    # Match Book.position_values then MarketExecutor._charge_borrow exactly.
    position_values = quantities * marks * multipliers
    return float(np.sum(np.maximum(-position_values, 0.0) * rates) * years * multiplier)


def _corporate_frontier(
    env: Any, index: int, before: dict[str, Any]
) -> tuple[list[Fraction], float]:
    dataset = env.dataset
    quantities = [Fraction(value) for value in before["exact_quantities"]]
    split = dataset.resolved_array("split_factor")[index]
    if np.any(split != 1.0):
        quantities = [
            q * exact_quantity(float(factor))
            for q, factor in zip(quantities, split, strict=True)
        ]
    inactive = ~dataset.resolved_array("asset_active")[index]
    reporting = np.asarray([project_quantity(q) for q in quantities], dtype=np.float64)
    proceeds = 0.0
    # Native tests the whole mask, then settles ALL inactive symbols.
    if np.any(inactive & (np.abs(reporting) > 1e-12)):
        multipliers = dataset.resolved_array("contract_multipliers")
        recovery = dataset.resolved_array("delisting_recovery")[index]
        proceeds = math.fsum(
            float(
                q
                * exact_quantity(float(price))
                * exact_quantity(float(multiplier))
                * exact_quantity(float(recovered))
            )
            for q, price, multiplier, recovered, settle in zip(
                quantities,
                dataset.open[index],
                multipliers,
                recovery,
                inactive,
                strict=True,
            )
            if settle
        )
        quantities = [
            Fraction(0) if settle else q
            for q, settle in zip(quantities, inactive, strict=True)
        ]
    return quantities, proceeds


def _native_signed_carry(
    env: Any, row: dict[str, Any], before: dict[str, Any]
) -> dict[str, Any]:
    """Validate captured arithmetic; no Book/MarketExecutor/forecast/RNG calls."""
    dataset, index = env.dataset, row["processing_index"]
    previous, native = before["book"], row["native"]
    execution, after = native["execution"], native["book"]
    multipliers = dataset.resolved_array("contract_multipliers")
    due = dataset.resolved_array("funding_due")[index]
    price_rates = dataset.resolved_array("funding_price_rate")[index]
    evidence = execution["funding_evidence"]

    # Runtime starts at zero and adds once per actual physical row. This closes
    # absent_due and early-return holes, even before normal-phase source checks.
    if len(evidence) > 1 or (not np.any(due) and evidence):
        raise ValueError("one native physical row has invalid funding coverage")
    funding_total = 0.0
    for boundary in evidence:
        source_funding = -float(
            np.dot(
                np.asarray(boundary["signed_quantities"], dtype=np.float64)
                * multipliers,
                price_rates * due.astype(np.float64),
            )
        )
        _amount(boundary["funding_amount"], source_funding, "funding boundary")
        funding_total += boundary["funding_amount"]
    _amount(execution["interval_funding"], funding_total, "funding interval")

    rate = float(dataset.resolved_array("cash_rate")[index])
    borrow_rates = dataset.resolved_array("borrow_rate")[index]
    dividend_source = dataset.resolved_array("dividend")[index]
    if rate == 0.0:
        _amount(execution["interval_cash_interest"], 0.0, "zero-rate cash interest")
    if env.execution_cost.borrow_rate_multiplier == 0.0 or not np.any(borrow_rates):
        _amount(execution["interval_borrow_cost"], 0.0, "zero-source borrow")
    if not np.any(dividend_source):
        _amount(execution["interval_dividend"], 0.0, "zero-source dividend")

    if after["termination_reason"] is not None:
        return {"coverage": "phase_independent_only_economic_end"}

    # This is a derived fresh-settled-lane invariant, not a general executor
    # restriction: reset empty; target keeps one or cancels/emits one; HOLD
    # cancels; independent-symbol MARKET allocation visits each order once.
    fills = [
        event
        for event in native["order_events"]
        if event["event_type"] in ("filled", "partial_fill")
    ]
    submitted = [
        event for event in native["order_events"] if event["event_type"] == "submitted"
    ]
    if (
        len(before["active_orders"]) > 1
        or len(native["active_orders"]) > 1
        or len(submitted) > 1
        or len(fills) > 1
    ):
        raise ValueError("actual fresh settled lane lost its single-order invariant")

    processing_years, gap_years = _fractions(env, row["start_index"], index)
    multiplier = env.execution_cost.borrow_rate_multiplier
    cash = float(previous["cash"])
    cash_before_gap = cash
    gap_interest = gap_borrow = 0.0
    if gap_years > 0.0:
        gap_interest = float(cash * rate * gap_years)
        cash += gap_interest
        gap_borrow = _borrow(
            np.asarray(previous["quantities"], dtype=np.float64),
            np.asarray(previous["mark_prices"], dtype=np.float64),
            multipliers,
            borrow_rates,
            gap_years,
            multiplier,
        )
        if gap_borrow > 0.0:
            cash -= gap_borrow
    # Native uses this rounded actual cash delta, not interest minus borrow.
    gap_delta = cash - cash_before_gap
    before_fill, recovery = _corporate_frontier(env, index, previous)
    cash += recovery

    quantities = np.asarray(after["quantities"], dtype=np.float64)
    marks = dataset.open[index].copy()
    if fills:
        event = fills[0]
        symbol = event["symbol_index"]
        if symbol != env.symbol_index:
            raise ValueError("actual single fill acquired another symbol")
        exact_delta = Fraction(after["exact_quantities"][symbol]) - before_fill[symbol]
        if project_quantity(exact_delta) != event["filled_quantity"]:
            raise ValueError("actual fill reporting differs from projected book delta")
        # _record_execution sums the whole native symbol vector, including zeros.
        exact_deltas = [
            Fraction(value) - q
            for value, q in zip(after["exact_quantities"], before_fill, strict=True)
        ]
        fill_prices = dataset.open[index].copy()
        fill_prices[symbol] = event["execution_price"]
        notional = math.fsum(
            project_quantity(q) * float(price) * float(contract_multiplier)
            for q, price, contract_multiplier in zip(
                exact_deltas,
                fill_prices,
                multipliers,
                strict=True,
            )
        )
        cash -= notional + float(execution["interval_cost"])
        marks[symbol] = event["execution_price"]
    else:
        _amount(execution["interval_cost"], 0.0, "cost without an actual fill")
        if after["exact_quantities"] != [str(q) for q in before_fill]:
            raise ValueError(
                "native no-fill row changed its exact post-corporate quantity"
            )

    dividend = float(np.dot(quantities * multipliers, dividend_source))
    cash += dividend
    basis = cash
    if abs(gap_delta) > 1e-12:
        basis -= gap_delta
    processing_interest = float(basis * rate * processing_years)
    cash += processing_interest
    processing_borrow = _borrow(
        quantities,
        marks,
        multipliers,
        borrow_rates,
        processing_years,
        multiplier,
    )
    if processing_borrow > 0.0:
        cash -= processing_borrow
    funding = -float(
        np.dot(quantities * multipliers, price_rates * due.astype(np.float64))
    )
    cash += funding

    _amount(execution["interval_dividend"], 0.0 + dividend, "dividend interval")
    _amount(
        execution["interval_cash_interest"],
        (0.0 + gap_interest) + processing_interest,
        "cash-interest interval",
    )
    _amount(
        execution["interval_borrow_cost"],
        (0.0 + gap_borrow) + processing_borrow,
        "borrow interval",
    )
    _amount(execution["interval_funding"], 0.0 + funding, "source funding interval")
    _amount(after["cash"], cash, "signed returned cash")
    return {"coverage": "normal_native_phases_exact", "recovery_cash": recovery}


def _native_sources(
    env: Any,
    row: dict[str, Any],
    before: dict[str, Any],
    *,
    target_identity: str | None = None,
) -> None:
    native, index, symbol = row["native"], row["processing_index"], env.symbol_index
    execution, book = native["execution"], native["book"]
    _account(env, {"book": book, "active_orders": native["active_orders"]}, index)
    fill_events = [
        event
        for event in native["order_events"]
        if event["event_type"] in ("filled", "partial_fill")
    ]
    _same(execution["fill_count"], len(fill_events))
    _same(
        execution["completed_fill_count"],
        sum(event["event_type"] == "filled" for event in fill_events),
    )
    due = env.dataset.resolved_array("funding_due")[index]
    evidence = execution["funding_evidence"]
    if len(evidence) != int(np.any(due)) and book["termination_reason"] is None:
        raise ValueError("actual funding boundary coverage differs from native source")
    for funding in evidence:
        for key, values in (
            ("funding_due", due),
            ("mark_prices", env.dataset.resolved_array("mark_price")[index]),
            (
                "contract_multipliers",
                env.dataset.resolved_array("contract_multipliers"),
            ),
            ("funding_rates", env.dataset.funding_rate[index]),
            (
                "funding_price_rates",
                env.dataset.resolved_array("funding_price_rate")[index],
            ),
        ):
            _same(funding[key], values.tolist())
        if book["termination_reason"] is None:
            _same(funding["signed_quantities"], book["quantities"])
        if not math.isclose(
            funding["funding_amount"],
            execution["interval_funding"],
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError("native funding amount differs from its interval")
    _native_signed_carry(env, row, before)
    capacities = execution["capacity_evidence"]
    if len(capacities) > 1:
        raise ValueError("independent account emitted another symbol capacity")
    reference = (
        index if env.execution_cost.processing_bar_volume_capacity else index - 1
    )
    prices = (
        env.dataset.open[reference]
        if env.execution_cost.processing_bar_volume_capacity
        else env.dataset.close[reference]
    )
    notional = float(env.dataset.market_notional(reference, prices)[symbol])
    participation = min(
        env.execution_cost.max_participation_rate,
        env.dataset.resolved_array("max_participation_rate")[index, symbol],
    )
    for capacity in capacities:
        _same(
            [
                capacity["processing_volume"],
                capacity["capacity_reference_price"],
                capacity["contract_multiplier"],
                capacity["participation_limit"],
                capacity["market_notional"],
            ],
            [
                float(env.dataset.volume[reference, symbol]),
                float(prices[symbol]),
                float(env.dataset.resolved_array("contract_multipliers")[symbol]),
                float(participation),
                notional,
            ],
        )
        if capacity["initial_capacity_notional"] != notional * participation:
            raise ValueError("native capacity declaration differs from source")
    previous = before["book"]
    for total, interval in (
        ("total_cost", "interval_cost"),
        ("funding_pnl", "interval_funding"),
        ("borrow_cost", "interval_borrow_cost"),
    ):
        if not math.isclose(
            book[total],
            previous[total] + execution[interval],
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError(
                "native accumulated accounting differs across returned rows"
            )
    if (
        book["peak_value"] < previous["peak_value"]
        or book["max_drawdown"] < previous["max_drawdown"]
    ):
        raise ValueError("native peak/drawdown moved backwards")
    accepted = row["accepted_orders"]
    if type(accepted) is not list:
        raise ValueError("native accepted order frontier must be an actual list")
    typed = [PendingOrder.from_mapping(order) for order in accepted]
    _same(accepted, [_pending(order) for order in typed])
    by_id = {order.order_id: order for order in typed}
    event_ids = {event["order_id"] for event in native["order_events"]}
    if len(by_id) != len(typed) or set(by_id) != event_ids:
        raise ValueError(
            "native accepted order witnesses must exactly cover returned event IDs"
        )
    for order in typed:
        if (
            order.intent.dataset_id != env.dataset.dataset_id
            or order.intent.execution_policy_digest
            != env.recipe["runtime_profile"]["economics_digest"]
            or order.intent.symbol_index != env.symbol_index
        ):
            raise ValueError(
                "actual accepted order changed full native source identity"
            )
        if any(
            event["order_id"] == order.order_id and event["event_type"] == "submitted"
            for event in native["order_events"]
        ) and (
            target_identity is None or order.intent.target_identity != target_identity
        ):
            raise ValueError(
                "submitted order differs from the actual risk-constrained decision"
            )
    if book["termination_reason"] is None:
        quantities, _ = _corporate_frontier(env, index, previous)
        previous_orders = {
            order["intent"]["order_id"]: PendingOrder.from_mapping(order)
            for order in before["active_orders"]
        }
        for order_id, order in by_id.items():
            previous_order = previous_orders.get(order_id)
            related = [
                event
                for event in native["order_events"]
                if event["order_id"] == order_id
            ]
            if previous_order is None and related[0]["event_type"] != "submitted":
                raise ValueError(
                    "accepted order began without an actual submitted event"
                )
            if previous_order is not None:
                _same(
                    order.intent.canonical_payload(),
                    previous_order.intent.canonical_payload(),
                )
            if order.exact_cumulative_filled_quantity is None or (
                previous_order is not None
                and previous_order.exact_cumulative_filled_quantity is None
            ):
                raise ValueError(
                    "native accepted order lacks exact cumulative quantity"
                )
            prior = Fraction(0)
            if previous_order is not None:
                previous_quantity = previous_order.exact_cumulative_filled_quantity
                if previous_quantity is None:
                    raise ValueError("prior order lacks exact cumulative quantity")
                prior = Fraction(previous_quantity)
            delta = Fraction(order.exact_cumulative_filled_quantity) - prior
            if order.intent.symbol_index != env.symbol_index:
                raise ValueError("accepted order acquired another symbol")
            reported_fills = math.fsum(event["filled_quantity"] for event in related)
            if not math.isclose(
                float(delta), reported_fills, rel_tol=1e-12, abs_tol=1e-12
            ):
                raise ValueError(
                    "exact accepted cumulative progress differs from actual event fill reporting"
                )
            latest = related[-1]
            _same(
                [order.status.value, order.remaining_quantity],
                [latest["new_status"], latest["remaining_quantity"]],
            )
        for event in native["order_events"]:
            if event["event_type"] in ("filled", "partial_fill"):
                symbol = event["symbol_index"]
                quantity = event["filled_quantity"]
                lot_size = max(
                    float(env.dataset.resolved_array("lot_size")[index, symbol]),
                    env.execution_cost.lot_size,
                )
                if lot_size == 0:
                    accepted_quantity = accepted_fill_quantity(quantity)
                    if quantity == -project_quantity(quantities[symbol]):
                        # This is the native no-lot full projected close rule.
                        accepted_quantity = -quantities[symbol]
                else:
                    # Join the exact returned book delta to its source lot units.
                    # A float projection can alias adjacent large counts, so it
                    # must never be used to guess or restrict the actual count.
                    delta_units = (
                        Fraction(book["exact_quantities"][symbol]) - quantities[symbol]
                    ) / exact_quantity(lot_size)
                    if delta_units.denominator != 1:
                        raise ValueError(
                            "native exact fill is not an integral source lot count"
                        )
                    accepted_quantity = accepted_fill_quantity(
                        quantity, lot_size=lot_size, lot_count=delta_units.numerator
                    )
                quantities[symbol] += accepted_quantity
        _same(book["exact_quantities"], [str(q) for q in quantities])
        starting, ending = previous["equity"], max(book["equity"], 0.0)
        net = max(ending / starting - 1.0, -1.0 + 1e-12)
        gross_value = (
            ending
            + execution["interval_cost"]
            - execution["interval_funding"]
            + execution["interval_borrow_cost"]
            - execution["interval_dividend"]
            - execution["interval_cash_interest"]
        )
        _same(
            [
                execution["interval_net_return"],
                execution["interval_log_return"],
                execution["interval_gross_return"],
            ],
            [net, math.log1p(net), gross_value / starting - 1.0],
        )
    states = {o["intent"]["order_id"]: o["status"] for o in before["active_orders"]}
    remaining = {
        o["intent"]["order_id"]: o["remaining_quantity"]
        for o in before["active_orders"]
    }
    for event in native["order_events"]:
        order_id = event["order_id"]
        if order_id not in states and event["event_type"] != "submitted":
            raise ValueError("native order appeared without actual submission")
        if order_id in states and event["previous_status"] != states[order_id]:
            raise ValueError("native order status chain changed across rows")
        states[order_id], remaining[order_id] = (
            event["new_status"],
            event["remaining_quantity"],
        )
    final_active = {
        o["intent"]["order_id"]: o["status"] for o in native["active_orders"]
    }
    terminal = {"filled", "cancelled", "rejected", "expired"}
    _same(
        final_active,
        {oid: status for oid, status in states.items() if status not in terminal},
    )
    _same(
        {
            o["intent"]["order_id"]: o["remaining_quantity"]
            for o in native["active_orders"]
        },
        {oid: remaining[oid] for oid in final_active},
    )


def _source_selection(env: Any, episode: int, row: dict[str, Any]) -> dict[str, Any]:
    arrays = env.dataset.identity_arrays()
    financial = {}
    for key in sorted(env._source_envelope["native_financial_arrays"]):
        values = (
            arrays[key]
            if key == "contract_multipliers"
            else arrays[key][row["start_index"] : row["processing_index"] + 1]
        )
        financial[key] = {
            "dtype": values.dtype.str,
            "shape": list(values.shape),
            "order": "C",
            "bytes_sha256": sha256(values.tobytes(order="C")).hexdigest(),
        }
    result = {
        "episode_id": episode,
        "kind": row["kind"],
        "start_index": row["start_index"],
        "processing_index": row["processing_index"],
        "native_financial_arrays": financial,
    }
    if row["kind"] == "actor":
        offset = row["start_index"] - env.start_index
        result["actor_features"] = {
            "dtype": env.dataset.features.dtype.str,
            "values": env.dataset.features[
                row["start_index"], env.symbol_index, list(env.feature_indices)
            ].tolist(),
            "available": env.dataset.feature_available[
                row["start_index"], env.symbol_index, list(env.feature_indices)
            ].tolist(),
        }
        result["forecast"] = env._source_envelope["actor_forecasts"][offset]
        result["cost"] = env._source_envelope["actor_cost_declarations"][offset]
        result["forecast_features"] = env._source_envelope[
            "actor_forecast_selected_features"
        ][offset]
    return result


def _observation_selection(
    env: Any, episode: int, index: int, witness: dict[str, Any]
) -> dict[str, Any]:
    """Pin actual environment preparation separately from later model consumption."""
    arrays, offset = env.dataset.identity_arrays(), index - env.start_index
    selected = {}
    for name in (
        "features",
        "feature_available",
        "feature_staleness",
        "feature_staleness_hours",
        "feature_missing_reason",
    ):
        values = arrays[name][index, env.symbol_index, list(env.feature_indices)]
        selected[name] = {
            "dtype": values.dtype.str,
            "shape": list(values.shape),
            "order": "C",
            "bytes_sha256": sha256(values.tobytes(order="C")).hexdigest(),
        }
    return {
        "kind": "observation_preparation",
        "episode_id": episode,
        "actor_index": index,
        "actual_witness_digest": content_digest(witness),
        "selected_actor_features": selected,
        "forecast": env._source_envelope["actor_forecasts"][offset],
        "forecast_features": env._source_envelope["actor_forecast_selected_features"][
            offset
        ],
        "cost": env._source_envelope["actor_cost_declarations"][offset],
    }


def _native_transition(
    env: Any,
    event: dict[str, Any],
    before: dict[str, Any],
    rows: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    try:
        encoded = bytes.fromhex(event["settled_transition_hex"])
        facts = json.loads(encoded)
    except (TypeError, ValueError) as error:
        raise ValueError("invalid actual native transition bytes") from error
    if (
        encoded != canonical_json_bytes(facts)
        or encoded.hex() != event["settled_transition_hex"]
    ):
        raise ValueError("native transition bytes must be exact canonical JSON")
    _closed(facts, _PAYLOAD_KEYS)
    identity = event["identity"]
    _same(
        {
            key: facts[key]
            for key in (
                "schema",
                "episode_id",
                "account_id",
                "actor_step",
                "actor_index",
                "bound_digest",
                "recipe_digest",
                "source_envelope_digest",
                "initial_capital",
                "truncated",
            )
        },
        {
            "schema": "allocation_settled_transition_v1",
            "episode_id": identity["episode_id"],
            "account_id": env.account_id,
            "actor_step": identity["actor_step"],
            "actor_index": identity["actor_index"],
            "bound_digest": env.bound.digest,
            "recipe_digest": env.recipe_digest,
            "source_envelope_digest": env._source_digest,
            "initial_capital": env.initial_capital,
            "truncated": False,
        },
    )
    expected_count = (
        1 + env.bound.clock.reserve_steps
        if identity["actor_step"] + 1 == env.bound.clock.actor_steps
        else 1
    )
    if type(facts["rows"]) is not list or not 1 <= len(facts["rows"]) <= expected_count:
        raise ValueError("native transition has missing or extra physical rows")
    opening = before["book"]["equity"]
    for offset, row in enumerate(facts["rows"]):
        kind = "actor" if offset == 0 else "reserve"
        _closed(row, _ROW_KEYS | ({"proposal"} if kind == "actor" else set()))
        index = identity["actor_index"] + offset
        _same(
            {
                k: row[k]
                for k in (
                    "kind",
                    "start_index",
                    "processing_index",
                    "clock_valid",
                    "returned_clock_types",
                    "opening_equity",
                    "before_state_digest",
                )
            },
            {
                "kind": kind,
                "start_index": index,
                "processing_index": index + 1,
                "clock_valid": True,
                "returned_clock_types": {"next_index": "int", "bars_advanced": "int"},
                "opening_equity": before["book"]["equity"],
                "before_state_digest": _state(
                    env, identity["episode_id"], index, before
                ),
            },
        )
        _integer(row["processing_index"])
        native = row["native"]
        if kind == "actor":
            proposal = _closed(
                row["proposal"],
                set(
                    "decision raw_action action raw_target_weight target_weight is_hold".split()
                ),
            )
            action = int(_tensor(event["supplied_action"], "<i8", ()).item())
            decision = _decision(env, proposal["decision"])
            _source_decision(env, index, before, decision)
            expected = decision.propose(action)
            decision_identity = (
                decision.baseline.decision_digest
                if expected.target_weight == decision.baseline.target_weight
                else expected.digest
            )
            _same(proposal, _json(expected))
            current = np.asarray(before["book"]["quantities"]) * np.asarray(
                before["book"]["mark_prices"]
            )
            current *= np.asarray(before["book"]["contract_multipliers"])
            current /= before["book"]["equity"]
            complete = native | {
                "decision_index": index,
                "decision_time_ns": _time(env, index),
                "processing_index": index + 1,
                "processing_time_ns": _time(env, index + 1),
                "symbol_index": env.symbol_index,
                "proposal": proposal,
                "proposal_digest": expected.digest,
                "decision_digest": decision.decision_digest,
                "risk": row["risk"],
                "risk_config": _json(env.risk_config),
                "current_weights": current.tolist(),
            }
            validate_allocation_execution_facts_v2(
                complete,
                env.recipe,
                dataset_id=env.dataset.dataset_id,
                action_code=action,
                actual_cost_payload=env.execution_cost.execution_policy_payload(),
            )
        else:
            validate_terminal_allocation_facts(
                native,
                dataset_id=env.dataset.dataset_id,
                economics_digest=env.recipe["runtime_profile"]["economics_digest"],
                symbol_index=env.symbol_index,
                start_index=index,
                start_time_ns=_time(env, index),
                processing_index=index + 1,
                processing_time_ns=_time(env, index + 1),
            )
            weights = np.asarray(before["book"]["quantities"]) * np.asarray(
                before["book"]["mark_prices"]
            )
            weights *= np.asarray(before["book"]["contract_multipliers"])
            weights /= before["book"]["equity"]
            expected_risk = PreTradeRisk(env.risk_config).constrain(
                np.zeros(env.dataset.n_symbols),
                current=weights,
                drawdown=before["book"]["max_drawdown"],
            )
            _same(row["risk"], _json(expected_risk))
            current = weights
            decision_identity = content_digest(
                {
                    "schema": "allocation_settled_reserve_decision_v1",
                    "bound_digest": env.bound.digest,
                    "episode_id": identity["episode_id"],
                    "index": index,
                }
            )
        target = np.asarray(row["risk"]["weights"])
        target_identity = (
            None
            if np.array_equal(target, current)
            else content_digest(
                {"decision": decision_identity, "target": target.tolist()}
            )
        )
        _native_sources(env, row, before, target_identity=target_identity)
        before = {"book": native["book"], "active_orders": native["active_orders"]}
        _same(
            row["after_state_digest"],
            _state(env, identity["episode_id"], index + 1, before),
        )
        rows.append(row)
    book, total_actor = before["book"], identity["actor_step"] + 1
    terminated = (
        total_actor == env.bound.clock.actor_steps
        or book["termination_reason"] is not None
    )
    _same(facts["terminated"], terminated)
    reward = net_equity_increment(
        opening, book["equity"], initial_capital=env.initial_capital
    )
    _same(
        [facts["reward"], event["reward"], event["terminated"], event["truncated"]],
        [reward, reward, terminated, False],
    )
    witness = _closed(facts["settlement_witness"], _WITNESS_KEYS)
    full = (
        total_actor == env.bound.clock.actor_steps
        and len(rows) == env.bound.clock.physical_horizon_steps
        and book["as_of_index"] == env.settlement_stop_index
    )
    settled = (
        full
        and book["termination_reason"] is None
        and not before["active_orders"]
        and all(Fraction(q) == 0 for q in book["exact_quantities"])
    )
    _same(
        witness,
        {
            "actor_steps_processed": total_actor,
            "physical_steps_processed": len(rows),
            "reserve_steps_processed": sum(r["kind"] == "reserve" for r in rows),
            "full_coverage": full,
            "settled": settled,
            "eligible": settled
            and book["max_drawdown"] <= env.bound.objective.maximum_drawdown,
            "maximum_drawdown": book["max_drawdown"],
            "frontier_index": book["as_of_index"],
            "native_frontier": before,
        },
    )
    return before, rows, facts


def _parameters(value: Any) -> dict[str, Any]:
    if type(value) is not dict or not value or any(type(k) is not str for k in value):
        raise ValueError("actual parameter snapshot must be a nonempty named mapping")
    for tensor in value.values():
        shape = tensor.get("shape")
        if type(shape) is not list or any(type(i) is not int or i < 0 for i in shape):
            raise ValueError("actual parameter shape changed")
        _tensor(tensor, "<f4", tuple(shape))
    return value


def _gae(
    buffer: dict[str, np.ndarray],
    last: np.ndarray,
    dones: np.ndarray,
    gamma: float,
    lam: float,
) -> None:
    # Exact pinned SB3 2.3.2 NumPy order. Do not cast bool final masks to float32.
    advantages = np.zeros_like(buffer["values"])
    last_gae_lam: Any = 0
    for step in reversed(range(len(advantages))):
        if step == len(advantages) - 1:
            next_non_terminal = 1.0 - dones
            next_values = last.flatten()
        else:
            next_non_terminal = 1.0 - buffer["episode_starts"][step + 1]
            next_values = buffer["values"][step + 1]
        delta = (
            buffer["rewards"][step]
            + gamma * next_values * next_non_terminal
            - buffer["values"][step]
        )
        last_gae_lam = delta + gamma * lam * next_non_terminal * last_gae_lam
        advantages[step] = last_gae_lam
    _same(_record(buffer["advantages"]), _record(advantages))
    _same(_record(buffer["returns"]), _record(advantages + buffer["values"]))


def _inputs(
    env: Any, value: Any, expected: list[dict[str, Any]], resets: list[dict[str, Any]]
) -> None:
    value = _closed(
        value,
        set(
            "schema observation_schema_digest width dtype events role_counts role_digests prepared_reset_without_actor prepared_reset_without_actor_count".split()
        ),
    )
    _same(
        {
            k: value[k]
            for k in ("schema", "observation_schema_digest", "width", "dtype", "events")
        },
        {
            "schema": "allocation_settled_input_observations_v1",
            "observation_schema_digest": content_digest(env.recipe["observation"]),
            "width": len(env.observation_schema.fields),
            "dtype": "little_endian_float32",
            "events": expected,
        },
    )
    counts, digests = {}, {}
    for role in _ROLES:
        digest, count = sha256(), 0
        for event in expected:
            if event["role"] == role:
                tensor = event["tensor"]
                header = canonical_json_bytes(
                    {k: v for k, v in event.items() if k != "tensor"}
                    | {"tensor": {k: v for k, v in tensor.items() if k != "hex"}}
                )
                data = bytes.fromhex(tensor["hex"])
                digest.update(
                    len(header).to_bytes(8, "big")
                    + header
                    + len(data).to_bytes(8, "big")
                    + data
                )
                count += 1
        counts[role], digests[role] = count, digest.hexdigest()
    actor_episodes = {
        v["identity"]["episode_id"] for v in expected if v["role"] == "actor"
    }
    unused = [
        {
            "identity": v["identity"],
            "prepared": v["prepared"],
            "tensor": v["observation"],
        }
        for v in resets
        if v["prepared"] and v["identity"]["episode_id"] not in actor_episodes
    ]
    _same(value["role_counts"], counts)
    _same(value["role_digests"], digests)
    _same(value["prepared_reset_without_actor"], unused)
    _same(value["prepared_reset_without_actor_count"], len(unused))


def validate_settled_transition_capture(
    env: SettledAllocationTradingEnv,
    capture: object,
    *,
    requested_timesteps: int,
    optimization: object,
    training_protocol: AllocationPPOTrainingProtocol,
) -> dict[str, Any]:
    """Independently close captured relations; no complete boolean is consulted."""
    try:
        return _validate(
            env,
            capture,
            requested_timesteps=requested_timesteps,
            optimization=optimization,
            training_protocol=training_protocol,
        )
    except (KeyError, TypeError, IndexError, OverflowError, RecursionError) as error:
        raise ValueError("invalid closed observed settled training capture") from error


def _validate(
    env: Any,
    capture: Any,
    *,
    requested_timesteps: int,
    optimization: Any,
    training_protocol: AllocationPPOTrainingProtocol,
) -> dict[str, Any]:
    _native(capture)
    _native(optimization)
    capture = _closed(
        capture,
        set(
            "schema declared events inputs successful_append_count returned_rollout_count returned_update_count failure_count".split()
        ),
    )
    _same(capture["schema"], "allocation_settled_observed_capture_v1")
    env.validate_binding()
    source_envelope = settled_training_source_receipt(env)
    _same(
        capture["declared"],
        {
            "recipe": env.recipe,
            "recipe_digest": env.recipe_digest,
            "objective": env.bound.objective.payload(),
            "settled_clock": env.bound.clock.payload(),
            "source_envelope": source_envelope,
        },
    )
    budget, n, width = (
        _integer(requested_timesteps, 1),
        env.bound.clock.rollout_steps,
        len(env.observation_schema.fields),
    )
    if budget % n:
        raise ValueError("successful append budget must be an exact rollout multiple")
    if not isinstance(training_protocol, AllocationPPOTrainingProtocol) or (
        training_protocol.n_steps,
        training_protocol.gamma,
        training_protocol.gae_lambda,
    ) != (n, env.bound.clock.gamma, env.bound.clock.gae_lambda):
        raise ValueError("settled observed update protocol differs from actor clock")
    events = capture["events"]
    if type(events) is not list or not events:
        raise ValueError("actual fit needs explicit chronological events")
    for ordinal, event in enumerate(events):
        kind = event.get("kind") if type(event) is dict else None
        if kind not in _EVENT_KEYS:
            raise ValueError(
                "failed or unknown capture cannot produce a successful bundle"
            )
        _closed(event, _EVENT_KEYS[kind] | {"kind", "ordinal"})
        if _integer(event["ordinal"]) != ordinal:
            raise ValueError("native event ordinal is missing/reordered")
    by_ordinal = {event["ordinal"]: event for event in events}
    episode = step = reset_number = append_count = rollout_number = update_number = 0
    current_account: dict[str, Any] | None = None
    current_identity: dict[str, Any] | None = None
    current_observation: np.ndarray | None = None
    pending_forward: dict[str, Any] | None = None
    pending_native: dict[str, Any] | None = None
    pending_stage: dict[str, Any] | None = None
    pending_critic: dict[str, Any] | None = None
    params_digest = params_after = None
    collecting = ended = False
    awaiting_reset = False
    native_rows: list[dict[str, Any]] = []
    episode_indices: list[int] = []
    episodes: list[dict[str, Any]] = []
    resets: list[dict[str, Any]] = []
    input_events: list[dict[str, Any]] = []
    source_selections: list[dict[str, Any]] = []
    append_events: list[dict[str, Any]] = []
    buffers: list[dict[str, Any]] = []
    optimizer_frames, epochs_total, successful_optimizer_calls = sha256(), 0, 0
    started = False

    def add_input(role: str, identity: dict[str, Any], tensor: dict[str, Any]) -> int:
        ordinal = len(input_events)
        input_events.append(
            {
                "kind": "input",
                "ordinal": ordinal,
                "role": role,
                "identity": identity,
                "tensor": tensor,
                "model_consumed": role != "terminal_sentinel",
            }
        )
        return ordinal

    def project_episode() -> dict[str, Any]:
        assert current_account is not None
        book = current_account["book"]
        return {
            "schema": "allocation_settled_episode_witness_v1",
            "episode_index": len(episodes),
            "actor_transition_indices": list(episode_indices),
            "processing_rows": [
                {
                    "kind": row["kind"],
                    "start_index": row["start_index"],
                    "processing_index": row["processing_index"],
                    "bars_advanced": row["native"]["execution"]["bars_advanced"],
                    "returned_clock_types": row["returned_clock_types"],
                }
                for row in native_rows
            ],
            "final_book": {key: book[key] for key in sorted(_LOWER_BOOK_KEYS)},
            "final_active_orders": [
                {
                    "order_id": order["intent"]["order_id"],
                    "status": order["status"],
                    "requested_quantity": order["intent"]["requested_quantity"],
                    "remaining_quantity": order["remaining_quantity"],
                    "exact_cumulative_filled_quantity": order[
                        "exact_cumulative_filled_quantity"
                    ],
                }
                for order in current_account["active_orders"]
            ],
            "failure": None,
        }

    for event in events:
        kind = event["kind"]
        if ended:
            raise ValueError("actual core capture continued after training_end")
        if kind == "reset":
            if (
                pending_stage is not None
                or (episode and not awaiting_reset)
                or (episode == 0 and started)
            ):
                raise ValueError("native reset preceded true terminal or initial setup")
            episode += 1
            reset_number = len(resets)
            step, native_rows, episode_indices = 0, [], []
            _same(
                [event["reset_ordinal"], event["prepared"]], [reset_number, episode > 1]
            )
            _identity(
                env,
                event["identity"],
                episode=episode,
                step=0,
                index=env.start_index,
                reset=reset_number,
            )
            _reset_account(env, event["native_account"])
            current_account, current_identity = (
                event["native_account"],
                event["identity"],
            )
            current_observation = _tensor(event["observation"], "<f4", (width,))
            expected_observation = _observation_witness(
                env, env.start_index, current_account, event["observation_witness"]
            )
            _same(_record(current_observation), _record(expected_observation))
            source_selections.append(
                _observation_selection(
                    env, episode, env.start_index, event["observation_witness"]
                )
            )
            resets.append(event)
            awaiting_reset = False
        elif kind == "training_start":
            if started or episode != 1:
                raise ValueError("actual initial setup/start ordering changed")
            _same(event["actual_timesteps"], 0)
            started = True
        elif kind == "rollout_start":
            if (
                not started
                or collecting
                or rollout_number != update_number
                or pending_stage is not None
            ):
                raise ValueError("actual rollout started before prior update closure")
            _same(
                [event["rollout"], event["actual_timesteps"]],
                [rollout_number, append_count],
            )
            collecting, pending_critic = True, None
        elif kind == "actor_forward":
            if (
                not collecting
                or pending_forward is not None
                or pending_stage is not None
                or pending_critic is not None
                or awaiting_reset
                or current_identity is None
                or current_observation is None
            ):
                raise ValueError(
                    "actor forward is missing, duplicated, or outside collection"
                )
            _same(event["identity"], current_identity)
            actor = _closed(
                event["actor"], {"observation", "action", "value", "log_prob"}
            )
            _same(
                _tensor(actor["observation"], "<f4", (1, width)).tobytes().hex(),
                current_observation.tobytes().hex(),
            )
            action = int(_tensor(actor["action"], "<i8", (1,))[0])
            if (
                not 0 <= action <= 3
                or float(_tensor(actor["log_prob"], "<f4", (1,))[0]) > 0
            ):
                raise ValueError(
                    "actual categorical action/log probability is outside contract"
                )
            _tensor(actor["value"], "<f4", (1, 1))
            _digest(event["parameters_digest"])
            if params_digest is not None:
                _same(event["parameters_digest"], params_digest)
            else:
                params_digest = event["parameters_digest"]
            _same(
                event["input_ordinal"],
                add_input("actor", event["identity"], actor["observation"]),
            )
            pending_forward = event
        elif kind == "env_transition":
            if (
                pending_forward is None
                or pending_native is not None
                or current_account is None
            ):
                raise ValueError("native step lacks exactly one actual actor")
            _same(event["identity"], pending_forward["identity"])
            _same(
                _tensor(event["supplied_action"], "<i8", ()).tobytes().hex(),
                pending_forward["actor"]["action"]["hex"],
            )
            current_account, native_rows, facts = _native_transition(
                env, event, current_account, native_rows
            )
            step += 1
            _identity(
                env,
                event["identity"],
                episode=episode,
                step=step - 1,
                index=env.start_index + step - 1,
                reset=reset_number,
            )
            for row in facts["rows"]:
                source_selections.append(_source_selection(env, episode, row))
            current_observation = _tensor(
                event["returned_observation"], "<f4", (width,)
            )
            if event["terminated"]:
                _same(event["observation_witness"], None)
                if np.any(current_observation):
                    raise ValueError("true terminal sentinel is nonzero")
                sentinel = event["identity"] | {
                    "physical_frontier_index": current_account["book"]["as_of_index"],
                    "actor_steps_processed": step,
                }
                add_input("terminal_sentinel", sentinel, event["returned_observation"])
                current_identity, awaiting_reset = None, True
            else:
                current_identity = dict(event["identity"]) | {
                    "actor_step": step,
                    "actor_index": env.start_index + step,
                    "time_ns": _time(env, env.start_index + step),
                }
                expected_observation = _observation_witness(
                    env,
                    env.start_index + step,
                    current_account,
                    event["observation_witness"],
                )
                _same(_record(current_observation), _record(expected_observation))
                source_selections.append(
                    _observation_selection(
                        env,
                        episode,
                        env.start_index + step,
                        event["observation_witness"],
                    )
                )
            pending_native = event
            # Final native frontier is saved before DummyVecEnv can reset it.
            pending_native = pending_native | {"episode_projection": project_episode()}
        elif kind == "stage":
            if (
                pending_forward is None
                or pending_native is None
                or pending_stage is not None
            ):
                raise ValueError("callback stage lacks actual forward/native return")
            if current_observation is None:
                raise ValueError("callback stage lacks an actual successor observation")
            _same(
                [event["sequence"], event["rollout"], event["local"]],
                [append_count, rollout_number, append_count - rollout_number * n],
            )
            _same(
                [event["actor_forward_ordinal"], event["env_transition_ordinal"]],
                [pending_forward["ordinal"], pending_native["ordinal"]],
            )
            _same(event["identity"], pending_forward["identity"])
            _same(
                {
                    k: event["actor"][k]
                    for k in ("observation", "action", "value", "log_prob")
                },
                pending_forward["actor"],
            )
            _same(
                event["actor"]["reward"],
                _record(np.asarray([pending_native["reward"]], dtype="<f4")),
            )
            _same(
                [event["done"], event["episode_start"]],
                [pending_native["terminated"], event["identity"]["actor_step"] == 0],
            )
            _same(event["next_identity"], current_identity)
            _same(
                event["next_observation"],
                _record(current_observation.reshape(1, width)),
            )
            _same(
                event["next_role"],
                "prepared_reset" if event["done"] else "live_successor",
            )
            _same(
                event["terminal"],
                pending_native["returned_observation"] if event["done"] else None,
            )
            if event["done"] and awaiting_reset:
                raise ValueError("native true terminal lacks actual prepared reset")
            _closed(
                event["actor"], {"observation", "action", "value", "log_prob", "reward"}
            )
            pending_stage = event
        elif kind == "append":
            if pending_stage is None or pending_native is None or not collecting:
                raise ValueError("successful append lacks a staged actual row")
            local = append_count - rollout_number * n
            _same(
                [
                    event[k]
                    for k in (
                        "sequence",
                        "rollout",
                        "local",
                        "pos_before",
                        "pos_after",
                        "stage_ordinal",
                        "actual_timesteps",
                    )
                ],
                [
                    append_count,
                    rollout_number,
                    local,
                    local,
                    local + 1,
                    pending_stage["ordinal"],
                    append_count + 1,
                ],
            )
            arguments, stored = (
                _closed(event["arguments"], set(_FIELDS[:6])),
                _closed(event["stored_row"], set(_FIELDS[:6])),
            )
            actor = pending_stage["actor"]
            for field, key, dtype, shape in (
                ("observations", "observation", "<f4", (1, width)),
                ("actions", "action", "<i8", (1, 1)),
                ("rewards", "reward", "<f4", (1,)),
                ("values", "value", "<f4", (1, 1)),
                ("log_probs", "log_prob", "<f4", (1,)),
            ):
                actual = _tensor(arguments[field], dtype, shape)
                _same(actual.tobytes().hex(), actor[key]["hex"])
                expected = actual.astype("<f4").reshape(
                    (1, width)
                    if field == "observations"
                    else (1, 1)
                    if field == "actions"
                    else (1,)
                )
                _same(stored[field], _record(expected))
            masks = _tensor(arguments["episode_starts"], "|b1", (1,))
            _same(bool(masks[0]), pending_stage["episode_start"])
            _same(stored["episode_starts"], _record(masks.astype("<f4")))
            append_events.append(event)
            append_count += 1
            actor_local = pending_stage["identity"]["actor_step"]
            if pending_stage["done"]:
                projection = pending_native["episode_projection"]
                projection["actor_transition_indices"] = list(range(actor_local + 1))
                episodes.append(projection)
            else:
                if actor_local != len(episode_indices):
                    raise ValueError("actual episode append prefix is incomplete")
                episode_indices.append(actor_local)
            pending_forward = pending_native = pending_stage = None
        elif kind == "critic":
            if (
                not collecting
                or pending_stage is not None
                or pending_forward is not None
                or pending_critic is not None
                or append_count != (rollout_number + 1) * n
            ):
                raise ValueError(
                    "actual critic endpoint is missing/reordered/duplicated"
                )
            if current_observation is None:
                raise ValueError("actual critic endpoint lacks its input observation")
            _same(event["identity"], current_identity)
            last_stage = by_ordinal[append_events[-1]["stage_ordinal"]]
            role = (
                "prepared_reset_boundary_critic"
                if last_stage["done"]
                else "live_boundary_critic"
            )
            _same(event["role"], role)
            _same(event["input"], _record(current_observation.reshape(1, width)))
            _tensor(event["result"], "<f4", (1, 1))
            _same(event["parameters_digest"], params_digest)
            _same(
                event["input_ordinal"],
                add_input(role, event["identity"], event["input"]),
            )
            pending_critic = event
        elif kind == "rollout":
            if not collecting or pending_critic is None or pending_stage is not None:
                raise ValueError(
                    "native compute lacks full append/actual critic closure"
                )
            _same(
                [
                    event[k]
                    for k in (
                        "rollout",
                        "first",
                        "last",
                        "critic_ordinal",
                        "actual_timesteps",
                    )
                ],
                [
                    rollout_number,
                    rollout_number * n,
                    append_count - 1,
                    pending_critic["ordinal"],
                    append_count,
                ],
            )
            _same(event["same_critic_result_object"], True)
            _same(event["last_values"], pending_critic["result"])
            dones = _tensor(event["dones"], "|b1", (1,))
            _same(
                bool(dones[0]), by_ordinal[append_events[-1]["stage_ordinal"]]["done"]
            )
            native_buffer = _closed(event["chronological_buffer"], set(_FIELDS))
            arrays = {}
            for field in _FIELDS:
                buffer_shape = (
                    (n, 1, width)
                    if field == "observations"
                    else (n, 1, 1)
                    if field == "actions"
                    else (n, 1)
                )
                arrays[field] = _tensor(native_buffer[field], "<f4", buffer_shape)
            for field in _FIELDS[:6]:
                expected = np.stack(
                    [
                        _tensor(
                            a["stored_row"][field],
                            "<f4",
                            (1, width)
                            if field == "observations"
                            else (1, 1)
                            if field == "actions"
                            else (1,),
                        )
                        for a in append_events[rollout_number * n : append_count]
                    ],
                    axis=0,
                )
                _same(native_buffer[field], _record(expected))
            _gae(
                arrays,
                _tensor(event["last_values"], "<f4", (1, 1)),
                dones,
                env.bound.clock.gamma,
                env.bound.clock.gae_lambda,
            )
            _same(
                content_digest(_parameters(event["parameters_before"])), params_digest
            )
            if params_after is not None:
                _same(event["parameters_before"], params_after)
            buffers.append(event)
            rollout_number += 1
        elif kind == "rollout_end":
            if not collecting or len(buffers) != rollout_number:
                raise ValueError("native rollout_end preceded returned compute")
            _same(
                [event["rollout"], event["actual_timesteps"]],
                [rollout_number - 1, append_count],
            )
            collecting = False
        elif kind == "update":
            if collecting or update_number + 1 != rollout_number:
                raise ValueError("actual update preceded returned buffer or duplicated")
            _same(
                [event["rollout"], event["actual_timesteps"]],
                [update_number, append_count],
            )
            calls, epochs = (
                _integer(event["optimizer_calls"], 0),
                _integer(event["epochs"], 1),
            )
            batches = n // training_protocol.batch_size
            if (
                epochs > training_protocol.n_epochs
                or not batches * (epochs - 1) <= calls <= batches * epochs
                or training_protocol.target_kl is None
                and (epochs != training_protocol.n_epochs or calls != epochs * batches)
                or epochs < training_protocol.n_epochs
                and calls == epochs * batches
            ):
                raise ValueError(
                    "observed per-update calls/epochs differ from fixed protocol"
                )
            epochs_total += epochs
            _same(event["n_updates"], epochs_total)
            for _ in range(calls):
                successful_optimizer_calls += 1
                frame = canonical_json_bytes(
                    {
                        "step": successful_optimizer_calls,
                        "rollout": update_number + 1,
                        "timesteps": append_count,
                    }
                )
                optimizer_frames.update(len(frame).to_bytes(8, "big") + frame)
            diagnostics = event["diagnostics"]
            if type(diagnostics) is not dict or any(
                type(k) is not str or not k.startswith("train/") for k in diagnostics
            ):
                raise ValueError("actual optimizer diagnostic schema changed")
            for name, value in diagnostics.items():
                if type(value) is dict:
                    _same(value, {"value": None, "reason": "zero_target_variance"})
                    returns = _tensor(
                        buffers[-1]["chronological_buffer"]["returns"], "<f4", (n, 1)
                    )
                    if name != "train/explained_variance" or np.var(returns) != 0:
                        raise ValueError(
                            "nullable native diagnostic lacks actual zero target variance"
                        )
                elif type(value) not in (int, float) or not math.isfinite(value):
                    raise ValueError(
                        "actual optimizer diagnostic is not a finite native scalar"
                    )
            params_after = _parameters(event["parameters_after"])
            if set(params_after) != set(buffers[-1]["parameters_before"]):
                raise ValueError("actual optimizer changed model parameter names")
            for name in params_after:
                _same(
                    {k: v for k, v in params_after[name].items() if k != "hex"},
                    {
                        k: v
                        for k, v in buffers[-1]["parameters_before"][name].items()
                        if k != "hex"
                    },
                )
            params_digest = content_digest(params_after)
            update_number += 1
        elif kind == "training_end":
            if (
                collecting
                or update_number != rollout_number
                or append_count != budget
                or not started
            ):
                raise ValueError(
                    "native final train/update/append closure is incomplete"
                )
            _same(event["actual_timesteps"], budget)
            if (
                pending_forward is not None
                or pending_stage is not None
                or pending_native is not None
            ):
                raise ValueError("training_end has an unadmitted returned row")
            ended = True
    if (
        not ended
        or append_count != budget
        or rollout_number != budget // n
        or update_number != rollout_number
    ):
        raise ValueError("actual fit lacks exact completed collection/update budget")
    if episode_indices:
        episodes.append(project_episode())
    if not episodes:
        raise ValueError("actual fit captured no actor-bearing episode")
    _same(
        {
            key: capture[key]
            for key in (
                "successful_append_count",
                "returned_rollout_count",
                "returned_update_count",
                "failure_count",
            )
        },
        {
            "successful_append_count": append_count,
            "returned_rollout_count": rollout_number,
            "returned_update_count": update_number,
            "failure_count": 0,
        },
    )
    _same(
        optimization,
        {
            "schema": "allocation_ppo_optimization_receipt_v1",
            "rollout_update_count": update_number,
            "successful_optimizer_step_calls": successful_optimizer_calls,
            "epoch_iterations": epochs_total,
            "optimizer_step_events_digest": optimizer_frames.hexdigest(),
            "final_capture": "after_final_train_v1",
        },
    )
    _inputs(env, capture["inputs"], input_events, resets)
    source = {
        "dataset_id": env.dataset.dataset_id,
        "symbol": env.dataset.symbols[env.symbol_index],
        "symbol_count": env.dataset.n_symbols,
        "symbol_index": env.symbol_index,
        "start_index": env.start_index,
        "stop_index": env.settlement_stop_index,
        "start_time_ns": _time(env, env.start_index),
        "terminal_time_ns": _time(env, env.settlement_stop_index),
    }
    return {
        "source": source,
        "episodes": episodes,
        "failure": None,
        "bindings": {
            "schema": "allocation_settled_upper_proof_bindings_v1",
            "source_consumption_digest": content_digest(
                {
                    "declared_envelope": source_envelope,
                    "actual_native_selections": source_selections,
                }
            ),
            "observation_consumption_digest": content_digest(capture["inputs"]),
            "transition_trace_digest": content_digest(
                {"declared": capture["declared"], "events": events}
            ),
            "buffer_digest": content_digest(
                [
                    {
                        "rollout": b["rollout"],
                        "chronological_buffer": b["chronological_buffer"],
                    }
                    for b in buffers
                ]
            ),
            "append_ledger_digest": content_digest(append_events),
        },
    }


__all__ = ["validate_settled_transition_capture"]
