from __future__ import annotations

import json
from dataclasses import asdict, fields, replace
from hashlib import sha256

import numpy as np
import pytest

from tools.market_b4_oracle import decision_identity
from tools.market_b4_oracle.decision_identity import (
    OwnedAccountComponents,
    OwnedIntentComponents,
    OwnedPendingOrder,
    build_actor_identity,
    canonical_json_bytes,
    owned_order_mapping,
)
from trade_rl.artifacts.hashing import content_digest
from trade_rl.evaluation.allocation_decision import AllocationDecision
from trade_rl.simulation import BookState
from trade_rl.simulation.orders.model import (
    OrderIntent,
    OrderStatus,
    OrderType,
    PendingOrder,
    TimeInForce,
)
from trade_rl.strategies.allocation import (
    AfterCostTargetAllocator,
    AllocationContext,
    AllocationInputs,
)
from trade_rl.strategies.allocation_action import (
    AllocationActionContract,
)


def _book() -> BookState:
    return BookState(
        quantities=np.asarray([1.5, -0.25], dtype=np.float64),
        cash=1000.0,
        mark_prices=np.asarray([10.0, 20.0], dtype=np.float64),
        peak_value=1100.0,
        contract_multipliers=np.asarray([2.0, 1.5], dtype=np.float64),
        max_drawdown=0.12,
        turnover_total=2.5,
        total_cost=0.25,
        funding_pnl=-0.125,
        fill_count=3,
        rebalance_events=2,
        returns_history=[-0.0, 0.01],
        borrow_cost=0.03,
        margin_used=44.0,
        maintenance_margin=0.01,
        maintenance_requirement=0.44,
        margin_deficit=0.0,
        insolvent=False,
        as_of_index=4,
        as_of_dataset_id="d" * 64,
        _exact_quantities=("3/2", "-1/4"),
    )


def _account(
    *,
    book: BookState | None = None,
    active_orders: tuple[OwnedPendingOrder, ...] = (),
    terminal_orders: tuple[OwnedPendingOrder, ...] = (),
) -> OwnedAccountComponents:
    source = _book() if book is None else book
    facts = {}
    for field in fields(source):
        if field.name == "_exact_quantities":
            continue
        value = getattr(source, field.name)
        facts[field.name] = value.tolist() if isinstance(value, np.ndarray) else value
    facts["exact_quantities"] = tuple(str(value) for value in source.exact_quantities)
    return OwnedAccountComponents(
        account_id="synthetic-two-symbol-account",
        dataset_id="d" * 64,
        symbols=("BTC-PERP", "ETH-PERP"),
        execution_policy_digest="e" * 64,
        risk_config={
            "drawdown_start": 0.10,
            "drawdown_stop": 0.20,
            "max_abs_weight": 1.0,
        },
        symbol_index=0,
        decision_index=4,
        decision_time=str(np.datetime64("2026-01-01T04:00:00", "ns")),
        book=facts,
        active_orders=active_orders,
        terminal_orders=terminal_orders,
    )


def _inputs() -> dict[str, object]:
    time = "2026-01-01T04:00:00.000000000"
    return {
        "symbol": "BTC-PERP",
        "decision_time": time,
        "available_at": time,
        "horizon_end": "2026-01-02T04:00:00.000000000",
        "source_identity": "synthetic-independent-source-components",
        "expected_simple_return": 0.08,
        "return_variance": 0.01,
        "buy_cost": 0.001,
        "sell_cost": 0.002,
        "exit_cost": 0.0005,
        "funding_return": -0.0002,
        "borrow_return": 0.0001,
        "cash_return": 0.0003,
        "return_unit": "expected_simple_return",
    }


def _allocator() -> dict[str, object]:
    return {
        "lower_weight": -0.5,
        "upper_weight": 0.8,
        "max_turnover": 0.3,
        "risk_aversion": 1.5,
    }


def _decision_fields() -> dict[str, object]:
    return {
        "feature_names": ("trend", "volatility"),
        "feature_values": (-0.0, 0.25),
        "max_drawdown": 0.12,
        "initial_capital": 1000.0,
        "remaining_steps": 7,
        "pending_gross": 0.125,
        "pending_count": 0,
    }


def _build(
    account: OwnedAccountComponents,
    *,
    action_code=2,
    risk_weights=(0.2, -0.0),
    target_weight=0.2,
    objective_value=0.01,
):
    preliminary = build_actor_identity(
        account,
        _inputs(),
        _allocator(),
        independent_target_weight=target_weight,
        independent_objective_value=objective_value,
        decision_fields=_decision_fields(),
        action_contract={"mode": "residual", "scale": 0.5},
        action_code=action_code,
        final_risk_weights=risk_weights,
    )
    # The independent target/objective above is the explicit synthetic owner
    # input; the native optimizer is invoked only to check payload parity.
    return preliminary


def _native_proposal(account: OwnedAccountComponents):
    seed = _build(account)
    context_data = json.loads(seed.context_bytes)
    context_data["decision_time"] = np.datetime64(context_data["decision_time"], "ns")
    context = AllocationContext(**context_data)
    inputs_data = _inputs()
    for name in ("decision_time", "available_at", "horizon_end"):
        inputs_data[name] = np.datetime64(inputs_data[name], "ns")
    inputs = AllocationInputs(**inputs_data)
    allocator = AfterCostTargetAllocator(**_allocator())
    proposal = allocator.propose(inputs, context)
    return seed, context, inputs, allocator, proposal


def test_two_symbol_context_proposal_decision_action_and_target_hashes_match_public_apis():
    account = _account()
    _, context, _, _, proposal = _native_proposal(account)
    built = _build(
        account,
        action_code=2,
        risk_weights=(0.2, -0.0),
        target_weight=proposal.target_weight,
        objective_value=proposal.objective_value,
    )
    assert built.proposal_digest == proposal.decision_digest
    assert built.context_state_digest == content_digest(
        json.loads(built.context_state_bytes)
    )
    assert built.context_state_digest == context.state_digest
    assert built.current_weights == pytest.approx(
        (float(_book().weights[0]), float(_book().weights[1]))
    )

    decision = AllocationDecision(
        baseline=proposal,
        action_contract=AllocationActionContract(mode="residual", scale=0.5),
        **_decision_fields(),
    )
    action = decision.propose(2)
    built_action = _build(
        account,
        action_code=2,
        risk_weights=(0.2, -0.0),
        target_weight=proposal.target_weight,
        objective_value=proposal.objective_value,
    )
    assert built_action.decision_digest == decision.decision_digest
    assert built_action.action_digest == action.digest
    assert built_action.actor_decision_identity == proposal.decision_digest
    assert built_action.target_identity == content_digest(
        {"decision": proposal.decision_digest, "target": [0.2, -0.0]}
    )
    assert json.loads(built_action.decision_bytes)["feature_values"] == [-0.0, 0.25]


def test_nonbaseline_action_uses_action_digest_before_full_risk_vector_wrapper():
    account = _account()
    _, context, _, _, baseline = _native_proposal(account)
    built = _build(
        account,
        action_code=1,
        risk_weights=(-0.4, -0.0),
        target_weight=baseline.target_weight,
        objective_value=baseline.objective_value,
    )
    decision = AllocationDecision(
        baseline=baseline,
        action_contract=AllocationActionContract(mode="residual", scale=0.5),
        **_decision_fields(),
    )
    action = decision.propose(1)
    assert action.target_weight != baseline.target_weight
    assert built.actor_decision_identity == action.digest
    assert built.target_identity == content_digest(
        {"decision": action.digest, "target": [-0.4, -0.0]}
    )


def test_native_order_id_and_pending_state_are_rebuilt_from_prior_owned_identity():
    first_account = _account()
    previous = _build(first_account, action_code=3, risk_weights=(0.4, -0.0))
    order_fields = {
        "dataset_id": first_account.dataset_id,
        "execution_policy_digest": first_account.execution_policy_digest,
        "symbol_index": 0,
        "requested_quantity": 0.75,
        "order_type": "market",
        "time_in_force": "gtc",
        "limit_price": None,
        "stop_price": None,
        "submit_index": 4,
        "eligible_index": 5,
        "expiry_index": None,
        "submission_reference_price": 10.0,
        "decision_equity": 1022.5,
        "replaced_order": None,
        "reduce_only": False,
    }
    built_intent = OwnedIntentComponents(
        identity_source=previous,
        **order_fields,
    )
    built_order = OwnedPendingOrder(
        intent=built_intent,
        remaining_quantity=0.75,
        exact_cumulative_filled_quantity="0",
    )
    native_intent = OrderIntent.create(
        dataset_id=first_account.dataset_id,
        target_identity=previous.target_identity,
        execution_policy_digest=first_account.execution_policy_digest,
        symbol_index=0,
        requested_quantity=0.75,
        order_type=OrderType.MARKET,
        time_in_force=TimeInForce.GTC,
        limit_price=None,
        stop_price=None,
        submit_index=4,
        eligible_index=5,
        expiry_index=None,
        submission_reference_price=10.0,
        decision_equity=1022.5,
    )
    native_order = PendingOrder.from_intent(native_intent)
    owned_payload = owned_order_mapping(built_order, first_account)
    assert canonical_json_bytes(owned_payload) == canonical_json_bytes(
        asdict(native_order)
    )
    assert owned_payload["intent"]["order_id"] == native_intent.order_id
    assert "target_identity" not in order_fields
    assert "order_id" not in order_fields


def test_full_pending_order_state_changes_context_digest_and_requires_builder_identity():
    account = _account()
    previous = _build(account, action_code=3, risk_weights=(0.4, -0.0))
    intent = OwnedIntentComponents(
        identity_source=previous,
        dataset_id=account.dataset_id,
        execution_policy_digest=account.execution_policy_digest,
        symbol_index=0,
        requested_quantity=0.5,
        order_type="market",
        time_in_force="gtc",
        limit_price=None,
        stop_price=None,
        submit_index=4,
        eligible_index=5,
        expiry_index=None,
        submission_reference_price=10.0,
        decision_equity=1022.5,
    )
    pending = OwnedPendingOrder(
        intent=intent,
        remaining_quantity=0.5,
        cumulative_filled_quantity=0.0,
        cumulative_filled_notional=0.0,
        status=OrderStatus.SUBMITTED,
        exact_cumulative_filled_quantity="0",
    )
    with_pending = _account(active_orders=(pending,))
    empty_identity = _build(account)
    pending_identity = _build(with_pending)
    assert empty_identity.context_state_digest != pending_identity.context_state_digest
    assert (
        json.loads(pending_identity.context_state_bytes)["active_orders"][0]["intent"][
            "target_identity"
        ]
        == previous.target_identity
    )
    assert pending_identity.context_state_digest == content_digest(
        json.loads(pending_identity.context_state_bytes)
    )

    forged = OwnedIntentComponents(
        identity_source="f" * 64,  # type: ignore[arg-type]
        dataset_id=account.dataset_id,
        execution_policy_digest=account.execution_policy_digest,
        symbol_index=0,
        requested_quantity=0.5,
        order_type="market",
        time_in_force="gtc",
        limit_price=None,
        stop_price=None,
        submit_index=4,
        eligible_index=5,
        expiry_index=None,
        submission_reference_price=10.0,
        decision_equity=1022.5,
    )
    with pytest.raises(ValueError, match="builder result"):
        owned_order_mapping(OwnedPendingOrder(forged, 0.5), account)


def test_terminal_history_is_rebuilt_with_prior_owned_identity():
    account = _account()
    previous = _build(account, action_code=1, risk_weights=(-0.4, -0.0))
    owned_intent = OwnedIntentComponents(
        identity_source=previous,
        dataset_id=account.dataset_id,
        execution_policy_digest=account.execution_policy_digest,
        symbol_index=0,
        requested_quantity=0.5,
        order_type="market",
        time_in_force="gtc",
        limit_price=None,
        stop_price=None,
        submit_index=4,
        eligible_index=5,
        expiry_index=None,
        submission_reference_price=10.0,
        decision_equity=1022.5,
    )
    native_intent = OrderIntent.create(
        dataset_id=account.dataset_id,
        target_identity=previous.target_identity,
        execution_policy_digest=account.execution_policy_digest,
        symbol_index=0,
        requested_quantity=0.5,
        order_type=OrderType.MARKET,
        time_in_force=TimeInForce.GTC,
        limit_price=None,
        stop_price=None,
        submit_index=4,
        eligible_index=5,
        expiry_index=None,
        submission_reference_price=10.0,
        decision_equity=1022.5,
    )
    native_terminal = PendingOrder.from_intent(native_intent).cancel(
        processing_index=5, reason="synthetic-cancel"
    )
    owned_terminal = OwnedPendingOrder(
        intent=owned_intent,
        remaining_quantity=0.5,
        status=OrderStatus.CANCELLED,
        last_processed_index=5,
        terminal_reason="synthetic-cancel",
        evidence_version=1,
        exact_cumulative_filled_quantity="0",
    )
    assert canonical_json_bytes(owned_order_mapping(owned_terminal, account)) == (
        canonical_json_bytes(asdict(native_terminal))
    )
    built = _build(_account(terminal_orders=(owned_terminal,)))
    state_payload = json.loads(built.context_state_bytes)
    terminal_rows = state_payload["terminal_orders"]
    assert len(terminal_rows) == 1
    assert terminal_rows[0]["intent"]["order_id"] == native_intent.order_id
    assert built.context_state_digest == content_digest(state_payload)


@pytest.mark.parametrize("scope_field", ["dataset_id", "execution_policy_digest"])
def test_terminal_mapping_cache_does_not_skip_account_scope_checks(
    scope_field: str,
) -> None:
    account = _account()
    previous = _build(account, action_code=1, risk_weights=(-0.4, -0.0))
    order = OwnedPendingOrder(
        intent=OwnedIntentComponents(
            identity_source=previous,
            dataset_id=account.dataset_id,
            execution_policy_digest=account.execution_policy_digest,
            symbol_index=0,
            requested_quantity=0.5,
            order_type="market",
            time_in_force="gtc",
            limit_price=None,
            stop_price=None,
            submit_index=4,
            eligible_index=5,
            expiry_index=None,
            submission_reference_price=10.0,
            decision_equity=1022.5,
        ),
        remaining_quantity=0.5,
        status=OrderStatus.CANCELLED,
        last_processed_index=5,
        terminal_reason="synthetic-cancel",
        evidence_version=1,
        exact_cumulative_filled_quantity="0",
    )
    mismatched_account = replace(account, **{scope_field: "f" * 64})

    class CachedMapping:
        @staticmethod
        def admits(candidate: OwnedPendingOrder) -> bool:
            return candidate is order

        @staticmethod
        def mapping_bytes(candidate: OwnedPendingOrder) -> bytes | None:
            return b"{}" if candidate is order else None

        @staticmethod
        def stage_mapping(candidate: OwnedPendingOrder, payload: bytes) -> None:
            raise AssertionError("a cache hit must not stage a replacement")

    with pytest.raises(ValueError, match="owned order source differs from account"):
        decision_identity._order_mapping(
            order,
            mismatched_account,
            terminal_memo=CachedMapping(),
            cache_terminal=True,
        )


@pytest.mark.parametrize(
    ("alias_field", "alias_value"),
    [
        ("trigger_index", {"nested": []}),
        ("last_processed_index", [3]),
    ],
)
def test_terminal_identity_rebuild_observes_mutable_progress_alias(
    alias_field: str, alias_value: dict[str, list[int]] | list[int]
) -> None:
    account = _account()
    previous = _build(account, action_code=1, risk_weights=(-0.4, -0.0))
    intent = OwnedIntentComponents(
        identity_source=previous,
        dataset_id=account.dataset_id,
        execution_policy_digest=account.execution_policy_digest,
        symbol_index=0,
        requested_quantity=0.5,
        order_type="market",
        time_in_force="gtc",
        limit_price=None,
        stop_price=None,
        submit_index=4,
        eligible_index=5,
        expiry_index=None,
        submission_reference_price=10.0,
        decision_equity=1022.5,
    )
    terminal = replace(
        OwnedPendingOrder(
            intent=intent,
            remaining_quantity=0.5,
            status=OrderStatus.CANCELLED,
            last_processed_index=5,
            terminal_reason="synthetic-cancel",
            evidence_version=1,
            exact_cumulative_filled_quantity="0",
        ),
        **{alias_field: alias_value},
    )
    account = _account(terminal_orders=(terminal,))

    before = _build(account)
    if isinstance(alias_value, dict):
        alias_value["nested"].append(7)
    else:
        alias_value.append(7)
    after = _build(account)

    before_row = json.loads(before.context_state_bytes)["terminal_orders"][0]
    after_row = json.loads(after.context_state_bytes)["terminal_orders"][0]
    assert before_row[alias_field] != after_row[alias_field]
    assert after_row[alias_field] == alias_value
    assert before.context_state_bytes != after.context_state_bytes
    assert before.context_state_digest != after.context_state_digest


def test_mapping_insertion_order_normalizes_and_equal_signed_zero_is_actor_hold():
    account = _account()
    original = _build(account)
    reordered = OwnedAccountComponents(
        account_id=account.account_id,
        dataset_id=account.dataset_id,
        symbols=account.symbols,
        execution_policy_digest=account.execution_policy_digest,
        risk_config=dict(reversed(tuple(account.risk_config.items()))),
        symbol_index=account.symbol_index,
        decision_index=account.decision_index,
        decision_time=account.decision_time,
        book=dict(reversed(tuple(account.book.items()))),
    )
    normalized = build_actor_identity(
        reordered,
        dict(reversed(tuple(_inputs().items()))),
        dict(reversed(tuple(_allocator().items()))),
        independent_target_weight=0.2,
        independent_objective_value=0.01,
        decision_fields=dict(reversed(tuple(_decision_fields().items()))),
        action_contract={"scale": 0.5, "mode": "residual"},
        action_code=2,
        final_risk_weights=(0.2, -0.0),
    )
    assert normalized.context_state_bytes == original.context_state_bytes
    assert normalized.proposal_digest == original.proposal_digest
    assert normalized.decision_digest == original.decision_digest
    assert normalized.target_identity == original.target_identity

    zero_book = BookState.zero(2, 1000.0)
    zero_book.quantities[:] = (0.0, -0.0)
    zero_book._exact_quantities = ("0", "0")
    zero_book.as_of_index = 4
    zero_book.as_of_dataset_id = "d" * 64
    zero_account = _account(book=zero_book)
    held = build_actor_identity(
        zero_account,
        _inputs(),
        _allocator(),
        independent_target_weight=0.2,
        independent_objective_value=0.01,
        decision_fields=_decision_fields(),
        action_contract={"mode": "residual", "scale": 0.5},
        action_code=2,
        final_risk_weights=(0.0, 0.0),
    )
    assert held.current_weights == (0.0, -0.0)
    assert held.target_identity is None


def test_actor_identity_reuses_canonical_account_state_encoding(monkeypatch):
    account = _account()
    original_encoder = decision_identity.canonical_json_bytes
    context_state_encodings: list[bytes] = []

    def count_context_state_encodings(value):
        encoded = original_encoder(value)
        if (
            isinstance(value, dict)
            and value.get("schema") == "independent_allocation_context_v1"
        ):
            context_state_encodings.append(encoded)
        return encoded

    monkeypatch.setattr(
        decision_identity, "canonical_json_bytes", count_context_state_encodings
    )
    built = _build(account)

    assert len(context_state_encodings) == 1
    assert built.context_state_bytes == context_state_encodings[0]
    assert built.context_state_bytes == original_encoder(
        json.loads(built.context_state_bytes)
    )
    # Pinned from the public builder before the one-encoding change.
    assert sha256(built.context_state_bytes).hexdigest() == (
        "81d1e989bfe0ea310e4a384f53cd16e9d225b4a0e6008e157edaad50174736fb"
    )
    assert built.context_state_digest == sha256(built.context_state_bytes).hexdigest()
    assert tuple(asdict(built)) == (
        "context_state_bytes",
        "context_state_digest",
        "context_bytes",
        "proposal_bytes",
        "proposal_digest",
        "decision_bytes",
        "decision_digest",
        "action_bytes",
        "action_digest",
        "actor_decision_identity",
        "target_identity",
        "current_weights",
    )

    changed_book = dict(account.book)
    changed_book["returns_history"] = [*account.book["returns_history"], 0.02]
    changed = _build(replace(account, book=changed_book))
    assert len(context_state_encodings) == 2
    assert changed.context_state_bytes == context_state_encodings[1]
    assert (
        changed.context_state_digest == sha256(changed.context_state_bytes).hexdigest()
    )
    assert changed.context_state_bytes != built.context_state_bytes
    assert changed.context_state_digest != built.context_state_digest


def test_canonical_order_normalization_signed_zero_and_mutation_are_explicit():
    assert canonical_json_bytes({"a": -0.0, "b": 1}) == canonical_json_bytes(
        {"b": 1, "a": -0.0}
    )
    assert canonical_json_bytes({"v": -0.0}) == canonical_json_bytes({"v": -0.0})
    assert canonical_json_bytes({"v": -0.0}) != canonical_json_bytes({"v": 0.0})

    account = _account()
    original = _build(account)
    mutable_book = dict(account.book)
    changed_account = OwnedAccountComponents(
        account_id=account.account_id,
        dataset_id=account.dataset_id,
        symbols=account.symbols,
        execution_policy_digest=account.execution_policy_digest,
        risk_config=account.risk_config,
        symbol_index=account.symbol_index,
        decision_index=account.decision_index,
        decision_time=account.decision_time,
        book=mutable_book,
    )
    before_bytes = original.context_state_bytes
    mutable_book["cash"] = 1001.0
    changed = _build(changed_account)
    assert original.context_state_bytes == before_bytes
    assert changed.context_state_digest != original.context_state_digest


@pytest.mark.parametrize(
    "changes",
    [
        {"unexpected_observed_digest": "a" * 64},
        {"decision_digest": "b" * 64},
    ],
)
def test_closed_input_mapping_rejects_injected_observed_digest_fields(changes):
    account = _account()
    with pytest.raises(ValueError, match="fields are not closed"):
        build_actor_identity(
            account,
            _inputs() | changes,
            _allocator(),
            independent_target_weight=0.2,
            independent_objective_value=0.01,
            decision_fields=_decision_fields(),
            action_contract={"mode": "residual", "scale": 0.5},
            action_code=2,
            final_risk_weights=(0.2, -0.0),
        )


def test_two_symbol_shape_and_finite_canonical_values_are_mandatory():
    account = _account()
    bad_book = dict(account.book)
    bad_book["quantities"] = [1.5]
    with pytest.raises(ValueError, match="two-symbol shape"):
        _build(
            OwnedAccountComponents(
                account_id=account.account_id,
                dataset_id=account.dataset_id,
                symbols=account.symbols,
                execution_policy_digest=account.execution_policy_digest,
                risk_config=account.risk_config,
                symbol_index=account.symbol_index,
                decision_index=account.decision_index,
                decision_time=account.decision_time,
                book=bad_book,
            )
        )
    with pytest.raises(ValueError, match="finite"):
        canonical_json_bytes({"invalid": float("nan")})


@pytest.mark.parametrize(
    ("field", "integer_value", "float_value"),
    [
        ("requested_quantity", 1, 1.0),
        ("decision_equity", 1022, 1022.0),
        ("submission_reference_price", 10, 10.0),
    ],
)
def test_order_identity_preserves_native_integer_vs_float_boundary_and_builder_rejects_int(
    field, integer_value, float_value
):
    account = _account()
    previous = _build(account, action_code=3, risk_weights=(0.4, -0.0))
    base = {
        "dataset_id": account.dataset_id,
        "target_identity": previous.target_identity,
        "execution_policy_digest": account.execution_policy_digest,
        "symbol_index": 0,
        "requested_quantity": 0.75,
        "order_type": OrderType.MARKET,
        "time_in_force": TimeInForce.GTC,
        "limit_price": None,
        "stop_price": None,
        "submit_index": 4,
        "eligible_index": 5,
        "expiry_index": None,
        "submission_reference_price": 10.0,
        "decision_equity": 1022.5,
    }
    native_integer = OrderIntent.create(**(base | {field: integer_value}))
    native_float = OrderIntent.create(**(base | {field: float_value}))
    assert native_integer.order_id != native_float.order_id

    owned_fields = {
        "identity_source": previous,
        "dataset_id": account.dataset_id,
        "execution_policy_digest": account.execution_policy_digest,
        "symbol_index": 0,
        "requested_quantity": 0.75,
        "order_type": "market",
        "time_in_force": "gtc",
        "limit_price": None,
        "stop_price": None,
        "submit_index": 4,
        "eligible_index": 5,
        "expiry_index": None,
        "submission_reference_price": 10.0,
        "decision_equity": 1022.5,
    }
    owned_fields[field] = integer_value
    with pytest.raises(ValueError, match="finite float"):
        owned_order_mapping(
            OwnedPendingOrder(OwnedIntentComponents(**owned_fields), 0.75), account
        )


@pytest.mark.parametrize(
    ("order_type", "time_in_force", "limit_price", "stop_price", "expiry"),
    [
        ("limit", "gtc", 10.0, None, None),
        ("stop_market", "gtc", None, 10.0, None),
        ("market", "day", None, None, 8),
        ("market", "gtc", 10.0, None, None),
        ("market", "gtc", None, 10.0, None),
    ],
)
def test_owned_identity_order_scope_is_market_gtc_without_price_or_expiry(
    order_type, time_in_force, limit_price, stop_price, expiry
):
    account = _account()
    previous = _build(account, action_code=3, risk_weights=(0.4, -0.0))
    intent = OwnedIntentComponents(
        identity_source=previous,
        dataset_id=account.dataset_id,
        execution_policy_digest=account.execution_policy_digest,
        symbol_index=0,
        requested_quantity=0.75,
        order_type=order_type,
        time_in_force=time_in_force,
        limit_price=limit_price,
        stop_price=stop_price,
        submit_index=4,
        eligible_index=5,
        expiry_index=expiry,
        submission_reference_price=10.0,
        decision_equity=1022.5,
    )
    with pytest.raises(ValueError, match="MARKET/GTC"):
        owned_order_mapping(OwnedPendingOrder(intent, 0.75), account)


def test_context_wealth_and_both_weights_match_native_binary64_reduction_exactly():
    book = BookState(
        quantities=np.asarray([0.1, 0.1], dtype=np.float64),
        cash=99987.65432110002,
        mark_prices=np.asarray([123.456789, 123.456789], dtype=np.float64),
        peak_value=100100.0,
        contract_multipliers=np.asarray([1.0, 1.0], dtype=np.float64),
        max_drawdown=0.1,
        turnover_total=0.0,
        total_cost=0.0,
        funding_pnl=0.0,
        fill_count=0,
        rebalance_events=0,
        returns_history=(),
        borrow_cost=0.0,
        margin_used=0.0,
        maintenance_margin=0.0,
        maintenance_requirement=0.0,
        margin_deficit=0.0,
        insolvent=False,
        as_of_index=4,
        as_of_dataset_id="d" * 64,
        _exact_quantities=("1/10", "1/10"),
    )
    account = _account(book=book)
    built = _build(account, risk_weights=(0.2, 0.1))
    context = json.loads(built.context_bytes)
    assert context["equity"].hex() == book.portfolio_value.hex()
    assert tuple(value.hex() for value in built.current_weights) == tuple(
        float(value).hex() for value in book.weights
    )
    assert built.context_state_digest == content_digest(
        json.loads(built.context_state_bytes)
    )


def test_signed_zero_weight_bit_is_preserved_in_full_identity_shape():
    zero_book = BookState.zero(2, 1000.0)
    zero_book.quantities[:] = (0.0, -0.0)
    zero_book._exact_quantities = ("0", "0")
    zero_book.as_of_index = 4
    zero_book.as_of_dataset_id = "d" * 64
    built = _build(_account(book=zero_book), risk_weights=(0.0, 0.0))
    assert tuple(value.hex() for value in built.current_weights) == (
        "0x0.0p+0",
        "-0x0.0p+0",
    )
    assert built.target_identity is None
