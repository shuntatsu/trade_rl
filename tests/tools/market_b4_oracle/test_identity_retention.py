from __future__ import annotations

import importlib.util
from pathlib import Path
from types import MappingProxyType, ModuleType
from typing import Any

import pytest

from tools.market_b4_oracle import decision_identity, frozen_market_replay
from tools.market_b4_oracle.decision_identity import (
    ActorIdentity,
    OwnedAccountComponents,
    OwnedIntentComponents,
    OwnedPendingOrder,
    canonical_json_bytes,
    owned_order_mapping,
)
from tools.market_b4_oracle.expected_intent import ExpectedIntent
from tools.market_replay_oracle.artifact import VerifiedDatasetContent


def _load_test_module(filename: str, module_name: str) -> ModuleType:
    path = Path(__file__).with_name(filename)
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"could not load existing synthetic fixture: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_IDENTITY_FIXTURES = _load_test_module(
    "test_decision_identity.py", "_identity_retention_identity_fixtures"
)
_REPLAY_FIXTURES = _load_test_module(
    "test_frozen_market_replay.py", "_identity_retention_replay_fixtures"
)


def _identity_fixture(name: str, *args: Any, **kwargs: Any) -> Any:
    return getattr(_IDENTITY_FIXTURES, name)(*args, **kwargs)


def _replay_fixture(name: str, *args: Any, **kwargs: Any) -> Any:
    return getattr(_REPLAY_FIXTURES, name)(*args, **kwargs)


def _synthetic_content_with_complete_funding(rows: int) -> VerifiedDatasetContent:
    content = _replay_fixture("_content", rows=rows)
    arrays = dict(content.arrays)
    for name in ("funding_rate", "funding_event_count", "funding_due"):
        arrays[name] = arrays[name].copy()
    for index in range(72, rows, 8):
        arrays["funding_rate"][index, :] = 0.0001
        arrays["funding_event_count"][index, :] = 1
        arrays["funding_due"][index, :] = True
    for name in ("funding_rate", "funding_event_count", "funding_due"):
        arrays[name].setflags(write=False)
    return VerifiedDatasetContent(
        content.dataset_id,
        MappingProxyType(arrays),
        content.dataset,
    )


def _intent_fields(
    account: OwnedAccountComponents,
    *,
    target_identity: str,
    submit_index: int,
    replaced_order_id: str | None = None,
) -> dict[str, Any]:
    return {
        "dataset_id": account.dataset_id,
        "target_identity": target_identity,
        "execution_policy_digest": account.execution_policy_digest,
        "symbol_index": 0,
        "requested_quantity": 0.75,
        "order_type": "market",
        "time_in_force": "gtc",
        "limit_price": None,
        "stop_price": None,
        "submit_index": submit_index,
        "eligible_index": submit_index + 1,
        "expiry_index": None,
        "submission_reference_price": 10.0,
        "decision_equity": 1022.5,
        "replaced_order_id": replaced_order_id,
    }


def test_compact_basis_requires_internal_factory_for_ordinary_construction() -> None:
    basis_type = getattr(decision_identity, "ActorIntentBasis", None)
    assert isinstance(basis_type, type)

    with pytest.raises(TypeError, match="internal factory"):
        basis_type()


def test_compact_basis_factory_rejects_incomplete_actor_identity() -> None:
    incomplete_identity = object.__new__(ActorIdentity)
    factory = decision_identity._actor_intent_basis_from_identity

    with pytest.raises(ValueError, match="target_identity"):
        factory(incomplete_identity)


@pytest.mark.parametrize("source_kind", ["actor_identity", "compact_basis"])
def test_order_mapping_rejects_missing_target_identity(source_kind: str) -> None:
    account = _identity_fixture("_account")
    identity = _identity_fixture(
        "_build", account, action_code=3, risk_weights=(0.4, -0.0)
    )
    if source_kind == "actor_identity":
        identity_source = object.__new__(ActorIdentity)
    else:
        basis_type = decision_identity.ActorIntentBasis
        identity_source = object.__new__(basis_type)
    fields = _intent_fields(
        account, target_identity=identity.target_identity, submit_index=4
    )
    order = OwnedPendingOrder(
        intent=OwnedIntentComponents(
            identity_source=identity_source,
            **{
                key: value
                for key, value in fields.items()
                if key not in {"target_identity", "replaced_order_id"}
            },
        ),
        remaining_quantity=0.75,
    )

    with pytest.raises(ValueError, match="target_identity"):
        owned_order_mapping(order, account)


def test_compact_intent_basis_is_builder_derived_and_keeps_replacement_hashes() -> None:
    account = _identity_fixture("_account")
    identity = _identity_fixture(
        "_build", account, action_code=3, risk_weights=(0.4, -0.0)
    )
    basis_type = getattr(decision_identity, "ActorIntentBasis", None)
    factory = getattr(decision_identity, "_actor_intent_basis_from_identity", None)

    assert isinstance(basis_type, type)
    assert callable(factory)
    basis = factory(identity)
    assert type(basis) is basis_type
    assert basis.target_identity == identity.target_identity
    assert not hasattr(basis, "context_state_bytes")
    assert factory(None) is None

    fake_source = type(
        "FakeActorIdentity", (), {"target_identity": identity.target_identity}
    )()
    with pytest.raises(TypeError):
        factory("f" * 64)
    with pytest.raises(TypeError):
        factory({"target_identity": identity.target_identity})
    with pytest.raises(TypeError):
        factory(fake_source)
    with pytest.raises(TypeError):
        basis_type(target_identity=identity.target_identity)

    invalid_sources = (
        None,
        "f" * 64,
        {"target_identity": identity.target_identity},
        fake_source,
    )
    for invalid_source in invalid_sources:
        forged_order = OwnedPendingOrder(
            intent=OwnedIntentComponents(
                identity_source=invalid_source,
                **{
                    key: value
                    for key, value in _intent_fields(
                        account,
                        target_identity=identity.target_identity,
                        submit_index=4,
                    ).items()
                    if key not in {"target_identity", "replaced_order_id"}
                },
            ),
            remaining_quantity=0.75,
        )
        with pytest.raises(ValueError, match="builder result"):
            owned_order_mapping(forged_order, account)

    first_fields = _intent_fields(
        account, target_identity=identity.target_identity, submit_index=4
    )
    first_expected = ExpectedIntent.create(**first_fields)
    first_owned = OwnedPendingOrder(
        intent=OwnedIntentComponents(
            identity_source=identity,
            **{
                key: value
                for key, value in first_fields.items()
                if key not in {"target_identity", "replaced_order_id"}
            },
        ),
        remaining_quantity=0.75,
    )
    second_fields = _intent_fields(
        account,
        target_identity=identity.target_identity,
        submit_index=5,
        replaced_order_id=first_expected.order_id,
    )
    second_expected = ExpectedIntent.create(**second_fields)
    legacy_second = OwnedPendingOrder(
        intent=OwnedIntentComponents(
            identity_source=identity,
            replaced_order=first_owned,
            **{
                key: value
                for key, value in second_fields.items()
                if key not in {"target_identity", "replaced_order_id"}
            },
        ),
        remaining_quantity=0.75,
    )
    compact_first = OwnedPendingOrder(
        intent=OwnedIntentComponents(
            identity_source=basis,
            **{
                key: value
                for key, value in first_fields.items()
                if key not in {"target_identity", "replaced_order_id"}
            },
        ),
        remaining_quantity=0.75,
    )
    compact_second = OwnedPendingOrder(
        intent=OwnedIntentComponents(
            identity_source=basis,
            replaced_order=compact_first,
            **{
                key: value
                for key, value in second_fields.items()
                if key not in {"target_identity", "replaced_order_id"}
            },
        ),
        remaining_quantity=0.75,
    )

    legacy_mapping = owned_order_mapping(legacy_second, account)
    compact_mapping = owned_order_mapping(compact_second, account)
    assert canonical_json_bytes(compact_mapping) == canonical_json_bytes(legacy_mapping)
    assert compact_mapping["intent"]["order_id"] == second_expected.order_id
    assert compact_mapping["intent"]["replaced_order_id"] == first_expected.order_id


def _run_synthetic_actor_prefix(
    steps: int,
    monkeypatch: pytest.MonkeyPatch,
    *,
    retain_full_identity: bool,
) -> tuple[Any, tuple[ActorIdentity, ...], tuple[Any, ...]]:
    actor_start = 24
    actor_stop = actor_start + steps
    terminal_stop = actor_stop + 1
    content = _synthetic_content_with_complete_funding(rows=terminal_stop + 1)
    window = frozen_market_replay.FrozenMarketWindow(
        actor_start=actor_start,
        actor_stop=actor_stop,
        terminal_stop=terminal_stop,
    )
    binding = frozen_market_replay.FrozenMarketBinding._comparison_fixture(
        content=content,
        window=window,
        selected_symbol_index=0,
        expected_horizon_seconds=3600,
        observation_feature_indices=tuple(range(6)),
        native_role="nonrl",
        initial_capital=100_000.0,
        scenario="COST2L0",
    )
    runner = frozen_market_replay.FrozenMarketReplay(
        content=content,
        policy=_replay_fixture("_policy"),
        binding=binding,
        reserve_plan_digest="c" * 64,
    )
    forecast_index = _replay_fixture("_forecast_index", through=actor_stop + 1)
    identities: list[ActorIdentity] = []
    advances: list[Any] = []
    original_builder = frozen_market_replay._build_actor_identity_core

    def observe_builder(*args: Any, **kwargs: Any) -> ActorIdentity:
        identity = original_builder(*args, **kwargs)
        reference_kwargs = {
            name: value for name, value in kwargs.items() if name != "terminal_memo"
        }
        reference = decision_identity.build_actor_identity(*args, **reference_kwargs)
        assert identity == reference
        for name in (
            "context_state_bytes",
            "context_bytes",
            "proposal_bytes",
            "decision_bytes",
            "action_bytes",
            "context_state_digest",
            "proposal_digest",
            "decision_digest",
            "action_digest",
            "actor_decision_identity",
            "target_identity",
        ):
            assert getattr(identity, name) == getattr(reference, name)
        identities.append(identity)
        return identity

    def retain_builder_result(identity: ActorIdentity | None) -> ActorIdentity | None:
        return identity

    with monkeypatch.context() as patch:
        patch.setattr(
            frozen_market_replay,
            "_build_actor_identity_core",
            observe_builder,
        )
        if retain_full_identity:
            patch.setattr(
                frozen_market_replay,
                "_actor_intent_basis_from_identity",
                retain_builder_result,
                raising=False,
            )
        for _ in range(steps):
            advances.append(runner.advance(forecast_index=forecast_index, role="NONRL"))
        reserve = runner.advance(forecast_index=forecast_index, role="NONRL")

    assert reserve.phase == "reserve"
    assert reserve.actor_identity is None
    advances.append(reserve)
    return runner, tuple(identities), tuple(advances)


@pytest.mark.parametrize("steps", [40, 160])
def test_synthetic_history_retains_only_compact_basis_with_exact_output_parity(
    steps: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    legacy, legacy_identities, legacy_advances = _run_synthetic_actor_prefix(
        steps, monkeypatch, retain_full_identity=True
    )
    compact, compact_identities, compact_advances = _run_synthetic_actor_prefix(
        steps, monkeypatch, retain_full_identity=False
    )

    assert len(legacy_identities) == len(compact_identities) == steps
    assert legacy_identities == compact_identities
    assert legacy_advances == compact_advances
    assert legacy.account_snapshot == compact.account_snapshot
    legacy_completion = legacy.completion()
    compact_completion = compact.completion()
    assert legacy_completion == compact_completion
    assert legacy_completion["status"] == "comparison_window_consumed"
    assert legacy_completion["terminal_flat"] is True
    assert legacy_completion["no_active_orders"] is True
    assert tuple(legacy._all_orders) == tuple(compact._all_orders)
    for order_id, compact_order in compact._all_orders.items():
        legacy_order = legacy._all_orders[order_id]
        assert compact_order.intent == legacy_order.intent
        assert compact_order.intent.order_id == order_id
        assert (
            compact_order.intent.replaced_order_id
            == legacy_order.intent.replaced_order_id
        )
        assert getattr(
            compact_order.actor_identity, "target_identity", None
        ) == getattr(legacy_order.actor_identity, "target_identity", None)

    basis_type = getattr(decision_identity, "ActorIntentBasis", None)
    assert isinstance(basis_type, type)
    assert compact._all_orders
    retained_full_contexts = {
        id(order.actor_identity): order.actor_identity
        for order in compact._all_orders.values()
        if isinstance(order.actor_identity, ActorIdentity)
    }
    assert retained_full_contexts == {}
    compact_basis_orders = [
        order
        for order in compact._all_orders.values()
        if order.actor_identity is not None
    ]
    assert compact_basis_orders
    assert all(
        type(order.actor_identity) is basis_type for order in compact_basis_orders
    )
    assert all(
        not hasattr(order.actor_identity, "context_state_bytes")
        for order in compact_basis_orders
    )
    actor_stop = 24 + steps
    assert all(
        order.intent.submit_index >= actor_stop
        for order in compact._all_orders.values()
        if order.actor_identity is None
    )

    legacy_retained = {
        id(order.actor_identity): order.actor_identity
        for order in legacy._all_orders.values()
        if isinstance(order.actor_identity, ActorIdentity)
    }
    legacy_full_bytes = sum(
        len(identity.context_state_bytes) for identity in legacy_retained.values()
    )
    compact_full_bytes = sum(
        len(identity.context_state_bytes)
        for identity in retained_full_contexts.values()
    )
    assert legacy_full_bytes > 0
    assert compact_full_bytes == 0
