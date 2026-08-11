from __future__ import annotations

from dataclasses import replace

import pytest

from trade_rl.artifacts.hashing import content_digest
from trade_rl.rl.instrument_episode_routing import (
    GENERIC_INSTRUMENT_ACTION_NAMES,
    GENERIC_INSTRUMENT_SYMBOL,
    GENERIC_INSTRUMENT_SYMBOLS,
    DeterministicBalancedInstrumentRouter,
    InstrumentDatasetBinding,
    InstrumentDatasetSplit,
    InstrumentEpisodeBinding,
)


def _digest(label: str) -> str:
    return content_digest(label)


def _binding(
    symbol: str,
    *,
    split: InstrumentDatasetSplit | str = InstrumentDatasetSplit.TRAIN,
    partition_digest: str | None = None,
) -> InstrumentDatasetBinding:
    return InstrumentDatasetBinding(
        concrete_symbol=symbol,
        source_dataset_id=_digest(f"source:{symbol}"),
        symbol_dataset_digest=_digest(f"dataset:{symbol}"),
        execution_metadata_digest=_digest(f"execution:{symbol}"),
        instrument_descriptor_digest=_digest(f"descriptor:{symbol}"),
        partition_digest=partition_digest or _digest("partition"),
        split=split,
    )


def test_generic_policy_contract_is_one_identity_free_instrument_slot() -> None:
    assert GENERIC_INSTRUMENT_SYMBOL == "INSTRUMENT"
    assert GENERIC_INSTRUMENT_SYMBOLS == ("INSTRUMENT",)
    assert GENERIC_INSTRUMENT_ACTION_NAMES == ("target_weight:INSTRUMENT",)


def test_dataset_binding_is_canonical_digest_bound_and_round_trips() -> None:
    binding = _binding("BTCUSDT")

    payload = binding.to_json_dict()

    assert payload["schema_version"] == "instrument_dataset_binding_v1"
    assert payload["binding_digest"] == binding.digest
    assert InstrumentDatasetBinding.from_json_dict(payload) == binding
    assert "INSTRUMENT" not in binding.digest_payload().values()


def test_dataset_binding_rejects_generic_symbol_invalid_digest_and_split() -> None:
    with pytest.raises(ValueError, match="concrete_symbol"):
        _binding("INSTRUMENT")
    with pytest.raises(ValueError, match="symbol_dataset_digest"):
        replace(_binding("BTCUSDT"), symbol_dataset_digest="not-a-digest")
    with pytest.raises(ValueError, match="split"):
        _binding("BTCUSDT", split="shadow")


def test_dataset_binding_strict_load_rejects_extra_or_tampered_fields() -> None:
    payload = _binding("BTCUSDT").to_json_dict()

    with pytest.raises(ValueError, match="fields"):
        InstrumentDatasetBinding.from_json_dict({**payload, "ticker_id": 7})

    tampered = dict(payload)
    tampered["concrete_symbol"] = "ETHUSDT"
    with pytest.raises(ValueError, match="digest"):
        InstrumentDatasetBinding.from_json_dict(tampered)


def test_router_is_input_order_independent_and_balanced_per_cycle() -> None:
    bindings = tuple(
        _binding(symbol)
        for symbol in ("BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT")
    )
    first = DeterministicBalancedInstrumentRouter(
        bindings,
        run_seed=41,
        environment_index=3,
        partition_digest=_digest("partition"),
    )
    reordered = DeterministicBalancedInstrumentRouter(
        tuple(reversed(bindings)),
        run_seed=41,
        environment_index=3,
        partition_digest=_digest("partition"),
    )

    first_routes = tuple(first.route(index) for index in range(2 * len(bindings)))
    reordered_routes = tuple(
        reordered.route(index) for index in range(2 * len(bindings))
    )

    assert tuple(route.binding.concrete_symbol for route in first_routes) == tuple(
        route.binding.concrete_symbol for route in reordered_routes
    )
    for cycle in range(2):
        cycle_routes = first_routes[cycle * len(bindings) : (cycle + 1) * len(bindings)]
        assert {route.binding.concrete_symbol for route in cycle_routes} == {
            binding.concrete_symbol for binding in bindings
        }
        assert tuple(route.routing_position for route in cycle_routes) == tuple(
            range(len(bindings))
        )
        assert {route.routing_cycle for route in cycle_routes} == {cycle}


def test_router_route_is_stateless_and_repeatable() -> None:
    router = DeterministicBalancedInstrumentRouter(
        tuple(_binding(symbol) for symbol in ("BTCUSDT", "ETHUSDT", "SOLUSDT")),
        run_seed=11,
        environment_index=0,
        partition_digest=_digest("partition"),
    )

    assert router.route(7) == router.route(7)
    assert router.route(7).completed_episode_count == 7
    assert router.route(7).routing_cycle == 2
    assert router.route(7).routing_position == 1


def test_router_identity_changes_with_environment_index_without_mutable_rng() -> None:
    bindings = tuple(_binding(symbol) for symbol in ("BTCUSDT", "ETHUSDT", "SOLUSDT"))
    first = DeterministicBalancedInstrumentRouter(
        bindings,
        run_seed=11,
        environment_index=0,
        partition_digest=_digest("partition"),
    )
    second = DeterministicBalancedInstrumentRouter(
        bindings,
        run_seed=11,
        environment_index=1,
        partition_digest=_digest("partition"),
    )

    assert first.digest != second.digest
    assert first.to_json_dict()["environment_index"] == 0
    assert second.to_json_dict()["environment_index"] == 1


def test_router_rejects_non_train_duplicate_and_foreign_partition_bindings() -> None:
    train = _binding("BTCUSDT")
    validation = _binding("ETHUSDT", split=InstrumentDatasetSplit.VALIDATION)
    foreign = _binding("SOLUSDT", partition_digest=_digest("another-partition"))

    with pytest.raises(ValueError, match="train"):
        DeterministicBalancedInstrumentRouter(
            (train, validation),
            run_seed=1,
            environment_index=0,
            partition_digest=_digest("partition"),
        )
    with pytest.raises(ValueError, match="unique"):
        DeterministicBalancedInstrumentRouter(
            (train, train),
            run_seed=1,
            environment_index=0,
            partition_digest=_digest("partition"),
        )
    with pytest.raises(ValueError, match="partition"):
        DeterministicBalancedInstrumentRouter(
            (train, foreign),
            run_seed=1,
            environment_index=0,
            partition_digest=_digest("partition"),
        )


def test_router_rejects_invalid_indices_and_seed_types() -> None:
    bindings = (_binding("BTCUSDT"),)

    with pytest.raises(ValueError, match="run_seed"):
        DeterministicBalancedInstrumentRouter(
            bindings,
            run_seed=True,
            environment_index=0,
            partition_digest=_digest("partition"),
        )
    with pytest.raises(ValueError, match="environment_index"):
        DeterministicBalancedInstrumentRouter(
            bindings,
            run_seed=1,
            environment_index=-1,
            partition_digest=_digest("partition"),
        )

    router = DeterministicBalancedInstrumentRouter(
        bindings,
        run_seed=1,
        environment_index=0,
        partition_digest=_digest("partition"),
    )
    with pytest.raises(ValueError, match="completed_episode_count"):
        router.route(-1)
    with pytest.raises(ValueError, match="completed_episode_count"):
        router.route(True)


def test_episode_binding_closes_route_dataset_and_episode_range() -> None:
    router = DeterministicBalancedInstrumentRouter(
        tuple(_binding(symbol) for symbol in ("BTCUSDT", "ETHUSDT")),
        run_seed=5,
        environment_index=2,
        partition_digest=_digest("partition"),
    )
    route = router.route(3)

    episode = InstrumentEpisodeBinding.from_route(
        route,
        episode_start=120,
        episode_stop=360,
    )

    payload = episode.to_json_dict()
    assert payload["episode_binding_digest"] == episode.digest
    assert payload["concrete_symbol"] == route.binding.concrete_symbol
    assert payload["environment_index"] == 2
    assert payload["completed_episode_count"] == 3
    assert payload["routing_cycle"] == 1
    assert payload["routing_position"] == 1
    assert InstrumentEpisodeBinding.from_json_dict(payload) == episode


def test_episode_binding_rejects_invalid_range_or_tampered_payload() -> None:
    router = DeterministicBalancedInstrumentRouter(
        (_binding("BTCUSDT"),),
        run_seed=5,
        environment_index=2,
        partition_digest=_digest("partition"),
    )
    route = router.route(0)

    with pytest.raises(ValueError, match="episode"):
        InstrumentEpisodeBinding.from_route(
            route,
            episode_start=12,
            episode_stop=12,
        )

    episode = InstrumentEpisodeBinding.from_route(
        route,
        episode_start=12,
        episode_stop=24,
    )
    tampered = episode.to_json_dict()
    tampered["routing_position"] = 3
    with pytest.raises(ValueError, match="digest|routing"):
        InstrumentEpisodeBinding.from_json_dict(tampered)
