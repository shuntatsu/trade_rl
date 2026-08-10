from __future__ import annotations

import pytest

from trade_rl.artifacts.hashing import content_digest
from trade_rl.rl.universal_episode_router import (
    DeterministicBalancedInstrumentRouter,
    InstrumentRoute,
)


def _router(*, environment_index: int = 0) -> DeterministicBalancedInstrumentRouter:
    return DeterministicBalancedInstrumentRouter(
        train_symbols=(
            "BTCUSDT",
            "ETHUSDT",
            "SOLUSDT",
            "XRPUSDT",
            "ADAUSDT",
        ),
        partition_digest=content_digest("partition"),
        run_seed=17,
        environment_index=environment_index,
    )


def test_router_uses_every_symbol_once_before_repetition() -> None:
    router = _router()
    count = len(router.train_symbols)

    first_cycle = tuple(router.route(index) for index in range(count))
    second_cycle = tuple(router.route(count + index) for index in range(count))

    assert {route.concrete_symbol for route in first_cycle} == set(router.train_symbols)
    assert len({route.concrete_symbol for route in first_cycle}) == count
    assert {route.concrete_symbol for route in second_cycle} == set(
        router.train_symbols
    )
    assert tuple(route.routing_cycle for route in first_cycle) == (0,) * count
    assert tuple(route.routing_position for route in first_cycle) == tuple(range(count))
    assert tuple(route.routing_cycle for route in second_cycle) == (1,) * count


def test_router_is_deterministic_and_routes_have_canonical_identity() -> None:
    first = _router()
    second = _router()

    observed = tuple(first.route(index) for index in range(17))

    assert observed == tuple(second.route(index) for index in range(17))
    assert all(isinstance(route, InstrumentRoute) for route in observed)
    assert all(
        route.digest == content_digest(route.to_json_dict()) for route in observed
    )
    assert first.digest == second.digest


def test_environment_index_is_part_of_router_identity() -> None:
    first = _router(environment_index=0)
    second = _router(environment_index=1)

    assert first.digest != second.digest
    assert first.to_json_dict()["environment_index"] == 0
    assert second.to_json_dict()["environment_index"] == 1


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("train_symbols", (), "train_symbols"),
        ("train_symbols", ("BTCUSDT", "BTCUSDT"), "unique"),
        ("partition_digest", "x", "SHA-256"),
        ("run_seed", -1, "run_seed"),
        ("environment_index", -1, "environment_index"),
    ],
)
def test_router_rejects_invalid_identity(
    field: str,
    value: object,
    match: str,
) -> None:
    payload: dict[str, object] = {
        "train_symbols": ("BTCUSDT",),
        "partition_digest": content_digest("partition"),
        "run_seed": 17,
        "environment_index": 0,
    }
    payload[field] = value

    with pytest.raises(ValueError, match=match):
        DeterministicBalancedInstrumentRouter(**payload)  # type: ignore[arg-type]


@pytest.mark.parametrize("completed_episode_count", [-1, True, 1.5])
def test_router_rejects_invalid_completed_episode_count(
    completed_episode_count: object,
) -> None:
    with pytest.raises(ValueError, match="completed_episode_count"):
        _router().route(completed_episode_count)  # type: ignore[arg-type]
