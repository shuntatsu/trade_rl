from __future__ import annotations

import pytest

from trade_rl.artifacts.hashing import content_digest
from trade_rl.rl.universal_instrument_binding import (
    GENERIC_INSTRUMENT_SYMBOL,
    GENERIC_INSTRUMENT_SYMBOLS,
    GENERIC_TARGET_WEIGHT_ACTION_NAMES,
    InstrumentDatasetBinding,
    InstrumentEpisodeBinding,
    validate_training_instrument_bindings,
)


def _digest(value: object) -> str:
    return content_digest(value)


def _binding(symbol: str, *, split: str = "train") -> InstrumentDatasetBinding:
    return InstrumentDatasetBinding(
        concrete_symbol=symbol,
        source_dataset_id=_digest((symbol, "source")),
        symbol_dataset_digest=_digest((symbol, "dataset")),
        execution_metadata_digest=_digest((symbol, "execution")),
        instrument_descriptor_digest=_digest((symbol, "descriptor")),
        split=split,
    )


def test_generic_policy_facing_contract_is_exactly_one_instrument() -> None:
    assert GENERIC_INSTRUMENT_SYMBOL == "INSTRUMENT"
    assert GENERIC_INSTRUMENT_SYMBOLS == ("INSTRUMENT",)
    assert GENERIC_TARGET_WEIGHT_ACTION_NAMES == ("target_weight:INSTRUMENT",)


def test_dataset_binding_is_canonical_and_round_trips() -> None:
    binding = _binding("BTCUSDT")

    assert InstrumentDatasetBinding.from_json_dict(binding.to_json_dict()) == binding
    assert binding.digest == content_digest(binding.to_json_dict())
    assert binding.to_json_dict() == {
        "concrete_symbol": "BTCUSDT",
        "execution_metadata_digest": _digest(("BTCUSDT", "execution")),
        "instrument_descriptor_digest": _digest(("BTCUSDT", "descriptor")),
        "schema_version": "instrument_dataset_binding_v1",
        "source_dataset_id": _digest(("BTCUSDT", "source")),
        "split": "train",
        "symbol_dataset_digest": _digest(("BTCUSDT", "dataset")),
    }


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("concrete_symbol", " ", "non-empty"),
        ("source_dataset_id", "x", "SHA-256"),
        ("symbol_dataset_digest", "x", "SHA-256"),
        ("execution_metadata_digest", "x", "SHA-256"),
        ("instrument_descriptor_digest", "x", "SHA-256"),
        ("split", "sealed", "split"),
    ],
)
def test_dataset_binding_rejects_invalid_identity(
    field: str,
    value: str,
    match: str,
) -> None:
    payload: dict[str, object] = {
        "concrete_symbol": "BTCUSDT",
        "source_dataset_id": _digest("source"),
        "symbol_dataset_digest": _digest("dataset"),
        "execution_metadata_digest": _digest("execution"),
        "instrument_descriptor_digest": _digest("descriptor"),
        "split": "train",
    }
    payload[field] = value

    with pytest.raises(ValueError, match=match):
        InstrumentDatasetBinding(**payload)  # type: ignore[arg-type]


def test_episode_binding_flattens_concrete_identity_for_telemetry() -> None:
    dataset_binding = _binding("ETHUSDT")
    episode = InstrumentEpisodeBinding(
        dataset_binding=dataset_binding,
        episode_start=12,
        episode_stop=44,
        episode_seed=123,
        environment_index=2,
        completed_episode_count=7,
        routing_cycle=1,
        routing_position=2,
    )

    payload = episode.to_json_dict()

    assert payload["concrete_symbol"] == "ETHUSDT"
    assert payload["source_dataset_id"] == dataset_binding.source_dataset_id
    assert payload["dataset_binding_digest"] == dataset_binding.digest
    assert payload["episode_start"] == 12
    assert payload["episode_stop"] == 44
    assert payload["routing_cycle"] == 1
    assert payload["routing_position"] == 2
    assert episode.digest == content_digest(payload)
    assert InstrumentEpisodeBinding.from_json_dict(payload) == episode


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("episode_start", -1, "episode_start"),
        ("episode_stop", 0, "episode_stop"),
        ("episode_seed", -1, "episode_seed"),
        ("environment_index", -1, "environment_index"),
        ("completed_episode_count", -1, "completed_episode_count"),
        ("routing_cycle", -1, "routing_cycle"),
        ("routing_position", -1, "routing_position"),
    ],
)
def test_episode_binding_rejects_invalid_lifecycle_values(
    field: str,
    value: int,
    match: str,
) -> None:
    payload: dict[str, object] = {
        "dataset_binding": _binding("BTCUSDT"),
        "episode_start": 0,
        "episode_stop": 10,
        "episode_seed": 5,
        "environment_index": 0,
        "completed_episode_count": 0,
        "routing_cycle": 0,
        "routing_position": 0,
    }
    payload[field] = value

    with pytest.raises(ValueError, match=match):
        InstrumentEpisodeBinding(**payload)  # type: ignore[arg-type]


def test_training_binding_validation_requires_exact_train_closure() -> None:
    train_symbols = ("BTCUSDT", "ETHUSDT", "SOLUSDT")
    bindings = tuple(_binding(symbol) for symbol in reversed(train_symbols))

    resolved = validate_training_instrument_bindings(train_symbols, bindings)

    assert tuple(resolved) == train_symbols
    assert resolved["BTCUSDT"] == _binding("BTCUSDT")


@pytest.mark.parametrize(
    "bindings",
    [
        (_binding("BTCUSDT"), _binding("ETHUSDT")),
        (
            _binding("BTCUSDT"),
            _binding("ETHUSDT"),
            _binding("SOLUSDT"),
            _binding("XRPUSDT"),
        ),
        (
            _binding("BTCUSDT"),
            _binding("ETHUSDT"),
            _binding("SOLUSDT", split="validation"),
        ),
        (
            _binding("BTCUSDT"),
            _binding("ETHUSDT"),
            _binding("ETHUSDT"),
        ),
    ],
)
def test_training_binding_validation_fails_closed_on_leakage_or_closure(
    bindings: tuple[InstrumentDatasetBinding, ...],
) -> None:
    with pytest.raises(ValueError, match="closure|train|duplicate"):
        validate_training_instrument_bindings(
            ("BTCUSDT", "ETHUSDT", "SOLUSDT"),
            bindings,
        )
