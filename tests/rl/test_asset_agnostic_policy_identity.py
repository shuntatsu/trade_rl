from __future__ import annotations

from types import SimpleNamespace

import pytest

from trade_rl.rl.policy_identity import (
    bind_sb3_policy_identity,
    validate_model_sb3_policy_identity,
    validate_sb3_policy_architecture_compatibility,
)
from trade_rl.rl.sequence_architecture import (
    sequence_architecture_identity,
    sequence_asset_binding_identity,
)
from trade_rl.rl.sequence_policy import SequencePolicyArchitecture


def _architecture(
    *, timeframe_layers: int = 1, n_symbols: int = 3
) -> SequencePolicyArchitecture:
    return SequencePolicyArchitecture(
        input_channels={"15m": 3, "1h": 3, "4h": 3, "1d": 3},
        window_lengths={"15m": 4, "1h": 4, "4h": 4, "1d": 4},
        latent_dims={"15m": 8, "1h": 8, "4h": 8, "1d": 8},
        asset_state_width=4,
        snapshot_width=8,
        n_symbols=n_symbols,
        d_model=24,
        timeframe_attention_heads=4,
        timeframe_attention_layers=timeframe_layers,
        asset_attention_heads=4,
        asset_attention_layers=1,
        dropout=0.0,
    )


def _model(architecture: SequencePolicyArchitecture) -> SimpleNamespace:
    return SimpleNamespace(
        policy=SimpleNamespace(
            features_extractor=SimpleNamespace(
                asset_encoder=SimpleNamespace(architecture=architecture)
            ),
            shared_actor_head="hierarchical_gate_target_v1",
            shared_actor_gate_temperature=1.0,
            action_distribution_name="masked_shared_squashed_diag_gaussian_v1",
            log_std=SimpleNamespace(shape=(1,)),
            use_sde=False,
        )
    )


def _assembly(symbols: tuple[str, ...]) -> SimpleNamespace:
    return SimpleNamespace(
        observation_encoder="hierarchical_sequence_v2",
        sequence_symbols=symbols,
        sequence_action_names=tuple(f"target_weight:{symbol}" for symbol in symbols),
        policy_actor_head="hierarchical_gate_target_v1",
        hierarchical_gate_temperature=1.0,
    )


def test_sequence_architecture_digest_does_not_bind_symbol_names() -> None:
    architecture = _architecture()

    first = sequence_architecture_identity(architecture)
    repeated = sequence_architecture_identity(architecture)

    assert first == repeated
    assert first.digest == repeated.digest
    assert "symbols" not in first.digest_payload()
    assert "action_names" not in first.digest_payload()


def test_asset_binding_is_separate_and_symbol_specific() -> None:
    first = sequence_asset_binding_identity(
        n_symbols=3,
        symbols=("BTCUSDT", "ETHUSDT", "BNBUSDT"),
        action_names=(
            "target_weight:BTCUSDT",
            "target_weight:ETHUSDT",
            "target_weight:BNBUSDT",
        ),
    )
    second = sequence_asset_binding_identity(
        n_symbols=3,
        symbols=("SOLUSDT", "XRPUSDT", "ADAUSDT"),
        action_names=(
            "target_weight:SOLUSDT",
            "target_weight:XRPUSDT",
            "target_weight:ADAUSDT",
        ),
    )

    assert first.digest != second.digest
    assert first.n_symbols == second.n_symbols == 3


def test_policy_architecture_is_compatible_across_symbol_bindings() -> None:
    architecture = _architecture()
    first_model = _model(architecture)
    second_model = _model(architecture)
    first = bind_sb3_policy_identity(
        first_model, _assembly(("BTCUSDT", "ETHUSDT", "BNBUSDT"))
    )
    second = bind_sb3_policy_identity(
        second_model, _assembly(("SOLUSDT", "XRPUSDT", "ADAUSDT"))
    )

    assert first["policy_architecture_digest"] == second["policy_architecture_digest"]
    assert first["asset_binding_digest"] != second["asset_binding_digest"]
    validate_sb3_policy_architecture_compatibility(second, first)
    with pytest.raises(ValueError, match="architecture identity mismatch"):
        validate_model_sb3_policy_identity(second_model, first)


def test_policy_architecture_compatibility_rejects_real_model_drift() -> None:
    expected = bind_sb3_policy_identity(
        _model(_architecture(timeframe_layers=1)),
        _assembly(("BTCUSDT", "ETHUSDT", "BNBUSDT")),
    )
    drifted = bind_sb3_policy_identity(
        _model(_architecture(timeframe_layers=2)),
        _assembly(("SOLUSDT", "XRPUSDT", "ADAUSDT")),
    )

    with pytest.raises(ValueError, match="architecture compatibility"):
        validate_sb3_policy_architecture_compatibility(drifted, expected)


def test_three_symbol_checkpoint_is_incompatible_with_one_symbol_policy() -> None:
    one_symbol_model = _model(_architecture(n_symbols=1))
    one_symbol_identity = bind_sb3_policy_identity(
        one_symbol_model,
        _assembly(("BTCUSDT",)),
    )
    three_symbol_model = _model(_architecture(n_symbols=3))
    three_symbol_identity = bind_sb3_policy_identity(
        three_symbol_model,
        _assembly(("BTCUSDT", "ETHUSDT", "BNBUSDT")),
    )

    assert one_symbol_identity["sequence_architecture_digest"] != (
        three_symbol_identity["sequence_architecture_digest"]
    )
    assert one_symbol_identity["asset_binding_digest"] != (
        three_symbol_identity["asset_binding_digest"]
    )
    with pytest.raises(ValueError, match="architecture compatibility"):
        validate_sb3_policy_architecture_compatibility(
            three_symbol_identity,
            one_symbol_identity,
        )
    with pytest.raises(ValueError, match="architecture identity mismatch"):
        validate_model_sb3_policy_identity(three_symbol_model, one_symbol_identity)


def test_asset_binding_rejects_non_target_weight_action_names() -> None:
    with pytest.raises(ValueError, match="target-weight action names"):
        sequence_asset_binding_identity(
            n_symbols=3,
            symbols=("BTCUSDT", "ETHUSDT", "BNBUSDT"),
            action_names=("BTCUSDT", "ETHUSDT", "BNBUSDT"),
        )
