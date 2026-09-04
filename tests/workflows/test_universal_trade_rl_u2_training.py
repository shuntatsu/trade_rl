from __future__ import annotations

from collections.abc import Mapping
from types import SimpleNamespace
from typing import Any, cast

import torch

from tests.rl.universal_trade_test_support import make_u1_market, make_u1_wrapper
from trade_rl.artifacts.hashing import content_digest
from trade_rl.integrations.sb3_model_assembly import resolve_sb3_policy_assembly
from trade_rl.rl.algorithm_configs import build_algorithm_config
from trade_rl.rl.observations import CURRENT_WEIGHT_SOURCE
from trade_rl.rl.policy_identity import bind_sb3_policy_identity
from trade_rl.rl.universal_instrument_binding import InstrumentDatasetBinding
from trade_rl.rl.universal_single_instrument_env import EpisodeRoutedSingleInstrumentEnv
from trade_rl.rl.universal_trade_contract import (
    UNIVERSAL_TRADE_OBSERVATION_SCHEMA,
    UNIVERSAL_TRADE_STATE_LAYOUT_SCHEMA,
)
from trade_rl.rl.universal_trade_observation import UNIVERSAL_TRADE_POLICY_STATE_FIELDS
from trade_rl.workflows.universal_trade_rl_u2_contract import (
    build_universal_trade_rl_u2_training_config,
)


def _u2_routed_u1_probe() -> EpisodeRoutedSingleInstrumentEnv:
    dataset = make_u1_market(symbol="BTCUSDT", n_bars=10_000)
    binding = InstrumentDatasetBinding(
        concrete_symbol="BTCUSDT",
        source_dataset_id=dataset.dataset_id,
        symbol_dataset_digest=content_digest({"fixture": "u2-source"}),
        execution_metadata_digest=content_digest({"fixture": "u2-execution"}),
        instrument_descriptor_digest=content_digest({"fixture": "u2-descriptor"}),
        split="train",
    )

    def factory(_binding: InstrumentDatasetBinding):
        return make_u1_wrapper(dataset=dataset)

    return EpisodeRoutedSingleInstrumentEnv(
        train_symbols=("BTCUSDT",),
        partition_digest=content_digest({"fixture": "u2-fit-partition"}),
        bindings=(binding,),
        environment_factory=factory,
        run_seed=0,
        environment_index=0,
        instrument_context_provider=None,
        v4_context_provider=None,
        training_contract_digest=content_digest({"fixture": "u2-training-contract"}),
    )


def _assembly(probe: EpisodeRoutedSingleInstrumentEnv):
    config = build_universal_trade_rl_u2_training_config()
    assembly = resolve_sb3_policy_assembly(
        probe=probe,
        identity={
            "action_size": 1,
            "action_names": ("target_weight:INSTRUMENT",),
        },
        config=config,
        algorithm_config=build_algorithm_config(config),
    )
    return config, assembly


def _u1_extractor(probe: EpisodeRoutedSingleInstrumentEnv, assembly: Any):
    extractor_type = cast(Any, assembly.policy_kwargs["features_extractor_class"])
    raw_kwargs = assembly.policy_kwargs["features_extractor_kwargs"]
    assert isinstance(raw_kwargs, Mapping)
    return extractor_type(probe.observation_space, **dict(raw_kwargs))


def test_u2_routed_u1_observation_has_explicit_sb3_sequence_adapter() -> None:
    probe = _u2_routed_u1_probe()
    try:
        observation, _info = probe.reset(seed=0)
        assert set(observation) == {
            "sequence_15m_values",
            "sequence_15m_available",
            "sequence_15m_staleness",
            "sequence_1h_values",
            "sequence_1h_available",
            "sequence_1h_staleness",
            "sequence_4h_values",
            "sequence_4h_available",
            "sequence_4h_staleness",
            "sequence_1d_values",
            "sequence_1d_available",
            "sequence_1d_staleness",
            "policy_state",
        }

        _config, assembly = _assembly(probe)
        extractor = assembly.policy_kwargs["features_extractor_class"]
        assert getattr(extractor, "__name__", "") == (
            "UniversalTradeU1SequenceFeatureExtractor"
        )
        assert assembly.sequence_metadata is not None
        assert assembly.sequence_metadata["schema_version"] == (
            "universal_trade_u1_sequence_adapter_v1"
        )
        assert assembly.sequence_symbols == ("INSTRUMENT",)
        assert assembly.sequence_action_names == ("target_weight:INSTRUMENT",)
        assert assembly.rollout_buffer_class is None
        assert assembly.rollout_buffer_kwargs is None
    finally:
        probe.close()


def test_u2_u1_sequence_adapter_constructs_and_forwards_exact_u1_observation() -> None:
    probe = _u2_routed_u1_probe()
    try:
        observation, _info = probe.reset(seed=0)
        config, assembly = _assembly(probe)
        extractor = _u1_extractor(probe, assembly)

        batched = {
            key: torch.as_tensor(value).unsqueeze(0)
            for key, value in observation.items()
        }
        encoded = extractor(batched)

        assert set(observation) == set(probe.observation_space.spaces)
        assert encoded.shape == (1, 2 * config.sequence_d_model + 128 + 2)
        assert torch.isfinite(encoded).all()
    finally:
        probe.close()


def test_u2_u1_policy_identity_binds_adapter_and_current_weight_location() -> None:
    probe = _u2_routed_u1_probe()
    try:
        _observation, _info = probe.reset(seed=0)
        _config, assembly = _assembly(probe)
        extractor = _u1_extractor(probe, assembly)
        model = SimpleNamespace(
            policy=SimpleNamespace(
                features_extractor=extractor,
                shared_actor_head=assembly.policy_actor_head,
                shared_actor_gate_temperature=assembly.hierarchical_gate_temperature,
                action_distribution_name="masked_shared_squashed_diag_gaussian_v1",
                log_std=SimpleNamespace(shape=(1,)),
                use_sde=False,
            )
        )

        identity = bind_sb3_policy_identity(model, assembly)

        assert assembly.sequence_metadata is not None
        adapter = identity["sequence_observation_adapter"]
        assert isinstance(adapter, dict)
        assert adapter == {
            "adapter_contract_digest": assembly.sequence_metadata[
                "adapter_contract_digest"
            ],
            "schema_version": "universal_trade_u1_sequence_adapter_identity_v1",
        }
        assert identity["sequence_observation_adapter_digest"] == content_digest(adapter)
        assert identity["current_weight_observation"] == {
            "bounds": (-1.0, 1.0),
            "dtype": "float32",
            "field": "current_weight",
            "field_index": UNIVERSAL_TRADE_POLICY_STATE_FIELDS.index("current_weight"),
            "key": "policy_state",
            "observation_schema": UNIVERSAL_TRADE_OBSERVATION_SCHEMA,
            "shape": (1,),
            "source": CURRENT_WEIGHT_SOURCE,
            "state_layout_schema": UNIVERSAL_TRADE_STATE_LAYOUT_SCHEMA,
        }
    finally:
        probe.close()
