from __future__ import annotations

from tests.rl.universal_trade_test_support import make_u1_market, make_u1_wrapper
from trade_rl.artifacts.hashing import content_digest
from trade_rl.integrations.sb3_model_assembly import resolve_sb3_policy_assembly
from trade_rl.rl.algorithm_configs import build_algorithm_config
from trade_rl.rl.universal_instrument_binding import InstrumentDatasetBinding
from trade_rl.rl.universal_single_instrument_env import EpisodeRoutedSingleInstrumentEnv
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
