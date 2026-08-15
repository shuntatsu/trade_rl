from __future__ import annotations

import copy

import numpy as np
import pytest

from trade_rl.learning.episode_oracle_teacher import OracleEpisodeContract
from trade_rl.workflows.universal_causal_alpha_contracts import (
    CausalAlphaEpisodePartition,
)
from trade_rl.workflows.universal_causal_alpha_v3_config import (
    CausalAlphaV3ResearchConfig,
)
from trade_rl.workflows.universal_causal_alpha_v3_signal import (
    split_causal_alpha_v3_partitions,
)


def _config_payload() -> dict[str, object]:
    return {
        "schema_version": "universal_causal_alpha_v3_research_config_v1",
        "nested_selection": {
            "signal_contract_count": 2,
            "minimum_economic_contract_count": 1,
        },
        "signal_gate": {
            "minimum_scope_count": 2,
            "minimum_scope_coverage": 1.0,
            "minimum_rank_ic_lower_ci": 0.0,
            "minimum_top_bottom_spread_lower_ci": 0.0,
            "minimum_direction_accuracy_excess_lower_ci": 0.0,
            "bootstrap_resamples": 100,
            "bootstrap_seed": 7,
            "bootstrap_block_size": 1,
        },
        "selection_gate": {
            "minimum_mean_gross_return": 0.0,
            "minimum_mean_net_return": 0.0,
            "minimum_symbol_episode_net_return": -0.05,
            "maximum_mean_turnover_per_day": 1.0,
            "maximum_unexplained_execution_rejections": 0,
            "minimum_positive_gross_episode_fraction": 0.5,
        },
        "candidates": [
            {
                "name": "baseline",
                "fit": {"ridge_strength": 0.1},
                "target": {
                    "target_magnitudes": [0.0, 0.025, 0.05, 0.1, 0.25],
                    "uncertainty_multiplier": 1.0,
                    "execution_cost_multiplier": 1.5,
                    "edge_margin": 0.001,
                    "alpha_rebalance_decisions": 4,
                    "strong_reversal_threshold": 0.02,
                    "max_target_delta": 0.125,
                },
            },
            {
                "name": "uncertainty-high",
                "fit": {"ridge_strength": 0.1},
                "target": {
                    "target_magnitudes": [0.0, 0.025, 0.05, 0.1, 0.25],
                    "uncertainty_multiplier": 1.5,
                    "execution_cost_multiplier": 1.5,
                    "edge_margin": 0.001,
                    "alpha_rebalance_decisions": 4,
                    "strong_reversal_threshold": 0.02,
                    "max_target_delta": 0.125,
                },
            },
        ],
    }


def _contract(dataset_id: str, episode: int, start: int) -> OracleEpisodeContract:
    return OracleEpisodeContract(
        dataset_id=dataset_id,
        episode_index=episode,
        start=start,
        stop=start + 11,
        initial_state_mode="cash",
        initial_weights=np.zeros(1, dtype=np.float64),
    )


def test_research_config_is_strict_and_semantically_unique() -> None:
    config = CausalAlphaV3ResearchConfig.from_mapping(_config_payload())

    assert config.nested_selection.signal_contract_count == 2
    assert len(config.candidates) == 2
    assert config.candidates[0].fit.digest == config.candidates[1].fit.digest
    assert config.candidates[0].semantic_digest != config.candidates[1].semantic_digest

    unknown = copy.deepcopy(_config_payload())
    unknown["surprise"] = True
    with pytest.raises(ValueError, match="field"):
        CausalAlphaV3ResearchConfig.from_mapping(unknown)

    duplicate = copy.deepcopy(_config_payload())
    duplicate["candidates"] = [
        duplicate["candidates"][0],
        {**duplicate["candidates"][0], "name": "same-semantics"},
    ]
    with pytest.raises(ValueError, match="semantic"):
        CausalAlphaV3ResearchConfig.from_mapping(duplicate)


def test_nested_partition_keeps_signal_selection_and_holdout_disjoint() -> None:
    dataset_id = "1" * 64
    contracts = tuple(_contract(dataset_id, index, index * 20) for index in range(4))
    partition = CausalAlphaEpisodePartition(
        contracts=contracts,
        selection_contracts=contracts[:-1],
        holdout_contract=contracts[-1],
        train_start=0,
        train_stop=contracts[-1].stop,
    )

    nested = split_causal_alpha_v3_partitions(
        {"BTCUSDT": partition},
        train_symbols=("BTCUSDT",),
        signal_contract_count=2,
        minimum_economic_contract_count=1,
    )["BTCUSDT"]

    assert tuple(item.episode_index for item in nested.signal_contracts) == (0, 1)
    assert tuple(item.episode_index for item in nested.economic_contracts) == (2,)
    assert nested.holdout_contract.episode_index == 3
    assert set(nested.signal_contract_digests).isdisjoint(
        nested.economic_contract_digests
    )
    assert nested.holdout_contract.digest not in set(nested.signal_contract_digests)
    assert nested.holdout_contract.digest not in set(nested.economic_contract_digests)


def test_nested_partition_rejects_insufficient_economic_scope() -> None:
    dataset_id = "2" * 64
    contracts = tuple(_contract(dataset_id, index, index * 20) for index in range(3))
    partition = CausalAlphaEpisodePartition(
        contracts=contracts,
        selection_contracts=contracts[:-1],
        holdout_contract=contracts[-1],
        train_start=0,
        train_stop=contracts[-1].stop,
    )

    with pytest.raises(ValueError, match="economic"):
        split_causal_alpha_v3_partitions(
            {"BTCUSDT": partition},
            train_symbols=("BTCUSDT",),
            signal_contract_count=2,
            minimum_economic_contract_count=1,
        )
