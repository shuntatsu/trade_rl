from __future__ import annotations

from dataclasses import replace

import pytest

from trade_rl.evaluation.experiments.ppo_global_btc_regime_prereg import (
    canonical_ppo_global_btc_regime_protocol,
)


def test_protocol_freezes_existing_controlled_factor_category() -> None:
    protocol = canonical_ppo_global_btc_regime_protocol()

    assert protocol.experiment_controlled_factor == "FEATURE_SET"
    assert protocol.baseline_observation_schema == "ppo_observation_v2"
    assert protocol.baseline_global_feature_names == ()
    assert protocol.local_observation_contract_changed is False


def test_protocol_freezes_exact_global_reference_channels() -> None:
    protocol = canonical_ppo_global_btc_regime_protocol()

    assert protocol.reference_observation_components == (
        "value",
        "available_and_finite",
        "normalized_staleness",
    )
    assert protocol.reference_observation_width == 3
    assert protocol.reference_value_source == "canonical_dataset.features"
    assert protocol.reference_available_source == "canonical_dataset.feature_available"
    assert protocol.reference_staleness_source == "canonical_dataset.feature_staleness"
    assert protocol.reference_usable_semantics == "feature_available_and_isfinite"
    assert protocol.reference_unavailable_value == 0.0
    assert protocol.reference_staleness_recomputed is False


def test_protocol_rejects_global_observation_contract_drift() -> None:
    protocol = canonical_ppo_global_btc_regime_protocol()
    mutations: tuple[dict[str, object], ...] = (
        {"experiment_controlled_factor": "PPO_TRAINING_BUDGET"},
        {"baseline_observation_schema": "ppo_observation_v3"},
        {"baseline_global_feature_names": ("existing_global",)},
        {"local_observation_contract_changed": True},
        {"reference_observation_components": ("value",)},
        {"reference_observation_width": 4},
        {"reference_value_source": "recomputed_returns"},
        {"reference_available_source": "custom_mask"},
        {"reference_staleness_source": "recomputed_age"},
        {"reference_usable_semantics": "finite_only"},
        {"reference_unavailable_value": 1.0},
        {"reference_staleness_recomputed": True},
    )
    for mutation in mutations:
        with pytest.raises(ValueError, match="preregistered"):
            replace(protocol, **mutation)
