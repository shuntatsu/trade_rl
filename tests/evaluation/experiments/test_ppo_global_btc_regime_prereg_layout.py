from __future__ import annotations

from dataclasses import replace

import pytest

from trade_rl.evaluation.experiments.ppo_global_btc_regime_prereg import (
    canonical_ppo_global_btc_regime_protocol,
)


def test_protocol_freezes_candidate_observation_layout() -> None:
    protocol = canonical_ppo_global_btc_regime_protocol()

    assert (
        protocol.candidate_observation_schema == "ppo_observation_v3_global_btc_regime"
    )
    assert protocol.candidate_observation_layout == (
        "local_values",
        "local_available",
        "local_staleness",
        "global_reference_value",
        "global_reference_available_and_finite",
        "global_reference_normalized_staleness",
        "current_intent",
        "current_weight",
    )
    assert protocol.global_reference_applies_to_all_ppo_symbols is True
    assert protocol.global_reference_timestamp_alignment == "same_dataset_row"
    assert protocol.global_reference_broadcast_is_symbol_independent is True


def test_protocol_rejects_candidate_layout_drift() -> None:
    protocol = canonical_ppo_global_btc_regime_protocol()
    mutations: tuple[dict[str, object], ...] = (
        {"candidate_observation_schema": "ppo_observation_v2"},
        {"candidate_observation_layout": protocol.candidate_observation_layout[::-1]},
        {"global_reference_applies_to_all_ppo_symbols": False},
        {"global_reference_timestamp_alignment": "previous_dataset_row"},
        {"global_reference_broadcast_is_symbol_independent": False},
    )
    for mutation in mutations:
        with pytest.raises(ValueError, match="preregistered"):
            replace(protocol, **mutation)
