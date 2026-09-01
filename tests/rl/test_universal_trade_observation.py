from __future__ import annotations

import importlib

import numpy as np
import pytest

from tests.rl.universal_trade_test_support import (
    make_runtime_snapshot,
    make_u1_feature_specs,
    make_u1_market,
)
from trade_rl.rl.universal_trade_contract import UniversalTradePolicyContract

_EXPECTED_KEYS = (
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
)
_FORBIDDEN_STATE_TOKENS = (
    "trend",
    "alpha",
    "shadow",
    "baseline",
    "remaining",
    "symbol",
    "dataset",
)


def _observation_module():
    try:
        return importlib.import_module("trade_rl.rl.universal_trade_observation")
    except ModuleNotFoundError:
        pytest.fail("Universal Trade U1 observation module is not implemented")


def test_u1_observation_has_exact_strategy_prior_free_layout() -> None:
    module = _observation_module()
    builder_type = getattr(module, "UniversalTradeObservationBuilder", None)
    assert builder_type is not None, "Universal Trade U1 observation builder is missing"

    dataset = make_u1_market()
    builder = builder_type(
        contract=UniversalTradePolicyContract(feature_specs=make_u1_feature_specs())
    )
    observation = builder.build(
        dataset=dataset,
        index=6000,
        runtime=make_runtime_snapshot(),
    )

    assert tuple(observation) == _EXPECTED_KEYS
    assert builder.observation_space.contains(observation)
    assert observation["policy_state"].dtype == np.float32

    source_indices = builder.source_indices(dataset=dataset, index=6000)
    assert tuple(source_indices) == ("15m", "1h", "4h", "1d")
    assert all(np.all(indices <= 6000) for indices in source_indices.values())

    state_fields = tuple(builder.policy_state_fields)
    assert state_fields
    assert all(
        token not in field.lower()
        for field in state_fields
        for token in _FORBIDDEN_STATE_TOKENS
    )
