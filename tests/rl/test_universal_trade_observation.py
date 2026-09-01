from __future__ import annotations

import importlib

import numpy as np
import pytest

from tests.rl.universal_trade_test_support import (
    make_runtime_snapshot,
    make_u1_feature_specs,
    make_u1_market,
)
from trade_rl.rl.universal_normalization import (
    build_universal_trade_sequence_normalizer,
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


def _builder():
    module = _observation_module()
    builder_type = getattr(module, "UniversalTradeObservationBuilder", None)
    assert builder_type is not None, "Universal Trade U1 observation builder is missing"
    return builder_type(
        contract=UniversalTradePolicyContract(feature_specs=make_u1_feature_specs())
    )


def test_u1_observation_has_exact_strategy_prior_free_layout() -> None:
    dataset = make_u1_market()
    builder = _builder()
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


def test_u1_policy_state_uses_fixed_dimensionless_transforms() -> None:
    builder = _builder()
    observation = builder.build(
        dataset=make_u1_market(),
        index=6000,
        runtime=make_runtime_snapshot(
            position_age_hours=24.0,
            pending_order_age_hours=48.0,
            pending_order_eligible_delay_hours=24.0,
            pending_order_expiry_distance_hours=72.0,
            mark_index_basis=0.01,
            borrow_rate=0.02,
        ),
    )
    state = dict(
        zip(builder.policy_state_fields, observation["policy_state"], strict=True)
    )

    assert state["position_age_days"] == pytest.approx(np.log1p(1.0))
    assert state["pending_order_age_days"] == pytest.approx(np.log1p(2.0))
    assert state["pending_order_eligible_delay_days"] == pytest.approx(np.log1p(1.0))
    assert state["pending_order_expiry_distance_days"] == pytest.approx(np.log1p(3.0))
    assert state["mark_index_basis"] == pytest.approx(np.tanh(1.0))
    assert state["borrow_rate"] == pytest.approx(np.tanh(0.02))


def test_u1_observation_normalizer_transforms_sequence_values_only() -> None:
    contract = UniversalTradePolicyContract(feature_specs=make_u1_feature_specs())
    btc = make_u1_market(
        symbol="BTCUSDT",
        n_bars=6200,
        feature_level=1.0,
    )
    eth = make_u1_market(
        symbol="ETHUSDT",
        n_bars=6200,
        feature_level=2.0,
    )
    cutoff = int(btc.timestamps[6000].astype("datetime64[ns]").astype(np.int64))
    normalizer = build_universal_trade_sequence_normalizer(
        symbol_datasets={"BTCUSDT": btc, "ETHUSDT": eth},
        contract=contract,
        source_dataset_digests=(
            ("BTCUSDT", "b" * 64),
            ("ETHUSDT", "e" * 64),
        ),
        knowledge_cutoff_ns=cutoff,
        universe_manifest_digest="a" * 64,
        provenance_digest="c" * 64,
    )

    module = _observation_module()
    builder_type = getattr(module, "UniversalTradeObservationBuilder", None)
    assert builder_type is not None, "Universal Trade U1 observation builder is missing"
    raw_builder = builder_type(contract=contract)
    normalized_builder = builder_type(contract=contract, normalizer=normalizer)
    runtime = make_runtime_snapshot()
    raw = raw_builder.build(dataset=btc, index=6000, runtime=runtime)
    normalized = normalized_builder.build(dataset=btc, index=6000, runtime=runtime)

    changed_values = False
    for timeframe in ("15m", "1h", "4h", "1d"):
        values_key = f"sequence_{timeframe}_values"
        available_key = f"sequence_{timeframe}_available"
        staleness_key = f"sequence_{timeframe}_staleness"
        feature_names = tuple(
            spec.name
            for spec in contract.feature_specs
            if spec.resolved_timeframe("15m") == timeframe
        )
        expected = normalizer.transform(
            timeframe,
            raw[values_key],
            raw[available_key].astype(np.bool_),
            feature_names=feature_names,
        )

        np.testing.assert_allclose(
            normalized[values_key],
            expected,
            atol=1e-7,
            rtol=0.0,
        )
        np.testing.assert_array_equal(normalized[available_key], raw[available_key])
        np.testing.assert_array_equal(normalized[staleness_key], raw[staleness_key])
        changed_values |= not np.array_equal(normalized[values_key], raw[values_key])

    assert changed_values
    np.testing.assert_array_equal(normalized["policy_state"], raw["policy_state"])
