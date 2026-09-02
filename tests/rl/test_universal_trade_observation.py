from __future__ import annotations

import importlib
from dataclasses import replace

import numpy as np
import pytest

from tests.rl.universal_trade_test_support import (
    make_runtime_snapshot,
    make_u1_feature_specs,
    make_u1_market,
)
from trade_rl.data.contracts import FeatureKind, FeatureSpec
from trade_rl.data.market import MarketDataset
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


def _builder_type():
    module = _observation_module()
    builder_type = getattr(module, "UniversalTradeObservationBuilder", None)
    assert builder_type is not None, "Universal Trade U1 observation builder is missing"
    return builder_type


def _builder():
    return _builder_type()(
        contract=UniversalTradePolicyContract(feature_specs=make_u1_feature_specs())
    )


def _reidentified(dataset: MarketDataset, **changes: object) -> MarketDataset:
    return replace(
        dataset,
        dataset_id="0" * 64,
        identity_payload_json=None,
        **changes,
    ).with_content_identity({"fixture": "universal_trade_u1_observation_mutation_v1"})


def _assert_observations_equal(
    left: dict[str, np.ndarray],
    right: dict[str, np.ndarray],
) -> None:
    assert tuple(left) == tuple(right)
    for key in left:
        np.testing.assert_array_equal(left[key], right[key], err_msg=key)


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


def test_u1_observation_ignores_all_future_market_mutations() -> None:
    decision_index = 6000
    original = make_u1_market(n_bars=6200)

    features = original.features.copy()
    features[decision_index + 1 :] += 10_000.0
    open_price = original.open.copy()
    high = original.high.copy()
    low = original.low.copy()
    close = original.close.copy()
    volume = original.volume.copy()
    funding_rate = original.funding_rate.copy()
    mark_price = original.resolved_array("mark_price").copy()
    index_price = original.resolved_array("index_price").copy()
    for array in (open_price, high, low, close, volume, mark_price, index_price):
        array[decision_index + 1 :] *= 1_000.0
    funding_rate[decision_index + 1 :] += 0.25

    mutated = _reidentified(
        original,
        features=features,
        open=open_price,
        high=high,
        low=low,
        close=close,
        volume=volume,
        funding_rate=funding_rate,
        mark_price=mark_price,
        index_price=index_price,
    )
    runtime = make_runtime_snapshot()
    builder = _builder()

    before = builder.build(
        dataset=original,
        index=decision_index,
        runtime=runtime,
    )
    after = builder.build(
        dataset=mutated,
        index=decision_index,
        runtime=runtime,
    )
    _assert_observations_equal(before, after)


def test_u1_observation_is_invariant_to_symbol_rename() -> None:
    builder = _builder()
    runtime = make_runtime_snapshot()
    btc = builder.build(
        dataset=make_u1_market(symbol="BTCUSDT"),
        index=6000,
        runtime=runtime,
    )
    renamed = builder.build(
        dataset=make_u1_market(symbol="FOOUSDT"),
        index=6000,
        runtime=runtime,
    )
    _assert_observations_equal(btc, renamed)


def test_u1_observation_is_invariant_to_price_units() -> None:
    builder = _builder()
    runtime = make_runtime_snapshot()
    base = builder.build(
        dataset=make_u1_market(price_scale=1.0),
        index=6000,
        runtime=runtime,
    )
    rescaled = builder.build(
        dataset=make_u1_market(price_scale=1_000.0),
        index=6000,
        runtime=runtime,
    )

    assert tuple(base) == tuple(rescaled)
    for key in base:
        np.testing.assert_allclose(base[key], rescaled[key], atol=1e-7, rtol=0.0)


def test_u1_observation_distinguishes_true_zero_from_unavailable_zero() -> None:
    decision_index = 6000
    source = make_u1_market(n_bars=6200)
    features = source.features.copy()
    features[decision_index, 0, 0] = 0.0
    available = source.feature_available.copy()
    available[decision_index, 0, 0] = True
    true_zero = _reidentified(source, features=features, feature_available=available)

    unavailable_mask = available.copy()
    unavailable_mask[decision_index, 0, 0] = False
    staleness_hours = source.feature_staleness_hours.copy()
    staleness_hours[decision_index, 0, 0] = 24.0
    staleness = source.feature_staleness.copy()
    staleness[decision_index, 0, 0] = 1.0
    unavailable_zero = _reidentified(
        source,
        features=features,
        feature_available=unavailable_mask,
        feature_staleness_hours=staleness_hours,
        feature_staleness=staleness,
    )

    builder = _builder()
    runtime = make_runtime_snapshot()
    observed = builder.build(
        dataset=true_zero,
        index=decision_index,
        runtime=runtime,
    )
    missing = builder.build(
        dataset=unavailable_zero,
        index=decision_index,
        runtime=runtime,
    )

    assert observed["sequence_15m_values"][0, -1, 0] == pytest.approx(0.0)
    assert missing["sequence_15m_values"][0, -1, 0] == pytest.approx(0.0)
    assert observed["sequence_15m_available"][0, -1, 0] == 1
    assert missing["sequence_15m_available"][0, -1, 0] == 0


def test_u1_observation_schema_digest_binds_feature_order_not_identity() -> None:
    specs = make_u1_feature_specs()
    extra = FeatureSpec(
        name="15m__ret_2",
        kind=FeatureKind.LOG_RETURN,
        lookback=2,
    )
    first = _builder_type()(
        contract=UniversalTradePolicyContract(
            feature_specs=(specs[0], extra, specs[1], specs[2], specs[3])
        )
    )
    reordered = _builder_type()(
        contract=UniversalTradePolicyContract(
            feature_specs=(extra, specs[0], specs[1], specs[2], specs[3])
        )
    )

    assert len(first.schema_digest) == 64
    assert len(first.state_layout_digest) == 64
    assert first.schema_digest != reordered.schema_digest
    assert first.state_layout_digest == reordered.state_layout_digest

    identity_builder = _builder()
    digest_before = identity_builder.schema_digest
    runtime = make_runtime_snapshot()
    identity_builder.build(
        dataset=make_u1_market(symbol="BTCUSDT"),
        index=6000,
        runtime=runtime,
    )
    identity_builder.build(
        dataset=make_u1_market(symbol="FOOUSDT"),
        index=6000,
        runtime=runtime,
    )
    assert identity_builder.schema_digest == digest_before


def test_u1_observation_rejects_normalizer_for_different_contract() -> None:
    specs = make_u1_feature_specs()
    fitted_contract = UniversalTradePolicyContract(
        feature_specs=(
            replace(specs[0], lookback=2),
            specs[1],
            specs[2],
            specs[3],
        )
    )
    btc = make_u1_market(symbol="BTCUSDT", n_bars=6200)
    eth = make_u1_market(symbol="ETHUSDT", n_bars=6200)
    cutoff = int(btc.timestamps[6000].astype("datetime64[ns]").astype(np.int64))
    normalizer = build_universal_trade_sequence_normalizer(
        symbol_datasets={"BTCUSDT": btc, "ETHUSDT": eth},
        contract=fitted_contract,
        source_dataset_digests=(("BTCUSDT", "b" * 64), ("ETHUSDT", "e" * 64)),
        knowledge_cutoff_ns=cutoff,
        universe_manifest_digest="a" * 64,
        provenance_digest="c" * 64,
    )

    expected_contract = UniversalTradePolicyContract(feature_specs=specs)
    with pytest.raises(ValueError, match="normalizer|contract"):
        _builder_type()(contract=expected_contract, normalizer=normalizer)


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

    builder_type = _builder_type()
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
