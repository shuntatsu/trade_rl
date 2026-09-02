from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from tests.rl.universal_trade_test_support import (
    make_runtime_snapshot,
    make_u1_feature_specs,
    make_u1_market,
)
from trade_rl.data.market import MarketDataset
from trade_rl.rl.universal_normalization import (
    build_universal_trade_sequence_normalizer,
)
from trade_rl.rl.universal_trade_contract import UniversalTradePolicyContract
from trade_rl.rl.universal_trade_observation import UniversalTradeObservationBuilder


def _reidentified(dataset: MarketDataset, **changes: object) -> MarketDataset:
    return replace(
        dataset,
        dataset_id="0" * 64,
        identity_payload_json=None,
        **changes,
    ).with_content_identity({"fixture": "universal_trade_u1_missing_value_v1"})


def _with_unavailable_placeholder(
    source: MarketDataset,
    *,
    decision_index: int,
    placeholder: float,
) -> MarketDataset:
    features = source.features.copy()
    features[decision_index, 0, 0] = placeholder
    available = source.feature_available.copy()
    available[decision_index, 0, 0] = False
    staleness_hours = source.feature_staleness_hours.copy()
    staleness_hours[decision_index, 0, 0] = 24.0
    staleness = source.feature_staleness.copy()
    staleness[decision_index, 0, 0] = 1.0
    return _reidentified(
        source,
        features=features,
        feature_available=available,
        feature_staleness_hours=staleness_hours,
        feature_staleness=staleness,
    )


def _builder(*, use_normalizer: bool) -> UniversalTradeObservationBuilder:
    contract = UniversalTradePolicyContract(feature_specs=make_u1_feature_specs())
    if not use_normalizer:
        return UniversalTradeObservationBuilder(contract=contract)

    btc = make_u1_market(symbol="BTCUSDT", n_bars=6200)
    eth = make_u1_market(symbol="ETHUSDT", n_bars=6200)
    cutoff = int(btc.timestamps[6000].astype("datetime64[ns]").astype(np.int64))
    normalizer = build_universal_trade_sequence_normalizer(
        symbol_datasets={"BTCUSDT": btc, "ETHUSDT": eth},
        contract=contract,
        source_dataset_digests=(("BTCUSDT", "b" * 64), ("ETHUSDT", "e" * 64)),
        knowledge_cutoff_ns=cutoff,
        universe_manifest_digest="a" * 64,
        provenance_digest="c" * 64,
    )
    return UniversalTradeObservationBuilder(contract=contract, normalizer=normalizer)


@pytest.mark.parametrize("use_normalizer", (False, True))
@pytest.mark.parametrize("placeholder", (-1e9, -1.0, 1.0, 1e9))
def test_unavailable_raw_placeholder_never_changes_policy_observation(
    *,
    use_normalizer: bool,
    placeholder: float,
) -> None:
    decision_index = 6000
    source = make_u1_market(n_bars=6200)
    baseline = _with_unavailable_placeholder(
        source,
        decision_index=decision_index,
        placeholder=0.0,
    )
    mutated = _with_unavailable_placeholder(
        source,
        decision_index=decision_index,
        placeholder=placeholder,
    )
    builder = _builder(use_normalizer=use_normalizer)
    runtime = make_runtime_snapshot()

    expected = builder.build(
        dataset=baseline,
        index=decision_index,
        runtime=runtime,
    )
    observed = builder.build(
        dataset=mutated,
        index=decision_index,
        runtime=runtime,
    )

    assert tuple(expected) == tuple(observed)
    for key in expected:
        np.testing.assert_array_equal(expected[key], observed[key], err_msg=key)
    assert observed["sequence_15m_available"][0, -1, 0] == 0
    assert observed["sequence_15m_values"][0, -1, 0] == pytest.approx(0.0)
