from __future__ import annotations

import numpy as np
import pytest

from tests.rl.universal_trade_test_support import make_u1_feature_specs, make_u1_market
from trade_rl.rl.universal_normalization import (
    build_universal_trade_sequence_normalizer,
)
from trade_rl.rl.universal_trade_contract import UniversalTradePolicyContract


def _normalizer():
    btc = make_u1_market(symbol="BTCUSDT", n_bars=5800, feature_level=1.0)
    eth = make_u1_market(symbol="ETHUSDT", n_bars=5800, feature_level=2.0)
    cutoff = min(
        int(btc.timestamps[-1].astype("datetime64[ns]").astype(np.int64)),
        int(eth.timestamps[-1].astype("datetime64[ns]").astype(np.int64)),
    )
    return build_universal_trade_sequence_normalizer(
        symbol_datasets={"BTCUSDT": btc, "ETHUSDT": eth},
        contract=UniversalTradePolicyContract(feature_specs=make_u1_feature_specs()),
        source_dataset_digests=(
            ("BTCUSDT", "b" * 64),
            ("ETHUSDT", "e" * 64),
        ),
        knowledge_cutoff_ns=cutoff,
        universe_manifest_digest="a" * 64,
        provenance_digest="f" * 64,
    )


def test_u1_normalizer_accepts_native_sequence_leading_axes() -> None:
    normalizer = _normalizer()
    values = np.asarray([[[1.0], [1.1], [1.2]]], dtype=np.float32)
    available = np.asarray([[[True], [False], [True]]], dtype=np.bool_)

    transformed = normalizer.transform(
        "15m",
        values,
        available,
        feature_names=("15m__ret",),
    )

    assert transformed.shape == values.shape
    assert transformed.dtype == np.float32
    assert transformed[0, 1, 0] == pytest.approx(0.0)
    assert np.isfinite(transformed).all()
