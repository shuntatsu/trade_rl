from __future__ import annotations

from datetime import datetime, timezone

import pytest

from trade_rl.data.contracts import (
    FeatureKind,
    FeatureSpec,
    InstrumentContract,
    MarketBuildConfig,
    NormalizationMode,
    VolumeUnit,
)


def test_instrument_contract_validates_lifetime_and_multiplier() -> None:
    listed = datetime(2026, 1, 1, tzinfo=timezone.utc)
    delisted = datetime(2026, 2, 1, tzinfo=timezone.utc)
    contract = InstrumentContract(
        symbol="BTCUSDT",
        listed_at=listed,
        delisted_at=delisted,
        volume_unit=VolumeUnit.CONTRACTS,
        contract_multiplier=0.001,
    )

    assert contract.symbol == "BTCUSDT"
    assert contract.canonical_payload()["volume_unit"] == "contracts"

    with pytest.raises(ValueError, match="delisted_at"):
        InstrumentContract(
            symbol="BTCUSDT",
            listed_at=listed,
            delisted_at=listed,
        )
    with pytest.raises(ValueError, match="contract_multiplier"):
        InstrumentContract(symbol="BTCUSDT", contract_multiplier=0.0)


def test_feature_spec_validates_causal_configuration() -> None:
    spec = FeatureSpec(
        name="ret_4",
        kind=FeatureKind.LOG_RETURN,
        lookback=4,
        normalization=NormalizationMode.ROLLING_ZSCORE,
        normalization_window=48,
        min_periods=12,
        max_staleness_hours=8.0,
    )

    assert spec.canonical_payload()["normalization"] == "rolling_zscore"

    with pytest.raises(ValueError, match="lookback"):
        FeatureSpec(name="bad", kind=FeatureKind.LOG_RETURN, lookback=0)
    with pytest.raises(ValueError, match="normalization_window"):
        FeatureSpec(
            name="bad",
            kind=FeatureKind.LOG_RETURN,
            normalization=NormalizationMode.ROLLING_ZSCORE,
            normalization_window=0,
        )
    with pytest.raises(ValueError, match="max_staleness_hours"):
        FeatureSpec(
            name="bad",
            kind=FeatureKind.FUNDING_BPS,
            max_staleness_hours=0.0,
        )


def test_market_build_config_requires_unique_feature_names() -> None:
    spec = FeatureSpec(name="ret", kind=FeatureKind.LOG_RETURN)
    with pytest.raises(ValueError, match="feature names"):
        MarketBuildConfig(base_timeframe="1h", features=(spec, spec))
