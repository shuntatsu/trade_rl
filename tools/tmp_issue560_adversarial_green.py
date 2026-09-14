from __future__ import annotations

from pathlib import Path


def replace_once(path: str, old: str, new: str) -> None:
    target = Path(path)
    text = target.read_text(encoding="utf-8")
    if text.count(old) != 1:
        raise SystemExit(
            f"expected patch site not found exactly once: {path}: {old[:120]!r}"
        )
    target.write_text(text.replace(old, new), encoding="utf-8")


replace_once(
    "trade_rl/data/contracts.py",
    '''        if len(set(names)) != len(names):
            raise ValueError("feature names must be unique")
        if any(spec.timeframe == self.base_timeframe for spec in self.features):
''',
    '''        if len(set(names)) != len(names):
            raise ValueError("feature names must be unique")
        if (
            any(
                FeatureKind(spec.kind) is FeatureKind.SIGNED_TAKER_QUOTE_FLOW
                for spec in self.features
            )
            and self.base_timeframe != "1h"
        ):
            raise ValueError(
                "signed taker quote flow requires the 1h base decision clock"
            )
        if any(spec.timeframe == self.base_timeframe for spec in self.features):
''',
)

replace_once(
    "trade_rl/data/build/builder.py",
    '''    MarketBuildConfig,
    MarketCalendarKind,
)
''',
    '''    MarketBuildConfig,
    MarketCalendarKind,
    VolumeUnit,
)
''',
)
replace_once(
    "trade_rl/data/build/builder.py",
    '''                if (
                    spec.kind is FeatureKind.SIGNED_TAKER_QUOTE_FLOW
                    and native_timeframe != "1h"
                ):
                    raise ValueError(
                        "signed taker quote flow is defined only on the 1h clock"
                    )
                if native_timeframe == self.config.base_timeframe:
''',
    '''                if spec.kind is FeatureKind.SIGNED_TAKER_QUOTE_FLOW:
                    if native_timeframe != "1h":
                        raise ValueError(
                            "signed taker quote flow is defined only on the 1h clock"
                        )
                    if VolumeUnit(contract.volume_unit) is not VolumeUnit.QUOTE_NOTIONAL:
                        raise ValueError(
                            "signed taker quote flow requires quote-notional volume semantics"
                        )
                if native_timeframe == self.config.base_timeframe:
''',
)
replace_once(
    "trade_rl/data/build/builder.py",
    '''                    values, available, age_hours, staleness = _carry_feature(
                        event_values,
                        event_valid,
                        symbol_active[:, symbol_index],
                        timestamps,
                        max_staleness_hours=spec.max_staleness_hours,
                    )
''',
    '''                    if spec.kind is FeatureKind.SIGNED_TAKER_QUOTE_FLOW:
                        values = event_values
                        available = event_valid
                        age_hours = np.full(
                            n_bars, spec.max_staleness_hours, dtype=np.float64
                        )
                        staleness = np.ones(n_bars, dtype=np.float64)
                        age_hours[event_valid] = 0.0
                        staleness[event_valid] = 0.0
                    else:
                        values, available, age_hours, staleness = _carry_feature(
                            event_values,
                            event_valid,
                            symbol_active[:, symbol_index],
                            timestamps,
                            max_staleness_hours=spec.max_staleness_hours,
                        )
''',
)

test_path = Path("tests/data/test_signed_taker_flow_identity_compat.py")
text = test_path.read_text(encoding="utf-8")
if "import pytest\n" not in text:
    text = text.replace("import numpy as np\n", "import numpy as np\nimport pytest\n", 1)
if "    VolumeUnit,\n" not in text:
    text = text.replace(
        '''    MarketBuildConfig,
)
''',
        '''    MarketBuildConfig,
    VolumeUnit,
)
''',
        1,
    )
text = text.replace(
    "def _series(*, taker: np.ndarray | None) -> RawMarketSeries:\n",
    "def _series(*, taker: np.ndarray | None, tradable: np.ndarray | None = None) -> RawMarketSeries:\n",
    1,
)
text = text.replace(
    "        tradable=np.ones(n, dtype=np.bool_),\n",
    '''        tradable=(
            np.ones(n, dtype=np.bool_) if tradable is None else tradable
        ),
''',
    1,
)
addition = '''


def _signed_flow_builder() -> MarketDatasetBuilder:
    return MarketDatasetBuilder(
        MarketBuildConfig(
            base_timeframe="1h",
            features=(
                FeatureSpec(
                    name="1h__signed_taker_quote_flow_24bar",
                    kind=FeatureKind.SIGNED_TAKER_QUOTE_FLOW,
                    lookback=24,
                ),
            ),
        )
    )


def test_builder_materializes_signed_taker_flow_only_when_raw_field_exists() -> None:
    builder = _signed_flow_builder()
    instruments = (
        InstrumentContract(
            symbol="BTCUSDT",
            volume_unit=VolumeUnit.QUOTE_NOTIONAL,
        ),
    )

    enriched = builder.build(
        InMemoryMarketDataSource(
            {"BTCUSDT": _series(taker=np.full(30, 6.0, dtype=np.float64))}
        ),
        instruments,
    )
    missing = builder.build(
        InMemoryMarketDataSource({"BTCUSDT": _series(taker=None)}),
        instruments,
    )

    assert enriched.feature_available[23, 0, 0]
    assert enriched.features[23, 0, 0] == np.float32(0.2)
    assert not np.any(missing.feature_available[:, 0, 0])
    assert not np.any(missing.features[:, 0, 0])


def test_builder_does_not_carry_signed_flow_across_invalid_current_window() -> None:
    tradable = np.ones(30, dtype=np.bool_)
    tradable[24] = False
    dataset = _signed_flow_builder().build(
        InMemoryMarketDataSource(
            {
                "BTCUSDT": _series(
                    taker=np.full(30, 6.0, dtype=np.float64),
                    tradable=tradable,
                )
            }
        ),
        (
            InstrumentContract(
                symbol="BTCUSDT",
                volume_unit=VolumeUnit.QUOTE_NOTIONAL,
            ),
        ),
    )

    assert dataset.feature_available[23, 0, 0]
    assert not dataset.feature_available[24, 0, 0]
    assert dataset.features[24, 0, 0] == 0.0


def test_builder_rejects_non_quote_volume_semantics_for_signed_flow() -> None:
    with pytest.raises(ValueError, match="quote.*notional"):
        _signed_flow_builder().build(
            InMemoryMarketDataSource(
                {"BTCUSDT": _series(taker=np.full(30, 6.0, dtype=np.float64))}
            ),
            (InstrumentContract(symbol="BTCUSDT"),),
        )


def test_signed_flow_market_config_rejects_non_1h_base_clock() -> None:
    with pytest.raises(ValueError, match="1h base decision clock"):
        MarketBuildConfig(
            base_timeframe="15m",
            features=(
                FeatureSpec(
                    name="1h__signed_taker_quote_flow_24bar",
                    kind=FeatureKind.SIGNED_TAKER_QUOTE_FLOW,
                    lookback=24,
                    timeframe="1h",
                ),
            ),
        )
'''
if "test_builder_does_not_carry_signed_flow_across_invalid_current_window" in text:
    raise SystemExit("permanent adversarial tests already exist unexpectedly")
test_path.write_text(text + addition, encoding="utf-8")
