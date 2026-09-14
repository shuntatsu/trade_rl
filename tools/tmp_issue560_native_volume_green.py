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
    "trade_rl/data/features/multitimeframe.py",
    '''    FeatureKind,
    FeatureSpec,
    InstrumentContract,
    timeframe_hours,
)
''',
    '''    FeatureKind,
    FeatureSpec,
    InstrumentContract,
    VolumeUnit,
    timeframe_hours,
)
''',
)
replace_once(
    "trade_rl/data/features/multitimeframe.py",
    '''    if spec.kind is FeatureKind.SIGNED_TAKER_QUOTE_FLOW and timeframe != "1h":
        raise ValueError("signed taker quote flow is defined only on the 1h clock")
    event_values, event_valid, event_available_at = _native_events(spec, raw, contract)
''',
    '''    if spec.kind is FeatureKind.SIGNED_TAKER_QUOTE_FLOW:
        if timeframe != "1h":
            raise ValueError("signed taker quote flow is defined only on the 1h clock")
        if VolumeUnit(contract.volume_unit) is not VolumeUnit.QUOTE_NOTIONAL:
            raise ValueError(
                "signed taker quote flow requires quote-notional volume semantics"
            )
    event_values, event_valid, event_available_at = _native_events(spec, raw, contract)
''',
)

path = Path("tests/data/test_signed_taker_flow_feature.py")
text = path.read_text(encoding="utf-8")
text = text.replace(
    '''    InstrumentContract,
    NormalizationMode,
)
''',
    '''    InstrumentContract,
    NormalizationMode,
    VolumeUnit,
)
''',
    1,
)
text = text.replace(
    '''        InstrumentContract(symbol="BTCUSDT"),
        timestamps,
''',
    '''        InstrumentContract(
            symbol="BTCUSDT",
            volume_unit=VolumeUnit.QUOTE_NOTIONAL,
        ),
        timestamps,
''',
    1,
)
addition = '''


def test_signed_taker_flow_native_alignment_rejects_non_quote_volume_semantics() -> None:
    n = 30
    timestamps = np.datetime64("2022-01-01T00:00:00", "ns") + np.arange(
        n
    ) * np.timedelta64(1, "h")
    raw = _raw(taker=np.full(n, 6.0, dtype=np.float64))

    with pytest.raises(ValueError, match="quote.*notional"):
        align_native_feature(
            _spec(),
            raw,
            InstrumentContract(symbol="BTCUSDT"),
            timestamps,
            np.ones(n, dtype=np.bool_),
            timeframe="1h",
        )
'''
if "test_signed_taker_flow_native_alignment_rejects_non_quote_volume_semantics" in text:
    raise SystemExit("native volume semantics regression test already exists")
path.write_text(text + addition, encoding="utf-8")
