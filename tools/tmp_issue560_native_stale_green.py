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
    '''    valid_indices = np.flatnonzero(event_valid)
    if valid_indices.size == 0:
        return values, available, age_hours, staleness

    event_time_ns = raw.timestamps.astype("datetime64[ns]").astype(np.int64)
    availability_ns = event_available_at.astype("datetime64[ns]").astype(np.int64)
    order = valid_indices[np.argsort(availability_ns[valid_indices], kind="stable")]
    base_ns = base_timestamps.astype("datetime64[ns]").astype(np.int64)

    cursor = 0
''',
    '''    valid_indices = np.flatnonzero(event_valid)
    if valid_indices.size == 0:
        return values, available, age_hours, staleness

    event_time_ns = raw.timestamps.astype("datetime64[ns]").astype(np.int64)
    availability_ns = event_available_at.astype("datetime64[ns]").astype(np.int64)
    base_ns = base_timestamps.astype("datetime64[ns]").astype(np.int64)

    if spec.kind is FeatureKind.SIGNED_TAKER_QUOTE_FLOW:
        for base_index, timestamp_ns in enumerate(base_ns):
            if not base_active[base_index]:
                continue
            event_index = int(np.searchsorted(event_time_ns, timestamp_ns, side="left"))
            if (
                event_index >= len(event_time_ns)
                or event_time_ns[event_index] != timestamp_ns
                or not event_valid[event_index]
                or availability_ns[event_index] > timestamp_ns
            ):
                continue
            values[base_index] = event_values[event_index]
            available[base_index] = True
            age_hours[base_index] = 0.0
            staleness[base_index] = 0.0
        return values, available, age_hours, staleness

    order = valid_indices[np.argsort(availability_ns[valid_indices], kind="stable")]
    cursor = 0
''',
)

path = Path("tests/data/test_signed_taker_flow_feature.py")
text = path.read_text(encoding="utf-8")
addition = '''


def test_signed_taker_flow_native_alignment_does_not_carry_invalid_window() -> None:
    n = 30
    timestamps = np.datetime64("2022-01-01T00:00:00", "ns") + np.arange(
        n
    ) * np.timedelta64(1, "h")
    tradable = np.ones(n, dtype=np.bool_)
    tradable[24] = False
    raw = _raw(taker=np.full(n, 6.0, dtype=np.float64))
    raw = RawMarketSeries(
        timestamps=raw.timestamps,
        available_at=raw.available_at,
        open=raw.open,
        high=raw.high,
        low=raw.low,
        close=raw.close,
        volume=raw.volume,
        funding_rate=raw.funding_rate,
        funding_available=raw.funding_available,
        funding_event_count=raw.funding_event_count,
        tradable=tradable,
        taker_buy_quote_volume=raw.taker_buy_quote_volume,
    )

    values, available, _, _ = align_native_feature(
        _spec(),
        raw,
        InstrumentContract(
            symbol="BTCUSDT",
            volume_unit=VolumeUnit.QUOTE_NOTIONAL,
        ),
        timestamps,
        np.ones(n, dtype=np.bool_),
        timeframe="1h",
    )

    assert available[23]
    assert not available[24]
    assert values[24] == 0.0
'''
if "test_signed_taker_flow_native_alignment_does_not_carry_invalid_window" in text:
    raise SystemExit("native stale-carry regression test already exists")
path.write_text(text + addition, encoding="utf-8")
