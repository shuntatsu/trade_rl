from pathlib import Path


def replace_once(path: str, old: str, new: str) -> None:
    source = Path(path)
    text = source.read_text(encoding="utf-8")
    if text.count(old) != 1:
        raise RuntimeError(f"{path}: expected exactly one patch site")
    source.write_text(text.replace(old, new, 1), encoding="utf-8")


module = "trade_rl/integrations/binance/dataset.py"
test = "tests/integrations/test_binance.py"

replace_once(
    module,
    '''    timestamps = np.asarray([item[0] for item in parsed], dtype=np.int64)
    if np.any(np.diff(timestamps) <= 0):
        raise ValueError("Binance kline timestamps must be strictly increasing")
    if np.any(np.diff(timestamps) != interval_ms):
        raise ValueError("Binance kline range must be complete and exactly regular")
''',
    '''    timestamps = np.asarray([item[0] for item in parsed], dtype=np.int64)
    deltas = np.diff(timestamps)
    if np.any(deltas <= 0):
        raise ValueError("Binance kline timestamps must be strictly increasing")
    expected_close_grid_origin = start_ms + interval_ms
    if np.any((timestamps - expected_close_grid_origin) % interval_ms != 0):
        raise ValueError("Binance kline timestamps must align to the requested interval grid")
    if np.any(deltas % interval_ms != 0):
        raise ValueError("Binance kline timestamp gaps must be integer interval multiples")
''',
)
replace_once(
    module,
    '''def _align_funding(
    timestamps: np.ndarray,
    events: Sequence[tuple[int, float]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Aggregate every funding event into its completed native bar."""

    timestamp_ms = timestamps.astype("datetime64[ms]").astype(np.int64)
    if timestamp_ms.size < 2:
        raise ValueError("funding alignment requires at least two native bars")
    intervals = np.diff(timestamp_ms)
    if np.any(intervals <= 0) or np.any(intervals != intervals[0]):
        raise ValueError("funding alignment requires a regular native clock")
    interval_ms = int(intervals[0])
''',
    '''def _align_funding(
    timestamps: np.ndarray,
    events: Sequence[tuple[int, float]],
    *,
    interval_ms: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Aggregate funding only into observed bars on the nominal native grid."""

    if isinstance(interval_ms, bool) or not isinstance(interval_ms, int) or interval_ms <= 0:
        raise ValueError("funding interval_ms must be a positive integer")
    timestamp_ms = timestamps.astype("datetime64[ms]").astype(np.int64)
    if timestamp_ms.size < 2:
        raise ValueError("funding alignment requires at least two native bars")
    intervals = np.diff(timestamp_ms)
    if np.any(intervals <= 0):
        raise ValueError("funding alignment requires strictly increasing native bars")
    if np.any(intervals % interval_ms != 0):
        raise ValueError("funding alignment timestamps must remain on the nominal grid")
''',
)
replace_once(
    module,
    '''        funding, funding_available, funding_event_count = _align_funding(
            timestamps,
            self._funding_events(symbol),
        )
''',
    '''        funding, funding_available, funding_event_count = _align_funding(
            timestamps,
            self._funding_events(symbol),
            interval_ms=interval_ms,
        )
''',
)
replace_once(
    test,
    '''from trade_rl.data.contracts import VolumeUnit
''',
    '''from trade_rl.data.build import MarketDatasetBuilder
from trade_rl.data.contracts import (
    FeatureKind,
    FeatureSpec,
    InstrumentContract,
    MarketBuildConfig,
    VolumeUnit,
)
''',
)

path = Path(test)
text = path.read_text(encoding="utf-8")
marker = "class SparseKlineTransport"
if marker in text:
    raise RuntimeError("native-gap regression already exists")
append = '''


class SparseKlineTransport:
    def __init__(self) -> None:
        self.start = datetime(2022, 2, 1, tzinfo=UTC)
        self.rows = [
            _kline(self.start, quote_volume=1_000.0, close=100.0),
            _kline(
                self.start + timedelta(hours=2),
                quote_volume=1_200.0,
                close=102.0,
            ),
        ]

    def load_klines(self, **_: object) -> tuple[list[list[Any]], str]:
        return self.rows, "fixture:sparse-klines"

    def load_funding_rates(self, **_: object) -> tuple[list[tuple[int, float]], str]:
        return [
            (_ms(self.start + timedelta(hours=1)), 0.0001),
            (_ms(self.start + timedelta(hours=2)), 0.0099),
            (_ms(self.start + timedelta(hours=3)), -0.0002),
        ], "fixture:sparse-funding"


def test_source_preserves_native_kline_gaps_and_does_not_shift_funding() -> None:
    start = datetime(2022, 2, 1, tzinfo=UTC)
    source = BinanceMarketDataSource(
        market="usds-m",
        interval="1h",
        start_time=start,
        end_time=start + timedelta(hours=3),
        transport=SparseKlineTransport(),
    )

    series = source.load("XRPUSDT")

    np.testing.assert_array_equal(
        series.timestamps,
        np.array(
            ["2022-02-01T01:00:00", "2022-02-01T03:00:00"],
            dtype="datetime64[ns]",
        ),
    )
    np.testing.assert_allclose(series.funding_rate, [0.0001, -0.0002])
    np.testing.assert_array_equal(series.funding_event_count, [1, 1])


def test_builder_marks_native_kline_gap_unavailable_without_imputation() -> None:
    start = datetime(2022, 2, 1, tzinfo=UTC)
    source = BinanceMarketDataSource(
        market="usds-m",
        interval="1h",
        start_time=start,
        end_time=start + timedelta(hours=3),
        transport=SparseKlineTransport(),
    )
    dataset = MarketDatasetBuilder(
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
    ).build(
        source,
        (
            InstrumentContract(
                symbol="XRPUSDT",
                volume_unit=VolumeUnit.QUOTE_NOTIONAL,
            ),
        ),
    )

    np.testing.assert_array_equal(
        dataset.timestamps,
        np.array(
            [
                "2022-02-01T01:00:00",
                "2022-02-01T02:00:00",
                "2022-02-01T03:00:00",
            ],
            dtype="datetime64[ns]",
        ),
    )
    np.testing.assert_array_equal(dataset.tradable[:, 0], [True, False, True])
    assert dataset.information_available is not None
    np.testing.assert_array_equal(
        dataset.information_available[:, 0],
        [True, False, True],
    )
    assert not np.any(dataset.feature_available[:, 0, 0])
'''
path.write_text(text + append, encoding="utf-8")
