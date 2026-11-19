"""Result-blind synthetic adversarial tests for opt-in source clock admission.

These must never access real Binance data, historical Dataset IDs or orders.
"""

from __future__ import annotations

from datetime import timedelta

import pytest

from tests.integrations.test_binance_source_close_policy import (
    HOUR,
    START,
    START_MS,
    SyntheticTransport,
    _rows,
)
from trade_rl.integrations.binance.dataset import (
    BinanceMarketDataSource,
    BinanceSourceClosePolicy,
    _parse_kline_rows,
)
from trade_rl.integrations.binance.derived_klines import (
    BaseIntervalAggregatingTransport,
)

_V2 = BinanceSourceClosePolicy.STRICT_SOURCE_CLOSE_V2


def test_policy_cannot_change_after_a_legacy_source_was_cached() -> None:
    source = BinanceMarketDataSource(
        market="usds-m",
        interval="1h",
        start_time=START,
        end_time=START + timedelta(hours=24),
        transport=SyntheticTransport(damage_at=6, delta=-HOUR),
    )
    assert len(source.load("BTCUSDT").timestamps) == 24

    with pytest.raises(AttributeError):
        source.source_close_policy = _V2

    assert source.source_close_policy is BinanceSourceClosePolicy.RECONSTRUCTED_V1
    # A fresh v2 instance must still reject the exact same raw source.
    new = BinanceMarketDataSource(
        market="usds-m",
        interval="1h",
        start_time=START,
        end_time=START + timedelta(hours=24),
        transport=SyntheticTransport(damage_at=6, delta=-HOUR),
        source_close_policy=_V2,
    )
    with pytest.raises(ValueError, match="source close_time"):
        new.load("BTCUSDT")


@pytest.mark.parametrize("unit_scale", (1, 1000))
@pytest.mark.parametrize("delta", (-1, 0, 1))
def test_v2_requires_exact_native_source_close_time(
    unit_scale: int, delta: int
) -> None:
    rows = _rows(count=24)
    for row in rows:
        row[0] = int(row[0]) * unit_scale
        row[6] = (int(row[6]) + 1) * unit_scale - 1
    rows[6][6] = int(rows[6][6]) + delta

    def parse() -> tuple[object, ...]:
        return _parse_kline_rows(
            rows,
            interval_ms=HOUR,
            start_ms=START_MS,
            end_ms=START_MS + 24 * HOUR,
            source_close_policy=_V2,
        )

    if delta:
        with pytest.raises(ValueError, match="source close_time"):
            parse()
    else:
        parsed = parse()
        assert len(parsed[0]) == 24


@pytest.mark.parametrize("open_unit_scale,close_unit_scale", ((1, 1000), (1000, 1)))
def test_v2_refuses_mixed_native_time_units(
    open_unit_scale: int, close_unit_scale: int
) -> None:
    rows = _rows(count=24)
    for row in rows:
        row[0] = int(row[0]) * open_unit_scale
        row[6] = (int(row[6]) + 1) * close_unit_scale - 1
    with pytest.raises(ValueError, match="source close_time"):
        _parse_kline_rows(
            rows,
            interval_ms=HOUR,
            start_ms=START_MS,
            end_ms=START_MS + 24 * HOUR,
            source_close_policy=_V2,
        )


def test_strict_v2_does_not_trust_opaque_wrapper_around_legacy_aggregator() -> None:
    unverified_aggregator = BaseIntervalAggregatingTransport(
        SyntheticTransport(damage_at=3, delta=-HOUR)
    )

    class OpaqueDelegator:
        def load_klines(self, **kwargs: object) -> object:
            return unverified_aggregator.load_klines(**kwargs)

        def load_funding_rates(self, **kwargs: object) -> object:
            return unverified_aggregator.load_funding_rates(**kwargs)

    source = BinanceMarketDataSource(
        market="usds-m",
        interval="1h",
        start_time=START,
        end_time=START + timedelta(hours=24),
        transport=OpaqueDelegator(),
        source_close_policy=_V2,
    )
    with pytest.raises(ValueError, match="(verified|aggregate|transport)"):
        source.load_timeframe("BTCUSDT", "4h")
    assert unverified_aggregator.dropped_windows == {}


def test_v2_refuses_unvalidated_aggregate_even_when_it_is_the_source_base() -> None:
    aggregator = BaseIntervalAggregatingTransport(
        SyntheticTransport(damage_at=3, delta=-HOUR)
    )
    source = BinanceMarketDataSource(
        market="usds-m",
        interval="4h",
        start_time=START,
        end_time=START + timedelta(hours=24),
        transport=aggregator,
        source_close_policy=_V2,
    )
    with pytest.raises(ValueError, match="aggregate.*source close"):
        source.load("BTCUSDT")
    assert aggregator.dropped_windows == {}


def test_aggregate_policy_cannot_mutate_after_v2_data_was_cached() -> None:
    aggregator = BaseIntervalAggregatingTransport(
        SyntheticTransport(),
        source_close_policy=_V2,
    )
    source = BinanceMarketDataSource(
        market="usds-m",
        interval="1h",
        start_time=START,
        end_time=START + timedelta(hours=24),
        transport=aggregator,
        source_close_policy=_V2,
    )
    assert len(source.load_timeframe("BTCUSDT", "4h").timestamps) == 6
    with pytest.raises(AttributeError):
        aggregator.source_close_policy = BinanceSourceClosePolicy.RECONSTRUCTED_V1
    assert aggregator.source_close_policy is _V2


@pytest.mark.parametrize("gap_policy", ("strict", "non_tradable_rows"))
@pytest.mark.parametrize("microsecond_offset", (-1, 0, 1))
def test_v2_spot_microsecond_rows_retain_native_precision(
    gap_policy: str, microsecond_offset: int
) -> None:
    """SPOT Vision 2025+ close_time is in microseconds, not milliseconds."""

    class NativeMicroTransport:
        def load_klines(self, **kwargs: object) -> tuple[list[list[object]], str]:
            raw = _rows(count=24)
            for row in raw:
                row[0] = int(row[0]) * 1000
                row[6] = (int(row[6]) + 1) * 1000 - 1
            raw[6][6] = int(raw[6][6]) + microsecond_offset
            return raw, "synthetic-spot-native-microseconds"

        def load_funding_rates(self, **kwargs: object) -> tuple[list[object], str]:
            return [], "synthetic"

    source = BinanceMarketDataSource(
        market="spot",
        interval="1h",
        start_time=START,
        end_time=START + timedelta(hours=24),
        transport=NativeMicroTransport(),
        gap_policy=gap_policy,
        source_close_policy=_V2,
    )
    if microsecond_offset:
        with pytest.raises(ValueError, match="source close_time"):
            source.load("BTCUSDT")
    else:
        assert len(source.load("BTCUSDT").timestamps) == 24


@pytest.mark.parametrize("microsecond_offset", (-1, 0, 1))
def test_v2_microsecond_components_cannot_be_laundered_in_4h_aggregation(
    microsecond_offset: int,
) -> None:
    from trade_rl.integrations.binance.derived_klines import aggregate_klines

    rows = _rows(count=4)
    for row in rows:
        row[0] = int(row[0]) * 1000
        row[6] = (int(row[6]) + 1) * 1000 - 1
    rows[3][6] = int(rows[3][6]) + microsecond_offset

    def aggregate() -> tuple[list[list[object]], int]:
        return aggregate_klines(
            rows,
            base_interval="1h",
            interval="4h",
            start_ms=START_MS,
            end_ms=START_MS + 4 * HOUR,
            source_close_policy=_V2,
        )

    if microsecond_offset:
        with pytest.raises(ValueError, match="source close_time"):
            aggregate()
    else:
        bars, missing = aggregate()
        assert len(bars) == 1 and missing == 0


def test_v2_rejects_opaque_4h_aggregator_on_nominal_4h_base_clock() -> None:
    """Nominal Dataset interval cannot authenticate wrapped derived source rows."""
    unchecked = BaseIntervalAggregatingTransport(
        SyntheticTransport(damage_at=3, delta=-HOUR)
    )

    class OpaqueDelegator:
        def load_klines(self, **kwargs: object) -> object:
            return unchecked.load_klines(**kwargs)

        def load_funding_rates(self, **kwargs: object) -> object:
            return unchecked.load_funding_rates(**kwargs)

    source = BinanceMarketDataSource(
        market="usds-m",
        interval="4h",
        start_time=START,
        end_time=START + timedelta(hours=24),
        transport=OpaqueDelegator(),
        source_close_policy=_V2,
    )
    with pytest.raises(ValueError, match="verified higher-timeframe transport"):
        source.load("BTCUSDT")
    assert unchecked.dropped_windows == {}


def test_v2_rejects_decorated_aggregator_that_claims_source_validation() -> None:
    """An inherited marker is not proof that overridden logic checked 1h rows."""

    from trade_rl.integrations.binance.derived_klines import aggregate_klines

    class UnvalidatedSubclass(BaseIntervalAggregatingTransport):
        def load_klines(self, **kwargs: object) -> object:
            rows, source = self._klines.load_klines(
                market=kwargs["market"],
                symbol=kwargs["symbol"],
                interval="1h",
                start_ms=kwargs["start_ms"],
                end_ms=kwargs["end_ms"],
                mode=kwargs["mode"],
            )
            bars, _ = aggregate_klines(
                rows,
                base_interval="1h",
                interval=str(kwargs["interval"]),
                start_ms=int(kwargs["start_ms"]),
                end_ms=int(kwargs["end_ms"]),
            )
            return bars, source

    unverified = UnvalidatedSubclass(
        SyntheticTransport(damage_at=3, delta=-HOUR),
        source_close_policy=_V2,
    )
    source = BinanceMarketDataSource(
        market="usds-m",
        interval="1h",
        start_time=START,
        end_time=START + timedelta(hours=24),
        transport=unverified,
        source_close_policy=_V2,
    )
    with pytest.raises(ValueError, match="verified higher-timeframe transport"):
        source.load_timeframe("BTCUSDT", "4h")


def test_v2_allows_maintained_native_higher_timeframe_transport(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A genuine native 4h source need not be aggregated from 1h components."""
    from trade_rl.integrations.binance.transport import BinancePublicTransport

    transport = BinancePublicTransport(allow_network=False)

    def load_native(**kwargs: object) -> tuple[list[list[object]], str]:
        assert kwargs["interval"] == "4h"
        rows = _rows(count=6)
        for i, row in enumerate(rows):
            open_ms = START_MS + i * 4 * HOUR
            row[0] = open_ms
            row[6] = open_ms + 4 * HOUR - 1
        return rows, "synthetic-native-4h"

    def no_funding(**kwargs: object) -> tuple[list[object], str]:
        return [], "synthetic-native-4h"

    monkeypatch.setattr(transport, "load_klines", load_native)
    monkeypatch.setattr(transport, "load_funding_rates", no_funding)
    source = BinanceMarketDataSource(
        market="spot",
        interval="1h",
        start_time=START,
        end_time=START + timedelta(hours=24),
        transport=transport,
        source_close_policy=_V2,
    )
    series = source.load_timeframe("BTCUSDT", "4h")
    assert len(series.timestamps) == 6
    assert all(series.tradable)


def test_v2_rejects_nested_unverified_aggregator_with_mismatched_base() -> None:
    """An outer v2 marker cannot authenticate inner legacy-derived 4h bars."""
    inner_legacy = BaseIntervalAggregatingTransport(
        SyntheticTransport(damage_at=3, delta=-HOUR),
        base_interval="1h",
    )
    outer = BaseIntervalAggregatingTransport(
        inner_legacy,
        base_interval="4h",
        source_close_policy=_V2,
    )
    source = BinanceMarketDataSource(
        market="usds-m",
        interval="1h",
        start_time=START,
        end_time=START + timedelta(hours=24),
        transport=outer,
        source_close_policy=_V2,
    )
    with pytest.raises(ValueError, match="(aggregate|source close|verified)"):
        source.load_timeframe("BTCUSDT", "4h")
    assert inner_legacy.dropped_windows == {}


@pytest.mark.parametrize(
    "value",
    (
        float(START_MS + HOUR - 1),
        True,
        " 1767259199999",
        "-1",
        "١٧٦٧٢٥٩١٩٩٩٩",
    ),
)
def test_v2_rejects_noninteger_source_close_type(value: object) -> None:
    from trade_rl.integrations.binance.source_clock import _check_source_close_time

    row = _rows(count=1)[0]
    row[6] = value
    with pytest.raises(ValueError, match="source close_time"):
        _check_source_close_time(
            row,
            open_ms=START_MS,
            interval_ms=HOUR,
            source_close_policy=_V2,
        )


@pytest.mark.parametrize(
    "invalid_open",
    (
        START_MS + 1,
        START_MS * 1000 + 1,
        START_MS * 1_000_000,
        START_MS // 1000,
    ),
)
def test_v2_rejects_invalid_source_open_unit_or_offset(invalid_open: int) -> None:
    from trade_rl.integrations.binance.source_clock import _check_source_close_time

    row = _rows(count=1)[0]
    row[0] = invalid_open
    with pytest.raises(ValueError, match="invalid native open time"):
        _check_source_close_time(
            row,
            open_ms=START_MS,
            interval_ms=HOUR,
            source_close_policy=_V2,
        )


def test_v2_allows_verified_native_4h_base_in_maintained_aggregator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An authentic native 4h owner is distinct from pre-aggregated opaque 4h."""
    from trade_rl.integrations.binance.transport import BinancePublicTransport

    native = BinancePublicTransport(allow_network=False)

    def native_four_hour(**kwargs: object) -> tuple[list[list[object]], str]:
        assert kwargs["interval"] == "4h"
        rows = _rows(count=6)
        for i, row in enumerate(rows):
            open_ms = START_MS + i * 4 * HOUR
            row[0] = open_ms
            row[6] = open_ms + 4 * HOUR - 1
        return rows, "synthetic-direct-native-4h"

    def no_funding(**kwargs: object) -> tuple[list[object], str]:
        return [], "synthetic"

    monkeypatch.setattr(native, "load_klines", native_four_hour)
    monkeypatch.setattr(native, "load_funding_rates", no_funding)

    aggregator = BaseIntervalAggregatingTransport(
        native,
        base_interval="4h",
        source_close_policy=_V2,
    )
    source = BinanceMarketDataSource(
        market="usds-m",
        interval="1h",
        start_time=START,
        end_time=START + timedelta(hours=24),
        transport=aggregator,
        source_close_policy=_V2,
    )
    result = source.load_timeframe("BTCUSDT", "4h")
    assert len(result.timestamps) == 6
    assert all(result.tradable)
