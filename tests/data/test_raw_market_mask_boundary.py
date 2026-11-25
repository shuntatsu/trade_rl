"""Raw input must not erase explicit missingness before native-clock admission."""

from dataclasses import replace

import numpy as np
import pytest

from trade_rl.data.source import RawMarketSeries


def _series() -> RawMarketSeries:
    closes = np.arange("2020-01-06", "2020-01-27", 7, dtype="datetime64[D]")
    values = np.array([100.0, 101.0, 102.0])
    return RawMarketSeries(
        timestamps=closes,
        available_at=closes,
        open=values,
        high=values + 1,
        low=values - 1,
        close=values,
        volume=np.ones(3),
        funding_rate=np.zeros(3),
        tradable=np.ones(3, dtype=bool),
        funding_available=np.zeros(3, dtype=bool),
        funding_event_count=np.zeros(3, dtype=np.int32),
    )


@pytest.mark.parametrize(
    "field",
    (
        "timestamps",
        "available_at",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "funding_rate",
        "tradable",
        "funding_available",
        "funding_event_count",
    ),
)
@pytest.mark.parametrize("view", (False, True))
def test_raw_input_rejects_masked_owner(field: str, view: bool) -> None:
    series = _series()
    masked = np.ma.array(getattr(series, field), mask=[False, True, False], copy=True)
    incoming = masked.astype(masked.dtype).view(np.ndarray) if view else masked
    if view:
        assert np.ma.isMaskedArray(incoming.base)
    with pytest.raises(ValueError, match=field + ".*masked array"):
        replace(series, **{field: incoming})
