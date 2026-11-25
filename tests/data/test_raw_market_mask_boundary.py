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


@pytest.mark.parametrize("wrapper", ["buffer", "opaque", "subclass"])
def test_raw_close_refuses_unverified_owners(wrapper):
    series = _series()
    masked = np.ma.array(series.close, mask=[False, True, False], copy=True)
    if wrapper == "buffer":
        incoming = memoryview(masked)
    elif wrapper == "opaque":
        incoming = np.lib.stride_tricks.as_strided(np.asarray(memoryview(masked)))
    else:

        class Foreign(np.ndarray):
            pass

        incoming = series.close.view(Foreign)
    with pytest.raises(ValueError, match="close.*(masked array|opaque|subclass)"):
        replace(series, close=incoming)


@pytest.mark.parametrize("backing", ["owned", "slice", "bytes", "bytearray", "buffer"])
def test_raw_close_preserves_closed_plain_ownership(backing):
    series = _series()
    if backing == "owned":
        incoming = series.close.copy()
    elif backing == "slice":
        incoming = np.repeat(series.close, 2)[::2]
    elif backing == "bytes":
        incoming = np.frombuffer(series.close.tobytes(), dtype=np.float64)
    elif backing == "bytearray":
        incoming = np.frombuffer(bytearray(series.close.tobytes()), dtype=np.float64)
    else:
        incoming = memoryview(series.close)
    np.testing.assert_array_equal(replace(series, close=incoming).close, series.close)


def test_native_owner_check_refuses_opaque_without_getter_invocation():
    from trade_rl.data.native_clock import _native_unmasked_array

    class Opaque:
        def __init__(self):
            self.arr = np.array([100.0, 101.0, 102.0])
            self.__array_interface__ = self.arr.__array_interface__
            self.guard = False

        @property
        def __class__(self):
            if self.guard:
                pytest.fail("foreign class getter called")
            return Opaque

        @property
        def base(self):
            pytest.fail("foreign base getter called")

    owner = Opaque()
    arr = np.asarray(owner)
    assert arr.base is owner
    owner.guard = True
    with pytest.raises(ValueError, match="opaque"):
        _native_unmasked_array(arr, field_name="close")


def test_native_clock_refuses_opaque_retained_mask():
    from trade_rl.data.native_clock import NativeBarClock, NativeClockKind

    closes = np.array(["2024-02-01", "2024-03-01"], dtype="datetime64[D]")
    # Buffer views retain the masked numeric owner before reinterpreting ticks.
    owner = np.ma.array(closes.view(np.int64), mask=[False, True], copy=True)
    wrapped = np.lib.stride_tricks.as_strided(np.asarray(memoryview(owner))).view(
        closes.dtype
    )
    with pytest.raises(ValueError, match="opaque"):
        NativeBarClock(NativeClockKind.UTC_MONTH).interval_bounds(wrapped)
