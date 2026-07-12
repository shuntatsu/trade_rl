import numpy as np
import pytest

from mars_lite.serving.market_time import resolve_completed_bar_endpoint


@pytest.mark.parametrize(
    ("timeframe", "timestamps", "now", "expected_end", "expected_age"),
    [
        (
            "1h",
            ["2026-07-12T08:00", "2026-07-12T09:00"],
            "2026-07-12T09:30",
            1,
            0.5,
        ),
        (
            "4h",
            ["2026-07-12T00:00", "2026-07-12T04:00"],
            "2026-07-12T07:00",
            1,
            3.0,
        ),
        (
            "1d",
            ["2026-07-10T00:00", "2026-07-11T00:00"],
            "2026-07-12T12:00",
            2,
            12.0,
        ),
    ],
)
def test_resolve_completed_bar_endpoint(
    timeframe: str,
    timestamps: list[str],
    now: str,
    expected_end: int,
    expected_age: float,
) -> None:
    endpoint = resolve_completed_bar_endpoint(
        np.asarray(timestamps, dtype="datetime64[ns]"),
        base_timeframe=timeframe,
        now_utc=np.datetime64(now, "ns"),
    )

    assert endpoint.end_exclusive == expected_end
    assert endpoint.data_age_hours == pytest.approx(expected_age)


def test_resolve_completed_bar_endpoint_rejects_no_completed_bar() -> None:
    with pytest.raises(ValueError, match="no completed bar"):
        resolve_completed_bar_endpoint(
            np.asarray(["2026-07-12T08:00"], dtype="datetime64[ns]"),
            base_timeframe="1h",
            now_utc=np.datetime64("2026-07-12T08:30", "ns"),
        )


@pytest.mark.parametrize(
    "timestamps",
    [
        ["2026-07-12T09:00", "2026-07-12T08:00"],
        ["2026-07-12T08:00", "2026-07-12T08:00"],
    ],
)
def test_resolve_completed_bar_endpoint_requires_strict_order(
    timestamps: list[str],
) -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        resolve_completed_bar_endpoint(
            np.asarray(timestamps, dtype="datetime64[ns]"),
            base_timeframe="1h",
            now_utc=np.datetime64("2026-07-12T12:00", "ns"),
        )


def test_resolve_completed_bar_endpoint_rejects_unknown_timeframe() -> None:
    with pytest.raises(ValueError, match="unsupported base timeframe"):
        resolve_completed_bar_endpoint(
            np.asarray(["2026-07-12T08:00"], dtype="datetime64[ns]"),
            base_timeframe="2h",
            now_utc=np.datetime64("2026-07-12T12:00", "ns"),
        )
