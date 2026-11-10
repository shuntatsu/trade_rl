"""Synthetic fixtures shared by the cross-asset tests (no network, no crypto)."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np

from trade_rl.integrations.cross_asset.catalog import SeriesSpec, series_by_key
from trade_rl.integrations.cross_asset.fetch import TransportError, fetch_all
from trade_rl.integrations.cross_asset.observations import ObsSeries, build_obs_series
from trade_rl.integrations.cross_asset.parsers import ParsedSeries
from trade_rl.integrations.cross_asset.store import CrossAssetStore

FETCHED_AT = datetime(2025, 7, 10, 12, 0, tzinfo=UTC)
SEEDED_KEYS = (
    "SP500",
    "NASDAQCOM",
    "NIKKEI225",
    "DTWEXBGS",
    "DEXJPUS",
    "DEXUSEU",
    "T10Y2Y",
    "VIXCLS",
)


class FakeTransport:
    def __init__(self, responses: dict[str, bytes | Exception]) -> None:
        self.responses = responses
        self.requested: list[str] = []

    def get(self, url: str) -> bytes:
        self.requested.append(url)
        for fragment, response in self.responses.items():
            if fragment in url:
                if isinstance(response, Exception):
                    raise response
                return response
        raise TransportError("no fake response", status=404)


def weekdays(count: int, start: str = "2023-01-02") -> np.ndarray:
    days: list[np.datetime64] = []
    current = np.datetime64(start, "D")
    while len(days) < count:
        if current.astype(datetime).weekday() < 5:
            days.append(current)
        current += np.timedelta64(1, "D")
    return np.asarray(days, dtype="datetime64[D]")


def random_walk(
    count: int, seed: int, *, start: float = 100.0, drift: float = 0.0005
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return start * np.exp(np.cumsum(drift + 0.01 * rng.standard_normal(count)))


def parsed(days: np.ndarray, values: np.ndarray) -> ParsedSeries:
    return ParsedSeries(
        dates=days,
        values=np.asarray(values, dtype=np.float64),
        rows_total=int(days.size),
        missing_rows=0,
        last_row_date=None,
    )


def obs_series(key: str, days: np.ndarray, values: np.ndarray) -> ObsSeries:
    spec: SeriesSpec = series_by_key(key)
    return build_obs_series(spec, parsed(days, values))


def fred_payload(key: str, days: np.ndarray, values: np.ndarray) -> bytes:
    rows = "".join(
        f"{day},{float(value)!r}\n" for day, value in zip(days, values, strict=True)
    )
    return f"observation_date,{key}\n{rows}".encode()


def clock(start: datetime = FETCHED_AT) -> Callable[[], datetime]:
    state = {"now": start}

    def now() -> datetime:
        current = state["now"]
        state["now"] = current + timedelta(seconds=1)
        return current

    return now


def seed_store(
    root: Path, *, count: int = 420, scale: float = 1.0, at: datetime = FETCHED_AT
) -> CrossAssetStore:
    """Write a synthetic FRED-shaped store with eight series."""

    store = CrossAssetStore(root)
    days = weekdays(count)
    responses: dict[str, bytes | Exception] = {}
    for offset, key in enumerate(SEEDED_KEYS):
        if key == "T10Y2Y":
            values = 0.5 + np.cumsum(
                0.02 * np.random.default_rng(offset).standard_normal(count)
            )
        else:
            values = random_walk(count, seed=offset) * scale
        responses[f"id={key}"] = fred_payload(key, days, values)
    specs = tuple(series_by_key(key) for key in SEEDED_KEYS)
    fetch_all(
        store,
        specs,
        FakeTransport(responses),
        now=clock(at),
        sleep=lambda _seconds: None,
    )
    return store
