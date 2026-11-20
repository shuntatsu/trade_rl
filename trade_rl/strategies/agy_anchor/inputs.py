"""Typed klines + funding input bundle of the agy anchor and its funding observations.

The anchor needs, per 4h bar close and per symbol, the spot and perpetual close, quote
volume and taker-buy quote volume, a "valid" flag per leg, and the raw funding events.
Nothing else: Hyperliquid, Binance open interest/long-short metrics and the 15m jump are
neutral in this variant (see ``tests/fixtures/agy_anchor/README.md``).

Timing contract (identical to the lab): a row is a *completed* 4h bar and is stamped with
its close time ``close_ns``; a funding event is known at row ``k`` iff its timestamp is
``<= close_ns[k]``. Row ``k`` of every derived array uses only rows ``<= k`` and events
``<= close_ns[k]``, so truncating the bundle never changes earlier rows.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from trade_rl.data.market import MarketDataset

_PAIR_SUFFIXES = (":spot", ":perp")


def _readonly(array: np.ndarray) -> np.ndarray:
    view = np.ascontiguousarray(array).view()
    view.setflags(write=False)
    return view


@dataclass(frozen=True, slots=True)
class AnchorInputs:
    """Klines + funding inputs on a 4h bar-close clock for ``K`` symbols."""

    symbols: tuple[str, ...]
    close_ns: np.ndarray
    perp_close: np.ndarray
    spot_close: np.ndarray
    perp_quote_volume: np.ndarray
    spot_quote_volume: np.ndarray
    perp_taker_buy_quote_volume: np.ndarray
    spot_taker_buy_quote_volume: np.ndarray
    perp_valid: np.ndarray
    spot_valid: np.ndarray
    funding_event_ns: tuple[np.ndarray, ...]
    funding_event_rate: tuple[np.ndarray, ...]

    def __post_init__(self) -> None:
        k = len(self.symbols)
        if k < 4:
            raise ValueError("the anchor needs at least four symbols")
        if len(set(self.symbols)) != k:
            raise ValueError("symbols must be unique")
        close_ns = np.asarray(self.close_ns, dtype=np.int64)
        if close_ns.ndim != 1 or close_ns.size < 2:
            raise ValueError("close_ns must be a one-dimensional clock of >= 2 rows")
        if np.any(np.diff(close_ns) <= 0):
            raise ValueError("close_ns must be strictly increasing")
        t = int(close_ns.size)
        object.__setattr__(self, "close_ns", _readonly(close_ns))
        for name in (
            "perp_close",
            "spot_close",
            "perp_quote_volume",
            "spot_quote_volume",
            "perp_taker_buy_quote_volume",
            "spot_taker_buy_quote_volume",
        ):
            array = np.asarray(getattr(self, name), dtype=np.float64)
            if array.shape != (t, k):
                raise ValueError(f"{name} must have shape ({t}, {k})")
            if not np.isfinite(array).all():
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name, _readonly(array))
        for name in ("perp_close", "spot_close"):
            if np.any(getattr(self, name) <= 0.0):
                raise ValueError(f"{name} must be strictly positive")
        for leg in ("perp", "spot"):
            quote = getattr(self, f"{leg}_quote_volume")
            taker = getattr(self, f"{leg}_taker_buy_quote_volume")
            if np.any(quote < 0.0) or np.any(taker < 0.0):
                raise ValueError(f"{leg} volumes must be non-negative")
            if np.any(taker > quote * (1.0 + 1e-9) + 1e-9):
                raise ValueError(f"{leg} taker-buy volume exceeds the quote volume")
        for name in ("perp_valid", "spot_valid"):
            flags = np.asarray(getattr(self, name))
            if flags.dtype != np.bool_ or flags.shape != (t, k):
                raise ValueError(f"{name} must be a boolean ({t}, {k}) array")
            object.__setattr__(self, name, _readonly(flags))
        if len(self.funding_event_ns) != k or len(self.funding_event_rate) != k:
            raise ValueError("funding events must be given for every symbol")
        events_ns: list[np.ndarray] = []
        events_rate: list[np.ndarray] = []
        last_close = int(close_ns[-1])
        for symbol, ev_ns, ev_rate in zip(
            self.symbols, self.funding_event_ns, self.funding_event_rate, strict=True
        ):
            ns = np.asarray(ev_ns, dtype=np.int64)
            rate = np.asarray(ev_rate, dtype=np.float64)
            if ns.ndim != 1 or rate.shape != ns.shape:
                raise ValueError(f"{symbol}: funding events must be 1-D and aligned")
            if not np.isfinite(rate).all():
                raise ValueError(f"{symbol}: funding rates must be finite")
            if np.any(np.diff(ns) <= 0):
                raise ValueError(
                    f"{symbol}: funding events must be strictly increasing"
                )
            if ns.size and int(ns[-1]) > last_close:
                raise ValueError(
                    f"{symbol}: a funding event is later than the last bar close"
                )
            events_ns.append(_readonly(ns))
            events_rate.append(_readonly(rate))
        object.__setattr__(self, "funding_event_ns", tuple(events_ns))
        object.__setattr__(self, "funding_event_rate", tuple(events_rate))

    @property
    def n_rows(self) -> int:
        return int(self.close_ns.size)

    def truncate(self, rows: int) -> AnchorInputs:
        """The first ``rows`` rows and only the funding events known at the last one."""
        if not 2 <= rows <= self.n_rows:
            raise ValueError("rows must lie in [2, n_rows]")
        last_close = int(self.close_ns[rows - 1])
        keep = [ns <= last_close for ns in self.funding_event_ns]
        return AnchorInputs(
            symbols=self.symbols,
            close_ns=self.close_ns[:rows],
            perp_close=self.perp_close[:rows],
            spot_close=self.spot_close[:rows],
            perp_quote_volume=self.perp_quote_volume[:rows],
            spot_quote_volume=self.spot_quote_volume[:rows],
            perp_taker_buy_quote_volume=self.perp_taker_buy_quote_volume[:rows],
            spot_taker_buy_quote_volume=self.spot_taker_buy_quote_volume[:rows],
            perp_valid=self.perp_valid[:rows],
            spot_valid=self.spot_valid[:rows],
            funding_event_ns=tuple(
                ns[m] for ns, m in zip(self.funding_event_ns, keep, strict=True)
            ),
            funding_event_rate=tuple(
                rate[m] for rate, m in zip(self.funding_event_rate, keep, strict=True)
            ),
        )


def funding_observations(
    inputs: AnchorInputs,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(last, mean of last 9 events, mean of last 21 events)`` known at each row close.

    Lab ``extract_real_data.compute_causal_funding_arrays`` (observed arrays only). The
    "3d" and "7d" means are over the last 9 and 21 *events* (count based), and a row
    before the first event is zero.
    """
    t, k = inputs.n_rows, len(inputs.symbols)
    last = np.zeros((t, k), dtype=np.float64)
    ma3 = np.zeros((t, k), dtype=np.float64)
    ma7 = np.zeros((t, k), dtype=np.float64)
    close_ns = np.asarray(inputs.close_ns)
    for s_idx in range(k):
        ev_ns = inputs.funding_event_ns[s_idx]
        ev_rate = inputs.funding_event_rate[s_idx]
        if ev_ns.size == 0:
            continue
        pref = np.concatenate(([0.0], np.cumsum(ev_rate)))
        cnt = np.searchsorted(ev_ns, close_ns, side="right")
        valid = cnt > 0
        idx_last = np.maximum(cnt - 1, 0)
        last[valid, s_idx] = ev_rate[idx_last[valid]]
        lo_3d = np.maximum(cnt - 9, 0)
        span_3d = np.maximum(cnt - lo_3d, 1)
        ma3[valid, s_idx] = (pref[cnt[valid]] - pref[lo_3d[valid]]) / span_3d[valid]
        lo_7d = np.maximum(cnt - 21, 0)
        span_7d = np.maximum(cnt - lo_7d, 1)
        ma7[valid, s_idx] = (pref[cnt[valid]] - pref[lo_7d[valid]]) / span_7d[valid]
    return last, ma3, ma7


def pair_names(symbols: Sequence[str]) -> tuple[str, ...]:
    """Pair names of an alternating ``X:spot, X:perp`` leg layout (strict)."""
    if len(symbols) % 2 or not symbols:
        raise ValueError("an alternating spot/perp layout has an even leg count")
    names: list[str] = []
    for index in range(0, len(symbols), 2):
        spot, perp = symbols[index], symbols[index + 1]
        base = spot.removesuffix(_PAIR_SUFFIXES[0])
        if (
            spot != base + _PAIR_SUFFIXES[0]
            or perp != base + _PAIR_SUFFIXES[1]
            or not base
        ):
            raise ValueError("legs must alternate '<pair>:spot', '<pair>:perp'")
        names.append(base)
    return tuple(names)


def from_market_dataset(
    dataset: MarketDataset,
    *,
    taker_buy_quote_volume: np.ndarray,
    funding_events: Sequence[tuple[np.ndarray, np.ndarray]] | None = None,
) -> AnchorInputs:
    """Build the bundle from an alternating 4h ``MarketDataset``.

    ``dataset.volume`` is the quote turnover of each leg and ``dataset.tradable`` the leg
    validity (a halted spot bar is non-tradable with zero volume). The taker-buy quote
    volume is not part of the dataset (nor of ``RawMarketSeries``); pass the ``(T, 2P)``
    sum of the four hourly kline taker-buy quote values per leg. ``funding_events`` are the exact ``(ns, rate)``
    arrays per pair (from the funding rows, including pre-window events for a warm
    start). When omitted each 4h row carrying exactly one funding event becomes an event
    stamped at that row's close, which preserves its bar membership but starts the
    count-based means cold at the first row; a row with two or more events is refused.
    """
    names = pair_names(dataset.symbols)
    taker = np.asarray(taker_buy_quote_volume, dtype=np.float64)
    if taker.shape != (dataset.n_bars, dataset.n_symbols):
        raise ValueError("taker_buy_quote_volume must match the dataset (T, legs)")
    close_ns = dataset.timestamps.astype("datetime64[ns]").astype(np.int64)
    volume = np.asarray(dataset.volume, dtype=np.float64)
    tradable = np.asarray(dataset.tradable, dtype=bool)
    if funding_events is None:
        rate_rows = np.asarray(dataset.funding_rate, dtype=np.float64)[:, 1::2]
        count_rows = np.asarray(dataset.resolved_array("funding_event_count"))[:, 1::2]
        if np.any(count_rows > 1):
            raise ValueError(
                "a 4h bar carries two or more funding events: pass exact funding_events"
            )
        events: list[tuple[np.ndarray, np.ndarray]] = []
        for column in range(len(names)):
            rows = np.flatnonzero(count_rows[:, column] == 1)
            events.append((close_ns[rows], rate_rows[rows, column]))
    else:
        if len(funding_events) != len(names):
            raise ValueError("funding_events must cover every pair")
        events = [(np.asarray(ns), np.asarray(rate)) for ns, rate in funding_events]
    spot, perp = slice(0, None, 2), slice(1, None, 2)
    return AnchorInputs(
        symbols=names,
        close_ns=close_ns,
        perp_close=np.asarray(dataset.close, dtype=np.float64)[:, perp],
        spot_close=np.asarray(dataset.close, dtype=np.float64)[:, spot],
        perp_quote_volume=volume[:, perp],
        spot_quote_volume=volume[:, spot],
        perp_taker_buy_quote_volume=taker[:, perp],
        spot_taker_buy_quote_volume=taker[:, spot],
        perp_valid=tradable[:, perp],
        spot_valid=tradable[:, spot],
        funding_event_ns=tuple(ns for ns, _ in events),
        funding_event_rate=tuple(rate for _, rate in events),
    )


__all__ = [
    "AnchorInputs",
    "from_market_dataset",
    "funding_observations",
    "pair_names",
]
