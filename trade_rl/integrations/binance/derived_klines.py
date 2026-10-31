"""Higher-timeframe klines aggregated from verified base-interval klines.

Opt-in only. The canonical transport fetches each native interval from its own
Binance archive; this module is for offline development builds whose local
archive roster holds only one base interval (1h). Every aggregated bar must be
complete: a window with any missing base bar is dropped, never interpolated, so
the gap policy of the consuming source decides how the hole is represented.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from trade_rl.integrations.binance.vision import _interval_ms, _normalize_epoch_ms

_KlineRow = Sequence[object]


def aggregate_klines(
    rows: Sequence[_KlineRow],
    *,
    base_interval: str,
    interval: str,
    start_ms: int,
    end_ms: int,
) -> tuple[list[list[object]], int]:
    """Aggregate ``base_interval`` kline rows into complete ``interval`` rows.

    Returns ``(aggregated_rows, dropped_windows)``. A window is aligned to
    ``start_ms`` and spans ``interval / base_interval`` consecutive base bars:
    open from the first bar, high/low as extremes, close from the last bar,
    base and quote volume as sums, close time as window end minus one
    millisecond. Windows with any missing base bar are dropped and counted.
    """

    base_ms = _interval_ms(base_interval)
    step = _interval_ms(interval)
    if step % base_ms != 0 or step < base_ms:
        raise ValueError("aggregated interval must be a multiple of the base interval")
    if start_ms % step != 0 or end_ms % step != 0:
        raise ValueError("aggregation range must align to the aggregated interval")
    parts_per_window = step // base_ms
    by_open: dict[int, _KlineRow] = {}
    for row in rows:
        if len(row) < 8:
            raise ValueError("Binance kline row must contain at least eight fields")
        open_ms = _normalize_epoch_ms(row[0])
        if open_ms in by_open:
            raise ValueError(f"duplicate Binance kline open time {open_ms}")
        by_open[open_ms] = row
    aggregated: list[list[object]] = []
    dropped = 0
    for open_ms in range(start_ms, end_ms, step):
        parts = [by_open.get(open_ms + k * base_ms) for k in range(parts_per_window)]
        complete = [part for part in parts if part is not None]
        if len(complete) != parts_per_window:
            dropped += 1
            continue
        aggregated.append(
            [
                open_ms,
                complete[0][1],
                repr(max(float(str(part[2])) for part in complete)),
                repr(min(float(str(part[3])) for part in complete)),
                complete[-1][4],
                repr(sum(float(str(part[5])) for part in complete)),
                open_ms + step - 1,
                repr(sum(float(str(part[7])) for part in complete)),
            ]
        )
    return aggregated, dropped


class BaseIntervalAggregatingTransport:
    """Serve every requested kline interval from one base-interval transport.

    ``load_klines`` fetches the ``base_interval`` rows from ``klines`` and
    aggregates them for any other interval; funding rates come from
    ``klines`` and exchange information from ``exchange_information`` (default:
    ``klines``). It never reaches the network on its own; offline-ness is the
    property of the wrapped transports.
    """

    def __init__(
        self,
        klines: Any,
        *,
        exchange_information: Any | None = None,
        base_interval: str = "1h",
    ) -> None:
        _interval_ms(base_interval)
        self._klines = klines
        self._exchange_information = (
            klines if exchange_information is None else exchange_information
        )
        self.base_interval = base_interval
        self.dropped_windows: dict[tuple[str, str], int] = {}

    def load_klines(
        self,
        *,
        market: Any,
        symbol: str,
        interval: str,
        start_ms: int,
        end_ms: int,
        mode: Any,
    ) -> tuple[list[list[object]], Any]:
        rows, source = self._klines.load_klines(
            market=market,
            symbol=symbol,
            interval=self.base_interval,
            start_ms=start_ms,
            end_ms=end_ms,
            mode=mode,
        )
        if interval == self.base_interval:
            return list(rows), source
        aggregated, dropped = aggregate_klines(
            rows,
            base_interval=self.base_interval,
            interval=interval,
            start_ms=start_ms,
            end_ms=end_ms,
        )
        self.dropped_windows[(symbol, interval)] = dropped
        return aggregated, source

    def load_funding_rates(self, **kwargs: Any) -> Any:
        return self._klines.load_funding_rates(**kwargs)

    def load_exchange_information(self, **kwargs: Any) -> Any:
        return self._exchange_information.load_exchange_information(**kwargs)


__all__ = ["BaseIntervalAggregatingTransport", "aggregate_klines"]
