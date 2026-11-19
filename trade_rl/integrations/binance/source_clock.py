"""Versioned raw Binance candle finality admission (opt-in new Dataset only)."""

from __future__ import annotations

from collections.abc import Sequence
from enum import StrEnum

from trade_rl.integrations.binance.vision import _normalize_epoch_ms

_KlineRow = Sequence[object]


class BinanceSourceClosePolicy(StrEnum):
    """Versioned admission of Binance source bar finality for new builds only."""

    RECONSTRUCTED_V1 = "reconstructed_v1"
    STRICT_SOURCE_CLOSE_V2 = "strict_source_close_v2"


def _check_source_close_time(
    row: _KlineRow,
    *,
    open_ms: int,
    interval_ms: int,
    source_close_policy: BinanceSourceClosePolicy,
) -> None:
    if source_close_policy is BinanceSourceClosePolicy.STRICT_SOURCE_CLOSE_V2:
        source_close_ms = _normalize_epoch_ms(row[6])
        if source_close_ms != open_ms + interval_ms - 1:
            raise ValueError(
                "Binance source close_time differs from expected interval end"
            )
