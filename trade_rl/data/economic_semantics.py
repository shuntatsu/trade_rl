"""Canonical explicit economic arrays shared by all market dataset adapters."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Mapping

import numpy as np

from trade_rl.data.contracts import InstrumentContract


def _readonly(
    value: object, *, shape: tuple[int, int], dtype: np.dtype, field: str
) -> np.ndarray:
    array = np.asarray(value, dtype=dtype)
    if array.ndim == 0:
        array = np.full(shape, array.item(), dtype=dtype)
    elif array.shape != shape:
        try:
            array = np.broadcast_to(array, shape)
        except ValueError as error:
            raise ValueError(
                f"{field} cannot be broadcast to economic-array shape"
            ) from error
    result = np.array(array, dtype=dtype, copy=True, order="C")
    if np.issubdtype(result.dtype, np.floating) and not np.isfinite(result).all():
        raise ValueError(f"{field} must contain only finite values")
    result.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True)
class MarketEconomicSemantics:
    """All execution/accounting arrays that must never depend on dataset defaults."""

    symbol_active: np.ndarray
    asset_active: np.ndarray
    tradable: np.ndarray
    information_available: np.ndarray
    available_at: np.ndarray
    fee_rate: np.ndarray
    maker_fee_rate: np.ndarray
    taker_fee_rate: np.ndarray
    spread_rate: np.ndarray
    max_participation_rate: np.ndarray
    minimum_notional: np.ndarray
    lot_size: np.ndarray
    tick_size: np.ndarray
    borrow_available: np.ndarray
    borrow_rate: np.ndarray
    funding_due: np.ndarray
    buy_allowed: np.ndarray
    sell_allowed: np.ndarray
    mark_price: np.ndarray
    index_price: np.ndarray

    def __post_init__(self) -> None:
        shapes = {getattr(self, item.name).shape for item in fields(self)}
        if len(shapes) != 1:
            raise ValueError("economic arrays must share one bar-by-symbol shape")
        for item in fields(self):
            value = getattr(self, item.name)
            if value.flags.writeable:
                raise ValueError(f"economic array {item.name} must be immutable")

    def market_dataset_kwargs(self) -> Mapping[str, np.ndarray]:
        return {item.name: getattr(self, item.name) for item in fields(self)}


def build_market_economic_semantics(
    *,
    timestamps: np.ndarray,
    instruments: tuple[InstrumentContract, ...],
    row_present: object,
    raw_tradable: object,
    source_information_available: object,
    available_at: object,
    close: object,
    funding_event_count: object,
    fee_rate: object = 0.0,
    maker_fee_rate: object = 0.0,
    taker_fee_rate: object = 0.0,
    spread_rate: object = 0.0,
    max_participation_rate: object = 1.0,
    borrow_available: object = True,
    borrow_rate: object = 0.0,
    buy_allowed: object = True,
    sell_allowed: object = True,
    mark_price: object | None = None,
    index_price: object | None = None,
) -> MarketEconomicSemantics:
    """Resolve one point-in-time, explicit economic contract for a dataset."""

    resolved_timestamps = np.asarray(timestamps, dtype="datetime64[ns]")
    if resolved_timestamps.ndim != 1 or resolved_timestamps.size < 2:
        raise ValueError("economic timestamps must be a rank-one market clock")
    if not instruments:
        raise ValueError("economic semantics require instruments")
    shape = (len(resolved_timestamps), len(instruments))
    rows = _readonly(
        row_present, shape=shape, dtype=np.dtype(np.bool_), field="row_present"
    )
    raw_trade = _readonly(
        raw_tradable, shape=shape, dtype=np.dtype(np.bool_), field="raw_tradable"
    )
    source_info = _readonly(
        source_information_available,
        shape=shape,
        dtype=np.dtype(np.bool_),
        field="source_information_available",
    )
    resolved_available_at = _readonly(
        available_at,
        shape=shape,
        dtype=np.dtype("datetime64[ns]"),
        field="available_at",
    )
    close_array = _readonly(
        close, shape=shape, dtype=np.dtype(np.float64), field="close"
    )
    funding_counts = _readonly(
        funding_event_count,
        shape=shape,
        dtype=np.dtype(np.int32),
        field="funding_event_count",
    )
    if np.any(funding_counts < 0):
        raise ValueError("funding_event_count must be non-negative")

    active = np.zeros(shape, dtype=np.bool_)
    tick = np.zeros(shape, dtype=np.float64)
    lot = np.zeros(shape, dtype=np.float64)
    minimum = np.zeros(shape, dtype=np.float64)
    for symbol_index, contract in enumerate(instruments):
        listed = np.datetime64(
            contract.listed_at.astimezone(__import__("datetime").UTC).replace(
                tzinfo=None
            ),
            "ns",
        )
        mask = resolved_timestamps >= listed
        if contract.delisted_at is not None:
            delisted = np.datetime64(
                contract.delisted_at.astimezone(__import__("datetime").UTC).replace(
                    tzinfo=None
                ),
                "ns",
            )
            mask &= resolved_timestamps < delisted
        active[:, symbol_index] = mask
        resolved_tick, resolved_lot, resolved_minimum = contract.execution_rule_arrays(
            resolved_timestamps
        )
        tick[:, symbol_index] = resolved_tick
        lot[:, symbol_index] = resolved_lot
        minimum[:, symbol_index] = resolved_minimum

    causal_time = resolved_available_at <= np.broadcast_to(
        resolved_timestamps[:, None], shape
    )
    active_ro = _readonly(
        active, shape=shape, dtype=np.dtype(np.bool_), field="symbol_active"
    )
    information = _readonly(
        source_info & rows & active & causal_time,
        shape=shape,
        dtype=np.dtype(np.bool_),
        field="information_available",
    )
    tradable = _readonly(
        raw_trade & rows & active,
        shape=shape,
        dtype=np.dtype(np.bool_),
        field="tradable",
    )
    resolved_borrow_available = _readonly(
        np.asarray(borrow_available, dtype=np.bool_) & active,
        shape=shape,
        dtype=np.dtype(np.bool_),
        field="borrow_available",
    )
    resolved_buy_allowed = _readonly(
        np.asarray(buy_allowed, dtype=np.bool_) & active,
        shape=shape,
        dtype=np.dtype(np.bool_),
        field="buy_allowed",
    )
    resolved_sell_allowed = _readonly(
        np.asarray(sell_allowed, dtype=np.bool_) & active,
        shape=shape,
        dtype=np.dtype(np.bool_),
        field="sell_allowed",
    )
    resolved_mark = (
        close_array
        if mark_price is None
        else _readonly(
            mark_price, shape=shape, dtype=np.dtype(np.float64), field="mark_price"
        )
    )
    resolved_index = (
        close_array
        if index_price is None
        else _readonly(
            index_price, shape=shape, dtype=np.dtype(np.float64), field="index_price"
        )
    )
    return MarketEconomicSemantics(
        symbol_active=active_ro,
        asset_active=active_ro,
        tradable=tradable,
        information_available=information,
        available_at=resolved_available_at,
        fee_rate=_readonly(
            fee_rate, shape=shape, dtype=np.dtype(np.float64), field="fee_rate"
        ),
        maker_fee_rate=_readonly(
            maker_fee_rate,
            shape=shape,
            dtype=np.dtype(np.float64),
            field="maker_fee_rate",
        ),
        taker_fee_rate=_readonly(
            taker_fee_rate,
            shape=shape,
            dtype=np.dtype(np.float64),
            field="taker_fee_rate",
        ),
        spread_rate=_readonly(
            spread_rate, shape=shape, dtype=np.dtype(np.float64), field="spread_rate"
        ),
        max_participation_rate=_readonly(
            max_participation_rate,
            shape=shape,
            dtype=np.dtype(np.float64),
            field="max_participation_rate",
        ),
        minimum_notional=_readonly(
            minimum, shape=shape, dtype=np.dtype(np.float64), field="minimum_notional"
        ),
        lot_size=_readonly(
            lot, shape=shape, dtype=np.dtype(np.float64), field="lot_size"
        ),
        tick_size=_readonly(
            tick, shape=shape, dtype=np.dtype(np.float64), field="tick_size"
        ),
        borrow_available=resolved_borrow_available,
        borrow_rate=_readonly(
            borrow_rate, shape=shape, dtype=np.dtype(np.float64), field="borrow_rate"
        ),
        funding_due=_readonly(
            funding_counts > 0,
            shape=shape,
            dtype=np.dtype(np.bool_),
            field="funding_due",
        ),
        buy_allowed=resolved_buy_allowed,
        sell_allowed=resolved_sell_allowed,
        mark_price=resolved_mark,
        index_price=resolved_index,
    )


__all__ = ["MarketEconomicSemantics", "build_market_economic_semantics"]
