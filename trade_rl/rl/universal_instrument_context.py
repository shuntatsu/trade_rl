"""Causal continuous instrument context for identity-free Universal policies."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC
from typing import Any, Final

import numpy as np

from trade_rl.artifacts.hashing import content_digest
from trade_rl.data.contracts import InstrumentContract, VolumeUnit
from trade_rl.data.universal_features import UNIVERSAL_INSTRUMENT_DESCRIPTOR_NAMES
from trade_rl.rl.universal_instrument_binding import InstrumentDatasetBinding

UNIVERSAL_INSTRUMENT_CONTEXT_SCHEMA: Final = "instrument_context_v1"
_TRAILING_WINDOW_NS: Final = 30 * 24 * 60 * 60 * 1_000_000_000
_EPSILON: Final = 1e-12


def _single_value(value: object, *, index: int, field: str) -> float:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 1 or index >= array.shape[0]:
        raise ValueError(f"{field} must be a sample-aligned single-symbol matrix")
    resolved = float(array[index, 0])
    if not math.isfinite(resolved):
        raise ValueError(f"{field} must be finite at the decision index")
    return resolved


def _decision_timestamp_ns(dataset: object, index: int) -> int:
    timestamps = np.asarray(getattr(dataset, "timestamps", None))
    if (
        timestamps.ndim != 1
        or not np.issubdtype(timestamps.dtype, np.datetime64)
        or index < 0
        or index >= timestamps.size
    ):
        raise ValueError("dataset timestamps do not cover the decision index")
    return int(timestamps.astype("datetime64[ns]").astype(np.int64)[index])


@dataclass(frozen=True, slots=True)
class CausalInstrumentContextProvider:
    """Derive nine ticker-free continuous descriptors at the current decision."""

    contracts: Mapping[str, InstrumentContract]

    def __post_init__(self) -> None:
        resolved = dict(self.contracts)
        if not resolved:
            raise ValueError("instrument context contracts must not be empty")
        if any(
            not isinstance(symbol, str)
            or not symbol
            or not isinstance(contract, InstrumentContract)
            or contract.symbol != symbol
            for symbol, contract in resolved.items()
        ):
            raise ValueError("instrument context contracts are invalid")
        object.__setattr__(self, "contracts", resolved)

    @property
    def schema_digest(self) -> str:
        return content_digest(
            {
                "descriptor_names": UNIVERSAL_INSTRUMENT_DESCRIPTOR_NAMES,
                "schema_version": UNIVERSAL_INSTRUMENT_CONTEXT_SCHEMA,
            }
        )

    @property
    def digest(self) -> str:
        return content_digest(
            {
                "contracts": tuple(
                    (
                        symbol,
                        contract.listed_at.astimezone(UTC).isoformat(),
                        contract.volume_unit.value,
                    )
                    for symbol, contract in sorted(self.contracts.items())
                ),
                "schema_digest": self.schema_digest,
            }
        )

    def __call__(
        self,
        environment: object,
        binding: InstrumentDatasetBinding,
    ) -> np.ndarray:
        if not isinstance(binding, InstrumentDatasetBinding):
            raise TypeError("binding must be InstrumentDatasetBinding")
        try:
            contract = self.contracts[binding.concrete_symbol]
        except KeyError as error:
            raise ValueError(
                "instrument context contract is missing for routed symbol"
            ) from error
        if contract.volume_unit is not VolumeUnit.QUOTE_NOTIONAL:
            raise ValueError(
                "instrument context requires quote-notional volume semantics"
            )

        dataset: Any = getattr(environment, "dataset", None)
        symbols = tuple(getattr(dataset, "symbols", ()))
        if symbols != (binding.concrete_symbol,):
            raise ValueError("instrument context dataset symbol does not match binding")
        volume_units = tuple(getattr(dataset, "volume_units", ()))
        if volume_units != (VolumeUnit.QUOTE_NOTIONAL,):
            raise ValueError(
                "instrument context requires quote-notional dataset volume"
            )
        index = getattr(environment, "current_index", None)
        if isinstance(index, bool) or not isinstance(index, int) or index < 0:
            raise ValueError("instrument context requires a valid current_index")
        timestamp_ns = _decision_timestamp_ns(dataset, index)
        listed_at = np.datetime64(
            contract.listed_at.astimezone(UTC).replace(tzinfo=None), "ns"
        ).astype(np.int64)
        listing_age_days = max(
            0.0, (timestamp_ns - int(listed_at)) / 86_400_000_000_000
        )

        timestamps_ns = (
            np.asarray(dataset.timestamps).astype("datetime64[ns]").astype(np.int64)
        )
        window_start = int(
            np.searchsorted(
                timestamps_ns, timestamp_ns - _TRAILING_WINDOW_NS, side="left"
            )
        )
        volume = np.asarray(dataset.volume, dtype=np.float64)
        if volume.ndim != 2 or volume.shape[1] != 1 or index >= volume.shape[0]:
            raise ValueError("instrument context volume shape is invalid")
        trailing_quote_notional = volume[window_start : index + 1, 0]
        if not np.isfinite(trailing_quote_notional).all() or np.any(
            trailing_quote_notional < 0.0
        ):
            raise ValueError("instrument context quote-notional history is invalid")

        mark = _single_value(dataset.mark_price, index=index, field="mark_price")
        if mark <= _EPSILON:
            mark = _single_value(dataset.close, index=index, field="close")
        if mark <= _EPSILON:
            raise ValueError("instrument context mark price must be positive")
        equity = float(
            getattr(getattr(environment, "hybrid", None), "portfolio_value", math.nan)
        )
        if not math.isfinite(equity) or equity <= _EPSILON:
            raise ValueError("instrument context requires positive portfolio equity")

        tick_size = _single_value(dataset.tick_size, index=index, field="tick_size")
        lot_size = _single_value(dataset.lot_size, index=index, field="lot_size")
        minimum_notional = _single_value(
            dataset.minimum_notional, index=index, field="minimum_notional"
        )
        fee_rate = _single_value(dataset.fee_rate, index=index, field="fee_rate")
        spread_rate = _single_value(
            dataset.spread_rate, index=index, field="spread_rate"
        )
        max_participation_rate = _single_value(
            dataset.max_participation_rate,
            index=index,
            field="max_participation_rate",
        )
        impact_rate = float(
            getattr(
                getattr(getattr(environment, "config", None), "execution_cost", None),
                "impact_rate",
                math.nan,
            )
        )
        if not math.isfinite(impact_rate) or impact_rate < 0.0:
            raise ValueError(
                "instrument context impact_rate must be finite and non-negative"
            )

        context = np.asarray(
            [
                math.log1p(listing_age_days),
                math.log1p(float(trailing_quote_notional.sum())),
                tick_size / mark,
                lot_size * mark / equity,
                minimum_notional / equity,
                fee_rate,
                spread_rate,
                impact_rate,
                max_participation_rate,
            ],
            dtype=np.float32,
        ).reshape(1, -1)
        if context.shape != (1, len(UNIVERSAL_INSTRUMENT_DESCRIPTOR_NAMES)):
            raise RuntimeError("instrument context descriptor width drifted")
        if not np.isfinite(context).all():
            raise ValueError("instrument context descriptors must be finite")
        return context


__all__ = [
    "UNIVERSAL_INSTRUMENT_CONTEXT_SCHEMA",
    "CausalInstrumentContextProvider",
]
