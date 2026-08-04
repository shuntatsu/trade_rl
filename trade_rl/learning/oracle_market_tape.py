"""Path-independent market inputs for bounded Oracle Bellman solvers."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Final

import numpy as np

from trade_rl.artifacts.hashing import content_digest
from trade_rl.data.contracts import VolumeUnit
from trade_rl.data.market import MarketDataset
from trade_rl.domain.common import require_sha256
from trade_rl.learning.oracle_bellman_contracts import OracleBellmanParameters

ORACLE_MARKET_TAPE_SCHEMA: Final = "oracle_market_tape_v1"

_ARRAY_FIELDS: Final = (
    "raw_position_factor",
    "equity_position_factor",
    "mark_open_ratio",
    "active",
    "tradable",
    "buy_allowed",
    "sell_allowed",
    "borrow_available",
    "market_notional",
    "participation_capacity",
    "minimum_notional",
    "base_unit_cost",
    "funding_due_rate",
    "borrow_rate",
    "dividend_open_ratio",
    "cash_rate",
    "elapsed_year_fraction",
)
_BOOL_FIELDS: Final = frozenset(
    {"active", "tradable", "buy_allowed", "sell_allowed", "borrow_available"}
)
_VECTOR_FIELDS: Final = frozenset({"cash_rate", "elapsed_year_fraction"})
_NONNEGATIVE_FIELDS: Final = frozenset(
    {"market_notional", "participation_capacity", "minimum_notional"}
)


def _array_identity(value: np.ndarray) -> dict[str, object]:
    contiguous = np.ascontiguousarray(value)
    return {
        "dtype": str(contiguous.dtype),
        "sha256": hashlib.sha256(contiguous.tobytes(order="C")).hexdigest(),
        "shape": tuple(int(size) for size in contiguous.shape),
    }


def _readonly_array(
    value: object,
    *,
    field: str,
    shape: tuple[int, ...],
    boolean: bool,
) -> np.ndarray:
    raw = np.asarray(value)
    if raw.shape != shape:
        raise ValueError(f"{field} shape does not match the market tape")
    if boolean:
        if not np.issubdtype(raw.dtype, np.bool_):
            raise ValueError(f"{field} must contain booleans")
        boolean_array = np.asarray(raw, dtype=np.bool_).copy(order="C")
        boolean_array.setflags(write=False)
        return boolean_array
    if not np.issubdtype(raw.dtype, np.number) or np.issubdtype(raw.dtype, np.bool_):
        raise ValueError(f"{field} must contain numeric values")
    numeric_array = np.asarray(raw, dtype=np.float64).copy(order="C")
    if not np.isfinite(numeric_array).all():
        raise ValueError(f"{field} must contain finite values")
    if field in _NONNEGATIVE_FIELDS and np.any(numeric_array < 0.0):
        raise ValueError(f"{field} must be non-negative")
    numeric_array.setflags(write=False)
    return numeric_array


@dataclass(frozen=True, slots=True)
class OracleMarketTape:
    """Immutable path-independent inputs aligned to one train range."""

    raw_position_factor: np.ndarray
    equity_position_factor: np.ndarray
    mark_open_ratio: np.ndarray
    active: np.ndarray
    tradable: np.ndarray
    buy_allowed: np.ndarray
    sell_allowed: np.ndarray
    borrow_available: np.ndarray
    market_notional: np.ndarray
    participation_capacity: np.ndarray
    minimum_notional: np.ndarray
    base_unit_cost: np.ndarray
    funding_due_rate: np.ndarray
    borrow_rate: np.ndarray
    dividend_open_ratio: np.ndarray
    cash_rate: np.ndarray
    elapsed_year_fraction: np.ndarray
    start: int
    stop: int
    dataset_id: str
    digest: str = ""
    schema_version: str = ORACLE_MARKET_TAPE_SCHEMA

    def __post_init__(self) -> None:
        dataset_id = require_sha256(self.dataset_id, field="dataset_id")
        if (
            isinstance(self.start, bool)
            or isinstance(self.stop, bool)
            or not isinstance(self.start, int)
            or not isinstance(self.stop, int)
            or self.start < 0
            or self.stop <= self.start + 1
        ):
            raise ValueError("market tape range must contain at least one decision")
        if self.schema_version != ORACLE_MARKET_TAPE_SCHEMA:
            raise ValueError("unsupported Oracle market tape schema")
        steps = self.stop - self.start - 1
        raw_position = np.asarray(self.raw_position_factor)
        if raw_position.ndim != 2 or raw_position.shape[0] != steps:
            raise ValueError("raw_position_factor shape does not match the market tape")
        symbol_count = raw_position.shape[1]
        if symbol_count <= 0:
            raise ValueError("market tape must contain at least one symbol")

        resolved: dict[str, np.ndarray] = {}
        for field in _ARRAY_FIELDS:
            shape = (steps,) if field in _VECTOR_FIELDS else (steps, symbol_count)
            resolved[field] = _readonly_array(
                getattr(self, field),
                field=field,
                shape=shape,
                boolean=field in _BOOL_FIELDS,
            )
        if np.any(resolved["raw_position_factor"] <= 0.0):
            raise ValueError("raw_position_factor must be positive")
        if np.any(resolved["equity_position_factor"] < 0.0):
            raise ValueError("equity_position_factor must be non-negative")
        if np.any(resolved["mark_open_ratio"] <= 0.0):
            raise ValueError("mark_open_ratio must be positive")

        expected = content_digest(
            {
                "arrays": {
                    field: _array_identity(resolved[field]) for field in _ARRAY_FIELDS
                },
                "dataset_id": dataset_id,
                "schema_version": self.schema_version,
                "start": self.start,
                "stop": self.stop,
            }
        )
        if self.digest and self.digest != expected:
            raise ValueError("Oracle market tape digest mismatch")
        for field, value in resolved.items():
            object.__setattr__(self, field, value)
        object.__setattr__(self, "dataset_id", dataset_id)
        object.__setattr__(self, "digest", expected)

    @property
    def steps(self) -> int:
        return self.stop - self.start - 1

    @property
    def symbol_count(self) -> int:
        return int(self.raw_position_factor.shape[1])

    @property
    def arrays(self) -> dict[str, np.ndarray]:
        return {field: getattr(self, field) for field in _ARRAY_FIELDS}


def _validate_range(
    dataset: MarketDataset,
    train_range: tuple[int, int],
) -> tuple[int, int]:
    if (
        len(train_range) != 2
        or isinstance(train_range[0], bool)
        or isinstance(train_range[1], bool)
        or not isinstance(train_range[0], int)
        or not isinstance(train_range[1], int)
    ):
        raise ValueError("market tape range must be a pair of integer indices")
    start, stop = train_range
    if not 0 <= start < stop - 1 < dataset.n_bars:
        raise ValueError("market tape range must contain in-dataset decisions")
    return start, stop


def oracle_open_market_factors(
    dataset: MarketDataset,
    close_index: int | np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return raw/equity position factors and active masks for next-open gaps."""

    raw_indices = np.asarray(close_index)
    if raw_indices.ndim > 1 or not np.issubdtype(raw_indices.dtype, np.integer):
        raise ValueError("close_index must contain integer indices")
    indices = np.asarray(raw_indices, dtype=np.int64)
    if np.any(indices < 0) or np.any(indices >= dataset.n_bars - 1):
        raise ValueError("close_index is outside the dataset")
    execution_indices = indices + 1
    previous_mark = dataset.resolved_array("mark_price")[indices]
    split = dataset.resolved_array("split_factor")[execution_indices]
    raw_position_factor = dataset.open[execution_indices] * split / previous_mark
    active = dataset.resolved_array("asset_active")[execution_indices]
    recovery = dataset.resolved_array("delisting_recovery")[execution_indices]
    equity_position_factor = np.where(
        active,
        raw_position_factor,
        raw_position_factor * recovery,
    )
    return raw_position_factor, equity_position_factor, active


def _market_notional_matrix(
    dataset: MarketDataset,
    *,
    close_indices: np.ndarray,
    execution_indices: np.ndarray,
) -> np.ndarray:
    volume = dataset.volume[close_indices]
    prices = dataset.open[execution_indices]
    multipliers = dataset.resolved_array("contract_multipliers")
    result = np.empty_like(volume, dtype=np.float64)
    for symbol_index, unit in enumerate(dataset.volume_units):
        if unit is VolumeUnit.QUOTE_NOTIONAL:
            result[:, symbol_index] = volume[:, symbol_index]
        elif unit is VolumeUnit.BASE_ASSET:
            result[:, symbol_index] = volume[:, symbol_index] * prices[:, symbol_index]
        else:
            result[:, symbol_index] = (
                volume[:, symbol_index]
                * multipliers[symbol_index]
                * prices[:, symbol_index]
            )
    return result


def build_oracle_market_tape(
    dataset: MarketDataset,
    train_range: tuple[int, int],
    parameters: OracleBellmanParameters,
) -> OracleMarketTape:
    """Precompute every market-side input independent of the chosen DP path."""

    if not isinstance(dataset, MarketDataset):
        raise ValueError("dataset must be MarketDataset")
    if not isinstance(parameters, OracleBellmanParameters):
        raise ValueError("parameters must be OracleBellmanParameters")
    start, stop = _validate_range(dataset, train_range)
    close_indices = np.arange(start, stop - 1, dtype=np.int64)
    execution_indices = close_indices + 1

    raw_position_factor, equity_position_factor, active = oracle_open_market_factors(
        dataset, close_indices
    )
    execution_open = dataset.open[execution_indices]
    market_notional = _market_notional_matrix(
        dataset,
        close_indices=close_indices,
        execution_indices=execution_indices,
    )
    participation_limit = np.minimum(
        dataset.resolved_array("max_participation_rate")[execution_indices],
        parameters.execution_cost.max_participation_rate,
    )
    venue_fee = (
        parameters.execution_cost.taker_fee_rate
        + dataset.resolved_array("taker_fee_rate")[execution_indices]
    )
    base_unit_cost = parameters.execution_cost.multiplier * (
        parameters.execution_cost.fee_rate
        + dataset.resolved_array("fee_rate")[execution_indices]
        + venue_fee
        + parameters.execution_cost.spread_rate
        + dataset.resolved_array("spread_rate")[execution_indices]
    )
    elapsed_year_fraction = np.asarray(
        [
            dataset.elapsed_year_fraction(int(close_index), int(execution_index))
            for close_index, execution_index in zip(
                close_indices, execution_indices, strict=True
            )
        ],
        dtype=np.float64,
    )

    return OracleMarketTape(
        raw_position_factor=raw_position_factor,
        equity_position_factor=equity_position_factor,
        mark_open_ratio=(
            dataset.resolved_array("mark_price")[execution_indices] / execution_open
        ),
        active=active,
        tradable=dataset.tradable[execution_indices],
        buy_allowed=dataset.resolved_array("buy_allowed")[execution_indices],
        sell_allowed=dataset.resolved_array("sell_allowed")[execution_indices],
        borrow_available=dataset.resolved_array("borrow_available")[execution_indices],
        market_notional=market_notional,
        participation_capacity=participation_limit * market_notional,
        minimum_notional=np.maximum(
            dataset.resolved_array("minimum_notional")[execution_indices],
            parameters.execution_cost.minimum_notional,
        ),
        base_unit_cost=base_unit_cost,
        funding_due_rate=(
            dataset.funding_rate[execution_indices]
            * dataset.resolved_array("funding_due")[execution_indices].astype(
                np.float64
            )
        ),
        borrow_rate=dataset.resolved_array("borrow_rate")[execution_indices],
        dividend_open_ratio=(
            dataset.resolved_array("dividend")[execution_indices] / execution_open
        ),
        cash_rate=dataset.resolved_array("cash_rate")[execution_indices],
        elapsed_year_fraction=elapsed_year_fraction,
        start=start,
        stop=stop,
        dataset_id=dataset.dataset_id,
    )


__all__ = [
    "ORACLE_MARKET_TAPE_SCHEMA",
    "OracleMarketTape",
    "build_oracle_market_tape",
    "oracle_open_market_factors",
]
