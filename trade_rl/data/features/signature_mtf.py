"""Causal native-clock rolling Path Signatures aligned only after computation.

Signatures are computed on a symbol's *native* completed-bar path. In particular,
15m points are not first sampled at a 1h decision clock. A bounded two-stack
Chen-product queue makes rolling windows linear in the number of native bars.
This is opt-in software machinery, not an economic or execution model.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from datetime import timezone
from itertools import product

import numpy as np

from trade_rl.artifacts import content_digest
from trade_rl.data.contracts import InstrumentContract, timeframe_hours
from trade_rl.data.identity import content_and_arrays_digest
from trade_rl.data.market import MarketDataset
from trade_rl.data.source import MultiTimeframeMarketDataSource, RawMarketSeries

_NS_PER_HOUR = 3_600_000_000_000
_SCHEMA = "native_mtf_path_signature_v1"


@dataclass(frozen=True, slots=True)
class SignatureClock:
    """One native timeframe's bounded Signature and as-of carry contract."""

    timeframe: str
    window_bars: int = 24
    depth: int = 2
    include_volume: bool = False
    max_staleness_hours: float = 24.0

    def __post_init__(self) -> None:
        timeframe_hours(self.timeframe)
        if (
            isinstance(self.window_bars, bool)
            or not isinstance(self.window_bars, int)
            or not 3 <= self.window_bars <= 512
        ):
            raise ValueError("signature window_bars must be in [3, 512]")
        if (
            isinstance(self.depth, bool)
            or not isinstance(self.depth, int)
            or not 1 <= self.depth <= 3
        ):
            raise ValueError("signature depth must be in [1, 3]")
        if not isinstance(self.include_volume, bool):
            raise ValueError("include_volume must be boolean")
        if (
            isinstance(self.max_staleness_hours, bool)
            or not isinstance(self.max_staleness_hours, (int, float))
            or not math.isfinite(self.max_staleness_hours)
            or self.max_staleness_hours <= 0.0
        ):
            raise ValueError("max_staleness_hours must be finite and positive")

    @property
    def names(self) -> tuple[str, ...]:
        labels = ("t", "p", "v") if self.include_volume else ("t", "p")
        prefix = (
            f"{self.timeframe}__path_sig_v1_w{self.window_bars}"
            f"_d{self.depth}_{''.join(labels)}"
        )
        return tuple(
            f"{prefix}_{''.join(word)}"
            for level in range(1, self.depth + 1)
            for word in product(labels, repeat=level)
        )

    def payload(self) -> dict[str, object]:
        return {
            "timeframe": self.timeframe,
            "window_bars": self.window_bars,
            "depth": self.depth,
            "include_volume": self.include_volume,
            "max_staleness_hours": float(self.max_staleness_hours),
        }


# The level-0 tensor is (1,). Level k has d**k components in word order.
type Tensor = tuple[np.ndarray, ...]


def _segment(displacement: np.ndarray, depth: int) -> Tensor:
    items = [np.array([1.0], dtype=np.float64), displacement]
    for level in range(2, depth + 1):
        items.append(np.kron(items[-1], displacement) / float(level))
    return tuple(items)


def _chen(left: Tensor, right: Tensor) -> Tensor:
    """Associative (in exact arithmetic) truncated tensor product."""
    depth = len(left) - 1
    result = [np.array([1.0], dtype=np.float64)]
    for level in range(1, depth + 1):
        total = np.zeros_like(right[level])
        for index in range(level + 1):
            total += np.kron(left[index], right[level - index])
        result.append(total)
    return tuple(result)


class _RollingChenQueue:
    """Amortized O(1) push/pop and O(1) aggregate per native segment."""

    def __init__(self, depth: int) -> None:
        self.depth = depth
        self.front: list[tuple[Tensor, Tensor]] = []
        self.back: list[tuple[Tensor, Tensor]] = []
        self.size = 0

    def clear(self) -> None:
        self.front.clear()
        self.back.clear()
        self.size = 0

    def push(self, delta: np.ndarray) -> None:
        segment = _segment(delta, self.depth)
        aggregate = _chen(self.back[-1][1], segment) if self.back else segment
        self.back.append((segment, aggregate))
        self.size += 1

    def pop_oldest(self) -> None:
        if not self.front:
            while self.back:
                segment, _ = self.back.pop()
                aggregate = _chen(segment, self.front[-1][1]) if self.front else segment
                self.front.append((segment, aggregate))
        if not self.front:
            raise ValueError("cannot pop from empty Signature queue")
        self.front.pop()
        self.size -= 1

    def signature(self) -> np.ndarray:
        if not self.size:
            raise ValueError("cannot read empty Signature queue")
        if self.front and self.back:
            levels = _chen(self.front[-1][1], self.back[-1][1])
        elif self.front:
            levels = self.front[-1][1]
        else:
            levels = self.back[-1][1]
        return np.concatenate(levels[1:])


def _native_signature_events(
    raw: RawMarketSeries,
    contract: InstrumentContract,
    clock: SignatureClock,
) -> tuple[np.ndarray, np.ndarray]:
    """Only on-time consecutive native closes may enter the rolling path."""
    assert raw.available_at is not None
    n = len(raw.timestamps)
    width = len(clock.names)
    values = np.zeros((n, width), dtype=np.float32)
    valid = np.zeros(n, dtype=np.bool_)
    step_ns = round(timeframe_hours(clock.timeframe) * _NS_PER_HOUR)
    dates = raw.timestamps.astype("datetime64[ns]").astype(np.int64)
    from_time = np.datetime64(
        contract.listed_at.astimezone(timezone.utc).replace(tzinfo=None), "ns"
    )
    good = (
        raw.tradable
        & (raw.available_at <= raw.timestamps)
        & (raw.timestamps >= from_time)
    )
    if contract.delisted_at is not None:
        until_time = np.datetime64(
            contract.delisted_at.astimezone(timezone.utc).replace(tzinfo=None),
            "ns",
        )
        good &= raw.timestamps < until_time
    if clock.include_volume:
        good &= raw.volume > 0.0

    queue = _RollingChenQueue(clock.depth)
    previous_price = 0.0
    previous_volume = 0.0
    previous_valid = False
    normalised_step = 1.0 / (clock.window_bars - 1)

    for index in range(n):
        if not good[index]:
            queue.clear()
            previous_valid = False
            continue
        price = math.log(float(raw.close[index]))
        volume = math.log(float(raw.volume[index])) if clock.include_volume else 0.0
        if not previous_valid or dates[index] - dates[index - 1] != step_ns:
            queue.clear()
        else:
            delta = np.array(
                [normalised_step, price - previous_price, volume - previous_volume]
                if clock.include_volume
                else [normalised_step, price - previous_price],
                dtype=np.float64,
            )
            queue.push(delta)
            if queue.size > clock.window_bars - 1:
                queue.pop_oldest()
            if queue.size == clock.window_bars - 1:
                signature = queue.signature()
                if np.isfinite(signature).all():
                    cast = signature.astype(np.float32)
                    if np.isfinite(cast).all():
                        values[index] = cast
                        valid[index] = True
        previous_price = price
        previous_volume = volume
        previous_valid = True
    return values, valid


def _align_events(
    raw: RawMarketSeries,
    events: np.ndarray,
    event_valid: np.ndarray,
    dataset: MarketDataset,
    symbol_index: int,
    max_staleness_hours: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Carry latest completed native event, never a future/native partial bar."""
    n = dataset.n_bars
    width = events.shape[1]
    values = np.zeros((n, width), dtype=np.float32)
    available = np.zeros((n, width), dtype=np.bool_)
    age = np.full((n, width), max_staleness_hours, dtype=np.float32)
    staleness = np.ones((n, width), dtype=np.float32)
    native_valid = np.flatnonzero(event_valid)
    if not native_valid.size:
        return values, available, age, staleness

    base_ns = dataset.timestamps.astype("datetime64[ns]").astype(np.int64)
    native_ns = raw.timestamps.astype("datetime64[ns]").astype(np.int64)
    chosen_positions = (
        np.searchsorted(native_ns[native_valid], base_ns, side="right") - 1
    )
    good_base = (
        dataset.resolved_array("symbol_active")[:, symbol_index]
        & dataset.resolved_array("information_available")[:, symbol_index]
        & dataset.tradable[:, symbol_index]
    )
    for base_index in np.flatnonzero(good_base & (chosen_positions >= 0)):
        native_index = int(native_valid[chosen_positions[base_index]])
        hours = float(base_ns[base_index] - native_ns[native_index]) / _NS_PER_HOUR
        if hours < -1e-12:
            raise ValueError("native Signature event is later than decision")
        age[base_index, :] = hours
        staleness[base_index, :] = min(hours / max_staleness_hours, 1.0)
        if hours <= max_staleness_hours + 1e-12:
            values[base_index, :] = events[native_index]
            available[base_index, :] = True
    return values, available, age, staleness


def with_native_multitimeframe_signatures(
    dataset: MarketDataset,
    source: MultiTimeframeMarketDataSource,
    instruments: tuple[InstrumentContract, ...],
    *,
    clocks: tuple[SignatureClock, ...],
) -> MarketDataset:
    """Compute independently per native clock, then as-of align to base decisions.

    Raw native source bytes are bound by content digests of the actually consumed
    columns. This does not establish provider publication/collection provenance.
    The destination must already be an identity-verified MarketDataset.
    """
    if not dataset.identity_verified:
        raise ValueError("native Signature requires identity-verified Dataset")
    if not isinstance(source, MultiTimeframeMarketDataSource):
        raise ValueError("source must provide native-timeframe bars")
    if tuple(contract.symbol for contract in instruments) != dataset.symbols:
        raise ValueError("instruments must exactly match Dataset symbol order")
    if not clocks or not all(isinstance(clock, SignatureClock) for clock in clocks):
        raise ValueError("clocks must be a nonempty tuple of SignatureClock")
    names = tuple(name for clock in clocks for name in clock.names)
    if len(set(names)) != len(names) or set(names) & set(dataset.feature_names):
        raise ValueError("native Signature feature names must be unique")

    output = np.zeros((dataset.n_bars, dataset.n_symbols, len(names)), dtype=np.float32)
    mask = np.zeros_like(output, dtype=np.bool_)
    age = np.ones_like(output, dtype=np.float32)
    staleness = np.ones_like(output, dtype=np.float32)
    source_digests: list[dict[str, str]] = []
    cache: dict[tuple[str, str], RawMarketSeries] = {}

    for symbol_index, contract in enumerate(instruments):
        offset = 0
        for clock in clocks:
            key = (contract.symbol, clock.timeframe)
            raw = cache.get(key)
            if raw is None:
                raw = source.load_timeframe(*key)
                cache[key] = raw
            if raw.timestamps[-1] > dataset.timestamps[-1]:
                raise ValueError(
                    "native Signature source extends beyond Dataset time scope"
                )
            assert raw.available_at is not None
            digest = content_and_arrays_digest(
                {
                    "schema": "native_signature_source_arrays_v1",
                    "symbol": contract.symbol,
                    "timeframe": clock.timeframe,
                    "instrument": contract.canonical_payload(),
                },
                (
                    ("timestamps", raw.timestamps),
                    ("available_at", raw.available_at),
                    ("close", raw.close),
                    ("volume", raw.volume),
                    ("tradable", raw.tradable),
                ),
            )
            source_digests.append(
                {
                    "symbol": contract.symbol,
                    "timeframe": clock.timeframe,
                    "sha256": digest,
                }
            )
            events, event_valid = _native_signature_events(raw, contract, clock)
            aligned = _align_events(
                raw,
                events,
                event_valid,
                dataset,
                symbol_index,
                clock.max_staleness_hours,
            )
            width = len(clock.names)
            stop = offset + width
            output[:, symbol_index, offset:stop] = aligned[0]
            mask[:, symbol_index, offset:stop] = aligned[1]
            age[:, symbol_index, offset:stop] = aligned[2]
            staleness[:, symbol_index, offset:stop] = aligned[3]
            offset = stop

    definition: dict[str, object] = {
        "schema": _SCHEMA,
        "source_dataset_id": dataset.dataset_id,
        "clock_specs": [clock.payload() for clock in clocks],
        "source_digests": source_digests,
        "window_semantics": "native_completed_consecutive_bars",
        "path_semantics": "piecewise_linear_normalised_bar_time_log_close_volume",
        "availability": "source_on_time_then_native_asof_on_base_decision",
    }
    missing = ~mask
    augmented = replace(
        dataset,
        identity_payload_json=None,
        features=np.concatenate((dataset.features, output), axis=2),
        feature_names=dataset.feature_names + names,
        feature_available=np.concatenate((dataset.feature_available, mask), axis=2),
        feature_staleness=np.concatenate(
            (dataset.resolved_array("feature_staleness"), staleness), axis=2
        ),
        feature_staleness_hours=np.concatenate(
            (dataset.resolved_array("feature_staleness_hours"), age), axis=2
        ),
        feature_missing_reason=np.concatenate(
            (
                dataset.resolved_array("feature_missing_reason"),
                missing.astype(np.int16),
            ),
            axis=2,
        ),
        feature_config_digest=content_digest(definition),
    )
    return augmented.with_content_identity(definition)


__all__ = ["SignatureClock", "with_native_multitimeframe_signatures"]
