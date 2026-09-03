"""Metadata-only time preregistration for Universal Trade RL U2."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final

from trade_rl.artifacts.hashing import content_digest
from trade_rl.domain.common import require_sha256
from trade_rl.workflows.universal_trade_rl_universe_manifest import (
    UniversalTradeRLUniverseManifest,
)

UNIVERSAL_TRADE_RL_U2_TIME_PARTITION_SCHEMA: Final = (
    "universal_trade_rl_u2_time_partition_v1"
)
U2_DECISION_STEP_NS: Final = 15 * 60 * 1_000_000_000
U2_BARS_PER_DAY: Final = 96
U2_MINIMUM_COMMON_BARS: Final = 600 * U2_BARS_PER_DAY
U2_SEEN_TIME_PROBE_BARS: Final = 60 * U2_BARS_PER_DAY
U2_EVALUATION_TILE_BARS: Final = 720 * 4
U2_MINIMUM_EVALUATION_TILES: Final = 2

_WINDOW_NAMES: Final = (
    "fit",
    "seen_time_probe",
    "development_future_1",
    "development_future_2",
    "admission_future",
)
_TILED_WINDOW_NAMES: Final = _WINDOW_NAMES[1:]
_WINDOW_KEYS: Final = (
    "name",
    "start_bar_index",
    "stop_bar_index_exclusive",
    "first_timestamp_ns",
    "last_timestamp_ns",
)
_TILE_KEYS: Final = (
    "source_window",
    "tile_index",
    "start_bar_index",
    "stop_bar_index_exclusive",
    "first_timestamp_ns",
    "last_timestamp_ns",
)
_PARTITION_KEYS: Final = (
    "schema_version",
    "universe_manifest_digest",
    "decision_step_ns",
    "common_first_timestamp_ns",
    "common_last_timestamp_ns",
    "common_bar_count",
    "windows",
    "tiles",
    "artifact_digest",
)


def _integer(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an integer and not boolean")
    return value


def _exact_mapping(
    value: object,
    *,
    keys: tuple[str, ...],
    field: str,
) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be an object with exact keys")
    result = {str(key): item for key, item in value.items()}
    if set(result) != set(keys) or len(result) != len(keys):
        raise ValueError(f"{field} must use exact keys")
    return result


def _sequence(value: object, *, field: str) -> tuple[object, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{field} must be an array")
    return tuple(value)


@dataclass(frozen=True, slots=True)
class UniversalTradeRLU2TimeWindow:
    name: str
    start_bar_index: int
    stop_bar_index_exclusive: int
    first_timestamp_ns: int
    last_timestamp_ns: int

    def __post_init__(self) -> None:
        if self.name not in _WINDOW_NAMES:
            raise ValueError("unsupported Universal Trade RL U2 time window")
        start = _integer(self.start_bar_index, field="time window start_bar_index")
        stop = _integer(
            self.stop_bar_index_exclusive,
            field="time window stop_bar_index_exclusive",
        )
        first = _integer(self.first_timestamp_ns, field="time window first_timestamp_ns")
        last = _integer(self.last_timestamp_ns, field="time window last_timestamp_ns")
        if start < 0 or stop <= start:
            raise ValueError("Universal Trade RL U2 time window bar range is invalid")
        if first < 0 or last < first:
            raise ValueError("Universal Trade RL U2 time window timestamps are invalid")
        expected_last = first + (stop - start - 1) * U2_DECISION_STEP_NS
        if last != expected_last:
            raise ValueError("Universal Trade RL U2 time window timestamps drifted")

    @property
    def bar_count(self) -> int:
        return self.stop_bar_index_exclusive - self.start_bar_index

    def to_payload(self) -> dict[str, object]:
        return {
            "name": self.name,
            "start_bar_index": self.start_bar_index,
            "stop_bar_index_exclusive": self.stop_bar_index_exclusive,
            "first_timestamp_ns": self.first_timestamp_ns,
            "last_timestamp_ns": self.last_timestamp_ns,
        }

    @classmethod
    def from_payload(cls, payload: object) -> UniversalTradeRLU2TimeWindow:
        values = _exact_mapping(payload, keys=_WINDOW_KEYS, field="U2 time window")
        name = values["name"]
        if not isinstance(name, str):
            raise ValueError("U2 time window name must be a string")
        return cls(
            name=name,
            start_bar_index=_integer(
                values["start_bar_index"], field="time window start_bar_index"
            ),
            stop_bar_index_exclusive=_integer(
                values["stop_bar_index_exclusive"],
                field="time window stop_bar_index_exclusive",
            ),
            first_timestamp_ns=_integer(
                values["first_timestamp_ns"], field="time window first_timestamp_ns"
            ),
            last_timestamp_ns=_integer(
                values["last_timestamp_ns"], field="time window last_timestamp_ns"
            ),
        )


@dataclass(frozen=True, slots=True)
class UniversalTradeRLU2EpisodeTile:
    source_window: str
    tile_index: int
    start_bar_index: int
    stop_bar_index_exclusive: int
    first_timestamp_ns: int
    last_timestamp_ns: int

    def __post_init__(self) -> None:
        if self.source_window not in _TILED_WINDOW_NAMES:
            raise ValueError("unsupported Universal Trade RL U2 tile source window")
        index = _integer(self.tile_index, field="U2 tile_index")
        start = _integer(self.start_bar_index, field="U2 tile start_bar_index")
        stop = _integer(
            self.stop_bar_index_exclusive,
            field="U2 tile stop_bar_index_exclusive",
        )
        first = _integer(self.first_timestamp_ns, field="U2 tile first_timestamp_ns")
        last = _integer(self.last_timestamp_ns, field="U2 tile last_timestamp_ns")
        if index < 0 or start < 0:
            raise ValueError("Universal Trade RL U2 tile index is invalid")
        if stop - start != U2_EVALUATION_TILE_BARS:
            raise ValueError("Universal Trade RL U2 tile must span exactly 720h")
        if first < 0 or last != first + (U2_EVALUATION_TILE_BARS - 1) * U2_DECISION_STEP_NS:
            raise ValueError("Universal Trade RL U2 tile timestamps drifted")

    @property
    def bar_count(self) -> int:
        return self.stop_bar_index_exclusive - self.start_bar_index

    def to_payload(self) -> dict[str, object]:
        return {
            "source_window": self.source_window,
            "tile_index": self.tile_index,
            "start_bar_index": self.start_bar_index,
            "stop_bar_index_exclusive": self.stop_bar_index_exclusive,
            "first_timestamp_ns": self.first_timestamp_ns,
            "last_timestamp_ns": self.last_timestamp_ns,
        }

    @classmethod
    def from_payload(cls, payload: object) -> UniversalTradeRLU2EpisodeTile:
        values = _exact_mapping(payload, keys=_TILE_KEYS, field="U2 episode tile")
        source_window = values["source_window"]
        if not isinstance(source_window, str):
            raise ValueError("U2 episode tile source_window must be a string")
        return cls(
            source_window=source_window,
            tile_index=_integer(values["tile_index"], field="U2 tile_index"),
            start_bar_index=_integer(
                values["start_bar_index"], field="U2 tile start_bar_index"
            ),
            stop_bar_index_exclusive=_integer(
                values["stop_bar_index_exclusive"],
                field="U2 tile stop_bar_index_exclusive",
            ),
            first_timestamp_ns=_integer(
                values["first_timestamp_ns"], field="U2 tile first_timestamp_ns"
            ),
            last_timestamp_ns=_integer(
                values["last_timestamp_ns"], field="U2 tile last_timestamp_ns"
            ),
        )


@dataclass(frozen=True, slots=True)
class UniversalTradeRLU2TimePartition:
    universe_manifest_digest: str
    common_first_timestamp_ns: int
    common_last_timestamp_ns: int
    common_bar_count: int
    windows: tuple[UniversalTradeRLU2TimeWindow, ...]
    tiles: tuple[UniversalTradeRLU2EpisodeTile, ...]
    decision_step_ns: int = U2_DECISION_STEP_NS
    schema_version: str = UNIVERSAL_TRADE_RL_U2_TIME_PARTITION_SCHEMA
    digest: str = ""

    def __post_init__(self) -> None:
        require_sha256(
            self.universe_manifest_digest,
            field="U2 time partition universe manifest digest",
        )
        if self.schema_version != UNIVERSAL_TRADE_RL_U2_TIME_PARTITION_SCHEMA:
            raise ValueError("unsupported Universal Trade RL U2 time partition schema")
        if self.decision_step_ns != U2_DECISION_STEP_NS:
            raise ValueError("Universal Trade RL U2 decision step must be exactly 15m")
        first = _integer(
            self.common_first_timestamp_ns,
            field="U2 common_first_timestamp_ns",
        )
        last = _integer(
            self.common_last_timestamp_ns,
            field="U2 common_last_timestamp_ns",
        )
        count = _integer(self.common_bar_count, field="U2 common_bar_count")
        if first < 0 or last < first or count < U2_MINIMUM_COMMON_BARS:
            raise ValueError("Universal Trade RL U2 common history must cover at least 600 days")
        if last != first + (count - 1) * U2_DECISION_STEP_NS:
            raise ValueError("Universal Trade RL U2 common 15m grid drifted")

        windows = tuple(self.windows)
        if tuple(window.name for window in windows) != _WINDOW_NAMES:
            raise ValueError("Universal Trade RL U2 time windows are not canonical")
        if any(not isinstance(window, UniversalTradeRLU2TimeWindow) for window in windows):
            raise TypeError("Universal Trade RL U2 time window contract is invalid")
        by_name = {window.name: window for window in windows}
        fit = by_name["fit"]
        seen = by_name["seen_time_probe"]
        d1 = by_name["development_future_1"]
        d2 = by_name["development_future_2"]
        admission = by_name["admission_future"]
        if fit.start_bar_index != 0 or admission.stop_bar_index_exclusive != count:
            raise ValueError("Universal Trade RL U2 partition does not close common history")
        if not (
            fit.stop_bar_index_exclusive == d1.start_bar_index
            and d1.stop_bar_index_exclusive == d2.start_bar_index
            and d2.stop_bar_index_exclusive == admission.start_bar_index
        ):
            raise ValueError("Universal Trade RL U2 primary time windows overlap or gap")
        if (
            seen.bar_count != U2_SEEN_TIME_PROBE_BARS
            or seen.stop_bar_index_exclusive != fit.stop_bar_index_exclusive
            or seen.start_bar_index < fit.start_bar_index
        ):
            raise ValueError("Universal Trade RL U2 seen-time probe contract drifted")
        for window in windows:
            expected_first = first + window.start_bar_index * U2_DECISION_STEP_NS
            expected_last = first + (window.stop_bar_index_exclusive - 1) * U2_DECISION_STEP_NS
            if (
                window.first_timestamp_ns != expected_first
                or window.last_timestamp_ns != expected_last
            ):
                raise ValueError("Universal Trade RL U2 window/common-grid identity drifted")

        tiles = tuple(self.tiles)
        if any(not isinstance(tile, UniversalTradeRLU2EpisodeTile) for tile in tiles):
            raise TypeError("Universal Trade RL U2 episode tile contract is invalid")
        for window_name in _TILED_WINDOW_NAMES:
            window = by_name[window_name]
            observed = tuple(tile for tile in tiles if tile.source_window == window_name)
            expected_count = window.bar_count // U2_EVALUATION_TILE_BARS
            if expected_count < U2_MINIMUM_EVALUATION_TILES:
                raise ValueError(
                    "Universal Trade RL U2 evaluation window requires at least two 720h tiles"
                )
            if len(observed) != expected_count:
                raise ValueError("Universal Trade RL U2 evaluation tile closure drifted")
            for index, tile in enumerate(observed):
                expected_start = window.start_bar_index + index * U2_EVALUATION_TILE_BARS
                if (
                    tile.tile_index != index
                    or tile.start_bar_index != expected_start
                    or tile.stop_bar_index_exclusive
                    != expected_start + U2_EVALUATION_TILE_BARS
                ):
                    raise ValueError("Universal Trade RL U2 evaluation tile order drifted")
                expected_first = first + expected_start * U2_DECISION_STEP_NS
                if tile.first_timestamp_ns != expected_first:
                    raise ValueError("Universal Trade RL U2 evaluation tile time drifted")
        if tuple(
            (tile.source_window, tile.tile_index) for tile in tiles
        ) != tuple(sorted((tile.source_window, tile.tile_index) for tile in tiles)):
            raise ValueError("Universal Trade RL U2 episode tiles are not canonical")

        object.__setattr__(self, "windows", windows)
        object.__setattr__(self, "tiles", tiles)
        expected = content_digest(self.to_payload(include_digest=False))
        if self.digest:
            require_sha256(self.digest, field="U2 time partition artifact digest")
            if self.digest != expected:
                raise ValueError("Universal Trade RL U2 time partition digest mismatch")
        object.__setattr__(self, "digest", expected)

    @property
    def fit_end_ns(self) -> int:
        return self.window("fit").last_timestamp_ns

    def window(self, name: str) -> UniversalTradeRLU2TimeWindow:
        for window in self.windows:
            if window.name == name:
                return window
        raise KeyError(name)

    def tiles_for(self, name: str) -> tuple[UniversalTradeRLU2EpisodeTile, ...]:
        if name not in _TILED_WINDOW_NAMES:
            raise KeyError(name)
        return tuple(tile for tile in self.tiles if tile.source_window == name)

    def to_payload(self, *, include_digest: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema_version": self.schema_version,
            "universe_manifest_digest": self.universe_manifest_digest,
            "decision_step_ns": self.decision_step_ns,
            "common_first_timestamp_ns": self.common_first_timestamp_ns,
            "common_last_timestamp_ns": self.common_last_timestamp_ns,
            "common_bar_count": self.common_bar_count,
            "windows": tuple(window.to_payload() for window in self.windows),
            "tiles": tuple(tile.to_payload() for tile in self.tiles),
        }
        if include_digest:
            payload["artifact_digest"] = self.digest
        return payload

    @classmethod
    def from_payload(cls, payload: object) -> UniversalTradeRLU2TimePartition:
        values = _exact_mapping(payload, keys=_PARTITION_KEYS, field="U2 time partition")
        schema = values["schema_version"]
        universe_digest = values["universe_manifest_digest"]
        artifact_digest = values["artifact_digest"]
        if not isinstance(schema, str):
            raise ValueError("U2 time partition schema_version must be a string")
        if not isinstance(universe_digest, str) or not isinstance(artifact_digest, str):
            raise ValueError("U2 time partition digest fields must be strings")
        windows = tuple(
            UniversalTradeRLU2TimeWindow.from_payload(item)
            for item in _sequence(values["windows"], field="U2 time partition windows")
        )
        tiles = tuple(
            UniversalTradeRLU2EpisodeTile.from_payload(item)
            for item in _sequence(values["tiles"], field="U2 time partition tiles")
        )
        return cls(
            universe_manifest_digest=universe_digest,
            common_first_timestamp_ns=_integer(
                values["common_first_timestamp_ns"],
                field="U2 common_first_timestamp_ns",
            ),
            common_last_timestamp_ns=_integer(
                values["common_last_timestamp_ns"],
                field="U2 common_last_timestamp_ns",
            ),
            common_bar_count=_integer(
                values["common_bar_count"], field="U2 common_bar_count"
            ),
            windows=windows,
            tiles=tiles,
            decision_step_ns=_integer(
                values["decision_step_ns"], field="U2 decision_step_ns"
            ),
            schema_version=schema,
            digest=artifact_digest,
        )


def _validate_dense_15m_source_metadata(manifest: UniversalTradeRLUniverseManifest) -> None:
    active_entries = tuple(entry for entry in manifest.entries if entry.role is not None)
    if not active_entries:
        raise ValueError("Universal Trade RL U2 requires active universe sources")
    for entry in active_entries:
        if (
            entry.first_timestamp_ns % U2_DECISION_STEP_NS != 0
            or entry.last_timestamp_ns % U2_DECISION_STEP_NS != 0
        ):
            raise ValueError(f"U2 source {entry.symbol} timestamps must align to 15m")
        delta = entry.last_timestamp_ns - entry.first_timestamp_ns
        if delta % U2_DECISION_STEP_NS != 0:
            raise ValueError(f"U2 source {entry.symbol} does not use a dense 15m grid")
        expected_rows = delta // U2_DECISION_STEP_NS + 1
        if entry.row_count != expected_rows:
            raise ValueError(
                f"U2 source {entry.symbol} row_count does not prove a dense 15m grid"
            )


def _window(
    *,
    name: str,
    start: int,
    stop: int,
    common_first_ns: int,
) -> UniversalTradeRLU2TimeWindow:
    return UniversalTradeRLU2TimeWindow(
        name=name,
        start_bar_index=start,
        stop_bar_index_exclusive=stop,
        first_timestamp_ns=common_first_ns + start * U2_DECISION_STEP_NS,
        last_timestamp_ns=common_first_ns + (stop - 1) * U2_DECISION_STEP_NS,
    )


def _tiles(
    window: UniversalTradeRLU2TimeWindow,
    *,
    common_first_ns: int,
) -> tuple[UniversalTradeRLU2EpisodeTile, ...]:
    count = window.bar_count // U2_EVALUATION_TILE_BARS
    if count < U2_MINIMUM_EVALUATION_TILES:
        raise ValueError(
            f"U2 {window.name} requires at least two complete 720h evaluation tiles"
        )
    return tuple(
        UniversalTradeRLU2EpisodeTile(
            source_window=window.name,
            tile_index=index,
            start_bar_index=(
                window.start_bar_index + index * U2_EVALUATION_TILE_BARS
            ),
            stop_bar_index_exclusive=(
                window.start_bar_index + (index + 1) * U2_EVALUATION_TILE_BARS
            ),
            first_timestamp_ns=(
                common_first_ns
                + (window.start_bar_index + index * U2_EVALUATION_TILE_BARS)
                * U2_DECISION_STEP_NS
            ),
            last_timestamp_ns=(
                common_first_ns
                + (
                    window.start_bar_index
                    + (index + 1) * U2_EVALUATION_TILE_BARS
                    - 1
                )
                * U2_DECISION_STEP_NS
            ),
        )
        for index in range(count)
    )


def build_universal_trade_rl_u2_time_partition(
    *,
    manifest: UniversalTradeRLUniverseManifest,
) -> UniversalTradeRLU2TimePartition:
    """Build U2 time preregistration using immutable U0 metadata only."""

    if not isinstance(manifest, UniversalTradeRLUniverseManifest):
        raise TypeError("U2 time partition requires a Universal Trade RL U0 manifest")
    _validate_dense_15m_source_metadata(manifest)
    active_entries = tuple(entry for entry in manifest.entries if entry.role is not None)
    common_first = max(entry.first_timestamp_ns for entry in active_entries)
    common_last = min(entry.last_timestamp_ns for entry in active_entries)
    if common_last < common_first:
        raise ValueError("Universal Trade RL U2 sources have no common history")
    common_delta = common_last - common_first
    if common_delta % U2_DECISION_STEP_NS != 0:
        raise ValueError("Universal Trade RL U2 common history is not a dense 15m grid")
    common_count = common_delta // U2_DECISION_STEP_NS + 1
    if common_count < U2_MINIMUM_COMMON_BARS:
        raise ValueError("Universal Trade RL U2 common history must cover at least 600 days")

    fit_stop = common_count * 6 // 10
    d1_stop = common_count * 7 // 10
    d2_stop = common_count * 8 // 10
    seen_start = fit_stop - U2_SEEN_TIME_PROBE_BARS
    if seen_start < 0:
        raise ValueError("Universal Trade RL U2 FIT window cannot hold 60-day seen probe")

    windows = (
        _window(name="fit", start=0, stop=fit_stop, common_first_ns=common_first),
        _window(
            name="seen_time_probe",
            start=seen_start,
            stop=fit_stop,
            common_first_ns=common_first,
        ),
        _window(
            name="development_future_1",
            start=fit_stop,
            stop=d1_stop,
            common_first_ns=common_first,
        ),
        _window(
            name="development_future_2",
            start=d1_stop,
            stop=d2_stop,
            common_first_ns=common_first,
        ),
        _window(
            name="admission_future",
            start=d2_stop,
            stop=common_count,
            common_first_ns=common_first,
        ),
    )
    tiles = tuple(
        tile
        for window in windows[1:]
        for tile in _tiles(window, common_first_ns=common_first)
    )
    tiles = tuple(sorted(tiles, key=lambda tile: (tile.source_window, tile.tile_index)))
    return UniversalTradeRLU2TimePartition(
        universe_manifest_digest=manifest.digest,
        common_first_timestamp_ns=common_first,
        common_last_timestamp_ns=common_last,
        common_bar_count=common_count,
        windows=windows,
        tiles=tiles,
    )


__all__ = [
    "UNIVERSAL_TRADE_RL_U2_TIME_PARTITION_SCHEMA",
    "U2_BARS_PER_DAY",
    "U2_DECISION_STEP_NS",
    "U2_EVALUATION_TILE_BARS",
    "U2_MINIMUM_COMMON_BARS",
    "U2_MINIMUM_EVALUATION_TILES",
    "U2_SEEN_TIME_PROBE_BARS",
    "UniversalTradeRLU2EpisodeTile",
    "UniversalTradeRLU2TimePartition",
    "UniversalTradeRLU2TimeWindow",
    "build_universal_trade_rl_u2_time_partition",
]
