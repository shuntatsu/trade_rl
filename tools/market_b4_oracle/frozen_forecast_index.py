from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, cast

from tools.market_b4_oracle.allocation_choice import AllocationAssumptions

from .frozen_forecast import (
    OwnedDecisionSnapshot,
    bind_frozen_forecast_assumptions,
)

_SCOPE = "FROZEN_FORECAST_INDEX_LOOKUP_ONLY"
_MAX_BYTES = 64 * 1024 * 1024
_MAX_PACKETS = 20_000
_MAX_VINTAGES = 1_024
_MAX_DEPTH = 64
_MAX_NUMBER_CHARS = 4096
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_MIN_NS = -(2**63)
_MAX_NS = 2**63 - 1
_STREAM_KEYS = frozenset(
    {
        "schema",
        "availability_semantics",
        "dataset_id",
        "vintages",
        "packets",
        "causal_scope_digest",
    }
)


def _ordinary_int(value: object, *, name: str, minimum: int = 1) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an ordinary integer >= {minimum}")
    return value


def _sha256(value: object, *, name: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("JSON object contains a duplicate key")
        result[key] = value
    return result


def _json_constant(value: str) -> object:
    raise ValueError(f"non-finite JSON constant {value!r} is unsupported")


def _json_float(value: str) -> float:
    if len(value) > _MAX_NUMBER_CHARS:
        raise ValueError("JSON numeric literal exceeds the 4096-character limit")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("JSON numeric literal is outside finite binary64 range")
    return result


def _json_int(value: str) -> int:
    if len(value) > _MAX_NUMBER_CHARS:
        raise ValueError("JSON numeric literal exceeds the 4096-character limit")
    result = int(value)
    try:
        finite = math.isfinite(float(result))
    except OverflowError as error:
        raise ValueError(
            "JSON integer literal is outside finite binary64 range"
        ) from error
    if not finite:
        raise ValueError("JSON integer literal is outside finite binary64 range")
    return result


def _check_nesting(text: str, *, max_depth: int) -> None:
    depth = 0
    in_string = False
    escaped = False
    for char in text:
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char in "[{":
            depth += 1
            if depth > max_depth:
                raise ValueError("JSON nesting exceeds the configured depth limit")
        elif char in "]}":
            depth -= 1
            if depth < 0:
                raise ValueError("JSON container delimiters are unbalanced")


@dataclass(frozen=True, slots=True)
class ForecastIndexLimits:
    max_bytes: int = _MAX_BYTES
    max_packets: int = _MAX_PACKETS
    max_vintages: int = _MAX_VINTAGES
    max_depth: int = _MAX_DEPTH

    def __post_init__(self) -> None:
        bounds = (
            ("max_bytes", self.max_bytes, 1, _MAX_BYTES),
            ("max_packets", self.max_packets, 1, _MAX_PACKETS),
            ("max_vintages", self.max_vintages, 1, _MAX_VINTAGES),
            ("max_depth", self.max_depth, 1, _MAX_DEPTH),
        )
        for name, value, lower, upper in bounds:
            if type(value) is not int or not lower <= value <= upper:
                raise ValueError(
                    f"{name} must be an ordinary integer in [{lower}, {upper}]"
                )


@dataclass(frozen=True, slots=True)
class ForecastIndexMetadata:
    scope: str
    pinned_bytes: int
    pinned_sha256: str
    packet_count: int
    vintage_count: int


class FrozenForecastIndex:
    """One-decode index for a caller-pinned frozen forecast stream.

    The raw byte identity is checked before UTF-8 or JSON parsing. The parsed
    tree and row indexes remain private. Lookups pass a tiny derived view of the
    selected packet/vintage pair through the existing semantic adapter; they do
    not expose model arrays or recompute a canonical digest for the full stream.
    """

    __slots__ = (
        "__metadata",
        "__top_metadata",
        "__payload",
        "__packet_index",
        "__vintage_index",
    )

    def __init__(
        self,
        raw_bytes: bytes,
        *,
        pinned_bytes: int,
        pinned_sha256: str,
        limits: ForecastIndexLimits | None = None,
    ) -> None:
        if type(raw_bytes) is not bytes:
            raise ValueError("forecast input must be immutable bytes")
        expected_bytes = _ordinary_int(pinned_bytes, name="pinned_bytes")
        expected_sha = _sha256(pinned_sha256, name="pinned_sha256")
        selected_limits = ForecastIndexLimits() if limits is None else limits
        if type(selected_limits) is not ForecastIndexLimits:
            raise ValueError("limits must be an exact ForecastIndexLimits value")
        if len(raw_bytes) != expected_bytes:
            raise ValueError("forecast byte count differs from its caller pin")
        if len(raw_bytes) > selected_limits.max_bytes:
            raise ValueError("forecast bytes exceed the configured input limit")
        actual_sha = hashlib.sha256(raw_bytes).hexdigest()
        if actual_sha != expected_sha:
            raise ValueError("forecast bytes differ from their caller SHA-256 pin")

        try:
            text = raw_bytes.decode("utf-8", errors="strict")
            _check_nesting(text, max_depth=selected_limits.max_depth)
            parsed = json.loads(
                text,
                object_pairs_hook=_json_object,
                parse_constant=_json_constant,
                parse_float=_json_float,
                parse_int=_json_int,
            )
        except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as error:
            raise ValueError(
                "forecast bytes are not bounded strict UTF-8 JSON"
            ) from error

        if type(parsed) is not dict or frozenset(parsed) != _STREAM_KEYS:
            raise ValueError("forecast stream must have its exact top-level fields")
        payload = cast(dict[str, object], parsed)
        if (
            payload["schema"] != "simple_return_ridge_stream_v1"
            or payload["availability_semantics"] != "declared_simulation_v1"
        ):
            raise ValueError("forecast stream schema or availability semantics differ")
        dataset_id = _sha256(payload["dataset_id"], name="dataset_id")
        causal_scope_digest = _sha256(
            payload["causal_scope_digest"], name="causal_scope_digest"
        )
        packets = payload["packets"]
        vintages = payload["vintages"]
        if type(packets) is not list or not packets:
            raise ValueError("forecast packets must be a nonempty JSON array")
        if type(vintages) is not list or not vintages:
            raise ValueError("forecast vintages must be a nonempty JSON array")
        if len(packets) > selected_limits.max_packets:
            raise ValueError("forecast packet count exceeds the configured limit")
        if len(vintages) > selected_limits.max_vintages:
            raise ValueError("forecast vintage count exceeds the configured limit")

        packet_index: dict[tuple[str, int], dict[str, object]] = {}
        for value in packets:
            if type(value) is not dict:
                raise ValueError("each forecast packet must be a JSON object")
            row = cast(dict[str, object], value)
            symbol = row.get("symbol")
            as_of = row.get("as_of")
            if type(symbol) is not str or not symbol.strip():
                raise ValueError("forecast packet symbol must be nonempty")
            if type(as_of) is not int or not _MIN_NS <= as_of <= _MAX_NS:
                raise ValueError("forecast packet as_of must be signed nanoseconds")
            key = (symbol, as_of)
            if key in packet_index:
                raise ValueError("forecast stream contains a duplicate packet key")
            packet_index[key] = row

        vintage_index: dict[str, dict[str, object]] = {}
        for value in vintages:
            if type(value) is not dict:
                raise ValueError("each forecast vintage must be a JSON object")
            row = cast(dict[str, object], value)
            digest = _sha256(row.get("digest"), name="vintage.digest")
            if digest in vintage_index:
                raise ValueError("forecast stream contains a duplicate vintage digest")
            vintage_index[digest] = row

        for row in packet_index.values():
            vintage_digest = _sha256(
                row.get("vintage_digest"), name="packet.vintage_digest"
            )
            if vintage_digest not in vintage_index:
                raise ValueError("forecast packet references an unresolved vintage")

        self.__payload = payload
        self.__top_metadata: Mapping[str, object] = MappingProxyType(
            {
                "schema": payload["schema"],
                "availability_semantics": payload["availability_semantics"],
                "dataset_id": dataset_id,
                "causal_scope_digest": causal_scope_digest,
            }
        )
        self.__packet_index: Mapping[tuple[str, int], dict[str, object]] = (
            MappingProxyType(packet_index)
        )
        self.__vintage_index: Mapping[str, dict[str, object]] = MappingProxyType(
            vintage_index
        )
        self.__metadata = ForecastIndexMetadata(
            scope=_SCOPE,
            pinned_bytes=expected_bytes,
            pinned_sha256=expected_sha,
            packet_count=len(packet_index),
            vintage_count=len(vintage_index),
        )

    @property
    def metadata(self) -> ForecastIndexMetadata:
        return self.__metadata

    def lookup(
        self,
        source: OwnedDecisionSnapshot,
        *,
        cost_assumptions: AllocationAssumptions,
    ) -> AllocationAssumptions:
        if type(source) is not OwnedDecisionSnapshot:
            raise ValueError("requires an exact owned-decision snapshot")
        key = (source.symbol, source.decision_time_ns)
        packet = self.__packet_index.get(key)
        if packet is None:
            raise ValueError("owned decision clock does not resolve to a packet")
        digest = _sha256(packet.get("vintage_digest"), name="packet.vintage_digest")
        vintage = self.__vintage_index.get(digest)
        if vintage is None:
            raise ValueError("packet does not resolve to a frozen vintage")
        selected_stream: dict[str, object] = dict(self.__top_metadata)
        selected_stream["packets"] = (packet,)
        selected_stream["vintages"] = (vintage,)
        return bind_frozen_forecast_assumptions(
            selected_stream,
            source=source,
            cost_assumptions=cost_assumptions,
        )


__all__ = [
    "ForecastIndexLimits",
    "ForecastIndexMetadata",
    "FrozenForecastIndex",
]
