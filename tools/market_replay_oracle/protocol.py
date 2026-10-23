"""Pin the fixed consumed-development protocol and derive its execution schedule."""

from __future__ import annotations

import hashlib
import re
import stat
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from tools.market_replay_oracle.artifact import VerifiedDatasetContent
from tools.market_replay_oracle.identity import canonical, json_object

MAX_PROTOCOL_BYTES = 1024 * 1024
_SOURCE_HEAD = "8bc2fd0cba83a02f5040159c1fc71e95fab774db"
_CANONICAL_PROTOCOL_SHA256 = (
    "a847aee427cbc99a79c8c9413296bc37f0b2d867171bd3244f108f49bd3fc479"
)
_TOP_LEVEL_KEYS = frozenset(
    {
        "accounts",
        "actor",
        "attempts",
        "budget",
        "capital",
        "concurrency",
        "configurations",
        "dataset",
        "decision",
        "economics",
        "funding",
        "policy",
        "progress_policy",
        "research",
        "resource_proposal",
        "roles",
        "scenarios",
        "schema",
        "scope",
        "source_head",
        "terminal",
    }
)
_SCHEMA = "issue810_tier_d_nonrl48_result_blind_v3"
_SCOPE = "CONSUMED_2025_DEVELOPMENT_NONRL48_ONLY"


@dataclass(frozen=True, slots=True)
class ProtocolPin:
    """Caller-supplied raw-byte and source-head pins for one protocol file."""

    sha256: str
    raw_bytes: int
    source_head: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.sha256, str)
            or re.fullmatch(r"[0-9a-f]{64}", self.sha256) is None
        ):
            raise ValueError("protocol pin requires a lowercase SHA-256")
        if (
            type(self.raw_bytes) is not int
            or not 0 < self.raw_bytes <= MAX_PROTOCOL_BYTES
        ):
            raise ValueError("protocol raw byte count must be within the 1 MiB limit")
        if (
            not isinstance(self.source_head, str)
            or re.fullmatch(r"[0-9a-f]{40}", self.source_head) is None
        ):
            raise ValueError("protocol pin requires a lowercase source commit SHA")


def _regular_protocol_file(path: Path) -> Path:
    if path.drive and not path.is_absolute():
        raise ValueError("drive-relative protocol paths are unsupported")
    # Keep the lexical path while making it absolute: resolve() would follow a
    # reparse point before the component walk could reject it.
    candidate = path if path.is_absolute() else Path.cwd() / path
    components = (candidate, *candidate.parents)
    for component in reversed(components):
        try:
            info = component.lstat()
        except OSError as error:
            raise ValueError("protocol path component is unavailable") from error
        if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
            raise ValueError("protocol path contains a link or reparse point")
        if component == candidate:
            if not stat.S_ISREG(info.st_mode):
                raise ValueError("protocol requires a regular file")
        elif not stat.S_ISDIR(info.st_mode):
            raise ValueError("protocol path ancestors must be directories")
    return candidate


def _validate_contract(value: object) -> dict[str, Any]:
    if type(value) is not dict or set(value) != _TOP_LEVEL_KEYS:
        raise ValueError("protocol top-level key closure mismatch")
    if value.get("schema") != _SCHEMA or value.get("scope") != _SCOPE:
        raise ValueError("unsupported protocol schema or scope")
    if value.get("source_head") != _SOURCE_HEAD:
        raise ValueError("protocol source head differs from the frozen source")
    try:
        digest = hashlib.sha256(canonical(value)).hexdigest()
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("protocol is not a canonical JSON object") from error
    if digest != _CANONICAL_PROTOCOL_SHA256:
        raise ValueError("protocol differs from the frozen financial and arm contract")
    return value


def load_protocol(path: Path, pin: ProtocolPin) -> dict[str, Any]:
    """Read and validate a pinned, bounded v3 protocol before returning it."""
    if not isinstance(path, Path) or not isinstance(pin, ProtocolPin):
        raise ValueError("a protocol Path and ProtocolPin are required")
    absolute_path = _regular_protocol_file(path)
    initial_size = absolute_path.stat().st_size
    if initial_size > MAX_PROTOCOL_BYTES:
        raise ValueError("protocol exceeds the 1 MiB limit")
    try:
        with absolute_path.open("rb") as stream:
            raw = stream.read(MAX_PROTOCOL_BYTES + 1)
    except OSError as error:
        raise ValueError("protocol file could not be read") from error
    _regular_protocol_file(absolute_path)
    if len(raw) > MAX_PROTOCOL_BYTES:
        raise ValueError("protocol exceeds the 1 MiB limit")
    if len(raw) != initial_size or len(raw) != pin.raw_bytes:
        raise ValueError("protocol raw byte count mismatch")
    if hashlib.sha256(raw).hexdigest() != pin.sha256:
        raise ValueError("protocol raw SHA-256 mismatch")

    # Parsing is intentionally after both raw-byte pin checks.
    document = json_object(raw)
    document = _validate_contract(document)
    if pin.source_head != _SOURCE_HEAD or document["source_head"] != pin.source_head:
        raise ValueError("protocol source head pin mismatch")
    return document


def _utc_second(value: np.datetime64) -> str:
    return np.datetime_as_string(value, unit="s")


def _utc_ns(value: np.datetime64) -> str:
    return np.datetime_as_string(value, unit="ns")


def _string_tuple(value: object) -> tuple[str, ...]:
    if not isinstance(value, (tuple, list)) or any(
        not isinstance(item, str) for item in value
    ):
        raise ValueError("verified dataset names must be string sequences")
    return tuple(value)


def _verified_clock(
    protocol: dict[str, Any], content: VerifiedDatasetContent
) -> np.ndarray[Any, Any]:
    dataset = protocol["dataset"]
    if not isinstance(content, VerifiedDatasetContent):
        raise ValueError("verified dataset content is required")
    if content.dataset_id != dataset["dataset_id"]:
        raise ValueError("verified dataset identity differs from the protocol")
    if not isinstance(content.dataset, Mapping):
        raise ValueError("verified dataset scalars are unavailable")
    scalars = content.dataset
    try:
        symbols = _string_tuple(scalars["symbols"])
        feature_names = _string_tuple(scalars["feature_names"])
        calendar_kind = scalars["calendar_kind"]
        nominal_bar_hours = scalars["nominal_bar_hours"]
        timestamps = np.asarray(content.arrays["timestamps"])
        features = np.asarray(content.arrays["features"])
    except (KeyError, TypeError) as error:
        raise ValueError("verified dataset fields are incomplete") from error
    if symbols != tuple(dataset["fit_symbols"]):
        raise ValueError("verified dataset symbol order differs from the protocol")
    if feature_names != tuple(dataset["feature_names"]):
        raise ValueError("verified dataset feature order differs from the protocol")
    if calendar_kind != "continuous_24_7":
        raise ValueError("only the frozen continuous UTC bar clock is supported")
    if nominal_bar_hours is not None and (
        type(nominal_bar_hours) not in {int, float} or nominal_bar_hours != 1.0
    ):
        raise ValueError("verified dataset cadence differs from the protocol")

    dimensions = tuple(dataset["dimensions"])
    if features.shape != dimensions or features.dtype.kind != "f":
        raise ValueError("verified dataset dimensions differ from the protocol")
    if (
        timestamps.shape != (dimensions[0],)
        or timestamps.dtype != np.dtype("datetime64[ns]")
        or np.isnat(timestamps).any()
    ):
        raise ValueError("verified UTC timestamp array has an invalid shape or dtype")
    start = np.datetime64(dataset["development_start"], "ns")
    expected = start + np.arange(dimensions[0]) * np.timedelta64(1, "h")
    if not np.array_equal(timestamps, expected):
        raise ValueError("verified UTC timestamps differ from the frozen hourly grid")
    return timestamps


def bind_schedule(
    protocol: dict[str, Any], content: VerifiedDatasetContent
) -> dict[str, Any]:
    """Bind the fixed 48-arm schedule to exact verified content timestamps."""
    document = _validate_contract(protocol)
    timestamps = _verified_clock(document, content)
    actor = document["actor"]
    terminal = document["terminal"]
    actor_start = actor["start"]
    actor_stop = actor["stop"]
    reserve_stop = terminal["stop_inclusive"]
    first_unused = reserve_stop + 1

    if (
        actor_stop - actor_start != actor["rows"]
        or reserve_stop != actor_stop + terminal["advances"]
        or first_unused != timestamps.size - 1
        or _utc_second(timestamps[actor_start]) != actor["start_utc"]
        or _utc_second(timestamps[actor_stop]) != actor["stop_utc"]
        or _utc_second(timestamps[reserve_stop]) != terminal["utc"]
    ):
        raise ValueError("actor or terminal indices do not match the verified clock")

    vintage_starts = actor["vintage_starts"]
    vintage_utc = actor["vintage_start_utc"]
    vintage_stops = [*vintage_starts[1:], actor_stop]
    if len(vintage_starts) != 3 or len(vintage_utc) != 3:
        raise ValueError("the fixed three-vintage schedule is required")
    vintages: list[dict[str, Any]] = []
    for start_index, stop_index, declared_start in zip(
        vintage_starts, vintage_stops, vintage_utc, strict=True
    ):
        start_time = timestamps[start_index]
        if _utc_second(start_time) != declared_start:
            raise ValueError("vintage UTC boundary differs from the verified clock")
        vintages.append(
            {
                "start_index": start_index,
                "stop_exclusive_index": stop_index,
                "rows": stop_index - start_index,
                "start_utc": _utc_second(start_time),
                "fit_cutoff_utc": _utc_ns(start_time - np.timedelta64(1, "ns")),
                "model_fit_time_utc": _utc_second(start_time),
            }
        )

    arms: list[dict[str, Any]] = []
    for configuration in document["configurations"]:
        for symbol_index, symbol in enumerate(document["dataset"]["fit_symbols"]):
            account = f"{configuration['id']}_{symbol}"
            for scenario in document["scenarios"]:
                for role in document["roles"]:
                    nonrl = role == "nonrl"
                    arms.append(
                        {
                            "configuration": configuration["id"],
                            "account": account,
                            "symbol": symbol,
                            "symbol_index": symbol_index,
                            "scenario": scenario,
                            "role": role,
                            "ridge_alpha": configuration["ridge_alpha"],
                            "forecast_horizon_seconds": (
                                configuration["horizon_hours"] * 3600
                            ),
                            "action_mode": "residual" if nonrl else "direct",
                            "action_code": 2 if nonrl else 0,
                            "action_scale": 1.0,
                        }
                    )

    if len(arms) != document["budget"]["native_arms"]:
        raise ValueError("expanded schedule does not match the frozen arm budget")
    actor_schedule = {
        "start_index": actor_start,
        "stop_exclusive_index": actor_stop,
        "advances": actor_stop - actor_start,
        "rows": actor["rows"],
        "start_utc": _utc_second(timestamps[actor_start]),
        "stop_utc": _utc_second(timestamps[actor_stop]),
    }
    terminal_schedule = {
        "start_index": actor_stop,
        "stop_inclusive_index": reserve_stop,
        "advances": reserve_stop - actor_stop,
        "total_advances_per_arm": actor["rows"] + terminal["advances"],
        "native_arms": len(arms),
        "matched_pairs": document["budget"]["matched_pairs"],
        "total_native_advances": len(arms) * (actor["rows"] + terminal["advances"]),
        "first_unused_index": first_unused,
        "first_unused_utc": _utc_second(timestamps[first_unused]),
    }
    if (
        terminal_schedule["advances"] != terminal["advances"]
        or terminal_schedule["total_advances_per_arm"]
        != document["budget"]["advances_per_arm"]
        or terminal_schedule["total_native_advances"]
        != document["budget"]["total_native_advances"]
    ):
        raise ValueError("expanded schedule does not match the frozen row budget")
    return {
        "actor": actor_schedule,
        "terminal": terminal_schedule,
        "vintages": vintages,
        "arms": arms,
    }
