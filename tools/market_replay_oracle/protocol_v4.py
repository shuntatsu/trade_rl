"""Separate exact-H protocol contract; frozen G/v3 remains byte-preserved."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np

from tools.market_replay_oracle.artifact import VerifiedDatasetContent
from tools.market_replay_oracle.identity import canonical, json_object
from tools.market_replay_oracle.protocol import (
    _TOP_LEVEL_KEYS,
    MAX_PROTOCOL_BYTES,
    ProtocolPin,
    _regular_protocol_file,
    _utc_ns,
    _utc_second,
    _verified_clock,
)
from tools.market_replay_oracle.protocol_source_profiles import (
    DEFAULT_PROTOCOL_SOURCE_PROFILE,
    ProtocolSourceProfileId,
    resolve_protocol_source_profile,
)

_DEFAULT_SOURCE_PROFILE = resolve_protocol_source_profile(
    DEFAULT_PROTOCOL_SOURCE_PROFILE
)
_SOURCE_HEAD = _DEFAULT_SOURCE_PROFILE.source_head
_CANONICAL_PROTOCOL_SHA256 = _DEFAULT_SOURCE_PROFILE.comparison_v4_canonical_sha256
_EXACT_FILL_EVIDENCE = {
    "positive_effective_lot": "order_event_v2_signed_integer_count_and_step",
    "zero_effective_lot": "order_event_v1_exact_reported_float",
    "nonfills": "order_event_v1_zero_fill",
    "source_transition": "fresh_attempt_required_no_v3_relabeling",
}


def _validate_contract_v4(
    value: object,
    *,
    source_profile: ProtocolSourceProfileId = DEFAULT_PROTOCOL_SOURCE_PROFILE,
) -> dict[str, Any]:
    profile = resolve_protocol_source_profile(source_profile)
    if type(value) is not dict or set(value) != _TOP_LEVEL_KEYS | {
        "exact_fill_evidence"
    }:
        raise ValueError("protocol v4 top-level key closure mismatch")
    if (
        value.get("schema") != "issue810_tier_d_nonrl48_result_blind_v4"
        or value.get("scope") != "CONSUMED_2025_DEVELOPMENT_NONRL48_ONLY"
        or value.get("source_head") != profile.source_head
    ):
        raise ValueError("unsupported v4 schema, scope or frozen source head")
    try:
        digest = hashlib.sha256(canonical(value)).hexdigest()
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("protocol v4 is not a canonical JSON object") from error
    if digest != profile.comparison_v4_canonical_sha256:
        raise ValueError(
            "protocol differs from the frozen v4 financial and arm contract"
        )
    return value


def load_protocol_v4(
    path: Path,
    pin: ProtocolPin,
    *,
    source_profile: ProtocolSourceProfileId = DEFAULT_PROTOCOL_SOURCE_PROFILE,
) -> dict[str, Any]:
    """Read a separately pinned, bounded v4 document; never reinterpret v3."""
    profile = resolve_protocol_source_profile(source_profile)
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
    document = _validate_contract_v4(json_object(raw), source_profile=source_profile)
    if (
        pin.source_head != profile.source_head
        or document["source_head"] != pin.source_head
    ):
        raise ValueError("protocol source head pin mismatch")
    return document


def bind_schedule_v4(
    protocol: dict[str, Any],
    content: VerifiedDatasetContent,
    *,
    source_profile: ProtocolSourceProfileId = DEFAULT_PROTOCOL_SOURCE_PROFILE,
) -> dict[str, Any]:
    """Expand v4 directly, preserving the fixed financial schedule and arm order."""
    document = _validate_contract_v4(protocol, source_profile=source_profile)
    # This helper validates only content identity/scalars/UTC grid; it does not
    # assert a G schema or source. Calling v3 bind_schedule would reject v4.
    timestamps = _verified_clock(document, content)
    actor, terminal = document["actor"], document["terminal"]
    start, stop = actor["start"], actor["stop"]
    reserve_stop = terminal["stop_inclusive"]
    first_unused = reserve_stop + 1
    if (
        stop - start != actor["rows"]
        or reserve_stop != stop + terminal["advances"]
        or first_unused != timestamps.size - 1
        or _utc_second(timestamps[start]) != actor["start_utc"]
        or _utc_second(timestamps[stop]) != actor["stop_utc"]
        or _utc_second(timestamps[reserve_stop]) != terminal["utc"]
    ):
        raise ValueError("actor or terminal indices do not match the verified clock")
    vintage_starts = actor["vintage_starts"]
    vintage_utc = actor["vintage_start_utc"]
    if len(vintage_starts) != 3 or len(vintage_utc) != 3:
        raise ValueError("the fixed three-vintage schedule is required")
    # Nanosecond subtraction is deliberate; fit excludes the vintage boundary.
    vintages = []
    for first, last, declared in zip(
        vintage_starts, [*vintage_starts[1:], stop], vintage_utc, strict=True
    ):
        time = timestamps[first]
        if _utc_second(time) != declared:
            raise ValueError("vintage UTC boundary differs from the verified clock")
        vintages.append(
            {
                "start_index": first,
                "stop_exclusive_index": last,
                "rows": last - first,
                "start_utc": _utc_second(time),
                "fit_cutoff_utc": _utc_ns(time - np.timedelta64(1, "ns")),
                "model_fit_time_utc": _utc_second(time),
            }
        )
    arms = []
    for configuration in document["configurations"]:
        for symbol_index, symbol in enumerate(document["dataset"]["fit_symbols"]):
            for scenario in document["scenarios"]:
                for role in document["roles"]:
                    nonrl = role == "nonrl"
                    arms.append(
                        {
                            "configuration": configuration["id"],
                            "account": f"{configuration['id']}_{symbol}",
                            "symbol": symbol,
                            "symbol_index": symbol_index,
                            "scenario": scenario,
                            "role": role,
                            "ridge_alpha": configuration["ridge_alpha"],
                            "forecast_horizon_seconds": configuration["horizon_hours"]
                            * 3600,
                            "action_mode": "residual" if nonrl else "direct",
                            "action_code": 2 if nonrl else 0,
                            "action_scale": 1.0,
                        }
                    )
    terminal_schedule = {
        "start_index": stop,
        "stop_inclusive_index": reserve_stop,
        "advances": reserve_stop - stop,
        "total_advances_per_arm": actor["rows"] + terminal["advances"],
        "native_arms": len(arms),
        "matched_pairs": document["budget"]["matched_pairs"],
        "total_native_advances": len(arms) * (actor["rows"] + terminal["advances"]),
        "first_unused_index": first_unused,
        "first_unused_utc": _utc_second(timestamps[first_unused]),
    }
    if (
        len(arms) != document["budget"]["native_arms"]
        or terminal_schedule["advances"] != terminal["advances"]
        or terminal_schedule["total_advances_per_arm"]
        != document["budget"]["advances_per_arm"]
        or terminal_schedule["total_native_advances"]
        != document["budget"]["total_native_advances"]
    ):
        raise ValueError("expanded schedule does not match the frozen arm/row budget")
    return {
        "actor": {
            "start_index": start,
            "stop_exclusive_index": stop,
            "advances": stop - start,
            "rows": actor["rows"],
            "start_utc": _utc_second(timestamps[start]),
            "stop_utc": _utc_second(timestamps[stop]),
        },
        "terminal": terminal_schedule,
        "vintages": vintages,
        "arms": arms,
    }
