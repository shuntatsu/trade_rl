"""Compare distinct native H/v3 and comparator H/v4 declarations, statically."""

from __future__ import annotations

import hashlib
from dataclasses import InitVar, dataclass, field
from pathlib import Path
from typing import Literal

from tools.market_replay_oracle import protocol_v4
from tools.market_replay_oracle.identity import canonical, json_object
from tools.market_replay_oracle.protocol import (
    _TOP_LEVEL_KEYS,
    MAX_PROTOCOL_BYTES,
    ProtocolPin,
    _regular_protocol_file,
)
from tools.market_replay_oracle.protocol_source_profiles import (
    DEFAULT_PROTOCOL_SOURCE_PROFILE,
    ProtocolSourceProfile,
    ProtocolSourceProfileId,
    protocol_source_profile_for_head,
    resolve_protocol_source_profile,
)

# Independently derived from H's committed tier_d_protocol.py declaration at
# d361b6e3c3cdc83a8f5f05da9f82e1f72b66e3e5, max_workers=4. The native
# public-loader control and the distinct v4 projection reproduce these bytes.
_DEFAULT_SOURCE_PROFILE = resolve_protocol_source_profile(
    DEFAULT_PROTOCOL_SOURCE_PROFILE
)
_H_NATIVE_CANONICAL_SHA256 = _DEFAULT_SOURCE_PROFILE.native_v3_canonical_sha256
_NATIVE_SCHEMA = "issue810_tier_d_nonrl48_result_blind_v3"


@dataclass(frozen=True, slots=True)
class DeclaredProtocolPair:
    """Immutable declaration snapshots and raw pins; no runtime attestation."""

    native_pin: ProtocolPin
    comparison_pin: ProtocolPin
    native_canonical: bytes
    comparison_canonical: bytes
    source_profile: InitVar[ProtocolSourceProfileId] = field(
        default=DEFAULT_PROTOCOL_SOURCE_PROFILE, kw_only=True
    )
    scope: Literal["STATIC_DECLARED_PROTOCOL_PAIR_ONLY"] = field(
        default="STATIC_DECLARED_PROTOCOL_PAIR_ONLY", init=False
    )

    def __post_init__(self, source_profile: ProtocolSourceProfileId) -> None:
        try:
            profile = resolve_protocol_source_profile(source_profile)
        except (TypeError, ValueError) as error:
            raise ValueError("unsupported pinned H protocol source profile") from error
        self._validate_profile(profile)

    def validate(self) -> None:
        """Revalidate the immutable snapshot against its closed pinned profile."""
        try:
            profile = protocol_source_profile_for_head(self.native_pin.source_head)
        except (AttributeError, TypeError, ValueError) as error:
            raise ValueError("unsupported pinned H protocol source profile") from error
        self._validate_profile(profile)

    def _validate_profile(self, profile: ProtocolSourceProfile) -> None:
        for pin, snapshot, expected in (
            (
                self.native_pin,
                self.native_canonical,
                profile.native_v3_canonical_sha256,
            ),
            (
                self.comparison_pin,
                self.comparison_canonical,
                profile.comparison_v4_canonical_sha256,
            ),
        ):
            if (
                not isinstance(pin, ProtocolPin)
                or pin.source_head != profile.source_head
                or type(snapshot) is not bytes
                or hashlib.sha256(snapshot).hexdigest() != expected
            ):
                raise ValueError("unsupported pinned H protocol declaration snapshot")


def _native_snapshot(
    path: Path, pin: ProtocolPin, profile: ProtocolSourceProfile
) -> bytes:
    """Read the independently frozen H/v3 declaration without native imports."""
    absolute = _regular_protocol_file(path)
    initial_size = absolute.stat().st_size
    if initial_size > MAX_PROTOCOL_BYTES:
        raise ValueError("native protocol exceeds the 1 MiB limit")
    try:
        with absolute.open("rb") as stream:
            raw = stream.read(MAX_PROTOCOL_BYTES + 1)
    except OSError as error:
        raise ValueError("native protocol file could not be read") from error
    _regular_protocol_file(absolute)
    if len(raw) > MAX_PROTOCOL_BYTES:
        raise ValueError("native protocol exceeds the 1 MiB limit")
    if len(raw) != initial_size or len(raw) != pin.raw_bytes:
        raise ValueError("native protocol raw byte count mismatch")
    if hashlib.sha256(raw).hexdigest() != pin.sha256:
        raise ValueError("native protocol raw SHA-256 mismatch")
    document = json_object(raw)
    if (
        set(document) != _TOP_LEVEL_KEYS
        or document.get("schema") != _NATIVE_SCHEMA
        or document.get("source_head") != profile.source_head
        or pin.source_head != profile.source_head
    ):
        raise ValueError("unsupported closed native H/v3 declaration or source pin")
    snapshot = canonical(document)
    if hashlib.sha256(snapshot).hexdigest() != profile.native_v3_canonical_sha256:
        raise ValueError("native declaration differs from the frozen H contract")
    return snapshot


def _distinct_files(native_path: Path, comparison_path: Path) -> None:
    native = _regular_protocol_file(native_path)
    comparison = _regular_protocol_file(comparison_path)
    try:
        same = native.samefile(comparison)
    except OSError as error:
        raise ValueError("protocol file identities could not be compared") from error
    if same:
        raise ValueError("native and comparison protocols require distinct files")


def load_declared_protocol_pair(
    v3_path: Path,
    v3_pin: ProtocolPin,
    v4_path: Path,
    v4_pin: ProtocolPin,
    *,
    source_profile: ProtocolSourceProfileId = DEFAULT_PROTOCOL_SOURCE_PROFILE,
) -> DeclaredProtocolPair:
    """Check two static declarations; later consumers require runtime binding."""
    profile = resolve_protocol_source_profile(source_profile)
    if not all(isinstance(path, Path) for path in (v3_path, v4_path)) or not all(
        isinstance(pin, ProtocolPin) for pin in (v3_pin, v4_pin)
    ):
        raise ValueError("two protocol Paths and independent ProtocolPins are required")
    _distinct_files(v3_path, v4_path)
    native_snapshot = _native_snapshot(v3_path, v3_pin, profile)
    comparison = protocol_v4.load_protocol_v4(
        v4_path, v4_pin, source_profile=profile.profile_id
    )
    comparison_snapshot = canonical(comparison)
    projection = dict(comparison)
    del projection["exact_fill_evidence"]
    projection["schema"] = _NATIVE_SCHEMA
    if canonical(projection) != native_snapshot:
        raise ValueError("native and comparison shared protocol fields differ")
    # Observe changes during this validation. This does not lock the files or
    # attest which inputs a subsequent native/comparison process consumes.
    if (
        _native_snapshot(v3_path, v3_pin, profile) != native_snapshot
        or canonical(
            protocol_v4.load_protocol_v4(
                v4_path, v4_pin, source_profile=profile.profile_id
            )
        )
        != comparison_snapshot
    ):
        raise ValueError("protocol declaration changed during validation")
    _distinct_files(v3_path, v4_path)
    return DeclaredProtocolPair(
        native_pin=v3_pin,
        comparison_pin=v4_pin,
        native_canonical=native_snapshot,
        comparison_canonical=comparison_snapshot,
        source_profile=profile.profile_id,
    )
