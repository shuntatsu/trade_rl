"""Closed source identities for paired H-native-v3 and exact-fill-v4 readers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

ProtocolSourceProfileId = Literal[
    "d361-native-v3-v4",
    "b631-native-candidate-v3-v4",
]

DEFAULT_PROTOCOL_SOURCE_PROFILE: ProtocolSourceProfileId = "d361-native-v3-v4"
B631_CANDIDATE_PROTOCOL_SOURCE_PROFILE: ProtocolSourceProfileId = (
    "b631-native-candidate-v3-v4"
)


@dataclass(frozen=True, slots=True)
class ProtocolSourceProfile:
    """Immutable canonical identities for one independently declared source."""

    profile_id: ProtocolSourceProfileId
    source_head: str
    native_v3_canonical_sha256: str
    comparison_v4_canonical_sha256: str


_PROTOCOL_SOURCE_PROFILES = (
    ProtocolSourceProfile(
        profile_id=DEFAULT_PROTOCOL_SOURCE_PROFILE,
        source_head="d361b6e3c3cdc83a8f5f05da9f82e1f72b66e3e5",
        native_v3_canonical_sha256=(
            "742e883b405a0d7f600998b820a190f872673f1a439d56dd3b8001a6cf017393"
        ),
        comparison_v4_canonical_sha256=(
            "59558244c17890ce8c4017cf3f0ffc144c133f781e9daec6367eb2afce7dea2a"
        ),
    ),
    ProtocolSourceProfile(
        profile_id=B631_CANDIDATE_PROTOCOL_SOURCE_PROFILE,
        source_head="b6318664c24da5a486baf0179d7a52fadaaa502d",
        native_v3_canonical_sha256=(
            "90f929781f8edb718d43ac46220030bdd6767ac4c856f7417a88606cc9bc6660"
        ),
        comparison_v4_canonical_sha256=(
            "e85aea4f1792a9a10d366c289b3f76ecaac2d548d210cdd30a449fb0d6c69852"
        ),
    ),
)


def resolve_protocol_source_profile(value: object) -> ProtocolSourceProfile:
    """Resolve only a declared profile identifier; no head or digest wildcards."""
    if type(value) is not str:
        raise ValueError("protocol source profile identifier is required")
    for profile in _PROTOCOL_SOURCE_PROFILES:
        if value == profile.profile_id:
            return profile
    raise ValueError("unsupported paired protocol source profile")


def protocol_source_profile_for_head(value: object) -> ProtocolSourceProfile:
    """Resolve an existing binding's exact source head in the closed profile set."""
    if type(value) is not str:
        raise ValueError("protocol source head is malformed")
    for profile in _PROTOCOL_SOURCE_PROFILES:
        if value == profile.source_head:
            return profile
    raise ValueError("protocol source head is outside the closed profile set")
