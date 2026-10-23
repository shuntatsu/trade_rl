"""Join pinned static inputs without executing a financial account."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any

from tools.market_replay_oracle.artifact import (
    DatasetContentPin,
    VerifiedDatasetContent,
    _read,
    _regular,
    load_pinned_v4,
)
from tools.market_replay_oracle.identity import canonical, json_object
from tools.market_replay_oracle.profiles import effective_profile
from tools.market_replay_oracle.protocol import (
    ProtocolPin,
    bind_schedule,
    load_protocol,
)
from tools.market_replay_oracle.source import NativeSourcePin, verify_native_source


@dataclass(frozen=True, slots=True)
class FrozenAttemptBinding:
    """Static comparison facts; never a financial or runtime authorization."""

    _canonical_payload: bytes = field(repr=False)
    content: VerifiedDatasetContent = field(repr=False, compare=False)
    _clock: tuple[int, int, int] = field(init=False, repr=False)
    _profile_index: Mapping[tuple[str, str, str], bytes] = field(
        init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        payload = json_object(self._canonical_payload)
        if canonical(payload) != self._canonical_payload:
            raise ValueError("binding requires canonical immutable payload bytes")
        clock = payload["clock"]
        start, stop, terminal = (
            clock["actor_start"],
            clock["actor_stop"],
            clock["settlement_stop"],
        )
        if any(type(value) is not int for value in (start, stop, terminal)) or not (
            0 <= start < stop < terminal
        ):
            raise ValueError("invalid bound actor/reserve clock")
        expected: dict[tuple[str, str, str], bytes] = {}
        for arm in payload["schedule"]["arms"]:
            key = (arm["account"], arm["scenario"], arm["role"])
            if key in expected:
                raise ValueError("duplicate bound account/scenario/role")
            profile = payload["profiles"][arm["scenario"]]
            expected[key] = canonical(
                {
                    "symbol_index": arm["symbol_index"],
                    "forecast_horizon_seconds": arm["forecast_horizon_seconds"],
                    "cost_policy": profile["cost_policy"],
                    "execution_policy_digest": profile["execution_policy_digest"],
                    "runtime_profile": profile["runtime_profile"],
                }
            )
        object.__setattr__(self, "_clock", (start, stop, terminal))
        object.__setattr__(self, "_profile_index", MappingProxyType(expected))

    @property
    def digest(self) -> str:
        return hashlib.sha256(self._canonical_payload).hexdigest()

    def payload(self) -> dict[str, Any]:
        """Return a fresh projection, leaving cached static facts immutable."""
        return json_object(self._canonical_payload)

    def progress(self, count: object) -> dict[str, Any]:
        start, stop, terminal = self._clock
        if type(count) is not int or not 1 <= count <= terminal - start:
            raise ValueError("advance count is outside the bound physical horizon")
        index = start + count
        return {
            "count": count,
            "processing_index": index,
            "phase": "actor" if index <= stop else "reserve",
        }

    def assert_arm_profile(
        self, account: str, scenario: str, role: str, observed: Mapping[str, object]
    ) -> None:
        """Compare complete static fields; a digest alone cannot qualify a row."""
        try:
            expected = self._profile_index[(account, scenario, role)]
            actual = canonical(dict(observed))
        except (KeyError, TypeError) as error:
            raise ValueError("unknown arm or malformed static profile") from error
        if actual != expected:
            raise ValueError("static native projection differs from bound inputs")


def _dataset_still_pinned(root: Path, pin: DatasetContentPin) -> None:
    for path in (root, *root.parents):
        _regular(path, directory=True)
    if {path.name for path in root.iterdir()} != {"manifest.json", "arrays.npz"}:
        raise ValueError("bound Dataset directory roster changed")
    _read(root / "manifest.json", pin.manifest_bytes, pin.manifest_sha256)
    _read(root / "arrays.npz", pin.arrays_bytes, pin.arrays_sha256)


def load_pinned_attempt(
    protocol_path: Path,
    protocol_pin: ProtocolPin,
    native_source_root: Path,
    native_source_pin: NativeSourcePin,
    dataset_root: Path,
    dataset_pin: DatasetContentPin,
) -> FrozenAttemptBinding:
    """Build expected static facts from explicit pins, not native result fields."""
    protocol = load_protocol(protocol_path, protocol_pin)
    if protocol_pin.source_head != native_source_pin.head_sha:
        raise ValueError("protocol and native source HEAD pins differ")
    dataset_root = Path(dataset_root)
    declared = protocol["dataset"]
    if (
        not dataset_root.is_absolute()
        or str(dataset_root) != declared["root"]
        or declared["dataset_id"] != dataset_pin.dataset_id
        or declared["input_sha256"]
        != {
            "manifest.json": dataset_pin.manifest_sha256,
            "arrays.npz": dataset_pin.arrays_sha256,
        }
    ):
        raise ValueError("caller Dataset root/pins differ from the pinned protocol")
    source_before = verify_native_source(native_source_root, native_source_pin)
    content = load_pinned_v4(dataset_root, dataset_pin)
    schedule = bind_schedule(protocol, content)
    start, stop = protocol["actor"]["start"], protocol["actor"]["stop"]
    profiles = {
        scenario: effective_profile(scenario, stop - start, protocol["policy"]["risk"])
        for scenario in protocol["scenarios"]
    }
    payload = {
        "schema": "independent_frozen_attempt_binding_v1",
        "scope": "STATIC_INPUT_COMPARISON_ONLY",
        "protocol": {
            "sha256": protocol_pin.sha256,
            "bytes": protocol_pin.raw_bytes,
            "source_head": protocol_pin.source_head,
        },
        "native_source": source_before,
        "dataset": {
            "dataset_id": dataset_pin.dataset_id,
            "manifest_sha256": dataset_pin.manifest_sha256,
            "arrays_sha256": dataset_pin.arrays_sha256,
            "manifest_bytes": dataset_pin.manifest_bytes,
            "arrays_bytes": dataset_pin.arrays_bytes,
        },
        "clock": {
            "actor_start": start,
            "actor_stop": stop,
            "settlement_stop": protocol["terminal"]["stop_inclusive"],
        },
        "schedule": schedule,
        "profiles": profiles,
    }
    _dataset_still_pinned(dataset_root, dataset_pin)
    if load_protocol(protocol_path, protocol_pin) != protocol:
        raise ValueError("pinned protocol changed during static binding")
    if verify_native_source(native_source_root, native_source_pin) != source_before:
        raise ValueError("native source changed during static binding")
    return FrozenAttemptBinding(canonical(payload), content)
