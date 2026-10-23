"""Bind exact-H static inputs and derive per-advance event comparison contexts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import InitVar, dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np

from tools.market_replay_oracle.artifact import (
    DatasetContentPin,
    load_pinned_v4,
)
from tools.market_replay_oracle.binding import (
    FrozenAttemptBinding,
    _dataset_still_pinned,
)
from tools.market_replay_oracle.events import OrderEventContext
from tools.market_replay_oracle.identity import canonical
from tools.market_replay_oracle.profiles import effective_profile
from tools.market_replay_oracle.protocol import ProtocolPin
from tools.market_replay_oracle.protocol_source_profiles import (
    DEFAULT_PROTOCOL_SOURCE_PROFILE,
    ProtocolSourceProfileId,
    resolve_protocol_source_profile,
)
from tools.market_replay_oracle.protocol_v4 import (
    _EXACT_FILL_EVIDENCE,
    bind_schedule_v4,
    load_protocol_v4,
)
from tools.market_replay_oracle.source import NativeSourcePin, verify_native_source


@dataclass(frozen=True, slots=True)
class ExactFillAttemptBinding(FrozenAttemptBinding):
    """Immutable expected event facts; no runtime, receipt or financial admission."""

    source_profile: InitVar[ProtocolSourceProfileId] = field(
        default=DEFAULT_PROTOCOL_SOURCE_PROFILE, kw_only=True
    )
    _event_arms: Mapping[tuple[str, str, str], tuple[int, str]] = field(
        init=False, repr=False, compare=False
    )

    def __post_init__(self, source_profile: ProtocolSourceProfileId) -> None:
        FrozenAttemptBinding.__post_init__(self)
        payload = self.payload()
        try:
            profile = resolve_protocol_source_profile(source_profile)
            source_head = payload["protocol"]["source_head"]
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("unsupported exact-fill source profile") from error
        if (
            payload.get("schema") != "independent_exact_fill_attempt_binding_v1"
            or payload.get("scope") != "STATIC_INPUT_COMPARISON_ONLY"
            or source_head != profile.source_head
            or payload["dataset"]["dataset_id"] != self.content.dataset_id
            or payload["exact_fill_evidence"] != _EXACT_FILL_EVIDENCE
        ):
            raise ValueError("unsupported exact-fill static binding")
        try:
            lots = self.content.arrays["lot_size"]
            times = self.content.arrays["timestamps"]
        except KeyError as error:
            raise ValueError(
                "verified lot and timestamp arrays are required"
            ) from error
        if (
            lots.shape != (9504, 2)
            or lots.dtype != np.dtype("<f8")
            or lots.flags.writeable
            or not np.isfinite(lots).all()
            or np.any(lots < 0.0)
            or times.shape != (9504,)
            or times.dtype != np.dtype("datetime64[ns]")
            or times.flags.writeable
            or np.isnat(times).any()
        ):
            raise ValueError("immutable verified lot/clock content is required")
        if self._clock[2] >= times.shape[0]:
            raise ValueError("bound settlement stop exceeds verified clock rows")
        expected: dict[tuple[str, str, str], tuple[int, str]] = {}
        for arm in payload["schedule"]["arms"]:
            profile = payload["profiles"][arm["scenario"]]
            # This frozen route has no extra venue grid, zero cost floor and
            # nominal stress. Adding one needs a separately frozen contract.
            if (
                profile["market_order_profile"] is not None
                or type(profile["cost_policy"]["lot_size"]) is not float
                or profile["cost_policy"]["lot_size"] != 0.0
                or canonical(profile["rule_stress"])
                != canonical(
                    {
                        "adverse_tick_rounding": False,
                        "lot_size_factor": 1.0,
                        "minimum_notional_factor": 1.0,
                        "name": "nominal",
                        "schema_version": "execution_rule_stress_v1",
                        "tick_size_factor": 1.0,
                    }
                )
            ):
                raise ValueError("unsupported effective-lot profile")
            symbol = arm["symbol_index"]
            if type(symbol) is not int or symbol not in (0, 1):
                raise ValueError("unsupported bound symbol index")
            expected[(arm["account"], arm["scenario"], arm["role"])] = (
                symbol,
                profile["execution_policy_digest"],
            )
        object.__setattr__(self, "_event_arms", MappingProxyType(expected))

    def event_context(
        self, account: str, scenario: str, role: str, count: object
    ) -> OrderEventContext:
        """Derive expected clocks and lot from inputs, never observed event fields."""
        processing = self.progress(count)["processing_index"]
        try:
            symbol, policy_digest = self._event_arms[(account, scenario, role)]
        except (KeyError, TypeError) as error:
            raise ValueError("unknown bound account/scenario/role") from error
        times = self.content.arrays["timestamps"]
        # max(row, 0.0) * 1.0 equals row for this validated nominal path.
        lot = float(self.content.arrays["lot_size"][processing, symbol])
        return OrderEventContext(
            dataset_id=self.content.dataset_id,
            execution_policy_digest=policy_digest,
            symbol_index=symbol,
            decision_index=processing - 1,
            decision_time_ns=int(times[processing - 1].astype("int64")),
            processing_index=processing,
            processing_time_ns=int(times[processing].astype("int64")),
            effective_lot_size=lot,
        )


def load_pinned_exact_fill_attempt(
    protocol_path: Path,
    protocol_pin: ProtocolPin,
    native_source_root: Path,
    native_source_pin: NativeSourcePin,
    dataset_root: Path,
    dataset_pin: DatasetContentPin,
    *,
    source_profile: ProtocolSourceProfileId = DEFAULT_PROTOCOL_SOURCE_PROFILE,
) -> ExactFillAttemptBinding:
    """Bind the separate H/v4 static contract; never start or relabel an attempt."""
    profile = resolve_protocol_source_profile(source_profile)
    protocol = load_protocol_v4(
        protocol_path, protocol_pin, source_profile=profile.profile_id
    )
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
    schedule = bind_schedule_v4(protocol, content, source_profile=profile.profile_id)
    start, stop = protocol["actor"]["start"], protocol["actor"]["stop"]
    profiles = {
        scenario: effective_profile(scenario, stop - start, protocol["policy"]["risk"])
        for scenario in protocol["scenarios"]
    }
    payload: dict[str, Any] = {
        "schema": "independent_exact_fill_attempt_binding_v1",
        "scope": "STATIC_INPUT_COMPARISON_ONLY",
        "protocol": {
            "sha256": protocol_pin.sha256,
            "bytes": protocol_pin.raw_bytes,
            "source_head": protocol_pin.source_head,
        },
        "exact_fill_evidence": protocol["exact_fill_evidence"],
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
    if (
        load_protocol_v4(protocol_path, protocol_pin, source_profile=profile.profile_id)
        != protocol
    ):
        raise ValueError("pinned protocol changed during static binding")
    if verify_native_source(native_source_root, native_source_pin) != source_before:
        raise ValueError("native source changed during static binding")
    return ExactFillAttemptBinding(
        canonical(payload), content, source_profile=profile.profile_id
    )
