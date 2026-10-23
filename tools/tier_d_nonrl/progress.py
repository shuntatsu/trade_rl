"""Strict native progress bindings; counts never qualify financial receipts."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .tier_d_protocol import json_sha256

_SPEC = {
    "configuration",
    "account",
    "scenario",
    "role",
    "plan_digest",
    "start_index",
    "actor_stop",
    "settlement_stop",
}
_RECORD = {
    "schema",
    "protocol_digest",
    "arm_id",
    "plan_digest",
    "event",
    "seq",
    "count",
    "total",
    "phase",
    "start_index",
    "processing_index",
}
_ENDS = {"native_returned", "native_failed"}


def _closed(value: Any, fields: set[str]) -> dict[str, Any]:
    if type(value) is not dict or set(value) != fields:
        raise ValueError("progress object has missing/foreign fields")
    return value


def validate_progress_policy(value: object) -> dict[str, Any]:
    raw = _closed(value, {"schema", "no_progress_seconds", "sample_seconds"})
    if raw["schema"] != "issue810_native_progress_v1" or any(
        type(raw[name]) is not int or raw[name] <= 0
        for name in ("no_progress_seconds", "sample_seconds")
    ):
        raise ValueError(
            "progress policy requires its schema and positive ordinary ints"
        )
    return dict(raw)


@dataclass(frozen=True)
class ArmProgressSpec:
    configuration: str
    account: str
    scenario: str
    role: str
    plan_digest: str
    start_index: int
    actor_stop: int
    settlement_stop: int

    def __post_init__(self) -> None:
        if any(
            type(value) is not str
            or not value
            or len(value) > 256
            or value.strip() != value
            or "\0" in value
            for value in (self.configuration, self.account, self.scenario, self.role)
        ) or self.role not in {"nonrl", "cash"}:
            raise ValueError(
                "progress spec requires ordinary nonempty identity strings"
            )
        if (
            type(self.plan_digest) is not str
            or len(self.plan_digest) != 64
            or set(self.plan_digest) - set("0123456789abcdef")
            or any(
                type(x) is not int
                for x in (self.start_index, self.actor_stop, self.settlement_stop)
            )
            or not 0 <= self.start_index < self.actor_stop < self.settlement_stop
        ):
            raise ValueError(
                "progress spec requires a plan SHA256 and ordered integer ranges"
            )

    @property
    def arm_id(self) -> str:
        return json_sha256(asdict(self))

    @property
    def target(self) -> int:
        return self.settlement_stop - self.start_index

    @property
    def payload(self) -> dict[str, Any]:
        return asdict(self) | {"arm_id": self.arm_id, "target": self.target}

    @classmethod
    def from_payload(cls, value: object) -> ArmProgressSpec:
        raw = _closed(value, _SPEC | {"arm_id", "target"})
        result = cls(**{name: raw[name] for name in _SPEC})
        if json_sha256(result.payload) != json_sha256(raw):
            raise ValueError("progress spec identity/target differs")
        return result


def roster_payload(
    protocol: dict[str, Any], specs: tuple[ArmProgressSpec, ...]
) -> dict[str, Any]:
    policy = validate_progress_policy(protocol.get("progress_policy"))
    if (
        not specs
        or len(specs) > 4096
        or any(type(s) is not ArmProgressSpec for s in specs)
    ):
        raise ValueError("progress roster must contain exact bounded arm specs")
    dimensions = []
    for name in ("accounts", "scenarios", "roles"):
        values = protocol.get(name)
        if (
            type(values) is not list
            or not values
            or any(type(v) is not str for v in values)
            or len(set(values)) != len(values)
        ):
            raise ValueError("progress protocol roster is invalid")
        dimensions.append(values)
    if len(dimensions[0]) * len(dimensions[1]) * len(dimensions[2]) > 4096:
        raise ValueError("progress roster exceeds bounded arm count")
    expected = {
        (a, s, r) for a in dimensions[0] for s in dimensions[1] for r in dimensions[2]
    }
    actual = [(s.account, s.scenario, s.role) for s in specs]
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError("progress roster must cover every declared arm exactly once")
    for spec in specs:
        if spec.configuration not in {
            row["id"] for row in protocol["configurations"]
        } or (spec.start_index, spec.actor_stop, spec.settlement_stop) != (
            protocol["actor"]["start"],
            protocol["actor"]["stop"],
            protocol["terminal"]["stop_inclusive"],
        ):
            raise ValueError("progress spec differs from declared configuration/window")
    return {
        "schema": "issue810_native_progress_roster_v1",
        "protocol_digest": json_sha256(protocol),
        "progress_policy": policy,
        "arms": [spec.payload for spec in specs],
    }


def validate_record(
    value: object,
    spec: ArmProgressSpec,
    protocol_digest: str,
    previous: dict[str, Any] | None,
) -> dict[str, Any]:
    raw = _closed(value, _RECORD)
    event, count = raw["event"], raw["count"]
    if (
        raw["schema"] != "issue810_native_progress_record_v1"
        or type(event) is not str
        or raw["arm_id"] != spec.arm_id
        or raw["plan_digest"] != spec.plan_digest
        or raw["protocol_digest"] != protocol_digest
        or any(
            type(raw[k]) is not int
            for k in ("seq", "count", "total", "start_index", "processing_index")
        )
        or not 0 <= count <= spec.target
        or raw["total"] != spec.target
    ):
        raise ValueError("malformed/misbound progress record")
    if previous is None:
        valid = event == "native_started" and raw["seq"] == count == 0
    else:
        valid = (
            previous["event"] not in _ENDS
            and event in _ENDS | {"native_advanced"}
            and raw["seq"] == previous["seq"] + 1
            and count == previous["count"] + (event == "native_advanced")
        )
    processing = spec.start_index + count
    advance = event == "native_advanced"
    if not valid or (
        raw["processing_index"] != processing
        or raw["start_index"] != processing - int(advance)
        or raw["phase"]
        != (
            ("actor" if processing <= spec.actor_stop else "reserve")
            if advance
            else None
        )
    ):
        raise ValueError("duplicate/out-of-order/phase/clock progress record")
    return dict(raw)
