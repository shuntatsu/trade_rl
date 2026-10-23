"""Immutable chronological training-window declarations for allocation PPO.

This module owns schedule identity only. It does not construct environments,
sample policies, open datasets, mutate accounts or claim independent market
experience from symbol slots or random crops.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal

from trade_rl._validation import require_sha256
from trade_rl.artifacts import canonical_json_bytes, content_digest

WindowRole = Literal["train", "validation", "held_out"]
_ROLES = ("train", "validation", "held_out")


def _native_json(value: object) -> None:
    if value is None or type(value) in (str, bool, int):
        return
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("schedule JSON numbers must be finite")
        return
    if type(value) is list:
        for item in value:
            _native_json(item)
        return
    if type(value) is dict:
        for key, item in value.items():
            if type(key) is not str:
                raise ValueError("schedule JSON keys must be native strings")
            _native_json(item)
        return
    raise ValueError("schedule payload must contain only native JSON values")


def _mapping(value: object, keys: set[str], field: str) -> dict[str, Any]:
    if type(value) is not dict or set(value) != keys:
        raise ValueError(f"{field} must contain exactly its declared fields")
    return value


def _integer(value: object, field: str) -> int:
    if type(value) is not int:
        raise ValueError(f"{field} must be a native integer")
    return value


def _index(value: object, field: str, *, minimum: int = 0) -> int:
    result = _integer(value, field)
    if result < minimum:
        raise ValueError(f"{field} must be >= {minimum}")
    return result


@dataclass(frozen=True, slots=True)
class AllocationTrainingWindow:
    """One complete account episode or one explicitly excluded evaluation window."""

    role: WindowRole
    dataset_id: str
    symbol: str
    start_index: int
    stop_index: int
    decision_start_ns: int
    terminal_time_ns: int
    source_digest: str

    def __post_init__(self) -> None:
        role: object = self.role
        if type(role) is not str or role not in _ROLES:
            raise ValueError("window role must be train, validation or held_out")
        require_sha256(self.dataset_id, field="dataset_id")
        require_sha256(self.source_digest, field="source_digest")
        if type(self.symbol) is not str or not self.symbol.strip():
            raise ValueError("window symbol must be nonempty")
        start = _index(self.start_index, "start_index")
        stop = _index(self.stop_index, "stop_index", minimum=1)
        decision = _integer(self.decision_start_ns, "decision_start_ns")
        terminal = _integer(self.terminal_time_ns, "terminal_time_ns")
        if start >= stop:
            raise ValueError("window start_index must precede stop_index")
        if decision >= terminal:
            raise ValueError("window decision clock must precede terminal clock")

    def payload(self) -> dict[str, object]:
        return {
            "schema": "allocation_training_window_v1",
            "role": self.role,
            "dataset_id": self.dataset_id,
            "symbol": self.symbol,
            "start_index": self.start_index,
            "stop_index": self.stop_index,
            "decision_start_ns": self.decision_start_ns,
            "terminal_time_ns": self.terminal_time_ns,
            "source_digest": self.source_digest,
        }

    def identity_payload(self) -> dict[str, object]:
        """Causal sampler identity excludes whole-Dataset lineage."""
        return {
            "schema": "allocation_training_window_identity_v1",
            "role": self.role,
            "symbol": self.symbol,
            "start_index": self.start_index,
            "stop_index": self.stop_index,
            "decision_start_ns": self.decision_start_ns,
            "terminal_time_ns": self.terminal_time_ns,
            "source_digest": self.source_digest,
        }

    @property
    def window_id(self) -> str:
        return content_digest(self.identity_payload())

    @classmethod
    def from_payload(cls, value: object) -> AllocationTrainingWindow:
        _native_json(value)
        data = _mapping(
            value,
            {
                "schema",
                "role",
                "dataset_id",
                "symbol",
                "start_index",
                "stop_index",
                "decision_start_ns",
                "terminal_time_ns",
                "source_digest",
            },
            "training window",
        )
        if data["schema"] != "allocation_training_window_v1":
            raise ValueError("unsupported allocation training window schema")
        try:
            result = cls(
                role=data["role"],
                dataset_id=data["dataset_id"],
                symbol=data["symbol"],
                start_index=data["start_index"],
                stop_index=data["stop_index"],
                decision_start_ns=data["decision_start_ns"],
                terminal_time_ns=data["terminal_time_ns"],
                source_digest=data["source_digest"],
            )
        except TypeError as error:
            raise ValueError("allocation training window is malformed") from error
        if canonical_json_bytes(result.payload()) != canonical_json_bytes(value):
            raise ValueError("training window differs from its canonical declaration")
        return result


@dataclass(frozen=True, slots=True)
class AllocationTrainingSchedule:
    """Result-blind roster; runtime reuse counts are recorded separately."""

    windows: tuple[AllocationTrainingWindow, ...]
    train_window_ids: tuple[str, ...]
    sampler: Literal["cyclic_declared_order_v1"] = "cyclic_declared_order_v1"
    reset_semantics: Literal["independent_account_v1"] = "independent_account_v1"
    rollout_boundary_semantics: Literal["continue_account_v1"] = "continue_account_v1"

    def __post_init__(self) -> None:
        if (
            type(self.windows) is not tuple
            or not self.windows
            or any(
                type(window) is not AllocationTrainingWindow for window in self.windows
            )
        ):
            raise ValueError("schedule requires immutable declared windows")
        if len({window.window_id for window in self.windows}) != len(self.windows):
            raise ValueError("schedule windows must have unique identities")
        if self.sampler != "cyclic_declared_order_v1":
            raise ValueError("unsupported allocation training sampler")
        if self.reset_semantics != "independent_account_v1":
            raise ValueError("training reset must mean a new independent account")
        if self.rollout_boundary_semantics != "continue_account_v1":
            raise ValueError("rollout boundaries must preserve the current account")

        by_id = {window.window_id: window for window in self.windows}
        training = [window for window in self.windows if window.role == "train"]
        expected = tuple(
            window.window_id
            for window in sorted(
                training,
                key=lambda window: (window.decision_start_ns, window.window_id),
            )
        )
        if (
            type(self.train_window_ids) is not tuple
            or not self.train_window_ids
            or any(type(identity) is not str for identity in self.train_window_ids)
            or len(set(self.train_window_ids)) != len(self.train_window_ids)
            or any(identity not in by_id for identity in self.train_window_ids)
            or any(
                by_id[identity].role != "train" for identity in self.train_window_ids
            )
        ):
            raise ValueError("training order must contain only unique train windows")
        if self.train_window_ids != expected:
            raise ValueError(
                "training windows must follow declared chronological order"
            )
        if set(self.train_window_ids) != {window.window_id for window in training}:
            raise ValueError(
                "training order must contain every train window exactly once"
            )

        grouped: dict[str, list[AllocationTrainingWindow]] = {}
        for window in self.windows:
            grouped.setdefault(window.symbol, []).append(window)
        for values in grouped.values():
            ordered = sorted(
                values, key=lambda window: (window.decision_start_ns, window.window_id)
            )
            for left, right in zip(ordered, ordered[1:]):
                if left.terminal_time_ns > right.decision_start_ns:
                    raise ValueError(
                        "schedule windows must not overlap for one symbol clock"
                    )

        train_terminal = max(window.terminal_time_ns for window in training)
        validation = [window for window in self.windows if window.role == "validation"]
        held_out = [window for window in self.windows if window.role == "held_out"]
        if any(
            window.decision_start_ns < train_terminal
            for window in validation + held_out
        ):
            raise ValueError("validation and held_out windows must follow training")
        if validation:
            validation_terminal = max(window.terminal_time_ns for window in validation)
            if any(
                window.decision_start_ns < validation_terminal for window in held_out
            ):
                raise ValueError("held_out windows must follow validation")

    @property
    def training_windows(self) -> tuple[AllocationTrainingWindow, ...]:
        by_id = {window.window_id: window for window in self.windows}
        return tuple(by_id[identity] for identity in self.train_window_ids)

    def payload(self) -> dict[str, object]:
        return {
            "schema": "allocation_training_schedule_v1",
            "sampler": self.sampler,
            "reset_semantics": self.reset_semantics,
            "rollout_boundary_semantics": self.rollout_boundary_semantics,
            "windows": [window.payload() for window in self.windows],
            "train_window_ids": list(self.train_window_ids),
        }

    @property
    def digest(self) -> str:
        return content_digest(self.payload())

    @classmethod
    def from_payload(cls, value: object) -> AllocationTrainingSchedule:
        _native_json(value)
        data = _mapping(
            value,
            {
                "schema",
                "sampler",
                "reset_semantics",
                "rollout_boundary_semantics",
                "windows",
                "train_window_ids",
            },
            "training schedule",
        )
        if data["schema"] != "allocation_training_schedule_v1":
            raise ValueError("unsupported allocation training schedule schema")
        if (
            type(data["windows"]) is not list
            or type(data["train_window_ids"]) is not list
        ):
            raise ValueError("training schedule arrays must be native lists")
        try:
            result = cls(
                windows=tuple(
                    AllocationTrainingWindow.from_payload(window)
                    for window in data["windows"]
                ),
                train_window_ids=tuple(data["train_window_ids"]),
                sampler=data["sampler"],
                reset_semantics=data["reset_semantics"],
                rollout_boundary_semantics=data["rollout_boundary_semantics"],
            )
        except TypeError as error:
            raise ValueError("allocation training schedule is malformed") from error
        if canonical_json_bytes(result.payload()) != canonical_json_bytes(value):
            raise ValueError("training schedule differs from its canonical declaration")
        return result


__all__ = ["AllocationTrainingSchedule", "AllocationTrainingWindow"]
