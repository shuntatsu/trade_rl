"""Content-addressed traversal and resume state for symbol-triplet training."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Final, cast

from trade_rl.artifacts.codec import canonical_json_bytes
from trade_rl.artifacts.hashing import content_digest
from trade_rl.domain.common import require_sha256
from trade_rl.workflows.symbol_triplet_manifest import (
    SYMBOL_TRIPLET_SIZE,
    SYMBOL_TRIPLET_SPLIT_COUNTS,
    SymbolTripletManifest,
    SymbolTripletSlot,
)

SYMBOL_TRIPLET_TRAINING_STAGE_SCHEMA: Final = "symbol_triplet_training_stage_v1"
SYMBOL_TRIPLET_TRAINING_PLAN_SCHEMA: Final = "symbol_triplet_training_plan_v1"
SYMBOL_TRIPLET_TRAINING_CURSOR_SCHEMA: Final = "symbol_triplet_training_cursor_v1"
_PLAN_IDENTITY_SCHEMA: Final = "symbol_triplet_training_plan_identity_v1"
_TRAIN_CYCLE_SCHEMA: Final = "symbol_triplet_train_cycle_v1"


def _non_negative_integer(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return value


def _positive_integer(value: object, *, field: str) -> int:
    resolved = _non_negative_integer(value, field=field)
    if resolved == 0:
        raise ValueError(f"{field} must be positive")
    return resolved


def _string_tuple(
    values: tuple[str, ...] | list[str],
    *,
    field: str,
    expected_size: int,
) -> tuple[str, ...]:
    resolved = tuple(values)
    if len(resolved) != expected_size:
        raise ValueError(f"{field} must contain exactly {expected_size} values")
    if any(not isinstance(value, str) or not value for value in resolved):
        raise ValueError(f"{field} must contain non-empty strings")
    if len(set(resolved)) != len(resolved):
        raise ValueError(f"{field} must be unique")
    return resolved


def _train_slot_payload(slot: SymbolTripletSlot) -> dict[str, object]:
    return {
        "source_slot_id": slot.slot_id,
        "source_triplet_id": slot.triplet_id,
        "symbols": slot.symbols,
        "train_split_slot": slot.split_slot,
    }


def _train_cycle_digest(slots: tuple[SymbolTripletSlot, ...]) -> str:
    return content_digest(
        {
            "schema_version": _TRAIN_CYCLE_SCHEMA,
            "slots": tuple(_train_slot_payload(slot) for slot in slots),
        }
    )


def _plan_identity_payload(
    *,
    manifest_digest: str,
    schedule_identity: str,
    train_cycle_digest: str,
    cycles: int,
    slot_symbols: tuple[str, ...],
) -> dict[str, object]:
    return {
        "cycles": cycles,
        "manifest_digest": manifest_digest,
        "schedule_identity": schedule_identity,
        "schema_version": _PLAN_IDENTITY_SCHEMA,
        "slot_symbols": slot_symbols,
        "train_cycle_digest": train_cycle_digest,
    }


@dataclass(frozen=True, slots=True)
class SymbolTripletTrainingStage:
    """One exact triplet binding in a multi-stage shared-policy training plan."""

    stage_index: int
    cycle_index: int
    train_split_slot: int
    source_slot_id: str
    source_triplet_id: str
    symbols: tuple[str, ...]
    slot_symbols: tuple[str, ...]
    plan_identity: str
    stage_id: str
    schema_version: str = SYMBOL_TRIPLET_TRAINING_STAGE_SCHEMA

    def __post_init__(self) -> None:
        stage_index = _non_negative_integer(self.stage_index, field="stage_index")
        cycle_index = _non_negative_integer(self.cycle_index, field="cycle_index")
        train_split_slot = _non_negative_integer(
            self.train_split_slot,
            field="train_split_slot",
        )
        if self.schema_version != SYMBOL_TRIPLET_TRAINING_STAGE_SCHEMA:
            raise ValueError("unsupported symbol-triplet training stage schema")
        require_sha256(self.source_slot_id, field="training_stage.source_slot_id")
        require_sha256(
            self.source_triplet_id,
            field="training_stage.source_triplet_id",
        )
        require_sha256(self.plan_identity, field="training_stage.plan_identity")
        require_sha256(self.stage_id, field="training_stage.stage_id")
        symbols = _string_tuple(
            self.symbols,
            field="training_stage.symbols",
            expected_size=SYMBOL_TRIPLET_SIZE,
        )
        slot_symbols = _string_tuple(
            self.slot_symbols,
            field="training_stage.slot_symbols",
            expected_size=SYMBOL_TRIPLET_SIZE,
        )
        expected_stage_id = content_digest(self.identity_payload())
        if self.stage_id != expected_stage_id:
            raise ValueError("symbol-triplet training stage identity mismatch")
        object.__setattr__(self, "stage_index", stage_index)
        object.__setattr__(self, "cycle_index", cycle_index)
        object.__setattr__(self, "train_split_slot", train_split_slot)
        object.__setattr__(self, "symbols", symbols)
        object.__setattr__(self, "slot_symbols", slot_symbols)

    def identity_payload(self) -> dict[str, object]:
        return {
            "cycle_index": self.cycle_index,
            "plan_identity": self.plan_identity,
            "schema_version": self.schema_version,
            "slot_symbols": self.slot_symbols,
            "source_slot_id": self.source_slot_id,
            "source_triplet_id": self.source_triplet_id,
            "stage_index": self.stage_index,
            "symbols": self.symbols,
            "train_split_slot": self.train_split_slot,
        }

    def to_json_dict(self) -> dict[str, object]:
        return {"stage_id": self.stage_id, **self.identity_payload()}


@dataclass(frozen=True, slots=True)
class SymbolTripletTrainingPlan:
    """Immutable repeated traversal over the manifest's balanced train split."""

    manifest_digest: str
    schedule_identity: str
    train_cycle_digest: str
    cycles: int
    slot_symbols: tuple[str, ...]
    plan_identity: str
    stages: tuple[SymbolTripletTrainingStage, ...]
    schema_version: str = SYMBOL_TRIPLET_TRAINING_PLAN_SCHEMA
    digest: str = ""

    def __post_init__(self) -> None:
        if self.schema_version != SYMBOL_TRIPLET_TRAINING_PLAN_SCHEMA:
            raise ValueError("unsupported symbol-triplet training plan schema")
        require_sha256(self.manifest_digest, field="training_plan.manifest_digest")
        require_sha256(
            self.schedule_identity,
            field="training_plan.schedule_identity",
        )
        require_sha256(
            self.train_cycle_digest,
            field="training_plan.train_cycle_digest",
        )
        cycles = _positive_integer(self.cycles, field="cycles")
        slot_symbols = _string_tuple(
            self.slot_symbols,
            field="training_plan.slot_symbols",
            expected_size=SYMBOL_TRIPLET_SIZE,
        )
        expected_plan_identity = content_digest(
            _plan_identity_payload(
                manifest_digest=self.manifest_digest,
                schedule_identity=self.schedule_identity,
                train_cycle_digest=self.train_cycle_digest,
                cycles=cycles,
                slot_symbols=slot_symbols,
            )
        )
        if self.plan_identity != expected_plan_identity:
            raise ValueError("symbol-triplet training plan identity mismatch")
        train_count = SYMBOL_TRIPLET_SPLIT_COUNTS["train"]
        expected_stage_count = cycles * train_count
        if len(self.stages) != expected_stage_count:
            raise ValueError("symbol-triplet training plan stage count mismatch")
        for stage_index, stage in enumerate(self.stages):
            if stage.stage_index != stage_index:
                raise ValueError(
                    "symbol-triplet training stage indices must be contiguous"
                )
            if stage.cycle_index != stage_index // train_count:
                raise ValueError("symbol-triplet training cycle index mismatch")
            if stage.train_split_slot != stage_index % train_count:
                raise ValueError("symbol-triplet training split slot mismatch")
            if stage.plan_identity != self.plan_identity:
                raise ValueError("symbol-triplet training stage plan identity mismatch")
            if stage.slot_symbols != slot_symbols:
                raise ValueError("symbol-triplet training stage slot mapping mismatch")
        first_cycle = self.stages[:train_count]
        expected_cycle_digest = content_digest(
            {
                "schema_version": _TRAIN_CYCLE_SCHEMA,
                "slots": tuple(
                    {
                        "source_slot_id": stage.source_slot_id,
                        "source_triplet_id": stage.source_triplet_id,
                        "symbols": stage.symbols,
                        "train_split_slot": stage.train_split_slot,
                    }
                    for stage in first_cycle
                ),
            }
        )
        if self.train_cycle_digest != expected_cycle_digest:
            raise ValueError("symbol-triplet training cycle digest mismatch")
        first_signature = tuple(
            (
                stage.source_slot_id,
                stage.source_triplet_id,
                stage.symbols,
                stage.train_split_slot,
            )
            for stage in first_cycle
        )
        for cycle_index in range(1, cycles):
            start = cycle_index * train_count
            cycle_signature = tuple(
                (
                    stage.source_slot_id,
                    stage.source_triplet_id,
                    stage.symbols,
                    stage.train_split_slot,
                )
                for stage in self.stages[start : start + train_count]
            )
            if cycle_signature != first_signature:
                raise ValueError("symbol-triplet training cycles must repeat exactly")
        expected_digest = content_digest(self.digest_payload())
        if self.digest and self.digest != expected_digest:
            raise ValueError("symbol-triplet training plan digest mismatch")
        object.__setattr__(self, "cycles", cycles)
        object.__setattr__(self, "slot_symbols", slot_symbols)
        object.__setattr__(self, "digest", expected_digest)

    @property
    def stage_count(self) -> int:
        return len(self.stages)

    def digest_payload(self) -> dict[str, object]:
        return {
            "cycles": self.cycles,
            "manifest_digest": self.manifest_digest,
            "plan_identity": self.plan_identity,
            "schedule_identity": self.schedule_identity,
            "schema_version": self.schema_version,
            "slot_symbols": self.slot_symbols,
            "stages": tuple(stage.to_json_dict() for stage in self.stages),
            "train_cycle_digest": self.train_cycle_digest,
        }

    def to_json_dict(self) -> dict[str, object]:
        return {"digest": self.digest, **self.digest_payload()}

    def validate_manifest(self, manifest: SymbolTripletManifest) -> None:
        if manifest.digest != self.manifest_digest:
            raise ValueError("symbol-triplet training manifest digest mismatch")
        if manifest.schedule_identity != self.schedule_identity:
            raise ValueError("symbol-triplet training schedule identity mismatch")
        train_slots = manifest.slots_for("train")
        if _train_cycle_digest(train_slots) != self.train_cycle_digest:
            raise ValueError("symbol-triplet training manifest cycle mismatch")
        for stage, slot in zip(
            self.stages[: len(train_slots)],
            train_slots,
            strict=True,
        ):
            if (
                stage.source_slot_id != slot.slot_id
                or stage.source_triplet_id != slot.triplet_id
                or stage.symbols != slot.symbols
                or stage.train_split_slot != slot.split_slot
            ):
                raise ValueError("symbol-triplet training manifest stage mismatch")


@dataclass(frozen=True, slots=True)
class SymbolTripletTrainingCursor:
    """Fail-closed resume position for one immutable training plan."""

    plan_digest: str
    stage_count: int
    next_stage_index: int
    last_completed_stage_id: str | None
    schema_version: str = SYMBOL_TRIPLET_TRAINING_CURSOR_SCHEMA
    digest: str = ""

    def __post_init__(self) -> None:
        if self.schema_version != SYMBOL_TRIPLET_TRAINING_CURSOR_SCHEMA:
            raise ValueError("unsupported symbol-triplet training cursor schema")
        require_sha256(self.plan_digest, field="training_cursor.plan_digest")
        stage_count = _positive_integer(self.stage_count, field="stage_count")
        next_stage_index = _non_negative_integer(
            self.next_stage_index,
            field="next_stage_index",
        )
        if next_stage_index > stage_count:
            raise ValueError("training cursor next stage is outside the plan")
        if next_stage_index == 0:
            if self.last_completed_stage_id is not None:
                raise ValueError(
                    "initial training cursor cannot have a completed stage"
                )
        else:
            if self.last_completed_stage_id is None:
                raise ValueError("advanced training cursor requires a completed stage")
            require_sha256(
                self.last_completed_stage_id,
                field="training_cursor.last_completed_stage_id",
            )
        expected_digest = content_digest(self.digest_payload())
        if self.digest and self.digest != expected_digest:
            raise ValueError("symbol-triplet training cursor digest mismatch")
        object.__setattr__(self, "stage_count", stage_count)
        object.__setattr__(self, "next_stage_index", next_stage_index)
        object.__setattr__(self, "digest", expected_digest)

    def digest_payload(self) -> dict[str, object]:
        return {
            "last_completed_stage_id": self.last_completed_stage_id,
            "next_stage_index": self.next_stage_index,
            "plan_digest": self.plan_digest,
            "schema_version": self.schema_version,
            "stage_count": self.stage_count,
        }

    def to_json_dict(self) -> dict[str, object]:
        return {"digest": self.digest, **self.digest_payload()}

    def validate_plan(self, plan: SymbolTripletTrainingPlan) -> None:
        if self.plan_digest != plan.digest:
            raise ValueError("symbol-triplet training cursor plan digest mismatch")
        if self.stage_count != plan.stage_count:
            raise ValueError("symbol-triplet training cursor stage count mismatch")
        if self.next_stage_index > plan.stage_count:
            raise ValueError("symbol-triplet training cursor is outside the plan")
        expected_last = (
            None
            if self.next_stage_index == 0
            else plan.stages[self.next_stage_index - 1].stage_id
        )
        if self.last_completed_stage_id != expected_last:
            raise ValueError("symbol-triplet training cursor completion mismatch")


def build_symbol_triplet_training_plan(
    manifest: SymbolTripletManifest,
    *,
    cycles: int,
    slot_symbols: tuple[str, ...],
) -> SymbolTripletTrainingPlan:
    """Build a deterministic repeated traversal over all balanced train slots."""

    resolved_cycles = _positive_integer(cycles, field="cycles")
    resolved_slots = _string_tuple(
        slot_symbols,
        field="slot_symbols",
        expected_size=SYMBOL_TRIPLET_SIZE,
    )
    train_slots = manifest.slots_for("train")
    train_cycle_digest = _train_cycle_digest(train_slots)
    plan_identity = content_digest(
        _plan_identity_payload(
            manifest_digest=manifest.digest,
            schedule_identity=manifest.schedule_identity,
            train_cycle_digest=train_cycle_digest,
            cycles=resolved_cycles,
            slot_symbols=resolved_slots,
        )
    )
    stages: list[SymbolTripletTrainingStage] = []
    for cycle_index in range(resolved_cycles):
        for slot in train_slots:
            stage_index = len(stages)
            identity_payload = {
                "cycle_index": cycle_index,
                "plan_identity": plan_identity,
                "schema_version": SYMBOL_TRIPLET_TRAINING_STAGE_SCHEMA,
                "slot_symbols": resolved_slots,
                "source_slot_id": slot.slot_id,
                "source_triplet_id": slot.triplet_id,
                "stage_index": stage_index,
                "symbols": slot.symbols,
                "train_split_slot": slot.split_slot,
            }
            stages.append(
                SymbolTripletTrainingStage(
                    stage_index=stage_index,
                    cycle_index=cycle_index,
                    train_split_slot=slot.split_slot,
                    source_slot_id=slot.slot_id,
                    source_triplet_id=slot.triplet_id,
                    symbols=slot.symbols,
                    slot_symbols=resolved_slots,
                    plan_identity=plan_identity,
                    stage_id=content_digest(identity_payload),
                )
            )
    plan = SymbolTripletTrainingPlan(
        manifest_digest=manifest.digest,
        schedule_identity=manifest.schedule_identity,
        train_cycle_digest=train_cycle_digest,
        cycles=resolved_cycles,
        slot_symbols=resolved_slots,
        plan_identity=plan_identity,
        stages=tuple(stages),
    )
    plan.validate_manifest(manifest)
    return plan


def initial_symbol_triplet_training_cursor(
    plan: SymbolTripletTrainingPlan,
) -> SymbolTripletTrainingCursor:
    cursor = SymbolTripletTrainingCursor(
        plan_digest=plan.digest,
        stage_count=plan.stage_count,
        next_stage_index=0,
        last_completed_stage_id=None,
    )
    cursor.validate_plan(plan)
    return cursor


def current_symbol_triplet_training_stage(
    plan: SymbolTripletTrainingPlan,
    cursor: SymbolTripletTrainingCursor,
) -> SymbolTripletTrainingStage | None:
    cursor.validate_plan(plan)
    if cursor.next_stage_index == plan.stage_count:
        return None
    return plan.stages[cursor.next_stage_index]


def advance_symbol_triplet_training_cursor(
    plan: SymbolTripletTrainingPlan,
    cursor: SymbolTripletTrainingCursor,
    *,
    completed_stage_id: str,
) -> SymbolTripletTrainingCursor:
    stage = current_symbol_triplet_training_stage(plan, cursor)
    if stage is None:
        raise ValueError("symbol-triplet training plan is already complete")
    if completed_stage_id != stage.stage_id:
        raise ValueError("completed stage does not match current training stage")
    advanced = SymbolTripletTrainingCursor(
        plan_digest=plan.digest,
        stage_count=plan.stage_count,
        next_stage_index=cursor.next_stage_index + 1,
        last_completed_stage_id=stage.stage_id,
    )
    advanced.validate_plan(plan)
    return advanced


def write_symbol_triplet_training_plan(
    path: str | Path,
    plan: SymbolTripletTrainingPlan,
) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(canonical_json_bytes(plan.to_json_dict()))
    return output


def write_symbol_triplet_training_cursor(
    path: str | Path,
    cursor: SymbolTripletTrainingCursor,
) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(canonical_json_bytes(cursor.to_json_dict()))
    return output


def _json_object(path: str | Path, *, field: str) -> dict[str, object]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{field} must be a JSON object")
    return dict(payload)


def load_symbol_triplet_training_plan(
    path: str | Path,
    *,
    manifest: SymbolTripletManifest,
) -> SymbolTripletTrainingPlan:
    payload = _json_object(path, field="symbol-triplet training plan")
    required = {
        "cycles",
        "digest",
        "manifest_digest",
        "plan_identity",
        "schedule_identity",
        "schema_version",
        "slot_symbols",
        "stages",
        "train_cycle_digest",
    }
    if set(payload) != required:
        raise ValueError("symbol-triplet training plan field closure mismatch")
    raw_stages = payload["stages"]
    if not isinstance(raw_stages, list):
        raise ValueError("symbol-triplet training plan stages must be a list")
    stage_fields = {
        "cycle_index",
        "plan_identity",
        "schema_version",
        "slot_symbols",
        "source_slot_id",
        "source_triplet_id",
        "stage_id",
        "stage_index",
        "symbols",
        "train_split_slot",
    }
    stages: list[SymbolTripletTrainingStage] = []
    for raw_stage in raw_stages:
        if not isinstance(raw_stage, dict) or set(raw_stage) != stage_fields:
            raise ValueError("symbol-triplet training stage field closure mismatch")
        raw_symbols = raw_stage["symbols"]
        raw_slot_symbols = raw_stage["slot_symbols"]
        if not isinstance(raw_symbols, list) or not isinstance(raw_slot_symbols, list):
            raise ValueError("symbol-triplet training stage symbols must be lists")
        stages.append(
            SymbolTripletTrainingStage(
                stage_index=raw_stage["stage_index"],
                cycle_index=raw_stage["cycle_index"],
                train_split_slot=raw_stage["train_split_slot"],
                source_slot_id=raw_stage["source_slot_id"],
                source_triplet_id=raw_stage["source_triplet_id"],
                symbols=tuple(raw_symbols),
                slot_symbols=tuple(raw_slot_symbols),
                plan_identity=raw_stage["plan_identity"],
                stage_id=raw_stage["stage_id"],
                schema_version=raw_stage["schema_version"],
            )
        )
    raw_plan_slots = payload["slot_symbols"]
    if not isinstance(raw_plan_slots, list):
        raise ValueError("symbol-triplet training plan slot symbols must be a list")
    plan = SymbolTripletTrainingPlan(
        manifest_digest=cast(str, payload["manifest_digest"]),
        schedule_identity=cast(str, payload["schedule_identity"]),
        train_cycle_digest=cast(str, payload["train_cycle_digest"]),
        cycles=cast(int, payload["cycles"]),
        slot_symbols=tuple(raw_plan_slots),
        plan_identity=cast(str, payload["plan_identity"]),
        stages=tuple(stages),
        schema_version=cast(str, payload["schema_version"]),
        digest=cast(str, payload["digest"]),
    )
    plan.validate_manifest(manifest)
    return plan


def load_symbol_triplet_training_cursor(
    path: str | Path,
    *,
    plan: SymbolTripletTrainingPlan,
) -> SymbolTripletTrainingCursor:
    payload = _json_object(path, field="symbol-triplet training cursor")
    required = {
        "digest",
        "last_completed_stage_id",
        "next_stage_index",
        "plan_digest",
        "schema_version",
        "stage_count",
    }
    if set(payload) != required:
        raise ValueError("symbol-triplet training cursor field closure mismatch")
    cursor = SymbolTripletTrainingCursor(
        plan_digest=cast(str, payload["plan_digest"]),
        stage_count=cast(int, payload["stage_count"]),
        next_stage_index=cast(int, payload["next_stage_index"]),
        last_completed_stage_id=cast(str | None, payload["last_completed_stage_id"]),
        schema_version=cast(str, payload["schema_version"]),
        digest=cast(str, payload["digest"]),
    )
    cursor.validate_plan(plan)
    return cursor


__all__ = [
    "SYMBOL_TRIPLET_TRAINING_CURSOR_SCHEMA",
    "SYMBOL_TRIPLET_TRAINING_PLAN_SCHEMA",
    "SYMBOL_TRIPLET_TRAINING_STAGE_SCHEMA",
    "SymbolTripletTrainingCursor",
    "SymbolTripletTrainingPlan",
    "SymbolTripletTrainingStage",
    "advance_symbol_triplet_training_cursor",
    "build_symbol_triplet_training_plan",
    "current_symbol_triplet_training_stage",
    "initial_symbol_triplet_training_cursor",
    "load_symbol_triplet_training_cursor",
    "load_symbol_triplet_training_plan",
    "write_symbol_triplet_training_cursor",
    "write_symbol_triplet_training_plan",
]
