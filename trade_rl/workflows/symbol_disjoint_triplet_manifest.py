"""Triplet schedules whose symbols remain disjoint across evaluation splits."""

from __future__ import annotations

import itertools
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Final, cast

from trade_rl.artifacts.codec import canonical_json_bytes
from trade_rl.artifacts.hashing import content_digest
from trade_rl.domain.common import require_sha256, require_unique_non_empty
from trade_rl.workflows.symbol_disjoint_manifest import (
    SymbolDisjointManifest,
    SymbolSplit,
)
from trade_rl.workflows.symbol_triplet_manifest import (
    SYMBOL_TRIPLET_SIZE,
    SymbolTripletMember,
    SymbolTripletSlot,
    TripletSplit,
)

SYMBOL_DISJOINT_TRIPLET_MANIFEST_SCHEMA: Final = "symbol_disjoint_triplet_manifest_v1"
_SPLITS: Final[tuple[SymbolSplit, ...]] = ("train", "validation", "test")


def _positive_count(symbol_count: int) -> int:
    if symbol_count < SYMBOL_TRIPLET_SIZE:
        raise ValueError(
            "each symbol-disjoint split must contain at least three symbols"
        )
    return len(tuple(itertools.combinations(range(symbol_count), SYMBOL_TRIPLET_SIZE)))


def _rank(seed: int, split: SymbolSplit, symbols: tuple[str, ...]) -> str:
    return content_digest(
        {
            "schema_version": "symbol_disjoint_triplet_rank_v1",
            "seed": seed,
            "split": split,
            "symbols": symbols,
        }
    )


def _balanced_order(
    symbols: tuple[str, ...], *, seed: int, split: SymbolSplit
) -> tuple[tuple[str, ...], ...]:
    remaining = set(itertools.combinations(symbols, SYMBOL_TRIPLET_SIZE))
    counts = {symbol: 0 for symbol in symbols}
    ordered: list[tuple[str, ...]] = []
    while remaining:

        def score(candidate: tuple[str, ...]) -> tuple[int, int, str]:
            projected = dict(counts)
            for symbol in candidate:
                projected[symbol] += 1
            values = tuple(projected.values())
            return (
                max(values) - min(values),
                sum(value * value for value in values),
                _rank(seed, split, candidate),
            )

        selected = min(remaining, key=score)
        remaining.remove(selected)
        ordered.append(selected)
        for symbol in selected:
            counts[symbol] += 1
    return tuple(ordered)


def _split_counts(
    train: tuple[str, ...],
    validation: tuple[str, ...],
    test: tuple[str, ...],
) -> Mapping[str, int]:
    return MappingProxyType(
        {
            "train": _positive_count(len(train)),
            "validation": _positive_count(len(validation)),
            "test": _positive_count(len(test)),
        }
    )


@dataclass(frozen=True, slots=True)
class SymbolDisjointTripletManifest:
    """All within-split triplets derived from one symbol-disjoint partition."""

    source_manifest_digest: str
    universe: tuple[str, ...]
    universe_digest: str
    seed: int
    train_symbols: tuple[str, ...]
    validation_symbols: tuple[str, ...]
    test_symbols: tuple[str, ...]
    schedule_identity: str
    slots: tuple[SymbolTripletSlot, ...]
    schema_version: str = SYMBOL_DISJOINT_TRIPLET_MANIFEST_SCHEMA
    digest: str = ""

    def __post_init__(self) -> None:
        if self.schema_version != SYMBOL_DISJOINT_TRIPLET_MANIFEST_SCHEMA:
            raise ValueError("unsupported symbol-disjoint triplet manifest schema")
        require_sha256(
            self.source_manifest_digest,
            field="symbol_disjoint_triplet.source_manifest_digest",
        )
        require_sha256(
            self.universe_digest,
            field="symbol_disjoint_triplet.universe_digest",
        )
        require_sha256(
            self.schedule_identity,
            field="symbol_disjoint_triplet.schedule_identity",
        )
        if (
            isinstance(self.seed, bool)
            or not isinstance(self.seed, int)
            or self.seed < 0
        ):
            raise ValueError("symbol-disjoint triplet seed must be non-negative")
        universe = tuple(
            require_unique_non_empty(
                tuple(self.universe), field="symbol_disjoint_triplet.universe"
            )
        )
        train = tuple(
            require_unique_non_empty(
                tuple(self.train_symbols), field="symbol_disjoint_triplet.train_symbols"
            )
        )
        validation = tuple(
            require_unique_non_empty(
                tuple(self.validation_symbols),
                field="symbol_disjoint_triplet.validation_symbols",
            )
        )
        test = tuple(
            require_unique_non_empty(
                tuple(self.test_symbols), field="symbol_disjoint_triplet.test_symbols"
            )
        )
        if not set(train).isdisjoint(validation) or not set(train).isdisjoint(test):
            raise ValueError("symbol-disjoint triplet splits must be disjoint")
        if not set(validation).isdisjoint(test):
            raise ValueError("symbol-disjoint triplet splits must be disjoint")
        if set(train) | set(validation) | set(test) != set(universe):
            raise ValueError("symbol-disjoint triplet split closure mismatch")
        order = {symbol: index for index, symbol in enumerate(universe)}
        for split_symbols in (train, validation, test):
            if tuple(sorted(split_symbols, key=order.__getitem__)) != split_symbols:
                raise ValueError("symbol-disjoint triplet split order mismatch")
        expected_universe_digest = content_digest(
            {
                "schema_version": "symbol_disjoint_triplet_universe_v1",
                "symbols": universe,
                "source_manifest_digest": self.source_manifest_digest,
            }
        )
        if self.universe_digest != expected_universe_digest:
            raise ValueError("symbol-disjoint triplet universe digest mismatch")
        counts = _split_counts(train, validation, test)
        expected_schedule_identity = content_digest(
            {
                "schema_version": "symbol_disjoint_triplet_schedule_identity_v1",
                "seed": self.seed,
                "source_manifest_digest": self.source_manifest_digest,
                "split_counts": counts,
                "test_symbols": test,
                "train_symbols": train,
                "universe_digest": self.universe_digest,
                "validation_symbols": validation,
            }
        )
        if self.schedule_identity != expected_schedule_identity:
            raise ValueError("symbol-disjoint triplet schedule identity mismatch")
        expected_total = sum(counts.values())
        if len(self.slots) != expected_total:
            raise ValueError("symbol-disjoint triplet slot count mismatch")
        expected_split_slot = {split: 0 for split in _SPLITS}
        observed: dict[SymbolSplit, set[tuple[str, ...]]] = {
            split: set() for split in _SPLITS
        }
        appearances: dict[SymbolSplit, dict[str, int]] = {
            "train": {symbol: 0 for symbol in train},
            "validation": {symbol: 0 for symbol in validation},
            "test": {symbol: 0 for symbol in test},
        }
        symbols_by_split = {
            "train": train,
            "validation": validation,
            "test": test,
        }
        for cycle_slot, slot in enumerate(self.slots):
            split = cast(SymbolSplit, slot.split)
            if split not in _SPLITS:
                raise ValueError("symbol-disjoint triplet split is invalid")
            if slot.cycle_slot != cycle_slot:
                raise ValueError(
                    "symbol-disjoint triplet cycle slots must be contiguous"
                )
            if slot.split_slot != expected_split_slot[split]:
                raise ValueError(
                    "symbol-disjoint triplet split slots must be contiguous"
                )
            expected_split_slot[split] += 1
            allowed = symbols_by_split[split]
            triplet_symbols = slot.symbols
            if len(triplet_symbols) != SYMBOL_TRIPLET_SIZE:
                raise ValueError("symbol-disjoint triplet must contain three symbols")
            if not set(triplet_symbols) <= set(allowed):
                raise ValueError("symbol-disjoint triplet crosses split boundaries")
            if tuple(sorted(triplet_symbols, key=order.__getitem__)) != triplet_symbols:
                raise ValueError("symbol-disjoint triplet member order mismatch")
            triplet_id = content_digest(
                {
                    "schema_version": "symbol_disjoint_triplet_identity_v1",
                    "source_manifest_digest": self.source_manifest_digest,
                    "split": split,
                    "symbols": triplet_symbols,
                }
            )
            if slot.triplet_id != triplet_id:
                raise ValueError("symbol-disjoint triplet identity mismatch")
            expected_slot_id = content_digest(
                {
                    "cycle_slot": cycle_slot,
                    "schema_version": "symbol_disjoint_triplet_slot_identity_v1",
                    "schedule_identity": self.schedule_identity,
                    "triplet_id": triplet_id,
                }
            )
            if slot.slot_id != expected_slot_id:
                raise ValueError("symbol-disjoint triplet slot identity mismatch")
            if len(slot.members) != SYMBOL_TRIPLET_SIZE:
                raise ValueError("symbol-disjoint triplet member count mismatch")
            for member_slot, member in enumerate(slot.members):
                if (
                    member.member_slot != member_slot
                    or member.symbol != triplet_symbols[member_slot]
                ):
                    raise ValueError("symbol-disjoint triplet member binding mismatch")
                symbol_id = content_digest(
                    {
                        "schema_version": "universe_symbol_identity_v1",
                        "symbol": member.symbol,
                        "universe_digest": self.universe_digest,
                    }
                )
                if member.symbol_id != symbol_id:
                    raise ValueError("symbol-disjoint triplet symbol identity mismatch")
                member_id = content_digest(
                    {
                        "member_slot": member_slot,
                        "schema_version": "symbol_disjoint_triplet_member_identity_v1",
                        "symbol_id": symbol_id,
                        "triplet_id": triplet_id,
                    }
                )
                if member.member_id != member_id:
                    raise ValueError("symbol-disjoint triplet member identity mismatch")
                appearances[split][member.symbol] += 1
            values = tuple(appearances[split].values())
            if max(values) - min(values) > 2:
                raise ValueError("symbol-disjoint triplet traversal is imbalanced")
            observed[split].add(triplet_symbols)
        for split in _SPLITS:
            expected = set(
                itertools.combinations(symbols_by_split[split], SYMBOL_TRIPLET_SIZE)
            )
            if observed[split] != expected:
                raise ValueError("symbol-disjoint triplet combination closure mismatch")
            values = tuple(appearances[split].values())
            if len(set(values)) != 1:
                raise ValueError(
                    "symbol-disjoint triplet final appearances are imbalanced"
                )
        expected_digest = content_digest(self.digest_payload())
        if self.digest and self.digest != expected_digest:
            raise ValueError("symbol-disjoint triplet manifest digest mismatch")
        object.__setattr__(self, "universe", universe)
        object.__setattr__(self, "train_symbols", train)
        object.__setattr__(self, "validation_symbols", validation)
        object.__setattr__(self, "test_symbols", test)
        object.__setattr__(self, "digest", expected_digest)

    @property
    def split_counts(self) -> dict[str, int]:
        return dict(
            _split_counts(
                self.train_symbols,
                self.validation_symbols,
                self.test_symbols,
            )
        )

    def symbols_for(self, split: SymbolSplit) -> tuple[str, ...]:
        if split == "train":
            return self.train_symbols
        if split == "validation":
            return self.validation_symbols
        if split == "test":
            return self.test_symbols
        raise ValueError("symbol-disjoint triplet split is invalid")

    def slots_for(self, split: TripletSplit) -> tuple[SymbolTripletSlot, ...]:
        if split not in _SPLITS:
            raise ValueError("symbol-disjoint triplet split is invalid")
        return tuple(slot for slot in self.slots if slot.split == split)

    def validate_source(self, source: SymbolDisjointManifest) -> None:
        if source.digest != self.source_manifest_digest:
            raise ValueError("symbol-disjoint triplet source manifest mismatch")
        expected = (
            source.source_universe,
            source.seed,
            source.train_symbols,
            source.validation_symbols,
            source.test_symbols,
        )
        actual = (
            self.universe,
            self.seed,
            self.train_symbols,
            self.validation_symbols,
            self.test_symbols,
        )
        if actual != expected:
            raise ValueError("symbol-disjoint triplet source binding mismatch")

    def digest_payload(self) -> dict[str, object]:
        return {
            "schedule_identity": self.schedule_identity,
            "schema_version": self.schema_version,
            "seed": self.seed,
            "slots": tuple(slot.digest_payload() for slot in self.slots),
            "source_manifest_digest": self.source_manifest_digest,
            "split_counts": self.split_counts,
            "test_symbols": self.test_symbols,
            "train_symbols": self.train_symbols,
            "triplet_size": SYMBOL_TRIPLET_SIZE,
            "universe": self.universe,
            "universe_digest": self.universe_digest,
            "validation_symbols": self.validation_symbols,
        }

    def to_json_dict(self) -> dict[str, object]:
        return {"digest": self.digest, **self.digest_payload()}


def build_symbol_disjoint_triplet_manifest(
    source: SymbolDisjointManifest,
) -> SymbolDisjointTripletManifest:
    universe = source.source_universe
    universe_digest = content_digest(
        {
            "schema_version": "symbol_disjoint_triplet_universe_v1",
            "symbols": universe,
            "source_manifest_digest": source.digest,
        }
    )
    counts = _split_counts(
        source.train_symbols,
        source.validation_symbols,
        source.test_symbols,
    )
    schedule_identity = content_digest(
        {
            "schema_version": "symbol_disjoint_triplet_schedule_identity_v1",
            "seed": source.seed,
            "source_manifest_digest": source.digest,
            "split_counts": counts,
            "test_symbols": source.test_symbols,
            "train_symbols": source.train_symbols,
            "universe_digest": universe_digest,
            "validation_symbols": source.validation_symbols,
        }
    )
    slots: list[SymbolTripletSlot] = []
    cycle_slot = 0
    for split in _SPLITS:
        for split_slot, triplet_symbols in enumerate(
            _balanced_order(source.symbols_for(split), seed=source.seed, split=split)
        ):
            triplet_id = content_digest(
                {
                    "schema_version": "symbol_disjoint_triplet_identity_v1",
                    "source_manifest_digest": source.digest,
                    "split": split,
                    "symbols": triplet_symbols,
                }
            )
            members: list[SymbolTripletMember] = []
            for member_slot, symbol in enumerate(triplet_symbols):
                symbol_id = content_digest(
                    {
                        "schema_version": "universe_symbol_identity_v1",
                        "symbol": symbol,
                        "universe_digest": universe_digest,
                    }
                )
                member_id = content_digest(
                    {
                        "member_slot": member_slot,
                        "schema_version": "symbol_disjoint_triplet_member_identity_v1",
                        "symbol_id": symbol_id,
                        "triplet_id": triplet_id,
                    }
                )
                members.append(
                    SymbolTripletMember(
                        member_slot=member_slot,
                        member_id=member_id,
                        symbol=symbol,
                        symbol_id=symbol_id,
                    )
                )
            slot_id = content_digest(
                {
                    "cycle_slot": cycle_slot,
                    "schema_version": "symbol_disjoint_triplet_slot_identity_v1",
                    "schedule_identity": schedule_identity,
                    "triplet_id": triplet_id,
                }
            )
            slots.append(
                SymbolTripletSlot(
                    cycle_slot=cycle_slot,
                    slot_id=slot_id,
                    split=cast(TripletSplit, split),
                    split_slot=split_slot,
                    triplet_id=triplet_id,
                    members=tuple(members),
                )
            )
            cycle_slot += 1
    manifest = SymbolDisjointTripletManifest(
        source_manifest_digest=source.digest,
        universe=universe,
        universe_digest=universe_digest,
        seed=source.seed,
        train_symbols=source.train_symbols,
        validation_symbols=source.validation_symbols,
        test_symbols=source.test_symbols,
        schedule_identity=schedule_identity,
        slots=tuple(slots),
    )
    manifest.validate_source(source)
    return manifest


def write_symbol_disjoint_triplet_manifest(
    path: str | Path, manifest: SymbolDisjointTripletManifest
) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(canonical_json_bytes(manifest.to_json_dict()))
    return output


def load_symbol_disjoint_triplet_manifest(
    path: str | Path, *, source: SymbolDisjointManifest | None = None
) -> SymbolDisjointTripletManifest:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("symbol-disjoint triplet manifest must be a JSON object")
    required = {
        "digest",
        "schedule_identity",
        "schema_version",
        "seed",
        "slots",
        "source_manifest_digest",
        "split_counts",
        "test_symbols",
        "train_symbols",
        "triplet_size",
        "universe",
        "universe_digest",
        "validation_symbols",
    }
    if set(payload) != required:
        raise ValueError("symbol-disjoint triplet manifest field closure mismatch")
    if payload["triplet_size"] != SYMBOL_TRIPLET_SIZE:
        raise ValueError("symbol-disjoint triplet size contract mismatch")
    raw_slots = payload["slots"]
    if not isinstance(raw_slots, list):
        raise ValueError("symbol-disjoint triplet slots must be a list")
    slots: list[SymbolTripletSlot] = []
    for raw_slot in raw_slots:
        if not isinstance(raw_slot, dict) or set(raw_slot) != {
            "cycle_slot",
            "members",
            "slot_id",
            "split",
            "split_slot",
            "triplet_id",
        }:
            raise ValueError("symbol-disjoint triplet slot field closure mismatch")
        raw_members = raw_slot["members"]
        if not isinstance(raw_members, list):
            raise ValueError("symbol-disjoint triplet members must be a list")
        members: list[SymbolTripletMember] = []
        for raw_member in raw_members:
            if not isinstance(raw_member, dict) or set(raw_member) != {
                "member_id",
                "member_slot",
                "symbol",
                "symbol_id",
            }:
                raise ValueError(
                    "symbol-disjoint triplet member field closure mismatch"
                )
            members.append(SymbolTripletMember(**raw_member))
        slots.append(
            SymbolTripletSlot(
                cycle_slot=raw_slot["cycle_slot"],
                slot_id=raw_slot["slot_id"],
                split=raw_slot["split"],
                split_slot=raw_slot["split_slot"],
                triplet_id=raw_slot["triplet_id"],
                members=tuple(members),
            )
        )
    sequence_fields = (
        "universe",
        "train_symbols",
        "validation_symbols",
        "test_symbols",
    )
    if any(not isinstance(payload[field], list) for field in sequence_fields):
        raise ValueError("symbol-disjoint triplet symbol fields must be lists")
    manifest = SymbolDisjointTripletManifest(
        source_manifest_digest=payload["source_manifest_digest"],
        universe=tuple(payload["universe"]),
        universe_digest=payload["universe_digest"],
        seed=payload["seed"],
        train_symbols=tuple(payload["train_symbols"]),
        validation_symbols=tuple(payload["validation_symbols"]),
        test_symbols=tuple(payload["test_symbols"]),
        schedule_identity=payload["schedule_identity"],
        slots=tuple(slots),
        schema_version=payload["schema_version"],
        digest=payload["digest"],
    )
    if payload["split_counts"] != manifest.split_counts:
        raise ValueError("symbol-disjoint triplet split count mismatch")
    if source is not None:
        manifest.validate_source(source)
    return manifest


__all__ = [
    "SYMBOL_DISJOINT_TRIPLET_MANIFEST_SCHEMA",
    "SymbolDisjointTripletManifest",
    "build_symbol_disjoint_triplet_manifest",
    "load_symbol_disjoint_triplet_manifest",
    "write_symbol_disjoint_triplet_manifest",
]
