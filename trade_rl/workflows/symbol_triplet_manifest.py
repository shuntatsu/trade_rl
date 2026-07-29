"""Deterministic, balanced train/validation/test symbol-triplet manifests."""

from __future__ import annotations

import itertools
import json
from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from types import MappingProxyType
from typing import Final, Literal, cast

from trade_rl.artifacts.codec import canonical_json_bytes
from trade_rl.artifacts.hashing import content_digest
from trade_rl.domain.common import require_sha256

SYMBOL_TRIPLET_MANIFEST_SCHEMA: Final = "symbol_triplet_manifest_v1"
SYMBOL_TRIPLET_UNIVERSE_SIZE: Final = 15
SYMBOL_TRIPLET_SIZE: Final = 3
SYMBOL_TRIPLET_CYCLE_SIZE: Final = 455
SYMBOL_TRIPLET_SPLIT_COUNTS: Final[Mapping[str, int]] = MappingProxyType(
    {
        "train": 319,
        "validation": 68,
        "test": 68,
    }
)

TripletSplit = Literal["train", "validation", "test"]
IndexTriplet = tuple[int, int, int]
Orbit = tuple[IndexTriplet, ...]


def _seed_rank(seed: int, label: str, value: object) -> str:
    return content_digest(
        {
            "label": label,
            "schema_version": "seeded_symbol_triplet_rank_v1",
            "seed": seed,
            "value": value,
        }
    )


def _validate_symbols(symbols: tuple[str, ...]) -> tuple[str, ...]:
    resolved = tuple(symbols)
    if len(resolved) != SYMBOL_TRIPLET_UNIVERSE_SIZE:
        raise ValueError("symbol triplet universe must contain exactly 15 symbols")
    if any(not isinstance(symbol, str) or not symbol for symbol in resolved):
        raise ValueError("symbol triplet universe symbols must be non-empty strings")
    if len(set(resolved)) != len(resolved):
        raise ValueError("symbol triplet universe symbols must be unique")
    return resolved


def _validate_seed(seed: int) -> int:
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("symbol triplet seed must be a non-negative integer")
    return seed


def _degrees(edges: tuple[IndexTriplet, ...] | list[IndexTriplet]) -> tuple[int, ...]:
    counts = [0] * SYMBOL_TRIPLET_UNIVERSE_SIZE
    for edge in edges:
        for symbol_index in edge:
            counts[symbol_index] += 1
    return tuple(counts)


@lru_cache(maxsize=1)
def _cyclic_orbits() -> tuple[tuple[Orbit, ...], Orbit]:
    all_edges = tuple(
        itertools.combinations(range(SYMBOL_TRIPLET_UNIVERSE_SIZE), SYMBOL_TRIPLET_SIZE)
    )
    seen: set[IndexTriplet] = set()
    orbits: list[Orbit] = []
    for edge in all_edges:
        if edge in seen:
            continue
        orbit = tuple(
            sorted(
                {
                    cast(
                        IndexTriplet,
                        tuple(
                            sorted(
                                (index + offset) % SYMBOL_TRIPLET_UNIVERSE_SIZE
                                for index in edge
                            )
                        ),
                    )
                    for offset in range(SYMBOL_TRIPLET_UNIVERSE_SIZE)
                }
            )
        )
        seen.update(orbit)
        orbits.append(orbit)
    full = tuple(orbit for orbit in orbits if len(orbit) == 15)
    short = tuple(orbit for orbit in orbits if len(orbit) == 5)
    if len(full) != 30 or len(short) != 1 or len(seen) != 455:
        raise RuntimeError("complete 15-symbol triplet orbit decomposition failed")
    return full, short[0]


@lru_cache(maxsize=1)
def _balanced_partial_subsets() -> tuple[
    dict[Orbit, tuple[tuple[IndexTriplet, ...], ...]],
    dict[Orbit, tuple[tuple[tuple[IndexTriplet, ...], tuple[int, ...]], ...]],
]:
    full_orbits, _ = _cyclic_orbits()
    train_subsets: dict[Orbit, tuple[tuple[IndexTriplet, ...], ...]] = {}
    validation_subsets: dict[
        Orbit, tuple[tuple[tuple[IndexTriplet, ...], tuple[int, ...]], ...]
    ] = {}
    for orbit in full_orbits:
        train_subsets[orbit] = tuple(
            subset
            for subset in itertools.combinations(orbit, 4)
            if sorted(_degrees(list(subset))) == [0] * 3 + [1] * 12
        )
        validation_subsets[orbit] = tuple(
            (subset, degrees)
            for subset in itertools.combinations(orbit, 8)
            if sorted(degrees := _degrees(list(subset))) == [1] * 6 + [2] * 9
        )
    return train_subsets, validation_subsets


def _partition_edges(seed: int) -> dict[TripletSplit, tuple[IndexTriplet, ...]]:
    full_orbits, short_orbit = _cyclic_orbits()
    train_subsets, validation_subsets = _balanced_partial_subsets()
    ranked_orbits = sorted(
        full_orbits, key=lambda orbit: _seed_rank(seed, "full-orbit", orbit)
    )

    selected: (
        tuple[
            Orbit,
            tuple[IndexTriplet, ...],
            Orbit,
            tuple[IndexTriplet, ...],
        ]
        | None
    ) = None
    for train_partial_orbit in ranked_orbits:
        ranked_train_subsets = sorted(
            train_subsets[train_partial_orbit],
            key=lambda subset: _seed_rank(seed, "train-partial", subset),
        )
        for train_partial in ranked_train_subsets:
            train_degrees = _degrees(list(train_partial))
            train_low_symbols = {
                index for index, count in enumerate(train_degrees) if count == 0
            }
            compatible: list[tuple[str, Orbit, tuple[IndexTriplet, ...]]] = []
            for validation_partial_orbit in ranked_orbits:
                if validation_partial_orbit == train_partial_orbit:
                    continue
                for validation_partial, degrees in validation_subsets[
                    validation_partial_orbit
                ]:
                    if all(degrees[index] == 2 for index in train_low_symbols):
                        compatible.append(
                            (
                                _seed_rank(
                                    seed,
                                    "validation-partial",
                                    (validation_partial_orbit, validation_partial),
                                ),
                                validation_partial_orbit,
                                validation_partial,
                            )
                        )
            if compatible:
                _, validation_partial_orbit, validation_partial = min(compatible)
                selected = (
                    train_partial_orbit,
                    train_partial,
                    validation_partial_orbit,
                    validation_partial,
                )
                break
        if selected is not None:
            break
    if selected is None:
        raise RuntimeError("balanced triplet split partition could not be constructed")

    train_partial_orbit, train_partial, validation_partial_orbit, validation_partial = (
        selected
    )
    remaining_orbits = [
        orbit
        for orbit in ranked_orbits
        if orbit not in {train_partial_orbit, validation_partial_orbit}
    ]
    train = [edge for orbit in remaining_orbits[:21] for edge in orbit] + list(
        train_partial
    )
    validation = [edge for orbit in remaining_orbits[21:25] for edge in orbit] + list(
        validation_partial
    )
    test = (
        [edge for orbit in remaining_orbits[25:] for edge in orbit]
        + list(short_orbit)
        + [edge for edge in train_partial_orbit if edge not in train_partial]
        + [edge for edge in validation_partial_orbit if edge not in validation_partial]
    )
    result: dict[TripletSplit, tuple[IndexTriplet, ...]] = {
        "train": tuple(train),
        "validation": tuple(validation),
        "test": tuple(test),
    }
    for split in ("train", "validation", "test"):
        edges = result[split]
        expected_count = SYMBOL_TRIPLET_SPLIT_COUNTS[split]
        degrees = _degrees(list(edges))
        if len(edges) != expected_count or max(degrees) - min(degrees) > 1:
            raise RuntimeError(f"{split} triplet partition is not balanced")
    return result


def _balanced_order(
    edges: tuple[IndexTriplet, ...], *, seed: int, split: TripletSplit
) -> tuple[IndexTriplet, ...]:
    remaining = set(edges)
    counts = [0] * SYMBOL_TRIPLET_UNIVERSE_SIZE
    ordered: list[IndexTriplet] = []
    while remaining:

        def score(edge: IndexTriplet) -> tuple[int, int, str]:
            projected = counts.copy()
            for index in edge:
                projected[index] += 1
            return (
                max(projected) - min(projected),
                sum(value * value for value in projected),
                _seed_rank(seed, f"{split}-edge", edge),
            )

        selected = min(remaining, key=score)
        remaining.remove(selected)
        ordered.append(selected)
        for index in selected:
            counts[index] += 1
    return tuple(ordered)


@dataclass(frozen=True, slots=True)
class SymbolTripletMember:
    member_slot: int
    member_id: str
    symbol: str
    symbol_id: str

    def digest_payload(self) -> dict[str, object]:
        return {
            "member_id": self.member_id,
            "member_slot": self.member_slot,
            "symbol": self.symbol,
            "symbol_id": self.symbol_id,
        }


@dataclass(frozen=True, slots=True)
class SymbolTripletSlot:
    cycle_slot: int
    slot_id: str
    split: TripletSplit
    split_slot: int
    triplet_id: str
    members: tuple[SymbolTripletMember, ...]

    @property
    def symbols(self) -> tuple[str, ...]:
        return tuple(member.symbol for member in self.members)

    def digest_payload(self) -> dict[str, object]:
        return {
            "cycle_slot": self.cycle_slot,
            "members": tuple(member.digest_payload() for member in self.members),
            "slot_id": self.slot_id,
            "split": self.split,
            "split_slot": self.split_slot,
            "triplet_id": self.triplet_id,
        }


@dataclass(frozen=True, slots=True)
class SymbolTripletManifest:
    universe: tuple[str, ...]
    universe_digest: str
    seed: int
    schedule_identity: str
    slots: tuple[SymbolTripletSlot, ...]
    schema_version: str = SYMBOL_TRIPLET_MANIFEST_SCHEMA
    digest: str = ""

    def __post_init__(self) -> None:
        universe = _validate_symbols(self.universe)
        seed = _validate_seed(self.seed)
        if self.schema_version != SYMBOL_TRIPLET_MANIFEST_SCHEMA:
            raise ValueError("unsupported symbol triplet manifest schema")
        require_sha256(self.universe_digest, field="triplet_manifest.universe_digest")
        require_sha256(
            self.schedule_identity, field="triplet_manifest.schedule_identity"
        )
        expected_universe_digest = content_digest(
            {
                "schema_version": "ordered_symbol_universe_v1",
                "symbols": universe,
            }
        )
        if self.universe_digest != expected_universe_digest:
            raise ValueError("symbol triplet universe digest mismatch")
        expected_schedule_identity = content_digest(
            {
                "schema_version": "symbol_triplet_schedule_identity_v1",
                "seed": seed,
                "split_counts": SYMBOL_TRIPLET_SPLIT_COUNTS,
                "universe_digest": self.universe_digest,
            }
        )
        if self.schedule_identity != expected_schedule_identity:
            raise ValueError("symbol triplet schedule identity mismatch")
        if len(self.slots) != SYMBOL_TRIPLET_CYCLE_SIZE:
            raise ValueError("symbol triplet manifest must contain all 455 triplets")

        expected_combinations = set(itertools.combinations(universe, 3))
        observed_combinations: set[tuple[str, ...]] = set()
        split_counts = {name: 0 for name in SYMBOL_TRIPLET_SPLIT_COUNTS}
        split_symbols = {
            name: {symbol: 0 for symbol in universe}
            for name in SYMBOL_TRIPLET_SPLIT_COUNTS
        }
        expected_split_slots = {name: 0 for name in SYMBOL_TRIPLET_SPLIT_COUNTS}
        for cycle_slot, slot in enumerate(self.slots):
            if slot.cycle_slot != cycle_slot:
                raise ValueError("symbol triplet cycle slots must be contiguous")
            if slot.split not in split_counts:
                raise ValueError("symbol triplet split is invalid")
            if slot.split_slot != expected_split_slots[slot.split]:
                raise ValueError("symbol triplet split slots must be contiguous")
            expected_split_slots[slot.split] += 1
            if len(slot.members) != SYMBOL_TRIPLET_SIZE:
                raise ValueError("symbol triplet slot must contain three members")
            symbols = slot.symbols
            if len(set(symbols)) != 3 or any(
                symbol not in universe for symbol in symbols
            ):
                raise ValueError("symbol triplet slot symbols are invalid")
            if tuple(sorted(symbols, key=universe.index)) != symbols:
                raise ValueError("symbol triplet members must follow universe order")
            triplet_id = content_digest(
                {
                    "schema_version": "symbol_triplet_identity_v1",
                    "symbols": symbols,
                    "universe_digest": self.universe_digest,
                }
            )
            if slot.triplet_id != triplet_id:
                raise ValueError("symbol triplet identity mismatch")
            expected_slot_id = content_digest(
                {
                    "cycle_slot": cycle_slot,
                    "schema_version": "symbol_triplet_slot_identity_v1",
                    "schedule_identity": self.schedule_identity,
                    "triplet_id": triplet_id,
                }
            )
            if slot.slot_id != expected_slot_id:
                raise ValueError("symbol triplet slot identity mismatch")
            for member_slot, member in enumerate(slot.members):
                if member.member_slot != member_slot:
                    raise ValueError("symbol triplet member slots must be contiguous")
                expected_symbol_id = content_digest(
                    {
                        "schema_version": "universe_symbol_identity_v1",
                        "symbol": member.symbol,
                        "universe_digest": self.universe_digest,
                    }
                )
                if member.symbol_id != expected_symbol_id:
                    raise ValueError("symbol triplet member symbol identity mismatch")
                expected_member_id = content_digest(
                    {
                        "member_slot": member_slot,
                        "schema_version": "symbol_triplet_member_identity_v1",
                        "symbol_id": member.symbol_id,
                        "triplet_id": triplet_id,
                    }
                )
                if member.member_id != expected_member_id:
                    raise ValueError("symbol triplet member identity mismatch")
                split_symbols[slot.split][member.symbol] += 1
            prefix_counts = tuple(split_symbols[slot.split].values())
            if max(prefix_counts) - min(prefix_counts) > 2:
                raise ValueError("symbol triplet split traversal is imbalanced")
            observed_combinations.add(symbols)
            split_counts[slot.split] += 1
        if observed_combinations != expected_combinations:
            raise ValueError("symbol triplet manifest does not close all combinations")
        if split_counts != SYMBOL_TRIPLET_SPLIT_COUNTS:
            raise ValueError("symbol triplet split counts mismatch")
        for counts in split_symbols.values():
            values = tuple(counts.values())
            if max(values) - min(values) > 1:
                raise ValueError(
                    "symbol triplet split symbol appearances are imbalanced"
                )
        expected_digest = content_digest(self.digest_payload())
        if self.digest and self.digest != expected_digest:
            raise ValueError("symbol triplet manifest digest mismatch")
        object.__setattr__(self, "universe", universe)
        object.__setattr__(self, "seed", seed)
        object.__setattr__(self, "digest", expected_digest)

    def digest_payload(self) -> dict[str, object]:
        return {
            "schedule_identity": self.schedule_identity,
            "schema_version": self.schema_version,
            "seed": self.seed,
            "slots": tuple(slot.digest_payload() for slot in self.slots),
            "split_counts": SYMBOL_TRIPLET_SPLIT_COUNTS,
            "triplet_size": SYMBOL_TRIPLET_SIZE,
            "universe": self.universe,
            "universe_digest": self.universe_digest,
        }

    def to_json_dict(self) -> dict[str, object]:
        return {"digest": self.digest, **self.digest_payload()}

    def slots_for(self, split: TripletSplit) -> tuple[SymbolTripletSlot, ...]:
        if split not in SYMBOL_TRIPLET_SPLIT_COUNTS:
            raise ValueError("symbol triplet split is invalid")
        return tuple(slot for slot in self.slots if slot.split == split)


def build_symbol_triplet_manifest(
    symbols: tuple[str, ...], *, seed: int
) -> SymbolTripletManifest:
    """Build one complete, balanced, seed-stable cycle over all 455 triplets."""

    universe = _validate_symbols(symbols)
    resolved_seed = _validate_seed(seed)
    universe_digest = content_digest(
        {"schema_version": "ordered_symbol_universe_v1", "symbols": universe}
    )
    schedule_identity = content_digest(
        {
            "schema_version": "symbol_triplet_schedule_identity_v1",
            "seed": resolved_seed,
            "split_counts": SYMBOL_TRIPLET_SPLIT_COUNTS,
            "universe_digest": universe_digest,
        }
    )
    partitions = _partition_edges(resolved_seed)
    slots: list[SymbolTripletSlot] = []
    cycle_slot = 0
    for split in ("train", "validation", "test"):
        typed_split: TripletSplit = split
        ordered = _balanced_order(
            partitions[typed_split], seed=resolved_seed, split=typed_split
        )
        for split_slot, indices in enumerate(ordered):
            triplet_symbols = tuple(universe[index] for index in indices)
            triplet_id = content_digest(
                {
                    "schema_version": "symbol_triplet_identity_v1",
                    "symbols": triplet_symbols,
                    "universe_digest": universe_digest,
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
                        "schema_version": "symbol_triplet_member_identity_v1",
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
                    "schema_version": "symbol_triplet_slot_identity_v1",
                    "schedule_identity": schedule_identity,
                    "triplet_id": triplet_id,
                }
            )
            slots.append(
                SymbolTripletSlot(
                    cycle_slot=cycle_slot,
                    slot_id=slot_id,
                    split=typed_split,
                    split_slot=split_slot,
                    triplet_id=triplet_id,
                    members=tuple(members),
                )
            )
            cycle_slot += 1
    return SymbolTripletManifest(
        universe=universe,
        universe_digest=universe_digest,
        seed=resolved_seed,
        schedule_identity=schedule_identity,
        slots=tuple(slots),
    )


def write_symbol_triplet_manifest(
    path: str | Path, manifest: SymbolTripletManifest
) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(canonical_json_bytes(manifest.to_json_dict()))
    return output


def load_symbol_triplet_manifest(path: str | Path) -> SymbolTripletManifest:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("symbol triplet manifest must be a JSON object")
    required = {
        "digest",
        "schedule_identity",
        "schema_version",
        "seed",
        "slots",
        "split_counts",
        "triplet_size",
        "universe",
        "universe_digest",
    }
    if set(payload) != required:
        raise ValueError("symbol triplet manifest field closure mismatch")
    if payload["split_counts"] != SYMBOL_TRIPLET_SPLIT_COUNTS:
        raise ValueError("symbol triplet manifest split contract mismatch")
    if payload["triplet_size"] != SYMBOL_TRIPLET_SIZE:
        raise ValueError("symbol triplet manifest size contract mismatch")
    raw_slots = payload["slots"]
    if not isinstance(raw_slots, list):
        raise ValueError("symbol triplet manifest slots must be a list")
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
            raise ValueError("symbol triplet slot field closure mismatch")
        raw_members = raw_slot["members"]
        if not isinstance(raw_members, list):
            raise ValueError("symbol triplet members must be a list")
        members: list[SymbolTripletMember] = []
        for raw_member in raw_members:
            if not isinstance(raw_member, dict) or set(raw_member) != {
                "member_id",
                "member_slot",
                "symbol",
                "symbol_id",
            }:
                raise ValueError("symbol triplet member field closure mismatch")
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
    raw_universe = payload["universe"]
    if not isinstance(raw_universe, list):
        raise ValueError("symbol triplet universe must be a list")
    return SymbolTripletManifest(
        universe=tuple(raw_universe),
        universe_digest=payload["universe_digest"],
        seed=payload["seed"],
        schedule_identity=payload["schedule_identity"],
        slots=tuple(slots),
        schema_version=payload["schema_version"],
        digest=payload["digest"],
    )


__all__ = [
    "SYMBOL_TRIPLET_CYCLE_SIZE",
    "SYMBOL_TRIPLET_MANIFEST_SCHEMA",
    "SYMBOL_TRIPLET_SIZE",
    "SYMBOL_TRIPLET_SPLIT_COUNTS",
    "SYMBOL_TRIPLET_UNIVERSE_SIZE",
    "SymbolTripletManifest",
    "SymbolTripletMember",
    "SymbolTripletSlot",
    "build_symbol_triplet_manifest",
    "load_symbol_triplet_manifest",
    "write_symbol_triplet_manifest",
]
