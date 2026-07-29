from __future__ import annotations

import itertools
import json
from pathlib import Path

import pytest

from trade_rl.workflows.symbol_triplet_manifest import (
    SYMBOL_TRIPLET_CYCLE_SIZE,
    SYMBOL_TRIPLET_SPLIT_COUNTS,
    SymbolTripletSlot,
    build_symbol_triplet_manifest,
    load_symbol_triplet_manifest,
    write_symbol_triplet_manifest,
)

SYMBOLS = tuple(f"ASSET-{index:02d}" for index in range(15))


def _appearances(slots: tuple[SymbolTripletSlot, ...]) -> dict[str, int]:
    counts = {symbol: 0 for symbol in SYMBOLS}
    for slot in slots:
        for symbol in slot.symbols:
            counts[symbol] += 1
    return counts


def test_manifest_closes_all_triplets_without_split_leakage() -> None:
    manifest = build_symbol_triplet_manifest(SYMBOLS, seed=20260729)

    assert len(manifest.slots) == SYMBOL_TRIPLET_CYCLE_SIZE
    observed = {slot.symbols for slot in manifest.slots}
    assert observed == set(itertools.combinations(SYMBOLS, 3))
    split_sets = {
        split: {slot.triplet_id for slot in manifest.slots_for(split)}
        for split in ("train", "validation", "test")
    }
    assert split_sets["train"].isdisjoint(split_sets["validation"])
    assert split_sets["train"].isdisjoint(split_sets["test"])
    assert split_sets["validation"].isdisjoint(split_sets["test"])
    assert {
        split: len(values) for split, values in split_sets.items()
    } == SYMBOL_TRIPLET_SPLIT_COUNTS


@pytest.mark.parametrize("seed", [0, 1, 42, 20260729, 2**63 - 1])
def test_each_split_has_the_minimum_possible_symbol_imbalance(seed: int) -> None:
    manifest = build_symbol_triplet_manifest(SYMBOLS, seed=seed)

    for split in ("train", "validation", "test"):
        running = {symbol: 0 for symbol in SYMBOLS}
        maximum_prefix_imbalance = 0
        slots = manifest.slots_for(split)
        for slot in slots:
            for symbol in slot.symbols:
                running[symbol] += 1
            maximum_prefix_imbalance = max(
                maximum_prefix_imbalance,
                max(running.values()) - min(running.values()),
            )
        counts = tuple(_appearances(slots).values())
        assert max(counts) - min(counts) <= 1
        assert maximum_prefix_imbalance <= 2
    complete_counts = tuple(_appearances(manifest.slots).values())
    assert set(complete_counts) == {91}


def test_seed_and_order_are_reproducible_and_identity_bound() -> None:
    first = build_symbol_triplet_manifest(SYMBOLS, seed=17)
    repeated = build_symbol_triplet_manifest(SYMBOLS, seed=17)
    changed_seed = build_symbol_triplet_manifest(SYMBOLS, seed=18)
    reordered = build_symbol_triplet_manifest(tuple(reversed(SYMBOLS)), seed=17)

    assert first == repeated
    assert first.digest == repeated.digest
    assert first.digest != changed_seed.digest
    assert [slot.triplet_id for slot in first.slots] != [
        slot.triplet_id for slot in changed_seed.slots
    ]
    assert {slot.symbols: slot.triplet_id for slot in first.slots} == {
        slot.symbols: slot.triplet_id for slot in changed_seed.slots
    }
    assert {
        member.symbol: member.symbol_id
        for slot in first.slots
        for member in slot.members
    } == {
        member.symbol: member.symbol_id
        for slot in changed_seed.slots
        for member in slot.members
    }
    assert {slot.slot_id for slot in first.slots}.isdisjoint(
        {slot.slot_id for slot in changed_seed.slots}
    )
    assert first.universe_digest != reordered.universe_digest
    assert first.digest != reordered.digest


def test_slot_member_and_symbol_identities_are_stable_and_closed() -> None:
    manifest = build_symbol_triplet_manifest(SYMBOLS, seed=7)

    assert [slot.cycle_slot for slot in manifest.slots] == list(range(455))
    assert len({slot.slot_id for slot in manifest.slots}) == 455
    assert len({slot.triplet_id for slot in manifest.slots}) == 455
    symbol_ids: dict[str, set[str]] = {symbol: set() for symbol in SYMBOLS}
    for slot in manifest.slots:
        assert [member.member_slot for member in slot.members] == [0, 1, 2]
        assert len({member.member_id for member in slot.members}) == 3
        for member in slot.members:
            symbol_ids[member.symbol].add(member.symbol_id)
    assert all(len(values) == 1 for values in symbol_ids.values())


def test_manifest_json_round_trip_and_tamper_rejection(tmp_path: Path) -> None:
    manifest = build_symbol_triplet_manifest(SYMBOLS, seed=99)
    path = write_symbol_triplet_manifest(tmp_path / "triplets.json", manifest)

    assert load_symbol_triplet_manifest(path) == manifest
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["slots"][0]["members"][0]["symbol"] = "SUBSTITUTED"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="symbols are invalid|identity mismatch"):
        load_symbol_triplet_manifest(path)


@pytest.mark.parametrize(
    "symbols,seed,message",
    [
        (SYMBOLS[:-1], 0, "exactly 15"),
        (SYMBOLS[:-1] + (SYMBOLS[0],), 0, "unique"),
        (SYMBOLS, -1, "non-negative"),
    ],
)
def test_manifest_rejects_invalid_universe_or_seed(
    symbols: tuple[str, ...], seed: int, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        build_symbol_triplet_manifest(symbols, seed=seed)
