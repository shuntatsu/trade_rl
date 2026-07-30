from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

from trade_rl.workflows.symbol_disjoint_manifest import build_symbol_disjoint_manifest
from trade_rl.workflows.symbol_disjoint_triplet_manifest import (
    build_symbol_disjoint_triplet_manifest,
    load_symbol_disjoint_triplet_manifest,
    write_symbol_disjoint_triplet_manifest,
)

_SYMBOLS = tuple(f"ASSET-{index:02d}" for index in range(15))


def _source():
    return build_symbol_disjoint_manifest(
        _SYMBOLS,
        seed=20260731,
        validation_count=3,
        test_count=3,
    )


def test_manifest_closes_combinations_inside_disjoint_splits_only() -> None:
    source = _source()
    manifest = build_symbol_disjoint_triplet_manifest(source)

    assert manifest.source_manifest_digest == source.digest
    assert manifest.split_counts == {"train": 84, "validation": 1, "test": 1}
    assert len(manifest.slots_for("train")) == 84
    assert len(manifest.slots_for("validation")) == 1
    assert len(manifest.slots_for("test")) == 1

    for split in ("train", "validation", "test"):
        allowed = set(source.symbols_for(split))
        slots = manifest.slots_for(split)
        assert slots
        assert all(set(slot.symbols) <= allowed for slot in slots)

    train_symbols = set(source.train_symbols)
    assert all(
        train_symbols.isdisjoint(slot.symbols)
        for split in ("validation", "test")
        for slot in manifest.slots_for(split)
    )


def test_train_cycle_is_balanced_deterministic_and_seed_bound() -> None:
    source = _source()
    first = build_symbol_disjoint_triplet_manifest(source)
    repeated = build_symbol_disjoint_triplet_manifest(source)
    changed = build_symbol_disjoint_triplet_manifest(
        build_symbol_disjoint_manifest(
            _SYMBOLS,
            seed=20260801,
            validation_count=3,
            test_count=3,
        )
    )

    assert first == repeated
    assert first.digest == repeated.digest
    assert first.digest != changed.digest
    counts = Counter(
        symbol for slot in first.slots_for("train") for symbol in slot.symbols
    )
    assert set(counts) == set(source.train_symbols)
    assert set(counts.values()) == {28}


def test_manifest_round_trips_and_rejects_cross_split_tampering(tmp_path: Path) -> None:
    source = _source()
    manifest = build_symbol_disjoint_triplet_manifest(source)
    path = write_symbol_disjoint_triplet_manifest(tmp_path / "manifest.json", manifest)

    assert load_symbol_disjoint_triplet_manifest(path, source=source) == manifest

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["train_symbols"][0] = payload["validation_symbols"][0]
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="disjoint|closure|digest|source"):
        load_symbol_disjoint_triplet_manifest(path, source=source)


def test_manifest_rejects_a_different_source_manifest(tmp_path: Path) -> None:
    source = _source()
    path = write_symbol_disjoint_triplet_manifest(
        tmp_path / "manifest.json",
        build_symbol_disjoint_triplet_manifest(source),
    )
    other = build_symbol_disjoint_manifest(
        _SYMBOLS,
        seed=17,
        validation_count=3,
        test_count=3,
    )
    with pytest.raises(ValueError, match="source"):
        load_symbol_disjoint_triplet_manifest(path, source=other)
