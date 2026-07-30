from __future__ import annotations

import json
from pathlib import Path

import pytest

from trade_rl.workflows.symbol_disjoint_manifest import (
    build_symbol_disjoint_manifest,
    load_symbol_disjoint_manifest,
    write_symbol_disjoint_manifest,
)

SYMBOLS = tuple(f"ASSET-{index:02d}" for index in range(15))


def test_manifest_is_seed_stable_order_independent_and_symbol_disjoint() -> None:
    first = build_symbol_disjoint_manifest(
        SYMBOLS,
        seed=20260730,
        validation_count=3,
        test_count=3,
    )
    repeated = build_symbol_disjoint_manifest(
        tuple(reversed(SYMBOLS)),
        seed=20260730,
        validation_count=3,
        test_count=3,
    )
    changed_seed = build_symbol_disjoint_manifest(
        SYMBOLS,
        seed=20260731,
        validation_count=3,
        test_count=3,
    )

    assert first == repeated
    assert first.digest == repeated.digest
    assert first.digest != changed_seed.digest
    assert len(first.train_symbols) == 9
    assert len(first.validation_symbols) == 3
    assert len(first.test_symbols) == 3
    assert set(first.train_symbols).isdisjoint(first.validation_symbols)
    assert set(first.train_symbols).isdisjoint(first.test_symbols)
    assert set(first.validation_symbols).isdisjoint(first.test_symbols)
    assert set(first.all_symbols) == set(SYMBOLS)


def test_triplets_are_generated_only_within_each_symbol_split() -> None:
    manifest = build_symbol_disjoint_manifest(
        SYMBOLS,
        seed=17,
        validation_count=3,
        test_count=3,
    )

    for split in ("train", "validation", "test"):
        split_symbols = set(manifest.symbols_for(split))
        triplets = manifest.combinations_for(split, size=3)
        assert triplets
        assert all(set(triplet) <= split_symbols for triplet in triplets)


def test_manifest_json_round_trip_and_overlap_tamper_rejection(tmp_path: Path) -> None:
    manifest = build_symbol_disjoint_manifest(
        SYMBOLS,
        seed=99,
        validation_count=3,
        test_count=3,
    )
    path = write_symbol_disjoint_manifest(tmp_path / "symbol-disjoint.json", manifest)

    assert load_symbol_disjoint_manifest(path) == manifest
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["validation_symbols"][0] = payload["train_symbols"][0]
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="disjoint|closure|digest"):
        load_symbol_disjoint_manifest(path)


@pytest.mark.parametrize(
    "symbols,validation_count,test_count,message",
    [
        (SYMBOLS[:8], 3, 3, "minimum"),
        (SYMBOLS, 2, 3, "minimum"),
        (SYMBOLS, 3, 0, "minimum"),
        (SYMBOLS[:-1] + (SYMBOLS[0],), 3, 3, "unique"),
    ],
)
def test_manifest_rejects_invalid_symbol_disjoint_contract(
    symbols: tuple[str, ...],
    validation_count: int,
    test_count: int,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        build_symbol_disjoint_manifest(
            symbols,
            seed=0,
            validation_count=validation_count,
            test_count=test_count,
        )
