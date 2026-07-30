from __future__ import annotations

import json
from pathlib import Path

import pytest

from trade_rl.workflows.symbol_triplet_manifest import build_symbol_triplet_manifest
from trade_rl.workflows.symbol_triplet_training_cursor import (
    advance_symbol_triplet_training_cursor,
    build_symbol_triplet_training_plan,
    current_symbol_triplet_training_stage,
    initial_symbol_triplet_training_cursor,
    load_symbol_triplet_training_cursor,
    load_symbol_triplet_training_plan,
    write_symbol_triplet_training_cursor,
    write_symbol_triplet_training_plan,
)

_SYMBOLS = (
    "BTCUSDT",
    "ETHUSDT",
    "BNBUSDT",
    "SOLUSDT",
    "XRPUSDT",
    "ADAUSDT",
    "DOGEUSDT",
    "LINKUSDT",
    "LTCUSDT",
    "BCHUSDT",
    "DOTUSDT",
    "AVAXUSDT",
    "UNIUSDT",
    "TRXUSDT",
    "ETCUSDT",
)
_SLOT_SYMBOLS = ("SLOT0", "SLOT1", "SLOT2")


def _manifest():
    return build_symbol_triplet_manifest(_SYMBOLS, seed=31)


def test_plan_repeats_the_balanced_train_split_in_manifest_order() -> None:
    manifest = _manifest()
    plan = build_symbol_triplet_training_plan(
        manifest,
        cycles=2,
        slot_symbols=_SLOT_SYMBOLS,
    )
    train_slots = manifest.slots_for("train")

    assert plan.cycles == 2
    assert plan.stage_count == 638
    assert len(plan.digest) == 64
    assert tuple(stage.source_slot_id for stage in plan.stages[:319]) == tuple(
        slot.slot_id for slot in train_slots
    )
    assert tuple(stage.source_slot_id for stage in plan.stages[319:]) == tuple(
        slot.slot_id for slot in train_slots
    )
    assert all(stage.slot_symbols == _SLOT_SYMBOLS for stage in plan.stages)
    assert all(len(stage.stage_id) == 64 for stage in plan.stages)


def test_plan_and_cursor_are_deterministic_and_resume_exactly(tmp_path: Path) -> None:
    manifest = _manifest()
    left = build_symbol_triplet_training_plan(
        manifest,
        cycles=2,
        slot_symbols=_SLOT_SYMBOLS,
    )
    right = build_symbol_triplet_training_plan(
        manifest,
        cycles=2,
        slot_symbols=_SLOT_SYMBOLS,
    )
    assert left == right

    cursor = initial_symbol_triplet_training_cursor(left)
    for _ in range(7):
        stage = current_symbol_triplet_training_stage(left, cursor)
        assert stage is not None
        cursor = advance_symbol_triplet_training_cursor(
            left,
            cursor,
            completed_stage_id=stage.stage_id,
        )

    plan_path = write_symbol_triplet_training_plan(tmp_path / "plan.json", left)
    cursor_path = write_symbol_triplet_training_cursor(
        tmp_path / "cursor.json",
        cursor,
    )
    loaded_plan = load_symbol_triplet_training_plan(plan_path, manifest=manifest)
    loaded_cursor = load_symbol_triplet_training_cursor(
        cursor_path,
        plan=loaded_plan,
    )

    assert loaded_plan == left
    assert loaded_cursor == cursor
    assert (
        current_symbol_triplet_training_stage(loaded_plan, loaded_cursor)
        == left.stages[7]
    )


def test_cursor_rejects_wrong_or_replayed_stage_completion() -> None:
    plan = build_symbol_triplet_training_plan(
        _manifest(),
        cycles=1,
        slot_symbols=_SLOT_SYMBOLS,
    )
    cursor = initial_symbol_triplet_training_cursor(plan)

    with pytest.raises(ValueError, match="current training stage"):
        advance_symbol_triplet_training_cursor(
            plan,
            cursor,
            completed_stage_id=plan.stages[1].stage_id,
        )

    completed = advance_symbol_triplet_training_cursor(
        plan,
        cursor,
        completed_stage_id=plan.stages[0].stage_id,
    )
    with pytest.raises(ValueError, match="current training stage"):
        advance_symbol_triplet_training_cursor(
            plan,
            completed,
            completed_stage_id=plan.stages[0].stage_id,
        )


def test_cursor_rejects_a_different_training_plan() -> None:
    manifest = _manifest()
    first = build_symbol_triplet_training_plan(
        manifest,
        cycles=1,
        slot_symbols=_SLOT_SYMBOLS,
    )
    second = build_symbol_triplet_training_plan(
        manifest,
        cycles=2,
        slot_symbols=_SLOT_SYMBOLS,
    )
    cursor = initial_symbol_triplet_training_cursor(first)

    with pytest.raises(ValueError, match="plan digest"):
        current_symbol_triplet_training_stage(second, cursor)


def test_completed_cursor_has_no_next_stage() -> None:
    plan = build_symbol_triplet_training_plan(
        _manifest(),
        cycles=1,
        slot_symbols=_SLOT_SYMBOLS,
    )
    cursor = initial_symbol_triplet_training_cursor(plan)
    for stage in plan.stages:
        cursor = advance_symbol_triplet_training_cursor(
            plan,
            cursor,
            completed_stage_id=stage.stage_id,
        )

    assert cursor.next_stage_index == plan.stage_count
    assert current_symbol_triplet_training_stage(plan, cursor) is None


def test_serialized_cursor_tampering_is_rejected(tmp_path: Path) -> None:
    plan = build_symbol_triplet_training_plan(
        _manifest(),
        cycles=1,
        slot_symbols=_SLOT_SYMBOLS,
    )
    cursor_path = write_symbol_triplet_training_cursor(
        tmp_path / "cursor.json",
        initial_symbol_triplet_training_cursor(plan),
    )
    payload = json.loads(cursor_path.read_text(encoding="utf-8"))
    payload["next_stage_index"] = 2
    cursor_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="cursor digest"):
        load_symbol_triplet_training_cursor(cursor_path, plan=plan)


def test_plan_rejects_non_positive_cycles() -> None:
    with pytest.raises(ValueError, match="cycles"):
        build_symbol_triplet_training_plan(
            _manifest(),
            cycles=0,
            slot_symbols=_SLOT_SYMBOLS,
        )
