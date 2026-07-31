from __future__ import annotations

from pathlib import Path

import pytest

from trade_rl.artifacts.codec import canonical_json_bytes
from trade_rl.workflows.symbol_triplet_manifest import build_symbol_triplet_manifest
from trade_rl.workflows.symbol_triplet_stage_state import (
    SymbolTripletStageStatePointer,
    SymbolTripletStageStateStore,
)
from trade_rl.workflows.symbol_triplet_training_cursor import (
    build_symbol_triplet_training_plan,
    initial_symbol_triplet_training_cursor,
)

_SYMBOLS = tuple(f"S{index:02d}" for index in range(15))


def _plan():
    manifest = build_symbol_triplet_manifest(_SYMBOLS, seed=31)
    return build_symbol_triplet_training_plan(
        manifest,
        cycles=1,
        slot_symbols=("SLOT0", "SLOT1", "SLOT2"),
    )


def test_pointer_load_validates_one_file_snapshot(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import trade_rl.workflows.symbol_triplet_stage_state as module

    plan = _plan()
    initial = initial_symbol_triplet_training_cursor(plan)
    store = SymbolTripletStageStateStore(tmp_path / "state", plan=plan)
    original_pointer = store.initialize(initial)
    replacement_pointer = SymbolTripletStageStatePointer(
        plan_digest=original_pointer.plan_digest,
        generation_digest=original_pointer.generation_digest,
        cursor_digest=original_pointer.cursor_digest,
        completion_digest=original_pointer.completion_digest,
        previous_pointer_digest=original_pointer.digest,
    )
    replacement_bytes = canonical_json_bytes(replacement_pointer.to_json_dict())
    original_json_object = module._json_object
    replaced = False

    def read_then_replace(path: Path, *, field: str) -> dict[str, object]:
        nonlocal replaced
        value = original_json_object(path, field=field)
        if path == store.current_path and not replaced:
            replaced = True
            module.atomic_replace_bytes(path, replacement_bytes)
        return value

    monkeypatch.setattr(module, "_json_object", read_then_replace)

    completion, cursor, pointer = store.load_current()

    assert completion is None
    assert cursor == initial
    assert pointer == original_pointer
    assert store.current_path.read_bytes() == replacement_bytes
