from __future__ import annotations

from pathlib import Path

import pytest

from trade_rl.artifacts.atomic_pointer import AtomicReplaceDurabilityError
from trade_rl.rl.checkpointing import publish_checkpoint
from trade_rl.workflows.symbol_triplet_manifest import build_symbol_triplet_manifest
from trade_rl.workflows.symbol_triplet_stage_orchestrator import (
    SymbolTripletStageCheckpoint,
    SymbolTripletStageCompletion,
)
from trade_rl.workflows.symbol_triplet_stage_state import SymbolTripletStageStateStore
from trade_rl.workflows.symbol_triplet_training_cursor import (
    advance_symbol_triplet_training_cursor,
    build_symbol_triplet_training_plan,
    initial_symbol_triplet_training_cursor,
)

_SYMBOLS = tuple(f"S{index:02d}" for index in range(15))
_SEEDS = (0, 1)


class _Policy:
    def save(self, path: str) -> None:
        Path(path).with_suffix(".zip").write_bytes(b"policy")


def _plan():
    manifest = build_symbol_triplet_manifest(_SYMBOLS, seed=31)
    return build_symbol_triplet_training_plan(
        manifest,
        cycles=1,
        slot_symbols=("SLOT0", "SLOT1", "SLOT2"),
    )


def _completion(plan, tmp_path: Path):
    references = []
    for seed in _SEEDS:
        manifest = publish_checkpoint(
            model=_Policy(),
            checkpoint_root=tmp_path / f"seed-{seed}",
            algorithm="ppo",
            seed=seed,
            requested_timestep=8,
            observed_timestep=8,
            environment_digest="1" * 64,
            training_config_digest="2" * 64,
        )
        references.append(
            SymbolTripletStageCheckpoint(
                seed=seed,
                checkpoint_root=manifest.policy_path.parent,
                checkpoint_digest=manifest.digest,
            )
        )
    return SymbolTripletStageCompletion(
        plan_digest=plan.digest,
        stage_id=plan.stages[0].stage_id,
        stage_index=0,
        training_seeds=_SEEDS,
        checkpoints=tuple(references),
    )


def test_generation_pointer_is_the_only_committed_state(tmp_path: Path) -> None:
    plan = _plan()
    initial = initial_symbol_triplet_training_cursor(plan)
    completion = _completion(plan, tmp_path / "checkpoints")
    advanced = advance_symbol_triplet_training_cursor(
        plan,
        initial,
        completed_stage_id=completion.stage_id,
        completion_digest=completion.digest,
    )
    store = SymbolTripletStageStateStore(tmp_path / "state", plan=plan)
    first = store.initialize(initial)

    pointer = store.commit(
        expected_cursor_digest=initial.digest,
        completion=completion,
        cursor=advanced,
    )
    loaded_completion, loaded_cursor, loaded_pointer = store.load_current()

    assert loaded_completion == completion
    assert loaded_cursor == advanced
    assert loaded_pointer == pointer
    assert pointer.previous_pointer_digest == first.digest
    assert pointer.cursor_digest == advanced.digest
    assert pointer.completion_digest == completion.digest


def test_pre_pointer_failure_leaves_orphan_generation_uncommitted(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import trade_rl.workflows.symbol_triplet_stage_state as module

    plan = _plan()
    initial = initial_symbol_triplet_training_cursor(plan)
    completion = _completion(plan, tmp_path / "checkpoints")
    advanced = advance_symbol_triplet_training_cursor(
        plan,
        initial,
        completed_stage_id=completion.stage_id,
        completion_digest=completion.digest,
    )
    store = SymbolTripletStageStateStore(tmp_path / "state", plan=plan)
    original = store.initialize(initial)

    monkeypatch.setattr(
        module,
        "atomic_replace_bytes",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("before replace")),
    )
    with pytest.raises(OSError, match="before replace"):
        store.commit(
            expected_cursor_digest=initial.digest,
            completion=completion,
            cursor=advanced,
        )

    loaded_completion, loaded_cursor, loaded_pointer = store.load_current()
    assert loaded_completion is None
    assert loaded_cursor == initial
    assert loaded_pointer == original
    assert len(tuple((store.root / "generations").iterdir())) == 2


def test_post_replace_durability_failure_keeps_new_generation_committed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import trade_rl.artifacts.atomic_pointer as atomic_pointer

    plan = _plan()
    initial = initial_symbol_triplet_training_cursor(plan)
    completion = _completion(plan, tmp_path / "checkpoints")
    advanced = advance_symbol_triplet_training_cursor(
        plan,
        initial,
        completed_stage_id=completion.stage_id,
        completion_digest=completion.digest,
    )
    store = SymbolTripletStageStateStore(tmp_path / "state", plan=plan)
    store.initialize(initial)

    monkeypatch.setattr(
        atomic_pointer,
        "_fsync_directory",
        lambda _path: (_ for _ in ()).throw(OSError("fsync failed")),
    )
    with pytest.raises(AtomicReplaceDurabilityError):
        store.commit(
            expected_cursor_digest=initial.digest,
            completion=completion,
            cursor=advanced,
        )

    loaded_completion, loaded_cursor, _ = store.load_current()
    assert loaded_completion == completion
    assert loaded_cursor == advanced


def test_stale_cursor_digest_cannot_advance_generation(tmp_path: Path) -> None:
    plan = _plan()
    initial = initial_symbol_triplet_training_cursor(plan)
    completion = _completion(plan, tmp_path / "checkpoints")
    advanced = advance_symbol_triplet_training_cursor(
        plan,
        initial,
        completed_stage_id=completion.stage_id,
        completion_digest=completion.digest,
    )
    store = SymbolTripletStageStateStore(tmp_path / "state", plan=plan)
    store.initialize(initial)
    store.commit(
        expected_cursor_digest=initial.digest,
        completion=completion,
        cursor=advanced,
    )

    with pytest.raises(ValueError, match="stale"):
        store.commit(
            expected_cursor_digest=initial.digest,
            completion=completion,
            cursor=advanced,
        )


def test_load_or_migrate_prefers_committed_generation_over_stale_legacy_cursor(
    tmp_path: Path,
) -> None:
    from trade_rl.workflows.symbol_triplet_stage_state import (
        load_or_migrate_symbol_triplet_stage_state,
    )
    from trade_rl.workflows.symbol_triplet_training_cursor import (
        write_symbol_triplet_training_cursor,
    )

    plan = _plan()
    initial = initial_symbol_triplet_training_cursor(plan)
    legacy_cursor_path = write_symbol_triplet_training_cursor(
        tmp_path / "cursor.json",
        initial,
    )
    state_root = tmp_path / "state"
    legacy_bytes = legacy_cursor_path.read_bytes()

    completion0, cursor0, _ = load_or_migrate_symbol_triplet_stage_state(
        plan=plan,
        state_root=state_root,
        legacy_cursor_path=legacy_cursor_path,
        legacy_completion_path=None,
    )
    assert completion0 is None
    assert cursor0 == initial

    completion = _completion(plan, tmp_path / "checkpoints")
    advanced = advance_symbol_triplet_training_cursor(
        plan,
        initial,
        completed_stage_id=completion.stage_id,
        completion_digest=completion.digest,
    )
    store = SymbolTripletStageStateStore(state_root, plan=plan)
    store.commit(
        expected_cursor_digest=initial.digest,
        completion=completion,
        cursor=advanced,
    )

    loaded_completion, loaded_cursor, _ = load_or_migrate_symbol_triplet_stage_state(
        plan=plan,
        state_root=state_root,
        legacy_cursor_path=legacy_cursor_path,
        legacy_completion_path=None,
    )
    assert loaded_completion == completion
    assert loaded_cursor == advanced
    assert legacy_cursor_path.read_bytes() == legacy_bytes


def test_orchestrator_commits_through_generation_store_without_mutating_legacy_files(
    tmp_path: Path,
) -> None:
    from trade_rl.workflows.symbol_triplet_stage_orchestrator import (
        build_symbol_triplet_stage_request,
        commit_symbol_triplet_stage_completion,
    )
    from trade_rl.workflows.symbol_triplet_training_cursor import (
        write_symbol_triplet_training_cursor,
    )

    plan = _plan()
    initial = initial_symbol_triplet_training_cursor(plan)
    legacy_cursor_path = write_symbol_triplet_training_cursor(
        tmp_path / "cursor.json",
        initial,
    )
    legacy_bytes = legacy_cursor_path.read_bytes()
    state_root = tmp_path / "state"
    SymbolTripletStageStateStore(state_root, plan=plan).initialize(initial)
    request = build_symbol_triplet_stage_request(
        plan,
        initial,
        training_seeds=_SEEDS,
        previous_completion=None,
    )
    assert request is not None
    checkpoint_roots = {
        checkpoint.seed: checkpoint.checkpoint_root
        for checkpoint in _completion(plan, tmp_path / "checkpoints").checkpoints
    }

    completion, advanced = commit_symbol_triplet_stage_completion(
        plan,
        initial,
        request=request,
        checkpoint_roots=checkpoint_roots,
        completion_path=tmp_path / "legacy-completion.json",
        cursor_path=legacy_cursor_path,
        stage_state_root=state_root,
    )

    loaded_completion, loaded_cursor, _ = SymbolTripletStageStateStore(
        state_root,
        plan=plan,
    ).load_current()
    assert loaded_completion == completion
    assert loaded_cursor == advanced
    assert legacy_cursor_path.read_bytes() == legacy_bytes
    assert not (tmp_path / "legacy-completion.json").exists()


def test_retry_repairs_partial_uncommitted_generation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import trade_rl.workflows.symbol_triplet_stage_state as module

    plan = _plan()
    initial = initial_symbol_triplet_training_cursor(plan)
    completion = _completion(plan, tmp_path / "checkpoints")
    advanced = advance_symbol_triplet_training_cursor(
        plan,
        initial,
        completed_stage_id=completion.stage_id,
        completion_digest=completion.digest,
    )
    store = SymbolTripletStageStateStore(tmp_path / "state", plan=plan)
    original_pointer = store.initialize(initial)
    original_write = module._write_exclusive
    failed = False

    def interrupt_completion(path: Path, payload: bytes) -> None:
        nonlocal failed
        if path.name == "completion.json" and not failed:
            failed = True
            raise OSError("interrupted generation")
        original_write(path, payload)

    monkeypatch.setattr(module, "_write_exclusive", interrupt_completion)
    with pytest.raises(OSError, match="interrupted generation"):
        store.commit(
            expected_cursor_digest=initial.digest,
            completion=completion,
            cursor=advanced,
        )
    assert store.load_current()[2] == original_pointer

    monkeypatch.setattr(module, "_write_exclusive", original_write)
    pointer = store.commit(
        expected_cursor_digest=initial.digest,
        completion=completion,
        cursor=advanced,
    )
    assert pointer.cursor_digest == advanced.digest
    assert store.load_current()[:2] == (completion, advanced)
