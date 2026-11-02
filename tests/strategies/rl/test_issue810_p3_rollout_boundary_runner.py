from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.issue810_p3_rollout_boundary_diagnostic import (
    RunCell,
    _cell_meets_acceptance,
    _execute_cells,
    _heldout_result_record,
    _verify_training_join_scope,
    planned_cells,
)


def _passing_cell_result() -> dict[str, object]:
    cues_by_window = {
        1: (-1, 0, 1, -1, 1, 0, -1, 1),
        2: (1, 0, -1, 1, -1, 0, 1, -1),
        3: (-1, 1, 0, -1, 1, 0, -1, 1),
    }
    return {
        "training": {"status": "completed", "joined_transition_count": 4096},
        "heldout": [
            {
                "window_index": index,
                "trained": {
                    "block_results": [
                        {
                            "block_index": block_index,
                            "cue": cue,
                            "actor_decision_index": index * 1025 + block_index * 128,
                            "native_processing_index": index * 1025
                            + block_index * 128
                            + 1,
                            "action_code": 2 if cue == 0 else 3,
                            "post_action_weight": _block_weight(cue, block_index),
                            "aligned": (
                                None
                                if cue == 0
                                else (
                                    _block_weight(cue, block_index) > 0.0
                                    if cue > 0
                                    else _block_weight(cue, block_index) < 0.0
                                )
                            ),
                            "flat_at_snapshot": cue == 0,
                            "pending_order_at_snapshot": False,
                        }
                        for block_index, cue in enumerate(cues_by_window[index])
                    ],
                    "aligned_nonzero_cues": 5,
                    "nonzero_cues": 6,
                    "neutral_block_snapshots": 2,
                    "neutral_flat_snapshots": 2,
                    "neutral_no_pending_snapshots": 2,
                    "neutral_flat": True,
                    "neutral_no_pending_orders": True,
                    "terminal_valid": True,
                    "terminal_log_return": 0.04,
                    "neutral_all_steps_flat_diagnostic": False,
                },
                "initialized": {
                    "terminal_valid": True,
                    "terminal_log_return": 0.01,
                },
                "fixed_long": {
                    "terminal_valid": True,
                    "terminal_log_return": 0.01,
                },
                "cash": {
                    "terminal_valid": True,
                    "terminal_total_cost": 0.0,
                    "terminal_log_return": 0.0,
                },
            }
            for index in (1, 2, 3)
        ],
    }


def test_protocol_is_exactly_six_nonadaptive_delay_seed_cells() -> None:
    cells = planned_cells()

    assert [(cell.payoff_transition, cell.seed) for cell in cells] == [
        (48, 0),
        (48, 7),
        (48, 17),
        (72, 0),
        (72, 7),
        (72, 17),
    ]
    assert len({cell.cell_id for cell in cells}) == 6
    assert all(isinstance(cell, RunCell) for cell in cells)


def test_heldout_result_uses_evaluator_actor_range_key() -> None:
    trained = {"actor_input_range_half_open": [1_025, 2_049]}
    initialized = {"actor_input_range_half_open": [1_025, 2_049]}
    fixed_long = {"actor_input_range_half_open": [1_025, 2_049]}
    cash = {"actor_input_range_half_open": [1_025, 2_049]}

    result = _heldout_result_record(
        window_index=1,
        dataset_id="synthetic-dataset-id",
        trained=trained,
        initialized=initialized,
        fixed_long=fixed_long,
        cash=cash,
    )

    assert result["actor_index_range"] == [1_025, 2_049]
    assert result["trained"] is trained
    assert result["initialized"] is initialized
    assert result["fixed_long"] is fixed_long
    assert result["cash"] is cash


def test_observation_transform_is_fitted_only_on_training_actor_prefix() -> None:
    from tools.issue810_p3_rollout_boundary_fixture import (
        build_synthetic_dataset,
        make_training_env,
    )
    from trade_rl.strategies.rl.portfolio_transform import (
        fit_portfolio_observation_transform,
    )

    dataset = build_synthetic_dataset(48)
    env = make_training_env(dataset, execution_seed=2)
    transform = fit_portfolio_observation_transform(env)

    assert transform.source_dataset_id == dataset.dataset_id
    assert transform.start_index == 0
    assert transform.stop_index == 1025
    assert transform.agent_stop_index == 1024
    assert env._session is None


def test_acceptance_requires_every_heldout_window_and_both_controls() -> None:
    cell = _passing_cell_result()

    assert _cell_meets_acceptance(cell) is True

    cell["heldout"][2]["trained"]["aligned_nonzero_cues"] = 4
    assert _cell_meets_acceptance(cell) is False

    cell = _passing_cell_result()
    cell["heldout"][0]["initialized"]["terminal_log_return"] = 0.021
    assert _cell_meets_acceptance(cell) is False

    cell = _passing_cell_result()
    cell["heldout"][1]["fixed_long"]["terminal_log_return"] = 0.021
    assert _cell_meets_acceptance(cell) is False

    cell = _passing_cell_result()
    cell["heldout"][0]["trained"]["block_results"][1]["pending_order_at_snapshot"] = (
        True
    )
    assert _cell_meets_acceptance(cell) is False


def test_cell_executor_records_one_failure_and_continues_without_retry(
    tmp_path: Path,
) -> None:
    root = tmp_path / "attempt"
    root.mkdir()
    calls: list[str] = []

    def run_cell(cell: RunCell, attempt_root: Path) -> dict[str, object]:
        calls.append(cell.cell_id)
        if cell.cell_id == "delay-48-seed-7":
            raise RuntimeError("synthetic injected fit failure")
        return _passing_cell_result()

    result = _execute_cells(root, run_cell)

    assert calls == [cell.cell_id for cell in planned_cells()]
    assert len(calls) == 6
    assert result["status"] == "completed"
    assert result["disposition"] == "NOT_ESTABLISHED"
    assert [cell["status"] for cell in result["cells"]] == [
        "completed",
        "failed",
        "completed",
        "completed",
        "completed",
        "completed",
    ]
    failure = json.loads((root / "cells" / "delay-48-seed-7" / "cell.json").read_text())
    assert failure["failure"]["error_type"] == "RuntimeError"
    assert failure["failure"]["error_message"] == "synthetic injected fit failure"


def test_join_scope_allows_reserved_settlement_row_but_not_actor_sample_from_it(
    tmp_path: Path,
) -> None:
    root = tmp_path / "training"
    event_root = root / "joins" / "events"
    event_root.mkdir(parents=True)
    native_root = root / "native" / "events"
    native_root.mkdir(parents=True)
    sample_root = root / "rollouts" / "rollout-000001" / "policy-samples" / "samples"
    sample_root.mkdir(parents=True)

    _write_join(
        root,
        event_root,
        native_root,
        sample_root,
        sequence=0,
        current_index=1,
        elapsed_intervals=1,
        actor_index=0,
    )
    # Last training decision starts at 1023; native settlement may advance
    # through reserved row 1025 without making that row an actor observation.
    _write_join(
        root,
        event_root,
        native_root,
        sample_root,
        sequence=1,
        current_index=1025,
        elapsed_intervals=2,
        actor_index=1023,
    )

    scope = _verify_training_join_scope(
        root,
        expected_count=2,
        start_index=0,
        actor_stop_exclusive=1024,
        reserved_stop_index=1025,
    )

    assert scope["actor_input_start_index"] == 0
    assert scope["actor_input_stop_exclusive"] == 1024
    assert scope["maximum_settlement_index"] == 1025
    assert scope["reserved_settlement_row_may_be_read"] is True

    # A sample whose native result maps to actor index 1024 is outside the
    # declared training window, even though its post-action state is row 1025.
    _write_join(
        root,
        event_root,
        native_root,
        sample_root,
        sequence=2,
        current_index=1025,
        elapsed_intervals=1,
        actor_index=1024,
    )
    with pytest.raises(ValueError, match="outside the training actor window"):
        _verify_training_join_scope(
            root,
            expected_count=3,
            start_index=0,
            actor_stop_exclusive=1024,
            reserved_stop_index=1025,
        )


def test_join_scope_rejects_terminal_padding_as_policy_input(tmp_path: Path) -> None:
    root = tmp_path / "training"
    event_root = root / "joins" / "events"
    event_root.mkdir(parents=True)
    native_root = root / "native" / "events"
    native_root.mkdir(parents=True)
    sample_root = root / "rollouts" / "rollout-000001" / "policy-samples" / "samples"
    sample_root.mkdir(parents=True)
    _write_join(
        root,
        event_root,
        native_root,
        sample_root,
        sequence=0,
        current_index=1,
        elapsed_intervals=1,
        actor_index=0,
        terminal_values=[1],
    )

    with pytest.raises(ValueError, match="terminal observation padding"):
        _verify_training_join_scope(
            root,
            expected_count=1,
            start_index=0,
            actor_stop_exclusive=1024,
            reserved_stop_index=1025,
        )


def test_join_scope_requires_batch_one_and_matching_terminal_flags(
    tmp_path: Path,
) -> None:
    root = tmp_path / "training"
    event_root = root / "joins" / "events"
    event_root.mkdir(parents=True)
    native_root = root / "native" / "events"
    native_root.mkdir(parents=True)
    sample_root = root / "rollouts" / "rollout-000001" / "policy-samples" / "samples"
    sample_root.mkdir(parents=True)
    _write_join(
        root,
        event_root,
        native_root,
        sample_root,
        sequence=0,
        current_index=1,
        elapsed_intervals=1,
        actor_index=0,
        forward_batch_size=2,
    )

    with pytest.raises(ValueError, match="not joined to its submitted native"):
        _verify_training_join_scope(
            root,
            expected_count=1,
            start_index=0,
            actor_stop_exclusive=1024,
            reserved_stop_index=1025,
        )

    sample = sample_root / "sample-000000000001.json"
    sample.unlink()
    native = native_root / "event-00000000.json"
    native.unlink()
    (event_root / "transition-000000000000.json").unlink()
    _write_join(
        root,
        event_root,
        native_root,
        sample_root,
        sequence=0,
        current_index=1,
        elapsed_intervals=1,
        actor_index=0,
        join_terminated=True,
    )

    with pytest.raises(ValueError, match="not joined to its submitted native"):
        _verify_training_join_scope(
            root,
            expected_count=1,
            start_index=0,
            actor_stop_exclusive=1024,
            reserved_stop_index=1025,
        )


def _write_join(
    root: Path,
    event_root: Path,
    native_root: Path,
    sample_root: Path,
    *,
    sequence: int,
    current_index: int,
    elapsed_intervals: int,
    actor_index: int,
    terminal_values: list[int] | None = None,
    forward_batch_size: int = 1,
    join_terminated: bool | None = None,
) -> None:
    import hashlib

    sample = {
        "schema": "portfolio_policy_sample_v2",
        "sample_id": sequence + 1,
        "batch_index": 0,
        "forward_batch_size": forward_batch_size,
        "raw_action_codes": [0],
        "inputs": {
            "clock": {"values": [actor_index / 1024.0, 0.0, 0.0]},
            "terminal": {"values": terminal_values or [0]},
        },
    }
    sample_path = sample_root / f"sample-{sequence + 1:012d}.json"
    sample_raw = json.dumps(sample, sort_keys=True, separators=(",", ":")).encode()
    sample_path.write_bytes(sample_raw)
    native = {
        "schema": "portfolio_episode_journal_v1",
        "sequence": sequence,
        "event": "step_result",
        "data": {
            "current_index": current_index,
            "elapsed_intervals": elapsed_intervals,
            "raw_action_codes": [0],
            "terminated": sequence == 1,
            "truncated": False,
        },
    }
    native_path = native_root / f"event-{sequence:08d}.json"
    native_raw = json.dumps(native, sort_keys=True, separators=(",", ":")).encode()
    native_path.write_bytes(native_raw)
    join = {
        "schema": "joint_portfolio_actor_native_join_v1",
        "sequence": sequence,
        "raw_action_codes": [0],
        "actor_sample": {
            "sample_id": sequence + 1,
            "path": sample_path.relative_to(root).as_posix(),
            "sha256": hashlib.sha256(sample_raw).hexdigest(),
        },
        "native_event": {
            "sequence": sequence,
            "path": native_path.relative_to(root).as_posix(),
            "sha256": hashlib.sha256(native_raw).hexdigest(),
        },
        "terminated": (sequence == 1 if join_terminated is None else join_terminated),
        "truncated": False,
    }
    join_path = event_root / f"transition-{sequence:012d}.json"
    join_path.write_text(json.dumps(join, sort_keys=True, separators=(",", ":")))


def _block_weight(cue: int, block_index: int) -> float:
    if cue == 0:
        return 0.0
    sign = 1.0 if cue > 0 else -1.0
    return -sign * 0.5 if block_index == 7 else sign * 0.5
