"""Run the fixed synthetic PPO rollout-boundary diagnostic for Issue 810.

This runner accepts only an output directory and expected source revision. It
constructs its synthetic Dataset in memory and never reads market-data paths.
Its disposition is a synthetic mechanism result, not economic evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

from trade_rl.artifacts.atomic_write import atomic_write_bytes
from trade_rl.artifacts.canonical import canonical_json_bytes

PROTOCOL_SCHEMA = "issue810_p3_rollout_boundary_diagnostic_v1"
SEEDS = (0, 7, 17)
PAYOFF_TRANSITIONS = (48, 72)
HELDOUT_WINDOWS = (1, 2, 3)
HELDOUT_CUES = {
    1: (-1, 0, 1, -1, 1, 0, -1, 1),
    2: (1, 0, -1, 1, -1, 0, 1, -1),
    3: (-1, 1, 0, -1, 1, 0, -1, 1),
}
TRAINING_TIMESTEPS = 4_096
ROLLOUT_STEPS = 64
BATCH_SIZE = 32
EPOCHS = 10
GAE_LAMBDA = 0.95
MIN_ALIGNED_NONZERO_CUES = 5
EXPECTED_NONZERO_CUES_PER_WINDOW = 6
MIN_TRAINED_LOG_RETURN = 0.03
MIN_CONTROL_UPLIFT = 0.02

PROTOCOL: dict[str, object] = {
    "schema": PROTOCOL_SCHEMA,
    "cells": [
        {"payoff_transition": delay, "seed": seed}
        for delay in PAYOFF_TRANSITIONS
        for seed in SEEDS
    ],
    "training": {
        "total_timesteps": TRAINING_TIMESTEPS,
        "n_steps": ROLLOUT_STEPS,
        "batch_size": BATCH_SIZE,
        "n_epochs": EPOCHS,
        "gamma": 1.0,
        "gae_lambda": GAE_LAMBDA,
        "device": "cpu",
        "execution_seed_offset": 2,
        "reset_seed_offset": 1,
        "ppo_seed_offset": 0,
    },
    "evaluation": {
        "heldout_window_indices": list(HELDOUT_WINDOWS),
        "heldout_cue_schedules": {
            str(index): list(cues) for index, cues in HELDOUT_CUES.items()
        },
        "deterministic_policy_actions": True,
        "fixed_long_action_each_actor_step": True,
        "cash_action_from_flat": "HOLD",
        "actor_observation_indices": "training window only",
        "transform_fit_range": "training actor prefix only, half-open",
        "optimizer_training_range": "training actor observations only",
        "reserved_settlement_row_may_be_read_for_training_close": True,
        "heldout_evaluation_occurs_after_fit_only": True,
    },
    "acceptance": {
        "min_aligned_nonzero_cues_per_heldout_window": MIN_ALIGNED_NONZERO_CUES,
        "expected_nonzero_cues_per_heldout_window": EXPECTED_NONZERO_CUES_PER_WINDOW,
        "min_trained_terminal_log_return": MIN_TRAINED_LOG_RETURN,
        "min_uplift_vs_initialized_and_fixed_long": MIN_CONTROL_UPLIFT,
        "neutral_blocks_flat_without_pending_orders": True,
        "terminal_flat_without_pending_orders": True,
        "all_cells_and_heldout_windows_must_pass": True,
    },
    "interpretation": (
        "synthetic PPO rollout-boundary learnability diagnostic only; "
        "not real-data, profitability, winner, or live-readiness evidence"
    ),
}
PROTOCOL_SHA256 = hashlib.sha256(canonical_json_bytes(PROTOCOL)).hexdigest()


@dataclass(frozen=True, slots=True)
class RunCell:
    payoff_transition: int
    seed: int

    @property
    def cell_id(self) -> str:
        return f"delay-{self.payoff_transition}-seed-{self.seed}"

    def to_payload(self) -> dict[str, int | str]:
        return {
            "cell_id": self.cell_id,
            "payoff_transition": self.payoff_transition,
            "seed": self.seed,
        }


def planned_cells() -> tuple[RunCell, ...]:
    """Return the six fixed cells in protocol order; there is no retry path."""

    return tuple(
        RunCell(payoff_transition=delay, seed=seed)
        for delay in PAYOFF_TRANSITIONS
        for seed in SEEDS
    )


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _write_json(path: Path, value: object, *, unique: bool = False) -> str:
    if unique and (path.exists() or path.is_symlink()):
        raise FileExistsError(f"diagnostic artifact already exists: {path}")
    payload = canonical_json_bytes(value)
    atomic_write_bytes(path, payload)
    return hashlib.sha256(payload).hexdigest()


def _strict_json(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    value = json.loads(raw)
    if type(value) is not dict:
        raise ValueError(f"expected a JSON object in {path.name}")
    if canonical_json_bytes(value) != raw:
        raise ValueError(f"noncanonical JSON in {path.name}")
    return value


def _cell_meets_acceptance(result: Mapping[str, object]) -> bool:
    """Fail closed unless every frozen per-window criterion is present and true."""

    training = result.get("training")
    heldout = result.get("heldout")
    if not isinstance(training, Mapping) or training.get("status") != "completed":
        return False
    if training.get("joined_transition_count") != TRAINING_TIMESTEPS:
        return False
    if type(heldout) is not list or len(heldout) != len(HELDOUT_WINDOWS):
        return False
    by_window: dict[int, Mapping[str, object]] = {}
    for item in heldout:
        if not isinstance(item, Mapping):
            return False
        window_index = item.get("window_index")
        if type(window_index) is not int or window_index in by_window:
            return False
        by_window[window_index] = item
    if set(by_window) != set(HELDOUT_WINDOWS):
        return False

    for window_index in HELDOUT_WINDOWS:
        item = by_window[window_index]
        trained = item.get("trained")
        initialized = item.get("initialized")
        fixed_long = item.get("fixed_long")
        cash = item.get("cash")
        if not all(
            isinstance(value, Mapping)
            for value in (trained, initialized, fixed_long, cash)
        ):
            return False
        assert isinstance(trained, Mapping)
        assert isinstance(initialized, Mapping)
        assert isinstance(fixed_long, Mapping)
        assert isinstance(cash, Mapping)
        blocks = trained.get("block_results")
        if type(blocks) is not list or len(blocks) != 8:
            return False
        nonzero_blocks: list[Mapping[str, object]] = []
        neutral_blocks: list[Mapping[str, object]] = []
        for block_index, block in enumerate(blocks):
            if not isinstance(block, Mapping):
                return False
            if (
                block.get("block_index") != block_index
                or type(block.get("cue")) is not int
                or type(block.get("actor_decision_index")) is not int
                or type(block.get("native_processing_index")) is not int
                or type(block.get("action_code")) is not int
                or _finite_number(block.get("post_action_weight")) is None
                or type(block.get("flat_at_snapshot")) is not bool
                or type(block.get("pending_order_at_snapshot")) is not bool
            ):
                return False
            if block["cue"] == 0:
                neutral_blocks.append(block)
                if block.get("aligned") is not None:
                    return False
            else:
                nonzero_blocks.append(block)
                if type(block.get("aligned")) is not bool:
                    return False
        if (
            len(nonzero_blocks) != EXPECTED_NONZERO_CUES_PER_WINDOW
            or len(neutral_blocks) != 2
        ):
            return False
        window_start = window_index * 1_025
        expected_cues = HELDOUT_CUES[window_index]
        for block_index, block in enumerate(blocks):
            expected_actor_index = window_start + block_index * 128
            if (
                block.get("cue") != expected_cues[block_index]
                or block.get("actor_decision_index") != expected_actor_index
                or block.get("native_processing_index") != expected_actor_index + 1
            ):
                return False
            weight = _finite_number(block.get("post_action_weight"))
            cue = expected_cues[block_index]
            expected_aligned = (
                None
                if cue == 0 or weight is None
                else (weight > 0.0 if cue > 0 else weight < 0.0)
            )
            expected_flat = weight is not None and abs(weight) <= 1e-12
            if (
                block.get("aligned") is not expected_aligned
                or block.get("flat_at_snapshot") is not expected_flat
            ):
                return False
        if sum(block.get("aligned") is True for block in nonzero_blocks) != trained.get(
            "aligned_nonzero_cues"
        ):
            return False
        if sum(
            block.get("flat_at_snapshot") is True for block in neutral_blocks
        ) != trained.get("neutral_flat_snapshots"):
            return False
        if sum(
            block.get("pending_order_at_snapshot") is False for block in neutral_blocks
        ) != trained.get("neutral_no_pending_snapshots"):
            return False
        if trained.get("neutral_block_snapshots") != len(neutral_blocks):
            return False
        if (
            trained.get("nonzero_cues") != EXPECTED_NONZERO_CUES_PER_WINDOW
            or type(trained.get("aligned_nonzero_cues")) is not int
            or trained["aligned_nonzero_cues"] < MIN_ALIGNED_NONZERO_CUES
            or trained.get("neutral_flat") is not True
            or trained.get("neutral_no_pending_orders") is not True
            or trained.get("terminal_valid") is not True
        ):
            return False
        trained_return = _finite_number(trained.get("terminal_log_return"))
        initialized_return = _finite_number(initialized.get("terminal_log_return"))
        fixed_long_return = _finite_number(fixed_long.get("terminal_log_return"))
        cash_return = _finite_number(cash.get("terminal_log_return"))
        if (
            trained_return is None
            or initialized_return is None
            or fixed_long_return is None
            or cash_return is None
            or initialized.get("terminal_valid") is not True
            or fixed_long.get("terminal_valid") is not True
            or cash.get("terminal_valid") is not True
            or _finite_number(cash.get("terminal_total_cost")) not in (0.0,)
            or abs(cash_return) > 1e-12
            or trained_return <= MIN_TRAINED_LOG_RETURN
            or trained_return - initialized_return <= MIN_CONTROL_UPLIFT
            or trained_return - fixed_long_return <= MIN_CONTROL_UPLIFT
        ):
            return False
    return True


def _finite_number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _execute_cells(
    output_root: Path,
    cell_runner: Callable[[RunCell, Path], Mapping[str, object]],
) -> dict[str, object]:
    """Run each preregistered cell once, retain failures, and continue the batch."""

    cells_root = output_root / "cells"
    cells_root.mkdir(parents=False, exist_ok=False)
    run_record: dict[str, object] = {
        "schema": "issue810_p3_rollout_boundary_run_v1",
        "protocol_sha256": PROTOCOL_SHA256,
        "status": "running",
        "disposition": "NOT_ESTABLISHED",
        "started_at": _utc_now(),
        "completed_at": None,
        "planned_cell_count": len(planned_cells()),
        "completed_cell_count": 0,
        "cells": [],
    }
    _write_json(output_root / "run.json", run_record, unique=True)
    records: list[dict[str, object]] = []

    for cell in planned_cells():
        attempt_root = cells_root / cell.cell_id
        attempt_root.mkdir(parents=False, exist_ok=False)
        record: dict[str, object]
        try:
            payload = dict(cell_runner(cell, attempt_root))
            canonical_json_bytes(payload)
            passed = _cell_meets_acceptance(payload)
            record = {
                **cell.to_payload(),
                "status": "completed",
                "acceptance_passed": passed,
                "result": payload,
            }
        except Exception as error:
            record = {
                **cell.to_payload(),
                "status": "failed",
                "acceptance_passed": False,
                "failure": {
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                },
            }
        _write_json(attempt_root / "cell.json", record, unique=True)
        records.append(record)
        run_record["completed_cell_count"] = len(records)
        run_record["cells"] = records.copy()
        _write_json(output_root / "run.json", run_record)

    all_passed = len(records) == len(planned_cells()) and all(
        record.get("status") == "completed" and record.get("acceptance_passed") is True
        for record in records
    )
    run_record.update(
        {
            "status": "completed",
            "disposition": (
                "SYNTHETIC_LEARNABILITY_SUPPORTED" if all_passed else "NOT_ESTABLISHED"
            ),
            "completed_at": _utc_now(),
            "completed_cell_count": len(records),
            "cells": records,
        }
    )
    _write_json(output_root / "run.json", run_record)
    return run_record


def _resolve_record_path(training_root: Path, value: object) -> Path:
    if type(value) is not str or not value:
        raise ValueError("training journal path is missing")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("training journal path escapes the attempt root")
    resolved = (training_root / relative).resolve(strict=True)
    if not resolved.is_relative_to(training_root.resolve(strict=True)):
        raise ValueError("training journal path escapes the attempt root")
    return resolved


def _read_join_records(training_root: Path) -> list[dict[str, Any]]:
    event_root = training_root / "joins" / "events"
    paths = sorted(event_root.glob("transition-*.json"))
    return [_strict_json(path) for path in paths]


def _verify_training_join_scope(
    training_root: Path,
    *,
    expected_count: int,
    start_index: int,
    actor_stop_exclusive: int,
    reserved_stop_index: int,
) -> dict[str, object]:
    """Verify every captured PPO action sample joins a train-window native step.

    The final training action may process the actor interval and one reserved
    settlement interval. Its decision index is recovered as current_index minus
    elapsed_intervals, so the reserved row is not misreported as an optimizer
    observation.
    """

    joins = _read_join_records(training_root)
    if len(joins) != expected_count:
        raise ValueError("joined PPO transition count differs from fixed budget")
    native_sequence = -1
    decision_indices: list[int] = []
    maximum_settlement_index = start_index
    for sequence, join in enumerate(joins):
        if (
            join.get("schema") != "joint_portfolio_actor_native_join_v1"
            or join.get("sequence") != sequence
        ):
            raise ValueError("actor/native join sequence is incomplete or invalid")
        actor_reference = join.get("actor_sample")
        native_reference = join.get("native_event")
        if not isinstance(actor_reference, Mapping) or not isinstance(
            native_reference, Mapping
        ):
            raise ValueError("actor/native join references are malformed")
        sample_path = _resolve_record_path(training_root, actor_reference.get("path"))
        native_path = _resolve_record_path(training_root, native_reference.get("path"))
        sample_bytes = sample_path.read_bytes()
        native_bytes = native_path.read_bytes()
        if hashlib.sha256(sample_bytes).hexdigest() != actor_reference.get(
            "sha256"
        ) or hashlib.sha256(native_bytes).hexdigest() != native_reference.get("sha256"):
            raise ValueError("actor/native join artifact digest mismatch")
        sample = _strict_json(sample_path)
        native = _strict_json(native_path)
        data = native.get("data")
        if (
            sample.get("schema") != "portfolio_policy_sample_v2"
            or sample.get("sample_id") != actor_reference.get("sample_id")
            or type(sample.get("batch_index")) is not int
            or sample.get("batch_index") != 0
            or type(sample.get("forward_batch_size")) is not int
            or sample.get("forward_batch_size") != 1
            or sample.get("raw_action_codes") != join.get("raw_action_codes")
            or native.get("event") != "step_result"
            or not isinstance(data, Mapping)
            or data.get("raw_action_codes") != join.get("raw_action_codes")
            or type(join.get("terminated")) is not bool
            or type(join.get("truncated")) is not bool
            or data.get("terminated") is not join.get("terminated")
            or data.get("truncated") is not join.get("truncated")
        ):
            raise ValueError(
                "policy sample is not joined to its submitted native action"
            )
        event_sequence = native.get("sequence")
        if type(event_sequence) is not int or event_sequence <= native_sequence:
            raise ValueError("native step references are stale or out of order")
        native_sequence = event_sequence
        current_index = data.get("current_index")
        elapsed = data.get("elapsed_intervals")
        if type(current_index) is not int or type(elapsed) is not int or elapsed < 1:
            raise ValueError("native step has invalid clock facts")
        decision_index = current_index - elapsed
        if not start_index <= decision_index < actor_stop_exclusive:
            raise ValueError("policy actor sample is outside the training actor window")
        if current_index > reserved_stop_index:
            raise ValueError("native training step advanced beyond reserved settlement")
        inputs = sample.get("inputs")
        if not isinstance(inputs, Mapping):
            raise ValueError("policy sample observation is missing")
        terminal = inputs.get("terminal")
        if (
            not isinstance(terminal, Mapping)
            or type(terminal.get("values")) is not list
            or terminal.get("values") != [0]
        ):
            raise ValueError("terminal observation padding entered an actor sample")
        clock = inputs.get("clock")
        if not isinstance(clock, Mapping) or type(clock.get("values")) is not list:
            raise ValueError("policy sample clock observation is missing")
        clock_values = clock["values"]
        if len(clock_values) != 3 or any(
            type(value) not in (int, float) or not math.isfinite(float(value))
            for value in clock_values
        ):
            raise ValueError("policy sample clock observation is malformed")
        expected_elapsed = (decision_index - start_index) / (
            actor_stop_exclusive - start_index
        )
        if not math.isclose(float(clock_values[0]), expected_elapsed, abs_tol=1e-7):
            raise ValueError(
                "policy observation clock does not match native decision index"
            )
        decision_indices.append(decision_index)
        maximum_settlement_index = max(maximum_settlement_index, current_index)

    return {
        "joined_transition_count": len(joins),
        "actor_input_start_index": start_index,
        "actor_input_stop_exclusive": actor_stop_exclusive,
        "actor_input_index_min": min(decision_indices),
        "actor_input_index_max": max(decision_indices),
        "maximum_settlement_index": maximum_settlement_index,
        "reserved_settlement_index": reserved_stop_index,
        "reserved_settlement_row_may_be_read": maximum_settlement_index
        == reserved_stop_index,
        "oos_observations_used_during_fit": False,
    }


def _load_checkpoint(path: Path) -> Any:
    try:
        sb3 = importlib.import_module("stable_baselines3")
        ppo = getattr(sb3, "PPO")
    except (ImportError, AttributeError) as error:
        raise RuntimeError("train-sb3 is required for the P3 diagnostic") from error
    return ppo.load(str(path), device="cpu")


def _heldout_result_record(
    *,
    window_index: int,
    dataset_id: str,
    trained: Mapping[str, object],
    initialized: Mapping[str, object],
    fixed_long: Mapping[str, object],
    cash: Mapping[str, object],
) -> dict[str, object]:
    actor_range = trained.get("actor_input_range_half_open")
    if (
        type(actor_range) is not list
        or len(actor_range) != 2
        or any(type(index) is not int for index in actor_range)
        or actor_range[0] >= actor_range[1]
    ):
        raise ValueError("trained held-out result omitted its actor input range")
    return {
        "window_index": window_index,
        "dataset_id": dataset_id,
        "actor_index_range": list(actor_range),
        "trained": trained,
        "initialized": initialized,
        "fixed_long": fixed_long,
        "cash": cash,
    }


def _evaluate_policy(
    *,
    policy_name: str,
    policy: Any | None,
    control_action: int | None,
    dataset: Any,
    window_index: int,
    transform: Any,
    training_env: Any,
    seed: int,
) -> dict[str, object]:
    from tools.issue810_p3_rollout_boundary_fixture import (
        BLOCK_TRANSITIONS,
        BLOCKS_PER_WINDOW,
        INITIAL_CAPITAL,
        WINDOW_STARTS,
        WINDOW_STOPS,
        make_oos_env,
    )
    from trade_rl.strategies.portfolio_policy_codec import ACTION_HOLD, ACTION_LONG

    if policy_name == "fixed_long" and control_action != ACTION_LONG:
        raise ValueError("fixed-long control must use literal ACTION_LONG")
    if policy_name == "cash" and control_action != ACTION_HOLD:
        raise ValueError("cash control must use literal ACTION_HOLD")
    env = make_oos_env(dataset, window_index, transform, training_env)
    start_index = WINDOW_STARTS[window_index]
    stop_index = WINDOW_STOPS[window_index]
    observation, reset_info = env.reset(seed=seed)
    if reset_info.get("current_index") != start_index:
        raise ValueError("held-out environment reset at an unexpected row")

    cues = tuple(
        int(dataset.features[start_index + block * BLOCK_TRANSITIONS, 0, 0])
        for block in range(BLOCKS_PER_WINDOW)
    )
    aligned = 0
    nonzero = 0
    neutral_block_snapshots = 0
    neutral_flat_snapshots = 0
    neutral_no_pending_snapshots = 0
    neutral_steps = 0
    neutral_flat_steps = 0
    neutral_no_pending_steps = 0
    actor_steps = 0
    terminal_info: Mapping[str, object] | None = None
    last_actor_index: int | None = None
    block_results: list[dict[str, object]] = []
    try:
        for actor_step in range(1_024):
            block_index = actor_step // BLOCK_TRANSITIONS
            cue = cues[block_index]
            if policy is None:
                action = np.asarray([control_action], dtype=np.int64)
            else:
                predicted, _ = policy.predict(observation, deterministic=True)
                action = np.asarray(predicted)
                if action.ndim == 0:
                    action = action.reshape(1)
                if action.shape != (1,) or action.dtype.kind not in "iu":
                    raise ValueError("PPO emitted an invalid synthetic action vector")
                action = action.astype(np.int64, copy=False)
            observation, _, terminated, truncated, info = env.step(action)
            actor_steps += 1
            elapsed = info.get("elapsed_intervals")
            current_index = info.get("current_index")
            if type(elapsed) is not int or type(current_index) is not int:
                raise ValueError("held-out native step omitted clock facts")
            last_actor_index = current_index - elapsed
            if actor_step % BLOCK_TRANSITIONS == 0:
                weight = float(observation["account"][0, 1])
                has_pending_order = bool(
                    np.any(observation["active_orders"][:, 0] > 0.0)
                )
                flat_snapshot = abs(weight) <= 1e-12
                if cue != 0:
                    nonzero += 1
                    block_aligned = (cue > 0 and weight > 0.0) or (
                        cue < 0 and weight < 0.0
                    )
                    aligned += int(block_aligned)
                else:
                    block_aligned = None
                    neutral_block_snapshots += 1
                    neutral_flat_snapshots += int(flat_snapshot)
                    neutral_no_pending_snapshots += int(not has_pending_order)
                block_results.append(
                    {
                        "block_index": block_index,
                        "cue": cue,
                        "actor_decision_index": start_index + actor_step,
                        "native_processing_index": current_index,
                        "elapsed_intervals": elapsed,
                        "action_code": int(action[0]),
                        "post_action_weight": weight,
                        "aligned": block_aligned,
                        "flat_at_snapshot": flat_snapshot,
                        "pending_order_at_snapshot": has_pending_order,
                    }
                )
            if cue == 0:
                neutral_steps += 1
                weight = float(observation["account"][0, 1])
                no_position = abs(weight) <= 1e-12
                no_pending = not bool(np.any(observation["active_orders"][:, 0] > 0.0))
                neutral_flat_steps += int(no_position)
                neutral_no_pending_steps += int(no_pending)
            if truncated:
                raise ValueError("synthetic evaluation unexpectedly truncated")
            if terminated:
                if actor_step != 1_023:
                    raise ValueError("held-out episode ended before all actor steps")
                terminal_info = info
                break
        if actor_steps != 1_024 or terminal_info is None:
            raise ValueError(
                "held-out episode did not complete its fixed actor horizon"
            )
        terminal_state = terminal_info.get("terminal_state")
        if not isinstance(terminal_state, Mapping):
            raise ValueError("native terminal state is missing")
        final_equity = _finite_number(terminal_state.get("portfolio_value"))
        initial_equity = float(INITIAL_CAPITAL)
        terminal_index = terminal_state.get("current_index")
        active_orders = terminal_state.get("active_orders")
        exact_quantities = terminal_state.get("exact_quantities")
        terminal_valid = (
            terminal_info.get("settlement_complete") is True
            and terminal_info.get("objective_endpoint_valid") is True
            and terminal_index == stop_index
            and exact_quantities == ("0",)
            and active_orders == ()
            and final_equity is not None
            and final_equity > 0.0
        )
        log_return = (
            math.log(final_equity / initial_equity)
            if final_equity is not None and final_equity > 0.0
            else None
        )
        neutral_flat = (
            neutral_block_snapshots > 0
            and neutral_flat_snapshots == neutral_block_snapshots
        )
        neutral_no_pending = (
            neutral_block_snapshots > 0
            and neutral_no_pending_snapshots == neutral_block_snapshots
        )
        return {
            "policy": policy_name,
            "window_index": window_index,
            "dataset_id": dataset.dataset_id,
            "actor_input_range_half_open": [start_index, stop_index - 1],
            "actor_steps": actor_steps,
            "last_actor_decision_index": last_actor_index,
            "reserved_settlement_index": stop_index,
            "cue_schedule": cues,
            "block_results": block_results,
            "aligned_nonzero_cues": aligned,
            "nonzero_cues": nonzero,
            "neutral_block_snapshots": neutral_block_snapshots,
            "neutral_flat_snapshots": neutral_flat_snapshots,
            "neutral_no_pending_snapshots": neutral_no_pending_snapshots,
            "neutral_flat_steps": neutral_flat_steps,
            "neutral_steps": neutral_steps,
            "neutral_no_pending_steps": neutral_no_pending_steps,
            "neutral_all_steps_flat_diagnostic": (
                neutral_steps > 0 and neutral_flat_steps == neutral_steps
            ),
            "neutral_all_steps_no_pending_diagnostic": (
                neutral_steps > 0 and neutral_no_pending_steps == neutral_steps
            ),
            "neutral_flat": neutral_flat,
            "neutral_no_pending_orders": neutral_no_pending,
            "terminal_valid": terminal_valid,
            "terminal_equity": final_equity,
            "initial_equity": initial_equity,
            "terminal_log_return": log_return,
            "terminal_max_drawdown": terminal_state.get("max_drawdown"),
            "terminal_total_cost": terminal_state.get("total_cost"),
            "terminal_active_orders": len(active_orders)
            if isinstance(active_orders, (tuple, list))
            else None,
            "terminal_exact_quantities": list(exact_quantities)
            if isinstance(exact_quantities, (tuple, list))
            else None,
            "terminal_elapsed_intervals": terminal_info.get("elapsed_intervals"),
        }
    finally:
        env.close()


def _run_real_cell(cell: RunCell, attempt_root: Path) -> Mapping[str, object]:
    from tools.issue810_p3_rollout_boundary_fixture import (
        PROTOCOL_SCHEMA as FIXTURE_SCHEMA,
    )
    from tools.issue810_p3_rollout_boundary_fixture import (
        WINDOW_STARTS,
        WINDOW_STOPS,
        build_synthetic_dataset,
        make_training_env,
    )
    from trade_rl.strategies.portfolio_policy_codec import ACTION_HOLD, ACTION_LONG
    from trade_rl.strategies.rl.portfolio_training import (
        SharedPortfolioPPOConfig,
        fit_shared_portfolio_ppo,
    )
    from trade_rl.strategies.rl.portfolio_transform import (
        fit_portfolio_observation_transform,
    )

    dataset = build_synthetic_dataset(cell.payoff_transition)
    training_env = make_training_env(dataset, execution_seed=cell.seed + 2)
    transform = fit_portfolio_observation_transform(training_env)
    actor_stop = training_env.stop_index - 1
    if (
        transform.source_dataset_id != dataset.dataset_id
        or transform.start_index != training_env.start_index
        or transform.stop_index != training_env.stop_index
        or transform.agent_stop_index != actor_stop
        or actor_stop - training_env.start_index != 1_024
    ):
        raise ValueError("training transform does not match the frozen actor prefix")
    training_root = attempt_root / "training"
    config = SharedPortfolioPPOConfig(
        total_timesteps=TRAINING_TIMESTEPS,
        n_steps=ROLLOUT_STEPS,
        batch_size=BATCH_SIZE,
        n_epochs=EPOCHS,
        reset_seed=cell.seed + 1,
        ppo_seed=cell.seed,
        gae_lambda=GAE_LAMBDA,
    )
    training_result = fit_shared_portfolio_ppo(
        training_env,
        transform=transform,
        output_root=training_root,
        config=config,
    )
    completion = _strict_json(training_root / "completion.json")
    context = _strict_json(training_root / "training_context.json")
    if (
        completion.get("status") != "completed"
        or completion.get("requested_timesteps") != TRAINING_TIMESTEPS
        or completion.get("realized_timesteps") != TRAINING_TIMESTEPS
        or completion.get("rollout_count") != TRAINING_TIMESTEPS // ROLLOUT_STEPS
        or completion.get("joined_transition_count") != TRAINING_TIMESTEPS
        or context.get("context", {}).get("dataset_id") != dataset.dataset_id
        or context.get("context", {}).get("start_index") != training_env.start_index
        or context.get("context", {}).get("stop_index") != training_env.stop_index
    ):
        raise ValueError(
            "native PPO completion does not match the fixed training scope"
        )
    joined_scope = _verify_training_join_scope(
        training_root,
        expected_count=TRAINING_TIMESTEPS,
        start_index=training_env.start_index,
        actor_stop_exclusive=actor_stop,
        reserved_stop_index=training_env.stop_index,
    )
    initial_checkpoint = training_root / "checkpoints" / "initial.zip"
    final_relative = completion.get("final_checkpoint_path")
    if type(final_relative) is not str:
        raise ValueError("completed PPO run omitted its final checkpoint path")
    final_checkpoint = (training_root / final_relative).resolve(strict=True)
    if not final_checkpoint.is_relative_to(training_root.resolve(strict=True)):
        raise ValueError("final PPO checkpoint escaped the training attempt root")
    final_digest = hashlib.sha256(final_checkpoint.read_bytes()).hexdigest()
    if final_digest != completion.get("final_checkpoint_sha256"):
        raise ValueError("final PPO checkpoint digest differs from completion receipt")
    initial_model = _load_checkpoint(initial_checkpoint)
    trained_model = _load_checkpoint(final_checkpoint)

    heldout_results: list[dict[str, object]] = []
    for window_index in HELDOUT_WINDOWS:
        initial = _evaluate_policy(
            policy_name="initialized",
            policy=initial_model,
            control_action=None,
            dataset=dataset,
            window_index=window_index,
            transform=transform,
            training_env=training_env,
            seed=cell.seed + 1,
        )
        trained = _evaluate_policy(
            policy_name="trained",
            policy=trained_model,
            control_action=None,
            dataset=dataset,
            window_index=window_index,
            transform=transform,
            training_env=training_env,
            seed=cell.seed + 1,
        )
        fixed_long = _evaluate_policy(
            policy_name="fixed_long",
            policy=None,
            control_action=ACTION_LONG,
            dataset=dataset,
            window_index=window_index,
            transform=transform,
            training_env=training_env,
            seed=cell.seed + 1,
        )
        cash = _evaluate_policy(
            policy_name="cash",
            policy=None,
            control_action=ACTION_HOLD,
            dataset=dataset,
            window_index=window_index,
            transform=transform,
            training_env=training_env,
            seed=cell.seed + 1,
        )
        heldout_results.append(
            _heldout_result_record(
                window_index=window_index,
                dataset_id=dataset.dataset_id,
                trained=trained,
                initialized=initial,
                fixed_long=fixed_long,
                cash=cash,
            )
        )

    return {
        "fixture_schema": FIXTURE_SCHEMA,
        "payoff_transition": cell.payoff_transition,
        "seed_mapping": {
            "ppo_seed": cell.seed,
            "reset_seed": cell.seed + 1,
            "execution_seed": cell.seed + 2,
        },
        "dataset": {
            "dataset_id": dataset.dataset_id,
            "generated_in_memory": True,
            "real_market_data_read": False,
            "shared_dataset_id_for_train_and_heldout": True,
            "train_fit_observation_range": [
                transform.start_index,
                transform.agent_stop_index,
            ],
            "train_actor_input_range_half_open": [
                training_env.start_index,
                actor_stop,
            ],
            "train_actor_input_stop_exclusive": actor_stop,
            "train_reserved_settlement_index": training_env.stop_index,
            "reserved_row_is_first_heldout_row": True,
            "heldout_ranges": [
                {
                    "window_index": index,
                    "start_index": WINDOW_STARTS[index],
                    "stop_index": WINDOW_STOPS[index],
                }
                for index in HELDOUT_WINDOWS
            ],
            "optimizer_input_scope": joined_scope,
            "heldout_observations_used_during_fit": False,
        },
        "training": {
            "status": "completed",
            "output_root": training_root.name,
            "requested_timesteps": training_result.requested_timesteps,
            "realized_timesteps": training_result.realized_timesteps,
            "rollout_count": training_result.rollout_count,
            "joined_transition_count": completion["joined_transition_count"],
            "optimizer_step_calls": training_result.optimizer_step_calls,
            "epoch_count": training_result.epoch_count,
            "initial_checkpoint_sha256": hashlib.sha256(
                initial_checkpoint.read_bytes()
            ).hexdigest(),
            "final_checkpoint_sha256": final_digest,
            "completion_sha256": training_result.completion_sha256,
        },
        "heldout": heldout_results,
        "interpretation": PROTOCOL["interpretation"],
    }


def _module_source_paths(repo_root: Path) -> dict[str, str]:
    names = (
        "trade_rl",
        "trade_rl.strategies.rl.portfolio_env",
        "trade_rl.strategies.rl.portfolio_transform",
        "trade_rl.strategies.rl.portfolio_training",
        "trade_rl.strategies.rl.portfolio_episode_journal",
        "trade_rl.strategies.rl.portfolio_policy_sample_journal",
        "tools.issue810_p3_rollout_boundary_fixture",
        "tools.issue810_p3_rollout_boundary_diagnostic",
    )
    paths: dict[str, str] = {}
    for name in names:
        module = importlib.import_module(name)
        module_file = getattr(module, "__file__", None)
        if type(module_file) is not str:
            raise RuntimeError(f"imported module has no source path: {name}")
        resolved = Path(module_file).resolve(strict=True)
        if not resolved.is_relative_to(repo_root):
            raise RuntimeError(f"module imported outside exact worktree: {name}")
        paths[name] = resolved.relative_to(repo_root).as_posix()
    return paths


def _source_preflight(expected_head: str) -> dict[str, object]:
    repo_root = Path(__file__).resolve().parents[1]
    if type(expected_head) is not str or len(expected_head) != 40:
        raise ValueError("--expected-head must be a full 40-character commit SHA")
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if head != expected_head.lower():
        raise RuntimeError("current HEAD differs from the predeclared source head")
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if status:
        raise RuntimeError("source worktree must be clean before the diagnostic")
    modules = _module_source_paths(repo_root)
    module_hashes = {
        name: hashlib.sha256((repo_root / relative).read_bytes()).hexdigest()
        for name, relative in modules.items()
        if relative.endswith(".py")
    }
    return {
        "git_head": head,
        "worktree_clean": True,
        "repo_root": str(repo_root),
        "module_paths": modules,
        "module_sha256": module_hashes,
    }


def run_diagnostic(output_root: str | Path, *, expected_head: str) -> dict[str, object]:
    repo_root = Path(__file__).resolve().parents[1]
    source = _source_preflight(expected_head)
    requested_root = Path(output_root).expanduser().absolute()
    if requested_root.exists() or requested_root.is_symlink():
        raise FileExistsError("diagnostic output root must be new")
    if not requested_root.parent.is_dir():
        raise FileNotFoundError("diagnostic output parent must already exist")
    resolved_root = requested_root.resolve(strict=False)
    if resolved_root.is_relative_to(repo_root):
        raise ValueError("diagnostic output root must be outside the source worktree")
    requested_root.mkdir(parents=False, exist_ok=False)
    protocol_record = {
        "protocol": PROTOCOL,
        "protocol_sha256": PROTOCOL_SHA256,
        "source": source,
        "output_root": str(resolved_root),
        "created_at": _utc_now(),
    }
    _write_json(requested_root / "protocol.json", protocol_record, unique=True)
    result = _execute_cells(requested_root, _run_real_cell)
    result["protocol_sha256"] = PROTOCOL_SHA256
    result["source"] = source
    result["interpretation"] = PROTOCOL["interpretation"]
    _write_json(requested_root / "run.json", result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--expected-head", required=True)
    args = parser.parse_args(argv)
    result = run_diagnostic(args.output_root, expected_head=args.expected_head)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("disposition") == "SYNTHETIC_LEARNABILITY_SUPPORTED" else 2


if __name__ == "__main__":
    sys.exit(main())
