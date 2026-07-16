from __future__ import annotations

import sys
from pathlib import Path

import apply_task2_oracle as base

ROOT = Path(__file__).resolve().parents[1]


def replace_once(path: str, old: str, new: str) -> None:
    target = ROOT / path
    text = target.read_text(encoding="utf-8")
    if old not in text:
        raise RuntimeError(f"missing supplemental anchor in {path}: {old[:120]!r}")
    target.write_text(text.replace(old, new, 1), encoding="utf-8")


def add_tests() -> None:
    base.add_tests()
    replace_once(
        "tests/learning/test_oracle_teacher.py",
        "    assert np.count_nonzero(targets[:, 0]) == 0\n    assert np.max(np.abs(targets).sum(axis=1)) <= config.max_gross\n",
        "    assert np.any(targets[:, 0] > 0.0)\n    assert np.max(targets[:, 0]) < config.max_abs_weight\n    assert np.max(np.abs(targets).sum(axis=1)) <= config.max_gross\n",
    )
    replace_once(
        "tests/learning/test_oracle_teacher.py",
        "    tradable[3:, 0] = False\n",
        "    tradable[1:, 0] = False\n",
    )
    replace_once(
        "tests/learning/test_oracle_teacher.py",
        "    assert np.count_nonzero(targets[2:, 0]) == 0\n",
        "    assert np.count_nonzero(targets[:, 0]) == 0\n",
    )


def add_implementation() -> None:
    base.add_implementation()
    replace_once(
        "trade_rl/learning/oracle_teacher.py",
        "    close_weights = np.zeros((steps, state_count, dataset.n_symbols), dtype=np.float64)\n    cash_index = int(np.flatnonzero(np.all(np.isclose(states, 0.0), axis=1))[0])\n",
        "    close_weights = np.zeros((steps, state_count, dataset.n_symbols), dtype=np.float64)\n    selected_targets = np.zeros_like(close_weights)\n    cash_index = int(np.flatnonzero(np.all(np.isclose(states, 0.0), axis=1))[0])\n",
    )
    replace_once(
        "trade_rl/learning/oracle_teacher.py",
        "            transition_valid, close_factor, candidate_close_weights, _ = (\n                _transition_matrices(\n",
        "            (\n                transition_valid,\n                close_factor,\n                candidate_close_weights,\n                candidate_effective_targets,\n            ) = (\n                _transition_matrices(\n",
    )
    replace_once(
        "trade_rl/learning/oracle_teacher.py",
        "            close_weights[step] = candidate_close_weights[\n                best_prior, np.arange(state_count)\n            ]\n        elif step == 0:\n",
        "            close_weights[step] = candidate_close_weights[\n                best_prior, np.arange(state_count)\n            ]\n            selected_targets[step] = candidate_effective_targets[\n                best_prior, np.arange(state_count)\n            ]\n        elif step == 0:\n",
    )
    replace_once(
        "trade_rl/learning/oracle_teacher.py",
        "        invalid = ~np.isfinite(scores[step])\n        close_weights[step, invalid] = 0.0\n\n    final_state = (\n",
        "        invalid = ~np.isfinite(scores[step])\n        close_weights[step, invalid] = 0.0\n        selected_targets[step, invalid] = 0.0\n\n    final_state = (\n",
    )
    replace_once(
        "trade_rl/learning/oracle_teacher.py",
        "    targets = states[state_path]\n",
        "    targets = (\n        selected_targets[np.arange(steps), state_path]\n        if config.signal_delay_decisions == 0\n        else states[state_path]\n    )\n",
    )


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in {"tests", "implementation"}:
        raise SystemExit("usage: apply_task2_oracle_fix.py tests|implementation")
    if sys.argv[1] == "tests":
        add_tests()
    else:
        add_implementation()
