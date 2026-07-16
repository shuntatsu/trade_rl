from __future__ import annotations

import sys
from pathlib import Path

import apply_task2_oracle as base

ROOT = Path(__file__).resolve().parents[1]


def replace_once(path: str, old: str, new: str) -> None:
    target = ROOT / path
    text = target.read_text(encoding="utf-8")
    if old not in text:
        raise RuntimeError(f"missing v3 anchor in {path}: {old[:120]!r}")
    target.write_text(text.replace(old, new, 1), encoding="utf-8")


def add_tests() -> None:
    base.add_tests()
    replace_once(
        "tests/learning/test_oracle_teacher.py",
        "    assert np.count_nonzero(targets[:, 0]) == 0\n    assert np.max(np.abs(targets).sum(axis=1)) <= config.max_gross\n",
        "    assert np.any(targets[:, 0] > 0.0)\n    assert np.max(np.abs(targets).sum(axis=1)) <= config.max_gross\n",
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
        "    approximation_contract: str = \"bounded_state_partial_fill_v1\"\n    schema_version: str = ORACLE_TEACHER_SCHEMA\n",
        "    approximation_contract: str = \"bounded_state_partial_fill_v1\"\n    control_tie_break_penalty: float = 1e-9\n    schema_version: str = ORACLE_TEACHER_SCHEMA\n",
    )
    replace_once(
        "trade_rl/learning/oracle_teacher.py",
        "        if self.approximation_contract != \"bounded_state_partial_fill_v1\":\n            raise ValueError(\"unsupported oracle approximation contract\")\n        cost = self.execution_cost\n",
        "        if self.approximation_contract != \"bounded_state_partial_fill_v1\":\n            raise ValueError(\"unsupported oracle approximation contract\")\n        if (\n            not math.isfinite(self.control_tie_break_penalty)\n            or self.control_tie_break_penalty <= 0.0\n        ):\n            raise ValueError(\n                \"oracle control_tie_break_penalty must be finite and positive\"\n            )\n        cost = self.execution_cost\n",
    )
    replace_once(
        "trade_rl/learning/oracle_teacher.py",
        "            transition_valid, close_factor, candidate_close_weights, _ = (\n                _transition_matrices(\n",
        "            (\n                transition_valid,\n                close_factor,\n                candidate_close_weights,\n                candidate_effective_targets,\n            ) = (\n                _transition_matrices(\n",
    )
    replace_once(
        "trade_rl/learning/oracle_teacher.py",
        "            candidate_scores = np.where(transition_valid, candidate_scores, -np.inf)\n            best_prior = np.argmax(candidate_scores, axis=0)\n",
        "            control_projection = np.abs(\n                states[None, :, :] - candidate_effective_targets\n            ).sum(axis=2)\n            candidate_scores -= (\n                config.control_tie_break_penalty * control_projection\n            )\n            candidate_scores = np.where(transition_valid, candidate_scores, -np.inf)\n            best_prior = np.argmax(candidate_scores, axis=0)\n",
    )
    replace_once(
        "trade_rl/learning/oracle_teacher.py",
        "    targets = states[state_path]\n",
        "    # Labels are bounded submitted targets. Realized partial/no-fill weights\n    # remain in the DP transition state and may drift outside the target grid.\n    targets = states[state_path]\n",
    )


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in {"tests", "implementation"}:
        raise SystemExit("usage: apply_task2_oracle_v3.py tests|implementation")
    if sys.argv[1] == "tests":
        add_tests()
    else:
        add_implementation()
