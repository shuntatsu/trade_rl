from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"
TARGET_TEST = "tests/evaluation/test_portfolio_family_comparison.py"
ATTEMPT_TEST = "tests/evaluation/test_portfolio_family_attempt.py"
SCENARIO_ATTEMPT_TEST = "tests/evaluation/test_portfolio_family_scenario_attempt_v2.py"
SCENARIO_REPLAY_TEST = "tests/evaluation/test_portfolio_family_scenario_replay_v2.py"
MANIFEST_TEST = "tests/evaluation/experiments/test_four_family_input_manifest.py"


def test_ppo_runtime_runs_four_family_actual_ppo_integration() -> None:
    workflow = WORKFLOW.read_text(encoding="utf-8")
    job = workflow.split("  ppo-runtime:\n", 1)[1].split("  guide:\n", 1)[0]

    assert (
        "- name: Install PPO runtime\n        run: uv sync --extra dev --extra train-sb3"
        in job
    )
    integration_step = job.split("      - name: Real SB3 PPO integration\n", 1)[1]
    command_match = re.search(r"^        run: (.+)$", integration_step, re.MULTILINE)

    assert command_match is not None
    selected_tests = command_match.group(1).split()
    assert TARGET_TEST in selected_tests
    assert ATTEMPT_TEST in selected_tests
    assert SCENARIO_ATTEMPT_TEST in selected_tests
    assert SCENARIO_REPLAY_TEST in selected_tests
    assert MANIFEST_TEST in selected_tests
