from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np

from research.issue519_portable_exp002_execute import (
    _compound,
    _formal_decision,
)
from trade_rl.evaluation.experiments import ExperimentDecisionKind


def test_direct_script_invocation_can_import_research_dependencies() -> None:
    root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [sys.executable, "research/issue519_portable_exp002_execute.py", "--help"],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_compound_reconstructs_ordered_total_return() -> None:
    values = np.asarray([0.10, -0.05, 0.02], dtype=np.float64)
    expected = (1.10 * 0.95 * 1.02) - 1.0
    assert _compound(values) == expected


def test_formal_decision_accepts_only_full_preregistered_gate() -> None:
    assert (
        _formal_decision(
            positive_effects=4,
            median_excess=0.01,
            candidate_positive_returns=5,
            candidate_median_turnover=858.0,
        )
        is ExperimentDecisionKind.ACCEPT_CANDIDATE
    )


def test_formal_decision_keep_baseline_on_three_or_fewer_positive_returns() -> None:
    assert (
        _formal_decision(
            positive_effects=5,
            median_excess=0.20,
            candidate_positive_returns=3,
            candidate_median_turnover=100.0,
        )
        is ExperimentDecisionKind.KEEP_BASELINE
    )


def test_formal_decision_keep_baseline_on_nonpositive_median_effect() -> None:
    assert (
        _formal_decision(
            positive_effects=5,
            median_excess=0.0,
            candidate_positive_returns=5,
            candidate_median_turnover=100.0,
        )
        is ExperimentDecisionKind.KEEP_BASELINE
    )


def test_formal_decision_inconclusive_in_middle_region() -> None:
    assert (
        _formal_decision(
            positive_effects=3,
            median_excess=0.01,
            candidate_positive_returns=4,
            candidate_median_turnover=900.0,
        )
        is ExperimentDecisionKind.INCONCLUSIVE
    )


def test_formal_decision_does_not_accept_equal_baseline_turnover() -> None:
    assert (
        _formal_decision(
            positive_effects=5,
            median_excess=0.01,
            candidate_positive_returns=5,
            candidate_median_turnover=858.3114067468092,
        )
        is ExperimentDecisionKind.INCONCLUSIVE
    )
