from __future__ import annotations

from importlib import import_module
from importlib.util import find_spec
from pathlib import Path

import pytest

from trade_rl.evaluation.experiments import ExperimentDecisionKind

MODULE = "research.issue519_portable_exp002_postverify"


def _module():
    spec = find_spec(MODULE)
    assert spec is not None, "Experiment 0002 primary postverifier is not implemented"
    return import_module(MODULE)


def test_independent_decision_matches_frozen_accept_boundary() -> None:
    module = _module()
    assert (
        module._independent_decision(
            positive_effects=4,
            median_excess=0.01,
            candidate_positive_returns=5,
            candidate_median_turnover=858.0,
        )
        is ExperimentDecisionKind.ACCEPT_CANDIDATE
    )


def test_independent_decision_keeps_on_preregistered_rejection_gate() -> None:
    module = _module()
    assert (
        module._independent_decision(
            positive_effects=5,
            median_excess=0.01,
            candidate_positive_returns=3,
            candidate_median_turnover=100.0,
        )
        is ExperimentDecisionKind.KEEP_BASELINE
    )


def test_independent_decision_is_inconclusive_between_gates() -> None:
    module = _module()
    assert (
        module._independent_decision(
            positive_effects=3,
            median_excess=0.01,
            candidate_positive_returns=4,
            candidate_median_turnover=900.0,
        )
        is ExperimentDecisionKind.INCONCLUSIVE
    )


def _write(root: Path, relative: str, value: bytes) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value)


def _minimal_immutable_pair(tmp_path: Path) -> tuple[Path, Path]:
    prereg = tmp_path / "prereg"
    result = tmp_path / "result"
    exact_files = (
        "bootstrap.json",
        "portable-exp002-prereg-index.json",
        "study/plan.json",
        "study/experiments/0002/definition.json",
        "dataset/manifest.json",
        "dataset/arrays.npz",
    )
    for relative in exact_files:
        _write(prereg, relative, relative.encode())
        _write(result, relative, relative.encode())
    for relative in (
        "study/baseline/evidence/meta.json",
        "study/experiments/0001/definition.json",
        "study/experiments/0001/candidate/meta.json",
        "study/experiments/0001/verification.json",
        "study/experiments/0001/comparison.json",
        "study/experiments/0001/decision.json",
    ):
        _write(prereg, relative, relative.encode())
        _write(result, relative, relative.encode())
    return prereg, result


def test_immutable_prefix_accepts_exact_prereg_result_prefix(tmp_path: Path) -> None:
    module = _module()
    prereg, result = _minimal_immutable_pair(tmp_path)
    module._assert_immutable_prefix(prereg, result)


def test_immutable_prefix_rejects_prior_experiment_tampering(tmp_path: Path) -> None:
    module = _module()
    prereg, result = _minimal_immutable_pair(tmp_path)
    _write(result, "study/experiments/0001/decision.json", b"tampered")
    with pytest.raises(RuntimeError, match="Experiment 0001"):
        module._assert_immutable_prefix(prereg, result)
