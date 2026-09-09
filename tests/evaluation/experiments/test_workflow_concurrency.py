from __future__ import annotations

import multiprocessing
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import pytest

from tests.evaluation.experiments.test_evidence import _config
from tests.evaluation.experiments.test_workflow import _with_baseline
from trade_rl.evaluation.experiments import (
    ControlledFactor,
    InvalidExperimentStateError,
    StudyFrozenError,
    StudyOutcome,
    define_experiment,
    freeze_study,
    inspect_study,
)


def _freeze_worker(root_text: str, start, results) -> None:
    if not start.wait(10.0):
        raise RuntimeError("race start was not released")
    try:
        freeze_study(
            root_text,
            outcome=StudyOutcome.NO_WINNER,
            rationale="Concurrent freeze race.",
            frozen_by="race-worker",
            frozen_at=datetime(2026, 9, 9, tzinfo=UTC),
        )
    except InvalidExperimentStateError:
        results.put("freeze:nonterminal")
    else:
        results.put("freeze:ok")


def _define_worker(
    root_text: str,
    dataset_text: str,
    baseline_digest: str,
    start,
    results,
) -> None:
    if not start.wait(10.0):
        raise RuntimeError("race start was not released")
    try:
        define_experiment(
            root_text,
            dataset_root=dataset_text,
            hypothesis="Concurrent definition race.",
            factor=ControlledFactor.PPO_TRAINING_BUDGET,
            candidate_config=replace(_config(), ppo_total_timesteps=64),
            baseline_evidence_digest=baseline_digest,
        )
    except StudyFrozenError:
        results.put("define:frozen")
    else:
        results.put("define:ok")


def test_freeze_and_definition_race_has_one_serializable_winner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, dataset_root, snapshot = _with_baseline(tmp_path, monkeypatch)
    assert snapshot.baseline is not None

    context = multiprocessing.get_context("spawn")
    start = context.Event()
    results = context.Queue()
    freeze = context.Process(
        target=_freeze_worker,
        args=(str(root), start, results),
    )
    define = context.Process(
        target=_define_worker,
        args=(
            str(root),
            str(dataset_root),
            snapshot.baseline.fingerprint,
            start,
            results,
        ),
    )

    freeze.start()
    define.start()
    start.set()

    freeze.join(20.0)
    define.join(20.0)
    assert freeze.exitcode == 0
    assert define.exitcode == 0

    observed = {results.get(timeout=5.0), results.get(timeout=5.0)}
    assert observed in (
        {"freeze:ok", "define:frozen"},
        {"freeze:nonterminal", "define:ok"},
    )

    rebuilt = inspect_study(root)
    if "freeze:ok" in observed:
        assert rebuilt.freeze is not None
        assert rebuilt.experiment_sequences == ()
    else:
        assert rebuilt.freeze is None
        assert rebuilt.experiment_sequences == (1,)
        assert rebuilt.terminal_sequences == ()
