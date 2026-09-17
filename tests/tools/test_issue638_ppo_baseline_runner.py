from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest


RUNNER = Path(".github/scripts/issue638_ppo_baseline_runner.py")


def _load_runner() -> ModuleType:
    spec = importlib.util.spec_from_file_location("issue638_ppo_baseline_runner", RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Evidence:
    arm = "baseline"
    seed = 2
    candidate_training_authorized = False

    def to_payload(self) -> dict[str, object]:
        return {
            "arm": self.arm,
            "seed": self.seed,
            "candidate_training_authorized": self.candidate_training_authorized,
        }


def test_runner_is_baseline_only_and_calls_evaluator_exactly_once() -> None:
    module = _load_runner()
    calls: list[dict[str, Any]] = []

    def evaluator(dataset: object, spec: object, *, arm: str, seed: int) -> _Evidence:
        calls.append({"dataset": dataset, "spec": spec, "arm": arm, "seed": seed})
        evidence = _Evidence()
        evidence.seed = seed
        return evidence

    dataset = object()
    spec = object()
    evidence = module.evaluate_baseline_seed(
        dataset,
        spec,
        seed=2,
        evaluator=evaluator,
    )
    assert evidence.arm == "baseline"
    assert evidence.candidate_training_authorized is False
    assert calls == [{"dataset": dataset, "spec": spec, "arm": "baseline", "seed": 2}]


@pytest.mark.parametrize("seed", [True, -1, 5, 1.0, "1"])
def test_runner_rejects_noncanonical_seed_before_evaluation(seed: object) -> None:
    module = _load_runner()
    calls = 0

    def evaluator(*args: object, **kwargs: object) -> object:
        nonlocal calls
        calls += 1
        raise AssertionError("evaluator must not run for invalid seed")

    with pytest.raises(ValueError, match="seed"):
        module.evaluate_baseline_seed(object(), object(), seed=seed, evaluator=evaluator)
    assert calls == 0


def test_canonical_evidence_bytes_reject_candidate_or_seed_drift() -> None:
    module = _load_runner()
    evidence = _Evidence()
    raw = module.canonical_baseline_evidence_bytes(evidence, seed=2)
    assert raw.endswith(b"\n")
    assert b'"arm":"baseline"' in raw
    assert b'"candidate_training_authorized":false' in raw

    evidence.arm = "candidate"
    with pytest.raises(ValueError, match="baseline"):
        module.canonical_baseline_evidence_bytes(evidence, seed=2)

    evidence.arm = "baseline"
    evidence.seed = 1
    with pytest.raises(ValueError, match="seed"):
        module.canonical_baseline_evidence_bytes(evidence, seed=2)
