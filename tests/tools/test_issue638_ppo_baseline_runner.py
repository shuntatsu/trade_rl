from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from trade_rl.artifacts.canonical import canonical_json_bytes

RUNNER = Path(".github/scripts/issue638_ppo_baseline_runner.py")


def _load_runner() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "issue638_ppo_baseline_runner", RUNNER
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Evidence:
    arm = "baseline"
    seed = 2
    training_layout = "sequential"
    rollout_steps_per_env = None
    candidate_training_authorized = False

    def to_payload(self) -> dict[str, object]:
        return {
            "arm": self.arm,
            "seed": self.seed,
            "training_layout": self.training_layout,
            "rollout_steps_per_env": self.rollout_steps_per_env,
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
        module.evaluate_baseline_seed(
            object(), object(), seed=seed, evaluator=evaluator
        )
    assert calls == 0


def test_canonical_evidence_bytes_reject_candidate_or_seed_drift() -> None:
    module = _load_runner()
    evidence = _Evidence()
    raw = module.canonical_baseline_evidence_bytes(evidence, seed=2)
    assert raw == (
        b'{"arm":"baseline","candidate_training_authorized":false,'
        b'"rollout_steps_per_env":null,"seed":2,"training_layout":"sequential"}'
    )
    assert b'"arm":"baseline"' in raw
    assert b'"candidate_training_authorized":false' in raw

    evidence.arm = "candidate"
    with pytest.raises(ValueError, match="baseline"):
        module.canonical_baseline_evidence_bytes(evidence, seed=2)

    evidence.arm = "baseline"
    evidence.seed = 1
    with pytest.raises(ValueError, match="seed"):
        module.canonical_baseline_evidence_bytes(evidence, seed=2)

    evidence.seed = 2
    evidence.training_layout = "interleaved"
    with pytest.raises(ValueError, match="sequential"):
        module.canonical_baseline_evidence_bytes(evidence, seed=2)


def test_activation_claim_is_canonical_pretraining_and_seed_bound() -> None:
    module = _load_runner()
    helper_head = "a" * 40
    helper_blob_sha = "b" * 40
    raw = module.activation_claim_bytes(
        seed=2,
        run_id=12345,
        run_attempt=1,
        helper_head=helper_head,
        helper_blob_sha=helper_blob_sha,
    )
    payload = json.loads(raw)
    assert raw == canonical_json_bytes(payload)
    assert payload["schema_version"] == "issue638_ppo_sequential_baseline_activation_v1"
    assert payload["issue_number"] == 638
    assert payload["seed"] == 2
    assert payload["workflow_run_id"] == 12345
    assert payload["workflow_run_attempt"] == 1
    assert payload["helper_head"] == helper_head
    assert payload["helper_blob_sha"] == helper_blob_sha
    assert payload["arm"] == "baseline"
    assert payload["training_layout"] == "sequential"
    assert payload["rollout_steps_per_env"] is None
    assert payload["caller_total_timesteps"] == 100_000
    assert payload["evaluator_started_at_claim"] is False
    assert payload["real_dataset_loaded_at_claim"] is False
    assert payload["result_artifact_published_at_claim"] is False
    assert payload["candidate_training_authorized"] is False
    assert payload["economic_result_interpreted"] is False
    assert payload["final_test_accessed"] is False
    assert payload["production_eligible"] is False
    assert payload["live_trading_authorized"] is False
    assert payload["merge_authorized"] is False


@pytest.mark.parametrize(
    ("run_id", "run_attempt", "helper_head", "helper_blob_sha"),
    [
        (0, 1, "a" * 40, "b" * 40),
        (True, 1, "a" * 40, "b" * 40),
        (1, 2, "a" * 40, "b" * 40),
        (1, True, "a" * 40, "b" * 40),
        (1, 1, "short", "b" * 40),
        (1, 1, "a" * 40, "short"),
    ],
)
def test_activation_claim_rejects_non_first_or_ambiguous_lineage(
    run_id: object,
    run_attempt: object,
    helper_head: object,
    helper_blob_sha: object,
) -> None:
    module = _load_runner()
    with pytest.raises(ValueError):
        module.activation_claim_bytes(
            seed=2,
            run_id=run_id,
            run_attempt=run_attempt,
            helper_head=helper_head,
            helper_blob_sha=helper_blob_sha,
        )


def test_slot_state_consumes_activation_before_training_and_blocks_reactivation() -> None:
    module = _load_runner()
    module._validate_slot_state(
        slot="empty", activation_count=0, result_count=0, fresh_count=0
    )
    module._validate_slot_state(
        slot="claimed", activation_count=1, result_count=0, fresh_count=0
    )
    module._validate_slot_state(
        slot="published", activation_count=1, result_count=1, fresh_count=0
    )

    invalid = (
        ("empty", 1, 0, 0),
        ("empty", 0, 1, 0),
        ("claimed", 0, 0, 0),
        ("claimed", 1, 1, 0),
        ("published", 0, 1, 0),
        ("published", 1, 0, 0),
        ("published", 1, 1, 1),
    )
    for slot, activation_count, result_count, fresh_count in invalid:
        with pytest.raises(SystemExit):
            module._validate_slot_state(
                slot=slot,
                activation_count=activation_count,
                result_count=result_count,
                fresh_count=fresh_count,
            )
