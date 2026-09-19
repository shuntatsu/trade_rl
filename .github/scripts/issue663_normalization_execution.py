"""Result-blind exactly-once helpers for Issue #663 normalized PPO execution."""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from dataclasses import replace
from hashlib import sha256
from pathlib import Path

import numpy as np

from trade_rl.artifacts import canonical_json_bytes, content_digest
from trade_rl.data.artifacts import load_market_dataset_artifact
from trade_rl.data.features.price_channels import with_price_channels
from trade_rl.evaluation.directional import evaluate_directional_arm
from trade_rl.evaluation.directional_study import development_indices
from trade_rl.evaluation.experiments import inspect_study
from trade_rl.simulation import ExecutionCostConfig
from trade_rl.strategies.rl.ppo import PPOIntentStrategy, fit_ppo_strategy
from trade_rl.strategies.rl.ppo_artifact import save_normalized_ppo

ISSUE_NUMBER = 663
ACTIVATION_ARTIFACT_NAME = "issue663-normalization-activation-v2"
ACTIVATION_SCHEMA = "issue663_normalization_activation_v2"
SEED_CLAIM_SCHEMA = "issue663_normalization_seed_claim_v2"
SEEDS = (0, 1, 2, 3, 4)
PPO_TIMESTEPS = 262_144


def _positive_int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _seed(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value not in SEEDS:
        raise ValueError("seed must be one of 0, 1, 2, 3, 4")
    return value


def _sha(value: object, *, field: str, prefixed: bool = False) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a SHA-256 string")
    prefix = "sha256:" if prefixed else ""
    raw = value.removeprefix(prefix) if prefixed else value
    if len(raw) != 64 or any(char not in "0123456789abcdef" for char in raw):
        raise ValueError(f"{field} must be a SHA-256 string")
    if prefixed and not value.startswith("sha256:"):
        raise ValueError(f"{field} must use the sha256: prefix")
    return value


def _head(value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 40
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise ValueError(
            "implementation_head must be a lowercase hexadecimal commit SHA"
        )
    return value


def _mapping(value: object, *, field: str) -> dict[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{field} must be a string-keyed mapping")
    return dict(value)


def _protocol(protocol: Mapping[str, object]) -> dict[str, object]:
    value = _mapping(protocol, field="protocol")
    seeds = value.get("candidate_seeds")
    if not isinstance(seeds, list) or tuple(seeds) != SEEDS:
        raise ValueError("protocol candidate seed roster differs from Issue 663")
    if value.get("unused_data_accessed") is not False:
        raise ValueError("protocol must keep unused data closed")
    if value.get("production_eligible") is not False:
        raise ValueError("protocol must keep production ineligible")
    return value


def claim_bytes(claim: Mapping[str, object]) -> bytes:
    return canonical_json_bytes(dict(claim))


def seed_claim_name(seed: int) -> str:
    return f"issue663-normalization-seed{_seed(seed)}-claim-v2"


def seed_result_name(seed: int) -> str:
    return f"issue663-normalization-seed{_seed(seed)}-v2"


def seed_fresh_name(seed: int) -> str:
    return f"issue663-normalization-seed{_seed(seed)}-fresh-v2"


def build_activation_claim(
    protocol: Mapping[str, object],
    *,
    workflow_run_id: int,
    workflow_run_attempt: int,
    implementation_head: str,
) -> dict[str, object]:
    protocol_value = _protocol(protocol)
    run_id = _positive_int(workflow_run_id, field="workflow_run_id")
    if isinstance(workflow_run_attempt, bool) or workflow_run_attempt != 1:
        raise ValueError("workflow_run_attempt must be exactly 1")
    return {
        "schema_version": ACTIVATION_SCHEMA,
        "issue_number": ISSUE_NUMBER,
        "workflow_run_id": run_id,
        "workflow_run_attempt": 1,
        "implementation_head": _head(implementation_head),
        "protocol_digest": content_digest(protocol_value),
        "source_dataset_id": protocol_value.get("source_dataset_id"),
        "source_artifact_digest": protocol_value.get("source_artifact_digest"),
        "control_run_id": protocol_value.get("control_run_id"),
        "candidate_seeds": list(SEEDS),
        "local_output_root_is_authority": False,
        "economic_execution_started": False,
        "economic_result_inspected": False,
        "unused_data_accessed": False,
        "final_test_accessed": False,
        "production_eligible": False,
        "live_trading_authorized": False,
    }


def build_seed_claim(
    protocol: Mapping[str, object],
    *,
    seed: int,
    workflow_run_id: int,
    workflow_run_attempt: int,
    implementation_head: str,
    activation_artifact_id: int,
    activation_artifact_digest: str,
    control_artifact_id: int,
    control_artifact_digest: str,
) -> dict[str, object]:
    protocol_value = _protocol(protocol)
    resolved_seed = _seed(seed)
    run_id = _positive_int(workflow_run_id, field="workflow_run_id")
    if isinstance(workflow_run_attempt, bool) or workflow_run_attempt != 1:
        raise ValueError("workflow_run_attempt must be exactly 1")
    return {
        "schema_version": SEED_CLAIM_SCHEMA,
        "issue_number": ISSUE_NUMBER,
        "seed": resolved_seed,
        "workflow_run_id": run_id,
        "workflow_run_attempt": 1,
        "implementation_head": _head(implementation_head),
        "protocol_digest": content_digest(protocol_value),
        "activation_artifact_id": _positive_int(
            activation_artifact_id, field="activation_artifact_id"
        ),
        "activation_artifact_digest": _sha(
            activation_artifact_digest,
            field="activation_artifact_digest",
            prefixed=True,
        ),
        "control_artifact_id": _positive_int(
            control_artifact_id, field="control_artifact_id"
        ),
        "control_artifact_digest": _sha(
            control_artifact_digest,
            field="control_artifact_digest",
            prefixed=True,
        ),
        "source_dataset_id": protocol_value.get("source_dataset_id"),
        "local_output_root_is_authority": False,
        "economic_execution_started": False,
        "candidate_result_inspected": False,
        "unused_data_accessed": False,
        "final_test_accessed": False,
        "production_eligible": False,
        "live_trading_authorized": False,
    }


def validate_remote_slot(
    payload: Mapping[str, object],
    *,
    artifact_name: str,
    state: str,
    workflow_run_id: int | None = None,
) -> dict[str, object] | None:
    data = _mapping(payload, field="artifact listing")
    count = data.get("total_count")
    artifacts = data.get("artifacts")
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise ValueError("artifact total_count must be a non-negative integer")
    if not isinstance(artifacts, list) or any(
        not isinstance(item, Mapping) for item in artifacts
    ):
        raise ValueError("artifact listing must contain an artifact array")
    if count != len(artifacts):
        raise ValueError("artifact total_count differs from artifact array length")
    matching = [dict(item) for item in artifacts if item.get("name") == artifact_name]
    if state == "empty":
        if matching:
            raise RuntimeError(f"{artifact_name} is already claimed")
        return None
    if state != "claimed":
        raise ValueError("state must be empty or claimed")
    if len(matching) != 1:
        raise RuntimeError(f"{artifact_name} must contain exactly one claim")
    artifact = matching[0]
    if artifact.get("expired") is not False:
        raise RuntimeError(f"{artifact_name} must be present and unexpired")
    expected_run = _positive_int(workflow_run_id, field="workflow_run_id")
    workflow = _mapping(artifact.get("workflow_run"), field="artifact workflow")
    if workflow.get("id") != expected_run:
        raise RuntimeError(f"{artifact_name} belongs to another workflow")
    return artifact


def seed_result_written_message(seed: int) -> str:
    return f"seed{_seed(seed)}: result bytes written; interpretation deferred"


def _factory(strategy: PPOIntentStrategy, feature_indices: tuple[int, ...]):
    def build() -> PPOIntentStrategy:
        return PPOIntentStrategy(
            strategy.policy,
            feature_indices=feature_indices,
            feature_normalizer=strategy.feature_normalizer,
        )

    return build


def run_candidate_seed(source: Path, output: Path, *, seed: int) -> dict[str, object]:
    """Fit/replay one frozen normalized PPO seed after its remote claim exists."""

    resolved_seed = _seed(seed)
    output.mkdir(parents=True, exist_ok=False)
    (output / "started.json").write_bytes(
        canonical_json_bytes(
            {
                "schema_version": "issue663_normalization_seed_started_v2",
                "issue_number": ISSUE_NUMBER,
                "seed": resolved_seed,
                "economic_result_interpreted": False,
                "unused_data_accessed": False,
                "final_test_accessed": False,
                "production_eligible": False,
                "live_trading_authorized": False,
            }
        )
    )
    try:
        torch = importlib.import_module("torch")
        torch.set_num_threads(1)
        dataset = with_price_channels(load_market_dataset_artifact(source / "dataset"))
        config = inspect_study(source / "study").plan.baseline_config
        cutoff = (
            int(np.searchsorted(dataset.timestamps, np.datetime64(config.fit_cutoff)))
            - 1
        )
        strategy = fit_ppo_strategy(
            dataset,
            feature_indices=config.feature_indices,
            fit_symbol_indices=config.fit_symbol_indices,
            start_index=0,
            stop_index=cutoff,
            gross_budget=0.1,
            total_timesteps=PPO_TIMESTEPS,
            seed=resolved_seed,
            initial_capital=10_000.0,
            execution_cost=replace(
                ExecutionCostConfig.zero(),
                processing_bar_volume_capacity=False,
            ),
            training_layout="sequential",
            rollout_steps_per_env=None,
            normalize_features=True,
        )
        model_bundle_digest = save_normalized_ppo(output / "model", strategy)
        start, stop = development_indices(dataset)
        factory = _factory(strategy, config.feature_indices)
        result = evaluate_directional_arm(
            dataset,
            factory,
            start_index=start,
            stop_index=stop,
        )
        if result["qualified"]:
            result["stress"] = [
                evaluate_directional_arm(
                    dataset,
                    factory,
                    start_index=start,
                    stop_index=stop,
                    cost_multiplier=2.0,
                ),
                evaluate_directional_arm(
                    dataset,
                    factory,
                    start_index=start,
                    stop_index=stop,
                    latency_bars=1,
                ),
            ]
            result["by_symbol"] = {
                symbol: evaluate_directional_arm(
                    dataset,
                    factory,
                    start_index=start,
                    stop_index=stop,
                    symbol_index=index,
                )
                for index, symbol in enumerate(dataset.symbols)
            }

        payload: dict[str, object] = dict(result)
        payload.update(
            {
                "evidence_schema": "issue663_normalization_seed_result_v2",
                "issue_number": ISSUE_NUMBER,
                "seed": resolved_seed,
                "model_bundle_digest": model_bundle_digest,
                "normalize_features": True,
                "training_layout": "sequential",
                "caller_total_timesteps": PPO_TIMESTEPS,
                "economic_result_interpreted": False,
                "unused_data_accessed": False,
                "final_test_accessed": False,
                "production_eligible": False,
                "live_trading_authorized": False,
            }
        )
        raw = canonical_json_bytes(payload)
        (output / "result.json").write_bytes(raw)
        (output / "result.sha256.json").write_bytes(
            canonical_json_bytes({"sha256": sha256(raw).hexdigest()})
        )
        print(seed_result_written_message(resolved_seed), flush=True)
        return payload
    except BaseException as error:
        (output / "failed.json").write_bytes(
            canonical_json_bytes(
                {
                    "schema_version": "issue663_normalization_seed_failure_v2",
                    "issue_number": ISSUE_NUMBER,
                    "seed": resolved_seed,
                    "error": repr(error),
                    "economic_result_interpreted": False,
                    "unused_data_accessed": False,
                    "final_test_accessed": False,
                    "production_eligible": False,
                    "live_trading_authorized": False,
                }
            )
        )
        raise
