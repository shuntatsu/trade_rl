"""Result-blind plan builder for the Issue 609 PPO global-BTC development evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import replace
from pathlib import Path
from typing import Any

from trade_rl.artifacts.canonical import canonical_json_bytes
from trade_rl.artifacts.hashing import content_digest
from trade_rl.data import (
    inspect_published_market_dataset_artifact,
    load_market_dataset_artifact,
)
from trade_rl.evaluation.experiments import (
    ControlledFactor,
    ResolvedRunConfig,
    create_study,
)
from trade_rl.evaluation.runs import CandidateRunConfig, resolve_candidate_run_spec
from trade_rl.strategies.rl.ppo import PPO_GLOBAL_BTC_REGIME_CONTEXT

SOURCE_BASELINE_RUN_ID = 34700123151
SOURCE_BASELINE_ARTIFACT_ID = 10301701698
SOURCE_BASELINE_ARTIFACT_NAME = "canonical-m2-portable-baseline-v1-34700123151"
SOURCE_BASELINE_ARTIFACT_DIGEST = (
    "sha256:f814fe4e205f8714c4344238911aae16e89ce0279908265feb1fdc85069b2a0a"
)
SOURCE_DATASET_ID = "d7a04ede97a1bb37b811c3e071f325fa007525a6040927e6793d8cc7c10f538f"
SOURCE_DATASET_ARTIFACT_DIGEST = (
    "77362e148c713840dda64e0ef70e663cce6611407eac31fefbb9fccca73ae8f8"
)
SOURCE_STUDY_DIGEST = "3d8404061a4082a8e9b3c786d9f5fc9a4347631dff39c201e3cba70470dfeb79"
SOURCE_BASELINE_EVIDENCE_FINGERPRINT = (
    "526b485d60b394739b7a8d03535e1cfa08b26fa920119c70ee53f05faa53dd29"
)

FACTOR_PREREG_HEAD = "d84235185c5e79327ea7ced7073f0026970ba46b"
FACTOR_PREREG_PROTOCOL_DIGEST = (
    "65a803034fca8180c165728022221d7b3c2fa5dcdd8349d3ca52c3a6adc72228"
)
IMPLEMENTATION_HEAD = "1ae494ce182189c5be61e8007dc956f912d8b9d5"
IMPLEMENTATION_TREE = "4f14c5a06505b8ff9811b473bf6dea296453a13b"
IMPLEMENTATION_INDEX_DIGEST = (
    "d5390a0025a92e4e2a1e987b7b7ac8ee6ef3974c5c8247b0256ae70c0d2d6e10"
)
IMPLEMENTATION_SEAL_RUN_ID = 35084553267
IMPLEMENTATION_SEAL_ARTIFACT_ID = 10441133239
IMPLEMENTATION_SEAL_ARTIFACT_DIGEST = (
    "sha256:8f8a3319f9213bf999b3ce2d911882fd9276daf575a7cde99ec33de9e8208b37"
)
IMPLEMENTATION_FRESH_ARTIFACT_ID = 10441194709
IMPLEMENTATION_FRESH_ARTIFACT_DIGEST = (
    "sha256:39f578321699d3deba1c1cc67ee9d5401c7a45573088d63b2c346f09721c7f52"
)

EVALUATION_PREREG_HEAD = "02fd08798b3f23cc3db13d96ac27778e93592019"
EVALUATION_PREREG_PROTOCOL_DIGEST = (
    "e52a19b859500ea01e900a1ea7209bd545c03acd08435d1ff615b7539ec75d2b"
)
EVALUATION_PREREG_SEAL_RUN_ID = 35088018055
EVALUATION_PREREG_SEAL_ARTIFACT_ID = 10443216354
EVALUATION_PREREG_SEAL_ARTIFACT_DIGEST = (
    "sha256:09c9033139b841b3e769ff1fb43fa291092f47fea5718c7ae396aebd71444986"
)
EVALUATION_PREREG_FRESH_ARTIFACT_ID = 10443355963
EVALUATION_PREREG_FRESH_ARTIFACT_DIGEST = (
    "sha256:6040f352bebb5c0da94a0c25f0a8e41665788826c980f512588089758d8ab44c"
)

EXECUTION_FRAMEWORK_HEAD = "ff332f0cc2dec46828aa1031d438f2f2f1705ba8"
EXECUTION_FRAMEWORK_TREE = "3d195e67a4ac3dff2f2ede4e11bd0bc2bccbfec6"
EXECUTION_FRAMEWORK_EXACT_VERIFY_RUN_ID = 35091958231
EXECUTION_FRAMEWORK_SEAL_RUN_ID = 35092272943
EXECUTION_FRAMEWORK_SEAL_ARTIFACT_ID = 10444158266
EXECUTION_FRAMEWORK_SEAL_ARTIFACT_DIGEST = (
    "sha256:f582fcd379be009fbabab1647031299bf046bd33d63dfe5d9fbed5cf9dcb66e4"
)
EXECUTION_FRAMEWORK_FRESH_ARTIFACT_ID = 10444820998
EXECUTION_FRAMEWORK_FRESH_ARTIFACT_DIGEST = (
    "sha256:b36fca9eecafef8d488dc80f0bfbacb843cc97699f331805e2d76b0bf682747c"
)
EXECUTION_FRAMEWORK_RECOVERY_RUN_ID = 35092562372
EXECUTION_FRAMEWORK_RECOVERY_ARTIFACT_ID = 10444891955
EXECUTION_FRAMEWORK_RECOVERY_ARTIFACT_DIGEST = (
    "sha256:54b1bccb360b985b15583a3ff6550b8342b8f6de4b8cc3f949a3d0a7ff44a9c8"
)

EXPECTED_SYMBOLS = ("BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT")
EXPECTED_PPO_SEEDS = (0, 1, 2, 3, 4)
EXPECTED_CHANGED_PATHS = (
    ("ppo_global_context",),
    ("ppo_observation_schema",),
    ("schema_version",),
)
RESEARCH_QUESTION = (
    "Does the sealed fit-scope-safe BTCUSDT 24h regime context robustly improve "
    "universal teacher-free PPO on the frozen portable development Dataset under "
    "identical execution/accounting/risk semantics?"
)


def _load_json_object(path: Path) -> dict[str, Any]:
    try:
        raw: Any = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid JSON: {path}") from error
    if not isinstance(raw, dict) or any(not isinstance(key, str) for key in raw):
        raise ValueError(f"JSON object required: {path}")
    return raw


def _require_equal(actual: object, expected: object, *, field: str) -> None:
    if type(actual) is not type(expected) or actual != expected:
        raise ValueError(f"{field} differs from frozen authority")


def _source_baseline_config(plan: dict[str, Any]) -> CandidateRunConfig:
    raw = plan.get("baseline_config")
    if not isinstance(raw, dict):
        raise ValueError("source Study baseline_config is malformed")
    _require_equal(raw.get("schema_version"), "resolved_run_config_v2", field="schema_version")
    _require_equal(raw.get("ppo_observation_schema"), "ppo_observation_v2", field="ppo_observation_schema")
    _require_equal(raw.get("ppo_global_feature_names"), [], field="ppo_global_feature_names")
    if "ppo_global_context" in raw:
        raise ValueError("source baseline unexpectedly defines PPO global context")
    return CandidateRunConfig(
        signal_name=str(raw["signal_name"]),
        feature_names=tuple(str(value) for value in raw["feature_names"]),
        fit_symbol_names=tuple(str(value) for value in raw["fit_symbol_names"]),
        fit_cutoff=str(raw["fit_cutoff"]),
        evaluation_start=str(raw["evaluation_start"]),
        evaluation_stop_exclusive=str(raw["evaluation_stop_exclusive"]),
        rule_entry_threshold=float(raw["rule_entry_threshold"]),
        rule_exit_threshold=float(raw["rule_exit_threshold"]),
        forecast_entry_threshold=float(raw["forecast_entry_threshold"]),
        forecast_exit_threshold=float(raw["forecast_exit_threshold"]),
        ppo_total_timesteps=int(raw["ppo_total_timesteps"]),
        ppo_seed=int(raw["ppo_seed"]),
        gross_budget=float(raw["gross_budget"]),
        initial_capital=float(raw["initial_capital"]),
    )


def _top_level_changed_paths(
    baseline: dict[str, object],
    candidate: dict[str, object],
) -> tuple[tuple[str, ...], ...]:
    changed = []
    for key in sorted(set(baseline) | set(candidate)):
        if key not in baseline or key not in candidate or baseline[key] != candidate[key]:
            changed.append((key,))
    return tuple(changed)


def _validate_source_authority(source_root: Path) -> tuple[dict[str, Any], CandidateRunConfig]:
    dataset_root = source_root / "dataset"
    source_study_root = source_root / "study"
    plan_path = source_study_root / "plan.json"
    manifest_path = source_study_root / "baseline" / "evidence" / "manifest.json"
    if not plan_path.is_file() or not manifest_path.is_file():
        raise ValueError("canonical baseline artifact is incomplete")

    artifact = inspect_published_market_dataset_artifact(dataset_root)
    dataset = load_market_dataset_artifact(dataset_root)
    _require_equal(dataset.dataset_id, SOURCE_DATASET_ID, field="source dataset id")
    _require_equal(artifact.artifact_digest, SOURCE_DATASET_ARTIFACT_DIGEST, field="dataset artifact digest")
    _require_equal(tuple(dataset.symbols), EXPECTED_SYMBOLS, field="dataset symbols")

    plan = _load_json_object(plan_path)
    _require_equal(content_digest(plan), SOURCE_STUDY_DIGEST, field="source Study digest")
    _require_equal(plan.get("dataset_id"), SOURCE_DATASET_ID, field="source Study dataset id")
    _require_equal(plan.get("dataset_artifact_digest"), SOURCE_DATASET_ARTIFACT_DIGEST, field="source Study dataset digest")
    _require_equal(plan.get("symbols"), list(EXPECTED_SYMBOLS), field="source Study symbols")
    _require_equal(plan.get("ppo_seeds"), list(EXPECTED_PPO_SEEDS), field="source Study PPO seeds")
    _require_equal(plan.get("n_bootstrap"), 2_000, field="source Study bootstrap count")
    _require_equal(plan.get("bootstrap_seed"), 1_729, field="source Study bootstrap seed")

    baseline_manifest = _load_json_object(manifest_path)
    _require_equal(
        baseline_manifest.get("fingerprint"),
        SOURCE_BASELINE_EVIDENCE_FINGERPRINT,
        field="source baseline EvidenceSet fingerprint",
    )
    _require_equal(
        baseline_manifest.get("ppo_seeds"),
        list(EXPECTED_PPO_SEEDS),
        field="source baseline PPO seeds",
    )
    return plan, _source_baseline_config(plan)


def build_plan(*, source_root: Path, output_root: Path, plan_run_id: int) -> None:
    source_plan, baseline_config = _validate_source_authority(source_root)
    dataset_root = source_root / "dataset"
    candidate_config = replace(
        baseline_config,
        ppo_global_context=PPO_GLOBAL_BTC_REGIME_CONTEXT,
    )

    if output_root.exists() and any(output_root.iterdir()):
        raise ValueError("output root must be empty")
    output_root.mkdir(parents=True, exist_ok=True)
    study_root = output_root / "study"
    snapshot = create_study(
        study_root,
        dataset_root=dataset_root,
        research_question=RESEARCH_QUESTION,
        baseline_config=baseline_config,
        ppo_seeds=EXPECTED_PPO_SEEDS,
        allowed_factors=(ControlledFactor.FEATURE_SET,),
        max_experiments=1,
        n_bootstrap=2_000,
        bootstrap_seed=1_729,
    )
    if snapshot.baseline_evidence_digest is not None or snapshot.experiment_sequences:
        raise ValueError("plan-only Study unexpectedly contains economic evidence")
    if snapshot.plan.dataset_id != SOURCE_DATASET_ID:
        raise ValueError("new Study dataset identity drift")
    if snapshot.plan.ppo_seeds != EXPECTED_PPO_SEEDS:
        raise ValueError("new Study PPO seed roster drift")
    if snapshot.plan.allowed_factors != (ControlledFactor.FEATURE_SET,):
        raise ValueError("new Study factor authority drift")
    if snapshot.plan.max_experiments != 1:
        raise ValueError("new Study experiment budget drift")

    artifact = inspect_published_market_dataset_artifact(dataset_root)
    dataset = load_market_dataset_artifact(dataset_root)
    candidate_spec = resolve_candidate_run_spec(
        dataset,
        dataset_artifact_schema=artifact.schema_version,
        dataset_artifact_digest=artifact.artifact_digest,
        config=candidate_config,
        execution_overlay=snapshot.plan.baseline_config.execution_overlay,
    )
    candidate_resolved = ResolvedRunConfig.from_candidate_spec(candidate_spec)
    baseline_payload = snapshot.plan.baseline_config.to_payload()
    candidate_payload = candidate_resolved.to_payload()
    changed_paths = _top_level_changed_paths(baseline_payload, candidate_payload)
    if changed_paths != EXPECTED_CHANGED_PATHS:
        raise ValueError(f"candidate resolved delta drift: {changed_paths!r}")
    if baseline_payload.get("ppo_global_feature_names") != []:
        raise ValueError("baseline global feature roster drift")
    if candidate_payload.get("ppo_global_feature_names") != []:
        raise ValueError("candidate introduced an unregistered global feature roster")
    if candidate_payload.get("ppo_global_context") != PPO_GLOBAL_BTC_REGIME_CONTEXT:
        raise ValueError("candidate PPO global context drift")
    if candidate_payload.get("ppo_observation_schema") != "ppo_observation_v3_global_btc_regime":
        raise ValueError("candidate observation schema drift")
    if candidate_payload.get("schema_version") != "resolved_run_config_v3":
        raise ValueError("candidate resolved schema drift")

    plan_path = study_root / "plan.json"
    plan_bytes = plan_path.read_bytes()
    plan_payload = _load_json_object(plan_path)
    if canonical_json_bytes(plan_payload) != plan_bytes:
        raise ValueError("new Study plan is not canonical JSON")
    study_digest = content_digest(plan_payload)
    if study_digest != snapshot.plan.digest:
        raise ValueError("new Study digest does not verify")

    lock_path = study_root / ".mutation.lock"
    if lock_path.exists():
        lock_path.unlink()

    requested_baseline = baseline_config.to_json_payload()
    requested_candidate = candidate_config.to_json_payload()
    requested_delta = _top_level_changed_paths(requested_baseline, requested_candidate)
    if requested_delta != (("ppo_global_context",),):
        raise ValueError("raw candidate config changes more than the sealed global context")

    execution_plan: dict[str, object] = {
        "schema_version": "issue609_global_btc_development_evaluation_plan_v1",
        "issue_number": 609,
        "plan_run_id": plan_run_id,
        "source_baseline_run_id": SOURCE_BASELINE_RUN_ID,
        "source_baseline_artifact_id": SOURCE_BASELINE_ARTIFACT_ID,
        "source_baseline_artifact_name": SOURCE_BASELINE_ARTIFACT_NAME,
        "source_baseline_artifact_api_digest": SOURCE_BASELINE_ARTIFACT_DIGEST.removeprefix("sha256:"),
        "source_dataset_id": SOURCE_DATASET_ID,
        "source_dataset_artifact_digest": SOURCE_DATASET_ARTIFACT_DIGEST,
        "source_study_digest": SOURCE_STUDY_DIGEST,
        "source_baseline_evidence_fingerprint": SOURCE_BASELINE_EVIDENCE_FINGERPRINT,
        "source_baseline_metrics_inspected_for_plan": False,
        "factor_prereg_head": FACTOR_PREREG_HEAD,
        "factor_prereg_protocol_digest": FACTOR_PREREG_PROTOCOL_DIGEST,
        "implementation_head": IMPLEMENTATION_HEAD,
        "implementation_tree": IMPLEMENTATION_TREE,
        "implementation_index_digest": IMPLEMENTATION_INDEX_DIGEST,
        "implementation_seal_run_id": IMPLEMENTATION_SEAL_RUN_ID,
        "implementation_seal_artifact_id": IMPLEMENTATION_SEAL_ARTIFACT_ID,
        "implementation_seal_artifact_api_digest": IMPLEMENTATION_SEAL_ARTIFACT_DIGEST.removeprefix("sha256:"),
        "implementation_fresh_artifact_id": IMPLEMENTATION_FRESH_ARTIFACT_ID,
        "implementation_fresh_artifact_api_digest": IMPLEMENTATION_FRESH_ARTIFACT_DIGEST.removeprefix("sha256:"),
        "evaluation_prereg_head": EVALUATION_PREREG_HEAD,
        "evaluation_prereg_protocol_digest": EVALUATION_PREREG_PROTOCOL_DIGEST,
        "evaluation_prereg_seal_run_id": EVALUATION_PREREG_SEAL_RUN_ID,
        "evaluation_prereg_seal_artifact_id": EVALUATION_PREREG_SEAL_ARTIFACT_ID,
        "evaluation_prereg_seal_artifact_api_digest": EVALUATION_PREREG_SEAL_ARTIFACT_DIGEST.removeprefix("sha256:"),
        "evaluation_prereg_fresh_artifact_id": EVALUATION_PREREG_FRESH_ARTIFACT_ID,
        "evaluation_prereg_fresh_artifact_api_digest": EVALUATION_PREREG_FRESH_ARTIFACT_DIGEST.removeprefix("sha256:"),
        "execution_framework_head": EXECUTION_FRAMEWORK_HEAD,
        "execution_framework_tree": EXECUTION_FRAMEWORK_TREE,
        "execution_framework_exact_verify_run_id": EXECUTION_FRAMEWORK_EXACT_VERIFY_RUN_ID,
        "execution_framework_seal_run_id": EXECUTION_FRAMEWORK_SEAL_RUN_ID,
        "execution_framework_seal_artifact_id": EXECUTION_FRAMEWORK_SEAL_ARTIFACT_ID,
        "execution_framework_seal_artifact_api_digest": EXECUTION_FRAMEWORK_SEAL_ARTIFACT_DIGEST.removeprefix("sha256:"),
        "execution_framework_fresh_artifact_id": EXECUTION_FRAMEWORK_FRESH_ARTIFACT_ID,
        "execution_framework_fresh_artifact_api_digest": EXECUTION_FRAMEWORK_FRESH_ARTIFACT_DIGEST.removeprefix("sha256:"),
        "execution_framework_recovery_run_id": EXECUTION_FRAMEWORK_RECOVERY_RUN_ID,
        "execution_framework_recovery_artifact_id": EXECUTION_FRAMEWORK_RECOVERY_ARTIFACT_ID,
        "execution_framework_recovery_artifact_api_digest": EXECUTION_FRAMEWORK_RECOVERY_ARTIFACT_DIGEST.removeprefix("sha256:"),
        "new_study_digest": study_digest,
        "new_study_plan_sha256": hashlib.sha256(plan_bytes).hexdigest(),
        "new_study_implementation_digest": snapshot.plan.implementation_digest,
        "new_study_runtime_environment_digest": snapshot.plan.runtime_environment_digest,
        "new_study_allowed_factors": [ControlledFactor.FEATURE_SET.value],
        "new_study_max_experiments": 1,
        "baseline_requested_config_digest": content_digest(requested_baseline),
        "candidate_requested_config_digest": content_digest(requested_candidate),
        "baseline_resolved_config_digest": content_digest(baseline_payload),
        "candidate_resolved_config_digest": content_digest(candidate_payload),
        "expected_resolved_changed_paths": [list(path) for path in EXPECTED_CHANGED_PATHS],
        "controlled_factor": ControlledFactor.FEATURE_SET.value,
        "semantic_factor": "ppo_global_btc_regime_context",
        "candidate_observation_schema": "ppo_observation_v3_global_btc_regime",
        "ppo_seeds": list(EXPECTED_PPO_SEEDS),
        "ppo_total_timesteps": 100_000,
        "n_bootstrap": 2_000,
        "bootstrap_seed": 1_729,
        "development_economic_execution_authorized_after_plan_audit": True,
        "candidate_result_inspected": False,
        "final_test_access_authorized": False,
        "operational_eligibility_established": False,
        "production_eligible": False,
        "live_trading_authorized": False,
        "merge_authorized": False,
        "source_plan_baseline_config_digest": content_digest(source_plan["baseline_config"]),
    }
    execution_plan["content_digest"] = content_digest(execution_plan)
    (output_root / "evaluation-plan.json").write_bytes(canonical_json_bytes(execution_plan))
    print("PLAN_WRITTEN=true")
    print("ECONOMIC_EXECUTION_PERFORMED=false")
    print("CANDIDATE_RESULT_INSPECTED=false")


def self_check() -> None:
    baseline = {
        "schema_version": "resolved_run_config_v2",
        "ppo_observation_schema": "ppo_observation_v2",
        "ppo_global_feature_names": [],
    }
    candidate = {
        **baseline,
        "schema_version": "resolved_run_config_v3",
        "ppo_observation_schema": "ppo_observation_v3_global_btc_regime",
        "ppo_global_context": PPO_GLOBAL_BTC_REGIME_CONTEXT,
    }
    assert _top_level_changed_paths(baseline, candidate) == EXPECTED_CHANGED_PATHS
    assert EXPECTED_PPO_SEEDS == (0, 1, 2, 3, 4)
    assert EVALUATION_PREREG_PROTOCOL_DIGEST == (
        "e52a19b859500ea01e900a1ea7209bd545c03acd08435d1ff615b7539ec75d2b"
    )
    assert EXECUTION_FRAMEWORK_RECOVERY_RUN_ID == 35092562372


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--plan-run-id", type=int)
    args = parser.parse_args()
    if args.self_check:
        self_check()
        print("SELF_CHECK=PASS")
        return
    if args.source_root is None or args.output_root is None or args.plan_run_id is None:
        parser.error("plan mode requires --source-root, --output-root, and --plan-run-id")
    if args.plan_run_id <= 0:
        parser.error("--plan-run-id must be positive")
    build_plan(
        source_root=args.source_root,
        output_root=args.output_root,
        plan_run_id=args.plan_run_id,
    )


if __name__ == "__main__":
    main()
