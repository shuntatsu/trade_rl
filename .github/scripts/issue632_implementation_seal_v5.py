"""Canonical result-blind implementation authority builder for Issue 632 v5."""

from __future__ import annotations

import argparse
import hashlib
import subprocess
from dataclasses import MISSING, fields
from pathlib import Path

from trade_rl.artifacts.canonical import canonical_json_bytes
from trade_rl.evaluation.experiments.ppo_interleaved_evaluation import (
    PPOReturnPathEvidence,
    PPOSeedEvidence,
    PPOSymbolEvidence,
    canonical_ppo_interleaved_evaluator_spec,
)

TARGET_HEAD = "97127952af3d01b71def695a4c8043eb301b2a33"
MAIN_HEAD = "c4fc52777ad0ea894d0d8f3a01337bcbbd5a26a8"
VERIFICATION_RUN_ID = 35_227_295_244
VERIFICATION_HARNESS_HEAD = "c48f416f11cf9a00f28be3f13343b29ff24e5427"
PR_NUMBER = 635
PROTOCOL_HEAD = "f1187dacae78e679a322cc53cbf03f3371f457b1"
PROTOCOL_MODULE_BLOB = "62a71b1ca8b7d22fdfc14282508754c44d96091a"
PROTOCOL_DIGEST = "a34aee66bf3f51ce02675b955f835c292b023770aab841f25b46815f9b399c2c"
PROTOCOL_SEAL_RUN_ID = 35_205_354_305
PROTOCOL_PRIMARY_ARTIFACT_ID = 10_489_866_637
PROTOCOL_PRIMARY_API_DIGEST = "4ca30f9884b642f32443bbb143b15e4fecbbc7a25d262e89d63649acbef00387"
PROTOCOL_FRESH_ARTIFACT_ID = 10_489_661_856
PROTOCOL_FRESH_API_DIGEST = "2c9d9a71db2315dc66237c9ba4b975f0bdf5a5234ad87cf8bbd05b947867afbb"
PROTOCOL_SEAL_JSON_SHA256 = "3e55fdcf772bbf4063d537e3f1912a5d1588611ce51e586180c8d1be13a186fe"
CAPABILITY_HEAD = "cddef3532dd582d0f1066a61006f64265f0cabbe"
SUPERSEDED_V4_RUN_ID = 35_227_707_295
SUPERSEDED_V4_FRESH_ARTIFACT_ID = 10_499_463_701

DURABLE_PATHS = (
    "tests/architecture/test_runs_capability_facade.py",
    "tests/evaluation/experiments/test_ppo_interleaved_direct_constructor_hardening.py",
    "tests/evaluation/experiments/test_ppo_interleaved_evaluation.py",
    "tests/evaluation/experiments/test_ppo_interleaved_evaluation_prereg.py",
    "tests/evaluation/experiments/test_ppo_interleaved_evaluation_prereg_hardening.py",
    "tests/evaluation/experiments/test_ppo_interleaved_evidence_codec.py",
    "trade_rl/evaluation/experiments/ppo_interleaved_evaluation.py",
    "trade_rl/evaluation/experiments/ppo_interleaved_evaluation_prereg.py",
    "trade_rl/evaluation/experiments/ppo_interleaved_evidence_codec.py",
    "trade_rl/evaluation/runs/__init__.py",
    "trade_rl/evaluation/runs/execute.py",
)

_FALSE_BOUNDARIES = (
    "baseline_training_authorized",
    "candidate_training_authorized",
    "economic_result_inspected",
    "final_test_accessed",
    "shared_cash_profitability_established",
    "production_eligible",
    "live_trading_authorized",
    "merge_authorized",
)


def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _schema_default(cls: type[object]) -> str:
    selected = [item for item in fields(cls) if item.name == "schema_version"]
    if len(selected) != 1 or selected[0].default is MISSING:
        raise ValueError(f"{cls.__name__} has no fixed schema_version")
    value = selected[0].default
    if not isinstance(value, str) or not value:
        raise ValueError(f"{cls.__name__} schema_version is invalid")
    return value


def build_authority(
    *, target_root: Path, authority_run_id: int, harness_head: str
) -> tuple[bytes, bytes]:
    if _git(target_root, "rev-parse", "HEAD") != TARGET_HEAD:
        raise ValueError("Issue 632 target head drifted")
    subprocess.run(
        ["git", "-C", str(target_root), "merge-base", "--is-ancestor", MAIN_HEAD, TARGET_HEAD],
        check=True,
    )
    if _git(target_root, "status", "--porcelain"):
        raise ValueError("Issue 632 target worktree is not clean")
    actual_paths = tuple(
        subprocess.check_output(
            ["git", "-C", str(target_root), "diff", "--name-only", f"{MAIN_HEAD}...{TARGET_HEAD}"],
            text=True,
        ).splitlines()
    )
    if actual_paths != DURABLE_PATHS:
        raise ValueError(f"Issue 632 durable scope drifted: {actual_paths!r}")

    spec = canonical_ppo_interleaved_evaluator_spec()
    expected = {
        "protocol_head": PROTOCOL_HEAD,
        "protocol_module_blob": PROTOCOL_MODULE_BLOB,
        "protocol_digest": PROTOCOL_DIGEST,
        "protocol_seal_run_id": PROTOCOL_SEAL_RUN_ID,
        "protocol_primary_artifact_id": PROTOCOL_PRIMARY_ARTIFACT_ID,
        "protocol_primary_artifact_digest": PROTOCOL_PRIMARY_API_DIGEST,
        "protocol_fresh_artifact_id": PROTOCOL_FRESH_ARTIFACT_ID,
        "protocol_fresh_artifact_digest": PROTOCOL_FRESH_API_DIGEST,
        "protocol_seal_json_sha256": PROTOCOL_SEAL_JSON_SHA256,
        "implementation_head": CAPABILITY_HEAD,
        "slippage_std": 0.0,
    }
    for name, value in expected.items():
        if getattr(spec, name) != value:
            raise ValueError(f"sealed evaluator authority drifted: {name}")
    spec_payload = spec.to_payload()
    for field in _FALSE_BOUNDARIES:
        if spec_payload.get(field) is not False:
            raise ValueError(f"result-blind boundary drifted: {field}")

    evidence_schemas = {
        "return_path": _schema_default(PPOReturnPathEvidence),
        "symbol": _schema_default(PPOSymbolEvidence),
        "seed": _schema_default(PPOSeedEvidence),
    }
    if evidence_schemas != {
        "return_path": "ppo_interleaved_return_path_evidence_v1",
        "symbol": "ppo_interleaved_symbol_evidence_v1",
        "seed": "ppo_interleaved_seed_evidence_v2",
    }:
        raise ValueError(f"evidence schema drifted: {evidence_schemas!r}")

    durable_files = [
        {"path": path, "blob_sha": _git(target_root, "rev-parse", f"{TARGET_HEAD}:{path}")}
        for path in DURABLE_PATHS
    ]
    implementation = {
        "schema_version": "issue632_ppo_interleaved_evaluator_authority_v5",
        "issue_number": 632,
        "pull_request": PR_NUMBER,
        "target_head": TARGET_HEAD,
        "target_tree_sha": _git(target_root, "rev-parse", f"{TARGET_HEAD}^{{tree}}"),
        "main_head": MAIN_HEAD,
        "verification_run_id": VERIFICATION_RUN_ID,
        "verification_harness_head": VERIFICATION_HARNESS_HEAD,
        "seal_harness_head": harness_head,
        "protocol_head": PROTOCOL_HEAD,
        "protocol_module_blob": PROTOCOL_MODULE_BLOB,
        "protocol_digest": PROTOCOL_DIGEST,
        "protocol_seal_run_id": PROTOCOL_SEAL_RUN_ID,
        "protocol_primary_artifact_id": PROTOCOL_PRIMARY_ARTIFACT_ID,
        "protocol_primary_artifact_digest": PROTOCOL_PRIMARY_API_DIGEST,
        "protocol_fresh_artifact_id": PROTOCOL_FRESH_ARTIFACT_ID,
        "protocol_fresh_artifact_digest": PROTOCOL_FRESH_API_DIGEST,
        "protocol_seal_json_sha256": PROTOCOL_SEAL_JSON_SHA256,
        "capability_head": CAPABILITY_HEAD,
        "evaluator_spec": spec_payload,
        "evaluator_spec_digest": spec.digest,
        "evidence_schemas": evidence_schemas,
        "evidence_hardening": {
            "raw_return_bool_alias_rejected": True,
            "common_strategy_period_identity_required": True,
            "slippage_identity_persisted": True,
        },
        "superseded_v4": {
            "run_id": SUPERSEDED_V4_RUN_ID,
            "primary_artifact_published": False,
            "fresh_artifact_id": SUPERSEDED_V4_FRESH_ARTIFACT_ID,
            "accepted_as_authority": False,
        },
        "durable_files": durable_files,
        "real_dataset_loaded": False,
        "ppo_training_performed": False,
        "economic_evaluation_performed": False,
        "economic_result_inspected": False,
        "unused_data_accessed": False,
        "final_test_accessed": False,
        "shared_cash_profitability_established": False,
        "production_eligible": False,
        "live_trading_authorized": False,
        "merge_authorized": False,
    }
    implementation_bytes = canonical_json_bytes(implementation)
    implementation_sha256 = _sha256(implementation_bytes)
    seal = {
        "schema_version": "issue632_ppo_interleaved_evaluator_seal_v5",
        "issue_number": 632,
        "authority_run_id": authority_run_id,
        "target_head": TARGET_HEAD,
        "main_head": MAIN_HEAD,
        "verification_run_id": VERIFICATION_RUN_ID,
        "verification_harness_head": VERIFICATION_HARNESS_HEAD,
        "seal_harness_head": harness_head,
        "implementation_sha256": implementation_sha256,
        "evaluator_spec_digest": spec.digest,
        "protocol_digest": PROTOCOL_DIGEST,
        "protocol_seal_run_id": PROTOCOL_SEAL_RUN_ID,
        "superseded_v4_run_id": SUPERSEDED_V4_RUN_ID,
        "real_dataset_loaded": False,
        "ppo_training_performed": False,
        "economic_evaluation_performed": False,
        "economic_result_inspected": False,
        "unused_data_accessed": False,
        "final_test_accessed": False,
        "shared_cash_profitability_established": False,
        "production_eligible": False,
        "live_trading_authorized": False,
        "merge_authorized": False,
    }
    seal_bytes = canonical_json_bytes(seal)
    return implementation_bytes, seal_bytes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--authority-run-id", type=int, required=True)
    parser.add_argument("--harness-head", required=True)
    args = parser.parse_args()
    if args.authority_run_id <= 0:
        raise SystemExit("authority run id must be positive")
    if len(args.harness_head) != 40:
        raise SystemExit("harness head must be a full Git SHA")
    implementation_bytes, seal_bytes = build_authority(
        target_root=args.target_root.resolve(),
        authority_run_id=args.authority_run_id,
        harness_head=args.harness_head,
    )
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "implementation.json").write_bytes(implementation_bytes)
    (args.output_dir / "seal.json").write_bytes(seal_bytes)
    print(f"IMPLEMENTATION_SHA256={_sha256(implementation_bytes)}")
    print(f"SEAL_SHA256={_sha256(seal_bytes)}")
    print("REAL_DATASET_LOADED=false")
    print("PPO_TRAINING_PERFORMED=false")
    print("ECONOMIC_EVALUATION_PERFORMED=false")
    print("ECONOMIC_RESULT_INSPECTED=false")
    print("FINAL_TEST_ACCESSED=false")


if __name__ == "__main__":
    main()
