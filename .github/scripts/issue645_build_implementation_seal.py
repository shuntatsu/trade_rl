#!/usr/bin/env python3
"""Build the result-blind Issue #645 corrected-accounting implementation authority."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

ISSUE_NUMBER = 645
BASE_HEAD = "c80652a126780580990bf46bb206155ba469e043"
CORRECTION_HEAD = "95a4831bfbd5dfac2b77371b0c99d6894edc4d3a"
IMPLEMENTATION_HEAD = "f93097e7747390dfc8dae1a866cf77351706c618"
VERIFICATION_RUN_ID = 35287091550
VERIFICATION_WORKFLOW_HEAD = "0acd3a8f3f5b7a5d6da9a7d9ef41d83d5a737b66"

PRODUCTION_PATHS = (
    "trade_rl/simulation/accounting.py",
    "trade_rl/simulation/execution.py",
    "trade_rl/simulation/liquidity.py",
    "trade_rl/simulation/orders/admission.py",
    "trade_rl/simulation/orders/model.py",
    "trade_rl/simulation/quantities.py",
    "trade_rl/simulation/stateful/symbol_fills.py",
)
SOURCE_EQUAL_SUPPORT_PATHS = (
    "tests/architecture/test_lean_simulation_layout.py",
    "tests/simulation/test_lot_accounting.py",
    "tests/simulation/test_stateful_execution_characterization.py",
)
LOCAL_SUPPORT_PATHS = ("guide/content/meta/code-map.json",)
UNCHANGED_DIRECTIONAL_PATHS = (
    "trade_rl/evaluation/directional_study.py",
    "trade_rl/evaluation/directional_selection.py",
    "trade_rl/evaluation/directional_candidates.py",
    "trade_rl/evaluation/directional.py",
)
EXPECTED_CHANGED_PATHS = tuple(
    sorted(PRODUCTION_PATHS + SOURCE_EQUAL_SUPPORT_PATHS + LOCAL_SUPPORT_PATHS)
)
EXPECTED_EXECUTOR_SOURCE_DIGEST = (
    "ea70e0094dcdd2cc98b72e6083767d6d6ea55d6c46321becbbe3e396685ff811"
)


def git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ("git", "-C", str(repo), *args),
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def blob(repo: Path, ref: str, path: str) -> str:
    return git(repo, "rev-parse", f"{ref}:{path}")


def canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def build(repo: Path) -> dict[str, object]:
    if git(repo, "rev-parse", "HEAD") != IMPLEMENTATION_HEAD:
        raise RuntimeError("implementation HEAD drift")

    changed = tuple(
        sorted(
            line
            for line in git(repo, "diff", "--name-only", BASE_HEAD, "HEAD").splitlines()
            if line
        )
    )
    if changed != EXPECTED_CHANGED_PATHS:
        raise RuntimeError(f"durable scope drift: {changed!r}")

    production_blobs: dict[str, str] = {}
    for path in PRODUCTION_PATHS:
        actual = blob(repo, "HEAD", path)
        expected = blob(repo, CORRECTION_HEAD, path)
        if actual != expected:
            raise RuntimeError(f"PR #643 production blob drift: {path}")
        production_blobs[path] = actual

    support_blobs: dict[str, str] = {}
    for path in SOURCE_EQUAL_SUPPORT_PATHS:
        actual = blob(repo, "HEAD", path)
        expected = blob(repo, CORRECTION_HEAD, path)
        if actual != expected:
            raise RuntimeError(f"PR #643 verification-support blob drift: {path}")
        support_blobs[path] = actual
    for path in LOCAL_SUPPORT_PATHS:
        support_blobs[path] = blob(repo, "HEAD", path)

    directional_blobs: dict[str, str] = {}
    for path in UNCHANGED_DIRECTIONAL_PATHS:
        actual = blob(repo, "HEAD", path)
        expected = blob(repo, BASE_HEAD, path)
        if actual != expected:
            raise RuntimeError(f"frozen directional semantic blob drift: {path}")
        directional_blobs[path] = actual

    code_map = json.loads((repo / "guide/content/meta/code-map.json").read_text())
    references = {
        item["symbol"]: item["source_sha256"] for item in code_map["code_references"]
    }
    if (
        references.get("trade_rl.simulation.execution.MarketExecutor")
        != EXPECTED_EXECUTOR_SOURCE_DIGEST
    ):
        raise RuntimeError("Guide MarketExecutor reviewed digest drift")

    return {
        "schema_version": "issue645_corrected_accounting_implementation_authority_v1",
        "issue_number": ISSUE_NUMBER,
        "factor": "pr643_exact_lot_accounting_correction_only",
        "base_head": BASE_HEAD,
        "correction_source_head": CORRECTION_HEAD,
        "implementation_head": IMPLEMENTATION_HEAD,
        "exact_verification": {
            "run_id": VERIFICATION_RUN_ID,
            "workflow_head": VERIFICATION_WORKFLOW_HEAD,
            "expected_conclusion": "success",
        },
        "changed_paths": list(changed),
        "production_blob_sha": production_blobs,
        "verification_support_blob_sha": support_blobs,
        "unchanged_directional_semantic_blob_sha": directional_blobs,
        "dataset_loaded": False,
        "model_fit_performed": False,
        "economic_replay_performed": False,
        "economic_result_inspected": False,
        "unused_data_accessed": False,
        "final_test_accessed": False,
        "production_eligible": False,
        "live_trading_authorized": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    authority = build(args.repo.resolve())
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = canonical_bytes(authority)
    (args.output / "manifest.json").write_bytes(manifest)
    seal = {
        "schema_version": "issue645_corrected_accounting_implementation_seal_v1",
        "manifest_sha256": hashlib.sha256(manifest).hexdigest(),
    }
    (args.output / "seal.json").write_bytes(canonical_bytes(seal))
    print(f"ISSUE645_IMPLEMENTATION_MANIFEST_SHA256={seal['manifest_sha256']}")
    print("DATASET_LOADED=false")
    print("MODEL_FIT_PERFORMED=false")
    print("ECONOMIC_REPLAY_PERFORMED=false")
    print("ECONOMIC_RESULT_INSPECTED=false")
    print("UNUSED_DATA_ACCESSED=false")
    print("FINAL_TEST_ACCESSED=false")
    print("PRODUCTION_ELIGIBLE=false")
    print("LIVE_TRADING_AUTHORIZED=false")


if __name__ == "__main__":
    main()
