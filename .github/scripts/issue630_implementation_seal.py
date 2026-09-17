from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import urllib.request
from pathlib import Path
from typing import Any

TARGET_SHA = "e0470b4e04fc16c14391e4cc327e27e1d612edc7"
BASE_SHA = "d18434799651cfc6c07e0840c40600dcf1dfa763"
VERIFICATION_RUN_ID = 35212581096
PR_NUMBER = 633
PROTOCOL_HEAD = "65eb3c90e0a5fe1cea952279c28024160838da08"
PROTOCOL_DIGEST = "c4942c190e507b2d437bd00646fd7ed09e2be5cff9b30fba6d1774b6f2af9c88"
PROTOCOL_SEAL_RUN_ID = 35211715033
PROTOCOL_PRIMARY_ARTIFACT_ID = 10491893357
PROTOCOL_PRIMARY_API_DIGEST = "sha256:0e3e1dc742c00d3a1625aa9ee8681aba86a676946083afd08eda783642dd6841"
PROTOCOL_FRESH_ARTIFACT_ID = 10492645772
PROTOCOL_FRESH_API_DIGEST = "sha256:ca5fc01ed41cad6fa2f2dc2cf9ef11e4fe7cd3b5b6f7d8bc86bea1eb86842f56"
PROTOCOL_SEAL_SHA256 = "b76ae3ac83109c2b6349330b0b9b0e70522320ae9bcdf1dd2a794f93668bd903"

DURABLE_FILES = [
    "tests/architecture/test_runs_capability_facade.py",
    "tests/evaluation/experiments/bootstrap/test_ridge_shared_cash_evaluation.py",
    "tests/evaluation/experiments/bootstrap/test_ridge_shared_cash_evaluation_falsification.py",
    "tests/evaluation/experiments/bootstrap/test_ridge_shared_cash_prereg.py",
    "tests/evaluation/experiments/bootstrap/test_ridge_shared_cash_prereg_execution_leverage.py",
    "trade_rl/evaluation/experiments/bootstrap/ridge_shared_cash_evaluation.py",
    "trade_rl/evaluation/experiments/bootstrap/ridge_shared_cash_prereg.py",
    "trade_rl/evaluation/runs/__init__.py",
    "trade_rl/evaluation/runs/execute.py",
    "trade_rl/strategies/forecasts/ridge_economic_gate.py",
]
FALSE_BOUNDARIES = (
    "shared_cash_economic_execution_performed",
    "shared_cash_economic_result_inspected",
    "unused_data_accessed",
    "final_test_accessed",
    "production_eligible",
    "live_trading_authorized",
    "merge_authorized",
)


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True).strip()


def _canonical(payload: object) -> bytes:
    return (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _api(path: str) -> dict[str, Any]:
    token = os.environ["GH_TOKEN"]
    request = urllib.request.Request(
        f"https://api.github.com/repos/{os.environ['GITHUB_REPOSITORY']}/{path}",
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    with urllib.request.urlopen(request) as response:
        value = json.load(response)
    if not isinstance(value, dict):
        raise SystemExit(f"GitHub API object expected for {path}")
    return value


def _normalize_digest(value: str) -> str:
    return value if value.startswith("sha256:") else f"sha256:{value}"


def build(output_dir: Path) -> None:
    from trade_rl.evaluation.experiments.bootstrap.ridge_shared_cash_evaluation import (
        canonical_ridge_shared_cash_evaluation_spec,
    )

    if _git("rev-parse", "HEAD") != TARGET_SHA:
        raise SystemExit("target HEAD drift")
    subprocess.run(["git", "merge-base", "--is-ancestor", BASE_SHA, TARGET_SHA], check=True)

    spec = canonical_ridge_shared_cash_evaluation_spec()
    expected = {
        "protocol_head": PROTOCOL_HEAD,
        "protocol_digest": PROTOCOL_DIGEST,
        "protocol_seal_run_id": PROTOCOL_SEAL_RUN_ID,
        "protocol_primary_artifact_id": PROTOCOL_PRIMARY_ARTIFACT_ID,
        "protocol_fresh_artifact_id": PROTOCOL_FRESH_ARTIFACT_ID,
        "protocol_seal_sha256": PROTOCOL_SEAL_SHA256,
        "execution_max_leverage": 1.0,
    }
    for name, value in expected.items():
        if getattr(spec, name) != value:
            raise SystemExit(f"spec authority drift: {name}")
    for name in (
        "unused_data_accessed",
        "final_test_accessed",
        "final_test_authorized",
        "operational_eligibility_established",
        "production_eligible",
        "live_trading_authorized",
        "merge_authorized",
    ):
        if getattr(spec, name) is not False:
            raise SystemExit(f"research boundary drift: {name}")

    actual_files = subprocess.check_output(
        ["git", "diff", "--name-only", BASE_SHA, TARGET_SHA], text=True
    ).splitlines()
    if actual_files != DURABLE_FILES:
        raise SystemExit(f"durable diff drift: {actual_files!r}")

    payload = {
        "schema_version": "issue630_shared_cash_evaluator_implementation_v1",
        "issue_number": 630,
        "pull_request": PR_NUMBER,
        "implementation_head": TARGET_SHA,
        "implementation_tree_sha": _git("rev-parse", f"{TARGET_SHA}^{{tree}}"),
        "carrier_base_head": BASE_SHA,
        "exact_verification_run_id": VERIFICATION_RUN_ID,
        "protocol_v3": {
            "head": PROTOCOL_HEAD,
            "digest": PROTOCOL_DIGEST,
            "seal_run_id": PROTOCOL_SEAL_RUN_ID,
            "primary_artifact_id": PROTOCOL_PRIMARY_ARTIFACT_ID,
            "primary_api_digest": PROTOCOL_PRIMARY_API_DIGEST,
            "fresh_artifact_id": PROTOCOL_FRESH_ARTIFACT_ID,
            "fresh_api_digest": PROTOCOL_FRESH_API_DIGEST,
            "seal_sha256": PROTOCOL_SEAL_SHA256,
            "execution_max_leverage": 1.0,
        },
        "durable_files": DURABLE_FILES,
        "blob_shas": {path: _git("rev-parse", f"{TARGET_SHA}:{path}") for path in DURABLE_FILES},
        "result_blind": True,
        "shared_cash_economic_execution_performed": False,
        "shared_cash_economic_result_inspected": False,
        "unused_data_accessed": False,
        "final_test_accessed": False,
        "production_eligible": False,
        "live_trading_authorized": False,
        "merge_authorized": False,
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    implementation = _canonical(payload)
    (output_dir / "implementation.json").write_bytes(implementation)
    implementation_sha = hashlib.sha256(implementation).hexdigest()
    seal_payload = {
        "schema_version": "issue630_shared_cash_evaluator_seal_v1",
        "implementation_head": TARGET_SHA,
        "implementation_sha256": implementation_sha,
        "exact_verification_run_id": VERIFICATION_RUN_ID,
        "result_blind": True,
        "shared_cash_economic_execution_performed": False,
        "shared_cash_economic_result_inspected": False,
        "unused_data_accessed": False,
        "final_test_accessed": False,
        "production_eligible": False,
        "live_trading_authorized": False,
        "merge_authorized": False,
    }
    seal = _canonical(seal_payload)
    (output_dir / "seal.json").write_bytes(seal)
    seal_sha = hashlib.sha256(seal).hexdigest()
    github_output = os.environ.get("GITHUB_OUTPUT")
    if github_output:
        with open(github_output, "a", encoding="utf-8") as handle:
            handle.write(f"implementation_sha256={implementation_sha}\n")
            handle.write(f"seal_sha256={seal_sha}\n")
    print(f"IMPLEMENTATION_SHA256={implementation_sha}")
    print(f"SEAL_SHA256={seal_sha}")
    print("SHARED_CASH_ECONOMIC_EXECUTION_PERFORMED=false")
    print("SHARED_CASH_ECONOMIC_RESULT_INSPECTED=false")


def audit(primary_dir: Path, fresh_dir: Path, closure_path: Path) -> None:
    primary_impl = (primary_dir / "implementation.json").read_bytes()
    fresh_impl = (fresh_dir / "implementation.json").read_bytes()
    primary_seal = (primary_dir / "seal.json").read_bytes()
    fresh_seal = (fresh_dir / "seal.json").read_bytes()
    if primary_impl != fresh_impl or primary_seal != fresh_seal:
        raise SystemExit("primary/fresh implementation seal bytes differ")
    implementation_sha = hashlib.sha256(primary_impl).hexdigest()
    seal_sha = hashlib.sha256(primary_seal).hexdigest()
    if implementation_sha != os.environ["IMPLEMENTATION_SHA256"]:
        raise SystemExit("implementation SHA output drift")
    if seal_sha != os.environ["SEAL_SHA256"]:
        raise SystemExit("seal SHA output drift")

    if _api(f"actions/runs/{VERIFICATION_RUN_ID}").get("conclusion") != "success":
        raise SystemExit("exact-head verification run drift")
    if _api(f"actions/runs/{PROTOCOL_SEAL_RUN_ID}").get("conclusion") != "success":
        raise SystemExit("protocol v3 seal run drift")
    pr = _api(f"pulls/{PR_NUMBER}")
    if (
        pr.get("state") != "open"
        or pr.get("draft") is not True
        or (pr.get("head") or {}).get("sha") != TARGET_SHA
        or (pr.get("base") or {}).get("sha") != BASE_SHA
    ):
        raise SystemExit("PR 633 Draft authority drift")

    for artifact_id, expected_digest in (
        (PROTOCOL_PRIMARY_ARTIFACT_ID, PROTOCOL_PRIMARY_API_DIGEST),
        (PROTOCOL_FRESH_ARTIFACT_ID, PROTOCOL_FRESH_API_DIGEST),
    ):
        metadata = _api(f"actions/artifacts/{artifact_id}")
        if (
            metadata.get("id") != artifact_id
            or metadata.get("digest") != expected_digest
            or (metadata.get("workflow_run") or {}).get("id") != PROTOCOL_SEAL_RUN_ID
            or metadata.get("expired") is not False
        ):
            raise SystemExit(f"protocol artifact authority drift: {artifact_id}")

    run_id = int(os.environ["GITHUB_RUN_ID"])
    seal_artifacts: list[tuple[str, int, str]] = []
    for name in ("PRIMARY", "FRESH"):
        artifact_id = int(os.environ[f"{name}_ARTIFACT_ID"])
        digest = _normalize_digest(os.environ[f"{name}_UPLOAD_DIGEST"])
        metadata = _api(f"actions/artifacts/{artifact_id}")
        if (
            metadata.get("id") != artifact_id
            or metadata.get("digest") != digest
            or (metadata.get("workflow_run") or {}).get("id") != run_id
            or metadata.get("expired") is not False
        ):
            raise SystemExit(f"{name.lower()} seal artifact authority drift")
        seal_artifacts.append((name.lower(), artifact_id, digest))

    implementation = json.loads(primary_impl)
    seal = json.loads(primary_seal)
    if implementation.get("implementation_head") != TARGET_SHA:
        raise SystemExit("sealed implementation HEAD drift")
    if seal.get("implementation_sha256") != implementation_sha:
        raise SystemExit("seal implementation SHA drift")
    for field in FALSE_BOUNDARIES:
        if implementation.get(field) is not False or seal.get(field) is not False:
            raise SystemExit(f"research boundary drift: {field}")

    closure = {
        "schema_version": "issue630_shared_cash_evaluator_seal_closure_v1",
        "seal_run_id": run_id,
        "implementation_head": TARGET_SHA,
        "implementation_sha256": implementation_sha,
        "seal_sha256": seal_sha,
        "primary_artifact_id": seal_artifacts[0][1],
        "primary_api_digest": seal_artifacts[0][2],
        "fresh_artifact_id": seal_artifacts[1][1],
        "fresh_api_digest": seal_artifacts[1][2],
        "primary_fresh_byte_equal": True,
        "result_blind": True,
        "shared_cash_economic_execution_performed": False,
        "shared_cash_economic_result_inspected": False,
        "unused_data_accessed": False,
        "final_test_accessed": False,
        "production_eligible": False,
        "live_trading_authorized": False,
        "merge_authorized": False,
    }
    closure_path.write_bytes(_canonical(closure))
    print("ISSUE630_IMPLEMENTATION_SEAL_CLOSURE=PASS")
    print("PRIMARY_FRESH_BYTE_EQUAL=true")
    print("SHARED_CASH_ECONOMIC_EXECUTION_PERFORMED=false")


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    build_parser = sub.add_parser("build")
    build_parser.add_argument("--output-dir", type=Path, required=True)
    audit_parser = sub.add_parser("audit")
    audit_parser.add_argument("--primary-dir", type=Path, required=True)
    audit_parser.add_argument("--fresh-dir", type=Path, required=True)
    audit_parser.add_argument("--closure", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "build":
        build(args.output_dir)
    else:
        audit(args.primary_dir, args.fresh_dir, args.closure)


if __name__ == "__main__":
    main()
