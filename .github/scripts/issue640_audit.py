"""Fresh no-refit audit for the one-shot Issue 640 development screen."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import urllib.request
from pathlib import Path
from typing import Any

from trade_rl.artifacts.canonical import canonical_json_bytes
from trade_rl.artifacts.hashing import content_digest
from trade_rl.evaluation.directional_candidates import ARMS
from trade_rl.evaluation.directional_selection import select_development_candidates

ISSUE = 640
SOURCE_SHA = "d4f1948ffb6bca44620773c2f67fc2f98b681d66"
SOURCE_CI_RUN = 35_231_705_047
PRODUCER_ARTIFACT_ID = 10_331_899_302
RECOVERY_ARTIFACT_ID = 10_332_575_500


def _json_bytes(path: Path) -> tuple[bytes, dict[str, Any]]:
    raw = path.read_bytes()
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    if canonical_json_bytes(value) != raw:
        raise ValueError(f"canonical JSON required: {path}")
    return raw, value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _api_artifact_names(*, repository: str, run_id: int, token: str) -> list[str]:
    request = urllib.request.Request(
        f"https://api.github.com/repos/{repository}/actions/runs/{run_id}/artifacts?per_page=100",
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        payload = json.load(response)
    if not isinstance(payload, dict) or not isinstance(payload.get("artifacts"), list):
        raise ValueError("workflow Artifact listing is unavailable")
    result: list[str] = []
    for item in payload["artifacts"]:
        if not isinstance(item, dict) or not isinstance(item.get("name"), str):
            raise ValueError("workflow Artifact entry is invalid")
        result.append(item["name"])
    return result


def audit(
    *,
    protocol_root: Path,
    arms_root: Path,
    summary_root: Path,
    output: Path,
    repository: str,
    run_id: int,
    token: str,
) -> dict[str, Any]:
    protocol_raw, protocol = _json_bytes(protocol_root / "protocol.json")
    protocol_digest = content_digest(protocol)
    _, protocol_digest_record = _json_bytes(protocol_root / "protocol.digest.json")
    if protocol_digest_record != {"digest": protocol_digest}:
        raise ValueError("protocol digest mismatch")
    _, authority = _json_bytes(protocol_root / "authority.json")
    if (
        authority.get("issue_number") != ISSUE
        or authority.get("source_head") != SOURCE_SHA
        or authority.get("source_ci_run") != SOURCE_CI_RUN
        or authority.get("producer_artifact_id") != PRODUCER_ARTIFACT_ID
        or authority.get("recovery_artifact_id") != RECOVERY_ARTIFACT_ID
        or authority.get("workflow_run_id") != run_id
        or authority.get("workflow_run_attempt") != 1
        or authority.get("economic_arm_started") is not False
        or authority.get("unused_data_accessed") is not False
        or authority.get("production_eligible") is not False
        or authority.get("live_trading_authorized") is not False
    ):
        raise ValueError("execution authority drifted")
    if tuple(protocol.get("arms", ())) != tuple(ARMS):
        raise ValueError("arm roster drifted")
    if (
        protocol.get("ppo_timesteps") != 262_144
        or protocol.get("ppo_training_layout") != "sequential"
        or protocol.get("maximum_drawdown") != 0.2
        or protocol.get("deleveraging_start") != 0.1
        or protocol.get("per_symbol_gross") != 0.1
        or protocol.get("account_gross") != 0.5
        or protocol.get("unused_data_accessed") is not False
        or protocol.get("production_eligible") is not False
    ):
        raise ValueError("frozen economic protocol drifted")

    results: dict[str, dict[str, Any]] = {}
    failed_arms: list[str] = []
    observed_exit_codes: dict[str, int] = {}
    for arm in ARMS:
        package = arms_root / f"issue640-directional-arm-{arm}-v1"
        _, attempt = _json_bytes(package / "attempt.json")
        exit_code = attempt.get("exit_code")
        if isinstance(exit_code, bool) or not isinstance(exit_code, int):
            raise ValueError(f"invalid exit code evidence: {arm}")
        observed_exit_codes[arm] = exit_code
        if (
            attempt.get("issue_number") != ISSUE
            or attempt.get("arm") != arm
            or attempt.get("source_head") != SOURCE_SHA
            or attempt.get("workflow_run_id") != run_id
            or attempt.get("workflow_run_attempt") != 1
            or attempt.get("unused_data_accessed") is not False
            or attempt.get("production_eligible") is not False
        ):
            raise ValueError(f"attempt identity drifted: {arm}")
        result_path = package / "result.json"
        failed_path = package / "failed.json"
        if exit_code != 0:
            if result_path.exists():
                raise ValueError(f"failed arm published a result: {arm}")
            failed_arms.append(arm)
            continue
        if not attempt.get("started") or not attempt.get("result_published"):
            raise ValueError(f"successful arm lacks started/result evidence: {arm}")
        if failed_path.exists() or not result_path.is_file():
            raise ValueError(f"successful arm has contradictory evidence: {arm}")
        result_raw, result = _json_bytes(result_path)
        _, digest_record = _json_bytes(package / "result.sha256.json")
        if digest_record != {"sha256": hashlib.sha256(result_raw).hexdigest()}:
            raise ValueError(f"result digest mismatch: {arm}")
        if (
            result.get("arm") != arm
            or result.get("protocol_digest") != protocol_digest
            or result.get("provenance") != protocol.get("provenance")
        ):
            raise ValueError(f"result protocol identity drifted: {arm}")
        model_hashes = result.get("model_sha256")
        if not isinstance(model_hashes, dict):
            raise ValueError(f"model digest map is missing: {arm}")
        for name, digest in model_hashes.items():
            if (
                not isinstance(name, str)
                or Path(name).name != name
                or not isinstance(digest, str)
                or _sha256(package / name) != digest
            ):
                raise ValueError(f"model digest mismatch: {arm}/{name}")
        results[arm] = result

    _, execution = _json_bytes(summary_root / "execution.json")
    if execution.get("arm_exit_codes") != observed_exit_codes:
        raise ValueError("run summary exit-code roster drifted")
    complete = len(results) == len(ARMS) and not failed_arms
    if execution.get("all_arm_commands_succeeded") is not complete:
        raise ValueError("run summary completeness drifted")

    selection_path = summary_root / "selection.json"
    selection_verified = False
    winner: str | None = None
    decision: str | None = None
    if complete:
        if not selection_path.is_file() or execution.get("selection_published") is not True:
            raise ValueError("complete study is missing selection")
        expected = select_development_candidates(results)
        expected["protocol_digest"] = protocol_digest
        expected["result_sha256"] = {
            arm: _sha256(arms_root / f"issue640-directional-arm-{arm}-v1" / "result.json")
            for arm in ARMS
        }
        if canonical_json_bytes(expected) != selection_path.read_bytes():
            raise ValueError("selection differs from independent reconstruction")
        selection_verified = True
        raw_winner = expected.get("winner")
        winner = raw_winner if isinstance(raw_winner, str) else None
        raw_decision = expected.get("decision")
        decision = raw_decision if isinstance(raw_decision, str) else None
    elif selection_path.exists() or execution.get("selection_published") is not False:
        raise ValueError("incomplete study published a selection")

    names = _api_artifact_names(repository=repository, run_id=run_id, token=token)
    expected_names = [
        "issue640-directional-protocol-v1",
        "issue640-directional-run-summary-v1",
        *(f"issue640-directional-arm-{arm}-v1" for arm in ARMS),
    ]
    for name in expected_names:
        if names.count(name) != 1:
            raise ValueError(f"workflow Artifact slot count mismatch: {name}")

    payload = {
        "schema_version": "issue640_directional_fresh_audit_v1",
        "issue_number": ISSUE,
        "workflow_run_id": run_id,
        "source_head": SOURCE_SHA,
        "protocol_sha256": hashlib.sha256(protocol_raw).hexdigest(),
        "protocol_digest": protocol_digest,
        "complete": complete,
        "failed_arms": failed_arms,
        "selection_verified": selection_verified,
        "decision": decision,
        "winner": winner,
        "unused_data_accessed": False,
        "production_eligible": False,
        "live_trading_authorized": False,
        "model_refit_performed": False,
    }
    output.mkdir(parents=True, exist_ok=False)
    (output / "audit.json").write_bytes(canonical_json_bytes(payload))
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--arms", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repository = os.environ.get("GITHUB_REPOSITORY")
    token = os.environ.get("GH_TOKEN")
    run_id = os.environ.get("GITHUB_RUN_ID")
    if not repository or not token or not run_id or not run_id.isdigit():
        raise SystemExit("GitHub workflow authority environment is required")
    payload = audit(
        protocol_root=args.protocol,
        arms_root=args.arms,
        summary_root=args.summary,
        output=args.output,
        repository=repository,
        run_id=int(run_id),
        token=token,
    )
    print(f"ISSUE640_COMPLETE={str(payload['complete']).lower()}")
    print(f"ISSUE640_SELECTION_VERIFIED={str(payload['selection_verified']).lower()}")
    print("ISSUE640_MODEL_REFIT_PERFORMED=false")


if __name__ == "__main__":
    main()
