"""Independent no-refit audit for the official Issue #640 directional study."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import urllib.parse
import urllib.request
from collections.abc import Mapping
from pathlib import Path
from types import ModuleType
from typing import Any

from trade_rl.artifacts import canonical_json_bytes, content_digest
from trade_rl.evaluation.directional_candidates import ARMS
from trade_rl.evaluation.directional_selection import select_development_candidates

ISSUE_NUMBER = 640
TARGET_SHA = "c80652a126780580990bf46bb206155ba469e043"
ACTIVATION_NAME = "issue640-directional-activation-v1"
PROTOCOL_NAME = "issue640-directional-protocol-v1"
SUMMARY_NAME = "issue640-directional-run-summary-v1"
OFFICIAL_REF = "run/issue640-directional-study-v1"
OFFICIAL_WORKFLOW_REF = (
    "shuntatsu/trade_rl/.github/workflows/issue640-directional-study-v1.yml"
    "@refs/heads/run/issue640-directional-study-v1"
)


def _canonical_object(path: Path) -> tuple[bytes, dict[str, Any]]:
    raw = path.read_bytes()
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    if canonical_json_bytes(value) != raw:
        raise ValueError(f"canonical JSON required: {path}")
    return raw, value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _api_json(repository: str, token: str, path: str) -> dict[str, Any]:
    request = urllib.request.Request(
        f"https://api.github.com/repos/{repository}/{path}",
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        payload = json.load(response)
    if not isinstance(payload, dict):
        raise ValueError(f"GitHub API object required: {path}")
    return payload


def _load_activation_helper(target: Path) -> ModuleType:
    path = target / ".github" / "scripts" / "issue640_directional_activation.py"
    spec = importlib.util.spec_from_file_location("issue640_directional_activation", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Issue 640 activation helper cannot be loaded")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _artifact_names(repository: str, token: str, run_id: int) -> list[str]:
    payload = _api_json(repository, token, f"actions/runs/{run_id}/artifacts?per_page=100")
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, list):
        raise ValueError("workflow artifact listing is unavailable")
    names: list[str] = []
    for item in artifacts:
        if not isinstance(item, Mapping) or not isinstance(item.get("name"), str):
            raise ValueError("invalid workflow artifact entry")
        names.append(item["name"])
    return names


def audit(
    *,
    target: Path,
    protocol_root: Path,
    activation_root: Path,
    arms_root: Path,
    summary_root: Path,
    output: Path,
    repository: str,
    token: str,
    run_id: int,
) -> dict[str, Any]:
    protocol_raw, protocol = _canonical_object(protocol_root / "protocol.json")
    _, digest_record = _canonical_object(protocol_root / "protocol.digest.json")
    protocol_digest = content_digest(protocol)
    if digest_record != {"digest": protocol_digest}:
        raise ValueError("protocol digest record drifted")

    activation = protocol.get("official_activation")
    if not isinstance(activation, Mapping):
        raise ValueError("official activation is missing from protocol")
    if (
        activation.get("workflow_run_id") != run_id
        or activation.get("workflow_run_number") != 1
        or activation.get("workflow_run_attempt") != 1
        or activation.get("workflow_event") != "push"
        or activation.get("ref_name") != OFFICIAL_REF
        or activation.get("workflow_ref") != OFFICIAL_WORKFLOW_REF
    ):
        raise ValueError("official activation identity drifted")
    head_sha = activation.get("head_sha")
    if (
        not isinstance(head_sha, str)
        or len(head_sha) != 40
        or any(character not in "0123456789abcdef" for character in head_sha)
    ):
        raise ValueError("official activation head SHA is invalid")

    runtime = protocol.get("required_runtime_packages")
    if not isinstance(runtime, Mapping):
        raise ValueError("required runtime package roster is missing")
    for name, version in {
        "lightgbm": "4.7.0",
        "stable-baselines3": "2.3.2",
        "torch": "2.4.1",
    }.items():
        if runtime.get(name) != version:
            raise ValueError(f"frozen runtime drifted: {name}")
    if not isinstance(runtime.get("scikit-learn"), str) or not runtime["scikit-learn"]:
        raise ValueError("scikit-learn runtime must be concrete")

    if (
        tuple(protocol.get("arms", ())) != tuple(ARMS)
        or protocol.get("ppo_timesteps") != 262_144
        or protocol.get("ppo_training_layout") != "sequential"
        or protocol.get("initial_capital") != 10_000.0
        or protocol.get("maximum_drawdown") != 0.2
        or protocol.get("deleveraging_start") != 0.1
        or protocol.get("per_symbol_gross") != 0.1
        or protocol.get("account_gross") != 0.5
        or protocol.get("unused_data_accessed") is not False
        or protocol.get("production_eligible") is not False
    ):
        raise ValueError("frozen directional protocol drifted")

    _, claim = _canonical_object(activation_root / "activation.json")
    helper = _load_activation_helper(target)
    helper.validate_activation_claim(
        claim,
        protocol,
        workflow_run_id=run_id,
        implementation_head=TARGET_SHA,
    )
    encoded_name = urllib.parse.quote(ACTIVATION_NAME, safe="")
    slot = _api_json(
        repository,
        token,
        f"actions/artifacts?name={encoded_name}&per_page=100",
    )
    helper.validate_remote_slot(slot, state="claimed", workflow_run_id=run_id)

    results: dict[str, dict[str, Any]] = {}
    exit_codes: dict[str, int] = {}
    failed_arms: list[str] = []
    for arm in ARMS:
        package = arms_root / f"issue640-directional-arm-{arm}-v1"
        _, attempt = _canonical_object(package / "attempt.json")
        exit_code = attempt.get("exit_code")
        if isinstance(exit_code, bool) or not isinstance(exit_code, int):
            raise ValueError(f"invalid arm exit code: {arm}")
        exit_codes[arm] = exit_code
        if (
            attempt.get("issue_number") != ISSUE_NUMBER
            or attempt.get("arm") != arm
            or attempt.get("workflow_run_id") != run_id
            or attempt.get("workflow_run_attempt") != 1
            or attempt.get("implementation_head") != TARGET_SHA
            or attempt.get("economic_result_interpreted") is not False
            or attempt.get("unused_data_accessed") is not False
            or attempt.get("production_eligible") is not False
            or attempt.get("live_trading_authorized") is not False
        ):
            raise ValueError(f"arm attempt identity drifted: {arm}")

        result_path = package / "result.json"
        failed_path = package / "failed.json"
        if exit_code != 0:
            if result_path.exists():
                raise ValueError(f"failed arm published a result: {arm}")
            failed_arms.append(arm)
            continue
        if (
            attempt.get("started") is not True
            or attempt.get("result_written") is not True
            or attempt.get("failure_written") is not False
            or failed_path.exists()
            or not result_path.is_file()
        ):
            raise ValueError(f"successful arm evidence is incomplete: {arm}")

        result_raw, result = _canonical_object(result_path)
        _, result_digest = _canonical_object(package / "result.sha256.json")
        if result_digest != {"sha256": hashlib.sha256(result_raw).hexdigest()}:
            raise ValueError(f"result SHA drifted: {arm}")
        if (
            result.get("arm") != arm
            or result.get("protocol_digest") != protocol_digest
            or result.get("provenance") != protocol.get("provenance")
            or result.get("production_eligible") is not False
        ):
            raise ValueError(f"result authority drifted: {arm}")
        model_hashes = result.get("model_sha256")
        if not isinstance(model_hashes, Mapping):
            raise ValueError(f"model hash map is missing: {arm}")
        for name, digest in model_hashes.items():
            if (
                not isinstance(name, str)
                or Path(name).name != name
                or not isinstance(digest, str)
                or _sha256(package / name) != digest
            ):
                raise ValueError(f"model digest mismatch: {arm}/{name}")
        results[arm] = result

    _, execution = _canonical_object(summary_root / "execution.json")
    complete = len(results) == len(ARMS) and not failed_arms
    if (
        execution.get("issue_number") != ISSUE_NUMBER
        or execution.get("workflow_run_id") != run_id
        or execution.get("implementation_head") != TARGET_SHA
        or execution.get("arm_exit_codes") != exit_codes
        or execution.get("all_arms_succeeded") is not complete
        or execution.get("economic_result_interpreted") is not False
        or execution.get("unused_data_accessed") is not False
        or execution.get("production_eligible") is not False
        or execution.get("live_trading_authorized") is not False
    ):
        raise ValueError("execution summary drifted")

    selection_path = summary_root / "selection.json"
    selection_verified = False
    decision: str | None = None
    winner: str | None = None
    if complete:
        if execution.get("selection_written") is not True or not selection_path.is_file():
            raise ValueError("complete study lacks immutable selection")
        expected = select_development_candidates(results)
        expected["protocol_digest"] = protocol_digest
        expected["result_sha256"] = {
            arm: _sha256(
                arms_root / f"issue640-directional-arm-{arm}-v1" / "result.json"
            )
            for arm in ARMS
        }
        if canonical_json_bytes(expected) != selection_path.read_bytes():
            raise ValueError("selection differs from independent reconstruction")
        selection_verified = True
        raw_decision = expected.get("decision")
        raw_winner = expected.get("winner")
        decision = raw_decision if isinstance(raw_decision, str) else None
        winner = raw_winner if isinstance(raw_winner, str) else None
    elif execution.get("selection_written") is not False or selection_path.exists():
        raise ValueError("incomplete study must not publish a selection")

    names = _artifact_names(repository, token, run_id)
    expected_names = [
        ACTIVATION_NAME,
        PROTOCOL_NAME,
        SUMMARY_NAME,
        *(f"issue640-directional-arm-{arm}-v1" for arm in ARMS),
    ]
    for name in expected_names:
        if names.count(name) != 1:
            raise ValueError(f"workflow artifact slot count drifted: {name}")

    payload = {
        "schema_version": "issue640_directional_official_audit_v1",
        "issue_number": ISSUE_NUMBER,
        "workflow_run_id": run_id,
        "implementation_head": TARGET_SHA,
        "protocol_sha256": hashlib.sha256(protocol_raw).hexdigest(),
        "protocol_digest": protocol_digest,
        "complete": complete,
        "failed_arms": failed_arms,
        "selection_verified": selection_verified,
        "decision": decision,
        "winner": winner,
        "model_refit_performed": False,
        "unused_data_accessed": False,
        "production_eligible": False,
        "live_trading_authorized": False,
    }
    output.mkdir(parents=True, exist_ok=False)
    (output / "audit.json").write_bytes(canonical_json_bytes(payload))
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--activation", type=Path, required=True)
    parser.add_argument("--arms", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    repository = os.environ.get("GITHUB_REPOSITORY")
    token = os.environ.get("GH_TOKEN")
    run_id_text = os.environ.get("GITHUB_RUN_ID")
    if not repository or not token or not run_id_text or not run_id_text.isdigit():
        raise SystemExit("GitHub workflow authority environment is required")

    payload = audit(
        target=args.target,
        protocol_root=args.protocol,
        activation_root=args.activation,
        arms_root=args.arms,
        summary_root=args.summary,
        output=args.output,
        repository=repository,
        token=token,
        run_id=int(run_id_text),
    )
    print(f"ISSUE640_COMPLETE={str(payload['complete']).lower()}")
    print(f"ISSUE640_SELECTION_VERIFIED={str(payload['selection_verified']).lower()}")
    print("ISSUE640_MODEL_REFIT_PERFORMED=false")


if __name__ == "__main__":
    main()
