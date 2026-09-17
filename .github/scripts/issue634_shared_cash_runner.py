from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

EVALUATOR_HEAD = "d1bf25fd66834289ffddb36063b0d5cef649f444"
EVALUATOR_VERIFY_RUN = 35213065827
EVALUATOR_PUBLICATION_RUN = 35214002467
EVALUATOR_IMPLEMENTATION_SHA = (
    "746d639ae4306d01c0765013d4f45882a419ceae3abf64e96ab0387d049d5d85"
)
EVALUATOR_SEAL_SHA = "82f931041be3a2b7aeedd85e1bc3b6ae683365af2ae28d6d3adb60ac0103231a"
EVALUATOR_PRIMARY_ID = 10493392861
EVALUATOR_PRIMARY_NAME = "issue630-final-authority-primary-v2"
EVALUATOR_PRIMARY_API_DIGEST = (
    "sha256:3b1f1e0f3001085545acc6008a3693d9978db33fc5d501abc58a8316efdd1baf"
)
EVALUATOR_FRESH_ID = 10493118083
EVALUATOR_FRESH_NAME = "issue630-final-authority-fresh-v2"
EVALUATOR_FRESH_API_DIGEST = (
    "sha256:fe727e1b7753b827e7ad5cee5e1bdb726f2d1a63321cf8e9b1234b52526591c7"
)
EVALUATOR_AUDIT_RUN = 35214226479
EVALUATOR_AUDIT_A_ID = 10494458180
EVALUATOR_AUDIT_A_NAME = "issue630-final-authority-audit-a-v1"
EVALUATOR_AUDIT_B_ID = 10494448132
EVALUATOR_AUDIT_B_NAME = "issue630-final-authority-audit-b-v1"
EVALUATOR_AUDIT_API_DIGEST = (
    "sha256:9480bf851092f48eb132c2f49d01fab13505fea3e2e729e17739a13d0953b581"
)

PROTOCOL_SEAL_RUN = 35211715033
PROTOCOL_PRIMARY_ID = 10491893357
PROTOCOL_PRIMARY_NAME = "issue627-shared-cash-prereg-v3-primary-35211715033"
PROTOCOL_PRIMARY_API_DIGEST = (
    "sha256:0e3e1dc742c00d3a1625aa9ee8681aba86a676946083afd08eda783642dd6841"
)
PROTOCOL_FRESH_ID = 10492645772
PROTOCOL_FRESH_NAME = "issue627-shared-cash-prereg-v3-fresh-35211715033"
PROTOCOL_FRESH_API_DIGEST = (
    "sha256:ca5fc01ed41cad6fa2f2dc2cf9ef11e4fe7cd3b5b6f7d8bc86bea1eb86842f56"
)

TRIGGER_RUN = 35201639813
TRIGGER_RESULT_ID = 10487739534
TRIGGER_RESULT_NAME = "issue626-ridge-economic-result-35201639813"
TRIGGER_RESULT_API_DIGEST = (
    "sha256:787362841f4ff9b68235f157ba82a9e02edc1e995b6bedb54ec05149b6d59148"
)
TRIGGER_FRESH_ID = 10488427586
TRIGGER_FRESH_NAME = "issue626-ridge-economic-fresh-verification-35201639813"
TRIGGER_FRESH_API_DIGEST = (
    "sha256:178bdd2575ab1d22d033a22a15f96b8a613fd786688b6d28647544352afee703"
)
TRIGGER_RESULT_DIGEST = (
    "78a39790c4d11dc903ac48f6044b0ebab46d2d6d26cbd8cf2c380be983b90d4b"
)
TRIGGER_RESULT_JSON_SHA = (
    "f6821eb123c4910178e190c378d86dbba486b13ede6c47fa8d4ef047ac78ac2a"
)

SOURCE_RUN = 34803217815
SOURCE_ID = 10331899302
SOURCE_NAME = "issue539-calibrated-causal-successor-v3-34803217815"
SOURCE_API_DIGEST = (
    "sha256:89e899427f23fa46929c8be1e71fd49abe0d1d465c7a7f796a0874426b885bce"
)
DATASET_ID = "6c0b040d317a1bb73a9273f4135879b31691634aa837f30f0eec005ac7531518"
DATASET_ARTIFACT_DIGEST = (
    "af481dd978db7d84cd3aa8ff4f5a35d8608ac44c755dd74f61e934105c02b6b7"
)
STUDY_DIGEST = "bfa2fcb307773f5384d7dcb884444164d6d3b575373d8f7dc1c810a61bf4c820"

COST_RUN = 35200841490
COST_ID = 10488425531
COST_NAME = "issue618-ridge-cost-authority-35200841490"
COST_API_DIGEST = (
    "sha256:7b4e0749b783fa29e50c3b92f0a21541148c80b8b4f4ed6b0586b0d5b52f64fb"
)
COST_AUTHORITY_DIGEST = (
    "bb33f36edcf69ba91257e85dc68c7d53db396867ce51ea204a6ee14bac5cec04"
)

RESULT_ARTIFACT_NAME = "issue634-ridge-shared-cash-result-v1"
FRESH_ARTIFACT_NAME = "issue634-ridge-shared-cash-fresh-v1"
AUDIT_ARTIFACT_NAME = "issue634-ridge-shared-cash-audit-v1"


def _canonical(payload: object) -> bytes:
    return (
        json.dumps(payload, allow_nan=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


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


def _assert_run_success(run_id: int, *, label: str) -> None:
    if _api(f"actions/runs/{run_id}").get("conclusion") != "success":
        raise SystemExit(f"{label} run is not successful")


def _assert_artifact(
    artifact_id: int,
    *,
    name: str,
    digest: str,
    run_id: int,
) -> None:
    meta = _api(f"actions/artifacts/{artifact_id}")
    if (
        meta.get("id") != artifact_id
        or meta.get("name") != name
        or meta.get("digest") != digest
        or (meta.get("workflow_run") or {}).get("id") != run_id
        or meta.get("expired") is not False
    ):
        raise SystemExit(f"immutable Artifact authority drift: {name}")


def _publication_jobs_are_valid() -> None:
    jobs = _api(f"actions/runs/{EVALUATOR_PUBLICATION_RUN}/jobs?per_page=100").get(
        "jobs"
    )
    if not isinstance(jobs, list):
        raise SystemExit("Issue 630 publication jobs unavailable")
    conclusions = {
        item.get("name"): item.get("conclusion")
        for item in jobs
        if isinstance(item, dict)
    }
    required = {
        "Recover primary publication from canonical generator": "success",
        "Recover fresh independent reconstruction": "success",
    }
    if any(conclusions.get(name) != expected for name, expected in required.items()):
        raise SystemExit("Issue 630 primary/fresh authority publication drift")


def authority_check(*, result_slot: str) -> None:
    if (
        subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        != EVALUATOR_HEAD
    ):
        raise SystemExit("exact evaluator HEAD drift")

    from trade_rl.evaluation.experiments.bootstrap.ridge_shared_cash_evaluation import (
        canonical_ridge_shared_cash_evaluation_spec,
    )

    spec = canonical_ridge_shared_cash_evaluation_spec()
    expected = {
        "issue_number": 630,
        "protocol_head": "65eb3c90e0a5fe1cea952279c28024160838da08",
        "protocol_digest": "c4942c190e507b2d437bd00646fd7ed09e2be5cff9b30fba6d1774b6f2af9c88",
        "protocol_seal_run_id": PROTOCOL_SEAL_RUN,
        "protocol_primary_artifact_id": PROTOCOL_PRIMARY_ID,
        "protocol_fresh_artifact_id": PROTOCOL_FRESH_ID,
        "trigger_run_id": TRIGGER_RUN,
        "trigger_result_artifact_id": TRIGGER_RESULT_ID,
        "trigger_fresh_artifact_id": TRIGGER_FRESH_ID,
        "trigger_result_digest": TRIGGER_RESULT_DIGEST,
        "successor_run_id": SOURCE_RUN,
        "successor_artifact_id": SOURCE_ID,
        "dataset_id": DATASET_ID,
        "dataset_artifact_digest": DATASET_ARTIFACT_DIGEST,
        "study_digest": STUDY_DIGEST,
        "cost_authority_run_id": COST_RUN,
        "cost_authority_artifact_id": COST_ID,
        "cost_authority_digest": COST_AUTHORITY_DIGEST,
        "execution_max_leverage": 1.0,
        "expected_n_periods": 17_544,
        "calendar_year_period_counts": ((2023, 8_760), (2024, 8_784)),
    }
    for field, value in expected.items():
        if getattr(spec, field) != value:
            raise SystemExit(f"sealed evaluator spec drift: {field}")
    for field in (
        "unused_data_accessed",
        "final_test_accessed",
        "final_test_authorized",
        "operational_eligibility_established",
        "production_eligible",
        "live_trading_authorized",
        "merge_authorized",
    ):
        if getattr(spec, field) is not False:
            raise SystemExit(f"research boundary drift: {field}")

    issue = _api("issues/634")
    if issue.get("state") != "open":
        raise SystemExit("Issue 634 must remain open before final audit")

    _assert_run_success(EVALUATOR_VERIFY_RUN, label="Issue 630 exact verification")
    _assert_run_success(EVALUATOR_AUDIT_RUN, label="Issue 630 independent audit")
    _assert_run_success(PROTOCOL_SEAL_RUN, label="Issue 627 protocol seal")
    _assert_run_success(TRIGGER_RUN, label="Issue 626 trigger")
    _assert_run_success(COST_RUN, label="PRE-PnL cost authority")
    _publication_jobs_are_valid()

    for args in (
        (
            PROTOCOL_PRIMARY_ID,
            PROTOCOL_PRIMARY_NAME,
            PROTOCOL_PRIMARY_API_DIGEST,
            PROTOCOL_SEAL_RUN,
        ),
        (
            PROTOCOL_FRESH_ID,
            PROTOCOL_FRESH_NAME,
            PROTOCOL_FRESH_API_DIGEST,
            PROTOCOL_SEAL_RUN,
        ),
        (
            EVALUATOR_PRIMARY_ID,
            EVALUATOR_PRIMARY_NAME,
            EVALUATOR_PRIMARY_API_DIGEST,
            EVALUATOR_PUBLICATION_RUN,
        ),
        (
            EVALUATOR_FRESH_ID,
            EVALUATOR_FRESH_NAME,
            EVALUATOR_FRESH_API_DIGEST,
            EVALUATOR_PUBLICATION_RUN,
        ),
        (
            EVALUATOR_AUDIT_A_ID,
            EVALUATOR_AUDIT_A_NAME,
            EVALUATOR_AUDIT_API_DIGEST,
            EVALUATOR_AUDIT_RUN,
        ),
        (
            EVALUATOR_AUDIT_B_ID,
            EVALUATOR_AUDIT_B_NAME,
            EVALUATOR_AUDIT_API_DIGEST,
            EVALUATOR_AUDIT_RUN,
        ),
        (
            TRIGGER_RESULT_ID,
            TRIGGER_RESULT_NAME,
            TRIGGER_RESULT_API_DIGEST,
            TRIGGER_RUN,
        ),
        (TRIGGER_FRESH_ID, TRIGGER_FRESH_NAME, TRIGGER_FRESH_API_DIGEST, TRIGGER_RUN),
        (SOURCE_ID, SOURCE_NAME, SOURCE_API_DIGEST, SOURCE_RUN),
        (COST_ID, COST_NAME, COST_API_DIGEST, COST_RUN),
    ):
        _assert_artifact(args[0], name=args[1], digest=args[2], run_id=args[3])

    encoded = urllib.parse.quote(RESULT_ARTIFACT_NAME, safe="")
    result_listing = _api(f"actions/artifacts?name={encoded}&per_page=100")
    count = result_listing.get("total_count")
    if result_slot == "empty" and count != 0:
        raise SystemExit("Issue 634 economic result slot is already consumed")
    if result_slot == "published" and count != 1:
        raise SystemExit("Issue 634 requires exactly one published result Artifact")


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SystemExit(f"JSON object expected: {path}")
    return value


def _validate_downloaded_inputs(
    *,
    successor: Path,
    cost: Path,
    evaluator_primary: Path,
    evaluator_fresh: Path,
    trigger_result: Path,
    trigger_fresh: Path,
) -> Any:
    from trade_rl.artifacts.hashing import content_digest
    from trade_rl.data import (
        inspect_published_market_dataset_artifact,
        load_market_dataset_artifact,
    )

    primary_impl = (evaluator_primary / "implementation.json").read_bytes()
    fresh_impl = (evaluator_fresh / "implementation.json").read_bytes()
    primary_seal = (evaluator_primary / "seal.json").read_bytes()
    fresh_seal = (evaluator_fresh / "seal.json").read_bytes()
    if primary_impl != fresh_impl or primary_seal != fresh_seal:
        raise SystemExit("Issue 630 primary/fresh authority bytes differ")
    if _sha256(primary_impl) != EVALUATOR_IMPLEMENTATION_SHA:
        raise SystemExit("Issue 630 implementation document SHA drift")
    if _sha256(primary_seal) != EVALUATOR_SEAL_SHA:
        raise SystemExit("Issue 630 implementation seal SHA drift")
    implementation = json.loads(primary_impl)
    if (
        not isinstance(implementation, dict)
        or implementation.get("implementation_head") != EVALUATOR_HEAD
    ):
        raise SystemExit("Issue 630 implementation authority HEAD drift")

    gate = _load_json(cost / "gate.json")
    authority_payload = _load_json(cost / "cost-authority.json")
    if content_digest(authority_payload) != COST_AUTHORITY_DIGEST:
        raise SystemExit("PRE-PnL cost authority digest drift")
    required_gate = {
        "gate_run_id": COST_RUN,
        "source_run_id": SOURCE_RUN,
        "source_artifact_id": SOURCE_ID,
        "dataset_id": DATASET_ID,
        "dataset_artifact_digest": DATASET_ARTIFACT_DIGEST,
        "checked_execution_rows": 17_545,
        "checked_symbols": 5,
        "fee_rate": 0.0005,
        "taker_fee_rate": 0.0,
        "spread_rate": 0.0002,
        "one_way_explicit_cost": 0.0007,
        "cost_authority_digest": COST_AUTHORITY_DIGEST,
        "dataset_only": True,
        "study_plan_contents_read": False,
        "evaluation_pnl_inspected": False,
        "evaluation_execution_authorized": False,
        "verified": True,
    }
    for field, value in required_gate.items():
        if gate.get(field) != value:
            raise SystemExit(f"PRE-PnL cost gate drift: {field}")

    published = inspect_published_market_dataset_artifact(successor / "dataset")
    if published.artifact_digest != DATASET_ARTIFACT_DIGEST:
        raise SystemExit("Dataset artifact digest drift")
    dataset = load_market_dataset_artifact(successor / "dataset")
    if dataset.dataset_id != DATASET_ID:
        raise SystemExit("Dataset ID drift")
    plan_payload = _load_json(successor / "study" / "plan.json")
    if content_digest(plan_payload) != STUDY_DIGEST:
        raise SystemExit("P&L-free Study plan digest drift")

    result_bytes = (trigger_result / "result.json").read_bytes()
    if _sha256(result_bytes) != TRIGGER_RESULT_JSON_SHA:
        raise SystemExit("Issue 626 result JSON SHA drift")
    trigger_payload = json.loads(result_bytes)
    if not isinstance(trigger_payload, dict):
        raise SystemExit("Issue 626 result payload invalid")
    if content_digest(trigger_payload) != TRIGGER_RESULT_DIGEST:
        raise SystemExit("Issue 626 result digest drift")
    if trigger_payload.get("research_status") != "PROMOTE_RESEARCH_REFERENCE":
        raise SystemExit("Issue 626 frozen trigger status drift")
    reconstructed = (trigger_fresh / "reconstructed-result.json").read_bytes()
    if reconstructed != result_bytes:
        raise SystemExit("Issue 626 primary/fresh result bytes differ")
    verification = _load_json(trigger_fresh / "verification.json")
    if verification.get("byte_identical") is not True:
        raise SystemExit("Issue 626 fresh verification drift")
    if verification.get("result_digest") != TRIGGER_RESULT_DIGEST:
        raise SystemExit("Issue 626 fresh result digest drift")
    return dataset


def _encode_result(result: Any) -> bytes:
    return _canonical(result.to_payload())


def _decode_arm(raw: object) -> Any:
    from trade_rl.evaluation.experiments.bootstrap.ridge_shared_cash_evaluation import (
        RidgeSharedCashArmEvidence,
    )

    if not isinstance(raw, dict):
        raise ValueError("shared-cash arm must be an object")
    expected = {
        "schema_version",
        "arm",
        "returns",
        "return_sha256",
        "total_return",
        "calendar_year_returns",
        "calendar_year_period_counts",
        "total_cost",
        "turnover_total",
        "max_drawdown",
        "termination_count",
        "termination_reasons",
        "n_periods",
    }
    if set(raw) != expected:
        raise ValueError("shared-cash arm keys differ from sealed schema")
    returns = raw["returns"]
    reasons = raw["termination_reasons"]
    year_returns = raw["calendar_year_returns"]
    year_counts = raw["calendar_year_period_counts"]
    if not isinstance(returns, list) or not isinstance(reasons, list):
        raise ValueError("shared-cash arm array fields are invalid")
    if not isinstance(year_returns, list) or not isinstance(year_counts, list):
        raise ValueError("calendar-year evidence arrays are invalid")

    def pair_list(values: list[object], value_key: str) -> tuple[tuple[Any, Any], ...]:
        pairs: list[tuple[Any, Any]] = []
        for item in values:
            if not isinstance(item, dict) or set(item) != {"year", value_key}:
                raise ValueError("calendar-year evidence item schema drift")
            pairs.append((item["year"], item[value_key]))
        return tuple(pairs)

    return RidgeSharedCashArmEvidence(
        schema_version=raw["schema_version"],
        arm=raw["arm"],
        returns=tuple(returns),
        return_sha256=raw["return_sha256"],
        total_return=raw["total_return"],
        calendar_year_returns=pair_list(year_returns, "return"),
        calendar_year_period_counts=pair_list(year_counts, "count"),
        total_cost=raw["total_cost"],
        turnover_total=raw["turnover_total"],
        max_drawdown=raw["max_drawdown"],
        termination_count=raw["termination_count"],
        termination_reasons=tuple(reasons),
        n_periods=raw["n_periods"],
    )


def decode_result(raw: bytes) -> Any:
    from trade_rl.evaluation.experiments.bootstrap.ridge_shared_cash_evaluation import (
        RidgeSharedCashEvaluation,
    )

    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError("shared-cash result must be an object")
    expected = {
        "schema_version",
        "spec_digest",
        "dataset_id",
        "symbols",
        "baseline",
        "candidate",
        "status",
        "unused_data_accessed",
        "final_test_accessed",
        "final_test_authorized",
        "operational_eligibility_established",
        "production_eligible",
        "live_trading_authorized",
        "merge_authorized",
    }
    if set(payload) != expected or not isinstance(payload["symbols"], list):
        raise ValueError("shared-cash result schema drift")
    result = RidgeSharedCashEvaluation(
        schema_version=payload["schema_version"],
        spec_digest=payload["spec_digest"],
        dataset_id=payload["dataset_id"],
        symbols=tuple(payload["symbols"]),
        baseline=_decode_arm(payload["baseline"]),
        candidate=_decode_arm(payload["candidate"]),
        status=payload["status"],
        unused_data_accessed=payload["unused_data_accessed"],
        final_test_accessed=payload["final_test_accessed"],
        final_test_authorized=payload["final_test_authorized"],
        operational_eligibility_established=payload[
            "operational_eligibility_established"
        ],
        production_eligible=payload["production_eligible"],
        live_trading_authorized=payload["live_trading_authorized"],
        merge_authorized=payload["merge_authorized"],
    )
    if _encode_result(result) != raw:
        raise ValueError("shared-cash result is not canonical")
    return result


def _provenance(result: Any, raw: bytes, *, publisher_run_id: int) -> dict[str, object]:
    return {
        "schema_version": "issue634_ridge_shared_cash_result_provenance_v1",
        "issue_number": 634,
        "publisher_run_id": publisher_run_id,
        "publisher_run_number": 1,
        "publisher_run_attempt": 1,
        "evaluator_head": EVALUATOR_HEAD,
        "evaluator_verification_run": EVALUATOR_VERIFY_RUN,
        "evaluator_publication_run": EVALUATOR_PUBLICATION_RUN,
        "evaluator_primary_artifact_id": EVALUATOR_PRIMARY_ID,
        "evaluator_primary_api_digest": EVALUATOR_PRIMARY_API_DIGEST,
        "evaluator_fresh_artifact_id": EVALUATOR_FRESH_ID,
        "evaluator_fresh_api_digest": EVALUATOR_FRESH_API_DIGEST,
        "evaluator_independent_audit_run": EVALUATOR_AUDIT_RUN,
        "protocol_seal_run": PROTOCOL_SEAL_RUN,
        "trigger_run": TRIGGER_RUN,
        "source_run": SOURCE_RUN,
        "source_artifact_id": SOURCE_ID,
        "source_artifact_api_digest": SOURCE_API_DIGEST,
        "cost_authority_run": COST_RUN,
        "cost_authority_artifact_id": COST_ID,
        "cost_authority_digest": COST_AUTHORITY_DIGEST,
        "result_digest": result.digest,
        "result_json_sha256": _sha256(raw),
        "sealed_evaluator_invocation_count": 1,
        "result_interpreted_before_publication": False,
        "unused_data_accessed": False,
        "final_test_accessed": False,
        "final_test_authorized": False,
        "operational_eligibility_established": False,
        "production_eligible": False,
        "live_trading_authorized": False,
        "merge_authorized": False,
    }


def evaluate_command(args: argparse.Namespace) -> None:
    authority_check(result_slot="empty")
    dataset = _validate_downloaded_inputs(
        successor=args.successor,
        cost=args.cost,
        evaluator_primary=args.evaluator_primary,
        evaluator_fresh=args.evaluator_fresh,
        trigger_result=args.trigger_result,
        trigger_fresh=args.trigger_fresh,
    )
    from trade_rl.evaluation.experiments.bootstrap.ridge_shared_cash_evaluation import (
        evaluate_ridge_shared_cash,
    )

    result = evaluate_ridge_shared_cash(dataset)
    raw = _encode_result(result)
    decode_result(raw)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "result.json").write_bytes(raw)
    (args.output / "provenance.json").write_bytes(
        _canonical(
            _provenance(
                result,
                raw,
                publisher_run_id=int(os.environ["GITHUB_RUN_ID"]),
            )
        )
    )
    print("ISSUE634_PUBLISHER_RESULT_WRITTEN=true")
    print("RESULT_INTERPRETED_BEFORE_PUBLICATION=false")


def reconstruct_command(args: argparse.Namespace) -> None:
    authority_check(result_slot="published")
    dataset = _validate_downloaded_inputs(
        successor=args.successor,
        cost=args.cost,
        evaluator_primary=args.evaluator_primary,
        evaluator_fresh=args.evaluator_fresh,
        trigger_result=args.trigger_result,
        trigger_fresh=args.trigger_fresh,
    )
    published = (args.published / "result.json").read_bytes()
    published_result = decode_result(published)
    provenance = _load_json(args.published / "provenance.json")
    if provenance.get("publisher_run_id") != int(os.environ["GITHUB_RUN_ID"]):
        raise SystemExit("publisher provenance run ID drift")
    if provenance.get("result_json_sha256") != _sha256(published):
        raise SystemExit("publisher result SHA drift")
    if provenance.get("result_digest") != published_result.digest:
        raise SystemExit("publisher result digest drift")
    if provenance.get("result_interpreted_before_publication") is not False:
        raise SystemExit("publisher interpretation boundary drift")

    from trade_rl.evaluation.experiments.bootstrap.ridge_shared_cash_evaluation import (
        evaluate_ridge_shared_cash,
    )

    reconstructed_result = evaluate_ridge_shared_cash(dataset)
    reconstructed = _encode_result(reconstructed_result)
    if reconstructed != published:
        raise SystemExit("fresh reconstruction differs from published result bytes")
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "reconstructed-result.json").write_bytes(reconstructed)
    verification = {
        "schema_version": "issue634_ridge_shared_cash_fresh_verification_v1",
        "publisher_run_id": int(os.environ["GITHUB_RUN_ID"]),
        "result_digest": reconstructed_result.digest,
        "result_json_sha256": _sha256(reconstructed),
        "publisher_provenance_sha256": _sha256(
            (args.published / "provenance.json").read_bytes()
        ),
        "byte_identical": True,
        "fresh_evaluator_invocation_count": 1,
        "result_interpreted_before_fresh_publication": False,
        "unused_data_accessed": False,
        "final_test_accessed": False,
        "production_eligible": False,
        "live_trading_authorized": False,
        "merge_authorized": False,
    }
    (args.output / "verification.json").write_bytes(_canonical(verification))
    print("ISSUE634_FRESH_BYTE_IDENTICAL=true")
    print("RESULT_INTERPRETED_BEFORE_FRESH_PUBLICATION=false")


def _assert_own_artifact(prefix: str, *, expected_name: str) -> tuple[int, str]:
    artifact_id = int(os.environ[f"{prefix}_ARTIFACT_ID"])
    expected_digest = os.environ[f"{prefix}_ARTIFACT_DIGEST"]
    expected_digest = (
        expected_digest
        if expected_digest.startswith("sha256:")
        else f"sha256:{expected_digest}"
    )
    meta = _api(f"actions/artifacts/{artifact_id}")
    if (
        meta.get("id") != artifact_id
        or meta.get("name") != expected_name
        or meta.get("digest") != expected_digest
        or (meta.get("workflow_run") or {}).get("id")
        != int(os.environ["GITHUB_RUN_ID"])
        or meta.get("expired") is not False
    ):
        raise SystemExit(f"Issue 634 {prefix.lower()} Artifact authority drift")
    return artifact_id, expected_digest


def audit_command(args: argparse.Namespace) -> None:
    authority_check(result_slot="published")
    published = (args.published / "result.json").read_bytes()
    reconstructed = (args.fresh / "reconstructed-result.json").read_bytes()
    if published != reconstructed:
        raise SystemExit("publisher/fresh result bytes differ")
    result = decode_result(published)
    provenance = _load_json(args.published / "provenance.json")
    verification = _load_json(args.fresh / "verification.json")
    if provenance.get("result_digest") != result.digest:
        raise SystemExit("publisher provenance result digest drift")
    if provenance.get("result_json_sha256") != _sha256(published):
        raise SystemExit("publisher provenance result SHA drift")
    if verification.get("result_digest") != result.digest:
        raise SystemExit("fresh verification result digest drift")
    if verification.get("result_json_sha256") != _sha256(published):
        raise SystemExit("fresh verification result SHA drift")
    if verification.get("byte_identical") is not True:
        raise SystemExit("fresh verification byte identity drift")

    publisher_id, publisher_digest = _assert_own_artifact(
        "PUBLISHER", expected_name=RESULT_ARTIFACT_NAME
    )
    fresh_id, fresh_digest = _assert_own_artifact(
        "FRESH", expected_name=FRESH_ARTIFACT_NAME
    )
    baseline = result.baseline
    candidate = result.candidate
    audit = {
        "schema_version": "issue634_ridge_shared_cash_final_audit_v1",
        "issue_number": 634,
        "workflow_run_id": int(os.environ["GITHUB_RUN_ID"]),
        "publisher_artifact_id": publisher_id,
        "publisher_api_digest": publisher_digest,
        "fresh_artifact_id": fresh_id,
        "fresh_api_digest": fresh_digest,
        "result_digest": result.digest,
        "result_json_sha256": _sha256(published),
        "status": result.status,
        "baseline": {
            "total_return": baseline.total_return,
            "calendar_year_returns": baseline.to_payload()["calendar_year_returns"],
            "total_cost": baseline.total_cost,
            "turnover_total": baseline.turnover_total,
            "max_drawdown": baseline.max_drawdown,
            "termination_count": baseline.termination_count,
            "n_periods": baseline.n_periods,
            "return_sha256": baseline.return_sha256,
        },
        "candidate": {
            "total_return": candidate.total_return,
            "calendar_year_returns": candidate.to_payload()["calendar_year_returns"],
            "total_cost": candidate.total_cost,
            "turnover_total": candidate.turnover_total,
            "max_drawdown": candidate.max_drawdown,
            "termination_count": candidate.termination_count,
            "n_periods": candidate.n_periods,
            "return_sha256": candidate.return_sha256,
        },
        "primary_fresh_byte_equal": True,
        "unused_data_accessed": False,
        "final_test_accessed": False,
        "final_test_authorized": False,
        "operational_eligibility_established": False,
        "production_eligible": False,
        "live_trading_authorized": False,
        "merge_authorized": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(_canonical(audit))
    print("ISSUE634_FINAL_AUDIT=PASS")
    print(f"RESEARCH_STATUS={result.status}")
    print(f"BASELINE_TOTAL_RETURN={baseline.total_return}")
    print(f"CANDIDATE_TOTAL_RETURN={candidate.total_return}")
    for year, value in baseline.calendar_year_returns:
        print(f"BASELINE_{year}_RETURN={value}")
    for year, value in candidate.calendar_year_returns:
        print(f"CANDIDATE_{year}_RETURN={value}")
    print(f"BASELINE_TOTAL_COST={baseline.total_cost}")
    print(f"CANDIDATE_TOTAL_COST={candidate.total_cost}")
    print(f"BASELINE_TURNOVER={baseline.turnover_total}")
    print(f"CANDIDATE_TURNOVER={candidate.turnover_total}")
    print(f"BASELINE_MAX_DRAWDOWN={baseline.max_drawdown}")
    print(f"CANDIDATE_MAX_DRAWDOWN={candidate.max_drawdown}")
    print(f"BASELINE_TERMINATION_COUNT={baseline.termination_count}")
    print(f"CANDIDATE_TERMINATION_COUNT={candidate.termination_count}")
    print("UNUSED_DATA_ACCESSED=false")
    print("FINAL_TEST_ACCESSED=false")


def self_check() -> None:
    from trade_rl.evaluation.experiments.bootstrap.ridge_shared_cash_evaluation import (
        RidgeSharedCashArmEvidence,
        RidgeSharedCashEvaluation,
        canonical_ridge_shared_cash_evaluation_spec,
        shared_cash_return_sha256,
    )
    from trade_rl.evaluation.metrics import compound_return

    spec = canonical_ridge_shared_cash_evaluation_spec()
    counts = spec.calendar_year_period_counts
    baseline_returns = (0.0,) * spec.expected_n_periods
    candidate_returns = (1e-6,) * spec.expected_n_periods

    def year_returns(values: tuple[float, ...]) -> tuple[tuple[int, float], ...]:
        offset = 0
        output: list[tuple[int, float]] = []
        for year, count in counts:
            output.append((year, compound_return(values[offset : offset + count])))
            offset += count
        return tuple(output)

    baseline = RidgeSharedCashArmEvidence(
        arm="baseline",
        returns=baseline_returns,
        return_sha256=shared_cash_return_sha256(baseline_returns),
        total_return=compound_return(baseline_returns),
        calendar_year_returns=year_returns(baseline_returns),
        calendar_year_period_counts=counts,
        total_cost=10.0,
        turnover_total=5.0,
        max_drawdown=0.2,
        termination_count=0,
        termination_reasons=(),
        n_periods=spec.expected_n_periods,
    )
    candidate = RidgeSharedCashArmEvidence(
        arm="candidate",
        returns=candidate_returns,
        return_sha256=shared_cash_return_sha256(candidate_returns),
        total_return=compound_return(candidate_returns),
        calendar_year_returns=year_returns(candidate_returns),
        calendar_year_period_counts=counts,
        total_cost=5.0,
        turnover_total=2.0,
        max_drawdown=0.1,
        termination_count=0,
        termination_reasons=(),
        n_periods=spec.expected_n_periods,
    )
    result = RidgeSharedCashEvaluation(
        spec_digest=spec.digest,
        dataset_id=spec.dataset_id,
        symbols=spec.symbols,
        baseline=baseline,
        candidate=candidate,
        status="QUALIFY_UNUSED_VALIDATION",
    )
    raw = _encode_result(result)
    recovered = decode_result(raw)
    if recovered.digest != result.digest or _encode_result(recovered) != raw:
        raise SystemExit("Issue 634 helper canonical self-check failed")
    print("ISSUE634_HELPER_SELF_CHECK=PASS")
    print("REAL_MARKET_DATA_ACCESSED=false")
    print("ECONOMIC_EVALUATOR_CALLED=false")


def add_input_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--successor", type=Path, required=True)
    parser.add_argument("--cost", type=Path, required=True)
    parser.add_argument("--evaluator-primary", type=Path, required=True)
    parser.add_argument("--evaluator-fresh", type=Path, required=True)
    parser.add_argument("--trigger-result", type=Path, required=True)
    parser.add_argument("--trigger-fresh", type=Path, required=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("authority-check")
    check.add_argument("--result-slot", choices=("empty", "published"), required=True)
    evaluate = commands.add_parser("evaluate")
    add_input_arguments(evaluate)
    evaluate.add_argument("--output", type=Path, required=True)
    reconstruct = commands.add_parser("reconstruct")
    add_input_arguments(reconstruct)
    reconstruct.add_argument("--published", type=Path, required=True)
    reconstruct.add_argument("--output", type=Path, required=True)
    audit = commands.add_parser("audit")
    audit.add_argument("--published", type=Path, required=True)
    audit.add_argument("--fresh", type=Path, required=True)
    audit.add_argument("--output", type=Path, required=True)
    commands.add_parser("self-check")
    args = parser.parse_args()
    if args.command == "authority-check":
        authority_check(result_slot=args.result_slot)
    elif args.command == "evaluate":
        evaluate_command(args)
    elif args.command == "reconstruct":
        reconstruct_command(args)
    elif args.command == "audit":
        audit_command(args)
    else:
        self_check()


if __name__ == "__main__":
    main()
