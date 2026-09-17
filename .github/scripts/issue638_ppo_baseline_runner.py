from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Callable, cast

ISSUE_NUMBER = 638
EVALUATOR_HEAD = "97127952af3d01b71def695a4c8043eb301b2a33"
MAIN_AUTHORITY = "c4fc52777ad0ea894d0d8f3a01337bcbbd5a26a8"
EVALUATOR_VERIFY_RUN = 35_227_295_244
EVALUATOR_VERIFY_HARNESS = "c48f416f11cf9a00f28be3f13343b29ff24e5427"
EVALUATOR_PR = 635
EVALUATOR_SEAL_RUN = 35_228_154_799
EVALUATOR_IMPLEMENTATION_SHA256 = (
    "42cbe8d537a0080a01c5e6eb4f5a21236d9cf4a340818865dd7b6f5c89e5b9e1"
)
EVALUATOR_SEAL_SHA256 = (
    "95bdd6cd1d7edf6d8963824f7005c3166c80fe82de1015b22801c318c556ee26"
)
EVALUATOR_PRIMARY_ID = 10_499_254_503
EVALUATOR_PRIMARY_NAME = "issue632-evaluator-v5-primary-35228154799"
EVALUATOR_PRIMARY_API_DIGEST = (
    "sha256:282f5598d97d068a8c9c3a1bcef7a2fde773d607b134e61fa5c535af43f72929"
)
EVALUATOR_FRESH_ID = 10_500_042_172
EVALUATOR_FRESH_NAME = "issue632-evaluator-v5-fresh-35228154799"
EVALUATOR_FRESH_API_DIGEST = (
    "sha256:6ea2e7ba796cd4775c7710ef899c3e99dd3ca3282256092838c686ee0ca516f3"
)
EVALUATOR_AUDIT_ID = 10_500_047_240
EVALUATOR_AUDIT_NAME = "issue632-evaluator-v5-audit-35228154799"
EVALUATOR_AUDIT_API_DIGEST = (
    "sha256:7d7b29dd483538792aedcd5249a1cc0bb5e7b2038c951f4ec1685c579fa11183"
)

PROTOCOL_HEAD = "f1187dacae78e679a322cc53cbf03f3371f457b1"
PROTOCOL_DIGEST = "a34aee66bf3f51ce02675b955f835c292b023770aab841f25b46815f9b399c2c"
PROTOCOL_SEAL_RUN = 35_205_354_305
PROTOCOL_PRIMARY_ID = 10_489_866_637
PROTOCOL_PRIMARY_NAME = "issue629-prereg-seal-primary-35205354305"
PROTOCOL_PRIMARY_API_DIGEST = (
    "sha256:4ca30f9884b642f32443bbb143b15e4fecbbc7a25d262e89d63649acbef00387"
)
PROTOCOL_FRESH_ID = 10_489_661_856
PROTOCOL_FRESH_NAME = "issue629-prereg-seal-fresh-35205354305"
PROTOCOL_FRESH_API_DIGEST = (
    "sha256:2c9d9a71db2315dc66237c9ba4b975f0bdf5a5234ad87cf8bbd05b947867afbb"
)
PROTOCOL_SEAL_SHA256 = (
    "3e55fdcf772bbf4063d537e3f1912a5d1588611ce51e586180c8d1be13a186fe"
)

SOURCE_RUN = 34_803_217_815
SOURCE_ID = 10_331_899_302
SOURCE_NAME = "issue539-calibrated-causal-successor-v3-34803217815"
SOURCE_API_DIGEST = (
    "sha256:89e899427f23fa46929c8be1e71fd49abe0d1d465c7a7f796a0874426b885bce"
)
SOURCE_VERIFY_RUN = 34_803_432_434
SOURCE_VERIFY_ID = 10_332_575_500
SOURCE_VERIFY_NAME = "issue539-calibrated-causal-fresh-verify-v3-34803432434"
SOURCE_VERIFY_API_DIGEST = (
    "sha256:2067c38da3c1f45927748e2cc7481f11268710b2b386a5bd3b9a289394937589"
)
DATASET_ID = "6c0b040d317a1bb73a9273f4135879b31691634aa837f30f0eec005ac7531518"
DATASET_ARTIFACT_DIGEST = (
    "af481dd978db7d84cd3aa8ff4f5a35d8608ac44c755dd74f61e934105c02b6b7"
)
STUDY_DIGEST = "bfa2fcb307773f5384d7dcb884444164d6d3b575373d8f7dc1c810a61bf4c820"

BASELINE_SEEDS = (0, 1, 2, 3, 4)
RESULT_PREFIX = "issue638-ppo-sequential-baseline-seed-"
FRESH_PREFIX = "issue638-ppo-sequential-baseline-fresh-seed-"
ACTIVATION_PREFIX = "issue638-ppo-sequential-baseline-activation-seed-"

Evaluator = Callable[..., Any]


def _canonical(payload: object) -> bytes:
    from trade_rl.artifacts.canonical import canonical_json_bytes

    return canonical_json_bytes(payload)


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _validate_seed(seed: object) -> int:
    if (
        isinstance(seed, bool)
        or not isinstance(seed, int)
        or seed not in BASELINE_SEEDS
    ):
        raise ValueError("seed must be one of the frozen baseline seeds 0..4")
    return seed


def _validate_baseline_evidence(evidence: object, *, seed: int) -> None:
    _validate_seed(seed)
    if getattr(evidence, "arm", None) != "baseline":
        raise ValueError("Issue 638 evidence must be baseline-only")
    if (
        getattr(evidence, "seed", None) != seed
        or type(getattr(evidence, "seed", None)) is not int
    ):
        raise ValueError("Issue 638 evidence seed differs from the authorized seed")
    if getattr(evidence, "training_layout", None) != "sequential":
        raise ValueError("Issue 638 evidence must use sequential training")
    if getattr(evidence, "rollout_steps_per_env", object()) is not None:
        raise ValueError(
            "Issue 638 sequential baseline must not use a rollout override"
        )
    if getattr(evidence, "candidate_training_authorized", None) is not False:
        raise ValueError("Issue 638 cannot authorize candidate training")


def evaluate_baseline_seed(
    dataset: object,
    spec: object,
    *,
    seed: object,
    evaluator: Evaluator | None = None,
) -> Any:
    """Call the sealed evaluator exactly once with the baseline arm hard-coded."""

    canonical_seed = _validate_seed(seed)
    selected_evaluator: Evaluator
    if evaluator is None:
        from trade_rl.evaluation.experiments.ppo_interleaved_evaluation import (
            evaluate_ppo_training_seed,
        )

        selected_evaluator = cast(Evaluator, evaluate_ppo_training_seed)
    else:
        selected_evaluator = evaluator
    evidence = selected_evaluator(dataset, spec, arm="baseline", seed=canonical_seed)
    _validate_baseline_evidence(evidence, seed=canonical_seed)
    return evidence


def canonical_baseline_evidence_bytes(evidence: object, *, seed: object) -> bytes:
    canonical_seed = _validate_seed(seed)
    _validate_baseline_evidence(evidence, seed=canonical_seed)
    to_payload = getattr(evidence, "to_payload", None)
    if not callable(to_payload):
        raise TypeError("evidence must expose to_payload()")
    payload = to_payload()
    if not isinstance(payload, dict):
        raise ValueError("evidence payload must be an object")
    return _canonical(payload)


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


def _assert_run_success(run_id: int, *, label: str) -> dict[str, Any]:
    run = _api(f"actions/runs/{run_id}")
    if run.get("status") != "completed" or run.get("conclusion") != "success":
        raise SystemExit(f"{label} is not successful")
    return run


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


def _assert_source_authority() -> None:
    """Validate the publisher-bound successor plus verification-only recovery."""

    producer = _api(f"actions/runs/{SOURCE_RUN}")
    if producer.get("status") != "completed" or producer.get("conclusion") != "failure":
        raise SystemExit("calibrated successor producer run history drift")
    jobs = _api(f"actions/runs/{SOURCE_RUN}/jobs?per_page=100").get("jobs")
    if not isinstance(jobs, list):
        raise SystemExit("calibrated successor producer jobs unavailable")
    conclusions = {
        item.get("name"): item.get("conclusion")
        for item in jobs
        if isinstance(item, dict)
    }
    required = {
        "Exact Human Guide": "success",
        "Exact Lean Core": "success",
        "Materialize and seal successor": "success",
        "Fresh independent reverify": "failure",
        "Audit fresh verification": "skipped",
    }
    if any(conclusions.get(name) != expected for name, expected in required.items()):
        raise SystemExit("calibrated successor producer job authority drift")
    _assert_artifact(
        SOURCE_ID,
        name=SOURCE_NAME,
        digest=SOURCE_API_DIGEST,
        run_id=SOURCE_RUN,
    )

    _assert_run_success(SOURCE_VERIFY_RUN, label="successor verification recovery")
    verify_jobs = _api(f"actions/runs/{SOURCE_VERIFY_RUN}/jobs?per_page=100").get(
        "jobs"
    )
    if not isinstance(verify_jobs, list):
        raise SystemExit("successor recovery jobs unavailable")
    verify_conclusions = {
        item.get("name"): item.get("conclusion")
        for item in verify_jobs
        if isinstance(item, dict)
    }
    for name in (
        "Fresh independent reverify existing bundle",
        "Audit recovered fresh verification",
    ):
        if verify_conclusions.get(name) != "success":
            raise SystemExit(f"successor recovered verification drift: {name}")
    _assert_artifact(
        SOURCE_VERIFY_ID,
        name=SOURCE_VERIFY_NAME,
        digest=SOURCE_VERIFY_API_DIGEST,
        run_id=SOURCE_VERIFY_RUN,
    )


def _artifact_count(name: str) -> int:
    encoded = urllib.parse.quote(name, safe="")
    listing = _api(f"actions/artifacts?name={encoded}&per_page=100")
    count = listing.get("total_count")
    if isinstance(count, bool) or not isinstance(count, int):
        raise SystemExit("GitHub Artifact count is unavailable")
    return count


def _result_name(seed: int) -> str:
    return f"{RESULT_PREFIX}{seed}-v1"


def _fresh_name(seed: int) -> str:
    return f"{FRESH_PREFIX}{seed}-v1"


def _activation_name(seed: int) -> str:
    return f"{ACTIVATION_PREFIX}{seed}-v1"


def _validate_sha40(value: object, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 40
        or value != value.lower()
        or any(ch not in "0123456789abcdef" for ch in value)
    ):
        raise ValueError(f"{label} must be a lowercase 40-character git SHA")
    return value


def activation_claim_bytes(
    *,
    seed: object,
    run_id: object,
    run_attempt: object,
    helper_head: object,
    helper_blob_sha: object,
) -> bytes:
    canonical_seed = _validate_seed(seed)
    if isinstance(run_id, bool) or not isinstance(run_id, int) or run_id <= 0:
        raise ValueError("run_id must be a positive integer")
    if type(run_attempt) is not int or run_attempt != 1:
        raise ValueError("run_attempt must be exactly 1")
    canonical_head = _validate_sha40(helper_head, label="helper_head")
    canonical_blob = _validate_sha40(helper_blob_sha, label="helper_blob_sha")
    return _canonical(
        {
            "schema_version": "issue638_ppo_sequential_baseline_activation_v1",
            "issue_number": ISSUE_NUMBER,
            "seed": canonical_seed,
            "workflow_run_id": run_id,
            "workflow_run_attempt": run_attempt,
            "helper_head": canonical_head,
            "helper_blob_sha": canonical_blob,
            "arm": "baseline",
            "training_layout": "sequential",
            "rollout_steps_per_env": None,
            "caller_total_timesteps": 100_000,
            "evaluator_started_at_claim": False,
            "real_dataset_loaded_at_claim": False,
            "result_artifact_published_at_claim": False,
            "candidate_training_authorized": False,
            "economic_result_interpreted": False,
            "final_test_accessed": False,
            "production_eligible": False,
            "live_trading_authorized": False,
            "merge_authorized": False,
        }
    )


def _validate_slot_state(
    *, slot: str, activation_count: int, result_count: int, fresh_count: int
) -> None:
    if slot not in {"empty", "claimed", "published"}:
        raise ValueError("slot must be empty, claimed, or published")
    counts = (activation_count, result_count, fresh_count)
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in counts
    ):
        raise SystemExit("Issue 638 Artifact slot counts are invalid")
    expected = {"empty": (0, 0, 0), "claimed": (1, 0, 0), "published": (1, 1, 0)}[slot]
    if counts != expected:
        raise SystemExit(f"Issue 638 seed Artifact slot state differs from {slot}")


def authority_check(*, seed: object, slot: str) -> None:
    canonical_seed = _validate_seed(seed)
    if slot not in {"empty", "claimed", "published"}:
        raise ValueError("slot must be empty, claimed, or published")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    if head != EVALUATOR_HEAD:
        raise SystemExit("exact Issue 632 evaluator HEAD drift")
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", MAIN_AUTHORITY, EVALUATOR_HEAD],
        check=True,
    )

    from trade_rl.evaluation.experiments.ppo_interleaved_evaluation import (
        canonical_ppo_interleaved_evaluator_spec,
    )

    spec = canonical_ppo_interleaved_evaluator_spec()
    expected = {
        "protocol_head": PROTOCOL_HEAD,
        "protocol_digest": PROTOCOL_DIGEST,
        "protocol_seal_run_id": PROTOCOL_SEAL_RUN,
        "protocol_primary_artifact_id": PROTOCOL_PRIMARY_ID,
        "protocol_fresh_artifact_id": PROTOCOL_FRESH_ID,
        "successor_bundle_run_id": SOURCE_RUN,
        "successor_bundle_artifact_id": SOURCE_ID,
        "dataset_id": DATASET_ID,
        "dataset_artifact_digest": DATASET_ARTIFACT_DIGEST,
        "study_digest": STUDY_DIGEST,
        "ppo_seeds": BASELINE_SEEDS,
        "ppo_total_timesteps": 100_000,
        "baseline_training_layout": "sequential",
        "baseline_rollout_steps_per_env": None,
        "candidate_training_layout": "interleaved",
        "candidate_rollout_steps_per_env": 384,
        "slippage_std": 0.0,
    }
    for field, value in expected.items():
        if getattr(spec, field) != value:
            raise SystemExit(f"sealed evaluator authority drift: {field}")
    for field in (
        "candidate_training_authorized",
        "economic_result_inspected",
        "final_test_accessed",
        "shared_cash_profitability_established",
        "production_eligible",
        "live_trading_authorized",
        "merge_authorized",
    ):
        if getattr(spec, field) is not False:
            raise SystemExit(f"research boundary drift: {field}")

    issue = _api(f"issues/{ISSUE_NUMBER}")
    if issue.get("state") != "open":
        raise SystemExit("Issue 638 must remain open during baseline publication")
    verify = _assert_run_success(EVALUATOR_VERIFY_RUN, label="Issue 632 verification")
    if verify.get("head_sha") != EVALUATOR_VERIFY_HARNESS:
        raise SystemExit("Issue 632 verification harness drift")
    jobs = _api(f"actions/runs/{EVALUATOR_VERIFY_RUN}/jobs?per_page=100").get("jobs")
    if not isinstance(jobs, list):
        raise SystemExit("Issue 632 verification jobs unavailable")
    job_conclusions = {
        item.get("name"): item.get("conclusion")
        for item in jobs
        if isinstance(item, dict)
    }
    for name in (
        "Draft PR authority",
        "Lean Core exact target",
        "Human Guide exact target",
    ):
        if job_conclusions.get(name) != "success":
            raise SystemExit(f"Issue 632 verification job drift: {name}")
    pr = _api(f"pulls/{EVALUATOR_PR}")
    if (
        pr.get("state") != "open"
        or pr.get("draft") is not True
        or (pr.get("head") or {}).get("sha") != EVALUATOR_HEAD
    ):
        raise SystemExit("Issue 632 Draft PR authority drift")

    _assert_run_success(EVALUATOR_SEAL_RUN, label="Issue 632 final seal")
    _assert_run_success(PROTOCOL_SEAL_RUN, label="Issue 629 protocol seal")
    _assert_source_authority()
    for args in (
        (
            EVALUATOR_PRIMARY_ID,
            EVALUATOR_PRIMARY_NAME,
            EVALUATOR_PRIMARY_API_DIGEST,
            EVALUATOR_SEAL_RUN,
        ),
        (
            EVALUATOR_FRESH_ID,
            EVALUATOR_FRESH_NAME,
            EVALUATOR_FRESH_API_DIGEST,
            EVALUATOR_SEAL_RUN,
        ),
        (
            EVALUATOR_AUDIT_ID,
            EVALUATOR_AUDIT_NAME,
            EVALUATOR_AUDIT_API_DIGEST,
            EVALUATOR_SEAL_RUN,
        ),
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
        (SOURCE_ID, SOURCE_NAME, SOURCE_API_DIGEST, SOURCE_RUN),
    ):
        _assert_artifact(args[0], name=args[1], digest=args[2], run_id=args[3])

    activation_count = _artifact_count(_activation_name(canonical_seed))
    result_count = _artifact_count(_result_name(canonical_seed))
    fresh_count = _artifact_count(_fresh_name(canonical_seed))
    _validate_slot_state(
        slot=slot,
        activation_count=activation_count,
        result_count=result_count,
        fresh_count=fresh_count,
    )


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SystemExit(f"JSON object expected: {path}")
    return value


def _validate_downloaded_authorities(
    *, successor: Path, evaluator_primary: Path, evaluator_fresh: Path
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
        raise SystemExit("Issue 632 primary/fresh authority bytes differ")
    if _sha256(primary_impl) != EVALUATOR_IMPLEMENTATION_SHA256:
        raise SystemExit("Issue 632 implementation SHA drift")
    if _sha256(primary_seal) != EVALUATOR_SEAL_SHA256:
        raise SystemExit("Issue 632 seal SHA drift")
    implementation = json.loads(primary_impl)
    seal = json.loads(primary_seal)
    if not isinstance(implementation, dict) or not isinstance(seal, dict):
        raise SystemExit("Issue 632 authority payload is invalid")
    if implementation.get("target_head") != EVALUATOR_HEAD:
        raise SystemExit("Issue 632 implementation target drift")
    if implementation.get("main_head") != MAIN_AUTHORITY:
        raise SystemExit("Issue 632 implementation main authority drift")
    if implementation.get("implementation_sha256") is not None:
        raise SystemExit("unexpected recursive implementation hash field")
    if seal.get("implementation_sha256") != EVALUATOR_IMPLEMENTATION_SHA256:
        raise SystemExit("Issue 632 seal implementation binding drift")
    for payload in (implementation, seal):
        for field in (
            "real_dataset_loaded",
            "ppo_training_performed",
            "economic_evaluation_performed",
            "economic_result_inspected",
            "unused_data_accessed",
            "final_test_accessed",
            "shared_cash_profitability_established",
            "production_eligible",
            "live_trading_authorized",
            "merge_authorized",
        ):
            if payload.get(field) is not False:
                raise SystemExit(f"Issue 632 seal boundary drift: {field}")

    published = inspect_published_market_dataset_artifact(successor / "dataset")
    if published.artifact_digest != DATASET_ARTIFACT_DIGEST:
        raise SystemExit("Dataset artifact digest drift")
    dataset = load_market_dataset_artifact(successor / "dataset")
    if dataset.dataset_id != DATASET_ID:
        raise SystemExit("Dataset ID drift")
    plan_payload = _load_json(successor / "study" / "plan.json")
    if content_digest(plan_payload) != STUDY_DIGEST:
        raise SystemExit("P&L-free Study plan digest drift")
    return dataset


def publish_seed(
    *,
    seed: object,
    successor: Path,
    evaluator_primary: Path,
    evaluator_fresh: Path,
    output: Path,
) -> None:
    canonical_seed = _validate_seed(seed)
    authority_check(seed=canonical_seed, slot="claimed")
    dataset = _validate_downloaded_authorities(
        successor=successor,
        evaluator_primary=evaluator_primary,
        evaluator_fresh=evaluator_fresh,
    )
    from trade_rl.artifacts.hashing import content_digest
    from trade_rl.evaluation.experiments.ppo_interleaved_evaluation import (
        canonical_ppo_interleaved_evaluator_spec,
    )

    spec = canonical_ppo_interleaved_evaluator_spec()
    evidence = evaluate_baseline_seed(dataset, spec, seed=canonical_seed)
    evidence_bytes = canonical_baseline_evidence_bytes(evidence, seed=canonical_seed)
    evidence_sha = _sha256(evidence_bytes)
    evidence_digest = content_digest(evidence.to_payload())
    provenance = {
        "schema_version": "issue638_ppo_sequential_baseline_provenance_v1",
        "issue_number": ISSUE_NUMBER,
        "seed": canonical_seed,
        "arm": "baseline",
        "training_layout": "sequential",
        "rollout_steps_per_env": None,
        "evaluator_head": EVALUATOR_HEAD,
        "evaluator_seal_run_id": EVALUATOR_SEAL_RUN,
        "evaluator_implementation_sha256": EVALUATOR_IMPLEMENTATION_SHA256,
        "evaluator_seal_sha256": EVALUATOR_SEAL_SHA256,
        "protocol_head": PROTOCOL_HEAD,
        "protocol_digest": PROTOCOL_DIGEST,
        "protocol_seal_run_id": PROTOCOL_SEAL_RUN,
        "source_run_id": SOURCE_RUN,
        "source_artifact_id": SOURCE_ID,
        "dataset_id": DATASET_ID,
        "dataset_artifact_digest": DATASET_ARTIFACT_DIGEST,
        "study_digest": STUDY_DIGEST,
        "evidence_sha256": evidence_sha,
        "evidence_digest": evidence_digest,
        "candidate_training_authorized": False,
        "economic_result_interpreted": False,
        "final_test_accessed": False,
        "production_eligible": False,
        "live_trading_authorized": False,
        "merge_authorized": False,
    }
    output.mkdir(parents=True, exist_ok=False)
    (output / "evidence.json").write_bytes(evidence_bytes)
    (output / "provenance.json").write_bytes(_canonical(provenance))
    print(f"ISSUE638_BASELINE_SEED_{canonical_seed}_PUBLISHED=true")
    print(f"EVIDENCE_SHA256={evidence_sha}")
    print("CANDIDATE_TRAINING_AUTHORIZED=false")
    print("ECONOMIC_RESULT_INTERPRETED=false")


def claim_seed(
    *,
    seed: object,
    run_id: object,
    run_attempt: object,
    helper_head: object,
    helper_blob_sha: object,
    output: Path,
) -> None:
    canonical_seed = _validate_seed(seed)
    authority_check(seed=canonical_seed, slot="empty")
    raw = activation_claim_bytes(
        seed=canonical_seed,
        run_id=run_id,
        run_attempt=run_attempt,
        helper_head=helper_head,
        helper_blob_sha=helper_blob_sha,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    output.write_bytes(raw)
    print(f"ISSUE638_BASELINE_SEED_{canonical_seed}_ACTIVATION_CLAIM_WRITTEN=true")
    print("REAL_DATASET_LOADED=false")
    print("PPO_TRAINING_PERFORMED=false")
    print("CANDIDATE_TRAINING_AUTHORIZED=false")


def verify_seed(*, seed: object, published: Path, output: Path) -> None:
    canonical_seed = _validate_seed(seed)
    authority_check(seed=canonical_seed, slot="published")
    from trade_rl.artifacts.hashing import content_digest
    from trade_rl.evaluation.experiments.ppo_interleaved_evidence_codec import (
        load_ppo_seed_evidence,
    )

    evidence_path = published / "evidence.json"
    provenance_path = published / "provenance.json"
    evidence = load_ppo_seed_evidence(evidence_path)
    _validate_baseline_evidence(evidence, seed=canonical_seed)
    provenance = _load_json(provenance_path)
    evidence_bytes = evidence_path.read_bytes()
    if (
        canonical_baseline_evidence_bytes(evidence, seed=canonical_seed)
        != evidence_bytes
    ):
        raise SystemExit("published baseline evidence is not canonical")
    evidence_sha = _sha256(evidence_bytes)
    evidence_digest = content_digest(evidence.to_payload())
    required = {
        "issue_number": ISSUE_NUMBER,
        "seed": canonical_seed,
        "arm": "baseline",
        "training_layout": "sequential",
        "rollout_steps_per_env": None,
        "evaluator_head": EVALUATOR_HEAD,
        "evaluator_seal_run_id": EVALUATOR_SEAL_RUN,
        "evaluator_implementation_sha256": EVALUATOR_IMPLEMENTATION_SHA256,
        "evaluator_seal_sha256": EVALUATOR_SEAL_SHA256,
        "protocol_head": PROTOCOL_HEAD,
        "protocol_digest": PROTOCOL_DIGEST,
        "protocol_seal_run_id": PROTOCOL_SEAL_RUN,
        "source_run_id": SOURCE_RUN,
        "source_artifact_id": SOURCE_ID,
        "dataset_id": DATASET_ID,
        "dataset_artifact_digest": DATASET_ARTIFACT_DIGEST,
        "study_digest": STUDY_DIGEST,
        "evidence_sha256": evidence_sha,
        "evidence_digest": evidence_digest,
        "candidate_training_authorized": False,
        "economic_result_interpreted": False,
        "final_test_accessed": False,
        "production_eligible": False,
        "live_trading_authorized": False,
        "merge_authorized": False,
    }
    for field, value in required.items():
        if provenance.get(field) != value:
            raise SystemExit(f"published baseline provenance drift: {field}")
    verification = {
        "schema_version": "issue638_ppo_sequential_baseline_verification_v1",
        "issue_number": ISSUE_NUMBER,
        "seed": canonical_seed,
        "evidence_sha256": evidence_sha,
        "evidence_digest": evidence_digest,
        "arm": "baseline",
        "training_layout": "sequential",
        "candidate_training_authorized": False,
        "ppo_retrained": False,
        "verified": True,
        "final_test_accessed": False,
        "production_eligible": False,
        "live_trading_authorized": False,
        "merge_authorized": False,
    }
    output.mkdir(parents=True, exist_ok=False)
    (output / "verification.json").write_bytes(_canonical(verification))
    print(f"ISSUE638_BASELINE_SEED_{canonical_seed}_FRESH_VERIFY=PASS")
    print("PPO_RETRAINED=false")
    print("CANDIDATE_TRAINING_AUTHORIZED=false")


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    authority = subparsers.add_parser("authority-check")
    authority.add_argument("--seed", type=int, required=True)
    authority.add_argument(
        "--slot", choices=("empty", "claimed", "published"), required=True
    )

    claim = subparsers.add_parser("claim-seed")
    claim.add_argument("--seed", type=int, required=True)
    claim.add_argument("--run-id", type=int, required=True)
    claim.add_argument("--run-attempt", type=int, required=True)
    claim.add_argument("--helper-head", required=True)
    claim.add_argument("--helper-blob-sha", required=True)
    claim.add_argument("--output", type=Path, required=True)

    publish = subparsers.add_parser("publish-seed")
    publish.add_argument("--seed", type=int, required=True)
    publish.add_argument("--successor", type=Path, required=True)
    publish.add_argument("--evaluator-primary", type=Path, required=True)
    publish.add_argument("--evaluator-fresh", type=Path, required=True)
    publish.add_argument("--output", type=Path, required=True)

    verify = subparsers.add_parser("verify-seed")
    verify.add_argument("--seed", type=int, required=True)
    verify.add_argument("--published", type=Path, required=True)
    verify.add_argument("--output", type=Path, required=True)

    args = parser.parse_args()
    if args.command == "authority-check":
        authority_check(seed=args.seed, slot=args.slot)
    elif args.command == "claim-seed":
        claim_seed(
            seed=args.seed,
            run_id=args.run_id,
            run_attempt=args.run_attempt,
            helper_head=args.helper_head,
            helper_blob_sha=args.helper_blob_sha,
            output=args.output,
        )
    elif args.command == "publish-seed":
        publish_seed(
            seed=args.seed,
            successor=args.successor,
            evaluator_primary=args.evaluator_primary,
            evaluator_fresh=args.evaluator_fresh,
            output=args.output,
        )
    elif args.command == "verify-seed":
        verify_seed(seed=args.seed, published=args.published, output=args.output)
    else:
        raise SystemExit("unsupported command")


if __name__ == "__main__":
    main()
