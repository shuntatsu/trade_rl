"""Fail-closed pre-compute authority reconstruction for Issue #610.

This verifier performs no candidate training and inspects no new economic result.
It binds the frozen baseline, Issue #607 implementation seal/fresh evidence, and
Issue #609 evaluation preregistration seal/fresh evidence, then resolves the
first executable candidate contract without executing it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from dataclasses import replace
from pathlib import Path
from typing import cast

from tools.tmp_issue610_ppo_global_btc_regime_evaluation import (
    EXPECTED_SEMANTIC_CHANGED_FIELDS,
    build_candidate_carrier_plan,
    candidate_run_config_from_resolved,
    validate_controlled_semantic_delta,
)
from trade_rl.artifacts.canonical import canonical_json_bytes
from trade_rl.artifacts.hashing import content_digest
from trade_rl.data import (
    inspect_published_market_dataset_artifact,
    load_market_dataset_artifact,
)
from trade_rl.evaluation.experiments import inspect_study, load_evidence_set
from trade_rl.evaluation.experiments.contracts import ResolvedRunConfig
from trade_rl.evaluation.runs import (
    build_candidate_run_provenance,
    resolve_candidate_run_spec,
)
from trade_rl.strategies.rl.ppo import (
    PPO_GLOBAL_BTC_REGIME_CONTEXT,
    PPO_GLOBAL_BTC_REGIME_OBSERVATION_SCHEMA,
    PPO_OBSERVATION_SCHEMA,
)

_EXPECTED_SYMBOLS = ("BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT")
_EXPECTED_SEEDS = (0, 1, 2, 3, 4)
_EVAL_BLOB_PATHS = (
    "trade_rl/evaluation/experiments/ppo_global_btc_regime_evaluation_prereg.py",
    "tests/evaluation/experiments/test_ppo_global_btc_regime_evaluation_prereg.py",
    "tests/evaluation/experiments/test_ppo_global_btc_regime_evaluation_authority.py",
    "tests/evaluation/experiments/test_ppo_global_btc_regime_evaluation_stage_separation.py",
)
_IMPL_FILES = {
    "implementation_or_guide": {
        "guide/content/meta/implementation-ppo.json",
        "guide/content/pages/implementation-ppo.md",
        "trade_rl/evaluation/experiments/contracts/run.py",
        "trade_rl/evaluation/experiments/delta.py",
        "trade_rl/evaluation/runs/candidate_suite.py",
        "trade_rl/evaluation/runs/config.py",
        "trade_rl/strategies/rl/ppo.py",
    },
    "permanent_oracle": {
        "tests/strategies/test_ppo_global_btc_regime.py",
        "tests/evaluation/test_issue607_candidate_config.py",
        "tests/evaluation/experiments/test_delta.py",
    },
}


def _env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"required environment variable is missing: {name}")
    return value


def _unique(root: Path, name: str) -> Path:
    matches = tuple(root.rglob(name))
    if len(matches) != 1:
        raise RuntimeError(
            f"expected exactly one {name} under {root}, found {len(matches)}"
        )
    return matches[0]


def _canonical_object(path: Path) -> tuple[dict[str, object], bytes]:
    raw = path.read_bytes()
    try:
        decoded = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"malformed JSON: {path}") from error
    if not isinstance(decoded, dict) or any(
        not isinstance(key, str) for key in decoded
    ):
        raise RuntimeError(f"JSON root is not a string-keyed object: {path}")
    if canonical_json_bytes(decoded) != raw:
        raise RuntimeError(f"JSON is not canonical: {path}")
    return cast(dict[str, object], decoded), raw


def _verify_embedded_digest(payload: dict[str, object], *, label: str) -> None:
    recorded = payload.get("content_digest")
    if not isinstance(recorded, str):
        raise RuntimeError(f"{label} content_digest missing")
    body = dict(payload)
    body.pop("content_digest")
    if content_digest(body) != recorded:
        raise RuntimeError(f"{label} content_digest mismatch")


def _require_false(payload: dict[str, object], *names: str) -> None:
    for name in names:
        if payload.get(name) is not False:
            raise RuntimeError(f"{name} must remain false")


def _verify_eval_authority(
    *, repo_root: Path, seal_root: Path, fresh_root: Path
) -> None:
    for name in ("protocol.json", "authority.json", "git-blobs.txt"):
        if (
            _unique(seal_root, name).read_bytes()
            != _unique(fresh_root, name).read_bytes()
        ):
            raise RuntimeError(f"Issue 609 seal/fresh byte mismatch: {name}")

    protocol, protocol_raw = _canonical_object(_unique(seal_root, "protocol.json"))
    protocol_digest = content_digest(protocol)
    if protocol_digest != _env("EVAL_PROTOCOL_DIGEST"):
        raise RuntimeError("Issue 609 protocol content digest mismatch")
    if hashlib.sha256(protocol_raw).hexdigest() != _env("EVAL_PROTOCOL_SHA256"):
        raise RuntimeError("Issue 609 protocol canonical JSON SHA-256 mismatch")

    authority, _ = _canonical_object(_unique(seal_root, "authority.json"))
    _verify_embedded_digest(authority, label="Issue 609 authority")
    expected = {
        "prereg_head": _env("EVAL_PREREG_HEAD"),
        "protocol_digest": _env("EVAL_PROTOCOL_DIGEST"),
        "protocol_json_sha256": _env("EVAL_PROTOCOL_SHA256"),
        "implementation_head": _env("IMPLEMENTATION_SHA"),
        "implementation_tree": _env("IMPLEMENTATION_TREE"),
        "implementation_index_digest": _env("IMPL_INDEX_DIGEST"),
        "source_dataset_id": _env("DATASET_ID"),
        "source_dataset_artifact_digest": _env("DATASET_ARTIFACT_DIGEST"),
        "source_study_digest": _env("STUDY_DIGEST"),
        "baseline_evidence_fingerprint": _env("BASELINE_FP"),
    }
    for key, value in expected.items():
        if authority.get(key) != value:
            raise RuntimeError(f"Issue 609 authority mismatch: {key}")
    _require_false(
        authority,
        "economic_execution_authorized",
        "economic_result_inspected",
        "final_test_access_authorized",
        "production_eligible",
        "live_trading_authorized",
        "merge_authorized",
    )

    expected_lines = []
    for path in _EVAL_BLOB_PATHS:
        blob = subprocess.check_output(
            [
                "git",
                "-C",
                str(repo_root),
                "rev-parse",
                f"{_env('EVAL_PREREG_HEAD')}:{path}",
            ],
            text=True,
        ).strip()
        expected_lines.append(f"{blob} {path}\n")
    if _unique(seal_root, "git-blobs.txt").read_text(encoding="utf-8") != "".join(
        expected_lines
    ):
        raise RuntimeError("Issue 609 git-blobs authority mismatch")


def _verify_impl_authority(
    *, repo_root: Path, seal_root: Path, fresh_root: Path
) -> None:
    for name in ("implementation-index.json", "authority.json"):
        if (
            _unique(seal_root, name).read_bytes()
            != _unique(fresh_root, name).read_bytes()
        ):
            raise RuntimeError(f"Issue 607 seal/fresh byte mismatch: {name}")

    index, index_raw = _canonical_object(
        _unique(seal_root, "implementation-index.json")
    )
    _verify_embedded_digest(index, label="Issue 607 implementation index")
    if index.get("content_digest") != _env("IMPL_INDEX_DIGEST"):
        raise RuntimeError("Issue 607 implementation index digest mismatch")
    if index.get("canonical_implementation_head") != _env("IMPLEMENTATION_SHA"):
        raise RuntimeError("Issue 607 implementation head mismatch")
    if index.get("canonical_implementation_tree_sha1") != _env("IMPLEMENTATION_TREE"):
        raise RuntimeError("Issue 607 implementation tree mismatch")

    files = index.get("files")
    if not isinstance(files, list):
        raise RuntimeError("Issue 607 implementation file index malformed")
    observed: dict[str, set[str]] = {key: set() for key in _IMPL_FILES}
    for entry in files:
        if not isinstance(entry, dict):
            raise RuntimeError("Issue 607 implementation file entry malformed")
        category = entry.get("category")
        path = entry.get("path")
        if (
            not isinstance(category, str)
            or category not in observed
            or not isinstance(path, str)
        ):
            raise RuntimeError("Issue 607 implementation file identity malformed")
        observed[category].add(path)
        raw = (repo_root / path).read_bytes()
        if hashlib.sha256(raw).hexdigest() != entry.get("sha256"):
            raise RuntimeError(f"Issue 607 file SHA-256 mismatch: {path}")
        blob = subprocess.check_output(
            [
                "git",
                "-C",
                str(repo_root),
                "rev-parse",
                f"{_env('IMPLEMENTATION_SHA')}:{path}",
            ],
            text=True,
        ).strip()
        if blob != entry.get("git_blob_sha1"):
            raise RuntimeError(f"Issue 607 git blob mismatch: {path}")
    if observed != _IMPL_FILES:
        raise RuntimeError("Issue 607 implementation file roster mismatch")

    authority, _ = _canonical_object(_unique(seal_root, "authority.json"))
    _verify_embedded_digest(authority, label="Issue 607 authority")
    if authority.get("canonical_implementation_head") != _env("IMPLEMENTATION_SHA"):
        raise RuntimeError("Issue 607 seal head mismatch")
    if authority.get("canonical_implementation_tree_sha1") != _env(
        "IMPLEMENTATION_TREE"
    ):
        raise RuntimeError("Issue 607 seal tree mismatch")
    if authority.get("implementation_index_content_digest") != _env(
        "IMPL_INDEX_DIGEST"
    ):
        raise RuntimeError("Issue 607 seal index digest mismatch")
    if (
        authority.get("implementation_index_sha256")
        != hashlib.sha256(index_raw).hexdigest()
    ):
        raise RuntimeError("Issue 607 seal index byte hash mismatch")
    _require_false(
        authority,
        "economic_factor_effect_executed",
        "economic_result_inspected",
        "final_test_access_authorized",
        "production_eligible",
        "live_trading_authorized",
        "merge_authorized",
    )


def _verify_baseline_fresh_binding(fresh_root: Path) -> None:
    report_path = _unique(fresh_root, "prereg-baseline-binding.json")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if not isinstance(report, dict):
        raise RuntimeError("baseline fresh binding report malformed")
    expected = {
        "baseline_artifact_id": int(_env("BASELINE_ID")),
        "baseline_artifact_digest": _env("BASELINE_DIGEST"),
        "dataset_id": _env("DATASET_ID"),
        "dataset_artifact_digest": _env("DATASET_ARTIFACT_DIGEST"),
        "study_digest": _env("STUDY_DIGEST"),
        "baseline_evidence_fingerprint": _env("BASELINE_FP"),
        "prereg_was_plan_only": True,
        "baseline_uses_exact_preregistered_plan": True,
        "profitability_interpreted": False,
        "winner_interpreted": False,
    }
    for key, value in expected.items():
        if report.get(key) != value:
            raise RuntimeError(f"baseline fresh binding mismatch: {key}")


def _build_report(
    *, baseline_root: Path
) -> tuple[dict[str, object], dict[str, object], dict[str, object], dict[str, object]]:
    dataset = load_market_dataset_artifact(baseline_root / "dataset")
    artifact = inspect_published_market_dataset_artifact(baseline_root / "dataset")
    snapshot = inspect_study(baseline_root / "study")
    loaded = load_evidence_set(baseline_root / "study" / "baseline" / "evidence")

    if dataset.dataset_id != _env("DATASET_ID"):
        raise RuntimeError("baseline Dataset ID mismatch")
    if artifact.artifact_digest != _env("DATASET_ARTIFACT_DIGEST"):
        raise RuntimeError("baseline Dataset artifact digest mismatch")
    if snapshot.plan.digest != _env("STUDY_DIGEST"):
        raise RuntimeError("baseline Study digest mismatch")
    if loaded.evidence.fingerprint != _env("BASELINE_FP"):
        raise RuntimeError("baseline EvidenceSet fingerprint mismatch")
    if (
        snapshot.baseline is None
        or snapshot.experiment_sequences
        or snapshot.terminal_sequences
        or snapshot.frozen
    ):
        raise RuntimeError("baseline Study state is not baseline-only mutable state")
    if (
        snapshot.plan.symbols != _EXPECTED_SYMBOLS
        or snapshot.plan.ppo_seeds != _EXPECTED_SEEDS
    ):
        raise RuntimeError("baseline symbol/seed roster mismatch")
    if (
        tuple(sorted(loaded.runs)) != _EXPECTED_SEEDS
        or loaded.evidence.ppo_seeds != _EXPECTED_SEEDS
    ):
        raise RuntimeError("baseline EvidenceSet seed roster mismatch")
    if snapshot.plan.n_bootstrap != 2_000 or snapshot.plan.bootstrap_seed != 1_729:
        raise RuntimeError("baseline bootstrap authority mismatch")

    baseline = snapshot.plan.baseline_config
    if baseline.schema_version != "resolved_run_config_v2":
        raise RuntimeError("baseline resolved schema mismatch")
    if baseline.ppo_observation_schema != PPO_OBSERVATION_SCHEMA:
        raise RuntimeError("baseline PPO observation schema mismatch")
    if (
        baseline.ppo_global_feature_names != ()
        or baseline.ppo_global_context is not None
    ):
        raise RuntimeError("baseline unexpectedly contains global PPO context")
    if baseline.ppo_total_timesteps != 100_000:
        raise RuntimeError("baseline PPO timestep budget mismatch")

    candidate = replace(
        baseline,
        schema_version="resolved_run_config_v3",
        ppo_observation_schema=PPO_GLOBAL_BTC_REGIME_OBSERVATION_SCHEMA,
        ppo_global_context=PPO_GLOBAL_BTC_REGIME_CONTEXT,
    )
    baseline_payload = baseline.to_payload()
    candidate_payload = candidate.to_payload()
    baseline_payload.pop("ppo_seed")
    candidate_payload.pop("ppo_seed")
    changed, violations = validate_controlled_semantic_delta(
        baseline_payload, candidate_payload
    )
    if changed != EXPECTED_SEMANTIC_CHANGED_FIELDS or violations:
        raise RuntimeError(f"candidate semantic delta invalid: {violations}")

    raw_candidate = candidate_run_config_from_resolved(candidate)
    spec = resolve_candidate_run_spec(
        dataset,
        dataset_artifact_schema=artifact.schema_version,
        dataset_artifact_digest=artifact.artifact_digest,
        config=raw_candidate,
        execution_overlay=baseline.execution_overlay,
    )
    resolved_again = ResolvedRunConfig.from_candidate_spec(spec)
    if resolved_again != candidate:
        raise RuntimeError(
            "executable candidate resolution differs from frozen candidate"
        )

    provenance = build_candidate_run_provenance()
    implementation_digest = provenance.get("implementation_digest")
    runtime_digest = provenance.get("runtime_environment_digest")
    if not isinstance(implementation_digest, str) or not isinstance(
        runtime_digest, str
    ):
        raise RuntimeError("candidate provenance digest malformed")
    carrier = build_candidate_carrier_plan(
        source_plan=snapshot.plan,
        candidate_config=candidate,
        implementation_digest=implementation_digest,
        runtime_environment_digest=runtime_digest,
    )
    if carrier.digest == snapshot.plan.digest:
        raise RuntimeError("candidate carrier unexpectedly reused source Study digest")

    report: dict[str, object] = {
        "schema_version": "issue610_precompute_authority_v2",
        "implementation_git_sha": _env("IMPLEMENTATION_SHA"),
        "implementation_tree_sha": _env("IMPLEMENTATION_TREE"),
        "helper_git_sha": _env("HELPER_SHA"),
        "implementation_index_digest": _env("IMPL_INDEX_DIGEST"),
        "evaluation_prereg_head": _env("EVAL_PREREG_HEAD"),
        "evaluation_protocol_digest": _env("EVAL_PROTOCOL_DIGEST"),
        "implementation_digest": implementation_digest,
        "runtime_environment_digest": runtime_digest,
        "dataset_id": dataset.dataset_id,
        "dataset_artifact_digest": artifact.artifact_digest,
        "source_study_digest": snapshot.plan.digest,
        "baseline_evidence_fingerprint": loaded.evidence.fingerprint,
        "candidate_carrier_study_digest": carrier.digest,
        "semantic_changed_fields": [list(path) for path in changed],
        "symbols": list(snapshot.plan.symbols),
        "ppo_seeds": list(snapshot.plan.ppo_seeds),
        "ppo_total_timesteps": candidate.ppo_total_timesteps,
        "n_bootstrap": snapshot.plan.n_bootstrap,
        "bootstrap_seed": snapshot.plan.bootstrap_seed,
        "prior_issue610_economic_artifacts": 0,
        "baseline_retrained": False,
        "candidate_trained": False,
        "economic_values_interpreted": False,
        "final_test_accessed": False,
        "production_eligible": False,
        "live_trading_authorized": False,
        "merge_authorized": False,
    }
    report["content_digest"] = content_digest(report)
    return report, baseline.to_payload(), candidate.to_payload(), carrier.to_payload()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--baseline-fresh-root", type=Path, required=True)
    parser.add_argument("--impl-seal-root", type=Path, required=True)
    parser.add_argument("--impl-fresh-root", type=Path, required=True)
    parser.add_argument("--eval-seal-root", type=Path, required=True)
    parser.add_argument("--eval-fresh-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    _verify_impl_authority(
        repo_root=args.repo_root,
        seal_root=args.impl_seal_root,
        fresh_root=args.impl_fresh_root,
    )
    _verify_eval_authority(
        repo_root=args.repo_root,
        seal_root=args.eval_seal_root,
        fresh_root=args.eval_fresh_root,
    )
    _verify_baseline_fresh_binding(args.baseline_fresh_root)
    report, baseline, candidate, carrier = _build_report(
        baseline_root=args.baseline_root
    )

    args.output_root.mkdir(parents=True, exist_ok=True)
    payloads = {
        "authority.json": report,
        "baseline-resolved.json": baseline,
        "candidate-resolved.json": candidate,
        "candidate-carrier-study.json": carrier,
    }
    for name, payload in payloads.items():
        (args.output_root / name).write_bytes(canonical_json_bytes(payload))
    print("ISSUE610_PRECOMPUTE_AUTHORITY_OK=true")
    print(f"CANDIDATE_CARRIER_STUDY_DIGEST={report['candidate_carrier_study_digest']}")
    print(f"AUTHORITY_CONTENT_DIGEST={report['content_digest']}")


if __name__ == "__main__":
    main()
