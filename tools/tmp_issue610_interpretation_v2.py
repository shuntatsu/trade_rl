"""Result-blind Issue 610 interpretation hardening.

This module is fixed before candidate economic evidence is inspected. It does not
train PPO. It strengthens only interpretation completeness: malformed termination
evidence is INVALID, while genuine new termination is a development gate failure;
and absolute candidate PPO profitability is published per symbol plus cross-symbol.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from statistics import median
from typing import Any, cast

from trade_rl.artifacts.canonical import canonical_json_bytes
from trade_rl.artifacts.hashing import content_digest
from trade_rl.evaluation.experiments.analysis import compare_evidence_sets
from tools.tmp_issue610_exact_once_orchestrator import (
    ISSUE_NUMBER,
    PRECOMPUTE_RUN_ID,
    SEEDS,
    SYMBOLS,
    UNAFFECTED_STRATEGIES,
    _load_canonical_object,
    _run_return_matrix,
    _strategy_entry,
    _strict_source,
    _termination_count,
    _termination_reasons,
    install_artifact_bridge,
    validate_precompute_authority,
)
from tools.tmp_issue610_ppo_global_btc_regime_evaluation import (
    EXPECTED_SEMANTIC_CHANGED_FIELDS,
    evaluate_frozen_gate,
    validate_candidate_ppo_returns,
    validate_controlled_semantic_delta,
    validate_unaffected_raw_returns,
)

RESULT_SCHEMA = "issue610_development_evaluation_result_v2"
ABSOLUTE_DIAGNOSTIC_SCHEMA = "issue610_absolute_candidate_ppo_diagnostic_v1"
RECOVERY_BINDING_SCHEMA = "issue610_candidate_recovery_binding_v1"


def classify_termination_evidence(
    baseline: Any,
    candidate: Any,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return (new-termination gate failures, malformed-evidence violations)."""

    new_termination: list[str] = []
    invalid: list[str] = []
    for seed in SEEDS:
        baseline_run = baseline.runs.get(seed)
        candidate_run = candidate.runs.get(seed)
        if baseline_run is None or candidate_run is None:
            invalid.append(f"termination run missing: seed={seed}")
            continue
        for symbol in SYMBOLS:
            label = f"seed={seed} symbol={symbol}"
            try:
                baseline_entry = _strategy_entry(baseline_run, symbol, "ppo")
                candidate_entry = _strategy_entry(candidate_run, symbol, "ppo")
                baseline_count = _termination_count(baseline_entry)
                candidate_count = _termination_count(candidate_entry)
                baseline_reasons = Counter(_termination_reasons(baseline_entry))
                candidate_reasons = Counter(_termination_reasons(candidate_entry))
            except ValueError as error:
                invalid.append(f"termination evidence malformed: {label}: {error}")
                continue
            if candidate_count > baseline_count:
                new_termination.append(f"new PPO termination count: {label}")
            if any(
                count > baseline_reasons.get(reason, 0)
                for reason, count in candidate_reasons.items()
            ):
                new_termination.append(f"new PPO termination reason: {label}")
    return tuple(new_termination), tuple(invalid)


def _finite_total_return(entry: Mapping[str, object]) -> float:
    metrics = entry.get("metrics")
    if not isinstance(metrics, Mapping):
        raise ValueError("PPO strategy metrics malformed")
    value = metrics.get("total_return")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("PPO total_return malformed")
    resolved = float(value)
    if not math.isfinite(resolved):
        raise ValueError("PPO total_return non-finite")
    return resolved


def absolute_candidate_ppo_diagnostic(
    candidate: Any,
    comparison: Mapping[str, object],
) -> dict[str, object]:
    """Publish the preregistered absolute-profit diagnostic without gating."""

    by_symbol: dict[str, object] = {}
    symbol_medians: list[float] = []
    for symbol in SYMBOLS:
        by_seed: dict[str, float] = {}
        values: list[float] = []
        for seed in SEEDS:
            run = candidate.runs.get(seed)
            if run is None:
                raise ValueError(f"candidate PPO run missing: seed={seed}")
            value = _finite_total_return(_strategy_entry(run, symbol, "ppo"))
            by_seed[str(seed)] = value
            values.append(value)
        symbol_median = float(median(values))
        symbol_medians.append(symbol_median)
        by_symbol[symbol] = {
            "by_seed_total_return": by_seed,
            "median_candidate_total_return": symbol_median,
        }

    cross_symbol = comparison.get("cross_symbol")
    if not isinstance(cross_symbol, Mapping):
        raise ValueError("comparison cross_symbol malformed")
    ppo_cross = cross_symbol.get("ppo")
    if not isinstance(ppo_cross, Mapping):
        raise ValueError("comparison PPO cross-symbol diagnostic missing")
    recorded = ppo_cross.get("median_candidate_total_return")
    if isinstance(recorded, bool) or not isinstance(recorded, (int, float)):
        raise ValueError("comparison PPO candidate median malformed")
    recorded_median = float(recorded)
    if not math.isfinite(recorded_median):
        raise ValueError("comparison PPO candidate median non-finite")
    computed_median = float(median(symbol_medians))
    if recorded_median != computed_median:
        raise ValueError("comparison PPO absolute candidate median mismatch")

    return {
        "schema_version": ABSOLUTE_DIAGNOSTIC_SCHEMA,
        "gates_development_decision": False,
        "by_symbol": by_symbol,
        "cross_symbol": {
            "symbol_count": len(symbol_medians),
            "positive_symbol_count": sum(value > 0.0 for value in symbol_medians),
            "negative_symbol_count": sum(value < 0.0 for value in symbol_medians),
            "zero_symbol_count": sum(value == 0.0 for value in symbol_medians),
            "median_candidate_total_return": computed_median,
            "worst_candidate_total_return": min(symbol_medians),
            "best_candidate_total_return": max(symbol_medians),
        },
    }


def validate_recovery_binding(
    candidate_root: Path,
    candidate_authority: Mapping[str, object],
) -> tuple[str, ...]:
    """Validate the candidate's result-blind recovery binding before interpretation."""

    binding_path = candidate_root / "recovery-binding.json"
    recovery_path = candidate_root / "recovery-authority.json"
    violations: list[str] = []
    try:
        binding, _ = _load_canonical_object(binding_path)
        recovery, _ = _load_canonical_object(recovery_path)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        return (f"candidate recovery authority malformed: {error}",)

    expected: dict[str, object] = {
        "schema_version": RECOVERY_BINDING_SCHEMA,
        "issue_number": ISSUE_NUMBER,
        "candidate_evidence_fingerprint": candidate_authority.get(
            "candidate_evidence_fingerprint"
        ),
        "candidate_authority_content_digest": candidate_authority.get("content_digest"),
        "recovery_authority_content_digest": recovery.get("content_digest"),
        "failed_run_candidate_training_performed": False,
        "recovery_candidate_training_performed": True,
        "economic_values_interpreted": False,
        "final_test_accessed": False,
        "production_eligible": False,
        "live_trading_authorized": False,
        "merge_authorized": False,
    }
    for field, value in expected.items():
        if type(binding.get(field)) is not type(value) or binding.get(field) != value:
            violations.append(f"candidate recovery binding mismatch: {field}")
    if recovery.get("recovery_candidate_execution_authorized") is not True:
        violations.append("recovery authority did not authorize candidate execution")
    if recovery.get("failed_run_candidate_training_performed") is not False:
        violations.append("recovery authority says prior run trained candidate")
    if recovery.get("failed_run_candidate_artifact_published") is not False:
        violations.append("recovery authority says prior run published candidate")
    return tuple(violations)


def interpret_candidate_v2(
    *,
    source_root: Path,
    candidate_root: Path,
    precompute_path: Path,
    output_root: Path,
    artifact_module_path: Path,
    interpretation_run_id: int,
    precompute_artifact_id: int,
    precompute_artifact_digest: str,
    candidate_artifact_id: int,
    candidate_artifact_digest: str,
) -> dict[str, object]:
    """Apply the sealed Issue 609 decision with complete diagnostic semantics."""

    install_artifact_bridge(artifact_module_path)
    precompute, _ = _load_canonical_object(precompute_path)
    precompute_violations = validate_precompute_authority(precompute)
    if precompute_violations:
        raise RuntimeError("; ".join(precompute_violations))

    _dataset, snapshot, baseline = _strict_source(source_root)
    candidate_authority, _ = _load_canonical_object(
        candidate_root / "candidate-authority.json"
    )
    if candidate_authority.get("candidate_training_performed") is not True:
        raise RuntimeError("candidate authority does not prove candidate training")
    if candidate_authority.get("economic_values_interpreted") is not False:
        raise RuntimeError("candidate artifact was already economically interpreted")
    if candidate_authority.get("final_test_accessed") is not False:
        raise RuntimeError("candidate artifact crossed final-test boundary")
    if candidate_authority.get("precompute_authority_content_digest") != precompute.get(
        "content_digest"
    ):
        raise RuntimeError("candidate/precompute authority binding mismatch")
    recovery_violations = validate_recovery_binding(candidate_root, candidate_authority)
    if recovery_violations:
        raise RuntimeError("; ".join(recovery_violations))

    candidate = __import__(
        "trade_rl.evaluation.experiments.evidence",
        fromlist=["load_evidence_set"],
    ).load_evidence_set(candidate_root / "evidence")
    if candidate.evidence.fingerprint != candidate_authority.get(
        "candidate_evidence_fingerprint"
    ):
        raise RuntimeError("candidate EvidenceSet authority mismatch")

    baseline_semantic = dict(baseline.semantic_config)
    candidate_semantic = dict(candidate.semantic_config)
    changed_paths, semantic_violations = validate_controlled_semantic_delta(
        baseline_semantic,
        candidate_semantic,
    )
    if changed_paths != EXPECTED_SEMANTIC_CHANGED_FIELDS:
        semantic_violations = tuple(semantic_violations) + (
            "interpreted semantic changed paths drift",
        )

    baseline_matrix = _run_return_matrix(baseline)
    candidate_matrix = _run_return_matrix(candidate)
    unaffected_violations = validate_unaffected_raw_returns(
        baseline_matrix,
        candidate_matrix,
        seeds=SEEDS,
        symbols=SYMBOLS,
        unaffected_strategies=UNAFFECTED_STRATEGIES,
    )
    ppo_violations = validate_candidate_ppo_returns(
        candidate_matrix,
        seeds=SEEDS,
        symbols=SYMBOLS,
    )
    new_termination, termination_validity = classify_termination_evidence(
        baseline,
        candidate,
    )

    comparison = compare_evidence_sets(
        baseline.runs,
        candidate.runs,
        n_bootstrap=snapshot.plan.n_bootstrap,
        bootstrap_seed=snapshot.plan.bootstrap_seed,
        schema_version="controlled_evidence_comparison_v2",
    )
    recorded_comparison_digest = comparison.get("analysis_digest")
    comparison_unsigned = dict(comparison)
    comparison_unsigned.pop("analysis_digest", None)
    if recorded_comparison_digest != content_digest(comparison_unsigned):
        raise RuntimeError("comparison analysis digest mismatch")

    validity_violations = (
        tuple(semantic_violations)
        + tuple(ppo_violations)
        + tuple(termination_validity)
    )
    gate = evaluate_frozen_gate(
        comparison,
        seeds=SEEDS,
        symbols=SYMBOLS,
        no_new_termination=not new_termination,
        unaffected_raw_returns_equal=not unaffected_violations,
        validity_violations=validity_violations,
    )
    absolute_diagnostic = absolute_candidate_ppo_diagnostic(candidate, comparison)

    result: dict[str, object] = {
        "schema_version": RESULT_SCHEMA,
        "issue_number": ISSUE_NUMBER,
        "interpretation_run_id": interpretation_run_id,
        "precompute_run_id": PRECOMPUTE_RUN_ID,
        "precompute_artifact_id": precompute_artifact_id,
        "precompute_artifact_api_digest": precompute_artifact_digest,
        "precompute_authority_content_digest": precompute["content_digest"],
        "candidate_artifact_id": candidate_artifact_id,
        "candidate_artifact_api_digest": candidate_artifact_digest,
        "candidate_authority_content_digest": candidate_authority["content_digest"],
        "candidate_evidence_fingerprint": candidate.evidence.fingerprint,
        "baseline_evidence_fingerprint": baseline.evidence.fingerprint,
        "comparison_schema": comparison["schema_version"],
        "comparison_digest": recorded_comparison_digest,
        "comparison": comparison,
        "semantic_changed_paths": [list(path) for path in changed_paths],
        "semantic_violations": list(semantic_violations),
        "candidate_ppo_violations": list(ppo_violations),
        "unaffected_raw_return_violations": list(unaffected_violations),
        "new_termination_violations": list(new_termination),
        "termination_validity_violations": list(termination_validity),
        "decision_gate": gate,
        "decision": gate["decision"],
        "absolute_candidate_profitability_diagnostic": absolute_diagnostic,
        "development_acceptance_establishes_profitability": False,
        "baseline_retrained": False,
        "candidate_retrained_during_interpretation": False,
        "economic_values_interpreted": True,
        "final_test_accessed": False,
        "operational_eligibility_established": False,
        "production_eligible": False,
        "live_trading_authorized": False,
        "merge_authorized": False,
    }
    result["content_digest"] = content_digest(result)
    output_root.mkdir(parents=True, exist_ok=False)
    (output_root / "result.json").write_bytes(canonical_json_bytes(result))
    print("ISSUE610_INTERPRETATION_V2=PASS")
    print("DECISION=" + str(gate["decision"]))
    print("FINAL_TEST_ACCESSED=false")
    print("PRODUCTION_ELIGIBLE=false")
    print("LIVE_TRADING_AUTHORIZED=false")
    return result


def self_check() -> None:
    if RESULT_SCHEMA != "issue610_development_evaluation_result_v2":
        raise RuntimeError("result schema drift")
    if ABSOLUTE_DIAGNOSTIC_SCHEMA != "issue610_absolute_candidate_ppo_diagnostic_v1":
        raise RuntimeError("absolute diagnostic schema drift")
    if tuple(UNAFFECTED_STRATEGIES) != (
        "trend",
        "mean_reversion",
        "ridge24",
        "lightgbm24",
        "cash",
        "constant_long",
        "constant_short",
    ):
        raise RuntimeError("unaffected strategy roster drift")
    print("ISSUE610_INTERPRETATION_V2_SELF_CHECK=PASS")


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("self-check")
    interpret = sub.add_parser("interpret")
    interpret.add_argument("--source-root", type=Path, required=True)
    interpret.add_argument("--candidate-root", type=Path, required=True)
    interpret.add_argument("--precompute", type=Path, required=True)
    interpret.add_argument("--output-root", type=Path, required=True)
    interpret.add_argument("--artifact-module", type=Path, required=True)
    interpret.add_argument("--interpretation-run-id", type=int, required=True)
    interpret.add_argument("--precompute-artifact-id", type=int, required=True)
    interpret.add_argument("--precompute-artifact-digest", required=True)
    interpret.add_argument("--candidate-artifact-id", type=int, required=True)
    interpret.add_argument("--candidate-artifact-digest", required=True)
    args = parser.parse_args()
    if args.command == "self-check":
        self_check()
        return
    if args.command == "interpret":
        interpret_candidate_v2(
            source_root=args.source_root,
            candidate_root=args.candidate_root,
            precompute_path=args.precompute,
            output_root=args.output_root,
            artifact_module_path=args.artifact_module,
            interpretation_run_id=args.interpretation_run_id,
            precompute_artifact_id=args.precompute_artifact_id,
            precompute_artifact_digest=args.precompute_artifact_digest,
            candidate_artifact_id=args.candidate_artifact_id,
            candidate_artifact_digest=args.candidate_artifact_digest,
        )
        return
    raise AssertionError(args.command)


if __name__ == "__main__":
    main()
