"""Recover Experiment 0002 from the immutable comparison-complete failure artifact.

The original authorized execution completed the candidate EvidenceSet, controlled
verification, and persisted comparison, then failed in a research-only cross-check
because recursively frozen JSON is exposed as ``MappingProxyType`` rather than
``dict``. Recovery never executes a candidate and never recomputes the persisted
comparison. It revalidates the exact immutable evidence, independently replays the
preregistered decision rule, and appends only the missing decision plus a result
index that remains interpretation-sealed until independent post-verification.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path

from research.issue519_portable_exp002 import (
    EXPECTED_BASELINE_FINGERPRINT,
    EXPECTED_DATASET_ARTIFACT_DIGEST,
    EXPECTED_DATASET_ID,
    EXPECTED_EXP001_CANDIDATE_FINGERPRINT,
    EXPECTED_EXP001_COMPARISON_DIGEST,
    EXPECTED_EXP001_DECISION_DIGEST,
    EXPECTED_EXP001_DEFINITION_DIGEST,
    EXPECTED_EXP001_VERIFICATION_DIGEST,
    EXPECTED_IMPLEMENTATION_DIGEST,
    EXPECTED_RUNTIME_DIGEST,
    EXPECTED_SEEDS,
    EXPECTED_STUDY_DIGEST,
    EXPECTED_SYMBOLS,
)
from research.issue519_portable_exp002_execute import (
    EXPECTED_CHANGED_PATHS,
    EXPECTED_DEFINITION_DIGEST,
    EXPECTED_REQUESTED_CANDIDATE_DIGEST,
    _independent_analysis,
)
from trade_rl.evaluation.experiments import (
    ControlledVerificationStatus,
    ExperimentDecisionKind,
    decide_experiment,
    inspect_study,
)
from trade_rl.evaluation.experiments.inspection import _reconstruct
from trade_rl.evaluation.experiments.store import StudyStore

EXPECTED_CANDIDATE_FINGERPRINT = (
    "f5ec53197a2f487ba11cc01a78edd4c93df8a2f172095e5903ee079ce5e9d0d0"
)
ORIGINAL_EXECUTION_RUN_ID = 34739994457
ORIGINAL_EXECUTION_HEAD = "45dd0667d8449c636d29b1c38bf39e15f7b7f572"
FAILURE_ARTIFACT_ID = 10313103451
FAILURE_ARTIFACT_DIGEST = (
    "sha256:e8b2f4e2b69d196fa4af327942e4eb5ef1473ba5f2c29bfe2e3d175ef4b34e51"
)
PREREG_RUN_ID = 34738896077
PREREG_ARTIFACT_ID = 10311508057
PREREG_ARTIFACT_DIGEST = (
    "sha256:566880f04bd0ba686dbe6796aff40709c70986f966844d8e11e000c99ef31057"
)
PREREG_VERIFY_ARTIFACT_ID = 10311728838
PREREG_VERIFY_ARTIFACT_DIGEST = (
    "sha256:37494b90782fd18250c20a0b33fe24b09c7c3874bced4ab7759459d6d2c4ac73"
)


def _mapping(value: object, *, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise RuntimeError(f"{field} malformed")
    return value


def _number(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError(f"{field} malformed")
    resolved = float(value)
    if not math.isfinite(resolved):
        raise RuntimeError(f"{field} non-finite")
    return resolved


def _nonnegative_int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise RuntimeError(f"{field} malformed")
    return value


def crosscheck_persisted_comparison(
    comparison: object,
    independent: Mapping[str, object],
) -> None:
    """Cross-check frozen Mapping-backed comparison against the raw-return oracle."""

    factor_effect = _mapping(
        getattr(comparison, "factor_effect", None),
        field="persisted comparison factor_effect",
    )
    cross_symbol = _mapping(
        factor_effect.get("cross_symbol"),
        field="persisted comparison cross_symbol",
    )
    mean_reversion = _mapping(
        cross_symbol.get("mean_reversion"),
        field="persisted mean-reversion summary",
    )
    formal = _mapping(
        independent.get("mean_reversion_formal_inputs"),
        field="independent formal inputs",
    )
    if mean_reversion.get("positive_symbol_count") != formal.get(
        "positive_factor_effect_symbol_count"
    ):
        raise RuntimeError("persisted mean-reversion positive count differs from raw oracle")
    persisted_median = _number(
        mean_reversion.get("median_excess_total_return"),
        field="persisted mean-reversion median",
    )
    independent_median = _number(
        formal.get("median_excess_total_return"),
        field="independent mean-reversion median",
    )
    if not math.isclose(
        persisted_median,
        independent_median,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise RuntimeError("persisted mean-reversion median differs from raw oracle")


def _normalize_deterministic_effects(value: object) -> dict[str, object]:
    effects = _mapping(value, field="recovery deterministic effects")
    normalized: dict[str, object] = {}
    for strategy, raw_effect in effects.items():
        effect = _mapping(raw_effect, field=f"recovery deterministic effect {strategy}")
        if set(effect) != {"by_symbol", "aggregate"}:
            raise RuntimeError("recovery deterministic effect schema drift")
        by_symbol = effect.get("by_symbol")
        if not isinstance(by_symbol, dict):
            raise RuntimeError("recovery deterministic by-symbol evidence malformed")
        aggregate = dict(
            _mapping(
                effect.get("aggregate"),
                field=f"recovery deterministic aggregate {strategy}",
            )
        )
        aggregate.pop("symbol_count", None)
        normalized[strategy] = {
            "by_symbol": by_symbol,
            "aggregate": aggregate,
        }
    return normalized


def normalize_independent_for_postverify(
    independent: Mapping[str, object],
) -> dict[str, object]:
    """Normalize execution-side evidence to the fresh postverifier oracle schema."""

    baseline_fp = independent.get("baseline_fingerprint")
    candidate_fp = independent.get("candidate_fingerprint")
    if not isinstance(baseline_fp, str) or not isinstance(candidate_fp, str):
        raise RuntimeError("recovery independent evidence fingerprint malformed")

    unaffected = _nonnegative_int(
        independent.get("unaffected_raw_return_equality_checks"),
        field="unaffected raw-return equality checks",
    )
    deterministic = _normalize_deterministic_effects(
        independent.get("deterministic_effects")
    )
    formal = independent.get("mean_reversion_formal_inputs")
    if not isinstance(formal, dict):
        raise RuntimeError("recovery formal inputs malformed")
    costs = independent.get("candidate_cost_semantics")
    if not isinstance(costs, dict):
        raise RuntimeError("recovery candidate cost semantics malformed")
    formal_decision = independent.get("formal_decision")
    if not isinstance(formal_decision, str):
        raise RuntimeError("recovery formal decision malformed")
    if independent.get("ppo_cross_symbol_metrics_used_for_decision") is not False:
        raise RuntimeError("PPO cross-symbol metrics unexpectedly used for decision")

    ppo_by_seed = _mapping(
        independent.get("ppo_by_seed_symbol"),
        field="recovery PPO evidence",
    )
    if set(ppo_by_seed) != {str(seed) for seed in EXPECTED_SEEDS}:
        raise RuntimeError("PPO seed evidence roster drift")
    ppo_checks = 0
    expected_symbols = set(EXPECTED_SYMBOLS)
    for seed in EXPECTED_SEEDS:
        per_symbol = _mapping(
            ppo_by_seed.get(str(seed)),
            field=f"recovery PPO seed {seed}",
        )
        if set(per_symbol) != expected_symbols:
            raise RuntimeError("PPO symbol evidence roster drift")
        for symbol in EXPECTED_SYMBOLS:
            entry = _mapping(
                per_symbol.get(symbol),
                field=f"recovery PPO {seed}/{symbol}",
            )
            baseline_total = _number(
                entry.get("baseline_total_return"),
                field=f"PPO baseline total return {seed}/{symbol}",
            )
            candidate_total = _number(
                entry.get("candidate_total_return"),
                field=f"PPO candidate total return {seed}/{symbol}",
            )
            excess = _number(
                entry.get("excess_total_return"),
                field=f"PPO excess total return {seed}/{symbol}",
            )
            if candidate_total != baseline_total or excess != 0.0:
                raise RuntimeError("PPO recovery evidence is not exact-zero effect")
            ppo_checks += 1
    if ppo_checks != len(EXPECTED_SEEDS) * len(EXPECTED_SYMBOLS):
        raise RuntimeError("PPO exact-zero effect check count drift")

    return {
        "baseline_evidence_fingerprint": baseline_fp,
        "candidate_evidence_fingerprint": candidate_fp,
        "unaffected_raw_return_equality_checks": unaffected,
        "ppo_exact_zero_effect_checks": ppo_checks,
        "deterministic_effects": deterministic,
        "mean_reversion_formal_inputs": formal,
        "candidate_cost_semantics": costs,
        "formal_decision": formal_decision,
    }


def _exp1_binding(experiment: object) -> tuple[object, object, object, object, object]:
    definition = getattr(experiment, "definition", None)
    candidate = getattr(experiment, "candidate", None)
    verification = getattr(experiment, "verification", None)
    comparison = getattr(experiment, "comparison", None)
    decision = getattr(experiment, "decision", None)
    evidence = getattr(candidate, "evidence", None)
    return (
        getattr(definition, "digest", None),
        getattr(evidence, "fingerprint", None),
        getattr(verification, "digest", None),
        getattr(comparison, "digest", None),
        getattr(decision, "digest", None),
    )


def recover(root: Path) -> dict[str, object]:
    """Append only the missing Experiment 0002 decision and recovery-bound index."""

    study_root = root / "study"
    snapshot = inspect_study(study_root)
    if snapshot.plan.digest != EXPECTED_STUDY_DIGEST:
        raise RuntimeError("recovery Study digest drift")
    if snapshot.plan.dataset_id != EXPECTED_DATASET_ID:
        raise RuntimeError("recovery Dataset ID drift")
    if snapshot.plan.dataset_artifact_digest != EXPECTED_DATASET_ARTIFACT_DIGEST:
        raise RuntimeError("recovery Dataset artifact digest drift")
    if snapshot.plan.implementation_digest != EXPECTED_IMPLEMENTATION_DIGEST:
        raise RuntimeError("recovery implementation digest drift")
    if snapshot.plan.runtime_environment_digest != EXPECTED_RUNTIME_DIGEST:
        raise RuntimeError("recovery runtime environment digest drift")
    if snapshot.baseline is None or snapshot.baseline.fingerprint != EXPECTED_BASELINE_FINGERPRINT:
        raise RuntimeError("recovery baseline fingerprint drift")
    if snapshot.experiment_sequences != (1, 2) or snapshot.terminal_sequences != (1,):
        raise RuntimeError("recovery input is not Exp2 comparison-complete / decision-missing")
    if snapshot.frozen:
        raise RuntimeError("recovery Study unexpectedly frozen")

    state = _reconstruct(StudyStore(study_root))
    if len(state.experiments) != 2:
        raise RuntimeError("recovery Study does not contain exactly two Experiments")
    exp1, exp2 = state.experiments
    expected_exp1 = (
        EXPECTED_EXP001_DEFINITION_DIGEST,
        EXPECTED_EXP001_CANDIDATE_FINGERPRINT,
        EXPECTED_EXP001_VERIFICATION_DIGEST,
        EXPECTED_EXP001_COMPARISON_DIGEST,
        EXPECTED_EXP001_DECISION_DIGEST,
    )
    if _exp1_binding(exp1) != expected_exp1:
        raise RuntimeError("Experiment 0001 binding drift before recovery")

    if exp2.definition.digest != EXPECTED_DEFINITION_DIGEST:
        raise RuntimeError("recovery Experiment 0002 definition drift")
    if exp2.definition.candidate_requested_config_digest != EXPECTED_REQUESTED_CANDIDATE_DIGEST:
        raise RuntimeError("recovery Experiment 0002 requested config drift")
    if exp2.failure is not None:
        raise RuntimeError("recovery input contains Experiment 0002 failure evidence")
    if exp2.candidate is None:
        raise RuntimeError("recovery Experiment 0002 candidate evidence missing")
    if exp2.candidate.evidence.fingerprint != EXPECTED_CANDIDATE_FINGERPRINT:
        raise RuntimeError("recovery Experiment 0002 candidate fingerprint drift")
    if exp2.verification is None:
        raise RuntimeError("recovery Experiment 0002 verification missing")
    if exp2.verification.status is not ControlledVerificationStatus.CONTROLLED:
        raise RuntimeError("recovery Experiment 0002 verification is not CONTROLLED")
    if exp2.verification.changed_paths != EXPECTED_CHANGED_PATHS:
        raise RuntimeError("recovery Experiment 0002 changed paths drift")
    if exp2.comparison is None:
        raise RuntimeError("recovery Experiment 0002 comparison evidence missing")
    if exp2.decision is not None:
        raise RuntimeError("recovery input already contains Experiment 0002 decision")
    if (root / "portable-exp002-result-index.json").exists():
        raise RuntimeError("recovery input already contains Experiment 0002 result index")

    candidate_fingerprint = exp2.candidate.evidence.fingerprint
    verification_digest = exp2.verification.digest
    comparison_digest = exp2.comparison.digest

    independent = _independent_analysis(study_root)
    if independent.get("candidate_fingerprint") != candidate_fingerprint:
        raise RuntimeError("independent candidate fingerprint differs during recovery")
    crosscheck_persisted_comparison(exp2.comparison, independent)
    normalized_independent = normalize_independent_for_postverify(independent)
    decision_kind = ExperimentDecisionKind(str(independent["formal_decision"]))
    decision = decide_experiment(
        study_root,
        2,
        decision=decision_kind,
        rationale=(
            "Mechanical replay of the preregistered Experiment 0002 decision rule from "
            "the original bound candidate/verification/comparison evidence; performance "
            "values remain sealed until independent post-verification passes."
        ),
        decided_by="openai-gpt-5.6-sol-recovery",
        decided_at=datetime.now(timezone.utc),
    )

    final_snapshot = inspect_study(study_root)
    if final_snapshot.plan.digest != EXPECTED_STUDY_DIGEST:
        raise RuntimeError("Study digest changed during Experiment 0002 recovery")
    if final_snapshot.experiment_sequences != (1, 2):
        raise RuntimeError("Experiment sequence changed during recovery")
    if final_snapshot.terminal_sequences != (1, 2):
        raise RuntimeError("recovered Experiment 0002 did not become terminal")
    if final_snapshot.frozen:
        raise RuntimeError("Study was frozen unexpectedly during recovery")

    recovered_state = _reconstruct(StudyStore(study_root))
    recovered_exp1, recovered_exp2 = recovered_state.experiments
    if _exp1_binding(recovered_exp1) != expected_exp1:
        raise RuntimeError("Experiment 0001 evidence changed during recovery")
    if recovered_exp2.candidate is None or recovered_exp2.candidate.evidence.fingerprint != candidate_fingerprint:
        raise RuntimeError("Experiment 0002 candidate evidence changed during recovery")
    if recovered_exp2.verification is None or recovered_exp2.verification.digest != verification_digest:
        raise RuntimeError("Experiment 0002 verification changed during recovery")
    if recovered_exp2.comparison is None or recovered_exp2.comparison.digest != comparison_digest:
        raise RuntimeError("Experiment 0002 comparison changed during recovery")
    if recovered_exp2.decision is None or recovered_exp2.decision.digest != decision.digest:
        raise RuntimeError("Experiment 0002 decision reconstruction mismatch")

    result_index: dict[str, object] = {
        "schema_version": "canonical_m2_portable_exp002_result_index_v1",
        "issue_number": 519,
        "execution_workflow_run_id": ORIGINAL_EXECUTION_RUN_ID,
        "execution_helper_git_sha": ORIGINAL_EXECUTION_HEAD,
        "prereg_run_id": PREREG_RUN_ID,
        "prereg_artifact_id": PREREG_ARTIFACT_ID,
        "prereg_artifact_digest": PREREG_ARTIFACT_DIGEST,
        "prereg_verifier_artifact_id": PREREG_VERIFY_ARTIFACT_ID,
        "prereg_verifier_artifact_digest": PREREG_VERIFY_ARTIFACT_DIGEST,
        "source_failure_artifact_id": FAILURE_ARTIFACT_ID,
        "source_failure_artifact_digest": FAILURE_ARTIFACT_DIGEST,
        "recovery_workflow_run_id": int(os.environ["GITHUB_RUN_ID"]),
        "recovery_helper_git_sha": os.environ["GITHUB_SHA"],
        "study_digest": EXPECTED_STUDY_DIGEST,
        "baseline_evidence_fingerprint": EXPECTED_BASELINE_FINGERPRINT,
        "candidate_evidence_fingerprint": candidate_fingerprint,
        "dataset_id": EXPECTED_DATASET_ID,
        "dataset_artifact_digest": EXPECTED_DATASET_ARTIFACT_DIGEST,
        "implementation_digest": EXPECTED_IMPLEMENTATION_DIGEST,
        "runtime_environment_digest": EXPECTED_RUNTIME_DIGEST,
        "definition_digest": EXPECTED_DEFINITION_DIGEST,
        "verification_digest": verification_digest,
        "verification_status": recovered_exp2.verification.status.value,
        "changed_paths": [list(path) for path in recovered_exp2.verification.changed_paths],
        "comparison_digest": comparison_digest,
        "decision_digest": recovered_exp2.decision.digest,
        "decision": recovered_exp2.decision.decision.value,
        "independent_raw_evidence": normalized_independent,
        "candidate_execution_count": 1,
        "candidate_rerun": False,
        "candidate_reexecuted_during_recovery": False,
        "comparison_recomputed_during_recovery": False,
        "study_frozen": False,
        "interpretation_authorized": False,
        "profitability_claimed": False,
        "winner_claimed": False,
        "final_test_authorized": False,
        "production_authorized": False,
    }
    (root / "portable-exp002-result-index.json").write_text(
        json.dumps(result_index, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return result_index


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    recover(args.root)


if __name__ == "__main__":
    main()


__all__ = [
    "crosscheck_persisted_comparison",
    "normalize_independent_for_postverify",
    "recover",
]
