"""Recover Issue #511 finalization from the immutable failed execution artifact.

The bound execution completed candidate/verification/comparison evidence and then
failed because a research helper required ``dict`` for recursively frozen JSON.
This recovery never executes a candidate or recomputes a persisted comparison. It
only revalidates the existing state, independently replays the preregistered rule,
and publishes the missing decision plus an explicitly recovery-bound result index.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research.issue511_portable_exp001_execute import (  # noqa: E402
    EXPECTED_BASELINE_FINGERPRINT,
    EXPECTED_DATASET_ARTIFACT_DIGEST,
    EXPECTED_DATASET_ID,
    EXPECTED_DEFINITION_DIGEST,
    EXPECTED_IMPLEMENTATION_DIGEST,
    EXPECTED_RUNTIME_DIGEST,
    EXPECTED_STUDY_DIGEST,
    _independent_analysis,
)
from trade_rl.evaluation.experiments import (  # noqa: E402
    ControlledVerificationStatus,
    ExperimentDecisionKind,
    decide_experiment,
    inspect_study,
)
from trade_rl.evaluation.experiments.inspection import _reconstruct  # noqa: E402
from trade_rl.evaluation.experiments.store import StudyStore  # noqa: E402

EXPECTED_CANDIDATE_FINGERPRINT = (
    "5e1ee8a530097e6b1c4571efe1d3edd9e867a2e77b8346a558ce9ed385e1f672"
)
ORIGINAL_EXECUTION_RUN_ID = 34707652616
ORIGINAL_EXECUTION_HEAD = "52e9c573fbee9ca24eac031f7dad5c6646ca76d8"
FAILURE_ARTIFACT_ID = 10304040458
FAILURE_ARTIFACT_DIGEST = (
    "sha256:0e67733694765a48f0ca47220cb4c5520bd2934043d5f1a4a60cf865f241735a"
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


def crosscheck_persisted_comparison(
    comparison: object,
    independent: Mapping[str, object],
) -> None:
    """Cross-check frozen MappingProxy-backed comparison against the raw oracle."""

    factor_effect = _mapping(
        getattr(comparison, "factor_effect", None), field="persisted comparison factor_effect"
    )
    cross_symbol = _mapping(
        factor_effect.get("cross_symbol"), field="persisted comparison cross_symbol"
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


def recover(root: Path) -> dict[str, object]:
    """Finish only the missing decision/index for the exact failed candidate evidence."""

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
    if snapshot.experiment_sequences != (1,) or snapshot.terminal_sequences:
        raise RuntimeError("recovery input is not comparison-complete / decision-missing")
    if snapshot.frozen:
        raise RuntimeError("recovery Study unexpectedly frozen")

    state = _reconstruct(StudyStore(study_root))
    experiment = state.experiments[0]
    if experiment.definition.digest != EXPECTED_DEFINITION_DIGEST:
        raise RuntimeError("recovery Experiment definition drift")
    if experiment.failure is not None:
        raise RuntimeError("recovery input contains terminal failure evidence")
    if experiment.candidate is None:
        raise RuntimeError("recovery candidate evidence missing")
    if experiment.candidate.evidence.fingerprint != EXPECTED_CANDIDATE_FINGERPRINT:
        raise RuntimeError("recovery candidate fingerprint drift")
    if experiment.verification is None:
        raise RuntimeError("recovery controlled verification missing")
    if experiment.verification.status is not ControlledVerificationStatus.CONTROLLED:
        raise RuntimeError("recovery verification is not CONTROLLED")
    if experiment.verification.changed_paths != (("rule_entry_threshold",),):
        raise RuntimeError("recovery changed path differs from preregistration")
    if experiment.comparison is None:
        raise RuntimeError("recovery comparison evidence missing")
    if experiment.decision is not None:
        raise RuntimeError("recovery input already contains a decision")

    independent = _independent_analysis(study_root)
    crosscheck_persisted_comparison(experiment.comparison, independent)
    decision_kind = ExperimentDecisionKind(str(independent["formal_decision"]))
    decision = decide_experiment(
        study_root,
        1,
        decision=decision_kind,
        rationale=(
            "Mechanical replay of the preregistered portable Experiment 0001 decision rule "
            "from the original bound candidate evidence; performance values remain sealed "
            "until independent post-verification passes."
        ),
        decided_by="openai-gpt-5.6-sol-recovery",
        decided_at=datetime.now(timezone.utc),
    )

    final_snapshot = inspect_study(study_root)
    if final_snapshot.plan.digest != EXPECTED_STUDY_DIGEST:
        raise RuntimeError("Study digest changed during recovery")
    if final_snapshot.experiment_sequences != (1,) or final_snapshot.terminal_sequences != (1,):
        raise RuntimeError("recovered Experiment did not reach one terminal decision")
    if final_snapshot.frozen:
        raise RuntimeError("Study was frozen unexpectedly during recovery")
    final_state = _reconstruct(StudyStore(study_root))
    recovered = final_state.experiments[0]
    if recovered.candidate is None or recovered.candidate.evidence.fingerprint != EXPECTED_CANDIDATE_FINGERPRINT:
        raise RuntimeError("candidate evidence changed during recovery")
    if recovered.verification is None or recovered.verification.digest != experiment.verification.digest:
        raise RuntimeError("verification evidence changed during recovery")
    if recovered.comparison is None or recovered.comparison.digest != experiment.comparison.digest:
        raise RuntimeError("comparison evidence changed during recovery")
    if recovered.decision is None or recovered.decision.digest != decision.digest:
        raise RuntimeError("recovered decision reconstruction mismatch")

    recovery_run_id = int(os.environ["GITHUB_RUN_ID"])
    recovery_head = os.environ["GITHUB_SHA"]
    result_index: dict[str, object] = {
        "schema_version": "canonical_m2_portable_exp001_recovered_result_index_v1",
        "issue_number": 511,
        "original_execution_run_id": ORIGINAL_EXECUTION_RUN_ID,
        "original_execution_helper_git_sha": ORIGINAL_EXECUTION_HEAD,
        "source_failure_artifact_id": FAILURE_ARTIFACT_ID,
        "source_failure_artifact_digest": FAILURE_ARTIFACT_DIGEST,
        "recovery_workflow_run_id": recovery_run_id,
        "recovery_helper_git_sha": recovery_head,
        "study_digest": EXPECTED_STUDY_DIGEST,
        "baseline_evidence_fingerprint": EXPECTED_BASELINE_FINGERPRINT,
        "candidate_evidence_fingerprint": EXPECTED_CANDIDATE_FINGERPRINT,
        "dataset_id": EXPECTED_DATASET_ID,
        "dataset_artifact_digest": EXPECTED_DATASET_ARTIFACT_DIGEST,
        "implementation_digest": EXPECTED_IMPLEMENTATION_DIGEST,
        "runtime_environment_digest": EXPECTED_RUNTIME_DIGEST,
        "definition_digest": EXPECTED_DEFINITION_DIGEST,
        "verification_digest": recovered.verification.digest,
        "verification_status": recovered.verification.status.value,
        "changed_paths": [list(path) for path in recovered.verification.changed_paths],
        "comparison_digest": recovered.comparison.digest,
        "decision_digest": recovered.decision.digest,
        "decision": recovered.decision.decision.value,
        "independent_raw_evidence": independent,
        "candidate_reexecuted_during_recovery": False,
        "comparison_recomputed_during_recovery": False,
        "study_frozen": False,
        "profitability_claimed": False,
        "winner_claimed": False,
        "final_test_authorized": False,
        "production_authorized": False,
    }
    (root / "portable-exp001-recovered-result-index.json").write_text(
        json.dumps(result_index, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    output = os.environ.get("GITHUB_OUTPUT")
    if output:
        with Path(output).open("a", encoding="utf-8") as handle:
            handle.write(
                "artifact_name="
                f"canonical-m2-portable-exp001-recovered-result-v1-{ORIGINAL_EXECUTION_RUN_ID}-{recovery_run_id}\n"
            )
    return result_index


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    recover(args.root)


if __name__ == "__main__":
    main()


__all__ = ["crosscheck_persisted_comparison", "recover"]
