"""Sealed pre-result preregistration for portable Canonical M2 Experiment 0003."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path

from research.issue522_portable_exp003 import (
    BASELINE_SIGNAL_INDEX,
    BASELINE_SIGNAL_NAME,
    CANDIDATE_SIGNAL_INDEX,
    CANDIDATE_SIGNAL_NAME,
    ENTRY_THRESHOLD,
    EXIT_THRESHOLD,
    EXPECTED_BASELINE_FINGERPRINT,
    EXPECTED_DATASET_ARTIFACT_DIGEST,
    EXPECTED_DATASET_ID,
    EXPECTED_EXP002_CANDIDATE_FINGERPRINT,
    EXPECTED_IMPLEMENTATION_DIGEST,
    EXPECTED_RUNTIME_DIGEST,
    EXPECTED_SEEDS,
    EXPECTED_STUDY_DIGEST,
    EXPECTED_SYMBOLS,
    SOURCE_RECOVERY_HEAD,
    SOURCE_RECOVERY_RUN_ID,
    SOURCE_RESULT_ARTIFACT_DIGEST,
    SOURCE_RESULT_ARTIFACT_ID,
    require_feature_index,
)
from trade_rl.artifacts.hashing import content_digest
from trade_rl.data import (
    inspect_published_market_dataset_artifact,
    load_market_dataset_artifact,
)
from trade_rl.evaluation.experiments import (
    ControlledFactor,
    ExperimentDecisionKind,
    define_experiment,
    inspect_study,
)
from trade_rl.evaluation.experiments.bootstrap.config import (
    load_canonical_m2_bootstrap_config,
)
from trade_rl.evaluation.experiments.inspection import _reconstruct
from trade_rl.evaluation.experiments.store import StudyStore
from trade_rl.evaluation.runs import build_candidate_run_provenance

ISSUE_NUMBER = 522
EXPERIMENT_SEQUENCE = 3
HYPOTHESIS = (
    "For the frozen large-liquid crypto universe, a four-day daily-return trend "
    "signal improves the trend strategy versus the rolling 24-hour return baseline "
    "without changing thresholds, risk, accounting, or any non-signal field."
)
FROZEN_BASELINE_TREND_MEDIAN_TURNOVER = 858.6801864417196
STRUCTURAL_RUN_ID = 34751469557
STRUCTURAL_HEAD = "9b924637b5835ccfe69fb68725ac2330fefa0b71"
STRUCTURAL_ARTIFACT_ID = 10316141669
STRUCTURAL_ARTIFACT_DIGEST = (
    "sha256:84be1fbdc039ecc6cd945d9d8c81ed594df424a23e112f1b34995bd07ec839c6"
)
SOURCE_FRESH_VERIFIER_ARTIFACT_ID = 10316125529
SOURCE_FRESH_VERIFIER_ARTIFACT_DIGEST = (
    "sha256:b2fcf6c924b4fbc3bd4ed5699ac372c8d518c7513f8c6232c52748a281c7a381"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def formal_rule_payload() -> dict[str, object]:
    """Return the exact, immutable Experiment 0003 decision rule."""

    return {
        "accept": {
            "positive_trend_factor_effect_symbols_min": 4,
            "median_trend_excess_total_return_strictly_positive": True,
            "candidate_trend_positive_total_return_symbols": 5,
            "candidate_trend_turnover_strictly_below_baseline": True,
        },
        "keep_baseline": {
            "positive_trend_factor_effect_symbols_max": 2,
            "median_trend_excess_total_return_non_positive": True,
            "candidate_trend_positive_total_return_symbols_max": 3,
        },
        "otherwise": "INCONCLUSIVE",
    }


def expected_prereg_index(
    *,
    definition_digest: str,
    requested_candidate_config_digest: str,
    structural_report_sha256: str,
) -> dict[str, object]:
    """Build the exact result-blind preregistration index payload."""

    return {
        "schema_version": "canonical_m2_portable_exp003_prereg_index_v1",
        "issue_number": ISSUE_NUMBER,
        "source_recovery_run_id": SOURCE_RECOVERY_RUN_ID,
        "source_recovery_head": SOURCE_RECOVERY_HEAD,
        "source_result_artifact_id": SOURCE_RESULT_ARTIFACT_ID,
        "source_result_artifact_digest": SOURCE_RESULT_ARTIFACT_DIGEST,
        "source_fresh_verifier_artifact_id": SOURCE_FRESH_VERIFIER_ARTIFACT_ID,
        "source_fresh_verifier_artifact_digest": SOURCE_FRESH_VERIFIER_ARTIFACT_DIGEST,
        "structural_diagnostic_run_id": STRUCTURAL_RUN_ID,
        "structural_diagnostic_head": STRUCTURAL_HEAD,
        "structural_diagnostic_artifact_id": STRUCTURAL_ARTIFACT_ID,
        "structural_diagnostic_artifact_digest": STRUCTURAL_ARTIFACT_DIGEST,
        "structural_report_sha256": structural_report_sha256,
        "study_digest": EXPECTED_STUDY_DIGEST,
        "baseline_evidence_fingerprint": EXPECTED_BASELINE_FINGERPRINT,
        "dataset_id": EXPECTED_DATASET_ID,
        "dataset_artifact_digest": EXPECTED_DATASET_ARTIFACT_DIGEST,
        "implementation_digest": EXPECTED_IMPLEMENTATION_DIGEST,
        "runtime_environment_digest": EXPECTED_RUNTIME_DIGEST,
        "prior_experiment_sequences": [1, 2],
        "prior_experiment_decisions": ["KEEP_BASELINE", "KEEP_BASELINE"],
        "experiment_sequence": EXPERIMENT_SEQUENCE,
        "definition_digest": definition_digest,
        "factor": "RULE_SIGNAL",
        "hypothesis": HYPOTHESIS,
        "requested_candidate_config_digest": requested_candidate_config_digest,
        "resolved_changed_paths": ["signal_index", "signal_name"],
        "baseline_signal_name": BASELINE_SIGNAL_NAME,
        "baseline_signal_index": BASELINE_SIGNAL_INDEX,
        "candidate_signal_name": CANDIDATE_SIGNAL_NAME,
        "candidate_signal_index": CANDIDATE_SIGNAL_INDEX,
        "rule_entry_threshold": ENTRY_THRESHOLD,
        "rule_exit_threshold": EXIT_THRESHOLD,
        "formal_target_strategy": "trend",
        "frozen_baseline_trend_median_turnover": FROZEN_BASELINE_TREND_MEDIAN_TURNOVER,
        "formal_decision_rule": formal_rule_payload(),
        "mean_reversion_side_effect_must_be_disclosed": True,
        "return_only_search_stop_rule": (
            "If Experiment 0003 is not ACCEPT_CANDIDATE, stop adjacent return-horizon, "
            "rule-threshold, long-only, and rule-signal permutations in this Study."
        ),
        "experiment_0002_trend_performance_inspected_for_design": False,
        "candidate_executed": False,
        "candidate_pnl_computed_before_preregistration": False,
        "result_inspected_before_preregistration": False,
        "profitability_claimed": False,
        "winner_claimed": False,
        "final_test_authorized": False,
        "production_authorized": False,
    }


def validate_prereg_index_payload(
    payload: object,
    *,
    definition_digest: str,
    requested_candidate_config_digest: str,
    structural_report_sha256: str,
) -> None:
    """Require exact key set and values; reject any post-result override surface."""

    if not isinstance(payload, dict):
        raise RuntimeError("preregistration index malformed")
    expected = expected_prereg_index(
        definition_digest=definition_digest,
        requested_candidate_config_digest=requested_candidate_config_digest,
        structural_report_sha256=structural_report_sha256,
    )
    if set(payload) != set(expected):
        raise RuntimeError("preregistration index key set drift")
    for key, value in expected.items():
        if payload.get(key) != value:
            raise RuntimeError(f"preregistration index drift: {key}")


def _load_structural_report(path: Path) -> tuple[dict[str, object], str]:
    if not path.is_file():
        raise RuntimeError("sealed structural diagnostic is missing")
    report = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(report, dict):
        raise RuntimeError("sealed structural diagnostic malformed")
    expected = {
        "schema_version": "canonical_m2_portable_exp003_structural_diagnostic_v1",
        "issue_number": ISSUE_NUMBER,
        "source_recovery_run_id": SOURCE_RECOVERY_RUN_ID,
        "source_recovery_head": SOURCE_RECOVERY_HEAD,
        "source_result_artifact_id": SOURCE_RESULT_ARTIFACT_ID,
        "source_result_artifact_digest": SOURCE_RESULT_ARTIFACT_DIGEST,
        "study_digest": EXPECTED_STUDY_DIGEST,
        "dataset_id": EXPECTED_DATASET_ID,
        "dataset_artifact_digest": EXPECTED_DATASET_ARTIFACT_DIGEST,
        "baseline_evidence_fingerprint": EXPECTED_BASELINE_FINGERPRINT,
        "experiment_sequences": [1, 2],
        "terminal_decisions": ["KEEP_BASELINE", "KEEP_BASELINE"],
        "rule_signal_allowed": True,
        "baseline_signal_name": BASELINE_SIGNAL_NAME,
        "baseline_signal_index": BASELINE_SIGNAL_INDEX,
        "candidate_signal_name": CANDIDATE_SIGNAL_NAME,
        "candidate_signal_index": CANDIDATE_SIGNAL_INDEX,
        "resolved_changed_fields": ["signal_index", "signal_name"],
        "rule_entry_threshold": ENTRY_THRESHOLD,
        "rule_exit_threshold": EXIT_THRESHOLD,
        "frozen_baseline_trend_median_turnover": FROZEN_BASELINE_TREND_MEDIAN_TURNOVER,
        "experiment_0002_trend_performance_inspected": False,
        "experiment_0003_candidate_executed": False,
        "experiment_0003_candidate_pnl_computed": False,
        "profitability_claimed": False,
        "winner_claimed": False,
        "final_test_authorized": False,
        "production_authorized": False,
    }
    for key, value in expected.items():
        if report.get(key) != value:
            raise RuntimeError(f"sealed structural diagnostic drift: {key}")
    if int(report.get("max_experiments", 0)) < EXPERIMENT_SEQUENCE:
        raise RuntimeError("sealed structural diagnostic has no Experiment 0003 budget")
    encoded = json.dumps(report, sort_keys=True)
    if "total_return" in encoded:
        raise RuntimeError("structural diagnostic unexpectedly contains return performance")
    return report, _sha256(path)


def _candidate_from_bootstrap(root: Path):
    bootstrap = load_canonical_m2_bootstrap_config(root / "bootstrap.json")
    baseline = bootstrap.baseline
    if baseline.signal_name != BASELINE_SIGNAL_NAME:
        raise RuntimeError("bootstrap baseline signal name drift")
    if baseline.rule_entry_threshold != ENTRY_THRESHOLD:
        raise RuntimeError("bootstrap entry threshold drift")
    if baseline.rule_exit_threshold != EXIT_THRESHOLD:
        raise RuntimeError("bootstrap exit threshold drift")
    candidate = replace(baseline, signal_name=CANDIDATE_SIGNAL_NAME)
    return candidate, content_digest(candidate.to_json_payload())


def _validate_source(root: Path, *, allow_exp003_definition: bool):
    dataset_root = root / "dataset"
    study_root = root / "study"
    dataset = load_market_dataset_artifact(dataset_root)
    artifact = inspect_published_market_dataset_artifact(dataset_root)
    snapshot = inspect_study(study_root)
    if dataset.dataset_id != EXPECTED_DATASET_ID:
        raise RuntimeError("Dataset ID drift")
    if artifact.artifact_digest != EXPECTED_DATASET_ARTIFACT_DIGEST:
        raise RuntimeError("Dataset artifact digest drift")
    if snapshot.plan.digest != EXPECTED_STUDY_DIGEST:
        raise RuntimeError("Study digest drift")
    if snapshot.plan.implementation_digest != EXPECTED_IMPLEMENTATION_DIGEST:
        raise RuntimeError("implementation digest drift")
    if snapshot.plan.runtime_environment_digest != EXPECTED_RUNTIME_DIGEST:
        raise RuntimeError("runtime digest drift")
    if snapshot.baseline is None or snapshot.baseline.fingerprint != EXPECTED_BASELINE_FINGERPRINT:
        raise RuntimeError("baseline fingerprint drift")
    if tuple(snapshot.plan.symbols) != EXPECTED_SYMBOLS:
        raise RuntimeError("symbol roster drift")
    if tuple(snapshot.plan.ppo_seeds) != EXPECTED_SEEDS:
        raise RuntimeError("PPO seed roster drift")
    if snapshot.plan.max_experiments < EXPERIMENT_SEQUENCE:
        raise RuntimeError("Study has no Experiment 0003 budget")
    if ControlledFactor.RULE_SIGNAL not in snapshot.plan.allowed_factors:
        raise RuntimeError("RULE_SIGNAL not allowed by Study")
    if snapshot.frozen:
        raise RuntimeError("Study unexpectedly frozen")
    expected_sequences = (1, 2, 3) if allow_exp003_definition else (1, 2)
    if snapshot.experiment_sequences != expected_sequences:
        raise RuntimeError("Experiment sequence drift")
    if snapshot.terminal_sequences != (1, 2):
        raise RuntimeError("prior terminal sequence drift")

    provenance = build_candidate_run_provenance()
    if provenance.get("implementation_digest") != EXPECTED_IMPLEMENTATION_DIGEST:
        raise RuntimeError("current implementation differs from frozen Study")
    if provenance.get("runtime_environment_digest") != EXPECTED_RUNTIME_DIGEST:
        raise RuntimeError("current runtime differs from frozen Study")

    require_feature_index(
        dataset.feature_names,
        name=BASELINE_SIGNAL_NAME,
        expected_index=BASELINE_SIGNAL_INDEX,
    )
    require_feature_index(
        dataset.feature_names,
        name=CANDIDATE_SIGNAL_NAME,
        expected_index=CANDIDATE_SIGNAL_INDEX,
    )

    state = _reconstruct(StudyStore(study_root))
    if len(state.experiments) != len(expected_sequences):
        raise RuntimeError("reconstructed Experiment count drift")
    for sequence, experiment in enumerate(state.experiments[:2], start=1):
        if experiment.failure is not None:
            raise RuntimeError(f"Experiment {sequence:04d} contains failure evidence")
        if experiment.decision is None:
            raise RuntimeError(f"Experiment {sequence:04d} decision missing")
        if experiment.decision.decision is not ExperimentDecisionKind.KEEP_BASELINE:
            raise RuntimeError(f"Experiment {sequence:04d} is not KEEP_BASELINE")
    exp2 = state.experiments[1]
    if exp2.candidate is None or exp2.candidate.evidence.fingerprint != EXPECTED_EXP002_CANDIDATE_FINGERPRINT:
        raise RuntimeError("Experiment 0002 candidate fingerprint drift")
    if exp2.verification is None or exp2.comparison is None:
        raise RuntimeError("Experiment 0002 verification/comparison missing")

    if allow_exp003_definition:
        exp3 = state.experiments[2]
        if any(
            value is not None
            for value in (
                exp3.candidate,
                exp3.verification,
                exp3.comparison,
                exp3.decision,
                exp3.failure,
            )
        ):
            raise RuntimeError("Experiment 0003 preregistration contains result evidence")
    return dataset, state


def _validate_definition(root: Path):
    _, state = _validate_source(root, allow_exp003_definition=True)
    definition = state.experiments[2].definition
    candidate, requested_digest = _candidate_from_bootstrap(root)
    if definition.sequence != EXPERIMENT_SEQUENCE:
        raise RuntimeError("Experiment 0003 sequence drift")
    if definition.factor is not ControlledFactor.RULE_SIGNAL:
        raise RuntimeError("Experiment 0003 factor drift")
    if definition.study_digest != EXPECTED_STUDY_DIGEST:
        raise RuntimeError("Experiment 0003 Study binding drift")
    if definition.baseline_evidence_digest != EXPECTED_BASELINE_FINGERPRINT:
        raise RuntimeError("Experiment 0003 baseline binding drift")
    if definition.hypothesis != HYPOTHESIS:
        raise RuntimeError("Experiment 0003 hypothesis drift")
    if definition.candidate_requested_config_digest != requested_digest:
        raise RuntimeError("Experiment 0003 requested config digest drift")
    baseline_payload = state.plan.baseline_config.to_payload()
    candidate_payload = definition.candidate_config.to_payload()
    changed = tuple(
        key
        for key in sorted(set(baseline_payload) | set(candidate_payload))
        if baseline_payload.get(key) != candidate_payload.get(key)
    )
    if changed != ("signal_index", "signal_name"):
        raise RuntimeError(f"Experiment 0003 changed unexpected fields: {changed!r}")
    if candidate_payload.get("signal_name") != CANDIDATE_SIGNAL_NAME:
        raise RuntimeError("resolved candidate signal name drift")
    if candidate_payload.get("signal_index") != CANDIDATE_SIGNAL_INDEX:
        raise RuntimeError("resolved candidate signal index drift")
    if candidate.signal_name != CANDIDATE_SIGNAL_NAME:
        raise RuntimeError("candidate signal reconstruction drift")
    return definition, requested_digest


def preregister(root: Path, structural_report_path: Path) -> dict[str, object]:
    """Append only the immutable Experiment 0003 definition and prereg index."""

    _validate_source(root, allow_exp003_definition=False)
    _, structural_sha = _load_structural_report(structural_report_path)
    candidate, requested_digest = _candidate_from_bootstrap(root)
    definition = define_experiment(
        root / "study",
        dataset_root=root / "dataset",
        hypothesis=HYPOTHESIS,
        factor=ControlledFactor.RULE_SIGNAL,
        candidate_config=candidate,
        baseline_evidence_digest=EXPECTED_BASELINE_FINGERPRINT,
    )
    verified_definition, verified_requested = _validate_definition(root)
    if verified_definition.digest != definition.digest:
        raise RuntimeError("Experiment 0003 definition reconstruction mismatch")
    if verified_requested != requested_digest:
        raise RuntimeError("Experiment 0003 requested config reconstruction mismatch")
    index = expected_prereg_index(
        definition_digest=definition.digest,
        requested_candidate_config_digest=requested_digest,
        structural_report_sha256=structural_sha,
    )
    path = root / "portable-exp003-prereg-index.json"
    if path.exists():
        raise RuntimeError("Experiment 0003 preregistration index already exists")
    path.write_text(
        json.dumps(index, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return index


def verify_prereg(
    root: Path,
    structural_report_path: Path,
    report_root: Path,
) -> dict[str, object]:
    """Independently reconstruct and verify the sealed preregistration."""

    definition, requested_digest = _validate_definition(root)
    _, structural_sha = _load_structural_report(structural_report_path)
    path = root / "portable-exp003-prereg-index.json"
    if not path.is_file():
        raise RuntimeError("Experiment 0003 preregistration index missing")
    payload = json.loads(path.read_text(encoding="utf-8"))
    validate_prereg_index_payload(
        payload,
        definition_digest=definition.digest,
        requested_candidate_config_digest=requested_digest,
        structural_report_sha256=structural_sha,
    )
    state = _reconstruct(StudyStore(root / "study"))
    exp3 = state.experiments[2]
    if any(
        value is not None
        for value in (
            exp3.candidate,
            exp3.verification,
            exp3.comparison,
            exp3.decision,
            exp3.failure,
        )
    ):
        raise RuntimeError("Experiment 0003 preregistration contains result evidence")
    report = {
        "schema_version": "canonical_m2_portable_exp003_prereg_verification_v1",
        "verified": True,
        "definition_digest": definition.digest,
        "requested_candidate_config_digest": requested_digest,
        "structural_report_sha256": structural_sha,
        "formal_target_strategy": "trend",
        "formal_decision_rule": formal_rule_payload(),
        "candidate_executed": False,
        "result_inspected": False,
        "experiment_0002_trend_performance_inspected_for_design": False,
        "execution_authorized": False,
        "profitability_claimed": False,
        "winner_claimed": False,
        "final_test_authorized": False,
        "production_authorized": False,
    }
    report_root.mkdir(parents=True, exist_ok=True)
    (report_root / "prereg-verification.json").write_text(
        json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    create = sub.add_parser("preregister")
    create.add_argument("--root", type=Path, required=True)
    create.add_argument("--structural-report", type=Path, required=True)
    verify = sub.add_parser("verify-prereg")
    verify.add_argument("--root", type=Path, required=True)
    verify.add_argument("--structural-report", type=Path, required=True)
    verify.add_argument("--report-root", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "preregister":
        preregister(args.root, args.structural_report)
    else:
        verify_prereg(args.root, args.structural_report, args.report_root)


if __name__ == "__main__":
    main()


__all__ = [
    "expected_prereg_index",
    "formal_rule_payload",
    "preregister",
    "validate_prereg_index_payload",
    "verify_prereg",
]
