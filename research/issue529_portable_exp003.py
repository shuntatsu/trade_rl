"""Result-blind preregistration for portable Canonical M2 Experiment 0003."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

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
    EXPECTED_DEFINITION_DIGEST as EXPECTED_EXP002_DEFINITION_DIGEST,
)
from research.issue519_portable_exp002_recovery import (
    EXPECTED_CANDIDATE_FINGERPRINT as EXPECTED_EXP002_CANDIDATE_FINGERPRINT,
)
from trade_rl.artifacts.hashing import content_digest
from trade_rl.data import (
    inspect_published_market_dataset_artifact,
    load_market_dataset_artifact,
)
from trade_rl.evaluation.experiments import (
    ControlledFactor,
    ControlledVerificationStatus,
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

ISSUE_NUMBER = 529
SOURCE_FORMAL_RUN_ID = 34750227256
SOURCE_FORMAL_HEAD = "43c54f822dfd156f7fc5e4cf40f9fce9d8cc8b91"
SOURCE_RESULT_ARTIFACT_ID = 10315084502
SOURCE_RESULT_ARTIFACT_DIGEST = (
    "sha256:9a85ec7fa40cabfb662556ef89780b140609e9b2e471006981c829800a043d1d"
)
SOURCE_FRESH_VERIFY_ARTIFACT_ID = 10316125529
SOURCE_FRESH_VERIFY_ARTIFACT_DIGEST = (
    "sha256:b2fcf6c924b4fbc3bd4ed5699ac372c8d518c7513f8c6232c52748a281c7a381"
)
EXPECTED_EXP002_DECISION_DIGEST = (
    "e42bed5163557892c69fd8256c277f8596423b4fdcc05684ffe495e33e3b496f"
)

BASELINE_FEATURE_NAMES = (
    "1h__log_return_1bar",
    "1h__log_return_4bar",
    "1h__log_return_24bar",
    "1h__realized_volatility_24bar",
    "1h__volume_zscore_24bar",
    "1h__funding_bps",
    "1h__rsi_14bar",
    "1h__macd_histogram_12_26_9",
    "4h__log_return_4bar",
    "4h__realized_volatility_24bar",
    "1d__log_return_1bar",
    "1d__realized_volatility_24bar",
)
BASELINE_FEATURE_INDICES = (0, 1, 2, 4, 5, 6, 7, 10, 60, 63, 114, 118)
CANDIDATE_FEATURE_NAME = "1h__cross_sectional_momentum_rank_24bar"
CANDIDATE_FEATURE_INDEX = 57
EXPECTED_FIT_AVAILABLE_ROWS = 17511
EXPECTED_FIT_TOTAL_ROWS = 17519
EXPECTED_EVAL_AVAILABLE_ROWS = 17545
EXPECTED_EVAL_TOTAL_ROWS = 17545
HYPOTHESIS = (
    "Adding only the causal 24h cross-sectional momentum rank helps the universal "
    "24h forecast distinguish relative leaders and laggards from common crypto-market "
    "drift, improving net-of-cost lightgbm24 returns across the frozen symbols."
)
UNAFFECTED_CONTROL_STRATEGIES = (
    "cash",
    "constant_long",
    "constant_short",
    "trend",
    "mean_reversion",
)


def build_candidate_feature_names(baseline: tuple[str, ...]) -> tuple[str, ...]:
    """Append the single preregistered feature and reject duplicate exposure."""

    names = tuple(baseline)
    if names != BASELINE_FEATURE_NAMES:
        if CANDIDATE_FEATURE_NAME in names:
            raise RuntimeError("candidate feature is already present")
        raise RuntimeError("baseline feature set drift")
    if CANDIDATE_FEATURE_NAME in names:
        raise RuntimeError("candidate feature is already present")
    result = (*names, CANDIDATE_FEATURE_NAME)
    if len(set(result)) != len(result):
        raise RuntimeError("candidate feature set contains duplicates")
    return result


def formal_decision(
    *,
    positive_effects: int,
    median_excess: float,
    candidate_positive_returns: int,
) -> ExperimentDecisionKind:
    """Apply the immutable Experiment 0003 lightgbm24 decision rule."""

    if (
        positive_effects >= 4
        and median_excess > 0.0
        and candidate_positive_returns == 5
    ):
        return ExperimentDecisionKind.ACCEPT_CANDIDATE
    if (
        positive_effects <= 2
        or median_excess <= 0.0
        or candidate_positive_returns <= 3
    ):
        return ExperimentDecisionKind.KEEP_BASELINE
    return ExperimentDecisionKind.INCONCLUSIVE


def _coverage(
    available: np.ndarray,
    mask: np.ndarray,
    symbols: tuple[str, ...],
) -> dict[str, object]:
    total = int(np.count_nonzero(mask))
    if total <= 0:
        raise RuntimeError("feature diagnostic period is empty")
    return {
        symbol: {
            "available_rows": int(np.count_nonzero(available[mask, symbol_index])),
            "total_rows": total,
            "coverage": float(np.mean(available[mask, symbol_index])),
        }
        for symbol_index, symbol in enumerate(symbols)
    }


def feature_diagnostics(
    dataset: object,
    *,
    fit_cutoff: np.datetime64,
    require_frozen_roster: bool = True,
) -> dict[str, object]:
    """Verify only identity/availability/finite-value facts, never return relation."""

    dataset_id = getattr(dataset, "dataset_id", None)
    symbols = tuple(getattr(dataset, "symbols", ()))
    timestamps = np.asarray(getattr(dataset, "timestamps", ()), dtype="datetime64[ns]")
    feature_names = tuple(getattr(dataset, "feature_names", ()))
    features = np.asarray(getattr(dataset, "features", ()))
    feature_available = np.asarray(getattr(dataset, "feature_available", ()))

    if require_frozen_roster:
        if dataset_id != EXPECTED_DATASET_ID:
            raise RuntimeError("Experiment 0003 Dataset ID drift")
        if symbols != EXPECTED_SYMBOLS:
            raise RuntimeError("Experiment 0003 symbol roster drift")
    if CANDIDATE_FEATURE_NAME not in feature_names:
        raise RuntimeError("Experiment 0003 candidate feature missing")
    index = feature_names.index(CANDIDATE_FEATURE_NAME)
    if require_frozen_roster and index != CANDIDATE_FEATURE_INDEX:
        raise RuntimeError("Experiment 0003 candidate feature index drift")
    if CANDIDATE_FEATURE_NAME in BASELINE_FEATURE_NAMES:
        raise RuntimeError("Experiment 0003 candidate feature already in baseline")

    if timestamps.ndim != 1:
        raise RuntimeError("Experiment 0003 timestamps malformed")
    if features.ndim != 3 or feature_available.shape != features.shape:
        raise RuntimeError("Experiment 0003 feature arrays malformed")
    if features.shape[0] != timestamps.size or features.shape[1] != len(symbols):
        raise RuntimeError("Experiment 0003 feature array shape drift")
    if index >= features.shape[2]:
        raise RuntimeError("Experiment 0003 feature index outside array")

    values = np.asarray(features[:, :, index], dtype=np.float64)
    available = np.asarray(feature_available[:, :, index], dtype=np.bool_)
    if np.any(available & ~np.isfinite(values)):
        raise RuntimeError("available Experiment 0003 feature contains non-finite values")

    cutoff = np.datetime64(fit_cutoff, "ns")
    fit_mask = timestamps < cutoff
    eval_mask = timestamps >= cutoff
    fit_coverage = _coverage(available, fit_mask, symbols)
    eval_coverage = _coverage(available, eval_mask, symbols)

    if require_frozen_roster:
        for symbol in EXPECTED_SYMBOLS:
            fit = fit_coverage[symbol]
            evaluation = eval_coverage[symbol]
            if not isinstance(fit, dict) or not isinstance(evaluation, dict):
                raise RuntimeError("Experiment 0003 feature coverage malformed")
            if (
                fit.get("available_rows") != EXPECTED_FIT_AVAILABLE_ROWS
                or fit.get("total_rows") != EXPECTED_FIT_TOTAL_ROWS
            ):
                raise RuntimeError("Experiment 0003 fit feature coverage drift")
            if (
                evaluation.get("available_rows") != EXPECTED_EVAL_AVAILABLE_ROWS
                or evaluation.get("total_rows") != EXPECTED_EVAL_TOTAL_ROWS
            ):
                raise RuntimeError("Experiment 0003 evaluation feature coverage drift")

    return {
        "feature_name": CANDIDATE_FEATURE_NAME,
        "feature_index": index,
        "feature_count": len(feature_names),
        "fit_cutoff": str(cutoff),
        "fit_coverage": fit_coverage,
        "evaluation_coverage": eval_coverage,
        "no_return_target_or_pnl_relation_inspected": True,
    }


def _paths(root: Path) -> tuple[Path, Path, Path]:
    dataset_root = root / "dataset"
    study_root = root / "study"
    bootstrap_path = root / "bootstrap.json"
    if not dataset_root.is_dir() or not study_root.is_dir() or not bootstrap_path.is_file():
        raise RuntimeError("portable Experiment 0002 result layout is incomplete")
    return dataset_root, study_root, bootstrap_path


def _source_result_index(root: Path) -> dict[str, object]:
    path = root / "portable-exp002-result-index.json"
    if not path.is_file():
        raise RuntimeError("Experiment 0002 source result index missing")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError("Experiment 0002 source result index malformed")
    return value


def _validate_source_index(root: Path, state: object) -> None:
    index = _source_result_index(root)
    if index.get("schema_version") != "canonical_m2_portable_exp002_result_index_v1":
        raise RuntimeError("Experiment 0002 result-index schema drift")
    expected = {
        "issue_number": 519,
        "study_digest": EXPECTED_STUDY_DIGEST,
        "baseline_evidence_fingerprint": EXPECTED_BASELINE_FINGERPRINT,
        "dataset_id": EXPECTED_DATASET_ID,
        "dataset_artifact_digest": EXPECTED_DATASET_ARTIFACT_DIGEST,
        "implementation_digest": EXPECTED_IMPLEMENTATION_DIGEST,
        "runtime_environment_digest": EXPECTED_RUNTIME_DIGEST,
        "definition_digest": EXPECTED_EXP002_DEFINITION_DIGEST,
        "candidate_evidence_fingerprint": EXPECTED_EXP002_CANDIDATE_FINGERPRINT,
        "decision_digest": EXPECTED_EXP002_DECISION_DIGEST,
        "decision": "KEEP_BASELINE",
        "verification_status": "CONTROLLED",
        "changed_paths": [["signal_index"], ["signal_name"]],
        "candidate_execution_count": 1,
        "candidate_rerun": False,
        "candidate_reexecuted_during_recovery": False,
        "comparison_recomputed_during_recovery": False,
        "profitability_claimed": False,
        "winner_claimed": False,
        "final_test_authorized": False,
        "production_authorized": False,
    }
    for key, expected_value in expected.items():
        if index.get(key) != expected_value:
            raise RuntimeError(f"Experiment 0002 source result-index drift: {key}")

    experiments = getattr(state, "experiments", ())
    if len(experiments) < 2:
        raise RuntimeError("Experiment 0002 state missing")
    exp2 = experiments[1]
    verification = getattr(exp2, "verification", None)
    comparison = getattr(exp2, "comparison", None)
    decision = getattr(exp2, "decision", None)
    if index.get("verification_digest") != getattr(verification, "digest", None):
        raise RuntimeError("Experiment 0002 verification/index binding drift")
    if index.get("comparison_digest") != getattr(comparison, "digest", None):
        raise RuntimeError("Experiment 0002 comparison/index binding drift")
    if index.get("decision_digest") != getattr(decision, "digest", None):
        raise RuntimeError("Experiment 0002 decision/index binding drift")


def _exp1_binding(experiment: object) -> tuple[object, object, object, object, object]:
    candidate = getattr(experiment, "candidate", None)
    evidence = getattr(candidate, "evidence", None)
    return (
        getattr(getattr(experiment, "definition", None), "digest", None),
        getattr(evidence, "fingerprint", None),
        getattr(getattr(experiment, "verification", None), "digest", None),
        getattr(getattr(experiment, "comparison", None), "digest", None),
        getattr(getattr(experiment, "decision", None), "digest", None),
    )


def _validate_source_study(root: Path, *, allow_exp003_definition: bool):
    dataset_root, study_root, _ = _paths(root)
    dataset = load_market_dataset_artifact(dataset_root)
    artifact = inspect_published_market_dataset_artifact(dataset_root)
    snapshot = inspect_study(study_root)

    if dataset.dataset_id != EXPECTED_DATASET_ID:
        raise RuntimeError("portable Dataset ID drift")
    if artifact.artifact_digest != EXPECTED_DATASET_ARTIFACT_DIGEST:
        raise RuntimeError("portable Dataset artifact digest drift")
    if snapshot.plan.digest != EXPECTED_STUDY_DIGEST:
        raise RuntimeError("portable Study digest drift")
    if snapshot.baseline is None or snapshot.baseline.fingerprint != EXPECTED_BASELINE_FINGERPRINT:
        raise RuntimeError("portable baseline EvidenceSet drift")
    if snapshot.plan.implementation_digest != EXPECTED_IMPLEMENTATION_DIGEST:
        raise RuntimeError("frozen implementation digest drift")
    if snapshot.plan.runtime_environment_digest != EXPECTED_RUNTIME_DIGEST:
        raise RuntimeError("frozen runtime digest drift")
    if tuple(snapshot.plan.ppo_seeds) != EXPECTED_SEEDS:
        raise RuntimeError("frozen PPO seed roster drift")
    if tuple(snapshot.plan.symbols) != EXPECTED_SYMBOLS:
        raise RuntimeError("frozen symbol roster drift")
    if ControlledFactor.FEATURE_SET not in snapshot.plan.allowed_factors:
        raise RuntimeError("FEATURE_SET is not allowed by frozen Study")
    if snapshot.plan.max_experiments < 3:
        raise RuntimeError("frozen Study has no Experiment 0003 budget")
    if snapshot.frozen:
        raise RuntimeError("portable Study unexpectedly frozen")

    expected_sequences = (1, 2, 3) if allow_exp003_definition else (1, 2)
    if snapshot.experiment_sequences != expected_sequences:
        raise RuntimeError("portable Study Experiment sequence drift")
    if snapshot.terminal_sequences != (1, 2):
        raise RuntimeError("prior Experiment terminal-state drift")

    provenance = build_candidate_run_provenance()
    if provenance.get("implementation_digest") != EXPECTED_IMPLEMENTATION_DIGEST:
        raise RuntimeError("current implementation differs from frozen Study")
    if provenance.get("runtime_environment_digest") != EXPECTED_RUNTIME_DIGEST:
        raise RuntimeError("current runtime differs from frozen Study")

    state = _reconstruct(StudyStore(study_root))
    exp1 = state.experiments[0]
    expected_exp1 = (
        EXPECTED_EXP001_DEFINITION_DIGEST,
        EXPECTED_EXP001_CANDIDATE_FINGERPRINT,
        EXPECTED_EXP001_VERIFICATION_DIGEST,
        EXPECTED_EXP001_COMPARISON_DIGEST,
        EXPECTED_EXP001_DECISION_DIGEST,
    )
    if _exp1_binding(exp1) != expected_exp1:
        raise RuntimeError("Experiment 0001 evidence binding drift")
    if exp1.decision.decision is not ExperimentDecisionKind.KEEP_BASELINE:
        raise RuntimeError("Experiment 0001 decision is not KEEP_BASELINE")
    if exp1.failure is not None:
        raise RuntimeError("Experiment 0001 unexpectedly contains failure evidence")

    exp2 = state.experiments[1]
    if exp2.definition.digest != EXPECTED_EXP002_DEFINITION_DIGEST:
        raise RuntimeError("Experiment 0002 definition drift")
    if exp2.definition.factor is not ControlledFactor.RULE_SIGNAL:
        raise RuntimeError("Experiment 0002 factor drift")
    if (
        exp2.candidate is None
        or exp2.candidate.evidence.fingerprint != EXPECTED_EXP002_CANDIDATE_FINGERPRINT
    ):
        raise RuntimeError("Experiment 0002 candidate evidence drift")
    if exp2.verification is None:
        raise RuntimeError("Experiment 0002 verification missing")
    if exp2.verification.status is not ControlledVerificationStatus.CONTROLLED:
        raise RuntimeError("Experiment 0002 verification is not CONTROLLED")
    if exp2.verification.changed_paths != (("signal_index",), ("signal_name",)):
        raise RuntimeError("Experiment 0002 changed-path drift")
    if exp2.comparison is None:
        raise RuntimeError("Experiment 0002 comparison missing")
    if exp2.decision is None or exp2.decision.digest != EXPECTED_EXP002_DECISION_DIGEST:
        raise RuntimeError("Experiment 0002 decision drift")
    if exp2.decision.decision is not ExperimentDecisionKind.KEEP_BASELINE:
        raise RuntimeError("Experiment 0002 decision is not KEEP_BASELINE")
    if exp2.failure is not None:
        raise RuntimeError("Experiment 0002 unexpectedly contains failure evidence")
    _validate_source_index(root, state)

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
            raise RuntimeError("Experiment 0003 prereg contains result evidence")
    return dataset, state


def _candidate_from_bootstrap(root: Path):
    _, _, bootstrap_path = _paths(root)
    config = load_canonical_m2_bootstrap_config(bootstrap_path)
    baseline = config.baseline
    if baseline.feature_names != BASELINE_FEATURE_NAMES:
        raise RuntimeError("frozen baseline feature set drift")
    candidate_names = build_candidate_feature_names(baseline.feature_names)
    candidate = replace(baseline, feature_names=candidate_names)
    return candidate, content_digest(candidate.to_json_payload())


def _validate_exp003_definition(root: Path):
    _, study_root, _ = _paths(root)
    state = _reconstruct(StudyStore(study_root))
    if len(state.experiments) != 3:
        raise RuntimeError("expected exactly three Experiments after preregistration")
    exp3 = state.experiments[2]
    definition = exp3.definition
    candidate, requested_digest = _candidate_from_bootstrap(root)
    if definition.sequence != 3 or definition.factor is not ControlledFactor.FEATURE_SET:
        raise RuntimeError("Experiment 0003 identity/factor drift")
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
    if changed != ("feature_indices", "feature_names"):
        raise RuntimeError(f"Experiment 0003 changed unexpected fields: {changed!r}")
    if tuple(candidate_payload.get("feature_names", ())) != (
        *BASELINE_FEATURE_NAMES,
        CANDIDATE_FEATURE_NAME,
    ):
        raise RuntimeError("Experiment 0003 resolved feature names drift")
    if tuple(candidate_payload.get("feature_indices", ())) != (
        *BASELINE_FEATURE_INDICES,
        CANDIDATE_FEATURE_INDEX,
    ):
        raise RuntimeError("Experiment 0003 resolved feature indices drift")
    if candidate.feature_names != (*BASELINE_FEATURE_NAMES, CANDIDATE_FEATURE_NAME):
        raise RuntimeError("Experiment 0003 candidate reconstruction drift")
    return definition, requested_digest


def preregister(root: Path, *, issue_number: int) -> dict[str, object]:
    """Append only the result-blind Experiment 0003 definition and prereg index."""

    if issue_number != ISSUE_NUMBER:
        raise RuntimeError("Experiment 0003 issue number drift")
    dataset, _ = _validate_source_study(root, allow_exp003_definition=False)
    candidate, requested_digest = _candidate_from_bootstrap(root)
    diagnostics = feature_diagnostics(
        dataset,
        fit_cutoff=candidate.fit_cutoff,
        require_frozen_roster=True,
    )
    dataset_root, study_root, _ = _paths(root)
    definition = define_experiment(
        study_root,
        dataset_root=dataset_root,
        hypothesis=HYPOTHESIS,
        factor=ControlledFactor.FEATURE_SET,
        candidate_config=candidate,
        baseline_evidence_digest=EXPECTED_BASELINE_FINGERPRINT,
    )
    _validate_source_study(root, allow_exp003_definition=True)
    verified_definition, verified_requested_digest = _validate_exp003_definition(root)
    if verified_definition.digest != definition.digest:
        raise RuntimeError("Experiment 0003 definition reconstruction mismatch")
    if verified_requested_digest != requested_digest:
        raise RuntimeError("Experiment 0003 requested config reconstruction mismatch")

    candidate_feature_names = (*BASELINE_FEATURE_NAMES, CANDIDATE_FEATURE_NAME)
    candidate_feature_indices = (*BASELINE_FEATURE_INDICES, CANDIDATE_FEATURE_INDEX)
    index: dict[str, object] = {
        "schema_version": "canonical_m2_portable_exp003_prereg_index_v1",
        "issue_number": ISSUE_NUMBER,
        "source_formal_run_id": SOURCE_FORMAL_RUN_ID,
        "source_formal_head": SOURCE_FORMAL_HEAD,
        "source_result_artifact_id": SOURCE_RESULT_ARTIFACT_ID,
        "source_result_artifact_digest": SOURCE_RESULT_ARTIFACT_DIGEST,
        "source_fresh_verify_artifact_id": SOURCE_FRESH_VERIFY_ARTIFACT_ID,
        "source_fresh_verify_artifact_digest": SOURCE_FRESH_VERIFY_ARTIFACT_DIGEST,
        "study_digest": EXPECTED_STUDY_DIGEST,
        "baseline_evidence_fingerprint": EXPECTED_BASELINE_FINGERPRINT,
        "dataset_id": EXPECTED_DATASET_ID,
        "dataset_artifact_digest": EXPECTED_DATASET_ARTIFACT_DIGEST,
        "implementation_digest": EXPECTED_IMPLEMENTATION_DIGEST,
        "runtime_environment_digest": EXPECTED_RUNTIME_DIGEST,
        "prior_experiment_sequences": [1, 2],
        "prior_experiment_decisions": ["KEEP_BASELINE", "KEEP_BASELINE"],
        "prior_exp002_decision_digest": EXPECTED_EXP002_DECISION_DIGEST,
        "experiment_sequence": 3,
        "definition_digest": definition.digest,
        "factor": "FEATURE_SET",
        "requested_candidate_config_digest": requested_digest,
        "resolved_changed_paths": ["feature_indices", "feature_names"],
        "baseline_feature_names": list(BASELINE_FEATURE_NAMES),
        "baseline_feature_indices": list(BASELINE_FEATURE_INDICES),
        "candidate_feature_name": CANDIDATE_FEATURE_NAME,
        "candidate_feature_index": CANDIDATE_FEATURE_INDEX,
        "candidate_feature_names": list(candidate_feature_names),
        "candidate_feature_indices": list(candidate_feature_indices),
        "formal_target_strategy": "lightgbm24",
        "accept_positive_factor_effect_symbols_min": 4,
        "accept_median_excess_total_return_strictly_positive": True,
        "accept_candidate_positive_total_return_symbols": 5,
        "keep_baseline_positive_factor_effect_symbols_max": 2,
        "keep_baseline_median_excess_total_return_non_positive": True,
        "keep_baseline_candidate_positive_total_return_symbols_max": 3,
        "ridge24_side_effect_must_be_disclosed": True,
        "ppo_side_effect_must_be_disclosed": True,
        "unaffected_control_strategies": list(UNAFFECTED_CONTROL_STRATEGIES),
        "unaffected_control_raw_return_equality_required": True,
        **diagnostics,
        "candidate_executed": False,
        "result_inspected_before_preregistration": False,
        "profitability_claimed": False,
        "winner_claimed": False,
        "final_test_authorized": False,
        "production_authorized": False,
    }
    path = root / "portable-exp003-prereg-index.json"
    if path.exists():
        raise RuntimeError("Experiment 0003 preregistration index already exists")
    path.write_text(
        json.dumps(index, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return index


def verify_prereg(root: Path, report_root: Path) -> dict[str, object]:
    """Reconstruct and independently verify sealed Experiment 0003 preregistration."""

    dataset, _ = _validate_source_study(root, allow_exp003_definition=True)
    definition, requested_digest = _validate_exp003_definition(root)
    candidate, candidate_digest = _candidate_from_bootstrap(root)
    if candidate_digest != requested_digest:
        raise RuntimeError("Experiment 0003 requested digest does not reproduce")
    diagnostics = feature_diagnostics(
        dataset,
        fit_cutoff=candidate.fit_cutoff,
        require_frozen_roster=True,
    )

    path = root / "portable-exp003-prereg-index.json"
    if not path.is_file():
        raise RuntimeError("Experiment 0003 preregistration index missing")
    index = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(index, dict):
        raise RuntimeError("Experiment 0003 preregistration index malformed")

    expected = {
        "schema_version": "canonical_m2_portable_exp003_prereg_index_v1",
        "issue_number": ISSUE_NUMBER,
        "source_formal_run_id": SOURCE_FORMAL_RUN_ID,
        "source_formal_head": SOURCE_FORMAL_HEAD,
        "source_result_artifact_id": SOURCE_RESULT_ARTIFACT_ID,
        "source_result_artifact_digest": SOURCE_RESULT_ARTIFACT_DIGEST,
        "source_fresh_verify_artifact_id": SOURCE_FRESH_VERIFY_ARTIFACT_ID,
        "source_fresh_verify_artifact_digest": SOURCE_FRESH_VERIFY_ARTIFACT_DIGEST,
        "study_digest": EXPECTED_STUDY_DIGEST,
        "baseline_evidence_fingerprint": EXPECTED_BASELINE_FINGERPRINT,
        "dataset_id": EXPECTED_DATASET_ID,
        "dataset_artifact_digest": EXPECTED_DATASET_ARTIFACT_DIGEST,
        "implementation_digest": EXPECTED_IMPLEMENTATION_DIGEST,
        "runtime_environment_digest": EXPECTED_RUNTIME_DIGEST,
        "prior_experiment_sequences": [1, 2],
        "prior_experiment_decisions": ["KEEP_BASELINE", "KEEP_BASELINE"],
        "prior_exp002_decision_digest": EXPECTED_EXP002_DECISION_DIGEST,
        "experiment_sequence": 3,
        "definition_digest": definition.digest,
        "factor": "FEATURE_SET",
        "requested_candidate_config_digest": requested_digest,
        "resolved_changed_paths": ["feature_indices", "feature_names"],
        "baseline_feature_names": list(BASELINE_FEATURE_NAMES),
        "baseline_feature_indices": list(BASELINE_FEATURE_INDICES),
        "candidate_feature_name": CANDIDATE_FEATURE_NAME,
        "candidate_feature_index": CANDIDATE_FEATURE_INDEX,
        "candidate_feature_names": [*BASELINE_FEATURE_NAMES, CANDIDATE_FEATURE_NAME],
        "candidate_feature_indices": [*BASELINE_FEATURE_INDICES, CANDIDATE_FEATURE_INDEX],
        "formal_target_strategy": "lightgbm24",
        "accept_positive_factor_effect_symbols_min": 4,
        "accept_median_excess_total_return_strictly_positive": True,
        "accept_candidate_positive_total_return_symbols": 5,
        "keep_baseline_positive_factor_effect_symbols_max": 2,
        "keep_baseline_median_excess_total_return_non_positive": True,
        "keep_baseline_candidate_positive_total_return_symbols_max": 3,
        "ridge24_side_effect_must_be_disclosed": True,
        "ppo_side_effect_must_be_disclosed": True,
        "unaffected_control_strategies": list(UNAFFECTED_CONTROL_STRATEGIES),
        "unaffected_control_raw_return_equality_required": True,
        "candidate_executed": False,
        "result_inspected_before_preregistration": False,
        "profitability_claimed": False,
        "winner_claimed": False,
        "final_test_authorized": False,
        "production_authorized": False,
    }
    expected_keys = set(expected) | set(diagnostics)
    if set(index) != expected_keys:
        raise RuntimeError("Experiment 0003 preregistration index key-set drift")
    for key, expected_value in expected.items():
        if index.get(key) != expected_value:
            raise RuntimeError(f"Experiment 0003 preregistration index drift: {key}")
    for key, expected_value in diagnostics.items():
        if index.get(key) != expected_value:
            raise RuntimeError(f"Experiment 0003 feature diagnostic drift: {key}")

    report_root.mkdir(parents=True, exist_ok=True)
    report = {
        "schema_version": "canonical_m2_portable_exp003_prereg_verification_v1",
        "study_digest": EXPECTED_STUDY_DIGEST,
        "baseline_evidence_fingerprint": EXPECTED_BASELINE_FINGERPRINT,
        "definition_digest": definition.digest,
        "requested_candidate_config_digest": requested_digest,
        "prior_exp002_decision_digest": EXPECTED_EXP002_DECISION_DIGEST,
        "resolved_changed_paths": ["feature_indices", "feature_names"],
        "candidate_feature_name": CANDIDATE_FEATURE_NAME,
        "candidate_feature_index": CANDIDATE_FEATURE_INDEX,
        **diagnostics,
        "prior_exp001_exp002_evidence_unchanged": True,
        "current_provenance_matches_frozen_study": True,
        "definition_is_result_blind": True,
        "candidate_absent": True,
        "verification_passed": True,
    }
    (report_root / "exp003-prereg-verification.json").write_text(
        json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    prereg = subparsers.add_parser("preregister")
    prereg.add_argument("--root", type=Path, required=True)
    prereg.add_argument("--issue-number", type=int, required=True)

    verify = subparsers.add_parser("verify-prereg")
    verify.add_argument("--root", type=Path, required=True)
    verify.add_argument("--report-root", type=Path, required=True)

    args = parser.parse_args()
    if args.command == "preregister":
        preregister(args.root, issue_number=args.issue_number)
    else:
        verify_prereg(args.root, args.report_root)


if __name__ == "__main__":
    main()


__all__ = [
    "BASELINE_FEATURE_NAMES",
    "CANDIDATE_FEATURE_NAME",
    "EXPECTED_DATASET_ID",
    "ExperimentDecisionKind",
    "build_candidate_feature_names",
    "feature_diagnostics",
    "formal_decision",
    "preregister",
    "verify_prereg",
]
