"""Pre-result preregistration for portable Canonical M2 Experiment 0002."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import replace
from pathlib import Path
from statistics import median

import numpy as np

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
    load_evidence_set,
)
from trade_rl.evaluation.experiments.bootstrap.config import (
    load_canonical_m2_bootstrap_config,
)
from trade_rl.evaluation.experiments.inspection import _reconstruct
from trade_rl.evaluation.experiments.store import StudyStore
from trade_rl.evaluation.runs import build_candidate_run_provenance

EXPECTED_STUDY_DIGEST = (
    "3d8404061a4082a8e9b3c786d9f5fc9a4347631dff39c201e3cba70470dfeb79"
)
EXPECTED_BASELINE_FINGERPRINT = (
    "526b485d60b394739b7a8d03535e1cfa08b26fa920119c70ee53f05faa53dd29"
)
EXPECTED_DATASET_ID = (
    "d7a04ede97a1bb37b811c3e071f325fa007525a6040927e6793d8cc7c10f538f"
)
EXPECTED_DATASET_ARTIFACT_DIGEST = (
    "77362e148c713840dda64e0ef70e663cce6611407eac31fefbb9fccca73ae8f8"
)
EXPECTED_IMPLEMENTATION_DIGEST = (
    "4e74c99ea86f347e024d701396ac58f24eb49affb30a6c283f385de2a839f8f7"
)
EXPECTED_RUNTIME_DIGEST = (
    "6dbc9cffd844837e17741ae30681f09a6d84b0a49dec011a4fc328f97ade1d85"
)
EXPECTED_EXP001_DEFINITION_DIGEST = (
    "8edbba0ad9cdb938b9b10a65e88b940dbbd4c37b1000fe29acadb5d209d52f28"
)
EXPECTED_EXP001_CANDIDATE_FINGERPRINT = (
    "5e1ee8a530097e6b1c4571efe1d3edd9e867a2e77b8346a558ce9ed385e1f672"
)
EXPECTED_EXP001_VERIFICATION_DIGEST = (
    "b331fbb8ba76b2ae08a1febe971f586eae90f73ccca38c5f0ea326b4c802bec4"
)
EXPECTED_EXP001_COMPARISON_DIGEST = (
    "6f70b9e3a462301fef9473e502bcf5ed875bd0042a1df0123f4e681c76b666bb"
)
EXPECTED_EXP001_DECISION_DIGEST = (
    "cf318e67013d208024e188317c38ae4b30017f449000d0ffa75c72aedbd71b3c"
)
EXPECTED_BASELINE_MEDIAN_TURNOVER = 858.3114067468092
EXPECTED_SYMBOLS = ("BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT")
EXPECTED_SEEDS = (0, 1, 2, 3, 4)
EXPECTED_DAILY_REFRESH_COUNTS = {symbol: 731 for symbol in EXPECTED_SYMBOLS}
EXPECTED_BASELINE_TRANSITIONS = {
    "BTCUSDT": 1387,
    "ETHUSDT": 1601,
    "BNBUSDT": 1601,
    "XRPUSDT": 1739,
    "ADAUSDT": 1835,
}
EXPECTED_CANDIDATE_TRANSITIONS = {
    "BTCUSDT": 423,
    "ETHUSDT": 425,
    "BNBUSDT": 416,
    "XRPUSDT": 457,
    "ADAUSDT": 396,
}
SOURCE_RECOVERY_RUN_ID = 34733780367
SOURCE_RECOVERY_HEAD = "17897b924dde12bbcc38a5740d15e96ed9aaf00c"
SOURCE_RESULT_ARTIFACT_ID = 10309859809
SOURCE_RESULT_ARTIFACT_DIGEST = (
    "sha256:cbada8936bc361e7dfcd05c6c213da97e3225a69dca7d0a79a0533e49fd2aaca"
)
BASELINE_SIGNAL_NAME = "1h__log_return_24bar"
BASELINE_SIGNAL_INDEX = 2
CANDIDATE_SIGNAL_NAME = "1d__log_return_1bar"
CANDIDATE_SIGNAL_INDEX = 114
HYPOTHESIS = (
    "Replacing the rolling hourly-updated 24h return signal with the point-in-time "
    "daily-updated 1-bar daily return signal reduces redundant mean-reversion state "
    "changes and turnover while preserving the same 24h return information at daily "
    "refreshes, improving net-of-cost mean-reversion returns across the frozen symbols."
)


def _paths(root: Path) -> tuple[Path, Path, Path]:
    dataset_root = root / "dataset"
    study_root = root / "study"
    bootstrap_path = root / "bootstrap.json"
    if not dataset_root.is_dir() or not study_root.is_dir() or not bootstrap_path.is_file():
        raise RuntimeError("portable recovered artifact layout is incomplete")
    return dataset_root, study_root, bootstrap_path


def _metric_number(metrics: object, name: str) -> float:
    if not isinstance(metrics, dict):
        raise RuntimeError("strategy metrics malformed")
    value = metrics.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError(f"strategy metric malformed: {name}")
    resolved = float(value)
    if not math.isfinite(resolved):
        raise RuntimeError(f"strategy metric non-finite: {name}")
    return resolved


def _strategy_entry(run, symbol_index: int, name: str) -> dict[str, object]:
    by_symbol = run.summary.get("by_symbol")
    if not isinstance(by_symbol, list) or len(by_symbol) != len(EXPECTED_SYMBOLS):
        raise RuntimeError("candidate summary symbol roster malformed")
    symbol = by_symbol[symbol_index]
    if not isinstance(symbol, dict) or symbol.get("symbol") != EXPECTED_SYMBOLS[symbol_index]:
        raise RuntimeError("candidate summary symbol ordering drift")
    strategies = symbol.get("strategies")
    if not isinstance(strategies, list):
        raise RuntimeError("candidate summary strategy roster malformed")
    matches = [
        item
        for item in strategies
        if isinstance(item, dict) and item.get("name") == name
    ]
    if len(matches) != 1:
        raise RuntimeError(f"strategy entry missing or duplicated: {name}")
    return matches[0]


def _intent_transition_count(
    signal: np.ndarray,
    available: np.ndarray,
    *,
    entry_threshold: float,
    exit_threshold: float,
) -> int:
    """Replay only the frozen mean-reversion intent state machine."""

    values = np.asarray(signal, dtype=np.float64).reshape(-1)
    availability = np.asarray(available, dtype=np.bool_).reshape(-1)
    if values.shape != availability.shape or values.size == 0:
        raise ValueError("signal and availability must be non-empty equal-length vectors")
    if not math.isfinite(entry_threshold) or entry_threshold <= 0.0:
        raise ValueError("entry_threshold must be finite and positive")
    if (
        not math.isfinite(exit_threshold)
        or exit_threshold < 0.0
        or exit_threshold >= entry_threshold
    ):
        raise ValueError("exit_threshold must be finite and below entry_threshold")

    # -1 = SHORT, 0 = FLAT, 1 = LONG.
    current = 0
    changes = 0
    for raw, is_available in zip(values, availability, strict=True):
        if not is_available or not math.isfinite(float(raw)):
            next_intent = 0
        else:
            value = float(raw)
            if current == 0:
                if value >= entry_threshold:
                    next_intent = -1
                elif value <= -entry_threshold:
                    next_intent = 1
                else:
                    next_intent = 0
            elif current == 1:
                if value >= entry_threshold:
                    next_intent = -1
                elif value >= -exit_threshold:
                    next_intent = 0
                else:
                    next_intent = 1
            else:
                if value <= -entry_threshold:
                    next_intent = 1
                elif value <= exit_threshold:
                    next_intent = 0
                else:
                    next_intent = -1
        if next_intent != current:
            changes += 1
        current = next_intent
    return changes


def _verify_daily_signal_refresh_alignment(
    rolling_signal: np.ndarray,
    daily_signal: np.ndarray,
    daily_available: np.ndarray,
    daily_staleness: np.ndarray,
) -> int:
    """Require exact rolling/daily 24h-return equality at daily refresh points."""

    rolling = np.asarray(rolling_signal, dtype=np.float64).reshape(-1)
    daily = np.asarray(daily_signal, dtype=np.float64).reshape(-1)
    available = np.asarray(daily_available, dtype=np.bool_).reshape(-1)
    staleness = np.asarray(daily_staleness, dtype=np.float64).reshape(-1)
    if not (
        rolling.shape == daily.shape == available.shape == staleness.shape
        and rolling.size > 0
    ):
        raise ValueError("signal alignment inputs must be non-empty equal-length vectors")
    if not np.isfinite(staleness).all() or np.any(staleness < 0.0):
        raise ValueError("daily staleness must be finite and non-negative")

    refresh = available & (staleness == 0.0)
    count = int(np.count_nonzero(refresh))
    if count == 0:
        raise RuntimeError("no daily signal refresh points found")
    rolling_refresh = rolling[refresh]
    daily_refresh = daily[refresh]
    if not np.isfinite(rolling_refresh).all() or not np.isfinite(daily_refresh).all():
        raise RuntimeError("daily/rolling 24h signal refresh contains non-finite values")
    if not np.array_equal(rolling_refresh, daily_refresh):
        raise RuntimeError("daily/rolling 24h signal mismatch")
    return count


def _baseline_turnover(study_root: Path) -> float:
    loaded = load_evidence_set(study_root / "baseline/evidence")
    if loaded.evidence.fingerprint != EXPECTED_BASELINE_FINGERPRINT:
        raise RuntimeError("loaded baseline EvidenceSet drift")
    if loaded.evidence.ppo_seeds != EXPECTED_SEEDS:
        raise RuntimeError("baseline PPO seed roster drift")
    if tuple(sorted(loaded.runs)) != EXPECTED_SEEDS:
        raise RuntimeError("baseline Run seed roster drift")

    reference: dict[str, np.ndarray] = {}
    turnovers: list[float] = []
    for seed in EXPECTED_SEEDS:
        run = loaded.runs[seed]
        for symbol_index, symbol in enumerate(EXPECTED_SYMBOLS):
            entry = _strategy_entry(run, symbol_index, "mean_reversion")
            return_key = entry.get("return_key")
            if not isinstance(return_key, str):
                raise RuntimeError("baseline mean-reversion return key malformed")
            values = run.returns.get(return_key)
            if values is None:
                raise RuntimeError("baseline mean-reversion raw returns missing")
            if seed == EXPECTED_SEEDS[0]:
                reference[symbol] = values.copy()
                turnovers.append(
                    _metric_number(entry.get("metrics"), "turnover_total")
                )
            elif not np.array_equal(reference[symbol], values):
                raise RuntimeError("baseline mean-reversion is seed-sensitive")
    resolved = float(median(turnovers))
    if resolved != EXPECTED_BASELINE_MEDIAN_TURNOVER:
        raise RuntimeError("frozen baseline mean-reversion turnover drift")
    return resolved


def _validate_recovered_study(root: Path, *, allow_exp002_definition: bool):
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
    if ControlledFactor.RULE_SIGNAL not in snapshot.plan.allowed_factors:
        raise RuntimeError("RULE_SIGNAL is not allowed by the frozen Study")
    if snapshot.plan.max_experiments < 2:
        raise RuntimeError("frozen Study has no Experiment 0002 budget")
    if snapshot.frozen:
        raise RuntimeError("portable Study unexpectedly frozen")

    expected_sequences = (1, 2) if allow_exp002_definition else (1,)
    if snapshot.experiment_sequences != expected_sequences:
        raise RuntimeError("portable Study Experiment sequence drift")
    if snapshot.terminal_sequences != (1,):
        raise RuntimeError("Experiment 0001 terminal-state drift")

    provenance = build_candidate_run_provenance()
    if provenance.get("implementation_digest") != EXPECTED_IMPLEMENTATION_DIGEST:
        raise RuntimeError("current implementation differs from frozen Study")
    if provenance.get("runtime_environment_digest") != EXPECTED_RUNTIME_DIGEST:
        raise RuntimeError("current runtime differs from frozen Study")

    state = _reconstruct(StudyStore(study_root))
    exp1 = state.experiments[0]
    if exp1.definition.digest != EXPECTED_EXP001_DEFINITION_DIGEST:
        raise RuntimeError("Experiment 0001 definition drift")
    if (
        exp1.candidate is None
        or exp1.candidate.evidence.fingerprint != EXPECTED_EXP001_CANDIDATE_FINGERPRINT
    ):
        raise RuntimeError("Experiment 0001 candidate evidence drift")
    if (
        exp1.verification is None
        or exp1.verification.digest != EXPECTED_EXP001_VERIFICATION_DIGEST
    ):
        raise RuntimeError("Experiment 0001 verification drift")
    if (
        exp1.comparison is None
        or exp1.comparison.digest != EXPECTED_EXP001_COMPARISON_DIGEST
    ):
        raise RuntimeError("Experiment 0001 comparison drift")
    if exp1.decision is None or exp1.decision.digest != EXPECTED_EXP001_DECISION_DIGEST:
        raise RuntimeError("Experiment 0001 decision drift")
    if exp1.decision.decision is not ExperimentDecisionKind.KEEP_BASELINE:
        raise RuntimeError("Experiment 0001 decision is not KEEP_BASELINE")
    if exp1.failure is not None:
        raise RuntimeError("Experiment 0001 unexpectedly contains failure evidence")

    if allow_exp002_definition:
        exp2 = state.experiments[1]
        if any(
            value is not None
            for value in (
                exp2.candidate,
                exp2.verification,
                exp2.comparison,
                exp2.decision,
                exp2.failure,
            )
        ):
            raise RuntimeError("Experiment 0002 preregistration contains result evidence")
    if _baseline_turnover(study_root) != EXPECTED_BASELINE_MEDIAN_TURNOVER:
        raise RuntimeError("baseline turnover reconstruction drift")
    return dataset, state


def _candidate_from_bootstrap(root: Path):
    _, _, bootstrap_path = _paths(root)
    config = load_canonical_m2_bootstrap_config(bootstrap_path)
    baseline = config.baseline
    if baseline.signal_name != BASELINE_SIGNAL_NAME:
        raise RuntimeError("frozen baseline signal name drift")
    if baseline.rule_entry_threshold != 0.01 or baseline.rule_exit_threshold != 0.0025:
        raise RuntimeError("frozen baseline rule thresholds drift")
    candidate = replace(baseline, signal_name=CANDIDATE_SIGNAL_NAME)
    return candidate, content_digest(candidate.to_json_payload())


def _signal_diagnostics(dataset, baseline_config) -> dict[str, object]:
    try:
        baseline_index = dataset.feature_names.index(BASELINE_SIGNAL_NAME)
        candidate_index = dataset.feature_names.index(CANDIDATE_SIGNAL_NAME)
    except ValueError as error:
        raise RuntimeError("Experiment 0002 signal feature is missing") from error
    if baseline_index != BASELINE_SIGNAL_INDEX or candidate_index != CANDIDATE_SIGNAL_INDEX:
        raise RuntimeError("Experiment 0002 signal feature index drift")

    timestamps = dataset.timestamps.astype("datetime64[ns]")
    start_matches = np.flatnonzero(timestamps == baseline_config.evaluation_start)
    stop_matches = np.flatnonzero(timestamps == baseline_config.evaluation_stop_exclusive)
    if start_matches.size != 1 or stop_matches.size != 1:
        raise RuntimeError("frozen evaluation timestamps no longer resolve exactly")
    start = int(start_matches[0])
    stop = int(stop_matches[0])
    if not start < stop:
        raise RuntimeError("frozen evaluation range is invalid")

    staleness = dataset.resolved_array("feature_staleness")
    refresh_counts: dict[str, int] = {}
    baseline_transitions: dict[str, int] = {}
    candidate_transitions: dict[str, int] = {}
    for symbol_index, symbol in enumerate(EXPECTED_SYMBOLS):
        rolling = dataset.features[start:stop, symbol_index, baseline_index]
        daily = dataset.features[start:stop, symbol_index, candidate_index]
        rolling_available = dataset.feature_available[
            start:stop, symbol_index, baseline_index
        ]
        daily_available = dataset.feature_available[
            start:stop, symbol_index, candidate_index
        ]
        daily_staleness = staleness[start:stop, symbol_index, candidate_index]
        refresh_counts[symbol] = _verify_daily_signal_refresh_alignment(
            rolling,
            daily,
            daily_available,
            daily_staleness,
        )
        baseline_transitions[symbol] = _intent_transition_count(
            rolling,
            rolling_available,
            entry_threshold=baseline_config.rule_entry_threshold,
            exit_threshold=baseline_config.rule_exit_threshold,
        )
        candidate_transitions[symbol] = _intent_transition_count(
            daily,
            daily_available,
            entry_threshold=baseline_config.rule_entry_threshold,
            exit_threshold=baseline_config.rule_exit_threshold,
        )

    if refresh_counts != EXPECTED_DAILY_REFRESH_COUNTS:
        raise RuntimeError("daily signal refresh roster drift")
    if baseline_transitions != EXPECTED_BASELINE_TRANSITIONS:
        raise RuntimeError("baseline intent-transition diagnostic drift")
    if candidate_transitions != EXPECTED_CANDIDATE_TRANSITIONS:
        raise RuntimeError("candidate intent-transition diagnostic drift")
    return {
        "daily_refresh_counts": refresh_counts,
        "baseline_intent_transitions": baseline_transitions,
        "candidate_intent_transitions": candidate_transitions,
        "baseline_intent_transition_median": float(
            median(baseline_transitions.values())
        ),
        "candidate_intent_transition_median": float(
            median(candidate_transitions.values())
        ),
    }


def _validate_exp002_definition(root: Path):
    _, study_root, _ = _paths(root)
    state = _reconstruct(StudyStore(study_root))
    if len(state.experiments) != 2:
        raise RuntimeError("expected exactly two Experiments after preregistration")
    exp2 = state.experiments[1]
    definition = exp2.definition
    candidate, requested_digest = _candidate_from_bootstrap(root)
    if definition.sequence != 2 or definition.factor is not ControlledFactor.RULE_SIGNAL:
        raise RuntimeError("Experiment 0002 identity/factor drift")
    if definition.study_digest != EXPECTED_STUDY_DIGEST:
        raise RuntimeError("Experiment 0002 Study binding drift")
    if definition.baseline_evidence_digest != EXPECTED_BASELINE_FINGERPRINT:
        raise RuntimeError("Experiment 0002 baseline binding drift")
    if definition.hypothesis != HYPOTHESIS:
        raise RuntimeError("Experiment 0002 hypothesis drift")
    if definition.candidate_requested_config_digest != requested_digest:
        raise RuntimeError("Experiment 0002 requested config digest drift")

    baseline_payload = state.plan.baseline_config.to_payload()
    candidate_payload = definition.candidate_config.to_payload()
    changed = tuple(
        key
        for key in sorted(set(baseline_payload) | set(candidate_payload))
        if baseline_payload.get(key) != candidate_payload.get(key)
    )
    if changed != ("signal_index", "signal_name"):
        raise RuntimeError(f"Experiment 0002 changed unexpected fields: {changed!r}")
    if candidate.signal_name != CANDIDATE_SIGNAL_NAME:
        raise RuntimeError("candidate signal reconstruction drift")
    if candidate_payload.get("signal_name") != CANDIDATE_SIGNAL_NAME:
        raise RuntimeError("resolved candidate signal name drift")
    if candidate_payload.get("signal_index") != CANDIDATE_SIGNAL_INDEX:
        raise RuntimeError("resolved candidate signal index drift")
    return definition, requested_digest


def preregister(root: Path, *, issue_number: int) -> dict[str, object]:
    """Append only the result-blind Experiment 0002 definition and prereg index."""

    dataset, _ = _validate_recovered_study(root, allow_exp002_definition=False)
    candidate, requested_digest = _candidate_from_bootstrap(root)
    diagnostics = _signal_diagnostics(dataset, candidate)
    dataset_root, study_root, _ = _paths(root)
    definition = define_experiment(
        study_root,
        dataset_root=dataset_root,
        hypothesis=HYPOTHESIS,
        factor=ControlledFactor.RULE_SIGNAL,
        candidate_config=candidate,
        baseline_evidence_digest=EXPECTED_BASELINE_FINGERPRINT,
    )
    _validate_recovered_study(root, allow_exp002_definition=True)
    verified_definition, verified_requested_digest = _validate_exp002_definition(root)
    if verified_definition.digest != definition.digest:
        raise RuntimeError("Experiment 0002 definition reconstruction mismatch")
    if verified_requested_digest != requested_digest:
        raise RuntimeError("Experiment 0002 requested config reconstruction mismatch")

    index: dict[str, object] = {
        "schema_version": "canonical_m2_portable_exp002_prereg_index_v1",
        "issue_number": issue_number,
        "source_recovery_run_id": SOURCE_RECOVERY_RUN_ID,
        "source_recovery_head": SOURCE_RECOVERY_HEAD,
        "source_recovered_result_artifact_id": SOURCE_RESULT_ARTIFACT_ID,
        "source_recovered_result_artifact_digest": SOURCE_RESULT_ARTIFACT_DIGEST,
        "study_digest": EXPECTED_STUDY_DIGEST,
        "baseline_evidence_fingerprint": EXPECTED_BASELINE_FINGERPRINT,
        "dataset_id": EXPECTED_DATASET_ID,
        "dataset_artifact_digest": EXPECTED_DATASET_ARTIFACT_DIGEST,
        "implementation_digest": EXPECTED_IMPLEMENTATION_DIGEST,
        "runtime_environment_digest": EXPECTED_RUNTIME_DIGEST,
        "prior_experiment_sequence": 1,
        "prior_experiment_decision": "KEEP_BASELINE",
        "prior_experiment_decision_digest": EXPECTED_EXP001_DECISION_DIGEST,
        "experiment_sequence": 2,
        "definition_digest": definition.digest,
        "factor": definition.factor.value,
        "requested_candidate_config_digest": requested_digest,
        "resolved_changed_paths": ["signal_index", "signal_name"],
        "baseline_signal_name": BASELINE_SIGNAL_NAME,
        "baseline_signal_index": BASELINE_SIGNAL_INDEX,
        "candidate_signal_name": CANDIDATE_SIGNAL_NAME,
        "candidate_signal_index": CANDIDATE_SIGNAL_INDEX,
        "rule_entry_threshold": candidate.rule_entry_threshold,
        "rule_exit_threshold": candidate.rule_exit_threshold,
        "frozen_baseline_mean_reversion_median_turnover": (
            EXPECTED_BASELINE_MEDIAN_TURNOVER
        ),
        "formal_target_strategy": "mean_reversion",
        "accept_positive_factor_effect_symbols_min": 4,
        "accept_median_excess_total_return_strictly_positive": True,
        "accept_candidate_positive_total_return_symbols": 5,
        "accept_candidate_turnover_strictly_below_baseline": True,
        "keep_baseline_positive_factor_effect_symbols_max": 2,
        "keep_baseline_median_excess_total_return_non_positive": True,
        "keep_baseline_candidate_positive_total_return_symbols_max": 3,
        "trend_side_effect_must_be_disclosed": True,
        **diagnostics,
        "candidate_executed": False,
        "result_inspected_before_preregistration": False,
        "profitability_claimed": False,
        "winner_claimed": False,
        "final_test_authorized": False,
        "production_authorized": False,
    }
    index_path = root / "portable-exp002-prereg-index.json"
    if index_path.exists():
        raise RuntimeError("Experiment 0002 preregistration index already exists")
    with index_path.open("x", encoding="utf-8") as handle:
        json.dump(index, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")
    print("PORTABLE_EXP002_PREREG=" + json.dumps(index, sort_keys=True))
    return index


def verify_prereg(root: Path, report_root: Path) -> dict[str, object]:
    """Independently reconstruct and verify the sealed Experiment 0002 prereg."""

    dataset, _ = _validate_recovered_study(root, allow_exp002_definition=True)
    definition, requested_digest = _validate_exp002_definition(root)
    candidate, candidate_digest = _candidate_from_bootstrap(root)
    if candidate_digest != requested_digest:
        raise RuntimeError("candidate requested digest does not independently reproduce")
    diagnostics = _signal_diagnostics(dataset, candidate)

    index_path = root / "portable-exp002-prereg-index.json"
    if not index_path.is_file():
        raise RuntimeError("Experiment 0002 preregistration index missing")
    index = json.loads(index_path.read_text(encoding="utf-8"))
    expected_pairs = {
        "schema_version": "canonical_m2_portable_exp002_prereg_index_v1",
        "issue_number": 519,
        "source_recovery_run_id": SOURCE_RECOVERY_RUN_ID,
        "source_recovery_head": SOURCE_RECOVERY_HEAD,
        "source_recovered_result_artifact_id": SOURCE_RESULT_ARTIFACT_ID,
        "source_recovered_result_artifact_digest": SOURCE_RESULT_ARTIFACT_DIGEST,
        "study_digest": EXPECTED_STUDY_DIGEST,
        "baseline_evidence_fingerprint": EXPECTED_BASELINE_FINGERPRINT,
        "dataset_id": EXPECTED_DATASET_ID,
        "dataset_artifact_digest": EXPECTED_DATASET_ARTIFACT_DIGEST,
        "implementation_digest": EXPECTED_IMPLEMENTATION_DIGEST,
        "runtime_environment_digest": EXPECTED_RUNTIME_DIGEST,
        "prior_experiment_sequence": 1,
        "prior_experiment_decision": "KEEP_BASELINE",
        "prior_experiment_decision_digest": EXPECTED_EXP001_DECISION_DIGEST,
        "experiment_sequence": 2,
        "definition_digest": definition.digest,
        "factor": "RULE_SIGNAL",
        "requested_candidate_config_digest": requested_digest,
        "resolved_changed_paths": ["signal_index", "signal_name"],
        "baseline_signal_name": BASELINE_SIGNAL_NAME,
        "baseline_signal_index": BASELINE_SIGNAL_INDEX,
        "candidate_signal_name": CANDIDATE_SIGNAL_NAME,
        "candidate_signal_index": CANDIDATE_SIGNAL_INDEX,
        "rule_entry_threshold": 0.01,
        "rule_exit_threshold": 0.0025,
        "candidate_executed": False,
        "result_inspected_before_preregistration": False,
        "profitability_claimed": False,
        "winner_claimed": False,
        "final_test_authorized": False,
        "production_authorized": False,
    }
    for key, expected in expected_pairs.items():
        if index.get(key) != expected:
            raise RuntimeError(f"Experiment 0002 preregistration index drift: {key}")
    for key, expected in diagnostics.items():
        if index.get(key) != expected:
            raise RuntimeError(f"Experiment 0002 diagnostic drift: {key}")

    report_root.mkdir(parents=True, exist_ok=True)
    report: dict[str, object] = {
        "schema_version": "canonical_m2_portable_exp002_prereg_verification_v1",
        "study_digest": EXPECTED_STUDY_DIGEST,
        "baseline_evidence_fingerprint": EXPECTED_BASELINE_FINGERPRINT,
        "definition_digest": definition.digest,
        "requested_candidate_config_digest": requested_digest,
        "prior_exp001_decision_digest": EXPECTED_EXP001_DECISION_DIGEST,
        "resolved_changed_paths": ["signal_index", "signal_name"],
        "daily_signal_refresh_equality_verified": True,
        **diagnostics,
        "prior_exp001_evidence_unchanged": True,
        "current_provenance_matches_frozen_study": True,
        "definition_is_result_blind": True,
        "candidate_absent": True,
        "verification_passed": True,
    }
    (report_root / "exp002-prereg-verification.json").write_text(
        json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print("PORTABLE_EXP002_PREREG_VERIFY=" + json.dumps(report, sort_keys=True))
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
    "_intent_transition_count",
    "_verify_daily_signal_refresh_alignment",
    "preregister",
    "verify_prereg",
]
