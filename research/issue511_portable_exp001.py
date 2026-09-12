"""Fail-closed preregistration/verifier for portable Canonical M2 Experiment 0001."""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import replace
from pathlib import Path
from statistics import median

import numpy as np

from trade_rl.artifacts.hashing import content_digest
from trade_rl.data import inspect_published_market_dataset_artifact, load_market_dataset_artifact
from trade_rl.evaluation.experiments import (
    ControlledFactor,
    define_experiment,
    inspect_study,
    load_evidence_set,
)
from trade_rl.evaluation.experiments.bootstrap.config import load_canonical_m2_bootstrap_config
from trade_rl.evaluation.experiments.inspection import _reconstruct
from trade_rl.evaluation.experiments.store import StudyStore
from trade_rl.evaluation.runs import build_candidate_run_provenance

EXPECTED_STUDY_DIGEST = "3d8404061a4082a8e9b3c786d9f5fc9a4347631dff39c201e3cba70470dfeb79"
EXPECTED_BASELINE_FINGERPRINT = "526b485d60b394739b7a8d03535e1cfa08b26fa920119c70ee53f05faa53dd29"
EXPECTED_DATASET_ID = "d7a04ede97a1bb37b811c3e071f325fa007525a6040927e6793d8cc7c10f538f"
EXPECTED_DATASET_ARTIFACT_DIGEST = "77362e148c713840dda64e0ef70e663cce6611407eac31fefbb9fccca73ae8f8"
EXPECTED_IMPLEMENTATION_DIGEST = "4e74c99ea86f347e024d701396ac58f24eb49affb30a6c283f385de2a839f8f7"
EXPECTED_RUNTIME_DIGEST = "6dbc9cffd844837e17741ae30681f09a6d84b0a49dec011a4fc328f97ade1d85"
EXPECTED_REQUESTED_CANDIDATE_DIGEST = "668f5386994efddc14f37c06e4031c86f01ff741c5083a02876b043d91e53dd6"
EXPECTED_SEEDS = (0, 1, 2, 3, 4)
EXPECTED_SYMBOLS = ("BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT")
HYPOTHESIS = (
    "Raising rule_entry_threshold from 0.01 to 0.02 reduces low-conviction "
    "mean-reversion entries enough to improve net-of-cost mean-reversion returns "
    "across the five frozen symbols while reducing turnover."
)


def _required_env(name: str) -> str:
    value = os.environ.get(name)
    if value is None or not value:
        raise RuntimeError(f"required environment variable missing: {name}")
    return value


def _paths(root: Path) -> tuple[Path, Path, Path]:
    dataset_root = root / "dataset"
    study_root = root / "study"
    bootstrap_path = root / "bootstrap.json"
    if not dataset_root.is_dir() or not study_root.is_dir() or not bootstrap_path.is_file():
        raise RuntimeError("portable baseline artifact layout is incomplete")
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
    matches = [item for item in strategies if isinstance(item, dict) and item.get("name") == name]
    if len(matches) != 1:
        raise RuntimeError(f"strategy entry missing or duplicated: {name}")
    return matches[0]


def _validate_baseline(root: Path, *, allow_definition: bool) -> tuple[object, float]:
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
    if snapshot.frozen:
        raise RuntimeError("portable Study unexpectedly frozen")
    expected_sequences = (1,) if allow_definition else ()
    if snapshot.experiment_sequences != expected_sequences or snapshot.terminal_sequences:
        raise RuntimeError("portable Study experiment lifecycle drift")

    provenance = build_candidate_run_provenance()
    if provenance.get("implementation_digest") != snapshot.plan.implementation_digest:
        raise RuntimeError("current implementation provenance differs from frozen Study")
    if provenance.get("runtime_environment_digest") != snapshot.plan.runtime_environment_digest:
        raise RuntimeError("current runtime provenance differs from frozen Study")

    loaded = load_evidence_set(study_root / "baseline" / "evidence")
    if loaded.evidence.fingerprint != EXPECTED_BASELINE_FINGERPRINT:
        raise RuntimeError("loaded baseline fingerprint drift")
    if tuple(sorted(loaded.runs)) != EXPECTED_SEEDS:
        raise RuntimeError("loaded baseline run roster drift")

    reference_returns: dict[str, np.ndarray] = {}
    turnovers: list[float] = []
    for seed in EXPECTED_SEEDS:
        run = loaded.runs[seed]
        for symbol_index, symbol in enumerate(EXPECTED_SYMBOLS):
            entry = _strategy_entry(run, symbol_index, "mean_reversion")
            return_key = entry.get("return_key")
            if not isinstance(return_key, str) or return_key not in run.returns:
                raise RuntimeError("mean_reversion raw returns missing")
            values = run.returns[return_key]
            key = symbol
            if seed == EXPECTED_SEEDS[0]:
                reference_returns[key] = values.copy()
                turnovers.append(_metric_number(entry.get("metrics"), "turnover_total"))
            elif not np.array_equal(reference_returns[key], values):
                raise RuntimeError("mean_reversion baseline is seed-sensitive")

    if len(turnovers) != len(EXPECTED_SYMBOLS):
        raise RuntimeError("baseline turnover roster incomplete")
    return snapshot, float(median(turnovers))


def _candidate_from_bootstrap(root: Path):
    _, _, bootstrap_path = _paths(root)
    config = load_canonical_m2_bootstrap_config(bootstrap_path)
    baseline = config.baseline
    if baseline.rule_entry_threshold != 0.01 or baseline.rule_exit_threshold != 0.0025:
        raise RuntimeError("baseline rule thresholds differ from frozen protocol")
    candidate = replace(baseline, rule_entry_threshold=0.02)
    digest = content_digest(candidate.to_json_payload())
    if digest != EXPECTED_REQUESTED_CANDIDATE_DIGEST:
        raise RuntimeError(f"candidate requested-config digest drift: {digest}")
    return candidate, digest


def _validate_definition(root: Path):
    _, study_root, _ = _paths(root)
    state = _reconstruct(StudyStore(study_root))
    if len(state.experiments) != 1:
        raise RuntimeError("expected exactly one preregistered Experiment")
    experiment = state.experiments[0]
    if any(
        value is not None
        for value in (
            experiment.candidate,
            experiment.verification,
            experiment.comparison,
            experiment.decision,
            experiment.failure,
        )
    ):
        raise RuntimeError("preregistration contains post-definition evidence")
    definition = experiment.definition
    if definition.sequence != 1 or definition.factor is not ControlledFactor.RULE_THRESHOLDS:
        raise RuntimeError("Experiment identity/factor drift")
    if definition.study_digest != EXPECTED_STUDY_DIGEST:
        raise RuntimeError("Experiment Study binding drift")
    if definition.baseline_evidence_digest != EXPECTED_BASELINE_FINGERPRINT:
        raise RuntimeError("Experiment baseline binding drift")
    if definition.hypothesis != HYPOTHESIS:
        raise RuntimeError("Experiment hypothesis drift")
    if definition.candidate_requested_config_digest != EXPECTED_REQUESTED_CANDIDATE_DIGEST:
        raise RuntimeError("Experiment requested-config digest drift")

    baseline_payload = state.plan.baseline_config.to_payload()
    candidate_payload = definition.candidate_config.to_payload()
    changed = tuple(
        key
        for key in sorted(set(baseline_payload) | set(candidate_payload))
        if baseline_payload.get(key) != candidate_payload.get(key)
    )
    if changed != ("rule_entry_threshold",):
        raise RuntimeError(f"resolved candidate changes unexpected fields: {changed!r}")
    if candidate_payload["rule_entry_threshold"] != 0.02:
        raise RuntimeError("resolved candidate rule entry threshold drift")
    return definition


def preregister(root: Path, issue_number: int) -> None:
    snapshot, baseline_turnover = _validate_baseline(root, allow_definition=False)
    candidate, requested_digest = _candidate_from_bootstrap(root)
    dataset_root, study_root, _ = _paths(root)
    definition = define_experiment(
        study_root,
        dataset_root=dataset_root,
        hypothesis=HYPOTHESIS,
        factor=ControlledFactor.RULE_THRESHOLDS,
        candidate_config=candidate,
        baseline_evidence_digest=EXPECTED_BASELINE_FINGERPRINT,
    )
    after = inspect_study(study_root)
    if after.plan.digest != snapshot.plan.digest:
        raise RuntimeError("Study digest changed during preregistration")
    _validate_baseline(root, allow_definition=True)
    verified_definition = _validate_definition(root)
    if verified_definition.digest != definition.digest:
        raise RuntimeError("Experiment definition reconstruction digest mismatch")

    index = {
        "schema_version": "canonical_m2_portable_exp001_prereg_index_v1",
        "issue_number": issue_number,
        "current_main_sha": _required_env("CURRENT_MAIN_SHA"),
        "source_baseline_run_id": int(_required_env("BASELINE_RUN_ID")),
        "source_baseline_artifact_id": int(_required_env("BASELINE_ARTIFACT_ID")),
        "source_baseline_artifact_digest": _required_env("BASELINE_ARTIFACT_DIGEST"),
        "study_digest": EXPECTED_STUDY_DIGEST,
        "baseline_evidence_fingerprint": EXPECTED_BASELINE_FINGERPRINT,
        "dataset_id": EXPECTED_DATASET_ID,
        "dataset_artifact_digest": EXPECTED_DATASET_ARTIFACT_DIGEST,
        "implementation_digest": EXPECTED_IMPLEMENTATION_DIGEST,
        "runtime_environment_digest": EXPECTED_RUNTIME_DIGEST,
        "experiment_sequence": 1,
        "definition_digest": definition.digest,
        "factor": definition.factor.value,
        "requested_candidate_config_digest": requested_digest,
        "resolved_changed_paths": ["rule_entry_threshold"],
        "baseline_rule_entry_threshold": 0.01,
        "candidate_rule_entry_threshold": 0.02,
        "rule_exit_threshold": 0.0025,
        "frozen_baseline_mean_reversion_median_turnover": baseline_turnover,
        "formal_target_strategy": "mean_reversion",
        "accept_positive_factor_effect_symbols_min": 4,
        "accept_median_excess_total_return_strictly_positive": True,
        "accept_candidate_positive_total_return_symbols": 5,
        "accept_candidate_turnover_strictly_below_baseline": True,
        "keep_baseline_positive_factor_effect_symbols_max": 2,
        "keep_baseline_median_excess_total_return_non_positive": True,
        "keep_baseline_candidate_positive_total_return_symbols_max": 3,
        "known_preportable_outcome": "KEEP_BASELINE",
        "candidate_executed": False,
        "result_inspected_before_preregistration": False,
        "ppo_cross_symbol_metrics_used_for_decision": False,
    }
    (root / "portable-exp001-prereg-index.json").write_text(
        json.dumps(index, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print("PORTABLE_EXP001_PREREG=" + json.dumps(index, sort_keys=True, allow_nan=False))
    output = os.environ.get("GITHUB_OUTPUT")
    if output:
        with Path(output).open("a", encoding="utf-8") as handle:
            handle.write(f"definition_digest={definition.digest}\n")
            handle.write(f"candidate_requested_digest={requested_digest}\n")
            handle.write(f"baseline_turnover={baseline_turnover:.17g}\n")
            handle.write(f"artifact_name=canonical-m2-portable-exp001-prereg-v1-{_required_env('GITHUB_RUN_ID')}\n")


def verify_prereg(root: Path, report_root: Path) -> None:
    _, baseline_turnover = _validate_baseline(root, allow_definition=True)
    candidate, requested_digest = _candidate_from_bootstrap(root)
    definition = _validate_definition(root)
    index_path = root / "portable-exp001-prereg-index.json"
    if not index_path.is_file():
        raise RuntimeError("preregistration index missing")
    index = json.loads(index_path.read_text(encoding="utf-8"))
    expected_pairs = {
        "schema_version": "canonical_m2_portable_exp001_prereg_index_v1",
        "study_digest": EXPECTED_STUDY_DIGEST,
        "baseline_evidence_fingerprint": EXPECTED_BASELINE_FINGERPRINT,
        "dataset_id": EXPECTED_DATASET_ID,
        "dataset_artifact_digest": EXPECTED_DATASET_ARTIFACT_DIGEST,
        "implementation_digest": EXPECTED_IMPLEMENTATION_DIGEST,
        "runtime_environment_digest": EXPECTED_RUNTIME_DIGEST,
        "definition_digest": definition.digest,
        "factor": "RULE_THRESHOLDS",
        "requested_candidate_config_digest": requested_digest,
        "candidate_executed": False,
        "result_inspected_before_preregistration": False,
        "ppo_cross_symbol_metrics_used_for_decision": False,
    }
    for key, expected in expected_pairs.items():
        if index.get(key) != expected:
            raise RuntimeError(f"preregistration index drift: {key}")
    stored_turnover = index.get("frozen_baseline_mean_reversion_median_turnover")
    if not isinstance(stored_turnover, (int, float)) or isinstance(stored_turnover, bool):
        raise RuntimeError("frozen baseline turnover missing")
    if float(stored_turnover) != baseline_turnover:
        raise RuntimeError("frozen baseline turnover does not independently reproduce")
    if candidate.rule_entry_threshold != 0.02:
        raise RuntimeError("candidate config reconstruction drift")

    report_root.mkdir(parents=True, exist_ok=True)
    report = {
        "schema_version": "canonical_m2_portable_exp001_prereg_verification_v1",
        "study_digest": EXPECTED_STUDY_DIGEST,
        "baseline_evidence_fingerprint": EXPECTED_BASELINE_FINGERPRINT,
        "definition_digest": definition.digest,
        "requested_candidate_config_digest": requested_digest,
        "baseline_mean_reversion_median_turnover": baseline_turnover,
        "current_provenance_matches_frozen_study": True,
        "definition_is_result_blind": True,
        "candidate_absent": True,
        "verification_passed": True,
    }
    (report_root / "prereg-verification.json").write_text(
        json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print("PORTABLE_EXP001_PREREG_VERIFY=" + json.dumps(report, sort_keys=True, allow_nan=False))


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
        preregister(args.root, args.issue_number)
    else:
        verify_prereg(args.root, args.report_root)


if __name__ == "__main__":
    main()
