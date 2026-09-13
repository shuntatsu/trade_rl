"""Pre-result structural diagnostics for portable Experiment 0003.

This module deliberately contains no candidate execution or candidate P&L
computation. It validates the frozen Study/Dataset authority, the preselected
feature identity, the baseline-only turnover reference, and signal-state
transition diagnostics by invoking the frozen production trend strategy.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from statistics import median

import numpy as np

from trade_rl.data import (
    inspect_published_market_dataset_artifact,
    load_market_dataset_artifact,
)
from trade_rl.evaluation.experiments import (
    ControlledFactor,
    ExperimentDecisionKind,
    inspect_study,
    load_evidence_set,
)
from trade_rl.evaluation.experiments.inspection import _reconstruct
from trade_rl.evaluation.experiments.store import StudyStore
from trade_rl.strategies.interface import StrategyObservation
from trade_rl.strategies.position_intent import PositionIntent
from trade_rl.strategies.rules.trend import TrendIntentConfig, TrendIntentStrategy

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
EXPECTED_SYMBOLS = ("BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT")
EXPECTED_SEEDS = (0, 1, 2, 3, 4)
EXPECTED_EXP002_CANDIDATE_FINGERPRINT = (
    "f5ec53197a2f487ba11cc01a78edd4c93df8a2f172095e5903ee079ce5e9d0d0"
)
SOURCE_RESULT_ARTIFACT_ID = 10315084502
SOURCE_RESULT_ARTIFACT_DIGEST = (
    "sha256:9a85ec7fa40cabfb662556ef89780b140609e9b2e471006981c829800a043d1d"
)
SOURCE_RECOVERY_RUN_ID = 34750227256
SOURCE_RECOVERY_HEAD = "43c54f822dfd156f7fc5e4cf40f9fce9d8cc8b91"

BASELINE_SIGNAL_NAME = "1h__log_return_24bar"
BASELINE_SIGNAL_INDEX = 2
CANDIDATE_SIGNAL_NAME = "1d__log_return_4bar"
CANDIDATE_SIGNAL_INDEX = 115
ENTRY_THRESHOLD = 0.01
EXIT_THRESHOLD = 0.0025


def require_feature_index(
    feature_names: Sequence[str],
    *,
    name: str,
    expected_index: int,
) -> int:
    """Require one exact feature identity at the preselected immutable index."""

    names = tuple(feature_names)
    if not name or any(not isinstance(item, str) or not item for item in names):
        raise ValueError("feature names must be non-empty strings")
    if (
        isinstance(expected_index, bool)
        or not isinstance(expected_index, int)
        or expected_index < 0
    ):
        raise ValueError("expected_index must be a non-negative integer")
    matches = tuple(index for index, item in enumerate(names) if item == name)
    if len(matches) != 1:
        raise RuntimeError(f"feature identity missing or duplicated: {name}")
    observed = matches[0]
    if observed != expected_index:
        raise RuntimeError(
            f"feature index drift for {name}: expected={expected_index} observed={observed}"
        )
    return observed


def trend_transition_count(
    signal: np.ndarray,
    available: np.ndarray,
    *,
    entry_threshold: float,
    exit_threshold: float,
) -> int:
    """Count intent changes by invoking the frozen production trend strategy."""

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

    strategy = TrendIntentStrategy(
        TrendIntentConfig(
            signal_index=0,
            entry_threshold=entry_threshold,
            exit_threshold=exit_threshold,
        )
    )
    current = PositionIntent.FLAT
    changes = 0
    for index, (raw, is_available) in enumerate(
        zip(values, availability, strict=True)
    ):
        observation = StrategyObservation(
            index=index,
            timestamp=np.datetime64(index, "ns"),
            symbol="STRUCTURAL_DIAGNOSTIC",
            features=np.asarray([raw], dtype=np.float64),
            feature_available=np.asarray([is_available], dtype=np.bool_),
            global_features=np.asarray([0.0], dtype=np.float64),
            global_feature_available=np.asarray([True], dtype=np.bool_),
            current_intent=current,
            current_weight=0.0,
        )
        next_intent = strategy.decide(observation)
        if next_intent is not current:
            changes += 1
        current = next_intent
    return changes


def _metric_number(metrics: object, field: str) -> float:
    if not isinstance(metrics, dict):
        raise RuntimeError("strategy metrics malformed")
    value = metrics.get(field)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError(f"strategy metric malformed: {field}")
    result = float(value)
    if not math.isfinite(result):
        raise RuntimeError(f"strategy metric non-finite: {field}")
    return result


def _strategy_entry(run: object, symbol_index: int, strategy: str) -> dict[str, object]:
    summary = getattr(run, "summary", None)
    if not isinstance(summary, dict):
        raise RuntimeError("run summary malformed")
    by_symbol = summary.get("by_symbol")
    if not isinstance(by_symbol, list) or len(by_symbol) != len(EXPECTED_SYMBOLS):
        raise RuntimeError("run symbol roster malformed")
    symbol = by_symbol[symbol_index]
    if not isinstance(symbol, dict) or symbol.get("symbol") != EXPECTED_SYMBOLS[symbol_index]:
        raise RuntimeError("run symbol ordering drift")
    strategies = symbol.get("strategies")
    if not isinstance(strategies, list):
        raise RuntimeError("run strategy roster malformed")
    matches = [
        item
        for item in strategies
        if isinstance(item, dict) and item.get("name") == strategy
    ]
    if len(matches) != 1:
        raise RuntimeError(f"strategy entry missing or duplicated: {strategy}")
    return matches[0]


def _baseline_trend_turnover(study_root: Path) -> tuple[float, dict[str, float]]:
    loaded = load_evidence_set(study_root / "baseline/evidence")
    if loaded.evidence.fingerprint != EXPECTED_BASELINE_FINGERPRINT:
        raise RuntimeError("baseline EvidenceSet fingerprint drift")
    if loaded.evidence.ppo_seeds != EXPECTED_SEEDS:
        raise RuntimeError("baseline seed roster drift")
    if tuple(sorted(loaded.runs)) != EXPECTED_SEEDS:
        raise RuntimeError("baseline Run roster drift")

    reference: dict[str, float] = {}
    for seed in EXPECTED_SEEDS:
        run = loaded.runs[seed]
        for symbol_index, symbol in enumerate(EXPECTED_SYMBOLS):
            turnover = _metric_number(
                _strategy_entry(run, symbol_index, "trend").get("metrics"),
                "turnover_total",
            )
            if seed == EXPECTED_SEEDS[0]:
                reference[symbol] = turnover
            elif turnover != reference[symbol]:
                raise RuntimeError(f"baseline trend turnover is seed-sensitive: {symbol}")
    values = [reference[symbol] for symbol in EXPECTED_SYMBOLS]
    return float(median(values)), reference


def _resolved_changed_fields(baseline: object, candidate: object) -> tuple[str, ...]:
    before_method = getattr(baseline, "to_payload", None)
    after_method = getattr(candidate, "to_payload", None)
    if not callable(before_method) or not callable(after_method):
        raise RuntimeError("resolved config does not expose to_payload")
    before = before_method()
    after = after_method()
    if not isinstance(before, dict) or not isinstance(after, dict):
        raise RuntimeError("resolved config payload malformed")
    keys = sorted(set(before) | set(after))
    return tuple(key for key in keys if before.get(key) != after.get(key))


def structural_diagnostic(root: Path) -> dict[str, object]:
    """Audit the chosen Exp3 hypothesis without reading candidate performance."""

    dataset_root = root / "dataset"
    study_root = root / "study"
    if not dataset_root.is_dir() or not study_root.is_dir():
        raise RuntimeError("portable result artifact layout is incomplete")

    artifact = inspect_published_market_dataset_artifact(dataset_root)
    dataset = load_market_dataset_artifact(dataset_root)
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
    if snapshot.experiment_sequences != (1, 2) or snapshot.terminal_sequences != (1, 2):
        raise RuntimeError("Study does not contain exactly two terminal development Experiments")
    if snapshot.frozen:
        raise RuntimeError("Study unexpectedly frozen")
    if snapshot.plan.max_experiments < 3:
        raise RuntimeError("Study has no Experiment 0003 budget")
    if ControlledFactor.RULE_SIGNAL not in snapshot.plan.allowed_factors:
        raise RuntimeError("RULE_SIGNAL is not allowed by the frozen Study")

    state = _reconstruct(StudyStore(study_root))
    if len(state.experiments) != 2:
        raise RuntimeError("Study state already contains Experiment 0003")
    exp1, exp2 = state.experiments
    for sequence, experiment in ((1, exp1), (2, exp2)):
        decision = experiment.decision
        if decision is None or decision.decision is not ExperimentDecisionKind.KEEP_BASELINE:
            raise RuntimeError(f"Experiment {sequence:04d} is not terminal KEEP_BASELINE")
        if experiment.failure is not None:
            raise RuntimeError(f"Experiment {sequence:04d} contains failure evidence")
    if exp2.candidate is None or exp2.candidate.evidence.fingerprint != EXPECTED_EXP002_CANDIDATE_FINGERPRINT:
        raise RuntimeError("Experiment 0002 candidate fingerprint drift")

    baseline = snapshot.plan.baseline_config
    if baseline.signal_name != BASELINE_SIGNAL_NAME or baseline.signal_index != BASELINE_SIGNAL_INDEX:
        raise RuntimeError("baseline RULE_SIGNAL identity drift")
    if baseline.rule_entry_threshold != ENTRY_THRESHOLD or baseline.rule_exit_threshold != EXIT_THRESHOLD:
        raise RuntimeError("baseline rule threshold drift")

    require_feature_index(
        dataset.feature_names,
        name=BASELINE_SIGNAL_NAME,
        expected_index=BASELINE_SIGNAL_INDEX,
    )
    candidate_index = require_feature_index(
        dataset.feature_names,
        name=CANDIDATE_SIGNAL_NAME,
        expected_index=CANDIDATE_SIGNAL_INDEX,
    )
    candidate_config = replace(
        baseline,
        signal_name=CANDIDATE_SIGNAL_NAME,
        signal_index=CANDIDATE_SIGNAL_INDEX,
    )
    changed = _resolved_changed_fields(baseline, candidate_config)
    if changed != ("signal_index", "signal_name"):
        raise RuntimeError(f"candidate resolved delta is not RULE_SIGNAL-only: {changed!r}")

    baseline_turnover, baseline_turnover_by_symbol = _baseline_trend_turnover(study_root)
    staleness = dataset.feature_staleness_hours
    if staleness is None:
        raise RuntimeError("Dataset lacks feature staleness-hour evidence")

    per_symbol: dict[str, object] = {}
    for symbol_index, symbol in enumerate(EXPECTED_SYMBOLS):
        baseline_signal = dataset.features[:, symbol_index, BASELINE_SIGNAL_INDEX]
        baseline_available = dataset.feature_available[:, symbol_index, BASELINE_SIGNAL_INDEX]
        candidate_signal = dataset.features[:, symbol_index, candidate_index]
        candidate_available = dataset.feature_available[:, symbol_index, candidate_index]
        candidate_staleness = staleness[:, symbol_index, candidate_index]
        available_staleness = candidate_staleness[candidate_available]
        if available_staleness.size == 0 or not np.isfinite(available_staleness).all():
            raise RuntimeError(f"candidate feature availability malformed: {symbol}")
        refresh = candidate_available & (candidate_staleness == 0.0)
        per_symbol[symbol] = {
            "baseline_transition_count": trend_transition_count(
                baseline_signal,
                baseline_available,
                entry_threshold=ENTRY_THRESHOLD,
                exit_threshold=EXIT_THRESHOLD,
            ),
            "candidate_transition_count": trend_transition_count(
                candidate_signal,
                candidate_available,
                entry_threshold=ENTRY_THRESHOLD,
                exit_threshold=EXIT_THRESHOLD,
            ),
            "candidate_available_count": int(np.count_nonzero(candidate_available)),
            "candidate_refresh_count": int(np.count_nonzero(refresh)),
            "candidate_max_staleness_hours_when_available": float(np.max(available_staleness)),
            "baseline_trend_turnover": baseline_turnover_by_symbol[symbol],
        }

    return {
        "schema_version": "canonical_m2_portable_exp003_structural_diagnostic_v1",
        "issue_number": 522,
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
        "max_experiments": snapshot.plan.max_experiments,
        "rule_signal_allowed": True,
        "baseline_signal_name": BASELINE_SIGNAL_NAME,
        "baseline_signal_index": BASELINE_SIGNAL_INDEX,
        "candidate_signal_name": CANDIDATE_SIGNAL_NAME,
        "candidate_signal_index": CANDIDATE_SIGNAL_INDEX,
        "resolved_changed_fields": list(changed),
        "rule_entry_threshold": ENTRY_THRESHOLD,
        "rule_exit_threshold": EXIT_THRESHOLD,
        "frozen_baseline_trend_median_turnover": baseline_turnover,
        "by_symbol": per_symbol,
        "experiment_0002_trend_performance_inspected": False,
        "experiment_0003_candidate_executed": False,
        "experiment_0003_candidate_pnl_computed": False,
        "profitability_claimed": False,
        "winner_claimed": False,
        "final_test_authorized": False,
        "production_authorized": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = structural_diagnostic(args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()


__all__ = [
    "require_feature_index",
    "structural_diagnostic",
    "trend_transition_count",
]
