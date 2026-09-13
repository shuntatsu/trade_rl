"""Independent post-artifact verifier for portable Experiment 0004.

This verifier deliberately does not import the Experiment 0004 execution helper.
It reconstructs preregistration/result state and raw EvidenceSets independently,
replays the frozen lightgbm24 decision rule, and requires the published result
to remain one-shot and interpretation-sealed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import median

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
from research.issue529_portable_exp004 import (
    AFFECTED_STRATEGIES,
    EXPECTED_EXP002_DECISION_DIGEST,
    EXPECTED_EXP003_BINDING,
    ISSUE_NUMBER,
    STOP_RULE,
    UNAFFECTED_CONTROL_STRATEGIES,
    formal_rule_payload,
)
from trade_rl.data import (
    inspect_published_market_dataset_artifact,
    load_market_dataset_artifact,
)
from trade_rl.evaluation.experiments import (
    ControlledFactor,
    ControlledVerificationStatus,
    ExperimentDecisionKind,
    inspect_study,
    load_evidence_set,
)
from trade_rl.evaluation.experiments.inspection import _reconstruct
from trade_rl.evaluation.experiments.store import StudyStore

STRATEGIES = (
    "cash",
    "constant_long",
    "constant_short",
    "trend",
    "mean_reversion",
    "ridge24",
    "lightgbm24",
    "ppo",
)
DETERMINISTIC = tuple(name for name in STRATEGIES if name != "ppo")
UNAFFECTED = frozenset(UNAFFECTED_CONTROL_STRATEGIES)
AFFECTED = frozenset(AFFECTED_STRATEGIES)
EXPECTED_CHANGED_PATHS = (("feature_indices",), ("feature_names",))


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tree_manifest(root: Path) -> dict[str, str]:
    if not root.is_dir():
        raise RuntimeError(f"immutable directory is missing: {root}")
    return {
        path.relative_to(root).as_posix(): _file_sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != ".mutation.lock"
    }


def _assert_same_file(prereg_root: Path, result_root: Path, relative: str) -> None:
    before = prereg_root / relative
    after = result_root / relative
    if not before.is_file() or not after.is_file():
        raise RuntimeError(f"immutable file is missing: {relative}")
    if _file_sha256(before) != _file_sha256(after):
        raise RuntimeError(f"immutable preregistration file changed: {relative}")


def _assert_same_tree(
    prereg_root: Path,
    result_root: Path,
    relative: str,
    *,
    label: str,
) -> None:
    if _tree_manifest(prereg_root / relative) != _tree_manifest(result_root / relative):
        raise RuntimeError(f"{label} immutable evidence changed after preregistration")


def _assert_immutable_prefix(prereg_root: Path, result_root: Path) -> None:
    for relative in (
        "bootstrap.json",
        "portable-exp002-result-index.json",
        "portable-exp003-result-index.json",
        "portable-exp004-prereg-index.json",
        "study/plan.json",
        "study/experiments/0004/definition.json",
        "dataset/manifest.json",
        "dataset/arrays.npz",
    ):
        _assert_same_file(prereg_root, result_root, relative)
    _assert_same_tree(prereg_root, result_root, "study/baseline", label="baseline")
    for sequence in (1, 2, 3):
        _assert_same_tree(
            prereg_root,
            result_root,
            f"study/experiments/{sequence:04d}",
            label=f"Experiment {sequence:04d}",
        )


def independent_decision(
    *,
    positive_effects: int,
    median_excess: float,
    candidate_positive_returns: int,
) -> ExperimentDecisionKind:
    """Independent copy of the preregistered Experiment 0004 rule."""

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


def validate_result_index_safety(index: object) -> None:
    """Require one-shot, sealed, non-promotional result-index semantics."""

    if not isinstance(index, dict):
        raise RuntimeError("Experiment 0004 result index malformed")
    if index.get("schema_version") != "canonical_m2_portable_exp004_result_index_v1":
        raise RuntimeError("Experiment 0004 result index schema drift")
    if index.get("issue_number") != ISSUE_NUMBER:
        raise RuntimeError("Experiment 0004 issue binding drift")
    if index.get("verification_status") != "CONTROLLED":
        raise RuntimeError("Experiment 0004 verification status drift")
    if index.get("changed_paths") != [["feature_indices"], ["feature_names"]]:
        raise RuntimeError("Experiment 0004 changed paths drift")
    if index.get("formal_target_strategy") != "lightgbm24":
        raise RuntimeError("Experiment 0004 formal target drift")
    if index.get("candidate_execution_count") != 1:
        raise RuntimeError("Experiment 0004 execution count drift")
    if index.get("candidate_rerun") is not False:
        raise RuntimeError("Experiment 0004 candidate rerun drift")
    if index.get("study_frozen") is not False:
        raise RuntimeError("Experiment 0004 Study frozen drift")
    if index.get("interpretation_authorized") is not False:
        raise RuntimeError("Experiment 0004 result interpretation is not sealed")
    for key in (
        "profitability_claimed",
        "winner_claimed",
        "final_test_authorized",
        "production_authorized",
    ):
        if index.get(key) is not False:
            raise RuntimeError(f"unsupported claim in Experiment 0004 result: {key}")


def _compound(values: np.ndarray) -> float:
    wealth = 1.0
    for value in np.asarray(values, dtype=np.float64).reshape(-1):
        resolved = float(value)
        if not math.isfinite(resolved) or resolved <= -1.0:
            raise RuntimeError("raw return is non-finite or <= -1")
        wealth *= 1.0 + resolved
    result = wealth - 1.0
    if not math.isfinite(result):
        raise RuntimeError("compounded return is non-finite")
    return result


def _metric(entry: object, field: str) -> float:
    if not isinstance(entry, dict):
        raise RuntimeError("strategy entry malformed")
    metrics = entry.get("metrics")
    if not isinstance(metrics, dict):
        raise RuntimeError("strategy metrics malformed")
    value = metrics.get(field)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError(f"strategy metric malformed: {field}")
    resolved = float(value)
    if not math.isfinite(resolved):
        raise RuntimeError(f"strategy metric non-finite: {field}")
    return resolved


def _metric_int(entry: object, field: str) -> int:
    if not isinstance(entry, dict):
        raise RuntimeError("strategy entry malformed")
    metrics = entry.get("metrics")
    if not isinstance(metrics, dict):
        raise RuntimeError("strategy metrics malformed")
    value = metrics.get(field)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise RuntimeError(f"strategy integer metric malformed: {field}")
    return value


def _entry(run: object, symbol_index: int, strategy: str) -> dict[str, object]:
    summary = getattr(run, "summary", None)
    if not isinstance(summary, dict):
        raise RuntimeError("run summary malformed")
    by_symbol = summary.get("by_symbol")
    if not isinstance(by_symbol, list) or len(by_symbol) != len(EXPECTED_SYMBOLS):
        raise RuntimeError("run symbol roster malformed")
    symbol_entry = by_symbol[symbol_index]
    if (
        not isinstance(symbol_entry, dict)
        or symbol_entry.get("symbol") != EXPECTED_SYMBOLS[symbol_index]
    ):
        raise RuntimeError("run symbol ordering drift")
    strategies = symbol_entry.get("strategies")
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


def _entry_values(
    run: object,
    symbol_index: int,
    strategy: str,
) -> tuple[dict[str, object], np.ndarray]:
    entry = _entry(run, symbol_index, strategy)
    key = entry.get("return_key")
    returns = getattr(run, "returns", None)
    if not isinstance(key, str) or not isinstance(returns, dict) or key not in returns:
        raise RuntimeError("strategy raw-return evidence missing")
    values = np.asarray(returns[key], dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise RuntimeError("strategy raw returns malformed")
    reconstructed = _compound(values)
    if not math.isclose(
        reconstructed,
        _metric(entry, "total_return"),
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise RuntimeError("persisted total_return differs from raw-return compounding")
    return entry, values


def _validate_seed_invariance(loaded: object) -> int:
    runs = getattr(loaded, "runs", None)
    if not isinstance(runs, dict) or tuple(sorted(runs)) != EXPECTED_SEEDS:
        raise RuntimeError("EvidenceSet seed roster drift")
    reference: dict[tuple[str, str], np.ndarray] = {}
    checks = 0
    for seed in EXPECTED_SEEDS:
        for symbol_index, symbol in enumerate(EXPECTED_SYMBOLS):
            for strategy in DETERMINISTIC:
                _, values = _entry_values(runs[seed], symbol_index, strategy)
                key = (symbol, strategy)
                if seed == EXPECTED_SEEDS[0]:
                    reference[key] = values.copy()
                else:
                    if not np.array_equal(reference[key], values):
                        raise RuntimeError(
                            f"deterministic raw returns are seed-sensitive: {symbol}/{strategy}"
                        )
                    checks += 1
    return checks


def _effect_summary(excesses: list[float]) -> dict[str, object]:
    if len(excesses) != len(EXPECTED_SYMBOLS):
        raise RuntimeError("factor-effect symbol roster incomplete")
    return {
        "positive_symbol_count": sum(value > 0.0 for value in excesses),
        "negative_symbol_count": sum(value < 0.0 for value in excesses),
        "zero_symbol_count": sum(value == 0.0 for value in excesses),
        "median_excess_total_return": float(median(excesses)),
        "worst_excess_total_return": min(excesses),
        "best_excess_total_return": max(excesses),
    }


def _independent_raw_analysis(result_root: Path) -> dict[str, object]:
    baseline = load_evidence_set(result_root / "study/baseline/evidence")
    candidate = load_evidence_set(
        result_root / "study/experiments/0004/candidate/evidence"
    )
    if baseline.evidence.fingerprint != EXPECTED_BASELINE_FINGERPRINT:
        raise RuntimeError("baseline EvidenceSet fingerprint drift")
    if baseline.evidence.ppo_seeds != EXPECTED_SEEDS:
        raise RuntimeError("baseline PPO seed roster drift")
    if candidate.evidence.ppo_seeds != EXPECTED_SEEDS:
        raise RuntimeError("candidate PPO seed roster drift")
    if tuple(sorted(baseline.runs)) != EXPECTED_SEEDS:
        raise RuntimeError("baseline Run roster drift")
    if tuple(sorted(candidate.runs)) != EXPECTED_SEEDS:
        raise RuntimeError("candidate Run roster drift")

    baseline_seed_checks = _validate_seed_invariance(baseline)
    candidate_seed_checks = _validate_seed_invariance(candidate)

    unaffected_checks = 0
    for seed in EXPECTED_SEEDS:
        for symbol_index, symbol in enumerate(EXPECTED_SYMBOLS):
            for strategy in UNAFFECTED:
                _, before = _entry_values(baseline.runs[seed], symbol_index, strategy)
                _, after = _entry_values(candidate.runs[seed], symbol_index, strategy)
                if not np.array_equal(before, after):
                    raise RuntimeError(
                        "unaffected raw returns changed: "
                        f"seed={seed} symbol={symbol} strategy={strategy}"
                    )
                unaffected_checks += 1

    deterministic_effects: dict[str, object] = {}
    for strategy in DETERMINISTIC:
        by_symbol: dict[str, object] = {}
        excesses: list[float] = []
        for symbol_index, symbol in enumerate(EXPECTED_SYMBOLS):
            before_entry, before = _entry_values(
                baseline.runs[EXPECTED_SEEDS[0]], symbol_index, strategy
            )
            after_entry, after = _entry_values(
                candidate.runs[EXPECTED_SEEDS[0]], symbol_index, strategy
            )
            baseline_total = _compound(before)
            candidate_total = _compound(after)
            excess = candidate_total - baseline_total
            excesses.append(excess)
            by_symbol[symbol] = {
                "baseline_total_return": baseline_total,
                "candidate_total_return": candidate_total,
                "excess_total_return": excess,
                "baseline_turnover": _metric(before_entry, "turnover_total"),
                "candidate_turnover": _metric(after_entry, "turnover_total"),
                "baseline_total_cost": _metric(before_entry, "total_cost"),
                "candidate_total_cost": _metric(after_entry, "total_cost"),
            }
        deterministic_effects[strategy] = {
            "by_symbol": by_symbol,
            "aggregate": _effect_summary(excesses),
        }

    ppo_effects_by_seed: dict[str, object] = {}
    for seed in EXPECTED_SEEDS:
        by_symbol: dict[str, object] = {}
        for symbol_index, symbol in enumerate(EXPECTED_SYMBOLS):
            before_entry, before = _entry_values(
                baseline.runs[seed], symbol_index, "ppo"
            )
            after_entry, after = _entry_values(
                candidate.runs[seed], symbol_index, "ppo"
            )
            baseline_total = _compound(before)
            candidate_total = _compound(after)
            by_symbol[symbol] = {
                "baseline_total_return": baseline_total,
                "candidate_total_return": candidate_total,
                "excess_total_return": candidate_total - baseline_total,
                "baseline_turnover": _metric(before_entry, "turnover_total"),
                "candidate_turnover": _metric(after_entry, "turnover_total"),
                "baseline_total_cost": _metric(before_entry, "total_cost"),
                "candidate_total_cost": _metric(after_entry, "total_cost"),
            }
        ppo_effects_by_seed[str(seed)] = by_symbol

    trading = 0
    positive_cost = 0
    cash = 0
    aggregate_cost = 0.0
    for seed in EXPECTED_SEEDS:
        run = candidate.runs[seed]
        for symbol_index, symbol in enumerate(EXPECTED_SYMBOLS):
            for strategy in STRATEGIES:
                entry, values = _entry_values(run, symbol_index, strategy)
                total_cost = _metric(entry, "total_cost")
                n_trades = _metric_int(entry, "n_trades")
                aggregate_cost += total_cost
                if n_trades > 0:
                    trading += 1
                    if total_cost <= 0.0:
                        raise RuntimeError(
                            "trading observation has non-positive cost: "
                            f"seed={seed} symbol={symbol} strategy={strategy}"
                        )
                    positive_cost += 1
                if strategy == "cash":
                    cash += 1
                    if n_trades != 0 or total_cost != 0.0:
                        raise RuntimeError("cash trade/cost semantics drift")
                    if not np.array_equal(values, np.zeros_like(values)):
                        raise RuntimeError("cash raw-return semantics drift")
    if trading == 0 or positive_cost != trading:
        raise RuntimeError("candidate positive-cost trading oracle failed")
    if cash != len(EXPECTED_SEEDS) * len(EXPECTED_SYMBOLS):
        raise RuntimeError("cash observation roster incomplete")

    target = deterministic_effects.get("lightgbm24")
    if not isinstance(target, dict):
        raise RuntimeError("lightgbm24 factor-effect evidence missing")
    by_symbol = target.get("by_symbol")
    aggregate = target.get("aggregate")
    if not isinstance(by_symbol, dict) or not isinstance(aggregate, dict):
        raise RuntimeError("lightgbm24 factor-effect evidence malformed")
    candidate_totals = [
        float(by_symbol[symbol]["candidate_total_return"])
        for symbol in EXPECTED_SYMBOLS
    ]
    positive_effects = int(aggregate["positive_symbol_count"])
    median_excess = float(aggregate["median_excess_total_return"])
    positive_returns = sum(value > 0.0 for value in candidate_totals)
    decision = independent_decision(
        positive_effects=positive_effects,
        median_excess=median_excess,
        candidate_positive_returns=positive_returns,
    )
    return {
        "baseline_evidence_fingerprint": baseline.evidence.fingerprint,
        "candidate_evidence_fingerprint": candidate.evidence.fingerprint,
        "unaffected_raw_return_equality_checks": unaffected_checks,
        "deterministic_seed_invariance_checks": {
            "baseline": baseline_seed_checks,
            "candidate": candidate_seed_checks,
        },
        "deterministic_effects": deterministic_effects,
        "ppo_effects_by_seed": ppo_effects_by_seed,
        "lightgbm24_formal_inputs": {
            "positive_factor_effect_symbol_count": positive_effects,
            "median_excess_total_return": median_excess,
            "candidate_positive_total_return_symbol_count": positive_returns,
        },
        "candidate_cost_semantics": {
            "aggregate_total_cost_across_seed_runs": aggregate_cost,
            "trading_observations": trading,
            "positive_cost_observations": positive_cost,
            "cash_observations": cash,
        },
        "formal_decision": decision.value,
    }


def _prior_binding(state: object, count: int = 3) -> tuple[tuple[object, ...], ...]:
    experiments = getattr(state, "experiments", None)
    if not isinstance(experiments, list | tuple) or len(experiments) < count:
        raise RuntimeError("Experiment state malformed")
    result: list[tuple[object, ...]] = []
    for experiment in experiments[:count]:
        candidate = getattr(experiment, "candidate", None)
        evidence = getattr(candidate, "evidence", None)
        result.append(
            (
                getattr(getattr(experiment, "definition", None), "digest", None),
                getattr(evidence, "fingerprint", None),
                getattr(getattr(experiment, "verification", None), "digest", None),
                getattr(getattr(experiment, "comparison", None), "digest", None),
                getattr(getattr(experiment, "decision", None), "digest", None),
            )
        )
    return tuple(result)


def _validate_prereg_index(prereg_root: Path) -> dict[str, object]:
    path = prereg_root / "portable-exp004-prereg-index.json"
    if not path.is_file():
        raise RuntimeError("Experiment 0004 preregistration index missing")
    index = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(index, dict):
        raise RuntimeError("Experiment 0004 preregistration index malformed")
    expected = {
        "schema_version": "canonical_m2_portable_exp004_prereg_index_v1",
        "issue_number": ISSUE_NUMBER,
        "experiment_sequence": 4,
        "factor": "FEATURE_SET",
        "resolved_changed_paths": ["feature_indices", "feature_names"],
        "formal_target_strategy": "lightgbm24",
        "formal_decision_rule": formal_rule_payload(),
        "affected_strategies": list(AFFECTED_STRATEGIES),
        "unaffected_control_strategies": list(UNAFFECTED_CONTROL_STRATEGIES),
        "stop_rule": STOP_RULE,
        "candidate_executed": False,
        "candidate_pnl_computed_before_preregistration": False,
        "result_inspected_before_preregistration": False,
        "execution_authorized": False,
        "profitability_claimed": False,
        "winner_claimed": False,
        "final_test_authorized": False,
        "production_authorized": False,
    }
    for key, expected_value in expected.items():
        if index.get(key) != expected_value:
            raise RuntimeError(f"Experiment 0004 preregistration index drift: {key}")
    return index


def _validate_data_and_study(
    prereg_root: Path,
    result_root: Path,
) -> tuple[object, object]:
    for root, label in ((prereg_root, "preregistration"), (result_root, "result")):
        dataset = load_market_dataset_artifact(root / "dataset")
        artifact = inspect_published_market_dataset_artifact(root / "dataset")
        snapshot = inspect_study(root / "study")
        if dataset.dataset_id != EXPECTED_DATASET_ID:
            raise RuntimeError(f"{label} Dataset ID drift")
        if artifact.artifact_digest != EXPECTED_DATASET_ARTIFACT_DIGEST:
            raise RuntimeError(f"{label} Dataset artifact digest drift")
        if snapshot.plan.digest != EXPECTED_STUDY_DIGEST:
            raise RuntimeError(f"{label} Study digest drift")
        if snapshot.plan.implementation_digest != EXPECTED_IMPLEMENTATION_DIGEST:
            raise RuntimeError(f"{label} implementation digest drift")
        if snapshot.plan.runtime_environment_digest != EXPECTED_RUNTIME_DIGEST:
            raise RuntimeError(f"{label} runtime digest drift")
        if snapshot.baseline is None or snapshot.baseline.fingerprint != EXPECTED_BASELINE_FINGERPRINT:
            raise RuntimeError(f"{label} baseline fingerprint drift")
        if snapshot.frozen:
            raise RuntimeError(f"{label} Study unexpectedly frozen")
    prereg_state = _reconstruct(StudyStore(prereg_root / "study"))
    result_state = _reconstruct(StudyStore(result_root / "study"))
    if _prior_binding(prereg_state) != _prior_binding(result_state):
        raise RuntimeError("prior Experiment bindings changed after preregistration")
    return prereg_state, result_state


def _validate_exact_prior_bindings(result_state: object) -> None:
    experiments = getattr(result_state, "experiments", None)
    if not isinstance(experiments, list | tuple) or len(experiments) != 4:
        raise RuntimeError("result does not contain exactly four Experiments")
    exp1, exp2, exp3, _ = experiments
    expected_exp1 = (
        EXPECTED_EXP001_DEFINITION_DIGEST,
        EXPECTED_EXP001_CANDIDATE_FINGERPRINT,
        EXPECTED_EXP001_VERIFICATION_DIGEST,
        EXPECTED_EXP001_COMPARISON_DIGEST,
        EXPECTED_EXP001_DECISION_DIGEST,
    )
    if _prior_binding(result_state, count=1)[0] != expected_exp1:
        raise RuntimeError("Experiment 0001 binding drift")
    if exp2.definition.digest != EXPECTED_EXP002_DEFINITION_DIGEST:
        raise RuntimeError("Experiment 0002 definition drift")
    if exp2.candidate is None or exp2.candidate.evidence.fingerprint != EXPECTED_EXP002_CANDIDATE_FINGERPRINT:
        raise RuntimeError("Experiment 0002 candidate binding drift")
    if exp2.decision is None or exp2.decision.digest != EXPECTED_EXP002_DECISION_DIGEST:
        raise RuntimeError("Experiment 0002 decision binding drift")
    expected_exp3 = (
        EXPECTED_EXP003_BINDING["definition_digest"],
        EXPECTED_EXP003_BINDING["candidate_evidence_fingerprint"],
        EXPECTED_EXP003_BINDING["verification_digest"],
        EXPECTED_EXP003_BINDING["comparison_digest"],
        EXPECTED_EXP003_BINDING["decision_digest"],
    )
    if _prior_binding(result_state)[2] != expected_exp3:
        raise RuntimeError("Experiment 0003 binding drift")
    if exp3.decision is None or exp3.decision.decision is not ExperimentDecisionKind.KEEP_BASELINE:
        raise RuntimeError("Experiment 0003 decision drift")


def _validate_exp4_state(
    prereg_state: object,
    result_state: object,
) -> tuple[object, object, object, object]:
    prereg_experiments = getattr(prereg_state, "experiments", None)
    result_experiments = getattr(result_state, "experiments", None)
    if not isinstance(prereg_experiments, list | tuple) or len(prereg_experiments) != 4:
        raise RuntimeError("preregistration Experiment state malformed")
    if not isinstance(result_experiments, list | tuple) or len(result_experiments) != 4:
        raise RuntimeError("result Experiment state malformed")
    before = prereg_experiments[3]
    after = result_experiments[3]
    if before.definition.digest != after.definition.digest:
        raise RuntimeError("Experiment 0004 definition drift")
    if before.definition.factor is not ControlledFactor.FEATURE_SET:
        raise RuntimeError("Experiment 0004 preregistered factor drift")
    if any(
        getattr(before, field, None) is not None
        for field in ("candidate", "verification", "comparison", "decision", "failure")
    ):
        raise RuntimeError("preregistration contains Experiment 0004 result evidence")
    if after.failure is not None:
        raise RuntimeError("Experiment 0004 result contains failure evidence")
    if any(
        item is None
        for item in (after.candidate, after.verification, after.comparison, after.decision)
    ):
        raise RuntimeError("Experiment 0004 terminal evidence incomplete")
    if after.verification.status is not ControlledVerificationStatus.CONTROLLED:
        raise RuntimeError("Experiment 0004 verification is not CONTROLLED")
    if after.verification.changed_paths != EXPECTED_CHANGED_PATHS:
        raise RuntimeError("Experiment 0004 changed paths drift")
    return after, after.verification, after.comparison, after.decision


def _crosscheck_comparison(comparison: object, raw: dict[str, object]) -> None:
    factor_effect = getattr(comparison, "factor_effect", None)
    if not hasattr(factor_effect, "get"):
        raise RuntimeError("persisted comparison factor_effect malformed")
    cross_symbol = factor_effect.get("cross_symbol")
    if not hasattr(cross_symbol, "get"):
        raise RuntimeError("persisted comparison cross_symbol malformed")
    lightgbm = cross_symbol.get("lightgbm24")
    if not hasattr(lightgbm, "get"):
        raise RuntimeError("persisted lightgbm24 comparison malformed")
    formal = raw.get("lightgbm24_formal_inputs")
    if not isinstance(formal, dict):
        raise RuntimeError("fresh lightgbm24 formal inputs malformed")
    if lightgbm.get("positive_symbol_count") != formal.get(
        "positive_factor_effect_symbol_count"
    ):
        raise RuntimeError("persisted lightgbm24 positive count differs from raw oracle")
    persisted_median = lightgbm.get("median_excess_total_return")
    fresh_median = formal.get("median_excess_total_return")
    if isinstance(persisted_median, bool) or not isinstance(persisted_median, (int, float)):
        raise RuntimeError("persisted lightgbm24 median malformed")
    if isinstance(fresh_median, bool) or not isinstance(fresh_median, (int, float)):
        raise RuntimeError("fresh lightgbm24 median malformed")
    if not math.isclose(
        float(persisted_median),
        float(fresh_median),
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise RuntimeError("persisted lightgbm24 median differs from raw oracle")


def _validate_result_index(
    result_root: Path,
    *,
    prereg_index: dict[str, object],
    raw: dict[str, object],
    exp4: object,
    verification: object,
    comparison: object,
    decision: object,
) -> dict[str, object]:
    path = result_root / "portable-exp004-result-index.json"
    if not path.is_file():
        raise RuntimeError("Experiment 0004 result index missing")
    index = json.loads(path.read_text(encoding="utf-8"))
    validate_result_index_safety(index)
    assert isinstance(index, dict)
    expected = {
        "study_digest": EXPECTED_STUDY_DIGEST,
        "baseline_evidence_fingerprint": EXPECTED_BASELINE_FINGERPRINT,
        "candidate_evidence_fingerprint": raw["candidate_evidence_fingerprint"],
        "dataset_id": EXPECTED_DATASET_ID,
        "dataset_artifact_digest": EXPECTED_DATASET_ARTIFACT_DIGEST,
        "implementation_digest": EXPECTED_IMPLEMENTATION_DIGEST,
        "runtime_environment_digest": EXPECTED_RUNTIME_DIGEST,
        "definition_digest": exp4.definition.digest,
        "requested_candidate_config_digest": exp4.definition.candidate_requested_config_digest,
        "verification_digest": verification.digest,
        "comparison_digest": comparison.digest,
        "decision_digest": decision.digest,
        "decision": decision.decision.value,
        "affected_strategies": list(AFFECTED_STRATEGIES),
        "unaffected_control_strategies": list(UNAFFECTED_CONTROL_STRATEGIES),
        "stop_rule": prereg_index["stop_rule"],
    }
    for key, expected_value in expected.items():
        if index.get(key) != expected_value:
            raise RuntimeError(f"Experiment 0004 result-index binding drift: {key}")
    if index.get("independent_raw_evidence") != raw:
        raise RuntimeError("Experiment 0004 result-index raw evidence drift")
    return index


def verify(
    prereg_root: Path,
    result_root: Path,
    report_root: Path,
) -> dict[str, object]:
    """Reverify the published result without importing the execution helper."""

    _assert_immutable_prefix(prereg_root, result_root)
    prereg_index = _validate_prereg_index(prereg_root)
    prereg_state, result_state = _validate_data_and_study(prereg_root, result_root)
    _validate_exact_prior_bindings(result_state)
    exp4, verification, comparison, decision = _validate_exp4_state(
        prereg_state,
        result_state,
    )
    raw = _independent_raw_analysis(result_root)
    _crosscheck_comparison(comparison, raw)
    independent = independent_decision(
        positive_effects=int(raw["lightgbm24_formal_inputs"]["positive_factor_effect_symbol_count"]),  # type: ignore[index]
        median_excess=float(raw["lightgbm24_formal_inputs"]["median_excess_total_return"]),  # type: ignore[index]
        candidate_positive_returns=int(raw["lightgbm24_formal_inputs"]["candidate_positive_total_return_symbol_count"]),  # type: ignore[index]
    )
    if decision.decision is not independent:
        raise RuntimeError("persisted Experiment 0004 decision differs from independent replay")
    result_index = _validate_result_index(
        result_root,
        prereg_index=prereg_index,
        raw=raw,
        exp4=exp4,
        verification=verification,
        comparison=comparison,
        decision=decision,
    )

    unaffected_expected = (
        len(EXPECTED_SEEDS) * len(EXPECTED_SYMBOLS) * len(UNAFFECTED)
    )
    if raw["unaffected_raw_return_equality_checks"] != unaffected_expected:
        raise RuntimeError("unaffected raw-return equality check-count drift")
    seed_expected = (
        (len(EXPECTED_SEEDS) - 1)
        * len(EXPECTED_SYMBOLS)
        * len(DETERMINISTIC)
    )
    seed_checks = raw["deterministic_seed_invariance_checks"]
    if not isinstance(seed_checks, dict):
        raise RuntimeError("deterministic seed-invariance evidence malformed")
    if seed_checks.get("baseline") != seed_expected or seed_checks.get("candidate") != seed_expected:
        raise RuntimeError("deterministic seed-invariance check-count drift")

    report = {
        "schema_version": "canonical_m2_portable_exp004_postverify_v1",
        "verification_passed": True,
        "immutable_preregistration_prefix_verified": True,
        "prior_exp001_exp002_exp003_evidence_unchanged": True,
        "controlled_delta_verified": True,
        "unaffected_raw_return_equality_verified": True,
        "unaffected_raw_return_equality_checks": unaffected_expected,
        "deterministic_seed_invariance_verified": True,
        "deterministic_seed_invariance_checks": seed_checks,
        "all_deterministic_total_returns_recomputed_from_raw": True,
        "ppo_effects_fully_disclosed_by_seed": True,
        "candidate_cost_semantics_verified": True,
        "independent_decision_replayed": True,
        "formal_target_strategy": "lightgbm24",
        "formal_decision": independent.value,
        "lightgbm24_formal_inputs": raw["lightgbm24_formal_inputs"],
        "deterministic_effects": raw["deterministic_effects"],
        "ppo_effects_by_seed": raw["ppo_effects_by_seed"],
        "candidate_cost_semantics": raw["candidate_cost_semantics"],
        "candidate_execution_count": result_index["candidate_execution_count"],
        "candidate_rerun": result_index["candidate_rerun"],
        "candidate_reexecuted": False,
        "result_interpretation_sealed": True,
        "interpretation_authorized": False,
        "profitability_claimed": False,
        "winner_claimed": False,
        "final_test_authorized": False,
        "production_authorized": False,
    }
    report_root.mkdir(parents=True, exist_ok=True)
    (report_root / "postverify.json").write_text(
        json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prereg-root", required=True, type=Path)
    parser.add_argument("--result-root", required=True, type=Path)
    parser.add_argument("--report-root", required=True, type=Path)
    args = parser.parse_args()
    verify(args.prereg_root, args.result_root, args.report_root)


if __name__ == "__main__":
    main()


__all__ = [
    "independent_decision",
    "validate_result_index_safety",
    "verify",
]
