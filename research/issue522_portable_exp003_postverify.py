"""Independent post-artifact verifier for portable Experiment 0003.

This verifier deliberately does not import the Experiment 0003 execution helper.
It reconstructs the preregistration/result state and raw EvidenceSets independently,
replays the frozen trend decision rule, and verifies that interpretation remains
sealed in the published result artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import median

import numpy as np

from research.issue522_portable_exp003 import (
    EXPECTED_BASELINE_FINGERPRINT,
    EXPECTED_DATASET_ARTIFACT_DIGEST,
    EXPECTED_DATASET_ID,
    EXPECTED_IMPLEMENTATION_DIGEST,
    EXPECTED_RUNTIME_DIGEST,
    EXPECTED_SEEDS,
    EXPECTED_STUDY_DIGEST,
    EXPECTED_SYMBOLS,
    _metric_number,
    _strategy_entry,
)
from research.issue522_portable_exp003_prereg import (
    FROZEN_BASELINE_TREND_MEDIAN_TURNOVER,
    ISSUE_NUMBER,
    validate_prereg_index_payload,
)
from trade_rl.data import (
    inspect_published_market_dataset_artifact,
    load_market_dataset_artifact,
)
from trade_rl.evaluation.experiments import (
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
UNAFFECTED = frozenset(
    {"cash", "constant_long", "constant_short", "ridge24", "lightgbm24", "ppo"}
)
EXPECTED_CHANGED_PATHS = (("signal_index",), ("signal_name",))


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
        "portable-exp003-prereg-index.json",
        "study/plan.json",
        "study/experiments/0003/definition.json",
        "dataset/manifest.json",
        "dataset/arrays.npz",
    ):
        _assert_same_file(prereg_root, result_root, relative)
    _assert_same_tree(prereg_root, result_root, "study/baseline", label="baseline")
    _assert_same_tree(
        prereg_root,
        result_root,
        "study/experiments/0001",
        label="Experiment 0001",
    )
    _assert_same_tree(
        prereg_root,
        result_root,
        "study/experiments/0002",
        label="Experiment 0002",
    )


def independent_decision(
    *,
    positive_effects: int,
    median_excess: float,
    candidate_positive_returns: int,
    candidate_median_turnover: float,
) -> ExperimentDecisionKind:
    """Independent copy of the preregistered Experiment 0003 decision rule."""

    accept = (
        positive_effects >= 4
        and median_excess > 0.0
        and candidate_positive_returns == 5
        and candidate_median_turnover < FROZEN_BASELINE_TREND_MEDIAN_TURNOVER
    )
    keep_baseline = (
        positive_effects <= 2
        or median_excess <= 0.0
        or candidate_positive_returns <= 3
    )
    if accept:
        return ExperimentDecisionKind.ACCEPT_CANDIDATE
    if keep_baseline:
        return ExperimentDecisionKind.KEEP_BASELINE
    return ExperimentDecisionKind.INCONCLUSIVE


def validate_result_index_safety_flags(index: object) -> None:
    """Require the published result to remain one-shot and interpretation-sealed."""

    if not isinstance(index, dict):
        raise RuntimeError("Experiment 0003 result index malformed")
    expected = {
        "schema_version": "canonical_m2_portable_exp003_result_index_v1",
        "issue_number": ISSUE_NUMBER,
        "verification_status": "CONTROLLED",
        "changed_paths": [["signal_index"], ["signal_name"]],
        "formal_target_strategy": "trend",
        "candidate_execution_count": 1,
        "candidate_rerun": False,
        "study_frozen": False,
        "interpretation_authorized": False,
        "profitability_claimed": False,
        "winner_claimed": False,
        "final_test_authorized": False,
        "production_authorized": False,
    }
    for key, value in expected.items():
        if index.get(key) != value:
            raise RuntimeError(f"Experiment 0003 result index drift: {key}")


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


def _require_int(metrics: object, field: str) -> int:
    if not isinstance(metrics, dict):
        raise RuntimeError("strategy metrics malformed")
    value = metrics.get(field)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise RuntimeError(f"strategy metric malformed: {field}")
    return value


def _entry_values(
    run: object,
    symbol_index: int,
    strategy: str,
) -> tuple[dict[str, object], np.ndarray]:
    entry = _strategy_entry(run, symbol_index, strategy)
    return_key = entry.get("return_key")
    returns = getattr(run, "returns", None)
    if not isinstance(return_key, str) or not isinstance(returns, dict):
        raise RuntimeError("strategy raw-return evidence malformed")
    values = returns.get(return_key)
    if values is None:
        raise RuntimeError("strategy raw returns missing")
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not np.isfinite(array).all():
        raise RuntimeError("strategy raw returns malformed")
    observed = _metric_number(entry.get("metrics"), "total_return")
    reconstructed = _compound(array)
    if not math.isclose(observed, reconstructed, rel_tol=1e-12, abs_tol=1e-12):
        raise RuntimeError("persisted total_return differs from raw-return compounding")
    return entry, array


def _validate_seed_invariance(loaded: object) -> int:
    runs = getattr(loaded, "runs", None)
    if not isinstance(runs, dict) or tuple(sorted(runs)) != EXPECTED_SEEDS:
        raise RuntimeError("EvidenceSet seed roster drift")
    references: dict[tuple[str, str], np.ndarray] = {}
    checks = 0
    for seed in EXPECTED_SEEDS:
        for symbol_index, symbol in enumerate(EXPECTED_SYMBOLS):
            for strategy in DETERMINISTIC:
                _, values = _entry_values(runs[seed], symbol_index, strategy)
                key = (symbol, strategy)
                if seed == EXPECTED_SEEDS[0]:
                    references[key] = values.copy()
                elif not np.array_equal(references[key], values):
                    raise RuntimeError(
                        f"deterministic raw returns are seed-sensitive: {symbol}/{strategy}"
                    )
                else:
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
        result_root / "study/experiments/0003/candidate/evidence"
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
                "baseline_turnover": _metric_number(
                    before_entry.get("metrics"), "turnover_total"
                ),
                "candidate_turnover": _metric_number(
                    after_entry.get("metrics"), "turnover_total"
                ),
            }
        deterministic_effects[strategy] = {
            "by_symbol": by_symbol,
            "aggregate": _effect_summary(excesses),
        }

    ppo_checks = 0
    for seed in EXPECTED_SEEDS:
        for symbol_index, symbol in enumerate(EXPECTED_SYMBOLS):
            _, before = _entry_values(baseline.runs[seed], symbol_index, "ppo")
            _, after = _entry_values(candidate.runs[seed], symbol_index, "ppo")
            if not np.array_equal(before, after):
                raise RuntimeError(
                    f"PPO changed under RULE_SIGNAL: seed={seed} symbol={symbol}"
                )
            ppo_checks += 1

    trading = 0
    positive_cost = 0
    cash = 0
    aggregate_cost = 0.0
    for seed in EXPECTED_SEEDS:
        run = candidate.runs[seed]
        for symbol_index, symbol in enumerate(EXPECTED_SYMBOLS):
            for strategy in STRATEGIES:
                entry, values = _entry_values(run, symbol_index, strategy)
                metrics = entry.get("metrics")
                total_cost = _metric_number(metrics, "total_cost")
                n_trades = _require_int(metrics, "n_trades")
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

    trend = deterministic_effects.get("trend")
    if not isinstance(trend, dict):
        raise RuntimeError("trend factor-effect evidence missing")
    by_symbol = trend.get("by_symbol")
    aggregate = trend.get("aggregate")
    if not isinstance(by_symbol, dict) or not isinstance(aggregate, dict):
        raise RuntimeError("trend factor-effect evidence malformed")
    candidate_totals = [
        float(by_symbol[symbol]["candidate_total_return"])
        for symbol in EXPECTED_SYMBOLS
    ]
    candidate_turnovers = [
        float(by_symbol[symbol]["candidate_turnover"])
        for symbol in EXPECTED_SYMBOLS
    ]
    positive_effects = int(aggregate["positive_symbol_count"])
    median_excess = float(aggregate["median_excess_total_return"])
    positive_returns = sum(value > 0.0 for value in candidate_totals)
    median_turnover = float(median(candidate_turnovers))
    decision = independent_decision(
        positive_effects=positive_effects,
        median_excess=median_excess,
        candidate_positive_returns=positive_returns,
        candidate_median_turnover=median_turnover,
    )
    return {
        "baseline_evidence_fingerprint": baseline.evidence.fingerprint,
        "candidate_evidence_fingerprint": candidate.evidence.fingerprint,
        "unaffected_raw_return_equality_checks": unaffected_checks,
        "deterministic_seed_invariance_checks": {
            "baseline": baseline_seed_checks,
            "candidate": candidate_seed_checks,
        },
        "ppo_exact_zero_effect_checks": ppo_checks,
        "deterministic_effects": deterministic_effects,
        "trend_formal_inputs": {
            "positive_factor_effect_symbol_count": positive_effects,
            "median_excess_total_return": median_excess,
            "candidate_positive_total_return_symbol_count": positive_returns,
            "candidate_median_turnover": median_turnover,
            "frozen_baseline_median_turnover": FROZEN_BASELINE_TREND_MEDIAN_TURNOVER,
        },
        "mean_reversion_side_effect": deterministic_effects["mean_reversion"],
        "candidate_cost_semantics": {
            "aggregate_total_cost_across_seed_runs": aggregate_cost,
            "trading_observations": trading,
            "positive_cost_observations": positive_cost,
            "cash_observations": cash,
        },
        "formal_decision": decision.value,
    }


def _mapping(value: object, *, field: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise RuntimeError(f"{field} malformed")
    return value


def _crosscheck_persisted_comparison(
    comparison: object,
    raw: dict[str, object],
) -> None:
    factor_effect = getattr(comparison, "factor_effect", None)
    if not hasattr(factor_effect, "get"):
        raise RuntimeError("persisted comparison factor_effect malformed")
    cross_symbol = factor_effect.get("cross_symbol")
    if not hasattr(cross_symbol, "get"):
        raise RuntimeError("persisted comparison cross_symbol malformed")
    trend = cross_symbol.get("trend")
    if not hasattr(trend, "get"):
        raise RuntimeError("persisted trend comparison malformed")
    formal = _mapping(raw.get("trend_formal_inputs"), field="fresh trend inputs")
    if trend.get("positive_symbol_count") != formal.get(
        "positive_factor_effect_symbol_count"
    ):
        raise RuntimeError("persisted trend positive count differs from raw oracle")
    persisted_median = trend.get("median_excess_total_return")
    fresh_median = formal.get("median_excess_total_return")
    if isinstance(persisted_median, bool) or not isinstance(
        persisted_median, (int, float)
    ):
        raise RuntimeError("persisted trend median malformed")
    if isinstance(fresh_median, bool) or not isinstance(fresh_median, (int, float)):
        raise RuntimeError("fresh trend median malformed")
    if not math.isclose(
        float(persisted_median),
        float(fresh_median),
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise RuntimeError("persisted trend median differs from raw oracle")


def _prior_binding(state: object) -> tuple[tuple[object, ...], tuple[object, ...]]:
    experiments = getattr(state, "experiments", None)
    if not isinstance(experiments, list | tuple) or len(experiments) < 2:
        raise RuntimeError("prior Experiment state malformed")
    bindings: list[tuple[object, ...]] = []
    for experiment in experiments[:2]:
        candidate = getattr(experiment, "candidate", None)
        evidence = getattr(candidate, "evidence", None)
        bindings.append(
            (
                getattr(getattr(experiment, "definition", None), "digest", None),
                getattr(evidence, "fingerprint", None),
                getattr(getattr(experiment, "verification", None), "digest", None),
                getattr(getattr(experiment, "comparison", None), "digest", None),
                getattr(getattr(experiment, "decision", None), "digest", None),
            )
        )
    return bindings[0], bindings[1]


def _load_prereg_index(prereg_root: Path) -> dict[str, object]:
    path = prereg_root / "portable-exp003-prereg-index.json"
    if not path.is_file():
        raise RuntimeError("Experiment 0003 preregistration index missing")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError("Experiment 0003 preregistration index malformed")
    definition_digest = payload.get("definition_digest")
    requested_digest = payload.get("requested_candidate_config_digest")
    structural_sha = payload.get("structural_report_sha256")
    if not all(isinstance(item, str) for item in (definition_digest, requested_digest, structural_sha)):
        raise RuntimeError("Experiment 0003 preregistration binding malformed")
    validate_prereg_index_payload(
        payload,
        definition_digest=definition_digest,
        requested_candidate_config_digest=requested_digest,
        structural_report_sha256=structural_sha,
    )
    return payload


def _validate_study_bindings(
    prereg_root: Path,
    result_root: Path,
) -> tuple[object, object, object, object]:
    prereg_dataset = load_market_dataset_artifact(prereg_root / "dataset")
    result_dataset = load_market_dataset_artifact(result_root / "dataset")
    prereg_artifact = inspect_published_market_dataset_artifact(prereg_root / "dataset")
    result_artifact = inspect_published_market_dataset_artifact(result_root / "dataset")
    if prereg_dataset.dataset_id != EXPECTED_DATASET_ID or result_dataset.dataset_id != EXPECTED_DATASET_ID:
        raise RuntimeError("Dataset ID drift")
    if (
        prereg_artifact.artifact_digest != EXPECTED_DATASET_ARTIFACT_DIGEST
        or result_artifact.artifact_digest != EXPECTED_DATASET_ARTIFACT_DIGEST
    ):
        raise RuntimeError("Dataset artifact digest drift")

    prereg_snapshot = inspect_study(prereg_root / "study")
    result_snapshot = inspect_study(result_root / "study")
    for snapshot, label in (
        (prereg_snapshot, "preregistration"),
        (result_snapshot, "result"),
    ):
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
    if prereg_snapshot.experiment_sequences != (1, 2, 3):
        raise RuntimeError("preregistration Experiment sequence drift")
    if prereg_snapshot.terminal_sequences != (1, 2):
        raise RuntimeError("preregistration is not result-blind for Experiment 0003")
    if result_snapshot.experiment_sequences != (1, 2, 3):
        raise RuntimeError("result Experiment sequence drift")
    if result_snapshot.terminal_sequences != (1, 2, 3):
        raise RuntimeError("Experiment 0003 result lifecycle incomplete")

    prereg_state = _reconstruct(StudyStore(prereg_root / "study"))
    result_state = _reconstruct(StudyStore(result_root / "study"))
    if _prior_binding(prereg_state) != _prior_binding(result_state):
        raise RuntimeError("prior Experiment bindings changed after preregistration")
    return prereg_state, result_state, prereg_snapshot, result_snapshot


def _validate_exp3_state(
    prereg_state: object,
    result_state: object,
) -> tuple[object, object]:
    prereg_experiments = getattr(prereg_state, "experiments", None)
    result_experiments = getattr(result_state, "experiments", None)
    if not isinstance(prereg_experiments, list | tuple) or len(prereg_experiments) != 3:
        raise RuntimeError("preregistration Experiment state malformed")
    if not isinstance(result_experiments, list | tuple) or len(result_experiments) != 3:
        raise RuntimeError("result Experiment state malformed")
    before = prereg_experiments[2]
    after = result_experiments[2]
    if getattr(before, "definition", None).digest != getattr(after, "definition", None).digest:
        raise RuntimeError("Experiment 0003 definition drift")
    if any(
        getattr(before, field, None) is not None
        for field in ("candidate", "verification", "comparison", "decision", "failure")
    ):
        raise RuntimeError("preregistration contains Experiment 0003 result evidence")
    if getattr(after, "failure", None) is not None:
        raise RuntimeError("Experiment 0003 result contains failure evidence")
    candidate = getattr(after, "candidate", None)
    verification = getattr(after, "verification", None)
    comparison = getattr(after, "comparison", None)
    decision = getattr(after, "decision", None)
    if any(item is None for item in (candidate, verification, comparison, decision)):
        raise RuntimeError("Experiment 0003 terminal evidence incomplete")
    if verification.status is not ControlledVerificationStatus.CONTROLLED:
        raise RuntimeError("Experiment 0003 verification is not CONTROLLED")
    if verification.changed_paths != EXPECTED_CHANGED_PATHS:
        raise RuntimeError("Experiment 0003 changed paths drift")
    return after, decision


def _validate_result_index(
    result_root: Path,
    *,
    prereg_index: dict[str, object],
    raw: dict[str, object],
    exp3: object,
    decision: object,
) -> dict[str, object]:
    path = result_root / "portable-exp003-result-index.json"
    if not path.is_file():
        raise RuntimeError("Experiment 0003 result index missing")
    index = json.loads(path.read_text(encoding="utf-8"))
    validate_result_index_safety_flags(index)
    assert isinstance(index, dict)

    definition = getattr(exp3, "definition", None)
    candidate = getattr(exp3, "candidate", None)
    verification = getattr(exp3, "verification", None)
    comparison = getattr(exp3, "comparison", None)
    candidate_evidence = getattr(candidate, "evidence", None)
    expected = {
        "study_digest": EXPECTED_STUDY_DIGEST,
        "baseline_evidence_fingerprint": EXPECTED_BASELINE_FINGERPRINT,
        "candidate_evidence_fingerprint": raw["candidate_evidence_fingerprint"],
        "dataset_id": EXPECTED_DATASET_ID,
        "dataset_artifact_digest": EXPECTED_DATASET_ARTIFACT_DIGEST,
        "implementation_digest": EXPECTED_IMPLEMENTATION_DIGEST,
        "runtime_environment_digest": EXPECTED_RUNTIME_DIGEST,
        "definition_digest": getattr(definition, "digest", None),
        "requested_candidate_config_digest": prereg_index[
            "requested_candidate_config_digest"
        ],
        "verification_digest": getattr(verification, "digest", None),
        "comparison_digest": getattr(comparison, "digest", None),
        "decision_digest": getattr(decision, "digest", None),
        "decision": getattr(getattr(decision, "decision", None), "value", None),
        "mean_reversion_side_effect_must_be_disclosed": True,
        "return_only_search_stop_rule": prereg_index["return_only_search_stop_rule"],
    }
    for key, value in expected.items():
        if index.get(key) != value:
            raise RuntimeError(f"Experiment 0003 result index drift: {key}")
    if getattr(candidate_evidence, "fingerprint", None) != raw["candidate_evidence_fingerprint"]:
        raise RuntimeError("candidate fingerprint differs from fresh raw evidence")
    if index.get("independent_raw_evidence") != raw:
        raise RuntimeError("persisted independent raw evidence differs from fresh oracle")
    if raw.get("formal_decision") != expected["decision"]:
        raise RuntimeError("fresh decision replay differs from persisted decision")
    if not isinstance(index.get("execution_workflow_run_id"), int) or index[
        "execution_workflow_run_id"
    ] <= 0:
        raise RuntimeError("execution_workflow_run_id malformed")
    for key in ("prereg_run_id", "prereg_artifact_id", "prereg_verifier_artifact_id"):
        if not isinstance(index.get(key), int) or index[key] <= 0:
            raise RuntimeError(f"{key} malformed")
    for key in ("prereg_artifact_digest", "prereg_verifier_artifact_digest"):
        value = index.get(key)
        if not isinstance(value, str) or not value.startswith("sha256:"):
            raise RuntimeError(f"{key} malformed")
    return index


def verify(prereg_root: Path, result_root: Path, report_root: Path) -> dict[str, object]:
    """Independently verify a completed Experiment 0003 result artifact."""

    _assert_immutable_prefix(prereg_root, result_root)
    prereg_index = _load_prereg_index(prereg_root)
    prereg_state, result_state, _, _ = _validate_study_bindings(
        prereg_root,
        result_root,
    )
    exp3, decision = _validate_exp3_state(prereg_state, result_state)
    raw = _independent_raw_analysis(result_root)
    candidate = getattr(exp3, "candidate", None)
    evidence = getattr(candidate, "evidence", None)
    if getattr(evidence, "fingerprint", None) != raw["candidate_evidence_fingerprint"]:
        raise RuntimeError("candidate fingerprint mismatch after state reconstruction")
    _crosscheck_persisted_comparison(getattr(exp3, "comparison", None), raw)
    result_index = _validate_result_index(
        result_root,
        prereg_index=prereg_index,
        raw=raw,
        exp3=exp3,
        decision=decision,
    )

    report: dict[str, object] = {
        "schema_version": "canonical_m2_portable_exp003_postverify_v1",
        "study_digest": EXPECTED_STUDY_DIGEST,
        "dataset_id": EXPECTED_DATASET_ID,
        "baseline_evidence_fingerprint": EXPECTED_BASELINE_FINGERPRINT,
        "candidate_evidence_fingerprint": raw["candidate_evidence_fingerprint"],
        "definition_digest": getattr(getattr(exp3, "definition", None), "digest", None),
        "verification_digest": getattr(getattr(exp3, "verification", None), "digest", None),
        "comparison_digest": getattr(getattr(exp3, "comparison", None), "digest", None),
        "decision_digest": getattr(decision, "digest", None),
        "formal_target_strategy": "trend",
        "independent_raw_evidence": raw,
        "result_index_execution_run_id": result_index.get("execution_workflow_run_id"),
        "immutable_preregistration_prefix_verified": True,
        "prior_exp001_exp002_evidence_unchanged": True,
        "controlled_delta_verified": True,
        "unaffected_raw_return_equality_verified": True,
        "deterministic_seed_invariance_verified": True,
        "independent_decision_replayed": True,
        "candidate_reexecuted": False,
        "result_interpretation_sealed": True,
        "interpretation_authorized": False,
        "verification_passed": True,
    }
    report_root.mkdir(parents=True, exist_ok=True)
    (report_root / "postverify.json").write_text(
        json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prereg-root", type=Path, required=True)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--report-root", type=Path, required=True)
    args = parser.parse_args()
    verify(args.prereg_root, args.result_root, args.report_root)


if __name__ == "__main__":
    main()


__all__ = [
    "FROZEN_BASELINE_TREND_MEDIAN_TURNOVER",
    "independent_decision",
    "validate_result_index_safety_flags",
    "verify",
]
