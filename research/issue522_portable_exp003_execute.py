"""Execute sealed portable Canonical M2 Experiment 0003 exactly once.

The helper consumes the independently verified preregistration artifact as the
only Experiment 0003 authority.  It changes no production code and does not
invent post-result thresholds.  The formal target is trend; mean-reversion is
reported only as the preregistered RULE_SIGNAL side effect.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from collections.abc import Mapping
from datetime import datetime, timezone
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
    EXPERIMENT_SEQUENCE,
    FROZEN_BASELINE_TREND_MEDIAN_TURNOVER,
    ISSUE_NUMBER,
    _validate_definition,
    validate_prereg_index_payload,
)
from trade_rl.evaluation.experiments import (
    ControlledVerificationStatus,
    ExperimentDecisionKind,
    compare_experiment,
    decide_experiment,
    inspect_study,
    load_evidence_set,
    record_experiment_failure,
    run_experiment,
    verify_experiment,
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


def compound(values: np.ndarray) -> float:
    """Reconstruct ordered compounded total return from raw interval returns."""

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


def formal_decision(
    *,
    positive_effects: int,
    median_excess: float,
    candidate_positive_returns: int,
    candidate_median_turnover: float,
) -> ExperimentDecisionKind:
    """Apply the immutable Experiment 0003 preregistered trend rule."""

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


def _entry_values(
    run: object,
    symbol_index: int,
    strategy: str,
) -> tuple[dict[str, object], np.ndarray]:
    entry = _strategy_entry(run, symbol_index, strategy)
    return_key = entry.get("return_key")
    if not isinstance(return_key, str):
        raise RuntimeError("strategy return_key malformed")
    returns = getattr(run, "returns", None)
    if not isinstance(returns, dict):
        raise RuntimeError("run raw-return payload malformed")
    values = returns.get(return_key)
    if values is None:
        raise RuntimeError("strategy raw returns missing")
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not np.isfinite(array).all():
        raise RuntimeError("strategy raw returns malformed")
    observed = _metric_number(entry.get("metrics"), "total_return")
    reconstructed = compound(array)
    if not math.isclose(observed, reconstructed, rel_tol=1e-12, abs_tol=1e-12):
        raise RuntimeError("persisted total_return differs from raw-return compounding")
    return entry, array


def _require_int(metrics: object, field: str) -> int:
    if not isinstance(metrics, dict):
        raise RuntimeError("strategy metrics malformed")
    value = metrics.get(field)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise RuntimeError(f"strategy metric malformed: {field}")
    return value


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


def independent_analysis(study_root: Path) -> dict[str, object]:
    """Recompute Experiment 0003 evidence directly from raw interval returns."""

    baseline = load_evidence_set(study_root / "baseline/evidence")
    candidate = load_evidence_set(
        study_root / "experiments/0003/candidate/evidence"
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
                _, before = _entry_values(
                    baseline.runs[seed], symbol_index, strategy
                )
                _, after = _entry_values(
                    candidate.runs[seed], symbol_index, strategy
                )
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
            before_total = compound(before)
            after_total = compound(after)
            excess = after_total - before_total
            excesses.append(excess)
            by_symbol[symbol] = {
                "baseline_total_return": before_total,
                "candidate_total_return": after_total,
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
        float(by_symbol[symbol]["candidate_total_return"])  # type: ignore[index]
        for symbol in EXPECTED_SYMBOLS
    ]
    candidate_turnovers = [
        float(by_symbol[symbol]["candidate_turnover"])  # type: ignore[index]
        for symbol in EXPECTED_SYMBOLS
    ]
    positive_effects = int(aggregate["positive_symbol_count"])
    median_excess = float(aggregate["median_excess_total_return"])
    positive_returns = sum(value > 0.0 for value in candidate_totals)
    median_turnover = float(median(candidate_turnovers))
    decision = formal_decision(
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


def _mapping(value: object, *, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(
        not isinstance(key, str) for key in value
    ):
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
    """Cross-check frozen Mapping-backed trend comparison against raw evidence."""

    factor_effect = _mapping(
        getattr(comparison, "factor_effect", None),
        field="persisted comparison factor_effect",
    )
    cross_symbol = _mapping(
        factor_effect.get("cross_symbol"),
        field="persisted comparison cross_symbol",
    )
    trend = _mapping(cross_symbol.get("trend"), field="persisted trend summary")
    formal = _mapping(
        independent.get("trend_formal_inputs"), field="independent trend inputs"
    )
    if trend.get("positive_symbol_count") != formal.get(
        "positive_factor_effect_symbol_count"
    ):
        raise RuntimeError("persisted trend positive count differs from raw oracle")
    persisted_median = _number(
        trend.get("median_excess_total_return"), field="persisted trend median"
    )
    independent_median = _number(
        formal.get("median_excess_total_return"), field="independent trend median"
    )
    if not math.isclose(
        persisted_median, independent_median, rel_tol=1e-12, abs_tol=1e-12
    ):
        raise RuntimeError("persisted trend median differs from raw oracle")


def _validate_prereg_index(
    root: Path,
    *,
    definition_digest: str,
    requested_candidate_config_digest: str,
) -> dict[str, object]:
    path = root / "portable-exp003-prereg-index.json"
    if not path.is_file():
        raise RuntimeError("Experiment 0003 preregistration index missing")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError("Experiment 0003 preregistration index malformed")
    structural_sha = payload.get("structural_report_sha256")
    if not isinstance(structural_sha, str):
        raise RuntimeError("sealed structural report digest missing")
    validate_prereg_index_payload(
        payload,
        definition_digest=definition_digest,
        requested_candidate_config_digest=requested_candidate_config_digest,
        structural_report_sha256=structural_sha,
    )
    return payload


def _prior_binding(state: object) -> tuple[tuple[object, ...], tuple[object, ...]]:
    experiments = getattr(state, "experiments", None)
    if not isinstance(experiments, list | tuple) or len(experiments) < 2:
        raise RuntimeError("prior Experiment state malformed")
    result: list[tuple[object, ...]] = []
    for experiment in experiments[:2]:
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
    return result[0], result[1]


def execute(root: Path) -> dict[str, object]:
    """Execute the sealed candidate once, verify it, and append a mechanical decision."""

    definition, requested_digest = _validate_definition(root)
    prereg_index = _validate_prereg_index(
        root,
        definition_digest=definition.digest,
        requested_candidate_config_digest=requested_digest,
    )
    study_root = root / "study"
    dataset_root = root / "dataset"
    state_before = _reconstruct(StudyStore(study_root))
    prior_before = _prior_binding(state_before)
    exp3_before = state_before.experiments[2]
    if any(
        item is not None
        for item in (
            exp3_before.candidate,
            exp3_before.verification,
            exp3_before.comparison,
            exp3_before.decision,
            exp3_before.failure,
        )
    ):
        raise RuntimeError("Experiment 0003 is not result-blind before execution")

    try:
        run_experiment(study_root, EXPERIMENT_SEQUENCE, dataset_root=dataset_root)
    except Exception as error:
        try:
            record_experiment_failure(
                study_root,
                EXPERIMENT_SEQUENCE,
                reason=(
                    "candidate execution failed before complete EvidenceSet: "
                    f"{type(error).__name__}: {error}"
                ),
                recorded_by="openai-gpt-5.6-sol",
                recorded_at=datetime.now(timezone.utc),
            )
        except Exception:
            pass
        raise

    verification = verify_experiment(study_root, EXPERIMENT_SEQUENCE)
    if verification.status is not ControlledVerificationStatus.CONTROLLED:
        raise RuntimeError(
            f"candidate delta is not CONTROLLED: {verification.violations!r}"
        )
    if verification.changed_paths != EXPECTED_CHANGED_PATHS:
        raise RuntimeError("controlled changed paths differ from preregistration")

    independent = independent_analysis(study_root)
    comparison = compare_experiment(study_root, EXPERIMENT_SEQUENCE)
    crosscheck_persisted_comparison(comparison, independent)
    decision_kind = ExperimentDecisionKind(str(independent["formal_decision"]))
    decision = decide_experiment(
        study_root,
        EXPERIMENT_SEQUENCE,
        decision=decision_kind,
        rationale=(
            "Mechanical replay of the preregistered Experiment 0003 trend decision rule; "
            "performance remains interpretation-sealed until independent post-verification."
        ),
        decided_by="openai-gpt-5.6-sol",
        decided_at=datetime.now(timezone.utc),
    )

    snapshot = inspect_study(study_root)
    if snapshot.plan.digest != EXPECTED_STUDY_DIGEST:
        raise RuntimeError("Study digest changed during Experiment 0003")
    if snapshot.experiment_sequences != (1, 2, 3):
        raise RuntimeError("Experiment sequence drift after Experiment 0003")
    if snapshot.terminal_sequences != (1, 2, 3):
        raise RuntimeError("Experiment 0003 did not reach one terminal decision")
    if snapshot.frozen:
        raise RuntimeError("Study was frozen unexpectedly")

    state_after = _reconstruct(StudyStore(study_root))
    if _prior_binding(state_after) != prior_before:
        raise RuntimeError("prior Experiment evidence changed during Experiment 0003")
    exp3 = state_after.experiments[2]
    if exp3.candidate is None:
        raise RuntimeError("terminal Experiment 0003 lacks candidate evidence")
    if exp3.verification is None or exp3.verification.digest != verification.digest:
        raise RuntimeError("persisted Experiment 0003 verification mismatch")
    if exp3.comparison is None or exp3.comparison.digest != comparison.digest:
        raise RuntimeError("persisted Experiment 0003 comparison mismatch")
    if exp3.decision is None or exp3.decision.digest != decision.digest:
        raise RuntimeError("persisted Experiment 0003 decision mismatch")

    result_index: dict[str, object] = {
        "schema_version": "canonical_m2_portable_exp003_result_index_v1",
        "issue_number": ISSUE_NUMBER,
        "execution_workflow_run_id": int(os.environ.get("GITHUB_RUN_ID", "0")),
        "execution_helper_git_sha": os.environ.get("GITHUB_SHA", "local"),
        "prereg_run_id": int(os.environ.get("PREREG_RUN_ID", "0")),
        "prereg_artifact_id": int(os.environ.get("PREREG_ARTIFACT_ID", "0")),
        "prereg_artifact_digest": os.environ.get("PREREG_ARTIFACT_DIGEST", ""),
        "prereg_verifier_artifact_id": int(
            os.environ.get("PREREG_VERIFY_ARTIFACT_ID", "0")
        ),
        "prereg_verifier_artifact_digest": os.environ.get(
            "PREREG_VERIFY_ARTIFACT_DIGEST", ""
        ),
        "study_digest": EXPECTED_STUDY_DIGEST,
        "baseline_evidence_fingerprint": EXPECTED_BASELINE_FINGERPRINT,
        "candidate_evidence_fingerprint": exp3.candidate.evidence.fingerprint,
        "dataset_id": EXPECTED_DATASET_ID,
        "dataset_artifact_digest": EXPECTED_DATASET_ARTIFACT_DIGEST,
        "implementation_digest": EXPECTED_IMPLEMENTATION_DIGEST,
        "runtime_environment_digest": EXPECTED_RUNTIME_DIGEST,
        "definition_digest": definition.digest,
        "requested_candidate_config_digest": requested_digest,
        "verification_digest": verification.digest,
        "verification_status": verification.status.value,
        "changed_paths": [list(path) for path in verification.changed_paths],
        "comparison_digest": comparison.digest,
        "decision_digest": decision.digest,
        "decision": decision.decision.value,
        "formal_target_strategy": "trend",
        "mean_reversion_side_effect_must_be_disclosed": True,
        "return_only_search_stop_rule": prereg_index["return_only_search_stop_rule"],
        "independent_raw_evidence": independent,
        "candidate_execution_count": 1,
        "candidate_rerun": False,
        "study_frozen": False,
        "interpretation_authorized": False,
        "profitability_claimed": False,
        "winner_claimed": False,
        "final_test_authorized": False,
        "production_authorized": False,
    }
    (root / "portable-exp003-result-index.json").write_text(
        json.dumps(result_index, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return result_index


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    execute(args.root)


if __name__ == "__main__":
    main()


__all__ = [
    "FROZEN_BASELINE_TREND_MEDIAN_TURNOVER",
    "compound",
    "crosscheck_persisted_comparison",
    "execute",
    "formal_decision",
    "independent_analysis",
]
