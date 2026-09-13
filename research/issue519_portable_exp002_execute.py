"""Execute portable Canonical M2 Experiment 0002 after the sealed prereg gate."""

from __future__ import annotations

import argparse
import json
import math
import os
from datetime import datetime, timezone
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
    SOURCE_RECOVERY_HEAD,
    SOURCE_RECOVERY_RUN_ID,
    SOURCE_RESULT_ARTIFACT_DIGEST,
    SOURCE_RESULT_ARTIFACT_ID,
    _metric_number,
    _strategy_entry,
    _validate_exp002_definition,
    _validate_recovered_study,
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

EXPECTED_BASELINE_MEDIAN_TURNOVER = 858.3114067468092
EXPECTED_DEFINITION_DIGEST = (
    "854814c85f3e0175c83cc7f3cd342cefa627780bb9d88374d948a7ec167fca87"
)
EXPECTED_REQUESTED_CANDIDATE_DIGEST = (
    "a966dd408f43c64c1833a32fa0507a8e6b7d7fa865fb88c73cd0c1d53a05fcd4"
)
EXPECTED_CHANGED_PATHS = (("signal_index",), ("signal_name",))
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
UNAFFECTED = frozenset(
    {"cash", "constant_long", "constant_short", "ridge24", "lightgbm24", "ppo"}
)
DETERMINISTIC = tuple(name for name in STRATEGIES if name != "ppo")


def _compound(values: np.ndarray) -> float:
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


def _formal_decision(
    *,
    positive_effects: int,
    median_excess: float,
    candidate_positive_returns: int,
    candidate_median_turnover: float,
) -> ExperimentDecisionKind:
    """Apply the immutable Experiment 0002 preregistered decision rule."""

    accept = (
        positive_effects >= 4
        and median_excess > 0.0
        and candidate_positive_returns == 5
        and candidate_median_turnover < EXPECTED_BASELINE_MEDIAN_TURNOVER
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
    run,
    symbol_index: int,
    strategy: str,
) -> tuple[dict[str, object], np.ndarray]:
    entry = _strategy_entry(run, symbol_index, strategy)
    return_key = entry.get("return_key")
    if not isinstance(return_key, str):
        raise RuntimeError("strategy return_key malformed")
    values = run.returns.get(return_key)
    if values is None or values.ndim != 1 or not np.isfinite(values).all():
        raise RuntimeError("strategy raw returns malformed")
    observed_total_return = _metric_number(entry.get("metrics"), "total_return")
    independently_compounded = _compound(values)
    if not math.isclose(
        observed_total_return,
        independently_compounded,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise RuntimeError("persisted total_return differs from raw-return compounding")
    return entry, values


def _require_int(metrics: object, field: str) -> int:
    if not isinstance(metrics, dict):
        raise RuntimeError("strategy metrics malformed")
    value = metrics.get(field)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise RuntimeError(f"strategy metric malformed: {field}")
    return value


def _validate_seed_invariance(loaded, strategies: tuple[str, ...]) -> None:
    reference: dict[tuple[str, str], np.ndarray] = {}
    for seed in EXPECTED_SEEDS:
        run = loaded.runs[seed]
        for symbol_index, symbol in enumerate(EXPECTED_SYMBOLS):
            for strategy in strategies:
                _, values = _entry_values(run, symbol_index, strategy)
                key = (symbol, strategy)
                if seed == EXPECTED_SEEDS[0]:
                    reference[key] = values.copy()
                elif not np.array_equal(reference[key], values):
                    raise RuntimeError(
                        f"deterministic strategy is seed-sensitive: {symbol}/{strategy}"
                    )


def _effect_summary(excesses: list[float]) -> dict[str, object]:
    if len(excesses) != len(EXPECTED_SYMBOLS):
        raise RuntimeError("factor-effect symbol roster incomplete")
    return {
        "symbol_count": len(excesses),
        "positive_symbol_count": sum(value > 0.0 for value in excesses),
        "negative_symbol_count": sum(value < 0.0 for value in excesses),
        "zero_symbol_count": sum(value == 0.0 for value in excesses),
        "median_excess_total_return": float(median(excesses)),
        "worst_excess_total_return": min(excesses),
        "best_excess_total_return": max(excesses),
    }


def _independent_analysis(study_root: Path) -> dict[str, object]:
    baseline = load_evidence_set(study_root / "baseline/evidence")
    candidate = load_evidence_set(
        study_root / "experiments/0002/candidate/evidence"
    )
    if baseline.evidence.fingerprint != EXPECTED_BASELINE_FINGERPRINT:
        raise RuntimeError("baseline fingerprint drift during Experiment 0002")
    if baseline.evidence.ppo_seeds != EXPECTED_SEEDS:
        raise RuntimeError("baseline PPO seed roster drift")
    if candidate.evidence.ppo_seeds != EXPECTED_SEEDS:
        raise RuntimeError("candidate PPO seed roster drift")
    if tuple(sorted(baseline.runs)) != EXPECTED_SEEDS:
        raise RuntimeError("baseline Run roster drift")
    if tuple(sorted(candidate.runs)) != EXPECTED_SEEDS:
        raise RuntimeError("candidate Run roster drift")

    _validate_seed_invariance(baseline, DETERMINISTIC)
    _validate_seed_invariance(candidate, DETERMINISTIC)

    unaffected_checks = 0
    for seed in EXPECTED_SEEDS:
        for symbol_index, symbol in enumerate(EXPECTED_SYMBOLS):
            for strategy in UNAFFECTED:
                _, baseline_values = _entry_values(
                    baseline.runs[seed], symbol_index, strategy
                )
                _, candidate_values = _entry_values(
                    candidate.runs[seed], symbol_index, strategy
                )
                if not np.array_equal(baseline_values, candidate_values):
                    raise RuntimeError(
                        "unaffected strategy raw-return mismatch: "
                        f"seed={seed} symbol={symbol} strategy={strategy}"
                    )
                unaffected_checks += 1

    deterministic_effects: dict[str, object] = {}
    for strategy in DETERMINISTIC:
        symbols: dict[str, object] = {}
        excesses: list[float] = []
        for symbol_index, symbol in enumerate(EXPECTED_SYMBOLS):
            baseline_entry, baseline_values = _entry_values(
                baseline.runs[EXPECTED_SEEDS[0]], symbol_index, strategy
            )
            candidate_entry, candidate_values = _entry_values(
                candidate.runs[EXPECTED_SEEDS[0]], symbol_index, strategy
            )
            baseline_total = _compound(baseline_values)
            candidate_total = _compound(candidate_values)
            excess = candidate_total - baseline_total
            excesses.append(excess)
            symbols[symbol] = {
                "baseline_total_return": baseline_total,
                "candidate_total_return": candidate_total,
                "excess_total_return": excess,
                "baseline_turnover": _metric_number(
                    baseline_entry.get("metrics"), "turnover_total"
                ),
                "candidate_turnover": _metric_number(
                    candidate_entry.get("metrics"), "turnover_total"
                ),
            }
        deterministic_effects[strategy] = {
            "by_symbol": symbols,
            "aggregate": _effect_summary(excesses),
        }

    ppo_by_seed_symbol: dict[str, object] = {}
    for seed in EXPECTED_SEEDS:
        per_symbol: dict[str, object] = {}
        for symbol_index, symbol in enumerate(EXPECTED_SYMBOLS):
            _, baseline_values = _entry_values(
                baseline.runs[seed], symbol_index, "ppo"
            )
            _, candidate_values = _entry_values(
                candidate.runs[seed], symbol_index, "ppo"
            )
            if not np.array_equal(baseline_values, candidate_values):
                raise RuntimeError(
                    f"PPO changed under RULE_SIGNAL: seed={seed} symbol={symbol}"
                )
            baseline_total = _compound(baseline_values)
            candidate_total = _compound(candidate_values)
            excess = candidate_total - baseline_total
            if excess != 0.0:
                raise RuntimeError("PPO exact raw-return equality produced nonzero effect")
            per_symbol[symbol] = {
                "baseline_total_return": baseline_total,
                "candidate_total_return": candidate_total,
                "excess_total_return": excess,
            }
        ppo_by_seed_symbol[str(seed)] = per_symbol

    aggregate_cost = 0.0
    trading_observations = 0
    positive_cost_observations = 0
    cash_observations = 0
    for seed in EXPECTED_SEEDS:
        run = candidate.runs[seed]
        for symbol_index, symbol in enumerate(EXPECTED_SYMBOLS):
            for strategy in STRATEGIES:
                entry, values = _entry_values(run, symbol_index, strategy)
                metrics = entry.get("metrics")
                diagnostics = entry.get("diagnostics")
                total_cost = _metric_number(metrics, "total_cost")
                n_trades = _require_int(metrics, "n_trades")
                if isinstance(diagnostics, dict):
                    if diagnostics.get("total_cost") != metrics.get("total_cost"):  # type: ignore[union-attr]
                        raise RuntimeError("metric/diagnostic total_cost mismatch")
                    if diagnostics.get("n_trades") != metrics.get("n_trades"):  # type: ignore[union-attr]
                        raise RuntimeError("metric/diagnostic n_trades mismatch")
                if total_cost < 0.0:
                    raise RuntimeError("candidate total_cost is negative")
                aggregate_cost += total_cost
                if n_trades > 0:
                    trading_observations += 1
                    if total_cost <= 0.0:
                        raise RuntimeError(
                            "candidate trading observation has non-positive cost: "
                            f"seed={seed} symbol={symbol} strategy={strategy}"
                        )
                    positive_cost_observations += 1
                if strategy == "cash":
                    cash_observations += 1
                    if n_trades != 0 or total_cost != 0.0:
                        raise RuntimeError("candidate cash trade/cost semantics drift")
                    if not np.array_equal(values, np.zeros_like(values)):
                        raise RuntimeError("candidate cash raw-return semantics drift")

    if trading_observations == 0:
        raise RuntimeError("candidate produced no trading observations")
    if positive_cost_observations != trading_observations:
        raise RuntimeError("not every candidate trading observation has positive cost")
    expected_cash = len(EXPECTED_SEEDS) * len(EXPECTED_SYMBOLS)
    if cash_observations != expected_cash:
        raise RuntimeError("candidate cash observation roster incomplete")

    mean_reversion = deterministic_effects.get("mean_reversion")
    if not isinstance(mean_reversion, dict):
        raise RuntimeError("mean-reversion evidence missing")
    by_symbol = mean_reversion.get("by_symbol")
    aggregate = mean_reversion.get("aggregate")
    if not isinstance(by_symbol, dict) or not isinstance(aggregate, dict):
        raise RuntimeError("mean-reversion evidence malformed")
    candidate_total_returns = [
        float(by_symbol[symbol]["candidate_total_return"])  # type: ignore[index]
        for symbol in EXPECTED_SYMBOLS
    ]
    candidate_turnovers = [
        float(by_symbol[symbol]["candidate_turnover"])  # type: ignore[index]
        for symbol in EXPECTED_SYMBOLS
    ]
    positive_effects = int(aggregate["positive_symbol_count"])
    median_excess = float(aggregate["median_excess_total_return"])
    candidate_positive_returns = sum(value > 0.0 for value in candidate_total_returns)
    candidate_median_turnover = float(median(candidate_turnovers))
    decision = _formal_decision(
        positive_effects=positive_effects,
        median_excess=median_excess,
        candidate_positive_returns=candidate_positive_returns,
        candidate_median_turnover=candidate_median_turnover,
    )

    return {
        "baseline_fingerprint": baseline.evidence.fingerprint,
        "candidate_fingerprint": candidate.evidence.fingerprint,
        "unaffected_raw_return_equality_checks": unaffected_checks,
        "deterministic_effects": deterministic_effects,
        "ppo_by_seed_symbol": ppo_by_seed_symbol,
        "ppo_cross_symbol_metrics_used_for_decision": False,
        "mean_reversion_formal_inputs": {
            "positive_factor_effect_symbol_count": positive_effects,
            "median_excess_total_return": median_excess,
            "candidate_positive_total_return_symbol_count": candidate_positive_returns,
            "candidate_median_turnover": candidate_median_turnover,
            "frozen_baseline_median_turnover": EXPECTED_BASELINE_MEDIAN_TURNOVER,
        },
        "candidate_cost_semantics": {
            "aggregate_total_cost_across_seed_runs": aggregate_cost,
            "trading_observations": trading_observations,
            "positive_cost_observations": positive_cost_observations,
            "cash_observations": cash_observations,
        },
        "formal_decision": decision.value,
    }


def _crosscheck_persisted_comparison(comparison, independent: dict[str, object]) -> None:
    factor_effect = comparison.factor_effect
    cross_symbol = factor_effect.get("cross_symbol")
    if not isinstance(cross_symbol, dict):
        raise RuntimeError("persisted comparison cross_symbol malformed")
    mean_reversion = cross_symbol.get("mean_reversion")
    if not isinstance(mean_reversion, dict):
        raise RuntimeError("persisted mean-reversion comparison malformed")
    formal = independent.get("mean_reversion_formal_inputs")
    if not isinstance(formal, dict):
        raise RuntimeError("independent formal inputs malformed")
    if mean_reversion.get("positive_symbol_count") != formal[
        "positive_factor_effect_symbol_count"
    ]:
        raise RuntimeError("persisted mean-reversion positive count differs from raw oracle")
    persisted_median = mean_reversion.get("median_excess_total_return")
    if isinstance(persisted_median, bool) or not isinstance(
        persisted_median, (int, float)
    ):
        raise RuntimeError("persisted mean-reversion median malformed")
    if not math.isclose(
        float(persisted_median),
        float(formal["median_excess_total_return"]),
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise RuntimeError("persisted mean-reversion median differs from raw oracle")


def _validate_prereg_index(root: Path, definition_digest: str, requested_digest: str) -> None:
    path = root / "portable-exp002-prereg-index.json"
    if not path.is_file():
        raise RuntimeError("Experiment 0002 preregistration index missing")
    index = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "definition_digest": definition_digest,
        "requested_candidate_config_digest": requested_digest,
        "experiment_sequence": 2,
        "factor": "RULE_SIGNAL",
        "resolved_changed_paths": ["signal_index", "signal_name"],
        "candidate_executed": False,
        "result_inspected_before_preregistration": False,
        "profitability_claimed": False,
        "winner_claimed": False,
        "final_test_authorized": False,
        "production_authorized": False,
    }
    for key, value in expected.items():
        if index.get(key) != value:
            raise RuntimeError(f"sealed preregistration drift: {key}")
    if index.get("source_recovery_run_id") != SOURCE_RECOVERY_RUN_ID:
        raise RuntimeError("sealed prereg source recovery run drift")
    if index.get("source_recovery_head") != SOURCE_RECOVERY_HEAD:
        raise RuntimeError("sealed prereg source recovery head drift")
    if index.get("source_recovered_result_artifact_id") != SOURCE_RESULT_ARTIFACT_ID:
        raise RuntimeError("sealed prereg source artifact id drift")
    if (
        index.get("source_recovered_result_artifact_digest")
        != SOURCE_RESULT_ARTIFACT_DIGEST
    ):
        raise RuntimeError("sealed prereg source artifact digest drift")


def execute(root: Path) -> None:
    """Execute Experiment 0002 once and publish decision evidence locally."""

    _, state_before = _validate_recovered_study(
        root,
        allow_exp002_definition=True,
    )
    definition, requested_digest = _validate_exp002_definition(root)
    if definition.digest != EXPECTED_DEFINITION_DIGEST:
        raise RuntimeError("sealed Experiment 0002 definition digest drift")
    if requested_digest != EXPECTED_REQUESTED_CANDIDATE_DIGEST:
        raise RuntimeError("sealed Experiment 0002 requested config digest drift")
    _validate_prereg_index(root, definition.digest, requested_digest)

    exp1_before = state_before.experiments[0]
    exp1_binding = (
        exp1_before.definition.digest,
        exp1_before.candidate.evidence.fingerprint if exp1_before.candidate else None,
        exp1_before.verification.digest if exp1_before.verification else None,
        exp1_before.comparison.digest if exp1_before.comparison else None,
        exp1_before.decision.digest if exp1_before.decision else None,
    )
    expected_exp1_binding = (
        EXPECTED_EXP001_DEFINITION_DIGEST,
        EXPECTED_EXP001_CANDIDATE_FINGERPRINT,
        EXPECTED_EXP001_VERIFICATION_DIGEST,
        EXPECTED_EXP001_COMPARISON_DIGEST,
        EXPECTED_EXP001_DECISION_DIGEST,
    )
    if exp1_binding != expected_exp1_binding:
        raise RuntimeError("Experiment 0001 binding drift before Experiment 0002")

    dataset_root = root / "dataset"
    study_root = root / "study"
    try:
        run_experiment(study_root, 2, dataset_root=dataset_root)
    except Exception as error:
        try:
            record_experiment_failure(
                study_root,
                2,
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

    verification = verify_experiment(study_root, 2)
    if verification.status is not ControlledVerificationStatus.CONTROLLED:
        raise RuntimeError(
            f"candidate delta is not CONTROLLED: {verification.violations!r}"
        )
    if verification.changed_paths != EXPECTED_CHANGED_PATHS:
        raise RuntimeError("controlled changed paths differ from preregistration")

    independent = _independent_analysis(study_root)
    comparison = compare_experiment(study_root, 2)
    _crosscheck_persisted_comparison(comparison, independent)

    formal = independent.get("mean_reversion_formal_inputs")
    if not isinstance(formal, dict):
        raise RuntimeError("formal mean-reversion evidence missing")
    decision_kind = ExperimentDecisionKind(str(independent["formal_decision"]))
    rationale = (
        "Frozen portable Experiment 0002 rule replay: "
        f"positive_effect_symbols={formal['positive_factor_effect_symbol_count']}/5, "
        f"median_excess_total_return={formal['median_excess_total_return']!r}, "
        "candidate_positive_total_return_symbols="
        f"{formal['candidate_positive_total_return_symbol_count']}/5, "
        f"candidate_median_turnover={formal['candidate_median_turnover']!r}, "
        "frozen_baseline_median_turnover="
        f"{formal['frozen_baseline_median_turnover']!r}."
    )
    decision = decide_experiment(
        study_root,
        2,
        decision=decision_kind,
        rationale=rationale,
        decided_by="openai-gpt-5.6-sol",
        decided_at=datetime.now(timezone.utc),
    )

    final_snapshot = inspect_study(study_root)
    if final_snapshot.plan.digest != EXPECTED_STUDY_DIGEST:
        raise RuntimeError("Study digest changed during Experiment 0002")
    if final_snapshot.experiment_sequences != (1, 2):
        raise RuntimeError("Experiment sequence drift after Experiment 0002")
    if final_snapshot.terminal_sequences != (1, 2):
        raise RuntimeError("Experiment 0002 did not reach one terminal decision")
    if final_snapshot.frozen:
        raise RuntimeError("Study was frozen unexpectedly")

    state = _reconstruct(StudyStore(study_root))
    exp1 = state.experiments[0]
    exp2 = state.experiments[1]
    exp1_after = (
        exp1.definition.digest,
        exp1.candidate.evidence.fingerprint if exp1.candidate else None,
        exp1.verification.digest if exp1.verification else None,
        exp1.comparison.digest if exp1.comparison else None,
        exp1.decision.digest if exp1.decision else None,
    )
    if exp1_after != expected_exp1_binding:
        raise RuntimeError("Experiment 0001 evidence changed during Experiment 0002")
    if exp2.candidate is None:
        raise RuntimeError("terminal Experiment 0002 lacks candidate evidence")
    if exp2.verification is None or exp2.verification.digest != verification.digest:
        raise RuntimeError("persisted Experiment 0002 verification mismatch")
    if exp2.comparison is None or exp2.comparison.digest != comparison.digest:
        raise RuntimeError("persisted Experiment 0002 comparison mismatch")
    if exp2.decision is None or exp2.decision.digest != decision.digest:
        raise RuntimeError("persisted Experiment 0002 decision mismatch")

    result_index: dict[str, object] = {
        "schema_version": "canonical_m2_portable_exp002_result_index_v1",
        "issue_number": 519,
        "execution_workflow_run_id": int(os.environ["GITHUB_RUN_ID"]),
        "execution_helper_git_sha": os.environ["GITHUB_SHA"],
        "prereg_run_id": int(os.environ["PREREG_RUN_ID"]),
        "prereg_artifact_id": int(os.environ["PREREG_ARTIFACT_ID"]),
        "prereg_artifact_digest": os.environ["PREREG_ARTIFACT_DIGEST"],
        "prereg_verifier_artifact_id": int(
            os.environ["PREREG_VERIFY_ARTIFACT_ID"]
        ),
        "prereg_verifier_artifact_digest": os.environ[
            "PREREG_VERIFY_ARTIFACT_DIGEST"
        ],
        "study_digest": EXPECTED_STUDY_DIGEST,
        "baseline_evidence_fingerprint": EXPECTED_BASELINE_FINGERPRINT,
        "candidate_evidence_fingerprint": exp2.candidate.evidence.fingerprint,
        "dataset_id": EXPECTED_DATASET_ID,
        "dataset_artifact_digest": EXPECTED_DATASET_ARTIFACT_DIGEST,
        "implementation_digest": EXPECTED_IMPLEMENTATION_DIGEST,
        "runtime_environment_digest": EXPECTED_RUNTIME_DIGEST,
        "definition_digest": definition.digest,
        "verification_digest": verification.digest,
        "verification_status": verification.status.value,
        "changed_paths": [list(path) for path in verification.changed_paths],
        "comparison_digest": comparison.digest,
        "decision_digest": decision.digest,
        "decision": decision.decision.value,
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
    (root / "portable-exp002-result-index.json").write_text(
        json.dumps(result_index, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print("PORTABLE_EXP002_RESULT=" + json.dumps(result_index, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    execute(args.root)


if __name__ == "__main__":
    main()


__all__ = [
    "_compound",
    "_formal_decision",
    "_independent_analysis",
    "execute",
]
