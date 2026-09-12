"""Execute portable Canonical M2 Experiment 0001 after the sealed prereg gate."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import median

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research.issue511_portable_exp001 import (  # noqa: E402
    EXPECTED_BASELINE_FINGERPRINT,
    EXPECTED_DATASET_ARTIFACT_DIGEST,
    EXPECTED_DATASET_ID,
    EXPECTED_IMPLEMENTATION_DIGEST,
    EXPECTED_REQUESTED_CANDIDATE_DIGEST,
    EXPECTED_RUNTIME_DIGEST,
    EXPECTED_SEEDS,
    EXPECTED_STUDY_DIGEST,
    EXPECTED_SYMBOLS,
    _metric_number,
    _strategy_entry,
    _validate_baseline,
    _validate_definition,
)
from trade_rl.evaluation.experiments import (  # noqa: E402
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
from trade_rl.evaluation.experiments.inspection import _reconstruct  # noqa: E402
from trade_rl.evaluation.experiments.store import StudyStore  # noqa: E402

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
EXPECTED_DEFINITION_DIGEST = "8edbba0ad9cdb938b9b10a65e88b940dbbd4c37b1000fe29acadb5d209d52f28"
EXPECTED_BASELINE_MEDIAN_TURNOVER = 858.3114067468092


def _compound(values: np.ndarray) -> float:
    wealth = 1.0
    for value in values:
        resolved = float(value)
        if not math.isfinite(resolved) or resolved <= -1.0:
            raise RuntimeError("raw return is non-finite or <= -1")
        wealth *= 1.0 + resolved
    result = wealth - 1.0
    if not math.isfinite(result):
        raise RuntimeError("compounded return is non-finite")
    return result


def _entry_values(run, symbol_index: int, strategy: str) -> tuple[dict[str, object], np.ndarray]:
    entry = _strategy_entry(run, symbol_index, strategy)
    return_key = entry.get("return_key")
    if not isinstance(return_key, str):
        raise RuntimeError("strategy return_key malformed")
    values = run.returns.get(return_key)
    if values is None or values.ndim != 1 or not np.isfinite(values).all():
        raise RuntimeError("strategy raw returns malformed")
    metrics = entry.get("metrics")
    observed_total_return = _metric_number(metrics, "total_return")
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
    baseline = load_evidence_set(study_root / "baseline" / "evidence")
    candidate = load_evidence_set(
        study_root / "experiments" / "0001" / "candidate" / "evidence"
    )
    if baseline.evidence.fingerprint != EXPECTED_BASELINE_FINGERPRINT:
        raise RuntimeError("baseline fingerprint drift during Experiment")
    if baseline.evidence.ppo_seeds != EXPECTED_SEEDS or candidate.evidence.ppo_seeds != EXPECTED_SEEDS:
        raise RuntimeError("Experiment PPO seed roster drift")
    if tuple(sorted(baseline.runs)) != EXPECTED_SEEDS or tuple(sorted(candidate.runs)) != EXPECTED_SEEDS:
        raise RuntimeError("Experiment Run roster drift")

    _validate_seed_invariance(baseline, DETERMINISTIC)
    _validate_seed_invariance(candidate, DETERMINISTIC)

    unaffected_checks = 0
    for seed in EXPECTED_SEEDS:
        for symbol_index, symbol in enumerate(EXPECTED_SYMBOLS):
            for strategy in UNAFFECTED:
                _, baseline_values = _entry_values(baseline.runs[seed], symbol_index, strategy)
                _, candidate_values = _entry_values(candidate.runs[seed], symbol_index, strategy)
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
            _, baseline_values = _entry_values(baseline.runs[seed], symbol_index, "ppo")
            _, candidate_values = _entry_values(candidate.runs[seed], symbol_index, "ppo")
            baseline_total = _compound(baseline_values)
            candidate_total = _compound(candidate_values)
            excess = candidate_total - baseline_total
            if excess != 0.0 or not np.array_equal(baseline_values, candidate_values):
                raise RuntimeError("PPO changed under RULE_THRESHOLDS")
            per_symbol[symbol] = {
                "baseline_total_return": baseline_total,
                "candidate_total_return": candidate_total,
                "excess_total_return": excess,
            }
        ppo_by_seed_symbol[str(seed)] = per_symbol

    candidate_aggregate_cost = 0.0
    candidate_trading_observations = 0
    candidate_positive_cost_observations = 0
    candidate_cash_observations = 0
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
                candidate_aggregate_cost += total_cost
                if n_trades > 0:
                    candidate_trading_observations += 1
                    if total_cost <= 0.0:
                        raise RuntimeError(
                            "candidate trading observation has non-positive cost: "
                            f"seed={seed} symbol={symbol} strategy={strategy}"
                        )
                    candidate_positive_cost_observations += 1
                if strategy == "cash":
                    candidate_cash_observations += 1
                    if n_trades != 0 or total_cost != 0.0 or not np.array_equal(
                        values, np.zeros_like(values)
                    ):
                        raise RuntimeError("candidate cash semantics drift")

    if candidate_trading_observations == 0:
        raise RuntimeError("candidate produced no trading observations")
    if candidate_positive_cost_observations != candidate_trading_observations:
        raise RuntimeError("not every candidate trading observation has positive cost")
    if candidate_cash_observations != len(EXPECTED_SEEDS) * len(EXPECTED_SYMBOLS):
        raise RuntimeError("candidate cash observation roster incomplete")

    mean_reversion = deterministic_effects["mean_reversion"]
    mean_reversion_by_symbol = mean_reversion["by_symbol"]
    mean_reversion_aggregate = mean_reversion["aggregate"]
    if not isinstance(mean_reversion_by_symbol, dict) or not isinstance(
        mean_reversion_aggregate, dict
    ):
        raise RuntimeError("mean-reversion independent evidence malformed")
    candidate_total_returns = [
        float(mean_reversion_by_symbol[symbol]["candidate_total_return"])  # type: ignore[index]
        for symbol in EXPECTED_SYMBOLS
    ]
    candidate_turnovers = [
        float(mean_reversion_by_symbol[symbol]["candidate_turnover"])  # type: ignore[index]
        for symbol in EXPECTED_SYMBOLS
    ]
    positive_effects = int(mean_reversion_aggregate["positive_symbol_count"])
    median_excess = float(mean_reversion_aggregate["median_excess_total_return"])
    candidate_positive_returns = sum(value > 0.0 for value in candidate_total_returns)
    candidate_median_turnover = float(median(candidate_turnovers))

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
        decision = ExperimentDecisionKind.ACCEPT_CANDIDATE
    elif keep_baseline:
        decision = ExperimentDecisionKind.KEEP_BASELINE
    else:
        decision = ExperimentDecisionKind.INCONCLUSIVE

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
            "aggregate_total_cost_across_seed_runs": candidate_aggregate_cost,
            "trading_observations": candidate_trading_observations,
            "positive_cost_observations": candidate_positive_cost_observations,
            "cash_observations": candidate_cash_observations,
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
    formal = independent["mean_reversion_formal_inputs"]
    if not isinstance(formal, dict):
        raise RuntimeError("independent formal inputs malformed")
    if mean_reversion.get("positive_symbol_count") != formal[
        "positive_factor_effect_symbol_count"
    ]:
        raise RuntimeError("persisted mean-reversion positive count differs from raw oracle")
    persisted_median = mean_reversion.get("median_excess_total_return")
    if not isinstance(persisted_median, (int, float)) or isinstance(persisted_median, bool):
        raise RuntimeError("persisted mean-reversion median malformed")
    if not math.isclose(
        float(persisted_median),
        float(formal["median_excess_total_return"]),
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise RuntimeError("persisted mean-reversion median differs from raw oracle")


def execute(root: Path) -> None:
    snapshot, baseline_turnover = _validate_baseline(root, allow_definition=True)
    definition = _validate_definition(root)
    if definition.digest != EXPECTED_DEFINITION_DIGEST:
        raise RuntimeError("sealed Experiment definition digest drift")
    if baseline_turnover != EXPECTED_BASELINE_MEDIAN_TURNOVER:
        raise RuntimeError("sealed baseline turnover drift")
    if snapshot.plan.digest != EXPECTED_STUDY_DIGEST:
        raise RuntimeError("sealed Study digest drift")
    prereg_index = json.loads(
        (root / "portable-exp001-prereg-index.json").read_text(encoding="utf-8")
    )
    if prereg_index.get("requested_candidate_config_digest") != EXPECTED_REQUESTED_CANDIDATE_DIGEST:
        raise RuntimeError("sealed candidate digest drift")
    if prereg_index.get("result_inspected_before_preregistration") is not False:
        raise RuntimeError("preregistration is not result-blind")

    dataset_root = root / "dataset"
    study_root = root / "study"
    try:
        run_experiment(study_root, 1, dataset_root=dataset_root)
    except Exception as error:
        try:
            record_experiment_failure(
                study_root,
                1,
                reason=f"candidate execution failed before complete EvidenceSet: {type(error).__name__}: {error}",
                recorded_by="openai-gpt-5.6-sol",
                recorded_at=datetime.now(timezone.utc),
            )
        except Exception:
            pass
        raise

    verification = verify_experiment(study_root, 1)
    if verification.status is not ControlledVerificationStatus.CONTROLLED:
        raise RuntimeError(f"candidate delta is not CONTROLLED: {verification.violations!r}")
    if verification.changed_paths != (("rule_entry_threshold",),):
        raise RuntimeError("controlled verification changed paths differ from preregistration")

    independent = _independent_analysis(study_root)
    comparison = compare_experiment(study_root, 1)
    _crosscheck_persisted_comparison(comparison, independent)

    decision_kind = ExperimentDecisionKind(str(independent["formal_decision"]))
    formal = independent["mean_reversion_formal_inputs"]
    assert isinstance(formal, dict)
    rationale = (
        "Frozen portable Experiment 0001 rule replay: "
        f"positive_effect_symbols={formal['positive_factor_effect_symbol_count']}/5, "
        f"median_excess_total_return={formal['median_excess_total_return']!r}, "
        f"candidate_positive_total_return_symbols={formal['candidate_positive_total_return_symbol_count']}/5, "
        f"candidate_median_turnover={formal['candidate_median_turnover']!r}, "
        f"frozen_baseline_median_turnover={formal['frozen_baseline_median_turnover']!r}."
    )
    decision = decide_experiment(
        study_root,
        1,
        decision=decision_kind,
        rationale=rationale,
        decided_by="openai-gpt-5.6-sol",
        decided_at=datetime.now(timezone.utc),
    )

    final_snapshot = inspect_study(study_root)
    if final_snapshot.plan.digest != EXPECTED_STUDY_DIGEST:
        raise RuntimeError("Study digest changed during Experiment")
    if final_snapshot.experiment_sequences != (1,) or final_snapshot.terminal_sequences != (1,):
        raise RuntimeError("Experiment did not reach exactly one terminal decision")
    if final_snapshot.frozen:
        raise RuntimeError("Study was frozen by a single Experiment unexpectedly")
    state = _reconstruct(StudyStore(study_root))
    experiment = state.experiments[0]
    if experiment.decision is None or experiment.decision.digest != decision.digest:
        raise RuntimeError("persisted decision reconstruction mismatch")
    if experiment.comparison is None or experiment.comparison.digest != comparison.digest:
        raise RuntimeError("persisted comparison reconstruction mismatch")
    if experiment.verification is None or experiment.verification.digest != verification.digest:
        raise RuntimeError("persisted verification reconstruction mismatch")
    if experiment.candidate is None:
        raise RuntimeError("terminal Experiment lacks candidate evidence")

    result_index = {
        "schema_version": "canonical_m2_portable_exp001_result_index_v1",
        "issue_number": 511,
        "execution_workflow_run_id": int(os.environ["GITHUB_RUN_ID"]),
        "execution_helper_git_sha": os.environ["GITHUB_SHA"],
        "prereg_run_id": int(os.environ["PREREG_RUN_ID"]),
        "prereg_artifact_id": int(os.environ["PREREG_ARTIFACT_ID"]),
        "prereg_artifact_digest": os.environ["PREREG_ARTIFACT_DIGEST"],
        "prereg_verifier_run_id": int(os.environ["PREREG_VERIFY_RUN_ID"]),
        "prereg_verifier_artifact_id": int(os.environ["PREREG_VERIFY_ARTIFACT_ID"]),
        "prereg_verifier_artifact_digest": os.environ["PREREG_VERIFY_ARTIFACT_DIGEST"],
        "study_digest": EXPECTED_STUDY_DIGEST,
        "baseline_evidence_fingerprint": EXPECTED_BASELINE_FINGERPRINT,
        "candidate_evidence_fingerprint": experiment.candidate.evidence.fingerprint,
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
        "study_frozen": False,
        "profitability_claimed": False,
        "winner_claimed": False,
        "final_test_authorized": False,
        "production_authorized": False,
    }
    (root / "portable-exp001-result-index.json").write_text(
        json.dumps(result_index, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print("PORTABLE_EXP001_RESULT=" + json.dumps(result_index, sort_keys=True, allow_nan=False))
    output = os.environ.get("GITHUB_OUTPUT")
    if output:
        with Path(output).open("a", encoding="utf-8") as handle:
            handle.write(f"candidate_fingerprint={experiment.candidate.evidence.fingerprint}\n")
            handle.write(f"verification_digest={verification.digest}\n")
            handle.write(f"comparison_digest={comparison.digest}\n")
            handle.write(f"decision_digest={decision.digest}\n")
            handle.write(f"decision={decision.decision.value}\n")
            handle.write(f"artifact_name=canonical-m2-portable-exp001-result-v1-{os.environ['GITHUB_RUN_ID']}\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    execute(args.root)


if __name__ == "__main__":
    main()
