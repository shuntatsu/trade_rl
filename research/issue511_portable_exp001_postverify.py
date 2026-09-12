"""Independent post-result verifier for portable Canonical M2 Experiment 0001."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import median

import numpy as np

from trade_rl.evaluation.experiments import (
    ControlledVerificationStatus,
    ExperimentDecisionKind,
    inspect_study,
    load_evidence_set,
)
from trade_rl.evaluation.experiments.inspection import _reconstruct
from trade_rl.evaluation.experiments.store import StudyStore

EXPECTED_STUDY_DIGEST = "3d8404061a4082a8e9b3c786d9f5fc9a4347631dff39c201e3cba70470dfeb79"
EXPECTED_BASELINE_FINGERPRINT = "526b485d60b394739b7a8d03535e1cfa08b26fa920119c70ee53f05faa53dd29"
EXPECTED_DATASET_ID = "d7a04ede97a1bb37b811c3e071f325fa007525a6040927e6793d8cc7c10f538f"
EXPECTED_DATASET_ARTIFACT_DIGEST = "77362e148c713840dda64e0ef70e663cce6611407eac31fefbb9fccca73ae8f8"
EXPECTED_IMPLEMENTATION_DIGEST = "4e74c99ea86f347e024d701396ac58f24eb49affb30a6c283f385de2a839f8f7"
EXPECTED_RUNTIME_DIGEST = "6dbc9cffd844837e17741ae30681f09a6d84b0a49dec011a4fc328f97ade1d85"
EXPECTED_DEFINITION_DIGEST = "8edbba0ad9cdb938b9b10a65e88b940dbbd4c37b1000fe29acadb5d209d52f28"
EXPECTED_REQUESTED_CANDIDATE_DIGEST = "668f5386994efddc14f37c06e4031c86f01ff741c5083a02876b043d91e53dd6"
EXPECTED_BASELINE_MEDIAN_TURNOVER = 858.3114067468092
SEEDS = (0, 1, 2, 3, 4)
SYMBOLS = ("BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT")
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


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tree_manifest(root: Path) -> dict[str, str]:
    if not root.is_dir():
        raise RuntimeError(f"expected directory missing: {root}")
    result: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise RuntimeError(f"symlink forbidden in immutable evidence: {path}")
        if path.is_file() and path.name != ".mutation.lock":
            result[path.relative_to(root).as_posix()] = _sha(path)
    return result


def _assert_immutable_prefix(prereg: Path, result: Path) -> None:
    for relative in (
        Path("bootstrap.json"),
        Path("portable-exp001-prereg-index.json"),
        Path("study/plan.json"),
        Path("study/experiments/0001/definition.json"),
        Path("dataset/manifest.json"),
        Path("dataset/arrays.npz"),
    ):
        left = prereg / relative
        right = result / relative
        if not left.is_file() or not right.is_file() or _sha(left) != _sha(right):
            raise RuntimeError(f"pre-result immutable file changed: {relative}")
    if _tree_manifest(prereg / "study/baseline") != _tree_manifest(
        result / "study/baseline"
    ):
        raise RuntimeError("baseline evidence changed after preregistration")


def _entry(run, symbol_index: int, strategy_name: str) -> tuple[dict[str, object], np.ndarray]:
    by_symbol = run.summary.get("by_symbol")
    if not isinstance(by_symbol, list) or len(by_symbol) != len(SYMBOLS):
        raise RuntimeError("symbol roster malformed")
    symbol_entry = by_symbol[symbol_index]
    if not isinstance(symbol_entry, dict) or symbol_entry.get("symbol") != SYMBOLS[symbol_index]:
        raise RuntimeError("symbol ordering drift")
    strategies = symbol_entry.get("strategies")
    if not isinstance(strategies, list):
        raise RuntimeError("strategy roster malformed")
    matches = [
        item
        for item in strategies
        if isinstance(item, dict) and item.get("name") == strategy_name
    ]
    if len(matches) != 1:
        raise RuntimeError(f"strategy missing or duplicated: {strategy_name}")
    entry = matches[0]
    key = entry.get("return_key")
    if not isinstance(key, str):
        raise RuntimeError("return_key malformed")
    values = run.returns.get(key)
    if values is None or values.ndim != 1 or not np.isfinite(values).all():
        raise RuntimeError("raw return vector malformed")
    return entry, values


def _number(mapping: object, field: str) -> float:
    if not isinstance(mapping, dict):
        raise RuntimeError("metrics malformed")
    value = mapping.get(field)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError(f"metric malformed: {field}")
    result = float(value)
    if not math.isfinite(result):
        raise RuntimeError(f"metric non-finite: {field}")
    return result


def _nonnegative_int(mapping: object, field: str) -> int:
    if not isinstance(mapping, dict):
        raise RuntimeError("metrics malformed")
    value = mapping.get(field)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise RuntimeError(f"metric malformed: {field}")
    return value


def _compound(values: np.ndarray) -> float:
    wealth = 1.0
    for value in values.tolist():
        resolved = float(value)
        if not math.isfinite(resolved) or resolved <= -1.0:
            raise RuntimeError("invalid raw return")
        wealth = wealth * (1.0 + resolved)
    return wealth - 1.0


def _assert_seed_invariant(loaded, strategy: str) -> None:
    for symbol_index, symbol in enumerate(SYMBOLS):
        reference: np.ndarray | None = None
        for seed in SEEDS:
            _, values = _entry(loaded.runs[seed], symbol_index, strategy)
            if reference is None:
                reference = values.copy()
            elif not np.array_equal(reference, values):
                raise RuntimeError(f"seed invariance failed: {symbol}/{strategy}")


def _aggregate(excesses: list[float]) -> dict[str, object]:
    return {
        "positive_symbol_count": sum(value > 0.0 for value in excesses),
        "negative_symbol_count": sum(value < 0.0 for value in excesses),
        "zero_symbol_count": sum(value == 0.0 for value in excesses),
        "median_excess_total_return": float(median(excesses)),
        "worst_excess_total_return": min(excesses),
        "best_excess_total_return": max(excesses),
    }


def verify(prereg_root: Path, result_root: Path, report_root: Path) -> None:
    _assert_immutable_prefix(prereg_root, result_root)

    prereg_snapshot = inspect_study(prereg_root / "study")
    result_snapshot = inspect_study(result_root / "study")
    if prereg_snapshot.plan.digest != EXPECTED_STUDY_DIGEST:
        raise RuntimeError("prereg Study digest drift")
    if result_snapshot.plan.digest != EXPECTED_STUDY_DIGEST:
        raise RuntimeError("result Study digest drift")
    if prereg_snapshot.baseline is None or result_snapshot.baseline is None:
        raise RuntimeError("baseline evidence missing")
    if (
        prereg_snapshot.baseline.fingerprint != EXPECTED_BASELINE_FINGERPRINT
        or result_snapshot.baseline.fingerprint != EXPECTED_BASELINE_FINGERPRINT
    ):
        raise RuntimeError("baseline fingerprint drift")
    if prereg_snapshot.experiment_sequences != (1,) or prereg_snapshot.terminal_sequences:
        raise RuntimeError("prereg lifecycle is not definition-only")
    if result_snapshot.experiment_sequences != (1,) or result_snapshot.terminal_sequences != (1,):
        raise RuntimeError("result lifecycle is not one terminal Experiment")
    if result_snapshot.frozen:
        raise RuntimeError("Study unexpectedly frozen")
    if result_snapshot.plan.dataset_id != EXPECTED_DATASET_ID:
        raise RuntimeError("Dataset ID binding drift")
    if result_snapshot.plan.dataset_artifact_digest != EXPECTED_DATASET_ARTIFACT_DIGEST:
        raise RuntimeError("Dataset artifact binding drift")
    if result_snapshot.plan.implementation_digest != EXPECTED_IMPLEMENTATION_DIGEST:
        raise RuntimeError("implementation provenance drift")
    if result_snapshot.plan.runtime_environment_digest != EXPECTED_RUNTIME_DIGEST:
        raise RuntimeError("runtime provenance drift")

    prereg_state = _reconstruct(StudyStore(prereg_root / "study"))
    result_state = _reconstruct(StudyStore(result_root / "study"))
    prereg_experiment = prereg_state.experiments[0]
    result_experiment = result_state.experiments[0]
    if prereg_experiment.definition.digest != EXPECTED_DEFINITION_DIGEST:
        raise RuntimeError("prereg definition digest drift")
    if result_experiment.definition.digest != EXPECTED_DEFINITION_DIGEST:
        raise RuntimeError("result definition digest drift")
    if prereg_experiment.definition.to_payload() != result_experiment.definition.to_payload():
        raise RuntimeError("Experiment definition changed after preregistration")
    if result_experiment.candidate is None:
        raise RuntimeError("candidate evidence missing")
    if result_experiment.verification is None:
        raise RuntimeError("controlled verification missing")
    if result_experiment.verification.status is not ControlledVerificationStatus.CONTROLLED:
        raise RuntimeError("Experiment verification is not CONTROLLED")
    if result_experiment.verification.changed_paths != (("rule_entry_threshold",),):
        raise RuntimeError("controlled changed path differs from preregistration")
    if result_experiment.comparison is None or result_experiment.decision is None:
        raise RuntimeError("comparison/decision evidence missing")

    prereg_index = json.loads(
        (prereg_root / "portable-exp001-prereg-index.json").read_text(encoding="utf-8")
    )
    if prereg_index.get("requested_candidate_config_digest") != EXPECTED_REQUESTED_CANDIDATE_DIGEST:
        raise RuntimeError("candidate requested digest drift")
    if float(prereg_index.get("frozen_baseline_mean_reversion_median_turnover")) != EXPECTED_BASELINE_MEDIAN_TURNOVER:
        raise RuntimeError("frozen baseline turnover drift")

    baseline = load_evidence_set(result_root / "study/baseline/evidence")
    candidate = load_evidence_set(
        result_root / "study/experiments/0001/candidate/evidence"
    )
    if baseline.evidence.fingerprint != EXPECTED_BASELINE_FINGERPRINT:
        raise RuntimeError("loaded baseline fingerprint drift")
    if tuple(sorted(baseline.runs)) != SEEDS or tuple(sorted(candidate.runs)) != SEEDS:
        raise RuntimeError("Run seed roster drift")

    for strategy in DETERMINISTIC:
        _assert_seed_invariant(baseline, strategy)
        _assert_seed_invariant(candidate, strategy)

    unaffected_checks = 0
    for seed in SEEDS:
        for symbol_index, symbol in enumerate(SYMBOLS):
            for strategy in UNAFFECTED:
                _, left = _entry(baseline.runs[seed], symbol_index, strategy)
                _, right = _entry(candidate.runs[seed], symbol_index, strategy)
                if not np.array_equal(left, right):
                    raise RuntimeError(
                        f"unaffected raw returns changed: seed={seed} symbol={symbol} strategy={strategy}"
                    )
                unaffected_checks += 1

    all_strategy_effects: dict[str, object] = {}
    mean_reversion_candidate_returns: list[float] = []
    mean_reversion_candidate_turnovers: list[float] = []
    mean_reversion_excesses: list[float] = []

    for strategy in DETERMINISTIC:
        by_symbol: dict[str, object] = {}
        excesses: list[float] = []
        for symbol_index, symbol in enumerate(SYMBOLS):
            baseline_entry, baseline_values = _entry(baseline.runs[0], symbol_index, strategy)
            candidate_entry, candidate_values = _entry(candidate.runs[0], symbol_index, strategy)
            baseline_total = _compound(baseline_values)
            candidate_total = _compound(candidate_values)
            if not math.isclose(
                _number(baseline_entry.get("metrics"), "total_return"),
                baseline_total,
                rel_tol=1e-12,
                abs_tol=1e-12,
            ):
                raise RuntimeError("baseline persisted total return mismatch")
            if not math.isclose(
                _number(candidate_entry.get("metrics"), "total_return"),
                candidate_total,
                rel_tol=1e-12,
                abs_tol=1e-12,
            ):
                raise RuntimeError("candidate persisted total return mismatch")
            excess = candidate_total - baseline_total
            excesses.append(excess)
            candidate_turnover = _number(candidate_entry.get("metrics"), "turnover_total")
            by_symbol[symbol] = {
                "baseline_total_return": baseline_total,
                "candidate_total_return": candidate_total,
                "excess_total_return": excess,
                "candidate_turnover": candidate_turnover,
            }
            if strategy == "mean_reversion":
                mean_reversion_candidate_returns.append(candidate_total)
                mean_reversion_candidate_turnovers.append(candidate_turnover)
                mean_reversion_excesses.append(excess)
        all_strategy_effects[strategy] = {
            "by_symbol": by_symbol,
            "aggregate": _aggregate(excesses),
        }

    ppo_zero_effects = 0
    for seed in SEEDS:
        for symbol_index, symbol in enumerate(SYMBOLS):
            baseline_entry, baseline_values = _entry(baseline.runs[seed], symbol_index, "ppo")
            candidate_entry, candidate_values = _entry(candidate.runs[seed], symbol_index, "ppo")
            if not np.array_equal(baseline_values, candidate_values):
                raise RuntimeError(f"PPO changed: seed={seed} symbol={symbol}")
            if _compound(candidate_values) - _compound(baseline_values) != 0.0:
                raise RuntimeError("PPO nonzero effect despite identical raw returns")
            if _number(baseline_entry.get("metrics"), "total_return") != _number(
                candidate_entry.get("metrics"), "total_return"
            ):
                raise RuntimeError("PPO persisted total return changed")
            ppo_zero_effects += 1

    if len(mean_reversion_excesses) != 5:
        raise RuntimeError("mean-reversion symbol evidence incomplete")
    positive_effects = sum(value > 0.0 for value in mean_reversion_excesses)
    median_excess = float(median(mean_reversion_excesses))
    positive_candidate_returns = sum(value > 0.0 for value in mean_reversion_candidate_returns)
    candidate_median_turnover = float(median(mean_reversion_candidate_turnovers))
    accept = (
        positive_effects >= 4
        and median_excess > 0.0
        and positive_candidate_returns == 5
        and candidate_median_turnover < EXPECTED_BASELINE_MEDIAN_TURNOVER
    )
    keep = (
        positive_effects <= 2
        or median_excess <= 0.0
        or positive_candidate_returns <= 3
    )
    expected_decision = (
        ExperimentDecisionKind.ACCEPT_CANDIDATE
        if accept
        else ExperimentDecisionKind.KEEP_BASELINE
        if keep
        else ExperimentDecisionKind.INCONCLUSIVE
    )
    if result_experiment.decision.decision is not expected_decision:
        raise RuntimeError(
            "persisted decision differs from independent frozen-rule replay: "
            f"expected={expected_decision.value} observed={result_experiment.decision.decision.value}"
        )

    factor_effect = result_experiment.comparison.factor_effect
    cross_symbol = factor_effect.get("cross_symbol")
    if not isinstance(cross_symbol, dict):
        raise RuntimeError("persisted comparison malformed")
    persisted_mr = cross_symbol.get("mean_reversion")
    if not isinstance(persisted_mr, dict):
        raise RuntimeError("persisted mean-reversion summary malformed")
    if persisted_mr.get("positive_symbol_count") != positive_effects:
        raise RuntimeError("persisted positive symbol count differs from raw oracle")
    persisted_median = persisted_mr.get("median_excess_total_return")
    if not isinstance(persisted_median, (int, float)) or isinstance(persisted_median, bool):
        raise RuntimeError("persisted median excess malformed")
    if not math.isclose(float(persisted_median), median_excess, rel_tol=1e-12, abs_tol=1e-12):
        raise RuntimeError("persisted mean-reversion median differs from raw oracle")

    aggregate_cost = 0.0
    trading_observations = 0
    positive_cost_observations = 0
    cash_observations = 0
    for seed in SEEDS:
        run = candidate.runs[seed]
        for symbol_index, symbol in enumerate(SYMBOLS):
            for strategy in STRATEGIES:
                entry, values = _entry(run, symbol_index, strategy)
                metrics = entry.get("metrics")
                cost = _number(metrics, "total_cost")
                trades = _nonnegative_int(metrics, "n_trades")
                aggregate_cost += cost
                if trades > 0:
                    trading_observations += 1
                    if cost <= 0.0:
                        raise RuntimeError(
                            f"non-positive realized cost: seed={seed} symbol={symbol} strategy={strategy}"
                        )
                    positive_cost_observations += 1
                if strategy == "cash":
                    cash_observations += 1
                    if trades != 0 or cost != 0.0 or not np.array_equal(
                        values, np.zeros_like(values)
                    ):
                        raise RuntimeError("cash semantics drift")
    if trading_observations == 0 or positive_cost_observations != trading_observations:
        raise RuntimeError("candidate trading cost oracle failed")
    if cash_observations != 25:
        raise RuntimeError("cash observation count drift")

    result_index_path = result_root / "portable-exp001-result-index.json"
    if not result_index_path.is_file():
        raise RuntimeError("result index missing")
    result_index = json.loads(result_index_path.read_text(encoding="utf-8"))
    index_expected = {
        "schema_version": "canonical_m2_portable_exp001_result_index_v1",
        "issue_number": 511,
        "study_digest": EXPECTED_STUDY_DIGEST,
        "baseline_evidence_fingerprint": EXPECTED_BASELINE_FINGERPRINT,
        "candidate_evidence_fingerprint": candidate.evidence.fingerprint,
        "dataset_id": EXPECTED_DATASET_ID,
        "dataset_artifact_digest": EXPECTED_DATASET_ARTIFACT_DIGEST,
        "implementation_digest": EXPECTED_IMPLEMENTATION_DIGEST,
        "runtime_environment_digest": EXPECTED_RUNTIME_DIGEST,
        "definition_digest": EXPECTED_DEFINITION_DIGEST,
        "verification_digest": result_experiment.verification.digest,
        "verification_status": "CONTROLLED",
        "comparison_digest": result_experiment.comparison.digest,
        "decision_digest": result_experiment.decision.digest,
        "decision": expected_decision.value,
        "study_frozen": False,
        "profitability_claimed": False,
        "winner_claimed": False,
        "final_test_authorized": False,
        "production_authorized": False,
    }
    for key, expected in index_expected.items():
        if result_index.get(key) != expected:
            raise RuntimeError(f"result index differs from independently rebuilt evidence: {key}")

    report_root.mkdir(parents=True, exist_ok=True)
    report = {
        "schema_version": "canonical_m2_portable_exp001_postverify_v1",
        "study_digest": EXPECTED_STUDY_DIGEST,
        "definition_digest": EXPECTED_DEFINITION_DIGEST,
        "baseline_evidence_fingerprint": EXPECTED_BASELINE_FINGERPRINT,
        "candidate_evidence_fingerprint": candidate.evidence.fingerprint,
        "verification_digest": result_experiment.verification.digest,
        "comparison_digest": result_experiment.comparison.digest,
        "decision_digest": result_experiment.decision.digest,
        "decision": expected_decision.value,
        "mean_reversion": {
            "positive_factor_effect_symbol_count": positive_effects,
            "median_excess_total_return": median_excess,
            "candidate_positive_total_return_symbol_count": positive_candidate_returns,
            "candidate_median_turnover": candidate_median_turnover,
            "frozen_baseline_median_turnover": EXPECTED_BASELINE_MEDIAN_TURNOVER,
        },
        "all_deterministic_strategy_effects": all_strategy_effects,
        "ppo_exact_zero_effect_seed_symbol_checks": ppo_zero_effects,
        "unaffected_raw_return_equality_checks": unaffected_checks,
        "candidate_cost_semantics": {
            "aggregate_total_cost_across_seed_runs": aggregate_cost,
            "trading_observations": trading_observations,
            "positive_cost_observations": positive_cost_observations,
            "cash_observations": cash_observations,
        },
        "ppo_cross_symbol_metrics_used_for_decision": False,
        "preregistration_immutable_prefix_verified": True,
        "profitability_proven": False,
        "winner_selected": False,
        "final_test_authorized": False,
        "production_authorized": False,
        "verification_passed": True,
    }
    (report_root / "postverify.json").write_text(
        json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print("PORTABLE_EXP001_POSTVERIFY=" + json.dumps(report, sort_keys=True, allow_nan=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prereg-root", type=Path, required=True)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--report-root", type=Path, required=True)
    args = parser.parse_args()
    verify(args.prereg_root, args.result_root, args.report_root)


if __name__ == "__main__":
    main()
