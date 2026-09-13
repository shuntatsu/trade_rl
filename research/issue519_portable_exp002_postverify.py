"""Independent post-artifact verifier for portable Experiment 0002."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import median

import numpy as np

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
EXPECTED_EXP002_DEFINITION_DIGEST = (
    "854814c85f3e0175c83cc7f3cd342cefa627780bb9d88374d948a7ec167fca87"
)
EXPECTED_REQUESTED_CANDIDATE_DIGEST = (
    "a966dd408f43c64c1833a32fa0507a8e6b7d7fa865fb88c73cd0c1d53a05fcd4"
)
EXPECTED_BASELINE_MEDIAN_TURNOVER = 858.3114067468092
EXPECTED_SEEDS = (0, 1, 2, 3, 4)
EXPECTED_SYMBOLS = ("BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT")
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
    prereg = prereg_root / relative
    result = result_root / relative
    if not prereg.is_file() or not result.is_file():
        raise RuntimeError(f"immutable file is missing: {relative}")
    if _file_sha256(prereg) != _file_sha256(result):
        raise RuntimeError(f"immutable preregistration prefix changed: {relative}")


def _assert_same_tree(
    prereg_root: Path,
    result_root: Path,
    relative: str,
    *,
    label: str,
) -> None:
    prereg = _tree_manifest(prereg_root / relative)
    result = _tree_manifest(result_root / relative)
    if prereg != result:
        raise RuntimeError(f"{label} immutable evidence changed after preregistration")


def _assert_immutable_prefix(prereg_root: Path, result_root: Path) -> None:
    """Require every pre-candidate authority/evidence byte to remain unchanged."""

    for relative in (
        "bootstrap.json",
        "portable-exp002-prereg-index.json",
        "study/plan.json",
        "study/experiments/0002/definition.json",
        "dataset/manifest.json",
        "dataset/arrays.npz",
    ):
        _assert_same_file(prereg_root, result_root, relative)
    _assert_same_tree(
        prereg_root,
        result_root,
        "study/baseline",
        label="baseline",
    )
    _assert_same_tree(
        prereg_root,
        result_root,
        "study/experiments/0001",
        label="Experiment 0001",
    )


def _independent_decision(
    *,
    positive_effects: int,
    median_excess: float,
    candidate_positive_returns: int,
    candidate_median_turnover: float,
) -> ExperimentDecisionKind:
    """Independent copy of the result-blind formal decision rule."""

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


def _metric(entry: object, name: str) -> float:
    if not isinstance(entry, dict):
        raise RuntimeError("strategy entry is malformed")
    metrics = entry.get("metrics")
    if not isinstance(metrics, dict):
        raise RuntimeError("strategy metrics are malformed")
    value = metrics.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError(f"strategy metric is malformed: {name}")
    resolved = float(value)
    if not math.isfinite(resolved):
        raise RuntimeError(f"strategy metric is non-finite: {name}")
    return resolved


def _metric_int(entry: object, name: str) -> int:
    if not isinstance(entry, dict):
        raise RuntimeError("strategy entry is malformed")
    metrics = entry.get("metrics")
    if not isinstance(metrics, dict):
        raise RuntimeError("strategy metrics are malformed")
    value = metrics.get(name)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise RuntimeError(f"strategy integer metric is malformed: {name}")
    return value


def _entry(run: object, symbol_index: int, strategy: str) -> dict[str, object]:
    summary = getattr(run, "summary", None)
    if not isinstance(summary, dict):
        raise RuntimeError("run summary is malformed")
    by_symbol = summary.get("by_symbol")
    if not isinstance(by_symbol, list) or len(by_symbol) != len(EXPECTED_SYMBOLS):
        raise RuntimeError("run symbol roster is malformed")
    symbol = by_symbol[symbol_index]
    if not isinstance(symbol, dict) or symbol.get("symbol") != EXPECTED_SYMBOLS[symbol_index]:
        raise RuntimeError("run symbol ordering drift")
    strategies = symbol.get("strategies")
    if not isinstance(strategies, list):
        raise RuntimeError("run strategy roster is malformed")
    matches = [
        item
        for item in strategies
        if isinstance(item, dict) and item.get("name") == strategy
    ]
    if len(matches) != 1:
        raise RuntimeError(f"strategy entry missing or duplicated: {strategy}")
    return matches[0]


def _entry_values(run: object, symbol_index: int, strategy: str) -> tuple[dict[str, object], np.ndarray]:
    entry = _entry(run, symbol_index, strategy)
    key = entry.get("return_key")
    returns = getattr(run, "returns", None)
    if not isinstance(key, str) or not isinstance(returns, dict) or key not in returns:
        raise RuntimeError("strategy raw-return evidence is missing")
    values = np.asarray(returns[key], dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise RuntimeError("strategy raw returns are malformed")
    total = _compound(values)
    if not math.isclose(total, _metric(entry, "total_return"), rel_tol=1e-12, abs_tol=1e-12):
        raise RuntimeError("persisted total return differs from raw returns")
    return entry, values


def _validate_deterministic_seed_invariance(loaded: object) -> int:
    runs = getattr(loaded, "runs", None)
    if not isinstance(runs, dict) or tuple(sorted(runs)) != EXPECTED_SEEDS:
        raise RuntimeError("EvidenceSet seed roster drift")
    checks = 0
    references: dict[tuple[str, str], np.ndarray] = {}
    for seed in EXPECTED_SEEDS:
        for symbol_index, symbol in enumerate(EXPECTED_SYMBOLS):
            for strategy in DETERMINISTIC:
                _, values = _entry_values(runs[seed], symbol_index, strategy)
                key = (symbol, strategy)
                if seed == EXPECTED_SEEDS[0]:
                    references[key] = values.copy()
                else:
                    if not np.array_equal(references[key], values):
                        raise RuntimeError(
                            f"deterministic raw returns are seed-sensitive: {symbol}/{strategy}"
                        )
                    checks += 1
    return checks


def _validate_study_bindings(prereg_root: Path, result_root: Path) -> tuple[object, object]:
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

    prereg = inspect_study(prereg_root / "study")
    result = inspect_study(result_root / "study")
    for snapshot, label in ((prereg, "preregistration"), (result, "result")):
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
    if prereg.experiment_sequences != (1, 2) or prereg.terminal_sequences != (1,):
        raise RuntimeError("preregistration lifecycle is not result-blind")
    if result.experiment_sequences != (1, 2) or result.terminal_sequences != (1, 2):
        raise RuntimeError("result lifecycle is incomplete")
    return prereg, result


def _validate_state(result_root: Path) -> tuple[object, object, object, object]:
    state = _reconstruct(StudyStore(result_root / "study"))
    if len(state.experiments) != 2:
        raise RuntimeError("result does not contain exactly two Experiments")
    exp1, exp2 = state.experiments
    exp1_binding = (
        exp1.definition.digest,
        exp1.candidate.evidence.fingerprint if exp1.candidate else None,
        exp1.verification.digest if exp1.verification else None,
        exp1.comparison.digest if exp1.comparison else None,
        exp1.decision.digest if exp1.decision else None,
    )
    expected_exp1 = (
        EXPECTED_EXP001_DEFINITION_DIGEST,
        EXPECTED_EXP001_CANDIDATE_FINGERPRINT,
        EXPECTED_EXP001_VERIFICATION_DIGEST,
        EXPECTED_EXP001_COMPARISON_DIGEST,
        EXPECTED_EXP001_DECISION_DIGEST,
    )
    if exp1_binding != expected_exp1:
        raise RuntimeError("Experiment 0001 binding drift")
    if exp2.definition.digest != EXPECTED_EXP002_DEFINITION_DIGEST:
        raise RuntimeError("Experiment 0002 definition digest drift")
    if exp2.definition.candidate_requested_config_digest != EXPECTED_REQUESTED_CANDIDATE_DIGEST:
        raise RuntimeError("Experiment 0002 requested config digest drift")
    if exp2.candidate is None or exp2.verification is None or exp2.comparison is None or exp2.decision is None:
        raise RuntimeError("Experiment 0002 terminal evidence is incomplete")
    if exp2.failure is not None:
        raise RuntimeError("Experiment 0002 unexpectedly contains failure evidence")
    if exp2.verification.status is not ControlledVerificationStatus.CONTROLLED:
        raise RuntimeError("Experiment 0002 is not CONTROLLED")
    if exp2.verification.changed_paths != EXPECTED_CHANGED_PATHS:
        raise RuntimeError("Experiment 0002 changed paths drift")
    return state, exp1, exp2, exp2.decision


def _independent_raw_analysis(result_root: Path) -> dict[str, object]:
    baseline = load_evidence_set(result_root / "study/baseline/evidence")
    candidate = load_evidence_set(result_root / "study/experiments/0002/candidate/evidence")
    if baseline.evidence.fingerprint != EXPECTED_BASELINE_FINGERPRINT:
        raise RuntimeError("baseline EvidenceSet fingerprint drift")
    if baseline.evidence.ppo_seeds != EXPECTED_SEEDS or candidate.evidence.ppo_seeds != EXPECTED_SEEDS:
        raise RuntimeError("PPO seed policy drift")
    _validate_deterministic_seed_invariance(baseline)
    _validate_deterministic_seed_invariance(candidate)

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
            before_total = _compound(before)
            after_total = _compound(after)
            excess = after_total - before_total
            excesses.append(excess)
            by_symbol[symbol] = {
                "baseline_total_return": before_total,
                "candidate_total_return": after_total,
                "excess_total_return": excess,
                "baseline_turnover": _metric(before_entry, "turnover_total"),
                "candidate_turnover": _metric(after_entry, "turnover_total"),
            }
        deterministic_effects[strategy] = {
            "by_symbol": by_symbol,
            "aggregate": {
                "positive_symbol_count": sum(value > 0.0 for value in excesses),
                "negative_symbol_count": sum(value < 0.0 for value in excesses),
                "zero_symbol_count": sum(value == 0.0 for value in excesses),
                "median_excess_total_return": float(median(excesses)),
                "worst_excess_total_return": min(excesses),
                "best_excess_total_return": max(excesses),
            },
        }

    ppo_checks = 0
    for seed in EXPECTED_SEEDS:
        for symbol_index, symbol in enumerate(EXPECTED_SYMBOLS):
            _, before = _entry_values(baseline.runs[seed], symbol_index, "ppo")
            _, after = _entry_values(candidate.runs[seed], symbol_index, "ppo")
            if not np.array_equal(before, after):
                raise RuntimeError(f"PPO changed under RULE_SIGNAL: seed={seed} symbol={symbol}")
            ppo_checks += 1

    trading = positive_cost = cash = 0
    aggregate_cost = 0.0
    for seed in EXPECTED_SEEDS:
        run = candidate.runs[seed]
        for symbol_index, symbol in enumerate(EXPECTED_SYMBOLS):
            for strategy in STRATEGIES:
                entry, values = _entry_values(run, symbol_index, strategy)
                total_cost = _metric(entry, "total_cost")
                n_trades = _metric_int(entry, "n_trades")
                diagnostics = entry.get("diagnostics")
                if isinstance(diagnostics, dict):
                    if diagnostics.get("total_cost") != entry["metrics"]["total_cost"]:  # type: ignore[index]
                        raise RuntimeError("metric/diagnostic total_cost mismatch")
                    if diagnostics.get("n_trades") != entry["metrics"]["n_trades"]:  # type: ignore[index]
                        raise RuntimeError("metric/diagnostic n_trades mismatch")
                aggregate_cost += total_cost
                if n_trades > 0:
                    trading += 1
                    if total_cost <= 0.0:
                        raise RuntimeError(
                            f"trading observation has non-positive cost: seed={seed} symbol={symbol} strategy={strategy}"
                        )
                    positive_cost += 1
                if strategy == "cash":
                    cash += 1
                    if n_trades != 0 or total_cost != 0.0 or not np.array_equal(values, np.zeros_like(values)):
                        raise RuntimeError("cash semantics drift")
    if trading == 0 or positive_cost != trading:
        raise RuntimeError("candidate positive-cost trading oracle failed")
    if cash != len(EXPECTED_SEEDS) * len(EXPECTED_SYMBOLS):
        raise RuntimeError("candidate cash oracle roster incomplete")

    mr = deterministic_effects["mean_reversion"]
    by_symbol = mr["by_symbol"]  # type: ignore[index]
    aggregate = mr["aggregate"]  # type: ignore[index]
    candidate_totals = [
        float(by_symbol[symbol]["candidate_total_return"])  # type: ignore[index]
        for symbol in EXPECTED_SYMBOLS
    ]
    candidate_turnovers = [
        float(by_symbol[symbol]["candidate_turnover"])  # type: ignore[index]
        for symbol in EXPECTED_SYMBOLS
    ]
    positive_effects = int(aggregate["positive_symbol_count"])  # type: ignore[index]
    median_excess = float(aggregate["median_excess_total_return"])  # type: ignore[index]
    positive_returns = sum(value > 0.0 for value in candidate_totals)
    median_turnover = float(median(candidate_turnovers))
    decision = _independent_decision(
        positive_effects=positive_effects,
        median_excess=median_excess,
        candidate_positive_returns=positive_returns,
        candidate_median_turnover=median_turnover,
    )
    return {
        "baseline_evidence_fingerprint": baseline.evidence.fingerprint,
        "candidate_evidence_fingerprint": candidate.evidence.fingerprint,
        "unaffected_raw_return_equality_checks": unaffected_checks,
        "ppo_exact_zero_effect_checks": ppo_checks,
        "deterministic_effects": deterministic_effects,
        "mean_reversion_formal_inputs": {
            "positive_factor_effect_symbol_count": positive_effects,
            "median_excess_total_return": median_excess,
            "candidate_positive_total_return_symbol_count": positive_returns,
            "candidate_median_turnover": median_turnover,
            "frozen_baseline_median_turnover": EXPECTED_BASELINE_MEDIAN_TURNOVER,
        },
        "candidate_cost_semantics": {
            "aggregate_total_cost_across_seed_runs": aggregate_cost,
            "trading_observations": trading,
            "positive_cost_observations": positive_cost,
            "cash_observations": cash,
        },
        "formal_decision": decision.value,
    }


def _validate_result_index(result_root: Path, raw: dict[str, object], decision: object) -> dict[str, object]:
    path = result_root / "portable-exp002-result-index.json"
    if not path.is_file():
        raise RuntimeError("Experiment 0002 result index missing")
    index = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "schema_version": "canonical_m2_portable_exp002_result_index_v1",
        "issue_number": 519,
        "study_digest": EXPECTED_STUDY_DIGEST,
        "baseline_evidence_fingerprint": EXPECTED_BASELINE_FINGERPRINT,
        "dataset_id": EXPECTED_DATASET_ID,
        "dataset_artifact_digest": EXPECTED_DATASET_ARTIFACT_DIGEST,
        "implementation_digest": EXPECTED_IMPLEMENTATION_DIGEST,
        "runtime_environment_digest": EXPECTED_RUNTIME_DIGEST,
        "definition_digest": EXPECTED_EXP002_DEFINITION_DIGEST,
        "verification_status": "CONTROLLED",
        "changed_paths": [["signal_index"], ["signal_name"]],
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
            raise RuntimeError(f"Experiment 0002 result index drift: {key}")
    candidate_fp = raw.get("candidate_evidence_fingerprint")
    if index.get("candidate_evidence_fingerprint") != candidate_fp:
        raise RuntimeError("candidate fingerprint differs from independent raw evidence")
    stored_raw = index.get("independent_raw_evidence")
    if stored_raw != raw:
        raise RuntimeError("persisted independent raw evidence differs from fresh oracle")
    persisted_decision = getattr(decision, "decision", None)
    if not isinstance(persisted_decision, ExperimentDecisionKind):
        raise RuntimeError("persisted decision is malformed")
    if raw.get("formal_decision") != persisted_decision.value:
        raise RuntimeError("fresh decision replay differs from persisted decision")
    if index.get("decision") != persisted_decision.value:
        raise RuntimeError("result index decision differs from persisted decision")
    if index.get("decision_digest") != getattr(decision, "digest", None):
        raise RuntimeError("result index decision digest drift")
    return index


def verify(prereg_root: Path, result_root: Path, report_root: Path) -> dict[str, object]:
    _assert_immutable_prefix(prereg_root, result_root)
    _validate_study_bindings(prereg_root, result_root)
    _, _, exp2, decision = _validate_state(result_root)
    raw = _independent_raw_analysis(result_root)
    if exp2.candidate.evidence.fingerprint != raw["candidate_evidence_fingerprint"]:
        raise RuntimeError("candidate fingerprint mismatch after state reconstruction")
    if exp2.verification.digest is None or exp2.comparison.digest is None:
        raise RuntimeError("Experiment 0002 binding evidence is incomplete")
    result_index = _validate_result_index(result_root, raw, decision)
    report: dict[str, object] = {
        "schema_version": "canonical_m2_portable_exp002_postverify_v1",
        "study_digest": EXPECTED_STUDY_DIGEST,
        "dataset_id": EXPECTED_DATASET_ID,
        "baseline_evidence_fingerprint": EXPECTED_BASELINE_FINGERPRINT,
        "candidate_evidence_fingerprint": raw["candidate_evidence_fingerprint"],
        "definition_digest": EXPECTED_EXP002_DEFINITION_DIGEST,
        "verification_digest": exp2.verification.digest,
        "comparison_digest": exp2.comparison.digest,
        "decision_digest": decision.digest,
        "independent_raw_evidence": raw,
        "result_index_execution_run_id": result_index.get("execution_workflow_run_id"),
        "immutable_preregistration_prefix_verified": True,
        "prior_exp001_evidence_unchanged": True,
        "controlled_delta_verified": True,
        "independent_decision_replayed": True,
        "candidate_reexecuted": False,
        "interpretation_authorized": False,
        "verification_passed": True,
    }
    report_root.mkdir(parents=True, exist_ok=True)
    (report_root / "postverify.json").write_text(
        json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print("EXP002_POSTVERIFY=" + json.dumps(report, sort_keys=True, allow_nan=False))
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


__all__ = ["_assert_immutable_prefix", "_independent_decision", "verify"]
