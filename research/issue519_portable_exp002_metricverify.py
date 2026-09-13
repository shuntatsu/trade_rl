"""Independent deterministic metric/seed invariance verifier for Experiment 0002."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from trade_rl.evaluation.experiments import load_evidence_set

FIELDS = ("total_return", "max_drawdown", "turnover_total", "total_cost")
EXPECTED_SEEDS = (0, 1, 2, 3, 4)
EXPECTED_SYMBOLS = ("BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT")
EXPECTED_STRATEGIES = (
    "cash",
    "constant_long",
    "constant_short",
    "trend",
    "mean_reversion",
    "ridge24",
    "lightgbm24",
)


def _metric(entry: object, field: str) -> float:
    if not isinstance(entry, dict):
        raise RuntimeError("strategy entry is malformed")
    metrics = entry.get("metrics")
    if not isinstance(metrics, dict):
        raise RuntimeError("strategy metrics are malformed")
    value = metrics.get(field)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError(f"metric is malformed: {field}")
    resolved = float(value)
    if not math.isfinite(resolved):
        raise RuntimeError(f"metric is non-finite: {field}")
    return resolved


def _strategy_entry(run: object, symbol: str, strategy: str) -> dict[str, object]:
    summary = getattr(run, "summary", None)
    if not isinstance(summary, dict):
        raise RuntimeError("run summary is malformed")
    by_symbol = summary.get("by_symbol")
    if not isinstance(by_symbol, list):
        raise RuntimeError("run symbol evidence is malformed")
    matches = [item for item in by_symbol if isinstance(item, dict) and item.get("symbol") == symbol]
    if len(matches) != 1:
        raise RuntimeError(f"symbol evidence missing or duplicated: {symbol}")
    strategies = matches[0].get("strategies")
    if not isinstance(strategies, list):
        raise RuntimeError("run strategy evidence is malformed")
    found = [
        item
        for item in strategies
        if isinstance(item, dict) and item.get("name") == strategy
    ]
    if len(found) != 1:
        raise RuntimeError(f"strategy evidence missing or duplicated: {symbol}/{strategy}")
    return found[0]


def assert_metric_seed_invariance(
    loaded: object,
    *,
    seeds: tuple[int, ...] = EXPECTED_SEEDS,
    symbols: tuple[str, ...] = EXPECTED_SYMBOLS,
    strategies: tuple[str, ...] = EXPECTED_STRATEGIES,
) -> int:
    """Require deterministic strategy metrics to be exact across frozen PPO seeds."""

    runs = getattr(loaded, "runs", None)
    if not isinstance(runs, dict) or tuple(sorted(runs)) != tuple(sorted(seeds)):
        raise RuntimeError("EvidenceSet Run seed roster differs from expected seeds")
    reference_seed = seeds[0]
    checks = 0
    for symbol in symbols:
        for strategy in strategies:
            reference = _strategy_entry(runs[reference_seed], symbol, strategy)
            for seed in seeds[1:]:
                observed = _strategy_entry(runs[seed], symbol, strategy)
                for field in FIELDS:
                    expected_value = _metric(reference, field)
                    observed_value = _metric(observed, field)
                    if observed_value != expected_value:
                        raise RuntimeError(
                            "deterministic metric seed invariance failed: "
                            f"seed={seed} symbol={symbol} strategy={strategy} "
                            f"field={field} expected={expected_value!r} "
                            f"observed={observed_value!r}"
                        )
                    checks += 1
    return checks


def build_metric_invariance_report(baseline: object, candidate: object) -> dict[str, object]:
    baseline_checks = assert_metric_seed_invariance(baseline)
    candidate_checks = assert_metric_seed_invariance(candidate)
    return {
        "schema_version": "issue519_exp002_metric_seed_invariance_verification_v1",
        "verified": True,
        "fields": list(FIELDS),
        "seeds": list(EXPECTED_SEEDS),
        "symbols": list(EXPECTED_SYMBOLS),
        "strategies": list(EXPECTED_STRATEGIES),
        "baseline_checks": baseline_checks,
        "candidate_checks": candidate_checks,
        "total_checks": baseline_checks + candidate_checks,
    }


def write_metric_invariance_report(
    baseline: object,
    candidate: object,
    *,
    report_root: str | Path,
) -> tuple[Path, str]:
    report = build_metric_invariance_report(baseline, candidate)
    root = Path(report_root)
    root.mkdir(parents=True, exist_ok=True)
    path = root / "metric-seed-invariance-verification.json"
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    path.write_text(text, encoding="utf-8")
    return path, hashlib.sha256(text.encode("utf-8")).hexdigest()


def verify_result(result_root: str | Path, report_root: str | Path) -> tuple[Path, str]:
    root = Path(result_root)
    baseline = load_evidence_set(root / "study/baseline/evidence")
    candidate = load_evidence_set(root / "study/experiments/0002/candidate/evidence")
    return write_metric_invariance_report(
        baseline,
        candidate,
        report_root=report_root,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--report-root", type=Path, required=True)
    args = parser.parse_args()
    path, digest = verify_result(args.result_root, args.report_root)
    print(
        "EXP002_METRIC_INVARIANCE="
        + json.dumps(
            {"report": str(path), "sha256": digest, "verified": True},
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()


__all__ = [
    "assert_metric_seed_invariance",
    "build_metric_invariance_report",
    "verify_result",
    "write_metric_invariance_report",
]
