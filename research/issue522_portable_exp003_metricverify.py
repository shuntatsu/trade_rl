"""Independent deterministic metric seed-invariance verifier for Experiment 0003."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from research.issue522_portable_exp003 import EXPECTED_SEEDS, EXPECTED_SYMBOLS
from trade_rl.evaluation.experiments import load_evidence_set

FIELDS = ("total_return", "max_drawdown", "turnover_total", "total_cost")
STRATEGIES = (
    "cash",
    "constant_long",
    "constant_short",
    "trend",
    "mean_reversion",
    "ridge24",
    "lightgbm24",
)


def _metric(run: object, symbol: str, strategy: str, field: str) -> float:
    summary = getattr(run, "summary", None)
    if not isinstance(summary, dict):
        raise RuntimeError("run summary malformed")
    by_symbol = summary.get("by_symbol")
    if not isinstance(by_symbol, list):
        raise RuntimeError("run symbol roster malformed")
    symbol_matches = [
        entry for entry in by_symbol if isinstance(entry, dict) and entry.get("symbol") == symbol
    ]
    if len(symbol_matches) != 1:
        raise RuntimeError(f"symbol entry missing or duplicated: {symbol}")
    entries = symbol_matches[0].get("strategies")
    if not isinstance(entries, list):
        raise RuntimeError("strategy roster malformed")
    matches = [
        entry for entry in entries if isinstance(entry, dict) and entry.get("name") == strategy
    ]
    if len(matches) != 1:
        raise RuntimeError(f"strategy entry missing or duplicated: {strategy}")
    metrics = matches[0].get("metrics")
    if not isinstance(metrics, dict):
        raise RuntimeError("strategy metrics malformed")
    value = metrics.get(field)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError(f"{field} malformed")
    result = float(value)
    if not math.isfinite(result):
        raise RuntimeError(f"{field} non-finite")
    return result


def assert_metric_seed_invariance(
    loaded: object,
    *,
    seeds: tuple[int, ...],
    symbols: tuple[str, ...],
    strategies: tuple[str, ...],
) -> int:
    runs = getattr(loaded, "runs", None)
    if not isinstance(runs, dict) or tuple(sorted(runs)) != tuple(sorted(seeds)):
        raise RuntimeError("metric verifier seed roster drift")
    reference_seed = seeds[0]
    checks = 0
    for seed in seeds[1:]:
        for symbol in symbols:
            for strategy in strategies:
                for field in FIELDS:
                    reference = _metric(runs[reference_seed], symbol, strategy, field)
                    actual = _metric(runs[seed], symbol, strategy, field)
                    if actual != reference:
                        raise RuntimeError(
                            f"seed-sensitive metric {field}: seed={seed} symbol={symbol} strategy={strategy}"
                        )
                    checks += 1
    return checks


def build_metric_invariance_report(baseline: object, candidate: object) -> dict[str, object]:
    baseline_checks = assert_metric_seed_invariance(
        baseline,
        seeds=EXPECTED_SEEDS,
        symbols=EXPECTED_SYMBOLS,
        strategies=STRATEGIES,
    )
    candidate_checks = assert_metric_seed_invariance(
        candidate,
        seeds=EXPECTED_SEEDS,
        symbols=EXPECTED_SYMBOLS,
        strategies=STRATEGIES,
    )
    expected = (len(EXPECTED_SEEDS) - 1) * len(EXPECTED_SYMBOLS) * len(STRATEGIES) * len(FIELDS)
    if baseline_checks != expected or candidate_checks != expected:
        raise RuntimeError("metric verifier check count drift")
    return {
        "schema_version": "issue522_exp003_metric_seed_invariance_verification_v1",
        "verified": True,
        "fields": list(FIELDS),
        "seeds": list(EXPECTED_SEEDS),
        "symbols": list(EXPECTED_SYMBOLS),
        "strategies": list(STRATEGIES),
        "baseline_checks": baseline_checks,
        "candidate_checks": candidate_checks,
        "total_checks": baseline_checks + candidate_checks,
    }


def write_metric_invariance_report(
    baseline: object,
    candidate: object,
    *,
    report_root: Path,
) -> tuple[Path, str]:
    report = build_metric_invariance_report(baseline, candidate)
    report_root.mkdir(parents=True, exist_ok=True)
    path = report_root / "metric-seed-invariance-verification.json"
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    path.write_text(text, encoding="utf-8")
    return path, hashlib.sha256(text.encode("utf-8")).hexdigest()


def verify(result_root: Path, report_root: Path) -> dict[str, object]:
    baseline = load_evidence_set(result_root / "study/baseline/evidence")
    candidate = load_evidence_set(result_root / "study/experiments/0003/candidate/evidence")
    report = build_metric_invariance_report(baseline, candidate)
    write_metric_invariance_report(baseline, candidate, report_root=report_root)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", required=True, type=Path)
    parser.add_argument("--report-root", required=True, type=Path)
    args = parser.parse_args()
    verify(args.result_root, args.report_root)


if __name__ == "__main__":
    main()


__all__ = [
    "assert_metric_seed_invariance",
    "build_metric_invariance_report",
    "verify",
    "write_metric_invariance_report",
]
