import hashlib
import json
from importlib import import_module
from importlib.util import find_spec
from types import SimpleNamespace

import pytest

MODULE = "research.issue519_portable_exp002_metricverify"
FIELDS = ("total_return", "max_drawdown", "turnover_total", "total_cost")
FROZEN_SEEDS = (0, 1, 2, 3, 4)
FROZEN_SYMBOLS = ("BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT")
FROZEN_STRATEGIES = (
    "cash",
    "constant_long",
    "constant_short",
    "trend",
    "mean_reversion",
    "ridge24",
    "lightgbm24",
)


def _module():
    spec = find_spec(MODULE)
    assert spec is not None, "Experiment 0002 metric invariance verifier is not implemented"
    return import_module(MODULE)


def _loaded(
    *,
    seeds: tuple[int, ...] = (0, 1),
    symbols: tuple[str, ...] = ("BTCUSDT",),
    strategies: tuple[str, ...] = ("trend", "mean_reversion"),
    changed_field: str | None = None,
):
    runs = {}
    for seed in seeds:
        by_symbol = []
        for symbol in symbols:
            strategy_entries = []
            for strategy in strategies:
                metrics = {
                    "total_return": 0.1,
                    "max_drawdown": -0.2,
                    "turnover_total": 3.0,
                    "total_cost": 4.0,
                }
                if (
                    seed == seeds[-1]
                    and symbol == symbols[-1]
                    and strategy == strategies[-1]
                    and changed_field is not None
                ):
                    metrics[changed_field] += 0.5
                strategy_entries.append({"name": strategy, "metrics": metrics})
            by_symbol.append({"symbol": symbol, "strategies": strategy_entries})
        runs[seed] = SimpleNamespace(summary={"by_symbol": by_symbol})
    return SimpleNamespace(runs=runs)


def _full_pair():
    baseline = _loaded(
        seeds=FROZEN_SEEDS,
        symbols=FROZEN_SYMBOLS,
        strategies=FROZEN_STRATEGIES,
    )
    candidate = _loaded(
        seeds=FROZEN_SEEDS,
        symbols=FROZEN_SYMBOLS,
        strategies=FROZEN_STRATEGIES,
    )
    return baseline, candidate


def test_metric_seed_invariance_accepts_all_required_metrics() -> None:
    module = _module()
    checks = module.assert_metric_seed_invariance(
        _loaded(),
        seeds=(0, 1),
        symbols=("BTCUSDT",),
        strategies=("trend", "mean_reversion"),
    )
    assert checks == 8


@pytest.mark.parametrize("field", FIELDS)
def test_metric_seed_invariance_rejects_each_required_metric(field: str) -> None:
    module = _module()
    with pytest.raises(RuntimeError, match=field):
        module.assert_metric_seed_invariance(
            _loaded(changed_field=field),
            seeds=(0, 1),
            symbols=("BTCUSDT",),
            strategies=("trend", "mean_reversion"),
        )


def test_full_report_requires_1120_baseline_candidate_metric_checks() -> None:
    module = _module()
    baseline, candidate = _full_pair()
    report = module.build_metric_invariance_report(baseline, candidate)
    assert report == {
        "schema_version": "issue519_exp002_metric_seed_invariance_verification_v1",
        "verified": True,
        "fields": list(FIELDS),
        "seeds": list(FROZEN_SEEDS),
        "symbols": list(FROZEN_SYMBOLS),
        "strategies": list(FROZEN_STRATEGIES),
        "baseline_checks": 560,
        "candidate_checks": 560,
        "total_checks": 1120,
    }


def test_write_report_is_deterministic_and_contains_no_performance_values(tmp_path) -> None:
    module = _module()
    baseline, candidate = _full_pair()
    report_path, digest = module.write_metric_invariance_report(
        baseline,
        candidate,
        report_root=tmp_path,
    )
    expected = module.build_metric_invariance_report(baseline, candidate)
    expected_text = json.dumps(expected, indent=2, sort_keys=True) + "\n"
    assert report_path == tmp_path / "metric-seed-invariance-verification.json"
    assert report_path.read_text(encoding="utf-8") == expected_text
    assert digest == hashlib.sha256(expected_text.encode("utf-8")).hexdigest()
    assert "candidate_total_return" not in expected_text
    assert "formal_decision" not in expected_text
