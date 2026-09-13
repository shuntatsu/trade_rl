from __future__ import annotations

from importlib import import_module
from importlib.util import find_spec
from types import SimpleNamespace

import pytest

MODULE = "research.issue529_portable_exp004_metricverify"


def _module():
    spec = find_spec(MODULE)
    assert spec is not None, "Experiment 0004 metric verifier is not implemented"
    return import_module(MODULE)


def _loaded(*, drift: bool = False):
    seeds = (0, 1, 2, 3, 4)
    symbols = ("BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT")
    strategies = (
        "cash",
        "constant_long",
        "constant_short",
        "trend",
        "mean_reversion",
        "ridge24",
        "lightgbm24",
    )
    runs = {}
    for seed in seeds:
        by_symbol = []
        for symbol_index, symbol in enumerate(symbols):
            entries = []
            for strategy_index, strategy in enumerate(strategies):
                base = float(symbol_index + strategy_index)
                if drift and seed == 4 and symbol == "BTCUSDT" and strategy == "lightgbm24":
                    base += 0.01
                entries.append(
                    {
                        "name": strategy,
                        "metrics": {
                            "total_return": base,
                            "max_drawdown": base + 0.1,
                            "turnover_total": base + 0.2,
                            "total_cost": base + 0.3,
                        },
                    }
                )
            by_symbol.append({"symbol": symbol, "strategies": entries})
        runs[seed] = SimpleNamespace(summary={"by_symbol": by_symbol})
    return SimpleNamespace(runs=runs)


def test_metric_invariance_report_has_exact_check_count() -> None:
    module = _module()
    report = module.build_metric_invariance_report(_loaded(), _loaded())
    assert report["baseline_checks"] == 560
    assert report["candidate_checks"] == 560
    assert report["total_checks"] == 1120
    assert report["verified"] is True


def test_metric_invariance_rejects_seed_sensitive_deterministic_metric() -> None:
    module = _module()
    with pytest.raises(RuntimeError, match="seed-sensitive metric"):
        module.build_metric_invariance_report(_loaded(), _loaded(drift=True))
