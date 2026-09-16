from __future__ import annotations

from types import SimpleNamespace

from tools.tmp_issue610_interpretation_precheck_v3 import (
    SEEDS,
    SYMBOLS,
    classify_strict_termination_evidence,
    partition_termination_violations,
)


def _loaded(*, reason: str | None = None) -> SimpleNamespace:
    runs: dict[int, SimpleNamespace] = {}
    for seed in SEEDS:
        by_symbol: list[dict[str, object]] = []
        for symbol in SYMBOLS:
            reasons = [] if reason is None else [reason]
            by_symbol.append(
                {
                    "symbol": symbol,
                    "strategies": [
                        {
                            "name": "ppo",
                            "diagnostics": {"termination_reasons": reasons},
                            "metrics": {"termination_count": len(reasons)},
                        }
                    ],
                }
            )
        runs[seed] = SimpleNamespace(
            summary={"symbols": list(SYMBOLS), "by_symbol": by_symbol}
        )
    return SimpleNamespace(runs=runs)


def _ppo_entry(
    loaded: SimpleNamespace, *, seed: int, symbol_index: int
) -> dict[str, object]:
    run = loaded.runs[seed]
    return run.summary["by_symbol"][symbol_index]["strategies"][0]


def test_strict_precheck_accepts_consistent_unchanged_terminations() -> None:
    baseline = _loaded(reason="risk_limit")
    candidate = _loaded(reason="risk_limit")
    assert classify_strict_termination_evidence(baseline, candidate) == ((), ())


def test_strict_precheck_routes_genuine_new_reason_to_gate_failure() -> None:
    baseline = _loaded(reason="risk_limit")
    candidate = _loaded(reason="risk_limit")
    entry = _ppo_entry(candidate, seed=0, symbol_index=0)
    entry["diagnostics"] = {"termination_reasons": ["economic_floor"]}

    new_termination, invalid = classify_strict_termination_evidence(baseline, candidate)
    assert invalid == ()
    assert new_termination == (
        "new PPO termination: seed=0 BTCUSDT reason=economic_floor",
    )


def test_count_reason_inconsistency_is_invalid_not_gate_failure() -> None:
    baseline = _loaded()
    candidate = _loaded()
    entry = _ppo_entry(candidate, seed=0, symbol_index=0)
    entry["metrics"] = {"termination_count": 1}

    new_termination, invalid = classify_strict_termination_evidence(baseline, candidate)
    assert new_termination == ()
    assert invalid == ("candidate PPO termination evidence malformed: seed=0 BTCUSDT",)


def test_duplicate_reason_is_invalid_not_gate_failure() -> None:
    baseline = _loaded()
    candidate = _loaded()
    entry = _ppo_entry(candidate, seed=0, symbol_index=0)
    entry["diagnostics"] = {"termination_reasons": ["risk_limit", "risk_limit"]}
    entry["metrics"] = {"termination_count": 2}

    new_termination, invalid = classify_strict_termination_evidence(baseline, candidate)
    assert new_termination == ()
    assert invalid == ("candidate PPO termination evidence malformed: seed=0 BTCUSDT",)


def test_symbol_roster_drift_is_invalid() -> None:
    baseline = _loaded()
    candidate = _loaded()
    candidate.runs[0].summary["symbols"] = list(reversed(SYMBOLS))

    new_termination, invalid = classify_strict_termination_evidence(baseline, candidate)
    assert new_termination == ()
    assert invalid == ("candidate termination symbol roster/order mismatch: seed=0",)


def test_partition_treats_unknown_violation_as_invalid() -> None:
    assert partition_termination_violations(
        (
            "new PPO termination: seed=0 BTCUSDT reason=risk_limit",
            "unexpected evidence shape",
        )
    ) == (
        ("new PPO termination: seed=0 BTCUSDT reason=risk_limit",),
        ("unexpected evidence shape",),
    )
