from __future__ import annotations

from types import SimpleNamespace

from tools.tmp_issue610_ppo_global_btc_regime_termination import (
    validate_no_new_ppo_terminations,
)


def _run(*, reason: str | None, symbols: tuple[str, ...] = ("BTCUSDT",)):
    strategies = []
    for _symbol in symbols:
        diagnostics = {"termination_reasons": [] if reason is None else [reason]}
        strategies.append(
            {
                "name": "ppo",
                "diagnostics": diagnostics,
                "metrics": {"termination_count": 0 if reason is None else 1},
            }
        )
    summary = {
        "symbols": list(symbols),
        "by_symbol": [
            {
                "symbol": symbol,
                "strategies": [strategy],
            }
            for symbol, strategy in zip(symbols, strategies, strict=True)
        ],
    }
    return SimpleNamespace(summary=summary)


def test_no_new_termination_when_candidate_matches_baseline() -> None:
    baseline = {0: _run(reason=None), 1: _run(reason="drawdown_stop")}
    candidate = {0: _run(reason=None), 1: _run(reason="drawdown_stop")}

    assert (
        validate_no_new_ppo_terminations(
            baseline,
            candidate,
            seeds=(0, 1),
            symbols=("BTCUSDT",),
        )
        == ()
    )


def test_new_candidate_termination_is_reported() -> None:
    baseline = {0: _run(reason=None)}
    candidate = {0: _run(reason="margin_call")}

    violations = validate_no_new_ppo_terminations(
        baseline,
        candidate,
        seeds=(0,),
        symbols=("BTCUSDT",),
    )
    assert violations == ("new PPO termination: seed=0 BTCUSDT reason=margin_call",)


def test_changed_termination_reason_is_new() -> None:
    baseline = {0: _run(reason="drawdown_stop")}
    candidate = {0: _run(reason="liquidation")}

    violations = validate_no_new_ppo_terminations(
        baseline,
        candidate,
        seeds=(0,),
        symbols=("BTCUSDT",),
    )
    assert violations == ("new PPO termination: seed=0 BTCUSDT reason=liquidation",)


def test_removed_baseline_termination_is_not_new() -> None:
    violations = validate_no_new_ppo_terminations(
        {0: _run(reason="drawdown_stop")},
        {0: _run(reason=None)},
        seeds=(0,),
        symbols=("BTCUSDT",),
    )
    assert violations == ()


def test_malformed_candidate_termination_evidence_fails_closed() -> None:
    malformed = _run(reason=None)
    malformed.summary["by_symbol"][0]["strategies"][0]["metrics"][
        "termination_count"
    ] = 1

    violations = validate_no_new_ppo_terminations(
        {0: _run(reason=None)},
        {0: malformed},
        seeds=(0,),
        symbols=("BTCUSDT",),
    )
    assert violations == (
        "candidate PPO termination evidence malformed: seed=0 BTCUSDT",
    )


def test_malformed_baseline_termination_evidence_fails_closed() -> None:
    malformed = _run(reason=None)
    malformed.summary["by_symbol"][0]["strategies"][0]["diagnostics"][
        "termination_reasons"
    ] = [""]

    violations = validate_no_new_ppo_terminations(
        {0: malformed},
        {0: _run(reason=None)},
        seeds=(0,),
        symbols=("BTCUSDT",),
    )
    assert violations == (
        "baseline PPO termination evidence malformed: seed=0 BTCUSDT",
    )


def test_seed_roster_must_be_exact_and_ordered() -> None:
    baseline = {0: _run(reason=None), 1: _run(reason=None)}
    candidate = {0: _run(reason=None), 1: _run(reason=None), 99: _run(reason=None)}

    assert validate_no_new_ppo_terminations(
        baseline,
        candidate,
        seeds=(0, 1),
        symbols=("BTCUSDT",),
    ) == ("candidate termination seed roster/order mismatch",)


def test_symbol_roster_must_be_exact_and_ordered() -> None:
    baseline = {0: _run(reason=None, symbols=("BTCUSDT", "ETHUSDT"))}
    candidate = {0: _run(reason=None, symbols=("ETHUSDT", "BTCUSDT"))}

    assert validate_no_new_ppo_terminations(
        baseline,
        candidate,
        seeds=(0,),
        symbols=("BTCUSDT", "ETHUSDT"),
    ) == ("candidate termination symbol roster/order mismatch: seed=0",)


def test_missing_or_duplicate_ppo_strategy_fails_closed() -> None:
    missing = _run(reason=None)
    missing.summary["by_symbol"][0]["strategies"] = []
    assert validate_no_new_ppo_terminations(
        {0: _run(reason=None)},
        {0: missing},
        seeds=(0,),
        symbols=("BTCUSDT",),
    ) == ("candidate PPO termination evidence malformed: seed=0 BTCUSDT",)

    duplicate = _run(reason=None)
    duplicate.summary["by_symbol"][0]["strategies"].append(
        duplicate.summary["by_symbol"][0]["strategies"][0].copy()
    )
    assert validate_no_new_ppo_terminations(
        {0: _run(reason=None)},
        {0: duplicate},
        seeds=(0,),
        symbols=("BTCUSDT",),
    ) == ("candidate PPO termination evidence malformed: seed=0 BTCUSDT",)
