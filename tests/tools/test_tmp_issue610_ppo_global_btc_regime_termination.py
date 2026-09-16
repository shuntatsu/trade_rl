from __future__ import annotations

from types import SimpleNamespace


def _module():
    return __import__(
        "tools.tmp_issue610_ppo_global_btc_regime_termination",
        fromlist=["*"],
    )


def _run(*, reason: str | None):
    diagnostics = {"termination_reasons": [] if reason is None else [reason]}
    strategy = {
        "name": "ppo",
        "diagnostics": diagnostics,
        "metrics": {"termination_count": 0 if reason is None else 1},
    }
    summary = {
        "symbols": ["BTCUSDT"],
        "by_symbol": [
            {
                "symbol": "BTCUSDT",
                "strategies": [strategy],
            }
        ],
    }
    return SimpleNamespace(summary=summary)


def test_no_new_termination_when_candidate_matches_baseline() -> None:
    module = _module()
    baseline = {0: _run(reason=None), 1: _run(reason="drawdown_stop")}
    candidate = {0: _run(reason=None), 1: _run(reason="drawdown_stop")}

    assert module.validate_no_new_ppo_terminations(
        baseline,
        candidate,
        seeds=(0, 1),
        symbols=("BTCUSDT",),
    ) == ()


def test_new_candidate_termination_is_reported() -> None:
    module = _module()
    baseline = {0: _run(reason=None)}
    candidate = {0: _run(reason="margin_call")}

    violations = module.validate_no_new_ppo_terminations(
        baseline,
        candidate,
        seeds=(0,),
        symbols=("BTCUSDT",),
    )
    assert violations == (
        "new PPO termination: seed=0 BTCUSDT reason=margin_call",
    )


def test_changed_termination_reason_is_new() -> None:
    module = _module()
    baseline = {0: _run(reason="drawdown_stop")}
    candidate = {0: _run(reason="liquidation")}

    violations = module.validate_no_new_ppo_terminations(
        baseline,
        candidate,
        seeds=(0,),
        symbols=("BTCUSDT",),
    )
    assert violations == (
        "new PPO termination: seed=0 BTCUSDT reason=liquidation",
    )


def test_malformed_termination_evidence_fails_closed() -> None:
    module = _module()
    malformed = _run(reason=None)
    malformed.summary["by_symbol"][0]["strategies"][0]["metrics"]["termination_count"] = 1

    violations = module.validate_no_new_ppo_terminations(
        {0: _run(reason=None)},
        {0: malformed},
        seeds=(0,),
        symbols=("BTCUSDT",),
    )
    assert violations == (
        "candidate PPO termination evidence malformed: seed=0 BTCUSDT",
    )
