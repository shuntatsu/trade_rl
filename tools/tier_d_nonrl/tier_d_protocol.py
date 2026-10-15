"""Result-blind declarations; content pins are reproducibility, not authority."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

PACKAGE = Path(__file__).resolve().parent
REPO = PACKAGE.parent.parent
ROOT = REPO / "output/tier-d-nonrl"
IMPLEMENTATION_MODEL = "gpt-6.1-sol"
IMPLEMENTATION_MODELS = (IMPLEMENTATION_MODEL, "gpt-6-luna")
TASK_ID = "/root/tier_d_nonrl_runner_worker"
DATASET_ROOT = Path(
    "C:/dev/trade_rl/.worktrees/nonrl-market-development-20261012/output/"
    "nonrl-market-development-20261012/task18-selected-six/"
    "task18-new-selected-six-feature-witness"
)
DATASET_ID = "e916ff716fb5e4b90d5805eaa79200824b09f103984fa845bbeaedb57bef194f"
INPUT_PINS = {
    "manifest.json": "6ac5099b98a32bd33e4f98c5ac5fb2dbffd19b0b9e3c7c3a9b5a8f6860810ee6",
    "arrays.npz": "25c97a4ebd1e81ac64f4bc7e78dce4ab102c522e548e30a2154fcfa0143842a3",
}
SCENARIOS = ("BASEL0", "COST2L0", "BASEL2")
ROLES = ("nonrl", "cash")
CONFIGURATIONS = ((24, 1.0), (24, 10.0), (72, 1.0), (72, 10.0))
CAPITAL = 100000.0


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_sha256(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def source_pins() -> dict[str, str]:
    return {
        path.relative_to(PACKAGE).as_posix(): file_sha256(path)
        for path in sorted(PACKAGE.rglob("*.py"))
    }


def source_head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
    ).strip()


def validate_committed_source(protocol: dict[str, Any]) -> None:
    """Market admission requires this frozen committed source before data IO."""
    if source_head() != protocol["source_head"]:
        raise ValueError(
            "frozen protocol source HEAD changed; use a fresh reviewed attempt"
        )
    if subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"], cwd=REPO
    ).strip():
        raise ValueError("market execution requires a clean tracked tree")
    current = {
        path.relative_to(REPO).as_posix(): path for path in PACKAGE.rglob("*.py")
    }
    tracked = {
        name.decode()
        for name in subprocess.check_output(
            ["git", "ls-files", "-z", "--", PACKAGE.relative_to(REPO).as_posix()],
            cwd=REPO,
        ).split(b"\0")
        if name and name.endswith(b".py")
    }
    if not current or set(current) != tracked:
        raise ValueError("every maintained runner Python file must be tracked")
    for relative, path in current.items():
        committed = subprocess.check_output(
            ["git", "show", f"HEAD:{relative}"], cwd=REPO
        )
        if path.is_symlink() or hashlib.sha256(committed).hexdigest() != file_sha256(
            path
        ):
            raise ValueError("runner source bytes differ from the committed HEAD")


def worker_count(value: int) -> int:
    if type(value) is not int or not 1 <= value <= 6:
        raise ValueError("concurrency max_workers must be an integer within 1..6")
    return value


def configuration_id(horizon: int, alpha: float) -> str:
    return f"H{horizon}_RIDGE_A{alpha:g}"


def market_protocol(*, max_workers: int = 1) -> dict[str, Any]:
    accounts = [
        f"{configuration_id(horizon, alpha)}_{symbol}"
        for horizon, alpha in CONFIGURATIONS
        for symbol in ("BTCUSDT", "ETHUSDT")
    ]
    return {
        "schema": "issue810_tier_d_nonrl48_result_blind_v3",
        "scope": "CONSUMED_2025_DEVELOPMENT_NONRL48_ONLY",
        "source_head": source_head(),
        "concurrency": {"max_workers": worker_count(max_workers)},
        "progress_policy": {
            "schema": "issue810_native_progress_v1",
            "no_progress_seconds": 600,
            "sample_seconds": 1,
        },
        "dataset": {
            "root": str(DATASET_ROOT),
            "dataset_id": DATASET_ID,
            "input_sha256": INPUT_PINS,
            "dimensions": [9504, 2, 6],
            "schema": "market_dataset_artifact_v4",
            "identity": "market_dataset_identity_v7",
            "build": "market_build_v3",
            "development_start": "2024-12-01T01:00:00",
            "development_stop_exclusive": "2026-01-01T01:00:00",
            "feature_names": [
                "1h__log_return_24bar",
                "4h__log_return_1bar",
                "1d__log_return_1bar",
                "1h__realized_volatility_24bar",
                "1h__volume_zscore_24bar",
                "1h__funding_bps",
            ],
            "fit_symbols": ["BTCUSDT", "ETHUSDT"],
            "unused_status": "CONSUMED_DEVELOPMENT_NEVER_UNUSED",
        },
        "configurations": [
            {"id": configuration_id(h, a), "horizon_hours": h, "ridge_alpha": a}
            for h, a in CONFIGURATIONS
        ],
        "accounts": accounts,
        "scenarios": list(SCENARIOS),
        "roles": list(ROLES),
        "capital": {
            "currency": "USDT",
            "fx_parity": 1.0,
            "per_account": CAPITAL,
            "combined_denominator": 2 * CAPITAL,
            "mode": "independent_symbol",
            "combination": "two nominal USDT books for ONE config/scenario only",
        },
        "actor": {
            "start": 2903,
            "stop": 9430,
            "rows": 6527,
            "start_utc": "2025-04-01T00:00:00",
            "stop_utc": "2025-12-28T23:00:00",
            "vintage_starts": [2903, 5087, 7295],
            "vintage_start_utc": [
                "2025-04-01T00:00:00",
                "2025-07-01T00:00:00",
                "2025-10-01T00:00:00",
            ],
            "fit_cutoff": "vintage start minus 1 ns; endpoints/publication strictly earlier",
            "model_fit_time": "vintage start; declared simulation, not measured latency",
            "inference_delay_seconds": 0,
        },
        "terminal": {
            "stop_inclusive": 9502,
            "advances": 72,
            "utc": "2025-12-31T23:00:00",
            "last_row_9503": "unused by actor/reserve",
        },
        "economics": {
            "dataset_fee": 0.0005,
            "dataset_spread": 0.0002,
            "overlay_fee": 0.0,
            "overlay_spread": 0.0,
            "impact_coefficient": 0.0001,
            "reference_notional": CAPITAL,
            "participation": 0.05,
            "borrow_available": True,
            "borrow_rate": 0.0,
            "cash_rate": 0.0,
            "stochastic_slippage": 0.0,
            "cost2": "multiplier2 doubles fee/spread/impact only; carry unchanged",
            "latency2": "BASE beliefs/forecasts; MARKET latency2 actual execution",
            "execution": "one native MarketExecutor/BookState ledger; prior-bar capacity",
            "venue_rules": "current snapshot assumption, not historical PIT",
        },
        "funding": {
            "marks": "bar-close proxy; rates times bar-close mark all rows",
            "source": "Task18 rates-only feature witness, not authenticated venue marks",
            "observed_24h_counts": {"minimum": 2, "maximum": 4},
            "coverage_bounds": {
                "lookback_hours": 24,
                "minimum_counted_events": 2,
                "maximum_counted_close_gap_hours": 9,
                "maximum_counted_close_age_hours": 8,
            },
            "zero_rate_events": "counted by event_count, not rate!=0; original dates unchanged",
            "clock_limit": "counted bar-close proxy may understate actual settlement age",
        },
        "policy": {
            "allocator": {
                "lower_weight": -1.0,
                "upper_weight": 1.0,
                "max_turnover": None,
                "risk_aversion": 0.0,
            },
            "risk": {
                "max_gross": 1.0,
                "max_abs_weight": 1.0,
                "max_turnover": None,
                "drawdown_start": 0.10,
                "drawdown_stop": 0.20,
                "emergency_turnover_override": True,
                "fail_closed_tolerance": 1e-10,
            },
            "nonrl_action": "residual KEEP_BASELINE (code2), scale1",
            "cash_action": "direct HOLD (code0) from flat initial book, scale1",
            "forecasts": "raw close-ratio simple-return labels; carry/costs in allocator",
            "resets_per_arm": 1,
            "observation_schema": "allocation_account_observation_v2",
            "max_active_order_slots": 8,
            "vintage_boundary": "continuous quantity/cash/pending/risk",
            "gamma": 1.0,
            "reward": "equity_delta_v1 / fixed V0",
            "rollout_steps": 32,
            "gae_lambda_unused_nonrl": 0.95,
            "fold_metadata": "execution segments only; checkpoint/selection placeholders UNUSED",
        },
        "budget": {
            "dataset_loads": 5,
            "cost_producers": 4,
            "public_prepare_fits": 4,
            "ridge_vintages": 12,
            "regression_features": 6,
            "matched_pairs": 24,
            "native_arms": 48,
            "advances_per_arm": 6599,
            "total_native_advances": 316752,
            "seeds": [],
            "hpo": False,
            "unattempted": "retain UNATTEMPTED; never fabricate a cash-zero receipt",
        },
        "resource_proposal": {
            "measured": False,
            "rss_bytes": 2 * 1024**3,
            "disk_bytes": 4 * 1024**3,
            "wall_seconds": 3600,
            "sample_seconds": 10,
            "semantics": "Windows owned process-tree RSS including redirector descendants; sampled stop; overshoot possible",
            "maximum_inflight_pairs": max_workers,
            "storage": "gzip full raw receipts; matched pairs retained by at most the declared in-flight worker count",
            "limitations": "native APIs retain arm receipts in active workers; full-matrix feasibility and the proposed resource/progress bounds are not established",
        },
        "decision": {
            "primary": "terminal after-cost NONRL minus matched CASH wealth / V0",
            "rank": "per config/scenario combined BTC+ETH difference / (2*V0)",
            "guardrails": "every receipt verified, flat, no pending; maximum DD<=20% per book",
            "stress": "BASEL0, COST2L0, BASEL2 each required and disclosed; no crossed cell",
            "adoption": "NOT_ESTABLISHED",
            "winner": "NOT_DECLARED",
            "minimum_operational_profit_threshold": "UNKNOWN; no adoption claim",
            "decomposition": "returned native price PnL / fill fee-spread-impact / funding / borrow-interest; terminal closing cost disclosed as subset",
            "operating_costs": "fixed infrastructure costs UNKNOWN; account net is pre-tax and before unknown operating costs",
        },
        "research": {
            "objective": "screen fixed NONRL terminal net profit against CASH",
            "hypothesis": "lagged selected observations may predict H24/H72 price returns",
            "causal_story": "persistence/reversion in consumed crypto prices is a conjecture",
            "falsifiers": "nonpositive CASH delta, DD breach, incomplete settlement/coverage",
            "can_prove": "conditional development comparison under declared financial proxies",
            "cannot_prove": "unused-future, maximum profit, historical calibration, live readiness",
            "review": "ROOT different-model result-blind review BEFORE actual costs/fits/PnL",
            "no": [
                "new features",
                "new dataset",
                "new algorithm",
                "LLM trading/teachers",
                "second ledger",
                "RL market fit",
                "sealed/unused reads",
                "network market",
                "live orders",
            ],
            "legacy": "six-fit PPO learnability stays FAILED incl fee-only seed17; Phase1 source unchanged",
        },
        "attempts": "normal technical debugging/retries; retain each failed attempt and unchanged thresholds",
    }
