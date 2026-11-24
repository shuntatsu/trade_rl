"""One-shot cell runner for the frozen, previously consumed regime-switch study.

No market artifact is loaded by the CLI before authenticated research review
and a separate exact-scope owner comment have both passed.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import subprocess
from dataclasses import asdict
from pathlib import Path
from typing import Literal, cast

import numpy as np

from tools.agent_repo.research_assurance import assurance_digest
from tools.core7_g4_comment import fetch_comment, parse_g4_url
from tools.research_assurance_github import _github_api_read, verify_github_review
from trade_rl.artifacts import canonical_json_bytes
from trade_rl.data.artifacts import load_published_market_dataset_artifact
from trade_rl.data.features.cloud_regime_context import build_cloud_regime_context
from trade_rl.data.features.cloud_regime_switch_context import (
    build_cloud_regime_switch_context,
)
from trade_rl.data.features.cloud_trend_context import build_cloud_trend_context
from trade_rl.data.features.divergence_mtf import build_divergence_cloud_mtf
from trade_rl.data.market import MarketDataset
from trade_rl.data.source import RawMarketSeries
from trade_rl.evaluation.divergence_cloud_diagnostics import (
    attribute_single_symbol_ledger,
)
from trade_rl.evaluation.replay import run_shared_portfolio_replay
from trade_rl.risk import PreTradeRisk, PreTradeRiskConfig
from trade_rl.simulation import ExecutionCostConfig
from trade_rl.simulation.quantities import parse_quantity
from trade_rl.strategies.allocation.buy_and_hold import BuyAndHoldPortfolioPolicy
from trade_rl.strategies.cloud_regime_short_gate import (
    CloudRegimeShortGateConfig,
    CloudRegimeShortGatePolicy,
)
from trade_rl.strategies.cloud_regime_switch import CloudRegimeSwitchPolicy
from trade_rl.strategies.cloud_trend_policy import CloudTrendConfig, CloudTrendPolicy
from trade_rl.strategies.divergence_cloud_follow import (
    CloudFollowConfig,
    CloudFollowPolicy,
)
from trade_rl.strategies.portfolio import PortfolioPolicy
from trade_rl.strategies.portfolio_adapter import CashPortfolioPolicy

REPOSITORY = "shuntatsu/trade_rl"
REPO_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = REPO_ROOT / "report/cloud-regime-switch-dev-protocol-v1.json"
INPUT_ROOT = Path(
    "/home/rei/デスクトップ/dev/trade_rl_divergence_real_dev_20261123/switch_dev_frozen_inputs_20261124"
)
OUTPUT_ROOT = INPUT_ROOT.parent / "switch_dev_results_v1"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def protocol() -> dict:
    return json.loads(PROTOCOL.read_bytes())


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=20,
    ).stdout.strip()


def authorization_body(packet: dict) -> str:
    p = protocol()
    return "\n".join(
        [
            "G4_AUTHORIZE_ONCE cloud_regime_switch_dev_v1",
            f"protocol_sha256: {_sha(PROTOCOL)}",
            f"author_packet_sha256: {assurance_digest(packet)}",
            f"implementation_head: {packet['identity']['implementation_head']}",
            *(
                f"dataset_{symbol}: {p['datasets'][symbol]['dataset_id']}"
                for symbol in p["symbols"]
            ),
        ]
    )


def _authorize(packet: dict, pr: int, review_id: int, owner_url: str) -> dict:
    head = _git("rev-parse", "HEAD")
    if _git("status", "--porcelain") or packet["identity"] != {
        "implementation_head": head,
        "protocol_head": head,
    }:
        raise ValueError("clean exact protocol/implementation HEAD required")
    if _sha(PROTOCOL) not in packet["thesis"]["limitations"]:
        raise ValueError("packet must bind exact protocol bytes")
    review = verify_github_review(
        packet,
        repository=REPOSITORY,
        pr_number=pr,
        review_id=review_id,
        api_read=_github_api_read,
    )
    if review.status != "PASS":
        raise ValueError("authenticated independent research review has not passed")
    _, number, _ = parse_g4_url(owner_url)
    if number != pr:
        raise ValueError("owner approval must belong to the exact review PR")
    owner = fetch_comment(owner_url)
    if owner.body.strip() != authorization_body(packet):
        raise ValueError("separate exact owner authorization is required")
    again = fetch_comment(owner_url)
    if (again.body_sha256, again.updated_at) != (owner.body_sha256, owner.updated_at):
        raise ValueError("owner authorization changed during admission")
    return {
        "head": head,
        "packet_digest": assurance_digest(packet),
        "review_id": review_id,
        "pr": pr,
        "owner_comment": owner_url,
        "owner_body_sha256": owner.body_sha256,
        "protocol_sha256": _sha(PROTOCOL),
    }


def _policy(ds: MarketDataset, arm: str) -> PortfolioPolicy:
    symbol = ds.symbols[0]
    if arm == "CASH":
        return CashPortfolioPolicy(symbols=ds.symbols)
    if arm == "BH25":
        return BuyAndHoldPortfolioPolicy(symbols=ds.symbols, gross_budget=0.25)
    if arm in ("S0", "S1", "S2"):
        modes = {"S0": "trend_only", "S1": "hybrid", "S2": "hybrid_divergence"}
        # Literal arms are selected by protocol; no result-dependent routing.
        mode = cast(Literal["trend_only", "hybrid", "hybrid_divergence"], modes[arm])
        return CloudRegimeSwitchPolicy(
            context=build_cloud_regime_switch_context(ds), mode=mode
        )
    usable = (
        ds.tradable
        & ds.resolved_array("symbol_active")
        & ds.resolved_array("information_available")
        & (ds.resolved_array("available_at") <= ds.timestamps[:, None])
    )[:, 0]
    raw = RawMarketSeries(
        timestamps=ds.timestamps,
        available_at=ds.resolved_array("available_at")[:, 0],
        open=ds.open[:, 0],
        high=ds.high[:, 0],
        low=ds.low[:, 0],
        close=ds.close[:, 0],
        volume=ds.volume[:, 0],
        funding_rate=np.zeros(ds.n_bars),
        funding_available=np.zeros(ds.n_bars, dtype=bool),
        tradable=usable,
    )
    signals = build_divergence_cloud_mtf(raw, indicators=("RSI", "MACD_HIST", "MFI"))
    if arm in ("C2", "C3"):
        return CloudFollowPolicy(
            symbol=symbol,
            signals=signals,
            config=CloudFollowConfig(
                mode="confirmed_cloud" if arm == "C2" else "divergence_exit_cooldown"
            ),
        )
    if arm == "R2":
        return CloudRegimeShortGatePolicy(
            symbol=symbol,
            signals=signals,
            context=build_cloud_regime_context(raw, signals),
            config=CloudRegimeShortGateConfig(mode="momentum_and_cloud_slope"),
        )
    if arm in ("T0", "T1", "T2"):
        mode2 = cast(
            Literal["all_exits", "weak_trend_exits", "weak_trend_atr_entry"],
            {"T0": "all_exits", "T1": "weak_trend_exits", "T2": "weak_trend_atr_entry"}[
                arm
            ],
        )
        return CloudTrendPolicy(
            symbol=symbol,
            signals=signals,
            context=build_cloud_trend_context(raw),
            config=CloudTrendConfig(mode=mode2),
        )
    raise ValueError("unregistered study arm")


def evaluate_cell(
    ds: MarketDataset, arm: str, scenario: str, *, start: int, stop: int
) -> tuple[dict, bytes]:
    """Native simulation only; the CLI owns real-input authorization and publication."""
    p = protocol()
    if (
        ds.n_symbols != 1
        or ds.bar_hours != 1
        or arm not in p["arms"]
        or scenario not in p["scenarios"]
    ):
        raise ValueError("unsupported dataset or study cell")
    initial = float(p["initial_capital_usdt"])
    out = run_shared_portfolio_replay(
        ds,
        _policy(ds, arm),
        start_index=start,
        stop_index=stop,
        initial_capital=initial,
        execution_cost=ExecutionCostConfig(
            **p["scenarios"][scenario],
            max_participation_rate=0.05,
            allow_short=True,
            max_leverage=1.0,
        ),
        risk=PreTradeRisk(
            PreTradeRiskConfig(
                max_gross=1.0,
                max_abs_weight=1.0,
                max_turnover=None,
                drawdown_start=1.0,
                drawdown_stop=1.0,
            )
        ),
        settle_terminal_position=True,
        capture_ledger_evidence=True,
    )
    if out.ledger_evidence is None:
        raise ValueError("native ledger evidence missing")
    ledger = out.ledger_evidence.intervals
    ints = ledger.intervals
    book = out.book
    attr = attribute_single_symbol_ledger(ints)
    if not math.isclose(
        attr.net_total_usdt, book.portfolio_value - initial, abs_tol=1e-7, rel_tol=1e-10
    ):
        raise ValueError("equity attribution mismatch")
    for field, expected in (
        ("interval_cost", book.total_cost),
        ("interval_funding", book.funding_pnl),
        ("interval_borrow_cost", book.borrow_cost),
    ):
        if not math.isclose(
            math.fsum(getattr(x, field) for x in ints),
            expected,
            abs_tol=1e-7,
            rel_tol=1e-10,
        ):
            raise ValueError("native flow mismatch: " + field)
    quantities = [parse_quantity(x.exact_quantities_before[0]) for x in ints]
    mark = ds.resolved_array("mark_price")
    multiplier = float(ds.resolved_array("contract_multipliers")[0])
    gross = [
        abs(float(q))
        * float(mark[x.start_index, 0])
        * multiplier
        / x.portfolio_value_before
        for q, x in zip(quantities, ints, strict=True)
        if x.portfolio_value_before > 0
    ]
    returns = np.asarray(out.returns.values, dtype=float)
    sd = float(np.std(returns)) if len(returns) else 0.0
    sharpe = float(np.mean(returns) / sd * np.sqrt(8760)) if sd > 1e-12 else None
    terminal_ok = bool(
        book.termination_reason is None
        and ints[-1].next_index == stop
        and all(parse_quantity(q) == 0 for q in ledger.terminal_exact_quantities)
        and not any(abs(q) > 0 for _, q in ledger.active_order_remainders)
    )
    # A passive control that never entered is not a valid buy-and-hold
    # comparison, especially when latency causes HOLD to cancel the order.
    valid = terminal_ok and (arm != "BH25" or any(q > 0 for q in quantities))
    raw_ledger = canonical_json_bytes(out.ledger_evidence.to_mapping())
    metrics = {
        "valid": valid,
        "terminal_ok": terminal_ok,
        "arm": arm,
        "scenario": scenario,
        "net_return_pct": 100 * (float(book.portfolio_value) / initial - 1),
        "max_drawdown_pct": 100 * float(book.max_drawdown),
        "sharpe_hourly": sharpe,
        "native_fills": int(book.fill_count),
        "turnover": float(book.turnover_total),
        "total_cost_usdt": float(book.total_cost),
        "funding_pnl_usdt": float(book.funding_pnl),
        "borrow_cost_usdt": float(book.borrow_cost),
        "directional_equity_buckets": asdict(attr),
        "pre_interval_long_hours": sum(q > 0 for q in quantities),
        "pre_interval_short_hours": sum(q < 0 for q in quantities),
        "mean_pre_interval_gross": math.fsum(gross) / len(gross) if gross else None,
        "termination_reason": ledger.termination_reason,
        "ledger_sha256": hashlib.sha256(raw_ledger).hexdigest(),
    }
    return metrics, gzip.compress(raw_ledger, mtime=0)


def _load_fixed(symbol: str) -> MarketDataset:
    p = protocol()
    if _sha(INPUT_ROOT / "source_receipt.json") != p["data_receipt_sha256"]:
        raise ValueError("input reconstruction receipt changed")
    bound = p["datasets"][symbol]
    root = INPUT_ROOT / symbol
    for file, key in (
        ("manifest.json", "manifest_sha256"),
        ("arrays.npz", "arrays_npz_sha256"),
    ):
        if _sha(root / file) != bound[key]:
            raise ValueError("frozen artifact file digest changed")
    art, ds = load_published_market_dataset_artifact(root)
    if (
        art.artifact_digest != bound["artifact_digest"]
        or ds.dataset_id != bound["dataset_id"]
        or ds.symbols != (symbol,)
    ):
        raise ValueError("dataset identity mismatch")
    return ds


def summarize_recorded_cells(admission: dict) -> dict:
    """Audit the complete frozen roster without reopening the market Dataset."""
    from tools.cloud_regime_switch_summary import (
        cell_name,
        decide_comparison,
        read_cell,
        roster,
    )

    p = protocol()
    expected = {cell_name(key): key for key in roster(p)}
    if OUTPUT_ROOT.is_symlink():
        raise ValueError("output root must not be a symlink")
    entries = list(OUTPUT_ROOT.iterdir()) if OUTPUT_ROOT.exists() else []
    cells = {}
    for path in entries:
        if path.name not in expected:
            raise ValueError("unregistered result directory: " + path.name)
        key = expected[path.name]
        symbol, year, arm, scenario = key
        first = np.datetime64(
            p["warmup_start"].removesuffix("Z"), "h"
        ) + np.timedelta64(1, "h")
        start = int(
            (np.datetime64(f"{year}-01-01", "h") - first) / np.timedelta64(1, "h")
        )
        stop = int(
            (np.datetime64(f"{year + 1}-01-01", "h") - first) / np.timedelta64(1, "h")
        )
        cells[key] = read_cell(
            path,
            admission=admission,
            dataset_id=p["datasets"][symbol]["dataset_id"],
            arm=arm,
            scenario=scenario,
            expected_start=start,
            expected_stop=stop,
        )
    return decide_comparison(p, cells)


def main() -> None:
    p = protocol()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--packet", required=True, type=Path)
    ap.add_argument("--pr", required=True, type=int)
    ap.add_argument("--review-id", required=True, type=int)
    ap.add_argument("--owner-comment-url", required=True)
    ap.add_argument("--summarize", action="store_true")
    ap.add_argument("--symbol", choices=p["symbols"])
    ap.add_argument("--year", type=int, choices=p["years"])
    ap.add_argument("--arm", choices=p["arms"])
    ap.add_argument("--scenario", choices=tuple(p["scenarios"]))
    a = ap.parse_args()
    cell_args = (a.symbol, a.year, a.arm, a.scenario)
    if a.summarize:
        if any(x is not None for x in cell_args):
            ap.error("summary does not accept a single-cell selection")
    elif any(x is None for x in cell_args):
        ap.error("cell mode requires symbol, year, arm, and scenario")
    admission = _authorize(
        json.loads(a.packet.read_bytes()), a.pr, a.review_id, a.owner_comment_url
    )
    if a.summarize:
        print(
            json.dumps(
                summarize_recorded_cells(admission), sort_keys=True, allow_nan=False
            )
        )
        return
    name = f"{a.symbol}_{a.year}_{a.arm}_{a.scenario}"
    # Match the summary reader: do not publish into an aliased study root.
    if OUTPUT_ROOT.is_symlink():
        raise ValueError("output root must not be a symlink")
    OUTPUT_ROOT.mkdir(exist_ok=True)
    destination = OUTPUT_ROOT / name
    destination.mkdir()  # exclusive claim: no silent rerun or overwrite
    with (destination / "attempt.json").open("xb") as f:
        f.write(canonical_json_bytes({"admission": admission, "cell": name}))
    try:
        ds = _load_fixed(a.symbol)
        start = int(np.searchsorted(ds.timestamps, np.datetime64(f"{a.year}-01-01")))
        stop = int(np.searchsorted(ds.timestamps, np.datetime64(f"{a.year + 1}-01-01")))
        if ds.timestamps[start] != np.datetime64(f"{a.year}-01-01") or ds.timestamps[
            stop
        ] != np.datetime64(f"{a.year + 1}-01-01"):
            raise ValueError("fixed UTC calendar boundary missing")
        metrics, ledger = evaluate_cell(ds, a.arm, a.scenario, start=start, stop=stop)
        with (destination / "ledger.json.gz").open("xb") as f:
            f.write(ledger)
        with (destination / "result.json").open("xb") as f:
            f.write(
                canonical_json_bytes(
                    {
                        "admission": admission,
                        "cell": name,
                        "dataset_id": ds.dataset_id,
                        "metrics": metrics,
                        "ledger_gzip_sha256": hashlib.sha256(ledger).hexdigest(),
                    }
                )
            )
    except BaseException as exc:
        with (destination / "failure.json").open("xb") as f:
            f.write(
                canonical_json_bytes(
                    {"cell": name, "exception_type": type(exc).__name__}
                )
            )
        raise
    print("CELL_RECORDED", name)


if __name__ == "__main__":
    main()
