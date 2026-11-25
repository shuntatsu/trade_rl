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
from collections.abc import Sequence
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, cast

import numpy as np

from tools.agent_repo.research_assurance import assurance_digest
from tools.core7_g4_comment import fetch_comment, parse_g4_url
from tools.research_assurance_github import _github_api_read, verify_github_review
from trade_rl.artifacts import canonical_json_bytes, content_digest
from trade_rl.artifacts.verified_file import open_regular_binary
from trade_rl.data.artifacts import load_published_market_dataset_artifact
from trade_rl.data.features.cloud_regime_context import build_cloud_regime_context
from trade_rl.data.features.cloud_regime_switch_context import (
    build_cloud_regime_switch_context,
)
from trade_rl.data.features.cloud_trend_context import build_cloud_trend_context
from trade_rl.data.features.divergence_mtf import build_divergence_cloud_mtf
from trade_rl.data.identity import (
    MARKET_DATASET_IDENTITY_SCHEMA,
    MARKET_DATASET_IDENTITY_SCHEMA_V7,
    parse_identity_json,
)
from trade_rl.data.market import MarketDataset
from trade_rl.data.source import RawMarketSeries
from trade_rl.evaluation.divergence_cloud_diagnostics import (
    attribute_single_symbol_ledger,
)
from trade_rl.evaluation.replay import run_shared_portfolio_replay
from trade_rl.evaluation.replay_exposure import mean_pre_interval_gross_from_ledger
from trade_rl.risk import PreTradeRisk, PreTradeRiskConfig
from trade_rl.simulation import ExecutionCostConfig, runtime_execution_policy_digest
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
FROZEN_PROTOCOL_SHA256 = (
    "4a8d3c70c10332bff9d023b493baa88a03f2ee513b11a3d777a7f563605cb0cf"
)
INPUT_ROOT = Path(
    "/home/rei/デスクトップ/dev/trade_rl_divergence_real_dev_20261123/switch_dev_frozen_inputs_20261124"
)
OUTPUT_ROOT = INPUT_ROOT.parent / "switch_dev_results_v1"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def protocol() -> dict:
    # Admission must bind the pre-registered bytes, not merely whichever
    # protocol a caller or edited packet happens to reference.
    raw = PROTOCOL.read_bytes()
    if hashlib.sha256(raw).hexdigest() != FROZEN_PROTOCOL_SHA256:
        raise ValueError("frozen protocol digest differs")
    return json.loads(raw)


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
            f"protocol_sha256: {FROZEN_PROTOCOL_SHA256}",
            f"author_packet_sha256: {assurance_digest(packet)}",
            f"implementation_head: {packet['identity']['implementation_head']}",
            *(
                f"dataset_{symbol}: {p['datasets'][symbol]['dataset_id']}"
                for symbol in p["symbols"]
            ),
        ]
    )


def _github_timestamp(value: object) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise ValueError("GitHub gate timestamp is missing or malformed")
    try:
        parsed = datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)
        if parsed.strftime("%Y-%m-%dT%H:%M:%SZ") != value:
            raise ValueError("noncanonical GitHub timestamp")
        return parsed
    except ValueError as exc:
        raise ValueError("GitHub gate timestamp is malformed") from exc


def _require_full_ci(head: str, pr: int) -> tuple[int, int, str]:
    """Require the exact-head review-event Full CI, not Fast Push or old Green."""
    root = f"repos/{REPOSITORY}/actions/runs"

    def inventory(payload: object, key: str) -> list[dict]:
        if not isinstance(payload, dict):
            raise ValueError("CI inventory is malformed")
        rows = payload.get(key)
        if (
            not isinstance(rows, list)
            or type(payload.get("total_count")) is not int
            or payload["total_count"] != len(rows)
            or any(not isinstance(row, dict) for row in rows)
        ):
            raise ValueError("CI inventory is incomplete or malformed")
        return rows

    def positive(value: object) -> int:
        if type(value) is not int or value <= 0:
            raise ValueError("CI run identity is malformed")
        return value

    def relevant(row: dict) -> bool:
        if (
            row.get("name") != "CI"
            or row.get("path") != ".github/workflows/ci.yml"
            or row.get("head_sha") != head
            or row.get("event") != "pull_request_review"
        ):
            return False
        prs = row.get("pull_requests")
        if not isinstance(prs, list):
            raise ValueError("CI pull-request association is malformed")
        return any(
            isinstance(item, dict)
            and item.get("number") == pr
            and isinstance(item.get("head"), dict)
            and item["head"].get("sha") == head
            for item in prs
        )

    def latest() -> dict:
        rows = inventory(
            _github_api_read(f"{root}?head_sha={head}&per_page=100"), "workflow_runs"
        )
        candidates = [row for row in rows if relevant(row)]
        if not candidates:
            raise ValueError("exact-head review-event Full CI is missing")
        ids = [positive(row.get("id")) for row in candidates]
        if len(ids) != len(set(ids)):
            raise ValueError("CI run identity is duplicated")
        return max(candidates, key=lambda row: positive(row.get("id")))

    run = latest()
    run_id = positive(run.get("id"))
    attempt = positive(run.get("run_attempt"))
    if run.get("status") != "completed" or run.get("conclusion") != "success":
        raise ValueError("latest exact-head review-event Full CI is not Green")
    jobs = inventory(
        _github_api_read(f"{root}/{run_id}/attempts/{attempt}/jobs?per_page=100"),
        "jobs",
    )
    for name in (
        "Lean Core",
        "PPO Runtime",
        "Human Guide",
        "Market Store PostgreSQL",
        "Generic Independent Research Review",
    ):
        matches = [job for job in jobs if job.get("name") == name]
        if len(matches) != 1 or any(
            matches[0].get(key) != expected
            for key, expected in (
                ("head_sha", head),
                ("run_id", run_id),
                ("run_attempt", attempt),
                ("status", "completed"),
                ("conclusion", "success"),
            )
        ):
            raise ValueError(f"exact-head CI job is not uniquely Green: {name}")
    final = _github_api_read(f"{root}/{run_id}")
    fields = ("id", "run_attempt", "status", "conclusion", "updated_at")
    _github_timestamp(run.get("updated_at"))
    if (
        not isinstance(final, dict)
        or not relevant(final)
        or any(final.get(field) != run[field] for field in fields)
        or any(
            latest_run.get(field) != run[field]
            for latest_run in [latest()]
            for field in fields
        )
    ):
        raise ValueError("CI run changed during admission")
    return run_id, attempt, run["updated_at"]


def _authorize(packet: dict, pr: int, review_id: int, owner_url: str) -> dict:
    head = _git("rev-parse", "HEAD")
    if _git("status", "--porcelain") or packet["identity"] != {
        "implementation_head": head,
        "protocol_head": head,
    }:
        raise ValueError("clean exact protocol/implementation HEAD required")
    protocol()
    if FROZEN_PROTOCOL_SHA256 not in packet["thesis"]["limitations"]:
        raise ValueError("packet must bind exact protocol bytes")
    review_path = f"repos/{REPOSITORY}/pulls/{pr}/reviews/{review_id}"
    review_snapshot: dict = {}

    def review_api(path: str) -> object:
        row = _github_api_read(path)
        if path == review_path and isinstance(row, dict):
            review_snapshot.clear()
            review_snapshot.update(row)
        return row

    review = verify_github_review(
        packet,
        repository=REPOSITORY,
        pr_number=pr,
        review_id=review_id,
        api_read=review_api,
    )
    if review.status != "PASS":
        raise ValueError("authenticated independent research review has not passed")
    reviewed_at = _github_timestamp(review_snapshot.get("submitted_at"))
    ci_run_id, ci_attempt, ci_updated_at = _require_full_ci(head, pr)
    _, number, comment_id = parse_g4_url(owner_url)
    if number != pr:
        raise ValueError("owner approval must belong to the exact review PR")
    owner_path = f"repos/{REPOSITORY}/issues/comments/{comment_id}"
    owner_metadata: dict = {}

    def owner_gh(args: Sequence[str]) -> str:
        if list(args) != ["api", owner_path]:
            raise ValueError("unexpected owner authorization lookup")
        row = _github_api_read(owner_path)
        if not isinstance(row, dict):
            raise ValueError("owner authorization metadata is malformed")
        owner_metadata.clear()
        owner_metadata.update(row)
        return json.dumps(row)

    owner = fetch_comment(owner_url, gh=owner_gh)
    created_at = owner_metadata.get("created_at")
    if created_at != owner.updated_at or _github_timestamp(created_at) <= max(
        reviewed_at, _github_timestamp(ci_updated_at)
    ):
        raise ValueError(
            "owner authorization must be newly created after review and Full CI"
        )
    if owner.body.strip() != authorization_body(packet):
        raise ValueError("separate exact owner authorization is required")

    def require_unchanged_owner() -> None:
        again = fetch_comment(owner_url, gh=owner_gh)
        if (again.body_sha256, again.updated_at, owner_metadata.get("created_at")) != (
            owner.body_sha256,
            owner.updated_at,
            created_at,
        ):
            raise ValueError("owner authorization changed during admission")

    require_unchanged_owner()
    # Repeat the authenticated gates after the owner lookup: the open PR,
    # latest CI attempt and local checkout can change independently.
    if _require_full_ci(head, pr) != (ci_run_id, ci_attempt, ci_updated_at):
        raise ValueError("Full CI changed during admission")
    final_review = verify_github_review(
        packet,
        repository=REPOSITORY,
        pr_number=pr,
        review_id=review_id,
        api_read=_github_api_read,
    )
    if (
        final_review.status != "PASS"
        or _github_api_read(review_path) != review_snapshot
    ):
        raise ValueError("independent research review or PR changed during admission")
    # Owner approval can be edited or withdrawn while CI/review is rechecked.
    # Re-read it before the final local checks; these independent reads do not
    # constitute an atomic snapshot across GitHub and the checkout.
    require_unchanged_owner()
    if _git("rev-parse", "HEAD") != head or _git("status", "--porcelain"):
        raise ValueError("local checkout changed during admission")
    protocol()
    return {
        "head": head,
        "packet_digest": assurance_digest(packet),
        "review_id": review_id,
        "ci_run_id": ci_run_id,
        "ci_run_attempt": ci_attempt,
        "ci_run_updated_at": ci_updated_at,
        "review_submitted_at": review_snapshot["submitted_at"],
        "owner_created_at": created_at,
        "pr": pr,
        "owner_comment": owner_url,
        "owner_body_sha256": owner.body_sha256,
        "protocol_sha256": FROZEN_PROTOCOL_SHA256,
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


def _scenario_cost(protocol_value: dict, scenario: str) -> ExecutionCostConfig:
    """Derive the native cost configuration only from the frozen protocol."""
    if scenario not in protocol_value["scenarios"]:
        raise ValueError("unregistered execution scenario")
    return ExecutionCostConfig(
        **protocol_value["scenarios"][scenario],
        max_participation_rate=0.05,
        allow_short=True,
        max_leverage=1.0,
    )


def _verified_funding_price_rate_present(
    raw: bytes, *, bound: dict, symbol: str
) -> bool:
    """Derive funding presence from the independently pinned manifest only.

    No array decode, prices, or second execution. The artifact's verified
    metadata is a contract, not an attestation of the original market.
    """
    if hashlib.sha256(raw).hexdigest() != bound["manifest_sha256"]:
        raise ValueError("frozen manifest digest mismatch")
    manifest = json.loads(raw)
    if type(manifest) is not dict or canonical_json_bytes(manifest) != raw:
        raise ValueError("noncanonical frozen manifest")
    scalars = manifest.get("dataset")
    arrays = manifest.get("arrays")
    if type(scalars) is not dict or type(arrays) is not dict:
        raise ValueError("frozen manifest metadata mismatch")
    if (
        manifest.get("artifact_digest") != bound["artifact_digest"]
        or content_digest({k: v for k, v in manifest.items() if k != "artifact_digest"})
        != bound["artifact_digest"]
        or manifest.get("dataset_id") != bound["dataset_id"]
        or scalars.get("dataset_id") != bound["dataset_id"]
        or scalars.get("symbols") != [symbol]
        or manifest.get("arrays_file") != "arrays.npz"
        or manifest.get("arrays_digest") != bound["arrays_npz_sha256"]
        or set(scalars).intersection(arrays)
    ):
        raise ValueError("frozen manifest identity mismatch")
    identity_text = scalars.get("identity_payload_json")
    if type(identity_text) is not str:
        raise ValueError("frozen manifest identity metadata missing")
    identity = parse_identity_json(identity_text)
    present = "funding_price_rate" in arrays
    expected_schema = (
        ("market_dataset_artifact_v4", MARKET_DATASET_IDENTITY_SCHEMA_V7)
        if present
        else ("market_dataset_artifact_v3", MARKET_DATASET_IDENTITY_SCHEMA)
    )
    if (manifest.get("schema_version"), identity.get("schema")) != expected_schema:
        raise ValueError("frozen manifest funding schema mismatch")
    return present


def _read_frozen_funding_presence(protocol_value: dict, symbol: str) -> bool:
    bound = protocol_value["datasets"][symbol]
    with open_regular_binary(
        INPUT_ROOT / symbol / "manifest.json", field="frozen dataset manifest"
    ) as handle:
        raw = handle.read(8 * 1024 * 1024 + 1)
    if len(raw) > 8 * 1024 * 1024:
        raise ValueError("frozen manifest exceeds metadata byte ceiling")
    return _verified_funding_price_rate_present(raw, bound=bound, symbol=symbol)


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
    cost = _scenario_cost(p, scenario)
    out = run_shared_portfolio_replay(
        ds,
        _policy(ds, arm),
        start_index=start,
        stop_index=stop,
        initial_capital=initial,
        execution_cost=cost,
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
        capture_exposure_evidence=True,
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
    expected_gross = math.fsum(gross) / len(gross) if gross else None
    restored_gross = mean_pre_interval_gross_from_ledger(
        raw_ledger,
        expected_ledger_digest=hashlib.sha256(raw_ledger).hexdigest(),
        expected_dataset_id=ds.dataset_id,
        expected_execution_policy_digest=runtime_execution_policy_digest(
            cost, funding_price_rate_present=ds.funding_price_rate is not None
        ),
        expected_symbols=ds.symbols,
        expected_start_index=start,
        expected_stop_index=stop,
    )
    if restored_gross != expected_gross:
        raise ValueError(
            "native exposure evidence differs from fixed reporting formula"
        )
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
        "mean_pre_interval_gross": expected_gross,
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
    funding_presence: dict[str, bool] = {}
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
        if symbol not in funding_presence:
            funding_presence[symbol] = _read_frozen_funding_presence(p, symbol)
        expected_policy = runtime_execution_policy_digest(
            _scenario_cost(p, scenario),
            funding_price_rate_present=funding_presence[symbol],
        )
        cells[key] = read_cell(
            path,
            admission=admission,
            dataset_id=p["datasets"][symbol]["dataset_id"],
            expected_symbol=symbol,
            expected_execution_policy_digest=expected_policy,
            arm=arm,
            scenario=scenario,
            expected_start=start,
            expected_stop=stop,
            expected_start_timestamp_ns=int(
                np.datetime64(f"{year}-01-01", "ns").astype(np.int64)
            ),
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
