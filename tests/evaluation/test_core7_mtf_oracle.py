"""Synthetic G2 independent Decimal ledger oracle, decision truth table, runner E2E, and open packet contract."""

from __future__ import annotations

import ast
import json
import subprocess
from dataclasses import replace
from decimal import Decimal, getcontext
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from tests.data.test_mtf6_signals import _make_alternating_dataset
from tests.evaluation.packet_pin_rule import evaluate_pin, git_blob_reader
from tools.agent_repo.research_assurance import assurance_digest, evaluate_assurance
from trade_rl.data.features.mtf6_signals import build_mtf6_signal_pack
from trade_rl.evaluation.core7_mtf_study import (
    INVALID,
    PASS_PARITY_ONLY,
    STOP_FAIL_NO_RESCUE,
    SYNTHETIC_ONLY_NOT_EVIDENCE,
    CellSummary,
    decide,
    protocol,
    run_causality_probe,
    run_study,
)
from trade_rl.evaluation.replay import run_shared_portfolio_replay
from trade_rl.risk.pretrade import PreTradeRisk
from trade_rl.strategies.rl.shadow_anchor_mtf import (
    ShadowAnchorMlpWeights,
    ShadowAnchorMtfConfig,
    ShadowAnchorMtfPolicy,
)

ROOT = Path(__file__).resolve().parents[2]
PACKET_PATH = ROOT / "report" / "issue810-core7-mtf-ppo-research-assurance-v5.json"
PACKET_V4_PATH = ROOT / "report" / "issue810-core7-mtf-ppo-research-assurance-v4.json"
PACKET_V3_PATH = ROOT / "report" / "issue810-core7-mtf-ppo-research-assurance-v3.json"
PACKET_V2_PATH = ROOT / "report" / "issue810-core7-mtf-ppo-research-assurance-v2.json"
PACKET_V1_PATH = ROOT / "report" / "issue810-core7-mtf-ppo-research-assurance-v1.json"
SPEC_PATH = ROOT / "docs" / "specs" / "core7-mtf-shadow-anchor-ppo.md"

PINNED_PROTOCOL_DIGEST = (
    "af34928c682de43afce8e203a42841ec7bdb89eb9407fe0b9399b7821e2ca013"
)
PINNED_PACKET_DIGEST = (
    "003a2767febabc9dbe2729b1a6e3efb0217a578471d2b9413766bfd496cf8a39"
)
ENTRY_POINTS = (
    "trade_rl/data/features/mtf6_signals.py",
    "trade_rl/strategies/rl/shadow_anchor_mtf.py",
    "trade_rl/evaluation/core7_mtf_study/",
)


def _dec(value: float) -> Decimal:
    return Decimal(repr(float(value)))


def _assert_close_decimal(actual: float, expected: Decimal, *, label: str) -> None:
    tol = max(Decimal("2e-9"), Decimal("2e-12") * abs(expected))
    diff = abs(_dec(actual) - expected)
    assert diff <= tol, (
        f"{label}: actual={actual}, expected={expected}, diff={diff}, tol={tol}"
    )


def test_independent_decimal_2pair_4leg_ledger_oracle() -> None:
    """Verify cash, equity, fee+spread+sqrt-impact cost, funding, turnover, and terminal flatness."""
    getcontext().prec = 50
    ds_raw = _make_alternating_dataset(
        n_bars=48, pairs=("BTCUSDT", "ETHUSDT"), seed=103
    )
    # Apply tight volume on a few bars so 1% participation cap actually binds on some fills
    vol = np.asarray(ds_raw.volume, dtype=np.float64).copy()
    vol[14:18, :] = 150.0
    # Zero spot funding, explicit non-zero perp funding
    fr = np.asarray(ds_raw.funding_rate, dtype=np.float64).copy()
    fr[:, 0::2] = 0.0
    fr[:, 1::2] = 0.00025
    ds = replace(
        ds_raw,
        dataset_id="c" * 64,
        volume=vol,
        funding_rate=fr,
    )

    pack = build_mtf6_signal_pack(ds)
    start_index = 10
    stop_index = 46
    initial_capital = 100_000.0
    cost_cfg = protocol.scenario_cost("BASE")
    risk = PreTradeRisk(protocol.risk_config())

    policy = ShadowAnchorMtfPolicy(
        signal_pack=pack,
        config=ShadowAnchorMtfConfig(
            mode="core7_sparse_ppo",
            start_index=start_index,
            fit_stop_index=start_index,
            rebalance_every_bars=6,
            deadband=0.04,
        ),
        weights=ShadowAnchorMlpWeights.zero_residual(fit_stop_index=start_index),
    )

    result = run_shared_portfolio_replay(
        ds,
        policy,
        start_index=start_index,
        stop_index=stop_index,
        initial_capital=initial_capital,
        execution_cost=cost_cfg,
        risk=risk,
        settle_terminal_position=True,
        capture_ledger_evidence=True,
    )
    assert result.ledger_evidence is not None
    intervals = result.ledger_evidence.intervals.intervals
    assert len(intervals) > 0

    n_syms = ds.n_symbols
    d_cash = Decimal(repr(initial_capital))
    d_qty = [Decimal("0")] * n_syms
    d_cost = Decimal("0")
    d_funding = Decimal("0")
    d_turnover = Decimal("0")

    fee_rate = _dec(cost_cfg.fee_rate)
    taker_rate = _dec(cost_cfg.taker_fee_rate)
    spread_rate = _dec(cost_cfg.spread_rate)
    impact_rate = _dec(cost_cfg.impact_rate)

    filled_count = 0
    for iv in intervals:
        prev_marks = [_dec(float(ds.close[iv.start_index, s])) for s in range(n_syms)]
        eq_before = d_cash + sum(q * m for q, m in zip(d_qty, prev_marks, strict=True))
        period_start_val = max(eq_before, Decimal("1e-12"))

        for ev in iv.order_events:
            if float(ev.filled_quantity) == 0.0:
                continue
            filled_count += 1
            s = ev.symbol_index
            dq = _dec(float(ev.filled_quantity))
            p_fill = _dec(float(ds.open[iv.next_index, s]))
            assert ev.execution_price is not None
            assert abs(float(ev.execution_price) - float(p_fill)) <= 1e-12

            notional = abs(dq * p_fill)
            ref_vol = _dec(float(ds.volume[iv.next_index - 1, s]))
            ref_close = _dec(float(ds.close[iv.next_index - 1, s]))
            notional_part = notional / (ref_vol * ref_close)
            qty_part = abs(dq) / ref_vol
            participation = max(notional_part, qty_part)
            assert abs(participation - _dec(ev.participation_rate)) <= Decimal("1e-12")
            unit_cost = (
                fee_rate + taker_rate + spread_rate + impact_rate * participation.sqrt()
            )
            fill_cost = notional * unit_cost

            d_cash -= dq * p_fill + fill_cost
            d_cost += fill_cost
            d_qty[s] += dq
            d_turnover += notional / period_start_val

        # End-of-bar funding
        bar_funding = Decimal("0")
        curr_marks = [_dec(float(ds.close[iv.next_index, s])) for s in range(n_syms)]
        for s in range(n_syms):
            fr_s = _dec(float(ds.funding_rate[iv.next_index, s]))
            bar_funding -= d_qty[s] * curr_marks[s] * fr_s
        d_cash += bar_funding
        d_funding += bar_funding

        d_equity = d_cash + sum(q * m for q, m in zip(d_qty, curr_marks, strict=True))

        _assert_close_decimal(iv.cash_after, d_cash, label=f"cash@{iv.next_index}")
        _assert_close_decimal(
            iv.portfolio_value_after, d_equity, label=f"equity@{iv.next_index}"
        )
        _assert_close_decimal(
            iv.total_cost_after, d_cost, label=f"cost@{iv.next_index}"
        )
        _assert_close_decimal(
            iv.funding_pnl_after, d_funding, label=f"funding@{iv.next_index}"
        )
        _assert_close_decimal(
            iv.turnover_total_after, d_turnover, label=f"turnover@{iv.next_index}"
        )

    assert filled_count > 0
    assert all(abs(q) <= Decimal("1e-12") for q in d_qty), (
        f"non-flat terminal qty: {d_qty}"
    )
    assert all(abs(float(q)) <= 1e-12 for q in result.book.quantities)


def _cell(
    arm: str,
    scenario: str,
    *,
    ret: float = 0.05,
    sharpe: float = 1.2,
    dd: float = 0.04,
    complete: bool = True,
    terminal_flat: bool = True,
    execution_valid: bool = True,
    causality_passed: bool = True,
    ledger_reconciled: bool = True,
    stop_reason: str | None = None,
    invalid_reasons: tuple[str, ...] = (),
) -> CellSummary:
    return CellSummary(
        arm=arm,
        scenario=scenario,
        total_return=ret,
        annual_sharpe=sharpe,
        max_drawdown=dd,
        total_cost=100.0,
        funding_pnl=10.0,
        turnover_total=2.0,
        complete=complete,
        terminal_flat=terminal_flat,
        execution_valid=execution_valid,
        causality_passed=causality_passed,
        ledger_reconciled=ledger_reconciled,
        stop_reason=stop_reason,
        invalid_reasons=invalid_reasons,
    )


def test_decision_truth_table_and_nonrl_regime_gate_diagnostics() -> None:
    cells: dict[tuple[str, str], CellSummary] = {}
    for arm in protocol.ARMS:
        for sc in protocol.SCENARIOS:
            if arm == "ANC_REGIME_GATE_NONRL":
                cells[(arm, sc)] = _cell(arm, sc, ret=0.06, sharpe=1.5, dd=0.025)
            elif arm == "CORE7_SPARSE_PPO":
                cells[(arm, sc)] = _cell(arm, sc, ret=0.05, sharpe=1.2, dd=0.04)
            else:
                cells[(arm, sc)] = _cell(arm, sc)

    res_pass = decide(cells, evidence_eligible=True)
    assert res_pass["outcome"] == PASS_PARITY_ONLY
    assert res_pass["edge_claim"] is False
    assert res_pass["production_eligible"] is False
    assert res_pass["live_trading_authorized"] is False
    diag = res_pass["rl_vs_control_diagnostics"]
    assert isinstance(diag, dict)
    core7_vs_gate = diag["CORE7_SPARSE_PPO"]["ANC_REGIME_GATE_NONRL"]["BASE"]
    assert core7_vs_gate["nonrl_regime_gate_dominates_dd"] is True
    assert core7_vs_gate["nonrl_regime_gate_dominates_sharpe"] is True
    assert "ANC_BIDIR_75_25" in diag["ORTH6_CUSH_PPO"]

    # Roundtrip serialization
    d0 = cells[("CORE7_SPARSE_PPO", "BASE")]
    assert CellSummary.from_dict(d0.to_dict()) == d0

    # Stop reason -> STOP_FAIL_NO_RESCUE
    cells_stop = dict(cells)
    cells_stop[("ORTH6_CUSH_PPO", "HIGH_COST")] = _cell(
        "ORTH6_CUSH_PPO", "HIGH_COST", stop_reason="drawdown_stop"
    )
    assert decide(cells_stop, evidence_eligible=True)["outcome"] == STOP_FAIL_NO_RESCUE

    # Missing or invalid -> INVALID
    for bad_override in (
        {"terminal_flat": False},
        {"causality_passed": False},
        {"ledger_reconciled": False},
        {"execution_valid": False},
        {"invalid_reasons": ("bad_cell",)},
    ):
        cells_inv = dict(cells)
        cells_inv[("ANC_BIDIR_75_25", "BASE")] = _cell(
            "ANC_BIDIR_75_25", "BASE", **bad_override
        )
        assert decide(cells_inv, evidence_eligible=True)["outcome"] == INVALID

    cells_missing = dict(cells)
    cells_missing.pop(("CASH", "DELAY_1BAR"))
    assert decide(cells_missing, evidence_eligible=True)["outcome"] == INVALID


def test_runner_fail_closed_refusals_causality_leak_trap_and_synthetic_e2e(
    tmp_path: Path,
) -> None:
    ds = _make_alternating_dataset(n_bars=50, pairs=("BTCUSDT", "ETHUSDT"), seed=88)
    digest = protocol.protocol_digest()

    with pytest.raises(ValueError, match="authorization_url is required"):
        run_study(ds, output_dir=tmp_path / "no_url", expected_protocol_digest=digest)

    with pytest.raises(ValueError, match="https://github.com/shuntatsu/trade_rl"):
        run_study(
            ds,
            output_dir=tmp_path / "bad_url",
            expected_protocol_digest=digest,
            authorization_url="https://evil.example/comment-1",
        )

    ds_syn = protocol.mark_synthetic_dataset(ds)
    with pytest.raises(ValueError, match="fit_stop_index <= start_index"):
        run_study(
            ds_syn,
            output_dir=tmp_path / "bad_fit",
            expected_protocol_digest=digest,
            allow_synthetic_unauthenticated=True,
            start_index=8,
            stop_index=45,
            fit_stop_index=10,
        )

    with pytest.raises(ValueError, match="ROSTER"):
        run_study(
            ds_syn,
            output_dir=tmp_path / "incomplete_roster",
            expected_protocol_digest=digest,
            allow_synthetic_unauthenticated=True,
            require_full_roster=True,
            start_index=8,
            stop_index=45,
        )

    ds_future = protocol.mark_synthetic_dataset(
        _make_alternating_dataset(
            n_bars=12,
            pairs=("BTCUSDT", "ETHUSDT"),
            start_iso="2026-07-30T12:00:00",
        )
    )
    with pytest.raises(ValueError, match="Pre-FS1 embargo"):
        run_study(
            ds_future,
            output_dir=tmp_path / "embargo",
            expected_protocol_digest=digest,
            allow_synthetic_unauthenticated=True,
            start_index=2,
            stop_index=10,
        )

    # Causality probe passes on clean causal signals and catches injected leakage
    probe = run_causality_probe(
        ds, cut_index=25, policy_mode="anc_regime_gate_nonrl", start_index=5
    )
    assert (
        probe.passed
        and probe.signal_max_abs_diff == 0.0
        and probe.proposal_max_abs_diff == 0.0
    )

    # Injected pack leakage (contaminated row at the cut) is detected by the helper
    from trade_rl.evaluation.core7_mtf_study import runner as study_runner

    pack_clean = build_mtf6_signal_pack(ds)
    leaked_c7 = np.asarray(pack_clean.core7_signals, dtype=np.float64).copy()
    leaked_c7[25, 0, 0] += 0.25  # cut_index contaminated
    pack_leaked = replace(pack_clean, core7_signals=leaked_c7)
    assert study_runner._max_prefix_pack_diff(
        pack_clean, pack_leaked, cut_index=25
    ) == pytest.approx(0.25)

    # Full 5-arm synthetic E2E + write-once refusal
    out_dir = tmp_path / "e2e_study"
    res = run_study(
        ds_syn,
        output_dir=out_dir,
        expected_protocol_digest=digest,
        allow_synthetic_unauthenticated=True,
        start_index=8,
        stop_index=48,
        fit_stop_index=8,
        rebalance_every_bars=6,
    )
    assert res["outcome"] == SYNTHETIC_ONLY_NOT_EVIDENCE
    assert res["edge_claim"] is False
    assert res["production_eligible"] is False
    assert res["live_trading_authorized"] is False
    assert (out_dir / "study-manifest.json").is_file()
    assert (out_dir / "decision.json").is_file()
    cells = res["cells"]
    assert isinstance(cells, dict)
    assert len(cells) == len(protocol.ARMS) * len(protocol.SCENARIOS)

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        run_study(
            ds_syn,
            output_dir=out_dir,
            expected_protocol_digest=digest,
            allow_synthetic_unauthenticated=True,
            start_index=8,
            stop_index=48,
            fit_stop_index=8,
        )


def _git(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args], cwd=ROOT, capture_output=True, text=True, check=False
    )


def _pinned_python_files(head: str) -> set[str]:
    listing = _git("ls-tree", "-r", "--name-only", head, "trade_rl", "tools")
    assert listing.returncode == 0, listing.stderr
    return {path for path in listing.stdout.split() if path.endswith(".py")}


def _module_name(path: str) -> str:
    return path[: -len(".py")].replace("/", ".").removesuffix(".__init__")


def _module_file(module: str, files: set[str]) -> str | None:
    base = module.replace(".", "/")
    for candidate in (f"{base}.py", f"{base}/__init__.py"):
        if candidate in files:
            return candidate
    return None


def _imported_modules(path: str, source: str) -> set[str]:
    package = (
        _module_name(path)
        if path.endswith("/__init__.py")
        else _module_name(path).rpartition(".")[0]
    )
    found: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                parts = package.split(".")
                parts = parts[: len(parts) - (node.level - 1)]
                module = ".".join([*parts, *([node.module] if node.module else [])])
            else:
                module = node.module or ""
            found.add(module)
            found.update(f"{module}.{alias.name}" for alias in node.names)
    return {name for name in found if name.split(".")[0] in {"trade_rl", "tools"}}


def _bound_code_paths(head: str) -> list[str]:
    files = _pinned_python_files(head)
    pending = [
        path
        for path in files
        if any(path == entry or path.startswith(entry) for entry in ENTRY_POINTS)
    ]
    assert pending, "no Core7 MTF entry point found at packet head"
    seen: set[str] = set()
    while pending:
        path = pending.pop()
        if path in seen:
            continue
        seen.add(path)
        parts = _module_name(path).split(".")
        for depth in range(1, len(parts)):
            init = "/".join(parts[:depth]) + "/__init__.py"
            if init in files:
                pending.append(init)
        source = _git("show", f"{head}:{path}")
        assert source.returncode == 0, source.stderr
        for module in _imported_modules(path, source.stdout):
            target = _module_file(module, files)
            if target is not None:
                pending.append(target)
    return sorted(seen)


def test_prior_v1_packet_is_immutable_history_and_unreviewed() -> None:
    """The v1 packet (protocol v1) is superseded by v2 and never re-pinned or edited."""
    record: dict[str, Any] = json.loads(PACKET_V1_PATH.read_text(encoding="utf-8"))
    assert assurance_digest(record) == (
        "ce369ac6d66687ba0c822c56bcfe5930c144dd287edda19894fb1e0d7cd70ea6"
    )
    assert record["identity"]["implementation_head"] == (
        "edc910ae4f8030523b4b5e39d63fa1cf7d006450"
    )
    verified = evaluate_assurance(record)
    assert verified.status == "UNREVIEWED"
    assert verified.errors == ()
    assert verified.economic_execution_authorized is False


def test_prior_v2_packet_is_immutable_history_and_unreviewed() -> None:
    """The v2 packet (protocol v2) is superseded by v3 and never re-pinned or edited."""
    record: dict[str, Any] = json.loads(PACKET_V2_PATH.read_text(encoding="utf-8"))
    assert assurance_digest(record) == (
        "474d77259f177694b6d5e74eca313bf9fa844ab6f7c1898c085c686e263ed057"
    )
    assert record["identity"]["implementation_head"] == (
        "8f49dfa024cf796c65949962aa55181a2a872bf1"
    )
    verified = evaluate_assurance(record)
    assert verified.status == "UNREVIEWED"
    assert verified.errors == ()
    assert verified.economic_execution_authorized is False


def test_prior_v3_packet_is_immutable_history_and_unreviewed() -> None:
    """The v3 packet (protocol v3) is superseded by v4 and never re-pinned or edited."""
    record: dict[str, Any] = json.loads(PACKET_V3_PATH.read_text(encoding="utf-8"))
    assert assurance_digest(record) == (
        "e66ef4f381318da2162b74fc20bf1042f9416788f4f33472f1de19fe2b4b35bc"
    )
    assert record["identity"]["implementation_head"] == (
        "9f95a8faf388e637305a0cc915fb0dadede7ae41"
    )
    verified = evaluate_assurance(record)
    assert verified.status == "UNREVIEWED"
    assert verified.errors == ()
    assert verified.economic_execution_authorized is False


def test_prior_v4_packet_is_immutable_history_and_unreviewed() -> None:
    """The frozen v4 packet must never be silently rewritten to pass new source."""
    record: dict[str, Any] = json.loads(PACKET_V4_PATH.read_text(encoding="utf-8"))
    assert assurance_digest(record) == (
        "c7bba24b8a3c3d1cd79cc16cc3c4f8311d9143edd0aa0602ae6f64d9234708e5"
    )
    assert record["identity"]["implementation_head"] == (
        "58d76a19a7265e24231ec69be7c7ec1f20394e1d"
    )
    verified = evaluate_assurance(record)
    assert verified.status == "UNREVIEWED"
    assert verified.errors == ()
    assert verified.economic_execution_authorized is False


def test_core7_mtf_packet_stays_unreviewed_and_pinned() -> None:
    record: dict[str, Any] = json.loads(PACKET_PATH.read_text(encoding="utf-8"))
    assert PINNED_PROTOCOL_DIGEST in record["mechanism"]["authorities"]["evidence"]
    assert protocol.protocol_digest() == PINNED_PROTOCOL_DIGEST
    assert assurance_digest(record) == PINNED_PACKET_DIGEST

    verified = evaluate_assurance(record)
    assert verified.status == "UNREVIEWED"
    assert verified.errors == ()
    assert verified.economic_execution_authorized is False
    assert record["claims"]["next_action_class"] == "synthetic_verification_only"
    assert record["claims"]["claim_level"] == "software_validity"
    assert record["claims"]["production_eligible"] is False
    assert record["claims"]["live_trading_authorized"] is False
    assert record["evidence"]["evidence_level"] == "software_validity"
    assert "anc_regime_gate_nonrl" in record["evidence"]["controls"]

    head = str(record["identity"]["implementation_head"])
    assert record["identity"]["protocol_head"] == head
    assert _git("cat-file", "-e", f"{head}^{{commit}}").returncode == 0

    def changed_since_head(paths: Any) -> list[str]:
        diff_committed = _git("diff", "--name-only", head, "HEAD", "--", *paths)
        assert diff_committed.returncode == 0, diff_committed.stderr
        diff_worktree = _git("diff", "--name-only", head, "--", *paths)
        assert diff_worktree.returncode == 0, diff_worktree.stderr
        return sorted(
            set(diff_committed.stdout.split()) | set(diff_worktree.stdout.split())
        )

    violations = evaluate_pin(
        head=head,
        closed_run=None,
        closure_at=_bound_code_paths,
        changed_since_head=changed_since_head,
        read_blob=git_blob_reader(ROOT),
        is_ancestor_of_head=lambda _: False,
        packet_head_at=lambda _, __: head,
    )
    assert violations == [], violations
    spec_text = SPEC_PATH.read_text(encoding="utf-8")
    assert PINNED_PROTOCOL_DIGEST in spec_text
    assert PINNED_PACKET_DIGEST in spec_text
    assert head in spec_text
