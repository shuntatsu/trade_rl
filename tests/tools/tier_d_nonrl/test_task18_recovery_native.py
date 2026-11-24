"""Closed two-arm native recovery conformance, never Task18 market evidence.

The fixture substitutes a distinct synthetic recipe only inside the test
processes. No caller-supplied Dataset, external market file or economic run is
accepted. Linux uses one owned child (no descendant-containment claim);
Windows uses the maintained native Job Object boundary.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
from unittest.mock import patch

import numpy as np

from tools.tier_d_nonrl import task18_recovery as recovery
from tools.tier_d_nonrl.task18_receipts import (
    ArmReceipt,
    ExternalArmVerificationReport,
    ExternalSourceVerificationReport,
    RunReceipt,
    SourceBinding,
    SourceFilePin,
)
from tools.tier_d_nonrl.task18_recipe import build_task18_protocol
from tools.tier_d_nonrl.tier_d_process_containment import launch_contained_process
from trade_rl.artifacts.atomic_write import atomic_write_bytes
from trade_rl.artifacts.canonical import canonical_json_bytes
from trade_rl.artifacts.verified_file import file_digest_and_size, read_verified_bytes
from trade_rl.data.market import MarketDataset
from trade_rl.evaluation.decision_value_ledger_oracle import (
    verify_paired_economic_ledger,
)
from trade_rl.evaluation.replay import run_shared_portfolio_replay
from trade_rl.simulation.execution import ExecutionCostConfig
from trade_rl.strategies.allocation.buy_and_hold import BuyAndHoldPortfolioPolicy
from trade_rl.strategies.portfolio_adapter import CashPortfolioPolicy

_SOURCE = "a" * 40  # Artificial binding, not a real repository attestation.
_RUN = "synthetic-native-recovery"
_SUCCESSOR_RUN = "synthetic-native-recovery-successor"
_WORKER_MODULE = "tests.tools.tier_d_nonrl.test_task18_recovery_native"
_SCHEMA = "synthetic_native_recovery_fixture_v1"
_FROZEN = canonical_json_bytes(
    {
        "schema": _SCHEMA,
        "symbol": "ARTIFICIAL",
        "prices": [100, 100, 110, 110, 110, 110, 110, 110],
        "initial_capital": 1000.0,
        "start": 0,
        "stop": 7,
        "nonrl_weight": 0.2,
        "execution_cost": "zero",
        "cash": "flat",
    }
)


def _sha(raw: bytes) -> str:
    return sha256(raw).hexdigest()


def _write(path: Path, value: object) -> bytes:
    raw = canonical_json_bytes(value)
    atomic_write_bytes(path, raw)
    return raw


def _fixture(raw: bytes):
    # Refuse arbitrary input even if someone can generate a new digest.
    assert raw == _FROZEN
    frozen = json.loads(raw)
    close = np.asarray(frozen["prices"], dtype=np.float64)[:, None]
    opening = np.vstack((close[0], close[:-1]))
    market = MarketDataset(
        dataset_id="0" * 64,
        symbols=(frozen["symbol"],),
        timestamps=np.datetime64("2026-01-01T00:00:00", "ns")
        + np.arange(len(close)) * np.timedelta64(1, "h"),
        features=np.zeros((len(close), 1, 1), dtype=np.float32),
        global_features=np.zeros((len(close), 1), dtype=np.float32),
        open=opening,
        high=np.maximum(opening, close),
        low=np.minimum(opening, close),
        close=close,
        volume=np.full_like(close, 1_000_000.0),
        funding_rate=np.zeros_like(close),
        tradable=np.ones_like(close, dtype=np.bool_),
        feature_available=np.ones((len(close), 1, 1), dtype=np.bool_),
        feature_names=("artificial",),
        global_feature_names=("artificial_global",),
        periods_per_year=8760,
    ).with_content_identity({"fixture": _SCHEMA})
    binding = SourceBinding(
        _SOURCE,
        (
            SourceFilePin(
                "tests/tools/tier_d_nonrl/test_task18_recovery_native.py",
                _sha(Path(__file__).read_bytes()),
            ),
        ),
        _sha(b"synthetic binding only"),
    )
    source_bytes = canonical_json_bytes(
        {"scope": _SCHEMA, "source_authentication": False}
    )
    report = ExternalSourceVerificationReport(
        _SOURCE,
        True,
        binding.files,
        binding.import_origin_manifest_sha256,
        1,
        _sha(source_bytes),
    )
    base = build_task18_protocol()
    pair = replace(
        base.pairs[0], configuration="ARTIFICIAL_BH", symbol=market.symbols[0]
    )
    arms = tuple(
        replace(a, configuration=pair.configuration, symbol=pair.symbol)
        for a in base.arms[:2]
    )
    protocol = replace(
        base,
        recipe_schema=_SCHEMA,
        scope="SYNTHETIC_RECOVERY_CONFORMANCE_ONLY",
        roster_dimension_note="One artificial pair; no historical Dataset.",
        historical_reference=replace(
            base.historical_reference,
            source_commit=_SOURCE,
            protocol_source_path=binding.files[0].path,
            protocol_source_sha256=binding.files[0].sha256,
            protocol_digest=_sha(raw),
            dataset_id=market.dataset_id,
            manifest_sha256=_sha(raw),
            arrays_sha256=_sha(close.tobytes()),
            source_file_pins=binding.files,
        ),
        dataset=replace(
            base.dataset,
            row_count=len(close),
            symbols=market.symbols,
            feature_names=market.feature_names,
            features_per_symbol=1,
            consumed_development_only=False,
        ),
        configurations=(replace(base.configurations[0], name=pair.configuration),),
        scenarios=base.scenarios[:1],
        pairs=(pair,),
        arms=arms,
        index_plan=replace(
            base.index_plan,
            actor_start_inclusive=frozen["start"],
            actor_stop_exclusive=frozen["stop"] - 1,
            terminal_stop_inclusive=frozen["stop"],
            terminal_advance_count=1,
            native_advances_per_arm=frozen["stop"] - frozen["start"],
            total_native_advances=2 * (frozen["stop"] - frozen["start"]),
            vintage_start_indices=(),
            ridge_vintage_count=0,
            public_cost_preparation_fit_count=0,
        ),
        economics=replace(
            base.economics,
            initial_capital_per_symbol_usdt=frozen["initial_capital"],
            base_fee_rate=0.0,
            base_spread_rate=0.0,
            impact_coefficient=0.0,
        ),
        gates=replace(
            base.gates,
            required_scenarios=(pair.scenario,),
            claim_boundary="Synthetic software conformance only.",
        ),
        nonrl_policy="synthetic buy-and-hold",
        cash_policy="synthetic flat cash",
    )
    ref = protocol.historical_reference
    run = RunReceipt(
        run_id=_RUN,
        protocol_digest=protocol.adapted_protocol_digest,
        historical_protocol_digest=ref.protocol_digest,
        dataset_id=ref.dataset_id,
        manifest_sha256=ref.manifest_sha256,
        arrays_sha256=ref.arrays_sha256,
        source_binding=binding,
        source_report=report,
        runtime_completed=False,
        arms=tuple(
            ArmReceipt(
                arm_id=arm.arm_id,
                pair_id=arm.pair_id,
                role=arm.role,
                configuration=arm.configuration,
                symbol=arm.symbol,
                scenario=arm.scenario,
                protocol_digest=protocol.adapted_protocol_digest,
                dataset_id=ref.dataset_id,
                manifest_sha256=ref.manifest_sha256,
                arrays_sha256=ref.arrays_sha256,
                source_binding_sha256=binding.manifest_sha256,
                status="RUNNING" if arm.role == "NONRL" else "NOT_ATTEMPTED",
                initial_equity_usdt=None,
                terminal_equity_usdt=None,
                max_drawdown=None,
                terminal_flat=None,
                pending_order_count=None,
                reason_code="RUN_IN_PROGRESS" if arm.role == "NONRL" else "NOT_STARTED",
            )
            for arm in arms
        ),
    )
    return frozen, market, protocol, run, source_bytes


def _execute(
    root: Path,
    arm: ArmReceipt,
    raw: bytes,
    *,
    run_id: str,
    predecessor_handoff: str | None = None,
) -> None:
    frozen, market, protocol, run, _ = _fixture(raw)
    assert arm in run.arms
    if arm.role == "NONRL":
        policy = BuyAndHoldPortfolioPolicy(
            symbols=market.symbols, gross_budget=frozen["nonrl_weight"]
        )
    else:
        assert arm.role == "CASH"
        policy = CashPortfolioPolicy(symbols=market.symbols)
    cost = ExecutionCostConfig.zero()
    result = run_shared_portfolio_replay(
        market,
        policy,
        start_index=frozen["start"],
        stop_index=frozen["stop"],
        initial_capital=frozen["initial_capital"],
        execution_cost=cost,
        settle_terminal_position=True,
        capture_ledger_evidence=True,
    )
    assert verify_paired_economic_ledger(
        result,
        market,
        expected_intervals=protocol.index_plan.native_advances_per_arm,
        initial_capital=frozen["initial_capital"],
        expected_start_index=frozen["start"],
        execution_cost=cost,
    )
    wrapped = result.ledger_evidence
    assert wrapped is not None
    ledger = wrapped.intervals
    fills = sum(
        event.filled_quantity != 0
        for iv in ledger.intervals
        for event in iv.order_events
    )
    assert (
        ledger.final_total_cost
        == ledger.final_funding_pnl
        == ledger.final_borrow_cost
        == 0.0
    )
    assert ledger.terminal_exact_quantities == ("0",)
    assert not ledger.active_order_remainders
    if arm.role == "CASH":
        assert fills == 0 and ledger.final_portfolio_value == frozen["initial_capital"]
    else:
        assert fills >= 2 and ledger.final_portfolio_value == 1020.0
    folder = root / "arms" / arm.arm_id
    folder.mkdir(parents=True, exist_ok=False)
    ledger_raw = _write(folder / "native-ledger.json", wrapped.to_mapping())
    result_raw = _write(
        folder / "result.json",
        {
            "scope": _SCHEMA,
            "run_id": run_id,
            "predecessor_handoff_sha256": predecessor_handoff,
            "arm_id": arm.arm_id,
            "advances": len(ledger.intervals),
            "fills": fills,
            "terminal_equity": ledger.final_portfolio_value,
        },
    )
    verifier_raw = _write(
        folder / "verification.json",
        {
            "scope": _SCHEMA,
            "ledger_sha256": _sha(ledger_raw),
            "source_backed_reconstruction": True,
            "zero_cost_oracle": True,
        },
    )
    assert run.source_report is not None
    evidence = ExternalArmVerificationReport(
        run_id=run_id,
        protocol_digest=arm.protocol_digest,
        source_report_binding_sha256=run.source_report.binding_sha256,
        arm_id=arm.arm_id,
        pair_id=arm.pair_id,
        role=arm.role,
        configuration=arm.configuration,
        symbol=arm.symbol,
        scenario=arm.scenario,
        dataset_id=arm.dataset_id,
        native_ledger_sha256=_sha(ledger_raw),
        arm_result_sha256=_sha(result_raw),
        verification_report_sha256=_sha(verifier_raw),
        expected_advances=protocol.index_plan.native_advances_per_arm,
        observed_advances=len(ledger.intervals),
        initial_equity_usdt=frozen["initial_capital"],
        terminal_equity_usdt=ledger.final_portfolio_value,
        max_drawdown=ledger.final_max_drawdown,
        terminal_settlement_complete=True,
        terminal_flat=True,
        pending_order_count=0,
        cost_oracle_passed=True,
    )
    complete = replace(
        arm,
        status="COMPLETED",
        reason_code=None,
        initial_equity_usdt=evidence.initial_equity_usdt,
        terminal_equity_usdt=evidence.terminal_equity_usdt,
        max_drawdown=evidence.max_drawdown,
        terminal_flat=True,
        pending_order_count=0,
        native_ledger_sha256=evidence.native_ledger_sha256,
        arm_result_sha256=evidence.arm_result_sha256,
        external_report=evidence,
    )
    # Receipt is published last, leaving the run slot RUNNING on disk.
    _write(folder / "receipt.json", complete.to_mapping())


def _prepare(base: Path) -> None:
    source = base / "stopped"
    source.mkdir()
    _, _, _, run, source_bytes = _fixture(_FROZEN)
    atomic_write_bytes(source / "frozen-plan.bin", _FROZEN)
    atomic_write_bytes(source / "source-verification.json", source_bytes)
    _write(source / "stopped-run.json", run.to_mapping())
    _execute(source, run.arms[0], _FROZEN, run_id=_RUN)
    # Deterministic stop point: durable completed NONRL receipt, before its
    # status promotion or any CASH dispatch. The parent owns this child.
    atomic_write_bytes(base / "ready", b"completed NONRL published")
    while True:
        time.sleep(0.1)


def _resume(base: Path, digest: str) -> None:
    source = base / "handoff"
    raw = read_verified_bytes(
        source / "frozen-plan.bin",
        expected_digest=_sha(_FROZEN),
        expected_size_bytes=len(_FROZEN),
        field="closed synthetic frozen plan",
    )
    _, _, protocol, run, _ = _fixture(raw)
    with patch.object(recovery, "build_task18_protocol", return_value=protocol):
        plan = recovery.load_staged_recovery(
            source,
            expected_source_commit=_SOURCE,
            expected_frozen_plan_sha256=_sha(_FROZEN),
            expected_handoff_sha256=digest,
        )
    assert plan.execution_authorized is False  # Only this closed fixture runs.
    assert plan.completed_arm_ids == (run.arms[0].arm_id,)
    assert plan.pending_arm_ids == (run.arms[1].arm_id,)
    output = base / "resumed"
    output.mkdir()
    for arm_id in plan.pending_arm_ids:
        arm = next(a for a in run.arms if a.arm_id == arm_id)
        _execute(
            output, arm, raw, run_id=_SUCCESSOR_RUN, predecessor_handoff=plan.sha256
        )


def _snapshot(root: Path) -> dict[str, bytes]:
    return {
        p.relative_to(root).as_posix(): p.read_bytes()
        for p in root.rglob("*")
        if p.is_file()
    }


def test_native_stopped_nonrl_receipt_is_retained_and_only_cash_resumes(tmp_path):
    base = tmp_path.resolve()
    args = [sys.executable, "-m", _WORKER_MODULE, "prepare", str(base)]
    cwd = Path(__file__).resolve().parents[3]
    with (base / "prepare.log").open("wb") as log:
        owned = (
            launch_contained_process(
                args, cwd=cwd, stdout=log, stderr=subprocess.STDOUT
            )
            if os.name == "nt"
            else None
        )
        child = (
            owned.process
            if owned is not None
            else subprocess.Popen(args, cwd=cwd, stdout=log, stderr=subprocess.STDOUT)
        )
        try:
            deadline = time.monotonic() + 30
            while not (base / "ready").is_file():
                assert child.poll() is None, (base / "prepare.log").read_text()
                assert time.monotonic() < deadline, (
                    "native fixture did not publish its receipt"
                )
                time.sleep(0.025)
            assert child.poll() is None
        finally:
            if owned is not None:
                owned.close(timeout=15)
            else:
                # The fixture never spawns descendants; no Linux tree proof.
                child.kill()
                child.wait(timeout=15)
        assert child.returncode not in (None, 0)
    source = base / "stopped"
    before = _snapshot(source)
    _, _, protocol, run, _ = _fixture(_FROZEN)
    pins = tuple(
        recovery.RecoveryFilePin(path, *file_digest_and_size(source / path))
        for path in sorted(before)
    )
    with patch.object(recovery, "build_task18_protocol", return_value=protocol):
        plan = recovery.inspect_stopped_recovery(
            source,
            pins,
            expected_source_commit=_SOURCE,
            expected_frozen_plan_sha256=_sha(_FROZEN),
        )
        assert plan.recovered_arm_ids == (run.arms[0].arm_id,)
        recovery.stage_stopped_recovery(
            source,
            base / "handoff",
            pins,
            expected_source_commit=_SOURCE,
            expected_frozen_plan_sha256=_sha(_FROZEN),
            expected_handoff_sha256=plan.sha256,
        )
    handoff = _snapshot(base / "handoff")
    assert (
        handoff["arms/" + run.arms[0].arm_id + "/receipt.json"]
        == before["arms/" + run.arms[0].arm_id + "/receipt.json"]
    )
    assert _snapshot(source) == before
    # Remove access to the original path before launching a fresh Python process.
    source.rename(base / "preserved-original")
    resumed = subprocess.run(
        [sys.executable, "-m", args[2], "resume", str(base), plan.sha256],
        cwd=cwd,
        capture_output=True,
        timeout=30,
        check=False,
    )
    assert resumed.returncode == 0, resumed.stderr.decode(errors="replace")
    assert _snapshot(base / "preserved-original") == before
    assert _snapshot(base / "handoff") == handoff
    assert {p.name for p in (base / "resumed" / "arms").iterdir()} == {
        run.arms[1].arm_id
    }
    result = json.loads(
        (base / "resumed" / "arms" / run.arms[1].arm_id / "result.json").read_bytes()
    )
    assert result == {
        "scope": _SCHEMA,
        "run_id": _SUCCESSOR_RUN,
        "predecessor_handoff_sha256": plan.sha256,
        "arm_id": run.arms[1].arm_id,
        "advances": 7,
        "fills": 0,
        "terminal_equity": 1000.0,
    }

    old_receipt = json.loads(before[f"arms/{run.arms[0].arm_id}/receipt.json"])
    new_receipt = json.loads(
        (base / "resumed" / "arms" / run.arms[1].arm_id / "receipt.json").read_bytes()
    )
    assert old_receipt["external_report"]["run_id"] == _RUN
    assert new_receipt["external_report"]["run_id"] == _SUCCESSOR_RUN
    assert new_receipt["protocol_digest"] == old_receipt["protocol_digest"]


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "prepare":
        _prepare(Path(sys.argv[2]))
    elif len(sys.argv) == 4 and sys.argv[1] == "resume":
        _resume(Path(sys.argv[2]), sys.argv[3])
    else:
        raise SystemExit("closed synthetic test worker only")
