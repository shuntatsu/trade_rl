"""Type-check a downstream consumer of a non-editable, installed core wheel."""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

_IMPORTS = """\
from datetime import datetime, UTC
from fractions import Fraction
from pathlib import Path
from typing import Any, assert_type
import numpy as np

from trade_rl.data import MarketDataset
from trade_rl.data.build import MarketDatasetBuilder
from trade_rl.data.contracts import FeatureKind, FeatureSpec, InstrumentContract, MarketBuildConfig
from trade_rl.data.features import calculate_feature_events
from trade_rl.data.source import MarketDataSource, RawMarketSeries
from trade_rl.data.timestamps import utc_timestamp_ns
from trade_rl.data.features.clocked_overlay import ClockedNativeFeatureGroup, ClockedNativeSource, append_clocked_native_feature_groups
from trade_rl.data.native_clock import NativeBarClock
from trade_rl.evaluation.replay import (
    SharedCashPortfolioReplayLedgerEvidence, SharedCashPortfolioReplayResult,
    run_shared_portfolio_replay,
)
from trade_rl.evaluation.clocked_native_replay import NativePublicationSlot, ClockedNativeReplayResult, clocked_native_decision_schedule, run_clocked_native_portfolio_replay
from trade_rl.evaluation.replay_exposure import ReplayExposureEvidence, mean_pre_interval_gross_from_ledger
from trade_rl.evaluation.runs.external_provenance import (
    ExternalPackageInput, ExternalSourceProvenance, PackageFilePin, PackageSourcePin,
    verify_external_source_provenance,
)
from trade_rl.risk import PreTradeRisk, PreTradeRiskConfig
from trade_rl.simulation import ExecutionCostConfig, runtime_execution_policy_digest
from trade_rl.simulation.quantities import parse_quantity
from trade_rl.simulation.diagnostics.fill_cash import NativeFillCashObservation
from trade_rl.simulation.training_identity import (
    training_execution_config_payload, training_execution_config_digest,
    training_risk_config_payload, training_risk_config_digest,
)
from trade_rl.strategies.portfolio import (
    PortfolioDecisionSnapshot, PortfolioPolicy, PortfolioTargetProposal,
)

class ExternalPolicy:
    def decide_portfolio(self, snapshot: PortfolioDecisionSnapshot) -> PortfolioTargetProposal:
        quantity = assert_type(parse_quantity(snapshot.account.exact_quantities[0]), Fraction)
        return PortfolioTargetProposal((0.0,), hold_current_quantity_symbols=(0,) if quantity else ())

policy: PortfolioPolicy = ExternalPolicy()
"""
_POSITIVE = (
    _IMPORTS
    + """
def consume(source: MarketDataSource, instruments: tuple[InstrumentContract, ...], raw: RawMarketSeries) -> None:
    spec = FeatureSpec("ret", FeatureKind.LOG_RETURN)
    config = MarketBuildConfig(base_timeframe="1h", features=(spec,))
    market = assert_type(MarketDatasetBuilder(config).build(source, instruments), MarketDataset)
    assert raw.funding_available is not None
    events = calculate_feature_events(
        spec, open_price=raw.open, high=raw.high, low=raw.low, close=raw.close,
        volume=raw.volume, funding_rate=raw.funding_rate, funding_available=raw.funding_available,
        row_present=np.ones(raw.close.shape, dtype=np.bool_), active=raw.tradable,
    )
    assert_type(events, tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], np.ndarray[Any, Any]])
    result = assert_type(run_shared_portfolio_replay(
        market, policy, start_index=0, stop_index=2, execution_cost=ExecutionCostConfig(),
        risk=PreTradeRisk(PreTradeRiskConfig()), capture_ledger_evidence=True,
    ), SharedCashPortfolioReplayResult)
    assert_type(result.ledger_evidence, SharedCashPortfolioReplayLedgerEvidence | None)
    assert_type(run_shared_portfolio_replay(market, policy, start_index=0, stop_index=2, settle_terminal_position=True, actor_decision_indices=(0,)), SharedCashPortfolioReplayResult)
    if result.ledger_evidence is not None:
        assert_type(result.ledger_evidence.exposure_evidence, ReplayExposureEvidence | None)
    assert_type(mean_pre_interval_gross_from_ledger(b"", expected_ledger_digest="a"*64, expected_dataset_id="b"*64, expected_execution_policy_digest="c"*64, expected_symbols=("A",), expected_start_index=0, expected_stop_index=2), float)
    assert_type(runtime_execution_policy_digest(ExecutionCostConfig(), funding_price_rate_present=False), str)
def external_sources(core: PackageSourcePin, research: PackageSourcePin, wheel: Path, package: Path) -> None:
    source = assert_type(ExternalPackageInput(core, wheel, package), ExternalPackageInput)
    other = ExternalPackageInput(research, wheel, package)
    result = assert_type(verify_external_source_provenance(
        source, other, runtime_config={"fee": 0.0},
        expected_core_pin_digest="a"*64, expected_research_pin_digest="b"*64,
        expected_runtime_config_digest="c"*64,
    ), ExternalSourceProvenance)
    assert_type(result.digest, str)
    assert_type(core.files, tuple[PackageFilePin, ...])
def clock_and_training(market: MarketDataset, native: ClockedNativeSource, clock: NativeBarClock) -> None:
    spec = FeatureSpec("higher", FeatureKind.LOG_RETURN)
    group = ClockedNativeFeatureGroup(clock, (native,), (spec,))
    assert_type(append_clocked_native_feature_groups(market, groups=(group,)), MarketDataset)
    assert_type(clocked_native_decision_schedule(market, group, start_index=0, stop_index=2), tuple[NativePublicationSlot, ...])
    scheduled = assert_type(run_clocked_native_portfolio_replay(market, policy, decision_group=group, start_index=0, stop_index=2, initial_capital=1000.0, execution_cost=ExecutionCostConfig()), ClockedNativeReplayResult)
    assert_type(run_clocked_native_portfolio_replay(market, policy, decision_group=group, start_index=0, stop_index=2, initial_capital=1000.0, execution_cost=ExecutionCostConfig(), execution_mode="linear_carry", terminal_reserve_bars=1, continue_pending_orders=True), ClockedNativeReplayResult)
    assert_type(scheduled.replay, SharedCashPortfolioReplayResult)
    assert_type(scheduled.policy_decision_indices, tuple[int, ...])
    assert_type(scheduled.slots, tuple[NativePublicationSlot, ...])
    assert_type(scheduled.slots[0].close_ns, int)
    assert_type(scheduled.slots[0].available_at_ns, int)
    assert_type(scheduled.slots[0].decision_index, int | None)
    assert_type(scheduled.slots[0].decision_time_ns, int | None)
    assert_type(training_execution_config_payload(ExecutionCostConfig()), dict[str, object])
    assert_type(training_execution_config_digest(ExecutionCostConfig()), str)
    assert_type(training_risk_config_payload(PreTradeRiskConfig()), dict[str, object])
    assert_type(training_risk_config_digest(PreTradeRiskConfig()), str)
    assert_type(utc_timestamp_ns(datetime(2024, 1, 1, tzinfo=UTC)), int)
    assert_type(utc_timestamp_ns(np.datetime64("2024-01-01")), int)
    rows: list[NativeFillCashObservation] = []
    assert_type(run_shared_portfolio_replay(market, policy, start_index=0, stop_index=2, fill_cash_observer=rows.append), SharedCashPortfolioReplayResult)
    assert_type(rows[0].fills[0].quantity_after, str)
"""
)
_NEGATIVE = (
    _IMPORTS
    + """
class WrongPolicy:
    def decide_portfolio(self, snapshot: PortfolioDecisionSnapshot) -> int:
        return 1

bad_policy: PortfolioPolicy = WrongPolicy()  # expect: assignment
parse_quantity(1.5)  # expect: arg-type
PortfolioTargetProposal(("wrong",))  # expect: arg-type
FeatureSpec("ret", FeatureKind.LOG_RETURN, lookback="wrong")  # expect: arg-type
ExecutionCostConfig(fee_rate="wrong")  # expect: arg-type
runtime_execution_policy_digest(ExecutionCostConfig(), funding_price_rate_present="unknown")  # expect: arg-type
runtime_execution_policy_digest(ExecutionCostConfig())  # expect: call-arg
mean_pre_interval_gross_from_ledger("bad", expected_ledger_digest="a"*64, expected_dataset_id="b"*64, expected_execution_policy_digest="c"*64, expected_symbols=("A",), expected_start_index=0, expected_stop_index=2)  # expect: arg-type
PreTradeRiskConfig(max_gross="wrong")  # expect: arg-type
PackageFilePin("file.py", "bad", "a"*64)  # expect: arg-type
training_execution_config_digest(PreTradeRiskConfig())  # expect: arg-type
training_risk_config_payload(ExecutionCostConfig())  # expect: arg-type
utc_timestamp_ns("2024-01-01")  # expect: arg-type
def bad_clock_group(clock: NativeBarClock, spec: FeatureSpec, market: MarketDataset) -> None:
    ClockedNativeFeatureGroup(clock, ("bad",), (spec,))  # expect: arg-type
    clocked_native_decision_schedule(market, clock, start_index=0, stop_index=2)  # expect: arg-type
    run_clocked_native_portfolio_replay(market, policy, decision_group=ClockedNativeFeatureGroup(clock, (), (spec,)), start_index=0, stop_index=2, initial_capital=1000.0, execution_cost=ExecutionCostConfig(), execution_mode="unknown")  # expect: arg-type
    run_clocked_native_portfolio_replay(market, policy, decision_group=clock, start_index=0, stop_index=2, initial_capital=1000.0, execution_cost=ExecutionCostConfig())  # expect: arg-type
    run_clocked_native_portfolio_replay(market, WrongPolicy(), decision_group=ClockedNativeFeatureGroup(clock, (), (spec,)), start_index=0, stop_index=2, initial_capital=1000.0, execution_cost=ExecutionCostConfig())  # expect: arg-type
    run_shared_portfolio_replay(market, policy, start_index=0, stop_index=2, fill_cash_observer=7)  # expect: arg-type
def bad_external(pin: PackageSourcePin, path: Path, source: ExternalPackageInput) -> None:
    ExternalPackageInput(pin, "bad", path)  # expect: arg-type
    verify_external_source_provenance(source, source, runtime_config={}, expected_core_pin_digest="a"*64, expected_research_pin_digest="b"*64)  # expect: call-arg
def bad_replay(market: MarketDataset) -> None:
    run_shared_portfolio_replay(market, WrongPolicy(), start_index=0, stop_index=2)  # expect: arg-type
"""
)


def verify_installed_typing(target_python: Path) -> None:
    target_python = target_python.absolute()
    env = {
        key: value
        for key, value in os.environ.items()
        if key not in {"MYPYPATH", "PYTHONPATH", "MYPY_CACHE_DIR"}
    }
    with tempfile.TemporaryDirectory(prefix="trade-rl-downstream-") as temporary:
        root = Path(temporary)
        subprocess.run(
            [
                str(target_python),
                "-I",
                "-c",
                "import sys, trade_rl; from pathlib import Path; "
                "p = Path(trade_rl.__file__).resolve(); "
                "assert p.is_relative_to(Path(sys.prefix).resolve()); "
                "assert p.with_name('py.typed').read_bytes() == b''",
            ],
            cwd=root,
            env=env,
            check=True,
        )
        (root / "mypy.ini").write_text("[mypy]\n", encoding="utf-8")
        for filename, source in (
            ("positive.py", _POSITIVE),
            ("negative.py", _NEGATIVE),
        ):
            (root / filename).write_text(source, encoding="utf-8")
            result = subprocess.run(
                [
                    sys.executable,
                    "-I",
                    "-m",
                    "mypy",
                    "--strict",
                    "--config-file",
                    str(root / "mypy.ini"),
                    "--python-executable",
                    str(target_python),
                    "--no-incremental",
                    "--no-pretty",
                    "--no-error-summary",
                    "--show-error-codes",
                    "--no-color-output",
                    filename,
                ],
                cwd=root,
                env=env,
                text=True,
                capture_output=True,
            )
            expected = [
                (line, match.group(1))
                for line, value in enumerate(source.splitlines(), start=1)
                if (match := re.search(r"# expect: ([a-z-]+)$", value))
            ]
            observed = re.findall(
                rf"^{filename}:(\d+): error: .* \[([a-z-]+)\]$",
                result.stdout,
                re.MULTILINE,
            )
            error_lines = [
                line for line in result.stdout.splitlines() if ": error:" in line
            ]
            if (
                result.returncode != (1 if expected else 0)
                or sorted((int(line), code) for line, code in observed)
                != sorted(expected)
                or len(error_lines) != len(expected)
                or result.stderr
            ):
                raise RuntimeError(
                    f"{filename}: unexpected downstream type-check result "
                    f"(expected {expected}, exit {result.returncode}):\n"
                    f"{result.stdout}{result.stderr}"
                )
            print(
                f"{filename}: downstream typing verified ({len(expected)} expected errors)"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target_python", type=Path)
    verify_installed_typing(parser.parse_args().target_python)
