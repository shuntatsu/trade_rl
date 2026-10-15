"""Artificial integration checks for the maintained development driver."""

from __future__ import annotations

import importlib
import importlib.util
import json
from pathlib import Path

import pytest

from tools.tier_d_nonrl.tier_d_fixture import write_artificial_dataset
from tools.tier_d_nonrl.tier_d_native import (
    cost_rows,
    execution_cost,
    synthetic_protocol,
)


def driver():
    assert importlib.util.find_spec("tools.tier_d_nonrl.tier_d_runner") is not None, (
        "Tier D bounded public-API driver is missing"
    )
    return importlib.import_module("tools.tier_d_nonrl.tier_d_runner")


def test_result_blind_protocol_declares_the_complete_48_arm_budget():
    protocol = driver().market_protocol()
    assert len(protocol["accounts"]) == 8
    assert protocol["scenarios"] == ["BASEL0", "COST2L0", "BASEL2"]
    assert protocol["roles"] == ["nonrl", "cash"]
    assert protocol["budget"]["native_arms"] == 48
    assert protocol["budget"]["total_native_advances"] == 316752
    assert protocol["capital"]["per_account"] == 100000
    assert protocol["capital"]["combined_denominator"] == 200000
    assert protocol["decision"]["adoption"] == "NOT_ESTABLISHED"
    assert protocol["decision"]["winner"] == "NOT_DECLARED"
    assert protocol["funding"]["coverage_bounds"] == {
        "lookback_hours": 24,
        "minimum_counted_events": 2,
        "maximum_counted_close_gap_hours": 9,
        "maximum_counted_close_age_hours": 8,
    }


def test_artificial_h72_pair_uses_native_continuous_books_and_latency2(tmp_path):
    tmp_path = tmp_path / "attempt"
    result = driver().run_synthetic(tmp_path)
    assert result["scope"] == "ARTIFICIAL_SOFTWARE_INTEGRATION_ONLY"
    assert result["completed_arms"] == 2
    assert result["attempted_arms"] == 2
    assert result["dataset_dimensions"] == [256, 2, 6]
    assert result["public_calls"] == {
        "dataset_loads": 2,
        "cost_producers": 1,
        "prepare_calls": 1,
        "ridge_vintages": 1,
    }
    group = result["pairs"][0]
    assert group["comparison_complete"]
    assert group["roles"]["cash"]["final_equity"] == 100000
    assert group["roles"]["cash"]["total_cost"] == 0
    for role in ("nonrl", "cash"):
        closure = result["native_closure"][role]
        assert closure["reset_count"] == 1
        assert closure["actor_rows"] == 32
        assert closure["closing_rows"] == 8
        assert closure["settlement_complete"]
        assert closure["final_quantities"] == [0, 0]
        assert closure["active_orders"] == 0
        assert closure["ledger_reward_sum"] == pytest.approx(
            (group["roles"][role]["final_equity"] - 100000) / 100000
        )
    # Native MARKET latency2 is eligible at submit128 + 2 = processing130.
    assert result["native_closure"]["nonrl"]["first_fill_processing_index"] == 130
    assert result["decision"]["adoption"] == "NOT_ESTABLISHED"
    assert result["decision"]["winner"] == "NOT_DECLARED"
    assert len(tuple(tmp_path.glob("receipts/*.json.gz"))) == 2
    costs = group["decomposition"]["nonrl"]
    assert costs["price_pnl"] == pytest.approx(0, abs=1e-8)
    assert costs["fee"] > 0
    assert costs["funding_pnl"] < 0
    assert costs["closing_execution_cost_subset"] > 0
    assert costs["reconciliation_residual"] == pytest.approx(0, abs=1e-8)


def test_market_invocation_refuses_before_real_input_without_external_review(tmp_path):
    with pytest.raises(ValueError, match="ROOT different-model review"):
        driver().run_market(tmp_path, review_path=None)
    assert not tuple(tmp_path.rglob("*.npz"))


def test_unattempted_arm_cannot_be_replaced_by_a_cash_zero(tmp_path):
    module = driver()
    attempt_root = tmp_path / "attempt"
    module.initialize_attempt(attempt_root, module.market_protocol())
    roster = json.loads((attempt_root / "attempt.json").read_text(encoding="utf-8"))
    assert len(roster["arms"]) == 48
    assert all(row["state"] == "UNATTEMPTED" for row in roster["arms"])
    assert all(row["receipt"] is None for row in roster["arms"])


def test_output_directory_is_inside_the_assigned_ignored_scope():
    module = driver()
    root = Path(module.__file__).resolve().parent
    assert root.name == "tier_d_nonrl"
    assert root.parent.name == "tools"


def test_cost2_doubles_only_execution_rates_with_nonzero_artificial_carry(tmp_path):
    dataset = write_artificial_dataset(tmp_path / "artificial-carry-oracle")
    protocol = synthetic_protocol(driver().market_protocol(), dataset)
    base = cost_rows(dataset, protocol, 72, "BASEL0")[0]
    stress = cost_rows(dataset, protocol, 72, "COST2L0")[0]
    # One 100000 reference order / 10000000 quote turnover = 1% participation.
    # Dataset .0005 fee + .0002 spread + .0001*sqrt(.01) = .00071 one way.
    assert base.buy_cost == pytest.approx(0.00071)
    assert base.sell_cost == pytest.approx(0.00071)
    assert base.exit_cost == pytest.approx(0.00071)
    assert stress.buy_cost == pytest.approx(0.00142)
    assert stress.sell_cost == pytest.approx(0.00142)
    assert stress.exit_cost == pytest.approx(0.00142)
    # Three artificial 8h events at .0001 each, extrapolated 72h/24h.
    assert base.funding_return == pytest.approx(0.0009)
    assert stress.funding_return == base.funding_return
    assert base.borrow_return == pytest.approx(0.0876 * 72 / 8760)
    assert stress.borrow_return == base.borrow_return
    assert execution_cost("BASEL0").fee_rate == 0
    assert execution_cost("BASEL0").spread_rate == 0


def test_technical_failure_retains_failed_and_unattempted_slots(tmp_path, monkeypatch):
    tmp_path = tmp_path / "attempt"
    module = driver()

    def fail(*args, **kwargs):
        raise OSError("artificial injected native-call failure")

    monkeypatch.setattr(module, "run_terminal_allocation_execution", fail)
    with pytest.raises(RuntimeError, match="artificial injected"):
        module.run_synthetic(tmp_path)
    state = json.loads((tmp_path / "attempt.json").read_text(encoding="utf-8"))
    assert state["status"] == "FAILED"
    assert [row["state"] for row in state["arms"]] == ["FAILED", "UNATTEMPTED"]
    assert all(row["receipt"] is None for row in state["arms"])
    assert not (tmp_path / "result.json").exists()
    assert (tmp_path / "progress.jsonl").is_file()


def test_synthetic_run_can_resume_and_reuse_completed_receipts(tmp_path):
    module = driver()
    source, destination = tmp_path / "first", tmp_path / "second"
    r1 = module.run_synthetic(source)
    before = {
        p.relative_to(source): p.read_bytes() for p in source.rglob("*") if p.is_file()
    }
    r2 = module.run_synthetic(destination, resume_from=source)
    assert r1["completed_arms"] == r2["completed_arms"] == 2
    assert r1["matrix_digest"] == r2["matrix_digest"]
    assert r1["pairs"] == r2["pairs"]
    assert r1["combined_ranking"] == r2["combined_ranking"]
    assert r2["public_calls"]["cost_producers"] == 0
    assert r2["public_calls"]["prepare_calls"] == 0
    assert r2["public_calls"]["ridge_vintages"] == 0
    assert {
        p.relative_to(source): p.read_bytes() for p in source.rglob("*") if p.is_file()
    } == before
    assert (destination / "resume.json").is_file()
    assert "arm_reused" in (destination / "progress.jsonl").read_text()


def test_synthetic_run_supports_parallel_execution(tmp_path, monkeypatch):
    tmp_path = tmp_path / "attempt"
    module = driver()
    original_synthetic_protocol = module.synthetic_protocol

    def multi_pair_protocol(base_protocol, dataset):
        p = original_synthetic_protocol(base_protocol, dataset)
        p["scenarios"] = ["BASEL0", "BASEL2"]
        p["budget"]["native_arms"] = 4
        p["concurrency"] = {"max_workers": 2}
        p["resource_proposal"]["maximum_inflight_pairs"] = 2
        return p

    monkeypatch.setattr(module, "synthetic_protocol", multi_pair_protocol)
    result = module.run_synthetic(tmp_path)
    assert result["completed_arms"] == 4
    assert len(result["pairs"]) == 2
    events = [
        json.loads(line)["event"]
        for line in (tmp_path / "progress.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line
    ]
    assert "parallel_execution_started" in events
