"""Regression tests for Phase 2 selection, isolation and diagnostic contracts."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

from tests.evaluation.test_dev_realdata_phase2_trials import (
    small_protocol,
    synthetic_market,
)
from trade_rl.evaluation.dev_realdata_study import runner, trial_study
from trade_rl.evaluation.dev_realdata_study.protocol import SCENARIOS
from trade_rl.evaluation.dev_realdata_study.trial_sources import DrawdownOverlayPolicy
from trade_rl.evaluation.dev_realdata_study.trials import ALL_ARMS
from trade_rl.evaluation.replay import SharedCashPortfolioReplayResult
from trade_rl.strategies.allocation.cost_aware import AllocationDecisionRecord
from trade_rl.strategies.portfolio import (
    PortfolioDecisionSnapshot,
    PortfolioTargetProposal,
)


def _outcome(
    arm: str,
    scenario: str,
    *,
    drawdown: float = 5.0,
    days: int = 36,
    day_offset: int = 0,
    terminated: bool = False,
) -> runner.ReplayOutcome:
    start = np.datetime64("2021-10-04T00:00:00", "ns")
    hours = (days - 1) * 24 + 1
    times = (
        (
            start
            + day_offset * np.timedelta64(1, "D")
            + np.arange(hours) * np.timedelta64(1, "h")
        )
        .astype("datetime64[ns]")
        .astype(np.int64)
    )
    equity = np.linspace(10_000, 11_200, hours)
    return runner.ReplayOutcome(
        arm,
        scenario,
        equity,
        times,
        {
            "terminal_net_profit_pct": 12.0,
            "max_drawdown_pct": drawdown,
            "reconciliation": {"passed": not terminated},
            "terminated": ["margin"] if terminated else [],
        },
    )


def _runs(
    *,
    missing_controls: tuple[str, ...] = (),
    matrix_failure: bool = False,
    candidate_days: int = 36,
    candidate_offset: int = 0,
    candidate_terminated: bool = False,
) -> dict[str, trial_study.ArmRun]:
    runs = {
        arm: trial_study.ArmRun(arm, {}, {"worker": {"status": "FAILED_TO_RUN"}}, None)
        for arm in ALL_ARMS
    }
    for arm in ("cash", "equal_weight_buy_and_hold", "t1_tsmom"):
        if arm in missing_controls:
            continue
        outcomes = {
            scenario: _outcome(
                arm,
                scenario,
                days=candidate_days if arm == "t1_tsmom" else 36,
                day_offset=candidate_offset if arm == "t1_tsmom" else 0,
                terminated=candidate_terminated if arm == "t1_tsmom" else False,
            )
            for scenario in SCENARIOS
        }
        probe: dict[str, object] | None = None
        matrix: dict[str, dict[str, object]] | None = None
        if arm == "t1_tsmom":
            probe = {
                "passed": True,
                "identical_pre_cut_trace": True,
                "pre_cut_trace_has_trades": True,
            }
            matrix = {
                scenario: {"passed": not matrix_failure or scenario != "cost_x2"}
                for scenario in SCENARIOS
            }
        runs[arm] = trial_study.ArmRun(arm, outcomes, {}, probe, matrix)
    return runs


def _assemble(**kwargs: Any) -> dict[str, Any]:
    result, _ = trial_study.assemble(
        synthetic_market(), small_protocol(), _runs(**kwargs)
    )
    return result


def test_custom_risk_protocol_is_used_by_both_final_labels() -> None:
    market = synthetic_market()
    tight = replace(small_protocol(), max_drawdown_limit=0.01)
    result, _ = trial_study.assemble(market, tight, _runs())
    assert result["label"] == "NO_WINNER"
    assert result["label_amended_rule"] == "NO_WINNER"
    assert (
        result["selection_amended_rule"]["arms"]["t1_tsmom"]["risk_gate_passed"]
        is False
    )


@pytest.mark.parametrize("missing_control", ["cash", "equal_weight_buy_and_hold"])
def test_an_invalid_control_blocks_any_development_promotion(
    missing_control: str,
) -> None:
    result = _assemble(missing_controls=(missing_control,))
    assert result["label"] == result["label_amended_rule"] == "NO_WINNER"
    assert result["selected_arm_amended_rule"] is None
    assert result["selection_amended_rule"]["arms"]["t1_tsmom"]["eligible"] is False
    assert missing_control in result["selection_amended_rule"]["invalid_controls"]


@pytest.mark.parametrize("missing", ["arm", "scenario"])
def test_partial_arm_report_is_ineligible_without_key_error(missing: str) -> None:
    from trade_rl.evaluation.dev_realdata_study.trial_labels import (
        labels_under_both_rules,
    )
    from trade_rl.evaluation.dev_realdata_study.trials import LABEL_FAILED_TO_RUN

    result = _assemble()
    if missing == "arm":
        del result["arms"]["t1_tsmom"]
    else:
        del result["arms"]["t1_tsmom"]["cost_x2"]
    for selection in labels_under_both_rules(result).values():
        assert selection["selected_arm"] is None
        assert selection["arms"]["t1_tsmom"]["status"] == LABEL_FAILED_TO_RUN


def test_stale_control_validity_cannot_bypass_replay_falsifier() -> None:
    from trade_rl.evaluation.dev_realdata_study.trial_labels import (
        labels_under_both_rules,
    )

    result = _assemble()
    assert result["validity"]["cash"] is True
    result["validity_detail"]["cash"]["no_termination_all_scenarios"] = False
    for selection in labels_under_both_rules(result).values():
        assert selection["label"] == "NO_WINNER"
        assert selection["selected_arm"] is None
        assert "cash" in selection["invalid_controls"]


def test_missing_causality_matrix_evidence_never_promotes_candidate() -> None:
    from trade_rl.evaluation.dev_realdata_study.trial_labels import (
        labels_under_both_rules,
    )

    result = _assemble()
    del result["validity_detail"]["t1_tsmom"]["causality_matrix_passed"]
    for selection in labels_under_both_rules(result).values():
        assert selection["validity"]["t1_tsmom"] is False
        assert selection["selected_arm"] is None


def test_bootstrap_omission_records_missing_arm_scenario() -> None:
    runs = _runs()
    del runs["t1_tsmom"].outcomes["cost_x2"]
    result, _ = trial_study.assemble(synthetic_market(), small_protocol(), runs)
    assert result["bootstrap_omissions"]["cost_x2"]["t1_tsmom"] == "missing_outcome"


def test_probe_matrix_failure_disqualifies_under_both_rules() -> None:
    result = _assemble(matrix_failure=True)
    assert result["label"] == result["label_amended_rule"] == "NO_WINNER"
    assert result["selection_pr_c_probe_rule"]["validity"]["t1_tsmom"] is False
    assert result["selection_amended_rule"]["validity"]["t1_tsmom"] is False


def test_causality_matrix_cut_is_the_effective_cut_not_a_constant() -> None:
    custom_cut = "2021-10-18T00:00:00"
    result, _ = trial_study.assemble(
        synthetic_market(), small_protocol(), _runs(), matrix_cut=custom_cut
    )
    assert result["causality_matrix_cut"] == custom_cut
    assert result["nonregistered_causality_cut_override"] is True
    assert result["status"] == "diagnostic_only"
    assert result["label"] == result["label_amended_rule"] == "NO_WINNER"
    assert (
        result["selection_amended_rule"]["selection_blocked_reason"]
        == "diagnostic_only"
    )


def test_run_phase2_propagates_the_effective_matrix_cut(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    custom_cut = "2021-10-18T00:00:00"

    def no_replays(
        _dataset: Any, _protocol: Any, arm: str, **_kwargs: Any
    ) -> trial_study.ArmRun:
        return trial_study.ArmRun(
            arm,
            {},
            {"worker": {"status": "FAILED_TO_RUN"}},
            None,
        )

    monkeypatch.setattr(trial_study, "run_arm", no_replays)
    result, _ = trial_study.run_phase2(
        synthetic_market(),
        small_protocol(),
        matrix_cut=custom_cut,
    )
    assert result["causality_matrix_cut"] == custom_cut
    assert result["status"] == "diagnostic_only"


def test_sequential_worker_exception_becomes_failed_arm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def run_or_fail(
        _dataset: Any, _protocol: Any, arm: str, **_kwargs: Any
    ) -> trial_study.ArmRun:
        if arm == "t1_tsmom":
            raise RuntimeError("synthetic arm crash")
        return trial_study.ArmRun(
            arm, {}, {"worker": {"status": "FAILED_TO_RUN"}}, None
        )

    monkeypatch.setattr(trial_study, "run_arm", run_or_fail)
    result, _ = trial_study.run_phase2(
        synthetic_market(), small_protocol(), parallel_loader=None
    )
    assert result["label"] == "NO_WINNER"
    assert result["failures"]["t1_tsmom"]["worker"]["status"] == "FAILED_TO_RUN"


def test_bootstrap_skips_terminated_shorter_paths_without_aborting() -> None:
    result = _assemble(candidate_days=4, candidate_terminated=True)
    assert result["label"] == result["label_amended_rule"] == "NO_WINNER"
    assert result["bootstrap"]["base"].get("t1_tsmom") is None
    assert result["bootstrap_omissions"]["base"]["t1_tsmom"] == "terminated"
    assert "equal_weight_buy_and_hold" in result["bootstrap"]["base"]


def test_partial_unterminated_replay_is_not_valid_evidence() -> None:
    result = _assemble(candidate_days=4)
    assert result["validity"]["t1_tsmom"] is False
    assert result["validity_detail"]["t1_tsmom"]["complete_clock_coverage"] is False
    assert result["label"] == result["label_amended_rule"] == "NO_WINNER"


def test_bootstrap_rejects_unpaired_timestamps_even_when_lengths_match() -> None:
    result = _assemble(candidate_offset=1)
    assert "t1_tsmom" not in result["bootstrap"]["base"]
    assert (
        result["bootstrap_omissions"]["base"]["t1_tsmom"] == "not_paired_with_controls"
    )


def test_pr_c_probe_is_derived_from_primitives_not_a_passed_flag() -> None:
    from trade_rl.evaluation.dev_realdata_study.trial_labels import (
        labels_under_both_rules,
    )

    result = _assemble()
    result["causality_probes"]["t1_tsmom"]["pre_cut_trace_has_trades"] = False
    # Simulate a stale or contradictory stored summary boolean.
    result["causality_probes"]["t1_tsmom"]["passed"] = True
    primary = labels_under_both_rules(
        result, {"causality_probes": result["causality_probes"]}
    )
    assert primary["label_pr_c_probe_rule"]["validity"]["t1_tsmom"] is False


@pytest.mark.parametrize("equity, expected_scale", [(85.0, 0.5), (75.0, 0.0)])
def test_t7_preserves_fail_closed_hold_when_overlay_would_trim(
    equity: float, expected_scale: float
) -> None:
    proposal = PortfolioTargetProposal(
        target_weights=(0.5, -0.4),
        hold_current_quantity_symbols=(0, 1),
    )
    stamp = int(np.datetime64("2023-01-03T00:00:00", "ns").astype(np.int64))
    record = AllocationDecisionRecord(
        index=10,
        timestamp_ns=stamp,
        outcome="fail_closed",
        reason="invalid_forecast",
        iterations=0,
        objective=None,
        traded_symbols=(),
    )

    class Inner:
        records = (record,)

        def decide_portfolio(self, _snapshot: Any) -> PortfolioTargetProposal:
            return proposal

    policy = DrawdownOverlayPolicy(
        Inner(),
        gross_max=0.95,
        decision_interval_hours=24,
        decision_anchor_hour_utc=0,
    )
    snapshot = SimpleNamespace(
        timestamp_ns=stamp,
        account=SimpleNamespace(portfolio_value=equity, peak_value=100.0),
    )
    result = policy.decide_portfolio(cast(PortfolioDecisionSnapshot, snapshot))
    assert result is proposal
    summary = policy.overlay_summary()
    assert policy.scales[-1][1] == pytest.approx(expected_scale)
    assert summary["n_fail_closed_hold_preserved"] == 1
    assert summary["n_zero_scale_fail_closed_hold_preserved"] == int(
        expected_scale == 0.0
    )
    if expected_scale == 0.0:
        # An already-exposed book is not silently liquidated when the
        # allocator cannot safely issue a new target. Risk remains observable.
        assert summary["latched_at_end"] is True
        assert summary["n_decisions_at_zero"] == 1


@pytest.mark.parametrize(
    "hours, anchor",
    [(0, 0), (-1, 0), (4, -1), (4, 4), (4, 8)],
)
def test_t7_rejects_schedule_that_can_never_fire(hours: int, anchor: int) -> None:
    class Inner:
        records: tuple[object, ...] = ()

    with pytest.raises(ValueError, match="interval|anchor"):
        DrawdownOverlayPolicy(
            Inner(),
            gross_max=0.95,
            decision_interval_hours=hours,
            decision_anchor_hour_utc=anchor,
        )


def test_pre_cut_trade_check_rejects_flat_no_fill_intervals() -> None:
    intervals = [
        SimpleNamespace(exact_quantities_after=("0", "0"), order_events=()),
        SimpleNamespace(exact_quantities_after=("0", "0"), order_events=()),
    ]
    result = SimpleNamespace(
        ledger_evidence=SimpleNamespace(intervals=SimpleNamespace(intervals=intervals))
    )
    assert (
        runner._pre_cut_has_trades(cast(SharedCashPortfolioReplayResult, result))
        is False
    )


def test_causality_trade_check_only_looks_at_actual_interval_rows() -> None:
    intervals = [
        SimpleNamespace(exact_quantities_after=("0", "0"), order_events=()),
        SimpleNamespace(exact_quantities_after=("1/5", "0"), order_events=()),
    ]
    # Decision count is intentionally unrelated to the interval count.
    result = SimpleNamespace(
        ledger_evidence=SimpleNamespace(intervals=SimpleNamespace(intervals=intervals)),
        decisions=[SimpleNamespace() for _ in range(20)],
    )
    assert (
        runner._pre_cut_has_trades(cast(SharedCashPortfolioReplayResult, result))
        is True
    )


def test_pre_cut_roundtrip_fill_counts_as_trading_even_if_flat() -> None:
    intervals = [
        SimpleNamespace(
            exact_quantities_after=("0",),
            order_events=(SimpleNamespace(filled_quantity=0.25),),
        )
    ]
    result = SimpleNamespace(
        ledger_evidence=SimpleNamespace(intervals=SimpleNamespace(intervals=intervals))
    )
    assert (
        runner._pre_cut_has_trades(cast(SharedCashPortfolioReplayResult, result))
        is True
    )


def test_cli_refuses_preexisting_and_overlapping_outputs_before_loading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from trade_rl.evaluation.dev_realdata_study import trial_cli

    def should_not_load(_root: Path) -> Any:
        raise AssertionError("no dataset work should begin for invalid output paths")

    monkeypatch.setattr(
        trial_cli, "load_published_market_dataset_artifact", should_not_load
    )
    out = tmp_path / "out"
    out.mkdir()
    (out / "equity.npz").write_bytes(b"preserved")
    argv = [
        "--dataset",
        str(tmp_path / "unused"),
        "--output-dir",
        str(out),
        "--summary-path",
        str(tmp_path / "summary.json"),
    ]
    assert trial_cli.main(argv) == 2
    assert (out / "equity.npz").read_bytes() == b"preserved"
    out2 = tmp_path / "new"
    argv[argv.index(str(out))] = str(out2)
    argv[argv.index(str(tmp_path / "summary.json"))] = str(out2 / "summary.json")
    assert trial_cli.main(argv) == 2
    assert not out2.exists()
    out3 = tmp_path / "new3"
    blocked_parent = tmp_path / "not_a_directory"
    blocked_parent.write_text("do not overwrite", encoding="utf-8")
    argv[argv.index(str(out2))] = str(out3)
    argv[argv.index(str(out2 / "summary.json"))] = str(blocked_parent / "summary.json")
    assert trial_cli.main(argv) == 2
    assert not out3.exists()
    assert blocked_parent.read_text(encoding="utf-8") == "do not overwrite"


def test_runner_records_interval_units_and_all_cashflow_components(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = synthetic_market()
    protocol = small_protocol()
    adapter = SimpleNamespace(history=[(0.07, 0.14)], calls=1, null_packets=0)
    monkeypatch.setattr(
        runner,
        "build_arm",
        lambda *_: runner.BuiltArm(
            policy=SimpleNamespace(),
            adapter=adapter,
            forecast_interval_hours=168,
        ),
    )
    diagnostics = SimpleNamespace(
        total_cost=1.0,
        funding_pnl=-2.0,
        borrow_cost=3.0,
        turnover_total=0.5,
        n_trades=1,
        termination_reasons=(),
    )
    monkeypatch.setattr(
        runner,
        "run_shared_portfolio_replay",
        lambda *_a, **_kw: SimpleNamespace(diagnostics=diagnostics),
    )
    monkeypatch.setattr(
        runner,
        "equity_path",
        lambda *_: (np.asarray([10_000.0, 10_100.0]), np.asarray([1, 2])),
    )
    monkeypatch.setattr(
        runner,
        "performance_summary",
        lambda *_: {"years": 1.0},
    )
    monkeypatch.setattr(
        runner,
        "reconcile_ledger",
        lambda *_: {
            "passed": True,
            "dividend_pnl": 4.0,
            "cash_interest_pnl": 5.0,
        },
    )
    monkeypatch.setattr(runner, "exposure_summary", lambda *_: {})
    result = runner.run_replay(
        dataset,
        protocol,
        "t8_blend_weekly",
        "base",
        {},
        start_index=0,
        stop_index=200,
    )
    forecast = cast(dict[str, Any], result.summary["forecast"])
    assert forecast["decision_interval_hours"] == 168
    assert forecast["mean_abs_mu_per_interval"] == pytest.approx(0.105)
    assert forecast["mean_abs_mu_per_day"] == pytest.approx(0.015)
    assert result.summary["gross_price_pnl_usdt"] == pytest.approx(97.0)
    assert result.summary["dividend_pnl_usdt"] == 4.0
    assert result.summary["cash_interest_pnl_usdt"] == 5.0
    adapter.history = []
    empty = runner.run_replay(
        dataset,
        protocol,
        "t8_blend_weekly",
        "base",
        {},
        start_index=0,
        stop_index=200,
    )
    forecast_empty = cast(dict[str, Any], empty.summary["forecast"])
    for key in (
        "mean_abs_mu_per_interval",
        "median_abs_mu_per_interval",
        "mean_abs_mu_per_day",
        "median_abs_mu_per_day",
        "share_abs_mu_above_one_way_cost",
    ):
        assert forecast_empty[key] is None
    import json

    json.dumps(empty.summary, allow_nan=False)


def test_compact_summary_retains_cashflow_and_native_forecast_units() -> None:
    """A compact result must still reconcile net profit to the price P&L."""
    row = {
        "terminal_net_profit_pct": 1.0,
        "cagr_pct": 0.5,
        "max_drawdown_pct": 0.3,
        "sharpe_daily": None,
        "turnover_total": 1.0,
        "turnover_per_year": 1.0,
        "execution_costs_usdt": 2.0,
        "funding_pnl_usdt": -3.0,
        "borrow_cost_usdt": 4.0,
        "dividend_pnl_usdt": 5.0,
        "cash_interest_pnl_usdt": 6.0,
        "gross_price_pnl_usdt": 98.0,
        "net_pnl_usdt": 100.0,
        "n_trades": 2,
        "per_year_return_pct": {"2023": 1.0},
        "reconciliation": {
            "passed": True,
            "terminal_flat": True,
            "terminal_residual_quantities": {},
            "terminal_residual_notional_usdt": 0.0,
        },
        "exposure": {"mean_gross_exposure": 0.2, "mean_net_exposure": 0.1},
        "forecast": {
            "share_abs_mu_above_one_way_cost": 0.5,
            "decision_interval_hours": 168,
            "mean_abs_mu_per_interval": 0.07,
            "median_abs_mu_per_interval": 0.035,
            "mean_abs_mu_per_day": 0.01,
            "median_abs_mu_per_day": 0.005,
        },
    }
    keys = (
        "schema_version",
        "status",
        "label",
        "label_pr_c_probe_rule",
        "label_amended_rule",
        "selected_arm_pr_c_probe_rule",
        "selected_arm_amended_rule",
        "label_note",
        "registry_digest",
        "base_protocol_digest",
        "dataset_id",
        "git_head",
        "trials",
        "validity",
        "validity_detail",
        "causality_probes",
        "causality_matrix",
        "causality_matrix_cut",
        "nonregistered_causality_cut_override",
        "failures",
        "bootstrap",
        "bootstrap_omissions",
        "bonferroni_comparisons",
        "selection_pr_c_probe_rule",
        "selection_amended_rule",
        "notes",
    )
    result: dict[str, Any] = {key: None for key in keys}
    legacy = dict(row)
    for name in ("borrow_cost_usdt", "dividend_pnl_usdt", "cash_interest_pnl_usdt"):
        del legacy[name]
    legacy["forecast"] = {
        "share_abs_mu_above_one_way_cost": 0.5,
        "mean_abs_mu_per_day": 0.01,
    }
    legacy["allocator_decisions"] = {"n_decisions": 2}
    result["arms"] = {
        "t8_blend_weekly": {"base": row},
        "legacy_day": {"base": legacy},
    }
    compact = trial_study.compact_phase2_summary(result)
    older = dict(result)
    older.pop("nonregistered_causality_cut_override")
    older.pop("bootstrap_omissions")
    older_compact = trial_study.compact_phase2_summary(older)
    # Unknown historical metadata must remain unknown, not be invented as
    # `False`/empty and not cause an otherwise readable summary to crash.
    assert older_compact["nonregistered_causality_cut_override"] is None
    assert older_compact["bootstrap_omissions"] is None
    saved = compact["arms"]["t8_blend_weekly"]["base"]
    assert saved["gross_price_pnl_usdt"] == pytest.approx(
        saved["net_pnl_usdt"]
        + saved["execution_costs_usdt"]
        - saved["funding_pnl_usdt"]
        + saved["borrow_cost_usdt"]
        - saved["dividend_pnl_usdt"]
        - saved["cash_interest_pnl_usdt"]
    )
    assert saved["decision_interval_hours"] == 168
    assert saved["mean_abs_mu_per_interval"] == pytest.approx(0.07)
    assert saved["mean_abs_mu_per_day"] == pytest.approx(0.01)
    old_saved = compact["arms"]["legacy_day"]["base"]
    assert "borrow_cost_usdt" not in old_saved
    assert "cash_interest_pnl_usdt" not in old_saved
    assert "decision_interval_hours" not in old_saved
    assert old_saved["fail_closed_hold_count"] is None
