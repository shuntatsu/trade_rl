"""Run the Phase 3 roster (Issue #810) on the shared development real-data runner.

One task replays one arm in all three scenarios and runs the real-data
future-perturbation probe and the per-scenario causality matrix for every arm except
CASH. A task that raises becomes a ``FAILED_TO_RUN`` row; nothing is retried with other
parameters. The label vocabulary is ``POST_HOC_PROMISING`` / ``NO_WINNER``; a
``WINNER`` cannot be produced here.
"""

from __future__ import annotations

import traceback
from collections.abc import Callable, Mapping
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from trade_rl.data.market import MarketDataset
from trade_rl.evaluation.dev_realdata_study import runner
from trade_rl.evaluation.dev_realdata_study.metrics import (
    bootstrap_excess,
    daily_equity,
    daily_returns,
    moving_block_indices,
    select_development_label,
)
from trade_rl.evaluation.dev_realdata_study.phase3_registry import (
    ALL_ARMS,
    BASELINE_ARMS,
    BONFERRONI_ARMS,
    CANDIDATE_POOL,
    CONTROL_ARMS,
    FAMILY_ARMS,
    IMPLEMENTATION_NOTES,
    LABEL_FAILED_TO_RUN,
    LABEL_NO_WINNER,
    LABEL_POST_HOC_PROMISING,
    P3_ARMS,
    P3_TRIALS,
    PASSIVE_COMPARATOR_ARM,
    RESULT_SCHEMA,
    registry_digest,
    roster_digest,
)
from trade_rl.evaluation.dev_realdata_study.phase3_sources import (
    build_phase3_arm,
    make_phase3_context,
)
from trade_rl.evaluation.dev_realdata_study.protocol import (
    ARM_BUY_AND_HOLD,
    ARM_CASH,
    DevStudyProtocol,
    protocol_digest,
)
from trade_rl.evaluation.dev_realdata_study.trial_labels import pr_c_probe_passed

Progress = Callable[[str], None]

# Same short pre-cut window as the Phase 2 causality matrix.
PROBE_MATRIX_CUT = "2023-03-01T00:00:00"

RULE_PR_C = "pr_c_probe_rule_requires_pre_cut_trades"
RULE_AMENDED = "amended_probe_rule_identical_decisions_only"


@dataclass
class ArmRun:
    """Everything one arm produced: outcomes per scenario, the probes, failures."""

    arm: str
    outcomes: dict[str, runner.ReplayOutcome]
    failures: dict[str, dict[str, str]]
    probe: dict[str, object] | None
    probe_matrix: dict[str, dict[str, object]] | None = None


def _failure(exc: BaseException) -> dict[str, str]:
    return {
        "status": LABEL_FAILED_TO_RUN,
        "error_type": type(exc).__name__,
        "error": str(exc)[:600],
        "traceback_tail": traceback.format_exc()[-1500:],
    }


def phase3_protocol(protocol: DevStudyProtocol) -> DevStudyProtocol:
    """The PR-C protocol with the Phase 3 rosters and the 21-arm Bonferroni count."""

    return replace(
        protocol,
        arms=ALL_ARMS,
        candidate_arms=CANDIDATE_POOL,
        bonferroni_comparisons=BONFERRONI_ARMS,
    )


def run_arm(
    dataset: MarketDataset,
    protocol: DevStudyProtocol,
    arm: str,
    *,
    start_index: int,
    stop_index: int,
    say: Progress,
    matrix_cut: str = PROBE_MATRIX_CUT,
) -> ArmRun:
    """All scenarios plus the causality probe and matrix for one arm."""

    outcomes: dict[str, runner.ReplayOutcome] = {}
    failures: dict[str, dict[str, str]] = {}
    probe: dict[str, object] | None = None
    probe_matrix: dict[str, dict[str, object]] | None = None
    context = make_phase3_context(dataset, protocol)
    for scenario in protocol.scenarios:
        say(f"replay {arm} / {scenario}")
        try:
            outcomes[scenario] = runner.run_replay(
                dataset,
                protocol,
                arm,
                scenario,
                context,
                start_index=start_index,
                stop_index=stop_index,
                arm_builder=build_phase3_arm,
            )
        except Exception as exc:  # recorded, never retried with other parameters
            failures[scenario] = _failure(exc)
            say(f"FAILED {arm} / {scenario}: {type(exc).__name__}: {exc}")
    if arm != ARM_CASH:
        say(f"causality probe {arm}")
        try:
            probe = runner.causality_probe(
                dataset,
                protocol,
                context,
                start_index=start_index,
                arm=arm,
                expect_trades=False,
                arm_builder=build_phase3_arm,
                sources_builder=make_phase3_context,
            )
        except Exception as exc:
            failures["causality_probe"] = _failure(exc)
            say(f"FAILED {arm} causality probe: {type(exc).__name__}: {exc}")
        say(f"causality matrix {arm}")
        try:
            probe_matrix = runner.causality_probe_matrix(
                dataset,
                protocol,
                context,
                start_index=start_index,
                cut=matrix_cut,
                arms=(arm,),
                expect_trades=False,
                arm_builder=build_phase3_arm,
                sources_builder=make_phase3_context,
            )[arm]
        except Exception as exc:
            failures["causality_matrix"] = _failure(exc)
            say(f"FAILED {arm} causality matrix: {type(exc).__name__}: {exc}")
    return ArmRun(arm, outcomes, failures, probe, probe_matrix)


def _arm_worker(
    loader: Callable[[], MarketDataset],
    protocol: DevStudyProtocol,
    arm: str,
    start_index: int,
    stop_index: int,
    matrix_cut: str,
) -> ArmRun:
    dataset = loader()
    return run_arm(
        dataset,
        protocol,
        arm,
        start_index=start_index,
        stop_index=stop_index,
        say=lambda message: print(message, flush=True),
        matrix_cut=matrix_cut,
    )


def run_phase3(
    dataset: MarketDataset,
    protocol: DevStudyProtocol,
    *,
    progress: Progress | None = None,
    parallel_loader: Callable[[], MarketDataset] | None = None,
    max_workers: int = 8,
    arms: tuple[str, ...] = ALL_ARMS,
    matrix_cut: str = PROBE_MATRIX_CUT,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Run the two baselines and the eight roster arms; return results and equity.

    ``protocol`` is the PR-C protocol (its digest is recorded). With
    ``parallel_loader`` each arm runs in its own process.
    """

    say = progress or (lambda message: None)
    if dataset.dataset_id != protocol.dataset_id:
        raise ValueError("dataset_id does not match the protocol")
    if tuple(arms) != ALL_ARMS:
        raise ValueError("arm roster differs from the registered Phase 3 roster")
    start_index = runner._row_of(dataset, protocol.evaluation_start)
    stop_index = runner._row_of(dataset, protocol.evaluation_end)

    runs: dict[str, ArmRun] = {}
    if parallel_loader is not None:
        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                arm: pool.submit(
                    _arm_worker,
                    parallel_loader,
                    protocol,
                    arm,
                    start_index,
                    stop_index,
                    matrix_cut,
                )
                for arm in arms
            }
            for arm, future in futures.items():
                try:
                    runs[arm] = future.result()
                except Exception as exc:  # worker crash: record, keep going
                    runs[arm] = ArmRun(arm, {}, {"worker": _failure(exc)}, None)
                say(f"arm {arm} complete")
    else:
        for arm in arms:
            runs[arm] = run_arm(
                dataset,
                protocol,
                arm,
                start_index=start_index,
                stop_index=stop_index,
                say=say,
                matrix_cut=matrix_cut,
            )
    return assemble(dataset, protocol, runs, matrix_cut=matrix_cut)


def _select(
    arms_table: Mapping[str, Mapping[str, Mapping[str, Any]]],
    validity: Mapping[str, bool],
    protocol: DevStudyProtocol,
    rule: str,
) -> dict[str, Any]:
    scenarios = protocol.scenarios
    runnable = tuple(
        arm
        for arm in CANDIDATE_POOL
        if all("terminal_net_profit_pct" in arms_table[arm][s] for s in scenarios)
    )
    selection = select_development_label(
        {
            arm: {
                scenario: {
                    "terminal_net_profit_pct": float(
                        arms_table[arm][scenario]["terminal_net_profit_pct"]
                    ),
                    "max_drawdown_pct": float(
                        arms_table[arm][scenario]["max_drawdown_pct"]
                    ),
                }
                for scenario in scenarios
            }
            for arm in runnable
        },
        validity,
        replace(protocol, candidate_arms=runnable),
    )
    rows: dict[str, Any] = dict(selection.rows)
    for arm in CANDIDATE_POOL:
        if arm not in runnable:
            rows[arm] = {
                "valid": False,
                "eligible": False,
                "status": LABEL_FAILED_TO_RUN,
            }
    label = (
        LABEL_POST_HOC_PROMISING
        if selection.selected_arm is not None
        else LABEL_NO_WINNER
    )
    return {
        "rule": rule,
        "label": label,
        "selected_arm": selection.selected_arm,
        "validity": {arm: bool(validity.get(arm, False)) for arm in CANDIDATE_POOL},
        "arms": {arm: rows[arm] for arm in CANDIDATE_POOL},
    }


def _interval(
    excess: np.ndarray, indices: np.ndarray, protocol: DevStudyProtocol
) -> dict[str, float]:
    base = bootstrap_excess(
        excess, indices, replace(protocol, bonferroni_comparisons=FAMILY_ARMS)
    ).to_mapping()
    wide = bootstrap_excess(
        excess, indices, replace(protocol, bonferroni_comparisons=BONFERRONI_ARMS)
    ).to_mapping()
    return {
        "mean_daily_excess": base["mean_daily_excess"],
        "annualized_mean_excess": base["annualized_mean_excess"],
        "ci95_low_annualized": base["ci95_low_annualized"],
        "ci95_high_annualized": base["ci95_high_annualized"],
        "bonferroni_family8_ci_low_annualized": base["bonferroni_ci_low_annualized"],
        "bonferroni_family8_ci_high_annualized": base["bonferroni_ci_high_annualized"],
        "bonferroni_all21_ci_low_annualized": wide["bonferroni_ci_low_annualized"],
        "bonferroni_all21_ci_high_annualized": wide["bonferroni_ci_high_annualized"],
    }


def assemble(
    dataset: MarketDataset,
    protocol: DevStudyProtocol,
    runs: dict[str, ArmRun],
    *,
    matrix_cut: str = PROBE_MATRIX_CUT,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Validity, bootstrap, selection and the result mapping from finished arm runs."""

    p3 = phase3_protocol(protocol)
    scenarios = protocol.scenarios
    arms_table: dict[str, dict[str, dict[str, Any]]] = {}
    for arm in ALL_ARMS:
        run = runs[arm]
        row: dict[str, dict[str, Any]] = {}
        for scenario in scenarios:
            if scenario in run.outcomes:
                row[scenario] = run.outcomes[scenario].summary
            else:
                row[scenario] = run.failures.get(
                    scenario,
                    run.failures.get(
                        "worker",
                        {"status": LABEL_FAILED_TO_RUN, "error": "no result"},
                    ),
                )
        arms_table[arm] = row

    def ran_everywhere(arm: str) -> bool:
        return all(scenario in runs[arm].outcomes for scenario in scenarios)

    validity_pr_c: dict[str, bool] = {}
    validity_amended: dict[str, bool] = {}
    validity_detail: dict[str, dict[str, object]] = {}
    for arm in ALL_ARMS:
        run = runs[arm]
        complete = ran_everywhere(arm)
        reconciled = complete and all(
            bool(arms_table[arm][s]["reconciliation"]["passed"]) for s in scenarios
        )
        no_termination = complete and all(
            not arms_table[arm][s]["terminated"] for s in scenarios
        )
        needs_probe = arm != ARM_CASH
        probe_identical = (not needs_probe) or (
            run.probe is not None and bool(run.probe["identical_pre_cut_trace"])
        )
        probe_traded = (not needs_probe) or (
            run.probe is not None and pr_c_probe_passed(run.probe)
        )
        matrix_ok = (not needs_probe) or (
            run.probe_matrix is not None
            and set(run.probe_matrix) == set(scenarios)
            and all(
                bool(c["identical_pre_cut_trace"]) for c in run.probe_matrix.values()
            )
        )
        replay_ok = bool(complete and reconciled and no_termination)
        validity_pr_c[arm] = bool(replay_ok and probe_traded and matrix_ok)
        validity_amended[arm] = bool(replay_ok and probe_identical and matrix_ok)
        validity_detail[arm] = {
            "all_scenarios_ran": complete,
            "ledger_reconciled_all_scenarios": reconciled,
            "no_termination_all_scenarios": no_termination,
            "causality_probe_identical": None if not needs_probe else probe_identical,
            "causality_probe_traded_before_cut": (
                None if not needs_probe else probe_traded
            ),
            "causality_matrix_identical_all_scenarios": (
                None if not needs_probe else matrix_ok
            ),
            "scenarios_failing_reconciliation": [
                s
                for s in scenarios
                if s in run.outcomes
                and not bool(arms_table[arm][s]["reconciliation"]["passed"])
            ],
        }

    bootstrap: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
    for scenario in scenarios:
        daily: dict[str, np.ndarray] = {}
        for arm in ALL_ARMS:
            outcome = runs[arm].outcomes.get(scenario)
            if outcome is not None:
                _, values = daily_equity(outcome.timestamps_ns, outcome.equity)
                daily[arm] = daily_returns(values)
        if ARM_CASH not in daily or ARM_BUY_AND_HOLD not in daily:
            bootstrap[scenario] = {}
            continue
        indices = moving_block_indices(
            len(daily[ARM_CASH]),
            block=p3.bootstrap_block_days,
            resamples=p3.bootstrap_resamples,
            seed=p3.bootstrap_seed,
        )
        table: dict[str, dict[str, dict[str, float]]] = {}
        for arm, series in daily.items():
            if arm == ARM_CASH:
                continue
            versus = {"vs_cash": _interval(series - daily[ARM_CASH], indices, p3)}
            if arm != ARM_BUY_AND_HOLD:
                versus["vs_equal_weight_buy_and_hold"] = _interval(
                    series - daily[ARM_BUY_AND_HOLD], indices, p3
                )
            if arm in CANDIDATE_POOL and PASSIVE_COMPARATOR_ARM in daily:
                versus["vs_p3_7_vol_scaled_passive"] = _interval(
                    series - daily[PASSIVE_COMPARATOR_ARM], indices, p3
                )
            table[arm] = versus
        bootstrap[scenario] = table

    primary = _select(arms_table, validity_pr_c, p3, RULE_PR_C)
    amended = _select(arms_table, validity_amended, p3, RULE_AMENDED)

    results: dict[str, Any] = {
        "schema_version": RESULT_SCHEMA,
        "status": "development_evidence_only_post_hoc_exploratory",
        "roster_digest": roster_digest(),
        "registry_digest": registry_digest(),
        "base_protocol_digest": protocol_digest(protocol),
        "dataset_id": dataset.dataset_id,
        "symbols": list(dataset.symbols),
        "git_head": runner._git_head(),
        "trials": [
            {
                "trial_id": spec.trial_id,
                "arm": spec.arm,
                "roster_text": spec.roster_text,
                "role": "control" if spec.arm in CONTROL_ARMS else "candidate",
                "status": "ran" if ran_everywhere(spec.arm) else LABEL_FAILED_TO_RUN,
            }
            for spec in P3_TRIALS
        ],
        "baselines": [
            {
                "arm": arm,
                "status": "ran" if ran_everywhere(arm) else LABEL_FAILED_TO_RUN,
            }
            for arm in BASELINE_ARMS
        ],
        "causality_probes": {
            arm: runs[arm].probe for arm in ALL_ARMS if arm != ARM_CASH
        },
        "causality_matrix": {
            arm: runs[arm].probe_matrix for arm in ALL_ARMS if arm != ARM_CASH
        },
        "causality_matrix_cut": matrix_cut,
        "failures": {arm: runs[arm].failures for arm in ALL_ARMS if runs[arm].failures},
        "validity_pr_c_probe_rule": validity_pr_c,
        "validity_amended_rule": validity_amended,
        "validity_detail": validity_detail,
        "arms": arms_table,
        "bootstrap": bootstrap,
        "bonferroni_family_arms": FAMILY_ARMS,
        "bonferroni_all_arms": BONFERRONI_ARMS,
        "selection_pr_c_probe_rule": primary,
        "selection_amended_rule": amended,
        "label": primary["label"],
        "label_pr_c_probe_rule": primary["label"],
        "label_amended_rule": amended["label"],
        "selected_arm_pr_c_probe_rule": primary["selected_arm"],
        "selected_arm_amended_rule": amended["selected_arm"],
        "label_note": (
            "primary label = PR-C probe rule (the arm must also trade before the "
            "probe cut); the amended rule is secondary and shown for continuity only"
        ),
        "implementation_notes": list(IMPLEMENTATION_NOTES),
        "notes": [
            "development evidence only; post hoc; never a WINNER and not a "
            "profitability claim",
            "the evaluation period was already used by PR-C, Phase 2 and the "
            "breakeven diagnostic; this is not confirmatory",
            "allocator cost c_i = fee + spread is a lower bound: participation impact "
            "is charged by the executor but not modelled by the allocator",
            "Bonferroni intervals are reported at 0.05/8 (the Phase 3 family) and "
            "0.05/21 (all development arms)",
        ],
    }
    equity_arrays: dict[str, np.ndarray] = {}
    for arm, run in runs.items():
        for scenario, outcome in run.outcomes.items():
            equity_arrays[f"{arm}__{scenario}__equity"] = outcome.equity
            equity_arrays[f"{arm}__{scenario}__timestamp_ns"] = outcome.timestamps_ns
    return results, equity_arrays


_SUMMARY_FIELDS = (
    "terminal_net_profit_pct",
    "cagr_pct",
    "max_drawdown_pct",
    "sharpe_daily",
    "turnover_total",
    "turnover_per_year",
    "execution_costs_usdt",
    "funding_pnl_usdt",
    "gross_price_pnl_usdt",
    "net_pnl_usdt",
    "n_trades",
    "per_year_return_pct",
)

_REPRODUCED_FIELDS = (
    "terminal_net_profit_pct",
    "max_drawdown_pct",
    "net_pnl_usdt",
    "execution_costs_usdt",
    "funding_pnl_usdt",
    "turnover_total",
    "n_trades",
)


def phase2_reproduction_check(
    results: Mapping[str, Any], phase2: Mapping[str, Any]
) -> dict[str, Any]:
    """Compare the re-run baselines with the committed Phase 2 result, exactly."""

    rows: dict[str, Any] = {}
    all_equal = True
    for arm in BASELINE_ARMS:
        for scenario in results["arms"][arm]:
            ours = results["arms"][arm][scenario]
            theirs = phase2["arms"][arm][scenario]
            diffs = {
                name: abs(float(ours[name]) - float(theirs[name]))
                for name in _REPRODUCED_FIELDS
            }
            equal = all(value == 0.0 for value in diffs.values())
            all_equal = all_equal and equal
            rows[f"{arm}/{scenario}"] = {"bit_identical": equal, "abs_diff": diffs}
    return {
        "source": "report/issue810-phase2-trial-results-20261029.json",
        "fields": list(_REPRODUCED_FIELDS),
        "all_bit_identical": all_equal,
        "rows": rows,
    }


def compact_phase3_summary(results: dict[str, Any]) -> dict[str, Any]:
    """Small tracked summary: one row per arm x scenario plus selection evidence."""

    rows: dict[str, dict[str, Any]] = {}
    for arm, scenarios in results["arms"].items():
        arm_rows: dict[str, Any] = {}
        for scenario, summary in scenarios.items():
            if summary.get("status") == LABEL_FAILED_TO_RUN:
                arm_rows[scenario] = dict(summary)
                continue
            row = {name: summary[name] for name in _SUMMARY_FIELDS}
            decisions = summary.get("allocator_decisions")
            row["fail_closed_hold_count"] = (
                None if decisions is None else decisions["fail_closed_hold_count"]
            )
            row["fail_closed_reasons"] = (
                None if decisions is None else decisions["fail_closed_reasons"]
            )
            row["n_allocator_decisions"] = (
                None if decisions is None else decisions["n_decisions"]
            )
            reconciliation = summary["reconciliation"]
            row["ledger_reconciled"] = reconciliation["passed"]
            row["terminal_flat"] = reconciliation["terminal_flat"]
            row["terminal_residual_quantities"] = reconciliation[
                "terminal_residual_quantities"
            ]
            row["terminal_residual_notional_usdt"] = reconciliation[
                "terminal_residual_notional_usdt"
            ]
            row["mean_gross_exposure"] = summary["exposure"]["mean_gross_exposure"]
            row["mean_net_exposure"] = summary["exposure"]["mean_net_exposure"]
            forecast = summary.get("forecast")
            if forecast is not None:
                row["share_abs_mu_above_one_way_cost"] = forecast[
                    "share_abs_mu_above_one_way_cost"
                ]
                row["mean_abs_mu_per_day"] = forecast["mean_abs_mu_per_day"]
            if "vol_target" in summary:
                row["vol_target"] = summary["vol_target"]
            arm_rows[scenario] = row
        rows[arm] = arm_rows
    keys = (
        "schema_version",
        "status",
        "label",
        "label_pr_c_probe_rule",
        "label_amended_rule",
        "selected_arm_pr_c_probe_rule",
        "selected_arm_amended_rule",
        "label_note",
        "roster_digest",
        "registry_digest",
        "base_protocol_digest",
        "dataset_id",
        "git_head",
        "trials",
        "baselines",
        "validity_pr_c_probe_rule",
        "validity_amended_rule",
        "validity_detail",
        "causality_probes",
        "causality_matrix",
        "causality_matrix_cut",
        "failures",
        "bootstrap",
        "bonferroni_family_arms",
        "bonferroni_all_arms",
        "selection_pr_c_probe_rule",
        "selection_amended_rule",
        "notes",
    )
    compact: dict[str, Any] = {key: results[key] for key in keys}
    compact["arms"] = rows
    compact["p3_arms"] = list(P3_ARMS)
    return compact


__all__ = [
    "PROBE_MATRIX_CUT",
    "ArmRun",
    "assemble",
    "compact_phase3_summary",
    "phase2_reproduction_check",
    "phase3_protocol",
    "run_arm",
    "run_phase3",
]
