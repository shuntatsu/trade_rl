"""Run the Phase 2 trial registry on the shared development real-data runner.

One task replays one arm in all three scenarios (so vintages are fitted once per
arm) and, for forecasting/allocator arms, runs the real-data future-perturbation
causality probe for that arm. Every task is isolated: an exception becomes a
``FAILED_TO_RUN`` row and the remaining arms still run. Nothing is retried with
other parameters and nothing here can emit a ``WINNER``.
"""

from __future__ import annotations

import traceback
from collections.abc import Callable
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
from trade_rl.evaluation.dev_realdata_study.protocol import (
    ARM_BUY_AND_HOLD,
    ARM_CASH,
    DevStudyProtocol,
    protocol_digest,
)
from trade_rl.evaluation.dev_realdata_study.trial_sources import (
    TrialContext,
    build_phase2_arm,
)
from trade_rl.evaluation.dev_realdata_study.trials import (
    ALL_ARMS,
    CANDIDATE_POOL,
    CONSTANTS,
    CONTROL_ARMS,
    IMPLEMENTATION_DECISIONS,
    LABEL_FAILED_TO_RUN,
    RESULT_SCHEMA,
    TRIAL_ARMS,
    TRIALS,
    registry_digest,
)

Progress = Callable[[str], None]


def make_context(dataset: MarketDataset, protocol: DevStudyProtocol) -> TrialContext:
    """Module-level so the causality probe can build sources for a perturbed copy."""

    return TrialContext(dataset, protocol)


@dataclass
class ArmRun:
    """Everything one arm produced: outcomes per scenario, the probe, failures."""

    arm: str
    outcomes: dict[str, runner.ReplayOutcome]
    failures: dict[str, dict[str, str]]
    probe: dict[str, object] | None


def _failure(exc: BaseException) -> dict[str, str]:
    return {
        "status": LABEL_FAILED_TO_RUN,
        "error_type": type(exc).__name__,
        "error": str(exc)[:600],
        "traceback_tail": traceback.format_exc()[-1500:],
    }


def run_arm(
    dataset: MarketDataset,
    protocol: DevStudyProtocol,
    arm: str,
    *,
    start_index: int,
    stop_index: int,
    say: Progress,
) -> ArmRun:
    """All scenarios and the causality probe for one arm; failures are recorded."""

    outcomes: dict[str, runner.ReplayOutcome] = {}
    failures: dict[str, dict[str, str]] = {}
    probe: dict[str, object] | None = None
    context = make_context(dataset, protocol)
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
                arm_builder=build_phase2_arm,
            )
        except Exception as exc:  # recorded, never retried with other parameters
            failures[scenario] = _failure(exc)
            say(f"FAILED {arm} / {scenario}: {type(exc).__name__}: {exc}")
    if arm not in CONTROL_ARMS:
        say(f"causality probe {arm}")
        try:
            probe = runner.causality_probe(
                dataset,
                protocol,
                context,
                start_index=start_index,
                arm=arm,
                arm_builder=build_phase2_arm,
                sources_builder=make_context,
            )
        except Exception as exc:
            failures["causality_probe"] = _failure(exc)
            say(f"FAILED {arm} causality probe: {type(exc).__name__}: {exc}")
    return ArmRun(arm, outcomes, failures, probe)


def _arm_worker(
    loader: Callable[[], MarketDataset],
    protocol: DevStudyProtocol,
    arm: str,
    start_index: int,
    stop_index: int,
) -> ArmRun:
    dataset = loader()
    return run_arm(
        dataset,
        protocol,
        arm,
        start_index=start_index,
        stop_index=stop_index,
        say=lambda message: print(message, flush=True),
    )


def phase2_protocol(protocol: DevStudyProtocol) -> DevStudyProtocol:
    """The PR-C protocol with the Phase 2 rosters and the 13-arm Bonferroni count."""

    return replace(
        protocol,
        arms=ALL_ARMS,
        candidate_arms=CANDIDATE_POOL,
        bonferroni_comparisons=CONSTANTS.bonferroni_arms,
    )


def run_phase2(
    dataset: MarketDataset,
    protocol: DevStudyProtocol,
    *,
    progress: Progress | None = None,
    parallel_loader: Callable[[], MarketDataset] | None = None,
    max_workers: int = 8,
    arms: tuple[str, ...] = ALL_ARMS,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Run the PR-C arms and the eight registered trials; return results and equity.

    ``protocol`` is the PR-C protocol (its digest is recorded and must match the
    registry). With ``parallel_loader`` each arm runs in its own process.
    """

    say = progress or (lambda message: None)
    if dataset.dataset_id != protocol.dataset_id:
        raise ValueError("dataset_id does not match the protocol")
    if tuple(arms) != ALL_ARMS:
        raise ValueError("arm roster differs from the registered Phase 2 roster")
    start_index = runner._row_of(dataset, protocol.evaluation_start)
    stop_index = runner._row_of(dataset, protocol.evaluation_end)

    runs: dict[str, ArmRun] = {}
    if parallel_loader is not None:
        heavy_first = sorted(arms, key=lambda arm: arm in CONTROL_ARMS)
        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                arm: pool.submit(
                    _arm_worker, parallel_loader, protocol, arm, start_index, stop_index
                )
                for arm in heavy_first
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
            )
    return assemble(dataset, protocol, runs)


def assemble(
    dataset: MarketDataset,
    protocol: DevStudyProtocol,
    runs: dict[str, ArmRun],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Validity, bootstrap, selection and the result mapping from finished arm runs."""

    p2 = phase2_protocol(protocol)
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

    validity: dict[str, bool] = {}
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
        needs_probe = arm not in CONTROL_ARMS
        probe_ok = (not needs_probe) or (
            run.probe is not None and bool(run.probe["passed"])
        )
        validity[arm] = bool(complete and reconciled and no_termination and probe_ok)
        validity_detail[arm] = {
            "all_scenarios_ran": complete,
            "ledger_reconciled_all_scenarios": reconciled,
            "no_termination_all_scenarios": no_termination,
            "causality_probe_passed": (
                None if not needs_probe else (run.probe or {}).get("passed", False)
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
            block=p2.bootstrap_block_days,
            resamples=p2.bootstrap_resamples,
            seed=p2.bootstrap_seed,
        )
        table: dict[str, dict[str, dict[str, float]]] = {}
        for arm, series in daily.items():
            if arm == ARM_CASH:
                continue
            versus = {
                "vs_cash": bootstrap_excess(
                    series - daily[ARM_CASH], indices, p2
                ).to_mapping()
            }
            if arm != ARM_BUY_AND_HOLD:
                versus["vs_equal_weight_buy_and_hold"] = bootstrap_excess(
                    series - daily[ARM_BUY_AND_HOLD], indices, p2
                ).to_mapping()
            table[arm] = versus
        bootstrap[scenario] = table

    runnable = tuple(arm for arm in CANDIDATE_POOL if ran_everywhere(arm))
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
        replace(p2, candidate_arms=runnable),
    )
    selection_rows: dict[str, Any] = dict(selection.rows)
    for arm in CANDIDATE_POOL:
        if arm not in runnable:
            selection_rows[arm] = {
                "valid": False,
                "eligible": False,
                "status": LABEL_FAILED_TO_RUN,
            }
    selection_mapping: dict[str, Any] = {
        "label": selection.label,
        "selected_arm": selection.selected_arm,
        "arms": {arm: selection_rows[arm] for arm in CANDIDATE_POOL},
    }

    results: dict[str, Any] = {
        "schema_version": RESULT_SCHEMA,
        "status": "development_evidence_only",
        "registry_digest": registry_digest(),
        "base_protocol_digest": protocol_digest(protocol),
        "dataset_id": dataset.dataset_id,
        "symbols": list(dataset.symbols),
        "git_head": runner._git_head(),
        "trials": [
            {
                "trial_id": spec.trial_id,
                "arm": spec.arm,
                "title": spec.title,
                "status": ("ran" if ran_everywhere(spec.arm) else LABEL_FAILED_TO_RUN),
            }
            for spec in TRIALS
        ],
        "causality_probes": {
            arm: runs[arm].probe for arm in ALL_ARMS if arm not in CONTROL_ARMS
        },
        "failures": {arm: runs[arm].failures for arm in ALL_ARMS if runs[arm].failures},
        "validity": validity,
        "validity_detail": validity_detail,
        "arms": arms_table,
        "bootstrap": bootstrap,
        "bonferroni_comparisons": p2.bonferroni_comparisons,
        "selection": selection_mapping,
        "label": selection.label,
        "implementation_decisions": list(IMPLEMENTATION_DECISIONS),
        "notes": [
            "development evidence only; never a WINNER",
            "the evaluation period was already used by the PR-C study and the "
            "breakeven diagnostic; this is not confirmatory",
            "allocator cost c_i = fee + spread is a lower bound: participation impact "
            "is charged by the executor but not modelled by the allocator",
            "fail_closed_hold_count counts allocator decisions that ended as a "
            "fail-closed HOLD",
            "Bonferroni intervals use level 0.05 / 13 (5 PR-C arms plus 8 trials)",
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


def compact_phase2_summary(results: dict[str, Any]) -> dict[str, Any]:
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
            row["n_allocator_decisions"] = (
                None if decisions is None else decisions["n_decisions"]
            )
            row["ledger_reconciled"] = summary["reconciliation"]["passed"]
            row["terminal_flat"] = summary["reconciliation"]["terminal_flat"]
            row["mean_gross_exposure"] = summary["exposure"]["mean_gross_exposure"]
            row["mean_net_exposure"] = summary["exposure"]["mean_net_exposure"]
            forecast = summary.get("forecast")
            if forecast is not None:
                row["share_abs_mu_above_one_way_cost"] = forecast[
                    "share_abs_mu_above_one_way_cost"
                ]
                row["mean_abs_mu_per_day"] = forecast["mean_abs_mu_per_day"]
            if "overlay" in summary:
                row["overlay"] = summary["overlay"]
            arm_rows[scenario] = row
        rows[arm] = arm_rows
    return {
        "schema_version": results["schema_version"],
        "status": results["status"],
        "label": results["label"],
        "registry_digest": results["registry_digest"],
        "base_protocol_digest": results["base_protocol_digest"],
        "dataset_id": results["dataset_id"],
        "git_head": results["git_head"],
        "trials": results["trials"],
        "validity": results["validity"],
        "validity_detail": results["validity_detail"],
        "causality_probes": results["causality_probes"],
        "failures": results["failures"],
        "arms": rows,
        "bootstrap": results["bootstrap"],
        "bonferroni_comparisons": results["bonferroni_comparisons"],
        "selection": results["selection"],
        "notes": results["notes"],
        "trial_arms": list(TRIAL_ARMS),
    }


__all__ = [
    "ArmRun",
    "assemble",
    "compact_phase2_summary",
    "make_context",
    "phase2_protocol",
    "run_arm",
    "run_phase2",
]
