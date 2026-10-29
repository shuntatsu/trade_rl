"""Post-hoc side-by-side labels for the Phase 2 result (no new study run).

Two full real-data executions exist. Execution 1 applied the PR-C causality-probe
validity rule, which requires the arm to trade before the probe cut; execution 2
applied an amended rule (identical decisions and traces, trades not required) that
was changed in the code while execution 1 was still running. The two rules give
different labels, so both are computed here by code from the committed replay
metrics plus execution 1's recorded probe outcomes. The PR-C rule is primary.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from trade_rl.evaluation.dev_realdata_study.metrics import select_development_label
from trade_rl.evaluation.dev_realdata_study.protocol import PROTOCOL
from trade_rl.evaluation.dev_realdata_study.trial_study import phase2_protocol
from trade_rl.evaluation.dev_realdata_study.trials import (
    CANDIDATE_POOL,
    CONTROL_ARMS,
    LABEL_FAILED_TO_RUN,
)

RULE_PR_C = "pr_c_probe_rule_requires_pre_cut_trades (execution 1)"
RULE_AMENDED = "amended_probe_rule_identical_decisions_only (execution 2)"


def _select(
    result: Mapping[str, Any], validity: Mapping[str, bool], rule: str
) -> dict[str, Any]:
    scenarios = PROTOCOL.scenarios
    runnable = tuple(
        arm
        for arm in CANDIDATE_POOL
        if all(
            "terminal_net_profit_pct" in result["arms"][arm][scenario]
            for scenario in scenarios
        )
    )
    selection = select_development_label(
        {
            arm: {
                scenario: {
                    "terminal_net_profit_pct": float(
                        result["arms"][arm][scenario]["terminal_net_profit_pct"]
                    ),
                    "max_drawdown_pct": float(
                        result["arms"][arm][scenario]["max_drawdown_pct"]
                    ),
                }
                for scenario in scenarios
            }
            for arm in runnable
        },
        validity,
        _pool_protocol(runnable),
    )
    rows: dict[str, Any] = dict(selection.rows)
    for arm in CANDIDATE_POOL:
        if arm not in runnable:
            rows[arm] = {
                "valid": False,
                "eligible": False,
                "status": LABEL_FAILED_TO_RUN,
            }
    return {
        "rule": rule,
        "label": selection.label,
        "selected_arm": selection.selected_arm,
        "validity": {arm: bool(validity.get(arm, False)) for arm in CANDIDATE_POOL},
        "arms": {arm: rows[arm] for arm in CANDIDATE_POOL},
    }


def _pool_protocol(runnable: tuple[str, ...]) -> Any:
    from dataclasses import replace

    return replace(phase2_protocol(PROTOCOL), candidate_arms=runnable)


def labels_under_both_rules(
    result: Mapping[str, Any], execution1: Mapping[str, Any]
) -> dict[str, Any]:
    """Recompute the selection under the PR-C rule and under the amended rule.

    ``result`` is the committed Phase 2 result (execution 2 metrics and validity
    parts); ``execution1`` is the committed execution-1 summary (its per-arm probe
    outcomes under the PR-C rule). Replay validity (ran, reconciled, flat, no
    termination) is taken from ``result``; only the probe rule differs.
    """

    replay_ok: dict[str, bool] = {}
    for arm, detail in result["validity_detail"].items():
        replay_ok[arm] = bool(
            detail["all_scenarios_ran"]
            and detail["ledger_reconciled_all_scenarios"]
            and detail["no_termination_all_scenarios"]
        )
    pr_c_validity = {
        arm: bool(
            replay_ok[arm]
            and (
                arm in CONTROL_ARMS
                or bool(execution1["causality_probes"][arm]["passed"])
            )
        )
        for arm in replay_ok
    }
    amended_validity = {arm: bool(value) for arm, value in result["validity"].items()}
    return {
        "label_preregistered_rule": _select(result, pr_c_validity, RULE_PR_C),
        "label_amended_rule": _select(result, amended_validity, RULE_AMENDED),
    }


def annotate_result(
    result: Mapping[str, Any],
    execution1: Mapping[str, Any],
    *,
    original_sha256: str,
) -> dict[str, Any]:
    """Return ``result`` with both labels side by side and the PR-C label primary."""

    both = labels_under_both_rules(result, execution1)
    annotated = {key: value for key, value in result.items() if key != "selection"}
    annotated["selection_amended_rule"] = result["selection"]
    annotated["selection_preregistered_rule"] = both["label_preregistered_rule"]
    annotated["label_preregistered_rule"] = both["label_preregistered_rule"]["label"]
    annotated["label_amended_rule"] = both["label_amended_rule"]["label"]
    annotated["selected_arm_preregistered_rule"] = both["label_preregistered_rule"][
        "selected_arm"
    ]
    annotated["selected_arm_amended_rule"] = both["label_amended_rule"]["selected_arm"]
    annotated["label"] = annotated["label_preregistered_rule"]
    annotated["label_note"] = (
        "primary label = PR-C probe rule (execution 1); the amended rule changed "
        "validity after execution 1 had started, see the provenance JSON"
    )
    annotated["post_hoc_annotation"] = {
        "applied_by": "trade_rl.evaluation.dev_realdata_study.trial_labels",
        "original_result_sha256": original_sha256,
        "original_label_field": result["label"],
        "replay_metrics_changed": False,
    }
    return annotated


__all__ = [
    "RULE_AMENDED",
    "RULE_PR_C",
    "annotate_result",
    "labels_under_both_rules",
]
