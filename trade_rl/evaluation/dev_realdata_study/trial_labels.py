"""Both Phase 2 labels, side by side, computed by code.

Two causality-probe validity rules exist. The PR-C probe rule (inherited from the
PR-C runner, not stated in the registry text) requires the arm to trade before the
probe cut; the amended rule (identical decisions and traces, trades not required)
was introduced in the code while real-data execution 1 was still running. They give
different labels, so ``labels_under_both_rules`` computes both from a result mapping:
the replay metrics and validity parts, plus the per-arm probe outcomes (either the
result's own ``causality_probes`` or a committed execution-1 summary). The PR-C
probe rule label is primary (``label``); a fresh study run therefore never reports
the amended label as its headline. Neither rule changes any replay metric.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import Any

from trade_rl.evaluation.dev_realdata_study.metrics import select_development_label
from trade_rl.evaluation.dev_realdata_study.protocol import PROTOCOL
from trade_rl.evaluation.dev_realdata_study.trials import (
    CANDIDATE_POOL,
    CONTROL_ARMS,
    LABEL_FAILED_TO_RUN,
)

RULE_PR_C = "pr_c_probe_rule_requires_pre_cut_trades"
RULE_AMENDED = "amended_probe_rule_identical_decisions_only"


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
        replace(PROTOCOL, candidate_arms=runnable),
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


def pr_c_probe_passed(probe: Mapping[str, Any]) -> bool:
    """PR-C probe rule: identical pre-cut trace AND the arm traded before the cut."""

    return bool(probe["identical_pre_cut_trace"] and probe["pre_cut_trace_has_trades"])


def labels_under_both_rules(
    result: Mapping[str, Any], execution1: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Recompute the selection under the PR-C probe rule and under the amended rule.

    Replay validity (ran, reconciled, flat, no termination) is taken from ``result``;
    only the probe rule differs. The PR-C-rule probe outcome comes from
    ``execution1["causality_probes"]`` when given (a committed execution-1 summary)
    and otherwise from the result's own per-arm probe records.
    """

    probes = (
        result["causality_probes"]
        if execution1 is None
        else execution1["causality_probes"]
    )
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
                or (
                    arm in probes
                    and probes[arm] is not None
                    and (
                        bool(probes[arm]["passed"])
                        if execution1 is not None
                        else pr_c_probe_passed(probes[arm])
                    )
                )
            )
        )
        for arm in replay_ok
    }
    amended_validity = {arm: bool(value) for arm, value in result["validity"].items()}
    return {
        "label_pr_c_probe_rule": _select(result, pr_c_validity, RULE_PR_C),
        "label_amended_rule": _select(result, amended_validity, RULE_AMENDED),
    }


def with_both_labels(
    result: Mapping[str, Any], execution1: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Return ``result`` with both labels; ``label`` is the PR-C probe rule label.

    Any ``selection`` / ``label`` already in ``result`` (the amended-rule outcome of
    the study's own selection step) is kept as ``selection_amended_rule`` and not
    used as the headline.
    """

    both = labels_under_both_rules(result, execution1)
    out = {
        key: value for key, value in result.items() if key not in {"selection", "label"}
    }
    primary = both["label_pr_c_probe_rule"]
    amended = both["label_amended_rule"]
    out["selection_pr_c_probe_rule"] = primary
    out["selection_amended_rule"] = amended
    out["label_pr_c_probe_rule"] = primary["label"]
    out["label_amended_rule"] = amended["label"]
    out["selected_arm_pr_c_probe_rule"] = primary["selected_arm"]
    out["selected_arm_amended_rule"] = amended["selected_arm"]
    out["label"] = primary["label"]
    out["label_note"] = (
        "primary label = PR-C probe rule (arm must also trade before the probe cut); "
        "the amended rule changed validity after real-data execution 1 had started, "
        "see the provenance JSON"
    )
    return out


def annotate_result(
    result: Mapping[str, Any],
    execution1: Mapping[str, Any] | None,
    *,
    original_sha256: str,
) -> dict[str, Any]:
    """Post-hoc: ``with_both_labels`` plus a record of the unchanged original file."""

    annotated = with_both_labels(result, execution1)
    annotated["post_hoc_annotation"] = {
        "applied_by": "trade_rl.evaluation.dev_realdata_study.trial_labels",
        "original_result_sha256": original_sha256,
        "original_label_field": result.get("label"),
        "replay_metrics_changed": False,
    }
    return annotated


__all__ = [
    "RULE_AMENDED",
    "RULE_PR_C",
    "annotate_result",
    "labels_under_both_rules",
    "pr_c_probe_passed",
    "with_both_labels",
]
