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

import math
from collections.abc import Mapping
from dataclasses import replace
from typing import Any

from trade_rl.evaluation.dev_realdata_study.metrics import select_development_label
from trade_rl.evaluation.dev_realdata_study.protocol import (
    LABEL_NO_WINNER,
    PROTOCOL,
    DevStudyProtocol,
)
from trade_rl.evaluation.dev_realdata_study.trials import (
    CANDIDATE_POOL,
    CONTROL_ARMS,
    LABEL_FAILED_TO_RUN,
)

RULE_PR_C = "pr_c_probe_rule_requires_pre_cut_trades"
RULE_AMENDED = "amended_probe_rule_identical_decisions_only"


def _has_finite_selection_metrics(row: object) -> bool:
    """A partial/nonfinite metric row is not valid selection evidence."""

    if not isinstance(row, Mapping):
        return False
    try:
        return all(
            math.isfinite(float(row[key]))
            for key in ("terminal_net_profit_pct", "max_drawdown_pct")
        )
    except (KeyError, TypeError, ValueError, OverflowError):
        return False


def _select(
    result: Mapping[str, Any],
    validity: Mapping[str, bool],
    rule: str,
    *,
    protocol: DevStudyProtocol,
) -> dict[str, Any]:
    scenarios = protocol.scenarios
    raw_arms = result.get("arms", {})
    arms_table = raw_arms if isinstance(raw_arms, Mapping) else {}
    runnable = tuple(
        arm
        for arm in CANDIDATE_POOL
        if isinstance(arms_table.get(arm), Mapping)
        and all(
            _has_finite_selection_metrics(arms_table[arm].get(scenario))
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
        replace(protocol, candidate_arms=runnable),
    )
    # A candidate cannot be promoted when either control has not produced
    # valid replay evidence, even though CASH's theoretical profit is zero.
    # Rule-specific validity, not stale cached labels, controls admission.
    invalid_controls = tuple(
        arm for arm in CONTROL_ARMS if not validity.get(arm, False)
    )
    diagnostic_only = result.get("status") == "diagnostic_only"
    rows: dict[str, Any] = dict(selection.rows)
    for arm in CANDIDATE_POOL:
        if arm not in runnable:
            rows[arm] = {
                "valid": False,
                "eligible": False,
                "status": LABEL_FAILED_TO_RUN,
            }
    if invalid_controls or diagnostic_only:
        for row in rows.values():
            row["eligible"] = False
            row["blocked_reason"] = (
                "invalid_control" if invalid_controls else "diagnostic_only"
            )
    blocked = bool(invalid_controls or diagnostic_only)
    return {
        "rule": rule,
        "label": LABEL_NO_WINNER if blocked else selection.label,
        "selected_arm": None if blocked else selection.selected_arm,
        "invalid_controls": list(invalid_controls),
        "selection_blocked_reason": (
            "invalid_control"
            if invalid_controls
            else "diagnostic_only"
            if diagnostic_only
            else None
        ),
        "validity": {arm: bool(validity.get(arm, False)) for arm in CANDIDATE_POOL},
        "arms": {arm: rows[arm] for arm in CANDIDATE_POOL},
    }


def pr_c_probe_passed(probe: Mapping[str, Any]) -> bool:
    """PR-C probe rule: identical pre-cut trace AND the arm traded before the cut."""

    return bool(
        probe.get("identical_pre_cut_trace", False)
        and probe.get("pre_cut_trace_has_trades", False)
    )


def labels_under_both_rules(
    result: Mapping[str, Any],
    execution1: Mapping[str, Any] | None = None,
    *,
    protocol: DevStudyProtocol = PROTOCOL,
) -> dict[str, Any]:
    """Recompute the selection under the PR-C probe rule and under the amended rule.

    Replay and matrix validity are shared. Only the pre-cut-trade
    requirement differs; a matrix failure never becomes valid through the
    historical PR-C trade rule. The PR-C-rule probe evidence comes from
    ``execution1["causality_probes"]`` when supplied, not its derived
    ``passed`` flag. This is forward-looking hardening; committed historical
    records remain unmodified.
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
            and detail.get("complete_clock_coverage", True)
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
                    and result["validity_detail"][arm].get(
                        "causality_matrix_passed", False
                    )
                    and pr_c_probe_passed(probes[arm])
                )
            )
        )
        for arm in replay_ok
    }
    amended_validity = {
        arm: bool(
            value
            and replay_ok.get(arm, False)
            and (
                arm in CONTROL_ARMS
                or result["validity_detail"][arm].get("causality_matrix_passed", False)
            )
        )
        for arm, value in result["validity"].items()
    }
    return {
        "label_pr_c_probe_rule": _select(
            result, pr_c_validity, RULE_PR_C, protocol=protocol
        ),
        "label_amended_rule": _select(
            result, amended_validity, RULE_AMENDED, protocol=protocol
        ),
    }


def with_both_labels(
    result: Mapping[str, Any],
    execution1: Mapping[str, Any] | None = None,
    *,
    protocol: DevStudyProtocol = PROTOCOL,
) -> dict[str, Any]:
    """Return ``result`` with both labels; ``label`` is the PR-C probe rule label.

    Any ``selection`` / ``label`` already in ``result`` (the amended-rule outcome of
    the study's own selection step) is kept as ``selection_amended_rule`` and not
    used as the headline.
    """

    both = labels_under_both_rules(result, execution1, protocol=protocol)
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
    protocol: DevStudyProtocol = PROTOCOL,
) -> dict[str, Any]:
    """Post-hoc: ``with_both_labels`` plus a record of the unchanged original file."""

    annotated = with_both_labels(result, execution1, protocol=protocol)
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
