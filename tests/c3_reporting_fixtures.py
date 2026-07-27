from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

from trade_rl.artifacts.codec import canonical_json_bytes
from trade_rl.artifacts.hashing import content_digest

SOURCE_RUN_DIGEST = "1" * 64
CORE_REPORT_DIGEST = "2" * 64
CONFIG_DIGEST = "3" * 64


def valid_summary_payload() -> dict[str, object]:
    folds = [
        {
            "effective_days": 30,
            "failure_reasons": [],
            "fold_id": f"fold-{index}",
            "mean_regret_margin": 0.03 + index * 0.001,
            "mean_spearman": 0.20 + index * 0.01,
            "mean_uplift": 0.010 + index * 0.001,
            "perfect_information_valid": True,
            "required_adverse_passed": True,
            "scenario_oracle_max_drawdown": 0.10 + index * 0.005,
            "selection_days": 30,
            "trend_max_drawdown": 0.12 + index * 0.005,
        }
        for index in range(6)
    ]
    execution_summaries = [
        {
            "execution_scenario": "adverse_spread_2x",
            "maximum_drawdown": 0.14,
            "mean_borrow_paid": 0.0001,
            "mean_fees": 0.001,
            "mean_fill_ratio": 0.92,
            "mean_filled_turnover": 0.40,
            "mean_funding_paid": 0.0002,
            "mean_gross_log_return": 0.009,
            "mean_impact_cost": 0.0005,
            "mean_spread_cost": 0.0007,
            "mean_total_economic_cost": 0.0025,
            "observation_count": 180,
            "policy_kind": "scenario_oracle",
            "termination_distribution": [["horizon", 180]],
            "total_cancel_replace_events": 12,
            "total_fill_count": 720,
            "total_pending_order_events": 8,
        },
        {
            "execution_scenario": "nominal",
            "maximum_drawdown": 0.12,
            "mean_borrow_paid": 0.0001,
            "mean_fees": 0.0008,
            "mean_fill_ratio": 0.96,
            "mean_filled_turnover": 0.38,
            "mean_funding_paid": 0.0002,
            "mean_gross_log_return": 0.012,
            "mean_impact_cost": 0.0003,
            "mean_spread_cost": 0.0004,
            "mean_total_economic_cost": 0.0018,
            "observation_count": 180,
            "policy_kind": "scenario_oracle",
            "termination_distribution": [["horizon", 180]],
            "total_cancel_replace_events": 8,
            "total_fill_count": 760,
            "total_pending_order_events": 4,
        },
    ]
    payload: dict[str, object] = {
        "all_perfect_information_valid": True,
        "all_required_adverse_passed": True,
        "anchor_max_share": 0.05,
        "bootstrap_block_days": 7,
        "bootstrap_resamples": 1000,
        "calibration_buckets": [
            {
                "bucket_index": 0,
                "maximum_score": 0.2,
                "minimum_score": 0.0,
                "predicted_loss_cvar": 0.02,
                "predicted_mean_advantage": 0.01,
                "realized_downside_mean": -0.005,
                "realized_mean_advantage": 0.008,
                "sample_count": 180,
            }
        ],
        "config_digest": CONFIG_DIGEST,
        "core_report_digest": CORE_REPORT_DIGEST,
        "effective_anchor_count": 64.0,
        "execution_summaries": execution_summaries,
        "failure_reasons": [],
        "folds": folds,
        "historical_coverage_fraction": 0.80,
        "mean_regret_margin": 0.033,
        "mean_spearman": 0.225,
        "mean_uplift": 0.0125,
        "neighbor_distance_p50": 0.10,
        "neighbor_distance_p90": 0.20,
        "neighbor_distance_p99": 0.30,
        "positive_uplift_folds": 6,
        "production_status": "NO-GO",
        "regret_margin_lower_ci": 0.010,
        "regret_margin_upper_ci": 0.050,
        "schema_version": "causal_scenario_c3_aggregate_summary_v1",
        "source_run_digest": SOURCE_RUN_DIGEST,
        "spearman_lower_ci": 0.10,
        "spearman_upper_ci": 0.35,
        "total_effective_days": 180,
        "total_selection_days": 180,
        "unique_anchor_count": 100,
        "uplift_lower_ci": 0.005,
        "uplift_p_value": 0.01,
        "uplift_upper_ci": 0.020,
        "worst_scenario_oracle_drawdown": 0.125,
        "worst_trend_drawdown": 0.145,
    }
    payload["summary_digest"] = content_digest(payload)
    return payload


def refreshed(payload: dict[str, object]) -> dict[str, object]:
    result = deepcopy(payload)
    result.pop("summary_digest", None)
    result["summary_digest"] = content_digest(result)
    return result


def write_summary(
    path: Path,
    payload: dict[str, object] | None = None,
    *,
    canonical: bool = True,
) -> dict[str, object]:
    resolved = valid_summary_payload() if payload is None else deepcopy(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    if canonical:
        path.write_bytes(canonical_json_bytes(resolved))
    else:
        path.write_text(json.dumps(resolved, indent=2), encoding="utf-8")
    return resolved
