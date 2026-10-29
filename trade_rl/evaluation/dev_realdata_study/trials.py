"""Frozen Phase 2 trial registry for the Issue #810 development real-data study.

The eight trials T1-T8 were fixed on 2026-10-29 BEFORE the PR-C (#1037) result
existed, and the implementation decisions below were fixed after PR-C but before
any T1-T8 result. This module is committed, together with its JSON export
(``report/issue810-phase2-trial-registry-20261029.json``), before the first real
data run. The PR-C result is context only: nothing here is tuned to it.

Same data, period, capital, costs, scenarios, metrics, selection rule and allocator
parameters as PR-C (:data:`~trade_rl.evaluation.dev_realdata_study.protocol.PROTOCOL`)
unless a trial states otherwise. Each trial is one fixed configuration, run once,
and reported whether it succeeds, fails the gate or fails to run.

Trials (expected-return sources, all feeding the same cost-aware allocator):

* T1 time-series momentum: blend of 30d and 90d past-return signs scaled by EWMA
  volatility to a 20% annualised per-asset target.
* T2 cross-sectional momentum: 90d return skipping the last day, demeaned.
* T3 funding carry tilt: minus the trailing 7d mean funding rate per 24h.
* T4 Ridge 24h on all 1h/4h/1d features (funding included).
* T5 LightGBM 24h on the 1h non-funding roster, fixed shallow parameters.
* T6 equal-weight blend of scale-normalised Ridge24 (PR-C), T1 and T3.
* T7 T6 plus an in-policy drawdown overlay with a latch.
* T8 T6 with weekly (168h, Monday 00:00 UTC) decisions.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Final

from trade_rl.evaluation.dev_realdata_study.protocol import (
    PROTOCOL,
    protocol_digest,
)

REGISTRY_SCHEMA: Final = "dev_realdata_phase2_trial_registry_v1"
RESULT_SCHEMA: Final = "dev_realdata_phase2_trial_results_v1"
REGISTRY_REPORT_PATH: Final = "report/issue810-phase2-trial-registry-20261029.json"
RESULT_REPORT_PATH: Final = "report/issue810-phase2-trial-results-20261029.json"

# Source kinds (keys of the expected-return-source registry in ``trial_sources``).
KIND_TSMOM: Final = "tsmom"
KIND_XSMOM: Final = "xsmom"
KIND_FUNDING_CARRY: Final = "funding_carry"
KIND_RIDGE_ALL: Final = "ridge_all_features"
KIND_LIGHTGBM: Final = "lightgbm_1h"
KIND_BLEND: Final = "blend_ridge24_tsmom_carry"

# Hours since the Unix epoch (a Thursday) modulo 168 that land on Monday 00:00 UTC.
WEEKLY_MONDAY_ANCHOR_HOUR: Final = 96


@dataclass(frozen=True)
class Phase2Constants:
    """Every numeric constant shared by the trial sources (fixed pre-run)."""

    # T1 time-series momentum.
    tsmom_windows_days: tuple[int, ...] = (30, 90)
    tsmom_vol_target: float = 0.20
    tsmom_vol_halflife_days: float = 60.0
    tsmom_vol_min_periods: int = 30
    tsmom_days_per_year: int = 365
    # mu = signal_cost_multiple * s * (vol_target / ann_vol) * one_way_cost.
    signal_cost_multiple: float = 2.0
    # T2 cross-sectional momentum.
    xsmom_lookback_days: int = 90
    xsmom_skip_days: int = 1
    # T3 funding carry.
    carry_window_days: int = 7
    # T4 feature roster: every dataset feature with one of these prefixes.
    ridge_all_prefixes: tuple[str, ...] = ("1h__", "4h__", "1d__")
    # T5 LightGBM (fixed; ``num_leaves = 2 ** max_depth``; other settings as the
    # repository's shallow LightGBM candidate).
    lgbm_n_estimators: int = 200
    lgbm_max_depth: int = 3
    lgbm_num_leaves: int = 8
    lgbm_learning_rate: float = 0.05
    lgbm_min_child_samples: int = 200
    lgbm_reg_lambda: float = 1.0
    lgbm_random_state: int = 0
    # T6/T7/T8 blend: causal trailing standard deviation over symbols and dates.
    blend_std_window_days: int = 365
    blend_min_dates: int = 10
    # T7 overlay.
    overlay_dd_start: float = 0.10
    overlay_dd_limit: float = 0.20
    overlay_release_fraction: float = 0.95
    # Multiple testing: 5 PR-C arms plus the 8 trials.
    bonferroni_arms: int = 13


CONSTANTS: Final = Phase2Constants()


@dataclass(frozen=True)
class TrialSpec:
    """One registered configuration (a row of the trial registry)."""

    trial_id: str
    arm: str
    title: str
    plan_text: str
    source_kind: str
    source_params: dict[str, Any] = field(default_factory=dict)
    decision_interval_hours: int = 24
    decision_anchor_hour_utc: int = 0
    long_only: bool = False
    overlay: str | None = None


TRIALS: Final = (
    TrialSpec(
        trial_id="T1",
        arm="t1_tsmom",
        title="time-series momentum",
        plan_text=(
            "mu_i = k * sign-weighted blend of 30d and 90d past returns, scaled by EWMA "
            "vol to a 20% annualized per-asset vol; k fixed so |mu| median ~ 2x "
            "one-way cost"
        ),
        source_kind=KIND_TSMOM,
        source_params={"windows_days": [30, 90], "vol_target": 0.20},
    ),
    TrialSpec(
        trial_id="T2",
        arm="t2_xsmom",
        title="cross-sectional momentum",
        plan_text=(
            "cross-sectional momentum (90d, skip last 1d) rank -> mu, dollar-neutral, "
            "gross 1.0"
        ),
        source_kind=KIND_XSMOM,
        source_params={"lookback_days": 90, "skip_days": 1},
    ),
    TrialSpec(
        trial_id="T3",
        arm="t3_funding_carry",
        title="funding carry tilt",
        plan_text=(
            "mu_i = -(trailing 7d mean funding rate per 24h) (short crowded longs)"
        ),
        source_kind=KIND_FUNDING_CARRY,
        source_params={"window_days": 7},
    ),
    TrialSpec(
        trial_id="T4",
        arm="t4_ridge24_all_features",
        title="Ridge 24h with funding and 4h/1d features",
        plan_text="Ridge 24h with added funding + 4h/1d features (if dataset has them)",
        source_kind=KIND_RIDGE_ALL,
        source_params={"horizon_hours": 24, "alpha": 10.0},
    ),
    TrialSpec(
        trial_id="T5",
        arm="t5_lightgbm24",
        title="LightGBM 24h",
        plan_text=(
            "LightGBM 24h (forecast-gbm extra), fixed params: 200 trees, depth 3, "
            "lr 0.05, min_leaf 200"
        ),
        source_kind=KIND_LIGHTGBM,
        source_params={"horizon_hours": 24},
    ),
    TrialSpec(
        trial_id="T6",
        arm="t6_blend",
        title="blend of Ridge24, T1 and T3",
        plan_text=("equal-weight blend of z-scored mu from Ridge24 + T1 + T3"),
        source_kind=KIND_BLEND,
        source_params={"components": ["ridge24_1h", "tsmom", "funding_carry"]},
    ),
    TrialSpec(
        trial_id="T7",
        arm="t7_blend_dd_overlay",
        title="T6 with drawdown overlay",
        plan_text=(
            "T6 with drawdown overlay (scale exposure linearly 1.0 -> 0.0 as running "
            "DD goes 10% -> 20%; latch until new equity high*0.95)"
        ),
        source_kind=KIND_BLEND,
        source_params={"components": ["ridge24_1h", "tsmom", "funding_carry"]},
        overlay="drawdown_latch",
    ),
    TrialSpec(
        trial_id="T8",
        arm="t8_blend_weekly",
        title="T6 with weekly decisions",
        plan_text="T6 with weekly (168h) decision cadence",
        source_kind=KIND_BLEND,
        source_params={"components": ["ridge24_1h", "tsmom", "funding_carry"]},
        decision_interval_hours=168,
        decision_anchor_hour_utc=WEEKLY_MONDAY_ANCHOR_HOUR,
    ),
)

TRIAL_ARMS: Final = tuple(trial.arm for trial in TRIALS)
if len(TRIALS) != 8 or len(set(TRIAL_ARMS)) != 8:  # pragma: no cover - import guard
    raise RuntimeError("the Phase 2 registry must hold exactly eight unique trials")

# Arms replayed in this study. The five PR-C arms are re-run in the same code path
# (deterministic) so every comparison is paired and the study is self-contained.
PR_C_ARMS: Final = PROTOCOL.arms
PR_C_CANDIDATE_ARMS: Final = PROTOCOL.candidate_arms
ALL_ARMS: Final = (*PR_C_ARMS, *TRIAL_ARMS)
CANDIDATE_POOL: Final = (*PR_C_CANDIDATE_ARMS, *TRIAL_ARMS)
CONTROL_ARMS: Final = (PR_C_ARMS[0], PR_C_ARMS[1])

LABEL_FAILED_TO_RUN: Final = "FAILED_TO_RUN"

IMPLEMENTATION_DECISIONS: Final = (
    "post-PR-C implementation decisions (fixed before any T1-T8 result):",
    "all signals use only marks and funding at or before the decision time; marks "
    "are the daily 00:00 UTC rows of the 1h dataset, funding is the funding_rate "
    "of due rows",
    "T1: r30 and r90 are the past 30d and 90d simple returns of daily marks; "
    "s = 0.5*sign(r30) + 0.5*sign(r90); ann_vol is the annualized (sqrt 365) EWMA "
    "(halflife 60d, normalized weights, non-centered) daily-return vol with at "
    "least 30 returns; mu_i = 2 * s * (0.20 / ann_vol_i) * one_way_cost, where "
    "one_way_cost = fee + spread = 0.0007",
    "T2: score = mark(t-1d)/mark(t-1d-90d) - 1 (a 90d return ending one day before "
    "the decision), demeaned across symbols and scaled so mean |mu| = 2*one_way_cost; "
    "an all-zero score gives mu = 0",
    "T2 net exposure: the allocator has no net-exposure constraint, so |sum w| <= 0.05 "
    "is NOT enforced; the mu are demeaned only (declared deviation); gross is the "
    "allocator's 0.95 rather than the plan's 1.0",
    "T3: mu_i = -(sum of funding_rate over due rows in (t-7d, t]) / 7, i.e. minus the "
    "trailing 7d mean funding rate per 24h",
    "T4: Ridge alpha 10, 24h horizon, expanding window and 30-day refit as PR-C, on "
    "every dataset feature (1h__ non-funding, funding, 4h__ and 1d__); mu = 0 where "
    "a symbol's packet is unavailable (for example XRP 4h/1d features after the "
    "2022 gap)",
    "T5: LightGBM 24h on the PR-C 1h__ non-funding roster with 200 trees, "
    "max_depth 3, num_leaves 8, learning_rate 0.05, min_child_samples 200, "
    "reg_lambda 1, random_state 0, n_jobs 1; if lightgbm cannot be imported the "
    "trial is recorded FAILED_TO_RUN and nothing is substituted",
    "T6: each component (Ridge24 as in PR-C, T1, T3), as a per-24h expected simple "
    "return, is divided by its causal trailing-365-day population std pooled over "
    "symbols and dates (dates up to and including the decision); the three are "
    "averaged and the average is multiplied by the causal trailing-365-day std of "
    "Ridge24 mu. T1/T3 histories start in the fit period; Ridge24 history can only "
    "start at the first vintage (2023-01-01), so mu = 0 until at least 10 daily "
    "Ridge24 dates exist. A symbol without a Ridge packet contributes 0 for that "
    "component",
    "T7: T6 mu and, in the policy, scale = clip((0.20 - DD) / 0.10, 0, 1) with "
    "DD = 1 - equity / snapshot peak_value at each scheduled decision; once DD >= "
    "20% the scale is latched at 0 until equity >= 0.95 * peak. The scale is applied "
    "as an exposure budget: if gross(target) > scale * gross_max the whole target is "
    "shrunk proportionally to scale * gross_max (hold-current-quantity flags are "
    "dropped for that decision); at scale 1 the allocator target is untouched. "
    "Consequence recognised before running: once latched the book is flat, equity "
    "cannot recover to 0.95*peak, so the latch is effectively terminal for the "
    "rest of the replay",
    "T8: T6 with 168h decisions at Monday 00:00 UTC (decision_anchor_hour_utc = 96 "
    "hours since epoch modulo 168); the allocator covariance is built from weekly "
    "marks (warmup from the same fit period) and T6 mu, defined per 24h, is "
    "multiplied by 7 so it is an expected return over the decision interval",
    "all eight trials are long/short, use the PR-C allocator parameters (gamma 5, "
    "halflife 60d, shrink 0.2, w_max 0.5, gross_max 0.95, min trade 0.5%), the PR-C "
    "execution costs, the three PR-C scenarios and the PR-C selection rule",
    "candidate pool for selection = the three PR-C allocator arms plus T1-T8 (CASH "
    "and buy-and-hold are controls); the label is DEVELOPMENT_PROMISING or "
    "NO_WINNER, never WINNER",
    "Bonferroni: confidence intervals are additionally reported at level 0.05/13 "
    "(5 PR-C arms plus 8 trials)",
    "validity for a trial = real-data future-perturbation causality probe passed "
    "(per trial) and every replay reconciled against raw fills with a flat terminal "
    "book and no termination",
    "a trial that raises is recorded with its error and counted as not eligible; "
    "nothing is substituted or retried with other parameters",
)


def registry_payload() -> dict[str, Any]:
    """Canonical JSON-able registry (without its own digest)."""

    return {
        "schema_version": REGISTRY_SCHEMA,
        "issue": 810,
        "registered": "2026-10-29",
        "registered_before": "the PR-C result and every T1-T8 result",
        "status": "registered",
        "base_protocol_digest": protocol_digest(PROTOCOL),
        "base_protocol_schema": PROTOCOL.schema_version,
        "dataset_id": PROTOCOL.dataset_id,
        "constants": json.loads(json.dumps(asdict(CONSTANTS), sort_keys=True)),
        "trials": [json.loads(json.dumps(asdict(trial))) for trial in TRIALS],
        "pr_c_arms": list(PR_C_ARMS),
        "candidate_pool": list(CANDIDATE_POOL),
        "controls": list(CONTROL_ARMS),
        "implementation_decisions": list(IMPLEMENTATION_DECISIONS),
        "result_report": RESULT_REPORT_PATH,
    }


def registry_digest() -> str:
    raw = json.dumps(
        registry_payload(), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def registry_json_bytes() -> bytes:
    payload = {**registry_payload(), "registry_digest": registry_digest()}
    return (
        json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")


__all__ = [
    "ALL_ARMS",
    "CANDIDATE_POOL",
    "CONSTANTS",
    "CONTROL_ARMS",
    "IMPLEMENTATION_DECISIONS",
    "KIND_BLEND",
    "KIND_FUNDING_CARRY",
    "KIND_LIGHTGBM",
    "KIND_RIDGE_ALL",
    "KIND_TSMOM",
    "KIND_XSMOM",
    "LABEL_FAILED_TO_RUN",
    "PR_C_ARMS",
    "PR_C_CANDIDATE_ARMS",
    "REGISTRY_REPORT_PATH",
    "REGISTRY_SCHEMA",
    "RESULT_REPORT_PATH",
    "RESULT_SCHEMA",
    "TRIALS",
    "TRIAL_ARMS",
    "WEEKLY_MONDAY_ANCHOR_HOUR",
    "Phase2Constants",
    "TrialSpec",
    "registry_digest",
    "registry_json_bytes",
    "registry_payload",
]
