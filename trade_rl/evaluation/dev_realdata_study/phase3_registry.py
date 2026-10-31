"""Frozen Phase 3 trial registry (turnover aversion and volatility targeting), Issue #810.

Status: POST_HOC_EXPLORATORY. The eight configurations were chosen by the planner
AFTER the Phase 2 results on the same development data were known (T4 had a large
gross edge eaten by costs and a 43% drawdown; T2 was cost-robust with a drawdown near
21%). They are therefore NOT preregistered with respect to the development data, and
the word "preregistered" must not be used for them. The only public pre-run record is
the Issue #810 comment reproduced verbatim in :data:`PUBLIC_RECORD_LINES`
(https://github.com/shuntatsu/trade_rl/issues/810#issuecomment-6493851921, created
2026-10-29T10:10:07Z, edited 2026-10-29T10:10:19Z only to correct the Bonferroni arm
count to 21). It was posted before any Phase 3 code existed. Its timestamps are
GitHub's; nothing in git alone proves them. The registry JSON
(``report/issue810-phase3-trial-registry-20261029.json``) is committed before the first
real-data run commit.

Same data, period, capital, costs, stresses, metrics and selection rule as PR-C and
Phase 2 (:data:`~trade_rl.evaluation.dev_realdata_study.protocol.PROTOCOL`), with the
PR-C probe rule (the arm must also trade before the probe cut) as the primary validity
rule and no amendments. Two mechanisms are added inside the cost-aware allocator:

* ``kappa``: the optimizer's L1 cost penalty is ``kappa * c_i`` (kappa = 3); execution
  costs are unchanged.
* ``VT12``: after the solve, if the ex-ante ``sqrt(w' Sigma w * periods_per_year)``
  exceeds 12%, the solved target is scaled by ``0.12 / sigma_p`` and only then does the
  existing min-trade rule run.

Everything the public record does not state is an implementation decision taken before
any real-data run and listed in :data:`IMPLEMENTATION_NOTES`; the notes are outside the
digest of the frozen roster (:func:`roster_digest`) but inside :func:`registry_digest`.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Final

from trade_rl.evaluation.dev_realdata_study.protocol import (
    ARM_BUY_AND_HOLD,
    ARM_CASH,
    PROTOCOL,
    protocol_digest,
)
from trade_rl.evaluation.dev_realdata_study.trials import WEEKLY_MONDAY_ANCHOR_HOUR

REGISTRY_SCHEMA: Final = "dev_realdata_phase3_trial_registry_v1"
RESULT_SCHEMA: Final = "dev_realdata_phase3_trial_results_v1"
REGISTRY_REPORT_PATH: Final = "report/issue810-phase3-trial-registry-20261029.json"
RESULT_REPORT_PATH: Final = "report/issue810-phase3-trial-results-20261029.json"
PROVENANCE_REPORT_PATH: Final = "report/issue810-phase3-trial-provenance-20261029.json"

PUBLIC_RECORD_URL: Final = (
    "https://github.com/shuntatsu/trade_rl/issues/810#issuecomment-6493851921"
)
PUBLIC_RECORD_CREATED_UTC: Final = "2026-10-29T10:10:07Z"
PUBLIC_RECORD_EDITED_UTC: Final = "2026-10-29T10:10:19Z"
PUBLIC_RECORD_EDIT_NOTE: Final = "edited only to correct the Bonferroni arm count to 21"

# The comment body, verbatim, one tuple element per line (joined with a newline).
PUBLIC_RECORD_LINES: Final = (
    "<!-- issue810-phase3-post-hoc-registry-20261029 -->",
    "## #810 Phase 3 trial registry — recorded BEFORE any Phase 3 code or run (post-hoc relative to Phase 2)",
    "",
    "This comment is the public timestamped record of the Phase 3 roster, written by the planner after the Phase 2 (PR #1042) results were known. It is therefore **POST_HOC_EXPLORATORY** with respect to the development data, not preregistered. No Phase 3 code exists yet.",
    "",
    "**Motivation (from Phase 2 numbers):** T4 (Ridge with all features) made +41.7k USDT gross but paid 16.5k in costs and drew down 43%. T2 (cross-sectional momentum) held up under cost stress with a drawdown of about 21%. T1 traded too little to matter. Hypothesis: turnover aversion plus ex-ante portfolio volatility targeting can turn gross edge into net profit that passes the drawdown gate.",
    "",
    "**Fixed protocol:** same data, period (2023-01-01..2026-08-01), capital, costs, stresses (cost x2, +1 bar latency), metrics and selection rule as Phase 2, using the PR-C probe rule with no amendments.",
    "",
    "**Two new causal, in-policy mechanisms:**",
    "- `kappa`: the optimizer's cost penalty is `kappa * c_i`, with kappa = 3. Execution costs are unchanged.",
    "- `VT12`: after the solve, if the ex-ante `sqrt(w' Sigma w * 365)` exceeds 12% annualized, scale the target `w` by `0.12 / sigma_p`, then apply the existing min-trade rule. `Sigma` is the allocator's own EWMA/shrunk covariance.",
    "",
    "**Roster (8 configs, each run once, all reported):**",
    "",
    "| ID | Config |",
    "|---|---|",
    "| P3-1 | T4 + kappa3 |",
    "| P3-2 | T4 + kappa3 + VT12 |",
    "| P3-3 | T2 + VT12 |",
    "| P3-4 | T6 + kappa3 + VT12 |",
    "| P3-5 | T4 weekly (168h, mu x7) + VT12 |",
    "| P3-6 | z-blend(T2, T4) + kappa3 + VT12 |",
    "| P3-7 | EW buy-and-hold + VT12 (vol-scaled passive comparator) |",
    "| P3-8 | z-blend(T2, T4) long-only + kappa3 + VT12 |",
    "",
    "**Multiple testing:** Bonferroni intervals reported over all 21 development arms (PR-C 5, including CASH and EW B&H, + T1–T8 + P3-1..P3-8), and per family.",
    "",
    "**Label vocabulary:** POST_HOC_PROMISING or NO_WINNER, never WINNER. A promising config would be frozen (exact code SHA plus params digest) as the candidate for one sealed forward evaluation on the unused window. That needs separate user authorization and is not run in this phase. No live or paper orders.",
    "",
)
PUBLIC_RECORD_BODY: Final = "\n".join(PUBLIC_RECORD_LINES)
PUBLIC_RECORD_SHA256: Final = (
    "347704933f29310d5c88bc55f3a95b6965271d4c7825b2c285acb9ef8222e334"
)

KAPPA: Final = 3.0
VT_TARGET_ANNUAL: Final = 0.12
BONFERRONI_ARMS: Final = 21
FAMILY_ARMS: Final = 8

KIND_ALLOCATOR: Final = "allocator"
KIND_PASSIVE_VT: Final = "passive_vol_target"

# Expected-return source keys (see ``phase3_sources``).
SOURCE_T4: Final = "t4_ridge24_all_features"
SOURCE_T4_WEEKLY: Final = "t4_ridge24_all_features_x7"
SOURCE_T2: Final = "t2_xsmom"
SOURCE_T6: Final = "t6_blend"
SOURCE_ZBLEND_T2_T4: Final = "zblend_t2_t4"


@dataclass(frozen=True)
class Phase3Spec:
    """One roster row. ``roster_text`` is the config text of the public record."""

    trial_id: str
    arm: str
    roster_text: str
    kind: str
    source: str | None
    kappa: float
    vol_target: float | None
    decision_interval_hours: int = 24
    decision_anchor_hour_utc: int = 0
    long_only: bool = False


P3_TRIALS: Final = (
    Phase3Spec(
        "P3-1", "p3_1_t4_kappa3", "T4 + kappa3", KIND_ALLOCATOR, SOURCE_T4, KAPPA, None
    ),
    Phase3Spec(
        "P3-2",
        "p3_2_t4_kappa3_vt12",
        "T4 + kappa3 + VT12",
        KIND_ALLOCATOR,
        SOURCE_T4,
        KAPPA,
        VT_TARGET_ANNUAL,
    ),
    Phase3Spec(
        "P3-3",
        "p3_3_t2_vt12",
        "T2 + VT12",
        KIND_ALLOCATOR,
        SOURCE_T2,
        1.0,
        VT_TARGET_ANNUAL,
    ),
    Phase3Spec(
        "P3-4",
        "p3_4_t6_kappa3_vt12",
        "T6 + kappa3 + VT12",
        KIND_ALLOCATOR,
        SOURCE_T6,
        KAPPA,
        VT_TARGET_ANNUAL,
    ),
    Phase3Spec(
        "P3-5",
        "p3_5_t4_weekly_vt12",
        "T4 weekly (168h, mu x7) + VT12",
        KIND_ALLOCATOR,
        SOURCE_T4_WEEKLY,
        1.0,
        VT_TARGET_ANNUAL,
        decision_interval_hours=168,
        decision_anchor_hour_utc=WEEKLY_MONDAY_ANCHOR_HOUR,
    ),
    Phase3Spec(
        "P3-6",
        "p3_6_zblend_t2t4_kappa3_vt12",
        "z-blend(T2, T4) + kappa3 + VT12",
        KIND_ALLOCATOR,
        SOURCE_ZBLEND_T2_T4,
        KAPPA,
        VT_TARGET_ANNUAL,
    ),
    Phase3Spec(
        "P3-7",
        "p3_7_ew_bh_vt12",
        "EW buy-and-hold + VT12 (vol-scaled passive comparator)",
        KIND_PASSIVE_VT,
        None,
        1.0,
        VT_TARGET_ANNUAL,
    ),
    Phase3Spec(
        "P3-8",
        "p3_8_zblend_t2t4_longonly_kappa3_vt12",
        "z-blend(T2, T4) long-only + kappa3 + VT12",
        KIND_ALLOCATOR,
        SOURCE_ZBLEND_T2_T4,
        KAPPA,
        VT_TARGET_ANNUAL,
        long_only=True,
    ),
)

P3_ARMS: Final = tuple(spec.arm for spec in P3_TRIALS)
if len(P3_TRIALS) != 8 or len(set(P3_ARMS)) != 8:  # pragma: no cover - import guard
    raise RuntimeError("the Phase 3 registry must hold exactly eight unique trials")

# CASH and unscaled equal-weight buy-and-hold are re-run in the same code path.
BASELINE_ARMS: Final = (ARM_CASH, ARM_BUY_AND_HOLD)
ALL_ARMS: Final = (*BASELINE_ARMS, *P3_ARMS)
# P3-7 is the vol-scaled passive comparator named in the public record, not a
# strategy candidate: it is reported and probed but cannot be the selected arm.
PASSIVE_COMPARATOR_ARM: Final = "p3_7_ew_bh_vt12"
CONTROL_ARMS: Final = (*BASELINE_ARMS, PASSIVE_COMPARATOR_ARM)
CANDIDATE_POOL: Final = tuple(arm for arm in P3_ARMS if arm != PASSIVE_COMPARATOR_ARM)

LABEL_POST_HOC_PROMISING: Final = "POST_HOC_PROMISING"
LABEL_NO_WINNER: Final = "NO_WINNER"
LABEL_FAILED_TO_RUN: Final = "FAILED_TO_RUN"

# Decided before any real-data run (not stated in the public record).
IMPLEMENTATION_NOTES: Final = (
    "implementation notes, fixed before any Phase 3 real-data execution:",
    "N1 kappa: the optimizer receives kappa * c_i with c_i = fee + spread = 0.0007 "
    "(the base-scenario one-way cost); the replay charges the unchanged scenario "
    "execution cost; the stress scenarios keep the base kappa * c_i like Phase 2",
    "N2 VT12: sigma_p = sqrt(w' Sigma w * periods_per_year) on the solved target "
    "before the min-trade rule, with Sigma the allocator's own EWMA (halflife 60d, "
    "shrink 0.2, floor 1e-8) covariance of decision-time marks including the current "
    "decision mark. periods_per_year = 8760 / decision_interval_hours: 365 for daily "
    "arms and 52.14 for the weekly P3-5 whose Sigma is a weekly-return covariance "
    "(the public record writes 365 for a daily Sigma; applying 365 to a weekly "
    "covariance would overstate the variance sevenfold, so this is a clarification "
    "of the same intent). If sigma_p <= 0.12 the target is unchanged. The min-trade "
    "rule can hold positions at exact quantity after the scaling, so the final book "
    "may sit slightly above the cap (final_vol is recorded per decision). A "
    "non-finite sigma_p fails closed (HOLD, reason invalid_volatility)",
    "N3 P3-5: the T4 source (Ridge alpha 10, 24h horizon, all 1h/4h/1d features, as "
    "T4) with every mu multiplied by 7; decisions every 168h at Monday 00:00 UTC "
    "(decision_anchor_hour_utc = 96, as T8); covariance from weekly marks with the "
    "same fit-period warmup rule as T8",
    "N4 z-blend(T2, T4): components are T2 (the registered 90d skip-1d demeaned score "
    "scaled to mean |mu| = 2 * one-way cost, daily) and T4 (Ridge24 all features, "
    "per-24h expected simple return); each component is divided by its causal "
    "trailing 365-day population std pooled over symbols and dates (dates up to and "
    "including the decision), the two are averaged (divided by 2) and multiplied by "
    "the same trailing std of T4; at least 10 T4 dates and, for T2, at least "
    "BLEND_MIN_POOLED_OBSERVATIONS = 30 finite pooled observations are required, "
    "otherwise mu = 0; this is the T6 construction with the components replaced",
    "N5 P3-7: a long-only equal-weight book (gross budget 0.99, as the buy-and-hold "
    "control) that keeps its relative composition (the weights drift with prices; "
    "there is no calendar rebalance back to equal weights) and re-sets its gross at "
    "every daily 00:00 UTC decision to 0.99 * min(1, 0.12 / sigma_p), where sigma_p "
    "is the ex-ante volatility of the current composition at gross 0.99 (same Sigma "
    "as N2); the first decision enters at equal weights; the existing min-trade rule "
    "(0.5% of equity per symbol) leaves smaller adjustments untouched; it is a "
    "control (vol-scaled passive comparator) and not a selection candidate",
    "N6 selection: candidates are the seven allocator arms P3-1..P3-6 and P3-8; CASH, "
    "unscaled equal-weight buy-and-hold and P3-7 are controls. The selection rule is "
    "the PR-C one (validity, max drawdown <= 20% in all three scenarios, net profit "
    "> 0 at base and cost x2, highest base net profit); the label names are "
    "POST_HOC_PROMISING / NO_WINNER",
    "N7 validity: the replay ran in all scenarios, the ledger reconciled to raw fills "
    "with a flat terminal book, no termination, the real-data future-perturbation "
    "probe at the PR-C cut (2023-07-01, base) identical, and the per-arm, "
    "per-scenario causality matrix at 2023-03-01 identical. Primary (PR-C probe "
    "rule): the arm must also have traded in the first half of the probe's trace "
    "rows (the PR-C 'pre_cut_trace_has_trades' definition, unchanged). Secondary, "
    "reported for continuity only: the same without the trade requirement. "
    "Including the causality matrix in both validity rules is stricter than "
    "Phase 2's primary rule and can only invalidate an arm, never promote one",
    "N8 Bonferroni: 95% percentile block-bootstrap intervals (20-day blocks, 2000 "
    "resamples, seed 0) of annualized daily excess versus CASH, versus unscaled "
    "equal-weight buy-and-hold and (for the candidates) versus P3-7 are reported "
    "unadjusted, at level 0.05/8 (the Phase 3 family) and at 0.05/21 (all "
    "development arms). The Phase 2 arms are not re-run here, so only the Phase 3 "
    "arms and the two baselines have intervals",
    "N9 baselines: CASH and unscaled equal-weight buy-and-hold are re-run in the "
    "same code and compared with the committed Phase 2 numbers (reproduction check)",
    "N10 the probe also records an explicit-split indicator "
    "(pre_cut_interval_rows_have_trades: any non-zero quantity in any pre-cut "
    "interval row) as a diagnostic only; it does not enter validity",
)


def roster_payload() -> dict[str, Any]:
    """The frozen roster only (no implementation notes)."""

    return {
        "schema_version": REGISTRY_SCHEMA,
        "issue": 810,
        "status": "post_hoc_exploratory",
        "base_protocol_digest": protocol_digest(PROTOCOL),
        "dataset_id": PROTOCOL.dataset_id,
        "kappa": KAPPA,
        "vol_target_annual": VT_TARGET_ANNUAL,
        "bonferroni_arms": BONFERRONI_ARMS,
        "family_arms": FAMILY_ARMS,
        "trials": [asdict(spec) for spec in P3_TRIALS],
        "baseline_arms": list(BASELINE_ARMS),
        "candidate_pool": list(CANDIDATE_POOL),
        "controls": list(CONTROL_ARMS),
    }


def _digest(payload: dict[str, Any]) -> str:
    raw = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def roster_digest() -> str:
    return _digest(roster_payload())


def registry_payload() -> dict[str, Any]:
    """Canonical JSON-able registry without its own digest."""

    return {
        **roster_payload(),
        "roster_digest": roster_digest(),
        "classification": (
            "POST_HOC_EXPLORATORY: chosen after the Phase 2 results on the same "
            "development data; not preregistered"
        ),
        "public_pre_run_record": {
            "url": PUBLIC_RECORD_URL,
            "created_utc": PUBLIC_RECORD_CREATED_UTC,
            "edited_utc": PUBLIC_RECORD_EDITED_UTC,
            "edit_note": PUBLIC_RECORD_EDIT_NOTE,
            "body_sha256": PUBLIC_RECORD_SHA256,
            "body_verbatim": PUBLIC_RECORD_BODY,
        },
        "implementation_notes": list(IMPLEMENTATION_NOTES),
        "result_report": RESULT_REPORT_PATH,
        "provenance_report": PROVENANCE_REPORT_PATH,
    }


def registry_digest() -> str:
    return _digest(registry_payload())


def registry_json_bytes() -> bytes:
    payload = {**registry_payload(), "registry_digest": registry_digest()}
    return (
        json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")


__all__ = [
    "ALL_ARMS",
    "BASELINE_ARMS",
    "BONFERRONI_ARMS",
    "CANDIDATE_POOL",
    "CONTROL_ARMS",
    "FAMILY_ARMS",
    "IMPLEMENTATION_NOTES",
    "KAPPA",
    "KIND_ALLOCATOR",
    "KIND_PASSIVE_VT",
    "LABEL_FAILED_TO_RUN",
    "LABEL_NO_WINNER",
    "LABEL_POST_HOC_PROMISING",
    "P3_ARMS",
    "P3_TRIALS",
    "PASSIVE_COMPARATOR_ARM",
    "PROVENANCE_REPORT_PATH",
    "PUBLIC_RECORD_BODY",
    "PUBLIC_RECORD_CREATED_UTC",
    "PUBLIC_RECORD_EDITED_UTC",
    "PUBLIC_RECORD_LINES",
    "PUBLIC_RECORD_SHA256",
    "PUBLIC_RECORD_URL",
    "REGISTRY_REPORT_PATH",
    "REGISTRY_SCHEMA",
    "RESULT_REPORT_PATH",
    "RESULT_SCHEMA",
    "SOURCE_T2",
    "SOURCE_T4",
    "SOURCE_T4_WEEKLY",
    "SOURCE_T6",
    "SOURCE_ZBLEND_T2_T4",
    "VT_TARGET_ANNUAL",
    "Phase3Spec",
    "registry_digest",
    "registry_json_bytes",
    "registry_payload",
    "roster_digest",
    "roster_payload",
]
