"""Frozen FS1 forward-study parameters (Issue #810), recorded before the window exists.

FS1 is a result-blind forward test of three configurations that were chosen from the
public Phase 2/3 development results (selection contamination is the reason a forward
test is needed at all, see ``docs/architecture/research-assurance.md``):

* ``FS1-A``: the Phase 3 P3-6 mechanism (z-blend of T2 and T4 expected returns, kappa 3,
  12% volatility target, long/short);
* ``FS1-B``: the Phase 3 P3-3 mechanism (T2 expected returns, 12% volatility target);
* ``FS1-C``: the Phase 3 P3-7 mechanism (equal-weight buy-and-hold with a 12% volatility
  target), a passive control that is never a candidate;
* CASH: the all-zero book (no yield is credited to cash).

The window 2026-11-01..2027-05-01 does not exist yet. Nothing in this module reads a
dataset, a clock or a result. It only freezes parameters and exposes their digest;
the sealed evaluator (:mod:`~trade_rl.evaluation.forward_study.fs1`) refuses to score
unless the digest, the code identity and the dataset seal all match.

``FROZEN_PARAMS_DIGEST`` is the SHA-256 of :func:`registry_payload` at the FS1 seal
commit. A test pins it; changing any frozen value is a new study (FS2), never an edit.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Final

from trade_rl.evaluation.dev_realdata_study.phase3_registry import (
    P3_TRIALS,
    PASSIVE_COMPARATOR_ARM,
)
from trade_rl.evaluation.dev_realdata_study.protocol import (
    ARM_CASH,
    PROTOCOL,
    SCENARIO_BASE,
    SCENARIO_COST_X2,
    SCENARIO_LATENCY,
    DevStudyProtocol,
)
from trade_rl.evaluation.dev_realdata_study.trials import CONSTANTS

REGISTRY_SCHEMA: Final = "forward_study_fs1_registry_v1"
RESULT_SCHEMA: Final = "forward_study_fs1_result_v1"
MANIFEST_SCHEMA: Final = "forward_study_fs1_dataset_seal_v1"
REGISTRY_REPORT_PATH: Final = "report/issue810-fs1-registry-20261030.json"

# Placeholder written into the digested protocol payload: the real dataset id exists
# only after the authorised fetch and is bound by the dataset seal manifest instead.
DATASET_ID_PLACEHOLDER: Final = "bound-by-the-fs1-dataset-seal-manifest"

LABEL_PASS: Final = "FORWARD_PASS"
LABEL_FAIL: Final = "FORWARD_FAIL"
LABEL_INVALID: Final = "FORWARD_INVALID"
LABELS: Final = (LABEL_PASS, LABEL_FAIL, LABEL_INVALID)

PASSIVE_BETTER: Final = "BETTER_THAN_PASSIVE"
PASSIVE_NOT_BETTER: Final = "NO_BETTER_THAN_PASSIVE"

ARM_FS1_A: Final = "fs1_a_zblend_t2t4_kappa3_vt12"
ARM_FS1_B: Final = "fs1_b_t2_vt12"
ARM_FS1_C: Final = "fs1_c_ew_bh_vt12"

ROLE_PRIMARY: Final = "primary_candidate"
ROLE_SECONDARY: Final = "secondary_candidate"
ROLE_PASSIVE: Final = "passive_control"
ROLE_CASH: Final = "cash_control"

SYMBOLS: Final = ("BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT")

CLAIM_LIMITS: Final = (
    "FS1 tests three configurations that were selected from public development "
    "results; it can falsify them out of sample but cannot re-select or tune them",
    "one window of about 181 days: a single regime, about nine independent 20-day "
    "blocks; a FORWARD_FAIL from a wide interval means 'not demonstrated', not "
    "'proved zero edge', and a FORWARD_PASS is not a durable-edge or profitability claim",
    "FORWARD_PASS is only the minimum bar: max drawdown within 20% in base and cost x2 "
    "AND the 95% block-bootstrap lower bound of annualised daily excess over CASH at "
    "cost x2 is positive. An arm can be FORWARD_PASS and NO_BETTER_THAN_PASSIVE",
    "the HEADLINE flag is edge_claim_supported: true only for FORWARD_PASS together with "
    "BETTER_THAN_PASSIVE (95% lower bound of the excess over FS1-C at cost x2 above zero) "
    "and survives_bonferroni_k2 (Bonferroni k=2 lower bound over CASH at cost x2 above "
    "zero). Without it the result supports no claim that the forecast or allocation "
    "mechanism adds value",
    "CASH earns no yield; costs are the fixed assumptions of the frozen protocol "
    "(allocator c_i is a lower bound; participation impact is charged by the executor)",
    "this is not the final-evaluation authorization chain, creates no WINNER and does "
    "not establish deployment (G5) eligibility; no live or paper order is placed",
)

# Dataset construction required by FS1. Compliance of a concrete builder run with this
# policy is procedural: the seal record names the builder commit and command. The
# manifest must carry this policy's digest.
FS1_BUILD_POLICY: Final[dict[str, object]] = {
    "schema_version": "fs1_dataset_build_policy_v1",
    "builder": "trade_rl.evaluation.dev_realdata_study.dataset_build gap policy and "
    "4h/1d aggregation, run on the FS1 span",
    "network": "only the separately authorized FS1 data fetch; never while scoring",
    "market": "usds_m",
    "symbols": list(SYMBOLS),
    "base_interval": "1h",
    "feature_timeframes": ["4h", "1d"],
    "start_time": "2021-01-01T00:00:00+00:00",
    "end_time": "2027-05-01T00:00:00+00:00",
    "gap_policy": "non_tradable_rows",
    "missing_base_bar": "non-tradable row (row_present=False), forward-filled marks",
    "missing_base_bar_funding": "funding events inside a missing bar are not accrued",
    "higher_clock_after_gap": (
        "window with any missing 1h bar is a price-carried placeholder with "
        "available_at=range end; recursive/rolling features touching it are unavailable"
    ),
    "higher_clock_source": "4h and 1d bars aggregated from complete 1h windows",
    "exchange_information": "exchange-rule snapshot recorded in the seal record, "
    "applied statically",
    "economics_profile": "bot_overlay_only_assumptions_v1",
    "baked_in_fees_and_spread": "zero; costs are applied by the replay",
}

PROTOCOL_NOTES: Final = (
    "FS1 reuses the Phase 3 arm builders, runner and metrics unchanged; only the "
    "window, the arm roster, the Bonferroni count and the causality cut differ",
    "the Ridge forecast sources refit every 30 days from the scoring start, expanding "
    "from the first dataset row; every refit uses only labels matured before its fit "
    "time, including bars of the scoring window that precede that fit time",
    "bars of the warmup-only window feed features, forecasts and covariance warmup "
    "but are never scored: the replay starts at the scoring start row",
    "CASH is the all-zero book and earns no interest",
)


@dataclass(frozen=True)
class Fs1Arm:
    """One FS1 roster row and the frozen Phase 3 mechanism it reuses."""

    fs1_id: str
    role: str
    phase3_arm: str
    lineage_trial_id: str
    mechanism: str


FS1_ARMS: Final = (
    Fs1Arm(ARM_CASH, ROLE_CASH, ARM_CASH, "CASH", "all-zero target; no yield"),
    Fs1Arm(
        ARM_FS1_A,
        ROLE_PRIMARY,
        "p3_6_zblend_t2t4_kappa3_vt12",
        "P3-6",
        "z-blend(T2, T4) + kappa3 + VT12, long/short",
    ),
    Fs1Arm(
        ARM_FS1_B,
        ROLE_SECONDARY,
        "p3_3_t2_vt12",
        "P3-3",
        "T2 + VT12, long/short",
    ),
    Fs1Arm(
        ARM_FS1_C,
        ROLE_PASSIVE,
        PASSIVE_COMPARATOR_ARM,
        "P3-7",
        "equal-weight buy-and-hold + VT12 (passive vol-targeted control)",
    ),
)
CANDIDATE_ARMS: Final = (ARM_FS1_A, ARM_FS1_B)
FS1_ARM_IDS: Final = tuple(arm.fs1_id for arm in FS1_ARMS)


@dataclass(frozen=True)
class Fs1Registry:
    """Every fixed FS1 parameter. The defaults ARE the registration."""

    schema_version: str = REGISTRY_SCHEMA
    issue: int = 810
    symbols: tuple[str, ...] = SYMBOLS
    # The dataset must start no later than this row (the protocol's covariance warmup
    # start) and end exactly at ``scoring_end``.
    dataset_first_row_latest: str = "2021-01-02T00:00:00"
    development_data_end: str = "2026-08-01T00:00:00"
    warmup_only_start: str = "2026-08-01T00:00:00"
    scoring_start: str = "2026-11-01T00:00:00"
    scoring_end: str = "2027-05-01T00:00:00"
    causality_cut: str = "2026-12-01T00:00:00"
    base_protocol: DevStudyProtocol = PROTOCOL
    arms: tuple[Fs1Arm, ...] = FS1_ARMS
    candidate_arms: tuple[str, ...] = CANDIDATE_ARMS
    passive_arm: str = ARM_FS1_C
    cash_arm: str = ARM_CASH
    max_drawdown_limit_pct: float = 20.0
    drawdown_scenarios: tuple[str, ...] = (SCENARIO_BASE, SCENARIO_COST_X2)
    excess_scenario: str = SCENARIO_COST_X2
    reported_scenarios: tuple[str, ...] = (
        SCENARIO_BASE,
        SCENARIO_COST_X2,
        SCENARIO_LATENCY,
    )
    bonferroni_k: int = 2
    settlement_margin_hours: int = 24
    build_policy: dict[str, object] = field(
        default_factory=lambda: dict(FS1_BUILD_POLICY)
    )
    min_trades: int = 1
    max_fail_closed_fraction: float = 0.10
    min_tradable_fraction: float = 0.99
    labels: tuple[str, ...] = LABELS
    claim_limits: tuple[str, ...] = CLAIM_LIMITS

    def protocol(self, dataset_id: str) -> DevStudyProtocol:
        """The runner protocol of this study for a concrete dataset id."""

        return replace(
            self.base_protocol,
            dataset_id=dataset_id,
            evaluation_start=self.scoring_start,
            evaluation_end=self.scoring_end,
            causality_cut=self.causality_cut,
            arms=tuple(arm.fs1_id for arm in self.arms),
            candidate_arms=self.candidate_arms,
            bonferroni_comparisons=self.bonferroni_k,
            deviations=PROTOCOL_NOTES,
        )

    def arm(self, fs1_id: str) -> Fs1Arm:
        for arm in self.arms:
            if arm.fs1_id == fs1_id:
                return arm
        raise ValueError(f"unknown FS1 arm: {fs1_id}")

    def payload(self) -> dict[str, Any]:
        """Canonical JSON-able parameters (digested)."""

        phase3 = {spec.arm: asdict(spec) for spec in P3_TRIALS}
        arms: list[dict[str, Any]] = []
        for arm in self.arms:
            row: dict[str, Any] = asdict(arm)
            row["phase3_spec"] = phase3.get(arm.phase3_arm)
            arms.append(row)
        protocol = self.protocol(DATASET_ID_PLACEHOLDER)
        return json.loads(
            json.dumps(
                {
                    "schema_version": self.schema_version,
                    "issue": self.issue,
                    "symbols": self.symbols,
                    "dataset_first_row_latest": self.dataset_first_row_latest,
                    "development_data_end": self.development_data_end,
                    "warmup_only_window": [self.warmup_only_start, self.scoring_start],
                    "scoring_window": [self.scoring_start, self.scoring_end],
                    "causality_cut": self.causality_cut,
                    "arms": arms,
                    "candidate_arms": self.candidate_arms,
                    "passive_arm": self.passive_arm,
                    "cash_arm": self.cash_arm,
                    "falsifiers": {
                        "max_drawdown_limit_pct": self.max_drawdown_limit_pct,
                        "drawdown_scenarios": self.drawdown_scenarios,
                        "excess_vs_cash_scenario": self.excess_scenario,
                        "excess_vs_cash_ci": "95% percentile block bootstrap lower bound > 0",
                        "excess_vs_passive_ci": "95% lower bound > 0 else NO_BETTER_THAN_PASSIVE",
                        "bonferroni_k": self.bonferroni_k,
                    },
                    "reported_scenarios": self.reported_scenarios,
                    "settlement_margin_hours": self.settlement_margin_hours,
                    "build_policy": self.build_policy,
                    "validity_guards": {
                        "min_trades": self.min_trades,
                        "max_fail_closed_fraction": self.max_fail_closed_fraction,
                        "min_tradable_fraction": self.min_tradable_fraction,
                    },
                    "labels": self.labels,
                    "protocol": protocol.canonical_payload(),
                    "phase2_constants": asdict(CONSTANTS),
                    "claim_limits": self.claim_limits,
                },
                sort_keys=True,
                allow_nan=False,
            )
        )

    def digest(self) -> str:
        return payload_digest(self.payload())

    def build_policy_digest(self) -> str:
        """SHA-256 the dataset seal manifest must carry for its construction."""

        return payload_digest(dict(self.build_policy))


def payload_digest(payload: dict[str, Any]) -> str:
    raw = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


FS1_REGISTRY: Final = Fs1Registry()

# SHA-256 of ``FS1_REGISTRY.payload()`` at the FS1 seal commit (pinned by a test).
FROZEN_PARAMS_DIGEST: Final = (
    "9350734ca6ba920ebfcb47cd7b81b633df5457bfcfe7da37eba6d96fab5f1c56"
)


def registry_payload() -> dict[str, Any]:
    return FS1_REGISTRY.payload()


def params_digest() -> str:
    return FS1_REGISTRY.digest()


def registry_json_bytes() -> bytes:
    """Committed export: the frozen payload plus its digest and the seal procedure."""

    payload = {
        **registry_payload(),
        "params_digest": params_digest(),
        "seal_procedure": [
            "1 this module is merged through the normal review path; no data exists",
            "2 a fresh result-blind G0-G2 review of the design record and code is "
            "recorded on the merging change before the seal",
            "3 the seal record on Issue #810 names the sealed code commit, the "
            "params digest above and, later, the dataset manifest SHA-256",
            "4 the user separately authorises (a) fetching the post-2026-08-31 data "
            "and (b) the single scoring run, each by an Issue #810 comment",
            "5 the seal record also names the dataset builder commit and command, the "
            "exchange-rule snapshot and the UTC fetch time; the manifest carries the "
            "build policy digest and the fetch time",
            "6 the dataset manifest is recorded before scoring; scoring refuses on "
            "any mismatch, before the scoring end plus the settlement margin, without "
            "an Issue #810 authorization comment URL or when the output already exists",
        ],
    }
    return (
        json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")


__all__ = [
    "ARM_CASH",
    "ARM_FS1_A",
    "ARM_FS1_B",
    "ARM_FS1_C",
    "CANDIDATE_ARMS",
    "CLAIM_LIMITS",
    "DATASET_ID_PLACEHOLDER",
    "FROZEN_PARAMS_DIGEST",
    "FS1_ARMS",
    "FS1_BUILD_POLICY",
    "FS1_ARM_IDS",
    "FS1_REGISTRY",
    "LABELS",
    "LABEL_FAIL",
    "LABEL_INVALID",
    "LABEL_PASS",
    "MANIFEST_SCHEMA",
    "PASSIVE_BETTER",
    "PASSIVE_NOT_BETTER",
    "PROTOCOL_NOTES",
    "REGISTRY_REPORT_PATH",
    "REGISTRY_SCHEMA",
    "RESULT_SCHEMA",
    "ROLE_CASH",
    "ROLE_PASSIVE",
    "ROLE_PRIMARY",
    "ROLE_SECONDARY",
    "SYMBOLS",
    "Fs1Arm",
    "Fs1Registry",
    "params_digest",
    "payload_digest",
    "registry_json_bytes",
    "registry_payload",
]
