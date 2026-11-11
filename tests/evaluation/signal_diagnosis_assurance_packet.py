"""Author packet (``research_assurance_v1``) for the SD0 signal-diagnosis protocol.

The packet is *rendered*, not committed. ``tools.research_assurance_github`` requires
the open PR head SHA to equal the packet's ``implementation_head`` and the review's
``commit_id``; a packet stored inside the PR tree cannot contain the SHA of the commit
that stores it. So the packet is built here for the exact final head, posted as an
author-only PR comment, and any later push needs a re-rendered packet.

``build_author_packet`` is deterministic: a reviewer can rebuild the posted packet for
the head under review and compare canonical digests.
"""

from __future__ import annotations

import json

from tools.agent_repo.research_assurance import assurance_digest
from trade_rl.evaluation.signal_diagnosis.protocol import (
    DATASET_ID,
    default_signal_diagnosis_protocol,
    protocol_digest,
)

REPORT_PATH = "report/issue1146-sd0-signal-diagnosis-protocol-v3-20261110.json"
ISSUE_URL = "https://github.com/shuntatsu/trade_rl/issues/1146"
SD1_ORACLES = (
    "planted-signal recovery",
    "shuffled-future null with FDR control",
    "common-level-only market with demeaned IC near zero, plus a heterogeneous-beta "
    "control that documents what demeaning does not remove",
    "tail-only signal with rank-versus-linear IC divergence",
)
CITED_TESTS = (
    "tests/evaluation/test_signal_diagnosis_protocol.py",
    "tests/evaluation/test_signal_diagnosis_verdict.py",
    "tests/integrations/market_store/test_embargo.py",
)


def build_author_packet(head: str) -> dict[str, object]:
    """The SD0 G0-G2 author packet bound to ``head`` and the protocol digest."""

    digest = protocol_digest(default_signal_diagnosis_protocol())
    bound = f"signal_diagnosis_protocol_v3 (SHA-256 {digest}, {REPORT_PATH})"
    return {
        "schema_version": "research_assurance_v1",
        "identity": {"protocol_head": head, "implementation_head": head},
        "thesis": {
            "final_decision": (
                "Objective: determine whether the current feature families carry a "
                "predictive signal that is stable, survives removal of the same-time "
                "cross-sectional mean (not heterogeneous market beta) and exceeds "
                "round-trip cost, before any new learner is tried (Issue #1146, "
                f"protocol digest {digest}), and decide whether to implement SD1 "
                "(synthetic measurement tools and oracles) and later ask the owner "
                "for a separate G4 lease to run SD2 (the fit-free real-data "
                "diagnosis). Operational target: whether further decision-learning "
                "work on the current features (the #1025 line) is justified or new "
                "information sources are needed. This packet authorizes no "
                "real-data execution, fit, replay, FS1 scoring, unused or final "
                "data read, or live order"
            ),
            "hypothesis": (
                "Issue #1146 (parent #810, follow-up of #1025). H1: at least one "
                "preregistered feature family has out-of-sample rank-IC predictive "
                "power above the drift baseline (expanding mean, monthly update) at "
                "the 24-bar horizon that remains after removing the same-time "
                "cross-sectional mean (the common level; heterogeneous market beta is "
                "not removed), has a positive increment in each of the four full "
                "calendar years 2022-2025 (2021, whose model decisions start in "
                "April, and the truncated 2026 are partial and reported only, and a "
                "full year counts only with 12 covered months), and under a simple "
                "threshold rule earns an after-cost payoff relative to a zero-cost "
                "drift-sign policy (not a beta-adjusted alpha) whose 95% 20-day "
                "moving-block-bootstrap interval excludes zero at 1x round-trip cost "
                "and whose point estimate is positive at 2x. Frozen as "
                f"{bound}. Dataset id {DATASET_ID}. Reused-development evidence only: "
                "the dataset is consumed development data through 2026-08-01, so any "
                "later result is a known-used, post hoc exploratory screening result; "
                "no result value exists for this packet"
            ),
            "economic_mechanism": (
                "Causal story: if the current features carry usable information about relative "
                "returns of the five USDT-M perpetuals (return and oscillator "
                "momentum or reversal, trend, volatility and range state, volume "
                "and flow, funding and crowding, cross-asset relative strength), a "
                "time-ordered out-of-sample fit should rank future returns better "
                "than the drift baseline even after the same-time cross-sectional "
                "mean is removed, and a large enough forecast should pay for the round "
                "trip. The diagnosis does not assume any mechanism is true; it "
                "tests whether any information survives before a new learner is "
                "tried"
            ),
            "counter_hypothesis": (
                "Counterfactuals: the drift baseline (expanding mean, monthly update), "
                "the cross-sectionally demeaned target, the year, regime and "
                "extreme-day slices and the SD1 shuffled-future null separate a real "
                "signal from its "
                "alternatives. H0a: predictive power is indistinguishable from the drift baseline, "
                "so the #1025 outcome is explained on the data side. H0b: apparent "
                "predictive power is explained by the common cross-sectional level "
                "or market direction, a few extreme days, concentration in particular "
                "years or symbols, or feature "
                "non-stationarity such as a funding_bps drift. H0c: a statistical "
                "signal exists but does not exceed round-trip cost"
            ),
            "information_gain": (
                "One preregistered pass separates four outcomes that call for "
                "different work: INVALID (fix and discard), NO_SIGNAL (stop the "
                "#1025 learner line on current features and look for new "
                "information sources), SIGNAL_NOT_TRADABLE (work on horizon, "
                "turnover and execution cost) and SIGNAL_CANDIDATE (freeze at most "
                "three candidates for a separate prospective validation request). "
                "On consumed data it can narrow the search; it cannot show an edge"
            ),
            "cheapest_falsifier": (
                "Before any market value is read: the SD0 protocol and decision-rule "
                "tests on hand-written synthetic inputs (tests/evaluation/"
                "test_signal_diagnosis_protocol.py, tests/evaluation/"
                "test_signal_diagnosis_verdict.py), then the four SD1 synthetic "
                "oracles that must pass before SD2: "
                + "; ".join(SD1_ORACLES)
                + ". For the economic question the cheapest falsifier is the single "
                "preregistered NO_SIGNAL outcome"
            ),
            "stop_rule": (
                "Stop the current-feature decision-learning line on NO_SIGNAL. Do "
                "not add features, horizons, targets, models, penalties, k, cost "
                "multipliers, thresholds or candidates after any result is seen; "
                "any addition is a new protocol version recorded with the full "
                "exploration history. Run once and save every cell. Do not retune "
                "the #1025 learner on this data. SD2 and later real-data execution "
                "requires a separate owner G4 lease and an independent result-blind "
                "review of this exact digest and heads from a different "
                "authenticated GitHub principal; without both, stop at SD1 "
                "synthetic verification. Never touch 2026-08-01T00:00Z onward (the "
                "FS1 warm-up span and the sealed window)"
            ),
            "stage": "discovery",
            "metric_proxy_rationale": (
                "Primary metric: the out-of-sample daily rank-IC increment over the "
                "drift baseline with Benjamini-Hochberg adjustment (q = 0.10; Holm "
                "FWER also reported). The p-value behind that gate is frozen in the "
                "protocol: a one-sided test of a mean daily rank-IC increment above "
                "zero (null: mean <= 0) by moving-block bootstrap with 20-day blocks, "
                "which cover the one-day overlap of 24-bar labels, 2000 resamples and "
                "seed 0; the correction is applied separately to the raw-target and "
                "the demeaned-target p-values, each over exactly the same 16 decision "
                "cells. Guardrails: the same increment must hold on the "
                "cross-sectionally demeaned target and be positive in each of the "
                "four full calendar years 2022-2025 (each with 12 months of at least "
                "15 valid daily increments), and a simple after-cost threshold-rule "
                "payoff relative to a zero-cost drift-sign policy "
                "(fee 5 bp plus spread 2 bp per side, round trip 14 bp) must have a "
                "20-day moving-block-bootstrap interval (2000 resamples, seed 0, "
                "95%) excluding zero at cost x1 and a positive point at cost x2. "
                "These are screening proxies, not profit: the cost model has no "
                "impact, funding, risk layer or canonical ledger, and at most three "
                "frozen candidates would be checked against native replay in a "
                "later card"
            ),
            "limitations": (
                "Assumptions: the five USDT-M perpetuals and 167 maintained features "
                "are the relevant universe; hourly bars with 24-bar labels are an "
                "adequate unit; a fee-plus-spread cost is a fair first screen. "
                "Development use: screening only, never selection of a production "
                "strategy. Unused-data policy: no unused data is touched, consumed "
                "data never returns to unused, and the FS1 warm-up span and sealed "
                "window are refused. "
                "Known-used and post hoc: the development dataset was consumed by "
                "earlier studies and the plan author has seen recorded aggregate "
                "results on it (#1025 TD8, the M38 forecast diagnostic and a local "
                "breakeven report) which motivated the demeaning and tail checks, so "
                "the protocol is not blind to those aggregates and no result can be "
                "an out-of-sample or unused-data claim. SD2 and later real-data "
                "execution needs a separate owner G4 lease; SD0 implements no "
                "runner and reads no market value. No result values exist for this "
                "packet. Author interpretations fixed in the protocol and open to "
                "reviewer challenge: the verdict corners the plan leaves unassigned "
                "(a year-unstable signal is NO_SIGNAL; an interval excluding zero "
                "with a non-positive 2x point, or an interval wholly below zero, is "
                "SIGNAL_NOT_TRADABLE); decision cells are family-by-model at the "
                "24-bar horizon with the standardise-and-clip transform and k = 1; "
                "BH is applied over the 16 decision cells; the cost-adjusted sign "
                "payoff target is the net payoff of a unit signed position; the "
                "first fit is 2021-04-01 so 2021 is a partial reported-only year and "
                "all four full years 2022-2025 must be positive (four of four, "
                "stricter than the plan's four of five, fixed result-blind after the "
                "review findings; a year with fewer than 12 covered months of at "
                "least 15 valid daily increments is ineligible and fails closed to "
                "NO_SIGNAL); the demeaned target removes only the common level and "
                "not heterogeneous market beta, and the beta-regression baseline is "
                "reported context that never decides; the drift-sign comparator is "
                "charged no cost, a deliberately favourable choice, and the D4 "
                "payoff relative to it is not a beta-adjusted alpha; 30 days stands "
                "for monthly; the inner CV has 5 forward-chaining folds purged by "
                "the label horizon; daily IC is pooled over the day's rows; "
                "families are defined by indicator kind and are arbitrary; five "
                "symbols and overlapping 24-hour labels give far fewer independent "
                "observations than rows; bars stamped 2026-08-01T00:00Z close the "
                "last development period and are inside the window. Prior exposure, "
                "cited from Issue #1146 section 1 and not recomputed here: a local "
                "breakeven report outside the repository (24-hour Ridge, alpha 10, "
                "2020-01 to 2026-08) recorded a pooled IC of 0.0675, a rank IC of "
                "0.0164 and a daily t of 2.31, and the M38 forecast diagnostic "
                "(Ridge, 2024-03) recorded mse_skill -0.017588, so the design is "
                "not blind to the gap between pooled and rank IC that motivated the "
                "tail check. Power trade-off, deliberate: the persistence rule is "
                "4 of 4 full years, stricter than the plan's 4 of 5; for a true "
                "signal that is positive in each year with probability 0.8, 4 of 4 "
                "passes with probability 0.41 while at least 4 of 5 would pass with "
                "0.74, so NO_SIGNAL is the likely label for a modest real effect and "
                "the rule favours few false positives on known-used data. This packet "
                "is UNREVIEWED"
            ),
        },
        "mechanism": {
            "chain": [
                "source",
                "availability",
                "feature_state",
                "model_strategy",
                "intent",
                "order",
                "fill",
                "accounting",
                "evidence",
                "decision",
            ],
            "authorities": {
                "unit": (
                    "forward log return over h hourly bars; costs in basis points "
                    "of notional; exposure is one signed unit"
                ),
                "time": (
                    "dataset available_at not later than the decision timestamp; a "
                    "label is usable only after t+h matures inside the window on the "
                    "bar-close-time basis; the window ends where the FS1 warm-up span "
                    "begins, derived from the FS1 registry; a decisive stability year "
                    "must start at or after the first usable fit and end inside the "
                    "window"
                ),
                "state": (
                    "no account or book state; the diagnosis is at signal level "
                    "and holds no position state"
                ),
                "sign": (
                    "positive rank-IC increment over the drift baseline on the raw "
                    "and the cross-sectionally demeaned target; payoff is signed "
                    "position times return minus absolute position times round-trip "
                    "cost, and D4 compares it with a zero-cost drift-sign policy"
                ),
                "risk": (
                    "no risk layer or sizing; unit exposure only; hard risk is "
                    "not exercised and not claimed"
                ),
                "execution": (
                    "simple fee plus spread on both sides, no impact, latency, "
                    "funding or capacity; not the native executor"
                ),
                "accounting": (
                    "none in SD0; D4 uses a simple payoff and is not the canonical "
                    "ledger; native replay is limited to at most three frozen "
                    "candidates in a later card"
                ),
                "evidence": (
                    "content-addressed protocol digest; single run with every cell "
                    "saved; verdict is a pure function of saved cell evidence under "
                    "frozen thresholds"
                ),
            },
            "changed_boundaries": ["time", "sign", "execution", "evidence"],
            "independent_oracles": {
                "time": {
                    "statement": (
                        "No evaluation window, label maturity or run request may "
                        "reach the FS1 warm-up span [2026-08-01T00:00Z, "
                        "2026-11-01T00:00Z) or the sealed window from "
                        "2026-11-01T00:00Z, labels must mature inside the window, and "
                        "a full stability year must lie wholly inside the out-of-sample "
                        "span that starts at the first usable fit"
                    ),
                    "counterexample": (
                        "A window ending one second after 2026-08-01T00:00Z, a "
                        "window covering the whole warm-up span, a window starting "
                        "at 2026-11-01, a naive datetime, a real_development "
                        "request without an authorization digest, a protocol that "
                        "lists 2021 (first fit in April) as a full year, a decisive "
                        "year with 11 covered months, a partial year supplied as "
                        "decisive evidence"
                    ),
                    "oracle": (
                        "tests/evaluation/test_signal_diagnosis_protocol.py "
                        "parametrizes literal-date windows, asserts every touching "
                        "window is refused and windows ending at 2026-08-01T00:00Z "
                        "are admitted, and asserts the protocol instants equal the "
                        "market-data store embargo constants that "
                        "tests/integrations/market_store/test_embargo.py pins to "
                        "the FS1 registry and the SQL triggers, so the checker does "
                        "not copy the dates. SD1 must add a leakage-injection "
                        "counterexample (a feature copying the matured label must "
                        "be flagged) before SD2"
                    ),
                    "known_limitations": (
                        "Checks interval arithmetic and the declared window only; "
                        "it does not prove historical point-in-time availability, "
                        "the dataset available_at contract (SD2 D0) or that the "
                        "SD1 label-maturity code is correct"
                    ),
                },
                "sign": {
                    "statement": (
                        "A signal needs a positive rank-IC increment on both the "
                        "raw and the cross-sectionally demeaned target; the "
                        "demeaned target subtracts the same-time cross-sectional "
                        "mean and does not remove heterogeneous market beta; rank IC "
                        "and linear IC are separate statistics"
                    ),
                    "counterexample": (
                        "A synthetic market where every symbol follows one common "
                        "factor with equal loading and no idiosyncratic signal; the "
                        "same market with unequal loadings (the demeaned target keeps "
                        "a residual, which the control documents); a planted "
                        "idiosyncratic signal; a signal present only in the "
                        "largest 1% of moves; labels shuffled in time"
                    ),
                    "oracle": (
                        "SD1 must show on seeded synthetic data with known truth: "
                        + "; ".join(SD1_ORACLES)
                        + ". SD0 checks only, on hand-written numbers in "
                        "tests/evaluation/test_signal_diagnosis_verdict.py, that "
                        "the rule rejects a cell whose demeaned IC is zero, "
                        "negative or not BH-significant"
                    ),
                    "known_limitations": (
                        "The four SD1 oracles are specified here and not "
                        "implemented in SD0; synthetic markets cannot show that "
                        "real beta, tails or non-stationarity behave like the "
                        "synthetic ones; a heterogeneous-beta residual in the "
                        "decisive target is a known, documented limit of v2"
                    ),
                },
                "execution": {
                    "statement": (
                        "The after-cost screen charges fee 5 bp plus spread 2 bp "
                        "per side (round trip 14 bp) at cost multipliers 1 and 2, "
                        "only when the forecast exceeds k times the round-trip "
                        "cost, and charges nothing while flat; the comparator "
                        "drift-sign policy is charged nothing"
                    ),
                    "counterexample": (
                        "A zero-cost or one-sided-cost rule, a rule that trades "
                        "below the threshold, a payoff that omits the round-trip "
                        "charge"
                    ),
                    "oracle": (
                        "tests/evaluation/test_signal_diagnosis_protocol.py checks "
                        "the cost fields against the development protocol fee and "
                        "spread and the 2*(fee+spread) identity; SD1 must supply a "
                        "slow independent reference for signed position times "
                        "return minus absolute position times round-trip cost on "
                        "hand-checkable rows"
                    ),
                    "known_limitations": (
                        "A simple cost model without impact, funding, borrow, "
                        "latency, capacity or a risk layer, and not the canonical "
                        "ledger; an after-cost screening payoff is not a P&L and a "
                        "candidate would still need native replay and a "
                        "prospective test"
                    ),
                },
                "evidence": {
                    "statement": (
                        "The verdict is a pure function of saved cell evidence "
                        "under thresholds frozen with a digest; changing a grid, "
                        "threshold or window, loading a protocol with a claim "
                        "field, or classifying incomplete evidence fails closed. "
                        "The Benjamini-Hochberg adjustment is computed inside the "
                        "verdict, separately for the raw and the demeaned target, "
                        "over exactly the 16 frozen decision cells from raw "
                        "one-sided p-values that carry the frozen procedure's "
                        "provenance; an already adjusted p-value cannot be supplied"
                    ),
                    "counterexample": (
                        "A protocol with a shrunken penalty grid and a matching "
                        "digest; a payload carrying an edge, G4 or live field; a "
                        "signal cell without D4 evidence; one tampered byte; a "
                        "duplicate JSON key; p-values with foreign or missing "
                        "provenance (another test, block length, resample count or "
                        "seed); a protocol whose bootstrap block is shorter than the "
                        "label overlap or whose BH family is not the decision cells"
                    ),
                    "oracle": (
                        "tests/evaluation/test_signal_diagnosis_protocol.py "
                        "(strict load, missing and extra field at every level, "
                        "claim-field refusal, frozen-reference rejection, empty and "
                        "altered grids) and tests/evaluation/"
                        "test_signal_diagnosis_verdict.py (every verdict boundary "
                        "with hand-written numbers: BH q equality, four of four "
                        "eligible full years, the 12-month coverage floor, refusal "
                        "of a partial start-up year, interval lower bound exactly "
                        "zero, strictly positive 2x point, candidate cap and "
                        "ranking)"
                    ),
                    "known_limitations": (
                        "Software-level on synthetic inputs; a frozen digest proves "
                        "the declaration cannot change silently, not that the "
                        "declared design is sound, and this packet is UNREVIEWED"
                    ),
                },
            },
        },
        "evidence": {
            "point_in_time": False,
            "common_accounting": False,
            "realistic_costs": False,
            "hard_risk": False,
            "terminal_state": False,
            "fit_development_unused_separated": False,
            "multi_symbol": False,
            "multi_period": False,
            "controls": [],
            "robustness": [],
            "independent_reconstruction": (
                "Software-level only. The verdict tests restate every boundary as "
                "literal numbers and the window tests as literal dates; the protocol "
                "constants are cross-checked against the existing development-study "
                "identity, the FS1 registry, the market-data store embargo, the "
                "maintained feature-definition function (167 names), the LightGBM "
                "constructor call and the PrequentialSchedule and "
                "moving_block_mean_test signatures. No market value has been read or "
                "computed and no D0-D5 numerics are implemented, so every "
                "evidence boolean here is false except no_development_rescue, which "
                "the protocol budget enforces"
            ),
            "no_development_rescue": True,
            "evidence_level": "software_validity",
        },
        "claims": {
            "claim_level": "software_validity",
            "permitted": [
                f"The signal-diagnosis protocol {bound}, its four-label decision "
                "rule and its run-admission contract are specified, frozen and "
                "software-verified on hand-written synthetic inputs only",
                "The protocol records the dataset as known-used and post hoc and "
                "refuses any window that touches the FS1 warm-up span or sealed "
                "window",
            ],
            "forbidden": [
                "any edge, profitability, G4, G5, unused-data, final-evaluation, "
                "live-trading or production claim",
                "running SD2 or any later real-data step, reading or computing any "
                "market value, fitting or replaying before a separate owner G4 "
                "lease and an independent review of this exact digest and heads",
                "touching 2026-08-01T00:00Z onward (the FS1 warm-up span or the "
                "sealed window)",
                "adding or changing a feature family, horizon, target, model, "
                "penalty, k, cost multiplier, threshold, block length or candidate "
                "cap after any result is seen without a new protocol version and a "
                "recorded exploration history",
                "reading a SIGNAL_CANDIDATE label as an edge",
                "calling any result beta-robust or beta-adjusted: the decisive target "
                "removes only the same-time cross-sectional mean, and the D4 payoff is "
                "relative to a zero-cost drift-sign policy, not a beta-adjusted alpha",
            ],
            "next_authorized_action": (
                "SD1 synthetic measurement tools and oracles (synthetic "
                "verification only) after an independent result-blind review of this "
                "exact packet digest and heads. Real-data SD2 and later execution "
                "requires a separate owner G4 lease; this packet authorizes none of "
                "it"
            ),
            "production_eligible": False,
            "live_trading_authorized": False,
            "decision_rule": (
                "After SD1 oracles hold and a G4 lease exists, one run is "
                "classified by the frozen table: INVALID if leakage, a dataset or "
                "protocol digest mismatch, an available_at violation or contact "
                "with the FS1 warm-up span or sealed window occurs; NO_SIGNAL if no "
                "family-by-model cell is positive and BH-significant (q = 0.10, "
                "frozen one-sided moving-block-bootstrap p-values) on "
                "both the raw and the cross-sectionally demeaned target with a "
                "positive demeaned increment in all four eligible full years "
                "2022-2025; SIGNAL_NOT_TRADABLE if signal cells "
                "exist but none has a cost-x1 interval of the payoff relative to the "
                "zero-cost drift-sign policy strictly above zero "
                "with a positive cost-x2 point; SIGNAL_CANDIDATE otherwise, keeping "
                "at most three cells ranked by interval lower bound. Every label "
                "stays known-used exploratory screening and is never an edge"
            ),
            "next_action_class": "synthetic_verification_only",
        },
    }


def canonical_packet_json(head: str) -> str:
    """The exact canonical JSON text to post as the author-only packet comment."""

    return json.dumps(
        build_author_packet(head),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def packet_digest(head: str) -> str:
    return assurance_digest(build_author_packet(head))
