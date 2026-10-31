"""Frozen preregistration for the Issue #810 P2 development real-data study.

This module is committed BEFORE the study is run on real data. Every parameter
below was fixed on 2026-10-29 before any allocator result existed; the runner
reads them from :data:`PROTOCOL` and never from the command line, and the result
summary records :func:`protocol_digest` so a result can be matched to this file.

Study (development evidence only, never a ``WINNER``):

* Data: the local development Dataset artifact (5 USDT-perpetual symbols, 1h
  bars 2021-01-01..2026-08-01, funding present, zero fees/spread baked in).
  Costs are applied by the replay's ``ExecutionCostConfig``.
* Forecast: pooled Ridge (alpha=10) on the ``1h__`` non-funding features,
  expanding window, 30-day refit from 2023-01-01, horizons 24h and 72h.
* Allocation: cost-aware multi-symbol allocator (``strategies.allocation``)
  with daily 00:00 UTC decisions inside ONE continuous shared-account replay
  per arm and scenario, evaluation 2023-01-01..2026-08-01, terminal settle flat.
* Arms (fixed roster of five): CASH; equal-weight buy-and-hold; allocator-24h
  long/short; allocator-72h long/short; allocator-24h long-only.
* Scenarios (all arms): base; cost x2 (fee and spread doubled); +1 bar order
  latency.
* Selection rule (result-blind): validity -> risk gate (max drawdown <= 20% in
  all three scenarios) -> highest base terminal net profit among the allocator
  arms that also beat CASH (terminal net profit > 0) under base and cost x2 ->
  ``DEVELOPMENT_PROMISING``; otherwise ``NO_WINNER``.

Recorded interpretation choices and deviations from the plan text (all decided
before any real-data result; see ``DEVIATIONS``):

1. No "canonical 12-feature roster" exists in the repository, so the plan's
   fallback applies: every ``1h__`` feature whose name does not contain
   ``funding`` (56 features), identical for all symbols.
2. ``activation_delay_hours`` is 0: this dataset's ``available_at`` equals the
   bar timestamp, so the plan's "1 if available_at lags timestamp" condition is
   not met and the first vintage (cutoff 2023-01-01T00:00) activates at once.
3. No ``MarketOrderProfile`` is passed: the Dataset already carries per-symbol
   lot size, tick size and minimum notional from the exchange-rule snapshot,
   and the profile would only repeat the same current-snapshot assumption.
4. The explicit ``ExecutionCostConfig`` is fee 0.05%/side, spread 0.02%,
   impact_rate 0.01% (repo default), participation cap 5% (repo default). The
   allocator's per-symbol cost ``c_i`` is the deterministic one-way
   ``fee + spread`` (0.07%); participation impact is not modelled by it.
5. Stresses change the market only: cost x2 doubles ``fee_rate`` and
   ``spread_rate`` (impact unchanged); latency sets ``order_latency_bars=1``.
   The allocator keeps its base ``c_i`` and decision times in every scenario
   (execution turns out worse than the policy assumed).
6. 72h forecasts: ``expm1(mu_log + 0.5 * s2)`` is taken at the 72h horizon and
   the resulting simple return is divided by 3 for the 24h decision interval.
7. Risk layer: the repo default ``PreTradeRisk.default_for_execution`` (max
   gross 1.0, no drawdown scaling, no turnover cap). The 20% drawdown limit is a
   selection gate only, not a trading rule.
8. Metrics: max drawdown on hourly mark-to-market equity; Sharpe, bootstrap and
   per-year figures on 00:00 UTC daily equity (Sharpe annualised by sqrt(365));
   CAGR over (end - start) / 365.25 days; turnover is the replay ledger's
   cumulative sum of absolute weight changes.
9. XRP lacks 120 bars (2022-02/04): the dataset marks them non-tradable and
   forward-fills marks, so the allocator holds XRP there; the covariance warmup
   uses those forward-filled daily marks.
10. Candidate pool for selection is the three allocator arms. CASH and
    buy-and-hold are controls. The rule as written does NOT require beating
    buy-and-hold; excess versus buy-and-hold is reported separately.
11. Bonferroni note: 7 arm-vs-benchmark comparisons per scenario (4 versus CASH,
    3 allocators versus buy-and-hold); adjusted intervals use level 0.05 / 7.
    Phase 2 later deflates by the total number of configurations it has run.
12. Validity includes a real-data causality probe (future prices/features after
    2023-07-01 perturbed; allocator-24h decisions and fills before that instant
    must be identical) plus ledger reconciliation of every replay. The embedded
    probe covers allocator-24h in the base scenario over 2023-01..2023-07 only and
    is applied as the global validity gate; ``runner.causality_probe_matrix``
    (added after the run) extends the same test to every arm and scenario.
13. The terminal-flat requirement is enforced by ``runner.reconcile_ledger``
    (part of the replay ``validity`` gate), not by a field of this frozen
    protocol; it is a runner validity rule, not a preregistered parameter.

Dataset build policy (recorded AFTER the first run, not part of the protocol
digest): the study Dataset is reproduced offline by
``dev_realdata_study.dataset_build`` under ``DATASET_BUILD_POLICY`` (missing 1h
bars are non-tradable rows with forward-filled marks and no funding accrual;
4h/1d bars are aggregated from complete 1h windows and a window touching a gap
is an ``available_at = range end`` placeholder, so recursive higher-clock
features stay unavailable afterwards). ``DevStudyProtocol`` and therefore
``protocol_digest`` are deliberately unchanged by that addition so the committed
result still matches this file.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Final

PROTOCOL_SCHEMA: Final = "dev_realdata_study_protocol_v1"

ARM_CASH: Final = "cash"
ARM_BUY_AND_HOLD: Final = "equal_weight_buy_and_hold"
ARM_ALLOCATOR_24H: Final = "allocator_24h_long_short"
ARM_ALLOCATOR_72H: Final = "allocator_72h_long_short"
ARM_ALLOCATOR_24H_LONG_ONLY: Final = "allocator_24h_long_only"

ARMS: Final = (
    ARM_CASH,
    ARM_BUY_AND_HOLD,
    ARM_ALLOCATOR_24H,
    ARM_ALLOCATOR_72H,
    ARM_ALLOCATOR_24H_LONG_ONLY,
)
CANDIDATE_ARMS: Final = (
    ARM_ALLOCATOR_24H,
    ARM_ALLOCATOR_72H,
    ARM_ALLOCATOR_24H_LONG_ONLY,
)

SCENARIO_BASE: Final = "base"
SCENARIO_COST_X2: Final = "cost_x2"
SCENARIO_LATENCY: Final = "latency_1bar"
SCENARIOS: Final = (SCENARIO_BASE, SCENARIO_COST_X2, SCENARIO_LATENCY)

LABEL_PROMISING: Final = "DEVELOPMENT_PROMISING"
LABEL_NO_WINNER: Final = "NO_WINNER"

DEVIATIONS: Final = (
    "features: no canonical 12-feature roster exists; all 1h__ non-funding features (56)",
    "activation_delay_hours=0: available_at equals the bar timestamp in this dataset",
    "no MarketOrderProfile: the dataset already carries lot/tick/min-notional rules",
    "allocator c_i = one-way fee+spread (0.07%); participation impact not modelled by it",
    "stresses alter the market only; the allocator keeps base c_i and schedule",
    "72h forecast: expm1(mu_log + s2/2) at 72h, then divided by 3",
    "risk layer is the repo default; the 20% drawdown limit is a selection gate only",
    "metrics: hourly equity for max drawdown, daily 00:00 UTC equity for the rest",
    "XRP gap bars are non-tradable with forward-filled marks (allocator holds)",
    "candidate pool is the three allocator arms; beating buy-and-hold is not required",
    "Bonferroni: 7 comparisons per scenario, adjusted level 0.05/7",
    "validity includes a real-data causality probe and per-replay ledger reconciliation",
)


@dataclass(frozen=True)
class DevStudyProtocol:
    """Every fixed parameter of the study; the defaults ARE the preregistration."""

    schema_version: str = PROTOCOL_SCHEMA
    issue: int = 810
    dataset_id: str = "0b95547145a8f35d4deaaea99479ccf93244080bff52ef969203f4fa721de174"
    # Evaluation window and account.
    evaluation_start: str = "2023-01-01T00:00:00"
    evaluation_end: str = "2026-08-01T00:00:00"
    covariance_warmup_start: str = "2021-01-02T00:00:00"
    initial_capital: float = 10_000.0
    # Forecasts.
    feature_prefix: str = "1h__"
    feature_exclude_substring: str = "funding"
    ridge_alpha: float = 10.0
    refit_interval_days: int = 30
    activation_delay_hours: int = 0
    horizons_hours: tuple[int, ...] = (24, 72)
    decision_interval_hours: int = 24
    decision_anchor_hour_utc: int = 0
    # Allocator.
    gamma: float = 5.0
    ewma_halflife_days: float = 60.0
    shrinkage: float = 0.2
    w_max: float = 0.5
    gross_max: float = 0.95
    min_trade_fraction: float = 0.005
    min_history_periods: int = 30
    buy_and_hold_gross_budget: float = 0.99
    # Execution (base) and stresses.
    fee_rate: float = 0.0005
    spread_rate: float = 0.0002
    impact_rate: float = 0.0001
    max_participation_rate: float = 0.05
    cost_stress_multiplier: float = 2.0
    latency_stress_bars: int = 1
    # Metrics.
    trading_days_per_year: int = 365
    cagr_days_per_year: float = 365.25
    bootstrap_block_days: int = 20
    bootstrap_resamples: int = 2000
    bootstrap_seed: int = 0
    bootstrap_alpha: float = 0.05
    bonferroni_comparisons: int = 7
    # Selection.
    max_drawdown_limit: float = 0.20
    reconciliation_tolerance: float = 1e-4
    # Causality probe.
    causality_cut: str = "2023-07-01T00:00:00"
    # Rosters.
    arms: tuple[str, ...] = ARMS
    candidate_arms: tuple[str, ...] = CANDIDATE_ARMS
    scenarios: tuple[str, ...] = SCENARIOS
    deviations: tuple[str, ...] = DEVIATIONS

    def canonical_payload(self) -> dict[str, object]:
        return json.loads(json.dumps(asdict(self), sort_keys=True))


PROTOCOL: Final = DevStudyProtocol()


def protocol_digest(protocol: DevStudyProtocol = PROTOCOL) -> str:
    """SHA-256 of the canonical JSON of every protocol field."""

    raw = json.dumps(
        protocol.canonical_payload(),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


__all__ = [
    "ARMS",
    "ARM_ALLOCATOR_24H",
    "ARM_ALLOCATOR_24H_LONG_ONLY",
    "ARM_ALLOCATOR_72H",
    "ARM_BUY_AND_HOLD",
    "ARM_CASH",
    "CANDIDATE_ARMS",
    "DEVIATIONS",
    "LABEL_NO_WINNER",
    "LABEL_PROMISING",
    "PROTOCOL",
    "PROTOCOL_SCHEMA",
    "SCENARIOS",
    "SCENARIO_BASE",
    "SCENARIO_COST_X2",
    "SCENARIO_LATENCY",
    "DevStudyProtocol",
    "protocol_digest",
]
