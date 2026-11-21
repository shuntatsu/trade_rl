"""Frozen constants and digest of the out-of-sample test ``agy_anchor_realistic_v1``.

The normative text is ``docs/specs/agy-anchor-realistic.md`` (including Amendment 1). The
signal under test is the port in :mod:`trade_rl.strategies.agy_anchor`; its constants are
*imported* from it, never re-typed. Two bindings make a change to the signal visible: the
SHA-256 of the parity fixture (``tests/fixtures/agy_anchor``) and a digest of the port's
source files are pinned here and enter :func:`protocol_digest`. Costs, risk and the
scenarios are the Core7 MTF study's, and the control and the low-turnover protocol it
rides on are the ones ``trend_long_spot_oos_v1`` reuses. Nothing here reads market data or
computes a return.
"""

from __future__ import annotations

import hashlib
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

from trade_rl.artifacts import canonical_json_bytes
from trade_rl.evaluation.carry_multisymbol.protocol import SPOT_HALTS
from trade_rl.evaluation.core7_mtf_study import protocol as core7
from trade_rl.evaluation.lowturn_screen import protocol as lowturn
from trade_rl.evaluation.trend_oos import protocol as trend
from trade_rl.risk.pretrade import PreTradeRiskConfig
from trade_rl.simulation.execution import ExecutionCostConfig
from trade_rl.strategies.agy_anchor import signals as anchor_signals
from trade_rl.strategies.agy_anchor import targets as anchor_targets

PROTOCOL_SCHEMA: Final = "agy_anchor_realistic_protocol_v1"
PROTOCOL_ID: Final = "agy_anchor_realistic_v1"

# --- the signal under test: bindings -------------------------------------------------
#: SHA-256 of ``tests/fixtures/agy_anchor/agy_anchor_klines_funding_2021.npz``.
FIXTURE_SHA256: Final = (
    "def5b1211e9de93f904943ad559979c7a08e4d70b0038cd7019f772fb9a96eae"
)
FIXTURE_RELATIVE_PATH: Final = (
    "tests/fixtures/agy_anchor/agy_anchor_klines_funding_2021.npz"
)
#: Digest of ``trade_rl/strategies/agy_anchor/*.py`` (see :func:`port_source_digest`).
PORT_SOURCE_DIGEST: Final = (
    "a89b8eb766d9d1cc3e01c23e1e600ac45c8d52aaecc4c7a1757cd34aa5de298c"
)

#: The reused lowturn/trend bindings (control arm and cash-account tooling).
LOWTURN_PROTOCOL_ID: Final = trend.LOWTURN_PROTOCOL_ID
LOWTURN_PROTOCOL_DIGEST: Final = trend.LOWTURN_PROTOCOL_DIGEST

# --- universe, clock, capital (imported) ---------------------------------------------
ROSTER: Final[tuple[str, ...]] = core7.ROSTER
LEG_ROSTER: Final[tuple[str, ...]] = core7.LEG_ROSTER
N_PAIRS: Final = len(ROSTER)
PERIODS_PER_YEAR: Final = lowturn.PERIODS_PER_YEAR
INITIAL_CAPITAL: Final = lowturn.INITIAL_CAPITAL
BAR_HOURS: Final = 4
SCENARIOS: Final[tuple[str, ...]] = lowturn.SCENARIOS
BASE_SCENARIO: Final = "BASE"
HIGH_COST_SCENARIO: Final = "HIGH_COST"
DELAY_SCENARIO: Final = "DELAY_1BAR"

# --- the anchor rule (imported from the port) and the frozen stride ------------------
REBALANCE_EVERY_BARS: Final = 18  # 3 days; chosen from known-used diagnostics
#: The port's own lab-4H "L1" timing: a decision at row ``t`` reads signal row ``t - 1``.
#: ``DELAY_1BAR`` lags the signal by one further bar (2). The controls use lowturn's
#: convention (0, and 1 for ``DELAY_1BAR``); they are constant rules.
ANCHOR_BASE_SIGNAL_LAG_BARS: Final = 1
DEADBAND: Final = anchor_targets.DEADBAND  # lab total-turnover deadband
ANCHOR_GROSS_CAP: Final = anchor_targets.GROSS_CAP
ANCHOR_W_MAX_LEG: Final = anchor_targets.W_MAX_LEG

CANDIDATE: Final = "AGY_ANCHOR"
CONTROL_CASH: Final = "CASH"
CONTROL_EW_SPOT: Final = "EW_HOLD_SPOT"
CONTROLS: Final[tuple[str, ...]] = (CONTROL_CASH, CONTROL_EW_SPOT)
ARMS: Final[tuple[str, ...]] = (CANDIDATE, *CONTROLS)

# --- execution -----------------------------------------------------------------------
MAX_LEVERAGE: Final = 2.5
PARTICIPATION_CAP: Final = 0.01
#: Amendment 1: no explicit perp initial-margin field exists; the engine's own
#: ``margin_used = gross_notional / max_leverage`` (``simulation/execution.py:756``) gives
#: ``1 / max_leverage``. A different rate is a new protocol.
INITIAL_MARGIN_RATE: Final = 0.4
CASH_LIMIT: Final = 1.0
#: Amendment 3: the engine books perp notional through cash like spot notional
#: (``simulation/accounting.py:820``), so the net signed exposure is capped as well.
#: Amendment 4: 3% below 1.0, because orders are sized at the decision close and filled at
#: the next bar, so a book at net 1.0 overshoots cash by the price gap plus costs.
NET_EXPOSURE_LIMIT: Final = 0.97
#: A cell whose cash net of cumulative execution costs is below minus this fraction of the
#: initial capital at any interval is invalid.
CASH_TOLERANCE_FRACTION: Final = 1e-6
REASON_NEGATIVE_CASH: Final = "negative_cash_borrowing"

# --- warm-up (Amendment 1) -----------------------------------------------------------
#: Bars a pair needs after both its spot and its perp are listed.
PAIR_WARMUP_BARS: Final = 180
#: Rows every stage must hold before its first decision.
STAGE_WARMUP_BARS: Final = 540

STAGE_DEV: Final = "DEV"
STAGE_BACKCAST: Final = "BACKCAST_2020"
STAGE_FORWARD: Final = "FORWARD"
STAGES: Final[tuple[str, ...]] = (STAGE_DEV, STAGE_BACKCAST, STAGE_FORWARD)
#: ``(load start, first decision, last decision bar, settles)`` in UTC; all are bar-close
#: timestamps of dataset rows. The last dataset row is two bars after the last decision
#: bar (the engine decides through ``stop - 2``, flattens on ``stop - 1``, marks on
#: ``stop``). DEV's last dataset row is the Tier D last row 2026-07-31T20:00Z, so no hour
#: at or after the 2026-08-01 embargo is read (Amendment 1).
STAGE_WINDOWS: Final[dict[str, dict[str, str]]] = {
    STAGE_DEV: {
        "load_start": "2021-10-01T00:00:00Z",
        "first_decision": "2021-12-30T04:00:00Z",
        "last_decision_bar": "2026-07-31T12:00:00Z",
        "settles": "2026-07-31T16:00:00Z",
    },
    STAGE_BACKCAST: {
        "load_start": "2020-01-01T00:00:00Z",
        "first_decision": "2020-05-01T00:00:00Z",
        "last_decision_bar": "2020-12-31T20:00:00Z",
        "settles": "2021-01-01T00:00:00Z",
    },
    STAGE_FORWARD: {
        "load_start": "2026-02-01T00:00:00Z",
        "first_decision": "2026-08-01T00:00:00Z",
        "last_decision_bar": "2027-04-30T20:00:00Z",
        "settles": "2027-05-01T00:00:00Z",
    },
}
FINAL_NOT_BEFORE: Final = "2027-05-02T00:00:00Z"
#: Owner scope relay (Issue #810) covering fetching and reading data for any pre-registered
#: out-of-sample / forward protocol. Bound for BACKCAST_2020 and FORWARD reads (owner-authored,
#: unedited between fetches); DEV reads known-used data and needs no relay of its own.
DATA_AUTHORIZATION_URL: Final = (
    "https://github.com/shuntatsu/trade_rl/issues/810#issuecomment-7040247247"
)
AUTHORIZED_STAGES: Final[tuple[str, ...]] = (STAGE_BACKCAST, STAGE_FORWARD)

# --- decision rule -------------------------------------------------------------------
#: Amendment 2: the owner's 25% research drawdown budget (quote in the spec). The ramp is
#: proportional to the earlier 0.10 / 0.20; trend_long_spot_oos_v1 stays at 0.20.
RISK_DRAWDOWN_START: Final = 0.125
RISK_DRAWDOWN_STOP: Final = 0.25
MAX_DRAWDOWN_LIMIT: Final = RISK_DRAWDOWN_STOP
MIN_BASE_SHARPE: Final = 0.50
LABEL_SUPPORTED: Final = "ANCHOR_SUPPORTED"
LABEL_FORWARD_ONLY: Final = "ANCHOR_FORWARD_ONLY"
LABEL_NOT_SUPPORTED: Final = "ANCHOR_NOT_SUPPORTED"
LABEL_INVALID: Final = "INVALID"
LABELS: Final[tuple[str, ...]] = (
    LABEL_SUPPORTED,
    LABEL_FORWARD_ONLY,
    LABEL_NOT_SUPPORTED,
    LABEL_INVALID,
)
LABEL_INTERIM: Final = "INTERIM_NOT_A_DECISION"
#: Stage-level labels of a single stage run (the combined label needs both stages).
STAGE_SUPPORTED: Final = "STAGE_SUPPORTED"
STAGE_NOT_SUPPORTED: Final = "STAGE_NOT_SUPPORTED"

BOOTSTRAP_KIND: Final = trend.BOOTSTRAP_KIND
BOOTSTRAP_BLOCK_BARS: Final = 42
BOOTSTRAP_RESAMPLES: Final = 10_000
BOOTSTRAP_SEED: Final = 20261121
BOOTSTRAP_CONFIDENCE: Final = 0.95
BOOTSTRAP_CHUNK: Final = 250

_SCENARIO_DELAY_BARS: Final[dict[str, int]] = {
    BASE_SCENARIO: 0,
    HIGH_COST_SCENARIO: 0,
    DELAY_SCENARIO: 1,
}

#: Exchange-wide spot outages declared result-blind from the BACKCAST_2020 coverage
#: report (spec Amendment 6). ``halts-from-coverage`` derived them from the report whose
#: SHA-256 is frozen below; coverage ran at code ``bed9c8de`` under digest ``ce469144``.
DECLARED_HALTS: Final[tuple[tuple[str, str], ...]] = (
    ("2020-02-09T02:00:00Z", "2020-02-09T03:00:00Z"),
    ("2020-02-19T12:00:00Z", "2020-02-19T17:00:00Z"),
    ("2020-03-04T10:00:00Z", "2020-03-04T11:00:00Z"),
    ("2020-04-25T02:00:00Z", "2020-04-25T04:00:00Z"),
    ("2020-06-28T02:00:00Z", "2020-06-28T05:00:00Z"),
    ("2020-11-30T06:00:00Z", "2020-11-30T07:00:00Z"),
    ("2020-12-21T14:00:00Z", "2020-12-21T18:00:00Z"),
    ("2020-12-25T02:00:00Z", "2020-12-25T03:00:00Z"),
)
COVERAGE_EVIDENCE_SHA256: Final[str | None] = (
    "fb8104aea3987be4ab5a4caa975b9f05f6135cb992955aa69073c3f6c29966eb"
)


#: The harness files whose bytes enter the digest (a change to any of them is a new digest).
HARNESS_PACKAGE_FILES: Final[tuple[str, ...]] = (
    "protocol.py",
    "cash.py",
    "decision.py",
    "data.py",
    "runner.py",
)
HARNESS_TOOL_RELATIVE_PATH: Final = "tools/agy_anchor_oos.py"


class ProtocolBindingError(RuntimeError):
    """A pinned binding (port, fixture, reused protocol, margin rate) no longer holds."""


def _normalised(data: bytes) -> bytes:
    return data.replace(b"\r\n", b"\n")


def port_source_digest() -> str:
    """SHA-256 over the port's ``*.py`` files (name and LF-normalised bytes), sorted."""
    root = Path(anchor_targets.__file__).resolve().parent
    digest = hashlib.sha256()
    for path in sorted(root.glob("*.py")):
        digest.update(path.name.encode("utf-8") + b"\0")
        digest.update(_normalised(path.read_bytes()))
        digest.update(b"\0")
    return digest.hexdigest()


def harness_source_digest(
    *, package_dir: Path | None = None, tool_path: Path | None = None
) -> str:
    """SHA-256 over the harness source (name and LF-normalised bytes), like the port digest.

    It covers ``protocol, cash, decision, data, runner`` of this package and
    ``tools/agy_anchor_oos.py``. It is *computed*, not pinned: protocol.py is one of the
    hashed files, so a pinned value would be self-referential. Any edit to the harness
    therefore changes the protocol digest. A missing file fails closed.
    """
    root = Path(__file__).resolve().parent if package_dir is None else package_dir
    tool = (
        Path(__file__).resolve().parents[3] / HARNESS_TOOL_RELATIVE_PATH
        if tool_path is None
        else tool_path
    )
    digest = hashlib.sha256()
    entries = [(name, root / name) for name in HARNESS_PACKAGE_FILES]
    entries.append((HARNESS_TOOL_RELATIVE_PATH, tool))
    for name, path in entries:
        if not path.is_file():
            raise ProtocolBindingError(f"harness source file {path} is missing")
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(_normalised(path.read_bytes()))
        digest.update(b"\0")
    return digest.hexdigest()


def fixture_path() -> Path:
    return Path(__file__).resolve().parents[3] / FIXTURE_RELATIVE_PATH


def verify_port_binding() -> None:
    current = port_source_digest()
    if current != PORT_SOURCE_DIGEST:
        raise ProtocolBindingError(
            f"agy_anchor port source digest {current} differs from the pinned "
            f"{PORT_SOURCE_DIGEST}: the signal changed"
        )


def verify_fixture_binding(path: Path | None = None) -> None:
    """Fail closed unless the parity fixture exists with the pinned SHA-256."""
    target = fixture_path() if path is None else path
    if not target.is_file():
        raise ProtocolBindingError(f"parity fixture {target} is missing")
    actual = hashlib.sha256(target.read_bytes()).hexdigest()
    if actual != FIXTURE_SHA256:
        raise ProtocolBindingError(
            f"parity fixture sha256 {actual} differs from the pinned {FIXTURE_SHA256}"
        )


def verify_lowturn_binding() -> None:
    trend.verify_lowturn_binding()


def verify_margin_binding() -> None:
    """The initial-margin rate must be the engine's ``1 / max_leverage`` in every scenario."""
    for name in SCENARIOS:
        cost = core7.scenario_cost(name)
        if cost.max_leverage != MAX_LEVERAGE:
            raise ProtocolBindingError("scenario max_leverage is not the frozen 2.5")
        if cost.max_participation_rate != PARTICIPATION_CAP:
            raise ProtocolBindingError("scenario participation cap is not 0.01")
        if abs(1.0 / cost.max_leverage - INITIAL_MARGIN_RATE) > 1e-15:
            raise ProtocolBindingError("initial margin rate is not 1 / max_leverage")
    core = core7.risk_config()
    rc = risk_config()
    if (core.max_gross, core.max_abs_weight, core.max_turnover) != (
        rc.max_gross,
        rc.max_abs_weight,
        rc.max_turnover,
    ):
        raise ProtocolBindingError("risk config differs from the frozen core7 caps")
    if (rc.drawdown_start, rc.drawdown_stop) != (0.125, 0.25):
        raise ProtocolBindingError(
            "risk drawdown ramp differs from the frozen 0.125 / 0.25"
        )
    if rc.drawdown_start * 2 != rc.drawdown_stop:
        raise ProtocolBindingError("the drawdown ramp must start at half of the stop")


def scenario_cost(scenario: str) -> ExecutionCostConfig:
    """The core7 ``ExecutionCostConfig`` of the scenario, unchanged."""
    return core7.scenario_cost(scenario)


def scenario_signal_lag_bars(scenario: str) -> int:
    """Extra signal lag in bars of a scenario (0, and 1 for ``DELAY_1BAR``)."""
    if scenario not in _SCENARIO_DELAY_BARS:
        raise ValueError(f"unknown scenario: {scenario!r}")
    return _SCENARIO_DELAY_BARS[scenario]


def arm_signal_lag_bars(arm: str, scenario: str) -> int:
    """Signal lag of ``arm`` in ``scenario``: the anchor adds its base L1 lag."""
    if arm not in ARMS:
        raise ValueError(f"unknown arm: {arm!r}")
    extra = scenario_signal_lag_bars(scenario)
    return extra + (ANCHOR_BASE_SIGNAL_LAG_BARS if arm == CANDIDATE else 0)


def risk_config() -> PreTradeRiskConfig:
    """The frozen risk config (Amendment 2): core7's gross/weight/turnover, 0.125 / 0.25."""
    return PreTradeRiskConfig(
        max_gross=2.5,
        max_abs_weight=0.55,
        max_turnover=2.5,
        drawdown_start=RISK_DRAWDOWN_START,
        drawdown_stop=RISK_DRAWDOWN_STOP,
        emergency_turnover_override=True,
    )


def declared_halt_intervals() -> tuple[tuple[datetime, datetime], ...]:
    out: list[tuple[datetime, datetime]] = []
    for lo_text, hi_text in DECLARED_HALTS:
        lo = datetime.fromisoformat(lo_text.replace("Z", "+00:00")).astimezone(UTC)
        hi = datetime.fromisoformat(hi_text.replace("Z", "+00:00")).astimezone(UTC)
        if lo >= hi or lo.minute or lo.second or hi.minute or hi.second:
            raise ProtocolBindingError(f"declared halt {lo_text}..{hi_text} is invalid")
        if out and lo < out[-1][1]:
            raise ProtocolBindingError("declared halts must be ordered and disjoint")
        out.append((lo, hi))
    evidence = COVERAGE_EVIDENCE_SHA256
    if out and (evidence is None or re.fullmatch(r"[0-9a-f]{64}", evidence) is None):
        raise ProtocolBindingError(
            "declared halts need the coverage.json SHA-256 they were derived from"
        )
    return tuple(out)


def stage_halt_intervals() -> tuple[tuple[datetime, datetime], ...]:
    merged = sorted([*SPOT_HALTS, *declared_halt_intervals()])
    for (_, a_hi), (b_lo, _) in zip(merged, merged[1:], strict=False):
        if b_lo < a_hi:
            raise ProtocolBindingError("registered and declared halts overlap")
    return tuple(merged)


def protocol_payload() -> dict[str, object]:
    """Canonical frozen protocol payload for digesting."""
    rc = risk_config()
    return {
        "schema_version": PROTOCOL_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "spec": "docs/specs/agy-anchor-realistic.md with Amendment 1",
        "signal": {
            "port": "trade_rl/strategies/agy_anchor",
            "port_source_digest": PORT_SOURCE_DIGEST,
            "parity_fixture_sha256": FIXTURE_SHA256,
            "parity_fixture_arrays": "fix_* (lab with the per-symbol N4 defect fixed)",
            "variant": "klines_funding_taker_oi_hyperliquid_jump15m_neutral",
            "carry_scale": anchor_targets.CARRY_SCALE,
            "alpha_scale": anchor_targets.ALPHA_SCALE,
            "w_max_leg": ANCHOR_W_MAX_LEG,
            "gross_cap": ANCHOR_GROSS_CAP,
            "blend_regime": anchor_signals.BLEND_REGIME,
            "blend_bidir": anchor_signals.BLEND_BIDIR,
            "min_valid": anchor_signals.MIN_VALID,
            "rebalance_every_bars": REBALANCE_EVERY_BARS,
            "deadband_total_turnover": DEADBAND,
            "stride_selection_disclosed": "3-day stride chosen from known-used agy "
            "diagnostics (+246% at 12bp against +220% daily)",
        },
        "reuses": {
            "lowturn_protocol_id": LOWTURN_PROTOCOL_ID,
            "lowturn_protocol_digest": LOWTURN_PROTOCOL_DIGEST,
            "control": "EW_HOLD_SPOT and CASH are lowturn LowturnPolicy families, "
            "wrapped in the same AnchorCashAccountPolicy as the anchor",
        },
        "data_authorization_url": DATA_AUTHORIZATION_URL,
        "data_authorization": {
            "url": DATA_AUTHORIZATION_URL,
            "stages": list(AUTHORIZED_STAGES),
            "scope": "owner scope relay: fetching and reading data for any pre-registered "
            "out-of-sample / forward protocol",
            "binding": "owner-authored and unedited between two fetches, for every "
            "BACKCAST_2020 and FORWARD read; recorded in the receipt",
            "dev": "known-used data; no relay needed",
        },
        "harness": {
            "files": [*HARNESS_PACKAGE_FILES, HARNESS_TOOL_RELATIVE_PATH],
            "harness_source_digest": harness_source_digest(),
            "binding": "computed, not pinned (protocol.py is hashed)",
        },
        "roster": list(ROSTER),
        "leg_layout": "alternating_spot_perp_even_index_spot_odd_index_perp",
        "arms": {"candidate": CANDIDATE, "controls": list(CONTROLS)},
        "scenarios": {
            name: {
                "cost": core7.scenario_cost(name).execution_policy_payload(),
                "extra_signal_lag_bars": scenario_signal_lag_bars(name),
                "anchor_signal_lag_bars": arm_signal_lag_bars(CANDIDATE, name),
                "control_signal_lag_bars": arm_signal_lag_bars(CONTROL_CASH, name),
            }
            for name in SCENARIOS
        },
        "stages": {name: dict(window) for name, window in STAGE_WINDOWS.items()},
        "warmup": {
            "pair_bars_after_both_markets_listed": PAIR_WARMUP_BARS,
            "stage_bars_before_first_decision": STAGE_WARMUP_BARS,
            "pair_listed_from_first_row": "available from the first row",
            "unavailable_pair_inputs": "zero volume, valid False, constant price; "
            "prices divided by one common factor (the spot close of the activation "
            "row) from activation on, only for pairs listed after the first row",
            "lab_demean_note": "lab all_ones cross-sectional demeans include "
            "unavailable pairs as zero columns",
        },
        "stage_row_convention": {
            "timestamps": "bar_close_of_the_4h_row",
            "last_dataset_row": "last_decision_bar + 2 bars",
            "rebalance_phase": f"(index - first_decision_index) % {REBALANCE_EVERY_BARS}",
        },
        "final_scoring_not_before": FINAL_NOT_BEFORE,
        "data": {
            "source": "binance_vision_spot_and_usdm_1h_klines_and_funding_"
            "checksum_verified",
            "aggregation": "hourly_to_4h_open_first_close_last_high_max_low_min_"
            "volume_sum_funding_sum_tradable_only_if_all_hours_tradable",
            "taker_buy": "sum of the four hourly taker_buy_quote_volume values of each "
            "leg, zero where the leg is not tradable; carried as a (T, legs) array "
            "beside the dataset, never inside it",
            "funding_events": "stamped at the close of the 4h bar holding them; "
            "count-based means start cold at the first row of the perp",
            "daily_fallback": "checksum_verified_daily_archives_only_provenance_recorded",
            "forward_fill": False,
            "spot_halts": [[lo.isoformat(), hi.isoformat()] for lo, hi in SPOT_HALTS],
            "declared_halts": {
                "intervals": [list(pair) for pair in DECLARED_HALTS],
                "coverage_json_sha256": COVERAGE_EVIDENCE_SHA256,
            },
            "perp_series": "strict_no_halt_calendar_no_gap_filled",
            "unlisted_rows": "asset_inactive_not_tradable_information_unavailable_"
            "constant_sentinel_price_no_invented_price",
            "partial_first_bar": "the 4h bar holding the first listed hour is unavailable",
            "dev_last_row": "2026-07-31T20:00:00Z (Tier D last row; embargo untouched)",
            "exchange_rules": "static_snapshot_not_point_in_time",
        },
        "risk": {
            "max_gross": rc.max_gross,
            "max_abs_weight": rc.max_abs_weight,
            "max_turnover": rc.max_turnover,
            "drawdown_start": rc.drawdown_start,
            "drawdown_stop": rc.drawdown_stop,
            "emergency_turnover_override": rc.emergency_turnover_override,
            "amendment": "2: owner 25% research drawdown budget; earlier protocols "
            "stay at 0.10 / 0.20",
        },
        "replay": {
            "path": "run_shared_portfolio_replay",
            "settle_terminal_position": True,
            "capture_ledger_evidence": True,
            "initial_capital": INITIAL_CAPITAL,
            "max_leverage": MAX_LEVERAGE,
            "participation_cap": PARTICIPATION_CAP,
            "summary": "core7 _summarize_replay_cell with execution_cost",
        },
        "cash_account": {
            "rule": "spot long gross + initial_margin_rate * perp gross <= cash_limit",
            "initial_margin_rate": INITIAL_MARGIN_RATE,
            "initial_margin_source": "engine margin_used = gross_notional / "
            "max_leverage (simulation/execution.py:756); no explicit perp field",
            "conservative": True,
            "different_rate": "a new protocol",
            "cash_limit": CASH_LIMIT,
            "net_exposure_limit": NET_EXPOSURE_LIMIT,
            "amendment_3": "the scale enforces both (a) spot long gross + "
            "initial_margin_rate * perp gross <= cash_limit and (b) the sum of signed "
            "weights <= net_exposure_limit, because the engine books every leg, perps "
            "included, through cash (accounting.py:820 cash_after = cash - (signed_"
            "notional + cost); accounting.py:387 portfolio_value = cash + position "
            "values), so cash is about capital * (1 - net exposure) minus costs",
            "amendment_4": "the net exposure limit is 0.97, a 3% buffer for the price "
            "gap between the decision close and the next-bar fill; the margin bound "
            "(a) stays at cash_limit 1.0 and the 1e-6 negative-cash tolerance is "
            "unchanged",
            "scale_factor": "the whole target is scaled proportionally by the tighter "
            "of the two factors; the negative-cash check is unchanged",
            "application": "all three arms (the controls are spot-only, so both bounds "
            "reduce to their spot gross), after the signal's own caps, on the effective "
            "weights (held legs at their current weight, others times risk_scale); "
            "proportional scaling; when binding all holds are dropped",
            "invalid_on_borrowing": {
                "reason": REASON_NEGATIVE_CASH,
                "rule": "max over ledger intervals of (-cash_after - total_cost_after) "
                "greater than tolerance_fraction * initial_capital",
                "tolerance_fraction": CASH_TOLERANCE_FRACTION,
            },
            "reported_not_gating": [
                "cap_bound_fraction_of_rebalance_decisions",
                "bound_by_margin_net_exposure_or_both_counts",
                "mean_pre_cap_measure",
                "mean_pre_cap_net",
                "min_cash",
                "max_borrowing",
            ],
        },
        "cell_validity": {
            "causality_probe": "per (arm, signal lag), over market rows and taker-buy "
            "volume",
            "ledger_reconciled": True,
            "causality_probe_passed": True,
            "not_insolvent": True,
            "terminal_book": "flat or only lean-core terminal dust",
        },
        "decision_rule": {
            "supported_stage": {
                "all_nine_cells_valid": True,
                "candidate_max_drawdown_below": MAX_DRAWDOWN_LIMIT,
                "candidate_base_return_above": 0.0,
                "candidate_base_sharpe_at_least": MIN_BASE_SHARPE,
                "candidate_high_cost_and_delay_returns_above": 0.0,
            },
            "labels": {
                LABEL_SUPPORTED: "SUPPORTED(FORWARD) and SUPPORTED(BACKCAST_2020)",
                LABEL_FORWARD_ONLY: "SUPPORTED(FORWARD) and not SUPPORTED(BACKCAST_2020)",
                LABEL_NOT_SUPPORTED: "otherwise, all cells valid",
                LABEL_INVALID: "any cell invalid",
            },
            "dev_stage": "diagnostic only; never gates, never tunes",
        },
        "reported_not_gating": {
            "per_calendar_year_returns": True,
            "turnover_cost_funding_max_drawdown": True,
            "cap_bound_fraction": True,
            "bootstrap": {
                "kind": BOOTSTRAP_KIND,
                "statistic": "mean per-bar BASE excess return over CASH",
                "block_mean_bars": BOOTSTRAP_BLOCK_BARS,
                "resamples": BOOTSTRAP_RESAMPLES,
                "seed": BOOTSTRAP_SEED,
                "chunk": BOOTSTRAP_CHUNK,
                "one_sided_confidence": BOOTSTRAP_CONFIDENCE,
                "gates": False,
            },
        },
        "sequencing": {
            "dev_reported_before_backcast": True,
            "backcast_requires_dev_run": True,
            "forward_final_requires_backcast_decision_same_digest": True,
            "forward_interim": LABEL_INTERIM,
            "forward_final_runs_once_not_before": FINAL_NOT_BEFORE,
            "no_change_to_any_constant_after_any_stage_result": True,
            "no_rescue_run": True,
            "higher_leverage": "a new protocol",
            "authorizes_orders": False,
        },
        "labels": list(LABELS),
    }


def protocol_digest() -> str:
    """SHA-256 hex digest of the canonical frozen protocol (bindings verified first)."""
    verify_port_binding()
    verify_fixture_binding()
    verify_lowturn_binding()
    verify_margin_binding()
    declared_halt_intervals()
    return hashlib.sha256(canonical_json_bytes(protocol_payload())).hexdigest()


__all__ = [
    "ANCHOR_BASE_SIGNAL_LAG_BARS",
    "ARMS",
    "BAR_HOURS",
    "BASE_SCENARIO",
    "BOOTSTRAP_BLOCK_BARS",
    "BOOTSTRAP_CHUNK",
    "BOOTSTRAP_CONFIDENCE",
    "BOOTSTRAP_KIND",
    "BOOTSTRAP_RESAMPLES",
    "BOOTSTRAP_SEED",
    "CANDIDATE",
    "CASH_LIMIT",
    "CASH_TOLERANCE_FRACTION",
    "CONTROLS",
    "CONTROL_CASH",
    "CONTROL_EW_SPOT",
    "AUTHORIZED_STAGES",
    "DATA_AUTHORIZATION_URL",
    "DEADBAND",
    "DELAY_SCENARIO",
    "FINAL_NOT_BEFORE",
    "FIXTURE_SHA256",
    "HARNESS_PACKAGE_FILES",
    "HIGH_COST_SCENARIO",
    "INITIAL_CAPITAL",
    "INITIAL_MARGIN_RATE",
    "LABELS",
    "LABEL_FORWARD_ONLY",
    "LABEL_INTERIM",
    "LABEL_INVALID",
    "LABEL_NOT_SUPPORTED",
    "LABEL_SUPPORTED",
    "LEG_ROSTER",
    "MAX_DRAWDOWN_LIMIT",
    "NET_EXPOSURE_LIMIT",
    "MIN_BASE_SHARPE",
    "PAIR_WARMUP_BARS",
    "PERIODS_PER_YEAR",
    "PORT_SOURCE_DIGEST",
    "PROTOCOL_ID",
    "RISK_DRAWDOWN_START",
    "RISK_DRAWDOWN_STOP",
    "REASON_NEGATIVE_CASH",
    "REBALANCE_EVERY_BARS",
    "ROSTER",
    "SCENARIOS",
    "STAGES",
    "STAGE_BACKCAST",
    "STAGE_DEV",
    "STAGE_FORWARD",
    "STAGE_NOT_SUPPORTED",
    "STAGE_SUPPORTED",
    "STAGE_WARMUP_BARS",
    "STAGE_WINDOWS",
    "ProtocolBindingError",
    "arm_signal_lag_bars",
    "declared_halt_intervals",
    "fixture_path",
    "harness_source_digest",
    "port_source_digest",
    "protocol_digest",
    "protocol_payload",
    "risk_config",
    "scenario_cost",
    "scenario_signal_lag_bars",
    "stage_halt_intervals",
    "verify_fixture_binding",
    "verify_lowturn_binding",
    "verify_margin_binding",
    "verify_port_binding",
]
