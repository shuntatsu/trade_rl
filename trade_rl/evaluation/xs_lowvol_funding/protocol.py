"""Frozen constants and digest of ``xs_lowvol_funding_v1``.

The normative text is ``docs/specs/xs-lowvol-funding.md``; every constant below is that
spec's, re-typed once here and nowhere else. Nothing here reads market data or computes a
return. Three bindings make a change visible: the SHA-256 of the frozen candidate list
(``candidates.json``), the reused ``trend_oos`` / ``lowturn_screen`` bindings, and a
*computed* digest of the harness source (the package files and the tool), so a change to
any of them is a new protocol digest.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

from trade_rl.artifacts import canonical_json_bytes
from trade_rl.evaluation.core7_mtf_study import protocol as core7
from trade_rl.evaluation.lowturn_screen import protocol as lowturn
from trade_rl.evaluation.trend_oos import protocol as trend
from trade_rl.risk.pretrade import PreTradeRiskConfig
from trade_rl.simulation.execution import ExecutionCostConfig

PROTOCOL_SCHEMA: Final = "xs_lowvol_funding_protocol_v1"
PROTOCOL_ID: Final = "xs_lowvol_funding_v1"
SPEC_PATH: Final = "docs/specs/xs-lowvol-funding.md"

# --- frozen candidate list ------------------------------------------------------------
CANDIDATES_RELATIVE_PATH: Final = (
    "trade_rl/evaluation/xs_lowvol_funding/candidates.json"
)
CANDIDATES_SHA256: Final = (
    "fc680208899cd4166c9923f9cf9e8f8ebe829010aae073a935de10d6e6fefdf6"
)
CANDIDATES_COUNT: Final = 441

#: The reused lowturn/trend bindings (the stationary bootstrap and the cash tooling).
LOWTURN_PROTOCOL_ID: Final = trend.LOWTURN_PROTOCOL_ID
LOWTURN_PROTOCOL_DIGEST: Final = trend.LOWTURN_PROTOCOL_DIGEST

# --- clock, capital, universe ----------------------------------------------------------
#: The 11 Tier D core pairs of the DEV stage (known-used data); perps only.
DEV_ROSTER: Final[tuple[str, ...]] = core7.ROSTER
BAR_HOURS: Final = 4
PERIODS_PER_YEAR: Final = lowturn.PERIODS_PER_YEAR
INITIAL_CAPITAL: Final = lowturn.INITIAL_CAPITAL
#: Weekly decisions at the 4h bar closing Monday 00:00 UTC: 7 days of 6 bars.
REBALANCE_EVERY_BARS: Final = 42
UNIVERSE_SIZE: Final = 30
MIN_ELIGIBLE: Final = 12
ELIGIBILITY_DAYS: Final = 90
VOLUME_DAYS: Final = 30
FUNDING_LOOKBACK_DAYS: Final = 3
MIN_FUNDING_EVENTS: Final = 9

# --- policy ----------------------------------------------------------------------------
SIGMA_BARS: Final = 180  # 30 days of 4h log returns
FUNDING_EVENTS: Final = 9  # f_3d: the last 9 settled funding rates
COV_BARS: Final = 360
#: Closes a name needs before it can be ranked and sized: 360 returns need 361 closes.
HISTORY_CLOSES: Final = COV_BARS + 1
LEG_DIVISOR: Final = 3
SCORE_WEIGHT_VOL: Final = 0.5
SCORE_WEIGHT_FUNDING: Final = 0.5
VOL_TARGET: Final = 0.20
COV_SHRINKAGE: Final = 0.5
MAX_ABS_WEIGHT: Final = 0.25
GROSS_CAP: Final = 2.0
#: The cash account of ``agy_anchor_oos`` (engine margin = gross / max_leverage = 0.4).
INITIAL_MARGIN_RATE: Final = 0.4
CASH_LIMIT: Final = 1.0
NET_EXPOSURE_LIMIT: Final = 0.97
MAX_LEVERAGE: Final = 2.5
PARTICIPATION_CAP: Final = 0.01
MINIMUM_NOTIONAL: Final = 10.0
MAINTENANCE_MARGIN_RATE: Final = 0.05
#: Extra adverse charge on a forced close (and on a written-off residual).
FORCED_CLOSE_ADVERSE: Final = 0.05
#: A held name is closed from this many 4h bars before its last available bar (about the
#: one-week public delisting notice); the lead uses knowledge of the archive end.
FORCED_CLOSE_LEAD_BARS: Final = 42
#: Amendment 2: the effective listing end of a perp is the close of its last bar with volume
#: (trailing zero-volume "zombie" bars are dropped); the forced-close lead counts back from it.
#: An unpublished monthly funding archive in a traded month is filled from the REST endpoint;
#: if that returns no event, the merged series must have no gap longer than this many hours there.
#: The same span is the minimum length of an interior zero-volume run turned into missing hours.
FUNDING_MAX_GAP_HOURS: Final = 24
#: A cell whose cash net of cumulative execution costs is below minus this fraction of the
#: initial capital at any interval is invalid.
CASH_TOLERANCE_FRACTION: Final = 1e-6
REASON_NEGATIVE_CASH: Final = "negative_cash_borrowing"
#: Rules for a symbol that is absent from the static exchangeInfo snapshot (a delisted
#: perpetual): the minimum notional is the scenario's 10 USDT floor.
FALLBACK_LOT_SIZE: Final = 1e-3
FALLBACK_MIN_NOTIONAL: Final = MINIMUM_NOTIONAL

# --- arms and scenarios -----------------------------------------------------------------
CANDIDATE: Final = "CANDIDATE"
ARM_LOWVOL: Final = "LOWVOL_ONLY"
ARM_FUNDING: Final = "FUNDING_ONLY"
ARM_EW_LONG: Final = "EW_LONG"
CONTROL_CASH: Final = "CASH"
ARMS: Final[tuple[str, ...]] = (
    CANDIDATE,
    ARM_LOWVOL,
    ARM_FUNDING,
    ARM_EW_LONG,
    CONTROL_CASH,
)
BASE_SCENARIO: Final = "BASE"
HIGH_COST_SCENARIO: Final = "HIGH_COST"
DELAY_SCENARIO: Final = "DELAY"
SCENARIOS: Final[tuple[str, ...]] = (BASE_SCENARIO, HIGH_COST_SCENARIO, DELAY_SCENARIO)
#: ``(fee, spread, impact coefficient, extra signal lag in bars)``.
_SCENARIO_TABLE: Final[dict[str, tuple[float, float, float, int]]] = {
    BASE_SCENARIO: (0.0005, 0.0003, 0.0002, 0),
    HIGH_COST_SCENARIO: (0.0005, 0.0008, 0.0006, 0),
    DELAY_SCENARIO: (0.0005, 0.0003, 0.0002, 1),
}

# --- stages -----------------------------------------------------------------------------
STAGE_DEV: Final = "DEV"
STAGE_DISJOINT: Final = "DISJOINT"
STAGE_FORWARD: Final = "FORWARD"
STAGES: Final[tuple[str, ...]] = (STAGE_DEV, STAGE_DISJOINT, STAGE_FORWARD)
#: ``(load start, first decision, last decision bar, settles)`` in UTC; all are bar-close
#: timestamps of dataset rows. The last dataset row is the settlement close (the engine
#: decides through stop - 2, flattens on stop - 1 and marks on stop); the last weekly
#: decision is the last Monday before it. Hourly opens stay strictly before the
#: 2026-08-01 embargo for DEV and DISJOINT.
STAGE_WINDOWS: Final[dict[str, dict[str, str]]] = {
    STAGE_DEV: {
        "load_start": "2022-01-01T00:00:00Z",
        "first_decision": "2022-04-04T00:00:00Z",
        "last_decision_bar": "2026-07-27T00:00:00Z",
        "settles": "2026-08-01T00:00:00Z",
    },
    STAGE_DISJOINT: {
        "load_start": "2021-01-01T00:00:00Z",
        "first_decision": "2021-04-05T00:00:00Z",
        "last_decision_bar": "2026-07-27T00:00:00Z",
        "settles": "2026-08-01T00:00:00Z",
    },
    STAGE_FORWARD: {
        "load_start": "2026-04-01T00:00:00Z",
        "first_decision": "2026-08-03T00:00:00Z",
        "last_decision_bar": "2027-04-26T00:00:00Z",
        "settles": "2027-05-01T00:00:00Z",
    },
}
FINAL_NOT_BEFORE: Final = "2027-05-08T00:00:00Z"
#: Owner scope relay (Issue #810) covering fetching and reading data for any pre-registered
#: out-of-sample / forward protocol. Bound for DISJOINT and FORWARD reads; DEV reads
#: known-used data and needs no relay of its own.
DATA_AUTHORIZATION_URL: Final = (
    "https://github.com/shuntatsu/trade_rl/issues/810#issuecomment-7040247247"
)
AUTHORIZED_STAGES: Final[tuple[str, ...]] = (STAGE_DISJOINT, STAGE_FORWARD)
#: The two halves of the DISJOINT window of condition C5 (inclusive calendar dates).
HALF_1: Final = ("2021-04-05", "2023-12-31")
HALF_2: Final = ("2024-01-01", "2026-07-31")
#: A FORWARD decision is scored on decisions from this instant.
FORWARD_SCORING_FROM: Final = "2026-08-03T00:00:00Z"

# --- decision rule ----------------------------------------------------------------------
RISK_DRAWDOWN_START: Final = 0.125
RISK_DRAWDOWN_STOP: Final = 0.25
MAX_DRAWDOWN_LIMIT: Final = 0.25
C2_MIN_BASE_SHARPE: Final = 0.6
C3_MIN_HIGH_COST_SHARPE: Final = 0.3
C3_MIN_DELAY_SHARPE: Final = 0.4
LABEL_SUPPORTED: Final = "XS_SUPPORTED"
LABEL_NOT_SUPPORTED: Final = "XS_NOT_SUPPORTED"
LABEL_INVALID: Final = "INVALID"
LABELS: Final[tuple[str, ...]] = (LABEL_SUPPORTED, LABEL_NOT_SUPPORTED, LABEL_INVALID)
LABEL_FORWARD_SUPPORTED: Final = "XS_FORWARD_SUPPORTED"
LABEL_FORWARD_NOT_SUPPORTED: Final = "XS_FORWARD_NOT_SUPPORTED"
LABEL_INTERIM: Final = "INTERIM_NOT_A_DECISION"
BOOTSTRAP_KIND: Final = trend.BOOTSTRAP_KIND
BOOTSTRAP_BLOCK_BARS: Final = 42
BOOTSTRAP_RESAMPLES: Final = 10_000
BOOTSTRAP_SEED: Final = 20261121
BOOTSTRAP_CONFIDENCE: Final = 0.90
BOOTSTRAP_CHUNK: Final = 250
#: Declared prior trials of the reported (never gating) deflated Sharpe.
PRIOR_TRIALS: Final = 40

#: Exchange-wide perpetual outages declared result-blind from the DISJOINT coverage report
#: (Amendment 1). The DISJOINT coverage report (code ef11e0b0) found no exchange-wide gap, so
#: no halt is declared; its SHA-256 is pinned as the evidence for that empty declaration.
DECLARED_HALTS: Final[tuple[tuple[str, str], ...]] = ()
COVERAGE_EVIDENCE_SHA256: Final[str | None] = (
    "757cbdd8f4d0a1ce7c783ea91bd3e7a16f719b5e5c3e8d00f1979e6a6b8b7d5c"
)
#: SHA-256 of the FORWARD symbol file, pinned by a result-blind amendment before any FORWARD
#: read; FORWARD is refused while it is None.
FORWARD_SYMBOLS_SHA256: Final[str | None] = (
    "48656a5dfb73491dacd2234490ba7c34c79b5692fcc91812a15095434900ba66"
)
FORWARD_SYMBOLS_RELATIVE_PATH: Final = (
    "trade_rl/evaluation/xs_lowvol_funding/forward_symbols.json"
)

#: The harness files whose bytes enter the digest (a change to any of them is a new digest).
HARNESS_PACKAGE_FILES: Final[tuple[str, ...]] = (
    "protocol.py",
    "universe.py",
    "policy.py",
    "data.py",
    "decision.py",
    "runner.py",
)
HARNESS_TOOL_RELATIVE_PATH: Final = "tools/xs_lowvol_funding.py"

#: FORWARD listing exclusions: stablecoin or gold bases and non-ASCII symbols.
_ASCII_SYMBOL: Final = re.compile(r"[A-Z0-9]+USDT")
_EXCLUDED_BASES: Final[frozenset[str]] = frozenset(
    {
        "USDC",
        "BUSD",
        "TUSD",
        "FDUSD",
        "USDP",
        "DAI",
        "USDE",
        "USD1",
        "XAU",
        "XAUT",
        "PAXG",
    }
)


class ProtocolBindingError(RuntimeError):
    """A pinned binding (candidate list, reused protocol, margin rate) no longer holds."""


def _normalised(data: bytes) -> bytes:
    return data.replace(b"\r\n", b"\n")


def candidates_path() -> Path:
    return Path(__file__).resolve().parents[3] / CANDIDATES_RELATIVE_PATH


def verify_candidates_binding(path: Path | None = None) -> None:
    """Fail closed unless the candidate list exists with the pinned SHA-256."""
    target = candidates_path() if path is None else path
    if not target.is_file():
        raise ProtocolBindingError(f"candidate list {target} is missing")
    actual = hashlib.sha256(target.read_bytes()).hexdigest()
    if actual != CANDIDATES_SHA256:
        raise ProtocolBindingError(
            f"candidate list sha256 {actual} differs from the pinned {CANDIDATES_SHA256}"
        )


def forward_symbols_path() -> Path:
    return Path(__file__).resolve().parents[3] / FORWARD_SYMBOLS_RELATIVE_PATH


def load_candidates(path: Path | None = None) -> tuple[str, ...]:
    """The frozen 441 candidate symbols (the SHA-256 is verified first)."""
    verify_candidates_binding(path)
    target = candidates_path() if path is None else path
    document = json.loads(target.read_bytes())
    symbols = document.get("perp_symbols")
    if (
        not isinstance(symbols, list)
        or len(symbols) != CANDIDATES_COUNT
        or len(set(symbols)) != CANDIDATES_COUNT
        or not all(isinstance(item, str) for item in symbols)
    ):
        raise ProtocolBindingError("candidate list is not 441 unique symbols")
    return tuple(symbols)


def forward_symbol_allowed(symbol: str) -> bool:
    """Whether a listed USDT perp may enter the FORWARD universe (spec exclusions).

    Only ASCII ``[A-Z0-9]+USDT`` symbols are admitted, minus stablecoin and gold bases.
    """
    if _ASCII_SYMBOL.fullmatch(symbol) is None:
        return False
    return symbol[: -len("USDT")] not in _EXCLUDED_BASES


def harness_source_digest(
    *, package_dir: Path | None = None, tool_path: Path | None = None
) -> str:
    """SHA-256 over the harness source (name and LF-normalised bytes).

    It covers the package modules named in ``HARNESS_PACKAGE_FILES`` and
    ``tools/xs_lowvol_funding.py``. It is *computed*, not pinned: protocol.py is one of
    the hashed files, so a pinned value would be self-referential. Any edit to the
    harness therefore changes the protocol digest. A missing file fails closed.
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


def verify_lowturn_binding() -> None:
    trend.verify_lowturn_binding()


def verify_margin_binding() -> None:
    """The initial-margin rate must be the engine's ``1 / max_leverage`` in every scenario."""
    for name in SCENARIOS:
        cost = scenario_cost(name)
        if cost.max_leverage != MAX_LEVERAGE:
            raise ProtocolBindingError("scenario max_leverage is not the frozen 2.5")
        if cost.max_participation_rate != PARTICIPATION_CAP:
            raise ProtocolBindingError("scenario participation cap is not 0.01")
        if abs(1.0 / cost.max_leverage - INITIAL_MARGIN_RATE) > 1e-15:
            raise ProtocolBindingError("initial margin rate is not 1 / max_leverage")
    rc = risk_config()
    if (rc.max_gross, rc.max_abs_weight, rc.max_turnover) != (GROSS_CAP, 0.25, 2.5):
        raise ProtocolBindingError("risk caps differ from the frozen 2.0 / 0.25 / 2.5")
    if (rc.drawdown_start, rc.drawdown_stop) != (0.125, 0.25):
        raise ProtocolBindingError("risk ramp differs from the frozen 0.125 / 0.25")


def scenario_cost(scenario: str) -> ExecutionCostConfig:
    """The ``ExecutionCostConfig`` of a scenario (spec table)."""
    if scenario not in _SCENARIO_TABLE:
        raise ValueError(f"unknown scenario: {scenario!r}")
    fee, spread, impact, _ = _SCENARIO_TABLE[scenario]
    return ExecutionCostConfig(
        fee_rate=fee,
        spread_rate=spread,
        impact_rate=impact,
        allow_short=True,
        max_leverage=MAX_LEVERAGE,
        maintenance_margin_rate=MAINTENANCE_MARGIN_RATE,
        max_participation_rate=PARTICIPATION_CAP,
        minimum_notional=MINIMUM_NOTIONAL,
        order_latency_bars=0,
        processing_bar_volume_capacity=False,
    )


def scenario_signal_lag_bars(scenario: str) -> int:
    """Extra signal lag in bars of a scenario (0, and 1 for ``DELAY``)."""
    if scenario not in _SCENARIO_TABLE:
        raise ValueError(f"unknown scenario: {scenario!r}")
    return _SCENARIO_TABLE[scenario][3]


def risk_config() -> PreTradeRiskConfig:
    """The frozen risk config; the risk layer can only shrink the book."""
    return PreTradeRiskConfig(
        max_gross=GROSS_CAP,
        max_abs_weight=MAX_ABS_WEIGHT,
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


def protocol_payload() -> dict[str, object]:
    """Canonical frozen protocol payload for digesting."""
    rc = risk_config()
    return {
        "schema_version": PROTOCOL_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "spec": SPEC_PATH,
        "candidates": {
            "path": CANDIDATES_RELATIVE_PATH,
            "sha256": CANDIDATES_SHA256,
            "count": CANDIDATES_COUNT,
        },
        "reuses": {
            "lowturn_protocol_id": LOWTURN_PROTOCOL_ID,
            "lowturn_protocol_digest": LOWTURN_PROTOCOL_DIGEST,
            "bootstrap": BOOTSTRAP_KIND,
        },
        "data_authorization": {
            "url": DATA_AUTHORIZATION_URL,
            "stages": list(AUTHORIZED_STAGES),
            "binding": "owner-authored and unedited between two fetches, for every "
            "DISJOINT and FORWARD read; recorded in the receipt",
            "dev": "known-used data; no relay needed",
        },
        "harness": {
            "files": [*HARNESS_PACKAGE_FILES, HARNESS_TOOL_RELATIVE_PATH],
            "harness_source_digest": harness_source_digest(),
            "binding": "computed, not pinned (protocol.py is hashed)",
        },
        "clock": {
            "bar_hours": BAR_HOURS,
            "periods_per_year": PERIODS_PER_YEAR,
            "timestamps": "bar_close_of_the_4h_row",
            "rebalance_every_bars": REBALANCE_EVERY_BARS,
            "decision": "the 4h bar closing Monday 00:00 UTC; orders fill at the next bar",
        },
        "universe": {
            "dev": list(DEV_ROSTER),
            "monthly_top": UNIVERSE_SIZE,
            "min_eligible": MIN_ELIGIBLE,
            "eligibility": "complete 1h perp bars for the 90 days strictly before the "
            "month start (or missing only inside a declared halt) and at least 9 "
            "settled funding events in the 3 days before it",
            "ranking": "perp quote volume over the 30 days strictly before the month "
            "start, ties by symbol name",
            "forward_listing": "all ASCII USDT perps listed at run time minus stablecoin "
            "and gold bases, supplied to the tool as a recorded file",
            "forward_symbols_sha256": FORWARD_SYMBOLS_SHA256,
            "forward_symbols_path": FORWARD_SYMBOLS_RELATIVE_PATH,
            "forward_excluded_bases": sorted(_EXCLUDED_BASES),
            "forward_ascii_rule": "[A-Z0-9]+USDT",
        },
        "policy": {
            "sigma_bars": SIGMA_BARS,
            "sigma": "sample standard deviation (ddof 1) of the last 180 4h log returns",
            "funding_events": FUNDING_EVENTS,
            "funding": "mean of the last 9 settled funding rates with funding time "
            "<= t (event rate = bar sum / bar event count)",
            "history_closes": HISTORY_CLOSES,
            "score": "0.5 * zr(-sigma) + 0.5 * zr(-f); zr = (average rank - mean "
            "rank) / population std of ranks",
            "legs": "k = floor(n / 3) highest and lowest scores; ties by symbol order",
            "leg_weights": "w_i = 1 / (k * sigma_i) so that sum |w| * sigma = 1 per leg",
            "vol_target": VOL_TARGET,
            "covariance": f"last {COV_BARS} 4h log returns of the held names, ddof 1, "
            f"shrunk {COV_SHRINKAGE} toward its diagonal, annualised with "
            f"{PERIODS_PER_YEAR} bars",
            "caps_in_order": [
                f"per_name_abs_weight_{MAX_ABS_WEIGHT}",
                f"gross_{GROSS_CAP}_scaled_proportionally",
                "cash_account_margin_and_net_exposure",
            ],
            "forced_close_zeroing": "after the vol target and before the caps",
            "terminal_unwind": "every arm targets 0 at the last weekly decision of the stage window (the last Monday decision on or before the last decision bar) and re-issues the flat target on every later bar",
            "ex_ante_vol_basis": "raw leg weights",
        },
        "cash_account": {
            "initial_margin_rate": INITIAL_MARGIN_RATE,
            "cash_limit": CASH_LIMIT,
            "net_exposure_limit": NET_EXPOSURE_LIMIT,
            "rule": "0.4 * perp gross <= 1.0 and net signed exposure <= 0.97, the "
            "whole target scaled by the tighter bound, on the effective weights",
            "invalid_on_borrowing": {
                "reason": REASON_NEGATIVE_CASH,
                "tolerance_fraction": CASH_TOLERANCE_FRACTION,
            },
        },
        "forced_close": {
            "rule": "a held name's target is 0 from the decision bar 42 4h bars before "
            "its last available bar onward (the lead uses knowledge of the archive end, "
            "modelling the public delisting announcement); the end is the effective "
            "end: the close of the last 1h bar with volume (Amendment 2)",
            "lead_bars": FORCED_CLOSE_LEAD_BARS,
            "adverse_charge": FORCED_CLOSE_ADVERSE,
            "booked": "cash debit at each fill bar of the forced close, applied to the "
            "equity path from that bar on",
            "unclosed_at_last_bar": "the run refuses (no write-off branch; disclosed "
            "limitation that would need an amendment)",
            "reported": ["count", "total_charge", "result_without_charge"],
        },
        "risk": {
            "max_gross": rc.max_gross,
            "max_abs_weight": rc.max_abs_weight,
            "max_turnover": rc.max_turnover,
            "drawdown_start": rc.drawdown_start,
            "drawdown_stop": rc.drawdown_stop,
            "emergency_turnover_override": rc.emergency_turnover_override,
        },
        "scenarios": {
            name: {
                "cost": scenario_cost(name).execution_policy_payload(),
                "extra_signal_lag_bars": scenario_signal_lag_bars(name),
            }
            for name in SCENARIOS
        },
        "fallback_exchange_rules": {
            "lot_size": FALLBACK_LOT_SIZE,
            "min_notional": FALLBACK_MIN_NOTIONAL,
            "applies_to": "symbols absent from the static exchangeInfo snapshot",
        },
        "arms": list(ARMS),
        "stages": {name: dict(window) for name, window in STAGE_WINDOWS.items()},
        "final_scoring_not_before": FINAL_NOT_BEFORE,
        "forward_scoring_from": FORWARD_SCORING_FROM,
        "data": {
            "source": "binance_vision_usdm_1h_klines_and_funding_checksum_verified",
            "aggregation": "hourly_to_4h_open_first_close_last_high_max_low_min_"
            "volume_sum_funding_sum_tradable_only_if_all_hours_tradable",
            "bar_states": "all four hours published: active and tradable; the rest "
            "explained by a declared halt: active, not tradable, preceding close; "
            "otherwise inactive (sentinel price, not tradable)",
            "daily_fallback": "checksum_verified_daily_archives_only_provenance_recorded",
            "forward_fill": False,
            "relisting_gap": "an interior run of at least 24 consecutive 1h bars with base "
            "and quote volume 0 and no funding event strictly inside is a delisting and "
            "relisting: its bars become missing hours before eligibility, forced close, "
            "funding admissibility and gap checks (Amendment 2; the gap then follows the "
            "existing unexplained-gap rules; a held name is force-closed when the gap "
            "starts, the same archive-knowledge exception as the delisting forced close); "
            "shorter runs and runs with a funding event are kept",
            "zombie_tail": "the trailing run of zero-volume 1h bars after a perp's last "
            "bar with volume is dropped as not listed before universe, forced-close, "
            "funding and gap rules (Amendment 2; uses archive-end knowledge like the "
            "forced-close exception); interior zero-volume bars are kept",
            "rest_funding": {
                "when": "a monthly funding archive is unpublished in a month with a bar "
                "of positive volume",
                "source": "binance_fapi_v1_fundingRate_public_raw_json_cached_sha256",
                "empty_month_admissible_if_max_gap_hours_at_most": FUNDING_MAX_GAP_HOURS,
                "otherwise": "refuse (fail closed; network only with --allow-network)",
            },
            "declared_halts": {
                "intervals": [list(pair) for pair in DECLARED_HALTS],
                "coverage_json_sha256": COVERAGE_EVIDENCE_SHA256,
            },
            "spot_data": "none",
            "exchange_rules": "static_snapshot_not_point_in_time",
            "mark_price": "bar_close_proxy",
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
        "decision_rule": {
            "invalid": [
                "cash net of cumulative execution costs below -1e-6 * capital",
                "a non-dust residual after terminal settlement",
                "a row outside the stage window or authorization",
                "an unreconciled ledger or a failed causality probe",
            ],
            "conditions": {
                "C1": "no INVALID cell in any arm or scenario",
                "C2": f"BASE Sharpe >= {C2_MIN_BASE_SHARPE}",
                "C3": f"HIGH_COST Sharpe >= {C3_MIN_HIGH_COST_SHARPE} and DELAY "
                f"Sharpe >= {C3_MIN_DELAY_SHARPE}",
                "C4": f"BASE max drawdown < {MAX_DRAWDOWN_LIMIT} on the dust- and "
                "charge-adjusted path",
                "C5": f"BASE Sharpe > 0 in both halves {list(HALF_1)} and {list(HALF_2)}",
                "C6": "one-sided lower bound of the mean BASE 4h net return > 0",
            },
            "labels": list(LABELS),
            "forward_labels": [LABEL_FORWARD_SUPPORTED, LABEL_FORWARD_NOT_SUPPORTED],
            "bootstrap": {
                "kind": BOOTSTRAP_KIND,
                "block_mean_bars": BOOTSTRAP_BLOCK_BARS,
                "resamples": BOOTSTRAP_RESAMPLES,
                "seed": BOOTSTRAP_SEED,
                "chunk": BOOTSTRAP_CHUNK,
                "one_sided_confidence": BOOTSTRAP_CONFIDENCE,
            },
            "reported_not_gating": [
                "per_year_returns",
                "turnover_and_cost_drag",
                "funding_pnl",
                "beta_and_correlation_to_ew_long",
                "leg_attribution",
                "forced_close_counts",
                f"deflated_sharpe_{PRIOR_TRIALS}_prior_trials",
            ],
            "dev_stage": "diagnostic only; never gates, never tunes",
        },
        "sequencing": {
            "disjoint_requires_dev_result_without_invalid_cell": True,
            "disjoint_single_use": "runs exactly once; the label, INVALID included, is final",
            "forward_requires_disjoint_decision_label": LABEL_SUPPORTED,
            "forward_final_not_before": FINAL_NOT_BEFORE,
            "stop_rule": "XS_NOT_SUPPORTED or INVALID ends the line; no constant is retuned",
            "no_change_to_any_constant_after_any_stage_result": True,
            "authorizes_orders": False,
        },
    }


def protocol_digest() -> str:
    """SHA-256 hex digest of the canonical frozen protocol (bindings verified first)."""
    verify_candidates_binding()
    verify_lowturn_binding()
    verify_margin_binding()
    declared_halt_intervals()
    return hashlib.sha256(canonical_json_bytes(protocol_payload())).hexdigest()


__all__ = [
    "ARMS",
    "ARM_EW_LONG",
    "ARM_FUNDING",
    "ARM_LOWVOL",
    "AUTHORIZED_STAGES",
    "BAR_HOURS",
    "BASE_SCENARIO",
    "BOOTSTRAP_BLOCK_BARS",
    "BOOTSTRAP_CHUNK",
    "BOOTSTRAP_CONFIDENCE",
    "BOOTSTRAP_RESAMPLES",
    "BOOTSTRAP_SEED",
    "C2_MIN_BASE_SHARPE",
    "C3_MIN_DELAY_SHARPE",
    "C3_MIN_HIGH_COST_SHARPE",
    "CANDIDATE",
    "CANDIDATES_COUNT",
    "CANDIDATES_SHA256",
    "CASH_LIMIT",
    "CASH_TOLERANCE_FRACTION",
    "CONTROL_CASH",
    "COV_BARS",
    "COV_SHRINKAGE",
    "COVERAGE_EVIDENCE_SHA256",
    "DATA_AUTHORIZATION_URL",
    "DECLARED_HALTS",
    "DELAY_SCENARIO",
    "DEV_ROSTER",
    "ELIGIBILITY_DAYS",
    "FALLBACK_LOT_SIZE",
    "FALLBACK_MIN_NOTIONAL",
    "FINAL_NOT_BEFORE",
    "FORCED_CLOSE_ADVERSE",
    "FORCED_CLOSE_LEAD_BARS",
    "FUNDING_MAX_GAP_HOURS",
    "FORWARD_SCORING_FROM",
    "FORWARD_SYMBOLS_RELATIVE_PATH",
    "FORWARD_SYMBOLS_SHA256",
    "FUNDING_EVENTS",
    "FUNDING_LOOKBACK_DAYS",
    "GROSS_CAP",
    "HALF_1",
    "HALF_2",
    "HARNESS_PACKAGE_FILES",
    "HARNESS_TOOL_RELATIVE_PATH",
    "HIGH_COST_SCENARIO",
    "HISTORY_CLOSES",
    "INITIAL_CAPITAL",
    "INITIAL_MARGIN_RATE",
    "LABELS",
    "LABEL_FORWARD_NOT_SUPPORTED",
    "LABEL_FORWARD_SUPPORTED",
    "LABEL_INTERIM",
    "LABEL_INVALID",
    "LABEL_NOT_SUPPORTED",
    "LABEL_SUPPORTED",
    "LEG_DIVISOR",
    "LOWTURN_PROTOCOL_DIGEST",
    "LOWTURN_PROTOCOL_ID",
    "MAINTENANCE_MARGIN_RATE",
    "MAX_ABS_WEIGHT",
    "MAX_DRAWDOWN_LIMIT",
    "MAX_LEVERAGE",
    "MIN_ELIGIBLE",
    "MIN_FUNDING_EVENTS",
    "MINIMUM_NOTIONAL",
    "NET_EXPOSURE_LIMIT",
    "PARTICIPATION_CAP",
    "PERIODS_PER_YEAR",
    "PRIOR_TRIALS",
    "PROTOCOL_ID",
    "REASON_NEGATIVE_CASH",
    "REBALANCE_EVERY_BARS",
    "RISK_DRAWDOWN_START",
    "RISK_DRAWDOWN_STOP",
    "SCENARIOS",
    "SCORE_WEIGHT_FUNDING",
    "SCORE_WEIGHT_VOL",
    "SIGMA_BARS",
    "STAGES",
    "STAGE_DEV",
    "STAGE_DISJOINT",
    "STAGE_FORWARD",
    "STAGE_WINDOWS",
    "UNIVERSE_SIZE",
    "VOL_TARGET",
    "VOLUME_DAYS",
    "ProtocolBindingError",
    "candidates_path",
    "declared_halt_intervals",
    "forward_symbol_allowed",
    "forward_symbols_path",
    "harness_source_digest",
    "load_candidates",
    "protocol_digest",
    "protocol_payload",
    "risk_config",
    "scenario_cost",
    "scenario_signal_lag_bars",
    "verify_candidates_binding",
    "verify_lowturn_binding",
    "verify_margin_binding",
]
