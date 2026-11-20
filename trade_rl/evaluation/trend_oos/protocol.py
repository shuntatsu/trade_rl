"""Frozen constants and digest of the out-of-sample test ``trend_long_spot_oos_v1``.

The normative text is ``docs/specs/trend-long-spot-oos.md``. The rule under test is
``TS_TREND_LONG_SPOT`` of ``lowturn_family_screen_v1`` and every value it trades with
is *imported* from :mod:`trade_rl.evaluation.lowturn_screen.protocol` (never re-typed),
so this protocol cannot drift away from it. The digest of the lowturn protocol that is
reused is pinned below and re-checked by :func:`verify_lowturn_binding`: a change to a
reused constant is detected instead of silently altering this protocol. Nothing in this
module reads market data or computes a return.
"""

from __future__ import annotations

import hashlib
import re
from datetime import UTC, datetime
from typing import Final

from trade_rl.artifacts import canonical_json_bytes
from trade_rl.evaluation.carry_multisymbol.protocol import SPOT_HALTS
from trade_rl.evaluation.lowturn_screen import protocol as lowturn
from trade_rl.risk.pretrade import PreTradeRiskConfig
from trade_rl.simulation.execution import ExecutionCostConfig

PROTOCOL_SCHEMA: Final = "trend_long_spot_oos_protocol_v1"
PROTOCOL_ID: Final = "trend_long_spot_oos_v1"

#: The lowturn protocol this one reuses, and the digest it had when this one was frozen
#: (``feat/lowturn-family-screen`` at ``7fb05514``).
LOWTURN_PROTOCOL_ID: Final = lowturn.PROTOCOL_ID
LOWTURN_PROTOCOL_DIGEST: Final = (
    "9de83d14ec756924cd9d5bc36da8a95fe7d3e3daf258c59a01584fd23eb2c433"
)

# --- reused unchanged from lowturn_family_screen_v1 (imported, not re-typed) ---------
ROSTER: Final[tuple[str, ...]] = lowturn.ROSTER
LEG_ROSTER: Final[tuple[str, ...]] = lowturn.LEG_ROSTER
N_PAIRS: Final = lowturn.N_PAIRS
LOOKBACK_BARS: Final[dict[str, int]] = dict(lowturn.LOOKBACK_BARS)
WARMUP_BARS: Final = lowturn.WARMUP_BARS
REBALANCE_EVERY_BARS: Final = lowturn.REBALANCE_EVERY_BARS
DEADBAND: Final = lowturn.DEADBAND
VOL_TARGET: Final = lowturn.VOL_TARGET
PER_NAME_CAP: Final = lowturn.PER_NAME_CAP
GROSS_CAP: Final = lowturn.GROSS_CAP
PERIODS_PER_YEAR: Final = lowturn.PERIODS_PER_YEAR
INITIAL_CAPITAL: Final = lowturn.INITIAL_CAPITAL
SCENARIOS: Final[tuple[str, ...]] = lowturn.SCENARIOS
BASE_SCENARIO: Final = "BASE"
HIGH_COST_SCENARIO: Final = "HIGH_COST"
DELAY_SCENARIO: Final = "DELAY_1BAR"

CANDIDATE: Final = "TS_TREND_LONG_SPOT"
CONTROL_CASH: Final = "CASH"
CONTROL_EW_SPOT: Final = "EW_HOLD_SPOT"
CONTROLS: Final[tuple[str, ...]] = (CONTROL_CASH, CONTROL_EW_SPOT)
ARMS: Final[tuple[str, ...]] = (CANDIDATE, *CONTROLS)

STAGES: Final[tuple[str, ...]] = ("BACKCAST", "FORWARD")
STAGE_BACKCAST: Final = "BACKCAST"
STAGE_FORWARD: Final = "FORWARD"

#: Bar duration of the 4h clock.
BAR_HOURS: Final = 4

# --- the stage windows, exactly as the spec table gives them -------------------------
#: ``(load start, first decision, last decision bar, settles)``; every instant is UTC.
#: The first decision, the last decision bar and the settlement instant are timestamps of
#: dataset rows (bar *close* times) in the repository's snapshot convention: a decision
#: at row ``t`` reads closes of rows ``<= t`` and trades in the next bar. The replay
#: engine decides on rows ``[first, stop - 2]``, places the flattening order on row
#: ``stop - 1`` and marks the book on row ``stop``, so the last dataset row is two bars
#: after the last decision bar (see :func:`trade_rl.evaluation.trend_oos.data.StageWindow`).
STAGE_WINDOWS: Final[dict[str, dict[str, str]]] = {
    STAGE_BACKCAST: {
        "load_start": "2018-01-01T00:00:00Z",
        "first_decision": "2018-07-01T00:00:00Z",
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
#: The FORWARD final scoring may not run before this instant.
FINAL_NOT_BEFORE: Final = "2027-05-02T00:00:00Z"
#: Owner comment that authorizes the out-of-sample and forward reads (Issue #810). The
#: FORWARD read path is bound to exactly this URL.
DATA_AUTHORIZATION_URL: Final = (
    "https://github.com/shuntatsu/trade_rl/issues/810#issuecomment-7028416072"
)

# --- the decision rule thresholds (frozen) -------------------------------------------
MAX_DRAWDOWN_LIMIT: Final = 0.20
MIN_BASE_SHARPE: Final = 0.30

# --- the reported, never-gating bootstrap (frozen) -----------------------------------
BOOTSTRAP_KIND: Final = "stationary_block_bootstrap_politis_romano_circular"
BOOTSTRAP_BLOCK_BARS: Final = 42
BOOTSTRAP_RESAMPLES: Final = 10_000
BOOTSTRAP_SEED: Final = 20261120
BOOTSTRAP_CONFIDENCE: Final = 0.95
#: Resamples are drawn in fixed chunks of this size from one generator; the chunk size
#: fixes the random stream, so it is part of the frozen definition.
BOOTSTRAP_CHUNK: Final = 250

LABEL_SUPPORTED: Final = "OOS_SUPPORTED"
LABEL_NOT_SUPPORTED: Final = "OOS_NOT_SUPPORTED"
LABEL_INVALID: Final = "INVALID"
LABELS: Final[tuple[str, ...]] = (LABEL_SUPPORTED, LABEL_NOT_SUPPORTED, LABEL_INVALID)
LABEL_INTERIM: Final = "INTERIM_NOT_A_DECISION"
CLAIM_BACKCAST: Final = LABEL_SUPPORTED
CLAIM_BOTH: Final = "OOS_AND_FORWARD_SUPPORTED"

_SCENARIO_DELAY_BARS: Final[dict[str, int]] = {
    BASE_SCENARIO: 0,
    HIGH_COST_SCENARIO: 0,
    DELAY_SCENARIO: 1,
}


#: Exchange-wide spot outages of the BACKCAST window, declared result-blind from the
#: coverage report whose SHA-256 is ``COVERAGE_EVIDENCE_SHA256``. Each entry is a
#: half-open ``(first missing hour open, end)`` UTC pair on whole hours; only ranges
#: that every symbol listed then is missing qualify. They are treated exactly as the
#: registered ``SPOT_HALTS`` (non-tradable rows with the preceding published close as
#: their mark, zero volume); it is the only carried-forward close the protocol allows.
DECLARED_HALTS: Final[tuple[tuple[str, str], ...]] = (
    ("2018-01-04T04:00:00Z", "2018-01-04T05:00:00Z"),
    ("2018-02-08T01:00:00Z", "2018-02-09T10:00:00Z"),
    ("2018-06-26T02:00:00Z", "2018-06-26T12:00:00Z"),
    ("2018-06-27T13:00:00Z", "2018-06-27T14:00:00Z"),
    ("2018-07-04T01:00:00Z", "2018-07-04T08:00:00Z"),
    ("2018-10-19T06:00:00Z", "2018-10-19T09:00:00Z"),
    ("2018-11-14T02:00:00Z", "2018-11-14T09:00:00Z"),
    ("2019-03-12T02:00:00Z", "2019-03-12T08:00:00Z"),
    ("2019-05-15T03:00:00Z", "2019-05-15T13:00:00Z"),
    ("2019-08-15T02:00:00Z", "2019-08-15T10:00:00Z"),
    ("2019-11-13T02:00:00Z", "2019-11-13T04:00:00Z"),
    ("2019-11-25T02:00:00Z", "2019-11-25T04:00:00Z"),
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
    "8680f82140fc6ce0b4c19076acebb4031b956bba52ac3940d9dc911a734c3d49"
)


class ProtocolBindingError(RuntimeError):
    """A constant this protocol reuses no longer matches the pinned lowturn protocol."""


def verify_lowturn_binding() -> None:
    """Fail closed unless the reused lowturn protocol has the pinned digest."""
    current = lowturn.protocol_digest()
    if current != LOWTURN_PROTOCOL_DIGEST:
        raise ProtocolBindingError(
            f"lowturn protocol digest {current} differs from the pinned "
            f"{LOWTURN_PROTOCOL_DIGEST}: the reused constants changed"
        )
    if lowturn.PROTOCOL_ID != LOWTURN_PROTOCOL_ID:
        raise ProtocolBindingError("lowturn protocol id changed")
    missing = [arm for arm in ARMS if arm not in lowturn.FAMILIES]
    if missing or CANDIDATE not in lowturn.CANDIDATE_FAMILIES:
        raise ProtocolBindingError(f"arms {missing} are not lowturn families")
    if set(CONTROLS) - set(lowturn.CONTROL_FAMILIES):
        raise ProtocolBindingError("controls are not lowturn control families")
    if tuple(SCENARIOS) != ("BASE", "HIGH_COST", "DELAY_1BAR"):
        raise ProtocolBindingError("lowturn scenarios changed")


def declared_halt_intervals() -> tuple[tuple[datetime, datetime], ...]:
    """The declared exchange-wide halts as UTC datetimes; validated, ordered, disjoint."""
    out: list[tuple[datetime, datetime]] = []
    for lo_text, hi_text in DECLARED_HALTS:
        lo = datetime.fromisoformat(lo_text.replace("Z", "+00:00")).astimezone(UTC)
        hi = datetime.fromisoformat(hi_text.replace("Z", "+00:00")).astimezone(UTC)
        if lo >= hi or lo.minute or lo.second or hi.minute or hi.second:
            raise ProtocolBindingError(f"declared halt {lo_text}..{hi_text} is invalid")
        if out and lo < out[-1][1]:
            raise ProtocolBindingError("declared halts must be ordered and disjoint")
        out.append((lo, hi))
    evidence: str | None = COVERAGE_EVIDENCE_SHA256
    if out and (evidence is None or re.fullmatch(r"[0-9a-f]{64}", evidence) is None):
        raise ProtocolBindingError(
            "declared halts need the coverage.json SHA-256 they were derived from"
        )
    return tuple(out)


def stage_halt_intervals() -> tuple[tuple[datetime, datetime], ...]:
    """Registered ``SPOT_HALTS`` plus the declared halts, ordered and disjoint."""
    merged = sorted([*SPOT_HALTS, *declared_halt_intervals()])
    for (_, a_hi), (b_lo, _) in zip(merged, merged[1:], strict=False):
        if b_lo < a_hi:
            raise ProtocolBindingError("registered and declared halts overlap")
    return tuple(merged)


def scenario_cost(scenario: str) -> ExecutionCostConfig:
    """The lowturn (core7 frozen) ``ExecutionCostConfig`` of the scenario, unchanged."""
    return lowturn.scenario_cost(scenario)


def scenario_signal_lag_bars(scenario: str) -> int:
    """Signal lag in bars; ``DELAY_1BAR`` lags every signal input by one bar."""
    if scenario not in _SCENARIO_DELAY_BARS:
        raise ValueError(f"unknown scenario: {scenario!r}")
    return lowturn.scenario_signal_lag_bars(scenario)


def risk_config() -> PreTradeRiskConfig:
    """The lowturn risk values (``core7_mtf_shadow_anchor_ppo_v4``), unchanged."""
    return lowturn.risk_config()


def protocol_payload() -> dict[str, object]:
    """Canonical frozen protocol payload for digesting."""
    rc = risk_config()
    return {
        "schema_version": PROTOCOL_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "claim_ceiling": {"backcast": CLAIM_BACKCAST, "both": CLAIM_BOTH},
        "reuses": {
            "protocol_id": LOWTURN_PROTOCOL_ID,
            "protocol_digest": LOWTURN_PROTOCOL_DIGEST,
            "code": "lowturn_screen policies and family_targets, unchanged "
            "(feat/lowturn-family-screen 7fb05514)",
        },
        "data_authorization_url": DATA_AUTHORIZATION_URL,
        "roster": list(ROSTER),
        "leg_layout": "alternating_spot_perp_even_index_spot_odd_index_perp",
        "arms": {"candidate": CANDIDATE, "controls": list(CONTROLS)},
        "scenarios": {
            name: {
                "cost": lowturn.scenario_cost(name).execution_policy_payload(),
                "signal_lag_bars": scenario_signal_lag_bars(name),
            }
            for name in SCENARIOS
        },
        "stages": {name: dict(window) for name, window in STAGE_WINDOWS.items()},
        "stage_row_convention": {
            "timestamps": "bar_close_of_the_4h_row",
            "first_decision": "the row whose close equals the first decision instant",
            "last_decision": "the row whose close equals the last decision bar",
            "settlement": "the flattening order is placed on the row closing at the "
            "settles instant, one row before the last row, which closes one 4h bar "
            "after it; the engine marks the terminal book there",
            "last_dataset_row": "last_decision_bar + 2 bars",
            "rebalance_phase": "(index - first_decision_index) % "
            f"{REBALANCE_EVERY_BARS}",
        },
        "final_scoring_not_before": FINAL_NOT_BEFORE,
        "data": {
            "source": "binance_vision_spot_1h_klines_checksum_verified",
            "aggregation": "hourly_to_4h_open_first_close_last_high_max_low_min_"
            "volume_sum_tradable_only_if_all_hours_tradable",
            "daily_fallback": "checksum_verified_daily_archives_only_provenance_recorded",
            "forward_fill": False,
            "spot_halts": [[lo.isoformat(), hi.isoformat()] for lo, hi in SPOT_HALTS],
            "declared_halts": {
                "intervals": [list(pair) for pair in DECLARED_HALTS],
                "coverage_json_sha256": COVERAGE_EVIDENCE_SHA256,
                "qualification": "only hours every symbol listed then is missing, "
                "after the month-from-daily-archives fallback",
                "treatment": "identical to SPOT_HALTS: non-tradable rows, zero volume, "
                "the preceding published close as the mark; the only carried-forward "
                "close allowed; any other gap is refused",
                "unusable_monthly_archive": "an unpublished or malformed monthly archive "
                "is rebuilt from checksum-verified daily archives; a day that is "
                "unpublished or malformed stays missing",
            },
            "perp_legs": "not_loaded_unavailable_in_every_row_neither_arm_reads_them",
            "unlisted_spot_rows": "asset_inactive_not_tradable_information_unavailable_"
            "constant_sentinel_price_no_invented_price",
            "unlisted_signal_inputs": "NaN_so_weight_0_until_every_lookback_close_exists",
            "partial_first_bar": "the 4h bar holding the first listed hour is unavailable",
            "sanity": "the first roster pair must be listed from the stage load start "
            "(an all-unavailable dataset means an unreachable source, not a result)",
            "exchange_rules": "static_snapshot_not_point_in_time",
        },
        "risk": {
            "max_gross": rc.max_gross,
            "max_abs_weight": rc.max_abs_weight,
            "max_turnover": rc.max_turnover,
            "drawdown_start": rc.drawdown_start,
            "drawdown_stop": rc.drawdown_stop,
            "emergency_turnover_override": rc.emergency_turnover_override,
        },
        "replay": {
            "path": "run_shared_portfolio_replay",
            "settle_terminal_position": True,
            "capture_ledger_evidence": True,
            "initial_capital": INITIAL_CAPITAL,
            "each_stage_starts_flat": True,
        },
        "rule": {
            "lookback_bars": dict(LOOKBACK_BARS),
            "vol_target": VOL_TARGET,
            "per_name_cap": PER_NAME_CAP,
            "gross_cap": GROSS_CAP,
            "rebalance_every_bars": REBALANCE_EVERY_BARS,
            "deadband": DEADBAND,
            "periods_per_year": PERIODS_PER_YEAR,
            "TS_TREND_LONG_SPOT": "spot legs, s = mean(sign r30, r90, r180) clipped "
            "to [0, 1], weight s * (vol_target / vol60) / 11, caps 0.40 per name and "
            "gross 2.0",
            "EW_HOLD_SPOT": "usable spot legs at 1/11",
            "CASH": "always flat",
        },
        "cell_validity": {
            "ledger_reconciled": True,
            "causality_probe_passed": True,
            "not_insolvent": True,
            "terminal_book": "flat or only lean-core terminal dust",
            "terminal_dust": "a residual whose terminal close the engine rejected for "
            "below_minimum_notional; valued at mark minus the modelled exit cost, which "
            "is deducted from the return; summary called with execution_cost",
        },
        "decision_rule": {
            "all_nine_cells_valid": True,
            "candidate_max_drawdown_below": MAX_DRAWDOWN_LIMIT,
            "candidate_base_return_above": 0.0,
            "candidate_base_sharpe_at_least": MIN_BASE_SHARPE,
            "candidate_high_cost_and_delay_returns_above": 0.0,
            "candidate_base_sharpe_above": CONTROL_EW_SPOT,
            "invalid_if": "any cell invalid",
        },
        "reported_not_gating": {
            "per_calendar_year_returns": True,
            "turnover_cost_max_drawdown": True,
            "bootstrap": {
                "kind": BOOTSTRAP_KIND,
                "statistic": "mean per-bar BASE excess return over the control",
                "controls": list(CONTROLS),
                "block_mean_bars": BOOTSTRAP_BLOCK_BARS,
                "resamples": BOOTSTRAP_RESAMPLES,
                "seed": BOOTSTRAP_SEED,
                "chunk": BOOTSTRAP_CHUNK,
                "one_sided_confidence": BOOTSTRAP_CONFIDENCE,
                "gates": False,
            },
        },
        "sequencing": {
            "backcast_runs_once": True,
            "forward_requires_backcast_supported": True,
            "forward_interim": LABEL_INTERIM,
            "forward_final_runs_once_not_before": FINAL_NOT_BEFORE,
            "no_parameter_sizing_leverage_symbol_window_or_cost_change": True,
        },
        "labels": list(LABELS),
    }


def protocol_digest() -> str:
    """SHA-256 hex digest of the canonical frozen protocol (binding verified first)."""
    verify_lowturn_binding()
    declared_halt_intervals()
    return hashlib.sha256(canonical_json_bytes(protocol_payload())).hexdigest()


__all__ = [
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
    "CLAIM_BACKCAST",
    "CLAIM_BOTH",
    "CONTROLS",
    "COVERAGE_EVIDENCE_SHA256",
    "DECLARED_HALTS",
    "CONTROL_CASH",
    "CONTROL_EW_SPOT",
    "DATA_AUTHORIZATION_URL",
    "DEADBAND",
    "DELAY_SCENARIO",
    "FINAL_NOT_BEFORE",
    "GROSS_CAP",
    "HIGH_COST_SCENARIO",
    "INITIAL_CAPITAL",
    "LABELS",
    "LABEL_INTERIM",
    "LABEL_INVALID",
    "LABEL_NOT_SUPPORTED",
    "LABEL_SUPPORTED",
    "LEG_ROSTER",
    "LOOKBACK_BARS",
    "LOWTURN_PROTOCOL_DIGEST",
    "LOWTURN_PROTOCOL_ID",
    "MAX_DRAWDOWN_LIMIT",
    "MIN_BASE_SHARPE",
    "N_PAIRS",
    "PERIODS_PER_YEAR",
    "PER_NAME_CAP",
    "PROTOCOL_ID",
    "PROTOCOL_SCHEMA",
    "REBALANCE_EVERY_BARS",
    "ROSTER",
    "SCENARIOS",
    "STAGES",
    "STAGE_BACKCAST",
    "STAGE_FORWARD",
    "STAGE_WINDOWS",
    "VOL_TARGET",
    "WARMUP_BARS",
    "ProtocolBindingError",
    "declared_halt_intervals",
    "protocol_digest",
    "protocol_payload",
    "risk_config",
    "scenario_cost",
    "scenario_signal_lag_bars",
    "stage_halt_intervals",
    "verify_lowturn_binding",
]
