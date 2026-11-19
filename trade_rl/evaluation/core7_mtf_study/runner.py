"""Thin study runner orchestrating Core7 MTF replay through ``run_shared_portfolio_replay``."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from trade_rl.artifacts import canonical_json_bytes
from trade_rl.data.features.mtf6_signals import (
    Mtf6SignalPack,
    build_mtf6_signal_pack,
    enforce_pre_fs1_boundary,
    validate_alternating_spot_perp_dataset,
)
from trade_rl.data.market import MarketDataset
from trade_rl.evaluation.core7_mtf_study import decision, protocol
from trade_rl.evaluation.core7_mtf_study.decision import CellSummary
from trade_rl.evaluation.replay import (
    SharedCashPortfolioReplayLedgerEvidence,
    SharedCashPortfolioReplayResult,
    run_shared_portfolio_replay,
)
from trade_rl.risk.pretrade import PreTradeRisk
from trade_rl.strategies.portfolio import (
    PortfolioAccountSnapshot,
    PortfolioDecisionSnapshot,
    PortfolioPolicy,
    PortfolioRiskLimits,
)
from trade_rl.strategies.rl.shadow_anchor_mtf import (
    CashHoldPortfolioPolicy,
    ShadowAnchorMlpWeights,
    ShadowAnchorMode,
    ShadowAnchorMtfConfig,
    ShadowAnchorMtfPolicy,
)

_PERIODS_PER_YEAR_4H = 2190.0
_RECONCILE_TOLERANCE = 1e-6


@dataclass(frozen=True, slots=True)
class CausalityProbeResult:
    """Result of future-bar perturbation causality probe at ``cut_index``."""

    cut_index: int
    policy_mode: str
    signal_max_abs_diff: float
    proposal_max_abs_diff: float
    passed: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "cut_index": self.cut_index,
            "policy_mode": self.policy_mode,
            "signal_max_abs_diff": self.signal_max_abs_diff,
            "proposal_max_abs_diff": self.proposal_max_abs_diff,
            "passed": self.passed,
        }


def _perturb_future_market_rows(
    dataset: MarketDataset,
    *,
    cut_index: int,
    seed: int,
) -> MarketDataset:
    if not 0 <= cut_index < dataset.n_bars - 1:
        raise ValueError("cut_index must satisfy 0 <= cut_index < dataset.n_bars - 1")
    rng = np.random.default_rng(seed)
    sl = slice(cut_index + 1, dataset.n_bars)
    shape = (dataset.n_bars - cut_index - 1, dataset.n_symbols)

    mult = np.exp(rng.normal(0.0, 0.25, size=shape))
    open_arr = np.asarray(dataset.open, dtype=np.float64).copy()
    high_arr = np.asarray(dataset.high, dtype=np.float64).copy()
    low_arr = np.asarray(dataset.low, dtype=np.float64).copy()
    close_arr = np.asarray(dataset.close, dtype=np.float64).copy()
    vol_arr = np.asarray(dataset.volume, dtype=np.float64).copy()
    fr_arr = np.asarray(dataset.funding_rate, dtype=np.float64).copy()

    open_arr[sl] = np.maximum(open_arr[sl] * mult, 1e-8)
    close_arr[sl] = np.maximum(
        close_arr[sl] * mult * np.exp(rng.normal(0.0, 0.12, size=shape)), 1e-8
    )
    spread_hi = 1.0 + np.abs(rng.normal(0.0, 0.05, size=shape))
    spread_lo = np.maximum(0.2, 1.0 - np.abs(rng.normal(0.0, 0.05, size=shape)))
    high_arr[sl] = np.maximum(
        np.maximum(open_arr[sl], close_arr[sl]) * spread_hi, high_arr[sl] * mult
    )
    low_arr[sl] = np.minimum(
        np.minimum(open_arr[sl], close_arr[sl]) * spread_lo, low_arr[sl] * mult
    )
    vol_arr[sl] = np.maximum(
        vol_arr[sl] * np.exp(rng.normal(0.0, 0.40, size=shape)), 1.0
    )
    fr_arr[sl] = fr_arr[sl] + rng.normal(0.0, 0.002, size=shape)

    perturbed_id = hashlib.sha256(
        f"{dataset.dataset_id}:perturbed_{cut_index}_{seed}".encode("utf-8")
    ).hexdigest()
    return replace(
        dataset,
        dataset_id=perturbed_id,
        open=open_arr,
        high=high_arr,
        low=low_arr,
        close=close_arr,
        volume=vol_arr,
        funding_rate=fr_arr,
    )


def _build_synthetic_prefix_snapshots(
    dataset: MarketDataset,
    *,
    start_index: int,
    stop_inclusive: int,
) -> tuple[PortfolioDecisionSnapshot, ...]:
    n_syms = dataset.n_symbols
    rc = protocol.risk_config()
    limits = PortfolioRiskLimits(
        max_gross=rc.max_gross,
        max_abs_weight=rc.max_abs_weight,
        max_turnover=rc.max_turnover,
        drawdown_start=rc.drawdown_start,
        drawdown_stop=rc.drawdown_stop,
        emergency_turnover_override=rc.emergency_turnover_override,
        fail_closed_tolerance=rc.fail_closed_tolerance,
    )
    snapshots: list[PortfolioDecisionSnapshot] = []
    zeros_f = (0.0,) * n_syms
    zeros_str = ("0",) * n_syms
    ones_f = (1.0,) * n_syms
    ts_ns = dataset.timestamps.astype("datetime64[ns]").astype(np.int64)
    feat_shape = dataset.feature_available.shape[2] if dataset.features.ndim == 3 else 0
    feat_avail = tuple((True,) * feat_shape for _ in range(n_syms))
    feat_stale = tuple((0.0,) * feat_shape for _ in range(n_syms))
    feat_rows = (
        tuple(tuple(0.0 for _ in range(feat_shape)) for _ in range(n_syms))
        if feat_shape > 0
        else tuple(() for _ in range(n_syms))
    )
    g_dim = dataset.global_features.shape[1] if dataset.global_features.ndim == 2 else 0
    g_feats = (0.0,) * g_dim
    g_avail = (True,) * g_dim

    for idx in range(start_index, stop_inclusive + 1):
        marks = tuple(float(dataset.close[idx, s]) for s in range(n_syms))
        account = PortfolioAccountSnapshot(
            cash=protocol.DEFAULT_INITIAL_CAPITAL,
            portfolio_value=protocol.DEFAULT_INITIAL_CAPITAL,
            peak_value=protocol.DEFAULT_INITIAL_CAPITAL,
            max_drawdown=0.0,
            exact_quantities=zeros_str,
            quantities=zeros_f,
            weights=zeros_f,
            mark_prices=marks,
            contract_multipliers=ones_f,
            turnover_total=0.0,
            total_cost=0.0,
            funding_pnl=0.0,
            borrow_cost=0.0,
            margin_used=0.0,
            maintenance_margin=0.0,
            maintenance_requirement=0.0,
            margin_deficit=0.0,
            insolvent=False,
            termination_reason=None,
        )
        snapshots.append(
            PortfolioDecisionSnapshot(
                index=idx,
                timestamp_ns=int(ts_ns[idx]),
                agent_stop_index=stop_inclusive + 1,
                remaining_decisions=(stop_inclusive - idx) + 1,
                symbols=dataset.symbols,
                feature_names=dataset.feature_names,
                global_feature_names=dataset.global_feature_names,
                features=feat_rows,
                feature_available=feat_avail,
                feature_staleness=feat_stale,
                global_features=g_feats,
                global_feature_available=g_avail,
                account=account,
                active_orders=(),
                risk=limits,
            )
        )
    return tuple(snapshots)


def _max_prefix_pack_diff(
    a: Mtf6SignalPack,
    b: Mtf6SignalPack,
    *,
    cut_index: int,
) -> float:
    sl = slice(0, cut_index + 1)
    arrays: tuple[tuple[np.ndarray, np.ndarray], ...] = (
        (a.core7_signals[sl], b.core7_signals[sl]),
        (a.orth6_signals[sl], b.orth6_signals[sl]),
        (a.w_aw_bidir_75_25[sl], b.w_aw_bidir_75_25[sl]),
        (a.c_weekly_iv[sl], b.c_weekly_iv[sl]),
        (a.vol_cushion_tail[sl], b.vol_cushion_tail[sl]),
        (a.regime_feats_41[sl], b.regime_feats_41[sl]),
        (a.opp_feats_6[sl], b.opp_feats_6[sl]),
        (a.gate_alpha_nonrl[sl], b.gate_alpha_nonrl[sl]),
        (a.gate_carry_nonrl[sl], b.gate_carry_nonrl[sl]),
    )
    max_diff = 0.0
    for arr_a, arr_b in arrays:
        if not (np.all(np.isfinite(arr_a)) and np.all(np.isfinite(arr_b))):
            return math.inf
        diff = float(np.max(np.abs(arr_a - arr_b), initial=0.0))
        if diff > max_diff:
            max_diff = diff
    return max_diff


def run_causality_probe(
    dataset: MarketDataset,
    *,
    cut_index: int,
    policy_mode: ShadowAnchorMode,
    start_index: int = 0,
    fit_stop_index: int = 0,
    weights: ShadowAnchorMlpWeights | None = None,
    rebalance_every_bars: int = protocol.DEFAULT_REBALANCE_EVERY_BARS,
    deadband: float = protocol.DEFAULT_DEADBAND,
    signal_lag_bars: int = 0,
    seed: int = 17,
    enforce_pre_fs1: bool = True,
) -> CausalityProbeResult:
    """Verify future bars ``[cut_index + 1:]`` cannot alter signals or proposals up to ``cut_index``."""
    if not 0 <= start_index <= cut_index < dataset.n_bars - 1:
        raise ValueError(
            "must satisfy 0 <= start_index <= cut_index < dataset.n_bars - 1"
        )
    if fit_stop_index > start_index:
        raise ValueError("fit_stop_index must not exceed start_index")

    pack_base = build_mtf6_signal_pack(dataset, enforce_pre_fs1=enforce_pre_fs1)
    perturbed = _perturb_future_market_rows(dataset, cut_index=cut_index, seed=seed)
    pack_pert = build_mtf6_signal_pack(perturbed, enforce_pre_fs1=enforce_pre_fs1)

    sig_diff = _max_prefix_pack_diff(pack_base, pack_pert, cut_index=cut_index)

    cfg = ShadowAnchorMtfConfig(
        mode=policy_mode,
        start_index=start_index,
        fit_stop_index=fit_stop_index,
        rebalance_every_bars=rebalance_every_bars,
        deadband=deadband,
        signal_lag_bars=signal_lag_bars,
    )
    pol_a = ShadowAnchorMtfPolicy(signal_pack=pack_base, config=cfg, weights=weights)
    pol_b = ShadowAnchorMtfPolicy(signal_pack=pack_pert, config=cfg, weights=weights)
    snaps = _build_synthetic_prefix_snapshots(
        dataset, start_index=start_index, stop_inclusive=cut_index
    )
    prop_diff = 0.0
    for snap in snaps:
        pa = np.asarray(pol_a.decide_portfolio(snap).target_weights, dtype=np.float64)
        pb = np.asarray(pol_b.decide_portfolio(snap).target_weights, dtype=np.float64)
        if not (np.all(np.isfinite(pa)) and np.all(np.isfinite(pb))):
            prop_diff = math.inf
            break
        diff = float(np.max(np.abs(pa - pb), initial=0.0))
        if diff > prop_diff:
            prop_diff = diff

    passed = sig_diff == 0.0 and prop_diff == 0.0
    return CausalityProbeResult(
        cut_index=cut_index,
        policy_mode=policy_mode,
        signal_max_abs_diff=sig_diff,
        proposal_max_abs_diff=prop_diff,
        passed=passed,
    )


def _check_ledger_reconciled(
    ledger: SharedCashPortfolioReplayLedgerEvidence | None,
) -> bool:
    if ledger is None:
        return False
    intervals = ledger.intervals.intervals
    if not intervals:
        return False
    for iv in intervals:
        vals = (
            iv.cash_before,
            iv.cash_after,
            iv.portfolio_value_before,
            iv.portfolio_value_after,
            iv.total_cost_before,
            iv.total_cost_after,
            iv.funding_pnl_before,
            iv.funding_pnl_after,
            iv.borrow_cost_before,
            iv.borrow_cost_after,
            iv.turnover_total_before,
            iv.turnover_total_after,
            iv.interval_cost,
            iv.interval_funding,
            iv.interval_net_return,
        )
        if not all(math.isfinite(v) for v in vals):
            return False
        if (
            abs((iv.total_cost_after - iv.total_cost_before) - iv.interval_cost)
            > _RECONCILE_TOLERANCE
        ):
            return False
        if (
            abs((iv.funding_pnl_after - iv.funding_pnl_before) - iv.interval_funding)
            > _RECONCILE_TOLERANCE
        ):
            return False
    return True


def _compute_annual_sharpe(returns: np.ndarray) -> float:
    arr = np.asarray(returns, dtype=np.float64)
    if arr.size < 2:
        return 0.0
    sd = float(np.std(arr, ddof=1))
    if sd <= 1e-12 or not math.isfinite(sd):
        return 0.0
    return float(np.mean(arr) / sd * math.sqrt(_PERIODS_PER_YEAR_4H))


def _arm_to_mode(arm: str) -> ShadowAnchorMode | None:
    mapping: dict[str, ShadowAnchorMode] = {
        "ANC_BIDIR_75_25": "anc_bidir_75_25",
        "ANC_REGIME_GATE_NONRL": "anc_regime_gate_nonrl",
        "CORE7_SPARSE_PPO": "core7_sparse_ppo",
        "ORTH6_CUSH_PPO": "orth6_cush_ppo",
    }
    return mapping.get(arm)


def _build_arm_policy(
    *,
    arm: str,
    signal_pack: Mtf6SignalPack,
    start_index: int,
    fit_stop_index: int,
    rebalance_every_bars: int,
    deadband: float,
    signal_lag_bars: int,
    weights: ShadowAnchorMlpWeights | None,
) -> PortfolioPolicy:
    if arm == "CASH":
        return CashHoldPortfolioPolicy()
    mode = _arm_to_mode(arm)
    if mode is None:
        raise ValueError(f"unsupported arm: {arm!r}")
    resolved_weights = (
        weights
        if weights is not None
        else (
            ShadowAnchorMlpWeights.zero_residual(fit_stop_index=fit_stop_index)
            if arm in protocol.RL_ARMS
            else None
        )
    )
    cfg = ShadowAnchorMtfConfig(
        mode=mode,
        start_index=start_index,
        fit_stop_index=fit_stop_index,
        rebalance_every_bars=rebalance_every_bars,
        deadband=deadband,
        signal_lag_bars=signal_lag_bars,
    )
    return ShadowAnchorMtfPolicy(
        signal_pack=signal_pack, config=cfg, weights=resolved_weights
    )


def _summarize_replay_cell(
    *,
    arm: str,
    scenario: str,
    result: SharedCashPortfolioReplayResult,
    initial_capital: float,
    causality_passed: bool,
) -> CellSummary:
    reasons: list[str] = []
    ledger = result.ledger_evidence
    reconciled = _check_ledger_reconciled(ledger)
    if not reconciled:
        reasons.append("ledger_not_reconciled")
    if not causality_passed:
        reasons.append("causality_probe_failed")

    terminal_flat = all(abs(float(q)) <= 1e-12 for q in result.book.quantities)
    if ledger is not None:
        terminal_flat = terminal_flat and all(
            abs(float(q)) <= 1e-12 for q in ledger.intervals.terminal_exact_quantities
        )
    if not terminal_flat:
        reasons.append("terminal_position_not_flat")

    term_reasons = result.diagnostics.termination_reasons
    stop_reason = term_reasons[0] if term_reasons else None
    ret_arr = np.asarray(result.returns.values, dtype=np.float64)
    total_return = float(result.book.portfolio_value / initial_capital - 1.0)
    complete = bool(ret_arr.size > 0 and stop_reason is None)
    execution_valid = bool(not result.book.insolvent and reconciled)

    return CellSummary(
        arm=arm,
        scenario=scenario,
        total_return=total_return,
        annual_sharpe=_compute_annual_sharpe(ret_arr),
        max_drawdown=float(result.book.max_drawdown),
        total_cost=float(result.diagnostics.total_cost),
        funding_pnl=float(result.diagnostics.funding_pnl),
        turnover_total=float(result.diagnostics.turnover_total),
        complete=complete,
        terminal_flat=terminal_flat,
        execution_valid=execution_valid,
        causality_passed=causality_passed,
        ledger_reconciled=reconciled,
        stop_reason=stop_reason,
        invalid_reasons=tuple(reasons),
    )


def _write_json_canonical(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(dict(payload)))


def run_study(
    dataset: MarketDataset,
    *,
    output_dir: Path,
    authorization_url: str | None = None,
    allow_synthetic_unauthenticated: bool = False,
    require_full_roster: bool | None = None,
    start_index: int = 0,
    stop_index: int | None = None,
    fit_stop_index: int = 0,
    weights_by_arm: Mapping[str, ShadowAnchorMlpWeights] | None = None,
    scenarios: Sequence[str] = protocol.SCENARIOS,
    causality_cut_index: int | None = None,
    rebalance_every_bars: int = protocol.DEFAULT_REBALANCE_EVERY_BARS,
    deadband: float = protocol.DEFAULT_DEADBAND,
    initial_capital: float = protocol.DEFAULT_INITIAL_CAPITAL,
    enforce_pre_fs1: bool = True,
) -> dict[str, object]:
    """Run the 5-arm Core7 MTF study exclusively via ``run_shared_portfolio_replay``."""
    resolved_auth_url: str | None = None
    if allow_synthetic_unauthenticated:
        if authorization_url is not None:
            resolved_auth_url = protocol.validate_authorization_url(authorization_url)
    else:
        if authorization_url is None:
            raise ValueError(
                "authorization_url is required unless allow_synthetic_unauthenticated=True"
            )
        resolved_auth_url = protocol.validate_authorization_url(authorization_url)

    if (output_dir / "decision.json").exists() or (
        output_dir / "study-manifest.json"
    ).exists():
        raise FileExistsError(
            f"refusing to overwrite existing study output in {output_dir}"
        )

    pair_names = validate_alternating_spot_perp_dataset(dataset)
    must_complete_roster = (
        (not allow_synthetic_unauthenticated)
        if require_full_roster is None
        else require_full_roster
    )
    roster_complete = pair_names == protocol.ROSTER
    if must_complete_roster and not roster_complete:
        raise ValueError(
            f"dataset pair roster {pair_names!r} does not match complete 11-symbol ROSTER"
        )

    if enforce_pre_fs1:
        enforce_pre_fs1_boundary(dataset)

    resolved_stop = dataset.n_bars - 1 if stop_index is None else stop_index
    if not 0 <= fit_stop_index <= start_index < resolved_stop < dataset.n_bars:
        raise ValueError(
            "indices must satisfy 0 <= fit_stop_index <= start_index < stop_index < dataset.n_bars"
        )
    if resolved_stop - start_index < 3:
        raise ValueError("study replay window requires stop_index - start_index >= 3")

    resolved_scenarios = tuple(scenarios)
    if not resolved_scenarios or len(set(resolved_scenarios)) != len(
        resolved_scenarios
    ):
        raise ValueError("scenarios must be a non-empty sequence without duplicates")
    for sc in resolved_scenarios:
        protocol.scenario_cost(sc)

    signal_pack = build_mtf6_signal_pack(dataset, enforce_pre_fs1=enforce_pre_fs1)
    cut_idx = (
        causality_cut_index
        if causality_cut_index is not None
        else start_index + max(0, (resolved_stop - start_index) // 2)
    )
    risk = PreTradeRisk(protocol.risk_config())

    probe_results: dict[str, CausalityProbeResult] = {}
    for arm in protocol.ARMS:
        mode = _arm_to_mode(arm)
        if mode is None:
            probe_results[arm] = CausalityProbeResult(
                cut_index=cut_idx,
                policy_mode="CASH",
                signal_max_abs_diff=0.0,
                proposal_max_abs_diff=0.0,
                passed=True,
            )
            continue
        arm_weights = None if weights_by_arm is None else weights_by_arm.get(arm)
        if arm_weights is None and arm in protocol.RL_ARMS:
            arm_weights = ShadowAnchorMlpWeights.zero_residual(
                fit_stop_index=fit_stop_index
            )
        probe_results[arm] = run_causality_probe(
            dataset,
            cut_index=cut_idx,
            policy_mode=mode,
            start_index=start_index,
            fit_stop_index=fit_stop_index,
            weights=arm_weights,
            rebalance_every_bars=rebalance_every_bars,
            deadband=deadband,
            enforce_pre_fs1=enforce_pre_fs1,
        )

    cells: dict[tuple[str, str], CellSummary] = {}
    for arm in protocol.ARMS:
        arm_weights = None if weights_by_arm is None else weights_by_arm.get(arm)
        causality_ok = probe_results[arm].passed
        for sc in resolved_scenarios:
            cost_cfg = protocol.scenario_cost(sc)
            delay_bars = protocol.scenario_signal_delay_bars(sc)
            policy = _build_arm_policy(
                arm=arm,
                signal_pack=signal_pack,
                start_index=start_index,
                fit_stop_index=fit_stop_index,
                rebalance_every_bars=rebalance_every_bars,
                deadband=deadband,
                signal_lag_bars=delay_bars,
                weights=arm_weights,
            )
            replay_result = run_shared_portfolio_replay(
                dataset,
                policy,
                start_index=start_index,
                stop_index=resolved_stop,
                initial_capital=initial_capital,
                execution_cost=cost_cfg,
                risk=risk,
                settle_terminal_position=True,
                capture_ledger_evidence=True,
            )
            cells[(arm, sc)] = _summarize_replay_cell(
                arm=arm,
                scenario=sc,
                result=replay_result,
                initial_capital=initial_capital,
                causality_passed=causality_ok,
            )

    decision_payload = decision.decide(cells, scenarios=resolved_scenarios)
    weights_digests = {
        arm: (
            ShadowAnchorMlpWeights.zero_residual(
                fit_stop_index=fit_stop_index
            ).weights_digest()
            if (weights_by_arm is None or arm not in weights_by_arm)
            else weights_by_arm[arm].weights_digest()
        )
        for arm in protocol.RL_ARMS
    }

    manifest: dict[str, object] = {
        "protocol_id": protocol.PROTOCOL_ID,
        "protocol_digest": protocol.protocol_digest(),
        "dataset_id": dataset.dataset_id,
        "pair_names": list(pair_names),
        "roster_complete": roster_complete,
        "start_index": start_index,
        "stop_index": resolved_stop,
        "fit_stop_index": fit_stop_index,
        "causality_cut_index": cut_idx,
        "scenarios": list(resolved_scenarios),
        "authorization_url": resolved_auth_url,
        "synthetic_only": allow_synthetic_unauthenticated,
        "enforce_pre_fs1": enforce_pre_fs1,
        "weights_digests": weights_digests,
        "causality_probes": {arm: res.to_dict() for arm, res in probe_results.items()},
    }
    full_decision: dict[str, object] = {
        **decision_payload,
        "protocol_id": protocol.PROTOCOL_ID,
        "protocol_digest": protocol.protocol_digest(),
        "dataset_id": dataset.dataset_id,
        "synthetic_only": allow_synthetic_unauthenticated,
        "authorization_url": resolved_auth_url,
        "roster_complete": roster_complete,
        "cells": {
            f"{arm}__{sc}": cell.to_dict() for (arm, sc), cell in sorted(cells.items())
        },
    }

    _write_json_canonical(output_dir / "study-manifest.json", manifest)
    _write_json_canonical(output_dir / "decision.json", full_decision)
    return full_decision


__all__ = [
    "CausalityProbeResult",
    "run_causality_probe",
    "run_study",
]
