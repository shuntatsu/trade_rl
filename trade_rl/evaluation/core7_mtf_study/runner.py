"""Thin study runner orchestrating Core7 MTF replay through ``run_shared_portfolio_replay``."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, fields, replace
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np

from trade_rl.data.features.mtf6_signals import (
    Mtf6SignalPack,
    build_mtf6_signal_pack,
    enforce_pre_fs1_boundary,
    validate_alternating_spot_perp_dataset,
)
from trade_rl.data.market import MarketDataset
from trade_rl.evaluation.core7_mtf_study import decision, protocol
from trade_rl.evaluation.core7_mtf_study.decision import CellSummary
from trade_rl.evaluation.experiments.store import StudyStore
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
_RECONCILE_ABS_TOLERANCE = 1e-6
_RECONCILE_REL_TOLERANCE = 1e-10
_HOLD_MISMATCH_DIFF = 1.0
_PROBE_ACCOUNT_SEED = 20_260_101
# every per-bar signal array of the pack (the clocks and names are not row data)
_PACK_ROW_FIELDS: tuple[str, ...] = tuple(
    f.name for f in fields(Mtf6SignalPack) if f.name not in ("pair_names", "timestamps")
)

PolicyFactory = Callable[
    [Mtf6SignalPack, ShadowAnchorMtfConfig, ShadowAnchorMlpWeights | None],
    PortfolioPolicy,
]


@dataclass(frozen=True, slots=True)
class CausalityProbeResult:
    """Result of future-bar perturbation causality probe at ``cut_index``."""

    cut_index: int
    policy_mode: str
    signal_max_abs_diff: float
    proposal_max_abs_diff: float
    passed: bool
    signal_lag_bars: int = 0
    decisions_probed: int = 0

    def to_dict(self) -> dict[str, object]:
        return {
            "cut_index": self.cut_index,
            "policy_mode": self.policy_mode,
            "signal_max_abs_diff": _json_safe(self.signal_max_abs_diff),
            "proposal_max_abs_diff": _json_safe(self.proposal_max_abs_diff),
            "passed": self.passed,
            "signal_lag_bars": self.signal_lag_bars,
            "decisions_probed": self.decisions_probed,
        }


def _json_safe(value: float) -> float:
    return value if math.isfinite(value) else 1.0e308


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
    # The perturbed copy is a payload-free view with its own derived identity: keeping
    # the source ``identity_payload_json`` would be inconsistent with the new arrays and
    # is rejected by ``MarketDataset``. The input dataset (and its identity) is untouched.
    return replace(
        dataset,
        dataset_id=perturbed_id,
        identity_payload_json=None,
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
    """Build decision snapshots along a deterministic, non-trivial account path.

    Weights random-walk and equity oscillates through the 10% ``drawdown_start`` so the
    HOLD, deadband, drawdown-release and cached-target branches of the policy all see
    both sides of their thresholds. The path never depends on the future market rows.
    """
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

    n_snaps = stop_inclusive - start_index + 1
    period = max(8, min(48, n_snaps // 2))
    rng = np.random.default_rng(_PROBE_ACCOUNT_SEED)
    weights = np.zeros(n_syms, dtype=np.float64)
    peak = protocol.DEFAULT_INITIAL_CAPITAL
    max_dd = 0.0
    for step, idx in enumerate(range(start_index, stop_inclusive + 1)):
        weights = np.clip(
            0.85 * weights + rng.normal(0.0, 0.12, size=n_syms), -0.5, 0.5
        )
        phase = 0.5 - 0.5 * math.cos(2.0 * math.pi * step / period)
        value = protocol.DEFAULT_INITIAL_CAPITAL * math.exp(-0.14 * phase)
        peak = max(peak, value)
        max_dd = max(max_dd, 1.0 - value / peak)
        marks = np.asarray(dataset.close[idx], dtype=np.float64)
        quantities = weights * value / marks
        cash = value - float(np.sum(quantities * marks))
        account = PortfolioAccountSnapshot(
            cash=cash,
            portfolio_value=value,
            peak_value=peak,
            max_drawdown=max_dd,
            exact_quantities=tuple(repr(float(q)) for q in quantities),
            quantities=tuple(float(q) for q in quantities),
            weights=tuple(float(w) for w in weights),
            mark_prices=tuple(float(m) for m in marks),
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
    """Largest absolute difference over rows ``[0, cut_index]`` of every pack array."""
    sl = slice(0, cut_index + 1)
    max_diff = 0.0
    for name in _PACK_ROW_FIELDS:
        arr_a = np.asarray(getattr(a, name)[sl], dtype=np.float64)
        arr_b = np.asarray(getattr(b, name)[sl], dtype=np.float64)
        if not (np.all(np.isfinite(arr_a)) and np.all(np.isfinite(arr_b))):
            return math.inf
        diff = float(np.max(np.abs(arr_a - arr_b), initial=0.0))
        if diff > max_diff:
            max_diff = diff
    return max_diff


_Decision = tuple[tuple[float, ...], tuple[int, ...]]


def _run_decisions(
    policy: PortfolioPolicy,
    snapshots: Sequence[PortfolioDecisionSnapshot],
    *,
    before_step: Callable[[int], None] | None = None,
) -> list[_Decision]:
    out: list[_Decision] = []
    for snap in snapshots:
        if before_step is not None:
            before_step(snap.index)
        proposal = policy.decide_portfolio(snap)
        out.append(
            (
                tuple(float(w) for w in proposal.target_weights),
                tuple(proposal.hold_current_quantity_symbols),
            )
        )
    return out


def _decision_diff(a: Sequence[_Decision], b: Sequence[_Decision]) -> float:
    worst = 0.0
    for (wa, ha), (wb, hb) in zip(a, b, strict=True):
        arr_a = np.asarray(wa, dtype=np.float64)
        arr_b = np.asarray(wb, dtype=np.float64)
        if not (np.all(np.isfinite(arr_a)) and np.all(np.isfinite(arr_b))):
            return math.inf
        diff = float(np.max(np.abs(arr_a - arr_b), initial=0.0))
        if ha != hb:
            diff = max(diff, _HOLD_MISMATCH_DIFF)
        worst = max(worst, diff)
    return worst


class _HybridPack:
    """Signal pack that is *base* up to the current decision row and *perturbed* after.

    ``future`` is built from a dataset perturbed from ``start_index + 1`` on. Before each
    decision at row ``i`` the rows ``[0, i]`` are restored to the base values in place, so
    only rows after ``i`` differ. A policy that reads any row after ``i`` therefore sees
    different numbers than the base policy for that exact decision.
    """

    def __init__(self, base: Mtf6SignalPack, future: Mtf6SignalPack) -> None:
        self._base = base
        self._arrays = {
            name: np.array(getattr(future, name), copy=True)
            for name in _PACK_ROW_FIELDS
        }
        self._restored = -1
        row_arrays: dict[str, Any] = dict(self._arrays)
        self.pack = replace(base, **row_arrays)

    def restore_through(self, index: int) -> None:
        lo = self._restored + 1
        if index < lo:
            return
        for name, arr in self._arrays.items():
            arr[lo : index + 1] = getattr(self._base, name)[lo : index + 1]
        self._restored = index


def _default_policy_factory(
    pack: Mtf6SignalPack,
    config: ShadowAnchorMtfConfig,
    weights: ShadowAnchorMlpWeights | None,
) -> PortfolioPolicy:
    return ShadowAnchorMtfPolicy(signal_pack=pack, config=config, weights=weights)


def _probe_policy(
    *,
    pack_base: Mtf6SignalPack,
    pack_cut: Mtf6SignalPack,
    pack_future: Mtf6SignalPack,
    snapshots: Sequence[PortfolioDecisionSnapshot],
    cut_index: int,
    config: ShadowAnchorMtfConfig,
    weights: ShadowAnchorMlpWeights | None,
    policy_factory: PolicyFactory | None,
) -> float:
    """Policy-level causality: proposals and HOLD tuples must ignore all later rows.

    * ``pack_cut`` (perturbed after ``cut_index``): decisions up to the cut must match.
    * ``pack_future`` (perturbed after ``start_index``) restored row-by-row: *every*
      decision, rebalance or not, must match the base run, so a lookahead of even one
      row at any bar is caught regardless of where the cut lands.
    """
    factory = policy_factory or _default_policy_factory
    base = _run_decisions(factory(pack_base, config, weights), snapshots)
    n_cut = sum(1 for snap in snapshots if snap.index <= cut_index)
    at_cut = _run_decisions(factory(pack_cut, config, weights), snapshots[:n_cut])
    hybrid = _HybridPack(pack_base, pack_future)
    every = _run_decisions(
        factory(hybrid.pack, config, weights),
        snapshots,
        before_step=hybrid.restore_through,
    )
    return max(_decision_diff(base[:n_cut], at_cut), _decision_diff(base, every))


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
    policy_factory: PolicyFactory | None = None,
) -> CausalityProbeResult:
    """Verify future bars cannot alter signals or proposals at any decision up to the cut.

    Signals are compared on every pack array up to ``cut_index``. Proposals (target weights
    and HOLD tuples) are compared at *every* decision in ``[start_index, cut_index]`` on a
    non-trivial account path, both against a dataset perturbed after the cut and against
    a pack whose rows after each decision are perturbed.
    """
    if not 0 <= start_index <= cut_index < dataset.n_bars - 1:
        raise ValueError(
            "must satisfy 0 <= start_index <= cut_index < dataset.n_bars - 1"
        )
    if fit_stop_index > start_index:
        raise ValueError("fit_stop_index must not exceed start_index")

    pack_base = build_mtf6_signal_pack(dataset, enforce_pre_fs1=True)
    pack_cut = build_mtf6_signal_pack(
        _perturb_future_market_rows(dataset, cut_index=cut_index, seed=seed),
        enforce_pre_fs1=True,
    )
    pack_future = build_mtf6_signal_pack(
        _perturb_future_market_rows(dataset, cut_index=start_index, seed=seed + 1),
        enforce_pre_fs1=True,
    )
    sig_diff = max(
        _max_prefix_pack_diff(pack_base, pack_cut, cut_index=cut_index),
        _max_prefix_pack_diff(pack_base, pack_future, cut_index=start_index),
    )
    snaps = _build_synthetic_prefix_snapshots(
        dataset, start_index=start_index, stop_inclusive=cut_index
    )
    cfg = ShadowAnchorMtfConfig(
        mode=policy_mode,
        start_index=start_index,
        fit_stop_index=fit_stop_index,
        rebalance_every_bars=rebalance_every_bars,
        deadband=deadband,
        signal_lag_bars=signal_lag_bars,
    )
    prop_diff = _probe_policy(
        pack_base=pack_base,
        pack_cut=pack_cut,
        pack_future=pack_future,
        snapshots=snaps,
        cut_index=cut_index,
        config=cfg,
        weights=weights,
        policy_factory=policy_factory,
    )
    return CausalityProbeResult(
        cut_index=cut_index,
        policy_mode=policy_mode,
        signal_max_abs_diff=sig_diff,
        proposal_max_abs_diff=prop_diff,
        passed=sig_diff == 0.0 and prop_diff == 0.0,
        signal_lag_bars=signal_lag_bars,
        decisions_probed=len(snaps),
    )


def _close(actual: float, expected: float, *, scale: float = 1.0) -> bool:
    tolerance = _RECONCILE_ABS_TOLERANCE + _RECONCILE_REL_TOLERANCE * max(
        abs(expected), abs(scale)
    )
    return math.isfinite(actual) and abs(actual - expected) <= tolerance


def _check_ledger_reconciled(
    ledger: SharedCashPortfolioReplayLedgerEvidence | None,
    dataset: MarketDataset,
    initial_capital: float,
) -> bool:
    """Rebuild every interval's cash, quantities, equity and turnover from raw fills.

    The check never trusts the book's own totals: each interval's cash is
    ``cash_before - sum(q * price * multiplier) - cost + funding - borrow + dividend +
    interest`` from the raw order events, quantities are ``before + fills``, equity is
    cash plus quantities at the next mark, and turnover is filled notional over the
    rebuilt start equity. Intervals must chain exactly, start from the initial capital
    flat, and the terminal aggregates must equal the last interval.
    """
    if ledger is None:
        return False
    led = ledger.intervals
    intervals = led.intervals
    if not intervals:
        return False
    multipliers = np.asarray(dataset.resolved_array("contract_multipliers"))
    marks = dataset.resolved_array("mark_price")
    n_syms = dataset.n_symbols

    first = intervals[0]
    if (
        first.start_index != led.start_index
        or not _close(first.cash_before, initial_capital)
        or not _close(first.portfolio_value_before, initial_capital)
        or any(Fraction(q) != 0 for q in first.exact_quantities_before)
        or first.total_cost_before != 0.0
        or first.funding_pnl_before != 0.0
        or first.borrow_cost_before != 0.0
        or first.turnover_total_before != 0.0
    ):
        return False

    previous = None
    cost_sum = 0.0
    for iv in intervals:
        scalars = (
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
            iv.interval_borrow_cost,
            iv.interval_dividend,
            iv.interval_cash_interest,
            iv.interval_net_return,
        )
        if not all(math.isfinite(v) for v in scalars):
            return False
        if previous is not None and (
            iv.start_index != previous.next_index
            or iv.exact_quantities_before != previous.exact_quantities_after
            or not _close(iv.cash_before, previous.cash_after)
            or not _close(iv.portfolio_value_before, previous.portfolio_value_after)
            or not _close(iv.total_cost_before, previous.total_cost_after)
            or not _close(iv.funding_pnl_before, previous.funding_pnl_after)
            or not _close(iv.borrow_cost_before, previous.borrow_cost_after)
            or not _close(iv.turnover_total_before, previous.turnover_total_after)
        ):
            return False
        if len(iv.exact_quantities_before) != n_syms:
            return False

        quantities = [Fraction(q) for q in iv.exact_quantities_before]
        marks_start = marks[iv.start_index]
        equity_before = float(iv.cash_before) + sum(
            float(quantities[s]) * float(marks_start[s]) * float(multipliers[s])
            for s in range(n_syms)
        )
        signed_fill_cash = 0.0
        filled_notional = 0.0
        for event in iv.order_events:
            if event.filled_quantity == 0.0:
                continue
            if event.execution_price is None:
                return False
            s = event.symbol_index
            quantities[s] += Fraction(str(event.filled_quantity))
            notional = (
                float(event.filled_quantity)
                * float(event.execution_price)
                * float(multipliers[s])
            )
            signed_fill_cash -= notional
            filled_notional += abs(notional)
        after = [Fraction(q) for q in iv.exact_quantities_after]
        if len(after) != n_syms or any(
            not _close(float(a), float(b))
            for a, b in zip(after, quantities, strict=True)
        ):
            return False

        expected_cash = (
            iv.cash_before
            + signed_fill_cash
            - iv.interval_cost
            + iv.interval_funding
            - iv.interval_borrow_cost
            + iv.interval_dividend
            + iv.interval_cash_interest
        )
        marks_next = marks[iv.next_index]
        expected_equity = float(iv.cash_after) + sum(
            float(after[s]) * float(marks_next[s]) * float(multipliers[s])
            for s in range(n_syms)
        )
        expected_turnover = filled_notional / max(equity_before, 1e-12)
        if (
            not _close(iv.cash_after, expected_cash, scale=initial_capital)
            or not _close(iv.portfolio_value_before, equity_before)
            or not _close(iv.portfolio_value_after, expected_equity)
            or not _close(iv.total_cost_after - iv.total_cost_before, iv.interval_cost)
            or not _close(
                iv.funding_pnl_after - iv.funding_pnl_before, iv.interval_funding
            )
            or not _close(
                iv.borrow_cost_after - iv.borrow_cost_before, iv.interval_borrow_cost
            )
            or not _close(
                iv.turnover_total_after - iv.turnover_total_before, expected_turnover
            )
        ):
            return False
        cost_sum += iv.interval_cost
        previous = iv

    assert previous is not None
    return bool(
        led.terminal_exact_quantities == previous.exact_quantities_after
        and _close(led.final_cash, previous.cash_after)
        and _close(led.final_portfolio_value, previous.portfolio_value_after)
        and _close(led.final_total_cost, previous.total_cost_after)
        and _close(led.final_total_cost, cost_sum)
        and _close(led.final_funding_pnl, previous.funding_pnl_after)
        and _close(led.final_borrow_cost, previous.borrow_cost_after)
        and _close(led.final_turnover_total, previous.turnover_total_after)
    )


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
    dataset: MarketDataset,
    initial_capital: float,
    causality_passed: bool,
) -> CellSummary:
    reasons: list[str] = []
    ledger = result.ledger_evidence
    reconciled = _check_ledger_reconciled(ledger, dataset, initial_capital)
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
    max_drawdown = float(result.book.max_drawdown)
    if stop_reason is None and max_drawdown >= protocol.risk_config().drawdown_stop:
        stop_reason = "max_drawdown_hard_stop"
    ret_arr = np.asarray(result.returns.values, dtype=np.float64)
    total_return = float(result.book.portfolio_value / initial_capital - 1.0)
    # a risk/economic stop is a STOP, not an integrity failure: the replay still ran
    complete = bool(ret_arr.size > 0)
    execution_valid = bool(not result.book.insolvent and reconciled)

    return CellSummary(
        arm=arm,
        scenario=scenario,
        total_return=total_return,
        annual_sharpe=_compute_annual_sharpe(ret_arr),
        max_drawdown=max_drawdown,
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
    """Publish once via the canonical Study owner, never follow output symlinks."""
    if ".." in path.parts:
        raise ValueError("study output path must not contain parent traversal")
    root = path.parent.absolute()
    # StudyStore's descriptor-bound mutation lock protects POSIX ancestor
    # swaps. Reject static ancestor links before its mkdir can create any
    # directories through them.
    if any(
        ancestor.is_symlink() or ancestor.is_junction()
        for ancestor in (root, *root.parents)
    ):
        raise ValueError("study output path must not contain symlinked directories")
    study = StudyStore(root)
    with study.mutation_lock():
        study.publish_json_once(path.name, dict(payload))


@dataclass(frozen=True, slots=True)
class _RunAuthorization:
    url: str | None
    receipt: protocol.AuthorizationReceipt | None
    synthetic_only: bool
    authenticated: bool
    must_complete_roster: bool

    @property
    def check(self) -> str:
        if self.url is None:
            return "none_synthetic_only"
        return "receipt_shape_and_binding" if self.authenticated else "url_format_only"

    @property
    def evidence_eligible(self) -> bool:
        return self.authenticated and not self.synthetic_only

    @property
    def non_evidence_label(self) -> str:
        return (
            decision.SYNTHETIC_ONLY_NOT_EVIDENCE
            if self.synthetic_only
            else decision.UNAUTHENTICATED_NOT_EVIDENCE
        )


def _authorize_run(
    dataset: MarketDataset,
    *,
    expected_protocol_digest: str,
    authorization_url: str | None,
    authorization_receipt: protocol.AuthorizationReceipt | None,
    allow_synthetic_unauthenticated: bool,
    require_full_roster: bool | None,
    scenarios: tuple[str, ...],
    enforce_pre_fs1: bool,
) -> _RunAuthorization:
    """Fail-closed run authorization.

    ``trade_rl`` is network-free, so the owner URL is only *format* checked here. A run
    is evidence-eligible only when the caller also passes an ``AuthorizationReceipt``
    (built by the tools layer after fetching the comment) whose URL equals
    ``authorization_url`` and whose quoted digest equals the frozen protocol digest; the
    runner validates that shape and binding but cannot prove the comment is genuine.
    A URL without a receipt runs as ``UNAUTHENTICATED_NOT_EVIDENCE``; a URL-less run needs
    ``allow_synthetic_unauthenticated`` plus the synthetic marker and runs as
    ``SYNTHETIC_ONLY_NOT_EVIDENCE``. The Pre-FS1 embargo is never waivable.
    """
    if expected_protocol_digest != protocol.protocol_digest():
        raise ValueError(
            "expected protocol digest does not match the frozen protocol digest "
            f"{protocol.protocol_digest()}"
        )
    if not enforce_pre_fs1:
        raise ValueError(
            "run_study always enforces the Pre-FS1 embargo; enforce_pre_fs1=False is refused"
        )
    if authorization_url is None and authorization_receipt is not None:
        raise ValueError("authorization_receipt requires authorization_url")
    if authorization_url is not None:
        resolved_url = protocol.validate_authorization_url(authorization_url)
        if scenarios != protocol.SCENARIOS:
            raise ValueError(
                f"scenarios must equal the frozen protocol scenarios {protocol.SCENARIOS!r}"
            )
        receipt = None
        if authorization_receipt is not None:
            receipt = protocol.validate_authorization_receipt(
                authorization_receipt, authorization_url=resolved_url
            )
        return _RunAuthorization(
            url=resolved_url,
            receipt=receipt,
            synthetic_only=False,
            authenticated=receipt is not None,
            must_complete_roster=True,
        )
    if not allow_synthetic_unauthenticated:
        raise ValueError(
            "authorization_url is required unless allow_synthetic_unauthenticated=True"
        )
    if not protocol.is_synthetic_dataset(dataset):
        raise ValueError(
            "a URL-less run requires a synthetic-marked dataset "
            "(protocol.mark_synthetic_dataset)"
        )
    return _RunAuthorization(
        url=None,
        receipt=None,
        synthetic_only=True,
        authenticated=False,
        must_complete_roster=bool(require_full_roster),
    )


def run_study(
    dataset: MarketDataset,
    *,
    output_dir: Path,
    expected_protocol_digest: str,
    authorization_url: str | None = None,
    authorization_receipt: protocol.AuthorizationReceipt | None = None,
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
    """Run the 5-arm Core7 MTF study exclusively via ``run_shared_portfolio_replay``.

    Authorization is fail-closed. ``expected_protocol_digest`` must equal the frozen
    protocol digest and the Pre-FS1 embargo is always enforced (``enforce_pre_fs1`` is
    kept for call compatibility and must be ``True``). With the owner G4 URL the run needs
    the complete roster and exactly ``protocol.SCENARIOS`` and yields the normal labels.
    Without a URL only a synthetic-marked dataset may run, and the outcome is then
    ``SYNTHETIC_ONLY_NOT_EVIDENCE`` (``INVALID`` stays ``INVALID``), never parity/stop
    evidence.
    """
    resolved_scenarios = tuple(scenarios)
    auth = _authorize_run(
        dataset,
        expected_protocol_digest=expected_protocol_digest,
        authorization_url=authorization_url,
        authorization_receipt=authorization_receipt,
        allow_synthetic_unauthenticated=allow_synthetic_unauthenticated,
        require_full_roster=require_full_roster,
        scenarios=resolved_scenarios,
        enforce_pre_fs1=enforce_pre_fs1,
    )

    if any(
        path.exists() or path.is_symlink()
        for path in (output_dir / "decision.json", output_dir / "study-manifest.json")
    ):
        raise FileExistsError(
            f"refusing to overwrite existing study output in {output_dir}"
        )

    pair_names = validate_alternating_spot_perp_dataset(dataset)
    roster_complete = pair_names == protocol.ROSTER
    if auth.must_complete_roster and not roster_complete:
        raise ValueError(
            f"dataset pair roster {pair_names!r} does not match complete 11-symbol ROSTER"
        )

    enforce_pre_fs1_boundary(dataset)

    resolved_stop = dataset.n_bars - 1 if stop_index is None else stop_index
    if not 0 <= fit_stop_index <= start_index < resolved_stop < dataset.n_bars:
        raise ValueError(
            "indices must satisfy 0 <= fit_stop_index <= start_index < stop_index < dataset.n_bars"
        )
    if resolved_stop - start_index < 3:
        raise ValueError("study replay window requires stop_index - start_index >= 3")

    if not resolved_scenarios or len(set(resolved_scenarios)) != len(
        resolved_scenarios
    ):
        raise ValueError("scenarios must be a non-empty sequence without duplicates")
    for sc in resolved_scenarios:
        protocol.scenario_cost(sc)

    signal_pack = build_mtf6_signal_pack(dataset, enforce_pre_fs1=True)
    cut_idx = (
        causality_cut_index
        if causality_cut_index is not None
        else start_index + max(0, (resolved_stop - start_index) // 2)
    )
    if not start_index <= cut_idx < dataset.n_bars - 1:
        raise ValueError("causality_cut_index must lie in [start_index, n_bars - 1)")
    risk = PreTradeRisk(protocol.risk_config())

    # Packs perturbed after the cut and after the start are built once and shared by
    # every (arm, signal lag) probe; signals are compared on all pack arrays.
    pack_cut = build_mtf6_signal_pack(
        _perturb_future_market_rows(dataset, cut_index=cut_idx, seed=17),
        enforce_pre_fs1=True,
    )
    pack_future = build_mtf6_signal_pack(
        _perturb_future_market_rows(dataset, cut_index=start_index, seed=18),
        enforce_pre_fs1=True,
    )
    signal_diff = max(
        _max_prefix_pack_diff(signal_pack, pack_cut, cut_index=cut_idx),
        _max_prefix_pack_diff(signal_pack, pack_future, cut_index=start_index),
    )
    probe_snapshots = _build_synthetic_prefix_snapshots(
        dataset, start_index=start_index, stop_inclusive=resolved_stop
    )

    arm_weights_resolved: dict[str, ShadowAnchorMlpWeights | None] = {}
    for arm in protocol.ARMS:
        w = None if weights_by_arm is None else weights_by_arm.get(arm)
        if w is None and arm in protocol.RL_ARMS:
            w = ShadowAnchorMlpWeights.zero_residual(fit_stop_index=fit_stop_index)
        arm_weights_resolved[arm] = w

    delays = sorted(
        {protocol.scenario_signal_delay_bars(sc) for sc in resolved_scenarios}
    )
    probe_by_arm_lag: dict[tuple[str, int], CausalityProbeResult] = {}
    for arm in protocol.ARMS:
        mode = _arm_to_mode(arm)
        for lag in delays:
            if mode is None:
                probe_by_arm_lag[(arm, lag)] = CausalityProbeResult(
                    cut_index=cut_idx,
                    policy_mode="CASH",
                    signal_max_abs_diff=signal_diff,
                    proposal_max_abs_diff=0.0,
                    passed=signal_diff == 0.0,
                    signal_lag_bars=lag,
                    decisions_probed=0,
                )
                continue
            cfg = ShadowAnchorMtfConfig(
                mode=mode,
                start_index=start_index,
                fit_stop_index=fit_stop_index,
                rebalance_every_bars=rebalance_every_bars,
                deadband=deadband,
                signal_lag_bars=lag,
            )
            prop_diff = _probe_policy(
                pack_base=signal_pack,
                pack_cut=pack_cut,
                pack_future=pack_future,
                snapshots=probe_snapshots,
                cut_index=cut_idx,
                config=cfg,
                weights=arm_weights_resolved[arm],
                policy_factory=None,
            )
            probe_by_arm_lag[(arm, lag)] = CausalityProbeResult(
                cut_index=cut_idx,
                policy_mode=mode,
                signal_max_abs_diff=signal_diff,
                proposal_max_abs_diff=prop_diff,
                passed=signal_diff == 0.0 and prop_diff == 0.0,
                signal_lag_bars=lag,
                decisions_probed=len(probe_snapshots),
            )

    cells: dict[tuple[str, str], CellSummary] = {}
    probe_results: dict[str, CausalityProbeResult] = {}
    for arm in protocol.ARMS:
        arm_weights = None if weights_by_arm is None else weights_by_arm.get(arm)
        for sc in resolved_scenarios:
            cost_cfg = protocol.scenario_cost(sc)
            delay_bars = protocol.scenario_signal_delay_bars(sc)
            probe = probe_by_arm_lag[(arm, delay_bars)]
            probe_results[f"{arm}__{sc}"] = probe
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
                dataset=dataset,
                initial_capital=initial_capital,
                causality_passed=probe.passed,
            )

    decision_payload = decision.decide(
        cells,
        scenarios=resolved_scenarios,
        evidence_eligible=auth.evidence_eligible,
        non_evidence_label=auth.non_evidence_label,
    )
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
    weights_are_zero_residual = {
        arm: weights_by_arm is None or arm not in weights_by_arm
        for arm in protocol.RL_ARMS
    }

    manifest: dict[str, object] = {
        "protocol_id": protocol.PROTOCOL_ID,
        "protocol_digest": protocol.protocol_digest(),
        "expected_protocol_digest": expected_protocol_digest,
        "dataset_id": dataset.dataset_id,
        "pair_names": list(pair_names),
        "roster_complete": roster_complete,
        "start_index": start_index,
        "stop_index": resolved_stop,
        "fit_stop_index": fit_stop_index,
        "causality_cut_index": cut_idx,
        "scenarios": list(resolved_scenarios),
        "authorization_url": auth.url,
        "authorization_check": auth.check,
        "authorization_authenticated": auth.authenticated,
        "authorization_receipt": None
        if auth.receipt is None
        else auth.receipt.to_dict(),
        "evidence_eligible": auth.evidence_eligible,
        "synthetic_only": auth.synthetic_only,
        "enforce_pre_fs1": True,
        "weights_digests": weights_digests,
        "weights_are_zero_residual": weights_are_zero_residual,
        "causality_probes": {key: res.to_dict() for key, res in probe_results.items()},
    }
    full_decision: dict[str, object] = {
        **decision_payload,
        "protocol_id": protocol.PROTOCOL_ID,
        "protocol_digest": protocol.protocol_digest(),
        "dataset_id": dataset.dataset_id,
        "synthetic_only": auth.synthetic_only,
        "authorization_url": auth.url,
        "authorization_check": auth.check,
        "authorization_authenticated": auth.authenticated,
        "roster_complete": roster_complete,
        "weights_are_zero_residual": weights_are_zero_residual,
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
