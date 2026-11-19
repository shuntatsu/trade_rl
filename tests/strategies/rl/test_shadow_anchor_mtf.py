"""Synthetic G2 tests for ShadowAnchorMtfPolicy: FM1 zero-residual parity, Non-RL gates, HOLD, risk/capacity."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from tests.data.test_mtf6_signals import _make_alternating_dataset
from trade_rl.data.features.mtf6_signals import build_mtf6_signal_pack
from trade_rl.data.market import MarketDataset
from trade_rl.evaluation.replay import (
    SharedCashPortfolioReplayResult,
    run_shared_portfolio_replay,
)
from trade_rl.risk.pretrade import PreTradeRisk, PreTradeRiskConfig
from trade_rl.simulation.execution import ExecutionCostConfig, MarketExecutor
from trade_rl.simulation.native_transition import validate_risk_execution_compatibility
from trade_rl.simulation.orders.model import OrderEvent
from trade_rl.strategies.portfolio import (
    PortfolioAccountSnapshot,
    PortfolioDecisionSnapshot,
    PortfolioRiskLimits,
)
from trade_rl.strategies.rl.shadow_anchor_mtf import (
    ShadowAnchorMlpWeights,
    ShadowAnchorMtfConfig,
    ShadowAnchorMtfPolicy,
    synthesize_alternating_leg_weights,
)


def _nonzero_weights(
    *, seed: int = 123, fit_stop_index: int = 0
) -> ShadowAnchorMlpWeights:
    rng = np.random.default_rng(seed)
    h = 16
    return ShadowAnchorMlpWeights(
        obs_mean_41=rng.normal(0.0, 0.1, size=41),
        obs_std_41=rng.uniform(0.5, 1.5, size=41),
        in_weight=rng.normal(0.0, 0.2, size=(h, 52)),
        in_bias=rng.normal(0.0, 0.05, size=h),
        in_ln_gamma=rng.uniform(0.8, 1.2, size=h),
        in_ln_beta=rng.normal(0.0, 0.02, size=h),
        res1_a_weight=rng.normal(0.0, 0.2, size=(h, h)),
        res1_a_bias=rng.normal(0.0, 0.05, size=h),
        res1_a_ln_gamma=rng.uniform(0.8, 1.2, size=h),
        res1_a_ln_beta=rng.normal(0.0, 0.02, size=h),
        res1_b_weight=rng.normal(0.0, 0.2, size=(h, h)),
        res1_b_bias=rng.normal(0.0, 0.05, size=h),
        res2_a_weight=rng.normal(0.0, 0.2, size=(h, h)),
        res2_a_bias=rng.normal(0.0, 0.05, size=h),
        res2_a_ln_gamma=rng.uniform(0.8, 1.2, size=h),
        res2_a_ln_beta=rng.normal(0.0, 0.02, size=h),
        res2_b_weight=rng.normal(0.0, 0.2, size=(h, h)),
        res2_b_bias=rng.normal(0.0, 0.05, size=h),
        mu_weight=rng.normal(0.0, 0.2, size=(7, h)),
        mu_bias=rng.normal(0.0, 0.1, size=7),
        obs_dim=52,
        act_dim=7,
        hidden=h,
        fit_stop_index=fit_stop_index,
    )


def _replay_cost(
    *, processing_bar_volume_capacity: bool = False
) -> ExecutionCostConfig:
    return ExecutionCostConfig(
        fee_rate=0.0004,
        spread_rate=0.0002,
        impact_rate=0.0002,
        allow_short=True,
        max_leverage=2.5,
        maintenance_margin_rate=0.05,
        max_participation_rate=0.01,
        minimum_notional=10.0,
        order_latency_bars=0,
        processing_bar_volume_capacity=processing_bar_volume_capacity,
    )


def _replay_risk() -> PreTradeRisk:
    return PreTradeRisk(
        PreTradeRiskConfig(
            max_gross=2.5,
            max_abs_weight=0.55,
            max_turnover=2.5,
            drawdown_start=0.10,
            drawdown_stop=0.20,
            emergency_turnover_override=True,
        )
    )


def test_fm1_zero_residual_equals_anchor_within_1e12() -> None:
    ds = _make_alternating_dataset(n_bars=70, pairs=("BTCUSDT", "ETHUSDT"), seed=31)
    pack = build_mtf6_signal_pack(ds)
    zw = ShadowAnchorMlpWeights.zero_residual(fit_stop_index=12)

    start_index = 12
    stop_index = 68
    cost = _replay_cost()
    risk = _replay_risk()

    res_anchor = run_shared_portfolio_replay(
        ds,
        ShadowAnchorMtfPolicy(
            signal_pack=pack,
            config=ShadowAnchorMtfConfig(
                mode="anc_bidir_75_25", start_index=start_index, fit_stop_index=12
            ),
        ),
        start_index=start_index,
        stop_index=stop_index,
        initial_capital=100_000.0,
        execution_cost=cost,
        risk=risk,
        settle_terminal_position=True,
        capture_ledger_evidence=True,
    )
    for rl_mode in ("core7_sparse_ppo", "orth6_cush_ppo"):
        res_rl = run_shared_portfolio_replay(
            ds,
            ShadowAnchorMtfPolicy(
                signal_pack=pack,
                config=ShadowAnchorMtfConfig(
                    mode=rl_mode, start_index=start_index, fit_stop_index=12
                ),
                weights=zw,
            ),
            start_index=start_index,
            stop_index=stop_index,
            initial_capital=100_000.0,
            execution_cost=cost,
            risk=risk,
            settle_terminal_position=True,
            capture_ledger_evidence=True,
        )
        assert (
            abs(res_rl.book.portfolio_value - res_anchor.book.portfolio_value) < 1e-12
        )
        for d_rl, d_an in zip(res_rl.decisions, res_anchor.decisions, strict=True):
            assert (
                max(
                    abs(a - b)
                    for a, b in zip(
                        d_rl.target_weights, d_an.target_weights, strict=True
                    )
                )
                < 1e-12
            )


def test_nonrl_regime_gate_throttling_and_mlp_determinism() -> None:
    ds = _make_alternating_dataset(n_bars=60, pairs=("BTCUSDT", "ETHUSDT"), seed=41)
    pack = build_mtf6_signal_pack(ds)

    ts_ns = ds.timestamps.astype("datetime64[ns]").astype(np.int64)
    limits = PortfolioRiskLimits(2.5, 0.55, 2.5, 0.10, 0.20, True, 1e-9)
    zeros = (0.0,) * ds.n_symbols
    snap = PortfolioDecisionSnapshot(
        index=24,
        timestamp_ns=int(ts_ns[24]),
        agent_stop_index=59,
        remaining_decisions=35,
        symbols=ds.symbols,
        feature_names=ds.feature_names,
        global_feature_names=ds.global_feature_names,
        features=tuple((0.0,) for _ in range(ds.n_symbols)),
        feature_available=tuple((True,) for _ in range(ds.n_symbols)),
        feature_staleness=tuple((0.0,) for _ in range(ds.n_symbols)),
        global_features=(0.0,),
        global_feature_available=(True,),
        account=PortfolioAccountSnapshot(
            cash=100_000.0,
            portfolio_value=100_000.0,
            peak_value=100_000.0,
            max_drawdown=0.0,
            exact_quantities=("0",) * ds.n_symbols,
            quantities=zeros,
            weights=zeros,
            mark_prices=tuple(float(v) for v in ds.close[24]),
            contract_multipliers=(1.0,) * ds.n_symbols,
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
        ),
        active_orders=(),
        risk=limits,
    )

    pol_anc = ShadowAnchorMtfPolicy(
        signal_pack=pack,
        config=ShadowAnchorMtfConfig(
            mode="anc_bidir_75_25", start_index=24, fit_stop_index=24
        ),
    )
    pol_gate = ShadowAnchorMtfPolicy(
        signal_pack=pack,
        config=ShadowAnchorMtfConfig(
            mode="anc_regime_gate_nonrl", start_index=24, fit_stop_index=24
        ),
    )
    w_anc = np.asarray(pol_anc.decide_portfolio(snap).target_weights, dtype=np.float64)
    w_gate = np.asarray(
        pol_gate.decide_portfolio(snap).target_weights, dtype=np.float64
    )

    g_a = float(pack.gate_alpha_nonrl[24])
    g_c = float(pack.gate_carry_nonrl[24])
    assert 0.45 <= g_a <= 1.0 and 0.30 <= g_c <= 1.0
    # Non-RL gate never exceeds unthrottled spot gross or perp gross bounds
    assert np.all(w_gate[0::2] >= 0.0)
    assert np.sum(np.abs(w_gate)) <= np.sum(np.abs(w_anc)) + 1e-12

    # Non-zero MLP weights digest changes when mutated and forward pass is deterministic
    w1 = _nonzero_weights(seed=77, fit_stop_index=24)
    w2 = replace(w1, mu_bias=w1.mu_bias + 0.01)
    assert w1.weights_digest() != w2.weights_digest()
    obs = np.linspace(-0.5, 0.5, 52, dtype=np.float64)
    np.testing.assert_allclose(
        w1.forward_deterministic(obs), w1.forward_deterministic(obs)
    )


def test_hold_and_deadband_and_drawdown_hold_release() -> None:
    ds = _make_alternating_dataset(n_bars=40, pairs=("BTCUSDT", "ETHUSDT"), seed=55)
    pack = build_mtf6_signal_pack(ds)
    pol = ShadowAnchorMtfPolicy(
        signal_pack=pack,
        config=ShadowAnchorMtfConfig(
            mode="anc_bidir_75_25",
            start_index=10,
            fit_stop_index=10,
            rebalance_every_bars=6,
            deadband=10.0,  # Forces deadband HOLD on rebalance bars when weights unchanged
        ),
    )
    ts_ns = ds.timestamps.astype("datetime64[ns]").astype(np.int64)
    limits = PortfolioRiskLimits(2.5, 0.55, 2.5, 0.10, 0.20, True, 1e-9)

    def _snap(idx: int, pv: float) -> PortfolioDecisionSnapshot:
        cur_w = (0.1, -0.1, 0.05, -0.05)
        return PortfolioDecisionSnapshot(
            index=idx,
            timestamp_ns=int(ts_ns[idx]),
            agent_stop_index=38,
            remaining_decisions=38 - idx,
            symbols=ds.symbols,
            feature_names=ds.feature_names,
            global_feature_names=ds.global_feature_names,
            features=tuple((0.0,) for _ in range(ds.n_symbols)),
            feature_available=tuple((True,) for _ in range(ds.n_symbols)),
            feature_staleness=tuple((0.0,) for _ in range(ds.n_symbols)),
            global_features=(0.0,),
            global_feature_available=(True,),
            account=PortfolioAccountSnapshot(
                cash=pv,
                portfolio_value=pv,
                peak_value=100_000.0,
                max_drawdown=max(0.0, 1.0 - pv / 100_000.0),
                exact_quantities=("1", "-1", "1", "-1"),
                quantities=(1.0, -1.0, 1.0, -1.0),
                weights=cur_w,
                mark_prices=tuple(float(v) for v in ds.close[idx]),
                contract_multipliers=(1.0,) * ds.n_symbols,
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
            ),
            active_orders=(),
            risk=limits,
        )

    # Bar 10: rebalance bar with huge deadband -> HOLD
    p10 = pol.decide_portfolio(_snap(10, 100_000.0))
    assert p10.hold_current_quantity_symbols == (0, 1, 2, 3)

    # Bar 11: non-rebalance bar without drawdown -> HOLD
    p11 = pol.decide_portfolio(_snap(11, 95_000.0))
    assert p11.hold_current_quantity_symbols == (0, 1, 2, 3)

    # Bar 12: non-rebalance bar WITH drawdown >= drawdown_start (12% >= 10%) -> release HOLD
    p12 = pol.decide_portfolio(_snap(12, 88_000.0))
    assert p12.hold_current_quantity_symbols == ()


def test_risk_execution_compatibility_refuses_leverage_mismatch() -> None:
    ds_base = _make_alternating_dataset(
        n_bars=55, pairs=("BTCUSDT", "ETHUSDT"), seed=61
    )
    risk_obj = _replay_risk()
    cost_bad = ExecutionCostConfig(
        max_leverage=1.0, allow_short=True, processing_bar_volume_capacity=False
    )
    with pytest.raises(ValueError, match="max_leverage"):
        validate_risk_execution_compatibility(
            risk_obj, MarketExecutor(ds_base, cost_bad)
        )

    cost_ok = _replay_cost(processing_bar_volume_capacity=False)
    validate_risk_execution_compatibility(risk_obj, MarketExecutor(ds_base, cost_ok))


def _fm3_replay(
    ds_base: MarketDataset,
    volume: np.ndarray,
    *,
    processing_bar_volume_capacity: bool,
    dataset_id: str,
) -> SharedCashPortfolioReplayResult:
    ds = replace(ds_base, dataset_id=dataset_id, volume=volume)
    return run_shared_portfolio_replay(
        ds,
        ShadowAnchorMtfPolicy(
            signal_pack=build_mtf6_signal_pack(ds),
            config=ShadowAnchorMtfConfig(
                mode="anc_bidir_75_25", start_index=10, fit_stop_index=10
            ),
        ),
        start_index=10,
        stop_index=20,
        initial_capital=100_000.0,
        execution_cost=_replay_cost(
            processing_bar_volume_capacity=processing_bar_volume_capacity
        ),
        risk=_replay_risk(),
        settle_terminal_position=False,
        capture_ledger_evidence=True,
    )


def _fills(result: SharedCashPortfolioReplayResult) -> list[tuple[int, OrderEvent]]:
    assert result.ledger_evidence is not None
    return [
        (interval.start_index, event)
        for interval in result.ledger_evidence.intervals.intervals
        for event in interval.order_events
        if event.filled_quantity != 0.0
    ]


def test_fm3_previous_bar_capacity_shock_discriminates_from_processing_bar() -> None:
    """Shock the volume of the bar a *clipped* rebalance fill executes in.

    The rebalance at bar 10 fills at bar 11 with a 1% cap. Bar 10's volume is made
    tight so the clip binds. With ``processing_bar_volume_capacity=False`` the cap reads
    the completed bar 10, so shocking bar 11 changes nothing; with ``True`` the cap
    reads bar 11, so the same shock must change the fills (positive control).
    """
    ds_base = _make_alternating_dataset(
        n_bars=40, pairs=("BTCUSDT", "ETHUSDT"), seed=61
    )
    vol_a = np.asarray(ds_base.volume, dtype=np.float64).copy()
    vol_a[10, :] = 120.0  # completed bar before the bar-11 fill: tight
    vol_b = vol_a.copy()
    vol_b[11, :] = 5e8  # the bar the fill executes in: shocked

    res_a = _fm3_replay(
        ds_base, vol_a, processing_bar_volume_capacity=False, dataset_id="a" * 64
    )
    res_b = _fm3_replay(
        ds_base, vol_b, processing_bar_volume_capacity=False, dataset_id="b" * 64
    )
    fills_a = _fills(res_a)
    # the clip binds on the rebalance fill: non-zero, partial, at the 1% cap
    first = [event for start, event in fills_a if start == 10]
    assert first, "rebalance at bar 10 must produce fills"
    for event in first:
        assert abs(event.filled_quantity) < abs(event.requested_quantity)
        assert event.participation_rate == pytest.approx(0.01, rel=1e-9)
    np.testing.assert_array_equal(res_a.book.quantities, res_b.book.quantities)
    assert res_a.book.portfolio_value == res_b.book.portfolio_value
    assert [e.filled_quantity for _, e in fills_a] == [
        e.filled_quantity for _, e in _fills(res_b)
    ]

    # positive control: the processing-bar flag reads bar 11, so the shock must matter
    res_a_true = _fm3_replay(
        ds_base, vol_a, processing_bar_volume_capacity=True, dataset_id="a" * 64
    )
    res_b_true = _fm3_replay(
        ds_base, vol_b, processing_bar_volume_capacity=True, dataset_id="b" * 64
    )
    assert not np.array_equal(res_a_true.book.quantities, res_b_true.book.quantities)
    assert [e.filled_quantity for _, e in _fills(res_a_true)] != [
        e.filled_quantity for _, e in _fills(res_b_true)
    ]


def test_replay_participation_cap_and_minimum_notional_bind() -> None:
    """Native 1% participation and 10 USDT minimum notional hold on a real replay."""
    ds_base = _make_alternating_dataset(
        n_bars=40, pairs=("BTCUSDT", "ETHUSDT"), seed=61
    )
    vol = np.asarray(ds_base.volume, dtype=np.float64).copy()
    vol[10, :] = 120.0  # clip at bar 11
    vol[16, :] = 2.0  # cap notional ~2 USDT (< 10) for the bar-17 fill
    res = _fm3_replay(
        ds_base, vol, processing_bar_volume_capacity=False, dataset_id="c" * 64
    )
    fills = _fills(res)
    assert fills
    cost = _replay_cost()
    assert any(
        event.participation_rate == pytest.approx(0.01, rel=1e-9) for _, event in fills
    ), "the 1% participation cap must bind on at least one fill"
    for _, event in fills:
        assert event.participation_rate <= cost.max_participation_rate + 1e-12
        assert event.filled_notional >= cost.minimum_notional - 1e-9
    # bar-16 rebalance: capacity below the minimum notional yields no fill at all
    assert [start for start, _ in fills if start == 16] == []
    vol_ctrl = vol.copy()
    vol_ctrl[16, :] = 150_000.0
    ctrl = _fills(
        _fm3_replay(
            ds_base,
            vol_ctrl,
            processing_bar_volume_capacity=False,
            dataset_id="d" * 64,
        )
    )
    assert any(start == 16 for start, _ in ctrl), (
        "control: without the tight volume the bar-16 rebalance trades"
    )


def test_fit_stop_after_start_refused() -> None:
    ds = _make_alternating_dataset(n_bars=24, pairs=("BTCUSDT", "ETHUSDT"), seed=9)
    pack = build_mtf6_signal_pack(ds)
    with pytest.raises(ValueError, match="lookahead forbidden"):
        ShadowAnchorMtfConfig(mode="core7_sparse_ppo", start_index=5, fit_stop_index=6)

    cfg = ShadowAnchorMtfConfig(
        mode="core7_sparse_ppo", start_index=5, fit_stop_index=5
    )
    w_leak = _nonzero_weights(seed=1, fit_stop_index=8)
    with pytest.raises(ValueError, match="exceeds start_index"):
        ShadowAnchorMtfPolicy(signal_pack=pack, config=cfg, weights=w_leak)


def _gross(weights: tuple[float, ...] | np.ndarray) -> float:
    return float(np.sum(np.abs(np.asarray(weights, dtype=np.float64))))


def test_drawdown_deleveraging_tracks_risk_scale_without_compounding() -> None:
    """After a drawdown breach the book follows ``risk_scale * intended`` gross.

    A non-rebalance bar must re-propose the cached pre-risk target so ``PreTradeRisk``
    scales it exactly once. Feeding risk-scaled current weights back through the risk
    controller compounds the scale every bar and then snaps back at the next rebalance.
    """
    ds = _make_alternating_dataset(n_bars=90, pairs=("BTCUSDT", "ETHUSDT"), seed=1)
    pack = build_mtf6_signal_pack(ds)
    start = 10
    risk = _replay_risk()
    res = run_shared_portfolio_replay(
        ds,
        ShadowAnchorMtfPolicy(
            signal_pack=pack,
            config=ShadowAnchorMtfConfig(
                mode="anc_bidir_75_25",
                start_index=start,
                fit_stop_index=start,
                alpha_scale=6.0,  # large enough to reach the 10% drawdown
            ),
        ),
        start_index=start,
        stop_index=88,
        initial_capital=100_000.0,
        execution_cost=_replay_cost(),
        risk=risk,
        settle_terminal_position=False,
        capture_ledger_evidence=True,
    )
    assert res.ledger_evidence is not None
    intervals = res.ledger_evidence.intervals.intervals
    decisions = res.decisions
    assert len(intervals) == len(decisions)

    intended: tuple[float, ...] | None = None
    breached_bars = 0
    previous_gross: float | None = None
    max_jump = 0.0
    for decision, interval in zip(decisions, intervals, strict=True):
        is_rebalance = (decision.index - start) % 6 == 0
        if is_rebalance:
            intended = decision.proposal_weights
        scale = risk.risk_scale(interval.max_drawdown_before)
        if scale >= 1.0 or intended is None:
            previous_gross = None
            continue
        breached_bars += 1
        target_gross = _gross(decision.target_weights)
        assert target_gross <= scale * _gross(intended) + 1e-9
        assert target_gross >= 0.95 * scale * _gross(intended)
        if not is_rebalance:
            # the proposal is the intended target or its deadband pre-image; either way
            # risk scales it exactly once
            assert target_gross <= scale * _gross(decision.proposal_weights) + 1e-9
        if previous_gross is not None:
            max_jump = max(max_jump, abs(target_gross - previous_gross))
        previous_gross = target_gross
    assert breached_bars >= 12, "the replay must exercise sustained deleveraging"
    assert max_jump <= 0.15, f"gross snapped by {max_jump}"


def _snapshot(
    ds: MarketDataset,
    *,
    index: int,
    weights: tuple[float, ...],
    portfolio_value: float,
    peak_value: float = 100_000.0,
) -> PortfolioDecisionSnapshot:
    ts_ns = ds.timestamps.astype("datetime64[ns]").astype(np.int64)
    n = ds.n_symbols
    return PortfolioDecisionSnapshot(
        index=index,
        timestamp_ns=int(ts_ns[index]),
        agent_stop_index=ds.n_bars - 1,
        remaining_decisions=ds.n_bars - 1 - index,
        symbols=ds.symbols,
        feature_names=ds.feature_names,
        global_feature_names=ds.global_feature_names,
        features=tuple((0.0,) for _ in range(n)),
        feature_available=tuple((True,) for _ in range(n)),
        feature_staleness=tuple((0.0,) for _ in range(n)),
        global_features=(0.0,),
        global_feature_available=(True,),
        account=PortfolioAccountSnapshot(
            cash=portfolio_value,
            portfolio_value=portfolio_value,
            peak_value=peak_value,
            max_drawdown=max(0.0, 1.0 - portfolio_value / peak_value),
            exact_quantities=("0",) * n,
            quantities=(0.0,) * n,
            weights=weights,
            mark_prices=tuple(float(v) for v in ds.close[index]),
            contract_multipliers=(1.0,) * n,
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
        ),
        active_orders=(),
        risk=PortfolioRiskLimits(2.5, 0.55, 2.5, 0.10, 0.20, True, 1e-9),
    )


def test_deadband_is_released_on_a_rebalance_bar_during_drawdown() -> None:
    """A sub-deadband rebalance holds when healthy but must trade when breached."""
    ds = _make_alternating_dataset(n_bars=40, pairs=("BTCUSDT", "ETHUSDT"), seed=55)
    pack = build_mtf6_signal_pack(ds)

    def make() -> ShadowAnchorMtfPolicy:
        return ShadowAnchorMtfPolicy(
            signal_pack=pack,
            config=ShadowAnchorMtfConfig(
                mode="anc_bidir_75_25",
                start_index=10,
                fit_stop_index=10,
                deadband=0.04,
            ),
        )

    zero = (0.0,) * ds.n_symbols
    proposed = make().decide_portfolio(
        _snapshot(ds, index=10, weights=zero, portfolio_value=100_000.0)
    )
    assert proposed.hold_current_quantity_symbols == ()
    target = proposed.target_weights
    assert _gross(target) > 0.1  # a non-trivial anchor so the comparison is meaningful

    # current weights == the proposal: |proposal - current| = 0 < deadband
    healthy = make().decide_portfolio(
        _snapshot(ds, index=10, weights=target, portfolio_value=100_000.0)
    )
    assert healthy.hold_current_quantity_symbols == (0, 1, 2, 3)
    assert healthy.target_weights == target

    breached = make().decide_portfolio(
        _snapshot(ds, index=10, weights=target, portfolio_value=88_000.0)
    )
    assert breached.hold_current_quantity_symbols == ()
    assert breached.target_weights == target


def _leg_inputs(
    n_pairs: int, *, carry: float, alpha: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    c = np.full(n_pairs, carry, dtype=np.float64)
    a = np.full(n_pairs, alpha, dtype=np.float64)
    a[1::2] *= -1.0
    on = np.ones(n_pairs, dtype=bool)
    return c, a, on, on.copy()


def test_leg_weight_projection_caps_spot_nonnegative_and_gross() -> None:
    """Direct unit oracle for ``synthesize_alternating_leg_weights``."""
    # oversized carry and alpha across 6 pairs: leg clips alone give 6.6 gross
    c, a, cm, pm = _leg_inputs(6, carry=3.0, alpha=3.0)
    legs = np.asarray(
        synthesize_alternating_leg_weights(
            c, a, c_scale=1.0, a_scale=1.0, carry_mask=cm, perp_mask=pm
        )
    )
    assert legs.shape == (12,)
    assert np.all(legs[0::2] >= 0.0)  # spot never short
    assert np.max(np.abs(legs)) <= 0.55 + 1e-12
    assert _gross(legs) == pytest.approx(2.5, abs=1e-9)  # gross cap binds exactly

    # the gross rescale preserves leg proportions (it is a uniform scale)
    raw = np.asarray(
        synthesize_alternating_leg_weights(
            c, a, c_scale=1.0, a_scale=1.0, carry_mask=cm, perp_mask=pm, gross_cap=99.0
        )
    )
    assert _gross(raw) == pytest.approx(6.6, abs=1e-9)
    np.testing.assert_allclose(legs, raw * (2.5 / 6.6), atol=1e-12)

    # a single extreme alpha must be leg-clipped at 0.55 (not merely gross-scaled)
    c1 = np.zeros(6)
    a1 = np.zeros(6)
    a1[0], a1[1] = 10.0, -0.01
    one = np.asarray(
        synthesize_alternating_leg_weights(
            c1, a1, c_scale=1.0, a_scale=1.0, carry_mask=cm, perp_mask=pm
        )
    )
    assert one[1] == pytest.approx(0.55, abs=1e-12)
    assert np.max(np.abs(one)) <= 0.55 + 1e-12

    # negative carry input never produces a negative spot leg or a short-carry perp
    neg = np.asarray(
        synthesize_alternating_leg_weights(
            np.full(6, -2.0),
            np.zeros(6),
            c_scale=1.0,
            a_scale=0.0,
            carry_mask=cm,
            perp_mask=pm,
        )
    )
    np.testing.assert_array_equal(neg, np.zeros(12))

    # masks: carry off removes spot/carry; perp off removes alpha
    off = np.zeros(6, dtype=bool)
    masked = np.asarray(
        synthesize_alternating_leg_weights(
            c, a, c_scale=1.0, a_scale=1.0, carry_mask=off, perp_mask=off
        )
    )
    np.testing.assert_array_equal(masked, np.zeros(12))
    spot_only = np.asarray(
        synthesize_alternating_leg_weights(
            c, a, c_scale=1.0, a_scale=1.0, carry_mask=cm, perp_mask=off
        )
    )
    assert np.all(spot_only[0::2] > 0.0) and np.all(spot_only[1::2] <= 0.0)


def test_breach_bars_do_not_churn_within_the_deadband() -> None:
    """Non-rebalance breach bars re-target only on real drift or a risk_scale change.

    An exact-quantity HOLD is impossible while ``risk_scale < 1`` (risk would scale the
    held weights again), so inside the deadband the policy proposes the pre-image that
    ``PreTradeRisk`` maps back onto the current weights. Bound: per-bar turnover stays
    bounded (<= 0.10), drift-sized trades are suppressed on several bars, and the
    non-rebalance cost is a small share of the breach-period cost.
    """
    ds = _make_alternating_dataset(n_bars=90, pairs=("BTCUSDT", "ETHUSDT"), seed=1)
    risk = _replay_risk()
    res = run_shared_portfolio_replay(
        ds,
        ShadowAnchorMtfPolicy(
            signal_pack=build_mtf6_signal_pack(ds),
            config=ShadowAnchorMtfConfig(
                mode="anc_bidir_75_25",
                start_index=10,
                fit_stop_index=10,
                alpha_scale=6.0,
            ),
        ),
        start_index=10,
        stop_index=88,
        initial_capital=100_000.0,
        execution_cost=_replay_cost(),
        risk=risk,
        settle_terminal_position=False,
        capture_ledger_evidence=True,
    )
    assert res.ledger_evidence is not None
    non_rebalance_turnover: list[float] = []
    non_rebalance_cost = 0.0
    breach_cost = 0.0
    previous_scale = 1.0
    for decision, interval in zip(
        res.decisions, res.ledger_evidence.intervals.intervals, strict=True
    ):
        scale = risk.risk_scale(interval.max_drawdown_before)
        scale_changed = scale != previous_scale
        previous_scale = scale
        if scale >= 1.0:
            continue
        breach_cost += interval.interval_cost
        # a bar on which risk_scale itself dropped legitimately de-levers
        if (decision.index - 10) % 6 != 0 and not scale_changed:
            non_rebalance_turnover.append(
                interval.turnover_total_after - interval.turnover_total_before
            )
            non_rebalance_cost += interval.interval_cost
    assert len(non_rebalance_turnover) >= 20
    assert max(non_rebalance_turnover) <= 0.10  # drift beyond the deadband re-targets
    assert sum(non_rebalance_turnover) / len(non_rebalance_turnover) <= 0.02
    assert sum(1 for t in non_rebalance_turnover if t == 0.0) >= 3
    assert non_rebalance_cost <= 0.10 * breach_cost


def test_breach_proposal_is_preimage_within_deadband_and_intended_otherwise() -> None:
    ds = _make_alternating_dataset(n_bars=40, pairs=("BTCUSDT", "ETHUSDT"), seed=55)
    pack = build_mtf6_signal_pack(ds)

    def make() -> ShadowAnchorMtfPolicy:
        return ShadowAnchorMtfPolicy(
            signal_pack=pack,
            config=ShadowAnchorMtfConfig(
                mode="anc_bidir_75_25", start_index=10, fit_stop_index=10
            ),
        )

    zero = (0.0,) * ds.n_symbols
    risk = _replay_risk()
    pol = make()
    intended = pol.decide_portfolio(
        _snapshot(ds, index=10, weights=zero, portfolio_value=100_000.0)
    ).target_weights
    scale = risk.risk_scale(0.12)
    assert 0.0 < scale < 1.0
    scaled = tuple(w * scale for w in intended)

    # current already at risk_scale * intended: risk must map the proposal onto current
    near = pol.decide_portfolio(
        _snapshot(ds, index=11, weights=scaled, portfolio_value=88_000.0)
    )
    assert near.hold_current_quantity_symbols == ()
    constrained = risk.constrain(
        np.asarray(near.target_weights), current=np.asarray(scaled), drawdown=0.12
    )
    np.testing.assert_allclose(constrained.weights, scaled, atol=1e-12)

    # current far from risk_scale * intended: the intended pre-risk target
    far = pol.decide_portfolio(
        _snapshot(ds, index=12, weights=zero, portfolio_value=88_000.0)
    )
    assert far.target_weights == intended

    # a deeper drawdown changes risk_scale: the intended target again (risk shrinks it)
    deeper = make()
    deeper.decide_portfolio(
        _snapshot(ds, index=10, weights=zero, portfolio_value=100_000.0)
    )
    shrink = deeper.decide_portfolio(
        _snapshot(ds, index=11, weights=scaled, portfolio_value=82_000.0)
    )
    assert shrink.target_weights == intended
