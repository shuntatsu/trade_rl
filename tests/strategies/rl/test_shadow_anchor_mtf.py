"""Synthetic G2 tests for ShadowAnchorMtfPolicy: FM1 zero-residual parity, Non-RL gates, HOLD, risk/capacity."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from tests.data.test_mtf6_signals import _make_alternating_dataset
from trade_rl.data.features.mtf6_signals import build_mtf6_signal_pack
from trade_rl.evaluation.replay import run_shared_portfolio_replay
from trade_rl.risk.pretrade import PreTradeRisk, PreTradeRiskConfig
from trade_rl.simulation.execution import ExecutionCostConfig, MarketExecutor
from trade_rl.simulation.native_transition import validate_risk_execution_compatibility
from trade_rl.strategies.portfolio import (
    PortfolioAccountSnapshot,
    PortfolioDecisionSnapshot,
    PortfolioRiskLimits,
)
from trade_rl.strategies.rl.shadow_anchor_mtf import (
    ShadowAnchorMlpWeights,
    ShadowAnchorMtfConfig,
    ShadowAnchorMtfPolicy,
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


def test_risk_execution_compatibility_and_fm3_previous_bar_capacity_and_stress() -> (
    None
):
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

    # FM3: processing_bar_volume_capacity=False ignores current-bar processing volume shock
    # Tighten volume around bar 14 so 1% capacity clips orders, then test future volume spike at 14
    vol_a = np.asarray(ds_base.volume, dtype=np.float64).copy()
    vol_a[13, :] = 12.0  # t-1 completed bar tight volume for bar 14 fill
    vol_b = vol_a.copy()
    vol_b[14, :] = (
        500_000.0  # future/processing bar volume shock at 14 (ignored when flag=False)
    )
    ds_a = replace(ds_base, dataset_id="a" * 64, volume=vol_a)
    ds_b = replace(ds_base, dataset_id="b" * 64, volume=vol_b)

    pack_a = build_mtf6_signal_pack(ds_a)
    pack_b = build_mtf6_signal_pack(ds_b)
    w_rl = _nonzero_weights(seed=5, fit_stop_index=10)
    res_a = run_shared_portfolio_replay(
        ds_a,
        ShadowAnchorMtfPolicy(
            signal_pack=pack_a,
            config=ShadowAnchorMtfConfig(
                mode="core7_sparse_ppo", start_index=10, fit_stop_index=10
            ),
            weights=w_rl,
        ),
        start_index=10,
        stop_index=15,  # Executes bars 11..14; bar 14 fill reads volume[13] == 12.0 on both!
        initial_capital=100_000.0,
        execution_cost=cost_ok,
        risk=_replay_risk(),
        settle_terminal_position=False,
        capture_ledger_evidence=True,
    )
    res_b = run_shared_portfolio_replay(
        ds_b,
        ShadowAnchorMtfPolicy(
            signal_pack=pack_b,
            config=ShadowAnchorMtfConfig(
                mode="core7_sparse_ppo", start_index=10, fit_stop_index=10
            ),
            weights=w_rl,
        ),
        start_index=10,
        stop_index=15,
        initial_capital=100_000.0,
        execution_cost=cost_ok,
        risk=_replay_risk(),
        settle_terminal_position=False,
        capture_ledger_evidence=True,
    )
    np.testing.assert_array_equal(res_a.book.quantities, res_b.book.quantities)
    assert res_a.book.portfolio_value == res_b.book.portfolio_value


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
