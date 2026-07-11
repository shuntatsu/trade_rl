import numpy as np
import pytest

from mars_lite.env.portfolio_env import PortfolioTradingEnv
from mars_lite.learning.baselines import simulate_strategy
from mars_lite.trading.pre_trade_risk import (
    PendingOrder,
    PreTradeRejection,
    PreTradeRiskConfig,
    PreTradeRiskVerifier,
)


def test_verifier_leverage_limit():
    verifier = PreTradeRiskVerifier(PreTradeRiskConfig(max_leverage=1.5))
    verifier.validate(np.array([0.5, 0.5, -0.5]), 1000.0)
    with pytest.raises(PreTradeRejection) as exc_info:
        verifier.validate(np.array([0.6, 0.5, -0.5]), 1000.0)
    assert exc_info.value.reason == "leverage_limit_exceeded"
    assert exc_info.value.details["gross_leverage"] == pytest.approx(1.6)


def test_verifier_single_weight_limit():
    verifier = PreTradeRiskVerifier(PreTradeRiskConfig(max_single_weight=0.4))
    verifier.validate(np.array([0.3, -0.3, 0.2]), 1000.0)
    with pytest.raises(PreTradeRejection) as exc_info:
        verifier.validate(np.array([0.45, -0.3, 0.2]), 1000.0)
    assert exc_info.value.reason == "single_weight_limit_exceeded"


def test_verifier_notional_and_position_limits():
    verifier = PreTradeRiskVerifier(
        PreTradeRiskConfig(max_notional=1200.0, max_position_pct=0.65)
    )
    verifier.validate(np.array([0.5, -0.6]), 1000.0)
    with pytest.raises(PreTradeRejection) as exc_info:
        verifier.validate(np.array([0.6, -0.7]), 1000.0)
    assert exc_info.value.reason in {
        "position_pct_limit_exceeded",
        "notional_limit_exceeded",
    }


def test_nan_and_inf_are_rejected():
    verifier = PreTradeRiskVerifier(PreTradeRiskConfig())
    with pytest.raises(PreTradeRejection) as exc:
        verifier.validate(np.array([0.1, np.nan]), 1000.0)
    assert exc.value.reason == "nan_or_inf_in_weights"


def test_net_exposure_limit():
    verifier = PreTradeRiskVerifier(PreTradeRiskConfig(max_net_exposure=0.5))
    with pytest.raises(PreTradeRejection) as exc:
        verifier.validate(np.array([0.4, 0.2]), 1000.0)
    assert exc.value.reason == "net_exposure_limit_exceeded"


def test_worst_case_uses_one_sided_fill_scenarios():
    verifier = PreTradeRiskVerifier(PreTradeRiskConfig(max_worst_case_notional=1_200.0))
    current = np.array([0.4, -0.2])
    target = np.array([0.5, -0.1])
    orders = [
        PendingOrder("BTCUSDT", "buy", 500.0),
        PendingOrder("BTCUSDT", "sell", 100.0),
        PendingOrder("ETHUSDT", "sell", 500.0),
    ]
    with pytest.raises(PreTradeRejection) as exc:
        verifier.validate(
            target,
            1000.0,
            symbols=["BTCUSDT", "ETHUSDT"],
            current_weights=current,
            open_orders=orders,
        )
    assert exc.value.reason == "worst_case_notional_exceeded"
    assert exc.value.details["worst_case_notional"] == pytest.approx(1700.0)


def test_reduce_only_pending_order_does_not_increase_worst_case():
    verifier = PreTradeRiskVerifier(PreTradeRiskConfig(max_worst_case_notional=500.0))
    verifier.validate(
        np.array([0.4]),
        1000.0,
        symbols=["BTCUSDT"],
        current_weights=np.array([0.4]),
        open_orders=[PendingOrder("BTCUSDT", "sell", 400.0, reduce_only=True)],
    )


def test_min_order_notional_uses_delta_not_target_position():
    verifier = PreTradeRiskVerifier(PreTradeRiskConfig(min_order_notional=10.0))
    with pytest.raises(PreTradeRejection) as exc:
        verifier.validate(
            np.array([0.505]),
            1000.0,
            current_weights=np.array([0.5]),
        )
    assert exc.value.reason == "min_order_notional_not_met"
    assert exc.value.details["order_notional"] == pytest.approx(5.0)


def test_unchanged_large_position_does_not_trigger_min_order():
    verifier = PreTradeRiskVerifier(PreTradeRiskConfig(min_order_notional=10.0))
    verifier.validate(np.array([0.5]), 1000.0, current_weights=np.array([0.5]))


def test_liquidity_cap_uses_delta_plus_open_orders():
    verifier = PreTradeRiskVerifier(
        PreTradeRiskConfig(symbol_liquidity_caps={"BTCUSDT": 100.0})
    )
    with pytest.raises(PreTradeRejection) as exc:
        verifier.validate(
            np.array([0.55]),
            1000.0,
            symbols=["BTCUSDT"],
            current_weights=np.array([0.5]),
            open_orders=[PendingOrder("BTCUSDT", "buy", 60.0)],
        )
    assert exc.value.reason == "symbol_liquidity_cap_exceeded"
    assert exc.value.details["execution_notional"] == pytest.approx(110.0)


def test_forbidden_symbol_allows_reduction_but_blocks_increase_or_flip():
    verifier = PreTradeRiskVerifier(PreTradeRiskConfig(forbidden_symbols={"DOGEUSDT"}))
    verifier.validate(
        np.array([0.1]),
        1000.0,
        symbols=["DOGEUSDT"],
        current_weights=np.array([0.2]),
    )
    verifier.validate(
        np.array([0.0]),
        1000.0,
        symbols=["DOGEUSDT"],
        current_weights=np.array([0.2]),
    )
    for target in (np.array([0.3]), np.array([-0.1])):
        with pytest.raises(PreTradeRejection) as exc:
            verifier.validate(
                target,
                1000.0,
                symbols=["DOGEUSDT"],
                current_weights=np.array([0.2]),
            )
        assert exc.value.reason == "forbidden_symbol"


class DummyFeatureSet:
    def __init__(self):
        self.symbols = ["BTCUSDT", "ETHUSDT"]
        self.n_symbols = 2
        self.n_bars = 10
        self.n_features = 2
        self.feature_names = ["feat1", "feat2"]
        self.close = np.ones((10, 2))
        self.features = np.zeros((10, 2, 2))
        self.global_features = np.zeros((10, 2))
        self.funding_rate = np.zeros((10, 2))


def test_env_integration_passes_current_weights_and_symbols():
    fs = DummyFeatureSet()
    verifier = PreTradeRiskVerifier(PreTradeRiskConfig(max_leverage=0.5))
    env = PortfolioTradingEnv(fs, pre_trade_verifier=verifier, initial_capital=100.0)
    env.reset()
    with pytest.raises(PreTradeRejection):
        env.step(np.array([0.3, 0.3]))


def test_simulate_strategy_integration():
    fs = DummyFeatureSet()
    verifier = PreTradeRiskVerifier(PreTradeRiskConfig(max_leverage=0.5))

    def risky_strategy(fs, t, w):
        return np.array([0.3, 0.3])

    with pytest.raises(PreTradeRejection):
        simulate_strategy(
            fs,
            risky_strategy,
            pre_trade_verifier=verifier,
            min_trade_delta=0.0,
        )
