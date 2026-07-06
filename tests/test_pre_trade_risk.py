import numpy as np
import pytest

from mars_lite.env.portfolio_env import PortfolioTradingEnv
from mars_lite.features.feature_pipeline import FeatureSet
from mars_lite.learning.baselines import simulate_strategy
from mars_lite.trading.pre_trade_risk import (
    PreTradeRejection,
    PreTradeRiskConfig,
    PreTradeRiskVerifier,
)


def test_verifier_leverage_limit():
    config = PreTradeRiskConfig(max_leverage=1.5)
    verifier = PreTradeRiskVerifier(config)

    # 正常
    verifier.validate(np.array([0.5, 0.5, -0.5]), 1000.0)

    # 超過
    with pytest.raises(PreTradeRejection) as exc_info:
        verifier.validate(np.array([0.6, 0.5, -0.5]), 1000.0)
    assert exc_info.value.reason == "leverage_limit_exceeded"
    assert exc_info.value.details["gross_leverage"] == 1.6


def test_verifier_single_weight_limit():
    config = PreTradeRiskConfig(max_single_weight=0.4)
    verifier = PreTradeRiskVerifier(config)

    # 正常
    verifier.validate(np.array([0.3, -0.3, 0.2]), 1000.0)

    # 超過
    with pytest.raises(PreTradeRejection) as exc_info:
        verifier.validate(np.array([0.45, -0.3, 0.2]), 1000.0)
    assert exc_info.value.reason == "single_weight_limit_exceeded"
    assert exc_info.value.details["max_single_weight_found"] == 0.45


def test_verifier_notional_limit():
    config = PreTradeRiskConfig(max_notional=1200.0)
    verifier = PreTradeRiskVerifier(config)

    # 正常 (portfolio_value * leverage = 1000.0 * 1.1 = 1100.0)
    verifier.validate(np.array([0.5, -0.6]), 1000.0)

    # 超過 (portfolio_value * leverage = 1000.0 * 1.3 = 1300.0)
    with pytest.raises(PreTradeRejection) as exc_info:
        verifier.validate(np.array([0.6, -0.7]), 1000.0)
    assert exc_info.value.reason == "notional_limit_exceeded"
    assert exc_info.value.details["total_notional"] == pytest.approx(1300.0)


def test_verifier_position_pct_limit():
    config = PreTradeRiskConfig(max_position_pct=0.25)
    verifier = PreTradeRiskVerifier(config)

    verifier.validate(np.array([0.2, -0.1]), 1000.0)

    with pytest.raises(PreTradeRejection) as exc_info:
        verifier.validate(np.array([0.3, -0.1]), 1000.0)
    assert exc_info.value.reason == "position_pct_limit_exceeded"
    assert exc_info.value.details["max_position_pct_found"] == pytest.approx(0.3)


def test_verifier_forbidden_symbol_rejection():
    config = PreTradeRiskConfig(forbidden_symbols={"DOGEUSDT"})
    verifier = PreTradeRiskVerifier(config)

    verifier.validate(
        np.array([0.0, 0.2]),
        1000.0,
        symbols=["BTCUSDT", "ETHUSDT"],
    )

    with pytest.raises(PreTradeRejection) as exc_info:
        verifier.validate(
            np.array([0.0, 0.2]),
            1000.0,
            symbols=["BTCUSDT", "DOGEUSDT"],
        )
    assert exc_info.value.reason == "forbidden_symbol"
    assert exc_info.value.details["symbol"] == "DOGEUSDT"


# 結合テスト用のダミー FeatureSet 作成
class DummyFeatureSet:
    def __init__(self):
        self.n_symbols = 2
        self.n_bars = 10
        self.n_features = 2
        self.feature_names = ["feat1", "feat2"]
        self.close = np.ones((10, 2))
        self.features = np.zeros((10, 2, 2))
        self.global_features = np.zeros((10, 2))
        self.funding_rate = np.zeros((10, 2))


def test_env_integration():
    fs = DummyFeatureSet()
    config_strict = PreTradeRiskConfig(max_leverage=0.5)
    verifier_strict = PreTradeRiskVerifier(config_strict)
    env_strict = PortfolioTradingEnv(
        fs, pre_trade_verifier=verifier_strict, initial_capital=100.0
    )
    env_strict.reset()

    # アクション [0.3, 0.3] は gross leverage = 0.6 になり、0.5 を超えるので Rejection になるはず。
    with pytest.raises(PreTradeRejection):
        env_strict.step(np.array([0.3, 0.3]))


def test_simulate_strategy_integration():
    fs = DummyFeatureSet()
    config = PreTradeRiskConfig(max_leverage=0.5)
    verifier = PreTradeRiskVerifier(config)

    # 常に一定のウェイト [0.3, 0.3] (leverage = 0.6) を返す戦略
    def risky_strategy(fs, t, w):
        return np.array([0.3, 0.3])

    with pytest.raises(PreTradeRejection):
        simulate_strategy(
            fs, risky_strategy, pre_trade_verifier=verifier, min_trade_delta=0.0
        )
