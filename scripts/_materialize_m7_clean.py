from __future__ import annotations

from pathlib import Path


def replace_once(path: str, old: str, new: str) -> None:
    file_path = Path(path)
    text = file_path.read_text(encoding="utf-8")
    if new in text:
        return
    if old not in text:
        raise RuntimeError(f"expected source block not found in {path}")
    file_path.write_text(text.replace(old, new, 1), encoding="utf-8")


def patch_baseline_risk_wiring() -> None:
    replace_once(
        "mars_lite/learning/baselines.py",
        """        if pre_trade_verifier is not None:\n            pre_trade_verifier.validate(target, value)\n\n        delta = target - weights\n        delta[np.abs(delta) < min_trade_delta] = 0.0\n        weights = weights + delta\n""",
        """        delta = target - weights\n        delta[np.abs(delta) < min_trade_delta] = 0.0\n        next_weights = weights + delta\n\n        if pre_trade_verifier is not None:\n            pre_trade_verifier.validate(\n                next_weights,\n                value,\n                symbols=fs.symbols,\n                current_weights=weights,\n            )\n\n        weights = next_weights\n""",
    )


def patch_legacy_replay_test() -> None:
    replace_once(
        "tests/test_adversarial_m4.py",
        """    # 注文がない場合、equity_curve は [initial_cash, final_equity] になる。\n    # どちらも 1000.0 なので returns は [0.0]。\n    assert len(result.returns) == 1\n    assert result.returns[0] == 0.0\n    assert result.sharpe == 0.0  # std が 0 なので 0.0 になるはず\n""",
        """    # 注文がなくても、ReplayResult は市場時刻の固定間隔グリッドを保つ。\n    # これにより約定数に依存せず、Sharpe の年率換算と比較系列が一貫する。\n    assert result.equity_timestamps == list(trades[\"timestamp\"])\n    assert result.equity_curve == [1_000.0] * len(trades)\n    assert result.returns == [0.0] * (len(trades) - 1)\n    assert result.sharpe == 0.0\n    assert result.annualization_factor == pytest.approx(365.25 * 24 * 60)\n""",
    )


def add_delta_integration_tests() -> None:
    path = Path("tests/test_pre_trade_risk.py")
    text = path.read_text(encoding="utf-8")
    if "test_env_integration_uses_execution_delta_for_minimum_order" in text:
        return
    text += """


def test_env_integration_uses_execution_delta_for_minimum_order():
    fs = DummyFeatureSet()
    verifier = PreTradeRiskVerifier(PreTradeRiskConfig(min_order_notional=10.0))
    env = PortfolioTradingEnv(
        fs,
        pre_trade_verifier=verifier,
        initial_capital=100.0,
        min_trade_delta=0.0,
        lambda_turnover=0.0,
    )
    env.reset(options={"start_idx": 0})
    env.step(np.array([0.2, 0.0]))
    with pytest.raises(PreTradeRejection) as exc:
        env.step(np.array([0.21, 0.0]))
    assert exc.value.reason == "min_order_notional_not_met"
    assert exc.value.details["symbol"] == "BTCUSDT"
    assert 0.0 < exc.value.details["order_notional"] < 10.0


def test_simulate_strategy_uses_post_threshold_execution_delta():
    fs = DummyFeatureSet()
    verifier = PreTradeRiskVerifier(PreTradeRiskConfig(min_order_notional=0.1))

    def small_rebalance_strategy(fs, t, w):
        if not w.any():
            return np.array([0.2, 0.0])
        return np.array([0.21, 0.0])

    with pytest.raises(PreTradeRejection) as exc:
        simulate_strategy(
            fs,
            small_rebalance_strategy,
            pre_trade_verifier=verifier,
            min_trade_delta=0.0,
        )
    assert exc.value.reason == "min_order_notional_not_met"
    assert exc.value.details["symbol"] == "BTCUSDT"
    assert 0.0 < exc.value.details["order_notional"] < 0.1


def test_simulate_strategy_validates_after_min_trade_filter():
    fs = DummyFeatureSet()
    verifier = PreTradeRiskVerifier(PreTradeRiskConfig(min_order_notional=0.1))

    def filtered_rebalance_strategy(fs, t, w):
        if not w.any():
            return np.array([0.2, 0.0])
        return np.array([0.21, 0.0])

    result = simulate_strategy(
        fs,
        filtered_rebalance_strategy,
        pre_trade_verifier=verifier,
        min_trade_delta=0.02,
    )
    assert result.n_bars == fs.n_bars - 1
"""
    path.write_text(text, encoding="utf-8")


def main() -> None:
    patch_baseline_risk_wiring()
    patch_legacy_replay_test()
    add_delta_integration_tests()


if __name__ == "__main__":
    main()
