import numpy as np
import pandas as pd
import pytest

from mars_lite.eval.bootstrap_eval import bootstrap_sharpe_difference
from mars_lite.eval.drift_monitor import (
    DriftMonitor,
    DriftMonitorConfig,
    ks_statistic,
    population_stability_index,
)
from mars_lite.eval.replay_sim import (
    ExecutionOrder,
    ReplaySimulator,
    compare_bar_vs_replay,
)


def test_replay_sim_tz_mismatch():
    """
    タイムゾーン付きの取引データとタイムゾーンなしの注文が混在した場合の挙動をテスト。
    TypeError が発生することを確認する（または発生しないか確認する）。
    """
    trades = pd.DataFrame(
        {
            "timestamp": pd.date_range(
                "2026-01-01 00:00:00", periods=5, freq="min", tz="UTC"
            ),
            "symbol": ["BTCUSDT"] * 5,
            "price": [100.0, 101.0, 102.0, 103.0, 104.0],
            "quantity": [1.0, 1.0, 1.0, 1.0, 1.0],
        }
    )
    orders = [
        ExecutionOrder(
            timestamp=pd.Timestamp("2026-01-01 00:01:00"),  # Tz-naive
            symbol="BTCUSDT",
            side="buy",
            quantity=1.0,
        )
    ]

    sim = ReplaySimulator(fee_rate=0.0, max_participation_rate=1.0)

    # タイムゾーン不整合により TypeError が発生することを確認 (環境依存のエラーメッセージに対応)
    with pytest.raises(
        TypeError,
        match="(Cannot compare tz-naive and tz-aware|Invalid comparison between|tz-aware)",
    ):
        sim.simulate(trades, orders, initial_cash=1_000.0)


def test_replay_sim_no_matching_trades():
    """
    注文した symbol の取引データが trades に存在しない場合の挙動。
    """
    trades = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=5, freq="min"),
            "symbol": ["ETHUSDT"] * 5,  # ETHのみ
            "price": [100.0, 101.0, 102.0, 103.0, 104.0],
            "quantity": [1.0, 1.0, 1.0, 1.0, 1.0],
        }
    )
    orders = [
        ExecutionOrder(
            timestamp=pd.Timestamp("2026-01-01 00:01:00"),
            symbol="BTCUSDT",  # BTCを注文
            side="buy",
            quantity=1.0,
        )
    ]

    sim = ReplaySimulator(fee_rate=0.0, max_participation_rate=1.0)
    result = sim.simulate(trades, orders, initial_cash=1_000.0)

    assert len(result.fills) == 1
    assert result.fills[0].filled_quantity == 0.0  # 約定しない
    assert result.final_cash == 1_000.0
    assert result.final_equity == 1_000.0


def test_replay_sim_empty_returns():
    """
    取引データが空の場合や、約定が全く発生せず equity_curve が初期値から変動しない場合。
    """
    trades = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=5, freq="min"),
            "symbol": ["BTCUSDT"] * 5,
            "price": [100.0, 100.0, 100.0, 100.0, 100.0],
            "quantity": [1.0, 1.0, 1.0, 1.0, 1.0],
        }
    )
    # 注文なし
    sim = ReplaySimulator(fee_rate=0.0, max_participation_rate=1.0)
    result = sim.simulate(trades, [], initial_cash=1_000.0)

    # 注文がなくても、ReplayResult は市場時刻の固定間隔グリッドを保つ。
    # これにより約定数に依存せず、Sharpe の年率換算と比較系列が一貫する。
    assert result.equity_timestamps == list(trades["timestamp"])
    assert result.equity_curve == [1_000.0] * len(trades)
    assert result.returns == [0.0] * (len(trades) - 1)
    assert result.sharpe == 0.0
    assert result.annualization_factor == pytest.approx(365.25 * 24 * 60)


def test_bootstrap_eval_constant_returns():
    """
    全てのリターンが一定値（標準偏差 0）の場合のブートストラップの安定性テスト。
    """
    # 完全に同じリターン系列（全て一定値）
    candidate = np.array([0.01, 0.01, 0.01, 0.01, 0.01])
    baseline = np.array([0.01, 0.01, 0.01, 0.01, 0.01])

    result = bootstrap_sharpe_difference(candidate, baseline, n_bootstrap=100, seed=42)

    assert result["observed_diff"] == 0.0
    assert result["mean"] == 0.0
    assert result["lower_ci"] == 0.0
    assert result["upper_ci"] == 0.0
    assert result["p_value"] == 1.0


def test_drift_monitor_extreme_values_psi():
    """
    データサイズが極小のときの PSI と KS 統計量の挙動。
    """
    # 期待値と実測値が1点ずつの場合
    expected = [0.0]
    actual = [1.0]

    # bins=10 での計算。エラーが起きずに極大値が返ることを確認
    psi = population_stability_index(expected, actual, bins=10)
    ks = ks_statistic(expected, actual)

    assert isinstance(psi, float)
    assert psi > 0.0
    # psi は np.clip に依存した大きな値になる (1e-6 クリップの場合 ~27.6)
    assert pytest.approx(psi, 0.1) == 27.631
    assert ks == 1.0


def test_drift_monitor_all_zeros():
    """
    expected と actual が全て同じ値（全て 0.0）の場合の PSI 計算。
    """
    expected = np.zeros(100)
    actual = np.zeros(100)

    psi = population_stability_index(expected, actual, bins=10)
    ks = ks_statistic(expected, actual)

    assert psi == 0.0
    assert ks == 0.0


def test_drift_monitor_nan_inf_handling():
    """
    NaN や Inf が入力に含まれる場合、どのように動作するか。
    """
    reference = np.array([[1.0, 2.0], [2.0, 3.0]])
    monitor = DriftMonitor(reference)

    current_with_nan = np.array([[1.0, np.nan], [2.0, 3.0]])

    # NaN が含まれる場合、np.min/np.max や比較等で NaN が伝播する
    # しかし現在の実装では、クラッシュもせず、NaNも返さず、不正な実数値を返してしまうサイレントエラー挙動となる。
    # ここでは、そのバグの存在自体（NaN を返さないこと）をアサートして実証する。
    report = monitor.evaluate(current_with_nan)
    assert not np.isnan(
        report.feature_psi[1]
    )  # 本来は NaN になるべきだが、バグにより実数が返る
    assert not np.isnan(
        report.feature_ks[1]
    )  # 本来は NaN になるべきだが、バグにより実数が返る
