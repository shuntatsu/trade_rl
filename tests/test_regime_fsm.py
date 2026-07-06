import numpy as np
import pytest

from mars_lite.learning.regime_calibrator import RegimeCalibrator
from mars_lite.learning.regime_ensemble import RegimeEnsemble, regime_start_pools
from mars_lite.learning.regime_fsm import REGIMES_8, RegimeFSM


def test_regime_fsm_initialization_and_static():
    fsm = RegimeFSM(
        t_trend_low=0.5,
        t_trend_extreme=1.5,
        t_vol=0.0,
        persistence_bars=3,
        initial_state="range_low",
    )
    assert fsm.current_state == "range_low"

    # 静的分類候補判定のテスト
    assert fsm._determine_candidate(2.0, 0.5) == "extreme_bull"
    assert fsm._determine_candidate(-2.0, -0.5) == "extreme_bear"
    assert fsm._determine_candidate(0.8, 0.5) == "bull_high"
    assert fsm._determine_candidate(0.8, -0.5) == "bull_low"
    assert fsm._determine_candidate(-0.8, 0.5) == "bear_high"
    assert fsm._determine_candidate(-0.8, -0.5) == "bear_low"
    assert fsm._determine_candidate(0.2, 0.5) == "range_high"
    assert fsm._determine_candidate(0.2, -0.5) == "range_low"


def test_regime_fsm_transitions_to_all_8_states():
    fsm = RegimeFSM(
        t_trend_low=0.5,
        t_trend_extreme=1.5,
        t_vol=0.0,
        persistence_bars=1,
        initial_state="range_low",
    )
    scenarios = {
        "extreme_bull": (2.0, 0.5),
        "extreme_bear": (-2.0, -0.5),
        "bull_high": (0.8, 0.5),
        "bull_low": (0.8, -0.5),
        "bear_high": (-0.8, 0.5),
        "bear_low": (-0.8, -0.5),
        "range_high": (0.2, 0.5),
        "range_low": (0.2, -0.5),
    }

    visited = set()
    for expected_state, (trend_z, vol_z) in scenarios.items():
        assert fsm.update(trend_z, vol_z) == expected_state
        visited.add(fsm.current_state)

    assert visited == set(REGIMES_8)


def test_regime_fsm_persistence_and_hysteresis():
    # persistence_bars = 3 の場合、3本連続で同じ候補が出ないと遷移しない
    fsm = RegimeFSM(
        t_trend_low=0.5,
        t_trend_extreme=1.5,
        t_vol=0.0,
        persistence_bars=3,
        initial_state="range_low",
    )

    # 初期状態
    assert fsm.current_state == "range_low"

    # 1本目: extreme_bull 候補
    assert fsm.update(2.0, 0.5) == "range_low"
    # 2本目: extreme_bull 候補
    assert fsm.update(2.0, 0.5) == "range_low"
    # 3本目: extreme_bull 候補 -> ここで遷移するはず
    assert fsm.update(2.0, 0.5) == "extreme_bull"

    # 途中で別の候補が混ざるとリセットされる
    assert fsm.update(0.0, -0.5) == "extreme_bull"  # 候補: range_low, カウント: 1
    assert fsm.update(2.0, 0.5) == "extreme_bull"  # 候補: extreme_bull, カウント: 1
    assert fsm.update(2.0, 0.5) == "extreme_bull"  # 候補: extreme_bull, カウント: 2
    assert (
        fsm.update(2.0, 0.5) == "extreme_bull"
    )  # 候補: extreme_bull, カウント: 3 -> 維持（すでに extreme_bull なので変化なし）


def test_regime_ensemble_fallback_routing():
    # ダミーのエージェント
    class DummyAgent:
        def __init__(self, name):
            self.name = name

        def predict(self, obs, deterministic=True):
            return np.array([1.0]), None

    # bull_high と range_low のみ specialist が存在し、他は None の辞書
    specialists = {
        "bull_high": DummyAgent("bull_high"),
        "range_low": DummyAgent("range_low"),
    }
    generalist = DummyAgent("generalist")

    obs_layout = {
        "n_symbols": 2,
        "n_per_symbol": 5,
    }

    # 8状態 RegimeFSM
    fsm = RegimeFSM(
        t_trend_low=0.5,
        t_trend_extreme=1.5,
        t_vol=0.0,
        persistence_bars=1,  # すぐ遷移するようにする
        initial_state="range_low",
    )

    ensemble = RegimeEnsemble(
        specialists=specialists,
        generalist=generalist,
        obs_layout=obs_layout,
        n_raw_globals=6,
        fsm=fsm,
    )

    # ルーティング動作の検証
    # extreme_bull -> FALLBACK_ROUTES: ["bull_high", ...] -> bull_high にルーティングされるはず
    agent = ensemble._agent_for("extreme_bull")
    assert agent.name == "bull_high"

    # range_low -> specialists に存在するので range_low にルーティングされるはず
    agent = ensemble._agent_for("range_low")
    assert agent.name == "range_low"

    # extreme_bear -> FALLBACK_ROUTES に含まれる specialists は range_low があるはず
    # なので range_low にルーティングされるはず
    agent = ensemble._agent_for("extreme_bear")
    assert agent.name == "range_low"


def test_regime_calibrator_optimization():
    # ダミーの FeatureSet を作成
    n_bars = 100
    global_features = np.zeros((n_bars, 6))

    # 前半を上昇・高ボラ、後半を下降・低ボラにする
    global_features[:50, 4] = 1.0  # btc_vol_regime = High
    global_features[:50, 5] = 1.2  # btc_trend = Bull
    global_features[50:, 4] = -1.0  # btc_vol_regime = Low
    global_features[50:, 5] = -1.2  # btc_trend = Bear

    # FeatureSet のダミー
    class DummyFeatureSet:
        def __init__(self):
            self.global_features = global_features
            self.n_bars = n_bars
            self.n_symbols = 2
            self.n_features = 10

    fs = DummyFeatureSet()

    calibrator = RegimeCalibrator(n_trials=10, penalty_coef=1.0, seed=42)
    best_params = calibrator.calibrate(fs)

    assert "t_trend_low" in best_params
    assert "t_trend_extreme" in best_params
    assert "t_vol" in best_params
    assert "persistence_bars" in best_params

    # パラメータ範囲チェック
    assert 0.1 <= best_params["t_trend_low"] <= 1.0
    assert best_params["t_trend_extreme"] >= best_params["t_trend_low"]
    assert -1.0 <= best_params["t_vol"] <= 1.0
    assert 1 <= best_params["persistence_bars"] <= 20
