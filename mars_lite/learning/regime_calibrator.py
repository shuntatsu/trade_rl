from typing import Any, Dict

import numpy as np
import optuna

from mars_lite.features.feature_pipeline import GLOBAL_FEATURES, FeatureSet
from mars_lite.learning.regime_fsm import REGIMES_8, RegimeFSM


class RegimeCalibrator:
    """
    Optuna を用いて 8状態 Regime FSM の閾値パラメータを自動較正するクラス。 (R5)
    目的関数：強気と弱気のリターン（btc_trend）の差、および高ボラと低ボラのボラティリティ（btc_vol_regime）の差を最大化し、遷移回数にペナルティを与える。
    """

    def __init__(self, n_trials: int = 100, penalty_coef: float = 1.0, seed: int = 42):
        self.n_trials = n_trials
        self.penalty_coef = penalty_coef
        self.seed = seed

    def calibrate(self, fs: FeatureSet) -> Dict[str, Any]:
        vol_idx = list(GLOBAL_FEATURES).index("btc_vol_regime")
        trend_idx = list(GLOBAL_FEATURES).index("btc_trend")

        vol_series = fs.global_features[:, vol_idx]
        trend_series = fs.global_features[:, trend_idx]

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        def objective(trial):
            t_trend_low = trial.suggest_float("t_trend_low", 0.1, 1.0)
            t_trend_extreme = trial.suggest_float(
                "t_trend_extreme", t_trend_low + 0.1, 2.5
            )
            t_vol = trial.suggest_float("t_vol", -1.0, 1.0)
            persistence_bars = trial.suggest_int("persistence_bars", 1, 20)

            fsm = RegimeFSM(
                t_trend_low=t_trend_low,
                t_trend_extreme=t_trend_extreme,
                t_vol=t_vol,
                persistence_bars=persistence_bars,
            )

            states = fsm.classify_series(trend_series, vol_series)

            # 各状態が1つも選ばれなかった場合などのガード
            bull_mask = np.isin(states, ["extreme_bull", "bull_high", "bull_low"])
            bear_mask = np.isin(states, ["extreme_bear", "bear_high", "bear_low"])
            high_mask = np.isin(states, ["bull_high", "bear_high", "range_high"])
            low_mask = np.isin(states, ["bull_low", "bear_low", "range_low"])

            if (
                not np.any(bull_mask)
                or not np.any(bear_mask)
                or not np.any(high_mask)
                or not np.any(low_mask)
            ):
                return -100.0

            mu_ret_bull = np.mean(trend_series[bull_mask])
            mu_ret_bear = np.mean(trend_series[bear_mask])
            mu_vol_high = np.mean(vol_series[high_mask])
            mu_vol_low = np.mean(vol_series[low_mask])

            sep_trend = mu_ret_bull - mu_ret_bear
            sep_vol = mu_vol_high - mu_vol_low

            # 遷移回数比率
            n = len(states)
            if n <= 1:
                p_trans = 0.0
            else:
                transitions = np.sum(states[1:] != states[:-1])
                p_trans = transitions / (n - 1)

            score = sep_trend + sep_vol - self.penalty_coef * p_trans
            return score

        sampler = optuna.samplers.TPESampler(seed=self.seed)
        study = optuna.create_study(direction="maximize", sampler=sampler)
        study.optimize(objective, n_trials=self.n_trials)

        return study.best_params
