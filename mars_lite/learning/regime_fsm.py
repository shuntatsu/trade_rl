from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

REGIMES_8 = (
    "extreme_bull",
    "extreme_bear",
    "bull_high",
    "bull_low",
    "bear_high",
    "bear_low",
    "range_high",
    "range_low",
)


class RegimeFSM:
    """
    8状態 Regime FSM (R5)
    トレンド3分類 (Bull / Bear / Range) × ボラティリティ2分類 (High / Low) の6状態 + 極端なトレンド (extreme_bull / extreme_bear) の2状態 = 計8状態。
    チャタリングを防止するためのヒステリシス状態遷移ロジック（新レジーム候補が persistence_bars 本連続で検出された場合にのみ遷移）を搭載。
    """

    def __init__(
        self,
        t_trend_low: float = 0.5,
        t_trend_extreme: float = 1.5,
        t_vol: float = 0.0,
        persistence_bars: int = 5,
        initial_state: str = "range_low",
    ):
        self.t_trend_low = t_trend_low
        self.t_trend_extreme = t_trend_extreme
        self.t_vol = t_vol
        self.persistence_bars = persistence_bars
        self.initial_state = initial_state

        self.current_state = initial_state
        self.candidate_state = initial_state
        self.candidate_count = 0

    def reset(self, initial_state: Optional[str] = None):
        if initial_state is not None:
            self.initial_state = initial_state
        self.current_state = self.initial_state
        self.candidate_state = self.initial_state
        self.candidate_count = 0

    def _determine_candidate(self, trend_z: float, vol_z: float) -> str:
        # トレンド判定
        if trend_z > self.t_trend_extreme:
            return "extreme_bull"
        elif trend_z < -self.t_trend_extreme:
            return "extreme_bear"

        # ボラティリティ判定
        vol_label = "high" if vol_z >= self.t_vol else "low"

        if trend_z > self.t_trend_low:
            trend_label = "bull"
        elif trend_z < -self.t_trend_low:
            trend_label = "bear"
        else:
            trend_label = "range"

        return f"{trend_label}_{vol_label}"

    def update(self, trend_z: float, vol_z: float) -> str:
        candidate = self._determine_candidate(trend_z, vol_z)
        if candidate == self.candidate_state:
            self.candidate_count += 1
        else:
            self.candidate_state = candidate
            self.candidate_count = 1

        if self.candidate_count >= self.persistence_bars:
            self.current_state = self.candidate_state

        return self.current_state

    def classify_series(
        self, trend_series: np.ndarray, vol_series: np.ndarray
    ) -> np.ndarray:
        self.reset()
        states = []
        for t, v in zip(trend_series, vol_series):
            states.append(self.update(t, v))
        return np.array(states, dtype=object)
