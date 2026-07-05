"""
ベースライン戦略モジュール

RLエージェントの成績は常にこれらと並記して評価する。
「④クロスセクショナルモメンタムルールに勝てないRLは存在意義がない」
という判定（ゲート2）に使う。

各ベースラインは weights(fs, t, current_weights) -> np.ndarray を実装し、
simulate_strategy() が環境と同一のコストモデルでバックテストする。
"""

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import numpy as np

from mars_lite.features.feature_pipeline import FeatureSet

WeightFn = Callable[[FeatureSet, int, np.ndarray], np.ndarray]


def flat_strategy(fs: FeatureSet, t: int, w: np.ndarray) -> np.ndarray:
    """①フラット（無ポジション）"""
    return np.zeros(fs.n_symbols)


def equal_weight_strategy(fs: FeatureSet, t: int, w: np.ndarray) -> np.ndarray:
    """②等ウェイトBuy&Hold"""
    return np.full(fs.n_symbols, 1.0 / fs.n_symbols)


def inverse_vol_strategy(fs: FeatureSet, t: int, w: np.ndarray) -> np.ndarray:
    """③ボラ逆数ウェイト（直近24バーの実現ボラ、日次リバランス）"""
    if t % 24 != 0 and w.any():
        return w
    start = max(0, t - 24)
    rets = np.diff(np.log(fs.close[start:t + 1]), axis=0)
    if len(rets) < 5:
        return np.full(fs.n_symbols, 1.0 / fs.n_symbols)
    vol = rets.std(axis=0)
    inv = 1.0 / np.clip(vol, 1e-6, None)
    return inv / inv.sum()


def cross_momentum_strategy(
    fs: FeatureSet, t: int, w: np.ndarray,
    lookback: int = 24, n_side: int = 2, rebalance_every: int = 24,
) -> np.ndarray:
    """④クロスセクショナルモメンタム（上位n_sideロング/下位n_sideショート）"""
    if t % rebalance_every != 0 and w.any():
        return w
    start = max(0, t - lookback)
    if t - start < lookback // 2:
        return np.zeros(fs.n_symbols)
    mom = np.log(fs.close[t] / fs.close[start])
    order = np.argsort(mom)
    weights = np.zeros(fs.n_symbols)
    weights[order[-n_side:]] = 0.5 / n_side
    weights[order[:n_side]] = -0.5 / n_side
    return weights


def trend_following_strategy(
    fs: FeatureSet, t: int, w: np.ndarray,
    lookback: int = 48, rebalance_every: int = 24,
) -> np.ndarray:
    """時系列モメンタム（ネット方向性あり。上昇相場では全ロング）"""
    if t % rebalance_every != 0 and w.any():
        return w
    start = max(0, t - lookback)
    if t - start < 4:
        return np.zeros(fs.n_symbols)
    mom = np.log(fs.close[t] / fs.close[start])
    scale = np.abs(mom).mean() + 1e-9
    raw = np.tanh(mom / scale)
    gross = np.abs(raw).sum()
    return raw / gross if gross > 1.0 else raw


BASELINES: Dict[str, WeightFn] = {
    "flat": flat_strategy,
    "equal_weight_bh": equal_weight_strategy,
    "inverse_vol": inverse_vol_strategy,
    "cross_momentum": cross_momentum_strategy,
    "trend_following": trend_following_strategy,
}


@dataclass
class StrategyResult:
    """戦略バックテストの結果"""
    name: str
    equity_curve: np.ndarray
    total_return: float
    sharpe: float
    max_drawdown: float
    turnover_total: float
    n_bars: int

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "total_return": self.total_return,
            "sharpe": self.sharpe,
            "max_drawdown": self.max_drawdown,
            "turnover_total": self.turnover_total,
            "n_bars": self.n_bars,
        }


def simulate_strategy(
    fs: FeatureSet,
    weight_fn: WeightFn,
    name: str = "strategy",
    fee_rate: float = 0.0005,
    spread_rate: float = 0.0002,
    impact_rate: float = 0.0001,
    min_trade_delta: float = 0.02,
    cost_multiplier: float = 1.0,
    start_idx: int = 0,
    end_idx: Optional[int] = None,
) -> StrategyResult:
    """
    PortfolioTradingEnvと同一のコストモデルで戦略をバックテスト
    """
    end_idx = end_idx if end_idx is not None else fs.n_bars - 1
    cost_per_turnover = (fee_rate + spread_rate + impact_rate) * cost_multiplier

    value = 1.0
    weights = np.zeros(fs.n_symbols)
    equity: List[float] = [value]
    rets: List[float] = []
    turnover_total = 0.0
    peak, max_dd = 1.0, 0.0

    for t in range(start_idx, end_idx):
        target = weight_fn(fs, t, weights)
        gross = np.abs(target).sum()
        if gross > 1.0:
            target = target / gross

        delta = target - weights
        delta[np.abs(delta) < min_trade_delta] = 0.0
        weights = weights + delta
        turnover = float(np.abs(delta).sum())
        turnover_total += turnover

        r_vec = fs.close[t + 1] / fs.close[t] - 1.0
        funding = float(np.sum(weights * fs.funding_rate[t + 1]))
        net = float(np.dot(weights, r_vec)) - turnover * cost_per_turnover - funding

        value *= (1.0 + net)
        rets.append(net)
        equity.append(value)
        peak = max(peak, value)
        max_dd = max(max_dd, 1.0 - value / peak)

    rets_arr = np.array(rets) if rets else np.zeros(1)
    sharpe = float(rets_arr.mean() / rets_arr.std() * np.sqrt(24 * 365)) \
        if rets_arr.std() > 0 else 0.0

    return StrategyResult(
        name=name,
        equity_curve=np.array(equity),
        total_return=value - 1.0,
        sharpe=sharpe,
        max_drawdown=max_dd,
        turnover_total=turnover_total,
        n_bars=len(rets),
    )


def run_all_baselines(fs: FeatureSet, **kwargs) -> Dict[str, StrategyResult]:
    """全ベースラインを同一条件でバックテスト"""
    return {
        name: simulate_strategy(fs, fn, name=name, **kwargs)
        for name, fn in BASELINES.items()
    }


def make_agent_weight_fn(agent, env) -> WeightFn:
    """
    学習済みPPOエージェントをWeightFnとしてラップ
    （simulate_strategyでベースラインと同一条件比較するため）

    注意: envはfsと同じFeatureSetで構築されたPortfolioTradingEnvであること。
    観測構築にenv内部状態（weights, value等）を使うため、simulate側の
    状態遷移と同期させる簡易実装として、envを内部で並走させる。
    """
    def weight_fn(fs: FeatureSet, t: int, w: np.ndarray) -> np.ndarray:
        # env内部を強制同期
        env.t = t
        env.weights = w.copy()
        obs = env._obs()
        action, _ = agent.predict(obs, deterministic=True)
        return env.project_weights(np.asarray(action, dtype=np.float64).flatten())

    return weight_fn
