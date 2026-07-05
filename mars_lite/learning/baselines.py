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


def oracle_dp_strategy(
    fs: FeatureSet,
    name: str = "oracle_dp",
    fee_rate: float = 0.0005,
    spread_rate: float = 0.0002,
    impact_rate: float = 0.0001,
    positions: tuple = (-1.0, 0.0, 1.0),
    allow_short: bool = True,
    cost_multiplier: float = 1.0,
    start_idx: int = 0,
    end_idx: Optional[int] = None,
) -> StrategyResult:
    """
    手数料込みの理論上限（オラクル）を動的計画法で厳密に求める。

    各銘柄に資本を 1/N ずつ割り当てた独立サブアカウントとして扱い、各
    サブアカウントで「未来を完全に知った上で、手数料を払ってでもポジションを
    持つ/反転させる価値があるか」をDP（Viterbi型のトレリス最短経路）で解く。

    - 状態: サブアカウント内ポジション p ∈ positions（既定 {-1,0,+1} = 全ショート/
      フラット/全ロング）。allow_short=Falseなら {0,+1}
    - バー報酬: log(1 + p·r_t)（対数で複利を加法化）
    - 遷移コスト: env と同じ (手数料+スプレッド)·|Δp| + impact·|Δp|^1.5 を
      log(1-cost) として減算。**微小な山谷は手数料に負けて取らない**ため、
      閾値を超える山と谷だけを取りにいく最適経路になる

    ポートフォリオ資産 = 各サブアカウント資産の平均（等資本のため加法的）。
    これは同一コスト・同一レバレッジ制約下での達成可能な上限であり、RLの
    「捕捉率 = RL収益 / オラクル収益」の分母に使える。
    """
    end_idx = end_idx if end_idx is not None else fs.n_bars - 1
    pos = np.array([p for p in positions if allow_short or p >= 0], dtype=np.float64)
    n_states = len(pos)
    impact_coef = (impact_rate / (0.1 ** 0.5))  # 線形率→sqrt則係数（execution.pyと一致）
    lin = (fee_rate + spread_rate) * cost_multiplier
    imp = impact_coef * cost_multiplier

    def trans_logcost(p_prev: float, p_new: float) -> float:
        d = abs(p_new - p_prev)
        if d == 0.0:
            return 0.0
        cost = lin * d + imp * (d ** 1.5)
        return np.log(max(1.0 - cost, 1e-9))

    n_sym = fs.n_symbols
    T = end_idx - start_idx
    if T < 1:
        raise ValueError("oracle_dp: 区間が短すぎます")

    # 各サブアカウントの資産推移（後で平均してポートフォリオ資産に）
    sleeve_equity = np.ones((T + 1, n_sym))
    turnover_total = 0.0

    for i in range(n_sym):
        r = fs.close[start_idx + 1:end_idx + 1, i] / fs.close[start_idx:end_idx, i] - 1.0
        # ホールド対数報酬 (T, n_states)
        hold = np.log(np.clip(1.0 + np.outer(r, pos), 1e-9, None))

        dp = np.full((T, n_states), -np.inf)
        ptr = np.zeros((T, n_states), dtype=np.int64)
        # t=0: 初期ポジション0（フラット）から遷移
        for j in range(n_states):
            dp[0, j] = trans_logcost(0.0, pos[j]) + hold[0, j]
        for t in range(1, T):
            for j in range(n_states):
                best_k, best_v = 0, -np.inf
                for k in range(n_states):
                    v = dp[t - 1, k] + trans_logcost(pos[k], pos[j])
                    if v > best_v:
                        best_v, best_k = v, k
                dp[t, j] = best_v + hold[t, j]
                ptr[t, j] = best_k

        # 最良終端からバックトラックして最適ポジション経路を復元
        path = np.zeros(T, dtype=np.int64)
        path[T - 1] = int(np.argmax(dp[T - 1]))
        for t in range(T - 1, 0, -1):
            path[t - 1] = ptr[t, path[t]]

        # 経路に沿ってサブアカウント資産を再構成（コスト→ホールドの順）
        prev_p = 0.0
        val = 1.0
        for t in range(T):
            p = pos[path[t]]
            d = abs(p - prev_p)
            turnover_total += d / n_sym          # ポートフォリオ比の回転
            cost = lin * d + imp * (d ** 1.5) if d > 0 else 0.0
            val *= (1.0 - cost) * (1.0 + p * r[t])
            sleeve_equity[t + 1, i] = val
            prev_p = p

    # ポートフォリオ資産 = サブアカウント資産の平均（等資本）
    equity = sleeve_equity.mean(axis=1)
    rets = np.diff(equity) / equity[:-1]
    sharpe = float(rets.mean() / rets.std() * np.sqrt(24 * 365)) \
        if rets.std() > 0 else 0.0
    peak = np.maximum.accumulate(equity)
    max_dd = float((1.0 - equity / peak).max())

    return StrategyResult(
        name=name,
        equity_curve=equity,
        total_return=float(equity[-1] - 1.0),
        sharpe=sharpe,
        max_drawdown=max_dd,
        turnover_total=float(turnover_total),
        n_bars=T,
    )


def run_all_baselines(fs: FeatureSet, include_oracle: bool = True,
                      **kwargs) -> Dict[str, StrategyResult]:
    """全ベースラインを同一条件でバックテスト（include_oracleでDPオラクル上限も併記）"""
    out = {
        name: simulate_strategy(fs, fn, name=name, **kwargs)
        for name, fn in BASELINES.items()
    }
    if include_oracle:
        oracle_kwargs = {k: v for k, v in kwargs.items()
                         if k in ("fee_rate", "spread_rate", "impact_rate",
                                  "cost_multiplier", "start_idx", "end_idx")}
        out["oracle_dp"] = oracle_dp_strategy(fs, **oracle_kwargs)
    return out


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
