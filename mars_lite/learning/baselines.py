"""
ベースライン戦略モジュール

RLエージェントの成績は常にこれらと並記して評価する。
「④クロスセクショナルモメンタムルールに勝てないRLは存在意義がない」
という判定（ゲート2）に使う。

各ベースラインは weights(fs, t, current_weights) -> np.ndarray を実装し、
simulate_strategy() が環境と同一のコストモデルでバックテストする。
"""

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from mars_lite.features.feature_pipeline import FeatureSet
from mars_lite.trading.execution import make_execution_model
from mars_lite.trading.post_processor import BARS_PER_YEAR_1H

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
    # 環境（PortfolioTradingEnv）・oracle_dp と厳密に同一の sqrt-impact + TWAP
    # 執行モデルを使う。以前は線形コスト（turnover×固定率）だったが、大きな
    # リバランスをするベースライン（cross_momentum/trend_following は ±0.5 の
    # ジャンプ）が RL/oracle より 7〜27% 安いコストしか払わず、ゲート2の
    # 「RL は全ベースラインに勝てるか」判定をベースライン有利に歪めていた。
    exec_model = make_execution_model(
        fee_rate=fee_rate, spread_rate=spread_rate,
        impact_rate=impact_rate, cost_multiplier=cost_multiplier,
    )

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
        cost = exec_model.cost_fraction(delta)
        net = float(np.dot(weights, r_vec)) - cost - funding

        value *= (1.0 + net)
        rets.append(net)
        equity.append(value)
        peak = max(peak, value)
        max_dd = max(max_dd, 1.0 - value / peak)

    rets_arr = np.array(rets) if rets else np.zeros(1)
    sharpe = float(rets_arr.mean() / rets_arr.std() * np.sqrt(BARS_PER_YEAR_1H)) \
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


def _true_returns(fs: FeatureSet, start_idx: int, end_idx: int) -> np.ndarray:
    """[start_idx, end_idx) 区間の1バー単純リターン (T, n_symbols)"""
    return fs.close[start_idx + 1:end_idx + 1] / fs.close[start_idx:end_idx] - 1.0


def oracle_dp_paths(
    fs: FeatureSet,
    signal: Optional[np.ndarray] = None,
    positions: tuple = (-1.0, 0.0, 1.0),
    allow_short: bool = True,
    fee_rate: float = 0.0005,
    spread_rate: float = 0.0002,
    impact_rate: float = 0.0001,
    cost_multiplier: float = 1.0,
    start_idx: int = 0,
    end_idx: Optional[int] = None,
    decision_every: int = 1,
) -> np.ndarray:
    """
    銘柄別の最適ポジション経路をDP（Viterbi型トレリス最短経路）で求める。

    各銘柄を独立サブアカウントとして扱い、「手数料を払ってでもポジションを
    持つ/反転させる価値があるか」を解く。`signal` を渡すとその値を
    バー報酬 log(1+p·signal_t) の駆動源として使う（ノイズ入りオラクル用。
    経路選択は signal に基づくが、損益実現は呼び出し側が真のリターンで
    別途行う）。省略時は真のリターン自体を signal として使う（完全オラクル）。

    decision_every > 1 の場合、ポジション変更は
    `t % decision_every == 0` のバーでのみ許可する
    （PortfolioTradingEnv.decision_every と同じ意味論）。低頻度アルファを
    毎バーの回転コストで削らないための整合。

    Returns:
        (T, n_symbols) の実現ポジション値配列（positions の要素）
    """
    end_idx = end_idx if end_idx is not None else fs.n_bars - 1
    pos = np.array([p for p in positions if allow_short or p >= 0], dtype=np.float64)
    n_states = len(pos)
    # 係数はExecutionModelのfactory関数から取得（コスト式の単一の正）。
    # DPの内側ループはスカラー演算のホットパスなのでcost_fraction呼び出しは
    # 使わず、係数のみ共有する。
    exec_model = make_execution_model(
        fee_rate=fee_rate, spread_rate=spread_rate,
        impact_rate=impact_rate, cost_multiplier=cost_multiplier,
    )
    lin = (exec_model.fee_rate + exec_model.spread_rate) * exec_model.cost_multiplier
    imp = exec_model.impact_coef * exec_model.cost_multiplier
    decision_every = max(1, int(decision_every))

    def trans_logcost(p_prev: float, p_new: float, is_decision_bar: bool) -> float:
        d = abs(p_new - p_prev)
        if d == 0.0:
            return 0.0
        if not is_decision_bar:
            return -np.inf  # 非意思決定バーではポジション変更を禁止
        cost = lin * d + imp * (d ** 1.5)
        return np.log(max(1.0 - cost, 1e-9))

    n_sym = fs.n_symbols
    T = end_idx - start_idx
    if T < 1:
        raise ValueError("oracle_dp: 区間が短すぎます")

    sig = signal if signal is not None else _true_returns(fs, start_idx, end_idx)
    paths = np.zeros((T, n_sym), dtype=np.float64)
    is_decision = [(t % decision_every == 0) for t in range(T)]

    for i in range(n_sym):
        s = sig[:, i]
        hold = np.log(np.clip(1.0 + np.outer(s, pos), 1e-9, None))

        dp = np.full((T, n_states), -np.inf)
        ptr = np.zeros((T, n_states), dtype=np.int64)
        for j in range(n_states):
            dp[0, j] = trans_logcost(0.0, pos[j], is_decision[0]) + hold[0, j]
        for t in range(1, T):
            for j in range(n_states):
                best_k, best_v = 0, -np.inf
                for k in range(n_states):
                    v = dp[t - 1, k] + trans_logcost(pos[k], pos[j], is_decision[t])
                    if v > best_v:
                        best_v, best_k = v, k
                dp[t, j] = best_v + hold[t, j]
                ptr[t, j] = best_k

        path_idx = np.zeros(T, dtype=np.int64)
        path_idx[T - 1] = int(np.argmax(dp[T - 1]))
        for t in range(T - 1, 0, -1):
            path_idx[t - 1] = ptr[t, path_idx[t]]
        paths[:, i] = pos[path_idx]

    return paths


def _simulate_positions(
    fs: FeatureSet,
    paths: np.ndarray,
    fee_rate: float = 0.0005,
    spread_rate: float = 0.0002,
    impact_rate: float = 0.0001,
    cost_multiplier: float = 1.0,
    start_idx: int = 0,
    end_idx: Optional[int] = None,
) -> Tuple[np.ndarray, float]:
    """
    銘柄別ポジション経路 paths (T, n_symbols) を**真のリターン**で実現し、
    サブアカウント平均のポートフォリオ資産曲線を返す。

    oracle_dp_paths の signal が真のリターンと異なる場合（ノイズ入り
    オラクル）に、選択された経路の実際の損益を正しく評価するために使う。
    """
    end_idx = end_idx if end_idx is not None else fs.n_bars - 1
    n_sym = fs.n_symbols
    T = end_idx - start_idx
    # 係数はExecutionModelのfactory関数から取得（コスト式の単一の正、oracle_dp_pathsと同じ）
    exec_model = make_execution_model(
        fee_rate=fee_rate, spread_rate=spread_rate,
        impact_rate=impact_rate, cost_multiplier=cost_multiplier,
    )
    lin = (exec_model.fee_rate + exec_model.spread_rate) * exec_model.cost_multiplier
    imp = exec_model.impact_coef * exec_model.cost_multiplier
    r = _true_returns(fs, start_idx, end_idx)

    sleeve_equity = np.ones((T + 1, n_sym))
    turnover_total = 0.0
    for i in range(n_sym):
        prev_p = 0.0
        val = 1.0
        for t in range(T):
            p = paths[t, i]
            d = abs(p - prev_p)
            turnover_total += d / n_sym
            cost = lin * d + imp * (d ** 1.5) if d > 0 else 0.0
            val *= (1.0 - cost) * (1.0 + p * r[t, i])
            sleeve_equity[t + 1, i] = val
            prev_p = p

    equity = sleeve_equity.mean(axis=1)
    return equity, turnover_total


def _equity_to_result(name: str, equity: np.ndarray, turnover_total: float) -> StrategyResult:
    rets = np.diff(equity) / equity[:-1]
    sharpe = float(rets.mean() / rets.std() * np.sqrt(BARS_PER_YEAR_1H)) \
        if rets.std() > 0 else 0.0
    peak = np.maximum.accumulate(equity)
    max_dd = float((1.0 - equity / peak).max())
    return StrategyResult(
        name=name, equity_curve=equity, total_return=float(equity[-1] - 1.0),
        sharpe=sharpe, max_drawdown=max_dd, turnover_total=float(turnover_total),
        n_bars=len(equity) - 1,
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
    decision_every: int = 1,
) -> StrategyResult:
    """
    手数料込みの理論上限（完全オラクル）を動的計画法で厳密に求める。

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

    注意: IC=1（完全な未来予知）を要求する到達不能な天井。現実的な比較には
    noisy_oracle_strategy（目標ICを指定した劣化オラクル）を併用すること。
    """
    paths = oracle_dp_paths(
        fs, signal=None, positions=positions, allow_short=allow_short,
        fee_rate=fee_rate, spread_rate=spread_rate, impact_rate=impact_rate,
        cost_multiplier=cost_multiplier, start_idx=start_idx, end_idx=end_idx,
        decision_every=decision_every,
    )
    equity, turnover_total = _simulate_positions(
        fs, paths, fee_rate=fee_rate, spread_rate=spread_rate,
        impact_rate=impact_rate, cost_multiplier=cost_multiplier,
        start_idx=start_idx, end_idx=end_idx,
    )
    return _equity_to_result(name, equity, turnover_total)


def calibrate_noise_to_ic(
    fwd_ret: np.ndarray,
    target_ic: float,
    seed: int = 0,
    n_avg: int = 5,
    tol: float = 0.005,
    max_iter: int = 40,
) -> float:
    """
    目標rank IC（ノイズ付き信号 vs 真の前方リターン）に近づくノイズsigmaを
    二分探索で求める。sigma=0でIC=1、sigma→∞でIC→0の単調減少関係を利用。
    """
    from mars_lite.features.signal_check import _rank_ic

    fwd = np.asarray(fwd_ret, dtype=np.float64).flatten()
    std = float(np.std(fwd))
    if std <= 1e-12:
        return 1.0
    if target_ic <= 0:
        return std * 50.0

    rng = np.random.default_rng(seed)

    def ic_at(sigma: float) -> float:
        ics = [
            _rank_ic(fwd + rng.normal(0.0, sigma, size=fwd.shape), fwd)
            for _ in range(n_avg)
        ]
        return float(np.mean(ics))

    lo, hi = 0.0, std * 50.0
    for _ in range(max_iter):
        mid = (lo + hi) / 2.0
        ic = ic_at(mid)
        if abs(ic - target_ic) < tol:
            return mid
        if ic > target_ic:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def noisy_oracle_strategy(
    fs: FeatureSet,
    target_ic: float = 0.05,
    seed: int = 0,
    n_draws: int = 3,
    name: Optional[str] = None,
    fee_rate: float = 0.0005,
    spread_rate: float = 0.0002,
    impact_rate: float = 0.0001,
    positions: tuple = (-1.0, 0.0, 1.0),
    allow_short: bool = True,
    cost_multiplier: float = 1.0,
    start_idx: int = 0,
    end_idx: Optional[int] = None,
    decision_every: int = 1,
) -> StrategyResult:
    """
    目標rank IC（既定0.05）だけ未来を知る「劣化オラクル」

    完全オラクル（IC=1、到達不能）ではなく、現実的に目指しうるIC水準の
    予知力しか持たない場合の理論上限を返す。複数ノイズドロー（n_draws）の
    平均を取り、単一乱数の偶然に左右されないようにする。捕捉率
    (RL収益 / このオラクル収益) の方が完全オラクル比よりも実務上の目標に近い。

    decision_every > 1 を指定すると、ポジション変更を decision_every バー
    毎に制限する（低頻度アルファに対して毎バー回転コストを払わせない）。
    「弱いICでも十分な頻度に落とせば黒字化するか」を定量的に測れる。
    """
    end_idx = end_idx if end_idx is not None else fs.n_bars - 1
    true_r = _true_returns(fs, start_idx, end_idx)
    sigma = calibrate_noise_to_ic(true_r, target_ic, seed=seed)

    rng = np.random.default_rng(seed)
    equity_draws, turnover_draws = [], []
    for _ in range(n_draws):
        noisy_signal = true_r + rng.normal(0.0, sigma, size=true_r.shape)
        paths = oracle_dp_paths(
            fs, signal=noisy_signal, positions=positions, allow_short=allow_short,
            fee_rate=fee_rate, spread_rate=spread_rate, impact_rate=impact_rate,
            cost_multiplier=cost_multiplier, start_idx=start_idx, end_idx=end_idx,
            decision_every=decision_every,
        )
        equity, turnover = _simulate_positions(
            fs, paths, fee_rate=fee_rate, spread_rate=spread_rate,
            impact_rate=impact_rate, cost_multiplier=cost_multiplier,
            start_idx=start_idx, end_idx=end_idx,
        )
        equity_draws.append(equity)
        turnover_draws.append(turnover)

    equity = np.mean(np.stack(equity_draws, axis=0), axis=0)
    turnover_total = float(np.mean(turnover_draws))
    name = name or f"oracle_ic{target_ic:.2f}"
    return _equity_to_result(name, equity, turnover_total)


def run_all_baselines(
    fs: FeatureSet, include_oracle: bool = True,
    noisy_oracle_ic: Optional[float] = None,
    **kwargs,
) -> Dict[str, StrategyResult]:
    """
    全ベースラインを同一条件でバックテスト

    include_oracle: 完全DPオラクル上限（到達不能な天井）も併記
    noisy_oracle_ic: 指定するとこの目標ICの劣化オラクル（現実的な天井）も併記
    """
    out = {
        name: simulate_strategy(fs, fn, name=name, **kwargs)
        for name, fn in BASELINES.items()
    }
    oracle_kwargs = {k: v for k, v in kwargs.items()
                     if k in ("fee_rate", "spread_rate", "impact_rate",
                              "cost_multiplier", "start_idx", "end_idx")}
    if include_oracle:
        out["oracle_dp"] = oracle_dp_strategy(fs, **oracle_kwargs)
    if noisy_oracle_ic is not None:
        out[f"oracle_ic{noisy_oracle_ic:.2f}"] = noisy_oracle_strategy(
            fs, target_ic=noisy_oracle_ic, **oracle_kwargs,
        )
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
