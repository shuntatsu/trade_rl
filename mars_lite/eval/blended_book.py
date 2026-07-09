"""
2スリーブ合成バックテスト（配分RL + トレンドフォロー・ベータ）

配分RLを market-neutral（cs_demean/beta-neutral）で学習すると、市場方向の
ベータ収益を構造的に取り逃す（docs/ARCHITECTURE.md §2.6の bull相場実験と
同種の問題）。ここではRLの**実行済みウェイト**（学習時と同一のenv/
DecisionPipelineを通した後の値、つまりtrain/serveと同じ経路）と、
learning.baselines.trend_following_strategy と同一のトレンドシグナルを
固定比率 trend_frac で合成し、単一ポートフォリオの複利計算で一括評価する。

会計方式はPortfolioTradingEnv.step()/learning.baselines.simulate_strategyと
厳密に同一にする（1つのポートフォリオ価値をdot(weight, return)で複利計算）。
learning.baselines._simulate_positionsは銘柄毎に独立した「サブ口座」を
平均する近似方式（oracle系の±1/0ポジション専用）であり、連続値の
レバレッジ配分ウェイトには使えない（複数銘柄で資本を共有する実際の
ポートフォリオ複利と一致しない）ため、ここでは使わない。

trend_frac=0 でRL単体、1でtrend_followingベースライン単体とほぼ一致する
（境界の整合性はテストで確認、完全一致ではないのは合成後のグロス射影と
no-trade-bandの適用有無の違いによる）。
"""

from typing import Dict, Optional

import numpy as np

from mars_lite.features.feature_pipeline import FeatureSet
from mars_lite.trading.post_processor import BARS_PER_YEAR_1H


def evaluate_blended_book(
    agent,
    fs: FeatureSet,
    trend_frac: float = 0.5,
    trend_lookback: int = 48,
    trend_rebalance_every: int = 24,
    fee_rate: float = 0.0005,
    spread_rate: float = 0.0002,
    impact_rate: float = 0.0001,
    min_trade_delta: float = 0.02,
    cost_multiplier: float = 1.0,
    name: Optional[str] = None,
    **env_kwargs,
) -> Dict:
    """
    RLの実行済みウェイトパスとトレンドフォロー・スリーブを

        w_blend[t] = (1 - trend_frac) * w_rl[t] + trend_frac * w_trend[t]

    で合成し（合成後グロスが1超なら射影）、PortfolioTradingEnvと同一の
    sqrt-impact執行コストモデルで単一ポートフォリオとして一括バックテストする。
    """
    from mars_lite.env.portfolio_env import PortfolioTradingEnv
    from mars_lite.learning.baselines import trend_following_strategy
    from mars_lite.trading.execution import make_execution_model

    env = PortfolioTradingEnv(fs, episode_bars=fs.n_bars - 2, **env_kwargs)
    obs, _ = env.reset(options={"start_idx": 0})
    done = False
    has_disagreement = hasattr(agent, "disagreement")

    rl_path = []
    trend_path = []
    w_trend = np.zeros(fs.n_symbols)
    while not done:
        t = env.t
        action, _ = agent.predict(obs, deterministic=True)
        if has_disagreement:
            env.disagreement = agent.disagreement(obs)
        obs, _, term, trunc, _info = env.step(action)
        rl_path.append(env.weights.copy())

        w_trend = trend_following_strategy(
            fs,
            t,
            w_trend,
            lookback=trend_lookback,
            rebalance_every=trend_rebalance_every,
        )
        trend_path.append(w_trend.copy())
        done = term or trunc

    rl_arr = np.asarray(rl_path, dtype=np.float64)
    trend_arr = np.asarray(trend_path, dtype=np.float64)
    blend = (1.0 - trend_frac) * rl_arr + trend_frac * trend_arr
    gross = np.abs(blend).sum(axis=1, keepdims=True)
    blend = np.where(gross > 1.0, blend / np.maximum(gross, 1e-9), blend)

    exec_model = make_execution_model(
        fee_rate=fee_rate,
        spread_rate=spread_rate,
        impact_rate=impact_rate,
        cost_multiplier=cost_multiplier,
    )
    T = blend.shape[0]
    value = 1.0
    weights = np.zeros(fs.n_symbols)
    equity = [value]
    rets = []
    turnover_total = 0.0
    peak, max_dd = 1.0, 0.0
    for t in range(T):
        target = blend[t]
        delta = target - weights
        delta[np.abs(delta) < min_trade_delta] = 0.0
        weights = weights + delta
        turnover_total += float(np.abs(delta).sum())

        r_vec = fs.close[t + 1] / fs.close[t] - 1.0
        funding = float(np.sum(weights * fs.funding_rate[t + 1]))
        cost = exec_model.cost_fraction(delta)
        net = float(np.dot(weights, r_vec)) - cost - funding

        value *= 1.0 + net
        rets.append(net)
        equity.append(value)
        peak = max(peak, value)
        max_dd = max(max_dd, 1.0 - value / peak)

    equity_arr = np.asarray(equity, dtype=np.float64)
    rets_arr = np.asarray(rets, dtype=np.float64) if rets else np.zeros(1)
    sharpe = (
        float(rets_arr.mean() / rets_arr.std() * np.sqrt(BARS_PER_YEAR_1H))
        if rets_arr.std() > 0
        else 0.0
    )
    result_name = name or f"blended_trend{trend_frac:.2f}"
    return {
        "name": result_name,
        "total_return": float(value - 1.0),
        "sharpe": sharpe,
        "max_drawdown": float(max_dd),
        "turnover_total": float(turnover_total),
        "n_bars": T,
        "equity_curve": equity_arr,
    }
