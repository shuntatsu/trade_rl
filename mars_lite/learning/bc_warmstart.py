"""
行動クローニング（BC）ウォームスタートモジュール

PPOをランダム初期化から始めると、報酬ノイズの中で「線形関係なら一瞬で
見つかる信号」の再発見に学習の大半を費やす。そこで、既知の強い教師方策
（連続クロスセクショナルモメンタム）を模倣する事前学習を行ってから
PPOで微調整する。PPOの仕事は「コスト・リスク調整の上乗せ」に絞られる。

教師方策はモデルフリー（当てはめ無し）なので過学習しない。実データで
予測力が無ければウォームスタートは単なる事前分布に留まり、PPO＋検証選択が
後で棄却する。
"""

from typing import Callable, Optional, Tuple

import numpy as np

# teacher_fn(fs, t, prev_weights) -> weights(gross<=1)
TeacherFn = Callable[[object, int, np.ndarray], np.ndarray]


def soft_momentum_teacher(lookback: int = 24) -> TeacherFn:
    """
    連続クロスセクショナルモメンタム教師

    直近lookbackバーのリターンでランク付けし、勝者ロング・敗者ショートの
    ゼロサム・グロス1ウェイトを毎バー返す（離散リバランス版より滑らか）。
    """
    def teacher(fs, t: int, prev: np.ndarray) -> np.ndarray:
        n = fs.n_symbols
        start = max(0, t - lookback)
        if t - start < 2:
            return np.zeros(n)
        mom = np.log(fs.close[t] / fs.close[start])
        # ランク中心化 → ゼロサム、グロス正規化
        order = np.argsort(np.argsort(mom)).astype(np.float64)
        centered = order - (n - 1) / 2.0
        denom = np.abs(centered).sum()
        if denom < 1e-12:
            return np.zeros(n)
        return centered / denom

    return teacher


def generate_teacher_dataset(
    fs, teacher_fn: TeacherFn, env_kwargs: Optional[dict] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    教師方策で環境を1エピソード走らせ (観測, 教師行動) を収集

    観測は後処理込みの実executedウェイト状態を反映するため、教師方策の
    分布上でBCが行われる（分布ずれを軽減）。
    """
    from mars_lite.env.portfolio_env import PortfolioTradingEnv

    env_kwargs = dict(env_kwargs or {})
    env = PortfolioTradingEnv(fs, episode_bars=fs.n_bars - 2, **env_kwargs)
    obs, _ = env.reset(options={"start_idx": 0})

    X, A = [], []
    done = False
    while not done:
        w = teacher_fn(fs, env.t, env.weights)
        X.append(obs.copy())
        A.append(w.astype(np.float32))
        obs, _, term, trunc, _ = env.step(w)
        done = term or trunc

    return np.asarray(X, dtype=np.float32), np.asarray(A, dtype=np.float32)


def bc_pretrain(
    agent,
    X: np.ndarray,
    A: np.ndarray,
    epochs: int = 15,
    lr: float = 1e-3,
    batch_size: int = 256,
    verbose: int = 0,
) -> object:
    """
    方策の行動分布平均を教師行動へMSE回帰する事前学習

    SB3 PPOの ActorCriticPolicy を前提。value関数は触らず、方策側のみ更新。
    """
    import torch
    import torch.nn.functional as F

    policy = agent.policy
    device = policy.device
    Xt = torch.as_tensor(X, dtype=torch.float32, device=device)
    At = torch.as_tensor(A, dtype=torch.float32, device=device)

    opt = torch.optim.Adam(policy.parameters(), lr=lr)

    n = len(Xt)
    for ep in range(epochs):
        perm = torch.randperm(n, device=device)
        total = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            dist = policy.get_distribution(Xt[idx])
            mean_action = dist.distribution.mean
            loss = F.mse_loss(mean_action, At[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.detach()) * len(idx)
        if verbose:
            print(f"[BC] epoch {ep + 1}/{epochs} mse={total / n:.5f}")

    return agent
