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


def ts_momentum_teacher(lookback: int = 48) -> TeacherFn:
    """
    時系列モメンタム教師（**ネット方向性を持つ**＝ベータ捕捉可能）

    クロスセクショナル教師（ridge/soft_momentum）はゼロサムで市場中立のため、
    「全銘柄が上がるだけ」の方向性ベータを捉えられない。この教師は各銘柄の
    過去リターンの絶対的な符号・強度でポジションを建て、ゼロサムにしない。
    上昇相場では全ロング、下落相場では全ショートに寄る（=トレンドフォロー）。
    """

    def teacher(fs, t: int, prev: np.ndarray) -> np.ndarray:
        n = fs.n_symbols
        start = max(0, t - lookback)
        if t - start < 4:
            return np.zeros(n)
        mom = np.log(fs.close[t] / fs.close[start])
        scale = np.abs(mom).mean() + 1e-9
        raw = np.tanh(mom / scale)  # 有界化・ネット方向性を保持
        gross = np.abs(raw).sum()
        return raw / gross if gross > 1.0 else raw

    return teacher


def ridge_teacher(
    train_fs,
    horizon: int = 4,
    lam: float = 10.0,
    target: str = "raw",
) -> TeacherFn:
    """
    データ駆動のRidge教師（アルファの型を仮定しない）

    ゲート1と同じRidge回帰を**学習スライスのみ**で当てはめ、
    その予測値のクロスセクショナル中心化ウェイトを教師とする。
    モメンタム型・平均回帰型どちらのアルファでも、ICゲートを通る
    信号があれば自動的にそれを模倣の出発点にできる。

    target="cs_demean" で学習すると、Ridge自体が市場中立な相対アルファ
    だけを当てにいく（予測後のクロスセクショナル中心化と整合する）。
    狭いユニバースでは市場全体の方向がフィットを汚染しうるため有効。

    注意: train_fs には学習用スライスだけを渡すこと（リーク防止）。
    """
    from mars_lite.features.signal_check import _pool, _ridge_fit

    X, y, _ = _pool(train_fs, horizon, target=target)
    w_ridge = _ridge_fit(X, y, lam)

    def teacher(fs, t: int, prev: np.ndarray) -> np.ndarray:
        feats = fs.features[t]  # (n_symbols, n_features)
        preds = np.hstack([feats, np.ones((len(feats), 1))]) @ w_ridge
        centered = preds - preds.mean()
        denom = np.abs(centered).sum()
        if denom < 1e-12:
            return np.zeros(fs.n_symbols)
        return centered / denom

    return teacher


def combined_teacher(
    train_fs,
    use_ridge: bool,
    use_trend: bool,
    horizon: int = 4,
    ridge_target: str = "raw",
) -> TeacherFn:
    """
    合成教師: Ridge（相対アルファ・ゼロサム）＋ 時系列モメンタム（方向性ベータ）

    ポジション = 市場中立の相対ベット + 方向性の市場エクスポージャ、という
    正しい分解。各成分は独立ゲートで有効化する:
      use_ridge: クロスセクショナルICが（マージン付きで）合格
      use_trend: 方向性トレンドが有意
    上昇相場では use_trend が effいてベータを捕捉、相対アルファ市場では
    use_ridge が効いて相対ベットを取る。両方でも自然に合算される。

    ridge_target="cs_demean" にすると、Ridge成分が方向性ベータ（ts_fn側で
    別途捕捉）と重複しない、純粋な相対アルファだけを学習する。
    """
    ridge_fn = (
        ridge_teacher(train_fs, horizon, target=ridge_target) if use_ridge else None
    )
    ts_fn = ts_momentum_teacher() if use_trend else None

    def teacher(fs, t: int, prev: np.ndarray) -> np.ndarray:
        w = np.zeros(fs.n_symbols)
        if ridge_fn is not None:
            w = w + ridge_fn(fs, t, prev)
        if ts_fn is not None:
            w = w + ts_fn(fs, t, prev)
        gross = np.abs(w).sum()
        return w / gross if gross > 1.0 else w

    return teacher


def dp_oracle_teacher(
    train_fs,
    noisy_ic: Optional[float] = None,
    seed: int = 0,
    fee_rate: float = 0.0005,
    spread_rate: float = 0.0002,
    impact_rate: float = 0.0001,
    cost_multiplier: float = 1.0,
    allow_short: bool = True,
) -> TeacherFn:
    """
    DPオラクル（特権教師）を模倣する蒸留教師（clairvoyant teacher distillation）

    未来を完全に（noisy_ic省略時）、または目標ICだけ劣化的に（noisy_ic指定時）
    知った上での最適ポジション経路を教師ラベルとする。studentは現在時刻の
    因果的特徴のみを観測して模倣するため、実際に再現できる度合いは結局
    データのIC（signal_check）で頭打ちになるが、限られた信号をより
    攻撃的（オラクル並みの機動性で）に使い切る出発点を与える。

    noisy_ic を指定すると、完全予知ではなく「学習可能な水準の予知力」を
    模倣させられる（perfect-future patternsの丸暗記を避ける）。

    注意: train_fsには学習スライスのみを渡すこと（リーク防止）。教師
    ラベル自体は非因果的（未来を見る）だが、模倣後の評価は必ずOOSで
    行うこと（教師の強さをRLの実力と誤解しない）。
    """
    from mars_lite.learning.baselines import (
        _true_returns,
        calibrate_noise_to_ic,
        oracle_dp_paths,
    )

    n_sym = train_fs.n_symbols
    end_idx = train_fs.n_bars - 1

    signal = None
    if noisy_ic is not None:
        true_r = _true_returns(train_fs, 0, end_idx)
        sigma = calibrate_noise_to_ic(true_r, noisy_ic, seed=seed)
        rng = np.random.default_rng(seed)
        signal = true_r + rng.normal(0.0, sigma, size=true_r.shape)

    paths = oracle_dp_paths(
        train_fs,
        signal=signal,
        allow_short=allow_short,
        fee_rate=fee_rate,
        spread_rate=spread_rate,
        impact_rate=impact_rate,
        cost_multiplier=cost_multiplier,
        start_idx=0,
        end_idx=end_idx,
    )  # (T, n_sym), T == end_idx

    def teacher(fs, t: int, prev: np.ndarray) -> np.ndarray:
        idx = min(t, len(paths) - 1)
        return paths[idx] / n_sym

    return teacher


def generate_teacher_dataset(
    fs,
    teacher_fn: TeacherFn,
    env_kwargs: Optional[dict] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    教師方策で環境を1エピソード走らせ (観測, 教師行動) を収集

    観測は後処理込みの実executedウェイト状態を反映するため、教師方策の
    分布上でBCが行われる（分布ずれを軽減）。
    """
    from mars_lite.env.portfolio_env import PortfolioTradingEnv

    env_kwargs = dict(env_kwargs or {})
    # 教師データはスライス全体を1エピソードで走査する。呼び出し側の
    # episode_bars / regime_start_pool（レジーム専門家用）は無視する。
    env_kwargs.pop("episode_bars", None)
    env_kwargs.pop("regime_start_pool", None)
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
            idx = perm[i : i + batch_size]
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
