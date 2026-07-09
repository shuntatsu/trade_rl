"""
PPO学習ライブラリ（ポートフォリオ配分エージェント）

scripts/train_portfolio.py から移動。CLIスクリプトとサーバーの両方から
同じ実装をimportして使う（以前はサーバーがsys.path操作でスクリプトを
ライブラリとしてimportしていた）。
"""

from typing import Optional

from mars_lite.env.portfolio_env import PortfolioTradingEnv
from mars_lite.features.feature_pipeline import FeatureSet
from mars_lite.models.portfolio_extractor import PortfolioExtractor


def make_env_fns(fs: FeatureSet, n_envs: int, seed: int, **env_kwargs):
    from stable_baselines3.common.monitor import Monitor

    def make_one(rank: int):
        def _init():
            env = PortfolioTradingEnv(fs, **env_kwargs)
            env.reset(seed=seed + rank)
            return Monitor(env)

        return _init

    return [make_one(i) for i in range(n_envs)]


def train_ppo(
    fs: FeatureSet,
    timesteps: int = 300_000,
    seed: int = 0,
    n_envs: int = 8,
    learning_rate: float = 3e-4,
    ent_coef: float = 0.002,
    gamma: float = 0.5,
    verbose: int = 0,
    callbacks=None,
    val_fs: FeatureSet = None,
    val_eval_freq: int = 20_000,
    bc_warmstart: bool = True,
    bc_epochs: int = 15,
    bc_teacher: str = "auto",
    extractor: str = "tfgated",
    horizon: int = 4,
    signal_target: str = "raw",
    oracle_noisy_ic: Optional[float] = None,
    net_size: str = "small",
    dropout: float = 0.0,
    net_arch=None,
    custom_teacher=None,
    custom_bc_dataset=None,
    **env_kwargs,
):
    """FeatureSetでPPOを学習して返す

    val_fs を渡すと検証スライスで定期評価し、最良時点のパラメータを
    最終モデルとして採用する（小データへの過学習対策）。
    val_fs 省略時は fs の末尾15%を自動で検証に割く。
    bc_warmstart=True でBC事前学習を行う。教師はbc_teacherで選択:
    ridge（デフォルト）= 学習スライスのRidge予測（アルファの型を仮定しない）、
    momentum = クロスモメンタム固定教師。

    signal_target: ICゲート判定とRidge教師の予測対象。"raw"（絶対リターン、
    既定）| "cs_demean"（バー毎に銘柄間平均を引いた市場中立の相対アルファ）|
    "vol_norm"。絶対リターンに信号が無く相対アルファだけが有意な市場では
    "cs_demean" を指定すると、ゲート判定・Ridge教師の両方が同じ市場中立の
    対象を見るようになる（gate1_diagnostic.py の診断結果に合わせる用途）。

    custom_teacher: 呼び出し側が構築済みの教師関数を直接渡す（bc_teacher選択
    ロジックを全てバイパス）。例: combined_teacher(fs, ..., ridge_target=
    "cs_demean") でtarget指定した合成教師を使いたい場合。
    custom_bc_dataset: (X, A) の教師データを直接渡す（generate_teacher_dataset
    もバイパス）。レジーム専門家など、教師データをfs全体ではなく特定の
    エピソード開始位置プールに限定したい場合に使う（PPO自身の経験分布との
    整合を保つため）。custom_teacher/bc_teacherより優先される。
    """
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import CallbackList
    from stable_baselines3.common.vec_env import DummyVecEnv

    from mars_lite.learning.val_selection import ValSelectionCallback

    if val_fs is None and fs.n_bars > 400:
        cut = int(fs.n_bars * 0.85)
        val_fs = fs.slice(cut, fs.n_bars)
        fs = fs.slice(0, cut)

    env = DummyVecEnv(make_env_fns(fs, n_envs, seed, **env_kwargs))
    probe = PortfolioTradingEnv(fs, **env_kwargs)

    # TFブロック構造を特徴名から導出（例: "15m_ret_z1" → TFプレフィックス）
    from mars_lite.features.feature_pipeline import TF_BLOCK_FEATURES

    tf_prefixes = []
    for name in fs.feature_names:
        p = name.split("_")[0]
        if p in ("15m", "30m", "1h", "4h", "1d") and p not in tf_prefixes:
            tf_prefixes.append(p)

    # net_size に応じた方策/価値ヘッド構成（extractorの規模と連動）。
    # small=実証済み（ARCHITECTURE.mdベンチ構成）、large=大容量（要再ベンチ）。
    if net_arch is None:
        net_arch = (
            dict(pi=[256, 256, 128], vf=[256, 256, 128])
            if net_size == "large"
            else dict(pi=[64, 64], vf=[64, 64])
        )

    if extractor == "tfgated" and tf_prefixes:
        from mars_lite.models.portfolio_extractor import TFGatedPortfolioExtractor

        policy_kwargs = {
            "features_extractor_class": TFGatedPortfolioExtractor,
            "features_extractor_kwargs": {
                **probe.obs_layout,
                "n_tf_blocks": len(tf_prefixes),
                "tf_block_size": len(TF_BLOCK_FEATURES),
                "size": net_size,
                "dropout": dropout,
            },
            "net_arch": net_arch,
        }
    else:
        policy_kwargs = {
            "features_extractor_class": PortfolioExtractor,
            "features_extractor_kwargs": {
                **probe.obs_layout,
                "size": net_size,
                "dropout": dropout,
            },
            "net_arch": net_arch,
        }

    def lr_schedule(progress_remaining: float) -> float:
        return learning_rate * progress_remaining

    agent = PPO(
        "MlpPolicy",
        env,
        policy_kwargs=policy_kwargs,
        learning_rate=lr_schedule,
        n_steps=256,
        batch_size=256,
        n_epochs=6,
        gamma=gamma,
        gae_lambda=0.9,
        ent_coef=ent_coef,
        vf_coef=0.5,
        max_grad_norm=0.5,
        seed=seed,
        device="cpu",
        verbose=verbose,
    )

    # BCウォームスタート（教師方策の模倣で方策を初期化）
    # bc_teacher="auto": 学習スライスのゲートで教師を自動選択
    #   クロスセクショナルIC合格 → ridge（相対アルファ）
    #   方向性トレンド合格     → ts_momentum（ベータ捕捉。上昇相場でB&Hに勝つ）
    #   どちらも無し           → BC無効（フラットで待つ＝安全）
    # ridge/momentum/ts_momentum を明示指定も可。
    if bc_warmstart:
        from mars_lite.learning.bc_warmstart import (
            bc_pretrain,
            generate_teacher_dataset,
            ridge_teacher,
            soft_momentum_teacher,
            ts_momentum_teacher,
        )

        teacher = custom_teacher
        if custom_bc_dataset is not None:
            pass  # 下でこのデータセットをそのまま使う（教師選択ロジックは全てスキップ）
        elif teacher is not None:
            pass  # 呼び出し側が既に構築した教師をそのまま使う（target指定等の柔軟性用）
        elif bc_teacher == "auto":
            from mars_lite.features.signal_check import run_signal_check, run_trend_gate
            from mars_lite.learning.bc_warmstart import combined_teacher

            ic = run_signal_check(fs, horizon=horizon, target=signal_target)
            trend = run_trend_gate(fs, horizon=horizon)
            # Ridgeは偽陽性回避のため閾値+マージンを要求
            use_ridge = ic.mean_oos_ic >= 0.025
            use_trend = trend["has_trend"]
            if use_ridge or use_trend:
                teacher = combined_teacher(
                    fs,
                    use_ridge=use_ridge,
                    use_trend=use_trend,
                    horizon=horizon,
                    ridge_target=signal_target,
                )
                if verbose:
                    comps = []
                    if use_ridge:
                        comps.append(f"ridge(ic={ic.mean_oos_ic:.3f})")
                    if use_trend:
                        comps.append(f"trend(t={trend['t_stat']:.1f})")
                    print(f"[BC auto] teacher = {' + '.join(comps)}")
            elif verbose:
                print("[BC auto] no gate passed -> BC disabled (flat prior)")
        elif bc_teacher == "ridge":
            teacher = ridge_teacher(fs, horizon=horizon, target=signal_target)
        elif bc_teacher == "ts_momentum":
            teacher = ts_momentum_teacher()
        elif bc_teacher == "oracle":
            from mars_lite.features.signal_check import run_signal_check
            from mars_lite.learning.bc_warmstart import dp_oracle_teacher

            ic = run_signal_check(fs, horizon=horizon, target=signal_target)
            if ic.mean_oos_ic >= 0.025:
                teacher = dp_oracle_teacher(fs, noisy_ic=oracle_noisy_ic)
                if verbose:
                    kind = (
                        f"noisy_ic={oracle_noisy_ic}"
                        if oracle_noisy_ic
                        else "perfect foresight"
                    )
                    print(
                        f"[BC oracle] IC gate passed (ic={ic.mean_oos_ic:.3f}), "
                        f"using DP-oracle teacher ({kind})"
                    )
            elif verbose:
                print(
                    f"[BC oracle] IC gate failed (ic={ic.mean_oos_ic:.3f}) "
                    "-> oracle teacher disabled (flat prior); "
                    "特権教師を模倣する意味がない（ノイズの丸暗記になる）"
                )
        else:
            teacher = soft_momentum_teacher()

        if custom_bc_dataset is not None:
            X, A = custom_bc_dataset
            bc_pretrain(agent, X, A, epochs=bc_epochs, verbose=verbose)
        elif teacher is not None:
            X, A = generate_teacher_dataset(fs, teacher, env_kwargs)
            bc_pretrain(agent, X, A, epochs=bc_epochs, verbose=verbose)

    val_cb = None
    if val_fs is not None:
        val_cb = ValSelectionCallback(
            val_fs,
            eval_freq=val_eval_freq,
            env_kwargs=env_kwargs,
            verbose=verbose,
        )
        callbacks = CallbackList(
            ([callbacks] if callbacks is not None else []) + [val_cb]
        )

    agent.learn(total_timesteps=timesteps, callback=callbacks, progress_bar=False)

    if val_cb is not None:
        agent = val_cb.restore_best(agent)
        if verbose:
            print(
                f"[train_ppo] Restored best-val model (score={val_cb.best_score:+.4f})"
            )
    return agent
