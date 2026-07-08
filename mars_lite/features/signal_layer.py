"""
因果的シグナルレイヤー（予測とトレードの責務分離）

背景（実測に基づく設計判断）: BC(Ridge教師)ウォームスタート直後の方策が
最も強く、PPOがそこからほぼ改善できないことが実データ検証で繰り返し
観測された。つまり「92特徴からの予測の発見」までRLに背負わせるのは
サンプル効率が悪すぎる。教師あり学習は毎バーに密なラベル（前方リターン）
があるため、予測はRLより遥かに効率良く学べる。

そこで責務を分離する:
  - 予測（アルファ生成）: このモジュール。Ridgeを「過去データのみで定期的に
    再学習 → 次の区間を予測」するローリング方式で、全バーの銘柄別アルファ
    予測系列を生成する（look-aheadなし）
  - トレード（サイジング・タイミング・コスト管理）: RL。観測にこの信号を
    受け取り、「予測を所与としていつ・どれだけ張るか」に専念する

因果性の保証:
  - 時刻tの信号は「t-horizon以前に確定したラベル」だけで学習したモデルの
    予測値。ラベル自体がhorizonバー先のリターンなので、学習データの右端は
    refit時点からhorizonバー手前で切る（embargo）
  - 標準化も学習窓の統計のみを使う
  - この性質により、fold分割**前**に全系列を一括計算してもリークしない
    （walk-forward検証でもそのまま使える）

観測への注入は augment_with_signals（append=92特徴+信号 / only=信号のみ）。
signal-onlyは観測次元を n_features → 信号数(既定1) に劇的に削減し、
RLの学習問題を大幅に単純化する。
"""

from typing import List, Optional

import numpy as np

from mars_lite.features.feature_pipeline import FeatureSet
from mars_lite.features.signal_check import (
    _forward_returns,
    _ridge_fit,
    _ridge_predict,
    _transform_target,
)


def causal_ridge_signal(
    fs: FeatureSet,
    horizon: int = 24,
    target: str = "raw",
    train_window: int = 4000,
    refit_every: int = 400,
    min_train_bars: int = 1000,
    lam: float = 10.0,
    clip: float = 5.0,
) -> np.ndarray:
    """
    ローリング再学習Ridgeによる因果的アルファ信号 (n_bars, n_symbols)

    各refit時点で「直近train_window バーのうち、ラベルが確定している
    （= horizonバー以上前の）サンプル」だけでRidgeを学習し、次の
    refit_everyバーの間はそのモデルで予測する。出力は学習窓の予測値
    標準偏差でz-score化してclipする（環境観測のスケールに揃える）。

    min_train_bars未満の序盤は信号0（ウォームアップ。エージェントは
    「信号なし=張らない」を学べる）。
    """
    n_bars, n_sym, n_feat = fs.features.shape
    fwd = _transform_target(_forward_returns(fs, horizon), target, fs)

    signal = np.zeros((n_bars, n_sym), dtype=np.float32)

    t = min_train_bars
    while t < n_bars:
        # 学習サンプル: [t-train_window, t-horizon) のバー（ラベル確定済み領域）
        lo = max(0, t - train_window)
        hi = t - horizon
        if hi - lo >= max(200, n_feat):
            X_tr = fs.features[lo:hi].reshape(-1, n_feat)
            y_tr = fwd[lo:hi].reshape(-1)
            m = np.isfinite(y_tr)
            if m.sum() >= max(200, n_feat):
                w = _ridge_fit(X_tr[m], y_tr[m], lam=lam)
                # 学習窓内の予測値で標準化パラメータを決める（因果的）
                p_tr = _ridge_predict(X_tr[m], w)
                sd = float(np.std(p_tr))
                sd = sd if sd > 1e-12 else 1.0
                mu = float(np.mean(p_tr))

                end = min(t + refit_every, n_bars)
                X_pred = fs.features[t:end].reshape(-1, n_feat)
                p = (_ridge_predict(X_pred, w) - mu) / sd
                signal[t:end] = (
                    np.clip(p, -clip, clip).reshape(end - t, n_sym).astype(np.float32)
                )
        t += refit_every

    return signal


def augment_with_signals(
    fs: FeatureSet,
    signals: np.ndarray,
    signal_names: Optional[List[str]] = None,
    only: bool = False,
) -> FeatureSet:
    """
    信号列 (n_bars, n_symbols) または (n_bars, n_symbols, k) を特徴量へ注入する

    only=False: 既存特徴の末尾に信号チャネルを追加（append）
    only=True : 特徴を信号チャネル**だけ**に置き換える（観測次元の劇的削減。
                RLは予測の発見から解放され、サイジング/タイミングに専念する）
    """
    if signals.ndim == 2:
        signals = signals[:, :, None]
    k = signals.shape[2]
    names = signal_names or [f"alpha_signal_{i}" for i in range(k)]
    if len(names) != k:
        raise ValueError(f"signal_names length {len(names)} != k {k}")
    if signals.shape[:2] != (fs.n_bars, fs.n_symbols):
        raise ValueError(
            f"signals shape {signals.shape[:2]} != (n_bars, n_symbols) "
            f"({fs.n_bars}, {fs.n_symbols})"
        )

    sig = signals.astype(np.float32)
    if only:
        new_features = sig
        new_names = list(names)
    else:
        new_features = np.concatenate([fs.features, sig], axis=2)
        new_names = list(fs.feature_names) + list(names)

    return FeatureSet(
        symbols=fs.symbols,
        timestamps=fs.timestamps,
        features=new_features,
        global_features=fs.global_features,
        close=fs.close,
        open_next=fs.open_next,
        funding_rate=fs.funding_rate,
        feature_names=new_names,
        global_feature_names=fs.global_feature_names,
    )


def apply_signal_layer(args, fs: FeatureSet, horizon: int) -> FeatureSet:
    """
    CLI引数からシグナルレイヤーを適用する（phase_train/phase_wf共用）

    --signal-layer off   : 何もしない（既定）
    --signal-layer append: 因果Ridge信号を特徴の末尾に追加
    --signal-layer only  : 特徴を信号だけに置き換え（RLはトレードに専念）

    因果的（各バーの信号は過去データのみで学習したモデルの出力）なので、
    fold分割前の全系列に一括適用してもwalk-forward検証のリークにならない。
    """
    mode = getattr(args, "signal_layer", "off")
    if mode == "off":
        return fs
    train_window = getattr(args, "signal_train_window", 4000)
    signal = causal_ridge_signal(
        fs,
        horizon=horizon,
        target=getattr(args, "target", "raw"),
        train_window=train_window,
        # ウォームアップは学習窓に比例させる（固定1000だと小さいデータセット
        # では学習スライスの大半が信号ゼロになり、方策が「常にフラット」を
        # 学んでしまう）
        min_train_bars=max(400, train_window // 4),
        refit_every=getattr(args, "signal_refit_every", 400),
    )
    fs2 = augment_with_signals(
        fs, signal, signal_names=[f"ridge_alpha_h{horizon}"], only=(mode == "only")
    )
    print(
        f"[Signal layer] mode={mode} ridge_alpha_h{horizon} "
        f"(features: {fs.n_features} -> {fs2.n_features})"
    )
    return fs2
