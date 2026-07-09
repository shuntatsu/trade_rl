"""
勾配ブースティング（LightGBM）による因果的アルファ予測器

背景（実測に基づく設計判断）: 線形Ridge予測器のクロスセクショナルOOS ICは
実データ（Binance holdout）で 0.0148 と合格ライン0.02を下回り、これが
システム全体の律速だった（RLは保守崩壊、MoneyManagerはtrend_followingに
未達）。勾配ブースティング木は表形式の金融データで線形回帰を上回りやすく、
文献ではクロスセクショナル・ランカーでIC≈0.05が報告されている。

このモジュールは signal_check._pool が返す (X, y) を Ridge と同一I/Oで消費し、
mars_lite.features.signal_layer.causal_ridge_signal と**同一の因果構造**
（ローリング再学習＋embargo）で銘柄別アルファ信号を生成する。因果構造を
再利用するため、リーク自己検査を作り直す必要がない:
  - 時刻tの信号は「t-horizon以前に確定したラベル」だけで学習したモデルの予測
  - 学習窓の右端は refit 時点から horizon バー手前で切る（embargo）
  - 標準化も学習窓の予測統計のみを使う（因果的）

LightGBM 未インストールでも core パイプラインが壊れないよう、import は
関数内に遅延させる（依存は pyproject の [research] extra）。
"""

from typing import Optional

import numpy as np

from mars_lite.features.feature_pipeline import FeatureSet
from mars_lite.features.signal_check import _forward_returns, _transform_target

# 弱シグナル・重ノイズの金融パネルデータ向けの保守的な既定ハイパーパラメータ。
# 過学習を強く抑える（浅い木・大きな葉最小サンプル・L2正則化・列/行サンプリング）。
DEFAULT_GBM_PARAMS = {
    "objective": "regression",
    "num_leaves": 15,
    "learning_rate": 0.03,
    "min_child_samples": 200,
    "feature_fraction": 0.6,
    "bagging_fraction": 0.7,
    "bagging_freq": 1,
    "lambda_l2": 10.0,
    "max_depth": 4,
    "verbosity": -1,
    "seed": 0,
}
DEFAULT_NUM_ROUNDS = 200


def fit_gbm(
    X: np.ndarray,
    y: np.ndarray,
    params: Optional[dict] = None,
    num_boost_round: int = DEFAULT_NUM_ROUNDS,
):
    """LightGBM回帰器を学習して Booster を返す（signal_check._ridge_fit のGBM版）。

    入力は signal_check._pool の出力（プールされたクロスセクショナル (X, y)）
    をそのまま渡せる。
    """
    import lightgbm as lgb

    p = dict(DEFAULT_GBM_PARAMS)
    if params:
        p.update(params)
    dtrain = lgb.Dataset(X, label=y, free_raw_data=False)
    return lgb.train(p, dtrain, num_boost_round=num_boost_round)


def predict_gbm(booster, X: np.ndarray) -> np.ndarray:
    return np.asarray(booster.predict(X), dtype=np.float64)


def causal_gbm_signal(
    fs: FeatureSet,
    horizon: int = 24,
    target: str = "raw",
    train_window: int = 4000,
    refit_every: int = 400,
    min_train_bars: int = 1000,
    clip: float = 5.0,
    params: Optional[dict] = None,
    num_boost_round: int = DEFAULT_NUM_ROUNDS,
) -> np.ndarray:
    """ローリング再学習LightGBMによる因果的アルファ信号 (n_bars, n_symbols)

    signal_layer.causal_ridge_signal と同一の窓・embargo・標準化ロジック。
    各refit時点で「直近train_windowバーのうちラベルが確定している
    （horizonバー以上前の）サンプル」だけで木を学習し、次のrefit_everyバーを
    予測する。出力は学習窓の予測値標準偏差でz-score化してclipする。
    """
    n_bars, n_sym, n_feat = fs.features.shape
    fwd = _transform_target(_forward_returns(fs, horizon), target, fs)

    signal = np.zeros((n_bars, n_sym), dtype=np.float32)

    t = min_train_bars
    while t < n_bars:
        lo = max(0, t - train_window)
        hi = t - horizon  # embargo: ラベル確定済み領域の右端
        if hi - lo >= max(200, n_feat):
            X_tr = fs.features[lo:hi].reshape(-1, n_feat)
            y_tr = fwd[lo:hi].reshape(-1)
            m = np.isfinite(y_tr)
            if m.sum() >= max(200, n_feat):
                booster = fit_gbm(
                    X_tr[m], y_tr[m], params=params, num_boost_round=num_boost_round
                )
                # 学習窓内の予測値で標準化パラメータを決める（因果的）
                p_tr = predict_gbm(booster, X_tr[m])
                sd = float(np.std(p_tr))
                sd = sd if sd > 1e-12 else 1.0
                mu = float(np.mean(p_tr))

                end = min(t + refit_every, n_bars)
                X_pred = fs.features[t:end].reshape(-1, n_feat)
                p = (predict_gbm(booster, X_pred) - mu) / sd
                signal[t:end] = (
                    np.clip(p, -clip, clip).reshape(end - t, n_sym).astype(np.float32)
                )
        t += refit_every

    return signal
