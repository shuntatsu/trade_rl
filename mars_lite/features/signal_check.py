"""
シグナル検証モジュール（ゲート1）

特徴量に「次kバーのリターン」への予測力（アルファ）があるかを
RL学習の前に検証する。予測力が無いデータでRLを学習させても
「取引しない」が最適解になるため、ここが不合格なら先に進まない。

手法:
- 単一特徴量のランクIC（Spearman相関）: 特徴量ごとの素の予測力
- Ridge回帰のウォークフォワードOOSランクIC: 特徴量を合成した予測力

判定基準（デフォルト）:
- 合成予測のOOSランクICの平均 >= 0.02 かつ 期間の6割以上で正
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from mars_lite.features.feature_pipeline import FeatureSet


@dataclass
class SignalReport:
    """ゲート1の判定レポート"""
    per_feature_ic: Dict[str, float]
    oos_ic_by_fold: List[float]
    mean_oos_ic: float
    positive_fold_ratio: float
    n_folds: int
    horizon: int
    passed: bool
    threshold: float

    def to_dict(self) -> Dict:
        return {
            "mean_oos_ic": self.mean_oos_ic,
            "positive_fold_ratio": self.positive_fold_ratio,
            "n_folds": self.n_folds,
            "horizon": self.horizon,
            "passed": self.passed,
            "threshold": self.threshold,
            "oos_ic_by_fold": self.oos_ic_by_fold,
            "top_features": dict(sorted(
                self.per_feature_ic.items(), key=lambda x: -abs(x[1])
            )[:10]),
        }

    def summary(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        top = sorted(self.per_feature_ic.items(), key=lambda x: -abs(x[1]))[:5]
        lines = [
            f"[Gate 1: Signal Check] {status}",
            f"  OOS rank IC: mean={self.mean_oos_ic:.4f} "
            f"(threshold={self.threshold}), positive folds="
            f"{self.positive_fold_ratio:.0%} ({self.n_folds} folds, horizon={self.horizon} bars)",
            "  Top features by |IC|:",
        ]
        lines += [f"    {name:<24} IC={ic:+.4f}" for name, ic in top]
        return "\n".join(lines)


def _rank(x: np.ndarray) -> np.ndarray:
    """1次元配列を[0,1]ランクに変換"""
    order = np.argsort(np.argsort(x))
    return order / max(len(x) - 1, 1)


def _rank_ic(pred: np.ndarray, target: np.ndarray) -> float:
    """SpearmanランクIC（scipy不使用）"""
    if len(pred) < 10 or np.std(pred) == 0 or np.std(target) == 0:
        return 0.0
    return float(np.corrcoef(_rank(pred), _rank(target))[0, 1])


def _forward_returns(fs: FeatureSet, horizon: int) -> np.ndarray:
    """次horizonバーの累積ログリターン (n_bars, n_symbols)、末尾はNaN"""
    log_close = np.log(fs.close)
    fwd = np.full_like(log_close, np.nan)
    fwd[:-horizon] = log_close[horizon:] - log_close[:-horizon]
    return fwd


def _pool(fs: FeatureSet, horizon: int):
    """全銘柄をプールした (X, y, bar_idx) を作成（末尾horizonバー除外）"""
    n_bars, n_sym, n_feat = fs.features.shape
    fwd = _forward_returns(fs, horizon)
    valid = n_bars - horizon
    X = fs.features[:valid].reshape(valid * n_sym, n_feat)
    y = fwd[:valid].reshape(valid * n_sym)
    bar_idx = np.repeat(np.arange(valid), n_sym)
    mask = np.isfinite(y)
    return X[mask], y[mask], bar_idx[mask]


def _ridge_fit(X: np.ndarray, y: np.ndarray, lam: float = 10.0) -> np.ndarray:
    """Ridge回帰の閉形式解（バイアス項付き）"""
    Xb = np.hstack([X, np.ones((len(X), 1))])
    A = Xb.T @ Xb + lam * np.eye(Xb.shape[1])
    return np.linalg.solve(A, Xb.T @ y)


def _ridge_predict(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    return np.hstack([X, np.ones((len(X), 1))]) @ w


def compute_feature_mask(
    fs: FeatureSet,
    horizon: int = 4,
    n_folds: int = 4,
    min_abs_ic: float = 0.01,
    min_sign_agreement: float = 0.75,
) -> Dict[str, object]:
    """
    IC安定性による特徴マスクを計算（時間軸の自動取捨選択）

    学習スライスを時間順にn_folds分割し、特徴ごとにfold別ランクICを計算。
    「平均|IC|がしきい値以上」かつ「ICの符号がfold間で一致」する特徴のみ残す。
    予測力のないTFブロックはここで自動的に落ちるため、多時間軸観測を
    「シグナルがあるTFだけ使う」形に縮退させられる。

    注意: fsには**学習スライスのみ**を渡すこと（選択リーク防止）。
    マスクは推論時にも同一適用が必要（モデルメタデータに保存する）。

    Returns:
        {mask: np.ndarray(bool, n_features), kept: [names], dropped: [names],
         per_feature: {name: {mean_abs_ic, sign_agreement}}}
    """
    X, y, bar_idx = _pool(fs, horizon)
    n_bars = int(bar_idx.max()) + 1
    edges = np.linspace(0, n_bars, n_folds + 1).astype(int)

    n_feat = X.shape[1]
    fold_ics = np.zeros((n_folds, n_feat))
    for k in range(n_folds):
        m = (bar_idx >= edges[k]) & (bar_idx < edges[k + 1])
        if m.sum() < 50:
            continue
        for j in range(n_feat):
            fold_ics[k, j] = _rank_ic(X[m][:, j], y[m])

    mean_abs = np.abs(fold_ics.mean(axis=0))
    signs = np.sign(fold_ics)
    dominant = np.sign(fold_ics.mean(axis=0))[None, :]
    agreement = (signs == dominant).mean(axis=0)

    mask = (mean_abs >= min_abs_ic) & (agreement >= min_sign_agreement)
    names = fs.feature_names
    return {
        "mask": mask,
        "kept": [n for n, m in zip(names, mask) if m],
        "dropped": [n for n, m in zip(names, mask) if not m],
        "per_feature": {
            n: {"mean_abs_ic": float(a), "sign_agreement": float(g)}
            for n, a, g in zip(names, mean_abs, agreement)
        },
    }


def run_leak_self_test(fs: FeatureSet, horizon: int = 4) -> Dict[str, object]:
    """
    リーク検出器自体の健全性を検査する自己テスト

    - shuffle検査: ターゲットを時間シャッフルするとIC≈0 になるべき
      （ならなければ評価コード側のバグ）
    - future-shift検査: 特徴を意図的に1バー未来へずらすとICが跳ね上がるべき
      （＝リーク検出器が実際にリークを捕捉できる証明）

    Returns:
        {shuffle_ic, base_ic, future_shift_ic, healthy: bool}
    """
    X, y, bar_idx = _pool(fs, horizon)

    # ベースIC（全特徴平均の素のIC）
    base_ic = float(np.mean([abs(_rank_ic(X[:, j], y)) for j in range(X.shape[1])]))

    # shuffle: yをシャッフル → 相関消失を期待
    rng = np.random.default_rng(0)
    y_shuf = rng.permutation(y)
    shuffle_ic = float(np.mean([abs(_rank_ic(X[:, j], y_shuf)) for j in range(X.shape[1])]))

    # future-shift: 特徴を未来方向へずらす（バー単位）→ リーク混入 → IC増大を期待
    n_bars = int(bar_idx.max()) + 1
    n_sym = fs.n_symbols
    fwd = _forward_returns(fs, horizon)
    valid = n_bars
    # 特徴を1バー未来にシフト（先読み）
    feat_shift = np.roll(fs.features, -1, axis=0)
    Xs = feat_shift[:valid].reshape(valid * n_sym, -1)
    ys = fwd[:valid].reshape(valid * n_sym)
    mask = np.isfinite(ys)
    Xs, ys = Xs[mask], ys[mask]
    future_ic = float(np.mean([abs(_rank_ic(Xs[:, j], ys)) for j in range(Xs.shape[1])]))

    healthy = shuffle_ic < 0.02 and future_ic > base_ic
    return {
        "shuffle_ic": shuffle_ic,
        "base_ic": base_ic,
        "future_shift_ic": future_ic,
        "healthy": bool(healthy),
    }


def run_signal_check(
    fs: FeatureSet,
    horizon: int = 4,
    n_folds: int = 5,
    purge_bars: int = 24,
    threshold: float = 0.02,
    min_positive_ratio: float = 0.6,
) -> SignalReport:
    """
    ウォークフォワードICゲートを実行

    Args:
        fs: FeatureSet
        horizon: 予測対象の先読みバー数（1h足なら4=4時間先）
        n_folds: ウォークフォワード分割数
        purge_bars: 学習と検証の間に挟むギャップ（リーク防止）
        threshold: 合格に必要な平均OOSランクIC
        min_positive_ratio: 正のICが必要なfoldの割合
    """
    X, y, bar_idx = _pool(fs, horizon)
    n_bars = int(bar_idx.max()) + 1

    # 特徴量ごとの素のIC（全期間）
    per_feature = {
        name: _rank_ic(X[:, j], y) for j, name in enumerate(fs.feature_names)
    }

    # ウォークフォワード: 前半で学習→purge→次区間で検証、をスライド
    fold_edges = np.linspace(0, n_bars, n_folds + 2).astype(int)
    oos_ics: List[float] = []
    for k in range(1, n_folds + 1):
        train_end = fold_edges[k]
        test_start = train_end + purge_bars
        test_end = fold_edges[k + 1]
        if test_start >= test_end or train_end < 50:
            continue

        train_mask = bar_idx < train_end
        test_mask = (bar_idx >= test_start) & (bar_idx < test_end)
        if train_mask.sum() < 100 or test_mask.sum() < 50:
            continue

        w = _ridge_fit(X[train_mask], y[train_mask])
        pred = _ridge_predict(X[test_mask], w)
        oos_ics.append(_rank_ic(pred, y[test_mask]))

    mean_ic = float(np.mean(oos_ics)) if oos_ics else 0.0
    pos_ratio = float(np.mean([ic > 0 for ic in oos_ics])) if oos_ics else 0.0
    passed = mean_ic >= threshold and pos_ratio >= min_positive_ratio

    return SignalReport(
        per_feature_ic=per_feature,
        oos_ic_by_fold=[float(x) for x in oos_ics],
        mean_oos_ic=mean_ic,
        positive_fold_ratio=pos_ratio,
        n_folds=len(oos_ics),
        horizon=horizon,
        passed=passed,
        threshold=threshold,
    )
