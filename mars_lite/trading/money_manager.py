"""
金銭管理アロケータ（責務分割: 予測は教師あり、サイジングはルール）

背景（実測に基づく設計判断）: 単一PPO方策に「予測（どの銘柄が上がるか）＋
銘柄選択＋サイジング＋タイミング」を全部背負わせると、弱いアルファ・保守
バイアス・信用割当の難しさが相互に増幅し、実データ3本すべてで
trend_following に勝てなかった（walk-forward中央値マイナス、gate2 FAIL）。
一方 mars_lite.learning.bc_warmstart には既にルール型アロケータ
（ridge_teacher=相対アルファ / ts_momentum_teacher=方向性ベータ /
combined_teacher=両者合成）が存在するが、BCウォームスタートの「教師」
としてしか使われず、戦略として評価もゲートもされていなかった。

このモジュールは combined_teacher を第一級の評価対象へ昇格させる:
  - 予測（アルファ生成）: Ridge を **学習スライスのみ** で当てはめる（因果的）
  - サイジング（金銭管理）: 相対アルファ（市場中立）＋方向性ベータ（トレンド）
    の分解に、ボラティリティ目標によるグロス調整（過去データのみで推定）を
    重ねる

これにより「教師あり予測＋ルールサイジングは、純トレンドフォロー/RLに
勝てるのか？」を、RL学習を一切回さず数秒で、既存の執行コストモデル
（simulate_strategy）で公平に検証できる。walk-forward/holdoutの
train区間で適合し test区間で評価するため look-ahead はない。
"""

from typing import Optional

import numpy as np

from mars_lite.features.feature_pipeline import FeatureSet
from mars_lite.learning.baselines import StrategyResult, WeightFn, simulate_strategy
from mars_lite.learning.bc_warmstart import combined_teacher
from mars_lite.trading.post_processor import BARS_PER_YEAR_1H


def _vol_targeted(
    teacher_fn: WeightFn,
    target_vol: float,
    lookback: int = 48,
    bars_per_year: int = BARS_PER_YEAR_1H,
    max_scale: float = 3.0,
) -> WeightFn:
    """teacher の出力グロスを、ポートフォリオ実現ボラが target_vol（年率）に
    近づくよう因果的にスケールするラッパ。

    時刻 t のスケールは close[:t] だけから推定した直近 lookback バーの
    ポートフォリオ・リターン標準偏差で決める（未来を見ない）。信号が弱く
    グロスが小さい局面では拡大、急変時は縮小する = 素朴なリスクパリティ的
    サイジング。max_scale で過剰レバレッジを抑える。
    """

    def fn(fs: FeatureSet, t: int, prev: np.ndarray) -> np.ndarray:
        w = teacher_fn(fs, t, prev)
        gross = float(np.abs(w).sum())
        if gross < 1e-9:
            return w
        start = max(1, t - lookback)
        if t - start < 8:
            return w
        # このウェイト構成を直近区間に当てはめた時のポートフォリオ・リターン系列
        rets = fs.close[start : t + 1] / fs.close[start - 1 : t] - 1.0  # (L, n_sym)
        port_rets = rets @ (w / gross)
        realized = float(np.std(port_rets)) * np.sqrt(bars_per_year)
        if realized < 1e-9:
            return w
        scale = float(np.clip(target_vol / realized, 0.0, max_scale))
        scaled = w * scale
        # グロス上限1.0を超えたら射影（レバレッジ暴走を防ぐ）
        sg = float(np.abs(scaled).sum())
        if sg > 1.0:
            scaled = scaled / sg
        return scaled

    return fn


def _turnover_controlled(
    teacher_fn: WeightFn,
    rebalance_every: int = 24,
    no_trade_band: float = 0.05,
) -> WeightFn:
    """回転抑制ラッパ（金銭管理の中核: コストで死なないようにする）。

    実データ検証で発見: combined_teacher は BC教師（学習ターゲット）由来で
    毎バー生Ridge予測へリバランスする設計のため、弱いIC（≈0.015）の実データ
    では turnover≈970 に暴走し、コストだけで -56% 溶かした（trend_following は
    24バー毎リバランスで turnover≈80）。ライブのアロケータには
      1. rebalance_every バー毎にのみ目標を再計算（それ以外は保持）
      2. 目標変化のグロスが no_trade_band 未満なら据え置き
    という回転抑制が必須。これは trend_following/cross_momentum と同じ構造。
    """

    def fn(fs: FeatureSet, t: int, prev: np.ndarray) -> np.ndarray:
        # 非リバランスバーは保持（初回=prevが全ゼロのときは計算する）
        if t % rebalance_every != 0 and np.abs(prev).sum() > 1e-9:
            return prev
        target = teacher_fn(fs, t, prev)
        if float(np.abs(target - prev).sum()) < no_trade_band:
            return prev
        return target

    return fn


def build_money_manager(
    train_fs: FeatureSet,
    horizon: int = 4,
    use_ridge: bool = True,
    use_trend: bool = True,
    ridge_target: str = "cs_demean",
    target_vol: float = 0.0,
    vol_lookback: int = 48,
    rebalance_every: int = 24,
    no_trade_band: float = 0.05,
    model: str = "ridge",
) -> WeightFn:
    """学習スライスのみから金銭管理アロケータ（weight_fn）を構築する。

    合成順序: combined_teacher（予測+分解）→ ボラ目標（サイジング）→
    回転抑制（保持+no-tradeバンド、実際に執行される最終ウェイト）。

    Args:
        train_fs: 適合に使う学習スライス（**これだけ**を使う=因果的）
        use_ridge: 相対アルファ成分（市場中立）を使うか
        use_trend: 方向性ベータ成分（時系列モメンタム）を使うか
        ridge_target: 予測対象。cs_demean=市場中立の相対アルファ推奨
        target_vol: >0 ならボラ目標スケーリングを重ねる（年率）。0で無効
        rebalance_every: 目標を再計算する間隔（バー数）。回転抑制の主ノブ
        no_trade_band: 目標変化グロスがこの値未満なら据え置く微小取引禁止帯
        model: アルファ予測器。ridge=線形（既定）/ gbm=LightGBM
    """
    teacher = combined_teacher(
        train_fs,
        use_ridge=use_ridge,
        use_trend=use_trend,
        horizon=horizon,
        ridge_target=ridge_target,
        model=model,
    )
    if target_vol and target_vol > 0:
        teacher = _vol_targeted(teacher, target_vol, lookback=vol_lookback)
    return _turnover_controlled(
        teacher, rebalance_every=rebalance_every, no_trade_band=no_trade_band
    )


def evaluate_money_manager(
    train_fs: FeatureSet,
    test_fs: FeatureSet,
    horizon: int = 4,
    use_ridge: bool = True,
    use_trend: bool = True,
    ridge_target: str = "cs_demean",
    target_vol: float = 0.0,
    rebalance_every: int = 24,
    no_trade_band: float = 0.05,
    model: str = "ridge",
    name: str = "money_manager",
    fee_rate: float = 0.0005,
    spread_rate: float = 0.0002,
    impact_rate: float = 0.0001,
    min_trade_delta: float = 0.02,
    cost_multiplier: float = 1.0,
) -> StrategyResult:
    """train_fs で適合し、test_fs で simulate_strategy（RL/ベースラインと同一の
    sqrt-impact執行モデル）を用いて評価する。

    train_fs と test_fs は列（特徴量）構成が一致している必要がある
    （同一パイプライン由来なら自動的に一致する）。
    """
    if train_fs.n_features != test_fs.n_features:
        raise ValueError(
            f"train/testの特徴量数が不一致: {train_fs.n_features} != "
            f"{test_fs.n_features}（同一パイプライン由来のスライスを渡すこと）"
        )
    teacher = build_money_manager(
        train_fs,
        horizon=horizon,
        use_ridge=use_ridge,
        use_trend=use_trend,
        ridge_target=ridge_target,
        target_vol=target_vol,
        rebalance_every=rebalance_every,
        no_trade_band=no_trade_band,
        model=model,
    )
    return simulate_strategy(
        test_fs,
        teacher,
        name=name,
        fee_rate=fee_rate,
        spread_rate=spread_rate,
        impact_rate=impact_rate,
        min_trade_delta=min_trade_delta,
        cost_multiplier=cost_multiplier,
    )
