"""
建玉クラウディング戦略（オーダーフロー/建玉を一次シグナルにした市場中立アルファ）

背景（実データ診断で発見）: このセッションでtrend・相対アルファ(Ridge/GBM)・
carry・RLをすべて反証した後、オーダーフロー/建玉系の特徴量を個別に
クロスセクショナルICで測ると、**建玉のクロスセクショナルz-score(csz_oi)が
24時間ホライズンで cs_demean IC ≈ -0.069**（閾値0.02の3倍超、GBMが92特徴から
得た0.022の3倍）という、セッション中最強のシグナルを持っていた。それまで
horizon=4で見ていたため見逃していた（h4では-0.030、h24で-0.069とホライズンで
単調に強くなる = 遅い実シグナルの特徴）。

経済的仮説: ある銘柄に相対的に建玉が積み上がる = レバレッジ・ポジションの
過密（多くの場合ロング過多）。過密なトレードはその後アンダーパフォームする
（ポジション解消・逆行）。よって「相対的に建玉が積み上がった銘柄をショート、
低い銘柄をロング」する市場中立戦略になる。

実データdev(6fold)検証: コスト0x/1x/2xすべてで中央値プラス
(+4.05%/+3.97%/+3.89%)、6/6 foldでtrend_followingに勝ち、bootstrap下限
+1.40(有意にTF超え)。turnoverが低くコスト非感応（建玉z-scoreは日次で動く
遅いシグナルのため）。funding(csz_funding)を混ぜると悪化したのでOI単独が最良。

因果性: csz_oi は各バーの同時刻クロスセクショナルz-score（同バーの全銘柄
データのみで計算）なので fs.features[t] を読むのは因果安全。feature_pipeline
のリーク自己検査(shuffle_ic≈0)で担保済み。

注意（データ品質）: Binanceの建玉(OI)はRESTの直近分 + data.binance.vision の
metrics ZIP フォールバックで構成される。OIに欠損・前方補完のアーティファクトが
あるとシグナルが偽物になりうるが、ICがh2→h24で単調に強くなる（ノイズなら
起きない）ことは実シグナルの傍証。ライブ運用前にOIデータの鮮度・整合性を
別途検証すること。

【重要・独立検証の結果（2023-08〜2025-03、7銘柄・別ユニバース）】
別の未接触期間で scripts/validate_crowding_signal.py により再検証したところ:
  - OIデータ品質は良好（nan0%・前方補完アーティファクトなし・timestamp単調）
  - csz_oi の h24 cs_demean IC は -0.046 と**再現**した（factorとしては実在）
  - **しかし戦略としては再現せず**: 6-fold walk-forward で cost2x median -3.58%,
    Sharpe -1.04, TF超え 1/6, bootstrap vs TF CI=[-7.14,-1.06]（有意に劣後）,
    DSR 0.000。同期間で trend_following は +14.18%(Sharpe2.24)と勝った。
これは「IC(統計的予測力) ≠ 収益性」の典型例。csz_oiは実在するが**弱い**
ファクター(IC≈-0.05〜-0.07)で、弱いファクターの戦略収益は局面依存で頑健でない。
dev(2024-11〜2026-07)の +3.89% はその期間に固有だった可能性が高く、DSR 0.229の
警告どおりだった。**この戦略を頑健な検証済みアルファとして扱ってはならない。**
"""

from typing import Callable

import numpy as np

from mars_lite.features.feature_pipeline import FeatureSet

WeightFn = Callable[[FeatureSet, int, np.ndarray], np.ndarray]

DEFAULT_REBALANCE_EVERY = 24  # 日次（1h足×24）。csz_oiは24hホライズンで最強
DEFAULT_GROSS = 1.0
DEFAULT_OI_FEATURE = "csz_oi"
DEFAULT_FUNDING_FEATURE = "csz_funding"


def make_crowding_strategy(
    rebalance_every: int = DEFAULT_REBALANCE_EVERY,
    gross: float = DEFAULT_GROSS,
    use_oi: bool = True,
    use_funding: bool = False,
    oi_feature: str = DEFAULT_OI_FEATURE,
    funding_feature: str = DEFAULT_FUNDING_FEATURE,
) -> WeightFn:
    """建玉/funding のクロスセクショナルz-scoreに基づく市場中立クラウディング戦略。

    クラウディングスコアが高い（相対的に建玉・fundingが積み上がった）銘柄を
    ショート、低い銘柄をロングする。CSデミーンで厳密に dollar-neutral（Σw=0）。

    Args:
        rebalance_every: リバランス間隔（バー数）。既定24（日次）
        gross: 目標グロス（Σ|w|）
        use_oi: 建玉クラウディング（csz_oi）を使うか
        use_funding: fundingクラウディング（csz_funding）を使うか。実データ
            では OI単独が最良で、funding併用は成績を悪化させた（既定off）
        oi_feature/funding_feature: 参照する特徴量名（FeatureSet.feature_names
            に存在する必要がある）
    """
    if not (use_oi or use_funding):
        raise ValueError("use_oi と use_funding の少なくとも一方を有効にすること")

    # feature名 -> index はFeatureSetごとに解決（sliceしても列順は不変）。
    _idx_cache: dict = {}

    def _indices(fs: FeatureSet):
        key = id(fs)
        if key not in _idx_cache:
            names = fs.feature_names
            oi_i = names.index(oi_feature) if use_oi else None
            fund_i = names.index(funding_feature) if use_funding else None
            _idx_cache[key] = (oi_i, fund_i)
        return _idx_cache[key]

    def teacher(fs: FeatureSet, t: int, prev: np.ndarray) -> np.ndarray:
        if t % rebalance_every != 0 and np.any(prev):
            return prev

        oi_i, fund_i = _indices(fs)
        score = np.zeros(fs.n_symbols)
        if oi_i is not None:
            score = score + fs.features[t][:, oi_i]
        if fund_i is not None:
            score = score + fs.features[t][:, fund_i]

        # クラウディング高い(score高) = ショート。CSデミーンで Σw=0 を厳密保証
        raw = -(score - score.mean())
        s = float(np.abs(raw).sum())
        if s < 1e-12:
            return np.zeros(fs.n_symbols)
        return raw / s * gross

    return teacher
