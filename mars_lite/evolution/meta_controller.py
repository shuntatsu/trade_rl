"""
メタコントローラモジュール

推論時に環境条件に基づいてMAP-Elitesアーカイブから最適な個体を選択
"""

from typing import Any, Dict, List, Optional, Tuple

from .map_elites import MAPElitesArchive


class MetaController:
    """
    メタコントローラ

    推論時に現在の市場環境を分析し、最適な個体を選択する。
    """

    def __init__(
        self,
        archive: MAPElitesArchive,
        sigma_bins: List[Tuple[float, float]] = None,
        volume_bins: List[Tuple[float, float]] = None,
    ):
        """
        Args:
            archive: MAP-Elitesアーカイブ
            sigma_bins: ボラティリティレジームの区間 [(low_min, low_max), (mid_min, mid_max), ...]
            volume_bins: 出来高レジームの区間
        """
        self.archive = archive

        # デフォルトのレジーム区間
        self.sigma_bins = sigma_bins or [
            (0.0, 0.01),  # 低ボラ
            (0.01, 0.03),  # 中ボラ
            (0.03, 1.0),  # 高ボラ
        ]
        self.volume_bins = volume_bins or [
            (0.0, 0.5),  # 薄い
            (0.5, 1.5),  # 普通
            (1.5, 10.0),  # 厚い
        ]

    def _classify_sigma(self, sigma: float) -> str:
        """ボラティリティをレジームに分類"""
        for i, (low, high) in enumerate(self.sigma_bins):
            if low <= sigma < high:
                return ["low", "medium", "high"][min(i, 2)]
        return "high"

    def _classify_volume(self, rel_volume: float) -> str:
        """相対出来高をレジームに分類"""
        for i, (low, high) in enumerate(self.volume_bins):
            if low <= rel_volume < high:
                return ["thin", "normal", "thick"][min(i, 2)]
        return "thick"

    def _regime_to_behavior(
        self,
        sigma_regime: str,
        volume_regime: str,
    ) -> Dict[str, float]:
        """
        レジームから推奨行動記述子を算出

        Args:
            sigma_regime: "low", "medium", "high"
            volume_regime: "thin", "normal", "thick"

        Returns:
            推奨行動記述子
        """
        # 高ボラ → 低aggressiveness（控えめに）
        # 厚い出来高 → 高aggressiveness（積極的に）
        # 高ボラ耐性：高ボラ時には高耐性個体を選ぶ

        aggressiveness_map = {
            ("low", "thin"): 0.3,
            ("low", "normal"): 0.5,
            ("low", "thick"): 0.7,
            ("medium", "thin"): 0.2,
            ("medium", "normal"): 0.4,
            ("medium", "thick"): 0.6,
            ("high", "thin"): 0.1,
            ("high", "normal"): 0.3,
            ("high", "thick"): 0.5,
        }

        volatility_tolerance_map = {
            "low": 0.5,
            "medium": 1.0,
            "high": 1.5,
        }

        return {
            "aggressiveness": aggressiveness_map.get(
                (sigma_regime, volume_regime), 0.4
            ),
            "volatility_tolerance": volatility_tolerance_map.get(sigma_regime, 1.0),
        }

    def select_individual(
        self,
        sigma: float,
        rel_volume: float,
    ) -> Optional[Any]:
        """
        環境条件に基づいて最適な個体を選択

        Args:
            sigma: 現在のボラティリティ
            rel_volume: 相対出来高（v / v_expected）

        Returns:
            選択された個体（なければNone）
        """
        sigma_regime = self._classify_sigma(sigma)
        volume_regime = self._classify_volume(rel_volume)

        target_behavior = self._regime_to_behavior(sigma_regime, volume_regime)

        # アーカイブから最近傍を取得
        individual = self.archive.get_nearest(target_behavior)

        return individual

    def select_with_info(
        self,
        sigma: float,
        rel_volume: float,
    ) -> Dict[str, Any]:
        """
        個体選択と追加情報を返す

        Args:
            sigma: 現在のボラティリティ
            rel_volume: 相対出来高

        Returns:
            選択結果と理由
        """
        sigma_regime = self._classify_sigma(sigma)
        volume_regime = self._classify_volume(rel_volume)
        target_behavior = self._regime_to_behavior(sigma_regime, volume_regime)

        individual = self.archive.get_nearest(target_behavior)

        return {
            "individual": individual,
            "sigma_regime": sigma_regime,
            "volume_regime": volume_regime,
            "target_behavior": target_behavior,
            "actual_behavior": individual.behavior_desc if individual else None,
        }


class SimpleRuleController:
    """
    シンプルなルールベースコントローラ

    MAP-Elitesアーカイブを使わず、ルールで個体を選択。
    デバッグ・ベースライン用。
    """

    def __init__(
        self,
        individuals: Dict[str, Any],
        default_id: str,
    ):
        """
        Args:
            individuals: id -> individual/agent のマッピング
            default_id: デフォルト個体ID
        """
        self.individuals = individuals
        self.default_id = default_id

    def select(
        self,
        sigma: float,
        rel_volume: float,
    ) -> Any:
        """
        ルールベースで個体を選択

        Args:
            sigma: ボラティリティ
            rel_volume: 相対出来高

        Returns:
            選択された個体
        """
        # シンプルなルール例
        if sigma > 0.03:
            # 高ボラ → 保守的な個体
            key = "conservative"
        elif rel_volume > 1.5:
            # 厚い出来高 → 積極的な個体
            key = "aggressive"
        else:
            key = self.default_id

        return self.individuals.get(key, self.individuals[self.default_id])
