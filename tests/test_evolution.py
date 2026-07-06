"""
進化戦略モジュールのテスト
"""

import pytest

from mars_lite.evolution.map_elites import (
    BehaviorDescriptor,
    MAPElitesArchive,
    compute_behavior_descriptors,
)
from mars_lite.evolution.meta_controller import MetaController
from mars_lite.learning.population import Individual, PopulationManager


class TestPopulationManager:
    """Population管理のテスト"""

    def test_initialize_population(self):
        """Populationが正しく初期化される"""
        pm = PopulationManager(population_size=8, seed=42)
        population = pm.initialize_population()

        assert len(population) == 8
        assert all(isinstance(ind, Individual) for ind in population)

    def test_hyperparams_in_range(self):
        """ハイパーパラメータが範囲内"""
        pm = PopulationManager(population_size=8, seed=42)
        pm.initialize_population()

        for ind in pm.population:
            for name, value in ind.hyperparams.items():
                min_val, max_val, _ = pm.hyperparam_ranges[name]
                assert min_val <= value <= max_val

    def test_pbt_step(self):
        """PBTステップが動作"""
        pm = PopulationManager(population_size=8, seed=42)
        pm.initialize_population()

        # fitnessを設定
        for i, ind in enumerate(pm.population):
            ind.fitness = float(i)

        # PBTステップ
        old_gen = pm.generation
        pm.pbt_step(exploit_ratio=0.25)

        assert pm.generation == old_gen + 1


class TestMAPElites:
    """MAP-Elitesのテスト"""

    @pytest.fixture
    def archive(self):
        descriptors = [
            BehaviorDescriptor("aggressiveness", 0.0, 1.0, 5),
            BehaviorDescriptor("volatility_tolerance", 0.0, 3.0, 5),
        ]
        return MAPElitesArchive(descriptors)

    def test_add_to_empty_cell(self, archive):
        """空セルへの追加"""
        ind = Individual(id="test1", hyperparams={"lr": 0.001})
        behavior = {"aggressiveness": 0.3, "volatility_tolerance": 1.0}

        added = archive.add(ind, behavior, fitness=0.5)

        assert added
        assert archive.get_by_behavior(behavior) is not None

    def test_replace_better_fitness(self, archive):
        """より良いfitnessで置換"""
        ind1 = Individual(id="test1", hyperparams={"lr": 0.001})
        ind2 = Individual(id="test2", hyperparams={"lr": 0.002})
        behavior = {"aggressiveness": 0.3, "volatility_tolerance": 1.0}

        archive.add(ind1, behavior, fitness=0.3)
        replaced = archive.add(ind2, behavior, fitness=0.8)

        assert replaced
        assert archive.get_by_behavior(behavior).id == "test2"

    def test_no_replace_worse_fitness(self, archive):
        """より悪いfitnessでは置換されない"""
        ind1 = Individual(id="test1", hyperparams={"lr": 0.001})
        ind2 = Individual(id="test2", hyperparams={"lr": 0.002})
        behavior = {"aggressiveness": 0.3, "volatility_tolerance": 1.0}

        archive.add(ind1, behavior, fitness=0.8)
        replaced = archive.add(ind2, behavior, fitness=0.3)

        assert not replaced
        assert archive.get_by_behavior(behavior).id == "test1"

    def test_get_nearest(self, archive):
        """最近傍セルの取得"""
        ind = Individual(id="test1", hyperparams={"lr": 0.001})
        archive.add(
            ind, {"aggressiveness": 0.5, "volatility_tolerance": 1.5}, fitness=0.5
        )

        # 近い位置で検索
        nearest = archive.get_nearest(
            {"aggressiveness": 0.6, "volatility_tolerance": 1.6}
        )

        assert nearest is not None
        assert nearest.id == "test1"


class TestMetaController:
    """メタコントローラのテスト"""

    @pytest.fixture
    def controller(self):
        descriptors = [
            BehaviorDescriptor("aggressiveness", 0.0, 1.0, 5),
            BehaviorDescriptor("volatility_tolerance", 0.0, 3.0, 5),
        ]
        archive = MAPElitesArchive(descriptors)

        # サンプル個体を追加
        for i in range(5):
            ind = Individual(id=f"ind_{i}", hyperparams={"lr": 0.001 * (i + 1)})
            behavior = {
                "aggressiveness": 0.2 * i,
                "volatility_tolerance": 0.6 * i,
            }
            archive.add(ind, behavior, fitness=0.5 + i * 0.1)

        return MetaController(archive)

    def test_select_individual(self, controller):
        """環境条件で個体が選択される"""
        ind = controller.select_individual(sigma=0.02, rel_volume=1.0)

        assert ind is not None
        assert isinstance(ind, Individual)

    def test_select_with_info(self, controller):
        """選択理由が返される"""
        result = controller.select_with_info(sigma=0.02, rel_volume=1.0)

        assert "individual" in result
        assert "sigma_regime" in result
        assert "volume_regime" in result


class TestBehaviorDescriptor:
    """行動記述子計算のテスト"""

    def test_compute_behavior_descriptors(self):
        """執行履歴から行動記述子が計算される"""
        import pandas as pd

        history = pd.DataFrame(
            {
                "quantity": [100, 150, 200, 50],
                "pov": [0.1, 0.15, 0.2, 0.05],
                "sigma": [0.01, 0.02, 0.03, 0.02],
            }
        )

        descriptors = compute_behavior_descriptors(history)

        assert "aggressiveness" in descriptors
        assert "volatility_tolerance" in descriptors
        assert 0 <= descriptors["aggressiveness"] <= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
