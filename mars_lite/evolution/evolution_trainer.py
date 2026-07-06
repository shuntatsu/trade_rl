"""
Evolution Training Loop

PBT-MAP-Elites の実行ループ。
Population 全体の学習・評価・進化を管理する。
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

from stable_baselines3 import PPO

from .behavior_utils import evaluate_agent_with_descriptors
from .grid_archive import GridArchive, Individual
from .pbt_manager import PBTManager


class EvolutionTrainer:
    """
    PBT-MAP-Elites Training Loop
    """

    def __init__(
        self,
        make_train_env_fn,
        make_eval_env_fn,
        base_hyperparams: Dict[str, Any],
        population_size: int = 25,
        steps_per_generation: int = 10000,
        eval_episodes: int = 3,
        output_dir: str = "outputs/evolution",
        grid_bins: int = 5,
        device: str = "auto",
    ):
        """
        Args:
            make_train_env_fn: 訓練環境を作成する関数
            make_eval_env_fn: 評価環境を作成する関数
            base_hyperparams: ベースハイパーパラメータ
            population_size: 集団サイズ
            steps_per_generation: 世代あたりの学習ステップ数
            eval_episodes: 評価エピソード数
            output_dir: 出力ディレクトリ
            grid_bins: グリッドの分割数
        """
        self.make_train_env_fn = make_train_env_fn
        self.make_eval_env_fn = make_eval_env_fn
        self.base_hyperparams = base_hyperparams
        self.population_size = population_size
        self.steps_per_generation = steps_per_generation
        self.eval_episodes = eval_episodes
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = device

        # Grid Archive
        self.archive = GridArchive(
            long_bias_bins=grid_bins, vol_exposure_bins=grid_bins
        )

        # PBT Manager
        self.pbt_manager = PBTManager(population_size=population_size)

        # Population
        self.population: List[Individual] = []

        # Agents (PPO instances)
        self.agents: List[Optional[PPO]] = [None] * population_size

    def initialize_population(self):
        """初期集団を生成"""
        print("[EvolutionTrainer] Initializing population...")
        self.population = self.pbt_manager.initialize_population(self.base_hyperparams)

        for i, ind in enumerate(self.population):
            print(f"  Individual {i}: lr={ind.hyperparams.get('learning_rate', 0):.2e}")

    def train_generation(
        self, generation: int, callbacks: List[Any] = None, abort_event=None
    ):
        """
        1世代分の学習

        Args:
            generation: 現在の世代番号
            callbacks: コールバックリスト
            abort_event: 停止イベント
        """
        print(f"\n{'=' * 60}")
        print(f"Generation {generation}")
        print(f"{'=' * 60}")

        # 各個体を学習
        for i, ind in enumerate(self.population):
            if abort_event and abort_event.is_set():
                print(f"[EvolutionTrainer] Aborting train_generation at individual {i}")
                return

            print(f"\n[Individual {i}] Training...")

            # Agent 作成 or ロード
            if self.agents[i] is None or ind.model_path is None:
                # 新規作成
                env = self.make_train_env_fn()

                self.agents[i] = PPO(
                    "MlpPolicy",
                    env,
                    learning_rate=ind.hyperparams.get("learning_rate", 3e-4),
                    gamma=ind.hyperparams.get("gamma", 0.99),
                    ent_coef=ind.hyperparams.get("ent_coef", 0.01),
                    clip_range=ind.hyperparams.get("clip_range", 0.2),
                    device=self.device,
                    verbose=1,
                )
            else:
                # 既存モデルをロード
                env = self.make_train_env_fn()
                self.agents[i] = PPO.load(ind.model_path, env=env, device=self.device)

            # 学習
            self.agents[i].learn(
                total_timesteps=self.steps_per_generation,
                callback=callbacks,
                log_interval=100,
            )
            ind.training_steps += self.steps_per_generation

            # モデル保存
            models_dir = self.output_dir / "models"
            models_dir.mkdir(parents=True, exist_ok=True)
            model_path = models_dir / f"gen{generation}_ind{i}.zip"
            self.agents[i].save(str(model_path))
            ind.model_path = str(model_path)

        print(f"\n[Generation {generation}] Training complete.")

    def evaluate_population(self, abort_event=None):
        """集団全体を評価"""
        print("\n[EvolutionTrainer] Evaluating population...")

        eval_env = self.make_eval_env_fn()

        for i, ind in enumerate(self.population):
            if abort_event and abort_event.is_set():
                print(
                    f"[EvolutionTrainer] Aborting evaluate_population at individual {i}"
                )
                return

            if self.agents[i] is None:
                continue

            # 評価
            result = evaluate_agent_with_descriptors(
                self.agents[i],
                eval_env,
                n_episodes=self.eval_episodes,
                abort_event=abort_event,
            )

            # Individual を更新
            ind.fitness = result["fitness"]
            ind.long_bias = result["long_bias"]
            ind.vol_exposure = result["vol_exposure"]

            # Archive に追加
            added = self.archive.add(ind)

            if added:
                self._log_elite(
                    ind, self.current_gen if hasattr(self, "current_gen") else -1
                )

            print(
                f"  Ind {i}: Fitness={ind.fitness:.2f}, "
                f"LongBias={ind.long_bias:.2f}, VolExp={ind.vol_exposure:.2f} "
                f"{'[ADDED]' if added else ''}"
            )

    def evolve_population(self):
        """PBT で集団を進化"""
        print("\n[EvolutionTrainer] Exploiting & Exploring...")
        self.population = self.pbt_manager.exploit_and_explore(
            self.population, self.archive
        )

    def run(
        self, n_generations: int = 10, callbacks: List[Any] = None, abort_event=None
    ):
        """
        Evolution Training の実行

        Args:
            n_generations: 世代数
            callbacks: コールバックリスト
            abort_event: 停止イベント
        """
        self.initialize_population()

        for gen in range(n_generations):
            if abort_event and abort_event.is_set():
                print(f"[EvolutionTrainer] Aborting run at generation {gen}")
                break

            # 学習
            self.current_gen = gen  # For logging interaction
            self.train_generation(gen, callbacks=callbacks, abort_event=abort_event)

            if abort_event and abort_event.is_set():
                print(
                    f"[EvolutionTrainer] Aborting run after generation {gen} training"
                )
                break

            # 評価
            self.evaluate_population(abort_event=abort_event)

            if abort_event and abort_event.is_set():
                print(
                    f"[EvolutionTrainer] Aborting run after evaluation at generation {gen}"
                )
                break

            # 統計表示
            stats = self.archive.get_stats()
            print("\n[Archive Stats]")
            print(f"  Coverage: {stats['coverage']:.1%}")
            print(f"  Individuals: {stats['num_individuals']}")
            print(f"  Max Fitness: {stats['max_fitness']:.2f}")
            print(f"  Mean Fitness: {stats['mean_fitness']:.2f}")

            # Archive 保存
            self.archive.save(self.output_dir / f"archive_gen{gen}")

            # 進化（最終世代以外）
            if gen < n_generations - 1:
                if abort_event and abort_event.is_set():
                    break
                self.evolve_population()

        print(f"\n{'=' * 60}")
        print("Evolution Training Complete!")
        print(f"{'=' * 60}")

        # 最終 Archive 保存
        self.archive.save(self.output_dir / "archive_final")

    def _log_elite(self, ind: Any, gen_idx: int) -> None:
        """ユーザー要望: シンプルなエリートメモをルートに保存"""
        try:
            log_path = Path("elites_memo.txt")  # Root of execution

            with open(log_path, "a", encoding="utf-8") as f:
                # Header if new
                if log_path.stat().st_size == 0:
                    f.write(
                        "Generation | Fitness | LongBias | VolExp | Model Path | Params\n"
                    )
                    f.write("-" * 80 + "\n")

                # Content
                # Extract interesting params (LR, etc)
                params_str = str(ind.hyperparams)

                line = (
                    f"Gen {gen_idx} | Fit {ind.fitness:.2f} | "
                    f"LB {ind.long_bias:.2f} | Vol {ind.vol_exposure:.2f} | "
                    f"{ind.model_path} | {params_str}\n"
                )
                f.write(line)

        except Exception as e:
            print(f"[EvolutionTrainer] Warining: Failed to write elite memo: {e}")
