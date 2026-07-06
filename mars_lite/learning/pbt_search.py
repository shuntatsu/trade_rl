"""
ハイパーパラメータのPopulation-Based Training（PBT）探索

γ=0.5のように手動で見つけた「効くパラメータ」がまだ埋まっている可能性が高い。
複数個体を並行学習し、定期的に「下位個体を上位個体で置換（exploit）＋摂動
（explore）」してハイパーパラメータ空間を進化的に探索する。fitnessは検証
スライスのスコア。

短時間で回すため、各個体は少ステップずつ学習し世代ごとにexploit/exploreする
軽量版PBT。探索対象: gamma, ent_coef, lambda_turnover, reward_scale,
learning_rate。
"""

import copy
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np

# 探索空間: 名前 -> (最小, 最大, log空間か)
DEFAULT_SEARCH_SPACE = {
    "gamma": (0.3, 0.95, False),
    "ent_coef": (0.0005, 0.02, True),
    "lambda_turnover": (0.01, 0.2, True),
    "reward_scale": (30.0, 300.0, True),
    "learning_rate": (1e-4, 6e-4, True),
}


def _sample(rng, space) -> Dict[str, float]:
    hp = {}
    for name, (lo, hi, log) in space.items():
        if log:
            hp[name] = float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
        else:
            hp[name] = float(rng.uniform(lo, hi))
    return hp


def _perturb(rng, hp, space, factor=1.3) -> Dict[str, float]:
    """explore: 各値を ×[1/factor, factor] で摂動し範囲にクリップ"""
    out = {}
    for name, v in hp.items():
        lo, hi, _ = space[name]
        mult = rng.choice([1.0 / factor, factor])
        out[name] = float(np.clip(v * mult, lo, hi))
    return out


@dataclass
class PBTResult:
    best_hp: Dict[str, float]
    best_score: float
    history: List[Dict] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "best_hp": self.best_hp,
            "best_score": self.best_score,
            "history": self.history,
        }


def run_pbt(
    train_eval_fn: Callable[[Dict[str, float], int], float],
    population_size: int = 6,
    n_generations: int = 4,
    exploit_frac: float = 0.34,
    space: Optional[Dict] = None,
    seed: int = 0,
    verbose: int = 1,
) -> PBTResult:
    """
    軽量PBTを実行

    Args:
        train_eval_fn: (hp, seed) -> validation_score を返す関数。
            通常は train_ppo で少ステップ学習し検証スコアを返すラッパー。
        population_size: 個体数
        n_generations: 世代数
        exploit_frac: 各世代で置換する下位個体の割合
        space: 探索空間（DEFAULT_SEARCH_SPACE）

    Returns:
        PBTResult（最良ハイパーパラメータ）
    """
    space = space or DEFAULT_SEARCH_SPACE
    rng = np.random.default_rng(seed)

    pop = [_sample(rng, space) for _ in range(population_size)]
    scores = [train_eval_fn(hp, seed + i) for i, hp in enumerate(pop)]
    history = []

    for gen in range(n_generations):
        order = np.argsort(scores)  # 昇順（最後が最良）
        n_exploit = max(1, int(population_size * exploit_frac))
        losers = order[:n_exploit]
        winners = order[-n_exploit:]

        for li, wi in zip(losers, rng.permutation(winners)):
            # exploit: 勝者のHPをコピー → explore: 摂動
            pop[li] = _perturb(
                rng, copy.deepcopy(pop[winners[-1] if wi is None else wi]), space
            )
            scores[li] = train_eval_fn(pop[li], seed + gen * 100 + li)

        best_i = int(np.argmax(scores))
        history.append(
            {
                "generation": gen,
                "best_score": float(scores[best_i]),
                "best_hp": {k: round(v, 6) for k, v in pop[best_i].items()},
            }
        )
        if verbose:
            hp = pop[best_i]
            print(
                f"[PBT gen {gen}] best={scores[best_i]:+.4f} "
                f"gamma={hp['gamma']:.2f} ent={hp['ent_coef']:.4f} "
                f"lam={hp['lambda_turnover']:.3f} rs={hp['reward_scale']:.0f}",
                flush=True,
            )

    best_i = int(np.argmax(scores))
    return PBTResult(
        best_hp=pop[best_i], best_score=float(scores[best_i]), history=history
    )
