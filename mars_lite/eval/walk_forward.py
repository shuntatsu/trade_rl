"""
ウォークフォワード評価モジュール

「学習期間 → purge → 検証期間」をスライドさせ、
各foldでRLエージェントを学習・OOS評価し、ベースラインと並記する。
複数シードの分布で報告する（1本の良い曲線を信用しない）。

出力: JSONレポート + エクイティカーブ図
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np

from mars_lite.eval.strategy_metrics import (
    infer_bars_per_year,
    reannualize_strategy_results,
)
from mars_lite.features.feature_pipeline import FeatureSet
from mars_lite.learning.baselines import StrategyResult, run_all_baselines
from mars_lite.trading.execution import FEE_KWARG_KEYS
from mars_lite.utils.metrics import deflated_sharpe_ratio

TrainFn = Callable[[FeatureSet, int], object]


@dataclass
class FoldResult:
    fold: int
    train_bars: int
    test_bars: int
    agent_by_seed: List[Dict]
    baselines: Dict[str, Dict]


@dataclass
class WalkForwardReport:
    folds: List[FoldResult] = field(default_factory=list)
    config: Dict = field(default_factory=dict)
    dsr: Dict = field(default_factory=dict)

    def summary(self) -> Dict:
        """fold横断のエージェント成績分布とベースライン比較"""
        agent_sharpes, agent_returns = [], []
        base_sharpes: Dict[str, List[float]] = {}
        for fold in self.folds:
            agent_sharpes += [item["sharpe"] for item in fold.agent_by_seed]
            agent_returns += [item["total_return"] for item in fold.agent_by_seed]
            for name, baseline in fold.baselines.items():
                base_sharpes.setdefault(name, []).append(baseline["sharpe"])

        def stats(values):
            if not values:
                return {}
            array = np.array(values)
            return {
                "mean": float(array.mean()),
                "median": float(np.median(array)),
                "min": float(array.min()),
                "max": float(array.max()),
            }

        return {
            "n_folds": len(self.folds),
            "agent_sharpe": stats(agent_sharpes),
            "agent_total_return": stats(agent_returns),
            "baseline_sharpe_mean": {
                key: float(np.mean(value)) for key, value in base_sharpes.items()
            },
        }

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": self.config,
            "summary": self.summary(),
            "deflated_sharpe": self.dsr,
            "folds": [
                {
                    "fold": fold.fold,
                    "train_bars": fold.train_bars,
                    "test_bars": fold.test_bars,
                    "agent_by_seed": fold.agent_by_seed,
                    "baselines": fold.baselines,
                }
                for fold in self.folds
            ],
        }
        with open(path, "w", encoding="utf-8") as file:
            json.dump(payload, file, indent=2, ensure_ascii=False)


def evaluate_agent_on_slice(
    agent, fs: FeatureSet, cost_multiplier: float = 1.0, **env_kwargs
) -> Dict:
    """FeatureSetスライス全体を1エピソードとして決定的評価"""
    from mars_lite.env.portfolio_env import PortfolioTradingEnv

    env = PortfolioTradingEnv(
        fs,
        episode_bars=fs.n_bars - 2,
        cost_multiplier=cost_multiplier,
        **env_kwargs,
    )
    obs, _ = env.reset(options={"start_idx": 0})
    done = False
    equity = [env.portfolio_value]
    info: Dict = {}
    has_disagreement = hasattr(agent, "disagreement")
    while not done:
        action, _ = agent.predict(obs, deterministic=True)
        if has_disagreement:
            env.disagreement = agent.disagreement(obs)
        obs, _, term, trunc, info = env.step(action)
        equity.append(env.portfolio_value)
        done = term or trunc

    return {
        "total_return": env.portfolio_value / env.initial_capital - 1.0,
        "sharpe": info.get("sharpe", 0.0),
        "max_drawdown": info.get("max_drawdown", 0.0),
        "n_trades": info.get("n_trades", 0),
        "turnover_total": info.get("turnover_total", 0.0),
        "funding_pnl": info.get("funding_pnl", 0.0),
        "hold_pct": info.get("hold_pct", 0.0),
        "equity_curve": [float(value) for value in equity],
    }


def run_walk_forward(
    fs: FeatureSet,
    train_fn: TrainFn,
    n_folds: int = 3,
    train_ratio_per_fold: float = 0.6,
    purge_bars: int = 24,
    seeds: Optional[List[int]] = None,
    cost_multiplier: float = 1.0,
    env_kwargs: Optional[Dict] = None,
    verbose: bool = True,
) -> WalkForwardReport:
    """
    ウォークフォワード検証を実行。

    fold k は [0, split_k) で学習し、purge後の [split_k+purge, split_{k+1}) で評価。
    """
    del train_ratio_per_fold
    seeds = seeds or [0, 1, 2]
    env_kwargs = env_kwargs or {}
    bars_per_year = infer_bars_per_year(fs)
    report = WalkForwardReport(
        config={
            "n_folds": n_folds,
            "purge_bars": purge_bars,
            "seeds": seeds,
            "cost_multiplier": cost_multiplier,
            "n_bars_total": fs.n_bars,
            "bars_per_year": bars_per_year,
        }
    )

    edges = np.linspace(int(fs.n_bars * 0.4), fs.n_bars, n_folds + 1).astype(int)
    fold_return_series: List[np.ndarray] = []
    trial_sharpes: List[float] = []

    for fold_number in range(n_folds):
        train_end = edges[fold_number]
        test_start = train_end + purge_bars
        test_end = edges[fold_number + 1]
        if test_end - test_start < 50:
            continue

        train_fs = fs.slice(0, train_end)
        test_fs = fs.slice(test_start, test_end)

        if verbose:
            print(
                f"[Fold {fold_number}] train: {train_fs.n_bars} bars, "
                f"test: {test_fs.n_bars} bars"
            )

        agent_results = []
        seed_returns: List[np.ndarray] = []
        seed_sharpes: List[float] = []
        for seed in seeds:
            agent = train_fn(train_fs, seed)
            result = evaluate_agent_on_slice(
                agent,
                test_fs,
                cost_multiplier=cost_multiplier,
                **env_kwargs,
            )
            result["seed"] = seed
            equity = np.asarray(result.pop("equity_curve", []), dtype=np.float64)
            sharpe = float(result.get("sharpe", 0.0))
            if len(equity) > 2:
                seed_returns.append(np.diff(np.log(np.clip(equity, 1e-9, None))))
                seed_sharpes.append(sharpe)
            trial_sharpes.append(sharpe)
            agent_results.append(result)
            if verbose:
                print(
                    f"  seed {seed}: ret={result['total_return']:+.4f} "
                    f"sharpe={result['sharpe']:+.2f} trades={result['n_trades']}"
                )

        if seed_returns:
            median_index = int(np.argsort(seed_sharpes)[len(seed_sharpes) // 2])
            fold_return_series.append(seed_returns[median_index])

        fee_kwargs = {
            key: env_kwargs[key] for key in FEE_KWARG_KEYS if key in env_kwargs
        }
        baseline_results = run_all_baselines(
            fs,
            cost_multiplier=cost_multiplier,
            start_idx=test_start,
            end_idx=test_end,
            **fee_kwargs,
        )
        baselines = {
            name: result.to_dict()
            for name, result in reannualize_strategy_results(
                baseline_results,
                bars_per_year=bars_per_year,
            ).items()
        }

        report.folds.append(
            FoldResult(
                fold=fold_number,
                train_bars=train_fs.n_bars,
                test_bars=test_fs.n_bars,
                agent_by_seed=agent_results,
                baselines=baselines,
            )
        )

    if fold_return_series:
        oos_returns = np.concatenate(fold_return_series)
        report.dsr = deflated_sharpe_ratio(
            oos_returns,
            trial_sharpes,
            annualization_factor=bars_per_year,
        )

    return report


def plot_comparison(
    agent_result: Dict,
    baseline_results: Dict[str, StrategyResult],
    output_path: Path,
    title: str = "Agent vs Baselines (OOS)",
) -> None:
    """エージェントとベースラインのエクイティカーブを描画"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if "equity_curve" in agent_result:
        curve = np.array(agent_result["equity_curve"])
        plt.plot(curve / curve[0], label="RL Agent", linewidth=2.2)
    for name, result in baseline_results.items():
        plt.plot(result.equity_curve, label=name, alpha=0.75)

    plt.title(title)
    plt.xlabel("Bars")
    plt.ylabel("Equity (normalized)")
    plt.legend()
    plt.grid(alpha=0.3)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=130)
    plt.close()
