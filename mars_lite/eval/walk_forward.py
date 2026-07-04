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

from mars_lite.features.feature_pipeline import FeatureSet
from mars_lite.learning.baselines import run_all_baselines, StrategyResult

# エージェント学習関数の型: (train_fs, seed) -> agent
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

    def summary(self) -> Dict:
        """fold横断のエージェント成績分布とベースライン比較"""
        agent_sharpes, agent_returns = [], []
        base_sharpes: Dict[str, List[float]] = {}
        for f in self.folds:
            agent_sharpes += [a["sharpe"] for a in f.agent_by_seed]
            agent_returns += [a["total_return"] for a in f.agent_by_seed]
            for name, b in f.baselines.items():
                base_sharpes.setdefault(name, []).append(b["sharpe"])

        def stats(xs):
            if not xs:
                return {}
            xs = np.array(xs)
            return {
                "mean": float(xs.mean()), "median": float(np.median(xs)),
                "min": float(xs.min()), "max": float(xs.max()),
            }

        return {
            "n_folds": len(self.folds),
            "agent_sharpe": stats(agent_sharpes),
            "agent_total_return": stats(agent_returns),
            "baseline_sharpe_mean": {
                k: float(np.mean(v)) for k, v in base_sharpes.items()
            },
        }

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": self.config,
            "summary": self.summary(),
            "folds": [
                {
                    "fold": f.fold,
                    "train_bars": f.train_bars,
                    "test_bars": f.test_bars,
                    "agent_by_seed": f.agent_by_seed,
                    "baselines": f.baselines,
                }
                for f in self.folds
            ],
        }
        with open(path, "w", encoding="utf-8") as fp:
            json.dump(payload, fp, indent=2, ensure_ascii=False)


def evaluate_agent_on_slice(
    agent, fs: FeatureSet, cost_multiplier: float = 1.0, **env_kwargs
) -> Dict:
    """FeatureSetスライス全体を1エピソードとして決定的評価"""
    from mars_lite.env.portfolio_env import PortfolioTradingEnv

    env = PortfolioTradingEnv(
        fs, episode_bars=fs.n_bars - 2,
        cost_multiplier=cost_multiplier, **env_kwargs,
    )
    obs, _ = env.reset(options={"start_idx": 0})
    done = False
    equity = [env.portfolio_value]
    info: Dict = {}
    # SeedEnsembleなら不一致度を毎ステップ後処理へ渡す（不確実時にグロス縮小）
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
        "equity_curve": [float(x) for x in equity],
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
    ウォークフォワード検証を実行

    fold k は [0, split_k) で学習し、purge後の [split_k+purge, split_{k+1}) で評価。
    """
    seeds = seeds or [0, 1, 2]
    env_kwargs = env_kwargs or {}
    report = WalkForwardReport(config={
        "n_folds": n_folds, "purge_bars": purge_bars,
        "seeds": seeds, "cost_multiplier": cost_multiplier,
        "n_bars_total": fs.n_bars,
    })

    edges = np.linspace(int(fs.n_bars * 0.4), fs.n_bars, n_folds + 1).astype(int)

    for k in range(n_folds):
        train_end = edges[k]
        test_start = train_end + purge_bars
        test_end = edges[k + 1]
        if test_end - test_start < 50:
            continue

        train_fs = fs.slice(0, train_end)
        test_fs = fs.slice(test_start, test_end)

        if verbose:
            print(f"[Fold {k}] train: {train_fs.n_bars} bars, test: {test_fs.n_bars} bars")

        agent_results = []
        for seed in seeds:
            agent = train_fn(train_fs, seed)
            res = evaluate_agent_on_slice(
                agent, test_fs, cost_multiplier=cost_multiplier, **env_kwargs
            )
            res["seed"] = seed
            res.pop("equity_curve", None)
            agent_results.append(res)
            if verbose:
                print(f"  seed {seed}: ret={res['total_return']:+.4f} "
                      f"sharpe={res['sharpe']:+.2f} trades={res['n_trades']}")

        baselines = {
            name: r.to_dict()
            for name, r in run_all_baselines(
                test_fs, cost_multiplier=cost_multiplier
            ).items()
        }

        report.folds.append(FoldResult(
            fold=k,
            train_bars=train_fs.n_bars,
            test_bars=test_fs.n_bars,
            agent_by_seed=agent_results,
            baselines=baselines,
        ))

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

    fig, ax = plt.subplots(figsize=(11, 6))
    if "equity_curve" in agent_result:
        curve = np.array(agent_result["equity_curve"])
        ax.plot(curve / curve[0], label="RL Agent", linewidth=2.2, color="#d62728")
    for name, r in baseline_results.items():
        ax.plot(r.equity_curve, label=name, alpha=0.75)

    ax.set_title(title)
    ax.set_xlabel("Bars (1h)")
    ax.set_ylabel("Equity (normalized)")
    ax.legend()
    ax.grid(alpha=0.3)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=130)
    plt.close(fig)
