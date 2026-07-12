import json
from types import SimpleNamespace

import numpy as np

from mars_lite.pipeline import evaluator


class _FeatureSet:
    n_bars = 200
    n_symbols = 1
    n_features = 1
    symbols = ["BTCUSDT"]
    feature_names = ["ret"]
    global_feature_names: list[str] = []
    timestamps = np.arange(200)
    features = np.ones((200, 1, 1), dtype=np.float64)
    close = np.ones((200, 1), dtype=np.float64)
    global_features = np.empty((200, 0), dtype=np.float64)

    def slice(self, start: int, stop: int):
        sliced = _FeatureSet()
        sliced.n_bars = stop - start
        return sliced


class _SignalReport:
    passed = True

    def summary(self) -> str:
        return "signal ok"

    def to_dict(self) -> dict[str, object]:
        return {"passed": True}


class _Baseline:
    def __init__(self, name: str, total_return: float = 0.0) -> None:
        self.name = name
        self.total_return = total_return
        self.sharpe = 0.0
        self.max_drawdown = 0.0
        self.turnover_total = 0.0

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "total_return": self.total_return,
            "sharpe": self.sharpe,
            "max_drawdown": self.max_drawdown,
            "turnover_total": self.turnover_total,
        }


class _Agent:
    def save(self, path: str) -> None:
        return None


def test_p0_report_records_effective_candidate_timing(tmp_path, monkeypatch) -> None:
    import mars_lite.data.sources as sources
    import mars_lite.features.feature_pipeline as feature_pipeline

    class _Source:
        symbols = ["BTCUSDT"]

        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

    class _Pipeline:
        def __init__(self, symbols) -> None:
            self.symbols = symbols

        def build(self, source):
            return _FeatureSet()

    monkeypatch.setattr(sources, "SyntheticSource", _Source)
    monkeypatch.setattr(feature_pipeline, "FeaturePipeline", _Pipeline)
    monkeypatch.setattr(evaluator, "run_signal_check", lambda fs, horizon: _SignalReport())
    monkeypatch.setattr(evaluator, "build_post_processor", lambda args, horizon: object())
    monkeypatch.setattr(evaluator, "build_env_kwargs", lambda args, pp, horizon: {})
    monkeypatch.setattr(evaluator, "train_ppo", lambda **kwargs: _Agent())
    monkeypatch.setattr(
        evaluator,
        "evaluate_agent_on_slice",
        lambda agent, fs, **kwargs: {
            "total_return": 0.1,
            "sharpe": 1.0,
            "max_drawdown": 0.0,
            "turnover_total": 1.0,
            "equity_curve": np.asarray([1.0, 1.1]),
        },
    )
    monkeypatch.setattr(
        evaluator,
        "run_all_baselines",
        lambda fs, **kwargs: {
            "equal_weight_bh": _Baseline("equal_weight_bh", total_return=0.0)
        },
    )
    monkeypatch.setattr(evaluator, "report_comparison", lambda *args, **kwargs: None)
    monkeypatch.setattr(evaluator, "plot_comparison", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        evaluator, "generate_and_save_manifest", lambda **kwargs: None
    )

    args = SimpleNamespace(
        days=90,
        alpha_strength=0.002,
        seed=0,
        horizon=12,
        timesteps=10,
        gamma=0.5,
        verbose=0,
        decision_every=4,
        min_trade_delta=0.04,
        lambda_turnover=0.04,
    )

    evaluator.phase_p0(args, tmp_path)

    report = json.loads((tmp_path / "p0_report.json").read_text(encoding="utf-8"))
    assert report["config"] == {
        "horizon": 12,
        "decision_every": 4,
        "days": 90,
    }
