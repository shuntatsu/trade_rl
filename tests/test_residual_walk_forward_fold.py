from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from mars_lite.eval.residual_walk_forward import ResidualFoldSpec
from mars_lite.pipeline import residual_walk_forward
from mars_lite.pipeline.residual_walk_forward import run_residual_fold


class _Slice:
    def __init__(self, name: str, n_bars: int):
        self.name = name
        self.n_bars = n_bars


class _FeatureSet:
    def slice(self, start: int, end: int) -> _Slice:
        return _Slice(f"slice:{start}:{end}", end - start)


class _Alpha:
    enabled = True
    dataset_identity = "alpha-inner-train-only"

    def save(self, path: Path) -> None:
        path.write_text("{}\n", encoding="utf-8")


class _FrozenAlphaFactory:
    fitted_on: object | None = None

    @classmethod
    def fit(cls, train_fs, **kwargs):
        cls.fitted_on = train_fs
        return _Alpha()


def _relative(cost: float) -> dict[str, object]:
    hybrid_returns = [0.001, -0.0005]
    shadow_returns = [0.0005, -0.00025]
    return {
        "hybrid": {
            "total_return": 0.02 - cost,
            "sharpe": 1.0,
            "max_drawdown": 0.1,
            "turnover_total": 1.0,
            "total_cost": cost,
            "funding_pnl": 0.0,
            "n_trades": 2,
            "n_base_bars": len(hybrid_returns),
        },
        "shadow": {
            "total_return": 0.01 - cost,
            "sharpe": 0.5,
            "max_drawdown": 0.1,
            "turnover_total": 1.0,
            "total_cost": cost,
            "funding_pnl": 0.0,
            "n_trades": 2,
            "n_base_bars": len(shadow_returns),
        },
        "paired": {
            "excess_total_return": 0.01,
            "excess_log_return": 0.01,
            "mean_base_bar_excess": 0.0,
            "p_value": 0.5,
            "lower_ci": 0.0,
            "upper_ci": 0.0,
            "block_size": 1,
        },
        "actions": {},
        "weight_stages": {},
        "execution": {},
        "return_series": {
            "kind": "base_bar",
            "hybrid": hybrid_returns,
            "shadow": shadow_returns,
        },
    }


def test_fold_fits_alpha_on_policy_train_and_reuses_selected_agent(
    monkeypatch,
    tmp_path: Path,
) -> None:
    calls: list[tuple[object, float]] = []
    selected_agent = object()

    monkeypatch.setattr(
        residual_walk_forward, "FrozenResidualAlpha", _FrozenAlphaFactory
    )
    monkeypatch.setattr(
        residual_walk_forward,
        "run_leak_self_test",
        lambda fs, horizon: {"healthy": True},
    )
    monkeypatch.setattr(residual_walk_forward, "walk_forward_ic", lambda *a, **k: {})
    monkeypatch.setattr(
        residual_walk_forward,
        "evaluate_residual_alpha_gate",
        lambda report: {"passed": True},
    )
    monkeypatch.setattr(
        residual_walk_forward,
        "with_history_context",
        lambda fs, start, end, history_bars: SimpleNamespace(
            feature_set=_Slice(f"context:{start}:{end}", end - start + history_bars),
            scored_bars=end - start,
            start_idx=history_bars,
        ),
    )
    monkeypatch.setattr(
        residual_walk_forward,
        "build_post_processor",
        lambda *a, **k: SimpleNamespace(cfg=SimpleNamespace(bars_per_year=8_760)),
    )
    monkeypatch.setattr(
        residual_walk_forward,
        "build_env_kwargs",
        lambda *a, **k: {
            "fee_rate": 0.0005,
            "spread_rate": 0.0002,
            "impact_rate": 0.0001,
            "decision_every": 4,
        },
    )
    monkeypatch.setattr(
        residual_walk_forward,
        "train_select_residual_candidates",
        lambda **kwargs: SimpleNamespace(
            development_results={"A": {}, "B": {}},
            development_cost2x_results={"A": {}, "B": {}},
            selection={"selected": "B", "policy_mode": "ppo_residual_ensemble"},
            selected_configuration="B",
            selected_agent=selected_agent,
            selected_policies=(),
            selected_model_path=tmp_path / "b.zip",
            selected_model_digest="a" * 64,
            selected_alpha_enabled=False,
        ),
    )

    def fake_evaluate(
        agent,
        fs,
        *,
        env_kwargs,
        bootstrap_seed,
        include_return_series,
    ):
        assert include_return_series is True
        calls.append((agent, float(env_kwargs.get("cost_multiplier", 1.0))))
        return _relative(float(env_kwargs.get("cost_multiplier", 1.0)) * 0.001)

    monkeypatch.setattr(residual_walk_forward, "evaluate_relative_agent", fake_evaluate)
    monkeypatch.setattr(residual_walk_forward, "run_all_baselines", lambda *a, **k: {})

    spec = ResidualFoldSpec(
        fold=1,
        policy_train_start=0,
        policy_train_end=640,
        checkpoint_validation_start=664,
        checkpoint_validation_end=700,
        configuration_selection_start=724,
        configuration_selection_end=800,
        outer_test_start=824,
        outer_test_end=1000,
        purge_bars=24,
    )
    args = SimpleNamespace(
        seed=10,
        ensemble=3,
        horizon=12,
        base_timeframe="1h",
        signal_model="gbm",
        noisy_oracle_ic=0.0,
    )

    report = run_residual_fold(
        fs=_FeatureSet(),
        spec=spec,
        args=args,
        output_dir=tmp_path,
    )

    assert getattr(_FrozenAlphaFactory.fitted_on, "name") == "slice:0:640"
    assert calls == [(selected_agent, 1.0), (selected_agent, 2.0)]
    assert report["selected_configuration"] == "B"
    assert report["selected_model_digest"] == "a" * 64
    assert report["outer_oos"]["model_digest_1x"] == "a" * 64
    assert report["outer_oos"]["model_digest_2x"] == "a" * 64
    assert report["outer_oos"]["same_selected_model_for_cost_scenarios"] is True
    assert report["split"]["outer_test_scored_bars"] == 176
    assert "_return_series_1x" in report
    assert "return_series" not in report["outer_oos"]["relative_1x"]
    assert (tmp_path / "residual_wf" / "fold_1" / "fold_report.json").is_file()
