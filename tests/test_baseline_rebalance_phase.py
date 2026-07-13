from __future__ import annotations

from mars_lite.eval.residual_walk_forward import ResidualFoldSpec
from mars_lite.pipeline import residual_walk_forward


def test_diagnostic_baselines_use_full_dataset_and_absolute_oos_bounds(
    monkeypatch,
) -> None:
    full_feature_set = object()
    captured: dict[str, object] = {}

    def fake_run_all(fs, **kwargs):
        captured["fs"] = fs
        captured.update(kwargs)
        return {}

    monkeypatch.setattr(residual_walk_forward, "run_all_baselines", fake_run_all)
    spec = ResidualFoldSpec(
        fold=0,
        policy_train_start=0,
        policy_train_end=700,
        checkpoint_validation_start=724,
        checkpoint_validation_end=824,
        configuration_selection_start=848,
        configuration_selection_end=948,
        outer_test_start=972,
        outer_test_end=1_072,
        purge_bars=24,
    )

    result = residual_walk_forward._diagnostic_baselines(
        fs=full_feature_set,
        spec=spec,
        env_kwargs={
            "fee_rate": 0.0005,
            "spread_rate": 0.0002,
            "impact_rate": 0.0001,
        },
        bars_per_year=8_760,
        cost_multiplier=1.0,
        noisy_oracle_ic=0.0,
    )

    assert result == {}
    assert captured["fs"] is full_feature_set
    assert captured["start_idx"] == 972
    assert captured["end_idx"] == 1_072
    assert captured["cost_multiplier"] == 1.0
