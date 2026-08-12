from pathlib import Path

import pytest

from trade_rl.workflows.universal_policy_stage_evaluation import (
    _action_diagnostics,
    _aggregate_stage,
    discover_universal_policy_artifacts,
)


def _write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def test_discovers_causal_stages_before_ordered_rollout_checkpoints(
    tmp_path: Path,
) -> None:
    for stage in ("random", "behavior_cloning", "behavior_cloning_critic"):
        _write(tmp_path / "policy-stages" / stage / "policy.zip", stage.encode())
    _write(tmp_path / "checkpoints/step-000000000384/policy.zip", b"384")
    _write(tmp_path / "checkpoints/step-000000000128/policy.zip", b"128")

    artifacts = discover_universal_policy_artifacts(tmp_path)

    assert [artifact.label for artifact in artifacts] == [
        "random",
        "behavior_cloning",
        "behavior_cloning_critic",
        "rollout_128",
        "rollout_384",
    ]
    assert all(len(artifact.file_digest) == 64 for artifact in artifacts)


def test_action_diagnostics_separate_churn_and_sign_flips() -> None:
    import numpy as np

    result = _action_diagnostics(np.asarray([[0.5], [0.4], [-0.2], [-0.2]]))

    assert result["absolute_target_delta_total"] == pytest.approx(1.2)
    assert result["absolute_target_delta_mean"] == pytest.approx(0.3)
    assert result["buy_sell_sign_flip_count"] == 1


def test_aggregate_preserves_economic_metrics() -> None:
    result = _aggregate_stage(
        [
            {
                "baseline_return": 0.04,
                "baseline_excess_return": -0.06,
                "turnover_multiple_per_day": 3.0,
                "performance": {
                    "gross_return": -0.01,
                    "net_return": -0.02,
                    "cost_total": 100.0,
                    "trade_count": 2,
                    "traded_step_count": 8,
                },
                "actions": {
                    "absolute_target_delta_total": 4.0,
                    "buy_sell_sign_flip_count": 3,
                },
            },
            {
                "baseline_return": 0.02,
                "baseline_excess_return": -0.03,
                "turnover_multiple_per_day": 1.0,
                "performance": {
                    "gross_return": 0.00,
                    "net_return": -0.01,
                    "cost_total": 50.0,
                    "trade_count": 1,
                    "traded_step_count": 4,
                },
                "actions": {
                    "absolute_target_delta_total": 2.0,
                    "buy_sell_sign_flip_count": 1,
                },
            },
        ]
    )

    assert result == {
        "symbol_count": 2,
        "mean_gross_return": pytest.approx(-0.005),
        "mean_net_return": pytest.approx(-0.015),
        "mean_baseline_return": pytest.approx(0.03),
        "mean_baseline_excess_return": pytest.approx(-0.045),
        "mean_turnover_multiple_per_day": pytest.approx(2.0),
        "total_execution_cost": pytest.approx(150.0),
        "total_closed_trade_count": 3,
        "total_traded_step_count": 12,
        "total_absolute_target_delta": pytest.approx(6.0),
        "total_buy_sell_sign_flip_count": 4,
    }
