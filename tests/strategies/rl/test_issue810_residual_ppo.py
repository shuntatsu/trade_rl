"""Real SB3/Torch checks of the synthetic residual-PPO study (PPO Runtime job).

These need the ``train-sb3`` extra and live under ``tests/strategies/rl/`` because
the PPO Runtime CI job selects that directory; the core job skips them. The budgets
are far below the frozen study budget and carry no research conclusion.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("stable_baselines3")

from trade_rl.evaluation.residual_rl.evaluate import (  # noqa: E402
    aggregate,
    constant_chooser,
    env_config,
    evaluate_chooser,
)
from trade_rl.evaluation.residual_rl.market import MarketSpec, Regime  # noqa: E402
from trade_rl.evaluation.residual_rl.protocol import (  # noqa: E402
    PROTOCOL_V1,
    ResidualProtocol,
)
from trade_rl.evaluation.residual_rl.study import run_smoke  # noqa: E402
from trade_rl.evaluation.residual_rl.train import (  # noqa: E402
    model_chooser,
    runtime_versions,
    train_residual_ppo,
)

ALLOCATOR_ALONE = PROTOCOL_V1.multipliers.index(1.0)


def _tiny_protocol() -> ResidualProtocol:
    return replace(
        PROTOCOL_V1,
        market=MarketSpec(horizon_days=40),
        ppo=replace(PROTOCOL_V1.ppo, n_steps=120, batch_size=40, total_timesteps=240),
    )


def test_runtime_versions_match_the_pinned_extra() -> None:
    versions = runtime_versions()
    assert versions["stable_baselines3"] == "2.3.2"
    assert versions["torch"].startswith("2.4.1")


def test_short_fit_is_deterministic_under_a_fixed_seed() -> None:
    protocol = _tiny_protocol()
    config = env_config(protocol, Regime.PLANTED_ALPHA)
    runs = []
    for _ in range(2):
        model = train_residual_ppo(protocol, Regime.PLANTED_ALPHA, 0)
        metrics, histogram = evaluate_chooser(
            config, (9000, 9001), model_chooser(model, deterministic=True)
        )
        runs.append(([m.net_log_return for m in metrics], histogram.tolist()))
    assert runs[0] == runs[1]
    assert sum(runs[0][1]) == 2 * 40
    other = train_residual_ppo(protocol, Regime.PLANTED_ALPHA, 1)
    assert other is not None


def test_fee_only_control_trains_toward_low_turnover() -> None:
    # Zero-alpha market: trading is a pure loss, so the learned throttle must cut
    # turnover well below the allocator's. 10,080 steps is a fifth of the frozen
    # budget and is only a learnability check, not the study.
    config = env_config(PROTOCOL_V1, Regime.ZERO_ALPHA)
    seeds = PROTOCOL_V1.seeds.calibration[:3]
    model = train_residual_ppo(
        PROTOCOL_V1, Regime.ZERO_ALPHA, 0, total_timesteps=10_080
    )
    trained, histogram = evaluate_chooser(
        config, seeds, model_chooser(model, deterministic=True)
    )
    allocator, _ = evaluate_chooser(config, seeds, constant_chooser(ALLOCATOR_ALONE))
    trained_turnover = aggregate(trained)["median_daily_turnover"]
    allocator_turnover = aggregate(allocator)["median_daily_turnover"]
    assert allocator_turnover > 0.2
    assert trained_turnover < 0.5 * allocator_turnover
    # It learned to prefer throttling: most decisions are below the allocator's m=1.
    assert histogram[:2].sum() > histogram[2:].sum()
    assert (
        aggregate(trained)["median_net_log_return"]
        > aggregate(allocator)["median_net_log_return"]
    )


def test_smoke_pipeline_records_both_regimes_without_claiming_evidence() -> None:
    record = run_smoke(PROTOCOL_V1, total_timesteps=1_008, n_seeds=1)
    assert set(record["regimes"]) == {"planted_alpha", "zero_alpha"}
    for by_seed in record["regimes"].values():
        assert set(by_seed) == {"seed0"}
        assert by_seed["seed0"]["n_episodes"] == 1.0
        assert np.isfinite(by_seed["seed0"]["median_net_log_return"])
    assert record["seeds"] == list(PROTOCOL_V1.seeds.calibration[:1])
