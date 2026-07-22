from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np

from trade_rl.integrations.training_telemetry import TrainingTelemetrySampler
from trade_rl.telemetry import read_training_telemetry


def _info(
    step: int,
    *,
    price: float = 100.0,
    before: tuple[float, ...] | None = (0.2,),
    after: tuple[float, ...] = (0.2,),
    exact_market: bool = True,
    terminated: bool = False,
    truncated: bool = False,
) -> dict[str, object]:
    book = SimpleNamespace(
        weights=np.asarray(after, dtype=np.float64),
        portfolio_value=1_000.0 + step,
        mark_prices=np.asarray([price], dtype=np.float64),
    )
    result: dict[str, object] = {
        "decision_step_index": step,
        "telemetry_symbol": "BTCUSDT",
        "telemetry_weights_after": np.asarray(after, dtype=np.float64),
        "hybrid_execution": SimpleNamespace(book=book),
        "portfolio_value_after": 1_000.0 + step,
        "baseline_portfolio_value_after": 1_000.0,
        "reward_total_scaled": 0.0,
        "drawdown_after": 0.0,
        "interval_cost": 0.0,
        "interval_net_return": 0.0,
        "telemetry_risk_reasons": (),
        "emergency_deleverage": False,
        "hybrid_terminated": terminated,
        "TimeLimit.truncated": truncated,
    }
    if before is not None:
        result["telemetry_weights_before"] = np.asarray(before, dtype=np.float64)
    if exact_market:
        result.update(
            {
                "telemetry_market_index": 100 + step,
                "telemetry_market_time": "2026-07-22T12:00:00.000000000",
                "telemetry_ohlc": (price, price, price, price),
            }
        )
    return result


def test_vector_environments_receive_distinct_episode_ids_and_rotate_after_done(
    tmp_path: Path,
) -> None:
    path = tmp_path / "training-telemetry.jsonl"
    sampler = TrainingTelemetrySampler(path, seed=3, sample_every=1)

    assert (
        sampler.consume(
            global_step=2,
            actions=np.asarray([[0.1], [0.2]], dtype=np.float32),
            rewards=np.asarray([0.0, 0.0], dtype=np.float32),
            dones=np.asarray([False, False]),
            infos=(_info(1), _info(1, price=600.0)),
        )
        == 2
    )
    assert (
        sampler.consume(
            global_step=4,
            actions=np.asarray([[0.0], [0.2]], dtype=np.float32),
            rewards=np.asarray([0.0, 0.0], dtype=np.float32),
            dones=np.asarray([True, False]),
            infos=(
                _info(2, terminated=True),
                _info(2, price=601.0),
            ),
        )
        == 2
    )
    assert (
        sampler.consume(
            global_step=6,
            actions=np.asarray([[0.3], [0.2]], dtype=np.float32),
            rewards=np.asarray([0.0, 0.0], dtype=np.float32),
            dones=np.asarray([False, False]),
            infos=(
                _info(0, price=200.0),
                _info(3, price=602.0),
            ),
        )
        == 2
    )
    sampler.close()

    items = read_training_telemetry(path, limit=20).items
    environment_zero = [item for item in items if item.environment_id == 0]
    environment_one = [item for item in items if item.environment_id == 1]

    assert environment_zero[0].episode_id == environment_zero[1].episode_id
    assert environment_zero[2].episode_id not in (
        None,
        environment_zero[1].episode_id,
    )
    assert len({item.episode_id for item in environment_one}) == 1
    assert environment_zero[0].episode_id != environment_one[0].episode_id


def test_terminal_transition_clears_visual_fallback_state_for_next_episode(
    tmp_path: Path,
) -> None:
    path = tmp_path / "training-telemetry.jsonl"
    sampler = TrainingTelemetrySampler(path, seed=5, sample_every=1)

    assert (
        sampler.consume(
            global_step=1,
            actions=np.asarray([[0.8]], dtype=np.float32),
            rewards=np.asarray([0.0], dtype=np.float32),
            dones=np.asarray([True]),
            infos=(
                _info(
                    8,
                    price=100.0,
                    before=(0.2,),
                    after=(0.8,),
                    terminated=True,
                ),
            ),
        )
        == 1
    )
    assert (
        sampler.consume(
            global_step=2,
            actions=np.asarray([[0.1]], dtype=np.float32),
            rewards=np.asarray([0.0], dtype=np.float32),
            dones=np.asarray([False]),
            infos=(
                _info(
                    0,
                    price=600.0,
                    before=None,
                    after=(0.1,),
                    exact_market=False,
                ),
            ),
        )
        == 1
    )
    sampler.close()

    first, second = read_training_telemetry(path, limit=10).items

    assert second.episode_id not in (None, first.episode_id)
    assert second.open == 600.0
    assert second.close == 600.0
    assert second.weights_before == (0.0,)
