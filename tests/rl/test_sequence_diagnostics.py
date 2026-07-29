from __future__ import annotations

import math
from types import SimpleNamespace

import torch
from gymnasium import spaces

from trade_rl.rl import sequence_diagnostics as diagnostics_module
from trade_rl.rl.gated_transformer import GatedTransformerStack
from trade_rl.rl.policies import SequenceAssetFeatureExtractor
from trade_rl.rl.sequence_diagnostics import (
    build_sequence_diagnostics_callback,
    sequence_diagnostics_payload,
)


def test_diagnostic_transformer_matches_normal_forward_and_masks_keys() -> None:
    torch.manual_seed(31)
    stack = GatedTransformerStack(
        d_model=8,
        heads=2,
        layers=2,
        ffn_multiplier=2,
        dropout=0.0,
        gate_bias=-2.0,
    ).eval()
    value = torch.randn(3, 5, 8)
    valid = torch.tensor(
        [
            [True, True, True, False, False],
            [True, True, False, True, False],
            [True, False, False, False, False],
        ]
    )
    with torch.no_grad():
        expected = stack(value, valid=valid)
        actual, weights = stack.diagnostic_forward(value, valid=valid)
    torch.testing.assert_close(actual, expected)
    assert len(weights) == 2
    assert weights[-1].shape == (3, 2, 5, 5)
    invalid_keys = (~valid)[:, None, None, :].expand_as(weights[-1])
    assert torch.count_nonzero(weights[-1].masked_select(invalid_keys)) == 0


def _extractor() -> SequenceAssetFeatureExtractor:
    timeframes = ("15m", "1h", "4h", "1d")
    feature_counts = {timeframe: 2 for timeframe in timeframes}
    window_lengths = {timeframe: 4 for timeframe in timeframes}
    observation_spaces: dict[str, spaces.Space] = {
        "current_snapshot": spaces.Box(-10.0, 10.0, shape=(3, 6)),
        "asset_state": spaces.Box(-10.0, 10.0, shape=(3, 4)),
        "global_state": spaces.Box(-10.0, 10.0, shape=(5,)),
        "active": spaces.Box(0.0, 1.0, shape=(3,)),
        "current_weights": spaces.Box(-1.0, 1.0, shape=(3,)),
    }
    for timeframe in timeframes:
        shape = (3, 4, 2)
        observation_spaces[f"sequence_{timeframe}_values"] = spaces.Box(
            -10.0, 10.0, shape=shape
        )
        observation_spaces[f"sequence_{timeframe}_available"] = spaces.Box(
            0.0, 1.0, shape=shape
        )
        observation_spaces[f"sequence_{timeframe}_staleness"] = spaces.Box(
            0.0, 100.0, shape=shape
        )
    return SequenceAssetFeatureExtractor(
        spaces.Dict(observation_spaces),
        feature_counts=feature_counts,
        window_lengths=window_lengths,
        snapshot_width=6,
        asset_state_width=4,
        global_width=5,
        n_symbols=3,
        sequence_tcn_capacity="compact",
        d_model=16,
        timeframe_attention_heads=4,
        timeframe_attention_layers=1,
        timeframe_ffn_multiplier=2,
        asset_attention_heads=4,
        asset_attention_layers=1,
        asset_ffn_multiplier=2,
        dropout=0.0,
    ).eval()


def _observations() -> dict[str, torch.Tensor]:
    torch.manual_seed(37)
    observations = {
        "current_snapshot": torch.randn(2, 3, 6),
        "asset_state": torch.randn(2, 3, 4),
        "global_state": torch.randn(2, 5),
        "active": torch.tensor(
            [[1.0, 1.0, 0.0], [1.0, 0.0, 1.0]],
            dtype=torch.float32,
        ),
        "current_weights": torch.zeros(2, 3, dtype=torch.float32),
    }
    for timeframe in ("15m", "1h", "4h", "1d"):
        observations[f"sequence_{timeframe}_values"] = torch.randn(2, 3, 4, 2)
        observations[f"sequence_{timeframe}_available"] = torch.ones(2, 3, 4, 2)
        observations[f"sequence_{timeframe}_staleness"] = torch.zeros(2, 3, 4, 2)
    observations["sequence_1d_available"][:, 2].zero_()
    observations["sequence_1d_staleness"][:, 2].fill_(100.0)
    return observations


def test_sequence_diagnostic_payload_is_finite_and_quality_aware() -> None:
    payload = sequence_diagnostics_payload(_extractor(), _observations())
    required = {
        "sequence/timeframe_attention/15m",
        "sequence/timeframe_attention/1h",
        "sequence/timeframe_attention/4h",
        "sequence/timeframe_attention/1d",
        "sequence/timeframe_attention_entropy",
        "sequence/timeframe_attention_max_share",
        "sequence/asset_attention_entropy",
        "sequence/timeframe_gate_mean",
        "sequence/asset_gate_mean",
        "sequence/gradient/available",
    }
    assert required <= payload.keys()
    assert all(math.isfinite(value) for value in payload.values())
    assert payload["sequence/timeframe_missing_ratio/1d"] > 0.0
    total_share = sum(
        payload[f"sequence/timeframe_attention/{timeframe}"]
        for timeframe in ("15m", "1h", "4h", "1d")
    )
    assert 0.0 < total_share <= 1.25


def test_sequence_diagnostics_callback_is_absent_when_disabled() -> None:
    assert (
        build_sequence_diagnostics_callback(enabled=False, rollout_interval=1) is None
    )


def test_sequence_diagnostics_callback_honors_rollout_interval(
    monkeypatch,
) -> None:
    calls: list[object] = []
    records: list[tuple[str, float]] = []

    monkeypatch.setattr(
        diagnostics_module,
        "sequence_diagnostics_payload",
        lambda extractor, observations: (
            calls.append((extractor, observations)) or {"sequence/probe": 1.0}
        ),
    )

    extractor = SimpleNamespace(
        asset_encoder=SimpleNamespace(timeframe_fusion=object())
    )

    class Policy:
        features_extractor = extractor

        @staticmethod
        def obs_to_tensor(observation):
            return observation, None

    callback = build_sequence_diagnostics_callback(enabled=True, rollout_interval=3)
    assert callback is not None
    callback.model = SimpleNamespace(
        policy=Policy(),
        _last_obs={"current_snapshot": torch.zeros(1)},
        logger=SimpleNamespace(record=lambda key, value: records.append((key, value))),
    )

    callback._on_rollout_end()
    callback._on_rollout_end()
    assert calls == []
    assert records == []

    callback._on_rollout_end()
    assert len(calls) == 1
    assert records == [("sequence/probe", 1.0)]
