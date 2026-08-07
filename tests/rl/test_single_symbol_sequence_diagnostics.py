from __future__ import annotations

import math

import torch
from gymnasium import spaces

from trade_rl.rl.policies import SequenceAssetFeatureExtractor
from trade_rl.rl.sequence_diagnostics import sequence_diagnostics_payload

_TIMEFRAMES = ("15m", "1h", "4h", "1d")


def _extractor() -> SequenceAssetFeatureExtractor:
    observation_spaces: dict[str, spaces.Space] = {
        "current_snapshot": spaces.Box(-10.0, 10.0, shape=(1, 6)),
        "asset_state": spaces.Box(-10.0, 10.0, shape=(1, 4)),
        "global_state": spaces.Box(-10.0, 10.0, shape=(5,)),
        "active": spaces.Box(0.0, 1.0, shape=(1,)),
        "current_weights": spaces.Box(-1.0, 1.0, shape=(1,)),
    }
    for timeframe in _TIMEFRAMES:
        shape = (1, 4, 2)
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
        feature_counts={timeframe: 2 for timeframe in _TIMEFRAMES},
        window_lengths={timeframe: 4 for timeframe in _TIMEFRAMES},
        snapshot_width=6,
        asset_state_width=4,
        global_width=5,
        n_symbols=1,
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
    torch.manual_seed(41)
    observations = {
        "current_snapshot": torch.randn(2, 1, 6),
        "asset_state": torch.randn(2, 1, 4),
        "global_state": torch.randn(2, 5),
        "active": torch.tensor([[1.0], [0.0]], dtype=torch.float32),
        "current_weights": torch.zeros(2, 1, dtype=torch.float32),
    }
    for timeframe in _TIMEFRAMES:
        observations[f"sequence_{timeframe}_values"] = torch.randn(2, 1, 4, 2)
        observations[f"sequence_{timeframe}_available"] = torch.ones(2, 1, 4, 2)
        observations[f"sequence_{timeframe}_staleness"] = torch.zeros(2, 1, 4, 2)
    return observations


def test_one_symbol_diagnostics_marks_cross_asset_metrics_not_applicable() -> None:
    extractor = _extractor()
    assert extractor.asset_encoder.cross_asset is None

    payload = sequence_diagnostics_payload(extractor, _observations())

    assert payload["sequence/asset_attention_entropy"] == 0.0
    assert payload["sequence/asset_attention_max_share"] == 0.0
    assert payload["sequence/asset_gate_mean"] == 0.0
    assert payload["sequence/asset_gate_saturation"] == 0.0
    assert payload["sequence/gradient/cross_asset"] == 0.0
    assert all(math.isfinite(value) for value in payload.values())
    assert payload["sequence/timeframe_attention_max_share"] > 0.0
