from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def replace_once(path: str, old: str, new: str) -> None:
    target = ROOT / path
    text = target.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise RuntimeError(
            f"{path}: expected one anchor, found {count}: {old[:100]!r}"
        )
    updated = text.replace(old, new, 1)
    ast.parse(updated, filename=path)
    target.write_text(updated, encoding="utf-8")


def update_gated_transformer() -> None:
    path = "trade_rl/rl/gated_transformer.py"
    replace_once(
        path,
        '''    def forward(self, value: torch.Tensor, *, valid: torch.Tensor) -> torch.Tensor:
        normalized = self.attention_norm(value)
        branch, _ = self.attention(
            normalized,
            normalized,
            normalized,
            key_padding_mask=~valid,
            need_weights=False,
        )
        value = self.attention_gate(value, branch)
        value = self._zero_invalid(value, valid)
        branch = self.ffn(self.ffn_norm(value))
        value = self.ffn_gate(value, branch)
        return self._zero_invalid(value, valid)
''',
        '''    def _forward(
        self,
        value: torch.Tensor,
        *,
        valid: torch.Tensor,
        need_weights: bool,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        normalized = self.attention_norm(value)
        branch, weights = self.attention(
            normalized,
            normalized,
            normalized,
            key_padding_mask=~valid,
            need_weights=need_weights,
            average_attn_weights=False,
        )
        value = self.attention_gate(value, branch)
        value = self._zero_invalid(value, valid)
        branch = self.ffn(self.ffn_norm(value))
        value = self.ffn_gate(value, branch)
        return self._zero_invalid(value, valid), weights

    def forward(self, value: torch.Tensor, *, valid: torch.Tensor) -> torch.Tensor:
        output, _ = self._forward(value, valid=valid, need_weights=False)
        return output

    def diagnostic_forward(
        self,
        value: torch.Tensor,
        *,
        valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        output, weights = self._forward(value, valid=valid, need_weights=True)
        if weights is None:
            raise RuntimeError("diagnostic attention weights were not produced")
        return output, weights
''',
    )
    replace_once(
        path,
        '''    def forward(self, value: torch.Tensor, *, valid: torch.Tensor) -> torch.Tensor:
        if value.ndim != 3 or value.shape[-1] != self.d_model:
            raise ValueError("transformer stack expects [batch, tokens, d_model]")
        if valid.shape != value.shape[:2]:
            raise ValueError("valid mask must match batch and token dimensions")
        valid = valid.to(device=value.device, dtype=torch.bool)
        if torch.any(~valid.any(dim=1)):
            raise ValueError("every batch row requires at least one valid token")
        value = value * valid.unsqueeze(-1).to(dtype=value.dtype)
        for block in self.blocks:
            value = block(value, valid=valid)
        value = self.output_norm(value)
        return value * valid.unsqueeze(-1).to(dtype=value.dtype)
''',
        '''    def _validated_inputs(
        self,
        value: torch.Tensor,
        valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if value.ndim != 3 or value.shape[-1] != self.d_model:
            raise ValueError("transformer stack expects [batch, tokens, d_model]")
        if valid.shape != value.shape[:2]:
            raise ValueError("valid mask must match batch and token dimensions")
        valid = valid.to(device=value.device, dtype=torch.bool)
        if torch.any(~valid.any(dim=1)):
            raise ValueError("every batch row requires at least one valid token")
        return value * valid.unsqueeze(-1).to(dtype=value.dtype), valid

    def forward(self, value: torch.Tensor, *, valid: torch.Tensor) -> torch.Tensor:
        value, valid = self._validated_inputs(value, valid)
        for block in self.blocks:
            value = block(value, valid=valid)
        value = self.output_norm(value)
        return value * valid.unsqueeze(-1).to(dtype=value.dtype)

    def diagnostic_forward(
        self,
        value: torch.Tensor,
        *,
        valid: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        value, valid = self._validated_inputs(value, valid)
        weights: list[torch.Tensor] = []
        for block in self.blocks:
            value, block_weights = block.diagnostic_forward(value, valid=valid)
            weights.append(block_weights)
        value = self.output_norm(value)
        output = value * valid.unsqueeze(-1).to(dtype=value.dtype)
        return output, tuple(weights)
''',
    )


def update_timeframe_fusion() -> None:
    path = "trade_rl/rl/timeframe_fusion.py"
    replace_once(
        path,
        '''    def forward(
        self,
        *,
        latents: Mapping[str, torch.Tensor],
        available: Mapping[str, torch.Tensor],
        staleness: Mapping[str, torch.Tensor],
        context: torch.Tensor,
    ) -> torch.Tensor:
        if tuple(latents) != self.timeframes:
            raise ValueError("latents must use ordered 15m/1h/4h/1d timeframes")
        if tuple(available) != self.timeframes or tuple(staleness) != self.timeframes:
            raise ValueError("quality planes must use ordered 15m/1h/4h/1d timeframes")
        if context.ndim != 3 or context.shape[-1] != self.d_model:
            raise ValueError("context must be [batch, assets, d_model]")
        batch, assets, _ = context.shape
        tokens = [self.context_norm(context)]
        valid_tokens = [
            torch.ones(batch, assets, dtype=torch.bool, device=context.device)
        ]
        duration = self.duration_encoder(
            self.duration_features.to(device=context.device, dtype=context.dtype)
        )
        identities = self.timeframe_embedding(
            torch.arange(len(self.timeframes), device=context.device)
        ).to(dtype=context.dtype)

        for index, timeframe in enumerate(self.timeframes):
            latent = latents[timeframe]
            if latent.shape != (batch, assets, self.latent_dims[timeframe]):
                raise ValueError("timeframe latent shape does not match architecture")
            plane = available[timeframe]
            stale = staleness[timeframe]
            if plane.shape[:2] != (batch, assets):
                raise ValueError("timeframe availability batch or asset shape mismatch")
            quality, has_any = _quality_summary(
                plane,
                stale,
                window_length=self.window_lengths[timeframe],
            )
            quality_token = self.quality_encoder(
                quality.to(device=latent.device, dtype=latent.dtype)
            )
            token = self.latent_projectors[timeframe](latent)
            token = token + quality_token + duration[index] + identities[index]
            tokens.append(token)
            valid_tokens.append(has_any.to(device=context.device))

        stacked = torch.stack(tokens, dim=2)
        valid = torch.stack(valid_tokens, dim=2)
        flattened = stacked.reshape(batch * assets, len(tokens), self.d_model)
        flattened_valid = valid.reshape(batch * assets, len(tokens))
        contextual = self.transformer(flattened, valid=flattened_valid)
        return contextual[:, 0].reshape(batch, assets, self.d_model)
''',
        '''    def _prepared_tokens(
        self,
        *,
        latents: Mapping[str, torch.Tensor],
        available: Mapping[str, torch.Tensor],
        staleness: Mapping[str, torch.Tensor],
        context: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, int, int]:
        if tuple(latents) != self.timeframes:
            raise ValueError("latents must use ordered 15m/1h/4h/1d timeframes")
        if tuple(available) != self.timeframes or tuple(staleness) != self.timeframes:
            raise ValueError("quality planes must use ordered 15m/1h/4h/1d timeframes")
        if context.ndim != 3 or context.shape[-1] != self.d_model:
            raise ValueError("context must be [batch, assets, d_model]")
        batch, assets, _ = context.shape
        tokens = [self.context_norm(context)]
        valid_tokens = [
            torch.ones(batch, assets, dtype=torch.bool, device=context.device)
        ]
        duration = self.duration_encoder(
            self.duration_features.to(device=context.device, dtype=context.dtype)
        )
        identities = self.timeframe_embedding(
            torch.arange(len(self.timeframes), device=context.device)
        ).to(dtype=context.dtype)

        for index, timeframe in enumerate(self.timeframes):
            latent = latents[timeframe]
            if latent.shape != (batch, assets, self.latent_dims[timeframe]):
                raise ValueError("timeframe latent shape does not match architecture")
            plane = available[timeframe]
            stale = staleness[timeframe]
            if plane.shape[:2] != (batch, assets):
                raise ValueError("timeframe availability batch or asset shape mismatch")
            quality, has_any = _quality_summary(
                plane,
                stale,
                window_length=self.window_lengths[timeframe],
            )
            quality_token = self.quality_encoder(
                quality.to(device=latent.device, dtype=latent.dtype)
            )
            token = self.latent_projectors[timeframe](latent)
            token = token + quality_token + duration[index] + identities[index]
            tokens.append(token)
            valid_tokens.append(has_any.to(device=context.device))

        stacked = torch.stack(tokens, dim=2)
        valid = torch.stack(valid_tokens, dim=2)
        flattened = stacked.reshape(batch * assets, len(tokens), self.d_model)
        flattened_valid = valid.reshape(batch * assets, len(tokens))
        return flattened, flattened_valid, batch, assets

    def forward(
        self,
        *,
        latents: Mapping[str, torch.Tensor],
        available: Mapping[str, torch.Tensor],
        staleness: Mapping[str, torch.Tensor],
        context: torch.Tensor,
    ) -> torch.Tensor:
        flattened, valid, batch, assets = self._prepared_tokens(
            latents=latents,
            available=available,
            staleness=staleness,
            context=context,
        )
        contextual = self.transformer(flattened, valid=valid)
        return contextual[:, 0].reshape(batch, assets, self.d_model)

    def diagnostic_forward(
        self,
        *,
        latents: Mapping[str, torch.Tensor],
        available: Mapping[str, torch.Tensor],
        staleness: Mapping[str, torch.Tensor],
        context: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...], torch.Tensor]:
        flattened, valid, batch, assets = self._prepared_tokens(
            latents=latents,
            available=available,
            staleness=staleness,
            context=context,
        )
        contextual, weights = self.transformer.diagnostic_forward(
            flattened,
            valid=valid,
        )
        output = contextual[:, 0].reshape(batch, assets, self.d_model)
        return output, weights, valid
''',
    )


def write_sequence_diagnostics() -> None:
    path = ROOT / "trade_rl/rl/sequence_diagnostics.py"
    path.write_text(
        '''"""Low-frequency diagnostics for hierarchical sequence policies."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import torch
from torch import nn

_TIMEFRAMES = ("15m", "1h", "4h", "1d")


def _gradient_norm(module: nn.Module) -> tuple[float, bool]:
    total = 0.0
    available = False
    for parameter in module.parameters():
        gradient = parameter.grad
        if gradient is None:
            continue
        available = True
        total += float(gradient.detach().float().square().sum().item())
    return math.sqrt(total), available


def _gate_metrics(stack: Any) -> tuple[float, float]:
    gates: list[torch.Tensor] = []
    for block in stack.blocks:
        gates.append(torch.sigmoid(block.attention_gate.gate.detach()).float())
        gates.append(torch.sigmoid(block.ffn_gate.gate.detach()).float())
    if not gates:
        return 0.0, 0.0
    values = torch.cat(gates)
    saturation = ((values <= 0.05) | (values >= 0.95)).float().mean()
    return float(values.mean().item()), float(saturation.item())


def _attention_distribution_metrics(
    weights: torch.Tensor,
    valid: torch.Tensor,
) -> tuple[torch.Tensor, float, float]:
    masked = weights * valid[:, None, :].to(dtype=weights.dtype)
    denominator = masked.sum(dim=-1, keepdim=True)
    has_any = denominator.squeeze(-1) > 0.0
    probabilities = masked / denominator.clamp_min(1e-12)
    entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(dim=-1)
    valid_count = valid.sum(dim=-1).clamp_min(1).to(dtype=weights.dtype)
    scale = valid_count.log().clamp_min(1.0)
    normalized_entropy = entropy / scale[:, None]
    maximum = probabilities.max(dim=-1).values
    if torch.any(has_any):
        entropy_value = float(normalized_entropy[has_any].mean().item())
        maximum_value = float(maximum[has_any].mean().item())
    else:
        entropy_value = 0.0
        maximum_value = 0.0
    return probabilities, entropy_value, maximum_value


def sequence_diagnostics_payload(
    feature_extractor: Any,
    observations: Mapping[str, torch.Tensor],
) -> dict[str, float]:
    """Run an explicit deterministic probe without changing the training forward path."""

    asset_encoder = getattr(feature_extractor, "asset_encoder", None)
    if asset_encoder is None or not hasattr(asset_encoder, "timeframe_fusion"):
        raise TypeError("sequence diagnostics require SequenceAssetFeatureExtractor")
    reference = observations["current_snapshot"]
    batch, assets, _ = reference.shape
    latents: dict[str, torch.Tensor] = {}
    available: dict[str, torch.Tensor] = {}
    staleness: dict[str, torch.Tensor] = {}

    was_training = bool(feature_extractor.training)
    feature_extractor.eval()
    try:
        with torch.no_grad():
            for timeframe in _TIMEFRAMES:
                values = observations[f"sequence_{timeframe}_values"].float()
                availability = observations[f"sequence_{timeframe}_available"].float()
                stale = observations[f"sequence_{timeframe}_staleness"].float()
                sequence = torch.cat(
                    (values, availability, torch.log1p(stale.clamp_min(0.0))),
                    dim=-1,
                )
                timestep_mask = (availability > 0.5).any(dim=-1)
                flattened = sequence.reshape(
                    batch * assets,
                    sequence.shape[2],
                    sequence.shape[3],
                )
                flattened_mask = timestep_mask.reshape(batch * assets, sequence.shape[2])
                encoded = asset_encoder.timeframe_encoders[timeframe](
                    flattened,
                    flattened_mask,
                )
                latents[timeframe] = encoded.reshape(batch, assets, -1)
                available[timeframe] = availability > 0.5
                staleness[timeframe] = stale

            context = asset_encoder.context_encoder(
                torch.cat(
                    (
                        asset_encoder.snapshot_encoder(reference.float()),
                        asset_encoder.asset_state_encoder(
                            observations["asset_state"].float()
                        ),
                    ),
                    dim=-1,
                )
            )
            fused, timeframe_weights, timeframe_valid = (
                asset_encoder.timeframe_fusion.diagnostic_forward(
                    latents=latents,
                    available=available,
                    staleness=staleness,
                    context=context,
                )
            )
            identities = torch.arange(assets, device=fused.device)
            fused = fused + asset_encoder.symbol_embedding(identities).unsqueeze(0)
            active = observations["active"].to(dtype=torch.bool)
            safe_active = active.clone()
            has_active = safe_active.any(dim=1)
            if torch.any(~has_active):
                safe_active[~has_active, 0] = True
                fused = fused.clone()
                fused[~has_active, 0] = 0.0
            _, asset_weights = asset_encoder.cross_asset.diagnostic_forward(
                fused,
                valid=safe_active,
            )
    finally:
        feature_extractor.train(was_training)

    timeframe_query = timeframe_weights[-1][:, :, 0, 1:]
    timeframe_key_valid = timeframe_valid[:, 1:]
    timeframe_probabilities, timeframe_entropy, timeframe_maximum = (
        _attention_distribution_metrics(timeframe_query, timeframe_key_valid)
    )
    asset_query = asset_weights[-1]
    asset_valid = safe_active.repeat_interleave(asset_query.shape[2], dim=0)
    asset_rows = asset_query.permute(0, 2, 1, 3).reshape(
        -1,
        asset_query.shape[1],
        asset_query.shape[-1],
    )
    _, asset_entropy, asset_maximum = _attention_distribution_metrics(
        asset_rows,
        asset_valid,
    )

    payload: dict[str, float] = {
        "sequence/timeframe_attention_entropy": timeframe_entropy,
        "sequence/timeframe_attention_max_share": timeframe_maximum,
        "sequence/asset_attention_entropy": asset_entropy,
        "sequence/asset_attention_max_share": asset_maximum,
    }
    for index, timeframe in enumerate(_TIMEFRAMES):
        valid = timeframe_key_valid[:, index]
        values = timeframe_probabilities[:, :, index]
        denominator = valid.sum().item() * values.shape[1]
        share = 0.0 if denominator == 0 else float(values.sum().item() / denominator)
        payload[f"sequence/timeframe_attention/{timeframe}"] = share
        payload[f"sequence/timeframe_missing_ratio/{timeframe}"] = float(
            1.0 - valid.float().mean().item()
        )

    timeframe_gate_mean, timeframe_gate_saturation = _gate_metrics(
        asset_encoder.timeframe_fusion.transformer
    )
    asset_gate_mean, asset_gate_saturation = _gate_metrics(asset_encoder.cross_asset)
    payload.update(
        {
            "sequence/timeframe_gate_mean": timeframe_gate_mean,
            "sequence/timeframe_gate_saturation": timeframe_gate_saturation,
            "sequence/asset_gate_mean": asset_gate_mean,
            "sequence/asset_gate_saturation": asset_gate_saturation,
        }
    )
    timeframe_gradient, timeframe_gradient_available = _gradient_norm(
        asset_encoder.timeframe_encoders
    )
    fusion_gradient, fusion_gradient_available = _gradient_norm(
        asset_encoder.timeframe_fusion
    )
    asset_gradient, asset_gradient_available = _gradient_norm(asset_encoder.cross_asset)
    payload.update(
        {
            "sequence/gradient/timeframe_encoder": timeframe_gradient,
            "sequence/gradient/timeframe_fusion": fusion_gradient,
            "sequence/gradient/cross_asset": asset_gradient,
            "sequence/gradient/available": float(
                timeframe_gradient_available
                or fusion_gradient_available
                or asset_gradient_available
            ),
        }
    )
    if any(not math.isfinite(value) for value in payload.values()):
        raise ValueError("sequence diagnostic payload contains non-finite values")
    return payload


def build_sequence_diagnostics_callback() -> Any:
    """Build a rollout-boundary TensorBoard probe that no-ops for other policies."""

    from stable_baselines3.common.callbacks import BaseCallback

    class SequenceDiagnosticsCallback(BaseCallback):
        def __init__(self) -> None:
            super().__init__(verbose=0)

        def _on_step(self) -> bool:
            return True

        def _on_rollout_end(self) -> None:
            policy = getattr(self.model, "policy", None)
            feature_extractor = getattr(policy, "features_extractor", None)
            if feature_extractor is None or not hasattr(
                getattr(feature_extractor, "asset_encoder", None),
                "timeframe_fusion",
            ):
                return
            last_observation = getattr(self.model, "_last_obs", None)
            if not isinstance(last_observation, dict):
                return
            tensor_observation, _ = policy.obs_to_tensor(last_observation)
            payload = sequence_diagnostics_payload(
                feature_extractor,
                tensor_observation,
            )
            for key, value in sorted(payload.items()):
                self.logger.record(key, value)

    return SequenceDiagnosticsCallback()


__all__ = ["build_sequence_diagnostics_callback", "sequence_diagnostics_payload"]
''',
        encoding="utf-8",
    )
    ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def update_checkpoint_callback() -> None:
    path = "trade_rl/rl/checkpointing.py"
    replace_once(
        path,
        '''    telemetry_callback = build_training_telemetry_callback(
        path=checkpoint_root.parent / "telemetry" / "training-telemetry.jsonl",
        seed=seed,
    )
    if not planned:
        return telemetry_callback
''',
        '''    telemetry_callback = build_training_telemetry_callback(
        path=checkpoint_root.parent / "telemetry" / "training-telemetry.jsonl",
        seed=seed,
    )
    from trade_rl.rl.sequence_diagnostics import build_sequence_diagnostics_callback

    diagnostics_callback = build_sequence_diagnostics_callback()
    if not planned:
        return CallbackList([telemetry_callback, diagnostics_callback])
''',
    )
    replace_once(
        path,
        "    return CallbackList([AtomicCheckpointCallback(), telemetry_callback])\n",
        '''    return CallbackList(
        [AtomicCheckpointCallback(), telemetry_callback, diagnostics_callback]
    )
''',
    )


def write_tests() -> None:
    path = ROOT / "tests/rl/test_sequence_diagnostics.py"
    path.write_text(
        '''from __future__ import annotations

import math

import torch
from gymnasium import spaces

from trade_rl.rl.gated_transformer import GatedTransformerStack
from trade_rl.rl.policies import SequenceAssetFeatureExtractor
from trade_rl.rl.sequence_diagnostics import sequence_diagnostics_payload


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
''',
        encoding="utf-8",
    )
    ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def main() -> None:
    update_gated_transformer()
    update_timeframe_fusion()
    write_sequence_diagnostics()
    update_checkpoint_callback()
    write_tests()


if __name__ == "__main__":
    main()
