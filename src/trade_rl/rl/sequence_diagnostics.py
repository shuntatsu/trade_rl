"""Low-frequency diagnostics for hierarchical sequence policies."""

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
    asset_weights: tuple[torch.Tensor, ...] = ()

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
                flattened_mask = timestep_mask.reshape(
                    batch * assets, sequence.shape[2]
                )
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
            active = observations["active"].to(dtype=torch.bool)
            asset_positions = torch.arange(assets, device=fused.device)
            has_active = active.any(dim=1)
            fallback = (~has_active).unsqueeze(1) & asset_positions.unsqueeze(0).eq(0)
            safe_active = active | fallback
            fused = torch.where(fallback.unsqueeze(-1), torch.zeros_like(fused), fused)
            cross_asset = asset_encoder.cross_asset
            if cross_asset is not None:
                _, raw_asset_weights = cross_asset.diagnostic_forward(
                    fused,
                    valid=safe_active,
                )
                asset_weights = tuple(raw_asset_weights)
    finally:
        feature_extractor.train(was_training)

    timeframe_query = timeframe_weights[-1][:, :, 0, 1:]
    timeframe_key_valid = timeframe_valid[:, 1:]
    timeframe_probabilities, timeframe_entropy, timeframe_maximum = (
        _attention_distribution_metrics(timeframe_query, timeframe_key_valid)
    )
    if asset_weights:
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
    else:
        asset_entropy = 0.0
        asset_maximum = 0.0

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
    cross_asset = asset_encoder.cross_asset
    if cross_asset is None:
        asset_gate_mean = 0.0
        asset_gate_saturation = 0.0
    else:
        asset_gate_mean, asset_gate_saturation = _gate_metrics(cross_asset)
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
    if cross_asset is None:
        asset_gradient = 0.0
        asset_gradient_available = False
    else:
        asset_gradient, asset_gradient_available = _gradient_norm(cross_asset)
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


def build_sequence_diagnostics_callback(
    *,
    enabled: bool,
    rollout_interval: int,
) -> Any | None:
    """Build a TensorBoard probe only when explicitly enabled."""

    if not isinstance(enabled, bool):
        raise ValueError("sequence diagnostics enabled must be a boolean")
    if (
        isinstance(rollout_interval, bool)
        or not isinstance(rollout_interval, int)
        or rollout_interval <= 0
    ):
        raise ValueError("sequence diagnostics rollout_interval must be positive")
    if not enabled:
        return None

    from stable_baselines3.common.callbacks import BaseCallback

    class SequenceDiagnosticsCallback(BaseCallback):
        def __init__(self) -> None:
            super().__init__(verbose=0)
            self.completed_rollouts = 0

        def _on_step(self) -> bool:
            return True

        def _on_rollout_end(self) -> None:
            self.completed_rollouts += 1
            if self.completed_rollouts % rollout_interval != 0:
                return
            policy = getattr(self.model, "policy", None)
            obs_to_tensor = getattr(policy, "obs_to_tensor", None)
            if not callable(obs_to_tensor):
                return
            feature_extractor = getattr(policy, "features_extractor", None)
            if feature_extractor is None or not hasattr(
                getattr(feature_extractor, "asset_encoder", None),
                "timeframe_fusion",
            ):
                return
            last_observation = getattr(self.model, "_last_obs", None)
            if not isinstance(last_observation, dict):
                return
            tensor_observation, _ = obs_to_tensor(last_observation)
            payload = sequence_diagnostics_payload(
                feature_extractor,
                tensor_observation,
            )
            for key, value in sorted(payload.items()):
                self.logger.record(key, value)

    return SequenceDiagnosticsCallback()


__all__ = ["build_sequence_diagnostics_callback", "sequence_diagnostics_payload"]
