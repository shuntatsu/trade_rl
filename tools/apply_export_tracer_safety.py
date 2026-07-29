from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def replace_once(relative: str, old: str, new: str) -> None:
    path = ROOT / relative
    source = path.read_text(encoding="utf-8")
    if new in source:
        return
    count = source.count(old)
    if count != 1:
        raise RuntimeError(f"{relative}: expected one anchor, observed {count}")
    path.write_text(source.replace(old, new, 1), encoding="utf-8")


def write_tests() -> None:
    test_path = ROOT / "tests/rl/test_export_tracer_safety.py"
    test_path.write_text(
        """from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from trade_rl.rl.sequence_policy import (
    MultiTimeframeAssetEncoder,
    SequencePolicyArchitecture,
)


class _TraceableAssetEncoder(nn.Module):
    def __init__(self, encoder: MultiTimeframeAssetEncoder) -> None:
        super().__init__()
        self.encoder = encoder

    def forward(
        self,
        sequence_15m: torch.Tensor,
        available_15m: torch.Tensor,
        staleness_15m: torch.Tensor,
        sequence_1h: torch.Tensor,
        available_1h: torch.Tensor,
        staleness_1h: torch.Tensor,
        sequence_4h: torch.Tensor,
        available_4h: torch.Tensor,
        staleness_4h: torch.Tensor,
        sequence_1d: torch.Tensor,
        available_1d: torch.Tensor,
        staleness_1d: torch.Tensor,
        snapshot: torch.Tensor,
        asset_state: torch.Tensor,
        active: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.encoder(
            sequences={
                "15m": sequence_15m,
                "1h": sequence_1h,
                "4h": sequence_4h,
                "1d": sequence_1d,
            },
            available={
                "15m": available_15m,
                "1h": available_1h,
                "4h": available_4h,
                "1d": available_1d,
            },
            staleness={
                "15m": staleness_15m,
                "1h": staleness_1h,
                "4h": staleness_4h,
                "1d": staleness_1d,
            },
            snapshot=snapshot,
            asset_state=asset_state,
            active=active,
        )


def _wrapper() -> _TraceableAssetEncoder:
    widths = {timeframe: (8, 8) for timeframe in ("15m", "1h", "4h", "1d")}
    architecture = SequencePolicyArchitecture(
        input_channels={timeframe: 1 for timeframe in widths},
        window_lengths={"15m": 4, "1h": 3, "4h": 2, "1d": 2},
        latent_dims={timeframe: 4 for timeframe in widths},
        asset_state_width=2,
        snapshot_width=2,
        n_symbols=2,
        d_model=8,
        timeframe_attention_heads=2,
        timeframe_attention_layers=1,
        timeframe_ffn_multiplier=2,
        asset_attention_heads=2,
        asset_attention_layers=1,
        asset_ffn_multiplier=2,
        dropout=0.0,
        encoder_widths=widths,
    )
    return _TraceableAssetEncoder(MultiTimeframeAssetEncoder(architecture)).eval()


def _inputs() -> tuple[torch.Tensor, ...]:
    torch.manual_seed(7)
    values: list[torch.Tensor] = []
    for window in (4, 3, 2, 2):
        sequence = torch.randn(2, 2, window, 1)
        available = torch.ones(2, 2, window, 1, dtype=torch.bool)
        staleness = torch.zeros(2, 2, window, 1)
        values.extend((sequence, available, staleness))
    values.extend(
        (
            torch.randn(2, 2, 2),
            torch.randn(2, 2, 2),
            torch.ones(2, 2, dtype=torch.bool),
        )
    )
    return tuple(values)


def _masked_cases(example: tuple[torch.Tensor, ...]) -> tuple[tuple[torch.Tensor, ...], ...]:
    partial = [value.clone() for value in example]
    partial[4][0, 1].zero_()
    partial[7][1, 0].zero_()
    partial[-1][0, 1] = False

    inactive = [value.clone() for value in example]
    inactive[-1].zero_()
    for position in (1, 4, 7, 10):
        inactive[position][1].zero_()

    return example, tuple(partial), tuple(inactive)


@pytest.mark.filterwarnings("error::torch.jit.TracerWarning")
def test_traced_asset_encoder_generalizes_across_availability_and_active_masks() -> None:
    wrapper = _wrapper()
    example = _inputs()
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, example, strict=False, check_trace=False)
        for inputs in _masked_cases(example):
            expected = wrapper(*inputs)
            actual = traced(*inputs)
            for expected_tensor, actual_tensor in zip(expected, actual, strict=True):
                np.testing.assert_allclose(
                    actual_tensor.detach().numpy(),
                    expected_tensor.detach().numpy(),
                    atol=1e-5,
                    rtol=0.0,
                )
""",
        encoding="utf-8",
    )
    replace_once(
        "tests/rl/test_policy_export.py",
        "def test_torchscript_actor_export_matches_sb3_prediction(tmp_path: Path) -> None:\n",
        '@pytest.mark.filterwarnings("error::torch.jit.TracerWarning")\n'
        "def test_torchscript_actor_export_matches_sb3_prediction(tmp_path: Path) -> None:\n",
    )
    replace_once(
        "tests/rl/test_policy_export.py",
        "def test_onnx_actor_export_matches_sb3_when_dependencies_exist(tmp_path: Path) -> None:\n",
        '@pytest.mark.filterwarnings("error::torch.jit.TracerWarning")\n'
        "def test_onnx_actor_export_matches_sb3_when_dependencies_exist(tmp_path: Path) -> None:\n",
    )
    replace_once(
        "tests/workflows/test_market_walk_forward.py",
        "def test_structured_walk_forward_trains_three_seed_ensemble_end_to_end(\n",
        '@pytest.mark.filterwarnings("error::torch.jit.TracerWarning")\n'
        "def test_structured_walk_forward_trains_three_seed_ensemble_end_to_end(\n",
    )
    replace_once(
        "tests/workflows/test_market_walk_forward.py",
        "import numpy as np\n",
        "import numpy as np\nimport pytest\n",
    )


def apply_implementation() -> None:
    export_context = ROOT / "trade_rl/rl/export_context.py"
    export_context.write_text(
        '''"""Keep eager validation outside traced and ONNX policy graphs."""

from __future__ import annotations

import torch


def graph_export_active() -> bool:
    """Return whether PyTorch is currently capturing an inference graph."""

    return bool(
        torch.jit.is_tracing()
        or torch.jit.is_scripting()
        or torch.onnx.is_in_onnx_export()
    )


__all__ = ["graph_export_active"]
''',
        encoding="utf-8",
    )

    replace_once(
        "trade_rl/rl/sequence_policy.py",
        "from trade_rl.rl.gated_transformer import GatedTransformerStack\n",
        "from trade_rl.rl.export_context import graph_export_active\n"
        "from trade_rl.rl.gated_transformer import GatedTransformerStack\n",
    )
    replace_once(
        "trade_rl/rl/sequence_policy.py",
        """    def forward_sequence(self, value: torch.Tensor) -> torch.Tensor:
        if value.ndim != 3:
            raise ValueError("timeframe input must be [batch, time, channels]")
        if value.shape[1] != self.window_length:
            raise ValueError("timeframe input length does not match architecture")
        return self.blocks(value.transpose(1, 2)).transpose(1, 2)

    def forward(
        self, value: torch.Tensor, available: torch.Tensor | None = None
    ) -> torch.Tensor:
        if available is None:
            encoded = self.forward_sequence(value)
            return self.projection(encoded[:, -1])
        if available.shape != value.shape[:2]:
            raise ValueError("availability mask must match batch and time dimensions")
        mask = available.to(dtype=torch.bool)
        positions = torch.arange(value.shape[1], device=value.device).expand_as(mask)
        indices = positions.masked_fill(~mask, -1).max(dim=1).values
        valid = indices >= 0
        if not torch.any(valid):
            return (value.sum(dim=(1, 2)).unsqueeze(1) * 0.0).expand(
                -1, self.latent_dim
            )
        valid_values = value[valid]
        encoded = self.forward_sequence(valid_values)
        valid_indices = indices[valid]
        selected = encoded[
            torch.arange(encoded.shape[0], device=value.device), valid_indices
        ]
        projected = self.projection(selected)
        output = projected.new_zeros((value.shape[0], self.latent_dim))
        batch_indices = torch.arange(value.shape[0], device=value.device)[valid]
        return output.index_copy(0, batch_indices, projected)
""",
        """    def forward_sequence(self, value: torch.Tensor) -> torch.Tensor:
        if not graph_export_active():
            if value.ndim != 3:
                raise ValueError("timeframe input must be [batch, time, channels]")
            if value.shape[1] != self.window_length:
                raise ValueError("timeframe input length does not match architecture")
        return self.blocks(value.transpose(1, 2)).transpose(1, 2)

    def forward(
        self, value: torch.Tensor, available: torch.Tensor | None = None
    ) -> torch.Tensor:
        if available is None:
            encoded = self.forward_sequence(value)
            return self.projection(encoded[:, -1])
        if not graph_export_active() and available.shape != value.shape[:2]:
            raise ValueError("availability mask must match batch and time dimensions")
        mask = available.to(dtype=torch.bool)
        positions = torch.arange(value.shape[1], device=value.device).expand_as(mask)
        indices = positions.masked_fill(~mask, -1).max(dim=1).values
        valid = indices >= 0
        safe_value = torch.where(mask.unsqueeze(-1), value, torch.zeros_like(value))
        encoded = self.forward_sequence(safe_value)
        safe_indices = indices.clamp_min(0)
        selected = encoded[
            torch.arange(encoded.shape[0], device=value.device), safe_indices
        ]
        projected = self.projection(selected)
        return projected * valid.unsqueeze(-1).to(dtype=projected.dtype)
""",
    )
    replace_once(
        "trade_rl/rl/sequence_policy.py",
        """        if snapshot.ndim != 3 or asset_state.ndim != 3 or active.ndim != 2:
            raise ValueError("asset encoder expects batched asset tensors")
        batch, assets, _ = snapshot.shape
        if assets != self.architecture.n_symbols:
            raise ValueError("asset count does not match architecture")
        if asset_state.shape[:2] != (batch, assets) or active.shape != (batch, assets):
            raise ValueError("asset tensors disagree on batch or asset dimensions")
""",
        """        if not graph_export_active():
            if snapshot.ndim != 3 or asset_state.ndim != 3 or active.ndim != 2:
                raise ValueError("asset encoder expects batched asset tensors")
        batch, assets, _ = snapshot.shape
        if not graph_export_active():
            if assets != self.architecture.n_symbols:
                raise ValueError("asset count does not match architecture")
            if asset_state.shape[:2] != (batch, assets) or active.shape != (
                batch,
                assets,
            ):
                raise ValueError("asset tensors disagree on batch or asset dimensions")
""",
    )
    replace_once(
        "trade_rl/rl/sequence_policy.py",
        """            if sequence.ndim != 4 or sequence.shape[:2] != (batch, assets):
                raise ValueError(
                    "sequence tensor has invalid batch or asset dimensions"
                )
            if sequence.shape[2] != self.architecture.window_lengths[timeframe]:
                raise ValueError("sequence length does not match architecture")
            if sequence.shape[-1] != self.architecture.input_channels[timeframe]:
                raise ValueError("sequence channel count does not match architecture")
            if (
                availability.ndim not in {3, 4}
                or availability.shape[:3] != sequence.shape[:3]
            ):
                raise ValueError("sequence availability shape is invalid")
            raw_staleness = staleness[timeframe]
            if raw_staleness.shape != availability.shape:
                raise ValueError("sequence staleness must match sequence availability")
""",
        """            if not graph_export_active():
                if sequence.ndim != 4 or sequence.shape[:2] != (batch, assets):
                    raise ValueError(
                        "sequence tensor has invalid batch or asset dimensions"
                    )
                if sequence.shape[2] != self.architecture.window_lengths[timeframe]:
                    raise ValueError("sequence length does not match architecture")
                if sequence.shape[-1] != self.architecture.input_channels[timeframe]:
                    raise ValueError("sequence channel count does not match architecture")
                if (
                    availability.ndim not in {3, 4}
                    or availability.shape[:3] != sequence.shape[:3]
                ):
                    raise ValueError("sequence availability shape is invalid")
            raw_staleness = staleness[timeframe]
            if not graph_export_active() and raw_staleness.shape != availability.shape:
                raise ValueError("sequence staleness must match sequence availability")
""",
    )
    replace_once(
        "trade_rl/rl/sequence_policy.py",
        """        active_mask = active.to(dtype=torch.bool)
        has_active = active_mask.any(dim=1)
        safe_mask = active_mask.clone()
        if torch.any(~has_active):
            safe_mask[~has_active, 0] = True
            fused = fused.clone()
            fused[~has_active, 0] = 0.0
        contextual = self.cross_asset(fused, valid=safe_mask)
""",
        """        active_mask = active.to(dtype=torch.bool)
        has_active = active_mask.any(dim=1)
        fallback = (~has_active).unsqueeze(1) & identities.unsqueeze(0).eq(0)
        safe_mask = active_mask | fallback
        fused = torch.where(fallback.unsqueeze(-1), torch.zeros_like(fused), fused)
        contextual = self.cross_asset(fused, valid=safe_mask)
""",
    )

    replace_once(
        "trade_rl/rl/timeframe_fusion.py",
        "from trade_rl.rl.gated_transformer import GatedTransformerStack\n",
        "from trade_rl.rl.export_context import graph_export_active\n"
        "from trade_rl.rl.gated_transformer import GatedTransformerStack\n",
    )
    replace_once(
        "trade_rl/rl/timeframe_fusion.py",
        """    if available.ndim not in {3, 4}:
        raise ValueError(
            "availability must be [batch, assets, time] or include channels"
        )
    if staleness.shape != available.shape:
        raise ValueError("staleness must match availability shape")
    if available.shape[2] != window_length:
        raise ValueError("quality plane window does not match architecture")
""",
        """    if not graph_export_active():
        if available.ndim not in {3, 4}:
            raise ValueError(
                "availability must be [batch, assets, time] or include channels"
            )
        if staleness.shape != available.shape:
            raise ValueError("staleness must match availability shape")
        if available.shape[2] != window_length:
            raise ValueError("quality plane window does not match architecture")
""",
    )
    replace_once(
        "trade_rl/rl/timeframe_fusion.py",
        """        if tuple(latents) != self.timeframes:
            raise ValueError("latents must use ordered 15m/1h/4h/1d timeframes")
        if tuple(available) != self.timeframes or tuple(staleness) != self.timeframes:
            raise ValueError("quality planes must use ordered 15m/1h/4h/1d timeframes")
        if context.ndim != 3 or context.shape[-1] != self.d_model:
            raise ValueError("context must be [batch, assets, d_model]")
""",
        """        if not graph_export_active():
            if tuple(latents) != self.timeframes:
                raise ValueError("latents must use ordered 15m/1h/4h/1d timeframes")
            if tuple(available) != self.timeframes or tuple(staleness) != self.timeframes:
                raise ValueError(
                    "quality planes must use ordered 15m/1h/4h/1d timeframes"
                )
            if context.ndim != 3 or context.shape[-1] != self.d_model:
                raise ValueError("context must be [batch, assets, d_model]")
""",
    )
    replace_once(
        "trade_rl/rl/timeframe_fusion.py",
        """            if latent.shape != (batch, assets, self.latent_dims[timeframe]):
                raise ValueError("timeframe latent shape does not match architecture")
            plane = available[timeframe]
            stale = staleness[timeframe]
            if plane.shape[:2] != (batch, assets):
                raise ValueError("timeframe availability batch or asset shape mismatch")
""",
        """            if not graph_export_active() and latent.shape != (
                batch,
                assets,
                self.latent_dims[timeframe],
            ):
                raise ValueError("timeframe latent shape does not match architecture")
            plane = available[timeframe]
            stale = staleness[timeframe]
            if not graph_export_active() and plane.shape[:2] != (batch, assets):
                raise ValueError("timeframe availability batch or asset shape mismatch")
""",
    )

    replace_once(
        "trade_rl/rl/gated_transformer.py",
        "from torch import nn\n",
        "from torch import nn\n\nfrom trade_rl.rl.export_context import graph_export_active\n",
    )
    replace_once(
        "trade_rl/rl/gated_transformer.py",
        """        if residual.shape != branch.shape:
            raise ValueError("residual and branch tensors must have identical shapes")
        if residual.ndim != 3 or residual.shape[-1] != self.gate.numel():
            raise ValueError("gated residual expects [batch, tokens, d_model]")
""",
        """        if not graph_export_active():
            if residual.shape != branch.shape:
                raise ValueError("residual and branch tensors must have identical shapes")
            if residual.ndim != 3 or residual.shape[-1] != self.gate.numel():
                raise ValueError("gated residual expects [batch, tokens, d_model]")
""",
    )
    replace_once(
        "trade_rl/rl/gated_transformer.py",
        """        if value.ndim != 3 or value.shape[-1] != self.d_model:
            raise ValueError("transformer stack expects [batch, tokens, d_model]")
        if valid.shape != value.shape[:2]:
            raise ValueError("valid mask must match batch and token dimensions")
        valid = valid.to(device=value.device, dtype=torch.bool)
        if torch.any(~valid.any(dim=1)):
            raise ValueError("every batch row requires at least one valid token")
""",
        """        if not graph_export_active():
            if value.ndim != 3 or value.shape[-1] != self.d_model:
                raise ValueError("transformer stack expects [batch, tokens, d_model]")
            if valid.shape != value.shape[:2]:
                raise ValueError("valid mask must match batch and token dimensions")
        valid = valid.to(device=value.device, dtype=torch.bool)
        if not graph_export_active() and torch.any(~valid.any(dim=1)):
            raise ValueError("every batch row requires at least one valid token")
""",
    )

    replace_once(
        "trade_rl/rl/policies.py",
        "from trade_rl.rl.observations import CURRENT_WEIGHT_SOURCE\n",
        "from trade_rl.rl.export_context import graph_export_active\n"
        "from trade_rl.rl.observations import CURRENT_WEIGHT_SOURCE\n",
    )
    replace_once(
        "trade_rl/rl/policies.py",
        """        if features.ndim != 2 or features.shape[1] != expected_width:
            raise ValueError("shared actor features do not match declared layout")
""",
        """        if not graph_export_active() and (
            features.ndim != 2 or features.shape[1] != expected_width
        ):
            raise ValueError("shared actor features do not match declared layout")
""",
    )
    replace_once(
        "trade_rl/rl/policies.py",
        """        if actor_latent.ndim != 2:
            raise ValueError("actor latent must be rank-two")
        expected = self.n_symbols * self.context_dim
        if actor_latent.shape[1] != expected:
            raise ValueError("actor latent does not match hierarchical head layout")
""",
        """        if not graph_export_active() and actor_latent.ndim != 2:
            raise ValueError("actor latent must be rank-two")
        expected = self.n_symbols * self.context_dim
        if not graph_export_active() and actor_latent.shape[1] != expected:
            raise ValueError("actor latent does not match hierarchical head layout")
""",
    )
    replace_once(
        "trade_rl/rl/policies.py",
        """        if mask.ndim != 2 or mask.shape[1] != self.action_dim:
            raise ValueError("active action mask does not match action dimensions")
""",
        """        if not graph_export_active() and (
            mask.ndim != 2 or mask.shape[1] != self.action_dim
        ):
            raise ValueError("active action mask does not match action dimensions")
""",
    )
    replace_once(
        "trade_rl/rl/policies.py",
        """    def hierarchical_actor_outputs(
        self, observations: dict[str, torch.Tensor]
    ) -> HierarchicalActorOutputs:
""",
        '''    def deterministic_actions(
        self, observations: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Return distribution-free deterministic target weights for export."""

        features = self.extract_features(observations)
        if isinstance(features, tuple):
            features = features[0]
        latent_pi = self.mlp_extractor.forward_actor(features)
        return torch.tanh(self.action_net(latent_pi))

    def hierarchical_actor_outputs(
        self, observations: dict[str, torch.Tensor]
    ) -> HierarchicalActorOutputs:
''',
    )

    replace_once(
        "trade_rl/rl/export.py",
        """class _DeterministicActor(nn.Module):
    def __init__(self, policy: Any) -> None:
        super().__init__()
        self.policy = policy

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.policy._predict(observation, deterministic=True)
""",
        """class _DeterministicActor(nn.Module):
    def __init__(self, policy: Any, *, algorithm: str) -> None:
        super().__init__()
        self.policy = policy
        self.direct_ppo = algorithm == "ppo"
        action_space = getattr(policy, "action_space", None)
        low = np.asarray(getattr(action_space, "low", -1.0), dtype=np.float32)
        high = np.asarray(getattr(action_space, "high", 1.0), dtype=np.float32)
        self.register_buffer("action_low", torch.as_tensor(low), persistent=False)
        self.register_buffer("action_high", torch.as_tensor(high), persistent=False)
        action_dist = getattr(policy, "action_dist", None)
        self.squash_output = action_dist.__class__.__name__.endswith(
            "SquashedDiagGaussianDistribution"
        )

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        if not self.direct_ppo:
            return self.policy._predict(observation, deterministic=True)
        features = self.policy.extract_features(observation)
        if isinstance(features, tuple):
            features = features[0]
        latent_pi = self.policy.mlp_extractor.forward_actor(features)
        actions = self.policy.action_net(latent_pi)
        if self.squash_output:
            actions = torch.tanh(actions)
        return torch.maximum(torch.minimum(actions, self.action_high), self.action_low)
""",
    )
    replace_once(
        "trade_rl/rl/export.py",
        "    actor = _DeterministicActor(policy).eval()\n",
        "    actor = _DeterministicActor(policy, algorithm=algorithm).eval()\n",
    )

    replace_once(
        "trade_rl/rl/structured_export.py",
        "from trade_rl.rl.policy_identity import (\n",
        "from trade_rl.rl.policies import SharedPerAssetActorCriticPolicy\n"
        "from trade_rl.rl.policy_identity import (\n",
    )
    replace_once(
        "trade_rl/rl/structured_export.py",
        """        policy: nn.Module,
""",
        """        policy: SharedPerAssetActorCriticPolicy,
""",
    )
    replace_once(
        "trade_rl/rl/structured_export.py",
        """        prediction = self.policy._predict(observation, deterministic=True)
        return prediction
""",
        """        return self.policy.deterministic_actions(observation)
""",
    )
    replace_once(
        "trade_rl/rl/structured_export.py",
        """    if not isinstance(policy, nn.Module):
        raise TypeError("structured export model policy must be a torch module")
""",
        """    if not isinstance(policy, SharedPerAssetActorCriticPolicy):
        raise TypeError("structured export requires the shared per-asset policy")
""",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("tests", "implementation", "all"))
    args = parser.parse_args()
    if args.mode in {"tests", "all"}:
        write_tests()
    if args.mode in {"implementation", "all"}:
        apply_implementation()


if __name__ == "__main__":
    main()
