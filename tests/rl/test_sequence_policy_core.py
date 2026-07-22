from __future__ import annotations

from dataclasses import dataclass

import torch

from trade_rl.rl.sequence_policy import (
    CausalTemporalBlock,
    MultiTimeframeAssetEncoder,
    SequencePolicyArchitecture,
)

torch.set_num_threads(1)


def test_causal_temporal_block_does_not_use_future_values() -> None:
    torch.manual_seed(7)
    block = CausalTemporalBlock(
        in_channels=4,
        out_channels=8,
        kernel_size=3,
        dilation=2,
        dropout=0.0,
    ).eval()
    baseline = torch.randn(2, 4, 24)
    mutated = baseline.clone()
    mutated[:, :, 13:] += 10_000.0

    with torch.no_grad():
        left = block(baseline)
        right = block(mutated)

    torch.testing.assert_close(left[:, :, :13], right[:, :, :13])


def test_compact_sequence_capacity_preserves_all_receptive_field_depths() -> None:
    import trade_rl.rl.sequence_policy as sequence_policy

    resolver = getattr(sequence_policy, "sequence_encoder_widths", None)
    assert callable(resolver)
    widths = resolver("compact")

    assert tuple(widths) == ("15m", "1h", "4h", "1d")
    assert tuple(map(len, widths.values())) == (6, 7, 6, 5)
    assert max(max(values) for values in widths.values()) == 112


def test_multitimeframe_encoder_shapes_and_parameter_budget() -> None:
    architecture = SequencePolicyArchitecture(
        input_channels={"15m": 18, "1h": 20, "4h": 16, "1d": 12},
        window_lengths={"15m": 12, "1h": 16, "4h": 10, "1d": 8},
        latent_dims={"15m": 192, "1h": 192, "4h": 160, "1d": 128},
        asset_state_width=40,
        snapshot_width=64,
        n_symbols=3,
        d_model=320,
        attention_heads=8,
        attention_layers=2,
    )
    encoder = MultiTimeframeAssetEncoder(architecture)
    batch, assets = 3, 3
    sequences = {
        "15m": torch.randn(batch, assets, 12, 18),
        "1h": torch.randn(batch, assets, 16, 20),
        "4h": torch.randn(batch, assets, 10, 16),
        "1d": torch.randn(batch, assets, 8, 12),
    }
    available = {
        key: torch.ones(value.shape[:-1], dtype=torch.bool)
        for key, value in sequences.items()
    }
    snapshot = torch.randn(batch, assets, 64)
    asset_state = torch.randn(batch, assets, 40)
    active = torch.tensor([[1, 1, 1], [1, 0, 1], [0, 0, 0]], dtype=torch.bool)

    tokens, pooled = encoder(
        sequences=sequences,
        available=available,
        snapshot=snapshot,
        asset_state=asset_state,
        active=active,
    )

    assert tokens.shape == (batch, assets, 320)
    assert pooled.shape == (batch, 320)
    assert torch.isfinite(tokens).all()
    assert torch.isfinite(pooled).all()
    assert torch.count_nonzero(tokens[2]) == 0
    assert torch.count_nonzero(pooled[2]) == 0
    parameter_count = sum(parameter.numel() for parameter in encoder.parameters())
    assert 2_000_000 <= parameter_count <= 10_000_000


def test_timeframe_receptive_fields_cover_declared_windows() -> None:
    architecture = SequencePolicyArchitecture(
        input_channels={"15m": 3, "1h": 3, "4h": 3, "1d": 3},
        window_lengths={"15m": 96, "1h": 168, "4h": 120, "1d": 60},
        latent_dims={"15m": 16, "1h": 16, "4h": 16, "1d": 16},
        asset_state_width=4,
        snapshot_width=8,
        n_symbols=3,
        d_model=24,
        attention_heads=4,
        attention_layers=1,
        dropout=0.0,
    )
    encoder = MultiTimeframeAssetEncoder(architecture)

    for timeframe, window in architecture.window_lengths.items():
        assert encoder.timeframe_encoders[timeframe].receptive_field >= window


def test_timeframe_projection_runs_only_after_causal_timestep_selection() -> None:
    from trade_rl.rl.sequence_policy import CausalTimeframeEncoder

    encoder = CausalTimeframeEncoder(
        4,
        8,
        window_length=12,
        widths=(8, 8, 8, 8),
        dropout=0.0,
    )
    projection_input_shapes: list[torch.Size] = []

    def capture_projection_input(
        _module: torch.nn.Module, args: tuple[torch.Tensor, ...]
    ) -> None:
        projection_input_shapes.append(args[0].shape)

    handle = encoder.projection.register_forward_pre_hook(capture_projection_input)
    try:
        encoder(
            torch.randn(3, 12, 4),
            torch.tensor(
                [
                    [True] * 12,
                    [True] * 7 + [False] * 5,
                    [False] * 12,
                ]
            ),
        )
    finally:
        handle.remove()

    assert projection_input_shapes == [torch.Size((3, 8))]


@dataclass(frozen=True)
class _ProjectionEquivalenceCase:
    legacy: torch.Tensor
    optimized: torch.Tensor
    legacy_input_gradient: torch.Tensor
    optimized_input_gradient: torch.Tensor
    legacy_parameter_gradients: dict[str, torch.Tensor]
    optimized_parameter_gradients: dict[str, torch.Tensor]


def _projection_equivalence_case(dtype: torch.dtype) -> _ProjectionEquivalenceCase:
    from trade_rl.rl.sequence_policy import CausalTimeframeEncoder

    torch.manual_seed(23)
    encoder = CausalTimeframeEncoder(
        4,
        8,
        window_length=12,
        widths=(8, 8, 8, 8),
        dropout=0.0,
    ).to(dtype=dtype)
    available = torch.tensor(
        [
            [True] * 12,
            [True] * 7 + [False] * 5,
            [False] * 12,
        ]
    )
    legacy_input = torch.randn(3, 12, 4, dtype=dtype, requires_grad=True)
    optimized_input = legacy_input.detach().clone().requires_grad_(True)

    legacy_sequence = encoder.projection(encoder.forward_sequence(legacy_input))
    positions = torch.arange(12).expand_as(available)
    indices = positions.masked_fill(~available, -1).max(dim=1).values
    safe = indices.clamp_min(0)
    legacy_selected = legacy_sequence[torch.arange(3), safe]
    legacy = torch.where(
        (indices >= 0).unsqueeze(1),
        legacy_selected,
        torch.zeros_like(legacy_selected),
    )
    legacy.square().sum().backward()
    assert legacy_input.grad is not None
    legacy_input_gradient = legacy_input.grad.detach().clone()
    legacy_parameter_gradients = {
        name: parameter.grad.detach().clone()
        for name, parameter in encoder.named_parameters()
        if parameter.grad is not None
    }

    encoder.zero_grad(set_to_none=True)
    optimized = encoder(optimized_input, available)
    optimized.square().sum().backward()
    assert optimized_input.grad is not None
    optimized_input_gradient = optimized_input.grad.detach().clone()
    optimized_parameter_gradients = {
        name: parameter.grad.detach().clone()
        for name, parameter in encoder.named_parameters()
        if parameter.grad is not None
    }

    return _ProjectionEquivalenceCase(
        legacy=legacy.detach().clone(),
        optimized=optimized.detach().clone(),
        legacy_input_gradient=legacy_input_gradient,
        optimized_input_gradient=optimized_input_gradient,
        legacy_parameter_gradients=legacy_parameter_gradients,
        optimized_parameter_gradients=optimized_parameter_gradients,
    )


def _relative_l2(left: torch.Tensor, right: torch.Tensor) -> float:
    denominator = max(
        float(torch.linalg.vector_norm(left)),
        float(torch.linalg.vector_norm(right)),
        1e-12,
    )
    return float(torch.linalg.vector_norm(left - right)) / denominator


def _assert_gradient_semantics(left: torch.Tensor, right: torch.Tensor) -> None:
    assert left.shape == right.shape
    assert left.dtype == right.dtype
    assert torch.isfinite(left).all()
    assert torch.isfinite(right).all()
    left_flat = left.reshape(-1)
    right_flat = right.reshape(-1)
    left_norm = float(torch.linalg.vector_norm(left_flat))
    right_norm = float(torch.linalg.vector_norm(right_flat))
    if left_norm == 0.0 or right_norm == 0.0:
        assert left_norm == right_norm == 0.0
        return
    cosine = float(torch.nn.functional.cosine_similarity(left_flat, right_flat, dim=0))
    assert cosine >= 0.999999
    assert _relative_l2(left, right) <= 2e-5


def test_projection_after_selection_matches_legacy_in_float64() -> None:
    case = _projection_equivalence_case(torch.float64)

    torch.testing.assert_close(case.optimized, case.legacy, rtol=1e-9, atol=1e-10)
    torch.testing.assert_close(
        case.optimized_input_gradient,
        case.legacy_input_gradient,
        rtol=1e-9,
        atol=1e-10,
    )
    assert case.optimized_parameter_gradients.keys() == (
        case.legacy_parameter_gradients.keys()
    )
    for name, gradient in case.optimized_parameter_gradients.items():
        torch.testing.assert_close(
            gradient,
            case.legacy_parameter_gradients[name],
            rtol=1e-9,
            atol=1e-10,
        )
    assert torch.count_nonzero(case.optimized[2]) == 0
    assert torch.count_nonzero(case.optimized_input_gradient[2]) == 0


def test_projection_after_selection_preserves_float32_gradient_semantics() -> None:
    case = _projection_equivalence_case(torch.float32)

    torch.testing.assert_close(case.optimized, case.legacy, rtol=1e-5, atol=2e-6)
    _assert_gradient_semantics(
        case.optimized_input_gradient,
        case.legacy_input_gradient,
    )
    assert case.optimized_parameter_gradients.keys() == (
        case.legacy_parameter_gradients.keys()
    )
    for name, gradient in case.optimized_parameter_gradients.items():
        _assert_gradient_semantics(gradient, case.legacy_parameter_gradients[name])
    assert torch.count_nonzero(case.optimized[2]) == 0
    assert torch.count_nonzero(case.optimized_input_gradient[2]) == 0
    assert torch.count_nonzero(case.optimized_input_gradient[:2]) > 0


def test_oldest_declared_history_can_influence_final_timeframe_latent() -> None:
    torch.manual_seed(17)
    from trade_rl.rl.sequence_policy import CausalTimeframeEncoder

    encoder = CausalTimeframeEncoder(
        2,
        8,
        window_length=96,
        widths=(8, 8, 8, 8, 8, 8),
        dropout=0.0,
    ).eval()
    baseline = torch.zeros(1, 96, 2)
    mutated = baseline.clone()
    mutated[:, 0, 0] = 10.0

    with torch.no_grad():
        left = encoder(baseline)
        right = encoder(mutated)

    assert not torch.allclose(left, right)


def test_asset_tokens_retain_symbol_specific_context() -> None:
    torch.manual_seed(19)
    architecture = SequencePolicyArchitecture(
        input_channels={"15m": 3, "1h": 3, "4h": 3, "1d": 3},
        window_lengths={"15m": 4, "1h": 4, "4h": 4, "1d": 4},
        latent_dims={"15m": 8, "1h": 8, "4h": 8, "1d": 8},
        asset_state_width=4,
        snapshot_width=8,
        n_symbols=3,
        d_model=24,
        attention_heads=4,
        attention_layers=1,
        dropout=0.0,
    )
    encoder = MultiTimeframeAssetEncoder(architecture).eval()
    sequences = {
        timeframe: torch.zeros(1, 3, 4, 3) for timeframe in architecture.input_channels
    }
    available = {
        timeframe: torch.ones(1, 3, 4, dtype=torch.bool)
        for timeframe in architecture.input_channels
    }
    snapshot = torch.zeros(1, 3, 8)
    snapshot[:, 1, 0] = 2.0
    state = torch.zeros(1, 3, 4)
    active = torch.ones(1, 3, dtype=torch.bool)

    with torch.no_grad():
        tokens, _ = encoder(
            sequences=sequences,
            available=available,
            snapshot=snapshot,
            asset_state=state,
            active=active,
        )

    assert tokens.shape == (1, 3, 24)
    assert not torch.allclose(tokens[:, 0], tokens[:, 1])
    assert not torch.allclose(tokens[:, 1], tokens[:, 2])


def test_shared_per_asset_action_head_reuses_parameters_and_is_equivariant() -> None:
    from trade_rl.rl.policies import SharedPerAssetActionHead

    torch.manual_seed(29)
    single = SharedPerAssetActionHead(
        n_symbols=1,
        token_dim=4,
        context_dim=9,
        hidden_dims=(7, 5),
    ).eval()
    shared = SharedPerAssetActionHead(
        n_symbols=3,
        token_dim=4,
        context_dim=9,
        hidden_dims=(7, 5),
    ).eval()
    shared.shared_head.load_state_dict(single.shared_head.state_dict())

    assert sum(parameter.numel() for parameter in single.parameters()) == sum(
        parameter.numel() for parameter in shared.parameters()
    )
    contexts = torch.randn(2, 3, 9)
    contexts[:, :, -1] = 1.0
    permutation = torch.tensor([2, 0, 1])
    with torch.no_grad():
        original = shared(contexts.reshape(2, -1))
        permuted = shared(contexts[:, permutation].reshape(2, -1))

    torch.testing.assert_close(permuted, original[:, permutation])


def test_shared_actor_masks_explicitly_inactive_assets() -> None:
    from trade_rl.rl.policies import SharedPerAssetActionHead

    head = SharedPerAssetActionHead(
        n_symbols=2,
        token_dim=3,
        context_dim=7,
        hidden_dims=(5,),
    ).eval()
    contexts = torch.randn(1, 2, 7)
    contexts[:, :, -1] = 1.0
    contexts[:, 1, -1] = 0.0

    with torch.no_grad():
        actions = head(contexts.reshape(1, -1))

    assert actions.shape == (1, 2)
    assert actions[0, 1].item() == 0.0


def test_shared_sequence_policy_installs_shared_actor_and_portfolio_critic() -> None:
    import numpy as np
    from gymnasium import spaces

    from trade_rl.rl.policies import (
        SequenceAssetFeatureExtractor,
        SharedPerAssetActionHead,
        SharedPerAssetActorCriticPolicy,
    )

    n_symbols = 2
    timeframes = ("15m", "1h", "4h", "1d")
    feature_counts = {timeframe: 2 for timeframe in timeframes}
    window_lengths = {timeframe: 3 for timeframe in timeframes}
    observation_spaces: dict[str, spaces.Space] = {
        "current_snapshot": spaces.Box(
            -1.0, 1.0, shape=(n_symbols, 8), dtype=np.float32
        ),
        "asset_state": spaces.Box(-1.0, 1.0, shape=(n_symbols, 4), dtype=np.float32),
        "global_state": spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32),
        "active": spaces.Box(0.0, 1.0, shape=(n_symbols,), dtype=np.float32),
    }
    for timeframe in timeframes:
        shape = (n_symbols, 3, 2)
        observation_spaces[f"sequence_{timeframe}_values"] = spaces.Box(
            -1.0, 1.0, shape=shape, dtype=np.float16
        )
        observation_spaces[f"sequence_{timeframe}_available"] = spaces.Box(
            0, 1, shape=shape, dtype=np.uint8
        )
        observation_spaces[f"sequence_{timeframe}_staleness"] = spaces.Box(
            0.0, 10.0, shape=shape, dtype=np.float16
        )
    observation_space = spaces.Dict(observation_spaces)
    policy = SharedPerAssetActorCriticPolicy(
        observation_space,
        spaces.Box(-1.0, 1.0, shape=(n_symbols,), dtype=np.float32),
        lambda _: 1e-3,
        net_arch={"pi": [11], "vf": [13]},
        features_extractor_class=SequenceAssetFeatureExtractor,
        features_extractor_kwargs={
            "feature_counts": feature_counts,
            "window_lengths": window_lengths,
            "snapshot_width": 8,
            "asset_state_width": 4,
            "global_width": 3,
            "n_symbols": n_symbols,
            "d_model": 16,
            "attention_heads": 4,
            "attention_layers": 1,
            "dropout": 0.0,
        },
        shared_actor_n_symbols=n_symbols,
        shared_actor_d_model=16,
        shared_actor_global_dim=128,
        shared_actor_net_arch=(11,),
    )

    assert isinstance(policy.action_net, SharedPerAssetActionHead)
    assert policy.mlp_extractor.latent_dim_vf == 13
    assert policy.action_net.n_symbols == n_symbols
    constructor = policy._get_constructor_parameters()
    assert constructor["shared_actor_n_symbols"] == n_symbols
    assert constructor["shared_actor_net_arch"] == (11,)


def test_partial_feature_availability_keeps_latest_timestep_usable() -> None:
    from gymnasium import spaces

    from trade_rl.rl.policies import SequenceAssetFeatureExtractor

    timeframes = ("15m", "1h", "4h", "1d")
    feature_counts = {timeframe: 2 for timeframe in timeframes}
    window_lengths = {timeframe: 3 for timeframe in timeframes}
    observation_spaces = {
        "current_snapshot": spaces.Box(-1.0, 1.0, shape=(1, 8), dtype=float),
        "asset_state": spaces.Box(-1.0, 1.0, shape=(1, 4), dtype=float),
        "global_state": spaces.Box(-1.0, 1.0, shape=(3,), dtype=float),
        "active": spaces.Box(0.0, 1.0, shape=(1,), dtype=float),
    }
    for timeframe in timeframes:
        shape = (1, 3, 2)
        observation_spaces[f"sequence_{timeframe}_values"] = spaces.Box(
            -1.0, 1.0, shape=shape, dtype=float
        )
        observation_spaces[f"sequence_{timeframe}_available"] = spaces.Box(
            0, 1, shape=shape, dtype="uint8"
        )
        observation_spaces[f"sequence_{timeframe}_staleness"] = spaces.Box(
            0.0, 10.0, shape=shape, dtype="float16"
        )
    extractor = SequenceAssetFeatureExtractor(
        spaces.Dict(observation_spaces),
        feature_counts=feature_counts,
        window_lengths=window_lengths,
        snapshot_width=8,
        asset_state_width=4,
        global_width=3,
        n_symbols=1,
        d_model=16,
        attention_heads=4,
        attention_layers=1,
        dropout=0.0,
    ).eval()
    observation = {
        "current_snapshot": torch.zeros(1, 1, 8),
        "asset_state": torch.zeros(1, 1, 4),
        "global_state": torch.zeros(1, 3),
        "active": torch.ones(1, 1),
    }
    for timeframe in timeframes:
        values = torch.zeros(1, 1, 3, 2)
        values[:, :, -1, 0] = 1.0
        availability = torch.ones(1, 1, 3, 2, dtype=torch.uint8)
        availability[:, :, -1, 1] = 0
        observation[f"sequence_{timeframe}_values"] = values
        observation[f"sequence_{timeframe}_available"] = availability
        observation[f"sequence_{timeframe}_staleness"] = torch.zeros(
            1, 1, 3, 2, dtype=torch.float16
        )

    captured: dict[str, torch.Tensor] = {}
    original = extractor.asset_encoder.timeframe_encoders["15m"].forward

    def capture(value: torch.Tensor, available: torch.Tensor | None = None):
        assert available is not None
        captured["mask"] = available.detach().clone()
        return original(value, available)

    extractor.asset_encoder.timeframe_encoders["15m"].forward = capture  # type: ignore[method-assign]
    with torch.no_grad():
        output = extractor(observation)

    assert output.shape == (1, 161)
    assert bool(captured["mask"][0, -1])


def test_shared_sequence_policy_uses_squashed_target_weight_distribution() -> None:
    import numpy as np
    from gymnasium import spaces
    from stable_baselines3.common.distributions import (
        SquashedDiagGaussianDistribution,
    )

    from trade_rl.rl.policies import (
        SequenceAssetFeatureExtractor,
        SharedPerAssetActorCriticPolicy,
    )

    n_symbols = 2
    timeframes = ("15m", "1h", "4h", "1d")
    feature_counts = {timeframe: 2 for timeframe in timeframes}
    window_lengths = {timeframe: 3 for timeframe in timeframes}
    components: dict[str, spaces.Space] = {
        "current_snapshot": spaces.Box(
            -10.0, 10.0, shape=(n_symbols, 8), dtype=np.float32
        ),
        "asset_state": spaces.Box(-10.0, 10.0, shape=(n_symbols, 4), dtype=np.float32),
        "global_state": spaces.Box(-10.0, 10.0, shape=(3,), dtype=np.float32),
        "active": spaces.Box(0.0, 1.0, shape=(n_symbols,), dtype=np.float32),
    }
    for timeframe in timeframes:
        shape = (n_symbols, 3, 2)
        components[f"sequence_{timeframe}_values"] = spaces.Box(
            -10.0, 10.0, shape=shape, dtype=np.float16
        )
        components[f"sequence_{timeframe}_available"] = spaces.Box(
            0, 1, shape=shape, dtype=np.uint8
        )
        components[f"sequence_{timeframe}_staleness"] = spaces.Box(
            0.0, 100.0, shape=shape, dtype=np.float16
        )
    observation_space = spaces.Dict(components)
    policy = SharedPerAssetActorCriticPolicy(
        observation_space,
        spaces.Box(-1.0, 1.0, shape=(n_symbols,), dtype=np.float32),
        lambda _: 1e-3,
        net_arch={"pi": [11], "vf": [13]},
        features_extractor_class=SequenceAssetFeatureExtractor,
        features_extractor_kwargs={
            "feature_counts": feature_counts,
            "window_lengths": window_lengths,
            "snapshot_width": 8,
            "asset_state_width": 4,
            "global_width": 3,
            "n_symbols": n_symbols,
            "d_model": 16,
            "attention_heads": 4,
            "attention_layers": 1,
            "dropout": 0.0,
        },
        shared_actor_n_symbols=n_symbols,
        shared_actor_d_model=16,
        shared_actor_global_dim=128,
        shared_actor_net_arch=(11,),
        log_std_init=-0.5,
    )
    assert isinstance(policy.action_dist, SquashedDiagGaussianDistribution)
    assert policy.action_distribution_name == "masked_shared_squashed_diag_gaussian"

    batch = 4
    observations: dict[str, torch.Tensor] = {}
    for key, space in observation_space.spaces.items():
        array = np.zeros((batch, *space.shape), dtype=space.dtype)
        if key == "active" or key.endswith("_available"):
            array.fill(1)
        observations[key] = torch.as_tensor(array)

    distribution = policy.get_distribution(observations)
    stochastic = distribution.get_actions(deterministic=False)
    deterministic = distribution.get_actions(deterministic=True)
    assert torch.all(stochastic <= 1.0)
    assert torch.all(stochastic >= -1.0)
    assert torch.all(deterministic <= 1.0)
    assert torch.all(deterministic >= -1.0)

    boundary = torch.tensor(
        [[0.999, -0.999], [-0.999, 0.999], [0.5, -0.5], [0.0, 0.0]],
        dtype=torch.float32,
    )
    values, log_prob, entropy = policy.evaluate_actions(observations, boundary)
    assert torch.isfinite(values).all()
    assert torch.isfinite(log_prob).all()
    assert entropy is None
