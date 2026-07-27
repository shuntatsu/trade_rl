from __future__ import annotations

import importlib
import importlib.util

import torch

_TIMEFRAMES = ("15m", "1h", "4h", "1d")


def _fusion_type():
    module_name = "trade_rl.rl.timeframe_fusion"
    assert importlib.util.find_spec(module_name) is not None, (
        "timeframe_fusion module is not implemented"
    )
    module = importlib.import_module(module_name)
    fusion_type = getattr(module, "CrossTimeframeFusion", None)
    assert fusion_type is not None, "CrossTimeframeFusion is not implemented"
    return fusion_type


def _fusion():
    return _fusion_type()(
        latent_dims={timeframe: 8 for timeframe in _TIMEFRAMES},
        window_lengths={timeframe: 4 for timeframe in _TIMEFRAMES},
        d_model=16,
        heads=4,
        layers=2,
        ffn_multiplier=3,
        dropout=0.0,
        gate_bias=-2.0,
    )


def _inputs(*, requires_grad: bool = False):
    torch.manual_seed(211)
    batch, assets, window, channels = 2, 3, 4, 2
    latents = {
        timeframe: torch.randn(
            batch,
            assets,
            8,
            requires_grad=requires_grad,
        )
        for timeframe in _TIMEFRAMES
    }
    available = {
        timeframe: torch.ones(
            batch,
            assets,
            window,
            channels,
            dtype=torch.bool,
        )
        for timeframe in _TIMEFRAMES
    }
    staleness = {
        timeframe: torch.zeros(batch, assets, window, channels)
        for timeframe in _TIMEFRAMES
    }
    context = torch.randn(batch, assets, 16, requires_grad=requires_grad)
    return latents, available, staleness, context


def test_cross_timeframe_fusion_returns_one_finite_token_per_asset() -> None:
    fusion = _fusion().eval()
    latents, available, staleness, context = _inputs()

    with torch.no_grad():
        output = fusion(
            latents=latents,
            available=available,
            staleness=staleness,
            context=context,
        )

    assert output.shape == (2, 3, 16)
    assert torch.isfinite(output).all()


def test_fully_missing_timeframe_is_data_invariant() -> None:
    fusion = _fusion().eval()
    latents, available, staleness, context = _inputs()
    available["4h"].zero_()
    mutated = {key: value.clone() for key, value in latents.items()}
    mutated["4h"] += 10_000.0
    stale_mutated = {key: value.clone() for key, value in staleness.items()}
    stale_mutated["4h"] += 10_000.0

    with torch.no_grad():
        left = fusion(
            latents=latents,
            available=available,
            staleness=staleness,
            context=context,
        )
        right = fusion(
            latents=mutated,
            available=available,
            staleness=stale_mutated,
            context=context,
        )

    torch.testing.assert_close(left, right)


def test_fully_missing_timeframe_receives_no_gradient() -> None:
    fusion = _fusion()
    latents, available, staleness, context = _inputs(requires_grad=True)
    available["1d"].zero_()

    output = fusion(
        latents=latents,
        available=available,
        staleness=staleness,
        context=context,
    )
    output.square().sum().backward()

    assert latents["1d"].grad is not None
    assert torch.count_nonzero(latents["1d"].grad) == 0
    for timeframe in ("15m", "1h", "4h"):
        assert latents[timeframe].grad is not None
        assert torch.count_nonzero(latents[timeframe].grad) > 0
    assert context.grad is not None
    assert torch.count_nonzero(context.grad) > 0


def test_no_timeframe_history_falls_back_to_finite_context() -> None:
    fusion = _fusion().eval()
    latents, available, staleness, context = _inputs()
    for value in available.values():
        value.zero_()

    with torch.no_grad():
        output = fusion(
            latents=latents,
            available=available,
            staleness=staleness,
            context=context,
        )

    assert output.shape == context.shape
    assert torch.isfinite(output).all()


def test_explicit_staleness_quality_can_change_fusion() -> None:
    fusion = _fusion().eval()
    latents, available, staleness, context = _inputs()
    changed = {key: value.clone() for key, value in staleness.items()}
    changed["15m"][:, :, -1, :] = 100.0

    with torch.no_grad():
        fresh = fusion(
            latents=latents,
            available=available,
            staleness=staleness,
            context=context,
        )
        stale = fusion(
            latents=latents,
            available=available,
            staleness=changed,
            context=context,
        )

    assert not torch.allclose(fresh, stale)


def test_timeframe_order_and_metadata_are_validated() -> None:
    fusion_type = _fusion_type()
    try:
        fusion_type(
            latent_dims={"1h": 8, "15m": 8, "4h": 8, "1d": 8},
            window_lengths={timeframe: 4 for timeframe in _TIMEFRAMES},
            d_model=16,
            heads=4,
            layers=2,
            ffn_multiplier=3,
            dropout=0.0,
            gate_bias=-2.0,
        )
    except ValueError as error:
        assert "15m/1h/4h/1d" in str(error)
    else:
        raise AssertionError("unordered timeframe metadata must fail closed")
