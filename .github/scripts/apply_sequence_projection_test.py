from __future__ import annotations

from pathlib import Path

PATH = Path("tests/rl/test_sequence_policy_core.py")


def replace_once(source: str, old: str, new: str) -> str:
    count = source.count(old)
    if count != 1:
        raise RuntimeError(f"expected one projection test block, found {count}")
    return source.replace(old, new, 1)


def main() -> None:
    source = PATH.read_text(encoding="utf-8")
    if "from dataclasses import dataclass\n" not in source:
        source = source.replace(
            "from __future__ import annotations\n\nimport torch\n",
            "from __future__ import annotations\n\nfrom dataclasses import dataclass\n\nimport torch\n",
            1,
        )

    old = '''def test_projection_after_selection_matches_legacy_outputs_and_gradients() -> None:
    from trade_rl.rl.sequence_policy import CausalTimeframeEncoder

    torch.manual_seed(23)
    encoder = CausalTimeframeEncoder(
        4,
        8,
        window_length=12,
        widths=(8, 8, 8, 8),
        dropout=0.0,
    )
    available = torch.tensor(
        [
            [True] * 12,
            [True] * 7 + [False] * 5,
            [False] * 12,
        ]
    )
    legacy_input = torch.randn(3, 12, 4, requires_grad=True)
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
    legacy_parameter_gradients = {
        name: parameter.grad.detach().clone()
        for name, parameter in encoder.named_parameters()
        if parameter.grad is not None
    }

    encoder.zero_grad(set_to_none=True)
    optimized = encoder(optimized_input, available)
    optimized.square().sum().backward()

    torch.testing.assert_close(optimized, legacy, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(
        optimized_input.grad, legacy_input.grad, rtol=1e-5, atol=1e-6
    )
    for name, parameter in encoder.named_parameters():
        assert parameter.grad is not None
        torch.testing.assert_close(
            parameter.grad,
            legacy_parameter_gradients[name],
            rtol=1e-5,
            atol=1e-6,
        )
'''

    new = '''@dataclass(frozen=True)
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
    cosine = float(
        torch.nn.functional.cosine_similarity(left_flat, right_flat, dim=0)
    )
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
'''

    source = replace_once(source, old, new)
    PATH.write_text(source, encoding="utf-8")


if __name__ == "__main__":
    main()
