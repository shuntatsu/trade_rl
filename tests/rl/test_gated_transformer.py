from __future__ import annotations

import importlib
import importlib.util

import torch


def _stack_type():
    module_name = "trade_rl.rl.gated_transformer"
    assert importlib.util.find_spec(module_name) is not None, (
        "gated_transformer module is not implemented"
    )
    module = importlib.import_module(module_name)
    stack_type = getattr(module, "GatedTransformerStack", None)
    assert stack_type is not None, "GatedTransformerStack is not implemented"
    return stack_type


def test_masked_tokens_cannot_change_unmasked_outputs() -> None:
    torch.manual_seed(101)
    stack = _stack_type()(
        d_model=24,
        heads=4,
        layers=2,
        ffn_multiplier=3,
        dropout=0.0,
        gate_bias=-2.0,
    ).eval()
    value = torch.randn(2, 5, 24)
    valid = torch.tensor(
        [
            [True, True, False, True, True],
            [True, False, True, True, False],
        ]
    )
    changed = value.clone()
    changed[~valid] += 10_000.0

    with torch.no_grad():
        left = stack(value, valid=valid)
        right = stack(changed, valid=valid)

    torch.testing.assert_close(left[valid], right[valid])
    assert torch.count_nonzero(left[~valid]) == 0
    assert torch.count_nonzero(right[~valid]) == 0


def test_masked_token_inputs_receive_no_gradient() -> None:
    torch.manual_seed(103)
    stack = _stack_type()(
        d_model=16,
        heads=4,
        layers=2,
        ffn_multiplier=3,
        dropout=0.0,
        gate_bias=-2.0,
    )
    value = torch.randn(1, 5, 16, requires_grad=True)
    valid = torch.tensor([[True, True, False, True, False]])

    output = stack(value, valid=valid)
    output.square().sum().backward()

    assert value.grad is not None
    assert torch.count_nonzero(value.grad[~valid]) == 0
    assert torch.count_nonzero(value.grad[valid]) > 0


def test_all_invalid_rows_fail_closed() -> None:
    stack = _stack_type()(
        d_model=16,
        heads=4,
        layers=1,
        ffn_multiplier=2,
        dropout=0.0,
        gate_bias=-2.0,
    )
    value = torch.randn(2, 3, 16)
    valid = torch.tensor([[True, False, False], [False, False, False]])

    try:
        stack(value, valid=valid)
    except ValueError as error:
        assert "at least one valid token" in str(error)
    else:
        raise AssertionError("all-invalid token rows must fail closed")


def test_residual_gates_start_at_configured_bias() -> None:
    stack = _stack_type()(
        d_model=16,
        heads=4,
        layers=2,
        ffn_multiplier=3,
        dropout=0.0,
        gate_bias=-1.75,
    )

    assert len(stack.blocks) == 2
    for block in stack.blocks:
        torch.testing.assert_close(
            block.attention_gate.gate,
            torch.full((16,), -1.75),
        )
        torch.testing.assert_close(
            block.ffn_gate.gate,
            torch.full((16,), -1.75),
        )


def test_stack_validates_architecture_dimensions() -> None:
    stack_type = _stack_type()
    invalid_cases = (
        {"d_model": 0, "heads": 1, "layers": 1, "ffn_multiplier": 2},
        {"d_model": 15, "heads": 4, "layers": 1, "ffn_multiplier": 2},
        {"d_model": 16, "heads": 4, "layers": 0, "ffn_multiplier": 2},
        {"d_model": 16, "heads": 4, "layers": 1, "ffn_multiplier": 0},
    )
    for kwargs in invalid_cases:
        try:
            stack_type(**kwargs, dropout=0.0, gate_bias=-2.0)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid architecture accepted: {kwargs}")
