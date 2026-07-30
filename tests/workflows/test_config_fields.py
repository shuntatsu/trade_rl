from __future__ import annotations

from dataclasses import dataclass

import pytest

from trade_rl.workflows.config_fields import (
    require_dataclass_fields,
    require_exact_fields,
)


def test_require_exact_fields_rejects_unknown_names() -> None:
    with pytest.raises(ValueError, match="config.*unknown fields.*typo"):
        require_exact_fields(
            {"required": 1, "typo": 2},
            required={"required"},
            optional={"optional"},
            field="config",
        )


def test_require_exact_fields_rejects_missing_required_names() -> None:
    with pytest.raises(ValueError, match="config.*missing required fields.*required"):
        require_exact_fields(
            {},
            required={"required"},
            optional=set(),
            field="config",
        )


def test_require_exact_fields_returns_a_copy() -> None:
    original = {"required": 1}
    resolved = require_exact_fields(
        original,
        required={"required"},
        optional=set(),
        field="config",
    )
    assert resolved == original
    assert resolved is not original


def test_require_dataclass_fields_rejects_shadowed_and_unknown_fields() -> None:
    @dataclass
    class Example:
        active: int
        shadowed: int = 0

    with pytest.raises(ValueError, match="example.*unknown fields.*shadowed, typo"):
        require_dataclass_fields(
            {"active": 1, "shadowed": 2, "typo": 3},
            Example,
            field="example",
            excluded={"shadowed"},
        )


def test_require_dataclass_fields_restores_json_lists_for_tuple_fields() -> None:
    @dataclass
    class Example:
        trigger_volume_fractions: tuple[float, float, float, float]
        label: str = "nominal"

    original = {
        "trigger_volume_fractions": [1.0, 0.5, 0.25, 0.0],
        "label": "stress",
    }

    resolved = require_dataclass_fields(original, Example, field="example")

    assert resolved == {
        "trigger_volume_fractions": (1.0, 0.5, 0.25, 0.0),
        "label": "stress",
    }
    assert original["trigger_volume_fractions"] == [1.0, 0.5, 0.25, 0.0]
