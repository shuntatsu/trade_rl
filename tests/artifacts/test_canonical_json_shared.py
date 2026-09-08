from __future__ import annotations

import pytest

from trade_rl.artifacts.canonical import canonical_json_bytes


@pytest.mark.parametrize(
    ("value", "expected"),
    (
        (
            {"unicode": "日本語", "nested": {"b": 2, "a": 1}},
            '{"nested":{"a":1,"b":2},"unicode":"日本語"}'.encode("utf-8"),
        ),
        (
            {"sequence": (1, 2, {"x": True})},
            b'{"sequence":[1,2,{"x":true}]}',
        ),
        (
            {"float": 0.125, "none": None},
            b'{"float":0.125,"none":null}',
        ),
    ),
)
def test_canonical_json_bytes_are_stable(value: object, expected: bytes) -> None:
    assert canonical_json_bytes(value) == expected


@pytest.mark.parametrize("value", (float("nan"), float("inf"), float("-inf")))
def test_canonical_json_rejects_non_finite_values(value: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        canonical_json_bytes({"value": value})
