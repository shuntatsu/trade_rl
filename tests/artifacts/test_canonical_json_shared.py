from __future__ import annotations

import hashlib

import pytest

from trade_rl.artifacts.codec import canonical_json_bytes as artifact_json
from trade_rl.catalog.contracts import cache_key_digest
from trade_rl.domain.canonical_json import canonical_json_bytes as domain_json


@pytest.mark.parametrize(
    "value",
    (
        {"unicode": "日本語", "nested": {"b": 2, "a": 1}},
        {"sequence": (1, 2, {"x": True})},
        {"float": 0.125, "none": None},
    ),
)
def test_artifact_and_domain_encoders_are_identical(value: object) -> None:
    assert artifact_json(value) == domain_json(value)


def test_catalog_cache_key_uses_shared_canonical_bytes() -> None:
    value = {"sequence": (1, 2), "unicode": "日本語"}
    assert cache_key_digest(value) == hashlib.sha256(domain_json(value)).hexdigest()


@pytest.mark.parametrize("value", (float("nan"), float("inf"), float("-inf")))
def test_shared_canonical_json_rejects_non_finite_values(value: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        domain_json({"value": value})
