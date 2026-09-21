import math
from fractions import Fraction

import pytest

import trade_rl.simulation.quantities as quantities
from trade_rl.simulation.quantities import (
    exact_quantity,
    parse_quantity,
    project_quantity,
)


def _cache_hooks(function):
    cache_clear = getattr(function, "cache_clear", None)
    cache_info = getattr(function, "cache_info", None)
    function_name = getattr(function, "__name__", type(function).__name__)
    assert callable(cache_clear), f"{function_name} should expose cache clearing"
    assert callable(cache_info), f"{function_name} should expose cache statistics"
    return cache_clear, cache_info


def test_exact_quantity_reuses_repeated_conversions_and_evicts_old_entries():
    cache_clear, cache_info = _cache_hooks(exact_quantity)

    cache_clear()
    try:
        first = exact_quantity(0.1)
        second = exact_quantity(0.1)

        assert first is second
        assert first == Fraction(1, 10)
        assert second == Fraction(1, 10)

        for value in range(10_000, 12_048):
            exact_quantity(float(value))

        info = cache_info()
        assert info.hits == 1
        assert info.maxsize == 2048
        assert info.currsize == info.maxsize

        # The least-recently-used entry was evicted after exceeding the bound.
        exact_quantity(0.1)
        assert cache_info().misses == 2050
    finally:
        cache_clear()


def test_parse_quantity_reuses_canonical_fractions_and_evicts_old_entries():
    cache_clear, cache_info = _cache_hooks(
        getattr(quantities, "_parse_quantity_cached", None)
    )

    cache_clear()
    try:
        first = parse_quantity("1/10")
        second = parse_quantity("1/10")

        assert first is second
        assert first == Fraction(1, 10)
        assert second == Fraction(1, 10)

        for value in range(10_000, 12_048):
            parse_quantity(str(value))

        info = cache_info()
        assert info.hits == 1
        assert info.maxsize == 2048
        assert info.currsize == info.maxsize

        parse_quantity("1/10")
        assert cache_info().misses == 2050
    finally:
        cache_clear()


def test_parse_quantity_validates_type_before_cache_lookup():
    with pytest.raises(
        ValueError,
        match="exact quantity must be a canonical rational string",
    ):
        parse_quantity([])  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("canonical", "alias"),
    (("1/10", "2/20"), ("1/10", "01/10"), ("1", "1.0"), ("0", "-0")),
)
def test_parse_quantity_rejects_noncanonical_alias_after_canonical_cache_hit(
    canonical: str, alias: str
) -> None:
    parse_quantity(canonical)

    with pytest.raises(ValueError, match="exact quantity must be canonical"):
        parse_quantity(alias)


def test_parse_quantity_does_not_cache_overflowing_or_oversized_values() -> None:
    cache_clear, cache_info = _cache_hooks(
        getattr(quantities, "_parse_quantity_cached", None)
    )
    project_cache_clear, project_cache_info = _cache_hooks(
        getattr(quantities, "_project_fraction_quantity", None)
    )
    cache_clear()
    project_cache_clear()
    try:
        overflowing = "1" + "0" * 400
        before_overflow = cache_info()
        for _ in range(2):
            with pytest.raises(ValueError, match="outside finite float range"):
                parse_quantity(overflowing)
        after_overflow = cache_info()
        assert after_overflow.currsize == before_overflow.currsize

        numerator = "1" + "0" * 500 + "1"
        denominator = "1" + "0" * 500
        oversized = f"{numerator}/{denominator}"
        before_oversized = cache_info()
        before_oversized_projection = project_cache_info()
        assert parse_quantity(oversized) == Fraction(int(numerator), int(denominator))
        assert cache_info() == before_oversized
        assert project_cache_info() == before_oversized_projection
    finally:
        cache_clear()
        project_cache_clear()


def test_parse_quantity_does_not_cache_string_subclasses() -> None:
    cache_clear, cache_info = _cache_hooks(
        getattr(quantities, "_parse_quantity_cached", None)
    )
    cache_clear()
    try:

        class QuantityString(str):
            pass

        before = cache_info()
        assert parse_quantity(QuantityString("1/10")) == Fraction(1, 10)
        assert cache_info() == before
    finally:
        cache_clear()


def test_project_quantity_caches_only_fraction_inputs_with_a_bounded_lru():
    cache_clear, cache_info = _cache_hooks(
        getattr(quantities, "_project_fraction_quantity", None)
    )

    cache_clear()
    try:
        value = Fraction(1, 10)
        assert project_quantity(value) == 0.1
        assert project_quantity(value) == 0.1

        for numerator in range(10_000, 12_048):
            project_quantity(Fraction(numerator, 7_919))

        info = cache_info()
        assert info.hits == 1
        assert info.maxsize == 2048
        assert info.currsize == info.maxsize

        project_quantity(value)
        assert cache_info().misses == 2050

        before_non_fraction = cache_info()
        assert project_quantity(1) == 1.0  # type: ignore[arg-type]
        assert project_quantity(1) == 1.0  # type: ignore[arg-type]
        assert cache_info() == before_non_fraction

        class FractionSubclass(Fraction):
            pass

        assert project_quantity(FractionSubclass(1, 10)) == 0.1
        assert cache_info() == before_non_fraction

        oversized = Fraction(10**1000 + 1, 10**1000)
        assert project_quantity(oversized) == 1.0
        assert cache_info() == before_non_fraction

        positive_underflow = project_quantity(Fraction(1, 10**400))
        negative_underflow = project_quantity(Fraction(-1, 10**400))
        assert positive_underflow == 0.0
        assert negative_underflow == 0.0
        assert math.copysign(1.0, positive_underflow) == 1.0
        assert math.copysign(1.0, negative_underflow) == -1.0
        assert abs(quantities.exact_quantity(positive_underflow)) <= Fraction(
            1, 10**400
        )
        assert abs(quantities.exact_quantity(negative_underflow)) <= Fraction(
            1, 10**400
        )
    finally:
        cache_clear()
