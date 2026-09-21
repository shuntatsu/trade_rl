from fractions import Fraction

from trade_rl.simulation.quantities import exact_quantity


def test_exact_quantity_reuses_repeated_conversions_and_evicts_old_entries():
    cache_clear = getattr(exact_quantity, "cache_clear", None)
    cache_info = getattr(exact_quantity, "cache_info", None)
    assert callable(cache_clear), "exact_quantity should expose its cache clear hook"
    assert callable(cache_info), "exact_quantity should expose cache statistics"

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
