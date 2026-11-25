"""Opt-in conceptual indicator-family confirmation before deduplication."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from trade_rl.data.features.divergence import DivergenceConfig, detect_divergences

FAMILIES = (("RSI", "price"), ("CCI", "price"), ("CMF", "flow"))


def _inputs():
    c = np.asarray(
        [11, 10.5, 10, 9, 8, 9, 10, 11, 12, 11, 10, 9.5, 8.5, 7, 7.5, 9, 10, 11],
        dtype=float,
    )
    a = np.asarray(
        [0, -1, -2, -3, -6, -5, -4, -2, 0, -1, -2, -1, -2, -3, -1, 0, 1, 2], dtype=float
    )
    b = a.copy()
    b[13] = -6
    b[14] = -4
    return c, {"RSI": a, "CCI": a.copy(), "CMF": b}


def test_family_gate_rejects_duplicate_votes_and_does_not_consume_pivot():
    c, bank = _inputs()
    cfg = DivergenceConfig(
        pivot_period=2, max_bars=30, min_indicators=2, validate_lines=False
    )
    baseline = detect_divergences(
        high=c + 0.2, low=c - 0.2, close=c, indicators=bank, config=cfg
    )
    mixed = detect_divergences(
        high=c + 0.2,
        low=c - 0.2,
        close=c,
        indicators=bank,
        config=replace(cfg, indicator_families=FAMILIES, min_families=2),
    )
    assert any(
        e.kind == "bull_regular" and e.pivot_at == 4 and e.observed_at == 14
        for e in baseline.events
    )
    assert not any(
        e.kind == "bull_regular" and e.pivot_at == 4 and e.observed_at == 14
        for e in mixed.events
    )
    confirmed = [
        e
        for e in mixed.events
        if e.kind == "bull_regular" and e.pivot_at == 4 and e.observed_at == 15
    ]
    assert {e.indicator for e in confirmed} == set(bank)


def test_family_gate_preserves_legacy_output_at_one_family():
    c, bank = _inputs()
    cfg = DivergenceConfig(
        pivot_period=2, max_bars=30, min_indicators=2, validate_lines=False
    )
    normal = detect_divergences(
        high=c + 0.2, low=c - 0.2, close=c, indicators=bank, config=cfg
    )
    explicit = detect_divergences(
        high=c + 0.2,
        low=c - 0.2,
        close=c,
        indicators=bank,
        config=replace(cfg, indicator_families=FAMILIES, min_families=1),
    )
    assert normal.events == explicit.events
    for key in ("bull_regular", "bear_regular", "bull_hidden", "bear_hidden"):
        np.testing.assert_array_equal(getattr(normal, key), getattr(explicit, key))


@pytest.mark.parametrize(
    "bad",
    [
        {"min_families": 2},
        {"indicator_families": (("RSI", ""),), "min_families": 1},
        {"indicator_families": (("RSI", "price"), ("RSI", "flow")), "min_families": 2},
        {"indicator_families": (("RSI", "price"), ("CCI", "price")), "min_families": 2},
        {"indicator_families": [("RSI", "price")], "min_families": 1},
        {"min_families": True},
    ],
)
def test_bad_family_configs_fail_closed(bad):
    with pytest.raises((ValueError, TypeError)):
        DivergenceConfig(**bad)


def test_incomplete_mapping_fails_closed():
    c, bank = _inputs()
    with pytest.raises(ValueError, match="exactly"):
        detect_divergences(
            high=c + 0.2,
            low=c - 0.2,
            close=c,
            indicators=bank,
            config=DivergenceConfig(indicator_families=(("RSI", "price"),)),
        )


@pytest.mark.parametrize("bearish", [False, True])
def test_three_family_requirement_and_named_quorum_are_both_enforced(
    bearish: bool,
) -> None:
    c, bank = _inputs()
    if bearish:
        c = 30 - c
        bank = {name: -values for name, values in bank.items()}
    kind = "bear_regular" if bearish else "bull_regular"
    mapping = (("RSI", "price"), ("CCI", "trend"), ("CMF", "flow"))
    cfg = DivergenceConfig(
        pivot_period=2,
        max_bars=30,
        min_indicators=2,
        validate_lines=False,
        indicator_families=mapping,
        min_families=3,
    )
    result = detect_divergences(
        high=c + 0.2,
        low=c - 0.2,
        close=c,
        indicators=bank,
        config=cfg,
    )
    assert not any(
        e.kind == kind and e.pivot_at == 4 and e.observed_at == 14
        for e in result.events
    )
    assert {
        e.indicator
        for e in result.events
        if e.kind == kind and e.pivot_at == 4 and e.observed_at == 15
    } == set(bank)
    stricter = detect_divergences(
        high=c + 0.2,
        low=c - 0.2,
        close=c,
        indicators=bank,
        config=replace(cfg, min_indicators=4),
    )
    assert not stricter.events
