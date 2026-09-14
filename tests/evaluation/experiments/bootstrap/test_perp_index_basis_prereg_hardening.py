from __future__ import annotations

from dataclasses import replace

import pytest

from trade_rl.evaluation.experiments.bootstrap.perp_index_basis_prereg import (
    canonical_perp_index_basis_protocol,
)


def test_protocol_freezes_exact_official_index_preflight_roster_and_checksum() -> (
    None
):
    protocol = canonical_perp_index_basis_protocol()

    assert protocol.full_preflight_months == tuple(
        f"{year}-{month:02d}"
        for year in (2021, 2022)
        for month in range(1, 13)
    )
    assert protocol.full_preflight_archive_root == (
        "https://data.binance.vision/data/futures/um/monthly/indexPriceKlines"
    )
    assert protocol.full_preflight_archive_url_template == (
        "https://data.binance.vision/data/futures/um/monthly/indexPriceKlines/"
        "{symbol}/1h/{symbol}-1h-{month}.zip"
    )
    assert protocol.full_preflight_checksum_suffix == ".CHECKSUM"
    assert protocol.full_preflight_checksum_required is True
    assert protocol.index_source_market == "USD_M"
    assert protocol.perpetual_source_market == "USD_M"
    assert protocol.perpetual_source_family == "klines"


def test_protocol_forbids_unregistered_regression_and_rank_rescues() -> None:
    protocol = canonical_perp_index_basis_protocol()

    assert protocol.rank_transform_allowed is False
    assert protocol.weighted_regression_allowed is False
    assert protocol.robust_regression_fallback_allowed is False

    mutations: tuple[dict[str, object], ...] = (
        {"full_preflight_months": protocol.full_preflight_months[:-1]},
        {"full_preflight_archive_root": "https://example.invalid"},
        {"full_preflight_archive_url_template": "https://example.invalid/{symbol}.zip"},
        {"full_preflight_checksum_suffix": ".sha256"},
        {"full_preflight_checksum_required": False},
        {"index_source_market": "COIN_M"},
        {"perpetual_source_market": "COIN_M"},
        {"perpetual_source_family": "markPriceKlines"},
        {"rank_transform_allowed": True},
        {"weighted_regression_allowed": True},
        {"robust_regression_fallback_allowed": True},
    )
    for mutation in mutations:
        with pytest.raises(ValueError, match="preregistered"):
            replace(protocol, **mutation)
