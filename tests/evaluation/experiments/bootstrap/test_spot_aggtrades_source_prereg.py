from __future__ import annotations

import json

import pytest

from trade_rl.artifacts.canonical import canonical_json_bytes
from trade_rl.evaluation.experiments.bootstrap.spot_aggtrades_source_prereg import (
    SpotAggTradesSourceProtocol,
    canonical_spot_aggtrades_source_protocol,
    load_spot_aggtrades_source_protocol_bytes,
)

EXPECTED_SYMBOLS = ("BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT")
EXPECTED_DATES = ("2021-01-15", "2021-07-15", "2022-01-15", "2022-07-15")
EXPECTED_HEADER = (
    "agg_trade_id",
    "price",
    "quantity",
    "first_trade_id",
    "last_trade_id",
    "timestamp",
    "buyer_is_maker",
    "best_price_match",
)
EXPECTED_REPORT_FIELDS = (
    "symbol",
    "date",
    "url",
    "checksum_url",
    "archive_available",
    "raw_zip_size_bytes",
    "raw_zip_sha256",
    "checksum_available",
    "checksum_text",
    "checksum_digest",
    "checksum_verified",
    "member_name",
    "header_present",
    "normalized_schema",
    "total_row_count",
    "usable_row_count",
    "provider_invalid_sentinel_count",
    "malformed_row_count",
    "first_usable_aggregate_id",
    "last_usable_aggregate_id",
    "first_usable_event_timestamp_ms",
    "last_usable_event_timestamp_ms",
    "usable_ids_strictly_increasing_unique",
    "usable_timestamps_nondecreasing",
    "timestamps_inside_requested_utc_date",
    "schema_valid",
)


def test_canonical_protocol_freezes_complete_result_blind_contract() -> None:
    protocol = canonical_spot_aggtrades_source_protocol()

    assert protocol.symbols == EXPECTED_SYMBOLS
    assert protocol.dates == EXPECTED_DATES
    assert protocol.planned_archive_count == 20
    assert protocol.market == "spot"
    assert protocol.source_family == "aggTrades"
    assert protocol.url_template == (
        "https://data.binance.vision/data/spot/daily/aggTrades/"
        "{symbol}/{symbol}-aggTrades-{date}.zip"
    )
    assert protocol.checksum_suffix == ".CHECKSUM"
    assert protocol.expected_header == EXPECTED_HEADER
    assert protocol.field_count == 8
    assert protocol.provider_invalid_sentinel == {
        "price": "0",
        "quantity": "0",
        "first_trade_id": "-1",
        "last_trade_id": "-1",
    }
    assert protocol.strict_boolean_tokens == ("False", "True")
    assert protocol.allowed_archive_report_fields == EXPECTED_REPORT_FIELDS
    assert protocol.pass_status == "PASS_SPOT_AGGTRADES_SOURCE"
    assert protocol.partial_status == "PARTIAL_SPOT_AGGTRADES_SOURCE"
    assert protocol.incompatible_status == "INCOMPATIBLE_SPOT_AGGTRADES_SOURCE"
    assert protocol.replacement_sources_allowed is False
    assert protocol.economic_values_allowed_in_report is False
    assert protocol.target_relation_allowed is False
    assert protocol.evaluation_pnl_inspected is False
    assert protocol.feature_hypothesis_selected is False
    assert protocol.production_eligible is False
    assert protocol.live_trading_authorized is False

    payload = protocol.to_payload()
    encoded = protocol.canonical_json_bytes
    assert encoded == canonical_json_bytes(payload)
    assert b"flow_imbalance" not in encoded
    assert b"beta" not in encoded
    forbidden_report_fields = {
        "flow_imbalance",
        "beta",
        "pnl",
        "price_mean",
        "quantity_mean",
        "notional_mean",
    }
    assert forbidden_report_fields.isdisjoint(protocol.allowed_archive_report_fields)
    assert payload["evaluation_pnl_inspected"] is False
    assert len(protocol.digest) == 64


def test_url_roster_is_exact_and_ordered() -> None:
    protocol = canonical_spot_aggtrades_source_protocol()
    urls = protocol.planned_urls
    assert len(urls) == 20
    assert urls[0].endswith("BTCUSDT/BTCUSDT-aggTrades-2021-01-15.zip")
    assert urls[3].endswith("BTCUSDT/BTCUSDT-aggTrades-2022-07-15.zip")
    assert urls[4].endswith("ETHUSDT/ETHUSDT-aggTrades-2021-01-15.zip")
    assert urls[-1].endswith("ADAUSDT/ADAUSDT-aggTrades-2022-07-15.zip")
    assert tuple(url + ".CHECKSUM" for url in urls) == protocol.planned_checksum_urls


def test_protocol_rejects_post_inspection_mutation() -> None:
    protocol = canonical_spot_aggtrades_source_protocol()
    payload = protocol.to_payload()

    mutations: list[tuple[str, object]] = [
        ("dates", ["2021-01-16", *list(EXPECTED_DATES[1:])]),
        ("field_count", 7),
        ("strict_boolean_tokens", ["true", "false"]),
        ("strict_boolean_tokens", ["1", "0"]),
        ("replacement_sources_allowed", True),
        ("economic_values_allowed_in_report", True),
        ("target_relation_allowed", True),
        ("evaluation_pnl_inspected", True),
        ("feature_hypothesis_selected", True),
        ("pass_status", "PASS"),
    ]
    for field, value in mutations:
        changed = dict(payload)
        changed[field] = value
        with pytest.raises(ValueError):
            SpotAggTradesSourceProtocol.from_payload(changed)

    changed = dict(payload)
    changed["provider_invalid_sentinel"] = {
        "price": "0",
        "quantity": "0",
        "first_trade_id": "0",
        "last_trade_id": "-1",
    }
    with pytest.raises(ValueError):
        SpotAggTradesSourceProtocol.from_payload(changed)

    changed = dict(payload)
    changed["allowed_archive_report_fields"] = [
        *EXPECTED_REPORT_FIELDS,
        "price_mean",
    ]
    with pytest.raises(ValueError):
        SpotAggTradesSourceProtocol.from_payload(changed)


def test_loader_requires_exact_canonical_json_bytes() -> None:
    protocol = canonical_spot_aggtrades_source_protocol()
    encoded = protocol.canonical_json_bytes
    loaded = load_spot_aggtrades_source_protocol_bytes(encoded)
    assert loaded == protocol
    assert loaded.digest == protocol.digest

    parsed = json.loads(encoded)
    noncanonical = json.dumps(parsed, indent=2, sort_keys=True).encode() + b"\n"
    assert noncanonical != encoded
    with pytest.raises(ValueError, match="canonical"):
        load_spot_aggtrades_source_protocol_bytes(noncanonical)

    with pytest.raises(ValueError):
        load_spot_aggtrades_source_protocol_bytes(encoded + b" ")
