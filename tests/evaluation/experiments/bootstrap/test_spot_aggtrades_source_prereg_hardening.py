from __future__ import annotations

import pytest

from trade_rl.evaluation.experiments.bootstrap.spot_aggtrades_source_prereg import (
    SpotAggTradesSourceProtocol,
    canonical_spot_aggtrades_source_protocol,
)


def test_protocol_freezes_archive_row_and_status_semantics() -> None:
    protocol = canonical_spot_aggtrades_source_protocol()

    assert protocol.checksum_must_name_exact_archive is True
    assert protocol.zip_exact_single_regular_csv_member is True
    assert protocol.member_name_must_match_planned_csv is True
    assert protocol.header_policy == "headerless_or_exact_expected_header"
    assert protocol.usable_aggregate_id_nonnegative_integer is True
    assert protocol.usable_price_quantity_finite_strictly_positive is True
    assert protocol.usable_trade_ids_nonnegative_first_le_last is True
    assert protocol.event_timestamp_nonnegative_millisecond_inside_requested_date is True
    assert protocol.provider_sentinel_requires_exact_four_field_match is True
    assert protocol.provider_sentinel_is_never_usable is True
    assert protocol.partial_or_near_sentinel_is_malformed is True
    assert protocol.usable_aggregate_ids_strictly_increasing_unique is True
    assert protocol.usable_timestamps_nondecreasing is True
    assert protocol.pass_requires_all_20_archives_and_checksums is True
    assert (
        protocol.partial_requires_missing_only_with_all_available_structurally_valid
        is True
    )
    assert protocol.incompatible_on_any_malformed_or_ambiguous_archive is True


def test_archive_and_gate_semantics_cannot_be_mutated_after_inspection() -> None:
    protocol = canonical_spot_aggtrades_source_protocol()
    payload = protocol.to_payload()

    mutations: tuple[tuple[str, object], ...] = (
        ("checksum_must_name_exact_archive", False),
        ("zip_exact_single_regular_csv_member", False),
        ("member_name_must_match_planned_csv", False),
        ("header_policy", "any_header"),
        ("usable_aggregate_id_nonnegative_integer", False),
        ("usable_price_quantity_finite_strictly_positive", False),
        ("usable_trade_ids_nonnegative_first_le_last", False),
        ("event_timestamp_nonnegative_millisecond_inside_requested_date", False),
        ("provider_sentinel_requires_exact_four_field_match", False),
        ("provider_sentinel_is_never_usable", False),
        ("partial_or_near_sentinel_is_malformed", False),
        ("usable_aggregate_ids_strictly_increasing_unique", False),
        ("usable_timestamps_nondecreasing", False),
        ("pass_requires_all_20_archives_and_checksums", False),
        (
            "partial_requires_missing_only_with_all_available_structurally_valid",
            False,
        ),
        ("incompatible_on_any_malformed_or_ambiguous_archive", False),
    )
    for field, value in mutations:
        changed = dict(payload)
        changed[field] = value
        with pytest.raises(ValueError):
            SpotAggTradesSourceProtocol.from_payload(changed)
