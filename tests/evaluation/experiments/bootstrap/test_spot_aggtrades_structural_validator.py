from __future__ import annotations

import hashlib
import io
import zipfile
from datetime import UTC, datetime

import pytest

from trade_rl.evaluation.experiments.bootstrap.spot_aggtrades_source_prereg import (
    canonical_spot_aggtrades_source_protocol,
)
from trade_rl.evaluation.experiments.bootstrap.spot_aggtrades_structural_validator import (
    build_structural_report,
    canonical_structural_report_bytes,
    decide_source_status,
    validate_archive_bytes,
)

PROTOCOL = canonical_spot_aggtrades_source_protocol()
SYMBOL = "BTCUSDT"
DATE = "2021-01-15"
URL = PROTOCOL.url_template.format(symbol=SYMBOL, date=DATE)
ARCHIVE_NAME = URL.rsplit("/", 1)[-1]
MEMBER_NAME = ARCHIVE_NAME.removesuffix(".zip") + ".csv"
DAY_START_MS = int(datetime(2021, 1, 15, tzinfo=UTC).timestamp() * 1000)


def _row(
    aggregate_id: object,
    price: object = "100.0",
    quantity: object = "2.0",
    first_trade_id: object = 10,
    last_trade_id: object = 11,
    timestamp: object = DAY_START_MS + 1_000,
    buyer_is_maker: object = "False",
    best_price_match: object = "True",
) -> list[object]:
    return [
        aggregate_id,
        price,
        quantity,
        first_trade_id,
        last_trade_id,
        timestamp,
        buyer_is_maker,
        best_price_match,
    ]


def _csv(rows: list[list[object]], *, header: bool = False) -> bytes:
    values: list[str] = []
    if header:
        values.append(",".join(PROTOCOL.expected_header))
    values.extend(",".join(str(value) for value in row) for row in rows)
    return ("\n".join(values) + "\n").encode("utf-8")


def _zip(
    rows: list[list[object]],
    *,
    header: bool = False,
    member_name: str = MEMBER_NAME,
    extra_member: bool = False,
) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr(member_name, _csv(rows, header=header))
        if extra_member:
            archive.writestr("extra.csv", b"x\n")
    return buffer.getvalue()


def _checksum(payload: bytes, *, name: str = ARCHIVE_NAME) -> bytes:
    digest = hashlib.sha256(payload).hexdigest()
    return f"{digest}  {name}\n".encode("ascii")


def _validate(
    rows: list[list[object]],
    *,
    header: bool = False,
    checksum: bytes | None = None,
    **zip_kwargs: object,
) -> dict[str, object]:
    payload = _zip(rows, header=header, **zip_kwargs)
    return validate_archive_bytes(
        PROTOCOL,
        symbol=SYMBOL,
        date=DATE,
        archive_bytes=payload,
        checksum_bytes=_checksum(payload) if checksum is None else checksum,
    )


def test_headerless_and_exact_header_are_same_strict_schema() -> None:
    rows = [_row(1), _row(2, timestamp=DAY_START_MS + 2_000)]
    no_header = _validate(rows)
    with_header = _validate(rows, header=True)

    assert no_header["header_present"] is False
    assert with_header["header_present"] is True
    for report in (no_header, with_header):
        assert report["schema_valid"] is True
        assert report["normalized_schema"] == list(PROTOCOL.expected_header)
        assert report["total_row_count"] == 2
        assert report["usable_row_count"] == 2
        assert report["provider_invalid_sentinel_count"] == 0
        assert report["malformed_row_count"] == 0
        assert report["usable_ids_strictly_increasing_unique"] is True
        assert report["usable_timestamps_nondecreasing"] is True
        assert report["timestamps_inside_requested_utc_date"] is True
        assert report["checksum_verified"] is True


def test_exact_provider_sentinel_is_unusable_and_ignored_for_order_checks() -> None:
    rows = [
        _row(1),
        _row(
            999,
            price="0",
            quantity="0",
            first_trade_id="-1",
            last_trade_id="-1",
            timestamp=DAY_START_MS + 1_500,
        ),
        _row(2, timestamp=DAY_START_MS + 2_000),
    ]
    report = _validate(rows)

    assert report["schema_valid"] is True
    assert report["total_row_count"] == 3
    assert report["usable_row_count"] == 2
    assert report["provider_invalid_sentinel_count"] == 1
    assert report["malformed_row_count"] == 0
    assert report["first_usable_aggregate_id"] == 1
    assert report["last_usable_aggregate_id"] == 2


def test_partial_or_near_sentinel_is_malformed() -> None:
    report = _validate(
        [
            _row(1),
            _row(
                2,
                price="0",
                quantity="0",
                first_trade_id="-1",
                last_trade_id=12,
                timestamp=DAY_START_MS + 2_000,
            ),
        ]
    )
    assert report["schema_valid"] is False
    assert report["provider_invalid_sentinel_count"] == 0
    assert report["malformed_row_count"] == 1


@pytest.mark.parametrize(
    "bad_row",
    [
        [1, "100", "2", 10, 11, DAY_START_MS + 1_000, "False"],
        _row(-1),
        _row(1.5),
        _row(1, price="0"),
        _row(1, quantity="nan"),
        _row(1, first_trade_id=-1),
        _row(1, first_trade_id=12, last_trade_id=11),
        _row(1, timestamp=-1),
        _row(1, timestamp=DAY_START_MS - 1),
        _row(1, buyer_is_maker="false"),
        _row(1, best_price_match="1"),
    ],
)
def test_malformed_rows_fail_closed(bad_row: list[object]) -> None:
    report = _validate([bad_row])
    assert report["schema_valid"] is False
    assert report["malformed_row_count"] >= 1
    assert report["usable_row_count"] == 0


def test_usable_order_and_timestamp_monotonicity_are_independent_oracles() -> None:
    duplicate_id = _validate([_row(2), _row(2, timestamp=DAY_START_MS + 2_000)])
    assert duplicate_id["schema_valid"] is False
    assert duplicate_id["usable_ids_strictly_increasing_unique"] is False

    decreasing_time = _validate(
        [
            _row(1, timestamp=DAY_START_MS + 2_000),
            _row(2, timestamp=DAY_START_MS + 1_000),
        ]
    )
    assert decreasing_time["schema_valid"] is False
    assert decreasing_time["usable_timestamps_nondecreasing"] is False


def test_wrong_member_extra_member_and_checksum_fail_closed() -> None:
    wrong_member = _validate([_row(1)], member_name="wrong.csv")
    assert wrong_member["schema_valid"] is False

    extra_member = _validate([_row(1)], extra_member=True)
    assert extra_member["schema_valid"] is False

    payload = _zip([_row(1)])
    wrong_name = validate_archive_bytes(
        PROTOCOL,
        symbol=SYMBOL,
        date=DATE,
        archive_bytes=payload,
        checksum_bytes=_checksum(payload, name="wrong.zip"),
    )
    assert wrong_name["checksum_verified"] is False
    assert wrong_name["schema_valid"] is False

    bad_digest = validate_archive_bytes(
        PROTOCOL,
        symbol=SYMBOL,
        date=DATE,
        archive_bytes=payload,
        checksum_bytes=f"{'0' * 64}  {ARCHIVE_NAME}\n".encode(),
    )
    assert bad_digest["checksum_verified"] is False
    assert bad_digest["schema_valid"] is False


def test_missing_archive_or_checksum_is_partial_not_incompatible() -> None:
    missing_archive = validate_archive_bytes(
        PROTOCOL,
        symbol=SYMBOL,
        date=DATE,
        archive_bytes=None,
        checksum_bytes=None,
    )
    payload = _zip([_row(1)])
    missing_checksum = validate_archive_bytes(
        PROTOCOL,
        symbol=SYMBOL,
        date=DATE,
        archive_bytes=payload,
        checksum_bytes=None,
    )

    assert missing_archive["archive_available"] is False
    assert missing_checksum["archive_available"] is True
    assert missing_checksum["checksum_available"] is False
    assert missing_checksum["schema_valid"] is True


def _all_good_reports() -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for symbol in PROTOCOL.symbols:
        for date in PROTOCOL.dates:
            url = PROTOCOL.url_template.format(symbol=symbol, date=date)
            name = url.rsplit("/", 1)[-1]
            member = name.removesuffix(".zip") + ".csv"
            timestamp = int(
                datetime.fromisoformat(date).replace(tzinfo=UTC).timestamp() * 1000
            )
            payload = _zip(
                [_row(1, timestamp=timestamp + 1_000)],
                member_name=member,
            )
            result.append(
                validate_archive_bytes(
                    PROTOCOL,
                    symbol=symbol,
                    date=date,
                    archive_bytes=payload,
                    checksum_bytes=_checksum(payload, name=name),
                )
            )
    return result


def test_frozen_status_gate_pass_partial_incompatible() -> None:
    good = _all_good_reports()
    assert decide_source_status(PROTOCOL, good) == PROTOCOL.pass_status

    missing = [dict(item) for item in good]
    missing[0].update(
        archive_available=False,
        checksum_available=False,
        checksum_verified=False,
    )
    assert decide_source_status(PROTOCOL, missing) == PROTOCOL.partial_status

    malformed = [dict(item) for item in good]
    malformed[0]["schema_valid"] = False
    assert decide_source_status(PROTOCOL, malformed) == PROTOCOL.incompatible_status


def test_canonical_report_is_structural_only_and_content_addressed() -> None:
    reports = _all_good_reports()
    report = build_structural_report(
        PROTOCOL,
        reports,
        protocol_head="9" * 40,
        protocol_seal_run_id=123,
        protocol_seal_artifact_id=456,
        protocol_seal_artifact_api_digest="a" * 64,
        protocol_fresh_artifact_id=789,
        protocol_fresh_artifact_api_digest="b" * 64,
    )
    encoded = canonical_structural_report_bytes(report)

    assert report["status"] == PROTOCOL.pass_status
    assert report["planned_archive_count"] == 20
    assert report["archive_reports"] == reports
    assert report["economic_values_inspected"] is False
    assert report["target_relation_computed"] is False
    assert report["evaluation_pnl_inspected"] is False
    assert report["feature_hypothesis_selected"] is False
    assert report["production_eligible"] is False
    assert report["live_trading_authorized"] is False
    assert isinstance(report["content_digest"], str)
    assert len(report["content_digest"]) == 64
    assert b"price_mean" not in encoded
    assert b"quantity_mean" not in encoded
    assert b"flow_imbalance" not in encoded
    assert b"beta" not in encoded
