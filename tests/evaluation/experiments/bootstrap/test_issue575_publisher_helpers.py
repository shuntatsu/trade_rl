from __future__ import annotations

import hashlib
import io
import math
import zipfile
from datetime import UTC, datetime

import numpy as np
import pytest

from tools import tmp_issue575_basis_publisher as publisher
from tools import tmp_issue575_basis_verifier as verifier


def _row(open_ms: int, *, close: float = 101.0) -> list[str]:
    return [
        str(open_ms),
        "100.0",
        "102.0",
        "99.0",
        str(close),
        "1.0",
        str(open_ms + publisher.INTERVAL_MS - 1),
        "1000.0",
        "1",
        "0.5",
        "500.0",
        "0",
    ]


def _archive(rows: list[list[str]], *, header: bool = False) -> bytes:
    lines: list[str] = []
    if header:
        lines.append(",".join(publisher.EXPECTED_HEADER))
    lines.extend(",".join(row) for row in rows)
    csv_bytes = ("\n".join(lines) + "\n").encode()
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr("BTCUSDT-1h-2021-01.csv", csv_bytes)
    return buffer.getvalue()


def _checksum(payload: bytes) -> bytes:
    digest = hashlib.sha256(payload).hexdigest()
    return f"{digest}  BTCUSDT-1h-2021-01.zip\n".encode()


@pytest.mark.parametrize("header", [False, True])
def test_publisher_and_verifier_normalize_same_perp_archive(
    monkeypatch: pytest.MonkeyPatch,
    header: bool,
) -> None:
    start_ms = int(datetime(2021, 1, 1, tzinfo=UTC).timestamp() * 1_000)
    payload = _archive(
        [_row(start_ms), _row(start_ms + publisher.INTERVAL_MS)], header=header
    )
    checksum = _checksum(payload)
    url = (
        "https://data.binance.vision/data/futures/um/monthly/klines/"
        "BTCUSDT/1h/BTCUSDT-1h-2021-01.zip"
    )
    expected_entry, expected_rows = publisher._parse_perp_archive(
        symbol="BTCUSDT",
        month="2021-01",
        payload=payload,
        checksum_payload=checksum,
        url=url,
    )

    def fake_fetch(request_url: str) -> bytes:
        if request_url == url:
            return payload
        if request_url == url + ".CHECKSUM":
            return checksum
        raise AssertionError(request_url)

    monkeypatch.setattr(verifier, "_fetch", fake_fetch)
    observed_entry, observed_rows = verifier._fresh_perp_archive(
        symbol="BTCUSDT", month="2021-01", url=url
    )
    assert observed_entry == expected_entry
    assert observed_rows == expected_rows
    assert expected_entry["missing_grid_rows"] == 742


def test_perp_parser_rejects_close_time_drift() -> None:
    start_ms = int(datetime(2021, 1, 1, tzinfo=UTC).timestamp() * 1_000)
    bad = _row(start_ms)
    bad[6] = str(start_ms + publisher.INTERVAL_MS)
    payload = _archive([bad])
    url = (
        "https://data.binance.vision/data/futures/um/monthly/klines/"
        "BTCUSDT/1h/BTCUSDT-1h-2021-01.zip"
    )
    with pytest.raises(RuntimeError, match="close-time"):
        publisher._parse_perp_archive(
            symbol="BTCUSDT",
            month="2021-01",
            payload=payload,
            checksum_payload=_checksum(payload),
            url=url,
        )


def _synthetic_rows() -> tuple[
    dict[str, list[tuple[int, float, float, float, float, float]]],
    dict[str, list[tuple[int, float]]],
]:
    start_ms = int(publisher.SOURCE_OPEN_START.timestamp() * 1_000)
    stop_ms = int(publisher.SOURCE_OPEN_END.timestamp() * 1_000)
    open_times = range(start_ms, stop_ms, publisher.INTERVAL_MS)
    perp: dict[str, list[tuple[int, float, float, float, float, float]]] = {}
    index: dict[str, list[tuple[int, float]]] = {}
    for symbol_index, symbol in enumerate(publisher.SYMBOLS):
        perp_rows: list[tuple[int, float, float, float, float, float]] = []
        index_rows: list[tuple[int, float]] = []
        index_close = 100.0 + symbol_index
        perp_close = index_close * 1.001
        for open_ms in open_times:
            perp_rows.append(
                (open_ms, perp_close, perp_close, perp_close, perp_close, 1_000.0)
            )
            index_rows.append((open_ms, index_close))
        perp[symbol] = perp_rows
        index[symbol] = index_rows
    return perp, index


def test_publisher_and_verifier_build_byte_identity_equivalent_dataset() -> None:
    perp, index = _synthetic_rows()
    digest = "a" * 64
    published = publisher._build_dataset(
        perp_rows=perp,
        index_rows=index,
        manifest_digest=digest,
    )
    reconstructed = verifier._rebuild_dataset(
        perp=perp,
        index=index,
        manifest_digest=digest,
    )
    assert published.dataset_id == reconstructed.dataset_id
    assert published.feature_names == ("1h__perp_index_log_basis_bps",)
    assert np.all(published.feature_available)
    expected = 10_000.0 * math.log(1.001)
    np.testing.assert_allclose(
        published.features[:, :, 0],
        np.float32(expected),
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_array_equal(
        published.features,
        reconstructed.features,
    )
