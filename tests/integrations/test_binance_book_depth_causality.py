from __future__ import annotations

import io
import zipfile

import numpy as np
import pytest

from trade_rl.integrations.binance import (
    BinanceBookDepthSeries,
    BinanceTransportError,
    parse_vision_book_depth_archive,
    validate_book_depth_reference_alignment,
)


def _payload() -> bytes:
    rows = ["timestamp,percentage,depth,notional"]
    for band in (-5, -4, -3, -2, -1, 1, 2, 3, 4, 5):
        depth = float(abs(band) * 10)
        price = 100.0 * (1.0 + band * 0.005)
        rows.append(f"2025-05-19 11:07:31,{band},{depth:.8f},{depth * price:.8f}")
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "BTCUSDT-bookDepth-2025-05-19.csv",
            "\n".join(rows) + "\n",
        )
    return buffer.getvalue()


def _series() -> BinanceBookDepthSeries:
    return parse_vision_book_depth_archive(_payload(), source="fixture")


def test_reference_alignment_rejects_future_available_reference() -> None:
    series = _series()

    with pytest.raises(BinanceTransportError, match="future|availability|available"):
        validate_book_depth_reference_alignment(
            series,
            np.array([100.0]),
            reference_available_at=series.timestamps + np.timedelta64(1, "s"),
            max_relative_deviation_rate=0.10,
        )


def test_reference_alignment_rejects_missing_availability_entry() -> None:
    series = _series()

    with pytest.raises(ValueError, match="reference_available_at"):
        validate_book_depth_reference_alignment(
            series,
            np.array([100.0]),
            reference_available_at=np.array([], dtype="datetime64[ns]"),
            max_relative_deviation_rate=0.10,
        )


def test_reference_alignment_rejects_nat_availability() -> None:
    series = _series()

    with pytest.raises(ValueError, match="NaT"):
        validate_book_depth_reference_alignment(
            series,
            np.array([100.0]),
            reference_available_at=np.array(["NaT"], dtype="datetime64[ns]"),
            max_relative_deviation_rate=0.10,
        )
