from __future__ import annotations

import io
import zipfile

import numpy as np
import pytest

from trade_rl.integrations.binance import (
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


def test_reference_alignment_rejects_future_available_reference() -> None:
    series = parse_vision_book_depth_archive(_payload(), source="fixture")

    with pytest.raises(BinanceTransportError, match="future|availability|available"):
        validate_book_depth_reference_alignment(
            series,
            np.array([100.0]),
            reference_available_at=series.timestamps + np.timedelta64(1, "s"),
            max_relative_deviation_rate=0.10,
        )
