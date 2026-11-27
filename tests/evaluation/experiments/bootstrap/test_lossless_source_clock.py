"""Synthetic source-clock admission oracles; never acquire market data."""

from dataclasses import replace
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from tests.evaluation.experiments.bootstrap.test_config import _valid_payload, _write
from tests.evaluation.experiments.bootstrap.test_final_window_config import _v3_payload
from trade_rl.evaluation.experiments.bootstrap.config import (
    load_canonical_m2_bootstrap_config,
)


@pytest.mark.parametrize("direct", [False, True])
@pytest.mark.parametrize(
    ("field", "raw"),
    [
        ("data_start", "1400-01-01T00:00:00+00:00"),
        ("data_start", "2500-01-01T00:00:00+00:00"),
        ("data_stop_exclusive", "2700-01-01T00:00:00+00:00"),
        ("final_evaluation_start", "2500-01-01T00:00:00+00:00"),
        ("final_evaluation_stop_exclusive", "2500-01-01T00:00:00+00:00"),
    ],
)
def test_bootstrap_rejects_unrepresentable_source_clock(
    tmp_path: Path, field: str, raw: str, direct: bool
) -> None:
    payload = _v3_payload() if field.startswith("final_") else _valid_payload()
    changes = {field: raw}
    if field == "final_evaluation_start":
        changes["final_evaluation_stop_exclusive"] = "2501-01-01T00:00:00+00:00"
    # An out-of-range instant must not be admitted under its wrapped int64
    # nanosecond projection, even if the projected chronology looks valid.
    with pytest.raises(ValueError, match=rf"{field}.*losslessly representable"):
        if direct:
            valid = load_canonical_m2_bootstrap_config(_write(tmp_path, payload))
            replace(
                valid,
                **{
                    key: datetime.fromisoformat(value) for key, value in changes.items()
                },
            )
        else:
            payload.update(changes)
            load_canonical_m2_bootstrap_config(_write(tmp_path, payload))


@pytest.mark.parametrize(
    ("field", "raw"),
    [
        ("data_start", "2024-01-01T00:00:00.000000001+00:00"),
        ("data_start", "2024-01-01T00:00:00,000000001Z"),
        ("data_stop_exclusive", "2025-01-01T00:00:00.0000000001Z"),
        ("data_start", "2024-01-01T00:00:00+00:00:00.000001"),
        ("data_start", "2024-01-01T00:00:00-00:00:00.000001"),
        ("data_start", "2024-01-01T09:00:00+09:00:00.000000001"),
        ("final_evaluation_start", "2025-01-01T00:00:00.000000001Z"),
        ("final_evaluation_stop_exclusive", "2025-04-01T00:00:00.000000001Z"),
    ],
)
def test_bootstrap_rejects_silently_truncated_clock_text(
    tmp_path: Path, field: str, raw: str
) -> None:
    payload = _v3_payload() if field.startswith("final_") else _valid_payload()
    payload[field] = raw
    with pytest.raises(ValueError, match=rf"{field}.*losslessly representable"):
        load_canonical_m2_bootstrap_config(_write(tmp_path, payload))


@pytest.mark.parametrize(
    "raw",
    [
        "2024-01-01T00:00:00.0000000000Z",
        "2024-01-01T09:00:00+09:00",
        "2023-12-31T19:00:00-05:00",
        "2024-01-01T09:00:00.25+09:00:00.25",
        "2024-01-01T00:00:00+00:00:00.000000000",
    ],
)
def test_exact_utc_aliases_keep_existing_config_digest(
    tmp_path: Path, raw: str
) -> None:
    payload = _valid_payload()
    expected = load_canonical_m2_bootstrap_config(_write(tmp_path, payload))
    payload["data_start"] = raw
    actual = load_canonical_m2_bootstrap_config(_write(tmp_path, payload))
    assert actual.digest == expected.digest
    assert actual.data_start == expected.data_start
    assert np.datetime64(actual.data_start.replace(tzinfo=None), "ns") == (
        np.datetime64("2024-01-01T00:00:00", "ns")
    )
