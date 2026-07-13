from __future__ import annotations

import json
from io import StringIO
from pathlib import Path

from trade_rl.cli.app import main
from trade_rl.data.artifact import load_market_dataset_artifact


def _write_market_csv(path: Path) -> None:
    rows = ["timestamp,available_at,open,high,low,close,volume,tradable"]
    previous = 100.0
    for hour in range(24):
        close = 100.0 + hour
        timestamp = f"2026-01-01T{hour:02d}:00:00Z"
        rows.append(
            f"{timestamp},{timestamp},{previous},{close + 1},{previous - 1},"
            f"{close},1000,true"
        )
        previous = close
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def test_data_build_cli_writes_reloadable_content_addressed_artifact(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "csv"
    source_root.mkdir()
    _write_market_csv(source_root / "BTCUSDT.csv")
    config_path = tmp_path / "market.json"
    config_path.write_text(
        json.dumps(
            {
                "source_root": "csv",
                "base_timeframe": "1h",
                "features": [
                    {
                        "name": "ret_1h",
                        "kind": "log_return",
                        "lookback": 1,
                        "max_staleness_hours": 4.0,
                    }
                ],
                "instruments": [
                    {
                        "symbol": "BTCUSDT",
                        "listed_at": "2026-01-01T00:00:00Z",
                        "volume_unit": "quote_notional",
                        "contract_multiplier": 1.0,
                    }
                ],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    output = tmp_path / "artifact"
    stdout = StringIO()

    exit_code = main(
        [
            "data",
            "build",
            "--config",
            str(config_path),
            "--output",
            str(output),
        ],
        stdout=stdout,
    )

    payload = json.loads(stdout.getvalue())
    dataset = load_market_dataset_artifact(output)
    assert exit_code == 0
    assert payload["schema"] == "market_dataset_build_result_v1"
    assert payload["dataset_id"] == dataset.dataset_id
    assert payload["artifact_digest"]
    assert dataset.symbols == ("BTCUSDT",)
    assert dataset.feature_names == ("ret_1h",)
    assert dataset.n_bars == 24
    assert (output / "manifest.json").is_file()
    assert (output / "arrays.npz").is_file()


def test_data_build_cli_is_deterministic_for_identical_inputs(tmp_path: Path) -> None:
    source_root = tmp_path / "csv"
    source_root.mkdir()
    _write_market_csv(source_root / "BTCUSDT.csv")
    config_path = tmp_path / "market.json"
    config_path.write_text(
        json.dumps(
            {
                "source_root": "csv",
                "base_timeframe": "1h",
                "features": [{"name": "ret", "kind": "log_return"}],
                "instruments": [
                    {
                        "symbol": "BTCUSDT",
                        "listed_at": "2026-01-01T00:00:00Z",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    outputs = (tmp_path / "first", tmp_path / "second")
    payloads: list[dict[str, object]] = []
    for output in outputs:
        stdout = StringIO()
        assert (
            main(
                [
                    "data",
                    "build",
                    "--config",
                    str(config_path),
                    "--output",
                    str(output),
                ],
                stdout=stdout,
            )
            == 0
        )
        payloads.append(json.loads(stdout.getvalue()))

    assert payloads[0] == payloads[1]
    assert (outputs[0] / "manifest.json").read_bytes() == (
        outputs[1] / "manifest.json"
    ).read_bytes()
    assert (outputs[0] / "arrays.npz").read_bytes() == (
        outputs[1] / "arrays.npz"
    ).read_bytes()
