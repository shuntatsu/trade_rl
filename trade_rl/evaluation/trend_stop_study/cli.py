"""Command line entry for the trend-stop study (writes new files only).

Nothing is tunable here: the protocol module fixes every parameter, including the
complete eleven-symbol roster (there is no exclusion option). The command refuses to
start without a well-formed G4 authorization comment URL, refuses a non-empty output
directory, refuses to run from a dirty Git working tree, and never touches the
network.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from trade_rl.data import MarketDataset
from trade_rl.evaluation.trend_stop_study.data import (
    build_vision_dataset,
    load_exported_dataset,
)
from trade_rl.evaluation.trend_stop_study.protocol import (
    PROTOCOL,
    TrendStopProtocol,
    validate_authorization_url,
)
from trade_rl.evaluation.trend_stop_study.runner import (
    capture_code_provenance,
    run_study,
)


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode(
        "utf-8"
    )


def write_outputs(
    output_dir: Path,
    result: Mapping[str, Any],
    cells: Mapping[str, Mapping[str, Any]],
    equity: Mapping[str, np.ndarray],
) -> list[Path]:
    """Write every cell, the equity arrays and the result; never overwrite."""

    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"{output_dir} is not empty; refusing to overwrite")
    (output_dir / "cells").mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for name in sorted(cells):
        path = output_dir / "cells" / f"{name}.json"
        with path.open("xb") as stream:
            stream.write(_json_bytes(cells[name]))
        written.append(path)
    if equity:
        equity_path = output_dir / "equity.npz"
        with equity_path.open("xb") as stream:
            np.savez_compressed(stream, **equity)
        written.append(equity_path)
    result_path = output_dir / "result.json"
    with result_path.open("xb") as stream:
        stream.write(_json_bytes(result))
    written.append(result_path)
    return written


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m trade_rl.evaluation.trend_stop_study",
        description=(
            "Run the preregistered trend-with-stops development study. Parameters "
            "are fixed by the protocol module; nothing is tunable here."
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--g4-authorization-url",
        required=True,
        help="owner's issue / pull-request comment URL authorizing this G4 run",
    )
    for timeframe in PROTOCOL.timeframes:
        parser.add_argument(
            f"--dataset-{timeframe}",
            type=Path,
            help=f"published {timeframe} Dataset artifact (market-store export output)",
        )
    parser.add_argument(
        "--vision-cache",
        type=Path,
        help="fallback: complete local Binance Vision cache (never the network)",
    )
    parser.add_argument(
        "--exchange-info-dir",
        type=Path,
        help="fallback: frozen exchange-information directory",
    )
    return parser


def _load_datasets(
    args: argparse.Namespace, protocol: TrendStopProtocol
) -> dict[str, MarketDataset]:
    datasets: dict[str, MarketDataset] = {}
    for timeframe in protocol.timeframes:
        path = getattr(args, f"dataset_{timeframe}")
        if path is not None:
            datasets[timeframe] = load_exported_dataset(path)
        elif args.vision_cache is not None and args.exchange_info_dir is not None:
            datasets[timeframe] = build_vision_dataset(
                protocol,
                timeframe,
                cache_root=args.vision_cache,
                exchange_info_dir=args.exchange_info_dir,
            )
        else:
            raise ValueError(
                f"no {timeframe} Dataset: pass --dataset-{timeframe} or both "
                "--vision-cache and --exchange-info-dir"
            )
    return datasets


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        authorization = validate_authorization_url(args.g4_authorization_url)
    except ValueError as error:
        print(error, file=sys.stderr)
        return 2
    output_dir: Path = args.output_dir
    if output_dir.exists() and any(output_dir.iterdir()):
        print("refusing to overwrite an existing study output", file=sys.stderr)
        return 2
    try:
        provenance = capture_code_provenance()
        provenance.require_clean()
        datasets = _load_datasets(args, PROTOCOL)
        result, cells, equity = run_study(
            datasets,
            PROTOCOL,
            g4_authorization_url=authorization,
            code_provenance=provenance,
            progress=lambda message: print(message, flush=True),
        )
    except ValueError as error:
        print(error, file=sys.stderr)
        return 2
    write_outputs(output_dir, result, cells, equity)
    print(f"label={result['label']} passing={result['passing_configurations']}")
    return 0 if result["label"] != "INVALID" else 3


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
