#!/usr/bin/env python3
"""Run the maintained three-asset native multi-timeframe research pipeline."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from trade_rl.data import publish_market_dataset_artifact
from trade_rl.integrations.binance import (
    BinanceMarket,
    BinancePublicTransport,
    BinanceTransportError,
    BinanceTransportMode,
    build_binance_market_dataset,
)

_SYMBOLS = ("BTCUSDT", "ETHUSDT", "BNBUSDT")
_NATIVE_TIMEFRAMES = ("15m", "1h", "4h", "1d")
_FEATURE_TIMEFRAMES = ("15m", "4h", "1d")
_START = "2024-12-01T00:00:00Z"
_END = "2026-06-01T00:00:00Z"
_TRAIN_RUN_COMMAND = ("train", "run")
_WALK_FORWARD_RUN_COMMAND = ("walk-forward", "run")
_FALLBACK_METADATA: dict[str, dict[str, object]] = {
    "BTCUSDT": {
        "listed_at": "2019-09-08T00:00:00Z",
        "tick_size": 0.1,
        "lot_size": 0.001,
        "minimum_notional": 5.0,
    },
    "ETHUSDT": {
        "listed_at": "2019-11-27T00:00:00Z",
        "tick_size": 0.01,
        "lot_size": 0.001,
        "minimum_notional": 5.0,
    },
    "BNBUSDT": {
        "listed_at": "2020-02-10T00:00:00Z",
        "tick_size": 0.01,
        "lot_size": 0.01,
        "minimum_notional": 5.0,
    },
}


def _parse_utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("time range must include a timezone")
    return parsed.astimezone(UTC)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _run_cli(arguments: list[str], *, root: Path, log_path: Path) -> dict[str, Any]:
    command = [sys.executable, "-m", "trade_rl.cli.app", *arguments]
    completed = subprocess.run(
        command,
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        "command: " + repr(command) + "\n\nstdout:\n" + completed.stdout
        + "\n\nstderr:\n" + completed.stderr,
        encoding="utf-8",
    )
    if completed.returncode != 0:
        raise RuntimeError(f"command failed; see {log_path}: {command!r}")
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError(f"command produced no JSON output: {command!r}")
    payload = json.loads(lines[-1])
    if not isinstance(payload, dict):
        raise RuntimeError(f"command JSON result must be an object: {command!r}")
    return dict(payload)


def _filter_number(filters: list[dict[str, object]], kind: str, *names: str) -> float:
    for item in filters:
        if item.get("filterType") != kind:
            continue
        for name in names:
            value = item.get(name)
            if isinstance(value, (str, int, float)) and not isinstance(value, bool):
                return float(value)
    raise ValueError(f"exchangeInfo is missing {kind} {names}")


def _metadata_from_exchange_info(payload: object) -> dict[str, dict[str, object]]:
    if not isinstance(payload, dict) or not isinstance(payload.get("symbols"), list):
        raise ValueError("exchangeInfo payload is invalid")
    by_symbol = {
        item.get("symbol"): item
        for item in payload["symbols"]
        if isinstance(item, dict) and isinstance(item.get("symbol"), str)
    }
    result: dict[str, dict[str, object]] = {}
    for symbol in _SYMBOLS:
        item = by_symbol.get(symbol)
        if not isinstance(item, dict) or not isinstance(item.get("filters"), list):
            raise ValueError(f"exchangeInfo has no active metadata for {symbol}")
        filters = [value for value in item["filters"] if isinstance(value, dict)]
        onboard = item.get("onboardDate")
        if not isinstance(onboard, (int, float)) or isinstance(onboard, bool):
            raise ValueError(f"exchangeInfo has no onboardDate for {symbol}")
        result[symbol] = {
            "listed_at": datetime.fromtimestamp(float(onboard) / 1000.0, tz=UTC).isoformat(),
            "tick_size": _filter_number(filters, "PRICE_FILTER", "tickSize"),
            "lot_size": _filter_number(filters, "LOT_SIZE", "stepSize"),
            "minimum_notional": _filter_number(
                filters,
                "MIN_NOTIONAL",
                "notional",
                "minNotional",
            ),
        }
    return result


def _resolve_metadata(
    transport: BinancePublicTransport,
    *,
    snapshot_path: Path,
) -> tuple[dict[str, dict[str, object]], str, str | None]:
    try:
        payload, source = transport.load_exchange_information(
            market=BinanceMarket.USDS_M
        )
        metadata = _metadata_from_exchange_info(payload)
        _write_json(
            snapshot_path,
            {"metadata": metadata, "payload": payload, "source": source},
        )
        return metadata, str(source), None
    except (BinanceTransportError, ValueError) as error:
        _write_json(
            snapshot_path,
            {
                "error": str(error),
                "metadata": _FALLBACK_METADATA,
                "source": "checked-in-fallback",
            },
        )
        return _FALLBACK_METADATA, "checked-in-fallback", str(error)


def _build_dataset(
    *,
    output: Path,
    transport: BinancePublicTransport,
    metadata: dict[str, dict[str, object]],
) -> dict[str, object]:
    result = build_binance_market_dataset(
        market=BinanceMarket.USDS_M,
        symbols=_SYMBOLS,
        interval="1h",
        feature_timeframes=_FEATURE_TIMEFRAMES,
        start_time=_parse_utc(_START),
        end_time=_parse_utc(_END),
        transport_mode=BinanceTransportMode.VISION,
        transport=transport,
        tick_sizes=tuple(float(metadata[symbol]["tick_size"]) for symbol in _SYMBOLS),
        lot_sizes=tuple(float(metadata[symbol]["lot_size"]) for symbol in _SYMBOLS),
        minimum_notionals=tuple(
            float(metadata[symbol]["minimum_notional"]) for symbol in _SYMBOLS
        ),
        listed_ats=tuple(
            _parse_utc(str(metadata[symbol]["listed_at"])) for symbol in _SYMBOLS
        ),
    )
    published = publish_market_dataset_artifact(output, result.dataset)
    if result.dataset.n_bars != 13_104:
        raise RuntimeError(
            f"expected 13,104 hourly bars, observed {result.dataset.n_bars}"
        )
    if result.dataset.symbols != _SYMBOLS:
        raise RuntimeError(f"unexpected symbol order: {result.dataset.symbols}")
    expected_features = (
        "15m__log_return_1bar",
        "15m__realized_volatility_4bar",
        "1h__log_return_1bar",
        "1h__log_return_1d",
        "1h__volume_zscore_1d",
        "1h__funding_bps",
        "4h__log_return_1bar",
        "4h__realized_volatility_6bar",
        "1d__log_return_1bar",
        "1d__log_return_7bar",
    )
    if result.dataset.feature_names != expected_features:
        raise RuntimeError(f"unexpected feature contract: {result.dataset.feature_names}")
    return {
        "artifact_digest": published.artifact_digest,
        "dataset_id": result.dataset.dataset_id,
        "feature_names": list(result.dataset.feature_names),
        "feature_timeframes": list(result.feature_timeframes),
        "n_bars": result.dataset.n_bars,
        "n_features": result.dataset.n_features,
        "n_symbols": result.dataset.n_symbols,
        "sources_used": list(result.sources_used),
        "symbols": list(result.dataset.symbols),
    }


def _require_file(path: Path) -> None:
    if not path.is_file() or path.stat().st_size <= 0:
        raise RuntimeError(f"required artifact file is missing or empty: {path}")


def _verify_training(path: Path) -> None:
    for relative in ("run.json", "ensemble.json", "environment.json"):
        _require_file(path / relative)
    for index in range(3):
        member = path / f"members/member-{index:03d}"
        _require_file(member / "policy.zip")
        checkpoints = tuple(member.glob("checkpoints/*.zip"))
        if not checkpoints:
            raise RuntimeError(f"member {index} has no retained checkpoints")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-root", type=Path, default=Path("var/binance-multitimeframe-full"))
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    work_root = args.work_root if args.work_root.is_absolute() else root / args.work_root
    if work_root.exists():
        shutil.rmtree(work_root)
    work_root.mkdir(parents=True)

    cache_root = work_root / "vision-cache"
    transport = BinancePublicTransport(
        timeout_seconds=60.0,
        max_attempts=4,
        retry_backoff_seconds=0.5,
        cache_root=cache_root,
    )
    metadata, metadata_source, metadata_error = _resolve_metadata(
        transport,
        snapshot_path=work_root / "exchange-info.json",
    )

    dataset_a_path = work_root / "dataset-a"
    dataset_b_path = work_root / "dataset-b"
    dataset_a = _build_dataset(
        output=dataset_a_path,
        transport=transport,
        metadata=metadata,
    )
    dataset_b = _build_dataset(
        output=dataset_b_path,
        transport=transport,
        metadata=metadata,
    )
    if dataset_a["dataset_id"] != dataset_b["dataset_id"]:
        raise RuntimeError("repeated multi-timeframe builds produced different dataset IDs")
    if dataset_a["artifact_digest"] != dataset_b["artifact_digest"]:
        raise RuntimeError("repeated multi-timeframe builds produced different artifact digests")

    artifact_root = work_root / "artifacts"
    training = _run_cli(
        [
            *_TRAIN_RUN_COMMAND,
            "--config",
            str(root / "examples/binance-multitimeframe/training-full.json"),
            "--dataset",
            str(dataset_a_path),
            "--output",
            str(artifact_root),
            "--run-id",
            "binance-multitimeframe-full-training",
        ],
        root=root,
        log_path=work_root / "training.log",
    )
    training_path = Path(str(training["artifact_path"]))
    if not training_path.is_absolute():
        training_path = root / training_path
    _verify_training(training_path)

    walk_forward = _run_cli(
        [
            *_WALK_FORWARD_RUN_COMMAND,
            "--config",
            str(root / "examples/binance-multitimeframe/walk-forward-full.json"),
            "--dataset",
            str(dataset_a_path),
            "--output",
            str(artifact_root),
            "--run-id",
            "binance-multitimeframe-full-walk-forward",
        ],
        root=root,
        log_path=work_root / "walk-forward.log",
    )
    walk_forward_path = Path(str(walk_forward["artifact_path"]))
    if not walk_forward_path.is_absolute():
        walk_forward_path = root / walk_forward_path
    _require_file(walk_forward_path / "run.json")

    summary = {
        "dataset": dataset_a,
        "dataset_repeat": dataset_b,
        "end_time": _END,
        "metadata_error": metadata_error,
        "metadata_source": metadata_source,
        "native_timeframes": list(_NATIVE_TIMEFRAMES),
        "production_status": "NO-GO",
        "schema": "binance_multitimeframe_full_research_v1",
        "start_time": _START,
        "training": training,
        "walk_forward": walk_forward,
    }
    _write_json(work_root / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
