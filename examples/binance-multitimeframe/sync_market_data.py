#!/usr/bin/env python3
"""Synchronize the maintained Binance Vision archive set into shared storage."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Callable

from trade_rl.integrations.binance import BinancePublicTransport
from trade_rl.integrations.binance_cache import (
    BinanceVisionCachePlan,
    sync_binance_vision_cache,
)

_EXAMPLE_DIR = Path(__file__).resolve().parent
if str(_EXAMPLE_DIR) not in sys.path:
    sys.path.insert(0, str(_EXAMPLE_DIR))

import full_research_pipeline as pipeline  # noqa: E402


def _parse_utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("maintained research timestamps must include a timezone")
    return parsed.astimezone(UTC)


def build_maintained_plan() -> BinanceVisionCachePlan:
    from trade_rl.integrations.binance_cache import plan_binance_vision_cache

    return plan_binance_vision_cache(
        market="usds-m",
        symbols=pipeline._SYMBOLS,
        intervals=pipeline._NATIVE_TIMEFRAMES,
        start_time=_parse_utc(pipeline._START),
        end_time=_parse_utc(pipeline._END),
    )


def _is_cache_payload(relative: Path) -> bool:
    if len(relative.parts) != 2 or relative.suffix != ".bin":
        return False
    directory, filename = relative.parts
    stem = Path(filename).stem
    hexadecimal = set("0123456789abcdef")
    return (
        len(directory) == 2
        and set(directory) <= hexadecimal
        and len(stem) == 64
        and set(stem) <= hexadecimal
        and stem.startswith(directory)
    )


def import_legacy_cache(*, source_root: Path, destination_root: Path) -> int:
    """Copy valid nonempty legacy cache files without overwriting new storage."""

    if not source_root.is_dir():
        return 0
    copied = 0
    for source in sorted(source_root.rglob("*.bin")):
        relative = source.relative_to(source_root)
        evidence_source = source.with_suffix(".json")
        if (
            not _is_cache_payload(relative)
            or source.stat().st_size <= 0
            or not evidence_source.is_file()
        ):
            continue
        destination = destination_root / relative
        if destination.exists():
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        shutil.copy2(evidence_source, destination.with_suffix(".json"))
        copied += 1
    return copied


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def run_sync(
    *,
    cache_root: Path,
    legacy_cache_root: Path | None = None,
    transport_factory: Callable[..., BinancePublicTransport] = BinancePublicTransport,
) -> dict[str, object]:
    root = cache_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    legacy_imported_count = 0
    if legacy_cache_root is not None:
        legacy_imported_count = import_legacy_cache(
            source_root=legacy_cache_root,
            destination_root=root,
        )
    plan = build_maintained_plan()
    transport = transport_factory(
        timeout_seconds=60.0,
        max_attempts=4,
        retry_backoff_seconds=0.5,
        cache_root=root,
    )
    report = sync_binance_vision_cache(plan, transport=transport)
    payload = {
        "schema_version": "binance_vision_cache_sync_v1",
        "legacy_imported_count": legacy_imported_count,
        **plan.to_dict(),
        **report.to_dict(),
    }
    _atomic_json(root / "sync-report.json", payload)
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path(
            os.environ.get(
                "TRADE_RL_MARKET_DATA_CACHE_ROOT",
                "/workspace/market-data/binance-vision",
            )
        ),
    )
    parser.add_argument(
        "--legacy-cache-root",
        type=Path,
        default=Path(
            os.environ.get(
                "TRADE_RL_LEGACY_MARKET_DATA_CACHE_ROOT",
                "/workspace/legacy-var/cache/binance-vision",
            )
        ),
    )
    args = parser.parse_args(argv)
    result = run_sync(
        cache_root=args.cache_root,
        legacy_cache_root=args.legacy_cache_root,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
