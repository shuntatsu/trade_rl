#!/usr/bin/env python3
"""Synchronize the maintained Binance Vision archive set into shared storage."""

from __future__ import annotations

import argparse
import json
import os
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
    transport_factory: Callable[..., BinancePublicTransport] = BinancePublicTransport,
) -> dict[str, object]:
    root = cache_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
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
    args = parser.parse_args(argv)
    result = run_sync(cache_root=args.cache_root)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
