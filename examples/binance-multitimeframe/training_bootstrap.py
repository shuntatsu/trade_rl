#!/usr/bin/env python3
"""Validate shared market archives before CUDA preflight and full research."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Callable

from trade_rl.integrations.binance_cache import (
    BinanceVisionCacheReport,
    require_complete_binance_vision_cache,
)

_EXAMPLE_DIR = Path(__file__).resolve().parent
if str(_EXAMPLE_DIR) not in sys.path:
    sys.path.insert(0, str(_EXAMPLE_DIR))

import full_run_entrypoint  # noqa: E402
import sync_market_data  # noqa: E402


def check_maintained_cache(cache_root: Path) -> BinanceVisionCacheReport:
    return require_complete_binance_vision_cache(
        sync_market_data.build_maintained_plan(),
        cache_root=cache_root,
    )


def ensure_cache_root_argument(argv: list[str], cache_root: Path) -> list[str]:
    result = list(argv)
    if "--cache-root" not in result:
        result.extend(("--cache-root", str(cache_root)))
    return result


def run_bootstrap(
    *,
    cache_root: Path,
    check_cache: Callable[[Path], object] = check_maintained_cache,
    full_entrypoint: Callable[[], int] = full_run_entrypoint.main,
    postgres_market_data: bool = False,
) -> int:
    if postgres_market_data:
        print(
            json.dumps(
                {
                    "cache_root": None,
                    "market_data_source": "postgresql_expanded_indicator_cache",
                    "schema_version": "training_market_cache_preflight_v2",
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return full_entrypoint()

    try:
        report = check_cache(cache_root)
    except FileNotFoundError as error:
        raise FileNotFoundError(
            f"{error}. Run `python examples/binance-multitimeframe/"
            "run_docker_training.py` or "
            "`docker compose -f compose.training.yaml run --rm market-data-sync` "
            "before starting trainer."
        ) from error

    if isinstance(report, BinanceVisionCacheReport):
        print(
            json.dumps(
                {
                    "cache_root": str(report.cache_root),
                    "cached_count": report.cached_count,
                    "planned_count": report.planned_count,
                    "schema_version": "training_market_cache_preflight_v1",
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
    return full_entrypoint()


def main() -> int:
    cache_root = Path(
        os.environ.get(
            "TRADE_RL_MARKET_DATA_CACHE_ROOT",
            "/workspace/market-data/binance-vision",
        )
    )
    sys.argv[:] = ensure_cache_root_argument(sys.argv, cache_root)
    return run_bootstrap(
        cache_root=cache_root,
        postgres_market_data=os.environ.get("TRADE_RL_POSTGRES_MARKET_DATA", "").lower()
        == "true",
    )


if __name__ == "__main__":
    raise SystemExit(main())
